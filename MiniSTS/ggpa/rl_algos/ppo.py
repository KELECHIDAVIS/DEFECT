'''
ppo.py -- proximal policy optimization for d.e.f.e.c.t.

parallels dqn.py in structure: two networks trained simultaneously
  - card actor-critic:   266-dim state -> policy over 11 card slots + state value
  - target actor-critic: 277-dim state (state + card one-hot) -> policy over 5 enemy slots + value

differences from dqn:
  - on-policy: collects full trajectories, then updates with the clipped
    surrogate objective over several epochs. no replay buffer, no target networks,
    no epsilon -- exploration comes from sampling the stochastic policy.
  - invalid actions are masked at the logit level (set to -1e9 before softmax)
    so they receive exactly zero probability during sampling AND during the
    log-prob computation of the update.
  - advantages computed with gae (generalized advantage estimation).

reward matches dqn exactly for a fair algorithm comparison:
  +1.0 + final_hp / 80 on fight win, -1.0 on loss, 0 otherwise (terminal only).

reference implementation style follows cleanrl's single-file ppo.
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# ---- hyperparameters ----
GAMMA = 0.99          # discount factor (matches dqn)
GAE_LAMBDA = 0.95     # gae smoothing
CLIP_EPS = 0.2        # ppo clip range
LR = 3e-4             # learning rate (matches dqn)
UPDATE_EPOCHS = 4     # passes over each rollout batch
MINIBATCH_SIZE = 128  # minibatch size within an update (matches dqn batch size)
ENT_COEF = 0.01       # entropy bonus -- keeps the policy from collapsing early
VF_COEF = 0.5         # value loss weight
MAX_GRAD_NORM = 0.5   # gradient clipping

N_ACTIONS = 11        # 10 card slots + end turn
N_ENEMIES = 5
N_OBSERVATIONS = 266
TARGET_INPUT_DIMS = N_OBSERVATIONS + N_ACTIONS  # 277

MASK_FILL = -1e9      # logit value for invalid actions


class ActorCritic(nn.Module):
    '''shared 256x256 trunk (same capacity as the dqn mlp) with policy and value heads'''

    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def masked_dist(self, x, mask):
        '''categorical distribution with invalid actions masked to zero probability'''
        logits, value = self.forward(x)
        logits = logits.masked_fill(mask == 0, MASK_FILL)
        return torch.distributions.Categorical(logits=logits), value


card_net = ActorCritic(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net = ActorCritic(TARGET_INPUT_DIMS, N_ENEMIES).to(device)

optimizer = optim.AdamW(
    list(card_net.parameters()) + list(target_net.parameters()),
    lr=LR, amsgrad=True
)


def build_target_state(state: torch.Tensor, card_action: int) -> torch.Tensor:
    '''concatenate state with one-hot card encoding -- identical to dqn.build_target_state'''
    card_one_hot = torch.zeros(N_ACTIONS, device=device)
    card_one_hot[card_action] = 1.0
    return torch.cat([state, card_one_hot.unsqueeze(0)], dim=1)  # (1, 277)


@torch.no_grad()
def select_action(state: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
    '''
    sample a card action from the masked policy.
    returns (action_idx, log_prob, value) -- log_prob and value are stored
    in the rollout buffer for the ppo update.
    deterministic=True (eval mode) takes the argmax instead of sampling.
    '''
    dist, value = card_net.masked_dist(state, mask.unsqueeze(0))
    if deterministic:
        action = dist.probs.argmax(dim=-1)
    else:
        action = dist.sample()
    return action.item(), dist.log_prob(action).item(), value.item()


@torch.no_grad()
def select_target(target_state: torch.Tensor, enemy_mask: torch.Tensor,
                  deterministic: bool = False):
    '''sample an enemy target from the masked target policy'''
    dist, value = target_net.masked_dist(target_state, enemy_mask.unsqueeze(0))
    if deterministic:
        action = dist.probs.argmax(dim=-1)
    else:
        action = dist.sample()
    return action.item(), dist.log_prob(action).item(), value.item()


class RolloutBuffer:
    '''
    on-policy storage for one update cycle (several full episodes).

    card steps and target steps are stored separately because they belong to
    different networks with different input dims. each target step remembers
    which card step it co-occurred with (its index within the current fight)
    so it inherits that step's return and advantage after gae -- the target
    choice and the card choice share credit for the same outcome.

    fights are treated as trajectory boundaries (done=True at fight end),
    mirroring dqn where each fight terminates an episode in the replay buffer.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        # card trajectory (flat across all fights in the batch)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.masks = []
        self.rewards = []
        self.dones = []
        # computed after each fight
        self.advantages = []
        self.returns = []

        # target trajectory
        self.t_states = []
        self.t_actions = []
        self.t_log_probs = []
        self.t_masks = []
        self.t_card_indices = []   # global index of the associated card step
        self.t_advantages = []
        self.t_returns = []

        # index where the current (unfinished) fight begins
        self._fight_start = 0

    def add_card_step(self, state, action, log_prob, value, mask, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask)
        self.rewards.append(reward)
        self.dones.append(done)

    def add_target_step(self, t_state, action, log_prob, mask):
        self.t_states.append(t_state)
        self.t_actions.append(action)
        self.t_log_probs.append(log_prob)
        self.t_masks.append(mask)
        # associate with the card step just added (the most recent one)
        self.t_card_indices.append(len(self.states) - 1)

    def finish_fight(self):
        '''
        compute gae advantages and returns for the fight that just ended.
        called after every fight (win or loss). terminal reward sits on the
        last step; bootstrap value is 0 because the fight is done.
        '''
        start, end = self._fight_start, len(self.states)
        if start == end:
            return

        rewards = self.rewards[start:end]
        values = self.values[start:end]

        advantages = [0.0] * (end - start)
        gae = 0.0
        for t in reversed(range(end - start)):
            next_value = values[t + 1] if t < (end - start - 1) else 0.0
            # within a fight no step is terminal except the last one
            next_non_terminal = 0.0 if t == (end - start - 1) else 1.0
            delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae

        returns = [a + v for a, v in zip(advantages, values)]
        self.advantages.extend(advantages)
        self.returns.extend(returns)

        # target steps inherit return/advantage from their card step
        for i, card_idx in enumerate(self.t_card_indices):
            if start <= card_idx < end and len(self.t_advantages) <= i:
                self.t_advantages.append(advantages[card_idx - start])
                self.t_returns.append(returns[card_idx - start])

        self._fight_start = end

    def __len__(self):
        return len(self.states)


buffer = RolloutBuffer()


def _ppo_loss(net, states, actions, old_log_probs, masks, advantages, returns):
    dist, values = net.masked_dist(states, masks)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.functional.mse_loss(values, returns)

    return policy_loss + VF_COEF * value_loss - ENT_COEF * entropy


def update():
    '''
    run the ppo update over everything currently in the buffer, then clear it.
    returns a small dict of diagnostics for logging.
    '''
    if len(buffer) == 0:
        return {}

    states = torch.cat(buffer.states).to(device)
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    masks = torch.stack(buffer.masks).to(device)
    advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(buffer.returns, dtype=torch.float32, device=device)

    # advantage normalization -- standard ppo stabilizer
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    has_targets = len(buffer.t_states) > 0
    if has_targets:
        t_states = torch.cat(buffer.t_states).to(device)
        t_actions = torch.tensor(buffer.t_actions, dtype=torch.long, device=device)
        t_old_log_probs = torch.tensor(buffer.t_log_probs, dtype=torch.float32, device=device)
        t_masks = torch.stack(buffer.t_masks).to(device)
        t_advantages = torch.tensor(buffer.t_advantages, dtype=torch.float32, device=device)
        t_returns = torch.tensor(buffer.t_returns, dtype=torch.float32, device=device)
        t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

    n = len(buffer)
    indices = np.arange(n)
    total_loss = 0.0
    n_updates = 0

    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            mb = indices[start:start + MINIBATCH_SIZE]
            mb_t = torch.tensor(mb, dtype=torch.long, device=device)

            loss = _ppo_loss(card_net, states[mb_t], actions[mb_t],
                             old_log_probs[mb_t], masks[mb_t],
                             advantages[mb_t], returns[mb_t])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(card_net.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_updates += 1

        # target network update -- whole target batch at once (usually small)
        if has_targets:
            loss = _ppo_loss(target_net, t_states, t_actions,
                             t_old_log_probs, t_masks, t_advantages, t_returns)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(target_net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    diagnostics = {
        'n_card_steps': n,
        'n_target_steps': len(buffer.t_states),
        'avg_loss': total_loss / max(n_updates, 1),
    }
    buffer.reset()
    return diagnostics


def save_model(path: str = 'ppo_model.pt'):
    torch.save({
        'card_net': card_net.state_dict(),
        'target_net': target_net.state_dict(),
    }, path)
    print(f'model saved to {path}')


def load_model(path: str = 'ppo_model.pt'):
    if not os.path.exists(path):
        print(f'no model found at {path} -- using random weights')
        return False
    checkpoint = torch.load(path, map_location=device)
    card_net.load_state_dict(checkpoint['card_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    print(f'model loaded from {path}')
    return True