
''' for this section this guide was closely followed: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html'''
import math 
import random 
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# use gpu 
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Device being used:", device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_mask'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[firstcard0exp, secondcard0exp...]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 10000
TAU = 0.005
LR = 3e-4
N_ACTIONS =11 

# Get number of actions from gym action space
n_actions = N_ACTIONS # max 10 card choices, then end turn 
# Get the number of state observations

n_observations = 266 # length of our flattened feature vector 

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)


steps_done = 0

#epsilon greedy action selection 
def select_action(state, mask): # takes in state and mask for valid actions 
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            # mask invalid actions so they can never be selected
            q_values[0][mask == 0] = float('-inf')
            return q_values.max(1).indices.view(1, 1)
    else:
        valid_actions = [i for i in range(11) if mask[i] == 1]
        return torch.tensor([[random.choice(valid_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values = target_net(non_final_next_states)
        # apply next state masks -- invalid actions set to -inf
        next_masks = torch.stack([m for m, s in zip(batch.next_mask, batch.next_state) if s is not None])
        next_q_values[next_masks == 0] = float('-inf')
        next_state_values[non_final_mask] = next_q_values.max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    

# target selection network -- input is state + card one-hot context
N_ENEMIES = 5
TARGET_INPUT_DIMS = n_observations + N_ACTIONS  # 266 + 11 = 277

target_policy_net = DQN(TARGET_INPUT_DIMS, N_ENEMIES).to(device)
target_target_net = DQN(TARGET_INPUT_DIMS, N_ENEMIES).to(device)
target_target_net.load_state_dict(target_policy_net.state_dict())
target_optimizer = optim.AdamW(target_policy_net.parameters(), lr=LR, amsgrad=True)
target_memory_buffer = ReplayMemory(50000)

def build_target_state(state: torch.Tensor, card_action: int) -> torch.Tensor:
    '''concatenate state with one-hot card encoding to form target network input'''
    card_one_hot = torch.zeros(N_ACTIONS, device=device)
    card_one_hot[card_action] = 1.0
    return torch.cat([state, card_one_hot.unsqueeze(0)], dim=1)  # (1, 277)

def select_target(target_state: torch.Tensor, enemy_mask: torch.Tensor) -> torch.Tensor:
    '''epsilon-greedy target selection -- no separate epsilon, shares card epsilon'''
    with torch.no_grad():
        q_values = target_policy_net(target_state)
        q_values[0][enemy_mask == 0] = float('-inf')
        return q_values.max(1).indices.view(1, 1)

def optimize_target_model():
    if len(target_memory_buffer) < BATCH_SIZE:
        return
    transitions = target_memory_buffer.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = target_policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q = target_target_net(non_final_next_states)
        next_masks = torch.stack([m for m, s in zip(batch.next_mask, batch.next_state)
                                  if s is not None])
        next_q[next_masks == 0] = float('-inf')
        next_state_values[non_final_mask] = next_q.max(1).values

    expected = (next_state_values * GAMMA) + reward_batch
    loss = nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))

    target_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(target_policy_net.parameters(), 100)
    target_optimizer.step()

# add to the bottom of dqn.py
def save_model(path: str = 'dqn_model.pt'):
    torch.save({
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'target_policy_net': target_policy_net.state_dict(),
        'target_target_net': target_target_net.state_dict(),
        'steps_done': steps_done,
    }, path)
    print(f'model saved to {path}')

def load_model(path: str = 'dqn_model.pt'):
    global steps_done
    if not os.path.exists(path):
        print(f'no model found at {path} -- using random weights')
        return False
    checkpoint = torch.load(path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    # target networks may not exist in older checkpoints
    if 'target_policy_net' in checkpoint:
        target_policy_net.load_state_dict(checkpoint['target_policy_net'])
        target_target_net.load_state_dict(checkpoint['target_target_net'])
    steps_done = checkpoint['steps_done']
    print(f'model loaded from {path} (steps_done={steps_done})')
    return True
