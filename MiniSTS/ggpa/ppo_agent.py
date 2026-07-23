from __future__ import annotations
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent

from action.action import EndAgentTurn, PlayCard
from ggpa.dqn_agent import DQNBot
from ggpa.rl_algos import ppo


class PPOBot(DQNBot):
    '''
    ppo agent. inherits the 266-dim state encoder, action mask builder, and
    target-selection plumbing from DQNBot -- both agents see the exact same
    state representation, so any performance difference is attributable to
    the algorithm, not the encoding.

    overrides:
      choose_card          -- samples from the masked ppo policy instead of
                              epsilon-greedy q-values, and records log_prob +
                              value for the rollout buffer
      choose_agent_target  -- samples from the ppo target policy

    in training mode the training loop reads current_log_prob / current_value
    (and pending_target) after each step to fill the rollout buffer.
    in eval mode actions are deterministic (argmax) and ppo_model.pt is loaded.
    '''

    def __init__(self, eval_mode: bool = False):
        # skip DQNBot.__init__'s dqn model loading -- init GGPA fields manually
        # by calling the grandparent chain through DQNBot with eval_mode=False,
        # then set our own eval flag and load ppo weights instead
        super().__init__(eval_mode=False)
        self.name = "PPOBot"
        self.eval_mode = eval_mode

        # per-step ppo bookkeeping (read by the training loop)
        self.current_log_prob = None
        self.current_value = None

        if eval_mode:
            ppo.load_model('ppo_model.pt')

    def choose_card(self, game_state: 'GameState', battle_state: 'BattleState') -> EndAgentTurn | PlayCard:
        state = self.get_state_vector(game_state, battle_state)
        mask = self.get_action_mask(game_state, battle_state)

        self.current_state = state
        self.current_mask = mask

        action_idx, log_prob, value = ppo.select_action(
            state, mask, deterministic=self.eval_mode)

        self.current_action = action_idx
        self.current_log_prob = log_prob
        self.current_value = value

        # action index 10 means end turn
        options = self.get_choose_card_options(game_state, battle_state)
        if action_idx == 10:
            return next(o for o in options if type(o) is EndAgentTurn)

        for option in options:
            if type(option) is not EndAgentTurn and option.get_card_index() == action_idx:
                return option

        # fallback -- if selected index has no valid play, end turn
        return next(o for o in options if type(o) is EndAgentTurn)

    def choose_agent_target(self, battle_state: 'BattleState', list_name: str,
                            agent_list: list['Agent']) -> 'Agent':
        if not agent_list:
            self.pending_target = None
            return None

        # single enemy -- no decision to make, nothing to store
        if len(agent_list) == 1:
            self.pending_target = None
            return agent_list[0]

        # enemy mask over all 5 slots -- only living enemies in agent_list are valid
        enemy_mask = torch.zeros(ppo.N_ENEMIES, device=ppo.device)
        for slot_idx, enemy in enumerate(battle_state.enemies):
            if enemy in agent_list and not enemy.is_dead():
                enemy_mask[slot_idx] = 1.0

        # 277-dim context: state + one-hot of the card just chosen
        target_state = ppo.build_target_state(self.current_state, self.current_action)

        chosen_slot, log_prob, _value = ppo.select_target(
            target_state, enemy_mask, deterministic=self.eval_mode)

        # store for the training loop -- pushed to the rollout buffer after
        # the reward for this step is known
        self.pending_target = {
            'target_state': target_state,
            'action': chosen_slot,
            'log_prob': log_prob,
            'enemy_mask': enemy_mask,
        }

        if chosen_slot < len(battle_state.enemies) and not battle_state.enemies[chosen_slot].is_dead():
            chosen_enemy = battle_state.enemies[chosen_slot]
            if chosen_enemy in agent_list:
                return chosen_enemy

        return agent_list[0]