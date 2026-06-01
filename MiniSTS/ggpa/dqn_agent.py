from __future__ import annotations
import random
from ggpa.ggpa import GGPA
from action.action import EndAgentTurn, PlayCard
from action.agent_targeted_action import DealAttackDamage, AddBlock, ApplyStatus
from config import CardType
from status_effecs import StatusEffectRepo
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card

from ggpa.rl_algos import dqn
import torch

# constants matching state vector reference doc
MAX_HAND_SIZE = 10      # max cards in hand at once
MAX_ENEMIES = 5         # max enemies in a single encounter
CARD_DIMS = 18          # dimensions per card slot
ENEMY_DIMS = 15         # dimensions per enemy slot
PLAYER_DIMS = 11        # dimensions for player state
N_ACTIONS = 11          # 10 card slots + end turn (index 10)

class DQNBot(GGPA):
    def __init__(self):
        super().__init__("DQNBot")
        # track current state and mask for use across choose_card and choose_agent_target
        self.current_state = None
        self.current_mask = None
        self.current_action = None

    def _encode_card(self, card, is_playable: bool) -> list:
        '''encode a single card into an 18-dim vector based on state vector reference'''
        vec = [0.0] * CARD_DIMS

        # index 0: is_present -- this is a real card not a padding slot
        vec[0] = 1.0

        # index 1: playable -- does the agent have enough mana to play this
        vec[1] = 1.0 if is_playable else 0.0

        # index 2: cost (raw 0-3, unnormalized -- same scale as energy)
        vec[2] = card.mana_cost.peek()

        # indices 3-7: one-hot card type encoding
        type_map = {
            CardType.ATTACK: 3,
            CardType.SKILL:  4,
            CardType.POWER:  5,
            CardType.STATUS: 6,
            CardType.CURSE:  7,
        }
        if card.card_type in type_map:
            vec[type_map[card.card_type]] = 1.0

        # scan card actions to extract damage, block, and status effects applied
        damage_val = 0.0
        hits_val = 1
        block_val = 0.0
        applies_vulnerable = 0
        applies_weak = 0
        applies_strength = 0
        draws_cards = 0
        exhausts = 0
        targets_single = 0
        targets_all = 0

        for action in card.actions:
            # check action type by class name since MiniSTS uses class-based actions
            action_class = type(action).__name__

            if action_class == 'AgentTargetedAction':
                # unwrap the targeted action to get the inner action
                inner = action.targeted
                inner_class = type(inner).__name__
                target_class = type(action.target).__name__

                # determine targeting type from target class name
                if 'All' in target_class:
                    targets_all = 1
                elif 'Choose' in target_class or 'Random' in target_class:
                    targets_single = 1
                elif 'Self' in target_class:
                    pass  # self-targeting, neither flag set

                # extract values from inner action
                if inner_class == 'DealAttackDamage':
                    damage_val = inner.val.peek()
                    hits_val = inner.times.peek() if hasattr(inner, 'times') else 1
                elif inner_class == 'AddBlock':
                    block_val = inner.val.peek()
                elif inner_class == 'ApplyStatus':
                    se = inner.status_effect
                    if se == StatusEffectRepo.VULNERABLE:
                        applies_vulnerable = int(inner.val.peek())
                    elif se == StatusEffectRepo.WEAK:
                        applies_weak = int(inner.val.peek())
                    elif se == StatusEffectRepo.STRENGTH:
                        applies_strength = int(inner.val.peek())
                elif inner_class == 'AndAgentTargeted':
                    # compound action -- check each sub-action
                    for sub in inner.targeted_set:
                        sub_class = type(sub).__name__
                        if sub_class == 'DealAttackDamage':
                            damage_val = sub.val.peek()
                            hits_val = sub.times.peek() if hasattr(sub, 'times') else 1
                        elif sub_class == 'AddBlock':
                            block_val = sub.val.peek()
                        elif sub_class == 'ApplyStatus':
                            se = sub.status_effect
                            if se == StatusEffectRepo.VULNERABLE:
                                applies_vulnerable = int(sub.val.peek())
                            elif se == StatusEffectRepo.WEAK:
                                applies_weak = int(sub.val.peek())
                            elif se == StatusEffectRepo.STRENGTH:
                                applies_strength = int(sub.val.peek())

            elif action_class == 'DrawCard':
                draws_cards = int(action.val.peek()) if hasattr(action, 'val') else 1

        # check if card exhausts by looking at card actions for Exhaust
        for action in card.actions:
            if type(action).__name__ == 'CardTargetedAction':
                if type(action.targeted).__name__ == 'Exhaust':
                    exhausts = 1

        # index 8: damage normalized by ceiling of 30
        vec[8] = damage_val / 30.0

        # index 9: number of hits (raw, usually 1)
        vec[9] = hits_val

        # index 10: block normalized by ceiling of 20
        vec[10] = block_val / 20.0

        # index 11: targets single enemy
        vec[11] = float(targets_single)

        # index 12: targets all enemies (aoe)
        vec[12] = float(targets_all)

        # index 13: vulnerable stacks applied
        vec[13] = applies_vulnerable

        # index 14: weak stacks applied
        vec[14] = applies_weak

        # index 15: strength modifier applied (can be negative)
        vec[15] = applies_strength

        # index 16: extra cards drawn on play
        vec[16] = draws_cards

        # index 17: card exhausts after playing
        vec[17] = float(exhausts)

        return vec

    def _encode_enemy(self, enemy, game_state, battle_state) -> list:
        '''encode a single enemy into a 15-dim vector based on state vector reference'''
        vec = [0.0] * ENEMY_DIMS

        # index 0: is_present -- this slot has a real enemy
        vec[0] = 1.0

        # index 1: hp normalized (current / max)
        vec[1] = enemy.health / enemy.max_health if enemy.max_health > 0 else 0.0

        # index 2: hp current normalized by ceiling of 100
        vec[2] = enemy.health / 100.0

        # index 3: block normalized by ceiling of 30
        vec[3] = enemy.block / 30.0

        # index 4-6: status effect stacks on enemy
        vec[4] = enemy.status_effect_state.get(StatusEffectRepo.VULNERABLE)
        vec[5] = enemy.status_effect_state.get(StatusEffectRepo.WEAK)
        vec[6] = enemy.status_effect_state.get(StatusEffectRepo.STRENGTH)

        # indices 7-12: multi-hot intent encoding
        # get the enemy's next intended action
        try:
            intention = enemy.get_intention(game_state, battle_state)
            intent_repr = repr(intention).lower()

            # multi-hot: multiple bits can be set simultaneously for combo intents
            vec[7] = 1.0 if 'dealattackdamage' in intent_repr or 'dealdamage' in intent_repr else 0.0  # intent_attack
            vec[8] = 1.0 if 'addblock' in intent_repr else 0.0                                         # intent_block
            vec[9] = 1.0 if 'applystatus' in intent_repr and 'strength' in intent_repr else 0.0        # intent_buff_self
            vec[10] = 1.0 if 'applystatus' in intent_repr and ('vulnerable' in intent_repr or 'weak' in intent_repr) else 0.0  # intent_debuff
            vec[11] = 0.0  # intent_sleep -- no sleep mechanic in current minists
            vec[12] = 0.0  # intent_unknown

            # index 13: attack damage normalized by ceiling of 30
            # extract damage value from intention if attacking
            if vec[7] == 1.0:
                for action in intention.actions if hasattr(intention, 'actions') else []:
                    if hasattr(action, 'targeted') and type(action.targeted).__name__ in ('DealAttackDamage', 'DealDamage'):
                        vec[13] = action.targeted.val.peek() / 30.0
                        break

            # index 14: attack times (hits per attack)
            if vec[7] == 1.0:
                for action in intention.actions if hasattr(intention, 'actions') else []:
                    if hasattr(action, 'targeted') and type(action.targeted).__name__ in ('DealAttackDamage', 'DealDamage'):
                        inner = action.targeted
                        vec[14] = inner.times.peek() if hasattr(inner, 'times') else 1.0
                        break

        except Exception:
            # if intent is unavailable set unknown flag
            vec[12] = 1.0

        return vec

    def get_state_vector(self, game_state: 'GameState', battle_state: 'BattleState') -> torch.Tensor:
        '''
        convert current game state to a 266-dim flat tensor
        structure: player state (11) + hand (10 x 18 = 180) + enemies (5 x 15 = 75)
        see state_vectors_reference.txt for full specification
        '''
        state = []
        player = battle_state.player
 
        # ---- SECTION 1: PLAYER STATE (11 dims) ----
 
        # index 0: current hp normalized by max hp
        state.append(player.health / player.max_health if player.max_health > 0 else 0.0)
 
        # index 1: max hp normalized by ceiling of 100
        state.append(player.max_health / 100.0)
 
        # index 2: current block normalized by ceiling of 30
        state.append(player.block / 30.0)
 
        # index 3: current mana (raw, unnormalized -- same scale as card cost)
        state.append(float(battle_state.mana))
 
        # index 4: max mana per turn from game state
        state.append(float(game_state.max_mana))
 
        # index 5-8: player status effect stacks
        state.append(float(player.status_effect_state.get(StatusEffectRepo.VULNERABLE)))
        state.append(float(player.status_effect_state.get(StatusEffectRepo.WEAK)))
 
        # frail is not in current minists statuseffectrepo -- default to 0 for forward compat
        state.append(0.0)  # frail placeholder
 
        # strength can be negative
        state.append(float(player.status_effect_state.get(StatusEffectRepo.STRENGTH)))
 
        # index 9-10: draw and discard pile ratios
        draw_size = len(battle_state.draw_pile)
        discard_size = len(battle_state.discard_pile)
        total = draw_size + discard_size
        state.append(draw_size / total if total > 0 else 0.0)    # draw ratio
        state.append(discard_size / total if total > 0 else 0.0)  # discard ratio
 
        # ---- SECTION 2: HAND (10 slots x 18 dims = 180 dims) ----
 
        # build set of playable card indices from valid options
        options = self.get_choose_card_options(game_state, battle_state)
        playable_indices = set()
        for option in options:
            if type(option) is not EndAgentTurn:
                playable_indices.add(option.get_card_index())
 
        for i in range(MAX_HAND_SIZE):
            if i < len(battle_state.hand):
                card = battle_state.hand[i]
                card_vec = self._encode_card(card, i in playable_indices)
            else:
                # padding slot -- all zeros
                card_vec = [0.0] * CARD_DIMS
            state.extend(card_vec)
 
        # ---- SECTION 3: ENEMY SLOTS (5 slots x 15 dims = 75 dims) ----
 
        for i in range(MAX_ENEMIES):
            if i < len(battle_state.enemies) and not battle_state.enemies[i].is_dead():
                enemy = battle_state.enemies[i]
                enemy_vec = self._encode_enemy(enemy, game_state, battle_state)
            else:
                # empty slot -- all zeros
                enemy_vec = [0.0] * ENEMY_DIMS
            state.extend(enemy_vec)
 
        # convert to tensor and add batch dimension
        return torch.tensor(state, dtype=torch.float32, device=dqn.device).unsqueeze(0)
 
    def get_action_mask(self, game_state: 'GameState', battle_state: 'BattleState') -> torch.Tensor:
        '''
        build an 11-element binary mask over valid actions
        indices 0-9 correspond to hand slots, index 10 is end turn
        1 = valid action, 0 = invalid (masked out before argmax)
        '''
        mask = torch.zeros(N_ACTIONS, device=dqn.device)
 
        # end turn is always valid
        mask[10] = 1.0
 
        # mark each hand slot as valid if it holds a playable card
        options = self.get_choose_card_options(game_state, battle_state)
        for option in options:
            if type(option) is not EndAgentTurn:
                idx = option.get_card_index()
                if idx < N_ACTIONS - 1:  # guard against out of range
                    mask[idx] = 1.0
 
        return mask
 
    def choose_card(self, game_state: 'GameState', battle_state: 'BattleState') -> EndAgentTurn | PlayCard:
        # build state vector and action mask for this step
        state = self.get_state_vector(game_state, battle_state)
        mask = self.get_action_mask(game_state, battle_state)
 
        # store for use in choose_agent_target (target selection happens after card selection)
        self.current_state = state
        self.current_mask = mask
 
        # select action via epsilon-greedy
        action_idx = dqn.select_action(state, mask).item()
        self.current_action = action_idx
 
        # action index 10 means end turn
        if action_idx == 10:
            return next(o for o in self.get_choose_card_options(game_state, battle_state)
                        if type(o) is EndAgentTurn)
 
        # otherwise find the PlayCard option at the selected hand index
        options = self.get_choose_card_options(game_state, battle_state)
        for option in options:
            if type(option) is not EndAgentTurn and option.get_card_index() == action_idx:
                return option
 
        # fallback -- if selected index has no valid play, end turn
        return next(o for o in options if type(o) is EndAgentTurn)
 
    def choose_agent_target(self, battle_state: 'BattleState', list_name: str, agent_list: list['Agent']) -> 'Agent':
        # for now pick randomly -- target selection will be improved when
        # multi-target autoregressive selection is added in a later phase
        return random.choice(agent_list)
 
    def choose_card_target(self, battle_state: 'BattleState', list_name: str, card_list: list['Card']) -> 'Card':
        return random.choice(card_list)
 