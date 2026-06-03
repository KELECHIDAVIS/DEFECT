#this agent always uses highest value attack on the enemy on the lowest health enemy 

from __future__ import annotations
import random
from ggpa.ggpa import GGPA
from action.action import EndAgentTurn, PlayCard
from config import CardType
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState
    from agent import Agent
    from card import Card
    from agent import Agent
    from card import Card

class AlwaysAttackBot(GGPA):
    def __init__(self):
        super().__init__("AlwaysAttackBot")

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> EndAgentTurn|PlayCard:
        options = self.get_choose_card_options(game_state, battle_state)
        
        # check all attack cards, select one with highest damage
        best_attack = None
        for option in options: 
            # skip if it's the end turn option
            if type(option) is EndAgentTurn: 
                continue
            
            current_card = battle_state.hand[option.get_card_index()]
            
            # skip if not an attack card
            if current_card.card_type != CardType.ATTACK:
                continue
            
            # if no attack found yet, set this as the first candidate
            if best_attack is None:
                best_attack = option
                continue
            
            # compare damage values, update best if current card hits harder
            best_attack_card = battle_state.hand[best_attack.card_index]
            if best_attack_card.actions[0].values[0].get() < current_card.actions[0].values[0].get():
                best_attack = option
        
        # if no attack cards available, end turn
        if best_attack is None:
            return next(o for o in options if type(o) is EndAgentTurn)
        
        return best_attack
    
    
    def choose_agent_target(self, battle_state: BattleState, list_name: str, 
                        agent_list: list[Agent]) -> Agent:
        # focus fire -- lowest hp enemy dies first, removing a damage source
        return min(agent_list, key=lambda e: e.health)
    
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return random.choice(card_list)