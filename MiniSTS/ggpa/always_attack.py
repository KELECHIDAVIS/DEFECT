#this agent always uses highest value attack on the lowest health enemy 

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
        best_attack = options[0] # could be end turn option
        for option in options: 
            #skip if it's the end turn option
            if type(option) is EndAgentTurn: 
                continue
            
            current_card = battle_state.hand[option.get_card_index()]
            
            if current_card != CardType.ATTACK:
                continue # skip if not attack 
            
            #if current attack has higher value then best change 
            best_attack_card = battle_state.hand[best_attack.card_index]
            
            #some cards have multiple actions (like bash: 8 damage and 2 vulnerable), so just get first action
            if best_attack_card.actions[0].values[0].get() < current_card.actions[0].values[0].get():
                best_attack = option # set option 
               
        
        return best_attack
    
    # attack lowest health enemy 
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return random.choice(agent_list)
    
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return random.choice(card_list)