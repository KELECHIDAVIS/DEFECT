from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import SwampLeech,  JawWorm 
from card import CardGen, CardRepo
import time
from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from ggpa.dqn_agent import DQNBot
from ggpa.human_input import HumanInput
from ggpa.always_attack import AlwaysAttackBot
from ggpa.rl_algos.training_logger import TrainingLogger

import torch


def run_dqn_episode(agent: DQNBot, game_state: GameState, battle_state: BattleState):
    '''
    run one full episode of dqn training using tick_player for per-step control.
    tick_player takes a pre-selected action directly -- it does NOT call choose_card
    internally, so our training loop controls the full step cycle.

    per-step reward breakdown:
      - current hp / 80.0 (normalized) #incentivize maximizing hp 
      - +1.0 bonus for winning the battle
      - -1.0 penalty for losing the battle
    '''
    from ggpa.rl_algos import dqn

    battle_state.initiate_log()
    step = 0
    episode_reward = 0.0
    prev_hp = battle_state.player.health

    while not battle_state.ended():

        # get state vector and action mask for this step
        state = agent.get_state_vector(game_state, battle_state)
        mask = agent.get_action_mask(game_state, battle_state)

        # choose action -- also stores current_action index on the agent
        action = agent.choose_card(game_state, battle_state)
        action_tensor = torch.tensor([[agent.current_action]], device=dqn.device)

        # execute action -- if EndAgentTurn, also runs enemy turn and draws next hand
        battle_state.tick_player(action)

        step_reward = 0.0
        done = battle_state.ended()

        if done:
            result = battle_state.get_end_result()
            final_hp = battle_state.player.health
            # win: +1.0 bonus + normalized hp remaining encourages winning with hp left
            # loss: -1.0 flat penalty
            step_reward = (1.0 + final_hp / 80.0) if result == 1 else -1.0
            next_state = None
            next_mask = torch.zeros(11, device=dqn.device)
        else:
            next_state = agent.get_state_vector(game_state, battle_state)
            next_mask = agent.get_action_mask(game_state, battle_state)

        episode_reward += step_reward

        # store transition in replay buffer
        reward_tensor = torch.tensor([step_reward], dtype=torch.float32, device=dqn.device)
        dqn.memory.push(state, action_tensor, next_state, reward_tensor, next_mask)

        # optimize policy network on a random batch from replay buffer
        dqn.optimize_model()

        # soft update target network
        target_state_dict = dqn.target_net.state_dict()
        policy_state_dict = dqn.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = (policy_state_dict[key] * dqn.TAU +
                                    target_state_dict[key] * (1 - dqn.TAU))
        dqn.target_net.load_state_dict(target_state_dict)

        step += 1

    result = battle_state.get_end_result()
    win = result == 1
    final_hp = battle_state.player.health

    return episode_reward, win, final_hp, step


def main():
    #agent = DQNBot()
    # agent = AlwaysAttackBot()
    # agent = BacktrackBot(4, False)
    # agent = RandomBot()
    agent = HumanInput(True) 
    # more episodes when gpu is available since training is faster
    num_episodes = 50
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 3000

    # training logger -- only used for dqn, saves to training_log.json
    logger = TrainingLogger(log_path='training_log.json', save_every=10) if isinstance(agent, DQNBot) else None
    # first battle is baseline jawworm 
    # then high health enemy where you should wait to attack when the time is right 
    # then two enemies that shows target prioritization 
    #then miniboss cultist where you have to kill as quick as possible 
    battles = [JawWorm, SwampLeech]
    for i_episode in range(num_episodes):
        # create game state ONCE per episode -- player HP carries over between fights
        game_state = GameState(Character.IRON_CLAD, agent, 0)
        game_state.set_deck(*CardRepo.get_scenario_0()[1])
        
        episode_reward = 0.0
        episode_won_all = True

        for battle in battles:
            start = time.time()

            if isinstance(agent, DQNBot):
                battle_state = BattleState(game_state, battle(game_state), verbose=Verbose.NO_LOG)
                battle_reward, win, final_hp, steps = run_dqn_episode(agent, game_state, battle_state)
                episode_reward += battle_reward

                if not win:
                    episode_won_all = False
                    break  # player died, skip remaining fights

                # burning blood -- heal 6 hp after each won fight, capped at max
                game_state.player.health = min(game_state.player.health + 6, game_state.player.max_health)

            else:
                battle_state = BattleState(game_state, battle(game_state), verbose=Verbose.LOG)
                battle_state.run()
                end = time.time()
                print(f"run ended in {end - start:.2f} seconds")

                if not battle_state.get_end_result() == 1:
                    break  # player died, skip remaining fights

                game_state.player.health = min(game_state.player.health + 6, game_state.player.max_health)

    
    
    # save final training log
    if logger is not None:
        logger.save()
        print(f'\ntraining complete. run plot_results.py to visualize.')
        print(f'to evaluate all agents: python3.11 evaluate.py')
        
    #save dqn model 
    if isinstance(agent, DQNBot):
        from ggpa.rl_algos import dqn
        dqn.save_model('dqn_model.pt')


if __name__ == '__main__':
    main()