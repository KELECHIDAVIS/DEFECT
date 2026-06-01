from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import AcidSlimeSmall, SpikeSlimeSmall, JawWorm
from card import CardGen, CardRepo
import time
from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from ggpa.dqn_agent import DQNBot
from ggpa.always_attack import AlwaysAttackBot
from ggpa.rl_algos.training_logger import TrainingLogger

import torch


def run_dqn_episode(agent: DQNBot, game_state: GameState, battle_state: BattleState):
    '''
    run one full episode of dqn training using tick_player for per-step control.
    tick_player takes a pre-selected action directly -- it does NOT call choose_card
    internally, so our training loop controls the full step cycle.

    per-step reward breakdown:
      - +damage dealt to enemies / 30.0 (normalized)
      - -hp lost by player / 80.0 (normalized)
      - +1.0 bonus for winning the battle
      - -1.0 penalty for losing the battle
    '''
    from ggpa.rl_algos import dqn

    battle_state.initiate_log()
    step = 0
    episode_reward = 0.0
    prev_hp = battle_state.player.health

    while not battle_state.ended():

        # record hp before action for reward computation
        prev_player_hp = battle_state.player.health
        prev_enemy_hp = sum(e.health for e in battle_state.enemies if not e.is_dead())

        # get state vector and action mask for this step
        state = agent.get_state_vector(game_state, battle_state)
        mask = agent.get_action_mask(game_state, battle_state)

        # choose action -- also stores current_action index on the agent
        action = agent.choose_card(game_state, battle_state)
        action_tensor = torch.tensor([[agent.current_action]], device=dqn.device)

        # execute action -- if EndAgentTurn, also runs enemy turn and draws next hand
        battle_state.tick_player(action)

        # compute step reward from hp changes
        new_player_hp = battle_state.player.health
        new_enemy_hp = sum(e.health for e in battle_state.enemies if not e.is_dead())

        hp_lost = prev_player_hp - new_player_hp
        damage_dealt = prev_enemy_hp - new_enemy_hp

        step_reward = (damage_dealt / 30.0) - (hp_lost / 80.0)

        # terminal reward -- add win/loss bonus and set next_state to None
        done = battle_state.ended()
        if done:
            result = battle_state.get_end_result()
            step_reward += float(result)  # +1.0 win, -1.0 loss
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

        # soft update target network weights toward policy network
        # theta_target = TAU * theta_policy + (1 - TAU) * theta_target
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
    agent = DQNBot()
    # agent = AlwaysAttackBot()
    # agent = BacktrackBot(4, False)
    # agent = RandomBot()

    # more episodes when gpu is available since training is faster
    num_episodes = 50
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 3000

    # training logger -- only used for dqn, saves to training_log.json
    logger = TrainingLogger(log_path='training_log.json', save_every=10) if isinstance(agent, DQNBot) else None

    for i_episode in range(num_episodes):
        game_state = GameState(Character.IRON_CLAD, agent, 0)
        game_state.set_deck(*CardRepo.get_scenario_0()[1])
        start = time.time()

        # dqn uses tick_player loop for per-step training control
        # all other agents use run() which handles the full episode automatically
        if isinstance(agent, DQNBot):
            battle_state = BattleState(game_state, JawWorm(game_state), verbose=Verbose.NO_LOG)
            episode_reward, win, final_hp, steps = run_dqn_episode(agent, game_state, battle_state)

            # log training metrics
            logger.log_episode(episode_reward, win, final_hp, steps)

            # print summary every 10 episodes
            if (i_episode + 1) % 10 == 0:
                logger.print_summary()

            end = time.time()
            print(f"episode {i_episode + 1}/{num_episodes}: "
                  f"{'WIN' if win else 'LOSE'} | "
                  f"hp: {final_hp}/80 | "
                  f"reward: {episode_reward:.3f} | "
                  f"steps: {steps} | "
                  f"{end - start:.2f}s")
            
            # # check convergence every 200 episodes
            # if (i_episode + 1) % 200 == 0 and len(logger.episode_wins) >= 400:
            #     recent = sum(logger.episode_wins[-200:]) / 200
            #     previous = sum(logger.episode_wins[-400:-200]) / 200
            #     if abs(recent - previous) < 0.02:
            #         print(f'converged at episode {i_episode + 1}')
            #         break
        else:
            battle_state = BattleState(game_state, JawWorm(game_state), verbose=Verbose.LOG)
            battle_state.run()
            end = time.time()
            print(f"run ended in {end - start:.2f} seconds")

    # save final training log
    if logger is not None:
        logger.save()
        print(f'\ntraining complete. run plot_results.py to visualize.')
        print(f'to evaluate all agents: python3.11 evaluate.py')


if __name__ == '__main__':
    main()