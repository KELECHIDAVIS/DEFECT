from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import SwampLeech, JawWorm, GoblinFiend, GoblinWizard
from card import CardGen, CardRepo
import time
from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from ggpa.dqn_agent import DQNBot
from ggpa.human_input import HumanInput
from ggpa.always_attack import AlwaysAttackBot
from ggpa.rl_algos.training_logger import TrainingLogger

import torch
from tqdm import tqdm


def run_dqn_episode(agent: DQNBot, game_state: GameState, battle_state: BattleState):
    '''
    run one full episode of dqn training using tick_player for per-step control.
    tick_player takes a pre-selected action directly -- it does NOT call choose_card
    internally, so our training loop controls the full step cycle.

    two networks are trained simultaneously:
      - card network: 266-dim state -> Q-values over 11 card slots
      - target network: 277-dim state (state + card one-hot) -> Q-values over 5 enemy slots
        (only trains when a multi-enemy fight requires target selection)

    reward:
      - +1.0 bonus + normalized hp remaining on win
      - -1.0 flat penalty on loss
    '''
    from ggpa.rl_algos import dqn

    battle_state.initiate_log()
    step = 0
    episode_reward = 0.0

    while not battle_state.ended():

        # get state vector and action mask for this step
        state = agent.get_state_vector(game_state, battle_state)
        mask = agent.get_action_mask(game_state, battle_state)

        # choose card -- also stores current_action and current_state on agent
        action = agent.choose_card(game_state, battle_state)
        action_tensor = torch.tensor([[agent.current_action]], device=dqn.device)

        # execute action -- if EndAgentTurn, also runs enemy turn and draws next hand
        # choose_agent_target is called internally by MiniSTS if the card needs a target
        # that call sets agent.pending_target if multi-enemy target selection occurred
        battle_state.tick_player(action)

        step_reward = 0.0
        done = battle_state.ended()

        if done:
            result = battle_state.get_end_result()
            final_hp = battle_state.player.health
            step_reward = (1.0 + final_hp / 80.0) if result == 1 else -1.0
            next_state = None
            next_mask = torch.zeros(dqn.N_ACTIONS, device=dqn.device)
        else:
            next_state = agent.get_state_vector(game_state, battle_state)
            next_mask = agent.get_action_mask(game_state, battle_state)

        episode_reward += step_reward
        reward_tensor = torch.tensor([step_reward], dtype=torch.float32, device=dqn.device)

        # ---- card network update ----
        dqn.memory.push(state, action_tensor, next_state, reward_tensor, next_mask)
        dqn.optimize_model()

        # soft update card target network
        target_sd = dqn.target_net.state_dict()
        policy_sd = dqn.policy_net.state_dict()
        for key in policy_sd:
            target_sd[key] = policy_sd[key] * dqn.TAU + target_sd[key] * (1 - dqn.TAU)
        dqn.target_net.load_state_dict(target_sd)

        # ---- target selection network update (only when multi-enemy target was chosen) ----
        if agent.pending_target is not None:
            pt = agent.pending_target

            if done:
                next_target_state = None
                next_enemy_mask = torch.zeros(dqn.N_ENEMIES, device=dqn.device)
            else:
                # next target state: updated state + same card one-hot context
                next_target_state = dqn.build_target_state(next_state, agent.current_action)
                next_enemy_mask = torch.zeros(dqn.N_ENEMIES, device=dqn.device)
                for slot_idx, enemy in enumerate(battle_state.enemies):
                    if not enemy.is_dead():
                        next_enemy_mask[slot_idx] = 1.0

            dqn.target_memory_buffer.push(
                pt['target_state'],
                pt['action'],
                next_target_state,
                reward_tensor,
                next_enemy_mask
            )
            dqn.optimize_target_model()

            # soft update target selection networks
            tgt_sd = dqn.target_target_net.state_dict()
            pol_sd = dqn.target_policy_net.state_dict()
            for key in pol_sd:
                tgt_sd[key] = pol_sd[key] * dqn.TAU + tgt_sd[key] * (1 - dqn.TAU)
            dqn.target_target_net.load_state_dict(tgt_sd)

            agent.pending_target = None  # clear after storing

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
    # agent = HumanInput(True)

    num_episodes = 50
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 3000

    # training logger -- only used for dqn
    logger = TrainingLogger(log_path='training_log.json', save_every=10) if isinstance(agent, DQNBot) else None

    # benchmark fight sequence
    # fight 1: jawworm (baseline)
    # fight 2: swamp leech (high hp, patience test)
    # fight 3: two enemies (target selection) 
    # fight 4: cultist with ritual (urgency test) -- add when ready
    # main.py and evaluate.py
    battles = [JawWorm, SwampLeech, (GoblinFiend, GoblinWizard)]
    pbar = tqdm(range(num_episodes), desc="Training", unit="ep")
    for i_episode in pbar:
        # game state created once per episode -- hp carries over between fights
        game_state = GameState(Character.IRON_CLAD, agent, 0)
        game_state.set_deck(*CardRepo.get_scenario_0()[1])

        episode_reward = 0.0
        episode_won_all = True
        final_hp = game_state.player.health
        total_steps = 0

        for battle in battles:
            # when creating battle state, unpack tuple if multi-enemy
            if isinstance(battle, tuple):
                enemies = [e(game_state) for e in battle]
            else:
                enemies = [battle(game_state)]

            if isinstance(agent, DQNBot):
                battle_state = BattleState(game_state, *enemies, verbose=Verbose.NO_LOG)
                battle_reward, win, final_hp, steps = run_dqn_episode(agent, game_state, battle_state)
                episode_reward += battle_reward
                total_steps += steps

                if not win:
                    episode_won_all = False
                    break

                # burning blood -- 6 hp heal after each won fight
                game_state.player.health = min(
                    game_state.player.health + 6,
                    game_state.player.max_health
                )

            else:
                start = time.time()
                battle_state = BattleState(game_state, *enemies, verbose=Verbose.LOG)
                battle_state.run()
                print(f"run ended in {time.time() - start:.2f} seconds")

                if not battle_state.get_end_result() == 1:
                    break

                game_state.player.health = min(
                    game_state.player.health + 6,
                    game_state.player.max_health
                )

        # log and update progress bar
        if isinstance(agent, DQNBot):
            logger.log_episode(episode_reward, episode_won_all, final_hp, total_steps)

            recent_wins = logger.episode_wins[-20:] if len(logger.episode_wins) >= 20 else logger.episode_wins
            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
            pbar.set_postfix({
                'win%': f'{win_rate:.0%}',
                'hp': final_hp,
                'reward': f'{episode_reward:.2f}',
            })

    # save training log and model
    if logger is not None:
        logger.save()
        print(f'\ntraining complete. run plot_results.py to visualize.')
        print(f'to evaluate all agents: python3.11 evaluate.py')

    if isinstance(agent, DQNBot):
        from ggpa.rl_algos import dqn
        dqn.save_model('dqn_model.pt')


if __name__ == '__main__':
    main()