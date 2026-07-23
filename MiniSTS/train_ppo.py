'''
train_ppo.py -- ppo training loop for d.e.f.e.c.t.

parallels main.py's dqn training exactly:
  - same four-fight benchmark (jawworm -> swampleech -> goblins -> cultist)
  - same terminal reward (+1 + hp/80 on fight win, -1 on loss)
  - same burning blood heal (+6 hp between fights)
  - same state encoding (inherited from DQNBot)

the difference is the update pattern: ppo is on-policy, so it collects
EPISODES_PER_UPDATE full episodes of experience, then runs the clipped
surrogate update over that batch, then discards it and collects fresh
experience with the updated policy.

usage:
    python3.11 train_ppo.py                    # full run (3000 episodes on gpu)
    python3.11 train_ppo.py --episodes 100     # short run for testing
'''
import argparse
import torch
from tqdm import tqdm

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import JawWorm, SwampLeech, GoblinFiend, GoblinWizard, Cultist
from card import CardRepo

from ggpa.ppo_agent import PPOBot
from ggpa.rl_algos import ppo
from ggpa.rl_algos.training_logger import TrainingLogger

EPISODES_PER_UPDATE = 8   # full 4-fight episodes collected per ppo update


def run_ppo_fight(agent: PPOBot, game_state: GameState, battle_state: BattleState):
    '''
    run one fight, storing every step in the ppo rollout buffer.
    mirrors run_dqn_episode in main.py -- tick_player gives per-step control,
    choose_agent_target is called internally by minists when a card needs a
    target and sets agent.pending_target.
    '''
    battle_state.initiate_log()
    step = 0
    fight_reward = 0.0

    while not battle_state.ended():
        # choose_card stores current_state/mask/action/log_prob/value on the agent
        action = agent.choose_card(game_state, battle_state)

        state = agent.current_state
        mask = agent.current_mask

        battle_state.tick_player(action)

        done = battle_state.ended()
        step_reward = 0.0
        if done:
            result = battle_state.get_end_result()
            final_hp = battle_state.player.health
            step_reward = (1.0 + final_hp / 80.0) if result == 1 else -1.0

        fight_reward += step_reward

        ppo.buffer.add_card_step(
            state=state,
            action=agent.current_action,
            log_prob=agent.current_log_prob,
            value=agent.current_value,
            mask=mask,
            reward=step_reward,
            done=done,
        )

        # target decision made during this step (multi-enemy fights only)
        if agent.pending_target is not None:
            pt = agent.pending_target
            ppo.buffer.add_target_step(
                t_state=pt['target_state'],
                action=pt['action'],
                log_prob=pt['log_prob'],
                mask=pt['enemy_mask'],
            )
            agent.pending_target = None

        step += 1

    # fight over -- compute gae for this trajectory segment
    ppo.buffer.finish_fight()

    result = battle_state.get_end_result()
    return fight_reward, result == 1, battle_state.player.health, step


def main():
    parser = argparse.ArgumentParser(description='d.e.f.e.c.t. ppo training')
    parser.add_argument('--episodes', type=int, default=None,
                        help='training episodes (default: 3000 on gpu, 50 on cpu)')
    parser.add_argument('--output', type=str, default='ppo_model.pt',
                        help='model save path (default: ppo_model.pt)')
    parser.add_argument('--log', type=str, default='training_log_ppo.json',
                        help='training log path (default: training_log_ppo.json)')
    args = parser.parse_args()

    num_episodes = args.episodes
    if num_episodes is None:
        num_episodes = 3000 if (torch.cuda.is_available() or
                                torch.backends.mps.is_available()) else 50

    agent = PPOBot(eval_mode=False)
    logger = TrainingLogger(log_path=args.log, save_every=10)

    battles = [JawWorm, SwampLeech, (GoblinFiend, GoblinWizard), Cultist]

    pbar = tqdm(range(num_episodes), desc="Training PPO", unit="ep")
    for i_episode in pbar:
        game_state = GameState(Character.IRON_CLAD, agent, 0)
        game_state.set_deck(*CardRepo.get_scenario_0()[1])

        episode_reward = 0.0
        episode_won_all = True
        final_hp = game_state.player.health
        total_steps = 0

        for battle in battles:
            if isinstance(battle, tuple):
                enemies = [e(game_state) for e in battle]
            else:
                enemies = [battle(game_state)]

            battle_state = BattleState(game_state, *enemies, verbose=Verbose.NO_LOG)
            fight_reward, win, final_hp, steps = run_ppo_fight(agent, game_state, battle_state)
            episode_reward += fight_reward
            total_steps += steps

            if not win:
                episode_won_all = False
                break

            # burning blood -- 6 hp heal after each won fight
            game_state.player.health = min(
                game_state.player.health + 6,
                game_state.player.max_health
            )

        logger.log_episode(episode_reward, episode_won_all, final_hp, total_steps)

        # on-policy update every EPISODES_PER_UPDATE episodes
        if (i_episode + 1) % EPISODES_PER_UPDATE == 0:
            ppo.update()

        recent_wins = logger.episode_wins[-20:] if len(logger.episode_wins) >= 20 else logger.episode_wins
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
        pbar.set_postfix({
            'win%': f'{win_rate:.0%}',
            'hp': final_hp,
            'reward': f'{episode_reward:.2f}',
        })

    # final update on any leftover experience, then save
    ppo.update()
    logger.save()
    ppo.save_model(args.output)
    print(f'\ntraining complete. evaluate with: python3.11 evaluate.py --agent ppo')


if __name__ == '__main__':
    main()