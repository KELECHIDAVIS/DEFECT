'''
training_logger.py -- saves dqn training metrics to training_log.json
import and use alongside main.py training loop.

usage in main.py:
    from ggpa.rl_algos.training_logger import TrainingLogger
    logger = TrainingLogger()

    # at end of each episode:
    logger.log_episode(episode_reward, win, player_hp, steps)

    # periodically save:
    logger.save()
'''

import json
import os
import numpy as np


class TrainingLogger:
    def __init__(self, log_path: str = 'training_log.json', save_every: int = 10):
        self.log_path = log_path
        self.save_every = save_every

        self.episode_rewards = []
        self.episode_wins = []
        self.episode_lengths = []
        self.episode_hp = []
        self.episodes = []

        # load existing log if present (to resume tracking)
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                existing = json.load(f)
            self.episode_rewards = existing.get('episode_rewards', [])
            self.episode_wins = existing.get('episode_wins', [])
            self.episode_lengths = existing.get('episode_lengths', [])
            self.episode_hp = existing.get('episode_hp', [])
            self.episodes = existing.get('episodes', [])
            print(f'resumed training log from episode {len(self.episodes)}')

    def log_episode(self, total_reward: float, win: bool, final_hp: int, steps: int):
        '''call at the end of each training episode'''
        ep_num = len(self.episodes) + 1
        self.episodes.append(ep_num)
        self.episode_rewards.append(total_reward)
        self.episode_wins.append(1 if win else 0)
        self.episode_lengths.append(steps)
        self.episode_hp.append(final_hp)

        if ep_num % self.save_every == 0:
            self.save()

    def get_running_win_rate(self, window: int = 20) -> list:
        '''compute running win rate with a sliding window'''
        if not self.episode_wins:
            return []
        wins = np.array(self.episode_wins, dtype=float)
        # use cumulative for early episodes, windowed for later ones
        rates = []
        for i in range(len(wins)):
            start = max(0, i - window + 1)
            rates.append(float(np.mean(wins[start:i + 1])))
        return rates

    def save(self):
        '''save log to json'''
        data = {
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'episode_wins': self.episode_wins,
            'episode_lengths': self.episode_lengths,
            'episode_hp': self.episode_hp,
            'running_win_rate': self.get_running_win_rate(),
            'total_episodes': len(self.episodes),
            'overall_win_rate': float(np.mean(self.episode_wins)) if self.episode_wins else 0.0,
        }
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        '''print recent training stats'''
        if not self.episodes:
            return
        recent = 20
        recent_wins = self.episode_wins[-recent:]
        recent_rewards = self.episode_rewards[-recent:]
        ep = len(self.episodes)
        print(f'episode {ep} | '
              f'win rate (last {recent}): {np.mean(recent_wins):.1%} | '
              f'avg reward (last {recent}): {np.mean(recent_rewards):.3f} | '
              f'total win rate: {np.mean(self.episode_wins):.1%}')
