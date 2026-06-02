'''
evaluate.py -- D.E.F.E.C.T. agent evaluation suite

runs all agents against the same seeded scenarios and saves
results to eval_results.json for visualization.

usage:
    python3.11 evaluate.py                        # evaluate all agents
    python3.11 evaluate.py --agent dqn            # evaluate only dqn
    python3.11 evaluate.py --agent llm            # evaluate llm (requires OPENAI_API_KEY)
    python3.11 evaluate.py --episodes 200         # run 200 eval episodes

llm agent setup:
    export OPENAI_API_KEY="your-key-here"
    python3.11 evaluate.py --agent llm
    get a key at platform.openai.com -- ~50 episodes costs very little
'''

import json
import random
import argparse
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import JawWorm
from card import CardRepo

from ggpa.random_bot import RandomBot
from ggpa.backtrack import BacktrackBot
from ggpa.always_attack import AlwaysAttackBot
from ggpa.dqn_agent import DQNBot

# llm agent requires openai api key -- import conditionally
try:
    from ggpa.chatgpt_bot import ChatGPTBot
    from ggpa.prompt2 import PromptOption
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# read api key from environment variable or auth.py fallback
try:
    from auth import GPT_AUTH
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', GPT_AUTH)
except ImportError:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)


# fixed seeds for reproducible evaluation -- all agents face identical scenarios
EVAL_SEEDS = [42, 123, 456, 789, 1000, 1337, 2024, 3141, 9999, 12345,
              111, 222, 333, 444, 555, 666, 777, 888, 999, 1111,
              2222, 3333, 4444, 5555, 6666, 7777, 8888, 9876, 5432, 1234,
              11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
              200, 300, 400, 500, 600, 700, 800, 900, 1500, 2500]


def run_episode(agent, seed: int) -> dict:
    '''
    run one evaluation episode with a fixed seed.
    returns a dict of metrics for this episode.
    '''
    random.seed(seed)

    game_state = GameState(Character.IRON_CLAD, agent, 0)
    game_state.set_deck(*CardRepo.get_scenario_0()[1])
    battle_state = BattleState(game_state, JawWorm(game_state), verbose=Verbose.NO_LOG)

    prev_hp = battle_state.player.max_health

    battle_state.initiate_log()

    # use tick_player loop for consistent per-step tracking across all agents
    steps = 0
    while not battle_state.ended():
        action = agent.choose_card(game_state, battle_state)
        battle_state.tick_player(action)
        steps += 1

    result = battle_state.get_end_result()
    final_hp = battle_state.player.health
    max_hp = battle_state.player.max_health

    return {
        'win': 1 if result == 1 else 0,
        'final_hp': final_hp,
        'max_hp': max_hp,
        'hp_ratio': final_hp / max_hp,
        'damage_taken': max_hp - final_hp,
        'turns': steps,
        'seed': seed,
    }


def evaluate_agent(agent, agent_name: str, seeds: list, verbose: bool = True) -> dict:
    '''
    run evaluation episodes for one agent across all seeds.
    returns aggregated results dict.
    '''
    episodes = []
    wins = 0

    if verbose:
        print(f'\nevaluating {agent_name} over {len(seeds)} episodes...')

    for i, seed in enumerate(seeds):
        try:
            result = run_episode(agent, seed)
        except Exception as e:
            # llm agent can fail on api errors -- log and skip
            print(f'  episode {i + 1} failed ({e}), skipping')
            continue

        episodes.append(result)
        wins += result['win']

        if verbose and (i + 1) % 10 == 0:
            win_rate = wins / len(episodes)
            print(f'  {i + 1}/{len(seeds)} -- running win rate: {win_rate:.1%}')

    if not episodes:
        print(f'  no episodes completed for {agent_name}')
        return {}

    n = len(episodes)
    win_rate = wins / n
    avg_hp = sum(e['final_hp'] for e in episodes) / n
    avg_hp_ratio = sum(e['hp_ratio'] for e in episodes) / n
    avg_damage = sum(e['damage_taken'] for e in episodes) / n
    avg_turns = sum(e['turns'] for e in episodes) / n

    winning_eps = [e for e in episodes if e['win'] == 1]
    avg_hp_when_winning = (sum(e['final_hp'] for e in winning_eps) / len(winning_eps)
                           if winning_eps else 0)

    if verbose:
        print(f'  results: win rate={win_rate:.1%} | '
              f'avg hp={avg_hp:.1f} | avg damage={avg_damage:.1f} | '
              f'avg turns={avg_turns:.1f}')

    return {
        'agent': agent_name,
        'n_episodes': n,
        'win_rate': win_rate,
        'wins': wins,
        'avg_final_hp': avg_hp,
        'avg_hp_ratio': avg_hp_ratio,
        'avg_damage_taken': avg_damage,
        'avg_turns': avg_turns,
        'avg_hp_when_winning': avg_hp_when_winning,
        'episodes': episodes,
    }


def build_agents(args) -> list:
    '''
    build list of (name, agent) tuples based on args.
    llm agents are only included if api key is available.
    '''
    agents = []

    if args.agent in ('all', 'random'):
        agents.append(('Random', RandomBot()))

    if args.agent in ('all', 'always_attack'):
        agents.append(('AlwaysAttack', AlwaysAttackBot()))

    if args.agent in ('all', 'backtrack'):
        agents.append(('Backtrack (depth 3)', BacktrackBot(3, False)))

    if args.agent in ('all', 'dqn'):
        dqn_agent = DQNBot(eval_mode=True)
        
        agents.append(('DQN', dqn_agent))

    if args.agent in ('all', 'llm', 'llm_cot'):
        if not LLM_AVAILABLE:
            print('warning: llm agent unavailable -- chatgpt_bot import failed')
        elif not OPENAI_API_KEY:
            print('warning: llm agent skipped -- set OPENAI_API_KEY to include it')
            print('         export OPENAI_API_KEY="your-key-here"')
            print('         get a key at platform.openai.com')
        else:
            if args.agent in ('all', 'llm'):
                # plain llm without chain of thought -- matches llm paper baseline
                # uses gpt-3.5-turbo -- same model as the original paper
                llm_agent = ChatGPTBot(
                    ChatGPTBot.ModelName.GPT_Turbo_35,
                    PromptOption.NONE,
                    0,      # history window (0 = stateless, best per llm paper)
                    False,  # no option outcomes
                    1       # num retries on invalid response
                )
                agents.append(('LLM', llm_agent))

            if args.agent in ('all', 'llm_cot'):
                # llm with chain of thought -- best llm variant from the paper
                # uses gpt-3.5-turbo -- same model as the original paper
                llm_cot_agent = ChatGPTBot(
                    ChatGPTBot.ModelName.GPT_Turbo_35,
                    PromptOption.CoT,
                    0,
                    False,
                    1
                )
                agents.append(('LLM CoT', llm_cot_agent))

    return agents


def main():
    parser = argparse.ArgumentParser(description='D.E.F.E.C.T. agent evaluation suite')
    parser.add_argument('--agent', type=str, default='all',
                        choices=['all', 'dqn', 'random', 'always_attack',
                                 'backtrack', 'llm', 'llm_cot'],
                        help='which agent to evaluate (default: all)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='number of evaluation episodes (default: 50)')
    parser.add_argument('--output', type=str, default='eval_results.json',
                        help='output file for results (default: eval_results.json)')
    args = parser.parse_args()

    seeds = EVAL_SEEDS[:args.episodes]

    # always start fresh -- delete existing results if present
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f'removed existing {args.output} -- starting fresh')

    results = {}

    agents_to_run = build_agents(args)
    if not agents_to_run:
        print('no agents to evaluate.')
        return

    start = time.time()
    for agent_name, agent in agents_to_run:
        agent_results = evaluate_agent(agent, agent_name, seeds, verbose=True)
        if agent_results:
            results[agent_name] = agent_results

    # save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f'\nresults saved to {args.output} ({elapsed:.1f}s)')
    print(f'\nsummary:')
    print(f'{"agent":<25} {"win rate":>10} {"avg hp":>10} {"avg dmg":>10} {"avg turns":>10}')
    print('-' * 65)
    for name, r in results.items():
        print(f'{name:<25} {r["win_rate"]:>10.1%} '
              f'{r["avg_final_hp"]:>10.1f} '
              f'{r["avg_damage_taken"]:>10.1f} '
              f'{r["avg_turns"]:>10.1f}')


if __name__ == '__main__':
    main()