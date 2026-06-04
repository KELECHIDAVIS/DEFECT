'''
evaluate.py -- D.E.F.E.C.T. agent evaluation suite

runs all agents against the same seeded multi-fight scenarios and saves
results to eval_results.json for visualization.

usage:
    python3.11 evaluate.py                        # evaluate all agents, full battle suite
    python3.11 evaluate.py --agent dqn            # evaluate only dqn
    python3.11 evaluate.py --agent llm            # evaluate llm (requires OPENAI_API_KEY)
    python3.11 evaluate.py --episodes 50          # run 50 eval episodes
    python3.11 evaluate.py --battles single       # jawworm only (faster, for llm)

llm agent setup:
    export OPENAI_API_KEY="your-key-here"
    python3.11 evaluate.py --agent llm --battles single
    note: multi-fight llm evaluation is slow (~1s per card choice, many more turns)
    recommend --battles single for llm to match the original paper setup
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
from agent import JawWorm, SwampLeech, GoblinFiend, GoblinWizard, Cultist
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

# battle suites -- mirrors main.py
# add new enemies here as the benchmark grows
BATTLE_SUITES = {
    'single': [JawWorm],
    'two':    [JawWorm, SwampLeech],
    'three': [JawWorm, SwampLeech, (GoblinFiend, GoblinWizard)],
    'four':  [JawWorm, SwampLeech, (GoblinFiend, GoblinWizard), Cultist],
}

BURNING_BLOOD_HEAL = 6  # ironclad starting relic


def run_episode(agent, seed: int, battles: list) -> dict:
    '''
    run one full multi-fight evaluation episode with a fixed seed.
    hp carries over between fights. burning blood heals 6 after each won fight.
    returns a dict of metrics for this episode.
    '''
    random.seed(seed)

    game_state = GameState(Character.IRON_CLAD, agent, 0)
    game_state.set_deck(*CardRepo.get_scenario_0()[1])

    total_steps = 0
    total_damage_taken = 0
    fights_won = 0
    per_fight_hp = []

    for fight_idx, battle_class in enumerate(battles):
        hp_before = game_state.player.health
        # when creating battle state, unpack tuple if multi-enemy
        if isinstance(battle_class, tuple):
            enemies = [e(game_state) for e in battle_class]
        else:
            enemies = [battle_class(game_state)]

        battle_state = BattleState(game_state, *enemies, verbose=Verbose.NO_LOG)
        battle_state.initiate_log()

        steps = 0
        while not battle_state.ended():
            action = agent.choose_card(game_state, battle_state)
            battle_state.tick_player(action)
            steps += 1

        total_steps += steps
        result = battle_state.get_end_result()
        final_hp = game_state.player.health
        per_fight_hp.append(final_hp)
        total_damage_taken += hp_before - final_hp

        if result != 1:
            # player died -- record loss and stop
            return {
                'win': 0,
                'fights_won': fights_won,
                'fights_total': len(battles),
                'final_hp': 0,
                'max_hp': game_state.player.max_health,
                'hp_ratio': 0.0,
                'damage_taken': total_damage_taken,
                'turns': total_steps,
                'per_fight_hp': per_fight_hp,
                'seed': seed,
            }

        fights_won += 1

        # burning blood -- heal 6 hp between fights
        if fight_idx < len(battles) - 1:
            game_state.player.health = min(
                game_state.player.health + BURNING_BLOOD_HEAL,
                game_state.player.max_health
            )

    # cleared all fights
    max_hp = game_state.player.max_health
    return {
        'win': 1,
        'fights_won': fights_won,
        'fights_total': len(battles),
        'final_hp': game_state.player.health,
        'max_hp': max_hp,
        'hp_ratio': game_state.player.health / max_hp,
        'damage_taken': total_damage_taken,
        'turns': total_steps,
        'per_fight_hp': per_fight_hp,
        'seed': seed,
    }


def evaluate_agent(agent, agent_name: str, seeds: list, battles: list,
                   verbose: bool = True) -> dict:
    '''
    run evaluation episodes for one agent across all seeds.
    returns aggregated results dict.
    '''
    episodes = []
    wins = 0

    if verbose:
        suite_names = ['+'.join(e.__name__ for e in b) if isinstance(b, tuple) else b.__name__ for b in battles]

        print(f'\nevaluating {agent_name} over {len(seeds)} episodes '
              f'({" → ".join(suite_names)})...')

    for i, seed in enumerate(seeds):
        try:
            result = run_episode(agent, seed, battles)
        except Exception as e:
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
    avg_fights_won = sum(e['fights_won'] for e in episodes) / n

    winning_eps = [e for e in episodes if e['win'] == 1]
    avg_hp_when_winning = (sum(e['final_hp'] for e in winning_eps) / len(winning_eps)
                           if winning_eps else 0)

    if verbose:
        print(f'  results: win rate={win_rate:.1%} | avg hp={avg_hp:.1f} | '
              f'avg damage={avg_damage:.1f} | avg turns={avg_turns:.1f} | '
              f'avg fights won={avg_fights_won:.1f}/{len(battles)}')

    return {
        'agent': agent_name,
        'n_episodes': n,
        'n_fights': len(battles),
        'win_rate': win_rate,
        'wins': wins,
        'avg_final_hp': avg_hp,
        'avg_hp_ratio': avg_hp_ratio,
        'avg_damage_taken': avg_damage,
        'avg_turns': avg_turns,
        'avg_fights_won': avg_fights_won,
        'avg_hp_when_winning': avg_hp_when_winning,
        'episodes': episodes,
    }


def build_agents(args) -> list:
    agents = []

    if args.agent in ('all', 'random'):
        agents.append(('Random', RandomBot()))

    if args.agent in ('all', 'always_attack'):
        agents.append(('AlwaysAttack', AlwaysAttackBot()))

    if args.agent in ('all', 'backtrack'):
        agents.append(('Backtrack (depth 3)', BacktrackBot(3, False)))

    if args.agent in ('all', 'dqn'):
        agents.append(('DQN', DQNBot(eval_mode=True)))

    if args.agent in ('all', 'llm', 'llm_cot'):
        if not LLM_AVAILABLE:
            print('warning: llm agent unavailable -- chatgpt_bot import failed')
        elif not OPENAI_API_KEY:
            print('warning: llm agent skipped -- set OPENAI_API_KEY to include it')
            print('         export OPENAI_API_KEY="your-key-here"')
        else:
            if args.agent in ('all', 'llm'):
                agents.append(('LLM', ChatGPTBot(
                    ChatGPTBot.ModelName.GPT_Turbo_35, PromptOption.NONE, 0, False, 1)))
            if args.agent in ('all', 'llm_cot'):
                agents.append(('LLM CoT', ChatGPTBot(
                    ChatGPTBot.ModelName.GPT_Turbo_35, PromptOption.CoT, 0, False, 1)))

    return agents


def main():
    parser = argparse.ArgumentParser(description='D.E.F.E.C.T. agent evaluation suite')
    parser.add_argument('--agent', type=str, default='all',
                        choices=['all', 'dqn', 'random', 'always_attack',
                                 'backtrack', 'llm', 'llm_cot'],
                        help='which agent to evaluate (default: all)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='number of evaluation episodes (default: 50)')
    parser.add_argument('--battles', type=str, default='four',
                        choices=list(BATTLE_SUITES.keys()),
                        help='battle suite to run (default: four)')
    parser.add_argument('--output', type=str, default='eval_results.json',
                        help='output file for results (default: eval_results.json)')
    args = parser.parse_args()

    seeds = EVAL_SEEDS[:args.episodes]
    battles = BATTLE_SUITES[args.battles]

    # warn about llm speed on multi-fight suites
    if args.battles != 'single' and args.agent in ('all', 'llm', 'llm_cot'):
        print(f'note: llm evaluation on {args.battles} suite will be slow (~1s per card '
              f'choice x many turns x {len(battles)} fights x {args.episodes} episodes)')
        print('      consider --battles single to match the original paper setup')

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
        agent_results = evaluate_agent(agent, agent_name, seeds, battles, verbose=True)
        if agent_results:
            results[agent_name] = agent_results

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f'\nresults saved to {args.output} ({elapsed:.1f}s)')
    print(f'\nbattle suite: {args.battles} ({" → ".join("+".join(e.__name__ for e in b) if isinstance(b, tuple) else b.__name__ for b in battles)})')

    print(f'{"agent":<25} {"win rate":>10} {"avg hp":>10} {"avg dmg":>10} '
          f'{"avg turns":>10} {"fights won":>12}')
    print('-' * 72)
    for name, r in results.items():
        print(f'{name:<25} {r["win_rate"]:>10.1%} '
              f'{r["avg_final_hp"]:>10.1f} '
              f'{r["avg_damage_taken"]:>10.1f} '
              f'{r["avg_turns"]:>10.1f} '
              f'{r["avg_fights_won"]:>10.1f}/{r["n_fights"]}')


if __name__ == '__main__':
    main()