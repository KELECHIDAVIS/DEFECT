'''
plot_results.py -- D.E.F.E.C.T. evaluation visualization

reads eval_results.json and generates comparison plots for all agents.
also reads training_log.json for dqn training curves if available.

usage:
    python3.11 plot_results.py                         # plot all
    python3.11 plot_results.py --input eval_results.json
    python3.11 plot_results.py --save                  # save plots to png
'''

import json
import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# dark theme matching d.e.f.e.c.t. aesthetic
plt.style.use('dark_background')

# color palette -- one distinct color per agent
AGENT_COLORS = {
    'Random':             '#6c757d',   # grey
    'AlwaysAttack':       '#e74c3c',   # red
    'Backtrack (depth 3)':'#f39c12',   # amber
    'DQN':                '#00d4ff',   # cyan
    'LLM':                '#9b59b6',   # purple
    'LLM CoT':            '#2ecc71',   # green
}

# fallback colors for any agent not in the palette
FALLBACK_COLORS = ['#e91e63', '#ff9800', '#4caf50', '#03a9f4', '#9c27b0']

BG_COLOR = '#0d1117'
PANEL_COLOR = '#161b22'
GRID_COLOR = '#30363d'
TEXT_COLOR = '#e6edf3'
ACCENT_COLOR = '#00d4ff'

# reference lines from the llm paper and design doc
LLM_PAPER_BACKTRACK_AVG_HP = 25.94   # backtrack agent in starter deck scenario
LLM_PAPER_LLM_COT_AVG_HP = 23.36    # llm cot agent in starter deck scenario
SKILLED_PLAYER_WIN_RATE = 0.20       # lower bound for skilled roguelite players
PRO_PLAYER_WIN_RATE = 0.50           # pro player win rate benchmark


def get_color(agent_name: str, idx: int) -> str:
    for key in AGENT_COLORS:
        if key.lower() in agent_name.lower():
            return AGENT_COLORS[key]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def style_axis(ax, title: str = '', xlabel: str = '', ylabel: str = ''):
    '''apply consistent dark styling to an axis'''
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)


def plot_win_rate(ax, results: dict):
    '''bar chart of win rates for all agents with reference lines'''
    agents = list(results.keys())
    win_rates = [results[a]['win_rate'] for a in agents]
    colors = [get_color(a, i) for i, a in enumerate(agents)]
    n = results[agents[0]]['n_episodes'] if agents else 0

    bars = ax.bar(agents, win_rates, color=colors, alpha=0.85,
                  edgecolor=GRID_COLOR, linewidth=0.5)

    # value labels on bars
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{rate:.0%}', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=9, fontweight='bold')

    # reference lines from design doc
    ax.axhline(y=SKILLED_PLAYER_WIN_RATE, color='#ffffff', linestyle='--',
               alpha=0.4, linewidth=1, label=f'skilled player ({SKILLED_PLAYER_WIN_RATE:.0%})')
    ax.axhline(y=PRO_PLAYER_WIN_RATE, color='#ffffff', linestyle=':',
               alpha=0.4, linewidth=1, label=f'pro player ({PRO_PLAYER_WIN_RATE:.0%})')

    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f'{v:.0%}' for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=8, loc='upper right',
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    style_axis(ax, title=f'win rate (n={n} episodes)', ylabel='win rate')


def plot_hp_distribution(ax, results: dict):
    '''overlapping hp distribution histograms -- mirrors llm paper figure style'''
    bins = np.linspace(0, 80, 17)  # 0 to 80 hp in 5hp bins

    for i, (agent_name, data) in enumerate(results.items()):
        color = get_color(agent_name, i)
        hp_values = [e['final_hp'] for e in data['episodes']]

        ax.hist(hp_values, bins=bins, alpha=0.5, color=color,
                label=agent_name, edgecolor='none')

        # mean line
        mean_hp = np.mean(hp_values)
        ax.axvline(x=mean_hp, color=color, linestyle='--',
                   linewidth=1.5, alpha=0.9)

    # reference lines from llm paper
    ax.axvline(x=LLM_PAPER_BACKTRACK_AVG_HP, color='#f39c12', linestyle=':',
               linewidth=1, alpha=0.6, label=f'llm paper backtrack ({LLM_PAPER_BACKTRACK_AVG_HP:.1f})')
    ax.axvline(x=LLM_PAPER_LLM_COT_AVG_HP, color='#9b59b6', linestyle=':',
               linewidth=1, alpha=0.6, label=f'llm paper cot ({LLM_PAPER_LLM_COT_AVG_HP:.1f})')

    ax.set_xlim(0, 85)
    ax.legend(fontsize=7, loc='upper left',
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    style_axis(ax, title='distribution of final player hp',
               xlabel='final player hp', ylabel='count')


def plot_avg_hp(ax, results: dict):
    '''grouped bar chart: avg hp overall vs avg hp when winning'''
    agents = list(results.keys())
    x = np.arange(len(agents))
    width = 0.35

    avg_hp_all = [results[a]['avg_final_hp'] for a in agents]
    avg_hp_win = [results[a]['avg_hp_when_winning'] for a in agents]
    colors = [get_color(a, i) for i, a in enumerate(agents)]

    bars1 = ax.bar(x - width / 2, avg_hp_all, width, label='all episodes',
                   color=colors, alpha=0.6, edgecolor=GRID_COLOR, linewidth=0.5)
    bars2 = ax.bar(x + width / 2, avg_hp_win, width, label='winning episodes only',
                   color=colors, alpha=1.0, edgecolor=GRID_COLOR, linewidth=0.5)

    # reference lines
    ax.axhline(y=LLM_PAPER_BACKTRACK_AVG_HP, color='#f39c12', linestyle='--',
               alpha=0.5, linewidth=1, label=f'llm paper backtrack avg')
    ax.axhline(y=LLM_PAPER_LLM_COT_AVG_HP, color='#9b59b6', linestyle='--',
               alpha=0.5, linewidth=1, label=f'llm paper cot avg')

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15)
    ax.set_ylim(0, 85)
    ax.legend(fontsize=8, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    style_axis(ax, title='average final hp', ylabel='hp remaining')


def plot_damage_taken(ax, results: dict):
    '''bar chart of average damage taken per episode'''
    agents = list(results.keys())
    avg_damage = [results[a]['avg_damage_taken'] for a in agents]
    colors = [get_color(a, i) for i, a in enumerate(agents)]

    bars = ax.bar(agents, avg_damage, color=colors, alpha=0.85,
                  edgecolor=GRID_COLOR, linewidth=0.5)

    for bar, dmg in zip(bars, avg_damage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{dmg:.1f}', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=9, fontweight='bold')

    ax.tick_params(axis='x', rotation=15)
    style_axis(ax, title='average damage taken', ylabel='damage taken (hp lost)')


def plot_turns(ax, results: dict):
    '''violin or box plot of turn count distribution per agent'''
    agents = list(results.keys())
    turn_data = [[e['turns'] for e in results[a]['episodes']] for a in agents]
    colors = [get_color(a, i) for i, a in enumerate(agents)]

    parts = ax.violinplot(turn_data, positions=range(len(agents)),
                          showmedians=True, showextrema=True)

    # style each violin
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor(color)

    for part_name in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[part_name].set_color(TEXT_COLOR)
        parts[part_name].set_linewidth(1)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=15)
    style_axis(ax, title='turn count distribution', ylabel='turns per episode')


def plot_win_rate_vs_episodes(ax, results: dict):
    '''
    running win rate over episodes for each agent.
    shows how win rate stabilizes as more episodes are run.
    useful for checking if 50 episodes is enough for stable estimates.
    '''
    for i, (agent_name, data) in enumerate(results.items()):
        color = get_color(agent_name, i)
        wins_cumulative = np.cumsum([e['win'] for e in data['episodes']])
        episodes_range = np.arange(1, len(data['episodes']) + 1)
        running_win_rate = wins_cumulative / episodes_range

        ax.plot(episodes_range, running_win_rate, color=color,
                label=agent_name, linewidth=1.5, alpha=0.9)

    ax.axhline(y=SKILLED_PLAYER_WIN_RATE, color='#ffffff', linestyle='--',
               alpha=0.3, linewidth=1)
    ax.axhline(y=PRO_PLAYER_WIN_RATE, color='#ffffff', linestyle=':',
               alpha=0.3, linewidth=1)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    style_axis(ax, title='running win rate over evaluation episodes',
               xlabel='episode', ylabel='win rate')


def plot_dqn_training_curve(ax, training_log_path: str):
    '''plot dqn training reward over episodes if training log exists'''
    if not os.path.exists(training_log_path):
        ax.text(0.5, 0.5, 'no training log found\nrun main.py with dqn agent first',
                ha='center', va='center', color=TEXT_COLOR, fontsize=10,
                transform=ax.transAxes)
        style_axis(ax, title='dqn training curve (not available)')
        return

    with open(training_log_path, 'r') as f:
        log = json.load(f)

    episodes = log.get('episodes', [])
    rewards = log.get('episode_rewards', [])
    win_rates = log.get('running_win_rate', [])

    if rewards:
        ax.plot(episodes, rewards, color=AGENT_COLORS['DQN'],
                alpha=0.4, linewidth=0.8, label='episode reward')
        # smoothed reward
        if len(rewards) >= 20:
            smoothed = np.convolve(rewards, np.ones(20) / 20, mode='valid')
            ax.plot(range(19, len(rewards)), smoothed, color=AGENT_COLORS['DQN'],
                    linewidth=2, label='smoothed (20-ep avg)')

    if win_rates:
        ax2 = ax.twinx()
        ax2.plot(episodes, win_rates, color='#2ecc71',
                 linewidth=1.5, alpha=0.8, label='win rate')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('win rate', color='#2ecc71', fontsize=9)
        ax2.tick_params(colors='#2ecc71')
        ax2.spines['right'].set_color(GRID_COLOR)

    ax.legend(fontsize=8, loc='upper left',
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    style_axis(ax, title='dqn training curve',
               xlabel='training episode', ylabel='episode reward')


def main():
    parser = argparse.ArgumentParser(description='D.E.F.E.C.T. evaluation visualization')
    parser.add_argument('--input', type=str, default='eval_results.json',
                        help='evaluation results file (default: eval_results.json)')
    parser.add_argument('--training_log', type=str, default='training_log.json',
                        help='dqn training log file (default: training_log.json)')
    parser.add_argument('--save', action='store_true',
                        help='save plots to png instead of displaying')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='directory to save plots (default: plots)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'error: {args.input} not found. run evaluate.py first.')
        sys.exit(1)

    with open(args.input, 'r') as f:
        results = json.load(f)

    if not results:
        print('no results found in file.')
        sys.exit(1)

    print(f'loaded results for agents: {list(results.keys())}')

    # ---- main comparison dashboard (2x3 grid) ----
    fig = plt.figure(figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle('D.E.F.E.C.T. -- Agent Evaluation Dashboard',
                 color=ACCENT_COLOR, fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_win_rate(ax1, results)
    plot_hp_distribution(ax2, results)
    plot_avg_hp(ax3, results)
    plot_damage_taken(ax4, results)
    plot_turns(ax5, results)
    plot_win_rate_vs_episodes(ax6, results)

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, 'defect_evaluation_dashboard.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
        print(f'saved: {path}')
    else:
        plt.show()

    # ---- dqn training curve (separate figure) ----
    fig2, ax_training = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax_training.set_facecolor(PANEL_COLOR)
    plot_dqn_training_curve(ax_training, args.training_log)
    fig2.suptitle('D.E.F.E.C.T. -- DQN Training Progress',
                  color=ACCENT_COLOR, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if args.save:
        path = os.path.join(args.output_dir, 'defect_training_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
        print(f'saved: {path}')
    else:
        plt.show()

    # ---- print summary table ----
    print(f'\n{"=" * 75}')
    print(f'{"agent":<25} {"win%":>8} {"avg hp":>8} {"avg dmg":>9} '
          f'{"avg turns":>10} {"hp(win)":>9}')
    print(f'{"-" * 75}')
    for name, r in results.items():
        print(f'{name:<25} {r["win_rate"]:>8.1%} '
              f'{r["avg_final_hp"]:>8.1f} '
              f'{r["avg_damage_taken"]:>9.1f} '
              f'{r["avg_turns"]:>10.1f} '
              f'{r["avg_hp_when_winning"]:>9.1f}')
    print(f'{"=" * 75}')
    print(f'reference: llm paper backtrack avg hp = {LLM_PAPER_BACKTRACK_AVG_HP} | '
          f'llm cot avg hp = {LLM_PAPER_LLM_COT_AVG_HP}')
    


if __name__ == '__main__':
    main()
