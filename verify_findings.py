'''
verify_findings.py -- D.E.F.E.C.T. claim verification

Recomputes every number in a results-log finding directly from the raw logs and
prints a pass/fail verdict per claim, so nothing in the paper rests on a number
someone eyeballed off a chart.

Finding 21 claims under test:
  C1  block-average table over 3000 episodes (win rate, HP, reward, turns)
  C2  PPO reaches >=96% win rate by episode 1000
  C3  PPO is flat after episode 1000 (tested, not asserted)
  C4  PPO reward plateaus at 5.47
  C5  DQN reward plateaus higher than PPO (requires DQN's training_log.json)
  C6  more episodes would not close the gap

C3 and C6 are the ones that actually need statistics. "Flat" is tested by fitting
an ordinary least squares trend to episodes 1000-3000 and checking whether the
95% confidence interval on the slope contains zero. If it does, the run is
plateaued and additional episodes are not expected to help.

usage:
    python3.11 verify_findings.py
    python3.11 verify_findings.py --ppo training_log_ppo.json --dqn training_log.json
    python3.11 verify_findings.py --save          # write plots to png
'''

import json
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('dark_background')
BG, PANEL, GRID, TEXT = '#0d1117', '#161b22', '#30363d', '#e6edf3'
PPO_C, DQN_C, OK_C = '#ff6ec7', '#00d4ff', '#2ecc71'

BLOCK = 500          # block size for the claimed table
PLATEAU_START = 1000  # claimed start of the plateau
CLAIMED = {          # the exact numbers written in Finding 21
    'blocks': [
        # (start, win, hp, reward, turns)
        (0,    0.61,  8.6, 4.01, 84.5),
        (500,  0.90, 12.7, 5.20, 72.5),
        (1000, 0.96, 12.7, 5.39, 64.0),
        (1500, 0.96, 12.8, 5.38, 62.7),
        (2000, 0.97, 13.7, 5.45, 62.0),
        (2500, 0.97, 13.7, 5.47, 61.0),
    ],
    'plateau_reward': 5.47,
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def arrays(log):
    return (np.array(log['episode_wins'], dtype=float),
            np.array(log['episode_hp'], dtype=float),
            np.array(log['episode_rewards'], dtype=float),
            np.array(log['episode_lengths'], dtype=float))


def ols_slope_ci(y, alpha_z=1.96):
    '''
    OLS slope of y against episode index, with a 95% CI.
    Returns (slope_per_1000_episodes, lo, hi).
    Pure numpy so this runs without scipy.
    '''
    n = len(y)
    x = np.arange(n, dtype=float)
    xbar, ybar = x.mean(), y.mean()
    sxx = ((x - xbar) ** 2).sum()
    slope = ((x - xbar) * (y - ybar)).sum() / sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    # residual standard error, 2 parameters estimated
    se_resid = np.sqrt((resid ** 2).sum() / (n - 2))
    se_slope = se_resid / np.sqrt(sxx)
    lo, hi = slope - alpha_z * se_slope, slope + alpha_z * se_slope
    # report per 1000 episodes -- easier to read than per episode
    return slope * 1000, lo * 1000, hi * 1000


def welch(a, b):
    '''Welch t statistic and approximate two-sided p via normal approximation'''
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (ma - mb) / np.sqrt(va + vb)
    # normal approximation is fine at n in the hundreds
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return ma, mb, t, p


def verdict(ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    return f'[{tag}] {detail}'


def check_finding_21(ppo, dqn, save, outdir):
    w, hp, r, t = arrays(ppo)
    n = len(w)
    print(f'\nPPO log: {n} episodes | overall win rate in file: '
          f"{ppo.get('overall_win_rate', float('nan')):.4f} | recomputed: {w.mean():.4f}")

    # ---- C1: block table ----
    print(f'\n--- C1: block averages (block = {BLOCK} episodes) ---')
    print(f'{"episodes":>12} {"win":>7} {"hp":>7} {"reward":>8} {"turns":>7}   vs claimed')
    all_ok = True
    for start, cw, chp, cr, ct in CLAIMED['blocks']:
        s, e = start, min(start + BLOCK, n)
        if s >= n:
            continue
        aw, ahp, ar, at = w[s:e].mean(), hp[s:e].mean(), r[s:e].mean(), t[s:e].mean()
        ok = (abs(aw - cw) <= 0.005 and abs(ahp - chp) <= 0.05
              and abs(ar - cr) <= 0.005 and abs(at - ct) <= 0.05)
        all_ok &= ok
        print(f'{s:>5}-{e:<6} {aw:>7.1%} {ahp:>7.1f} {ar:>8.2f} {at:>7.1f}   '
              f'{"ok" if ok else f"MISMATCH (claimed {cw:.0%}/{chp}/{cr}/{ct})"}')
    print(verdict(all_ok, 'block table reproduces the logged data'))

    # ---- C2: 96% by episode 1000 ----
    print('\n--- C2: reaches >=96% win rate by episode 1000 ---')
    blk = w[PLATEAU_START:PLATEAU_START + BLOCK].mean()
    print(f'win rate over episodes {PLATEAU_START}-{PLATEAU_START + BLOCK}: {blk:.1%}')
    print(verdict(blk >= 0.96, f'claim was >=96%, measured {blk:.1%}'))

    # ---- C3: flat after episode 1000 ----
    print(f'\n--- C3: plateau test, OLS trend over episodes {PLATEAU_START}-{n} ---')
    flat_ok = True
    for label, series, unit in [('reward', r, ''), ('win rate', w, ''),
                                ('hp', hp, ' hp'), ('turns', t, ' turns')]:
        sl, lo, hi = ols_slope_ci(series[PLATEAU_START:])
        contains_zero = lo <= 0 <= hi
        print(f'  {label:<9} slope {sl:+7.4f}{unit} per 1000 eps  '
              f'95% CI [{lo:+.4f}, {hi:+.4f}]  '
              f'{"flat" if contains_zero else "TRENDING"}')
        if label in ('reward', 'win rate'):
            flat_ok &= contains_zero
    print(verdict(flat_ok, 'reward and win rate show no significant trend after '
                           f'episode {PLATEAU_START}'))
    if not flat_ok:
        sl, lo, hi = ols_slope_ci(r[PLATEAU_START:])
        print('       the run was still improving. projected reward if the trend held:')
        for extra in (3000, 6000, 12000):
            print(f'         +{extra:>6,} episodes -> {r[-BLOCK:].mean() + sl * extra / 1000:.3f}')
        print('       treat this as extrapolation, not prediction. it does mean the'
              ' "converged" wording is too strong.')

    # ---- C4: plateau reward value ----
    print('\n--- C4: reward plateau value ---')
    last = r[-BLOCK:]
    ci = 1.96 * last.std(ddof=1) / np.sqrt(len(last))
    print(f'mean reward over last {BLOCK} episodes: {last.mean():.3f} +/- {ci:.3f} (95% CI)')
    print(verdict(abs(last.mean() - CLAIMED['plateau_reward']) <= 0.005,
                  f'claimed {CLAIMED["plateau_reward"]}'))

    # ---- internal consistency: reward should track the win flag ----
    print('\n--- internal consistency: reward vs win flag ---')
    won, lost = r[w == 1], r[w == 0]
    if len(lost) > 0:
        print(f'  won  episodes: n={len(won):4} mean reward {won.mean():.3f} '
              f'min {won.min():.3f}')
        print(f'  lost episodes: n={len(lost):4} mean reward {lost.mean():.3f} '
              f'max {lost.max():.3f}')
        sep = won.min() > lost.max()
        print(verdict(sep, 'win/loss reward ranges are cleanly separated'
                           if sep else 'ranges overlap, inspect the reward accounting'))
    # a full 4-fight win must score at least 4.0 (four wins x >=1.0 each)
    bad = int(((w == 1) & (r < 4.0)).sum())
    print(verdict(bad == 0, f'{bad} won episodes scored below the 4.0 floor for four wins'))

    # ---- C5 / C6: DQN comparison ----
    print('\n--- C5: DQN reward plateau (the number I estimated off the training curve) ---')
    if dqn is None:
        print('  DQN training log not found. C5 IS UNVERIFIED.')
        print('  The "roughly 5.8-6.0" figure in Finding 21 was read off the plotted')
        print('  curve, not computed. Re-run with --dqn training_log.json and replace')
        print('  that range with the measured value before it goes in the paper.')
        plot(ppo, None, save, outdir)
        return
    dw, dhp, dr, dt = arrays(dqn)
    dlast = dr[-BLOCK:]
    dci = 1.96 * dlast.std(ddof=1) / np.sqrt(len(dlast))
    print(f'  DQN last {BLOCK} episodes: reward {dlast.mean():.3f} +/- {dci:.3f} | '
          f'win {dw[-BLOCK:].mean():.1%} | hp {dhp[-BLOCK:].mean():.1f} | '
          f'turns {dt[-BLOCK:].mean():.1f}')
    ma, mb, tstat, p = welch(dlast, last)
    print(f'  DQN - PPO reward gap: {ma - mb:+.3f}  (Welch t={tstat:.2f}, p={p:.2e})')
    print(verdict(ma > mb and p < 0.05,
                  'DQN plateaus at a significantly higher reward than PPO'))

    print('\n--- C6: would more episodes close the gap? ---')
    sl, lo, hi = ols_slope_ci(r[PLATEAU_START:])
    gap = ma - mb
    if abs(sl) < 1e-9:
        print('  PPO trend is exactly flat.')
    else:
        eps_needed = gap / (sl / 1000)
        print(f'  PPO reward trend after ep {PLATEAU_START}: {sl:+.4f} per 1000 episodes')
        print(f'  gap to close: {gap:.3f}')
        if sl <= 0:
            print('  trend is flat or negative, so extrapolation never closes the gap')
        else:
            print(f'  at the observed trend it would take ~{eps_needed:,.0f} more '
                  f'episodes ({eps_needed / len(r):.0f}x the current run)')
    print(verdict(True, 'reported as an extrapolation, not a proof; state it that way'))

    plot(ppo, dqn, save, outdir)


def plot(ppo, dqn, save, outdir):
    w, hp, r, t = arrays(ppo)
    series = [('episode reward', r, None), ('win rate', w, None),
              ('final hp', hp, None), ('turns per episode', t, None)]
    if dqn is not None:
        dw, dhp, dr, dt = arrays(dqn)
        series = [('episode reward', r, dr), ('win rate', w, dw),
                  ('final hp', hp, dhp), ('turns per episode', t, dt)]

    def smooth(y, k=100):
        if len(y) < k:
            return np.arange(len(y)), y
        s = np.convolve(y, np.ones(k) / k, mode='valid')
        return np.arange(k - 1, len(y)), s

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.suptitle('Finding 21 verification -- 100-episode moving averages',
                 color='#00d4ff', fontsize=14, fontweight='bold')

    for ax, (label, p_series, d_series) in zip(axes.flat, series):
        ax.set_facecolor(PANEL)
        x, y = smooth(p_series)
        ax.plot(x, y, color=PPO_C, lw=1.8, label='PPO')
        if d_series is not None:
            dx, dy = smooth(d_series)
            ax.plot(dx, dy, color=DQN_C, lw=1.8, label='DQN')
        ax.axvline(PLATEAU_START, color=TEXT, ls='--', alpha=0.4, lw=1)
        ax.text(PLATEAU_START, ax.get_ylim()[1], ' claimed plateau start',
                color=TEXT, fontsize=8, va='top')
        # fitted trend over the claimed plateau region
        if len(p_series) > PLATEAU_START:
            seg = p_series[PLATEAU_START:]
            xs = np.arange(len(seg))
            c = np.polyfit(xs, seg, 1)
            ax.plot(xs + PLATEAU_START, np.polyval(c, xs), color=OK_C,
                    ls=':', lw=2, label='PPO trend (plateau region)')
        ax.set_title(label, color=TEXT, fontsize=11, fontweight='bold')
        ax.set_xlabel('training episode', color=TEXT, fontsize=9)
        ax.grid(True, color=GRID, alpha=0.5, lw=0.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    plt.tight_layout()
    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, 'finding21_verification.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f'\nsaved: {path}')
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description='verify results-log findings from raw logs')
    ap.add_argument('--ppo', default='training_log_ppo.json')
    ap.add_argument('--dqn', default='training_log.json')
    ap.add_argument('--finding', type=int, default=21)
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--outdir', default='plots')
    args = ap.parse_args()

    ppo = load(args.ppo)
    if ppo is None:
        print(f'error: {args.ppo} not found')
        return
    dqn = load(args.dqn)

    print('=' * 78)
    print(f'FINDING {args.finding} VERIFICATION')
    print('=' * 78)

    if args.finding == 21:
        check_finding_21(ppo, dqn, args.save, args.outdir)
    else:
        print(f'finding {args.finding} not implemented yet')


if __name__ == '__main__':
    main()