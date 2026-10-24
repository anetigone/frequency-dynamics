"""Plot downloaded follow-up JSON/CSV; no Torch, images, or training needed."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def plot(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ['#2864a5', '#d47722', '#38835e']
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    a, b, c, d = axes.flat
    confirmation = root / 'natural_confirmation_cifar10'
    summary = read(confirmation / 'confirmation_summary/summary.json')
    rows = [r for r in summary['fixed_window_results'] if r['representation'] == 'encoder']
    seeds = summary['design']['seeds']
    for offset, window, marker, color in [(-.09, 200, 'o', colors[0]), (.09, 750, 's', colors[1])]:
        selected = [next(r for r in rows if r['seed'] == s and r['early_epoch'] == window) for s in seeds]
        if not all(r['available'] for r in selected):
            raise ValueError('This review figure requires complete confirmation coverage')
        a.scatter(np.arange(len(seeds)) + offset, [r['joint_margin'] for r in selected],
                  label='200 -> 500' if window == 200 else '750 -> 2000', marker=marker, c=color, s=47, zorder=3)
    a.axhline(.001, color='#a52a2a', ls='--', lw=1, label='Required margin: 0.001')
    a.set_xticks(range(len(seeds)), seeds)
    a.set(xlabel='New RNG seed', ylabel='Minimum of the four risk gaps',
          title='(a) Fixed-window confirmation: 1/5 seeds')
    a.legend(fontsize=8, loc='lower left')

    # Display the sole successful confirmation seed, while panel (a) retains
    # all planned seeds and both windows.
    winners = [r for r in rows if r['full_rebound']]
    if not winners:
        b.text(.5, .5, 'No successful fixed window', ha='center', transform=b.transAxes)
    else:
        witness = winners[0]
        steps = [math.ceil(epoch * 50000 / 256) for epoch in
                 (0, witness['early_epoch'], witness['late_epoch'])]
        states = [read(confirmation / f'seed_{witness["seed"]}' / f'evaluation_{s}.json') for s in steps]
        for split, label, color, marker in [('train', 'Train optimum', colors[0], 'o'),
                                           ('evaluation', 'Test error', colors[1], 's')]:
            values = [r['readouts']['encoder'][f'moment_{split}_mse'][r['task_names'].index('antidiagonal_high_energy')]
                      for r in states]
            b.plot([r['epoch'] for r in states], np.array(values)-values[0], marker=marker, color=color, label=label)
        b.axhline(0, color='gray', lw=1)
        b.set(xlabel='Exposure-equivalent epoch', ylabel='Risk minus initialization',
              title=f'(b) Fixed task, seed {witness["seed"]}: improve then exceed start')
        b.legend(fontsize=8)

    with (root / 'natural_metric_cifar10/common_progress_pairs.csv').open(newline='') as handle:
        comparisons = list(csv.DictReader(handle))
    for seed, color in zip((30, 31, 32), colors):
        selected = sorted((r for r in comparisons if int(r['seed']) == seed and
                           float(r['source_epoch']) == 750 and r['split'] == 'evaluation' and
                           r['risk_pullback_minus_sgd']), key=lambda r: float(r['drop']))
        c.plot([float(r['drop']) for r in selected], [float(r['risk_pullback_minus_sgd']) for r in selected],
               'o-', color=color, label=f'Seed {seed}')
    c.axhline(0, color='gray', lw=1)
    c.set(xlabel='Fixed audit-loss drop from epoch 750', ylabel='Test risk: pullback minus SGD',
          title='(c) Metric intervention at similar audit progress')
    c.legend(fontsize=8)
    c.text(.02, .98, 'First crossings; actual losses are not identical', va='top', fontsize=8, transform=c.transAxes)

    values, accuracies = [], []
    for seed, color in zip((30, 31, 32), colors):
        result = read(root / 'natural_head_counterfactual_cifar10' / f'seed_{seed}/result.json')
        readings = [result['full_encoder']['5000'], result['oracle_head']['5000']['refitted_ols'],
                    result['cross_heads']['5000']['3000'], result['original_head']['5000']]
        risks = [r['test']['onehot_mse'] for r in readings]
        values.append(risks)
        accuracies.append([r['test']['accuracy'] for r in readings])
        d.plot(range(4), risks, 'o-', lw=.85, ms=4, alpha=.65, color=color)
    means = [statistics.mean(x) for x in zip(*values)]
    mean_accuracy = [statistics.mean(x) for x in zip(*accuracies)]
    d.scatter(range(4), means, marker='D', s=42, c='black', label='Three-seed mean', zorder=4)
    for index, risk, acc in zip(range(4), means, mean_accuracy):
        d.annotate(f'{100*acc:.2f}%', (index, risk), xytext=(0, 9), textcoords='offset points',
                   ha='center', fontsize=8)
    d.set_xticks(range(4), ['Full encoder\n64 dims', 'Train-label oracle\n8 dims', 'Epoch-3000 head\n8 dims',
                          'Epoch-5000 head\n8 dims'])
    d.set(ylabel='Test one-hot MSE', title='(d) Same epoch-5000 encoder, different readouts')
    d.legend(fontsize=8, loc='lower right')
    d.text(.02, .98, 'Percent labels: test accuracy', va='top', fontsize=8, transform=d.transAxes)
    d.margins(y=.18)
    for axis in axes.flat:
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.tight_layout(pad=2)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'followup_overview.{suffix}', dpi=190, bbox_inches='tight')
    plt.close(fig)
    print(output / 'followup_overview.png')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root / 'experiments/results/empirical/mechanism_followup')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot(args.root, args.output or args.root / 'review_20261024')
