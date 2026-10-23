"""Plot the downloaded results, without model inference or training."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--until-epoch', type=float)
    parser.add_argument('--moment-task', default='x_high_energy')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-textures', action='store_true')
    args = parser.parse_args()
    summary = args.snapshot / 'natural_moments_cifar10' / 'summary'
    classes = read(summary / 'class_readouts.csv')
    nearest = read(summary / 'knn.csv')
    risks = read(summary / 'moment_risks.csv')
    if args.until_epoch is not None:
        protocol = json.loads((summary.parent / 'protocol.json').read_text(encoding='utf-8'))
        last_step = math.ceil(args.until_epoch * protocol['split']['train_size'] /
                              protocol['arguments']['batch_size'])
        classes = [r for r in classes if int(r['step']) <= last_step]
        nearest = [r for r in nearest if int(r['step']) <= last_step]
        risks = [r for r in risks if int(r['step']) <= last_step]
    output = args.output or args.snapshot / 'review'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    methods = [('OLS', classes, '#202020'), ('raw', nearest, '#D55E00'),
               ('standardized', nearest, '#0072B2'), ('whitened', nearest, '#009E73')]
    for method, source, color in methods:
        rows = [r for r in source if r['representation'] == 'encoder' and
                (method == 'OLS' or r['method'] == method)]
        steps = sorted({int(r['step']) for r in rows})
        epochs, averages, lower, upper = [], [], [], []
        for step in steps:
            selected = [r for r in rows if int(r['step']) == step]
            values = [100 * float(r['evaluation_accuracy'] if method == 'OLS' else r['accuracy'])
                      for r in selected]
            epochs.append(float(selected[0]['epoch']))
            averages.append(sum(values) / len(values))
            lower.append(min(values))
            upper.append(max(values))
        label = 'Affine OLS' if method == 'OLS' else f'{method.capitalize()} kNN'
        axes[0].plot(epochs, averages, color=color, label=label, linewidth=2,
                     marker='o', markersize=3)
        axes[0].fill_between(epochs, lower, upper, color=color, alpha=.09)
    axes[0].set(title='(a) Class readouts of the same encoder', ylabel='Official test accuracy (%)')
    axes[0].legend(frameon=False, fontsize=9)
    for seed, color in zip([30, 31, 32], ['#0072B2', '#D55E00', '#009E73']):
        for split, style in [('train', '-'), ('evaluation', '--')]:
            rows = sorted([r for r in risks if int(r['seed']) == seed and
                           r['representation'] == 'encoder' and r['split'] == split and
                           r['task'] == args.moment_task], key=lambda r: int(r['step']))
            initial = float(rows[0]['risk'])
            label = f'Seed {seed}' if split == 'train' else None
            axes[1].plot([float(r['epoch']) for r in rows],
                         [float(r['risk']) - initial for r in rows], style, color=color,
                         linewidth=1.8, label=label)
    axes[1].axhline(0, color='.4', linewidth=.8)
    axes[1].axhspan(-.001, .001, color='.5', alpha=.10)
    axes[1].plot([], [], '-', color='.25', label='Train empirical optimum')
    axes[1].plot([], [], '--', color='.25', label='Official test error')
    task_title = args.moment_task.replace('_', ' ')
    axes[1].set(title=f'(b) Fixed {task_title} task',
                ylabel='Encoder normalized MSE minus initial MSE')
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xscale('symlog', linthresh=1)
        axis.set_xlabel('Exposure-equivalent epoch')
        axis.grid(axis='y', alpha=.2)
    horizon = max(float(r['epoch']) for r in classes)
    fig.suptitle(f'CIFAR-10: three fixed seeds, evaluations through {horizon:.0f} epochs', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'cifar_readout_and_moment.png', dpi=180)
    fig.savefig(output / 'cifar_readout_and_moment.pdf')
    plt.close(fig)
    if args.skip_textures:
        print(f'Review plots: {output}')
        return

    rows = read(args.snapshot / 'texture_rebound_50_seeds' / 'trajectory_summary' / 'task_frequencies.csv')
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    fig, axis = plt.subplots(figsize=(12, 4.4))
    for offset, mode, color in [(-.18, 'square', '#0072B2'), (0., 'linear', '#D55E00'),
                                (.18, 'frozen_front', '#009E73')]:
        selected = [next(r for r in rows if r['mode'] == mode and r['representation'] == 'encoder'
                         and r['metric'] == 'full_rebound' and r['task'] == task) for task in tasks]
        values = [float(r['completed_seed_frequency']) for r in selected]
        low = [max(0., v - float(r['wilson95_lower'])) for v, r in zip(values, selected)]
        high = [max(0., float(r['wilson95_upper']) - v) for v, r in zip(values, selected)]
        axis.errorbar([i + offset for i in range(len(tasks))], values, yerr=[low, high],
                      color=color, marker='o', linestyle='none', markersize=4, capsize=3,
                      label=mode)
    axis.set_xticks(range(len(tasks)), tasks, rotation=28, ha='right')
    axis.set_ylim(-.025, .45)
    axis.set_ylabel('Sampled full-rebound frequency')
    axis.set_title('Encoder rebound per fixed task: 50 paired seeds, 4,000 updates')
    axis.grid(axis='y', alpha=.2)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output / 'texture_task_frequencies.png', dpi=180)
    fig.savefig(output / 'texture_task_frequencies.pdf')
    plt.close(fig)
    print(f'Review plots: {output}')


if __name__ == '__main__':
    main()
