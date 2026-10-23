"""Five new CIFAR seeds, with fixed moment task and two fixed rebound windows.

Default plan reads no images. Training/evaluation delegate to the existing
natural-image driver; this file adds only the fixed-window analysis. The three
discovery seeds are never pooled into the confirmation denominator.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from natural_moment_experiment import epoch_step, read_json, write_json
from summarize_natural_moments import wilson, write_csv

TASK = 'antidiagonal_high_energy'
WINDOWS = [(200, 500), (750, 2000)]
EPOCHS = [0, 200, 500, 750, 2000]
SEEDS = [40, 41, 42, 43, 44]
DELTA = .001


def design(seeds):
    return dict(seeds=seeds, discovery_seeds_excluded=[30, 31, 32], task=TASK,
                windows=WINDOWS, threshold=DELTA, epochs=2000, batch_size=256,
                primary_representation='encoder',
                primary_rule='Same seed/task/window: both train and held-out improvement '
                             'and late excess above initialization must be strictly > delta',
                provenance='Task and windows chosen from previous exploratory seeds; '
                           'new RNG trajectories on the same official 50k/10k split, '
                           'not untouched new test images or a public preregistration',
                secondary='Projection, other ten tasks and source-local slopes are descriptive',
                selection='Fixed five-checkpoint grid; no seed replacement or outcome-dependent extension',
                knn='Skipped: this experiment tests OLS readability')


def delegated_command(args, phase):
    return [sys.executable, '-B', str(Path(__file__).with_name('natural_moment_experiment.py')),
            phase, '--dataset', 'cifar10', '--data-root', str(args.data_root),
            '--output', str(args.output), '--seeds', *map(str, args.seeds),
            '--epochs', '2000', '--eval-epochs', *map(str, EPOCHS),
            '--skip-knn', '--batch-size', '256', '--lr', '.002', '--ratio', '1',
            '--device', args.device, '--threads', str(args.threads),
            '--feature-batch', str(args.feature_batch)]


def fixed_event(initial, early, late, representation):
    rows = (initial, early, late)
    task_indices = [r['task_names'].index(TASK) for r in rows]
    gaps, risks = {}, {}
    for split in ('train', 'evaluation'):
        values = [r['readouts'][representation][f'moment_{split}_mse'][i]
                  for r, i in zip(rows, task_indices)]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite fixed-window risk')
        risks.update({f'{split}_{name}': value for name, value in
                      zip(('initial', 'early', 'late'), values)})
        gaps[f'{split}_improvement'] = values[0] - values[1]
        gaps[f'{split}_late_excess'] = values[2] - values[0]
    return dict(**risks, **gaps, joint_margin=min(gaps.values()),
                full_rebound=all(g > DELTA for g in gaps.values()),
                train_rebound=gaps['train_improvement'] > DELTA and gaps['train_late_excess'] > DELTA,
                evaluation_rebound=gaps['evaluation_improvement'] > DELTA and gaps['evaluation_late_excess'] > DELTA)


def summarize(args):
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        stored = protocol['arguments']
        if stored['seeds'] != args.seeds or stored['batch_size'] != 256 or stored['epochs'] != 2000:
            raise ValueError('Use the same seeds and fixed budget as the training output')
        split = protocol['split']
        if (stored['dataset'] != 'cifar10' or split['train_size'] != 50000 or
                split['evaluation_size'] != 10000 or split['classes'] != 10):
            raise ValueError('Confirmation requires the original official CIFAR 50k/10k split')
    rows, unions = [], []
    for seed in args.seeds:
        directory = args.output / f'seed_{seed}'
        result = read_json(directory / 'result.json') if (directory / 'result.json').exists() else {}
        history = {r['step']: r for r in result.get('history', [])}
        decisions = {rep: [] for rep in ('encoder', 'projection')}
        for early, late in WINDOWS:
            steps = [epoch_step(e, 50000, 256) for e in (0, early, late)]
            paths = [directory / f'evaluation_{s}.json' for s in steps]
            missing = [p.name for p in paths if not p.exists()]
            evaluations = None if missing else [read_json(p) for p in paths]
            if evaluations and any(r['seed'] != seed or r['step'] != s
                                   for r, s in zip(evaluations, steps)):
                raise ValueError('Evaluation seed/step differs from its directory or checkpoint name')
            for rep in decisions:
                row = dict(seed=seed, representation=rep, task=TASK,
                           early_epoch=early, late_epoch=late, available=not missing,
                           missing=' '.join(missing), full_rebound=None)
                if evaluations:
                    row.update(fixed_event(*evaluations, rep))
                    losses = [history.get(s, {}).get('loss') for s in steps]
                    row['audit_loss_initial'], row['audit_loss_early'], row['audit_loss_late'] = losses
                    row['audit_endpoint_descent'] = (losses[0] > losses[1] > losses[2]
                        if all(v is not None for v in losses) else None)
                    for label, evaluation in zip(('initial', 'early', 'late'), evaluations):
                        row[f'{label}_rank'] = evaluation['readouts'][rep]['rank']
                decisions[rep].append(row['full_rebound'])
                rows.append(row)
        for rep, values in decisions.items():
            unions.append(dict(seed=seed, representation=rep,
                               all_windows_available=all(v is not None for v in values),
                               at_least_one_rebound=True if any(v is True for v in values)
                               else False if all(v is not None for v in values) else None))
    counts = []
    for rep in ('encoder', 'projection'):
        for early, late in WINDOWS:
            selected = [r for r in rows if r['representation'] == rep and
                        r['early_epoch'] == early and r['late_epoch'] == late]
            n = sum(r['available'] for r in selected)
            k = sum(r['full_rebound'] is True for r in selected)
            counts.append(dict(representation=rep, window=f'{early}->{late}', events=k,
                               planned_seeds=len(args.seeds), observed_seeds=n,
                               unresolved_seeds=len(args.seeds)-n,
                               wilson95_complete=wilson(k, n) if n == len(args.seeds) else None))
        selected = [r for r in unions if r['representation'] == rep]
        n = sum(r['at_least_one_rebound'] is not None for r in selected)
        k = sum(r['at_least_one_rebound'] is True for r in selected)
        counts.append(dict(representation=rep, window='fixed two-window union', events=k,
                           planned_seeds=len(args.seeds), observed_seeds=n,
                           unresolved_seeds=len(args.seeds)-n,
                           wilson95_complete=wilson(k, n) if n == len(args.seeds) else None))
    output = args.output / 'confirmation_summary'
    write_json(output / 'summary.json', dict(design=design(args.seeds), counts=counts,
               fixed_window_results=rows, seed_union=unions,
               uncertainty='Wilson intervals are reported only for complete planned coverage; '
                           f'N={len(args.seeds)}. Missing is not a confirmed failure.'))
    write_csv(output / 'fixed_windows.csv', rows)
    write_csv(output / 'seed_union.csv', unions)
    lines = ['# Fixed-window confirmation', '',
             'Discovery seeds 30–32 are excluded. Primary: encoder, antidiagonal_high_energy, delta=0.001.',
             'Both train/held-out gaps must pass in the same fixed window. '
             'Missing runs remain visible; they are not classified as failures.', '',
             '| Representation | Fixed window | Successes / planned seeds | Observed | Unresolved |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in counts:
        lines.append(f"| {row['representation']} | {row['window']} | "
                     f"{row['events']}/{row['planned_seeds']} | {row['observed_seeds']} | {row['unresolved_seeds']} |")
    lines.extend(['', 'Counts are seed-level, not task/checkpoint-level independent trials. '
                  'This tests RNG repeatability on a previously inspected official image split. '
                  'It does not establish broad task typicality or a precise event prevalence.'])
    (output / 'findings.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Fixed-window results: {output / "findings.md"}', flush=True)
    return counts


def smoke():
    # Test that train/test cannot borrow successful windows from each other,
    # equality at delta fails, and missing planned seeds remain unresolved.
    import tempfile
    from types import SimpleNamespace

    def row(risk, train=None):
        return dict(task_names=[TASK], readouts={rep: dict(
            moment_train_mse=[risk if train is None else train],
            moment_evaluation_mse=[risk], rank=1) for rep in ('encoder', 'projection')})

    assert fixed_event(row(.4), row(.39), row(.42), 'encoder')['full_rebound']
    assert not fixed_event(row(.4), row(.4-DELTA/2), row(.42), 'encoder')['full_rebound']
    assert not fixed_event(row(.4), row(.42, .39), row(.39, .42), 'encoder')['full_rebound']
    # A constructed exact-threshold gap must fail the strict inequality.
    assert not fixed_event(row(0.), row(-DELTA), row(DELTA), 'encoder')['full_rebound']
    with tempfile.TemporaryDirectory(prefix='confirmation_smoke_') as temp:
        args = SimpleNamespace(output=Path(temp), seeds=[40, 41])
        directory = args.output / 'seed_40'
        for epoch, value in ((0, .4), (200, .39), (500, .42), (750, .39), (2000, .42)):
            step = epoch_step(epoch, 50000, 256)
            write_json(directory / f'evaluation_{step}.json', dict(seed=40, step=step, **row(value)))
        counts = summarize(args)
        primary = next(r for r in counts if r['representation'] == 'encoder' and
                       r['window'] == 'fixed two-window union')
        assert primary['events'] == 1 and primary['planned_seeds'] == 2 and primary['unresolved_seeds'] == 1
    print('PASS: fixed joint windows, strict threshold, missing coverage; synthetic JSON only')


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', default='plan', choices=['plan', 'train', 'evaluate', 'summarize', 'smoke'])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--data-root', type=Path, default=root)
    parser.add_argument('--output', type=Path, default=root / 'experiments/results/empirical/natural_confirmation_cifar10')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--feature-batch', type=int, default=256)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) & {30, 31, 32}:
        parser.error('Distinct new seeds are required; discovery seeds 30–32 must remain separate')
    if min(args.threads, args.feature_batch) < 1:
        parser.error('Thread and feature batch sizes must be positive')
    if args.phase == 'smoke':
        smoke()
    elif args.phase == 'plan':
        print(json.dumps(dict(design=design(args.seeds), output=str(args.output),
            steps_per_seed=390625, training_hours_at_0_01110=390625 * len(args.seeds) * .01110 / 3600,
            train_command=delegated_command(args, 'train'),
            action='plan only; no data access or training'), indent=2))
    elif args.phase == 'summarize':
        summarize(args)
        # Keep all eleven tasks descriptive; the primary counts above never
        # use the generic trajectory search in this existing summarizer.
        if (args.output / 'protocol.json').exists():
            subprocess.run(delegated_command(args, 'summarize'), check=True)
    else:
        subprocess.run(delegated_command(args, args.phase), check=True)


if __name__ == '__main__':
    main()
