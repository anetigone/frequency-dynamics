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
from itertools import combinations

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
    partial = pair_gaps(row(.4), row(.38), row(.39), 'encoder', 0)
    assert partial['category'] == 'improvement_only' and partial['joint_drawup'] > DELTA
    assert not partial['full_rebound']
    mismatch = pair_gaps(row(.4), row(.42, .38), row(.39, .42), 'encoder', 0)
    assert not mismatch['full_rebound']
    assert pair_gaps(row(0.), row(-DELTA), row(DELTA), 'encoder', 0)['category'] == 'neither'
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


def pair_gaps(initial, early, late, rep, index):
    gaps = {}
    for split in ('train', 'evaluation'):
        r0, rs, rt = [r['readouts'][rep][f'moment_{split}_mse'][index]
                      for r in (initial, early, late)]
        gaps.update({f'{split}_initial': r0, f'{split}_early': rs, f'{split}_late': rt,
                     f'{split}_improvement': r0-rs, f'{split}_excess': rt-r0,
                     f'{split}_drawup': rt-rs})
    improvement = min(gaps[f'{s}_improvement'] for s in ('train', 'evaluation'))
    excess = min(gaps[f'{s}_excess'] for s in ('train', 'evaluation'))
    drawup = min(gaps[f'{s}_drawup'] for s in ('train', 'evaluation'))
    category = ('complete' if improvement > DELTA and excess > DELTA else
                'improvement_only' if improvement > DELTA else
                'excess_only' if excess > DELTA else 'neither')
    return dict(**gaps, joint_improvement=improvement, joint_excess=excess,
                joint_margin=min(improvement, excess), joint_drawup=drawup,
                full_rebound=category == 'complete', category=category)


def analyze_events(args):
    """Read saved JSON only; keep the original confirmation outputs intact."""
    destination = args.output / 'event_analysis'
    destination.mkdir(parents=True, exist_ok=True)
    points, pairs, windows, paths, intervals, coverage = [], [], [], [], [], []
    for seed in args.seeds:
        directory = args.output / f'seed_{seed}'
        steps = [epoch_step(e, 50000, 256) for e in EPOCHS]
        missing = [s for s in steps if not (directory/f'evaluation_{s}.json').exists()]
        coverage.append(dict(seed=seed, complete=not missing, missing_steps=' '.join(map(str, missing))))
        if missing:
            continue
        evaluations = {e: read_json(directory/f'evaluation_{s}.json') for e, s in zip(EPOCHS, steps)}
        initial = evaluations[0]
        names = initial['task_names']
        for e, value in evaluations.items():
            if value['seed'] != seed or value['step'] != epoch_step(e, 50000, 256) or value['task_names'] != names:
                raise ValueError('Inconsistent evaluation seed, step or task order')
        prediction = read_json(directory/'prediction.json') if (directory/'prediction.json').exists() else None
        if prediction and (prediction['seed'] != seed or prediction['task_names'] != names):
            raise ValueError('Prediction seed/tasks do not match evaluations')
        result = read_json(directory/'result.json') if (directory/'result.json').exists() else {}
        crossing = next((r['epoch'] for r in result.get('history', []) if r['loss'] <= -3), None)
        for rep in ('encoder', 'projection'):
            for j, task in enumerate(names):
                for e, value in evaluations.items():
                    row = dict(seed=seed, representation=rep, task=task, epoch=e)
                    for split in ('train', 'evaluation'):
                        risk = value['readouts'][rep][f'moment_{split}_mse'][j]
                        if not math.isfinite(risk):
                            raise ValueError('Nonfinite risk')
                        row[f'{split}_risk'] = risk
                        row[f'{split}_minus_initial'] = risk-initial['readouts'][rep][f'moment_{split}_mse'][j]
                    points.append(row)
                local_pairs = []
                # Retrospective complete events exclude time zero as the early point.
                for a, b in combinations(EPOCHS[1:], 2):
                    row = dict(seed=seed, representation=rep, task=task, early_epoch=a, late_epoch=b,
                               **pair_gaps(initial, evaluations[a], evaluations[b], rep, j))
                    pairs.append(row)
                    local_pairs.append(row)
                    if (a, b) in WINDOWS:
                        windows.append(row.copy())
                for a, b in zip(EPOCHS, EPOCHS[1:]):
                    intervals.append(dict(seed=seed, representation=rep, task=task, source_epoch=a,
                        late_epoch=b, **pair_gaps(initial, evaluations[a], evaluations[b], rep, j)))
                best = max(local_pairs, key=lambda r: r['joint_margin'])
                # Common-pair drawup permits time zero, unlike complete rebound.
                drawups = [(a, b, pair_gaps(initial, evaluations[a], evaluations[b], rep, j))
                           for a, b in combinations(EPOCHS, 2)]
                draw = max(drawups, key=lambda r: r[2]['joint_drawup'])
                later_draw = max((r for r in drawups if r[0] > 0), key=lambda r: r[2]['joint_drawup'])
                joint_improvement = max(min(initial['readouts'][rep][f'moment_{s}_mse'][j] -
                    evaluations[e]['readouts'][rep][f'moment_{s}_mse'][j] for s in ('train', 'evaluation'))
                    for e in EPOCHS[1:])
                joint_excess = max(min(evaluations[e]['readouts'][rep][f'moment_{s}_mse'][j] -
                    initial['readouts'][rep][f'moment_{s}_mse'][j] for s in ('train', 'evaluation'))
                    for e in EPOCHS[1:])
                geom = initial.get('geometry', {}).get(rep, {}).get('moments', {}).get('tasks', [])
                initial_train = initial['readouts'][rep]['moment_train_mse'][j]
                slope = prediction['slopes'][rep][j] if prediction else None
                halves = [h[rep][j] for h in prediction['half_slopes']] if prediction else []
                change200 = evaluations[200]['readouts'][rep]['moment_train_mse'][j]-initial_train
                paths.append(dict(seed=seed, representation=rep, task=task,
                    initial_train_risk=initial_train, a0=geom[j]['a0'] if geom else None,
                    sqrt_risk_product=math.sqrt(max(0., initial_train*(1-initial_train))),
                    initial_slope=slope, half_slope_first=halves[0] if halves else None,
                    half_slope_second=halves[1] if halves else None,
                    train_change_to_200=change200,
                    initial_sign_matches_to_200=(slope*change200 > 0) if slope is not None else None,
                    first_audit_epoch_below_minus3=crossing,
                    max_joint_improvement=joint_improvement, max_joint_excess=joint_excess,
                    sampled_full_rebound=best['full_rebound'], best_joint_margin=best['joint_margin'],
                    best_early_epoch=best['early_epoch'], best_late_epoch=best['late_epoch'],
                    common_pair_max_drawup=draw[2]['joint_drawup'], drawup_source_epoch=draw[0],
                    drawup_late_epoch=draw[1], joint_drawup_event=draw[2]['joint_drawup'] > DELTA,
                    max_train_drawup=max(r[2]['train_drawup'] for r in drawups),
                    max_evaluation_drawup=max(r[2]['evaluation_drawup'] for r in drawups),
                    after_initial_joint_drawup=later_draw[2]['joint_drawup'],
                    after_initial_drawup_source_epoch=later_draw[0],
                    after_initial_drawup_late_epoch=later_draw[1],
                    fixed_union_rebound=any(r['full_rebound'] for r in local_pairs if
                                           (r['early_epoch'], r['late_epoch']) in WINDOWS)))
    for name, rows in [('points', points), ('all_pairs', pairs), ('fixed_windows', windows),
                       ('adjacent_changes', intervals), ('path_events', paths), ('coverage', coverage)]:
        write_csv(destination/f'{name}.csv', rows)
    primary = [r for r in paths if r['representation'] == 'encoder' and r['task'] == TASK]
    lines = ['# Event decomposition', '',
        'Saved JSON only. Primary task, threshold and fixed windows are unchanged. '
        'All eleven tasks and both representations are retained in CSV. '
        'Any-pair analyses and audit loss -3 are retrospective diagnostics. '
        'Training and test must share each witness pair; neither adds independent repetitions.', '',
        '| Seed | Max joint improvement | Max joint excess | Common-pair max drawup | Fixed union | Any-pair full |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for r in primary:
        lines.append(f"| {r['seed']} | {r['max_joint_improvement']:.6f} | {r['max_joint_excess']:.6f} | "
                     f"{r['common_pair_max_drawup']:.6f} | {r['fixed_union_rebound']} | {r['sampled_full_rebound']} |")
    full = sum(r['fixed_union_rebound'] for r in primary)
    any_full = sum(r['sampled_full_rebound'] for r in primary)
    draw_count = sum(r['after_initial_joint_drawup'] > DELTA for r in primary)
    lines += ['', f'Primary encoder: {len(primary)}/{len(args.seeds)} planned seeds observed; '
        f'fixed-window union {full}, any-pair complete events {any_full}, '
        f'common-pair drawup after a positive early time {draw_count}. '
        'These are different events; no thresholds/windows or confirmation denominators are changed.', '',
        'Improvement/excess maxima may occur at different times: their conjunction is not a full event. '
        'Common-pair drawup is max over ordered pairs of min(train change, test change); '
        'it is distinct from independently maximizing each split or from crossing initialization.', '',
        '| Seed | 200/500 category | 750/2000 category | Initial slope | Training change to 200 |',
        '| --- | --- | --- | ---: | ---: |']
    for r in primary:
        categories = [next(w['category'] for w in windows if w['seed'] == r['seed'] and
            w['representation'] == 'encoder' and w['task'] == TASK and w['early_epoch'] == a) for a, _ in WINDOWS]
        slope = f"{r['initial_slope']:.3g}" if r['initial_slope'] is not None else 'missing'
        lines.append(f"| {r['seed']} | {categories[0]} | {categories[1]} | {slope} | {r['train_change_to_200']:.6f} |")
    lines += ['', '| Seed | Positive-time drawup pair | Common train/test drawup | First audit L<=-3 epoch |',
              '| --- | --- | ---: | ---: |']
    for r in primary:
        lines.append(f"| {r['seed']} | {r['after_initial_drawup_source_epoch']}→{r['after_initial_drawup_late_epoch']} | "
                     f"{r['after_initial_joint_drawup']:.6f} | {r['first_audit_epoch_below_minus3']} |")
    lines += ['', 'An improvement-only fixed window can still have substantial drawup (seed 44). '
        'A positive drawup does not require complete rebound. Initial slopes are local; '
        'stable half-estimate signs do not ensure agreement over 200 epochs. '
        'For a deterministic unit-variance scalar task, a0=sqrt(R0*(1-R0)); '
        'its magnitude does not retain the orientation of motion.', '',
        'Signed geometric terms require checkpoint features, absent from risk-only JSON. '
        'Use the geometry phase on the server, then geometry-summary separately. '
        'No geometric terms are inferred from risk gaps alone.']
    (destination/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Event decomposition: {destination / "findings.md"}')


def geometry(args):
    """Read checkpoints and extract official train features only; never train."""
    import natural_moment_experiment as base
    source = args.source or args.output
    protocol = read_json(source/'protocol.json')
    n, batch = protocol['split']['train_size'], protocol['arguments']['batch_size']
    if protocol['split']['dataset'] != 'CIFAR-10' or n != 50000 or protocol['split']['evaluation_size'] != 10000:
        raise ValueError('Use the existing official CIFAR split')
    requested = [(seed, e, epoch_step(e, n, batch)) for seed in args.seeds for e in EPOCHS]
    missing = [str(source/f'seed_{s}/step_{t}.pt') for s, _, t in requested
               if not (source/f'seed_{s}/step_{t}.pt').exists()]
    if missing:
        raise FileNotFoundError('Run geometry where existing checkpoints reside; no training/download is needed. '
                                f'Missing {len(missing)} checkpoint(s), first: {missing[0]}')
    base.load_runtime()
    torch = base.torch
    torch.set_num_threads(args.threads)
    from natural_moment_eval import fixed_moment_dictionary, signed_geometry
    args.dataset, args.image_size = 'cifar10', 32
    data, _, train, _, split = base.load_images(args)
    if split != protocol['split']:
        raise ValueError('Data preprocessing differs from checkpoint protocol')
    data = data[train.to(data.device)]
    indices = torch.arange(len(data))
    names, targets = fixed_moment_dictionary(data, batch_size=args.feature_batch)
    destination = args.output/'geometry_analysis'
    settings = dict(source=str(source.resolve()), seeds=args.seeds, epochs=EPOCHS, task_names=names,
        data='official train only; targets scaled by training variance',
        identity='delta_risk = lost_readable - gained_residual + signed_cross',
        interpretation='retrospective empirical identity, not a finite-time prediction',
        new_training_updates=0)
    path = destination/'protocol.json'
    if path.exists() and read_json(path) != settings:
        raise ValueError('Use a separate output for a different geometry source/settings')
    write_json(path, settings)
    for seed in args.seeds:
        output = destination/f'seed_{seed}.json'
        if output.exists():
            print(f'seed={seed}: retaining existing geometry', flush=True)
            continue
        previous, previous_epoch, rows, checkpoints = None, None, [], []
        for e in EPOCHS:
            step = epoch_step(e, n, batch)
            file = source/f'seed_{seed}/step_{step}.pt'
            state = torch.load(file, map_location='cpu', weights_only=True)
            if state['seed'] != seed or state['step'] != step:
                raise ValueError(f'Checkpoint seed/step differs: {file}')
            model = base.NaturalSquare().to(args.device)
            model.load_state_dict(state['model'])
            model.eval()
            current = dict(zip(('encoder', 'projection'), base.extract(model, data, args.feature_batch)))
            checkpoints.append(dict(epoch=e, step=step, path=str(file.resolve())))
            if previous is not None:
                for rep in ('encoder', 'projection'):
                    value = signed_geometry(previous[rep], current[rep], targets, indices)
                    recorded = [read_json(source/f'seed_{seed}/evaluation_{s}.json')
                                for s in [epoch_step(previous_epoch, n, batch), step]]
                    for j, task in enumerate(names):
                        risks = [v['readouts'][rep]['moment_train_mse'][v['task_names'].index(task)] for v in recorded]
                        rows.append(dict(seed=seed, representation=rep, task=task,
                            source_epoch=previous_epoch, late_epoch=e,
                            initial_rank=value['initial_rank'], current_rank=value['current_rank'],
                            distance=value['distance_frobenius_over_sqrt2'], **value['tasks'][j],
                            saved_initial_risk=risks[0], saved_current_risk=risks[1],
                            saved_initial_error=value['tasks'][j]['initial_risk']-risks[0],
                            saved_current_error=value['tasks'][j]['current_risk']-risks[1],
                            saved_delta_error=value['tasks'][j]['delta_risk']-(risks[1]-risks[0])))
            previous, previous_epoch = current, e
        write_json(output, dict(seed=seed, state='complete', checkpoints=checkpoints, rows=rows))
        print(f'seed={seed}: signed geometry complete; no training', flush=True)


def geometry_summary(args):
    destination = args.output/'geometry_analysis'
    rows, missing = [], []
    for seed in args.seeds:
        path = destination/f'seed_{seed}.json'
        if not path.exists():
            missing.append(seed)
        else:
            rows.extend(read_json(path)['rows'])
    if not rows:
        raise FileNotFoundError('No signed geometry results; run geometry on the checkpoint server first')
    write_csv(destination/'signed_terms.csv', rows)
    lines = ['# Signed training-risk decomposition', '', f'Unavailable planned seeds: {missing}.',
        'Exact retrospective training identity; no held-out projector or prediction claim.', '',
        '| Seed | Epochs | Lost readable | Gained residual | Signed cross | Delta risk |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['representation'] == 'encoder' and r['task'] == TASK:
            lines.append(f"| {r['seed']} | {r['source_epoch']}→{r['late_epoch']} | {r['lost_readable']:.6f} | "
                         f"{r['gained_residual']:.6f} | {r['signed_cross']:.6f} | {r['delta_risk']:.6f} |")
    lines += ['', f"Maximum identity error: {max(abs(r['identity_error']) for r in rows):.3g}.",
        f"Maximum delta discrepancy vs saved OLS: {max(abs(r['saved_delta_error']) for r in rows):.3g}.",
        'The first two terms are nonnegative in exact arithmetic; tiny signed roundoff is retained. '
        'All tasks/representations are in signed_terms.csv. Cancellation is reported without attributing '
        'a unique dynamical cause or treating adjacent intervals as independent samples.']
    (destination/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Signed geometry summary: {destination / "findings.md"}')


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', default='plan', choices=[
        'plan', 'train', 'evaluate', 'summarize', 'smoke', 'analyze',
        'geometry-plan', 'geometry', 'geometry-summary'])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--data-root', type=Path, default=root)
    parser.add_argument('--output', type=Path, default=root / 'experiments/results/empirical/natural_confirmation_cifar10')
    parser.add_argument('--source', type=Path, help='Geometry checkpoint directory; defaults to --output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--feature-batch', type=int, default=256)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) & {30, 31, 32}:
        parser.error('Distinct new seeds are required; discovery seeds 30–32 must remain separate')
    if min(args.threads, args.feature_batch) < 1:
        parser.error('Thread and feature batch sizes must be positive')
    if args.phase == 'analyze':
        analyze_events(args)
    elif args.phase == 'geometry-plan':
        source = args.source or args.output
        paths = [source/f'seed_{s}/step_{epoch_step(e, 50000, 256)}.pt' for s in args.seeds for e in EPOCHS]
        print(json.dumps(dict(source=str(source), output=str(args.output/'geometry_analysis'),
            epochs=EPOCHS, planned_checkpoints=len(paths), available_checkpoints=sum(p.exists() for p in paths),
            pairs=list(zip(EPOCHS, EPOCHS[1:])), training_updates=0,
            action='plan only; geometry requires existing server checkpoints and cached CIFAR; no downloads',
            missing_checkpoints=[str(p) for p in paths if not p.exists()]), indent=2))
    elif args.phase == 'geometry':
        geometry(args)
    elif args.phase == 'geometry-summary':
        geometry_summary(args)
    elif args.phase == 'smoke':
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
