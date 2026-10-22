"""Summarize all available fixed natural-image moment tasks without selecting curves."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trajectory_event(points, threshold):
    """A sampled path event, with each improving time strictly before its late time.

    An event requires R(0)-R(early)>delta AND R(late)-R(0)>delta.
    Its occurrence is conditional on the saved grid/horizon; it is not an unbiased
    estimate of a continuous-time minimum or a population-risk generalization claim.
    """
    points = sorted(points)
    if len(points) < 3 or points[0][0] != 0:
        return dict(available=False, full_rebound=None)
    initial = points[0][1]
    minimum, early = initial, 0
    best_joint, best_pair, max_drawup = -math.inf, (0, 0), 0.
    for step, risk in points[1:]:
        if early > 0:
            joint = min(initial - minimum, risk - initial)
            if joint > best_joint:
                best_joint, best_pair = joint, (early, step)
        max_drawup = max(max_drawup, risk - minimum)
        if risk < minimum:
            minimum, early = risk, step
    if not math.isfinite(best_joint):
        best_joint = 0.
    lookup = dict(points)
    return dict(available=True, full_rebound=bool(best_joint > threshold),
                joint_margin=best_joint, max_drawup=max_drawup,
                max_improvement=initial - min(v for _, v in points),
                final_minus_initial=points[-1][1] - initial,
                selected_early_step=best_pair[0], selected_late_step=best_pair[1],
                selected_improvement=initial - lookup[best_pair[0]],
                selected_late_excess=lookup[best_pair[1]] - initial,
                observations=len(points), last_step=points[-1][0])


def wilson(events, n):
    if not n:
        return None, None
    z, p = 1.959963984540054, events / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center-radius, center+radius


def summarize(suite):
    suite = Path(suite)
    protocol = json.loads((suite / 'protocol.json').read_text(encoding='utf-8'))
    seeds = protocol['arguments']['seeds']
    threshold = protocol['arguments']['risk_threshold']
    output = suite / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    risks, classes, geometry, knn, local, coverage, audit, sensitivity, dominant = [], [], [], [], [], [], [], [], []
    trajectories = {}
    all_tasks = []
    for seed in seeds:
        directory = suite / f'seed_{seed}'
        result_path = directory / 'result.json'
        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
        evaluations = [json.loads(p.read_text(encoding='utf-8')) for p in directory.glob('evaluation_*.json')]
        evaluations.sort(key=lambda row: row['step'])
        available = {r['step'] for r in evaluations}
        coverage.append(dict(seed=seed, training_state=result.get('state', 'missing'),
                             completed_step=result.get('completed_step', 0),
                             evaluated_checkpoints=len(available), planned_checkpoints=len(protocol['evaluation_steps']),
                             missing_checkpoints=' '.join(str(s) for s in protocol['evaluation_steps'] if s not in available),
                             failure_records=' '.join(p.name for p in directory.glob('failure_step_*.json'))))
        for row in result.get('history', []):
            audit.append(dict(seed=seed, **row))
        for row in evaluations:
            names = row['task_names']
            all_tasks = names
            for representation in ['encoder', 'projection']:
                readings = row['readouts'][representation]
                spectrum = readings.get('training_feature_spectrum', {})
                leading = readings.get('leading_component_moments')
                if leading:
                    for index, task in enumerate(names):
                        dominant.append(dict(seed=seed, step=row['step'], epoch=row['epoch'],
                                             representation=representation, task=task,
                                             pc1_variance_share=spectrum['pc1_variance_share'],
                                             pc1_task_train_r2=leading['train_r2'][index],
                                             pc1_task_evaluation_mse=leading['evaluation_normalized_mse'][index]))
                for split in ['train', 'evaluation']:
                    for index, task in enumerate(names):
                        risk = readings[f'moment_{split}_mse'][index]
                        risks.append(dict(seed=seed, step=row['step'], epoch=row['epoch'],
                                          representation=representation, split=split, task=task, risk=risk))
                        trajectories.setdefault((seed, representation, split, task), []).append((row['step'], risk))
                classes.append(dict(seed=seed, step=row['step'], epoch=row['epoch'],
                                    representation=representation, rank=readings['rank'],
                                    input_dtype=readings.get('input_dtype'),
                                    singular_value_ratio=readings.get('singular_value_ratio'),
                                    train_mse=readings['class_train_onehot_mse'],
                                    evaluation_mse=readings['class_evaluation_onehot_mse'],
                                    train_accuracy=readings['class_train_accuracy'],
                                    evaluation_accuracy=readings['class_evaluation_accuracy'],
                                    feature_spectrum=json.dumps(spectrum)))
                for cutoff, check in readings.get('rank_threshold_checks', {}).items():
                    for split in ['train', 'evaluation']:
                        for index, task in enumerate(names):
                            sensitivity.append(dict(seed=seed, step=row['step'], representation=representation,
                                                    cutoff=cutoff, rank=check['rank'], split=split, task=task,
                                                    risk=check[f'moment_{split}_mse'][index]))
                        sensitivity.append(dict(seed=seed, step=row['step'], representation=representation,
                                                cutoff=cutoff, rank=check['rank'], split=split, task='class_onehot_mse',
                                                risk=check[f'class_{split}_onehot_mse']))
                for task_type, geometry_row in row['geometry'][representation].items():
                    for index, task_row in enumerate(geometry_row['tasks']):
                        geometry.append(dict(seed=seed, step=row['step'], representation=representation,
                                             task_type=task_type,
                                             task=names[index] if task_type == 'moments' else f'class_{index}',
                                             distance_op=geometry_row['distance_op'],
                                             distance_frobenius_over_sqrt2=geometry_row.get('distance_frobenius_over_sqrt2'),
                                             initial_rank=geometry_row['initial_rank'],
                                             current_rank=geometry_row['current_rank'], **task_row))
                    aggregate = geometry_row.get('raw_aggregate')
                    if aggregate:
                        geometry.append(dict(seed=seed, step=row['step'], representation=representation,
                                             task_type=task_type, task='raw_aggregate',
                                             risk_units='sum_of_onehot_squared_errors' if task_type == 'classes'
                                             else 'sum_of_raw_moment_squared_errors',
                                             class_mean_delta=aggregate['delta_risk'] / protocol['split']['classes']
                                             if task_type == 'classes' else None,
                                             class_mean_bound=aggregate['bound'] / protocol['split']['classes']
                                             if task_type == 'classes' and aggregate['bound'] is not None else None,
                                             **aggregate))
                for method, value in row.get('knn', {}).get(representation, {}).items():
                    if isinstance(value, dict):
                        knn.append(dict(seed=seed, step=row['step'], epoch=row['epoch'],
                                        representation=representation, method=method, **value))
        prediction_path = directory / 'prediction.json'
        if prediction_path.exists():
            prediction = json.loads(prediction_path.read_text(encoding='utf-8'))
            for representation in ['encoder', 'projection']:
                for index, task in enumerate(prediction['task_names']):
                    delta = prediction['slopes'][representation][index]
                    halves = [half[representation][index] for half in prediction['half_slopes']]
                    full_rank = all(half['rank'][representation]['full_column_rank']
                                    for half in prediction['half_slopes'])
                    tolerance = protocol['local_slope_tolerance']
                    stable = (full_rank and min(abs(s) for s in halves) > tolerance
                              and halves[0] * halves[1] > 0)
                    points = sorted(trajectories.get((seed, representation, 'train', task), []))
                    observed = points[1][1]-points[0][1] if len(points) > 1 else None
                    local.append(dict(seed=seed, representation=representation, task=task,
                                      predicted_local_slope_per_update=delta,
                                      half1_slope=halves[0], half2_slope=halves[1], half_sign_stable=stable,
                                      source_full_column_rank=full_rank, slope_tolerance=tolerance,
                                      predicted_direction=('improve' if delta < 0 else 'worsen') if stable else 'unresolved',
                                      first_recorded_delta=observed,
                                      first_recorded_step=points[1][0] if len(points) > 1 else None,
                                      sign_agreement=(observed*delta > 0) if observed is not None and stable else None,
                                      interpretation='Analytic local derivative along estimated mean update; '
                                      'not finite-rebound prediction'))
    events = [dict(seed=key[0], representation=key[1], split=key[2], task=key[3],
                   **trajectory_event(points, threshold)) for key, points in trajectories.items()]
    sensitivity_paths = {}
    for row in sensitivity:
        if row['task'] != 'class_onehot_mse':
            key = (row['seed'], row['representation'], row['split'], row['task'], row['cutoff'])
            sensitivity_paths.setdefault(key, []).append((row['step'], row['risk']))
    default_events = {(r['seed'], r['representation'], r['split'], r['task']): r for r in events}
    sensitivity_events = []
    for key, points in sensitivity_paths.items():
        event = trajectory_event(points, threshold)
        default = default_events[key[:4]]
        sensitivity_events.append(dict(seed=key[0], representation=key[1], split=key[2], task=key[3],
                                       cutoff=key[4], **event,
                                       differs_from_default=event['full_rebound'] != default['full_rebound']))
    rates = []
    for representation in ['encoder', 'projection']:
        for split in ['train', 'evaluation']:
            for task in all_tasks:
                rows = [r for r in events if (r['representation'], r['split'], r['task']) ==
                        (representation, split, task) and r['available']]
                successes = sum(r['full_rebound'] for r in rows)
                endpoints = [r['last_step'] for r in rows]
                common_horizon = len(set(endpoints)) == 1 if endpoints else False
                lo, hi = wilson(successes, len(rows)) if common_horizon else (None, None)
                rates.append(dict(representation=representation, split=split, task=task,
                                  observed_events=successes, observed_seeds=len(rows), planned_seeds=len(seeds),
                                  wilson_lower=lo, wilson_upper=hi,
                                  common_last_step=common_horizon,
                                  earliest_last_step=min(endpoints) if endpoints else None,
                                  latest_last_step=max(endpoints) if endpoints else None))
    for name, rows in [('moment_risks', risks), ('class_readouts', classes), ('geometry_bounds', geometry),
                       ('knn', knn), ('local_predictions', local), ('coverage', coverage), ('audit', audit),
                       ('trajectory_events', events), ('task_frequencies', rates), ('rank_sensitivity', sensitivity),
                       ('rank_sensitivity_events', sensitivity_events), ('dominant_statistics', dominant)]:
        write_csv(output / f'{name}.csv', rows)
    summary = dict(protocol=protocol, coverage=coverage, frequencies=rates,
                   events=events, evaluations=len(classes)//2,
                   scope='All available fixed tasks and checkpoints; independent unit is seed. '
                   'Train risks are empirical OLS optima; held-out curves are finite-sample errors. '
                   'Sampled minimum-based events are descriptive and depend on grid and horizon.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Natural-image moment experiments', '',
             f'Dataset: {protocol["split"]["dataset"]}; planned seeds: {seeds}; '
             f'budget: {protocol["arguments"]["epochs"]:g} exposure-equivalent epochs.', '',
             'No risk is reported as exact natural-image population risk. All affine OLS fits and '
             'normalizations use the official training split. Held-out labels do not choose tasks, '
             'hyperparameters, checkpoints or stopping times.', '',
             'A sampled full rebound requires an earlier risk below initialization and a later risk '
             f'above initialization, each by more than {threshold:g} training-target variance units. '
             'Maximum drawup is also retained without requiring the later risk to exceed initialization. '
             'Observed frequencies and Wilson intervals use independent seeds, not tasks/checkpoints. '
             'Missing seeds and partial coverage remain explicit; unequal horizons cannot support '
             'a common-horizon incidence estimate.', '',
             '| Seed | Training state | Saved update | Evaluated / planned checkpoints |',
             '| --- | --- | ---: | ---: |']
    for row in coverage:
        lines.append(f'| {row["seed"]} | {row["training_state"]} | {row["completed_step"]} | '
                     f'{row["evaluated_checkpoints"]}/{row["planned_checkpoints"]} |')
    lines.extend(['', '| Fixed task | Encoder train events / observed seeds | Projection train events / observed seeds |',
                  '| --- | ---: | ---: |'])
    for task in all_tasks:
        counts = []
        for representation in ['encoder', 'projection']:
            row = next(r for r in rates if (r['representation'], r['split'], r['task']) ==
                       (representation, 'train', task))
            counts.append(f'{row["observed_events"]}/{row["observed_seeds"]}')
        lines.append(f'| {task} | {counts[0]} | {counts[1]} |')
    lines.extend(['', '| Seed / representation | Initial → latest held-out class OLS accuracy | '
                  'Class MSE change | Latest raw / standardized / whitened KNN accuracy |',
                  '| --- | ---: | ---: | --- |'])
    for seed in seeds:
        for representation in ['encoder', 'projection']:
            rows = sorted([r for r in classes if r['seed'] == seed and r['representation'] == representation],
                          key=lambda r: r['step'])
            if not rows:
                continue
            accuracies = []
            latest_knn = max((r['step'] for r in knn if r['seed'] == seed and
                              r['representation'] == representation), default=None)
            for method in ['raw', 'standardized', 'whitened']:
                value = next((r['accuracy'] for r in knn if r['seed'] == seed and
                              r['representation'] == representation and r['step'] == latest_knn and
                              r['method'] == method), None)
                accuracies.append(f'{100*value:.2f}%' if value is not None else 'missing')
            first, last = rows[0], rows[-1]
            lines.append(f'| {seed} / {representation} | {100*first["evaluation_accuracy"]:.2f}% → '
                         f'{100*last["evaluation_accuracy"]:.2f}% | '
                         f'{last["evaluation_mse"]-first["evaluation_mse"]:+.6g} | '
                         f'{" / ".join(accuracies)} |')
    disagreements = sum(r['differs_from_default'] for r in sensitivity_events)
    lines.extend(['', f'Rank-cutoff sensitivity: {disagreements} task/seed/representation/split/cutoff '
                  'event decisions differ from the default solver. See rank_sensitivity_events.csv '
                  'and the underlying risks; this count is not a count of independent samples.'])
    lines.extend(['', 'The initialization derivative predicts a local sign only; disagreement between '
                  'two independent gradient half-estimates is marked unresolved. Agreement with the first '
                  'recorded observation does not validate a finite-horizon turning-time or rebound prediction. '
                  'The sample-function-space geometry bound is evaluated on the training distribution; '
                  'it cannot bound held-out changes. Read bound-to-observed ratios before interpreting '
                  'a small class change as a quantitative consistency check.', '',
                  'Raw/standardized/whitened Euclidean kNN comparisons separate sensitivity to feature '
                  'scaling from OLS readability. This is not a strict replication of another paper.'])
    (output / 'findings.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if risks:
        plot(output, risks, classes, audit, all_tasks, geometry)
    print(f'Summary: {output / "findings.md"}', flush=True)
    return summary


def plot(output, risks, classes, audit, tasks, geometry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        for representation, style in [('encoder', '-'), ('projection', '--')]:
            for seed in sorted({r['seed'] for r in risks}):
                rows = [r for r in risks if r['task'] == task and r['seed'] == seed and
                        r['representation'] == representation and r['split'] == 'evaluation']
                axis.plot([r['epoch'] for r in rows], [r['risk'] for r in rows], style,
                          label=f'{representation} seed {seed}', alpha=.8)
        axis.set_title(task, fontsize=9)
        axis.set_xscale('symlog', linthresh=1)
        axis.set_xlabel('Exposure-equivalent epoch')
        axis.set_ylabel('Held-out normalized MSE')
    for axis in list(axes.flat)[len(tasks):]:
        axis.axis('off')
    if tasks:
        axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / 'moment_trajectories.png', dpi=160)
    fig.savefig(output / 'moment_trajectories.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for seed in sorted({r['seed'] for r in classes}):
        for representation, style in [('encoder', '-'), ('projection', '--')]:
            rows = [r for r in classes if r['seed'] == seed and r['representation'] == representation]
            axes[0].plot([r['epoch'] for r in rows], [r['evaluation_accuracy'] for r in rows], style,
                         label=f'{representation} seed {seed}')
        rows = [r for r in audit if r['seed'] == seed]
        axes[1].plot([r['epoch'] for r in rows], [r['loss'] for r in rows], label=f'seed {seed}')
    for axis in axes:
        axis.set_xscale('symlog', linthresh=1)
        axis.set_xlabel('Exposure-equivalent epoch')
        axis.legend(fontsize=6)
    axes[0].set_ylabel('Held-out class OLS accuracy')
    axes[1].set_ylabel('Fixed-view audit GUPA loss')
    fig.tight_layout()
    fig.savefig(output / 'classes_and_objective.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for seed in sorted({r['seed'] for r in classes}):
        for representation, style in [('encoder', '-'), ('projection', '--')]:
            rows = [r for r in classes if r['seed'] == seed and r['representation'] == representation]
            axes[0].plot([r['epoch'] for r in rows],
                         [json.loads(r['feature_spectrum'])['pc1_variance_share'] for r in rows], style,
                         label=f'{representation} seed {seed}')
            moved = [r for r in geometry if r['seed'] == seed and r['representation'] == representation
                     and r['task_type'] == 'moments' and r['task'] == tasks[0]]
            by_step = {r['step']: r['epoch'] for r in rows}
            axes[1].plot([by_step[r['step']] for r in moved],
                         [r['distance_frobenius_over_sqrt2'] for r in moved], style,
                         label=f'{representation} seed {seed}')
    for axis in axes:
        axis.set_xscale('symlog', linthresh=1)
        axis.set_xlabel('Exposure-equivalent epoch')
        axis.legend(fontsize=6)
    axes[0].set_ylabel('Leading clean-feature variance share')
    axes[1].set_ylabel('Empirical subspace displacement (Frobenius / sqrt(2))')
    fig.tight_layout()
    fig.savefig(output / 'geometry_and_dominance.png', dpi=160)
    fig.savefig(output / 'geometry_and_dominance.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    summarize(parser.parse_args().suite)
