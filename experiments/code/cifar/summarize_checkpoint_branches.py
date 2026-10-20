"""Summarize fixed-step and audited-loss branch contrasts without loading Torch."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import argparse
import hashlib
import json
import math
import statistics


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None, values=values)


def summarize(directory):
    protocol = read_json(directory / 'protocol.json')
    args, jobs = protocol['arguments'], protocol['jobs']
    if protocol.get('test_used'):
        raise ValueError('Validation-only study required')
    expected_steps = sorted({0, args['steps'], *range(args['eval_every'], args['steps'], args['eval_every'])})
    runs, evaluations = {}, {}
    for job in jobs:
        folder = directory / job['id']
        run = read_json(folder / 'result.json')
        evaluation = read_json(folder / 'evaluation.json')
        digest = hashlib.sha256((folder / 'result.json').read_bytes()).hexdigest()
        if run['job'] != job or evaluation['job'] != job or evaluation['result_sha256'] != digest:
            raise ValueError('Branch metadata or evaluated-result identity mismatch')
        if run['steps'] != args['steps'] or run['fixed_steps'] != expected_steps or run.get('test_used') or evaluation.get('test_used'):
            raise ValueError('Incomplete or incompatible branch budget/splits')
        if run['source'] != protocol['sources'][str(job['seed'])]:
            raise ValueError('Source provenance differs from protocol')
        snapshot_steps = [row['step'] for row in run['snapshots']]
        if len(set(snapshot_steps)) != len(snapshot_steps) or [r['step'] for r in evaluation['rows']] != snapshot_steps:
            raise ValueError('Missing or duplicated snapshot evaluation')
        if not set(expected_steps).issubset(snapshot_steps):
            raise ValueError('Missing fixed checkpoint')
        for row in run['history']:
            if not math.isfinite(row['loss']):
                raise ValueError('Nonfinite audit loss')
        history = run['history']
        expected_audits = sorted(set(range(0, args['steps'] + 1, args['audit_every'])) | set(expected_steps))
        if [h['step'] for h in history] != expected_audits:
            raise ValueError('Incomplete fixed-audit history')
        eligible = {target for target in args['loss_levels'] if history[0]['loss'] > target
                    and any(h['loss'] <= target for h in history)}
        if len(run['loss_crossings']) != len(eligible) or {c['target_loss'] for c in run['loss_crossings']} != eligible:
            raise ValueError('Missing or duplicate first-crossing record')
        for crossing in run['loss_crossings']:
            audit = next((r['audit'] for r in run['snapshots'] if r['step'] == crossing['step']), None)
            if audit is None or crossing['target_loss'] not in args['loss_levels'] or crossing['observed_loss'] != audit['loss']:
                raise ValueError('Loss crossing differs from saved audit')
            target = crossing['target_loss']
            first = next((r for r in history if r['loss'] <= target), None)
            if history[0]['loss'] <= target or first is None or first['step'] != crossing['step']:
                raise ValueError('Loss match is not the first post-fork downward crossing')
        key = (job['seed'], job['name'])
        if key in runs:
            raise ValueError('Duplicate seed/branch')
        runs[key], evaluations[key] = run, {r['step']: r for r in evaluation['rows']}
    seeds = args['seeds']
    names = protocol.get('branch_names', ['ratio_0.1', 'baseline', 'ratio_10', 'slow', 'fast'])
    if 'baseline' not in names or len(set(names)) != len(names):
        raise ValueError('Unique branch names including baseline required')
    if set(runs) != {(s, n) for s in seeds for n in names}:
        raise ValueError('Missing planned seed or branch')
    for seed in seeds:
        group = [runs[seed, n] for n in names]
        for key in ['initial_state_sha256', 'batch_index_sha256', 'sampled_views_sha256', 'final_generator_sha256']:
            if len({r[key] for r in group}) != 1:
                raise ValueError(f'Branch pairing mismatch: seed={seed}, {key}')
        if any(evaluations[seed, n][0] != evaluations[seed, 'baseline'][0] for n in names):
            raise ValueError('Initial probes differ across branches')
        if any(r['history'][0] != group[0]['history'][0] for r in group):
            raise ValueError('Initial fixed-audit moments differ')

    fixed = []
    for name in names:
        for step in expected_steps:
            for rep in ['encoder', 'projection']:
                entry = dict(branch=name, step=step, representation=rep, seeds=seeds)
                for metric in ['accuracy', 'onehot_mse']:
                    values = [evaluations[s, name][step][rep]['validation'][metric] for s in seeds]
                    initial = [evaluations[s, name][0][rep]['validation'][metric] for s in seeds]
                    baseline = [evaluations[s, 'baseline'][step][rep]['validation'][metric] for s in seeds]
                    entry[metric] = describe(values)
                    entry[metric + '_minus_fork'] = describe([v - a for v, a in zip(values, initial)])
                    entry[metric + '_minus_baseline'] = describe([v - a for v, a in zip(values, baseline)])
                for metric in ['loss', 'clean_energy']:
                    entry['audit_' + metric] = describe([next(h for h in runs[s, name]['history'] if h['step'] == step)[metric] for s in seeds])
                fixed.append(entry)

    contrasts = [(n, 'baseline') for n in names if n != 'baseline']
    contrasts += [(n, control) for n in ['ratio_0.1', 'ratio_10'] if n in names
                  for control in ['slow', 'fast'] if control in names]
    matched = []
    for name, control in contrasts:
        for target in args['loss_levels']:
            pairs, missing = [], []
            for seed in seeds:
                left = next((c for c in runs[seed, name]['loss_crossings'] if c['target_loss'] == target), None)
                right = next((c for c in runs[seed, control]['loss_crossings'] if c['target_loss'] == target), None)
                if left is None or right is None:
                    missing.append(dict(seed=seed, reason='target_not_crossed_after_fork',
                                        missing_branches=[n for n, row in [(name, left), (control, right)] if row is None]))
                    continue
                tolerance = args['loss_tolerance']
                overshoots = [target - c['observed_loss'] for c in [left, right]]
                mismatch = abs(left['observed_loss'] - right['observed_loss'])
                if min(overshoots) < 0 or max(overshoots) > tolerance or mismatch > tolerance:
                    missing.append(dict(seed=seed, reason='loss_tolerance_exceeded',
                                        losses=[left['observed_loss'], right['observed_loss']], overshoots=overshoots))
                    continue
                pairs.append(dict(seed=seed, branch_step=left['step'], control_step=right['step'],
                                  branch_loss=left['observed_loss'], control_loss=right['observed_loss'],
                                  absolute_loss_mismatch=mismatch))
            for rep in ['encoder', 'projection']:
                row = dict(branch=name, control=control, target_loss=target, representation=rep,
                           available_seeds=[p['seed'] for p in pairs], missing=missing, matches=pairs)
                for metric in ['accuracy', 'onehot_mse']:
                    row[metric + '_branch_minus_control'] = describe([
                        evaluations[p['seed'], name][p['branch_step']][rep]['validation'][metric]
                        - evaluations[p['seed'], control][p['control_step']][rep]['validation'][metric] for p in pairs])
                matched.append(row)
    dataset_name = protocol.get('dataset_name', 'CIFAR-10')
    result = dict(dataset_name=dataset_name, n_seeds=len(seeds), seeds=seeds, paired_streams_and_initial_states_valid=True,
                  fixed_steps=fixed, loss_matched=matched, test_used=False,
                  interpretation='Exploratory single-task finite-sample OLS; seed-level descriptive summaries. Missing matches retained; no interpolation or automatic causal conclusion.')
    (directory / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = [f'# {dataset_name} 配对训练诊断', '',
             '验证集 OLS，按配对种子汇总。MSE 为主要终点，越低越好；准确率差用百分点。',
             '同 checkpoint 使用新配对随机流，不声称恢复旧轨迹。整体快慢与共同 loss 只是进度对照。', '',
             '| 分支 | 表示 | 最终准确率 % | 相对分叉 pp | 相对 baseline pp | 相对 baseline MSE |',
             '|---|---|---:|---:|---:|---:|']
    for row in fixed:
        if row['step'] == args['steps']:
            lines.append(f'| {row["branch"]} | {row["representation"]} | {100 * row["accuracy"]["mean"]:.3f} | '
                         f'{100 * row["accuracy_minus_fork"]["mean"]:+.3f} | {100 * row["accuracy_minus_baseline"]["mean"]:+.3f} | '
                         f'{row["onehot_mse_minus_baseline"]["mean"]:+.8f} |')
    lines += ['', '## 共同 loss（projection）', '',
              '| 分支 − 对照 | 目标 loss | 可用种子 | 缺失种子 | accuracy 差 pp | MSE 差 |',
              '|---|---:|---|---|---:|---:|']
    for row in matched:
        if row['representation'] != 'projection':
            continue
        accuracy = row['accuracy_branch_minus_control']['mean']
        mse = row['onehot_mse_branch_minus_control']['mean']
        accuracy_text = f'{100 * accuracy:+.3f}' if accuracy is not None else 'NA'
        mse_text = f'{mse:+.8f}' if mse is not None else 'NA'
        lines.append(f'| {row["branch"]} − {row["control"]} | {row["target_loss"]:g} | '
                     f'{row["available_seeds"]} | {[m["seed"] for m in row["missing"]]} | {accuracy_text} | {mse_text} |')
    lines += ['', '逐种子数值、实际 loss、超调和缺失原因见 summary.json。',
              '不自动据均值差宣布机制成立或失效；单个分类任务不能证明多任务取舍。', '']
    (directory / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=RESULTS / 'cifar' / 'checkpoint_branches_development')
    args = parser.parse_args()
    result = summarize(args.directory)
    print(f'Summarized {result["n_seeds"]} paired seeds: {args.directory / "summary.json"}')
