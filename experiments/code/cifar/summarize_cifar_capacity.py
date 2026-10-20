"""Summarize paired validation-only capacity diagnostics; run apart from Torch."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import argparse
import json
import statistics


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None, values=values)


def paired_difference(left, right, representation):
    return right[representation]['validation']['accuracy'] - left[representation]['validation']['accuracy']


def summarize(directory):
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    evaluations, runs = {}, {}
    for job in protocol['jobs']:
        path = directory / job['id']
        if not (path / 'result.json').exists() or not (path / 'evaluation.json').exists():
            raise ValueError(f"Incomplete planned run/evaluation: {job['id']}")
        run = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        evaluation = json.loads((path / 'evaluation.json').read_text(encoding='utf-8'))
        if run['job'] != job or evaluation['job'] != job:
            raise ValueError('Run metadata differs from locked plan')
        if run.get('test_used') or evaluation.get('test_used'):
            raise ValueError('These diagnostics must use training/validation only')
        if run['steps'] != protocol['arguments']['steps']:
            raise ValueError('Incomplete training budget')
        if [row['step'] for row in evaluation['fixed_steps']] != run['fixed_steps']:
            raise ValueError('Incomplete fixed-step evaluation')
        runs[job['id']], evaluations[job['id']] = run, evaluation
    seeds = sorted({job['seed'] for job in protocol['jobs']})
    # Check actual shared streams, and initial-function matching only within shape.
    for seed in seeds:
        group = [run for run in runs.values() if run['job']['seed'] == seed]
        for key in ['batch_index_sha256', 'augmentation_generator_final_sha256']:
            if len({run[key] for run in group}) != 1:
                raise ValueError(f'Paired random stream mismatch: {seed}/{key}')
        for architecture in {run['job']['architecture'] for run in group}:
            same_shape = [run for run in group if run['job']['architecture'] == architecture]
            if len({run['calibration']['initial_state_sha256'] for run in same_shape}) != 1:
                raise ValueError(f'Paired initialization mismatch: {seed}/{architecture}')
    groups = sorted({(j['objective'], j['architecture'], j['ratio']) for j in protocol['jobs']}, key=str)
    fixed = []
    for objective, architecture, ratio in groups:
        members = [r for r in evaluations.values() if (r['job']['objective'], r['job']['architecture'], r['job']['ratio']) == (objective, architecture, ratio)]
        for step in [r['step'] for r in members[0]['fixed_steps']]:
            rows = [next(r for r in member['fixed_steps'] if r['step'] == step) for member in members]
            initial = [next(r for r in member['fixed_steps'] if r['step'] == 0) for member in members]
            for representation in ['encoder', 'projection']:
                entry = dict(objective=objective, architecture=architecture, ratio=ratio, step=step,
                             representation=representation, seeds=[m['job']['seed'] for m in members])
                for split in ['train', 'validation']:
                    for metric in ['accuracy', 'onehot_mse', 'mean_true_class_margin']:
                        entry[split + '_' + metric] = describe([r[representation][split][metric] for r in rows])
                entry['validation_accuracy_change_from_initial'] = describe([paired_difference(a, b, representation) for a, b in zip(initial, rows)])
                for metric in ['participation_ratio', 'entropy_effective_rank', 'total_variance']:
                    entry[metric] = describe([r[representation]['training_feature_spectrum'][metric] for r in rows])
                entry['design_rank_with_intercept'] = describe([r[representation]['design_rank_with_intercept'] for r in rows])
                if objective != 'unsupervised':
                    entry['trained_classifier'] = {split: {metric: describe([r['trained_classifier'][split][metric] for r in rows])
                                                           for metric in ['accuracy', 'cross_entropy']} for split in ['train', 'validation']}
                fixed.append(entry)
    # Ratio comparisons use paired seeds, within each architecture only.
    ratios = sorted(protocol['arguments']['ratios'])
    unsupervised_architectures = sorted({r['job']['architecture'] for r in evaluations.values() if r['job']['objective'] == 'unsupervised'})
    ratio_effects, architecture_effects = [], []
    if len(ratios) > 1:
        for architecture in unsupervised_architectures:
            for level in [None, *protocol['arguments']['energy_levels']]:
                for representation in ['encoder', 'projection']:
                    values, mse_values, available, missing = [], [], [], []
                    for seed in seeds:
                        pair = []
                        for ratio in [ratios[0], ratios[-1]]:
                            member = next((r for r in evaluations.values() if r['job']['objective'] == 'unsupervised' and r['job']['architecture'] == architecture and r['job']['seed'] == seed and r['job']['ratio'] == ratio), None)
                            if member is None:
                                continue
                            row = member['fixed_steps'][-1] if level is None else next((e['metrics'] for e in member['energy_checkpoints'] if e['target_ell'] == level), None)
                            if row is not None:
                                pair.append(row)
                        if len(pair) == 2:
                            values.append(paired_difference(*pair, representation))
                            mse_values.append(pair[1][representation]['validation']['onehot_mse'] - pair[0][representation]['validation']['onehot_mse'])
                            available.append(seed)
                        else:
                            missing.append(seed)
                    ratio_effects.append(dict(architecture=architecture, target_ell=level,
                                             representation=representation, low_ratio=ratios[0], high_ratio=ratios[-1],
                                             available_seeds=available, missing_seeds=missing,
                                             high_minus_low_accuracy=describe(values), high_minus_low_onehot_mse=describe(mse_values)))
    for architecture in unsupervised_architectures:
        if architecture == 'baseline':
            continue
        for ratio in ratios:
            for representation in ['encoder', 'projection']:
                values, available = [], []
                for seed in seeds:
                    pair = [next((r for r in evaluations.values() if r['job']['objective'] == 'unsupervised' and r['job']['architecture'] == a and r['job']['seed'] == seed and r['job']['ratio'] == ratio), None) for a in ['baseline', architecture]]
                    if all(pair):
                        values.append(paired_difference(pair[0]['fixed_steps'][-1], pair[1]['fixed_steps'][-1], representation))
                        available.append(seed)
                architecture_effects.append(dict(architecture=architecture, ratio=ratio, representation=representation,
                                                 seeds=available, final_accuracy_minus_baseline=describe(values)))
    output = dict(protocol=protocol, run_count=len(evaluations), paired_streams_and_initializations_valid=True,
                  fixed_steps=fixed, ratio_effects=ratio_effects, architecture_effects=architecture_effects,
                  energy_observations=[dict(job=member['job'], **{key: row[key] for key in ['target_ell', 'observed_ell', 'step']})
                                       for member in evaluations.values() for row in member['energy_checkpoints']],
                  test_used=False, interpretation='Development diagnostics. Supervised results are attainable references, not capacity upper bounds. No cross-K common-loss equivalence.')
    (directory / 'summary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    final_step = protocol['arguments']['steps']
    lines = ['# CIFAR 容量诊断（开发阶段）', '',
             '仅训练 / 验证集；统计单位为配对种子，初始状态单列。有监督结果不构成理论容量上限。', '',
             '| 目标 | 架构 | r | 表示 | 训练 OLS % | 验证 OLS % | 相对初始变化 pp | 有效秩 |',
             '|---|---|---:|---|---:|---:|---:|---:|']
    for row in fixed:
        if row['step'] != final_step:
            continue
        lines.append(f"| {row['objective']} | {row['architecture']} | {row['ratio']} | {row['representation']} | {100*row['train_accuracy']['mean']:.2f} | {100*row['validation_accuracy']['mean']:.2f} | {100*row['validation_accuracy_change_from_initial']['mean']:+.2f} | {row['entropy_effective_rank']['mean']:.2f} |")
    lines += ['', '逐种子值、平方误差、margin、谱、共同能量缺失种子和配对差异见 summary.json；各运行的 evaluation.json 保留每类混淆矩阵及监督分类器的 CE / 准确率。',
              '不同 K 的总能量与 GUPA loss 不代表相同训练进度；共同能量配比比较仅在各架构内部解释。']
    (directory / 'findings.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


def plot_curves(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    architectures = sorted({r['architecture'] for r in output['fixed_steps'] if r['objective'] == 'unsupervised'})
    if not architectures:
        return
    fig, axes = plt.subplots(1, len(architectures), figsize=(5 * len(architectures), 3.8), squeeze=False, constrained_layout=True)
    for ax, architecture in zip(axes[0], architectures):
        for ratio in sorted({r['ratio'] for r in output['fixed_steps'] if r['objective'] == 'unsupervised'}):
            rows = sorted([r for r in output['fixed_steps'] if r['objective'] == 'unsupervised' and r['architecture'] == architecture and r['ratio'] == ratio and r['representation'] == 'projection'], key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows], [100 * r['validation_accuracy']['mean'] for r in rows], 'o-', label=f'r={ratio:g}')
        ax.set(title=architecture, xlabel='SGD updates', ylabel='Projection validation OLS accuracy (%)')
        ax.legend()
    fig.savefig(directory / 'capacity_validation.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=RESULTS / 'cifar' / 'capacity_development')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    output = summarize(args.directory)
    if not args.no_plots:
        plot_curves(args.directory, output)
    print(f"Summarized {output['run_count']} planned diagnostic runs; test_used=False")


if __name__ == '__main__':
    main()
