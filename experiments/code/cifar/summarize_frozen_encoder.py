"""Paired-seed summary of heads fitted to identical frozen CIFAR encoders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import argparse
import json
import statistics


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values)>1 else None, values=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=RESULTS/'cifar'/'frozen_encoder_heads')
    args = parser.parse_args()
    protocol = json.loads((args.directory/'protocol.json').read_text(encoding='utf-8'))
    results = []
    for job in protocol['jobs']:
        path = args.directory/job['id']/'result.json'
        if not path.exists():
            raise ValueError(f'Incomplete frozen-head run: {job["id"]}')
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['job'] != job or result['steps'] != protocol['arguments']['steps']:
            raise ValueError('Run differs from the planned source/budget')
        if not result['optimization']['encoder_unchanged_exactly'] or result['test_used']:
            raise ValueError('Encoder changed or test labels used')
        results.append(result)
    rows = []
    for step in sorted({r['job']['source_step'] for r in results}):
        group = [r for r in results if r['job']['source_step']==step]
        for method in ['encoder_full_ols','source_projection_ols','rank8_ls','trained_projection_ols','trained_classifier']:
            row = dict(source_step=step, method=method, seeds=[r['job']['seed'] for r in group])
            for split in ['train','validation']:
                metrics = [r['reduced_rank']['direct_'+split] if method=='rank8_ls' else r[method][split] for r in group]
                row[split+'_accuracy'] = describe([m['accuracy'] for m in metrics])
                loss = 'cross_entropy' if method=='trained_classifier' else 'onehot_mse'
                row[split+'_'+loss] = describe([m[loss] for m in metrics])
            row['validation_accuracy_gain_over_source_projection'] = describe([
                (r['reduced_rank']['direct_validation']['accuracy'] if method=='rank8_ls' else r[method]['validation']['accuracy'])
                - r['source_projection_ols']['validation']['accuracy'] for r in group])
            rows.append(row)
    summary = dict(protocol=protocol, count=len(results), rows=rows, test_used=False,
                   encoder_frozen_in_all_runs=True,
                   note='Same source encoder within each seed/source-step comparison. Rank-8 LS optimizes TRAIN MSE, not validation accuracy.')
    (args.directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 冻结 encoder 的 head 对照','','每个 source step / seed 内 encoder 完全相同；仅 projection 和分类器可训练。',
           'rank8_ls 是训练平方误差最优参照，不是验证准确率上限。','','| encoder 来源步数 | head / probe | n | 训练准确率 % | 验证准确率 % | 相对原 head pp |',
           '|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['source_step']} | {row['method']} | {len(row['seeds'])} | {100*row['train_accuracy']['mean']:.2f} | {100*row['validation_accuracy']['mean']:.2f} | {100*row['validation_accuracy_gain_over_source_projection']['mean']:+.2f} |")
    lines+=['','OLS 与训练出来的 CE 分类器分开报告。仅使用训练标签拟合，验证集不参与求解、选停或调参。',
            '逐种子值和损失见 summary.json；原模型 kernel 未更新的精确检查见各 result.json。']
    (args.directory/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Summarized {len(results)} frozen-encoder states')


if __name__=='__main__':
    main()
