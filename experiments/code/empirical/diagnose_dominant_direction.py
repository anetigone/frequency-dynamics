"""Identify the dominant clean-feature direction using saved checkpoints; no training/download."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS
from diagnose_natural_images import load_images, read, standardized_design, write_csv


def image_statistics(images, mean, std):
    """Recover grayscale pixels; retain squared normalized mean because the encoder is quadratic."""
    import torch
    x = images.double()
    raw = x*std+mean
    average = raw.mean((1, 2, 3))
    second = raw.square().mean((1, 2, 3))
    variance = (second-average.square()).clamp_min(0)
    dx = raw[:, :, :, 1:]-raw[:, :, :, :-1]
    dy = raw[:, :, 1:, :]-raw[:, :, :-1, :]
    values = dict(pixel_mean=average, pixel_mean_squared=average.square(),
                  pixel_rms=second.sqrt(), pixel_mean_square=second,
                  pixel_std=variance.sqrt(), pixel_variance=variance,
                  gradient_energy=.5*(dx.square().mean((1, 2, 3))+dy.square().mean((1, 2, 3))),
                  normalized_mean_squared=x.mean((1, 2, 3)).square(),
                  normalized_mean_square=x.square().mean((1, 2, 3)))
    return torch.stack(list(values.values()), 1), list(values)


def pearson(a, b):
    a, b = a-a.mean(), b-b.mean()
    denominator = (a.square().sum()*b.square().sum()).sqrt()
    return float((a*b).sum()/denominator) if denominator > 0 else None


def nuisance_fits(markers, names, target, train, validation):
    """Fit PC1 from image statistics on train; validation is never used to choose a statistic."""
    import torch
    rows = []
    for name, columns in [(name, [i]) for i, name in enumerate(names)]+[('all_statistics_linear', list(range(len(names))))]:
        design = standardized_design(markers[:, columns], train)
        fit = torch.linalg.lstsq(design[train], target[train, None], driver='gelsd')
        prediction = (design@fit.solution)[:, 0]
        row = dict(statistic=name, design_rank=int(fit.rank))
        for split, index in [('train', train), ('validation', validation)]:
            denominator = (target[index]-target[train].mean()).square().sum()
            row[split+'_prediction_r2_train_mean_reference'] = float(1-(target[index]-prediction[index]).square().sum()/denominator) if denominator > 0 else None
            row[split+'_pearson'] = pearson(markers[index, columns[0]], target[index]) if len(columns)==1 else None
        rows.append(row)
    return rows


def pca_features(features, train):
    import torch
    mean = features[train].double().mean(0)
    centered = features.double()-mean
    _, singular, vh = torch.linalg.svd(centered[train], full_matrices=False)
    rank = int((singular > torch.finfo(centered.dtype).eps*max(centered[train].shape)*singular.max()).sum())
    basis = vh[:rank].T.contiguous()
    # Deterministic loading sign; no image statistic or labels select the orientation.
    if rank:
        sign = torch.sign(basis[basis.abs().argmax(0), torch.arange(rank)])
        basis = basis*sign
    scores = centered@basis
    variance = scores[train].square().mean(0)
    return scores, basis, mean, variance


def probe_components(scores, labels, train, validation, classes):
    import torch
    target = torch.nn.functional.one_hot(labels, classes).double()
    subsets = dict(all=scores, pc1=scores[:, :1], remainder=scores[:, 1:],
                   constant=scores[:, :0])
    constant = target[train].mean(0)
    rows = []
    for component, matrix in subsets.items():
        design = standardized_design(matrix, train) if matrix.shape[1] else torch.ones(len(matrix), 1, dtype=torch.float64)
        fit = torch.linalg.lstsq(design[train], target[train], driver='gelsd')
        prediction = design@fit.solution
        row = dict(component=component, dimensions=matrix.shape[1], design_rank=int(fit.rank))
        for split, index in [('train', train), ('validation', validation)]:
            mse = (prediction[index]-target[index]).square().mean()
            baseline = (target[index]-constant).square().mean()
            row[split+'_accuracy'] = float((prediction[index].argmax(1)==labels[index]).double().mean())
            row[split+'_onehot_mse'] = float(mse)
            row[split+'_constant_skill'] = float(1-mse/baseline) if baseline > 0 else None
        rows.append(row)
    return rows


def component_consistency(first, second, variance, train, validation):
    """Normalize view differences by the same checkpoint's TRAIN clean variance, never view variance."""
    rows = []
    for component, section in [('all', slice(None)), ('pc1', slice(0, 1)), ('remainder', slice(1, None))]:
        clean_var = variance[section]
        for split, index in [('train', train), ('validation', validation)]:
            a, b = first[index, section], second[index, section]
            per_coordinate = .5*(a-b).square().mean(0)
            total = clean_var.sum()
            correlations = [pearson(a[:, j], b[:, j]) for j in range(a.shape[1])]
            correlations = [v for v in correlations if v is not None]
            rows.append(dict(component=component, split=split, dimensions=len(clean_var),
                             clean_train_variance=float(total), alignment=float(per_coordinate.sum()),
                             alignment_over_clean_train_variance=float(per_coordinate.sum()/total) if total > 0 else None,
                             clean_whitened_alignment_per_dimension=float((per_coordinate/clean_var).mean()) if len(clean_var) else None,
                             mean_view_pearson=sum(correlations)/len(correlations) if correlations else None))
    return rows


def infer_features(data, kernel, device, batch, augment_fn=None, view_seed=0):
    import torch
    from torch.nn import functional as F
    clean, first, second = [], [], []
    generator = torch.Generator().manual_seed(view_seed)
    with torch.inference_mode():
        for start in range(0, len(data), batch):
            x = data[start:start+batch]
            clean.append(F.conv2d(x.to(device), kernel).square().mean((-2, -1)).cpu())
            if augment_fn is not None:
                # CPU augmentation uses the original dataset function and the same stream at every checkpoint.
                a, b = augment_fn(x, generator), augment_fn(x, generator)
                first.append(F.conv2d(a.to(device), kernel).square().mean((-2, -1)).cpu())
                second.append(F.conv2d(b.to(device), kernel).square().mean((-2, -1)).cpu())
    return tuple(torch.cat(items).double() if items else None for items in [clean, first, second])


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    protocol = read(args.suite/'protocol.json')
    if protocol.get('test_used'):
        raise ValueError('Expected validation-only source protocol')
    dataset = 'dtd' if protocol.get('dataset_name')=='DTD' else 'cifar'
    classes = 47 if dataset=='dtd' else 10
    data, labels, train, validation = load_images(args, dataset, protocol)
    split = protocol.get('split', {})
    mean_key, std_key = ('mean', 'std') if dataset=='dtd' else ('training_mean', 'training_std')
    # These exact normalization statistics must be saved by the source, not inferred from validation.
    mean, std = split[mean_key], split[std_key]
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0 or not bool(data.isfinite().all()):
        raise ValueError('Invalid image normalization/data')
    marker_batches = []
    for start in range(0, len(data), 256):
        batch_markers, names = image_statistics(data[start:start+256], mean, std)
        marker_batches.append(batch_markers)
    markers = torch.cat(marker_batches)
    if dataset=='dtd':
        from dtd_pilot import augment_dtd as augment
    else:
        from cifar_local import augment
    correlations, probes, consistency, pca_rows, evolution = [], [], [], [], []
    for job in protocol['jobs']:
        branch = job.get('name', job['id'])
        if args.seeds and job['seed'] not in args.seeds or args.branches and branch not in args.branches:
            continue
        directory = args.suite/job['id']
        result = read(directory/'result.json')
        if result.get('job') != job:
            raise ValueError(f'{directory}: job mismatch')
        snapshots = {s['step']: s['file'] for s in result.get('snapshots', []) if s.get('fixed')}
        if not snapshots:
            snapshots = {s: f'step_{s}.pt' for s in result['fixed_steps']}
        steps = args.steps if args.steps is not None else [0, max(snapshots)]
        reference = {}
        for step in sorted(set([0, *steps])):
            if step not in snapshots:
                raise ValueError(f'{directory}: no fixed step {step}')
            payload = torch.load(directory/snapshots[step], map_location='cpu', weights_only=True)
            if payload.get('job') != job or payload.get('branch_step', payload.get('step')) != step:
                raise ValueError(f'{directory}: checkpoint metadata mismatch')
            kernel, weight = payload['model']['kernel'].to(args.device), payload['model']['weight'].double()
            if kernel.ndim != 4 or weight.ndim != 2 or weight.shape[1] != kernel.shape[0] or not bool(kernel.isfinite().all() and weight.isfinite().all()):
                raise ValueError('Invalid model tensors')
            features = infer_features(data, kernel, args.device, args.feature_batch, augment, args.view_seed)
            for rep in ['encoder', 'projection']:
                x, a, b = features if rep=='encoder' else tuple(t@weight.T for t in features)
                if not all(bool(t.isfinite().all()) for t in [x, a, b]):
                    raise ValueError('Nonfinite inferred features')
                scores, basis, center, variance = pca_features(x, train)
                if not len(variance):
                    raise ValueError('Zero-rank representation')
                if step==0:
                    reference[rep] = (scores[:, 0].clone(), basis.clone(), center.clone(), variance.clone())
                if step not in steps:
                    continue
                identity = dict(job=job['id'], seed=job['seed'], branch=branch, step=step, representation=rep)
                correlations.extend(dict(**identity, **r) for r in nuisance_fits(markers, names, scores[:, 0], train, validation))
                probes.extend(dict(**identity, **r) for r in probe_components(scores, labels, train, validation, classes))
                va, vb = (a-center)@basis, (b-center)@basis
                consistency.extend(dict(**identity, **r) for r in component_consistency(va, vb, variance, train, validation))
                initial_score, initial_basis, initial_center, initial_variance = reference[rep]
                initial_axes_scores = (x-initial_center)@initial_basis
                for split_name, index in [('train', train), ('validation', validation)]:
                    evolution.append(dict(**identity, split=split_name,
                        initial_vs_current_pc1_score_pearson=pearson(initial_score[index], scores[index, 0]),
                        current_variance_on_initial_pc1=float(initial_axes_scores[index, 0].var(unbiased=False)),
                        current_total_variance=float(x[index].var(0, unbiased=False).sum()),
                        initial_pc1_train_variance=float(initial_variance[0])))
                pca_rows.append(dict(**identity, train_rank=len(variance), train_variance=variance.tolist(),
                                     train_pc1_variance_share=float(variance[0]/variance.sum()),
                                     train_mean=center.tolist(), pc1_loading=basis[:, 0].tolist()))
            print(f'Dominant-direction diagnosis: {job["id"]} step={step}', flush=True)
    if not pca_rows:
        raise ValueError('No checkpoints selected')
    return dict(dataset=dataset, source=str(args.suite.resolve()), view_seed=args.view_seed,
                augmentation_device='cpu', feature_batch=args.feature_batch,
                image_statistic_names=names, normalization=dict(mean=mean, std=std),
                scope='Exploratory clean TRAIN PCA, original split, train-fitted OLS; one paired augmentation stream shared across checkpoints; no causal identification.',
                nuisance_correlations=correlations, component_probes=probes,
                view_consistency=consistency, pca=pca_rows, pc1_evolution=evolution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--output', type=Path, help='Default: SUITE/dominant_direction; separate from previous diagnosis')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--branches', nargs='+', default=['baseline'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 12])
    parser.add_argument('--steps', nargs='+', type=int, help='Default: step 0 and final fixed checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--feature-batch', type=int, default=128)
    parser.add_argument('--view-seed', type=int, default=20261023)
    args = parser.parse_args()
    if min(args.threads, args.feature_batch)<1 or args.steps and min(args.steps)<0:
        parser.error('Positive threads/batch and nonnegative steps required')
    args.output = args.output or args.suite/'dominant_direction'
    if args.output.resolve()==args.suite.resolve():
        parser.error('Output must differ from source suite')
    result = run(args)
    serialized = json.dumps(result, indent=2, allow_nan=False)+'\n'
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ['nuisance_correlations', 'component_probes', 'view_consistency', 'pc1_evolution']:
        write_csv(args.output/f'{name}.csv', result[name])
    (args.output/'diagnostic.json').write_text(serialized, encoding='utf-8')
    lines = ['# Dominant-direction diagnosis', '', result['scope'], '',
             '## Clean PCA', '', '| Job | Step | Representation | Rank | PC1 variance share |', '|---|---:|---|---:|---:|']
    for r in result['pca']:
        lines.append(f'| {r["job"]} | {r["step"]} | {r["representation"]} | {r["train_rank"]} | {r["train_pc1_variance_share"]:.6g} |')
    lines += ['', '## Reading the outputs', '',
              '- nuisance_correlations.csv: every prespecified statistic is reported; combined and single-statistic LS are fitted on train only. Validation prediction R2 can be negative; Pearson squared is not out-of-sample prediction R2.',
              '- component_probes.csv: all, PC1, orthogonal remainder, and constant use the same train-label OLS rule. PCA is fitted without labels on clean train features, without column scaling. OLS subsequently standardizes its input columns for numerical conditioning.',
              '- view_consistency.csv: alignment divided by clean TRAIN variance removes global output scale. Whitened alignment averages variance-normalized error over retained clean PCs. PC1/remainder alignment sums to all alignment.',
              '- pc1_evolution.csv: compare current PC1 scores with initialization, and current variance on the fixed initial PC1 loading. Each checkpoint also gets its own train PCA; no validation-fitted PCA or component selection.',
              '- Strong correlation with image energy does not establish a nuisance or causal shortcut; inspect class probes and view consistency together. Tail probes may generalize poorly.',
              '- One paired view stream is a descriptive diagnostic, not an independent augmentation replication. Keep --feature-batch and --view-seed fixed for direct comparisons.', '']
    (args.output/'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Outputs: {args.output.resolve()}')


if __name__=='__main__':
    main()
