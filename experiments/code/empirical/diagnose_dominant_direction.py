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


def zero_dc_kernel(kernel):
    """Remove each input-channel spatial DC response without changing the source tensor."""
    return kernel-kernel.mean((-2, -1), keepdim=True)


def view_objective(first, second):
    """The saved experiment's GUPA formula on a fixed pair of views, in double precision."""
    import torch
    a, b = first.double(), second.double()
    if len(a)<2:
        raise ValueError('View covariance requires at least two images')
    ac, bc = a-a.mean(0), b-b.mean(0)
    covariance = (ac.T@ac+bc.T@bc)/(2*(len(a)-1))
    shifted = torch.eye(a.shape[1], dtype=a.dtype)+covariance
    alignment = .5*(a-b).square().sum(1).mean()
    logdet = -.5*torch.linalg.slogdet(shifted)[1]
    trace = -.5*torch.trace(torch.linalg.solve(shifted, covariance))
    return dict(loss=float(alignment+logdet+trace), alignment=float(alignment),
                uniformity_logdet=float(logdet), uniformity_trace=float(trace),
                view_covariance_eigenvalues=torch.linalg.eigvalsh(covariance).flip(0).tolist())


def ablation_comparisons(probes):
    rows = []
    original = {(r['job'], r['step'], r['representation'], r['component']): r
                for r in probes if r['variant']=='original'}
    for r in probes:
        if r['variant']!='zero_dc':
            continue
        base = original[r['job'], r['step'], r['representation'], r['component']]
        row = {k: r[k] for k in ['job', 'seed', 'branch', 'step', 'representation', 'component']}
        for split in ['train', 'validation']:
            for metric in ['accuracy', 'onehot_mse', 'constant_skill']:
                key = split+'_'+metric
                row[key+'_zero_dc_minus_original'] = r[key]-base[key] if r[key] is not None and base[key] is not None else None
        rows.append(row)
    return rows


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
    kernel_rows, objectives = [], []
    dc_ablation = getattr(args, 'dc_ablation', False)
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
            original_energy = None
            variants = [('original', kernel)]
            if dc_ablation:
                variants.append(('zero_dc', zero_dc_kernel(kernel)))
            for variant, active_kernel in variants:
                identity_base = dict(job=job['id'], seed=job['seed'], branch=branch, step=step, variant=variant)
                if step in steps and dc_ablation:
                    original_norm = float(kernel.double().square().sum())
                    kernel_rows.append(dict(**identity_base,
                        max_abs_spatial_sum=float(active_kernel.double().sum((-2, -1)).abs().max()),
                        squared_norm=float(active_kernel.double().square().sum()),
                        squared_norm_ratio_to_original=float(active_kernel.double().square().sum())/original_norm if original_norm else None))
                features = infer_features(data, active_kernel, args.device, args.feature_batch, augment, args.view_seed)
                for rep in ['encoder', 'projection']:
                    x, a, b = features if rep=='encoder' else tuple(t@weight.T for t in features)
                    if not all(bool(t.isfinite().all()) for t in [x, a, b]):
                        raise ValueError('Nonfinite inferred features')
                    scores, basis, center, variance = pca_features(x, train)
                    if step==0:
                        reference[variant, rep] = (scores[:, 0].clone() if len(variance) else None, basis.clone(), center.clone(), variance.clone())
                    if step not in steps:
                        continue
                    identity = dict(**identity_base, representation=rep)
                    if len(variance):
                        correlations.extend(dict(**identity, **r) for r in nuisance_fits(markers, names, scores[:, 0], train, validation))
                    probes.extend(dict(**identity, **r) for r in probe_components(scores, labels, train, validation, classes))
                    va, vb = (a-center)@basis, (b-center)@basis
                    consistency.extend(dict(**identity, **r) for r in component_consistency(va, vb, variance, train, validation))
                    initial_score, initial_basis, initial_center, initial_variance = reference[variant, rep]
                    initial_axes_scores = (x-initial_center)@initial_basis
                    for split_name, index in [('train', train), ('validation', validation)]:
                        evolution.append(dict(**identity, split=split_name,
                            initial_vs_current_pc1_score_pearson=pearson(initial_score[index], scores[index, 0]) if initial_score is not None and len(variance) else None,
                            current_variance_on_initial_pc1=float(initial_axes_scores[index, 0].var(unbiased=False)) if len(initial_variance) else None,
                            current_total_variance=float(x[index].var(0, unbiased=False).sum()),
                            initial_pc1_train_variance=float(initial_variance[0]) if len(initial_variance) else None))
                    pca_rows.append(dict(**identity, train_rank=len(variance), train_variance=variance.tolist(),
                                         train_pc1_variance_share=float(variance[0]/variance.sum()) if len(variance) else None,
                                         train_mean=center.tolist(), pc1_loading=basis[:, 0].tolist() if len(variance) else []))
                    if rep=='projection' and dc_ablation:
                        energy = float((x[train]-x[train].mean(0)).square().sum(1).mean())
                        if variant=='original':
                            original_energy = energy
                        scales = [('raw', 1.)]
                        if energy>0 and original_energy>0:
                            scales.append(('original_train_clean_energy', math.sqrt(original_energy/energy)))
                        for mode, scale in scales:
                            for split_name, index in [('train', train), ('validation', validation)]:
                                objectives.append(dict(**identity, split=split_name, scale_mode=mode,
                                    output_scale=scale, clean_train_energy=energy*scale**2,
                                    **view_objective(a[index]*scale, b[index]*scale)))
            print(f'Dominant-direction diagnosis: {job["id"]} step={step}', flush=True)
    if not pca_rows:
        raise ValueError('No checkpoints selected')
    return dict(dataset=dataset, source=str(args.suite.resolve()), view_seed=args.view_seed,
                augmentation_device='cpu', feature_batch=args.feature_batch,
                image_statistic_names=names, normalization=dict(mean=mean, std=std),
                scope='Exploratory clean TRAIN PCA, original split, train-fitted OLS; one paired augmentation stream shared across checkpoints. '+
                      ('Post-hoc DC ablation changes the function and is not a matched-function training intervention.' if dc_ablation else 'No causal identification.'),
                dc_ablation=dc_ablation,
                nuisance_correlations=correlations, component_probes=probes,
                view_consistency=consistency, pca=pca_rows, pc1_evolution=evolution,
                kernel_statistics=kernel_rows, view_objective=objectives,
                ablation_comparisons=ablation_comparisons(probes) if dc_ablation else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--output', type=Path, help='Default: SUITE/dominant_direction, or SUITE/dc_ablation with --dc-ablation')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--branches', nargs='+', default=['baseline'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 12])
    parser.add_argument('--steps', nargs='+', type=int, help='Default: step 0 and final fixed checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--feature-batch', type=int, default=128)
    parser.add_argument('--view-seed', type=int, default=20261023)
    parser.add_argument('--dc-ablation', action='store_true', help='Compare original vs spatially zero-mean kernels with the same head and paired views')
    args = parser.parse_args()
    if min(args.threads, args.feature_batch)<1 or args.steps and min(args.steps)<0:
        parser.error('Positive threads/batch and nonnegative steps required')
    args.output = args.output or args.suite/('dc_ablation' if args.dc_ablation else 'dominant_direction')
    if args.output.resolve()==args.suite.resolve():
        parser.error('Output must differ from source suite')
    result = run(args)
    serialized = json.dumps(result, indent=2, allow_nan=False)+'\n'
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ['nuisance_correlations', 'component_probes', 'view_consistency', 'pc1_evolution',
                 'kernel_statistics', 'view_objective', 'ablation_comparisons']:
        write_csv(args.output/f'{name}.csv', result[name])
    (args.output/'diagnostic.json').write_text(serialized, encoding='utf-8')
    lines = ['# Dominant-direction diagnosis', '', result['scope'], '',
             '## Clean PCA', '', '| Job | Step | Variant | Representation | Rank | PC1 variance share |', '|---|---:|---|---|---:|---:|']
    for r in result['pca']:
        share = format(r['train_pc1_variance_share'], '.6g') if r['train_pc1_variance_share'] is not None else 'NA'
        lines.append(f'| {r["job"]} | {r["step"]} | {r["variant"]} | {r["representation"]} | {r["train_rank"]} | {share} |')
    lines += ['', '## Reading the outputs', '',
              '- nuisance_correlations.csv: every prespecified statistic is reported; combined and single-statistic LS are fitted on train only. Validation prediction R2 can be negative; Pearson squared is not out-of-sample prediction R2.',
              '- component_probes.csv: all, PC1, orthogonal remainder, and constant use the same train-label OLS rule. PCA is fitted without labels on clean train features, without column scaling. OLS subsequently standardizes its input columns for numerical conditioning.',
              '- view_consistency.csv: alignment divided by clean TRAIN variance removes global output scale. Whitened alignment averages variance-normalized error over retained clean PCs. PC1/remainder alignment sums to all alignment.',
              '- pc1_evolution.csv: compare current PC1 scores with initialization, and current variance on the fixed initial PC1 loading. Each checkpoint also gets its own train PCA; no validation-fitted PCA or component selection.',
              '- Strong correlation with image energy does not establish a nuisance or causal shortcut; inspect class probes and view consistency together. Tail probes may generalize poorly.',
              '- One paired view stream is a descriptive diagnostic, not an independent augmentation replication. Keep --feature-batch and --view-seed fixed for direct comparisons.', '']
    if args.dc_ablation:
        lines += ['## DC ablation', '',
                  '- zero_dc subtracts each kernel spatial mean in memory; head, source checkpoint and inputs are unchanged. No kernel norm renormalization or retraining.',
                  '- Each variant fits its own train PCA and probes; component coordinates therefore differ. Full-representation paired differences are in ablation_comparisons.csv; train and validation metrics are both retained.',
                  '- view_objective.csv contains projection GUPA terms on all fixed training/validation views, not the original mini-batch audit. Raw loss changes with scale. original_train_clean_energy rescales outputs by one train-only scalar to match the original checkpoint clean variance; it does not match covariance shape or initial functions.',
                  '- kernel_statistics.csv records remaining spatial sums and kernel squared norm changes. Zero-rank ablated features yield constant probes and missing PC1 correlations rather than failure.', '']
        lines += ['| Job | Step | Representation | Validation accuracy change pp | Validation MSE change |',
                  '|---|---:|---|---:|---:|']
        for r in result['ablation_comparisons']:
            if r['component']=='all':
                lines.append(f'| {r["job"]} | {r["step"]} | {r["representation"]} | {100*r["validation_accuracy_zero_dc_minus_original"]:+.4f} | {r["validation_onehot_mse_zero_dc_minus_original"]:+.8g} |')
        lines.append('')
    (args.output/'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Outputs: {args.output.resolve()}')


if __name__=='__main__':
    main()
