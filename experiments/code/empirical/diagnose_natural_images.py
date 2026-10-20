"""Read saved natural-image runs; optionally diagnose existing checkpoints. No training/download."""
import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spectrum_values(values):
    if not values:
        return {}
    if any(not math.isfinite(v) or v < -1e-10 for v in values):
        raise ValueError('Invalid covariance spectrum')
    values = sorted((max(0., v) for v in values), reverse=True)
    total = sum(values)
    p = [v / total for v in values if v > 0] if total else []
    return dict(total_variance=total, top1_variance_share=p[0] if p else None,
                participation_ratio=1 / sum(v*v for v in p) if p else 0.,
                entropy_effective_rank=math.exp(-sum(v*math.log(v) for v in p)) if p else 0.)


def saved_records(suite, classes, seeds, branches):
    probes, history, issues = [], [], []
    directories = sorted(p for p in suite.iterdir() if p.is_dir() and
                         ((p / 'result.json').exists() or (p / 'evaluation.json').exists()))
    for directory in directories:
        result = read(directory / 'result.json') if (directory / 'result.json').exists() else {}
        evaluation = read(directory / 'evaluation.json') if (directory / 'evaluation.json').exists() else {}
        job = result.get('job', evaluation.get('job', {}))
        branch = job.get('name', job.get('id', directory.name))
        if seeds and job.get('seed') not in seeds or branches and branch not in branches:
            continue
        identity = dict(job=directory.name, seed=job.get('seed'), branch=branch)
        if not result:
            issues.append(f'{directory.name}: result.json missing')
        if not evaluation:
            issues.append(f'{directory.name}: evaluation.json missing')
        if result.get('job') and evaluation.get('job') and result['job'] != evaluation['job']:
            raise ValueError(f'{directory}: result/evaluation job mismatch')
        if result.get('test_used') or evaluation.get('test_used'):
            raise ValueError(f'{directory}: expected validation-only diagnostics')
        for item in result.get('history', []):
            row = dict(**identity, **{k: v for k, v in item.items() if isinstance(v, (int, float))})
            eig = item.get('view_covariance_eigenvalues')
            if eig is not None:
                row.update(spectrum_values(eig))
                row['view_covariance_eigenvalues'] = json.dumps(eig)
            if all(k in item for k in ['loss', 'alignment', 'uniformity_logdet', 'uniformity_trace']):
                row['loss_decomposition_residual'] = item['loss'] - sum(item[k] for k in
                    ['alignment', 'uniformity_logdet', 'uniformity_trace'])
            history.append(row)
        rows = evaluation.get('rows', evaluation.get('fixed_steps', []))
        for item in rows:
            if not all(rep in item for rep in ['encoder', 'projection']):
                continue
            enc, proj = item['encoder'], item['projection']
            delta = enc['train']['onehot_mse'] - proj['train']['onehot_mse']
            if delta > 1e-9:
                issues.append(f'{directory.name} step {item["step"]}: encoder train MSE exceeds projection by {delta:.6g}; check numerical solve')
            for rep, fit in [('encoder', enc), ('projection', proj)]:
                row = dict(**identity, step=item['step'], representation=rep,
                           encoder_minus_projection_train_mse=delta,
                           design_rank_with_intercept=fit.get('design_rank_with_intercept'))
                for split in ['train', 'validation']:
                    for key in ['accuracy', 'onehot_mse', 'mean_true_class_margin']:
                        if key in fit[split]:
                            row[f'{split}_{key}'] = fit[split][key]
                    row[f'{split}_balanced_constant_skill'] = 1-fit[split]['onehot_mse']/((classes-1)/classes**2)
                row['validation_minus_train_mse'] = fit['validation']['onehot_mse']-fit['train']['onehot_mse']
                singular = fit.get('singular_values', [])
                rank = fit.get('design_rank_with_intercept', 0)
                if singular:
                    row['singular_values'] = json.dumps(singular)
                    row['design_condition_at_reported_rank'] = singular[0]/singular[rank-1] if rank and singular[rank-1] > 0 else None
                spec = fit.get('training_feature_spectrum', {})
                row.update(spectrum_values(spec.get('covariance_eigenvalues', [])))
                if spec:
                    row['training_covariance_eigenvalues'] = json.dumps(spec.get('covariance_eigenvalues', []))
                probes.append(row)
    for rows, grouping, keys in [(probes, ('job', 'representation'), ('train_onehot_mse', 'validation_onehot_mse', 'validation_accuracy')),
                                  (history, ('job',), ('loss', 'alignment', 'uniformity_logdet', 'uniformity_trace', 'clean_energy', 'energy'))]:
        initial = {}
        for row in sorted(rows, key=lambda r: r['step']):
            key = tuple(row[k] for k in grouping)
            start = initial.setdefault(key, row.copy())
            row['reference_step'] = start['step']
            for metric in keys:
                if metric in row and metric in start:
                    row[metric+'_minus_reference'] = row[metric]-start[metric]
            for energy in ['clean_energy', 'energy']:
                if row.get(energy) is not None and start.get(energy, 0) > 0:
                    row[energy+'_ratio_to_reference'] = row[energy]/start[energy]
    return probes, history, issues


def load_images(args, dataset, protocol):
    """Use exactly the prepared DTD split or the existing CIFAR 45k/5k split."""
    import torch
    if dataset == 'dtd':
        from dtd_hf import load_prepared
        size = protocol.get('arguments', {}).get('image_size', 64)
        root = args.data_root or EXPERIMENTS / 'data' / 'dtd_hf'
        data, labels, train, validation, metadata = load_prepared(root, size)
        split = protocol.get('split', {})
        for key, observed in [('mean', metadata['mean']), ('std', metadata['std']),
                              ('train_size', len(train)), ('validation_size', len(validation))]:
            if key in split and not math.isclose(split[key], observed, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f'DTD {key} differs from saved protocol')
        return data, labels, train, validation
    # Only the five existing training batches; never open test_batch or download.
    root = args.data_root or EXPERIMENTS.parent
    directory = root if (root / 'data_batch_1').is_file() else root / 'cifar-10-batches-py'
    import numpy as np
    batches = []
    targets = []
    for i in range(1, 6):
        with (directory / f'data_batch_{i}').open('rb') as stream:
            payload = pickle.load(stream, encoding='bytes')
        batches.append(payload[b'data'])
        targets.extend(payload[b'labels'])
    rgb = torch.from_numpy(np.concatenate(batches)).reshape(-1, 3, 32, 32).float()/255
    labels = torch.tensor(targets)
    if rgb.shape != (50000, 3, 32, 32) or not torch.equal(torch.bincount(labels), torch.full((10,), 5000)):
        raise ValueError('Unexpected CIFAR training data shape/class counts')
    data = rgb[:, :1]*.2989 + rgb[:, 1:2]*.587 + rgb[:, 2:3]*.114
    split = protocol.get('split', {})
    generator = torch.Generator().manual_seed(split.get('split_seed', 20261019))
    train, validation = [], []
    for label in range(10):
        index = torch.where(labels == label)[0]
        index = index[torch.randperm(len(index), generator=generator)]
        train.append(index[:4500])
        validation.append(index[4500:])
    train, validation = torch.cat(train), torch.cat(validation)
    mean, std = float(data[train].mean()), float(data[train].std())
    for key, value in [('training_mean', mean), ('training_std', std), ('train_size', len(train)), ('validation_size', len(validation))]:
        if key in split and not math.isclose(value, split[key], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f'CIFAR {key} differs from saved protocol')
    return (data-mean)/std, labels, train, validation


def standardized_design(x, train):
    import torch
    x = x.double()
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-12)
    return torch.cat([torch.ones(len(x), 1, dtype=x.dtype), (x-mean)/std], 1)


def functional_change(initial, current, train, validation):
    """Compare the same image rows; fit on training only, evaluate on existing validation."""
    import torch
    a, b = initial.double(), current.double()
    # Best shared scalar plus per-coordinate intercept: narrower than arbitrary remixing.
    ac, bc = a[train]-a[train].mean(0), b[train]-b[train].mean(0)
    alpha = (ac*bc).sum()/ac.square().sum().clamp_min(1e-30)
    scalar_prediction = (a-a[train].mean(0))*alpha+b[train].mean(0)
    design = standardized_design(a, train)
    fit = torch.linalg.lstsq(design[train], b[train], driver='gelsd').solution
    prediction = design@fit
    result = dict(shared_scale=float(alpha))
    for split, index in [('train', train), ('validation', validation)]:
        denominator = (b[index]-b[train].mean(0)).square().sum()
        result[f'{split}_scalar_relative_squared_residual'] = float((scalar_prediction[index]-b[index]).square().sum()/denominator) if denominator > 0 else None
        result[f'{split}_affine_relative_squared_residual'] = float((prediction[index]-b[index]).square().sum()/denominator) if denominator > 0 else None
        bases = []
        for x in [a[index], b[index]]:
            x = x-x.mean(0)
            u, singular, _ = torch.linalg.svd(x, full_matrices=False)
            tolerance = torch.finfo(x.dtype).eps*max(x.shape)*singular.max()
            bases.append(u[:, singular > tolerance])
        u, v = bases
        overlap = u.T@v
        result[f'{split}_initial_rank'] = u.shape[1]
        result[f'{split}_current_rank'] = v.shape[1]
        result[f'{split}_sample_projector_distance'] = math.sqrt(max(0., u.shape[1]+v.shape[1]-2*float(overlap.square().sum())))
        # Cross-Gram singular values are cosines of principal angles; no n-by-n projector.
        cosines = torch.linalg.svdvals(overlap).clamp(0, 1)
        result[f'{split}_largest_principal_angle_degrees'] = float(torch.rad2deg(torch.acos(cosines.min()))) if len(cosines) else None
    return result


def head_metrics(features, projection, labels, train, validation, classes):
    """Full affine OLS, original head OLS and global rank-K affine LS, without SGD."""
    import torch
    y = torch.nn.functional.one_hot(labels, classes).double()
    x = standardized_design(features, train)[:, 1:]
    u, singular, vh = torch.linalg.svd(x[train], full_matrices=False)
    rank = int((singular > torch.finfo(x.dtype).eps*max(x[train].shape)*singular.max()).sum())
    target_mean = y[train].mean(0)
    coefficient = u[:, :rank].T@(y[train]-target_mean)
    left, values, right = torch.linalg.svd(coefficient, full_matrices=False)
    k = min(projection.shape[1], len(values))
    full = (vh[:rank].T/singular[:rank])@coefficient
    reduced = (vh[:rank].T/singular[:rank])@(left[:, :k]@(values[:k, None]*right[:k]))
    design = standardized_design(projection, train)
    predictions = dict(encoder_full_ols=x@full+target_mean,
                       encoder_rank_k_ls=x@reduced+target_mean,
                       original_projection_ols=design@torch.linalg.lstsq(design[train], y[train], driver='gelsd').solution)
    rows = []
    for method, scores in predictions.items():
        row = dict(method=method, k=projection.shape[1], encoder_rank=rank)
        for name, index in [('train', train), ('validation', validation)]:
            row[f'{name}_accuracy'] = float((scores[index].argmax(1)==labels[index]).double().mean())
            row[f'{name}_onehot_mse'] = float((scores[index]-y[index]).square().mean())
        rows.append(row)
    return rows


def checkpoint_analysis(args, dataset, protocol):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(args.threads)
    data, labels, train, validation = load_images(args, dataset, protocol)
    if not bool(data.isfinite().all()):
        raise ValueError('Nonfinite image tensor')
    changes, heads = [], []
    jobs = protocol.get('jobs', [])
    if not jobs:
        raise ValueError('--checkpoints requires a protocol.json with jobs')
    for job in jobs:
        branch = job.get('name', job['id'])
        if args.seeds and job['seed'] not in args.seeds or args.branches and branch not in args.branches:
            continue
        folder = args.suite/job['id']
        result = read(folder/'result.json')
        snapshots = {s['step']: s['file'] for s in result.get('snapshots', []) if s.get('fixed')}
        if not snapshots:
            snapshots = {s: f'step_{s}.pt' for s in result['fixed_steps']}
        steps = args.steps if args.steps is not None else sorted(snapshots)
        for step in [0, *steps]:
            if step not in snapshots:
                raise ValueError(f'{job["id"]}: fixed checkpoint step {step} not saved')
        initial = None
        for step in sorted(set([0, *steps])):
            path = folder/snapshots[step]
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('job') != job or payload.get('branch_step', payload.get('step')) != step:
                raise ValueError(f'{path}: job/step metadata mismatch')
            state = payload['model']
            kernel, weight = state['kernel'].to(args.device), state['weight'].double()
            if kernel.ndim != 4 or weight.ndim != 2 or weight.shape[1] != kernel.shape[0]:
                raise ValueError(f'{path}: incompatible model shape')
            if not bool(kernel.isfinite().all() and weight.isfinite().all()):
                raise ValueError(f'{path}: nonfinite model parameters')
            with torch.inference_mode():
                features = torch.cat([F.conv2d(data[i:i+args.feature_batch].to(args.device), kernel).square().mean((-2, -1)).cpu()
                                      for i in range(0, len(data), args.feature_batch)]).double()
                projected = features@weight.T
            if not bool(features.isfinite().all() and projected.isfinite().all()):
                raise ValueError(f'{path}: nonfinite features')
            if step == 0:
                initial = (features, projected)
            if step in steps:
                identity = dict(job=job['id'], seed=job['seed'], branch=branch, step=step)
                for rep, start, now in [('encoder', initial[0], features), ('projection', initial[1], projected)]:
                    changes.append(dict(**identity, representation=rep, **functional_change(start, now, train, validation)))
                heads.extend(dict(**identity, **row) for row in head_metrics(features, projected, labels, train, validation, 47 if dataset == 'dtd' else 10))
                print(f'Diagnosed {job["id"]} step={step}', flush=True)
    if not changes:
        raise ValueError('No checkpoints selected; check --seeds/--branches')
    return changes, heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path, help='Existing dtd_pilot, CIFAR branch or capacity result directory')
    parser.add_argument('--output', type=Path, help='Default: SUITE/diagnosis; original files are not changed')
    parser.add_argument('--dataset', choices=['auto', 'dtd', 'cifar'], default='auto')
    parser.add_argument('--checkpoints', action='store_true', help='Also run inference and closed-form LS on saved checkpoints; never train')
    parser.add_argument('--data-root', type=Path, help='DTD prepared tensor folder, or CIFAR training-batch parent')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--feature-batch', type=int, default=128)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: all saved seeds')
    parser.add_argument('--branches', nargs='+', help='Branch name or capacity job id; default: all saved jobs')
    parser.add_argument('--steps', type=int, nargs='+', help='Checkpoint mode only; default: all fixed steps; reference is always step 0')
    args = parser.parse_args()
    if not args.suite.is_dir():
        parser.error('suite directory does not exist')
    if min(args.feature_batch, args.threads) < 1 or args.steps and min(args.steps) < 0:
        parser.error('Positive batch/threads and nonnegative steps required')
    args.output = args.output or args.suite/'diagnosis'
    if args.output.resolve() == args.suite.resolve():
        parser.error('--output must differ from the source suite')
    protocol = read(args.suite/'protocol.json') if (args.suite/'protocol.json').exists() else {}
    summary = read(args.suite/'summary.json') if (args.suite/'summary.json').exists() else {}
    if protocol.get('test_used'):
        parser.error('Expected a validation-only source protocol')
    detected = protocol.get('dataset_name', summary.get('dataset_name', 'CIFAR-10'))
    dataset = ('dtd' if detected == 'DTD' else 'cifar') if args.dataset == 'auto' else args.dataset
    if args.dataset != 'auto' and detected == 'DTD' and dataset != 'dtd':
        parser.error('Dataset differs from saved DTD metadata')
    classes = 47 if dataset == 'dtd' else 10
    probes, history, issues = saved_records(args.suite, classes, args.seeds, args.branches)
    changes, heads = [], []
    if args.checkpoints:
        changes, heads = checkpoint_analysis(args, dataset, protocol)
    if not probes:
        issues.append('No per-seed evaluation records found; summary-only fallback cannot diagnose train/validation gaps or numerical spectra.')
    aggregate = []
    for row in summary.get('fixed_steps', []):
        seed_values = row.get('seeds', [])
        branch = row.get('branch', row.get('architecture', 'unknown'))
        if args.branches and branch not in args.branches:
            continue
        for i, seed in enumerate(seed_values):
            if args.seeds and seed not in args.seeds:
                continue
            item = dict(seed=seed, branch=branch, step=row['step'], representation=row['representation'])
            for key in ['objective', 'architecture', 'ratio']:
                if key in row:
                    item[key] = row[key]
            for metric in ['accuracy', 'onehot_mse', 'audit_loss', 'audit_clean_energy', 'validation_accuracy', 'validation_onehot_mse']:
                if metric in row:
                    item[metric] = row[metric]['values'][i]
            aggregate.append(item)
    if not probes and not history and not aggregate:
        raise ValueError('No readable diagnostic records found in suite')
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = dict(probes=probes, audit_history=history, checkpoint_changes=changes,
                   checkpoint_heads=heads, summary_values=aggregate)
    for name, rows in outputs.items():
        write_csv(args.output/f'{name}.csv', rows)
    content = dict(dataset=dataset, source=str(args.suite.resolve()),
                   balanced_constant_mse=(classes-1)/classes**2, issues=issues,
                   scope='Existing validation split only; no training/download/hash computation. Seed rows, not independent checkpoints.', **outputs)
    (args.output/'diagnostic.json').write_text(json.dumps(content, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [f'# {dataset.upper()} saved-run diagnosis', '',
             f'Balanced constant MSE: {(classes-1)/classes**2:.12g}.', '',
             f'Probe rows: {len(probes)}; audit rows: {len(history)}; functional comparisons: {len(changes)}.', '',
             '## Findings', '']
    for issue in issues:
        lines.append('- '+issue)
    if probes:
        finals = {}
        for row in probes:
            key = (row['job'], row['representation'])
            if key not in finals or row['step'] > finals[key]['step']:
                finals[key] = row
        lines += ['', '| Job | Representation | Final step | Train MSE | Validation MSE | Validation accuracy % | Constant skill |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for row in finals.values():
            lines.append(f'| {row["job"]} | {row["representation"]} | {row["step"]} | {row["train_onehot_mse"]:.8g} | {row["validation_onehot_mse"]:.8g} | {100*row["validation_accuracy"]:.4f} | {row["validation_balanced_constant_skill"]:.6g} |')
    lines += ['', '## Interpretation', '',
              '- Positive encoder-minus-projection training MSE above 1e-9 needs numerical evaluation checks. Validation ordering may reverse through generalization error.',
              '- Compare loss decomposition with top1 variance share and the full saved spectra. Energy growth alone does not establish pure scaling.',
              '- A small scalar residual is compatible with shared scaling; a small affine residual allows feature remixing. Sample projector distance measures column spaces on the same images, not the theorem frequency-coordinate projector.',
              '- Principal angles cover the smaller-rank space; inspect both ranks and projector distance when ranks differ.',
              '- Rank-K fitting uses training labels and is a diagnostic reference, not an unsupervised method. No automatic causal verdict or seed-level significance test is produced.',
              '- Summary fallback retains only available saved seed values. Missing raw train metrics and loss components remain missing.', '']
    (args.output/'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Diagnostic outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
