"""Task-conditioned local risk predictions and paired short CIFAR SGD branches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import argparse
import copy
import json
import math
import statistics

import torch
from cifar_local import augment
from cifar_checkpoint_branches import (load_source, source_metadata, source_paths,
                                      learning_rates, read_json, write_json)
from diagnose_natural_images import load_images
from square_images import empirical_loss


def probe(z, labels, train, validation, classes=10):
    """Differentiable full-rank affine OLS; includes refitting in validation derivative."""
    z = z.double()
    mean = z[train].mean(0)
    scale = z[train].std(0).clamp_min(1e-12)
    design = torch.cat((torch.ones_like(z[:, :1]), (z - mean) / scale), 1)
    x = design[train]
    singular = torch.linalg.svdvals(x.detach())
    rank = int((singular > torch.finfo(x.dtype).eps * max(x.shape) * singular[0]).sum())
    if rank != x.shape[1]:
        raise ValueError('Rank-deficient OLS: the full-rank local derivative is undefined here')
    target = torch.nn.functional.one_hot(labels, classes).double()
    q, r = torch.linalg.qr(x, mode='reduced')
    coefficient = torch.linalg.solve_triangular(r, q.T @ target[train], upper=True)
    scores = design @ coefficient
    risks = {name: (scores[index] - target[index]).square().mean()
             for name, index in [('train', train), ('validation', validation)]}
    metrics = {name: dict(onehot_mse=float(risks[name].detach()),
                         accuracy=float((scores[index].argmax(1) == labels[index]).double().mean()))
               for name, index in [('train', train), ('validation', validation)]}
    metrics['design_rank'] = rank
    metrics['design_condition'] = float(singular[0] / singular[-1])
    return risks['validation'], metrics


@torch.no_grad()
def clean_projection(model, data, batch):
    return torch.cat([model(data[i:i+batch].to(device=model.kernel.device, dtype=model.kernel.dtype)).cpu()
                      for i in range(0, len(data), batch)])


def evaluate(model, data, labels, train, validation, batch):
    with torch.no_grad():
        return probe(clean_projection(model, data, batch), labels, train, validation)[1]


def risk_gradient(model, data, labels, train, validation, batch):
    # Only the small CPU feature matrix retains the QR graph. Replay convolution in
    # chunks with its feature adjoint, avoiding a graph for all 50k images on GPU.
    z = clean_projection(model, data, batch).requires_grad_(True)
    risk, metrics = probe(z, labels, train, validation)
    adjoint, = torch.autograd.grad(risk, z)
    parameters = (model.kernel, model.weight)
    gradients = [torch.zeros_like(p) for p in parameters]
    for i in range(0, len(data), batch):
        output = model(data[i:i+batch].to(device=model.kernel.device, dtype=model.kernel.dtype))
        part = torch.autograd.grad((output * adjoint[i:i+batch].to(output.device)).sum(), parameters)
        for total, value in zip(gradients, part):
            total.add_(value)
    return gradients, metrics


def training_gradient(model, data, train, generator, batch, batches):
    parameters = (model.kernel, model.weight)
    gradients = [torch.zeros_like(p) for p in parameters]
    losses = []
    # Keep the dataset on CPU; only actual batches and their views reside on GPU.
    for _ in range(batches):
        index = train[torch.randint(len(train), (batch,), generator=generator)]
        first, second = augment(data[index], generator), augment(data[index], generator)
        first, second = [v.to(device=model.kernel.device, dtype=model.kernel.dtype) for v in (first, second)]
        loss = empirical_loss(model(first), model(second))
        part = torch.autograd.grad(loss, parameters)
        for total, value in zip(gradients, part):
            total.add_(value / batches)
        losses.append(float(loss.detach()))
    if not all(bool(g.isfinite().all()) for g in gradients):
        raise FloatingPointError('Nonfinite training gradient')
    return gradients, sum(losses) / len(losses)


@torch.no_grad()
def update(model, directions, multiplier=1.):
    for parameter, direction in zip((model.kernel, model.weight), directions):
        parameter.add_(direction, alpha=multiplier)
    if not all(bool(p.isfinite().all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite updated parameters')


def finite_difference(model, directions, analytic, data, labels, train, validation, args):
    rows = []
    for block in range(2):
        direction = [torch.zeros_like(v) for v in directions]
        direction[block] = directions[block]
        for epsilon in args.fd_scales:
            values = []
            for sign in [-1, 1]:
                changed = copy.deepcopy(model)
                update(changed, direction, sign * epsilon)
                values.append(evaluate(changed, data, labels, train, validation,
                                       args.feature_batch)['validation']['onehot_mse'])
            observed = (values[1] - values[0]) / (2 * epsilon)
            error = abs(observed - analytic[block])
            rows.append(dict(block=['encoder', 'head'][block], epsilon=epsilon,
                             analytic=analytic[block], central_difference=observed,
                             absolute_error=error,
                             passed=error <= args.fd_atol + args.fd_rtol * abs(analytic[block])))
    return rows


def choose_branches(a, b, fd_pass, args):
    threshold = -a / b if a * b < 0 else None
    branches = [dict(name='baseline', ratio=1., speed=1.),
                dict(name='slow', ratio=1., speed=.5),
                dict(name='fast', ratio=1., speed=2.),
                dict(name='ratio_0.1', ratio=.1, speed=1.),
                dict(name='ratio_10', ratio=10., speed=1.)]
    reason = 'no positive threshold'
    if threshold is not None:
        if not fd_pass:
            reason = 'finite-difference check failed; no threshold-selected branches'
        elif not (args.ratio_bounds[0] <= threshold / 2 and threshold * 2 <= args.ratio_bounds[1]):
            reason = 'threshold bracket outside preset ratio bounds'
        elif min(abs(a), abs(b)) <= args.signal_floor:
            reason = 'block contribution below preset numerical signal floor'
        else:
            reason = 'eligible'
            branches += [dict(name='below_threshold', ratio=threshold / 2, speed=1.),
                         dict(name='above_threshold', ratio=threshold * 2, speed=1.)]
    return threshold, reason, branches


def run_repeat(args, seed, source_step, repeat, model, source, data, labels, train, validation,
               risk_gradients, initial):
    directory = args.output / f'seed_{seed}_step_{source_step}_repeat_{repeat}'
    if (directory / 'result.json').exists():
        return read_json(directory / 'result.json')
    stream_seed = args.future_seed + seed * 1000 + repeat
    generator = torch.Generator().manual_seed(stream_seed)
    gradients, source_loss = training_gradient(model, data, train, generator, args.batch_size, args.gradient_batches)
    # First update uses this same averaged gradient in every branch. Subsequent
    # updates use ordinary single minibatches from the same reset future stream.
    base = args.lr * source['common_mobility']
    directions = [-base * g for g in gradients]
    coefficients = [float((r * d).sum()) for r, d in zip(risk_gradients, directions)]
    fd = finite_difference(model, directions, coefficients, data, labels, train, validation, args)
    fd_pass = all(row['passed'] for row in fd)
    a, b = coefficients
    threshold, reason, branches = choose_branches(a, b, fd_pass, args)
    prediction = dict(seed=seed, source_step=source_step, repeat=repeat, stream_seed=stream_seed,
                      source=source, initial=initial, source_unsupervised_loss=source_loss,
                      encoder_coefficient=a, head_coefficient=b, positive_threshold=threshold,
                      eligibility=reason, finite_difference=fd, branches=branches,
                      gradient_norms=[float(g.norm()) for g in gradients],
                      base_relative_parameter_updates=[float(d.norm()/p.detach().norm().clamp_min(1e-30))
                                                       for d, p in zip(directions, (model.kernel, model.weight))])
    write_json(directory / 'prediction.json', prediction)  # Before any branch outcomes.
    outcomes = []
    for branch in branches:
        changed = copy.deepcopy(model)
        future = torch.Generator().manual_seed(stream_seed)
        rates = learning_rates(args.lr, source['common_mobility'], branch)
        predicted = branch['speed'] * 2 * (a + branch['ratio'] * b) / (1 + branch['ratio'])
        for step in range(1, max(args.horizons) + 1):
            grad, loss = training_gradient(changed, data, train, future, args.batch_size,
                                           args.gradient_batches if step == 1 else 1)
            update(changed, [-rate * g for rate, g in zip(rates, grad)])
            if step in args.horizons:
                metrics = evaluate(changed, data, labels, train, validation, args.feature_batch)
                delta = metrics['validation']['onehot_mse'] - initial['validation']['onehot_mse']
                target = step * predicted
                sign_match = (delta * target > 0) if min(abs(delta), abs(target)) > args.signal_floor else None
                outcomes.append(dict(**branch, step=step, metrics=metrics, observed_delta=delta,
                                     predicted_first_step_delta=predicted,
                                     frozen_tangent_extrapolation=target,
                                     residual_from_frozen_tangent=delta-target,
                                     sign_match=sign_match, prediction_valid=fd_pass,
                                     last_pre_update_batch_loss=loss,
                                     accuracy_delta=metrics['validation']['accuracy']-initial['validation']['accuracy']))
        print(f'seed={seed} source={source_step} repeat={repeat} {branch["name"]} complete', flush=True)
    for row in outcomes:
        baseline = next(o for o in outcomes if o['name'] == 'baseline' and o['step'] == row['step'])
        row['delta_minus_baseline'] = row['observed_delta'] - baseline['observed_delta']
        row['accuracy_delta_minus_baseline'] = row['accuracy_delta'] - baseline['accuracy_delta']
    reversals = []
    if reason == 'eligible':
        for horizon in args.horizons:
            pair = [next(o for o in outcomes if o['name'] == name and o['step'] == horizon)
                    for name in ['below_threshold', 'above_threshold']]
            resolved = all(abs(o['observed_delta']) > args.signal_floor for o in pair)
            reversals.append(dict(step=horizon, resolved=resolved,
                                  observed_opposite_signs=pair[0]['observed_delta']*pair[1]['observed_delta'] < 0 if resolved else None,
                                  both_predicted_signs_match=all(o['sign_match'] is True for o in pair) if resolved else None))
    result = dict(prediction=prediction, outcomes=outcomes, threshold_reversals=reversals, test_used=False)
    write_json(directory / 'result.json', result)
    return result


def summarize(output):
    runs = [read_json(p) for p in sorted(output.glob('seed_*_step_*_repeat_*/result.json'))]
    protocol = read_json(output / 'protocol.json')['arguments']
    expected = len(protocol['seeds']) * len(protocol['source_steps']) * protocol['repeats']
    per_seed = []
    for seed in protocol['seeds']:
        selected = [r for r in runs if r['prediction']['seed'] == seed]
        if not selected:
            continue
        per_seed.append(dict(seed=seed, states_and_repeats=len(selected),
                             positive_threshold_fraction=sum(r['prediction']['positive_threshold'] is not None for r in selected)/len(selected),
                             eligible_fraction=sum(r['prediction']['eligibility'] == 'eligible' for r in selected)/len(selected),
                             fd_pass_fraction=sum(all(x['passed'] for x in r['prediction']['finite_difference']) for r in selected)/len(selected)))
    cells = []
    keys = sorted({(r['prediction']['source_step'], o['name'], o['step']) for r in runs for o in r['outcomes']})
    for source_step, branch, horizon in keys:
        values = []
        for seed in protocol['seeds']:
            rows = [o for r in runs if r['prediction']['seed'] == seed and r['prediction']['source_step'] == source_step
                    for o in r['outcomes'] if o['name'] == branch and o['step'] == horizon]
            if rows:
                values.append(dict(seed=seed, n_repeats=len(rows),
                                   delta=statistics.mean(o['observed_delta'] for o in rows),
                                   accuracy_delta=statistics.mean(o['accuracy_delta'] for o in rows),
                                   delta_minus_baseline=statistics.mean(o['delta_minus_baseline'] for o in rows),
                                   valid_predictions=sum(o['prediction_valid'] for o in rows),
                                   sign_matches=sum(o['sign_match'] is True and o['prediction_valid'] for o in rows),
                                   sign_evaluable=sum(o['sign_match'] is not None and o['prediction_valid'] for o in rows)))
        deltas = [v['delta'] for v in values]
        cells.append(dict(source_step=source_step, branch=branch, horizon=horizon, per_seed=values,
                          mean_delta=statistics.mean(deltas),
                          seed_sample_std=statistics.stdev(deltas) if len(deltas) > 1 else None))
    reversals = [dict(seed=r['prediction']['seed'], source_step=r['prediction']['source_step'],
                      repeat=r['prediction']['repeat'], **item) for r in runs for item in r['threshold_reversals']]
    summary = dict(completed=len(runs), expected=expected, complete=len(runs) == expected,
                   threshold_reversals=reversals,
                   per_seed=per_seed, cells=cells, test_used=False,
                   scope='Exploratory label-conditioned predictions; repeats/checkpoints are not independent seeds')
    write_json(output / 'summary.json', summary)
    lines = ['# CIFAR local risk direction', '', f'Completed state/repeat jobs: {len(runs)}/{expected}.',
             'MSE deltas are validation risk minus the same source state; negative is better.',
             'Average repeats within each seed first. Threshold branches are conditional on eligibility.',
             'Step > 1 uses a frozen initial tangent, not a new derivative prediction at each step.', '',
             '| Source step | Branch | Horizon | Seeds | Mean MSE delta | Seed SD |',
             '|---|---|---|---|---|---|']
    for row in cells:
        lines.append(f'| {row["source_step"]} | {row["branch"]} | {row["horizon"]} | {len(row["per_seed"])} | '
                     f'{row["mean_delta"]:.6g} | {row["seed_sample_std"]} |')
    (output / 'findings.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['plan', 'run', 'summarize'], default='plan')
    parser.add_argument('--source', type=Path, default=RESULTS / 'cifar' / 'capacity_development')
    parser.add_argument('--source-kind', choices=['capacity', 'local'], default='capacity')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, default=RESULTS / 'cifar' / 'risk_direction_development')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 12])
    parser.add_argument('--source-steps', nargs='+', type=int, default=[0, 1000, 4000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 20])
    parser.add_argument('--gradient-batches', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--feature-batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--future-seed', type=int, default=941000)
    parser.add_argument('--fd-scales', nargs=2, type=float, default=[.1, .05])
    parser.add_argument('--fd-atol', type=float, default=1e-10)
    parser.add_argument('--fd-rtol', type=float, default=.02)
    parser.add_argument('--signal-floor', type=float, default=1e-10)
    parser.add_argument('--ratio-bounds', nargs=2, type=float, default=[.001, 1000.])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.phase == 'summarize':
        summarize(args.output)
        return
    if any(v < 1 for v in [args.repeats, args.gradient_batches, args.feature_batch, *args.horizons]) or args.batch_size < 2:
        parser.error('Positive budgets and batch-size >= 2 required')
    for name in ['seeds', 'source_steps', 'horizons']:
        values = getattr(args, name)
        if min(values) < 0 or len(set(values)) != len(values):
            parser.error(f'{name} must be unique and nonnegative')
    if 1 not in args.horizons or args.future_seed < 0:
        parser.error('Include horizon 1 and use a nonnegative future seed')
    if any(not math.isfinite(v) or v <= 0 for v in [args.lr, args.fd_atol, args.fd_rtol, args.signal_floor,
                                                   *args.fd_scales, *args.ratio_bounds]):
        parser.error('Rates, tolerances, FD scales and ratio bounds must be positive finite')
    if args.ratio_bounds[0] >= args.ratio_bounds[1] or args.fd_scales[0] == args.fd_scales[1]:
        parser.error('Ordered ratio bounds and distinct FD scales required')
    if args.output.resolve() == args.source.resolve() or args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('Use an output directory separate from the source study')
    arguments = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'phase'}
    if args.phase == 'plan':
        print(json.dumps(dict(arguments=arguments, state_repeat_jobs=len(args.seeds)*len(args.source_steps)*args.repeats,
                              scope='plan only; no data/checkpoint access; at most 7 branches per state/repeat'), indent=2))
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_protocol = read_json(args.source / 'protocol.json')
    original = source_protocol.get('arguments', source_protocol)
    if original.get('lr') != args.lr or original.get('batch_size') != args.batch_size:
        raise ValueError('Use source base learning rate and batch size')
    protocol = dict(arguments=arguments, precision='float64 model, loss, OLS and derivatives; CPU float32 augmentation',
                    first_update='mean of gradient_batches minibatch gradients; subsequent single-batch SGD',
                    labels='training OLS fit and validation risk derivative/threshold selection; oracle diagnostic',
                    split='reuse existing source 45k/5k split, no new partition, no test access', test_used=False)
    path = args.output / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Output settings differ; use a separate output directory')
    # Check all requested sources before loading images or computing derivatives.
    for step in args.source_steps:
        args.source_step = step
        for seed in args.seeds:
            source_metadata(args, seed)
            if not source_paths(args, seed)[0].is_file():
                raise FileNotFoundError(source_paths(args, seed)[0])
    data, labels, train, validation = load_images(args, 'cifar', source_protocol)
    write_json(path, protocol)
    for seed in args.seeds:
        for step in args.source_steps:
            args.source_step = step
            if all((args.output / f'seed_{seed}_step_{step}_repeat_{r}' / 'result.json').exists() for r in range(args.repeats)):
                continue
            source = source_metadata(args, seed)
            model = load_source(args, seed, args.device).double()
            gradient, initial = risk_gradient(model, data, labels, train, validation, args.feature_batch)
            if not all(bool(g.isfinite().all()) for g in gradient):
                raise FloatingPointError('Nonfinite risk derivative')
            for repeat in range(args.repeats):
                run_repeat(args, seed, step, repeat, model, source, data, labels, train, validation, gradient, initial)
    summarize(args.output)


if __name__ == '__main__':
    main()
