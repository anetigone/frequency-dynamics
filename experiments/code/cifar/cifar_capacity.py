"""Plan, train and evaluate validation-only CIFAR capacity diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

import argparse
import copy
import hashlib
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from cifar_local import NaturalSquare, augment, load_data
from square_images import empirical_loss

ARCHITECTURES = {'baseline': (64, 8), 'wide': (256, 8), 'less_compressed': (64, 32)}
OBJECTIVES = ('unsupervised', 'supervised_encoder', 'supervised_projection')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def make_jobs(seeds, suite, architectures, ratios):
    jobs = []
    for seed in seeds:
        if suite in ('all', 'unsupervised'):
            for architecture in architectures:
                c, k = ARCHITECTURES[architecture]
                for ratio in ratios:
                    jobs.append(dict(seed=seed, objective='unsupervised', architecture=architecture,
                                     channels=c, outputs=k, ratio=ratio))
        if suite in ('all', 'supervised'):
            for objective in OBJECTIVES[1:]:
                jobs.append(dict(seed=seed, objective=objective, architecture='baseline',
                                 channels=64, outputs=8, ratio=None))
    for job in jobs:
        suffix = f"_r{job['ratio']:g}" if job['ratio'] is not None else ''
        job['id'] = f"seed_{job['seed']}_{job['architecture']}_{job['objective']}{suffix}"
    return jobs


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize(seed, channels, outputs, calibration_x, draws=16, record_state_hash=True):
    """Same rule as the original experiment, generalized to C and K."""
    device = calibration_x.device
    generator = torch.Generator().manual_seed(seed)
    model = NaturalSquare(channels, outputs).to(device)
    with torch.no_grad():
        model.kernel.copy_(torch.randn(model.kernel.shape, generator=generator).to(device) / 11)
        model.weight.copy_(torch.randn(model.weight.shape, generator=generator).to(device) / math.sqrt(channels))
        z = model(calibration_x)
        energy = (z - z.mean(0)).square().sum(1).mean()
        if float(energy) <= 0:
            raise ValueError('Degenerate initialization calibration')
        model.weight.mul_(math.sqrt(.01 / float(energy)))
    z = model(calibration_x)
    z = z - z.mean(0)
    generator = torch.Generator(device=device).manual_seed(720000 + seed)
    estimates = []
    for _ in range(draws):
        signs = torch.randint(2, z.shape, generator=generator, device=device).float() * 2 - 1
        grad = torch.autograd.grad((z * signs).sum() / math.sqrt(len(z)), model.kernel, retain_graph=True)[0]
        estimates.append(float(grad.square().sum()))
    with torch.no_grad():
        f = model.features(calibration_x)
        head = outputs * float((f - f.mean(0)).square().sum(1).mean())
        front = sum(estimates) / draws
        if min(front, head) <= 0:
            raise ValueError('Degenerate Jacobian trace calibration')
        scale = (front / head) ** (1 / 6)
        before = model(calibration_x).clone()
        model.kernel.mul_(scale)
        model.weight.mul_(scale ** -2)
        error = float((model(calibration_x) - before).abs().max())
        initial_energy = float((before - before.mean(0)).square().sum(1).mean())
    calibration = dict(scale=scale, matched_function_max_error=error, initial_energy=initial_energy,
                       front_trace_after_estimate=front * scale ** -2,
                       head_trace_after=head * scale ** 4, hutchinson_draws=draws,
                       common_mobility=1 / (front * scale ** -2 + head * scale ** 4))
    if record_state_hash:
        calibration['initial_state_sha256'] = state_hash(model)
    return model, calibration


def supervised_head(model, objective, calibration_x, seed):
    """Frozen initial feature scaling; diagnostic classifier only, never in SSL."""
    with torch.no_grad():
        f = model.features(calibration_x)
        x = f if objective == 'supervised_encoder' else f @ model.weight.T
        mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    classifier = nn.Linear(x.shape[1], 10).to(x.device)
    generator = torch.Generator().manual_seed(920000 + seed)
    with torch.no_grad():
        classifier.weight.copy_(.01 * torch.randn(classifier.weight.shape, generator=generator).to(x.device))
        classifier.bias.zero_()
    return classifier, mean, std


def make_optimizer(model, classifier, calibration, job, args):
    if job['objective'] == 'unsupervised':
        mobility, ratio = calibration['common_mobility'], job['ratio']
        return torch.optim.SGD([
            {'params': [model.kernel], 'lr': args.lr * 2 * mobility / (1 + ratio)},
            {'params': [model.weight], 'lr': args.lr * 2 * mobility * ratio / (1 + ratio)}])
    parameters = [model.kernel]
    if job['objective'] == 'supervised_projection':
        parameters.append(model.weight)
    # Positive controls are feasibility references, not matched SSL dynamics.
    groups = [{'params': parameters, 'lr': args.supervised_lr},
              {'params': classifier.parameters(), 'lr': args.classifier_lr}]
    if getattr(args, 'supervised_optimizer', 'adam') == 'adam':
        optimizer = torch.optim.Adam(groups, weight_decay=0.)
    else:
        optimizer = torch.optim.SGD(groups, momentum=args.supervised_momentum)
    optimizer.supervised_grad_clip = getattr(args, 'supervised_grad_clip', 1.)
    return optimizer


def update(model, classifier, standardizer, optimizer, x1, x2, targets, objective):
    optimizer.zero_grad(set_to_none=True)
    if objective == 'unsupervised':
        loss = empirical_loss(model(x1), model(x2))
    else:
        mean, std = standardizer
        def logits(x):
            f = model.features(x)
            z = f if objective == 'supervised_encoder' else f @ model.weight.T
            return classifier((z - mean) / std)
        loss = .5 * (F.cross_entropy(logits(x1), targets) + F.cross_entropy(logits(x2), targets))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training loss; no automatic retry')
    loss.backward()
    active = list(model.parameters()) + ([] if classifier is None else list(classifier.parameters()))
    if any(p.grad is not None and not bool(p.grad.isfinite().all()) for p in active):
        raise FloatingPointError(f'Nonfinite gradients ({objective})')
    optimizer.last_step_diagnostics = None
    if objective != 'unsupervised':
        limit = getattr(optimizer, 'supervised_grad_clip', 0.)
        parameters = [p for p in active if p.grad is not None]
        if limit > 0:
            try:
                norm = torch.nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
            except RuntimeError as error:
                raise FloatingPointError(f'Nonfinite supervised gradient norm: {error}') from error
        else:
            norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().double().norm() for p in parameters]))
        optimizer.last_step_diagnostics = dict(gradient_norm_before_clip=float(norm),
                                               gradient_clipped=bool(limit > 0 and float(norm) > limit),
                                               gradient_clip_limit=limit)
    optimizer.step()
    if any(not bool(p.isfinite().all()) for p in active):
        raise FloatingPointError('Nonfinite parameters')
    return float(loss.detach())


def save_checkpoint(path, model, classifier, standardizer, job, step):
    torch.save(dict(model=model.state_dict(), classifier=None if classifier is None else classifier.state_dict(),
                    standardizer=None if standardizer is None else tuple(x.detach() for x in standardizer),
                    job=job, step=step), path)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def failure_report(directory, model, classifier, standardizer, optimizer, job, step, error, history):
    def stats(tensor):
        tensor = tensor.detach()
        def number(value):
            result = float(value)
            return result if math.isfinite(result) else None
        return dict(all_finite=bool(tensor.isfinite().all()), max_abs=number(tensor.abs().max()),
                    norm=number(tensor.double().norm()))
    parameters = dict(model.named_parameters())
    if classifier is not None:
        parameters.update({f'classifier.{key}': value for key, value in classifier.named_parameters()})
    report = dict(job=job, step=step, error=str(error), learning_rates=[g['lr'] for g in optimizer.param_groups],
                  parameters={key: stats(value) for key, value in parameters.items()},
                  gradients={key: stats(value.grad) for key, value in parameters.items() if value.grad is not None},
                  recent_history=history[-10:],
                  standardizer=None if standardizer is None else dict(mean=stats(standardizer[0]), std=stats(standardizer[1])),
                  status='failed; not a completed run or resumable optimizer state')
    write_json(directory / 'failure.json', report)
    save_checkpoint(directory / 'failure_state.pt', model, classifier, standardizer, job, step)


def train_job(job, args, data, labels, train_indices, calibration_x):
    directory = args.output / job['id']
    if (directory / 'result.json').exists():
        print(f"Already complete: {job['id']}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Starting {job['id']}: objective={job['objective']}, steps={args.steps}", flush=True)
    model, calibration = initialize(job['seed'], job['channels'], job['outputs'], calibration_x, args.trace_draws)
    classifier, standardizer = None, None
    if job['objective'] != 'unsupervised':
        classifier, mean, std = supervised_head(model, job['objective'], calibration_x, job['seed'])
        standardizer = (mean, std)
    optimizer = make_optimizer(model, classifier, calibration, job, args)
    print(f"Optimizer: {type(optimizer).__name__}; learning rates={[g['lr'] for g in optimizer.param_groups]}; "
          f"supervised gradient clip={getattr(optimizer, 'supervised_grad_clip', None)}", flush=True)
    save_checkpoint(directory / 'step_0.pt', model, classifier, standardizer, job, 0)
    generator = torch.Generator(device=data.device).manual_seed(820000 + job['seed'])
    indices = train_indices.to(data.device)
    history, crossings = [], []
    max_loss, clipped_updates, max_gradient_norm = -math.inf, 0, 0.
    batch_hash = hashlib.sha256()
    fixed_steps = sorted(set([0, *args.checkpoints, args.steps]))
    synchronize(data.device)
    started = time.perf_counter()
    if data.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(data.device)
    for step in range(1, args.steps + 1):
        index = indices[torch.randint(len(indices), (args.batch_size,), generator=generator, device=data.device)]
        batch_hash.update(index.cpu().numpy().tobytes())
        x1, x2 = augment(data[index], generator), augment(data[index], generator)
        # SSL never reads labels, even though its batches are shared with positive controls.
        target = None if classifier is None else labels[index.cpu()].to(data.device)
        try:
            loss = update(model, classifier, standardizer, optimizer, x1, x2, target, job['objective'])
        except FloatingPointError as error:
            failure_report(directory, model, classifier, standardizer, optimizer, job, step, error, history)
            raise FloatingPointError(f"{job['id']} failed at step {step}/{args.steps}: {error}; see {directory / 'failure.json'}") from error
        max_loss = max(max_loss, loss)
        diagnostics = optimizer.last_step_diagnostics
        if diagnostics is not None:
            clipped_updates += int(diagnostics['gradient_clipped'])
            max_gradient_norm = max(max_gradient_norm, diagnostics['gradient_norm_before_clip'])
        if step % args.energy_every == 0 or step == args.steps:
            with torch.no_grad():
                z = model(calibration_x)
                energy = float((z - z.mean(0)).square().sum(1).mean())
            if not math.isfinite(energy) or energy <= 0:
                error = FloatingPointError('Nonpositive/nonfinite calibration energy')
                failure_report(directory, model, classifier, standardizer, optimizer, job, step, error, history)
                raise FloatingPointError(f"{job['id']} failed at step {step}: {error}")
            ell = math.log(energy / calibration['initial_energy'])
            history.append(dict(step=step, loss=loss, energy=energy, ell=ell,
                                optimization=diagnostics))
            if job['objective'] == 'unsupervised':
                for level in args.energy_levels:
                    if ell >= level and not any(row['target_ell'] == level for row in crossings):
                        filename = f'energy_{level:g}.pt'
                        save_checkpoint(directory / filename, model, classifier, standardizer, job, step)
                        crossings.append(dict(target_ell=level, observed_ell=ell, step=step, file=filename))
        if step in fixed_steps:
            save_checkpoint(directory / f'step_{step}.pt', model, classifier, standardizer, job, step)
        if step % 500 == 0:
            print(f"{job['id']}: {step}/{args.steps}", flush=True)
    synchronize(data.device)
    result = dict(job=job, steps=args.steps, calibration=calibration, history=history,
                  optimization=dict(optimizer=type(optimizer).__name__, learning_rates=[g['lr'] for g in optimizer.param_groups],
                                    supervised_gradient_clip=getattr(optimizer, 'supervised_grad_clip', None),
                                    clipped_updates=clipped_updates, max_gradient_norm_before_clip=max_gradient_norm,
                                    max_training_loss=max_loss),
                  fixed_steps=fixed_steps, energy_checkpoints=crossings,
                  batch_index_sha256=batch_hash.hexdigest(),
                  augmentation_generator_final_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
                  seconds=time.perf_counter() - started,
                  peak_memory_GiB=torch.cuda.max_memory_allocated(data.device) / 2**30 if data.device.type == 'cuda' else None,
                  test_used=False)
    write_json(directory / 'result.json', result)
    print(f"Completed {job['id']}; result.json saved", flush=True)


def score_metrics(scores, targets):
    onehot = F.one_hot(targets, 10).double()
    predicted = scores.argmax(1)
    mask = F.one_hot(targets, 10).bool()
    margin = scores[mask] - scores.masked_fill(mask, -torch.inf).max(1).values
    confusion = torch.bincount(targets * 10 + predicted, minlength=100).reshape(10, 10)
    return dict(accuracy=float((predicted == targets).double().mean()),
                onehot_mse=float((scores - onehot).square().mean()),
                mean_true_class_margin=float(margin.mean()), confusion=confusion.tolist())


def spectrum(matrix):
    matrix = matrix.double() - matrix.double().mean(0)
    eigenvalues = torch.linalg.eigvalsh(matrix.T @ matrix / len(matrix)).clamp_min(0).flip(0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total if total > 0 else eigenvalues
    positive = probabilities[probabilities > 0]
    return dict(covariance_eigenvalues=eigenvalues.tolist(), total_variance=total,
                participation_ratio=float(1 / probabilities.square().sum()) if total > 0 else 0.,
                entropy_effective_rank=float(torch.exp(-(positive * positive.log()).sum())) if total > 0 else 0.)


def fit_probe(features, labels, train, validation):
    matrix = features.double()
    mean, std = matrix[train].mean(0), matrix[train].std(0).clamp_min(1e-12)
    design = torch.cat([torch.ones(len(matrix), 1, dtype=torch.float64), (matrix - mean) / std], 1)
    fit = torch.linalg.lstsq(design[train], F.one_hot(labels[train], 10).double(), driver='gelsd')
    return dict(train=score_metrics(design[train] @ fit.solution, labels[train]),
                validation=score_metrics(design[validation] @ fit.solution, labels[validation]),
                design_rank_with_intercept=int(fit.rank), singular_values=fit.singular_values.tolist(),
                training_feature_spectrum=spectrum(matrix[train]))


@torch.no_grad()
def evaluate_checkpoint(path, data, labels, train, validation, feature_batch=256):
    checkpoint = torch.load(path, map_location=data.device, weights_only=True)
    job = checkpoint['job']
    model = NaturalSquare(job['channels'], job['outputs']).to(data.device)
    model.load_state_dict(checkpoint['model'])
    features = torch.cat([model.features(data[i:i + feature_batch]).cpu() for i in range(0, len(data), feature_batch)])
    projection = features.double() @ model.weight.detach().cpu().double().T
    result = dict(step=checkpoint['step'], encoder=fit_probe(features, labels, train, validation),
                  projection=fit_probe(projection, labels, train, validation))
    if checkpoint['classifier'] is not None:
        x = features if job['objective'] == 'supervised_encoder' else projection
        mean, std = (v.cpu().double() for v in checkpoint['standardizer'])
        head = checkpoint['classifier']
        scores = ((x.double() - mean) / std) @ head['weight'].cpu().double().T + head['bias'].cpu().double()
        result['trained_classifier'] = {}
        for name, indices in [('train', train), ('validation', validation)]:
            result['trained_classifier'][name] = dict(accuracy=float((scores[indices].argmax(1) == labels[indices]).double().mean()),
                                                       cross_entropy=float(F.cross_entropy(scores[indices], labels[indices])))
    return result


def protocol_for(args, jobs, split):
    return dict(arguments={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'phase'},
                jobs=jobs, split=split, source_validation=False,
                torch_version=str(torch.__version__), device_name=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else 'cpu',
                test_used=False, scope='development diagnosis; supervised references are not capacity bounds',
                matching='same initial function only within architecture/seed; same batch/augmentation stream across jobs for each seed',
                energy='log total centered calibration variance relative to own initial value; proxy only, no cross-K equivalence claim',
                initialization='C,K-specific equal Gram traces at r=1; weighted initial total mobility fixed across ratios within shape',
                augmentation='reflect crop +/-4, horizontal flip, contrast .8..1.2, Gaussian .05; two shared views per update',
                supervised='clean initial feature scaling fixed; CE on two views; explicit optimizer and gradient clipping, recorded per run')


def prepare_protocol(args, jobs, split):
    """Resume in place: skip completed jobs, restart incomplete jobs, no source hashes."""
    current = protocol_for(args, jobs, split)
    lock = args.output / 'protocol.json'
    if lock.exists():
        previous = json.loads(lock.read_text(encoding='utf-8'))
        if previous['jobs'] != jobs or previous['split'] != split:
            raise ValueError('Existing jobs or data split differ; keep the original training arguments')
        for name in ['steps', 'checkpoints', 'energy_levels', 'energy_every', 'batch_size', 'lr', 'trace_draws']:
            if previous['arguments'][name] != current['arguments'][name]:
                raise ValueError(f'Existing training configuration differs: {name}; keep the original value')
        completed = []
        for job in jobs:
            path = args.output / job['id'] / 'result.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['job'] != job or result['steps'] != args.steps:
                    raise ValueError(f'Completed run configuration differs: {job["id"]}')
                completed.append(job)
        legacy_defaults = dict(supervised_optimizer='sgd', supervised_grad_clip=0.)
        supervised_keys = ['supervised_lr', 'classifier_lr', 'supervised_momentum', *legacy_defaults]
        supervised_changed = any(previous['arguments'].get(key, legacy_defaults.get(key)) != current['arguments'].get(key, legacy_defaults.get(key)) for key in supervised_keys)
        if supervised_changed and any(job['objective'] != 'unsupervised' for job in completed):
            raise ValueError('Completed supervised runs use different optimizer settings; preserve their original settings or use a separate output')
        if previous != current:
            snapshot = args.output / 'protocol_before_resume.json'
            if not snapshot.exists():
                snapshot.write_bytes(lock.read_bytes())
            events_path = args.output / 'resume_events.json'
            events = json.loads(events_path.read_text(encoding='utf-8')) if events_path.exists() else []
            events.append(dict(completed_runs_kept=[job['id'] for job in completed],
                               previous_arguments=previous['arguments'], current_arguments=current['arguments'],
                               source_validation=False, incomplete_runs='restart from initialization'))
            write_json(events_path, events)
            print(f'Resuming existing directory; keeping {len(completed)} completed runs; no source-hash validation', flush=True)
    write_json(lock, current)
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['plan', 'train', 'evaluate'], default='plan')
    parser.add_argument('--output', type=Path, default=RESULTS / 'cifar' / 'capacity_development')
    parser.add_argument('--suite', choices=['all', 'unsupervised', 'supervised'], default='all')
    parser.add_argument('--architectures', nargs='+', choices=list(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 12])
    parser.add_argument('--ratios', nargs='+', type=float, default=[.1, 1., 10.])
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--checkpoints', nargs='+', type=int, default=[1000, 2000, 3000])
    parser.add_argument('--energy-levels', nargs='+', type=float, default=[.5, 1., 2.])
    parser.add_argument('--energy-every', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--supervised-optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--supervised-grad-clip', type=float, default=1., help='supervised-only gradient norm limit; 0 disables')
    parser.add_argument('--supervised-lr', type=float, default=.0003)
    parser.add_argument('--classifier-lr', type=float, default=.001)
    parser.add_argument('--supervised-momentum', type=float, default=.9)
    parser.add_argument('--trace-draws', type=int, default=16)
    parser.add_argument('--feature-batch', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.trace_draws < 1 or args.energy_every < 1 or args.feature_batch < 1:
        parser.error('Positive steps/draws/intervals required; batch size >=2')
    if any(not math.isfinite(v) or v <= 0 for v in [*args.ratios, args.lr, args.supervised_lr, args.classifier_lr, *args.energy_levels]):
        parser.error('Ratios, learning rates and energy levels must be positive and finite')
    if not 0 <= args.supervised_momentum < 1:
        parser.error('Momentum must be in [0,1)')
    if not math.isfinite(args.supervised_grad_clip) or args.supervised_grad_clip < 0:
        parser.error('Supervised gradient clipping must be finite and nonnegative')
    for values in [args.seeds, args.ratios, args.architectures, args.checkpoints, args.energy_levels]:
        if len(values) != len(set(values)):
            parser.error('Duplicate configuration entries are not allowed')
    args.checkpoints = sorted(step for step in args.checkpoints if 0 < step <= args.steps)
    jobs = make_jobs(args.seeds, args.suite, args.architectures, args.ratios)
    if args.phase == 'plan':
        print(json.dumps(dict(run_count=len(jobs), jobs=jobs, output=str(args.output),
                              test_used=False, status='plan only; no data loaded, no training'), indent=2))
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu for implementation checks')
    if args.phase == 'evaluate':
        protocol = json.loads((args.output / 'protocol.json').read_text(encoding='utf-8'))
        pending = [j['id'] for j in protocol['jobs'] if not (args.output / j['id'] / 'result.json').exists()]
        if pending:
            raise ValueError(f'{len(pending)} runs incomplete; evaluate only after the planned suite finishes')
        data, labels, train, validation, split = load_data(EXPERIMENTS.parent, args.device)
        if split != protocol['split']:
            raise ValueError('Data split/normalization differs from training protocol')
        for job in protocol['jobs']:
            directory = args.output / job['id']
            destination = directory / 'evaluation.json'
            if destination.exists():
                continue
            result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            fixed = [dict(file=f'step_{step}.pt', **evaluate_checkpoint(directory / f'step_{step}.pt', data, labels, train, validation, args.feature_batch))
                     for step in result['fixed_steps']]
            energy = [dict(**row, metrics=evaluate_checkpoint(directory / row['file'], data, labels, train, validation, args.feature_batch))
                      for row in result['energy_checkpoints']]
            write_json(destination, dict(job=job, fixed_steps=fixed, energy_checkpoints=energy, test_used=False))
            print(f"Evaluated {job['id']}", flush=True)
        return
    data, labels, train, validation, split = load_data(EXPERIMENTS.parent, args.device)
    prepare_protocol(args, jobs, split)
    generator = torch.Generator().manual_seed(20261096)
    calibration_indices = train[torch.randperm(len(train), generator=generator)[:256]]
    calibration_x = data[calibration_indices.to(data.device)]
    for job in jobs:
        train_job(job, args, data, labels, train, calibration_x)


if __name__ == '__main__':
    main()
