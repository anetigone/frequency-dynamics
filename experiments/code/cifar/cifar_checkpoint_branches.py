"""Validation-only paired CIFAR branches from a saved r=1 square-model state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

import argparse
import hashlib
import json
import math

import torch
from cifar_local import NaturalSquare, augment, load_data
from cifar_capacity import fit_probe
from square_images import empirical_loss


BRANCHES = [dict(name='ratio_0.1', ratio=.1, speed=1.),
            dict(name='baseline', ratio=1., speed=1.),
            dict(name='ratio_10', ratio=10., speed=1.),
            dict(name='slow', ratio=1., speed=.5),
            dict(name='fast', ratio=1., speed=2.)]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_paths(args, seed):
    name = f'seed_{seed}_baseline_unsupervised_r1' if args.source_kind == 'capacity' else f'seed_{seed}_r1'
    directory = args.source / name
    filename = 'initial.pt' if args.source_kind == 'local' and args.source_step == 0 else f'step_{args.source_step}.pt'
    return directory / filename, directory / 'result.json'


def make_jobs(args):
    return [dict(id=f'seed_{seed}_{branch["name"]}', seed=seed, **branch) for seed in args.seeds for branch in BRANCHES]


def source_metadata(args, seed):
    checkpoint, result_path = source_paths(args, seed)
    result = read_json(result_path)
    if result['steps'] < args.source_step:
        raise ValueError('Requested source step exceeds completed source budget')
    if args.source_kind == 'capacity':
        expected = dict(seed=seed, architecture='baseline', objective='unsupervised', ratio=1., channels=64, outputs=8)
        if result['job'] != dict(id=f'seed_{seed}_baseline_unsupervised_r1', **expected):
            raise ValueError('Source is not the requested baseline r=1 run')
        if 'optimization' in result:
            optimization = result['optimization']
            if not isinstance(optimization, dict) or optimization.get('optimizer') != 'SGD':
                raise ValueError('Only the existing stateless unsupervised SGD sources are supported')
        else:
            # Capacity runs before the supervised-optimizer fix did not log this
            # field. Their unsupervised path already used the same stateless SGD;
            # supervised_momentum in the protocol never applied to that path.
            protocol = read_json(args.source / 'protocol.json')
            if result['job'] not in protocol.get('jobs', []):
                raise ValueError('Legacy capacity source job is missing from its source protocol')
    elif result['seed'] != seed or result['ratio'] != 1.:
        raise ValueError('Source seed/ratio mismatch')
    mobility = result['calibration']['common_mobility']
    if not math.isfinite(mobility) or mobility <= 0:
        raise ValueError('Invalid source mobility')
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
                result_sha256=file_hash(result_path), common_mobility=mobility)


def load_source(args, seed, device):
    path, _ = source_paths(args, seed)
    payload = torch.load(path, map_location=device, weights_only=True)
    if args.source_kind == 'capacity':
        if payload['step'] != args.source_step or payload['job']['seed'] != seed or payload['job']['ratio'] != 1.:
            raise ValueError('Source checkpoint metadata mismatch')
        state = payload['model']
    else:
        state = payload
    model = NaturalSquare().to(device)
    model.load_state_dict(state, strict=True)
    if any(not bool(p.isfinite().all()) for p in model.parameters()):
        raise ValueError('Nonfinite source parameters')
    return model


def learning_rates(base_lr, mobility, job):
    # Keep the original initial-Jacobian calibration; do not recalibrate at the fork.
    scale = base_lr * 2 * mobility * job['speed'] / (1 + job['ratio'])
    return [scale, scale * job['ratio']]


def make_audit(data, train, size, seed, augment_fn=augment):
    selection = torch.Generator().manual_seed(seed)
    indices = train[torch.randperm(len(train), generator=selection)[:size]]
    clean = data[indices.to(data.device)]
    generator = torch.Generator(device=data.device).manual_seed(seed + 1)
    first, second = augment_fn(clean, generator), augment_fn(clean, generator)
    metadata = dict(indices=indices.tolist(), first_view_sha256=tensor_hash(first),
                    second_view_sha256=tensor_hash(second), seed=seed)
    return (clean, first, second), metadata


@torch.no_grad()
def audit_model(model, audit):
    clean, first, second = audit
    # Double precision small-matrix moment arithmetic; convolution remains float32.
    z1, z2 = model(first).double(), model(second).double()
    c1, c2 = z1 - z1.mean(0), z2 - z2.mean(0)
    covariance = (c1.T @ c1 + c2.T @ c2) / (2 * (len(z1) - 1))
    shifted = torch.eye(len(covariance), device=z1.device, dtype=z1.dtype) + covariance
    alignment = .5 * (z1 - z2).square().sum(1).mean()
    logdet = -.5 * torch.linalg.slogdet(shifted)[1]
    trace = -.5 * torch.trace(torch.linalg.solve(shifted, covariance))
    z = model(clean).double()
    result = dict(loss=float(alignment + logdet + trace), alignment=float(alignment),
                  uniformity_logdet=float(logdet), uniformity_trace=float(trace),
                  clean_energy=float((z - z.mean(0)).square().sum(1).mean()),
                  view_covariance_eigenvalues=torch.linalg.eigvalsh(covariance).flip(0).cpu().tolist())
    if not all(math.isfinite(result[k]) for k in ['loss', 'alignment', 'uniformity_logdet', 'uniformity_trace', 'clean_energy']):
        raise FloatingPointError('Nonfinite fixed-audit moments')
    return result


def save_state(path, model, optimizer, generator, job, step):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    generator_state=generator.get_state(), job=job, branch_step=step), path)


def train_job(args, job, source, data, train, audit, source_loader=None, augment_fn=augment):
    directory = args.output / job['id']
    if (directory / 'result.json').exists():
        previous = read_json(directory / 'result.json')
        if previous['job'] != job or previous['steps'] != args.steps or previous['source'] != source:
            raise ValueError('Completed branch differs from locked job')
        print(f'Already complete: {job["id"]}', flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    model = (source_loader or load_source)(args, job['seed'], data.device)
    initial_hash = state_hash(model.state_dict())
    rates = learning_rates(args.lr, source['common_mobility'], job)
    optimizer = torch.optim.SGD([dict(params=[model.kernel], lr=rates[0]),
                                 dict(params=[model.weight], lr=rates[1])], momentum=0., weight_decay=0.)
    # A new, paired future stream, not a reconstruction of the original source stream.
    generator = torch.Generator(device=data.device).manual_seed(args.future_seed + job['seed'])
    batch_hash, view_hash = hashlib.sha256(), hashlib.sha256()
    fixed_steps = sorted({0, args.steps, *range(args.eval_every, args.steps, args.eval_every)})
    history, snapshots, crossings = [], [], []
    initial_loss = None
    indices = train.to(data.device)
    step = 0
    try:
        for step in range(args.steps + 1):
            if step:
                selected = indices[torch.randint(len(indices), (args.batch_size,), generator=generator, device=data.device)]
                batch_hash.update(selected.cpu().numpy().tobytes())
                first, second = augment_fn(data[selected], generator), augment_fn(data[selected], generator)
                # Hash actual views occasionally, plus every batch index and final RNG state.
                if step == 1 or step % args.audit_every == 0 or step == args.steps:
                    view_hash.update(bytes.fromhex(tensor_hash(first)))
                    view_hash.update(bytes.fromhex(tensor_hash(second)))
                optimizer.zero_grad(set_to_none=True)
                loss = empirical_loss(model(first), model(second))
                if not bool(loss.isfinite()):
                    raise FloatingPointError('Nonfinite branch training loss')
                loss.backward()
                if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite/missing branch gradient')
                optimizer.step()
                if any(not bool(p.isfinite().all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite branch parameters')
            if step % args.audit_every == 0 or step in fixed_steps:
                metrics = audit_model(model, audit)
                if initial_loss is None:
                    initial_loss = metrics['loss']
                reached = [level for level in args.loss_levels if initial_loss > level >= metrics['loss']
                           and not any(row['target_loss'] == level for row in crossings)]
                row = dict(step=step, **metrics)
                history.append(row)
                if step in fixed_steps or reached:
                    filename = f'step_{step}.pt'
                    save_state(directory / filename, model, optimizer, generator, job, step)
                    snapshots.append(dict(step=step, file=filename, audit=metrics, fixed=step in fixed_steps))
                    for level in reached:
                        crossings.append(dict(target_loss=level, observed_loss=metrics['loss'],
                                              overshoot=level - metrics['loss'], step=step, file=filename))
            if step and (step % args.eval_every == 0 or step == args.steps):
                print(f'{job["id"]}: {step}/{args.steps}; audit loss={history[-1]["loss"]:.6f}', flush=True)
    except (FloatingPointError, RuntimeError) as error:
        write_json(directory / 'failure.json', dict(job=job, step=step, error=str(error), status='failed; restart from source'))
        raise
    write_json(directory / 'result.json', dict(job=job, steps=args.steps, source=source,
               initial_state_sha256=initial_hash, learning_rates=rates, fixed_steps=fixed_steps,
               history=history, snapshots=snapshots, loss_crossings=crossings,
               batch_index_sha256=batch_hash.hexdigest(), sampled_views_sha256=view_hash.hexdigest(),
               final_generator_sha256=tensor_hash(generator.get_state()), test_used=False))
    (directory / 'failure.json').unlink(missing_ok=True)


@torch.no_grad()
def evaluate_job(args, job, data, labels, train, validation, probe_fn=fit_probe):
    directory = args.output / job['id']
    run = read_json(directory / 'result.json')
    result_hash = file_hash(directory / 'result.json')
    destination = directory / 'evaluation.json'
    if destination.exists():
        old = read_json(destination)
        if old.get('result_sha256') != result_hash or old['job'] != job:
            raise ValueError('Stale branch evaluation; use a fresh output directory')
        return
    model = NaturalSquare().to(data.device)
    rows = []
    for snapshot in run['snapshots']:
        payload = torch.load(directory / snapshot['file'], map_location=data.device, weights_only=True)
        if payload['job'] != job or payload['branch_step'] != snapshot['step']:
            raise ValueError('Branch checkpoint metadata mismatch')
        model.load_state_dict(payload['model'])
        features = torch.cat([model.features(data[i:i + args.feature_batch]).cpu() for i in range(0, len(data), args.feature_batch)])
        projection = features.double() @ model.weight.detach().cpu().double().T
        rows.append(dict(step=snapshot['step'], encoder=probe_fn(features, labels, train, validation),
                         projection=probe_fn(projection, labels, train, validation)))
    write_json(destination, dict(job=job, result_sha256=result_hash, rows=rows, test_used=False))


def prepare_protocol(args, jobs, sources, split, audit_metadata, source_protocol):
    if source_protocol['split'] != split:
        raise ValueError('Source and current split/normalization differ')
    source_args = source_protocol.get('arguments', source_protocol)
    if source_args.get('lr') != args.lr or source_args.get('batch_size') != args.batch_size:
        raise ValueError('Use the source base learning rate and batch size; speed branches apply explicit multipliers')
    protocol = dict(schema_version=1, arguments={k: str(v.resolve()) if isinstance(v, Path) else v
                    for k, v in vars(args).items() if k != 'phase'}, jobs=jobs,
                    sources=sources, split=split, audit=audit_metadata,
                    torch_version=str(torch.__version__), test_used=False,
                    scope='Exploratory single-task checkpoint branches; new paired future RNG, stateless SGD; not exact continuation',
                    primary_metric='validation onehot_mse; lower is better',
                    loss_matching='first downward audit-grid crossing after fork; no interpolation; overshoot retained')
    path = args.output / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Existing branch protocol differs; use a separate output')
    write_json(path, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['plan', 'train', 'evaluate'], default='plan')
    parser.add_argument('--source', type=Path, default=RESULTS / 'cifar' / 'capacity_development')
    parser.add_argument('--source-kind', choices=['capacity', 'local'], default='capacity')
    parser.add_argument('--output', type=Path, default=RESULTS / 'cifar' / 'checkpoint_branches_development')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 12])
    parser.add_argument('--source-step', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--audit-every', type=int, default=50)
    parser.add_argument('--audit-size', type=int, default=512)
    parser.add_argument('--audit-seed', type=int, default=20261020)
    parser.add_argument('--future-seed', type=int, default=930000)
    parser.add_argument('--loss-levels', type=float, nargs='+', default=[-.05, -.1, -.2, -.3, -.4])
    parser.add_argument('--loss-tolerance', type=float, default=.01)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--feature-batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if any(getattr(args, k) < 1 for k in ['steps', 'eval_every', 'audit_every', 'batch_size', 'feature_batch']) or args.audit_size < 2 or args.batch_size < 2:
        parser.error('Positive budgets/batch sizes required; moments require at least two images')
    if args.source_step < 0 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Nonnegative source step and unique nonnegative seeds required')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.loss_tolerance) or args.loss_tolerance <= 0:
        parser.error('Positive finite learning rate and loss tolerance required')
    if len(set(args.loss_levels)) != len(args.loss_levels) or not all(math.isfinite(x) for x in args.loss_levels):
        parser.error('Unique finite loss levels required')
    if args.output.resolve() == args.source.resolve() or args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('Output must be separate from the source study')
    jobs = make_jobs(args)
    if args.phase == 'plan':
        print(json.dumps(dict(jobs=jobs, source_checkpoints=[str(source_paths(args, seed)[0]) for seed in args.seeds],
                              source_step=args.source_step, branch_steps=args.steps,
                              status='plan only; no data, checkpoint or GPU access'), indent=2))
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = {str(seed): source_metadata(args, seed) for seed in args.seeds}
    source_protocol = read_json(args.source / 'protocol.json')
    data, labels, train, validation, split = load_data(EXPERIMENTS.parent, args.device)
    if args.audit_size > len(train):
        raise ValueError('Audit size exceeds training split')
    audit, audit_metadata = make_audit(data, train, args.audit_size, args.audit_seed)
    prepare_protocol(args, jobs, sources, split, audit_metadata, source_protocol)
    if args.phase == 'evaluate':
        # Require the complete training suite before any branch label evaluation.
        for job in jobs:
            run = read_json(args.output / job['id'] / 'result.json')
            if run['job'] != job or run['steps'] != args.steps or run['source'] != sources[str(job['seed'])]:
                raise ValueError('Incomplete or incompatible planned branch')
        for job in jobs:
            evaluate_job(args, job, data, labels, train, validation)
    else:
        for job in jobs:
            train_job(args, job, sources[str(job['seed'])], data, train, audit)


if __name__ == '__main__':
    main()
