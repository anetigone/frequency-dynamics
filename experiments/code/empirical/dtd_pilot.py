"""Hugging Face DTD square-model learning pilot and optional paired checkpoint forks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

import argparse
import json
import math
from functools import partial

from dtd_hf import REPO, REVISION, FILES, download, prepare, load_prepared


def augment_dtd(x, generator):
    import torch
    from torch.nn import functional as F
    size, batch = x.shape[-1], len(x)
    padding = max(1, size // 8)
    padded = F.pad(x, (padding,) * 4, mode='reflect')
    offset = torch.randint(2 * padding + 1, (batch, 2), generator=generator, device=x.device)
    y = torch.arange(size, device=x.device)[None, :, None] + offset[:, 0, None, None]
    z = torch.arange(size, device=x.device)[None, None, :] + offset[:, 1, None, None]
    result = padded[torch.arange(batch, device=x.device)[:, None, None], 0, y, z][:, None]
    flip = torch.rand(batch, 1, 1, 1, generator=generator, device=x.device) < .5
    result = torch.where(flip, result.flip(-1), result)
    contrast = .9 + .2 * torch.rand(batch, 1, 1, 1, generator=generator, device=x.device)
    return result * contrast + .01 * torch.randn(result.shape, generator=generator, device=x.device)


def fit_probe_dtd(features, labels, train, validation):
    import torch
    from torch.nn import functional as F
    from cifar_capacity import spectrum
    matrix = features.double()
    mean, std = matrix[train].mean(0), matrix[train].std(0).clamp_min(1e-12)
    design = torch.cat((torch.ones(len(matrix), 1, dtype=torch.float64), (matrix - mean) / std), 1)
    fit = torch.linalg.lstsq(design[train], F.one_hot(labels[train], 47).double(), driver='gelsd')
    def metrics(indices):
        scores, target = design[indices] @ fit.solution, labels[indices]
        mask = F.one_hot(target, 47).bool()
        return dict(accuracy=float((scores.argmax(1) == target).double().mean()),
                    onehot_mse=float((scores - mask.double()).square().mean()),
                    mean_true_class_margin=float((scores[mask] - scores.masked_fill(mask, -torch.inf).max(1).values).mean()),
                    confusion=torch.bincount(target * 47 + scores.argmax(1), minlength=47 ** 2).reshape(47, 47).tolist())
    return dict(train=metrics(train), validation=metrics(validation), design_rank_with_intercept=int(fit.rank),
                singular_values=fit.singular_values.tolist(), training_feature_spectrum=spectrum(matrix[train]))


def make_jobs(seeds, branching=False, dc_free=False):
    from cifar_checkpoint_branches import BRANCHES
    branches = BRANCHES if branching else [b for b in BRANCHES if b['name'] not in ['slow', 'fast']]
    if dc_free:
        if branching:
            raise ValueError('DC-free pilot is a from-initialization r=1 study')
        branches = [b for b in branches if b['name'] == 'baseline']
    return [dict(id=f'seed_{seed}_{branch["name"]}', seed=seed, **branch) for seed in seeds for branch in branches]


def initialize_sources(args, data, train, data_config):
    import torch
    from cifar_capacity import initialize
    indices = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(20261021))[:min(256, len(train))]]
    calibration = data[indices.to(data.device)]
    sources = {}
    dc_free = getattr(args, 'dc_free', False)
    for seed in args.seeds:
        path = args.output / 'initializations' / f'seed_{seed}.pt'
        if path.exists():
            previous = torch.load(path, map_location='cpu', weights_only=True)
            identity = previous['identity']
            if identity['seed'] != seed or identity.get('data_config', data_config) != data_config:
                raise ValueError('Existing DTD initialization settings differ')
            if bool(identity.get('dc_free', False)) != dc_free:
                raise ValueError('Existing DTD initialization kernel constraint differs; use a separate output directory')
            info = dict(identity['calibration'])
            info.pop('initial_state_sha256', None)
        else:
            extra = dict(zero_dc=True) if dc_free else {}
            model, info = initialize(seed, 64, 8, calibration, args.trace_draws, record_state_hash=False, **extra)
            path.parent.mkdir(parents=True, exist_ok=True)
            identity = dict(seed=seed, data_config=data_config, calibration=info)
            if dc_free:
                identity['dc_free'] = True
            torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()},
                       identity=identity), path)
        sources[str(seed)] = dict(checkpoint=str(path.resolve()), common_mobility=info['common_mobility'], calibration=info)
    return sources


def branch_sources(args, split):
    from cifar_checkpoint_branches import read_json
    protocol = read_json(args.source / 'protocol.json')
    original = protocol['arguments']
    if protocol.get('dataset_name') != 'DTD' or protocol['split'] != split or original['branching'] or original.get('dc_free', False):
        raise ValueError('DTD forks require a matching from-initialization pilot source')
    for key in ['lr', 'batch_size', 'image_size']:
        if original[key] != getattr(args, key):
            raise ValueError(f'Source and branch {key} differ')
    if original['steps'] < args.source_step:
        raise ValueError('Requested fork exceeds source budget')
    sources = {}
    for seed in args.seeds:
        directory = args.source / f'seed_{seed}_baseline'
        result = read_json(directory / 'result.json')
        if result['job']['seed'] != seed or result['job']['name'] != 'baseline' or result['steps'] != original['steps']:
            raise ValueError('Incomplete or incorrect DTD source baseline')
        snapshot = next((s for s in result['snapshots'] if s['step'] == args.source_step and s['fixed']), None)
        if snapshot is None:
            raise ValueError('Fork source must be a saved fixed-step pilot checkpoint')
        path = directory / snapshot['file']
        sources[str(seed)] = dict(checkpoint=str(path.resolve()),
                common_mobility=result['source']['common_mobility'], source_step=args.source_step)
    return sources


def load_model(args, seed, device, sources):
    import torch
    from cifar_local import NaturalSquare
    payload = torch.load(sources[str(seed)]['checkpoint'], map_location=device, weights_only=True)
    if args.branching:
        if payload['branch_step'] != args.source_step or payload['job']['seed'] != seed or payload['job']['name'] != 'baseline':
            raise ValueError('DTD fork checkpoint metadata mismatch')
        if payload['optimizer']['state']:
            raise ValueError('Only stateless source SGD is supported')
    model = NaturalSquare().to(device)
    model.load_state_dict(payload['model'], strict=True)
    if any(not bool(p.isfinite().all()) for p in model.parameters()):
        raise ValueError('Nonfinite DTD source parameters')
    return model


def lock_protocol(args, jobs, sources, split, audit):
    import torch
    from cifar_checkpoint_branches import read_json, write_json
    protocol = dict(schema_version=1, dataset_name='DTD',
                    arguments={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                               if k != 'phase' and not (k == 'dc_free' and not v)},
                    branch_names=list(dict.fromkeys(j['name'] for j in jobs)), jobs=jobs, sources=sources,
                    split=split, audit=audit, torch_version=str(torch.__version__), test_used=False,
                    primary_metric='validation onehot_mse, 47 classes; accuracy is secondary',
                    scope='Exploratory grayscale texture pilot; matched parameter starts and paired future streams',
                    augmentation='reflection crop +/-size/8, horizontal flip, contrast .9..1.1, Gaussian .01; no per-image normalization',
                    source_rng='new paired future stream; fork does not reconstruct the historical source RNG')
    if getattr(args, 'dc_free', False):
        protocol.update(kernel_constraint='spatial_zero_mean',
                        scope='Exploratory from-initialization DC-free r=1 training; not matched-function to the original pilot',
                        calibration='Clean energy 0.01; front mobility uses projected VJPs; both blocks equalized without task labels',
                        constraint_update='project kernel gradients and reproject kernels after each stateless SGD update',
                        source_rng='same pilot future_seed rule; no replay hashes or historical-stream reconstruction')
    destination = args.output / 'protocol.json'
    if destination.exists() and read_json(destination) != protocol:
        raise ValueError('Existing DTD protocol differs; keep arguments or use a new output directory')
    write_json(destination, protocol)
    return protocol


def run(args):
    import torch
    from cifar_checkpoint_branches import make_audit, train_job, evaluate_job, read_json
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, labels, train, validation, metadata = load_prepared(args.data_root, args.image_size)
    data = data.to(args.device)
    split = dict(dataset='DTD', repo_id=metadata['repo_id'], revision=metadata['revision'],
                 train_size=len(train), validation_size=len(validation),
                 image_size=args.image_size, mean=metadata['mean'], std=metadata['std'], test_used=False)
    dc_free = getattr(args, 'dc_free', False)
    jobs = make_jobs(args.seeds, args.branching, dc_free)
    # Evaluate never creates or reinitializes missing sources.
    if args.phase == 'evaluate':
        existing = read_json(args.output / 'protocol.json')
        sources = existing['sources']
    else:
        sources = branch_sources(args, split) if args.branching else initialize_sources(args, data, train, split)
    if args.audit_size > len(train):
        raise ValueError('Audit size exceeds training split')
    audit, audit_metadata = make_audit(data, train, args.audit_size, args.audit_seed,
                                     augment_fn=augment_dtd, record_hashes=not dc_free)
    lock_protocol(args, jobs, sources, split, audit_metadata)
    loader = partial(load_model, sources=sources)
    if args.phase == 'train':
        for job in jobs:
            train_job(args, job, sources[str(job['seed'])], data, train, audit,
                      source_loader=loader, augment_fn=augment_dtd, zero_dc=dc_free, record_hashes=not dc_free)
    else:
        for job in jobs:
            result = read_json(args.output / job['id'] / 'result.json')
            if result['job'] != job or result['steps'] != args.steps or result['source'] != sources[str(job['seed'])]:
                raise ValueError('Complete all planned DTD training before evaluation')
        for job in jobs:
            evaluate_job(args, job, data, labels, train, validation, probe_fn=fit_probe_dtd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['plan', 'download', 'prepare', 'train', 'evaluate', 'summarize'], default='plan')
    parser.add_argument('--data-root', type=Path, default=EXPERIMENTS / 'data' / 'dtd_hf')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--branching', action='store_true')
    parser.add_argument('--dc-free', action='store_true', help='From-initialization spatial-zero-mean kernels; r=1 only')
    parser.add_argument('--source', type=Path, default=RESULTS / 'empirical' / 'dtd_pilot')
    parser.add_argument('--source-step', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 12])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--audit-every', type=int, default=10)
    parser.add_argument('--audit-size', type=int, default=256)
    parser.add_argument('--audit-seed', type=int, default=20261022)
    parser.add_argument('--future-seed', type=int)
    parser.add_argument('--loss-levels', type=float, nargs='+', default=[-.05, -.1, -.2, -.3, -.4])
    parser.add_argument('--loss-tolerance', type=float, default=.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--feature-batch', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--trace-draws', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--download-workers', type=int, default=8)
    args = parser.parse_args()
    if args.dc_free and args.branching:
        parser.error('--dc-free is a from-initialization r=1 study; do not combine with --branching')
    args.output = args.output or RESULTS / 'empirical' / ('dtd_dc_free' if args.dc_free else 'dtd_branches' if args.branching else 'dtd_pilot')
    args.steps = args.steps if args.steps is not None else (1000 if args.branching else 2000)
    args.eval_every = args.eval_every if args.eval_every is not None else (250 if args.branching else 500)
    args.future_seed = args.future_seed if args.future_seed is not None else (950000 if args.branching else 940000)
    if args.image_size < 16 or min(args.batch_size, args.audit_size) < 2:
        parser.error('Image size >=16 and moment batches >=2 required')
    if min(args.steps, args.eval_every, args.audit_every, args.feature_batch, args.trace_draws, args.download_workers) < 1:
        parser.error('Positive budgets, intervals, trace draws and worker count required')
    if args.source_step < 0 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Unique nonnegative seeds and nonnegative source step required')
    if min(args.lr, args.loss_tolerance) <= 0 or not all(math.isfinite(v) for v in [args.lr, args.loss_tolerance, *args.loss_levels]):
        parser.error('Finite settings and positive learning rate/tolerance required')
    if len(set(args.loss_levels)) != len(args.loss_levels):
        parser.error('Loss levels must be unique')
    if args.branching and (args.output.resolve() == args.source.resolve() or args.output.resolve().is_relative_to(args.source.resolve())):
        parser.error('Fork output must be separate from source pilot')
    if args.phase == 'plan':
        # No Torch/PIL/HF import, network access or data reads.
        branches = [('ratio_0.1', .1, 1.), ('baseline', 1., 1.), ('ratio_10', 10., 1.)]
        if args.dc_free:
            branches = [('baseline', 1., 1.)]
        if args.branching:
            branches += [('slow', 1., .5), ('fast', 1., 2.)]
        print(json.dumps(dict(dataset=REPO, revision=REVISION, files=list(FILES),
              image_selection='official partition 1: 1880 train + 1880 validation JPEGs; no test images',
              jobs=[dict(seed=s, branch=n, ratio=r, speed=v) for s in args.seeds for n, r, v in branches],
              steps=args.steps, image_size=args.image_size, output=str(args.output), test_used=False,
              kernel_constraint='spatial_zero_mean' if args.dc_free else 'unconstrained',
              status='plan only; no download, data read or training'), indent=2))
    elif args.phase == 'download':
        paths = download(args.data_root, workers=args.download_workers)
        print(json.dumps(paths, indent=2))
    elif args.phase == 'prepare':
        print(prepare(args.data_root, args.image_size))
    elif args.phase == 'summarize':
        from summarize_checkpoint_branches import summarize
        result = summarize(args.output)
        print(f'Summarized {result["n_seeds"]} paired seeds: {args.output / "summary.json"}')
    else:
        run(args)


if __name__ == '__main__':
    main()
