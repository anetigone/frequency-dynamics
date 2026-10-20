"""Frozen-encoder CIFAR head diagnosis: unchanged source head, optimal LS rank-K, and CE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

import argparse
import json
import math

import torch
from torch.nn import functional as F

from cifar_local import NaturalSquare, augment, load_data
from cifar_capacity import fit_probe, score_metrics, supervised_head, write_json


def reduced_rank(features, labels, train, validation, outputs):
    """Global rank-K affine least-squares fit on TRAIN labels, using a linear raw-feature head."""
    x = features.double()
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-12)
    z = (x - mean) / std
    y = F.one_hot(labels[train], 10).double()
    y_mean = y.mean(0)
    u, singular, vh = torch.linalg.svd(z[train], full_matrices=False)
    tolerance = torch.finfo(z.dtype).eps * max(z[train].shape) * singular.max()
    rank = int((singular > tolerance).sum())
    coefficients = u[:, :rank].T @ (y - y_mean)
    left, values, right = torch.linalg.svd(coefficients, full_matrices=False)
    kept = min(outputs, rank, len(values))
    # z @ basis @ decoder + y_mean; convert to raw linear projection plus decoder bias.
    basis = (vh[:rank].T / singular[:rank]) @ left[:, :kept]
    decoder = values[:kept, None] * right[:kept]
    projection = torch.zeros(x.shape[1], outputs, dtype=torch.float64)
    projection[:, :kept] = basis / std[:, None]
    classifier = torch.zeros(outputs, 10, dtype=torch.float64)
    classifier[:kept] = decoder
    bias = y_mean - mean @ projection @ classifier
    scores = x @ projection @ classifier + bias
    return dict(projection=projection, classifier=classifier, bias=bias,
                feature_rank=rank, retained_components=kept,
                train=score_metrics(scores[train], labels[train]),
                validation=score_metrics(scores[validation], labels[validation]))


@torch.no_grad()
def features_for(model, data, batch):
    return torch.cat([model.features(data[i:i + batch]).cpu() for i in range(0, len(data), batch)])


@torch.no_grad()
def classifier_metrics(features, model, classifier, mean, std, labels, train, validation):
    x = features.double() @ model.weight.detach().cpu().double().T
    logits = ((x - mean.cpu().double()) / std.cpu().double()) @ classifier.weight.detach().cpu().double().T + classifier.bias.detach().cpu().double()
    return {name: dict(accuracy=float((logits[index].argmax(1) == labels[index]).double().mean()),
                       cross_entropy=float(F.cross_entropy(logits[index], labels[index])))
            for name, index in [('train', train), ('validation', validation)]}


def train_frozen_head(model, data, labels, train, calibration, seed, steps, batch_size, lr, classifier_lr, clip):
    """Only W and the classifier belong to the optimizer; the pixel kernel is immutable."""
    model.kernel.requires_grad_(False)
    model.weight.requires_grad_(True)
    original_kernel = model.kernel.detach().clone()
    original_weight = model.weight.detach().clone()
    classifier, mean, std = supervised_head(model, 'supervised_projection', calibration, seed)
    optimizer = torch.optim.Adam([{'params': [model.weight], 'lr': lr},
                                  {'params': classifier.parameters(), 'lr': classifier_lr}], weight_decay=0.)
    generator = torch.Generator(device=data.device).manual_seed(820000 + seed)
    indices = train.to(data.device)
    history, clipped = [], 0
    maximum_norm, maximum_loss = 0., -math.inf
    for step in range(1, steps + 1):
        index = indices[torch.randint(len(indices), (batch_size,), generator=generator, device=data.device)]
        x1, x2 = augment(data[index], generator), augment(data[index], generator)
        with torch.no_grad():
            f1, f2 = model.features(x1), model.features(x2)
        target = labels[index.cpu()].to(data.device)
        optimizer.zero_grad(set_to_none=True)
        a = classifier((f1 @ model.weight.T - mean) / std)
        b = classifier((f2 @ model.weight.T - mean) / std)
        loss = .5 * (F.cross_entropy(a, target) + F.cross_entropy(b, target))
        if not bool(loss.isfinite()):
            raise FloatingPointError(f'Frozen head seed={seed} step={step}: nonfinite CE')
        loss.backward()
        parameters = [model.weight, *classifier.parameters()]
        if any(not bool(p.grad.isfinite().all()) for p in parameters):
            raise FloatingPointError(f'Frozen head seed={seed} step={step}: nonfinite gradient')
        try:
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True))
        except RuntimeError as error:
            raise FloatingPointError(f'Frozen head seed={seed} step={step}: nonfinite gradient norm') from error
        clipped += int(norm > clip)
        maximum_norm, maximum_loss = max(maximum_norm, norm), max(maximum_loss, float(loss.detach()))
        optimizer.step()
        if any(not bool(p.isfinite().all()) for p in parameters):
            raise FloatingPointError(f'Frozen head seed={seed} step={step}: nonfinite parameter')
        if step == 1 or step % 100 == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss.detach()), gradient_norm_before_clip=norm))
        if step % 500 == 0:
            print(f'Frozen head seed={seed}: {step}/{steps}', flush=True)
    if model.kernel.grad is not None or not torch.equal(model.kernel, original_kernel):
        raise AssertionError('Encoder changed during frozen-head training')
    return classifier, mean, std, dict(history=history, clipped_updates=clipped,
                                      max_gradient_norm_before_clip=maximum_norm, max_training_loss=maximum_loss,
                                      encoder_unchanged_exactly=True,
                                      projection_changed=not torch.equal(original_weight, model.weight))


def run_state(args, job, data, labels, train, validation, calibration):
    directory = args.output / job['id']
    if (directory / 'result.json').exists():
        print(f"Already complete: {job['id']}", flush=True)
        return
    print(f"Starting {job['id']}; source={job['checkpoint']}", flush=True)
    payload = torch.load(job['checkpoint'], map_location=args.device, weights_only=True)
    source_job = payload['job']
    expected = dict(seed=job['seed'], architecture='baseline', objective='unsupervised', ratio=args.source_ratio,
                    channels=64, outputs=8)
    if any(source_job.get(key) != value for key, value in expected.items()) or payload['step'] != job['source_step']:
        raise ValueError('Source checkpoint metadata differs from requested baseline encoder')
    model = NaturalSquare(64, 8).to(args.device)
    model.load_state_dict(payload['model'])
    model.eval()
    model.kernel.requires_grad_(False)
    raw_features = features_for(model, data, args.feature_batch)
    original_projection = raw_features.double() @ model.weight.detach().cpu().double().T
    full_probe = fit_probe(raw_features, labels, train, validation)
    original_probe = fit_probe(original_projection, labels, train, validation)
    optimal = reduced_rank(raw_features, labels, train, validation, 8)
    optimal_projection = raw_features.double() @ optimal['projection']
    optimal_probe = fit_probe(optimal_projection, labels, train, validation)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(dict(projection=optimal['projection'], classifier=optimal['classifier'], bias=optimal['bias'],
                    source_checkpoint=job['checkpoint'], training_labels_only=True), directory / 'reduced_rank_head.pt')
    try:
        classifier, mean, std, optimization = train_frozen_head(model, data, labels, train, calibration,
                                                               job['seed'], args.steps, args.batch_size,
                                                               args.lr, args.classifier_lr, args.grad_clip)
    except FloatingPointError as error:
        write_json(directory / 'failure.json', dict(job=job, error=str(error), status='failed, no completed result'))
        raise
    learned_projection = raw_features.double() @ model.weight.detach().cpu().double().T
    learned_probe = fit_probe(learned_projection, labels, train, validation)
    # Re-extract a fixed subset to check that the cached features are still current.
    check = model.features(data[:min(args.feature_batch, len(data))]).detach().cpu()
    if not torch.allclose(check, raw_features[:len(check)], rtol=2e-5, atol=2e-6):
        raise AssertionError('Cached encoder features differ after frozen-head training')
    learned_classifier = classifier_metrics(raw_features, model, classifier, mean, std, labels, train, validation)
    torch.save(dict(model=model.state_dict(), classifier=classifier.state_dict(), standardizer=(mean, std),
                    source_checkpoint=job['checkpoint'], encoder_frozen=True), directory / 'head_final.pt')
    write_json(directory / 'result.json', dict(job=job, steps=args.steps, encoder_full_ols=full_probe,
                 source_projection_ols=original_probe,
                 reduced_rank=dict(feature_rank=optimal['feature_rank'], retained_components=optimal['retained_components'],
                                   direct_train=optimal['train'], direct_validation=optimal['validation'], projection_ols=optimal_probe),
                 trained_projection_ols=learned_probe, trained_classifier=learned_classifier,
                 optimization=optimization, test_used=False,
                 interpretation='Same frozen encoder for every head. Rank-K LS is globally training-MSE-optimal, not validation-accuracy-optimal.'))
    print(f"Completed {job['id']}; encoder unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['plan', 'train'], default='plan')
    parser.add_argument('--source', type=Path, default=RESULTS / 'cifar' / 'capacity_development')
    parser.add_argument('--output', type=Path, default=RESULTS / 'cifar' / 'frozen_encoder_heads')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 12])
    parser.add_argument('--source-steps', nargs='+', type=int, default=[0, 4000])
    parser.add_argument('--source-ratio', type=float, default=1.)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--feature-batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--classifier-lr', type=float, default=.001)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.feature_batch) < 1 or any(not math.isfinite(v) or v <= 0 for v in [args.lr, args.classifier_lr, args.grad_clip, args.source_ratio]):
        parser.error('Positive finite learning rates, clipping, batch sizes, ratio and budget required')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.source_steps)) != len(args.source_steps) or any(v < 0 for v in args.source_steps):
        parser.error('Unique seeds and unique nonnegative source steps required')
    if args.source.resolve() == args.output.resolve() or args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('Use an output separate from the original capacity study')
    jobs = [dict(id=f'seed_{seed}_source_{step}', seed=seed, source_step=step,
                 checkpoint=str((args.source / f'seed_{seed}_baseline_unsupervised_r{args.source_ratio:g}' / f'step_{step}.pt').resolve()))
            for seed in args.seeds for step in args.source_steps]
    if args.phase == 'plan':
        print(json.dumps(dict(jobs=jobs, run_count=len(jobs), outputs=8, source_validation=False,
                              status='plan only; checkpoints not required yet'), indent=2))
        return
    missing = [job['checkpoint'] for job in jobs if not (args.output/job['id']/'result.json').exists()
               and not Path(job['checkpoint']).is_file()]
    if missing:
        raise FileNotFoundError('Run on the server containing the source checkpoints; summary.json alone is insufficient: ' + ', '.join(missing))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, labels, train, validation, split = load_data(EXPERIMENTS.parent, args.device)
    source_protocol = json.loads((args.source / 'protocol.json').read_text(encoding='utf-8'))
    if source_protocol['split'] != split:
        raise ValueError('Source and current data split/normalization differ')
    config = dict(arguments={k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items() if k!='phase'},
                  jobs=jobs, split=split, source_validation=False, test_used=False,
                  objective='frozen encoder, supervised projection/classifier only',
                  reduced_rank='training-label-only global affine rank-8 least-squares reference')
    lock = args.output / 'protocol.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing frozen-head configuration differs; preserve the original arguments')
    write_json(lock, config)
    generator = torch.Generator().manual_seed(20261096)
    calibration = data[train[torch.randperm(len(train), generator=generator)[:256]].to(data.device)]
    for job in jobs:
        run_state(args, job, data, labels, train, validation, calibration)


if __name__ == '__main__':
    main()
