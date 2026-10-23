"""Fixed moment tasks and long-horizon GUPA on official natural-image splits.

Default is a data-free plan. Real training requires the explicit ``train`` phase.
Use ``--until-epoch 200`` for the first stage of the fixed 5,000-epoch budget;
the same command with 5,000 resumes model, stateless SGD and minibatch/view RNG.
``smoke`` uses small synthetic images only, without downloads or real training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

def load_runtime():
    # Keep data-free planning and matplotlib summaries out of the Torch process.
    # This also avoids mixing the Windows Torch/NumPy OpenMP runtimes in plotting.
    global torch, F, initialize, audit_model, make_audit, NaturalSquare, augment, augment_dtd, empirical_loss
    import torch
    from torch.nn import functional as F
    from cifar_capacity import initialize
    from cifar_checkpoint_branches import audit_model, make_audit
    from cifar_local import NaturalSquare, augment
    from dtd_pilot import augment_dtd
    from square_images import empirical_loss


EVAL_EPOCHS = [0, .1, .25, .5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50,
               60, 80, 100, 125, 150, 175, 200, 300, 500, 750, 1000,
               1500, 2000, 3000, 4000, 5000]
KNN_EPOCHS = [0, 10, 20, 50, 80, 100, 125, 150, 200, 500, 1000, 2000, 5000]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_tensor(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def epoch_step(epoch, n, batch):
    return int(math.ceil(epoch * n / batch))


def grids(args, n):
    steps = epoch_step(args.epochs, n, args.batch_size)
    evaluations = sorted({0, steps, *(epoch_step(e, n, args.batch_size)
                                     for e in args.eval_epochs if 0 <= e <= args.epochs)})
    knn = [] if getattr(args, 'skip_knn', False) else sorted({
        0, steps, *(epoch_step(e, n, args.batch_size)
                    for e in args.knn_epochs if 0 <= e <= args.epochs)})
    # Every KNN observation is also a full moment/OLS observation.
    return steps, sorted(set(evaluations) | set(knn)), knn


def load_images(args):
    if args.dataset == 'dtd':
        from dtd_hf import load_prepared
        data, labels, train, evaluation, info = load_prepared(args.data_root, args.image_size)
        split = dict(dataset='DTD', train_size=len(train), evaluation_size=len(evaluation),
                     evaluation_split='official partition 1 validation', classes=47,
                     image_size=args.image_size, normalization_mean=info['mean'],
                     normalization_std=info['std'], test_used=False)
    else:
        from torchvision.datasets import CIFAR10
        arrays, targets = [], []
        for training in [True, False]:
            dataset = CIFAR10(str(args.data_root), train=training, download=False)
            rgb = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255
            arrays.append(rgb[:, :1] * .2989 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114)
            targets.append(torch.tensor(dataset.targets))
        mean, std = arrays[0].mean(), arrays[0].std()
        data = (torch.cat(arrays) - mean) / std
        labels = torch.cat(targets)
        train, evaluation = torch.arange(50000), torch.arange(50000, 60000)
        split = dict(dataset='CIFAR-10', train_size=50000, evaluation_size=10000,
                     evaluation_split='official test, fixed checkpoints without model selection',
                     classes=10, image_size=32, normalization_mean=float(mean),
                     normalization_std=float(std), test_used=True)
    if not bool(data.isfinite().all()) or data.ndim != 4 or data.shape[1] != 1:
        raise ValueError('Images must be finite grayscale NCHW tensors')
    if len(torch.unique(labels[train])) != split['classes']:
        raise ValueError('Training class count differs from the dataset')
    return data.to(args.device), labels, train, evaluation, split


def protocol(args, split):
    steps, evaluations, knn = grids(args, split['train_size'])
    keys = ['dataset', 'seeds', 'epochs', 'batch_size', 'lr', 'ratio', 'trace_draws',
            'prediction_batches', 'knn_k', 'risk_threshold', 'audit_size', 'audit_every']
    if getattr(args, 'skip_knn', False):
        keys.append('skip_knn')
    return dict(arguments={k: getattr(args, k) for k in keys}, split=split,
                steps=steps, evaluation_steps=evaluations, knn_steps=knn,
                architecture='64 x 11x11 valid convolution, square, global average, linear 8-output head',
                epoch_definition='step * batch_size / official training size; iid sampling with replacement',
                optimizer='SGD, no momentum, weight decay, clipping or normalization',
                initialization='Independent Gaussian filters/heads, energy .01 and label-free Jacobian-trace calibration',
                task_definition='Fixed 11x11 patch second-moment functionals; see task_names and task_definitions.json',
                prediction='Analytic OLS directional derivative at initialization along an independent '
                'averaged unlabelled-gradient update; two half-estimates check sign stability. '
                'These are local predictions, not finite-time rebound predictions',
                local_slope_tolerance=1e-12,
                evaluation='Affine unregularized OLS fitted on the complete official train split; '
                'train empirical optimum and held-out error reported separately, neither is exact population risk',
                knn='Euclidean kNN: raw, train-coordinate-standardized, train-whitened; no k tuning',
                selection='All seeds, fixed tasks and checkpoints; held-out labels never used for updates, '
                'initialization, task choice, stopping, or hyperparameter/checkpoint selection',
                caveat='An architecture/objective/time-scale comparison, not an exact reproduction '
                'of another paper\'s augmentation, optimizer, initialization or kNN settings')


def check_protocol(args, split, create=False):
    expected = protocol(args, split)
    path = args.output / 'protocol.json'
    if path.exists():
        if read_json(path) != expected:
            raise ValueError('Experiment parameters differ; retain them or use a separate output directory')
    elif create:
        write_json(path, expected)
    else:
        raise FileNotFoundError('Run train first to create the protocol and checkpoints')
    return expected


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def training_step(model, generator, data, train_device, args, rates, augment_fn):
    index = train_device[torch.randint(len(train_device), (args.batch_size,),
                                     generator=generator, device=data.device)]
    first, second = augment_fn(data[index], generator), augment_fn(data[index], generator)
    model.zero_grad(set_to_none=True)
    loss = empirical_loss(model(first), model(second))
    if not bool(loss.isfinite()):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    with torch.no_grad():
        for parameter, rate in zip([model.kernel, model.weight], rates):
            if parameter.grad is None or not bool(parameter.grad.isfinite().all()):
                raise FloatingPointError('Nonfinite or missing training gradient')
            parameter.add_(parameter.grad, alpha=-rate)
            if not bool(parameter.isfinite().all()):
                raise FloatingPointError('Nonfinite training parameter')
    return float(loss.detach())


def prediction_state(model, calibration, seed, data, train, args, augment_fn):
    """Independent gradient average, before any trajectory outcomes are measured."""
    generator = torch.Generator(device=data.device).manual_seed(930000 + seed)
    halves = [[torch.zeros_like(p) for p in model.parameters()] for _ in range(2)]
    sizes = [(args.prediction_batches + 1) // 2, args.prediction_batches // 2]
    indices = train.to(data.device)
    for iteration in range(args.prediction_batches):
        chosen = indices[torch.randint(len(indices), (args.batch_size,),
                                      generator=generator, device=data.device)]
        first, second = augment_fn(data[chosen], generator), augment_fn(data[chosen], generator)
        loss = empirical_loss(model(first), model(second))
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        half = int(iteration >= sizes[0])
        for total, gradient in zip(halves[half], gradients):
            total.add_(gradient.detach() / sizes[half])
    base = args.lr * 2 * calibration['common_mobility'] / (1 + args.ratio)
    directions = [{name: (-rate * gradient).cpu() for (name, _), gradient, rate
                   in zip(model.named_parameters(), half, [base, base * args.ratio])} for half in halves]
    return dict(half_directions=directions, half_sizes=sizes, seed=seed,
                gradient_batches=args.prediction_batches,
                interpretation='Analytic initial OLS derivative along an independently estimated '
                'mean SGD update; split-half sign check, no finite-time rebound prediction')


def train_seed(args, seed, data, train, split, settings):
    directory = args.output / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / 'latest.pt'
    augment_fn = augment_dtd if args.dataset == 'dtd' else augment
    generator = torch.Generator(device=data.device).manual_seed(820000 + seed)
    model = NaturalSquare().to(data.device)
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if state['seed'] != seed:
            raise ValueError('Checkpoint seed differs')
        model.load_state_dict(state['model'])
        generator.set_state(state['generator_state'])
        start, calibration, history = state['step'], state['calibration'], state['history']
    else:
        choose = torch.randperm(len(train), generator=torch.Generator().manual_seed(20261096))
        calibration_x = data[train[choose[:min(256, len(train))]].to(data.device)]
        model, calibration = initialize(seed, 64, 8, calibration_x, args.trace_draws,
                                        record_state_hash=False)
        start, history = 0, []
        save_tensor(directory / 'prediction.pt', prediction_state(model, calibration, seed, data,
                                                                  train, args, augment_fn))
    base = args.lr * 2 * calibration['common_mobility'] / (1 + args.ratio)
    rates = [base, base * args.ratio]
    target = min(settings['steps'], epoch_step(args.until_epoch if args.until_epoch is not None
                                             else args.epochs, len(train), args.batch_size))
    if target < start:
        print(f'seed={seed}: already at step {start}, requested {target}; retaining checkpoints', flush=True)
        return
    audit, _ = make_audit(data, train, min(args.audit_size, len(train)), 20261023,
                         augment_fn=augment_fn, record_hashes=False)
    indices = train.to(data.device)
    evaluations = set(settings['evaluation_steps'])
    synchronize(args.device)
    begin = time.perf_counter()
    step = start

    def save_progress():
        state = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     generator_state=generator.get_state().cpu(), seed=seed, step=step,
                     calibration=calibration, history=history)
        save_tensor(latest, state)
        if step in evaluations and not (directory / f'step_{step}.pt').exists():
            save_tensor(directory / f'step_{step}.pt', state)
        write_json(directory / 'result.json', dict(seed=seed, completed_step=step,
                   target_steps=settings['steps'], state='complete' if step == settings['steps'] else 'partial',
                   exposed_epochs=step * args.batch_size / len(train), calibration=calibration,
                   learning_rates=rates, history=history))

    try:
        if start == 0:
            history.append(dict(step=0, epoch=0., **audit_model(model, audit)))
            save_progress()
        elif start in evaluations and not (directory / f'step_{start}.pt').exists():
            save_progress()
        for step in range(start + 1, target + 1):
            batch_loss = training_step(model, generator, data, indices, args, rates, augment_fn)
            if step % args.audit_every == 0 or step in evaluations or step == target:
                history.append(dict(step=step, epoch=step * args.batch_size / len(train),
                                    batch_loss=batch_loss, **audit_model(model, audit)))
            if step % args.save_every == 0 or step in evaluations or step == target:
                save_progress()
                synchronize(args.device)
                seconds = time.perf_counter() - begin
                print(f'seed={seed} step={step}/{settings["steps"]} '
                      f'epoch={step * args.batch_size / len(train):.3f} '
                      f'loss={history[-1]["loss"]:.6g} '
                      f'seconds/update={seconds / max(1, step - start):.5f}', flush=True)
    except (FloatingPointError, RuntimeError) as error:
        write_json(directory / f'failure_step_{step}.json',
                   dict(seed=seed, attempted_step=step, error=str(error),
                        resume='latest.pt retains the last successfully saved state; no automatic retry'))
        raise


def extract(model, data, batch):
    with torch.no_grad():
        features = torch.cat([model.features(data[i:i + batch]).cpu()
                              for i in range(0, len(data), batch)])
    if not bool(features.isfinite().all()):
        raise FloatingPointError('Nonfinite clean features')
    return features, features.double() @ model.weight.detach().cpu().double().T


def extract_direction(model, data, direction, batch):
    kernel = direction['kernel'].to(data.device)
    weight = direction['weight'].double()
    with torch.no_grad():
        feature_direction = torch.cat([
            (2 * F.conv2d(data[i:i + batch], model.kernel) *
             F.conv2d(data[i:i + batch], kernel)).mean((-2, -1)).cpu()
            for i in range(0, len(data), batch)]).double()
    features, _ = extract(model, data, batch)
    projection_direction = feature_direction @ model.weight.detach().cpu().double().T + features.double() @ weight.T
    return feature_direction, projection_direction


def evaluate(args, data, labels, train, heldout, split, settings):
    from natural_moment_eval import (fixed_moment_dictionary, fit_readouts, geometry_bound,
                                     knn_metrics, directional_readout_slopes)
    names, targets = fixed_moment_dictionary(data, batch_size=args.feature_batch)
    write_json(args.output / 'task_definitions.json',
               dict(task_names=names, source='natural_moment_eval.fixed_moment_dictionary',
                    scale='Targets standardized using official training moments only',
                    scope='Finite-image empirical and held-out risks, not known population risks'))
    max_step = epoch_step(args.until_epoch if args.until_epoch is not None else args.epochs,
                          len(train), args.batch_size)
    class_targets = F.one_hot(labels, split['classes']).double()
    for seed in args.seeds:
        directory = args.output / f'seed_{seed}'
        initial_path = directory / 'step_0.pt'
        if not initial_path.exists():
            print(f'seed={seed}: missing initialization; retained as missing in summary', flush=True)
            continue
        model = NaturalSquare().to(data.device)
        model.load_state_dict(torch.load(initial_path, map_location='cpu', weights_only=True)['model'])
        initial = extract(model, data, args.feature_batch)
        initial_readouts = fit_readouts(*initial, targets, labels, train, heldout, split['classes'])
        prediction_path = directory / 'prediction.json'
        if not prediction_path.exists():
            prediction = torch.load(directory / 'prediction.pt', map_location='cpu', weights_only=True)
            half_slopes = []
            for direction in prediction['half_directions']:
                tangent = extract_direction(model, data, direction, args.feature_batch)
                half_slopes.append(directional_readout_slopes(*initial, *tangent, targets, train))
            average = {rep: [(a * prediction['half_sizes'][0] + b * prediction['half_sizes'][1]) /
                             prediction['gradient_batches'] for a, b in zip(half_slopes[0][rep], half_slopes[1][rep])]
                       for rep in ['encoder', 'projection']}
            write_json(prediction_path, dict(seed=seed, task_names=names,
                       slopes=average, half_slopes=half_slopes,
                       gradient_batches=prediction['gradient_batches'],
                       interpretation=prediction['interpretation']))
        for step in settings['evaluation_steps']:
            if step > max_step:
                continue
            checkpoint = directory / f'step_{step}.pt'
            destination = directory / f'evaluation_{step}.json'
            if not checkpoint.exists():
                continue
            if destination.exists():
                continue
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
            current = initial if step == 0 else extract(model, data, args.feature_batch)
            row = dict(seed=seed, step=step, epoch=step * args.batch_size / len(train), task_names=names,
                       readouts=initial_readouts if step == 0 else fit_readouts(*current, targets, labels,
                                                                            train, heldout, split['classes']),
                       geometry={}, knn={})
            for index, representation in enumerate(['encoder', 'projection']):
                row['geometry'][representation] = dict(
                    moments=geometry_bound(initial[index], current[index], targets, train),
                    classes=geometry_bound(initial[index], current[index], class_targets, train))
                if step in settings['knn_steps']:
                    row['knn'][representation] = knn_metrics(current[index], labels, train, heldout,
                                                            split['classes'], device=args.device,
                                                            k=args.knn_k, query_batch=args.knn_batch)
            write_json(destination, row)
            print(f'Evaluated seed={seed} epoch={row["epoch"]:.3f} step={step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=['plan', 'train', 'evaluate', 'summarize', 'smoke'], default='plan')
    parser.add_argument('--dataset', choices=['cifar10', 'dtd'], default='cifar10')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[30, 31, 32])
    parser.add_argument('--epochs', type=float, default=5000)
    parser.add_argument('--until-epoch', type=float)
    parser.add_argument('--eval-epochs', type=float, nargs='+', default=EVAL_EPOCHS)
    parser.add_argument('--knn-epochs', type=float, nargs='+', default=KNN_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--ratio', type=float, default=1.)
    parser.add_argument('--trace-draws', type=int, default=16)
    parser.add_argument('--prediction-batches', type=int, default=16)
    parser.add_argument('--feature-batch', type=int, default=256)
    parser.add_argument('--audit-size', type=int, default=256)
    parser.add_argument('--audit-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--knn-k', type=int, default=20)
    parser.add_argument('--skip-knn', action='store_true', help='Skip expensive pairwise kNN; retain OLS and geometry')
    parser.add_argument('--knn-batch', type=int, default=128)
    parser.add_argument('--risk-threshold', type=float, default=.001)
    parser.add_argument('--image-size', type=int, default=64, help='DTD only; use its prepared image size')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    numeric = [args.epochs, args.lr, args.ratio, args.risk_threshold, *args.eval_epochs, *args.knn_epochs]
    if args.until_epoch is not None:
        numeric.append(args.until_epoch)
    if not all(math.isfinite(value) for value in numeric) or min([*args.eval_epochs, *args.knn_epochs]) < 0:
        parser.error('Epochs, rates and thresholds must be finite; observation epochs must be nonnegative')
    if (min(args.epochs, args.lr, args.ratio) <= 0 or
        min(args.batch_size, args.trace_draws, args.prediction_batches, args.feature_batch,
            args.audit_size, args.audit_every, args.save_every, args.knn_k, args.knn_batch, args.threads) < 1 or
        args.batch_size < 2 or args.prediction_batches < 2 or args.risk_threshold < 0 or
        args.until_epoch is not None and not 0 < args.until_epoch <= args.epochs):
        parser.error('Positive finite budgets, rates, sizes and valid --until-epoch are required')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seeds must be distinct')
    args.output = args.output or RESULTS / 'empirical' / f'natural_moments_{args.dataset}'
    args.data_root = args.data_root or (EXPERIMENTS.parent if args.dataset == 'cifar10'
                                       else EXPERIMENTS / 'data' / 'dtd_hf')
    if args.phase == 'plan':
        n = 50000 if args.dataset == 'cifar10' else 1880
        steps, evaluations, knn = grids(args, n)
        print(json.dumps(dict(dataset=args.dataset, seeds=args.seeds, epochs=args.epochs,
                   steps_per_seed=steps, total_updates=steps * len(args.seeds),
                   first_stage_epoch=args.until_epoch, evaluation_steps=evaluations,
                   knn_steps=knn, output=str(args.output),
                   action='plan only; no dataset access, download or training'), indent=2))
        return
    if args.phase == 'summarize':
        from summarize_natural_moments import summarize
        summarize(args.output)
        return
    load_runtime()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.phase == 'smoke':
        synthetic_smoke(args)
        return
    data, labels, train, heldout, split = load_images(args)
    settings = check_protocol(args, split, create=args.phase == 'train')
    if args.phase == 'train':
        for seed in args.seeds:
            train_seed(args, seed, data, train, split, settings)
    else:
        evaluate(args, data, labels, train, heldout, split, settings)


def synthetic_smoke(original):
    """Exercise pause/resume, evaluation and RNG continuity on tiny fake data."""
    import tempfile
    import subprocess
    args = copy.copy(original)
    args.device = 'cpu'
    args.seeds, args.epochs, args.batch_size = [0], 1., 4
    args.eval_epochs, args.knn_epochs = [0, .5, 1], [0, 1]
    args.trace_draws, args.prediction_batches, args.knn_k = 2, 2, 3
    args.audit_size, args.audit_every, args.save_every, args.feature_batch = 8, 2, 2, 16
    generator = torch.Generator().manual_seed(20261022)
    data = torch.randn(60, 1, 32, 32, generator=generator)
    labels, train, heldout = torch.arange(60) % 10, torch.arange(40), torch.arange(40, 60)
    split = dict(dataset='synthetic smoke only', train_size=40, evaluation_size=20, classes=10,
                 evaluation_split='synthetic', image_size=32, test_used=False)
    parent = RESULTS / 'checks'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='natural_moment_smoke_', dir=parent) as temp:
        args.output = Path(temp) / 'resumed'
        settings = check_protocol(args, split, create=True)
        args.until_epoch = .5
        train_seed(args, 0, data, train, split, settings)
        args.until_epoch = 1.
        train_seed(args, 0, data, train, split, settings)
        resumed = torch.load(args.output / 'seed_0' / 'latest.pt', weights_only=True)
        evaluate(args, data, labels, train, heldout, split, settings)
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('summarize_natural_moments.py')),
                        str(args.output)], check=True)
        args.output = Path(temp) / 'uninterrupted'
        settings = check_protocol(args, split, create=True)
        train_seed(args, 0, data, train, split, settings)
        direct = torch.load(args.output / 'seed_0' / 'latest.pt', weights_only=True)
        for name in resumed['model']:
            torch.testing.assert_close(resumed['model'][name], direct['model'][name], rtol=0, atol=0)
        assert torch.equal(resumed['generator_state'], direct['generator_state'])
        report = dict(passed=True, images='60 synthetic random images; no real dataset loaded',
                      train_steps=10, pause_step=5, resumed_matches_uninterrupted=True,
                      phases=['train', 'pause/resume', 'evaluate', 'summarize'])
        write_json(parent / 'natural_moment_smoke.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
