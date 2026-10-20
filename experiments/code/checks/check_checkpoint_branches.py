"""Synthetic CPU integration checks for causal CIFAR branch plumbing and summaries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import copy
import tempfile
from types import SimpleNamespace

import torch
from cifar_local import initialize, augment
from square_images import empirical_loss
from cifar_checkpoint_branches import (BRANCHES, read_json, write_json, make_jobs,
    make_audit, audit_model, source_metadata, load_source, learning_rates, prepare_protocol,
    train_job, evaluate_job)
from summarize_checkpoint_branches import summarize


def rejects(function, text):
    try:
        function()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError(text)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random = torch.Generator().manual_seed(1234)
    data = torch.randn(60, 1, 32, 32, generator=random)
    labels = torch.arange(60) % 10
    train, validation = torch.arange(40), torch.arange(40, 60)
    root = RESULTS / 'checks'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint_branches_', dir=root) as temporary:
        folder = Path(temporary)
        args = SimpleNamespace(phase='train', source=folder / 'source', source_kind='local',
            output=folder / 'branches', seeds=[0, 1], source_step=1000, steps=2,
            eval_every=1, audit_every=1, audit_size=12, audit_seed=123,
            future_seed=930000, loss_levels=[1e6, -1e6], loss_tolerance=.01,
            batch_size=4, feature_batch=20, lr=.002, device='cpu')
        split = dict(train_size=40, validation_size=20, test_used=False)
        audit, audit_metadata = make_audit(data, train, args.audit_size, args.audit_seed)
        again, again_metadata = make_audit(data, train, args.audit_size, args.audit_seed)
        assert audit_metadata == again_metadata and all(torch.equal(a, b) for a, b in zip(audit, again))
        for seed in args.seeds:
            model, calibration = initialize(seed, data[:12], 'cpu')
            directory = args.source / f'seed_{seed}_r1'
            directory.mkdir(parents=True)
            torch.save(model.state_dict(), directory / 'step_1000.pt')
            write_json(directory / 'result.json', dict(seed=seed, ratio=1., steps=4000, calibration=calibration))
        sources = {str(s): source_metadata(args, s) for s in args.seeds}
        # A near-initial loss tests actual first crossing; extreme levels test missing matches.
        model = load_source(args, 0, 'cpu')
        args.loss_levels.extend(audit_model(load_source(args, seed, 'cpu'), audit)['loss'] - 1e-9
                                for seed in args.seeds)
        torch.testing.assert_close(torch.tensor(audit_model(model, audit)['loss'], dtype=torch.float64),
                                   empirical_loss(model(audit[1]).double(), model(audit[2]).double()).detach())
        source_protocol = dict(split=split, lr=args.lr, batch_size=args.batch_size)
        jobs = make_jobs(args)
        prepare_protocol(args, jobs, sources, split, audit_metadata, source_protocol)
        changed = copy.copy(args); changed.future_seed += 1
        rejects(lambda: prepare_protocol(changed, jobs, sources, split, audit_metadata, source_protocol), 'Changed RNG accepted')
        for job in jobs:
            train_job(args, job, sources[str(job['seed'])], data, train, audit)
        # Baseline is exactly ordinary source-mobility SGD for the same first batch/views.
        baseline = next(j for j in jobs if j['seed'] == 0 and j['name'] == 'baseline')
        manual = load_source(args, 0, 'cpu')
        generator = torch.Generator().manual_seed(args.future_seed)
        index = train[torch.randint(len(train), (args.batch_size,), generator=generator)]
        first, second = augment(data[index], generator), augment(data[index], generator)
        grads = torch.autograd.grad(empirical_loss(manual(first), manual(second)), (manual.kernel, manual.weight))
        rates = learning_rates(args.lr, sources['0']['common_mobility'], baseline)
        with torch.no_grad():
            for parameter, grad, lr in zip((manual.kernel, manual.weight), grads, rates):
                parameter.add_(grad, alpha=-lr)
        actual = torch.load(args.output / baseline['id'] / 'step_1.pt', weights_only=True)
        for key, value in manual.state_dict().items():
            torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0)
        assert not actual['optimizer']['state']  # Plain SGD has no hidden momentum state.
        for job in jobs:
            evaluate_job(args, job, data, labels, train, validation)
        summary = summarize(args.output)
        assert summary['n_seeds'] == 2 and summary['paired_streams_and_initial_states_valid']
        assert all(row['accuracy_minus_fork']['mean'] == 0 for row in summary['fixed_steps'] if row['step'] == 0)
        assert all(row['available_seeds'] == [] for row in summary['loss_matched'] if row['target_loss'] in [1e6, -1e6])
        assert any(run['loss_crossings'] for run in [read_json(args.output / j['id'] / 'result.json') for j in jobs])
        # Restarting an unfinished branch from source is deterministic and independent of branch order.
        replay_args = copy.copy(args); replay_args.output = folder / 'replay'
        train_job(replay_args, baseline, sources['0'], data, train, audit)
        assert read_json(replay_args.output / baseline['id'] / 'result.json') == read_json(args.output / baseline['id'] / 'result.json')
        # Tampering with actual pairing and missing evaluations must not produce a summary.
        run_path = args.output / baseline['id'] / 'result.json'
        original = read_json(run_path)
        corrupt = copy.deepcopy(original); corrupt['sampled_views_sha256'] = 'bad'
        write_json(run_path, corrupt)
        evaluation_path = args.output / baseline['id'] / 'evaluation.json'
        evaluation = read_json(evaluation_path)
        rejects(lambda: summarize(args.output), 'Unpaired views accepted')
        write_json(run_path, original); write_json(evaluation_path, evaluation)
        bad_evaluation = copy.deepcopy(evaluation); bad_evaluation['rows'].pop()
        write_json(evaluation_path, bad_evaluation)
        rejects(lambda: summarize(args.output), 'Missing snapshot accepted')
        write_json(evaluation_path, evaluation)
        # Force tiny tolerance: crossing matches must become explicit missing pairs.
        protocol_path = args.output / 'protocol.json'
        protocol = read_json(protocol_path)
        protocol['arguments']['loss_tolerance'] = 1e-20
        write_json(protocol_path, protocol)
        restricted = summarize(args.output)
        assert any(m['reason'] == 'loss_tolerance_exceeded' for r in restricted['loss_matched'] for m in r['missing'])
        # Capacity checkpoint format and metadata are checked separately from raw local state dicts.
        capacity = copy.copy(args); capacity.source = folder / 'capacity'; capacity.source_kind = 'capacity'
        directory = capacity.source / 'seed_0_baseline_unsupervised_r1'
        directory.mkdir(parents=True)
        source = load_source(args, 0, 'cpu')
        job = dict(id='seed_0_baseline_unsupervised_r1', seed=0, architecture='baseline',
                   objective='unsupervised', ratio=1., channels=64, outputs=8)
        torch.save(dict(model=source.state_dict(), job=job, step=1000), directory / 'step_1000.pt')
        write_json(directory / 'result.json', dict(job=job, steps=4000, optimization=dict(optimizer='SGD'),
                   calibration=dict(common_mobility=sources['0']['common_mobility'])))
        source_metadata(capacity, 0)
        loaded = load_source(capacity, 0, 'cpu')
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in source.state_dict().items())
        # Real server unsupervised results may predate optimization logging.
        result_path = directory / 'result.json'
        legacy = read_json(result_path)
        del legacy['optimization']
        write_json(result_path, legacy)
        write_json(capacity.source / 'protocol.json', dict(jobs=[job],
                   arguments=dict(lr=.002, batch_size=4, supervised_momentum=.9)))
        source_metadata(capacity, 0)
        legacy_model = load_source(capacity, 0, 'cpu')
        assert all(torch.equal(v, legacy_model.state_dict()[k]) for k, v in source.state_dict().items())
        write_json(capacity.source / 'protocol.json', dict(jobs=[]))
        rejects(lambda: source_metadata(capacity, 0), 'Unregistered legacy source accepted')
        for optimizer in ['Adam', None]:
            invalid = copy.deepcopy(legacy)
            invalid['optimization'] = dict(optimizer=optimizer)
            write_json(result_path, invalid)
            rejects(lambda: source_metadata(capacity, 0), 'Unsupported explicit optimizer accepted')
    write_json(root / 'checkpoint_branch_checks.json', dict(status='passed', device='cpu',
        checks=['two-seed five-branch synthetic train/evaluate/summarize', 'exact baseline SGD first step',
                'identical starts and actual paired future streams', 'deterministic replay',
                'fixed audit loss equals empirical objective', 'source protocol mismatch rejection',
                'missing and overshot loss matches retained', 'corrupted pairing and incomplete evaluation rejected',
                'local and capacity source checkpoint formats',
                'legacy capacity results without optimization metadata',
                'legacy source protocol membership and explicit optimizer rejection'], real_cifar_training=False))
    print('Checkpoint branch integration checks passed')


if __name__ == '__main__':
    main()
