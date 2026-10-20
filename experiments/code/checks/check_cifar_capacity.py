"""Synthetic checks of calibration, pairing, supervised paths, OLS and summaries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

import argparse
import json
import subprocess
import tempfile
import copy
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cifar_capacity import (ARCHITECTURES, evaluate_checkpoint, fit_probe, initialize,
                            make_jobs, protocol_for, train_job, write_json,
                            prepare_protocol, make_optimizer, update)
from cifar_local import initialize as original_initialize


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(20261019)
    data = torch.randn(60, 1, 32, 32, generator=generator)
    labels = torch.arange(60) % 10
    train, validation = torch.arange(40), torch.arange(40, 60)
    cal = data[train[:12]]
    old, _ = original_initialize(0, cal, 'cpu')
    new, calibration = initialize(0, 64, 8, cal, 16)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=2e-6, atol=1e-7)
    assert calibration['matched_function_max_error'] < 1e-5
    assert abs(calibration['initial_energy'] - .01) < 1e-7
    assert abs(calibration['front_trace_after_estimate'] / calibration['head_trace_after'] - 1) < 1e-10
    # New finite checks/logging must not change stable SSL updates .
    legacy_path = EXPERIMENTS / 'archive' / 'cifar_capacity_before_supervised_fix_20261020.py'
    spec = importlib.util.spec_from_file_location('legacy_capacity', legacy_path)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    for ratio in [.1, 1., 10.]:
        old_model, old_calibration = legacy.initialize(0, 64, 8, cal, 2)
        new_model, new_calibration = initialize(0, 64, 8, cal, 2)
        job = dict(objective='unsupervised', ratio=ratio)
        settings = SimpleNamespace(lr=.002)
        old_optimizer = legacy.make_optimizer(old_model, None, old_calibration, job, settings)
        new_optimizer = make_optimizer(new_model, None, new_calibration, job, settings)
        for offset in [0, 4]:
            x1, x2 = data[offset:offset+4], data[offset+1:offset+5]
            a = legacy.update(old_model, None, None, old_optimizer, x1, x2, None, 'unsupervised')
            b = update(new_model, None, None, new_optimizer, x1, x2, None, 'unsupervised')
            assert a == b
            assert all(torch.equal(value, new_model.state_dict()[name]) for name, value in old_model.state_dict().items())
    # Validation labels cannot affect OLS training fit or training-only spectrum.
    features = torch.randn(60, 8, generator=generator)
    first = fit_probe(features, labels, train, validation)
    changed = labels.clone(); changed[validation] = (changed[validation] + 3) % 10
    second = fit_probe(features, changed, train, validation)
    assert first['train'] == second['train']
    assert first['training_feature_spectrum'] == second['training_feature_spectrum']
    parent = (RESULTS / 'checks').resolve()
    assert parent.is_relative_to(EXPERIMENTS.resolve())
    with tempfile.TemporaryDirectory(prefix='capacity_check_', dir=parent) as temporary:
        directory = Path(temporary).resolve()
        assert directory.is_relative_to(parent)
        args = SimpleNamespace(phase='train', output=directory, suite='all', architectures=list(ARCHITECTURES),
                               seeds=[0], ratios=[.1, 1., 10.], steps=2, checkpoints=[1], energy_levels=[1e-8, 100.],
                               energy_every=1, batch_size=4, lr=.002, supervised_lr=.001,
                               classifier_lr=.005, supervised_momentum=.9, trace_draws=2,
                               feature_batch=16, device='cpu')
        jobs = make_jobs(args.seeds, args.suite, args.architectures, args.ratios)
        assert len(jobs) == 11 and len({j['id'] for j in jobs}) == 11
        split = dict(synthetic=True, train_size=40, validation_size=20, test_used=False)
        write_json(directory / 'protocol.json', protocol_for(args, jobs, split))
        for job in jobs:
            train_job(job, args, data, labels, train, cal)
            run_dir = directory / job['id']
            run = json.loads((run_dir / 'result.json').read_text())
            fixed = [dict(file=f'step_{step}.pt', **evaluate_checkpoint(run_dir / f'step_{step}.pt', data, labels, train, validation, 16)) for step in run['fixed_steps']]
            energy = [dict(**row, metrics=evaluate_checkpoint(run_dir / row['file'], data, labels, train, validation, 16)) for row in run['energy_checkpoints']]
            write_json(run_dir / 'evaluation.json', dict(job=job, fixed_steps=fixed, energy_checkpoints=energy, test_used=False))
            initial = torch.load(run_dir / 'step_0.pt', weights_only=True)
            final = torch.load(run_dir / 'step_2.pt', weights_only=True)
            assert not torch.equal(initial['model']['kernel'], final['model']['kernel'])
            if job['objective'] == 'supervised_encoder':
                assert torch.equal(initial['model']['weight'], final['model']['weight'])
            else:
                assert not torch.equal(initial['model']['weight'], final['model']['weight'])
            if job['objective'] != 'unsupervised':
                assert 'trained_classifier' in fixed[-1]
        command = [sys.executable, '-B', str(EXPERIMENTS / 'run.py'), 'summarize_cifar_capacity', str(directory)]
        process = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace')
        assert process.returncode == 0, process.stderr
        report = json.loads((directory / 'summary.json').read_text())
        assert report['run_count'] == 11 and report['paired_streams_and_initializations_valid']
        assert (directory / 'capacity_validation.png').exists()
        absent = [r for r in report['ratio_effects'] if r['target_ell'] == 100.]
        assert absent and all(r['missing_seeds'] == [0] for r in absent)
        # Old-directory resume ignores source hashes and keeps all completed SSL bytes.
        previous = json.loads((directory / 'protocol.json').read_text())
        previous['source_sha256'] = {'cifar_capacity.py': 'old-source-hash-is-not-validated'}
        previous['arguments']['supervised_lr'] = .01
        previous['arguments']['classifier_lr'] = .05
        write_json(directory / 'protocol.json', previous)
        pending_supervised = []
        saved_ssl = {}
        for job in jobs:
            result_path = directory / job['id'] / 'result.json'
            if job['objective'] == 'unsupervised':
                saved_ssl[job['id']] = result_path.read_bytes()
            else:
                pending = result_path.with_suffix('.pending')
                result_path.rename(pending)
                pending_supervised.append((result_path, pending))
        updated = prepare_protocol(args, jobs, split)
        assert updated['source_validation'] is False
        assert (directory / 'protocol_before_resume.json').exists()
        events = json.loads((directory / 'resume_events.json').read_text())
        assert len(events[-1]['completed_runs_kept']) == 9
        with patch('cifar_capacity.update', side_effect=AssertionError('Completed SSL unexpectedly retrained')):
            for job in jobs:
                if job['objective'] == 'unsupervised':
                    train_job(job, args, data, labels, train, cal)
                    assert (directory / job['id'] / 'result.json').read_bytes() == saved_ssl[job['id']]
        wrong = copy.copy(args); wrong.lr *= 2
        try:
            prepare_protocol(wrong, jobs, split)
        except ValueError as error:
            assert 'configuration differs: lr' in str(error)
        else:
            raise AssertionError('Changed unsupervised configuration was accepted')
        for result_path, pending in pending_supervised:
            pending.rename(result_path)
        # Failed runs get explicit job/step diagnostics and cannot become completed.
        with tempfile.TemporaryDirectory(prefix='capacity_failure_', dir=parent) as failure_temp:
            failure_args = copy.copy(args); failure_args.output = Path(failure_temp).resolve()
            assert failure_args.output.is_relative_to(parent)
            with patch('cifar_capacity.update', side_effect=FloatingPointError('injected nonfinite loss')):
                try:
                    train_job(jobs[0], failure_args, data, labels, train, cal)
                except FloatingPointError as error:
                    assert jobs[0]['id'] in str(error) and 'step 1/2' in str(error)
                else:
                    raise AssertionError('Numerical failure was swallowed')
            failure_dir = failure_args.output / jobs[0]['id']
            failure = json.loads((failure_dir / 'failure.json').read_text())
            assert failure['step'] == 1 and failure['job'] == jobs[0]
            assert not (failure_dir / 'result.json').exists()
        # Incomplete planned evaluations must fail, rather than silently reducing n.
        evaluation_path = directory / jobs[0]['id'] / 'evaluation.json'
        evaluation_path.rename(evaluation_path.with_suffix('.saved'))
        process = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace')
        assert process.returncode != 0 and 'Incomplete planned run/evaluation' in process.stderr
    result = dict(synthetic_only=True, cifar_training_started=False, test_metrics_used=False,
                  original_baseline_initialization_matches=True, calibration_balanced=True,
                  validation_labels_excluded_from_probe_fit=True, synthetic_training_runs=11,
                  paired_initializations_and_random_streams=True, supervised_paths_checked=True,
                  actual_ols_and_plot_summary=True, missing_energy_seeds_reported=True,
                  incomplete_suite_rejected=True, temporary_files_removed=True)
    result.update(old_directory_resume_without_source_hashes=True, completed_ssl_runs_skipped_unchanged=True, changed_training_config_rejected=True,
                  contextual_failure_report=True, legacy_ssl_updates_bitwise_identical=True)
    write_json(RESULTS / 'checks' / 'cifar_capacity_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
