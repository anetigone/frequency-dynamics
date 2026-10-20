"""Synthetic checks of calibration, pairing, supervised paths, OLS and summaries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS

import argparse
import json
import subprocess
import tempfile
from types import SimpleNamespace

import torch

from cifar_capacity import (ARCHITECTURES, evaluate_checkpoint, fit_probe, initialize,
                            make_jobs, protocol_for, train_job, write_json)
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
                               energy_every=1, batch_size=4, lr=.002, supervised_lr=.01,
                               classifier_lr=.05, supervised_momentum=.9, trace_draws=2,
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
    write_json(RESULTS / 'checks' / 'cifar_capacity_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
