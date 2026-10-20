"""Synthetic DC-free calibration/training/evaluation checks. No download or real training."""
import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS
import torch
from cifar_capacity import initialize
from cifar_local import NaturalSquare
from dtd_hf import REPO, REVISION
from dtd_pilot import run, make_jobs
from summarize_checkpoint_branches import summarize
from diagnose_natural_images import saved_records


def no_hash_keys(value):
    if isinstance(value, dict):
        assert not any('sha256' in k or 'md5' in k for k in value)
        for item in value.values():
            no_hash_keys(item)
    elif isinstance(value, list):
        for item in value:
            no_hash_keys(item)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert len(make_jobs([0, 1, 12], dc_free=True))==3
    assert all(j['ratio']==1 and j['name']=='baseline' for j in make_jobs([0, 1, 12], dc_free=True))
    data = torch.randn(94, 1, 16, 16, generator=torch.Generator().manual_seed(77))
    # Per-image offsets deliberately create a substantial unconstrained DC response.
    data += torch.linspace(-2, 2, len(data))[:, None, None, None]
    labels = torch.arange(47).repeat(2)
    train, validation = torch.arange(47), torch.arange(47, 94)
    calibration = data[:12]
    model, info = initialize(0, 64, 8, calibration, draws=3, record_state_hash=False, zero_dc=True)
    assert model.kernel.detach().mean((-2, -1)).abs().max()<1e-8
    assert abs(info['initial_energy']-.01)<1e-7
    # Independently recompute the actual tangent-block mobility AFTER parameter balancing.
    z = model(calibration)
    z = z-z.mean(0)
    generator = torch.Generator().manual_seed(720000)
    projected_traces = []
    for _ in range(3):
        signs = torch.randint(2, z.shape, generator=generator).float()*2-1
        grad = torch.autograd.grad((z*signs).sum()/math.sqrt(len(z)), model.kernel, retain_graph=True)[0]
        projected = grad-grad.mean((-2, -1), keepdim=True)
        projected_traces.append(float(projected.square().sum()))
    actual_front = sum(projected_traces)/len(projected_traces)
    f = model.features(calibration).detach()
    actual_head = 8*float((f-f.mean(0)).square().sum(1).mean())
    assert math.isclose(actual_front, info['front_trace_after_estimate'], rel_tol=2e-5)
    assert math.isclose(actual_front, actual_head, rel_tol=2e-5)
    assert math.isclose(info['common_mobility'], 1/(actual_front+actual_head), rel_tol=2e-5)
    # The constrained model is insensitive to independent constant shifts of each image.
    shift = torch.linspace(-1, 1, 12)[:, None, None, None]
    with torch.no_grad():
        assert torch.allclose(model(calibration), model(calibration+shift), atol=2e-7, rtol=2e-5)
    parent = RESULTS/'checks'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dtd_dc_free_', dir=parent) as temporary:
        folder = Path(temporary)
        args = SimpleNamespace(phase='train', data_root=folder/'unused', output=folder/'dc_free',
            source=folder/'unused_source', branching=False, dc_free=True, source_step=500, seeds=[0],
            steps=6, eval_every=3, audit_every=1, audit_size=12, audit_seed=20261022, future_seed=940000,
            loss_levels=[-.05, -.1], loss_tolerance=.01, batch_size=8, feature_batch=32,
            image_size=16, trace_draws=3, lr=.002, device='cpu', download_workers=2)
        metadata = dict(repo_id=REPO, revision=REVISION, mean=0., std=1.)
        payload = (data, labels, train, validation, metadata)
        with patch('dtd_pilot.load_prepared', return_value=payload), \
             patch('cifar_checkpoint_branches.state_hash', side_effect=AssertionError('Unexpected state hash')), \
             patch('cifar_checkpoint_branches.tensor_hash', side_effect=AssertionError('Unexpected tensor hash')), \
             patch('cifar_capacity.state_hash', side_effect=AssertionError('Unexpected initialization hash')):
            run(args)
            first = torch.load(args.output/'seed_0_baseline'/'step_6.pt', weights_only=True)
            result = json.loads((args.output/'seed_0_baseline'/'result.json').read_text())
            assert result['kernel_constraint']=='spatial_zero_mean'
            assert max(r['kernel_max_abs_spatial_mean'] for r in result['history'])<1e-8
            for snapshot in result['snapshots']:
                state = torch.load(args.output/'seed_0_baseline'/snapshot['file'], weights_only=True)['model']
                assert state['kernel'].mean((-2, -1)).abs().max()<1e-8
            protocol = json.loads((args.output/'protocol.json').read_text())
            assert protocol['branch_names']==['baseline']
            for record in [result, protocol]:
                no_hash_keys(record)
            # Completed runs are reused; changed task labels cannot alter SSL/calibration.
            run(args)
            replay = copy.copy(args)
            replay.output = folder/'labels_changed'
            changed = (labels+9)%47
            with patch('dtd_pilot.load_prepared', return_value=(data, changed, train, validation, metadata)):
                run(replay)
            second = torch.load(replay.output/'seed_0_baseline'/'step_6.pt', weights_only=True)
            assert all(torch.equal(first['model'][k], second['model'][k]) for k in first['model'])
            assert torch.equal(first['generator_state'], second['generator_state'])
            # Evaluate/summarize use existing generic checkpoints and the original 47-class probe.
            args.phase='evaluate'
            run(args)
            summary = summarize(args.output)
            assert summary['kernel_constraint']=='spatial_zero_mean'
            assert summary['cross_branch_pairing_applicable'] is False and summary['loss_matched']==[]
            assert len(summary['fixed_steps'])==6
            assert all('train_onehot_mse_minus_initial' in r and 'onehot_mse_minus_initial' in r for r in summary['fixed_steps'])
            probes, history, issues = saved_records(args.output, 47, None, None)
            assert len(probes)==6 and len(history)==7 and not issues
            # A DC-free invocation cannot silently adopt an existing unconstrained initialization.
            identity_path = args.output/'initializations'/'seed_0.pt'
            initialization = torch.load(identity_path, weights_only=True)
            initialization['identity'].pop('dc_free')
            torch.save(initialization, identity_path)
            args.phase='train'
            try:
                run(args)
            except ValueError as error:
                assert 'constraint differs' in str(error)
            else:
                raise AssertionError('Mismatched initialization was accepted')
    print('Passed: constrained mobility/energy calibration, brightness-shift invariance, zero-mean training, label independence, reuse, no hashes, evaluation and summary compatibility.')


if __name__=='__main__':
    main()
