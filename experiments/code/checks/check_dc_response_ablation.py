"""Check DC removal, fixed-head probing and paired loss comparisons without real data/training."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS
import torch
from torch.nn import functional as F
import diagnose_dominant_direction as diagnostic


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    kernel = torch.randn(4, 1, 3, 3, dtype=torch.float64)
    original = kernel.clone()
    ablated = diagnostic.zero_dc_kernel(kernel)
    assert torch.equal(kernel, original)
    assert ablated.sum((-2, -1)).abs().max()<1e-14
    assert torch.allclose(diagnostic.zero_dc_kernel(ablated), ablated, atol=1e-14, rtol=0)
    constant_part = kernel.mean((-2, -1), keepdim=True).expand_as(kernel)
    assert abs(float(kernel.square().sum()-ablated.square().sum()-constant_part.square().sum()))<1e-12
    images = torch.randn(16, 1, 12, 12, dtype=torch.float64)
    shifts = torch.randn(16, 1, 1, 1, dtype=torch.float64)*3
    response = F.conv2d(images, ablated)
    shifted = F.conv2d(images+shifts, ablated)
    assert torch.allclose(response, shifted, atol=1e-13, rtol=1e-12)
    assert F.conv2d(images*0+1, ablated).abs().max()<1e-14
    assert not torch.allclose(F.conv2d(images, kernel), F.conv2d(images+shifts, kernel))
    # GUPA terms must agree with the actual training objective, not a normalized substitute.
    from square_images import empirical_loss
    a, b = torch.randn(25, 8, dtype=torch.float64), torch.randn(25, 8, dtype=torch.float64)
    terms = diagnostic.view_objective(a, b)
    assert abs(terms['loss']-float(empirical_loss(a, b)))<1e-12
    assert abs(terms['loss']-sum(terms[k] for k in ['alignment', 'uniformity_logdet', 'uniformity_trace']))<1e-12
    # Zero-DC kernels can legitimately remove the entire model, yielding constant probes.
    data = torch.randn(40, 1, 16, 16)
    labels = torch.arange(40)%4
    train, validation = torch.arange(24), torch.arange(24, 40)
    parent = RESULTS/'checks'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dc_ablation_', dir=parent) as temporary:
        suite = Path(temporary)
        job = dict(id='seed_0_baseline', seed=0, name='baseline')
        directory = suite/job['id']
        directory.mkdir()
        protocol = dict(dataset_name='DTD', jobs=[job], split=dict(mean=.5, std=.25))
        (suite/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
        snapshots = []
        source_states = {}
        for step in [0, 2]:
            # Initial state is purely DC; final state includes spatial variation.
            state = dict(kernel=torch.full((4, 1, 3, 3), .125) if step==0 else torch.randn(4, 1, 3, 3)/3,
                         weight=torch.randn(2, 4))
            source_states[step] = {k: v.clone() for k, v in state.items()}
            torch.save(dict(model=state, job=job, branch_step=step), directory/f'step_{step}.pt')
            snapshots.append(dict(step=step, fixed=True, file=f'step_{step}.pt'))
        (directory/'result.json').write_text(json.dumps(dict(job=job, snapshots=snapshots)), encoding='utf-8')
        args = argparse.Namespace(suite=suite, data_root=None, threads=1, feature_batch=8,
                                  branches=['baseline'], seeds=[0], steps=[0, 2], device='cpu',
                                  view_seed=123, dc_ablation=True)
        original_loader = diagnostic.load_images
        try:
            diagnostic.load_images = lambda *_: (data, labels, train, validation)
            result = diagnostic.run(args)
            args.dc_ablation = False
            baseline = diagnostic.run(args)
            previous_argv = sys.argv
            try:
                sys.argv = ['diagnose_dominant_direction', str(suite), '--dc-ablation', '--seeds', '0',
                            '--steps', '0', '2', '--feature-batch', '8', '--view-seed', '123']
                diagnostic.main()
            finally:
                sys.argv = previous_argv
            saved = json.loads((suite/'dc_ablation'/'diagnostic.json').read_text(encoding='utf-8'))
            assert saved['dc_ablation'] is True
            assert (suite/'dc_ablation'/'ablation_comparisons.csv').is_file()
            assert 'Validation accuracy change pp' in (suite/'dc_ablation'/'findings.md').read_text(encoding='utf-8')
        finally:
            diagnostic.load_images = original_loader
        assert len(result['pca'])==8 and len(result['ablation_comparisons'])==16
        assert len(result['kernel_statistics'])==4
        zero = [r for r in result['pca'] if r['variant']=='zero_dc' and r['step']==0]
        assert all(r['train_rank']==0 and r['train_pc1_variance_share'] is None for r in zero)
        for row in result['component_probes']:
            if row['variant']=='original':
                previous = next(r for r in baseline['component_probes'] if all(row[k]==r[k] for k in ['step','representation','component']))
                assert abs(row['validation_onehot_mse']-previous['validation_onehot_mse'])<1e-12
            if row['variant']=='zero_dc' and row['step']==0:
                assert row['validation_constant_skill']==0
        objectives = result['view_objective']
        for step in [0, 2]:
            rows = [r for r in objectives if r['step']==step and r['scale_mode']=='original_train_clean_energy']
            energies = [r['clean_train_energy'] for r in rows]
            assert max(energies)-min(energies)<1e-12
        assert not any(r['variant']=='zero_dc' and r['step']==0 and r['scale_mode']=='original_train_clean_energy' for r in objectives)
        for step, expected in source_states.items():
            state = torch.load(directory/f'step_{step}.pt', weights_only=True)['model']
            assert all(torch.equal(state[k], v) for k, v in expected.items())
        json.dumps(result, allow_nan=False)
    print('Passed: shift invariance, DC projection/norm identity, unchanged source, exact GUPA, paired baseline, energy matching and zero-rank ablation.')


if __name__=='__main__':
    main()
