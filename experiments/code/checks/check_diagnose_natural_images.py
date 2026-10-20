"""Synthetic mathematical and saved-checkpoint checks; no dataset download or training."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS
import torch
import diagnose_natural_images as diagnostic


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    train, validation = torch.arange(70), torch.arange(70, 120)
    x = torch.randn(120, 5, dtype=torch.float64)
    scaled = diagnostic.functional_change(x, x*3+2, train, validation)
    assert abs(scaled['shared_scale']-3) < 1e-12
    assert scaled['validation_scalar_relative_squared_residual'] < 1e-24
    assert scaled['validation_sample_projector_distance'] < 1e-6
    remix = x@torch.diag(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64))+4
    changed = diagnostic.functional_change(x, remix, train, validation)
    assert changed['validation_scalar_relative_squared_residual'] > .01
    assert changed['validation_affine_relative_squared_residual'] < 1e-24
    assert changed['validation_sample_projector_distance'] < 1e-6
    independent = diagnostic.functional_change(x, torch.randn_like(x), train, validation)
    assert independent['validation_affine_relative_squared_residual'] > .5
    assert independent['validation_sample_projector_distance'] > 2
    # An exactly realizable balanced 47-class target gives a known global rank-8 optimum.
    labels = torch.arange(47).repeat(2)
    features = torch.nn.functional.one_hot(labels, 47).double()
    t, v = torch.arange(47), torch.arange(47, 94)
    projection = features@torch.randn(47, 8, dtype=torch.float64)
    heads = {r['method']: r for r in diagnostic.head_metrics(features, projection, labels, t, v, 47)}
    assert heads['encoder_full_ols']['train_onehot_mse'] < 1e-25
    assert abs(heads['encoder_rank_k_ls']['train_onehot_mse']-38/47**2) < 1e-12
    assert heads['encoder_rank_k_ls']['train_onehot_mse'] <= heads['original_projection_ols']['train_onehot_mse']+1e-12
    assert abs(heads['encoder_rank_k_ls']['validation_onehot_mse']-38/47**2) < 1e-12
    assert diagnostic.spectrum_values([4., 0., 0.])['participation_ratio'] == 1
    # Exercise both saved evaluation schemas and complete checkpoint inference on synthetic images.
    parent = RESULTS/'checks'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='natural_diagnosis_', dir=parent) as temporary:
        suite = Path(temporary)
        job = dict(id='seed_0_baseline', seed=0, name='baseline')
        folder = suite/job['id']
        folder.mkdir()
        images = torch.randn(30, 1, 12, 12)
        targets = torch.arange(30)%10
        a, b = torch.arange(20), torch.arange(20, 30)
        state = dict(kernel=torch.randn(4, 1, 3, 3)/3, weight=torch.randn(2, 4))
        snapshots = []
        for step in [0, 2]:
            model = {k: value.clone() for k, value in state.items()}
            if step:
                model['weight'] *= 2
            filename = f'step_{step}.pt'
            torch.save(dict(model=model, job=job, branch_step=step), folder/filename)
            snapshots.append(dict(step=step, file=filename, fixed=True))
        run = dict(job=job, fixed_steps=[0, 2], snapshots=snapshots,
                   history=[dict(step=0, loss=-.3, alignment=.2, uniformity_logdet=-.3,
                                 uniformity_trace=-.2, clean_energy=1., view_covariance_eigenvalues=[1., 0.])])
        (folder/'result.json').write_text(json.dumps(run), encoding='utf-8')
        fit = dict(train=dict(accuracy=.3, onehot_mse=.07), validation=dict(accuracy=.2, onehot_mse=.08),
                   design_rank_with_intercept=3, singular_values=[5., 2., 1.],
                   training_feature_spectrum=dict(covariance_eigenvalues=[2., 1.]))
        row = dict(step=0, encoder=fit, projection=fit)
        for schema in ['rows', 'fixed_steps']:
            (folder/'evaluation.json').write_text(json.dumps(dict(job=job, **{schema: [row]})), encoding='utf-8')
            probes, history, issues = diagnostic.saved_records(suite, 10, None, None)
            assert len(probes) == 2 and len(history) == 1 and not issues
            assert abs(history[0]['loss_decomposition_residual']) < 1e-12
        args = argparse.Namespace(suite=suite, seeds=None, branches=None, steps=[0, 2],
                                  threads=1, device='cpu', feature_batch=7)
        original_loader = diagnostic.load_images
        try:
            diagnostic.load_images = lambda *_: (images, targets, a, b)
            changes, checkpoint_heads = diagnostic.checkpoint_analysis(args, 'cifar', dict(jobs=[job]))
        finally:
            diagnostic.load_images = original_loader
        assert len(changes) == 4 and len(checkpoint_heads) == 6
        projected = next(r for r in changes if r['step']==2 and r['representation']=='projection')
        assert abs(projected['shared_scale']-2) < 1e-10
        assert projected['validation_scalar_relative_squared_residual'] < 1e-20
        # Serialization must not silently emit NaN/Infinity.
        json.dumps(dict(changes=changes, heads=checkpoint_heads), allow_nan=False)
    print('Passed: scale/remix/subspace distinctions, exact 47-class rank-8 optimum, both saved schemas, synthetic checkpoint inference.')


if __name__ == '__main__':
    main()
