"""Synthetic checks of dominant-direction attribution; no real data or training."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS
import torch
import diagnose_dominant_direction as diagnostic


def main():
    torch.set_num_threads(1)
    torch.manual_seed(24)
    brightness = torch.tensor([1., -1., -1., 1.], dtype=torch.float64).repeat(40)
    labels = torch.tensor([0, 1, 0, 1]).repeat(40)
    semantic = labels.double()*2-1
    features = torch.stack([100*brightness, semantic], 1)
    train, validation = torch.arange(80), torch.arange(80, 160)
    scores, basis, center, variance = diagnostic.pca_features(features, train)
    assert abs(float(variance[0]/variance.sum())-10000/10001)<1e-12
    markers = brightness[:, None]
    fits = diagnostic.nuisance_fits(markers, ['brightness'], scores[:, 0], train, validation)
    assert fits[0]['validation_prediction_r2_train_mean_reference'] > 1-1e-12
    probes = {r['component']: r for r in diagnostic.probe_components(scores, labels, train, validation, 2)}
    assert probes['pc1']['validation_accuracy']==.5
    assert probes['remainder']['validation_accuracy']==1
    assert probes['all']['validation_onehot_mse']<1e-25
    rank_one = diagnostic.probe_components(scores[:, :1], labels, train, validation, 2)
    assert next(r for r in rank_one if r['component']=='remainder')['dimensions']==0
    # Validation data must not influence the training PCA.
    perturbed = features.clone()
    perturbed[validation] *= torch.tensor([.1, 10000.])
    assert torch.allclose(diagnostic.pca_features(perturbed, train)[1], basis)
    noise = torch.randn_like(scores)*.1
    first, second = scores+noise, scores-noise
    consistency = diagnostic.component_consistency(first, second, variance, train, validation)
    rescaled = diagnostic.component_consistency(first*9, second*9, variance*81, train, validation)
    for original, changed in zip(consistency, rescaled):
        assert abs(original['alignment_over_clean_train_variance']-changed['alignment_over_clean_train_variance'])<1e-12
    for split in ['train', 'validation']:
        rows = {r['component']: r for r in consistency if r['split']==split}
        assert abs(rows['all']['alignment']-rows['pc1']['alignment']-rows['remainder']['alignment'])<1e-12
    # Constant grayscale images have exactly zero within-image contrast and gradient energy.
    pixels = torch.tensor([0., .25, .5, 1.], dtype=torch.float64)
    images = ((pixels[:, None, None, None].expand(-1, 1, 8, 8)-.5)/.25)
    values, names = diagnostic.image_statistics(images, .5, .25)
    assert torch.allclose(values[:, names.index('pixel_mean')], pixels)
    assert values[:, names.index('pixel_std')].abs().max()==0
    assert values[:, names.index('gradient_energy')].abs().max()==0
    # Full synthetic saved-checkpoint flow, including original DTD augmentation and seed reset.
    parent = RESULTS/'checks'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dominant_direction_', dir=parent) as temporary:
        suite = Path(temporary)
        job = dict(id='seed_0_baseline', seed=0, name='baseline')
        directory = suite/job['id']
        directory.mkdir()
        protocol = dict(dataset_name='DTD', jobs=[job], split=dict(mean=.5, std=.25))
        (suite/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
        snapshots = []
        for step in [0, 2]:
            state = dict(kernel=torch.randn(4, 1, 3, 3)/3, weight=torch.randn(2, 4))
            torch.save(dict(model=state, job=job, branch_step=step), directory/f'step_{step}.pt')
            snapshots.append(dict(step=step, fixed=True, file=f'step_{step}.pt'))
        (directory/'result.json').write_text(json.dumps(dict(job=job, snapshots=snapshots)), encoding='utf-8')
        data = torch.randn(40, 1, 16, 16)
        original_loader = diagnostic.load_images
        args = argparse.Namespace(suite=suite, data_root=None, threads=1, feature_batch=9,
                                  branches=['baseline'], seeds=[0], steps=[0, 2], device='cpu', view_seed=123)
        try:
            diagnostic.load_images = lambda *_: (data, torch.arange(40)%4, torch.arange(24), torch.arange(24, 40))
            result = diagnostic.run(args)
        finally:
            diagnostic.load_images = original_loader
        assert len(result['pca'])==4 and len(result['component_probes'])==16
        assert len(result['view_consistency'])==24 and len(result['nuisance_correlations'])==40
        json.dumps(result, allow_nan=False)
        # Shared CPU RNG reproduces views across checkpoints and branch models.
        kernel = torch.randn(4, 1, 3, 3)
        from dtd_pilot import augment_dtd
        first_run = diagnostic.infer_features(data, kernel, 'cpu', 9, augment_dtd, 123)
        repeat = diagnostic.infer_features(data, kernel, 'cpu', 9, augment_dtd, 123)
        assert all(torch.equal(a, b) for a, b in zip(first_run, repeat))
    print('Passed: dominant nuisance versus semantic tail, train-only PCA, scale-free consistency, additive alignment, image statistics, synthetic checkpoint flow, paired views.')


if __name__=='__main__':
    main()
