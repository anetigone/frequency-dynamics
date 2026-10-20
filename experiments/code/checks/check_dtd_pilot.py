"""Offline synthetic-JPEG DTD download/prepare/pilot/fork integration checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import copy
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import dtd_hf
from dtd_pilot import augment_dtd, fit_probe_dtd, run
from cifar_checkpoint_branches import write_json, read_json
from summarize_checkpoint_branches import summarize


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parent = RESULTS / 'checks'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dtd_pilot_', dir=parent) as temporary:
        folder = Path(temporary)
        official = dtd_hf.partition()['splits']
        snapshot = folder / 'snapshot'
        samples, selected_paths = [], []
        for split in ['train', 'val', 'test']:
            for index, name in enumerate(official[split]):
                label, filename = name.split('/')
                relative = 'data/data_0/' + filename
                samples.append(dict(filepath=relative, tags=[], key_attributes=dict(label=label)))
                if split == 'test':
                    continue
                pixels = np.random.default_rng((0 if split == 'train' else 1880) + index).integers(0, 256, (8, 8, 3), dtype=np.uint8)
                path = snapshot / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path, format='JPEG')
                selected_paths.append(relative)
        write_json(snapshot / 'samples.json', dict(samples=samples))
        write_json(snapshot / 'metadata.json', dict(default_classes=dtd_hf.CLASSES))
        paths = {name: snapshot / name for name in ['samples.json', 'metadata.json']}
        digests = {name: dtd_hf.sha256(path) for name, path in paths.items()}
        calls, snapshot_calls = [], []
        def fake_hub(**kwargs):
            assert kwargs['repo_id'] == dtd_hf.REPO and kwargs['revision'] == dtd_hf.REVISION
            assert kwargs['repo_type'] == 'dataset'
            assert kwargs['headers']['Accept-Encoding'] == 'identity'
            calls.append(kwargs)
            return paths[kwargs['filename']]
        def fake_snapshot(**kwargs):
            assert kwargs['repo_id'] == dtd_hf.REPO and kwargs['revision'] == dtd_hf.REVISION
            assert kwargs['repo_type'] == 'dataset' and set(kwargs['allow_patterns']) == set(selected_paths)
            assert kwargs['headers']['Accept-Encoding'] == 'identity'
            snapshot_calls.append(kwargs)
            return snapshot
        # All bytes are generated locally; the patched download function never accesses the network.
        with patch.object(dtd_hf, 'FILES', digests), patch('huggingface_hub.hf_hub_download', side_effect=fake_hub), \
             patch('huggingface_hub.snapshot_download', side_effect=fake_snapshot):
            data_root = folder / 'prepared'
            assert dtd_hf.download(data_root)['images'] == 3760
            assert len(calls) == 2
            destination = dtd_hf.prepare(data_root, 16)
            assert len(calls) == 4 and all(c['local_files_only'] for c in calls[2:])
            assert snapshot_calls[-1]['local_files_only']
            dtd_hf.prepare(data_root, 16)
            assert len(calls) == 4  # Existing prepared tensors are loaded entirely offline.
            data, labels, train, validation, metadata = dtd_hf.load_prepared(data_root, 16)
            assert len(train) == len(validation) == 1880 and metadata['test_used'] is False
            assert abs(float(data[train].mean())) < 1e-5 and abs(float(data[train].std()) - 1) < 1e-5
            a = augment_dtd(data[:4], torch.Generator().manual_seed(123))
            b = augment_dtd(data[:4], torch.Generator().manual_seed(123))
            assert a.shape == (4, 1, 16, 16) and torch.equal(a, b)
            features = torch.randn(3760, 8, generator=torch.Generator().manual_seed(12))
            first = fit_probe_dtd(features, labels, train, validation)
            changed = labels.clone(); changed[validation] = (changed[validation] + 17) % 47
            second = fit_probe_dtd(features, changed, train, validation)
            assert first['train'] == second['train']
            assert len(first['validation']['confusion']) == 47
            args = SimpleNamespace(phase='train', data_root=data_root, output=folder / 'pilot',
                source=folder / 'pilot', branching=False, source_step=1, seeds=[0], steps=2,
                eval_every=1, audit_every=1, audit_size=12, audit_seed=1234, future_seed=940000,
                loss_levels=[1e6, -1e6], loss_tolerance=.01, batch_size=4, feature_batch=256,
                image_size=16, trace_draws=2, lr=.002, device='cpu', download_workers=2)
            run(args)
            args.phase = 'evaluate'; run(args)
            pilot = summarize(args.output)
            assert pilot['dataset_name'] == 'DTD' and len(pilot['fixed_steps']) == 18
            assert read_json(args.output / 'protocol.json')['branch_names'] == ['ratio_0.1', 'baseline', 'ratio_10']
            branching = copy.copy(args)
            branching.phase = 'train'; branching.branching = True
            branching.output = folder / 'forks'; branching.future_seed = 950000
            run(branching)
            branching.phase = 'evaluate'; run(branching)
            forks = summarize(branching.output)
            assert forks['paired_streams_and_initial_states_valid'] and len(forks['fixed_steps']) == 30
            for row in forks['fixed_steps']:
                if row['step'] == 0:
                    assert row['accuracy_minus_baseline']['mean'] == row['onehot_mse_minus_baseline']['mean'] == 0
            # Corrupted cache and incompatible label schemas must fail instead of changing the protocol.
            sidecar = destination.with_suffix('.json')
            content = read_json(sidecar)
            content['tensor_sha256'] = 'bad'
            write_json(sidecar, content)
            try:
                dtd_hf.load_prepared(data_root, 16)
            except ValueError:
                pass
            else:
                raise AssertionError('Corrupt tensor cache accepted')
            write_json(paths['metadata.json'], dict(default_classes=list(reversed(dtd_hf.CLASSES))))
            try:
                dtd_hf.select_samples(paths['samples.json'], paths['metadata.json'])
            except ValueError:
                pass
            else:
                raise AssertionError('Incompatible 47-class order accepted')
    write_json(parent / 'dtd_pilot_checks.json', dict(status='passed', device='cpu',
        real_dtd_downloaded=False, real_dtd_training=False,
        checks=['pinned HF metadata/snapshot API mocks and train/validation-only JPEG download',
                'identity content encoding on metadata and snapshot requests', 'offline prepare and cache reuse',
                'official partition 1 image names and 47-class per-split counts', 'training-only image normalization',
                '47-class probe and validation-label isolation', 'deterministic resolution-aware augmentation',
                'three-ratio pilot train/evaluate/summarize', 'five-branch checkpoint fork train/evaluate/summarize',
                'source and future-stream pairing', 'corrupt cache and class-order rejection']))
    print('DTD offline integration checks passed; no real data downloaded or trained')


if __name__ == '__main__':
    main()
