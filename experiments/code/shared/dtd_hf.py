"""Pinned Voxel51 DTD JPEG mirror with bundled official partition-1 metadata."""
import io
import json
import time
from pathlib import Path, PurePosixPath

REPO = 'Voxel51/Describable-Textures-Dataset'
REVISION = 'e850d7c773fa71940567cb53da4270b89fb3737a'
# Some server environments pair a newer httpx2 decoder with an older Brotli
# backend whose process() cannot accept output_buffer_limit. Request raw HTTP
# bodies for metadata, API listings and JPEGs; preserve native HF cache/resume.
DOWNLOAD_HEADERS = {'Accept-Encoding': 'identity'}
FILES = ('samples.json', 'metadata.json')
PARTITION = Path(__file__).resolve().parents[2] / 'resources' / 'dtd_partition1.json'
CLASSES = ('banded blotchy braided bubbly bumpy chequered cobwebbed cracked crosshatched crystalline '
           'dotted fibrous flecked freckled frilly gauzy grid grooved honeycombed interlaced knitted '
           'lacelike lined marbled matted meshed paisley perforated pitted pleated polka-dotted porous '
           'potholed scaly smeared spiralled sprinkled stained stratified striped studded swirly veined '
           'waffled woven wrinkled zigzagged').split()


def partition():
    from collections import Counter
    info = json.loads(PARTITION.read_text(encoding='utf-8'))
    if info['partition'] != 1:
        raise ValueError('Only the bundled official DTD partition 1 is supported')
    all_names = []
    for name in ['train', 'val', 'test']:
        entries = info['splits'][name]
        if len(entries) != 1880 or Counter(x.split('/')[0] for x in entries) != Counter({c: 40 for c in CLASSES}):
            raise ValueError('Official DTD split size/class counts mismatch')
        for entry in entries:
            p = PurePosixPath(entry)
            if len(p.parts) != 2 or p.parts[0] not in CLASSES or p.suffix != '.jpg' or '..' in p.parts:
                raise ValueError('Invalid official DTD image name')
        all_names.extend(entries)
    if len(set(all_names)) != 5640:
        raise ValueError('Official DTD splits overlap')
    return info


def select_samples(samples_path, metadata_path):
    from collections import Counter
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if metadata['default_classes'] != CLASSES:
        raise ValueError('Voxel51 DTD class order mismatch')
    records = json.loads(Path(samples_path).read_text(encoding='utf-8'))['samples']
    by_name = {}
    for record in records:
        label = record['key_attributes']['label']
        path = PurePosixPath(record['filepath'])
        if label not in CLASSES or path.is_absolute() or '..' in path.parts or not str(path).startswith('data/'):
            raise ValueError('Invalid Voxel51 DTD image metadata')
        name = label + '/' + path.name
        if not path.name.startswith(label + '_') or path.suffix != '.jpg' or name in by_name:
            raise ValueError('Duplicate or inconsistent Voxel51 DTD image name')
        by_name[name] = dict(name=name, path=str(path), label=CLASSES.index(label))
    if len(by_name) != 5640 or Counter(r['label'] for r in by_name.values()) != Counter({i: 120 for i in range(47)}):
        raise ValueError('Voxel51 DTD must contain 120 images for each of 47 classes')
    info = partition()
    if set(by_name) != {x for rows in info['splits'].values() for x in rows}:
        raise ValueError('Voxel51 image names do not match the official DTD partition')
    # The dataset viewer calls all 5640 samples "train"; use the official split instead.
    return [dict(**by_name[name], split=split) for split in ['train', 'val'] for name in info['splits'][split]]


def download(root, workers=8, offline=False):
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(root)
    cache = str(root / 'hub')
    metadata_paths = {}
    for filename in FILES:
        path = Path(hf_hub_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                    filename=filename, cache_dir=cache, local_files_only=offline,
                    headers=DOWNLOAD_HEADERS))
        metadata_paths[filename] = path
    rows = select_samples(metadata_paths['samples.json'], metadata_paths['metadata.json'])
    for attempt in range(3):
        try:
            snapshot = Path(snapshot_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                cache_dir=cache, allow_patterns=[r['path'] for r in rows], max_workers=workers,
                local_files_only=offline, headers=DOWNLOAD_HEADERS))
            break
        except Exception:
            if offline or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    missing = [r['path'] for r in rows if not (snapshot / r['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'DTD image missing: {missing[0]}; rerun download')
    return dict(snapshot=str(snapshot), samples_json=str(metadata_paths['samples.json']),
                metadata_json=str(metadata_paths['metadata.json']), images=len(rows),
                repo_id=REPO, revision=REVISION, partition=1, test_downloaded=False)


def preprocess_image(raw, image_size):
    from PIL import Image, ImageOps
    import numpy as np
    import torch
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert('L')
        short = image_size + image_size // 8
        scale = short / min(image.size)
        dimensions = (max(image_size, round(image.width * scale)), max(image_size, round(image.height * scale)))
        image = image.resize(dimensions, Image.Resampling.BICUBIC)
        left, top = (image.width - image_size) // 2, (image.height - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))
        return torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))[None]


def decode_images(snapshot, rows, image_size):
    import torch
    expected = partition()['splits']
    if [r['name'] for r in rows] != expected['train'] + expected['val']:
        raise ValueError('Downloaded DTD images differ from official train/validation')
    images, targets = [], []
    for row in rows:
        path = PurePosixPath(row['path'])
        if path.is_absolute() or '..' in path.parts or not str(path).startswith('data/'):
            raise ValueError('Invalid cached DTD image path')
        if row['label'] != CLASSES.index(row['name'].split('/')[0]):
            raise ValueError('DTD label differs from official image category')
        raw = (Path(snapshot) / path).read_bytes()
        images.append(preprocess_image(raw, image_size))
        targets.append(row['label'])
    return torch.stack(images), torch.tensor(targets), 1880


def prepare(root, image_size=64):
    import torch
    root = Path(root)
    destination = root / f'prepared_gray_{image_size}.pt'
    if destination.exists():
        load_prepared(root, image_size)
        return destination
    cached = download(root, offline=True)
    rows = select_samples(cached['samples_json'], cached['metadata_json'])
    images, labels, train_stop = decode_images(cached['snapshot'], rows, image_size)
    training = images[:train_stop].float() / 255
    mean, std = float(training.mean()), float(training.std())
    if not std > 0:
        raise ValueError('Degenerate training normalization')
    metadata = dict(repo_id=REPO, revision=REVISION, classes=CLASSES,
                    partition=1,
                    image_size=image_size, train_size=train_stop, validation_size=len(labels) - train_stop,
                    mean=mean, std=std, preprocessing='EXIF transpose, grayscale, shorter-side 9/8 size bicubic, center crop',
                    test_downloaded=False, test_used=False,
                    split='Official DTD partition 1, using bundled train1.txt/val1.txt file lists')
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pt.tmp')
    torch.save(dict(images=images, labels=labels, metadata=metadata), temporary)
    temporary.replace(destination)
    sidecar = destination.with_suffix('.json')
    sidecar.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return destination


def load_prepared(root, image_size=64):
    import torch
    path = Path(root) / f'prepared_gray_{image_size}.pt'
    if not path.exists():
        raise FileNotFoundError('DTD is not prepared; run dtd_pilot download, then dtd_pilot prepare')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    if metadata['repo_id'] != REPO or metadata['revision'] != REVISION or metadata['classes'] != CLASSES:
        raise ValueError('Prepared DTD provenance mismatch')
    if metadata['image_size'] != image_size or metadata['test_used']:
        raise ValueError('Prepared DTD configuration mismatch')
    if metadata['partition'] != 1:
        raise ValueError('Prepared DTD official partition mismatch')
    n = metadata['train_size']
    if payload['images'].shape != (3760, 1, image_size, image_size) or payload['images'].dtype != torch.uint8:
        raise ValueError('Unexpected prepared DTD image shape/dtype')
    labels = payload['labels']
    if labels.shape != (3760,) or n != 1880 or metadata['validation_size'] != 1880:
        raise ValueError('Unexpected DTD split size')
    for subset in [labels[:n], labels[n:]]:
        if not torch.equal(torch.bincount(subset, minlength=47), torch.full((47,), 40)):
            raise ValueError('Prepared DTD class counts mismatch')
    images = payload['images'].float() / 255
    mean, std = float(images[:n].mean()), float(images[:n].std())
    if abs(mean - metadata['mean']) > 1e-7 or abs(std - metadata['std']) > 1e-7:
        raise ValueError('Prepared DTD training statistics mismatch')
    return (images - mean) / std, labels, torch.arange(n), torch.arange(n, len(labels)), metadata
