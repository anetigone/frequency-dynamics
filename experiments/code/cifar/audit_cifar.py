"""Verify standard CIFAR-10 Python batches before deserializing them."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import hashlib
import json
from pathlib import Path

import numpy as np
import torchvision
from torchvision.datasets import CIFAR10


def audit(root):
    directory = root / CIFAR10.base_folder
    expected = list(CIFAR10.train_list) + list(CIFAR10.test_list)
    expected.append((CIFAR10.meta["filename"], CIFAR10.meta["md5"]))
    checks = []
    for filename, digest in expected:
        path = directory / filename
        observed = hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None
        checks.append({"file": filename, "expected_md5": digest, "observed_md5": observed,
                       "matches_standard": digest == observed})
    if not all(c["matches_standard"] for c in checks):
        raise ValueError(json.dumps(checks, indent=2))
    # Only load pickle after every file matches torchvision's standard checksums.
    train = CIFAR10(str(root), train=True, download=False)
    test = CIFAR10(str(root), train=False, download=False)
    assert train.data.shape == (50000, 32, 32, 3)
    assert test.data.shape == (10000, 32, 32, 3)
    assert train.data.dtype == test.data.dtype == np.uint8
    assert np.array_equal(np.bincount(train.targets), np.full(10, 5000))
    assert np.array_equal(np.bincount(test.targets), np.full(10, 1000))
    return {"standard_cifar10_python": True, "root": str(root.resolve()),
            "torchvision": torchvision.__version__, "checks": checks,
            "train_shape": list(train.data.shape), "test_shape": list(test.data.shape),
            "dtype": str(train.data.dtype), "classes": train.classes,
            "train_class_counts": np.bincount(train.targets).tolist(),
            "test_class_counts": np.bincount(test.targets).tolist()}


if __name__ == "__main__":
    root = EXPERIMENTS.parent
    result = audit(root)
    output = result_path("cifar_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
