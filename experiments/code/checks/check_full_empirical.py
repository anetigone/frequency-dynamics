"""Finite-batch fast pixel-parameter loss/gradient vs actual image network."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import json

import torch

from full_empirical import ROOT, batch_loss, build
from square_images import empirical_loss


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(91827)
    c1, c2 = [torch.randn(128, 16, 2, generator=generator) for _ in range(2)]
    e1, e2 = c1.square().sum(-1) / 2, c2.square().sum(-1) / 2
    rows = []
    for kernel in [32, 11]:
        for mode in ["square", "frozen_front"]:
            model, _ = build(0, mode, 1., kernel)
            fast = batch_loss(model, e1, e2)
            gf = torch.autograd.grad(fast, model.parameters())
            spatial = empirical_loss(model.image(model.image.images(c1)), model.image(model.image.images(c2)))
            gs = torch.autograd.grad(spatial, model.parameters())
            error = max(float((a - b).abs().max()) for a, b in zip(gf, gs))
            assert error < 1e-12 and abs(float(fast.detach() - spatial.detach())) < 1e-12
            rows.append({"kernel": kernel, "mode": mode, "max_parameter_gradient_error": error})
    output = result_path("full_empirical_checks.json")
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
