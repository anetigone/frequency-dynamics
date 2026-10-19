"""Restore actual complete-empirical-training checkpoints; never population states."""
from pathlib import Path

import torch

from full_empirical import build


def restore(path):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    model, _ = build(state["seed"], state["mode"], state["ratio"], state["kernel_size"])
    with torch.no_grad():
        model.front.copy_(state["front"])
        model.weight.copy_(state["weight"])
    model.mobility = state["mobility"]
    return model, state


def training_data(seed):
    generator = torch.Generator().manual_seed(510000 + seed)
    return torch.randn(32768, 16, 2, generator=generator, dtype=torch.float64)


def draw_batch(data, n, generator):
    clean = data[torch.randint(len(data), (n,), generator=generator)]
    c1 = clean + .2**.5 * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    c2 = clean + .2**.5 * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    return c1.square().sum(-1) / 2, c2.square().sum(-1) / 2
