"""Matched-A population controls, positive variable-step GD in physical parameters.

Linear/direct controls take the SAME standardized power features q as input.
They do not purport to be ordinary linear networks on raw pixels.
"""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from square_images import excess_risks, population_loss, texture_tasks
from train_textures import initialize


def control(base, mode, noise, tasks, names, step):
    network = copy.deepcopy(base)
    with torch.no_grad():
        a0, b0 = network.effective()
        a0, b0 = a0.clone(), b0.clone()
    if mode == "direct":
        a = torch.nn.Parameter(a0)
        parameters = [a]
        effective = lambda: (a, torch.eye(a.shape[1], dtype=a.dtype))
    elif mode == "linear":
        b = torch.nn.Parameter(b0)
        w = torch.nn.Parameter(network.weight.detach().clone())
        parameters = [b, w]
        effective = lambda: (w @ b, b)
    else:
        if mode == "frozen_head":
            network.weight.requires_grad_(False)
        if mode == "frozen_front":
            network.kernel.requires_grad_(False)
        parameters = [p for p in [network.kernel, network.weight] if p.requires_grad]
        effective = network.effective
    with torch.no_grad():
        assert torch.allclose(effective()[0], a0, atol=1e-14, rtol=1e-12)
        v0 = float(a0.square().sum())
        risk0 = excess_risks(a0, tasks)[0]
    history, updates, physical_time = [], 0, 0.
    for target in [0., 1., 2., 3.]:
        while True:
            a, b = effective()
            energy = a.square().sum()
            ell = math.log(float(energy.detach()) / v0)
            loss = population_loss(a, noise)
            if target - ell < 1e-8:
                with torch.no_grad():
                    risks, values, rank = excess_risks(a, tasks)
                    enc, _, enc_rank = excess_risks(b, tasks)
                row = {"mode": mode, "ell": ell, "target": target, "steps": updates,
                       "physical_time": physical_time, "loss": float(loss.detach()),
                       "rank": rank, "encoder_rank": enc_rank,
                       "view_energy": (1 + noise)**2 * float(energy.detach())}
                row.update({"risk_" + name: float(r) for name, r in zip(names, risks)})
                row.update({"encoder_risk_" + name: float(r) for name, r in zip(names, enc)})
                history.append(row)
                if mode == "direct":
                    assert (risks - risk0).abs().max() < 1e-11
                break
            grads = torch.autograd.grad(loss, parameters, retain_graph=True)
            dv = torch.autograd.grad(energy, parameters)
            rate = float((sum(-(x * y).sum() for x, y in zip(grads, dv)) / energy).detach())
            if rate <= 0:
                raise ArithmeticError("Cannot parameterize this control by increasing energy")
            dt = min(step, .5 * (target - ell)) / rate
            with torch.no_grad():
                for parameter, grad in zip(parameters, grads):
                    parameter.add_(grad, alpha=-dt)
            physical_time += dt
            updates += 1
            if updates > 30000:
                raise RuntimeError("Control exceeded update budget")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--step", type=float, default=.004)
    parser.add_argument("--balance-init", action="store_true")
    parser.add_argument("--output", type=Path, default=result_path("texture_controls"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(dtype="float64", device="cpu", channels=8, outputs=1,
                             kernel_size=32, backend="fft", initial_energy=.001,
                             balance_init=args.balance_init)
    names, tasks = texture_tasks(torch.float64, "cpu")
    for seed in args.seeds:
        base = initialize(config, seed)
        for mode in ["square_joint", "direct", "linear", "frozen_head", "frozen_front"]:
            history = control(base, mode, .2, tasks, names, args.step)
            for row in history:
                row["seed"] = seed
            path = args.output / f"seed_{seed}_{mode}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f"seed={seed} mode={mode} updates={history[-1]['steps']}", flush=True)
    (args.output / "config.json").write_text(json.dumps({
        "seeds": args.seeds, "log_energy_step": args.step,
        "balance_init": args.balance_init,
        "objective": "population GUPA, noise=.2", "initial_mapping": "matched exactly in each seed",
        "algorithm": "positive variable physical-step GD; step chosen from log-energy growth, not risk",
        "linear_input": "standardized frequency powers q, not raw pixels",
        "scope": "two-seed mechanism controls, not robustness evidence"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
