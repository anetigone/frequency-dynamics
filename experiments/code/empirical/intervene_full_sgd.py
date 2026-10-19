"""Repeated empirical SGD forks from complete empirical-training checkpoints."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from empirical_subspace_interventions import coefficients, critical
from full_empirical import ROOT, update
from sgd_state_tools import draw_batch, restore, training_data
from square_images import empirical_loss, excess_risks, texture_tasks
from subspace_mechanism import projector


def run_state(args, seed, mode, step):
    directory = args.output / f"seed_{seed}_{mode}_step{step}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        return
    model, state = restore(args.source / f"seed_{seed}_{mode}_r1_stream0" / f"checkpoint_{step}.pt")
    names, tasks = texture_tasks(torch.float64, "cpu")
    primary = names.index("x_high")
    dh, dw, _ = coefficients(model, tasks)
    threshold = critical(float(dh[primary]), float(dw[primary]))
    ratios = set([.1, 1., 10.])
    if threshold is not None:
        ratios.update([threshold / 2, threshold * 2])
    ratios = sorted(ratios)
    prediction = {"seed": seed, "mode": mode, "training_step": step, "ratios": ratios,
                  "lr": .002, "checkpoint_source": "complete empirical SGD",
                  "coefficients": {name: {"front": float(h), "head": float(w), "critical_ratio": critical(float(h), float(w))}
                                   for name, h, w in zip(names, dh, dw)},
                  "global_mobility": model.mobility, "known_task_beta": True}
    (directory / "predictions.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    with torch.no_grad():
        a0, b0 = model.effective()
        baseline = excess_risks(a0, tasks)[0]
        p0 = projector(b0)[0]
    data = training_data(seed)
    outcomes, thresholds = [], []
    for repeat in range(args.repeats):
        generator = torch.Generator().manual_seed(1100000 + seed * 100000 + step * 100 + repeat)
        branches = {r: copy.deepcopy(model) for r in ratios}
        horizons = [1, 8, 64] if step == 2000 and repeat < args.long_repeats else [1]
        first_changes = {}
        empirical_reference = None
        for horizon in range(1, max(horizons) + 1):
            e1, e2 = draw_batch(data, 256, generator)
            if horizon == 1:
                effective = a0.detach().clone().requires_grad_(True)
                gradient = torch.autograd.grad(empirical_loss(e1 @ effective.T, e2 @ effective.T), effective)[0]
                eh, ew, _ = coefficients(model, tasks, gradient)
                empirical_reference = [critical(float(h), float(w)) for h, w in zip(eh, ew)]
            for ratio, branch in branches.items():
                update(branch, e1, e2, .002, ratio)
                if horizon in horizons:
                    with torch.no_grad():
                        a, b = branch.effective()
                        actual = excess_risks(a, tasks)[0] - baseline
                        predicted = horizon * .002 * 2 * model.mobility / (1 + ratio) * (dh + ratio * dw)
                        drift = float((projector(b)[0] - p0).norm())
                    for j, name in enumerate(names):
                        outcomes.append({"repeat": repeat, "ratio": ratio, "horizon": horizon, "task": name,
                                         "predicted_change": float(predicted[j]), "observed_change": float(actual[j]),
                                         "encoder_displacement": drift})
                    if horizon == 1 and ratio in [.1, 1., 10.]:
                        first_changes[ratio] = actual
        x = torch.tensor([[1., r] for r in [.1, 1., 10.]])
        ys = torch.stack([first_changes[r] * (1 + r) / (.002 * 2 * model.mobility) for r in [.1, 1., 10.]])
        fit = torch.linalg.lstsq(x, ys).solution
        for j, name in enumerate(names):
            thresholds.append({"repeat": repeat, "task": name,
                               "population_prediction": prediction["coefficients"][name]["critical_ratio"],
                               "empirical_derivative_reference": empirical_reference[j],
                               "one_step_fitted_threshold": critical(float(fit[0, j]), float(fit[1, j]))})
    result = {"seed": seed, "mode": mode, "training_step": step, "predictions": prediction,
              "repeats": args.repeats, "long_repeats": args.long_repeats if step == 2000 else 0,
              "outcomes": outcomes, "thresholds": thresholds,
              "scope": "known-task population geometry vs repeated true finite-data SGD at empirically trained states"}
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"empirical state seed={seed} {mode} step={step} repeats={args.repeats} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=result_path("full_sgd_main"))
    parser.add_argument("--output", type=Path, default=result_path("full_sgd_interventions"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2, 12)))
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--long-repeats", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"source": str(args.source.resolve()), "seeds": args.seeds,
                "steps": [500, 2000, 3500], "models": ["square", "linear", "frozen_front"],
                "repeats": args.repeats, "long_repeats_at_midpoint": args.long_repeats,
                "horizons_at_midpoint": [1, 8, 64], "lr": .002, "batch": 256,
                "scope": "checkpoints and future updates both empirical SGD; common batches across ratio branches",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    lock = args.output / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise ValueError("Existing locked intervention protocol differs")
    lock.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for seed in args.seeds:
        for mode in protocol["models"]:
            for step in protocol["steps"]:
                run_state(args, seed, mode, step)


if __name__ == "__main__":
    main()
