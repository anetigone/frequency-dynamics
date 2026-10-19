"""Conditional mean projector step vs mean squared step; no zero-drift assumption."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import json
from pathlib import Path

import torch

from full_empirical import ROOT, learning_factors, update
from sgd_state_tools import draw_batch, restore, training_data
from square_images import excess_risks, population_loss, texture_tasks
from subspace_mechanism import projector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=result_path("full_sgd_development_v2"))
    parser.add_argument("--output", type=Path, default=result_path("batch_drift_diagnostic"))
    parser.add_argument("--repeats", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    names, tasks = texture_tasks(torch.float64, "cpu")
    rows = []
    for seed in [0, 1]:
        data = training_data(seed)
        for mode in ["square", "linear", "frozen_front"]:
            path = args.source / f"seed_{seed}_{mode}_r1_stream0" / "checkpoint_2000.pt"
            model, saved = restore(path)
            original = [p.detach().clone() for p in (model.front, model.weight)]
            with torch.no_grad():
                a, b = model.effective()
                p0 = projector(b)[0]
                risk0 = excess_risks(b, tasks)[0]
            # Finite population GD step is a reference, not presumed zero.
            loss = population_loss(model.effective()[0], .2)
            gradients = torch.autograd.grad(loss, model.parameters())
            with torch.no_grad():
                for parameter, gradient, factor in zip(model.parameters(), gradients, learning_factors(model, 1.)):
                    parameter.add_(gradient, alpha=-.002 * factor)
                population_delta = projector(model.effective()[1])[0] - p0
            for batch in [256, 1024, 16384]:
                generator = torch.Generator().manual_seed(930000 + seed * 100 + batch)
                deltas, changes = [], []
                for _ in range(args.repeats):
                    with torch.no_grad():
                        model.front.copy_(original[0])
                        model.weight.copy_(original[1])
                    e1, e2 = draw_batch(data, batch, generator)
                    update(model, e1, e2, .002, 1.)
                    with torch.no_grad():
                        b = model.effective()[1]
                        deltas.append(projector(b)[0] - p0)
                        changes.append(excess_risks(b, tasks)[0] - risk0)
                stack = torch.stack(deltas)
                mean = stack.mean(0)
                second = float(stack.square().sum((1, 2)).mean())
                variance = max(0., second - float(mean.square().sum()))
                row = {"seed": seed, "mode": mode, "checkpoint_step": saved["step"],
                       "batch_size": batch, "repeats": args.repeats,
                       "mean_delta_projector": mean.tolist(), "norm_mean_delta_projector": float(mean.norm()),
                       "mean_squared_norm_delta_projector": second, "projector_step_variance_trace": variance,
                       "rms_sampling_error_of_mean": (variance / args.repeats)**.5,
                       "population_finite_step_norm": float(population_delta.norm()),
                       "mean_minus_population_norm": float((mean - population_delta).norm()),
                       "cosine_mean_vs_population": float((mean * population_delta).sum() / (mean.norm() * population_delta.norm()))
                           if float(population_delta.norm()) > 1e-12 and float(mean.norm()) > 0 else None,
                       "mean_encoder_task_risk_change": dict(zip(names, torch.stack(changes).mean(0).tolist()))}
                rows.append(row)
                print(f"seed={seed} {mode} batch={batch} mean_norm={row['norm_mean_delta_projector']:.3g} "
                      f"variance={variance:.3g}", flush=True)
    result = {"scope": "conditional independent batches at two development states, not independent-seed evidence",
              "dataset": "same fixed finite training set as complete training; fresh batch/augmentation draws",
              "finite_step_size": .002, "rows": rows}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
