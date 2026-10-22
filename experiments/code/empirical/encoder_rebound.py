"""Fixed-task, fixed-time encoder rebound on independent texture initializations.

Train actual pixel filters with finite-dataset SGD using the existing exact
periodic-image forward identity. Labels and population risks enter evaluation
only. Run summarize_encoder_rebound.py separately to plot all trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS

import torch

from full_empirical import build, metrics, update
from square_images import texture_tasks
from subspace_mechanism import projector


@torch.no_grad()
def evaluate(model, initial, tasks, names, step):
    a, b = model.effective()
    row = {"step": step, **metrics(a, b, initial, tasks, names)}
    checks = {}
    for name, matrix in (("encoder", b), ("projection", a)):
        _, values, vh = torch.linalg.svd(matrix, full_matrices=False)
        relative = {"default": max(matrix.shape) * torch.finfo(matrix.dtype).eps,
                    "1e-12": 1e-12, "1e-10": 1e-10, "1e-08": 1e-8}
        checks[name] = {}
        for label, cutoff in relative.items():
            rank = int((values > cutoff * values.max()).sum())
            risks = (tasks.square().sum(1)
                     - (tasks @ vh[:rank].T).square().sum(1)).clamp_min(0)
            checks[name][label] = {"relative_threshold": cutoff, "rank": rank,
                                   "risks": dict(zip(names, risks.tolist()))}
    row["rank_threshold_checks"] = checks
    return row


def write_result(entry):
    result = {key: value for key, value in entry.items()
              if key not in ("model", "directory", "initial")}
    (entry["directory"] / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(entry, step):
    model = entry["model"]
    torch.save({"front": model.front.detach().clone(),
                "weight": model.weight.detach().clone(), "step": step,
                "mode": model.mode, "ratio": entry["ratio"], "seed": entry["seed"],
                "mobility": model.mobility, "scale": model.scale},
               entry["directory"] / f"checkpoint_{step}.pt")


def train_seed(args, seed, names, tasks):
    data_draws = torch.Generator().manual_seed(510000 + seed)
    data = torch.randn(32768, 16, 2, generator=data_draws)
    draws = torch.Generator().manual_seed(610000 + seed * 100)
    entries = []
    reference_a = None
    for mode in args.models:
        for ratio in args.ratios:
            directory = args.output / f"seed_{seed}_{mode}_r{ratio:g}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "result.json"
            if path.exists():
                status = json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")
                print(f"seed={seed} mode={mode} ratio={ratio:g} existing status={status}; "
                      "retaining result (use a new output directory for another run)", flush=True)
                continue
            model, calibration = build(seed, mode, ratio, kernel=32)
            with torch.no_grad():
                a, b = model.effective()
                if reference_a is None:
                    reference_a = a.detach().clone()
                if not torch.allclose(a, reference_a, rtol=1e-11, atol=1e-13):
                    raise ValueError("Paired models must have the same initial effective A")
                initial = {"energy": float(a.square().sum()),
                           "pa": projector(a)[0], "pb": projector(b)[0]}
            entry = {"model": model, "directory": directory, "initial": initial,
                     "seed": seed, "mode": mode, "ratio": ratio, "status": "running",
                     "initial_A": a.detach().tolist(), "initial_B": b.detach().tolist(),
                     "initial_front": model.front.detach().tolist(),
                     "initial_weight": model.weight.detach().tolist(),
                     "calibration": calibration,
                     "history": [evaluate(model, initial, tasks, names, 0)]}
            entries.append(entry)
            checkpoint(entry, 0)
            write_result(entry)
    if not entries:
        return
    start = time.perf_counter()
    fixed_steps = {args.early_step, args.late_step, args.steps}
    for step in range(1, args.steps + 1):
        clean = data[torch.randint(len(data), (args.batch_size,), generator=draws)]
        c1 = clean + math.sqrt(.2) * torch.randn(clean.shape, generator=draws)
        c2 = clean + math.sqrt(.2) * torch.randn(clean.shape, generator=draws)
        e1, e2 = c1.square().sum(-1) / 2, c2.square().sum(-1) / 2
        for entry in entries:
            if entry["status"] != "running":
                continue
            model = entry["model"]
            try:
                for _ in range(args.microsteps):
                    batch_loss = update(model, e1, e2, args.lr / args.microsteps, entry["ratio"])
                if (not math.isfinite(batch_loss)
                        or not all(bool(torch.isfinite(p).all()) for p in model.parameters())):
                    raise FloatingPointError("Nonfinite training loss or parameters")
                if step % args.record_every == 0 or step in fixed_steps:
                    row = evaluate(model, entry["initial"], tasks, names, step)
                    row["empirical_batch_loss"] = batch_loss
                    entry["history"].append(row)
                if step in fixed_steps:
                    checkpoint(entry, step)
                    write_result(entry)
            except (FloatingPointError, RuntimeError) as error:
                entry.update(status="numerical_failure", failure_step=step,
                             error=f"{type(error).__name__}: {error}")
                write_result(entry)
                print(f"seed={seed} mode={entry['mode']} failed step={step}: {error}", flush=True)
        if step % 1000 == 0 or step == args.steps:
            print(f"seed={seed} step={step}/{args.steps}", flush=True)
        if not any(entry["status"] == "running" for entry in entries):
            break
    for entry in entries:
        if entry["status"] == "running":
            entry["status"] = "complete"
        entry["seconds_for_seed_group"] = time.perf_counter() - start
        write_result(entry)
    print(f"seed={seed} finished seconds={time.perf_counter()-start:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "empirical/encoder_rebound_validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20, 26)))
    parser.add_argument("--models", nargs="+", choices=("square", "linear", "frozen_front"),
                        default=["square", "linear", "frozen_front"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--early-step", type=int, default=1000)
    parser.add_argument("--late-step", type=int, default=4000)
    parser.add_argument("--amplitude-threshold", type=float, default=.001)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument("--microsteps", type=int, default=1,
                        help="Two substeps on the same batch are a numerical check, not extra independent runs")
    args = parser.parse_args()
    if not 0 < args.early_step < args.late_step <= args.steps:
        parser.error("Require 0 < early-step < late-step <= steps")
    if (args.record_every <= 0 or args.batch_size < 2 or args.microsteps < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.amplitude_threshold) or args.amplitude_threshold < 0
            or any(not math.isfinite(r) or r <= 0 for r in args.ratios)):
        parser.error("Require finite positive learning settings and a nonnegative amplitude threshold")
    if len(args.seeds) != len(set(args.seeds)) or len(args.models) != len(set(args.models)) or len(args.ratios) != len(set(args.ratios)):
        parser.error("Seeds, models and ratios must be unique")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    names, tasks = texture_tasks(torch.float64, "cpu")
    protocol = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    protocol.update(phase="small_validation", tasks=names, task_vectors=tasks.tolist(),
                    dtype="float64", device="cpu", channels=8, outputs=1, kernel_size=32,
                    noise=.2, training_set_size=32768, initial_energy=.001,
                    data_seed_rule="510000 + seed", batch_view_seed_rule="610000 + 100*seed",
                    initialization="Independent N(0,1/16) spectral filter coordinates and N(0,1) heads; "
                    "initial A energy=.001, then label-free equal-trace factor calibration. "
                    "Not iid Gaussian after calibration, not the theorem neighborhood.",
                    training="Finite-dataset ordinary SGD; true pixel filters; exact periodic-image forward; "
                    "paired data, minibatches and views across models; no task labels in updates",
                    evaluation="Exact population unregularized optimal linear risk for the known synthetic distribution",
                    primary="Per-fixed-task frequency of R(0)-R(early)>amplitude_threshold AND "
                    "R(late)-R(0)>amplitude_threshold. No empirical-minimum selection.",
                    limits="Times and amplitude chosen using historical seeds 2..11; fixed before new runs. "
                    "Six-seed validation, not random-initialization typicality or a uniform task-family theorem. "
                    "Finite-dataset linear control is not constrained to preserve encoder row space.",
                    stop_rule="Fixed steps; include failures and missing comparisons; no outcome-driven expansion")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Experiment parameters differ from this directory; choose a new output to preserve its results")
    path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for seed in args.seeds:
        train_seed(args, seed, names, tasks)


if __name__ == "__main__":
    main()
