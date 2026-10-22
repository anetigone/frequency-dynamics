"""Fifty new texture seeds with fixed-time and sampled-trajectory rebound reports.

Defaults to a read-only plan. Use --stage train or all on the training server.
The existing six-seed results are never used as additional repetitions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from summarize_natural_moments import trajectory_event, wilson, write_csv


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results/empirical/texture_rebound_50_seeds"
MODELS = ["square", "linear", "frozen_front"]


def planned_steps(args):
    return sorted({0, args.early_step, args.late_step, args.steps,
                   *range(args.record_every, args.steps + 1, args.record_every)})


def settings(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "stage"}


def train(args):
    # Import Torch only for explicitly requested training, not for --stage plan.
    import torch
    from encoder_rebound import train_seed
    from square_images import texture_tasks

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    names, tasks = texture_tasks(torch.float64, "cpu")
    protocol = dict(settings(args), phase="rebound_confirmation", tasks=names,
                    task_vectors=tasks.tolist(), record_steps=planned_steps(args),
                    dtype="float64", device="cpu", channels=8, outputs=1,
                    kernel_size=32, noise=.2, training_set_size=32768,
                    initial_energy=.001, data_seed_rule="510000 + seed",
                    batch_view_seed_rule="610000 + 100*seed",
                    initialization="Independent N(0,1/16) spectral filter coordinates and N(0,1) heads; "
                    "initial A energy=.001, then label-free Jacobian-trace factor calibration and mobility. "
                    "Calibration reads model parameters and unlabeled moments only, no labels, task vectors or risks.",
                    training="Same original finite-dataset ordinary SGD, true pixel filters, paired data/minibatches/views; "
                    "population risks enter evaluation only; CPU float64, one thread.",
                    evaluation="Exact population optimal unregularized linear risks for the known synthetic distribution.",
                    primary="Per-fixed-task sampled full rebound: max over 0<s<t<=T of "
                    "min(R(0)-R(s),R(t)-R(0)) > amplitude_threshold. Fixed sample grid and horizon.",
                    secondary="Original fixed-time two-gap event and ordinary maximum drawup; all tasks and models retained.",
                    limits="Frequency is conditional on this calibrated initialization distribution, SGD settings, "
                    "fixed task, saved time grid and horizon. Tasks/checkpoints sharing a seed are not independent.",
                    stop_rule="Fixed seed list and step budget; do not expand based on observed outcomes. "
                    "Existing result files are retained, including partial or failed runs; no automatic reruns.")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Parameters differ from existing protocol; choose a new output directory")
    path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    train_args = SimpleNamespace(**vars(args))
    for seed in args.seeds:
        train_seed(train_args, seed, names, tasks)


def summarize(directory):
    from summarize_encoder_rebound import analyze_task, finite, load_runs, summarize_directory

    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    # Preserve the original analysis, figures, numerical-rank checks and initial pairing.
    fixed = summarize_directory(directory)
    runs = load_runs(directory, protocol)
    expected = protocol.get("record_steps")
    if expected is None:
        expected = planned_steps(SimpleNamespace(**protocol))
    threshold = protocol["amplitude_threshold"]
    output = directory / "trajectory_summary"
    output.mkdir(parents=True, exist_ok=True)
    coverage, rows, sensitivity = [], [], []
    for run in runs:
        history = run.get("history", [])
        present = {point["step"] for point in history}
        missing = [step for step in expected if step not in present]
        complete = run["status"] in ("complete", "completed") and not missing
        coverage.append(dict(seed=run["seed"], mode=run["mode"], ratio=run["ratio"],
                             run_status=run["status"], complete_record_grid=complete,
                             observed_states=len(present), planned_states=len(expected),
                             last_step=max(present, default=None),
                             missing_steps=" ".join(map(str, missing)),
                             failure_step=run.get("failure_step"), error=run.get("error")))
        for representation, task in itertools.product(("encoder", "projection"), protocol["tasks"]):
            points = [(point["step"], point.get(representation + "_risks", {}).get(task))
                      for point in history]
            points = [(step, value) for step, value in points if finite(value)]
            event = trajectory_event(points, threshold)
            eligible = complete and len(points) == len(expected) and event["available"]
            old = analyze_task(run, representation, task, protocol["early_step"],
                               protocol["late_step"], threshold)
            row = dict(seed=run["seed"], mode=run["mode"], ratio=run["ratio"],
                       representation=representation, task=task, run_status=run["status"],
                       complete_record_grid=complete, eligible_for_incidence=eligible,
                       amplitude_threshold=threshold, **event,
                       drawup_event=event.get("max_drawup", 0.) > threshold if event["available"] else None,
                       fixed_early_step=protocol["early_step"], fixed_late_step=protocol["late_step"],
                       fixed_early_improvement=old["early_improvement"], fixed_late_excess=old["late_excess"],
                       fixed_category=old["category"], fixed_full_rebound=old["category"] == "complete_rebound",
                       observed_event_in_partial_path=event.get("full_rebound") if not eligible else None)
            rows.append(row)
            cutoffs = sorted({cutoff for point in history for cutoff in
                              point.get("rank_threshold_checks", {}).get(representation, {})})
            for cutoff in cutoffs:
                checked = [(point["step"], point.get("rank_threshold_checks", {}).get(representation, {})
                            .get(cutoff, {}).get("risks", {}).get(task)) for point in history]
                checked = [(step, value) for step, value in checked if finite(value)]
                checked_event = trajectory_event(checked, threshold)
                sensitivity.append(dict(seed=run["seed"], mode=run["mode"], ratio=run["ratio"],
                                        representation=representation, task=task, cutoff=cutoff,
                                        complete_record_grid=complete and len(checked) == len(expected),
                                        **checked_event, differs_from_default=(
                                            checked_event.get("full_rebound") != event.get("full_rebound"))))
    rates = []
    for mode, ratio, representation, task in itertools.product(
            protocol["models"], protocol["ratios"], ("encoder", "projection"), protocol["tasks"]):
        group = [row for row in rows if (row["mode"], row["ratio"], row["representation"], row["task"])
                 == (mode, ratio, representation, task)]
        available = [row for row in group if row["eligible_for_incidence"]]
        excluded = [row for row in group if not row["eligible_for_incidence"]]
        for metric in ("full_rebound", "drawup_event", "fixed_full_rebound"):
            successes = sum(row[metric] for row in available)
            lo, hi = wilson(successes, len(available))
            rates.append(dict(mode=mode, ratio=ratio, representation=representation, task=task,
                              metric=metric, events=successes, completed_seeds=len(available),
                              planned_seeds=len(group), excluded_seeds=len(excluded),
                              excluded_seed_list=" ".join(str(row["seed"]) for row in excluded),
                              completed_seed_frequency=successes / len(available) if available else None,
                              observed_events_over_planned_seeds=successes / len(group),
                              wilson95_lower=lo, wilson95_upper=hi,
                              interval_scope="Complete-seed conditional frequency; missing/failures are not failures of rebound"))
    families = []
    for mode, ratio, representation in itertools.product(protocol["models"], protocol["ratios"],
                                                         ("encoder", "projection")):
        for seed in protocol["seeds"]:
            group = [row for row in rows if (row["mode"], row["ratio"], row["representation"], row["seed"])
                     == (mode, ratio, representation, seed)]
            families.append(dict(seed=seed, mode=mode, ratio=ratio, representation=representation,
                                 complete_record_grid=all(row["eligible_for_incidence"] for row in group),
                                 any_fixed_task_full_rebound=any(row["full_rebound"] is True for row in group),
                                 full_rebound_task_count=sum(row["full_rebound"] is True for row in group),
                                 fixed_task_count=len(group), scope="Descriptive correlated-task union, not per-task incidence"))
    for name, values in (("coverage", coverage), ("trajectory_events", rows),
                         ("task_frequencies", rates), ("rank_sensitivity_events", sensitivity),
                         ("seed_task_union", families)):
        write_csv(output / f"{name}.csv", values)
    summary = dict(protocol=protocol, coverage=coverage, frequencies=rates, events=rows,
                   task_union=families, fixed_summary=str(directory / "encoder_rebound_summary/summary.json"),
                   fixed_status_counts=fixed["run_status_counts"],
                   independent_unit="seed, for each fixed task/model/ratio/representation",
                   full_rebound="max_{0<s<t<=T} min(R0-Rs,Rt-R0)>delta on the saved grid",
                   ordinary_drawup="max_{0<=s<t<=T}(Rt-Rs)>delta; does not require crossing initialization",
                   interval_scope="Wilson 95% pointwise descriptive intervals conditional on complete recorded seeds. "
                   "No pooling of ten tasks or checkpoints; no simultaneous-coverage claim.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Fifty-seed texture rebound supplement", "",
             f"Planned independent seeds: {protocol['seeds']}; models: {protocol['models']}; horizon: {protocol['steps']} updates.",
             "The seed is the independent repetition for each fixed task. All ten fixed tasks and all models are reported.", "",
             "Sampled full rebound is max over 0<s<t<=T of min(R(0)-R(s),R(t)-R(0)) > delta. "
             "This removes the choice of two comparison times, but remains conditional on the prespecified observation grid and horizon. "
             "Ordinary drawup is max over 0<=s<t<=T of R(t)-R(s), and need not end above initialization. "
             "The original fixed-time event remains in encoder_rebound_summary and trajectory_events.csv.", "",
             "Initialization calibration reads no labels, task directions or task risks; it uses the initial effective map energy and Jacobian traces only. "
             "The resulting initialization distribution is calibrated and must not be described as ordinary unmodified iid Gaussian initialization.", "",
             "| Model | Representation | Fixed task | Metric | Events / complete / planned | Pointwise Wilson 95% CI |",
             "| --- | --- | --- | --- | ---: | --- |"]
    for group in rates:
        ci = (f"[{group['wilson95_lower']:.3f}, {group['wilson95_upper']:.3f}]"
              if group["wilson95_lower"] is not None else "unavailable")
        lines.append(f"| {group['mode']} (r={group['ratio']:g}) | {group['representation']} | {group['task']} | "
                     f"{group['metric']} | {group['events']}/{group['completed_seeds']}/{group['planned_seeds']} | {ci} |")
    lines += ["", "Failures, missing results and partial runs remain in coverage.csv and each planned-seed denominator. "
              "The CI and completed-seed frequency use complete recorded paths only; excluded seeds are listed in task_frequencies.csv. "
              "Partial-path events remain descriptive and never become completed trials. These intervals are pointwise and conditional on "
              "completion; they do not correct selection induced by numerical failures or supply simultaneous coverage across tasks.", "",
              f"Trajectory rank-cutoff event disagreements: {sum(row['differs_from_default'] for row in sensitivity)}; "
              "see rank_sensitivity_events.csv. Exact population risks apply only to the known synthetic input distribution. "
              "Finite-data SGD is not covered by the isotropic population linear-conservation statement.", ""]
    (output / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Trajectory summary: {output / 'findings.md'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "train", "summary", "all"), default="plan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 150)))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--early-step", type=int, default=1000)
    parser.add_argument("--late-step", type=int, default=4000)
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument("--amplitude-threshold", type=float, default=.001)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microsteps", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.early_step < args.late_step <= args.steps:
        parser.error("Require 0 < early-step < late-step <= steps")
    if (args.record_every < 1 or args.batch_size < 2 or args.microsteps < 1 or
            not math.isfinite(args.lr) or args.lr <= 0 or
            not math.isfinite(args.amplitude_threshold) or args.amplitude_threshold < 0 or
            any(not math.isfinite(r) or r <= 0 for r in args.ratios)):
        parser.error("Require finite positive settings and a nonnegative threshold")
    if any(len(values) != len(set(values)) for values in (args.seeds, args.models, args.ratios)):
        parser.error("Seeds, models and ratios must be unique")
    if args.stage == "plan":
        print(json.dumps(dict(stage="plan", arguments=settings(args),
                              planned_runs=len(args.seeds) * len(args.models) * len(args.ratios),
                              record_steps=planned_steps(args),
                              action="No training or writes; explicitly select --stage train/all on the server"),
                         indent=2, allow_nan=False))
        return
    if args.stage == "all":
        # Torch training and NumPy/Matplotlib summary use separate runtimes.
        # In particular, this avoids duplicate OpenMP DLLs in the local env.
        forwarded = []
        for key, value in vars(args).items():
            if key != "stage":
                forwarded.append("--" + key.replace("_", "-"))
                forwarded.extend(str(item) for item in (value if isinstance(value, list) else [value]))
        for stage in ("train", "summary"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--stage", stage,
                            *forwarded], check=True)
        return
    if args.stage == "train":
        train(args)
    if args.stage == "summary":
        summarize(args.output)


if __name__ == "__main__":
    main()
