"""Summarize fixed-time encoder rebound without selecting tasks or minima.

Standalone NumPy/Matplotlib process; importing the trainer is unnecessary.
For old full_sgd_main results this is explicitly exploratory reanalysis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ("complete_rebound", "improvement_only", "deterioration_only", "neither",
              "missing_comparison", "numerical_failure")
REPRESENTATIONS = ("encoder", "projection")
COLORS = {"square": "#0072B2", "linear": "#D55E00", "frozen_front": "#009E73"}


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def describe(values):
    values = [float(v) for v in values if finite(v)]
    if not values:
        return None
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": min(values), "max": max(values)}


def classify(early_gap, late_gap, threshold):
    early, late = early_gap > threshold, late_gap > threshold
    if early and late:
        return "complete_rebound"
    if early:
        return "improvement_only"
    if late:
        return "deterioration_only"
    return "neither"


def fixed_row(history, step):
    return next((row for row in history if row.get("step") == step), None)


def risk(row, representation, task):
    return None if row is None else row.get(representation + "_risks", {}).get(task)


def threshold_result(row, representation, threshold):
    if row is None:
        return None
    return row.get("rank_threshold_checks", {}).get(representation, {}).get(threshold)


def sensitivity(history, representation, task, early_step, late_step, amplitude):
    selected = [fixed_row(history, step) for step in (0, early_step, late_step)]
    if any(row is None for row in selected):
        return {}
    thresholds = set.intersection(*[
        set(row.get("rank_threshold_checks", {}).get(representation, {})) for row in selected])
    out = {}
    for threshold in sorted(thresholds, key=lambda label: -math.inf if label == "default" else float(label)):
        entries = [threshold_result(row, representation, threshold) for row in selected]
        risks = [entry.get("risks", {}).get(task) for entry in entries]
        if not all(finite(v) for v in risks):
            continue
        early_gap, late_gap = risks[0] - risks[1], risks[2] - risks[0]
        out[threshold] = {"relative_threshold": entries[0].get("relative_threshold"),
                          "ranks": [entry.get("rank") for entry in entries],
                          "early_improvement": early_gap, "late_excess": late_gap,
                          "category": classify(early_gap, late_gap, amplitude)}
    return out


def analyze_task(run, representation, task, early_step, late_step, amplitude):
    history = run.get("history", [])
    selected = [fixed_row(history, step) for step in (0, early_step, late_step)]
    values = [risk(row, representation, task) for row in selected]
    losses = [point.get("population_loss_for_evaluation_only") if point else None for point in selected]
    row = {"seed": run["seed"], "mode": run["mode"], "ratio": run["ratio"],
           "representation": representation, "task": task, "run_status": run["status"],
           "initial_risk": values[0], "early_risk": values[1], "late_risk": values[2],
           "initial_view_energy": selected[0].get("view_energy") if selected[0] else None,
           "early_view_energy": selected[1].get("view_energy") if selected[1] else None,
           "late_view_energy": selected[2].get("view_energy") if selected[2] else None,
           "early_inside_T_energy_region": selected[1].get("inside_T_energy_region") if selected[1] else None,
           "late_inside_T_energy_region": selected[2].get("inside_T_energy_region") if selected[2] else None,
           "initial_population_loss": losses[0], "early_population_loss": losses[1], "late_population_loss": losses[2],
           "comparison_population_loss_strictly_decreases": (losses[0] > losses[1] > losses[2]
                                                             if all(finite(value) for value in losses) else None),
           "early_step": early_step, "late_step": late_step, "amplitude_threshold": amplitude,
           "early_improvement": None, "late_excess": None,
           "category": "missing_comparison", "descriptive_minimum_step": None,
           "descriptive_minimum_risk": None, "descriptive_best_improvement": None,
           "descriptive_late_from_minimum": None,
           "rank_threshold_category_stable": None, "rank_threshold_checks": {}}
    failed = run["status"] not in ("complete", "completed", "missing_result", "running", "incomplete", "interrupted")
    if failed or any(value is not None and not finite(value) for value in values):
        row["category"] = "numerical_failure"
    elif run["status"] in ("complete", "completed") and all(finite(value) for value in values):
        row["early_improvement"] = values[0] - values[1]
        row["late_excess"] = values[2] - values[0]
        row["category"] = classify(row["early_improvement"], row["late_excess"], amplitude)
    row["early_margin_above_threshold"] = (row["early_improvement"] - amplitude
                                            if finite(row["early_improvement"]) else None)
    row["late_margin_above_threshold"] = (row["late_excess"] - amplitude
                                           if finite(row["late_excess"]) else None)
    row["near_threshold_complete_rebound"] = (
        min(row["early_improvement"], row["late_excess"]) <= 1.25 * amplitude
        if row["category"] == "complete_rebound" else False)
    row["initial_to_early_loss_change"] = (losses[1] - losses[0]
                                            if all(finite(v) for v in losses[:2]) else None)
    row["early_to_late_loss_change"] = (losses[2] - losses[1]
                                         if all(finite(v) for v in losses[1:]) else None)
    observed = [(point["step"], risk(point, representation, task)) for point in history
                if finite(risk(point, representation, task))]
    if observed:
        min_step, min_risk = min(observed, key=lambda pair: pair[1])
        row.update(descriptive_minimum_step=min_step, descriptive_minimum_risk=min_risk)
        if finite(values[0]):
            row["descriptive_best_improvement"] = values[0] - min_risk
        if finite(values[2]):
            row["descriptive_late_from_minimum"] = values[2] - min_risk
    checks = sensitivity(history, representation, task, early_step, late_step, amplitude)
    row["rank_threshold_checks"] = checks
    if checks:
        row["rank_threshold_category_stable"] = all(
            item["category"] == row["category"] for item in checks.values())
    return row


def geometry_summary(run, task_rows=()):
    history = run.get("history", [])
    result = {"seed": run["seed"], "mode": run["mode"], "ratio": run["ratio"],
              "status": run["status"], "last_recorded_step": max(
                  (row["step"] for row in history), default=None)}
    for representation in REPRESENTATIONS:
        ranks = [row.get(representation + "_rank") for row in history]
        ranks = [int(rank) for rank in ranks if rank is not None]
        minima, relative_minima = [], []
        for row in history:
            values = row.get(representation + "_singular_values", [])
            rank = row.get(representation + "_rank", 0)
            if rank and len(values) >= rank and finite(values[rank - 1]) and finite(values[0]):
                minima.append(values[rank - 1])
                if values[0] > 0:
                    relative_minima.append(values[rank - 1] / values[0])
        threshold_history = {}
        for row in history:
            checks = row.get("rank_threshold_checks", {}).get(representation, {})
            default = checks.get("default")
            if default is None:
                continue
            for cutoff, check in checks.items():
                item = threshold_history.setdefault(cutoff, {
                    "relative_threshold": check.get("relative_threshold"), "recorded_states": 0,
                    "recorded_ranks": [], "maximum_risk_difference_vs_default": 0.,
                    "maximum_difference_step": None, "maximum_difference_task": None})
                item["recorded_states"] += 1
                if check["rank"] not in item["recorded_ranks"]:
                    item["recorded_ranks"].append(check["rank"])
                for task, value in check.get("risks", {}).items():
                    reference = default.get("risks", {}).get(task)
                    if finite(value) and finite(reference):
                        difference = abs(value - reference)
                        if difference > item["maximum_risk_difference_vs_default"]:
                            item.update(maximum_risk_difference_vs_default=difference,
                                        maximum_difference_step=row["step"], maximum_difference_task=task)
        for item in threshold_history.values():
            item["recorded_ranks"].sort()
        for cutoff, item in threshold_history.items():
            decisions = [row for row in task_rows if row["representation"] == representation
                         and row["category"] in CATEGORIES[:4] and cutoff in row["rank_threshold_checks"]]
            item["fixed_comparison_tasks_checked"] = len(decisions)
            item["fixed_comparison_categories_unchanged"] = (all(
                row["rank_threshold_checks"][cutoff]["category"] == row["category"] for row in decisions)
                if decisions else None)
        result[representation] = {"recorded_ranks": sorted(set(ranks)),
                                  "minimum_nonzero_singular_value": min(minima, default=None),
                                  "minimum_relative_nonzero_singular_value": min(relative_minima, default=None),
                                  "rank_threshold_sensitivity_all_recorded_states": threshold_history}
    return result


def load_runs(directory, protocol):
    runs = []
    for seed, mode, ratio in itertools.product(protocol["seeds"], protocol["models"], protocol["ratios"]):
        name = f"seed_{seed}_{mode}_r{ratio:g}"
        candidates = [directory / name / "result.json"]
        if "stream" in protocol:
            candidates.append(directory / (name + f"_stream{protocol['stream']}") / "result.json")
        path = next((path for path in candidates if path.exists()), candidates[0])
        if path.exists():
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                run = {"history": [], "status": "unreadable_result", "error": str(error)}
        else:
            run = {"history": [], "status": "missing_result"}
        # Older complete-trajectory files have no status field.
        run.update(seed=seed, mode=mode, ratio=ratio, source=str(path))
        run.setdefault("status", "complete")
        runs.append(run)
    return runs


def initial_comparisons(runs, protocol):
    """Measure matching from the stored calibrated B, independently of A."""
    comparisons = []
    tasks = np.asarray(protocol.get("task_vectors", []), dtype=float)

    def geometry(matrix):
        matrix = np.asarray(matrix, dtype=float)
        _, values, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.sum(values > max(matrix.shape) * np.finfo(float).eps * values.max()))
        return matrix, vh[:rank].T @ vh[:rank], values, rank

    for seed, ratio in itertools.product(protocol["seeds"], protocol["ratios"]):
        paired = [run for run in runs if run["seed"] == seed and run["ratio"] == ratio
                  and "initial_A" in run and "initial_B" in run]
        for left, right in itertools.combinations(paired, 2):
            a, pa, _, _ = geometry(left["initial_A"])
            c, pc, _, _ = geometry(right["initial_A"])
            b, pb, sb, rb = geometry(left["initial_B"])
            d, pd, sd, rd = geometry(right["initial_B"])
            calculated_risk_difference = None
            if tasks.size and tasks.shape[1] == b.shape[1]:
                calculated_risk_difference = float(np.max(np.abs(
                    np.einsum("ti,ij,tj->t", tasks, pb - pd, tasks))))
            first_left, first_right = fixed_row(left["history"], 0), fixed_row(right["history"], 0)
            stored_difference = max((abs(value - first_right["encoder_risks"][task])
                                     for task, value in first_left["encoder_risks"].items()), default=None)
            comparisons.append({"seed": seed, "ratio": ratio, "left_mode": left["mode"],
                                "right_mode": right["mode"],
                                "initial_A_max_abs_difference": float(np.max(np.abs(a - c))),
                                "projection_projector_frobenius_difference": float(np.linalg.norm(pa - pc)),
                                "encoder_projector_frobenius_difference": float(np.linalg.norm(pb - pd)),
                                "encoder_task_risk_max_abs_difference": calculated_risk_difference,
                                "stored_encoder_task_risk_max_abs_difference": stored_difference,
                                "left_encoder_rank": rb, "right_encoder_rank": rd,
                                "initial_B_max_abs_difference": float(np.max(np.abs(b - d))),
                                "encoder_singular_value_max_abs_difference": float(np.max(np.abs(sb - sd))),
                                "left_to_right_encoder_frobenius_norm_ratio": float(np.linalg.norm(b) / np.linalg.norm(d))})
    return comparisons


def refinement_comparisons(runs, protocol, directories):
    """Keep two-half-step checks separate from the original seed denominator."""
    original = {(run["seed"], run["mode"], run["ratio"]): run for run in runs}
    comparisons = []
    early, late, amplitude = (protocol[key] for key in ("early_step", "late_step", "amplitude_threshold"))
    for directory in directories:
        directory = Path(directory).resolve()
        other_protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        for fine in load_runs(directory, other_protocol):
            key = (fine["seed"], fine["mode"], fine["ratio"])
            if key not in original:
                continue
            base = original[key]
            base_history = {point["step"]: point for point in base.get("history", [])}
            fine_history = {point["step"]: point for point in fine.get("history", [])}
            common_steps = sorted(base_history.keys() & fine_history.keys())
            for representation in REPRESENTATIONS:
                maximum = 0.
                for step in common_steps:
                    for task in protocol["tasks"]:
                        x, y = risk(base_history[step], representation, task), risk(fine_history[step], representation, task)
                        if finite(x) and finite(y):
                            maximum = max(maximum, abs(x - y))
                for task in protocol["tasks"]:
                    left = analyze_task(base, representation, task, early, late, amplitude)
                    right = analyze_task(fine, representation, task, early, late, amplitude)
                    valid = left["category"] in CATEGORIES[:4] and right["category"] in CATEGORIES[:4]
                    comparisons.append({"seed": key[0], "mode": key[1], "ratio": key[2],
                                        "representation": representation, "task": task,
                                        "refinement_directory": str(directory),
                                        "base_microsteps": protocol.get("microsteps"),
                                        "refinement_microsteps": other_protocol.get("microsteps"),
                                        "shared_recorded_steps": len(common_steps),
                                        "maximum_absolute_risk_difference_all_tasks_states": maximum,
                                        "base_early_improvement": left["early_improvement"],
                                        "base_late_excess": left["late_excess"],
                                        "refined_early_improvement": right["early_improvement"],
                                        "refined_late_excess": right["late_excess"],
                                        "base_category": left["category"], "refined_category": right["category"],
                                        "category_changed": left["category"] != right["category"] if valid else None,
                                        "complete_rebound_changed": ((left["category"] == "complete_rebound")
                                            != (right["category"] == "complete_rebound")) if valid else None})
    return comparisons


def write_rows(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def plot_trajectories(runs, tasks, protocol, output):
    early, late = protocol["early_step"], protocol["late_step"]
    modes, ratios = protocol["models"], protocol["ratios"]
    styles = ("-", "--", ":", "-.")
    ratio_styles = {ratio: styles[index % len(styles)] for index, ratio in enumerate(ratios)}
    fig, axes = plt.subplots(len(tasks), 2, figsize=(11.6, 2.05 * len(tasks)),
                             squeeze=False, constrained_layout=True)
    for index, task in enumerate(tasks):
        for column, representation in enumerate(REPRESENTATIONS):
            ax = axes[index, column]
            for run in runs:
                history = [row for row in run.get("history", []) if finite(risk(row, representation, task))]
                if not history:
                    continue
                color = COLORS.get(run["mode"], "#555555")
                ax.plot([row["step"] for row in history], [risk(row, representation, task) for row in history],
                        color=color, linestyle=ratio_styles[run["ratio"]], alpha=.55, linewidth=.8)
                for step in (0, early, late):
                    point = fixed_row(history, step)
                    if point is not None:
                        ax.plot(step, risk(point, representation, task), ".", color=color, ms=3, alpha=.7)
            ax.axvline(early, color="#666666", linestyle=":", linewidth=.7)
            ax.axvline(late, color="#666666", linestyle=":", linewidth=.7)
            ax.set(title=f"{task} | {representation}", ylabel="Exact population risk")
            ax.grid(alpha=.15)
            if index == len(tasks) - 1:
                ax.set_xlabel("SGD updates")
    for mode, ratio in itertools.product(modes, ratios):
        axes[0, 0].plot([], [], color=COLORS.get(mode, "#555555"), linestyle=ratio_styles[ratio],
                        label=f"{mode}, r={ratio:g}")
    axes[0, 0].legend(fontsize=7, ncol=min(3, len(modes)))
    fig.suptitle("All fixed tasks and all available runs; dots mark initial / fixed early / fixed late states", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"risk_trajectories.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(tasks), 2, figsize=(11.6, 2.05 * len(tasks)),
                             squeeze=False, constrained_layout=True)
    for index, task in enumerate(tasks):
        for column, representation in enumerate(REPRESENTATIONS):
            ax = axes[index, column]
            for run in runs:
                history = [row for row in run.get("history", []) if finite(risk(row, representation, task))]
                baseline = risk(fixed_row(history, 0), representation, task)
                if not history or not finite(baseline):
                    continue
                color = COLORS.get(run["mode"], "#555555")
                ax.plot([row["step"] for row in history],
                        [risk(row, representation, task) - baseline for row in history],
                        color=color, linestyle=ratio_styles[run["ratio"]], alpha=.55, linewidth=.8)
                for step in (early, late):
                    point = fixed_row(history, step)
                    if point is not None:
                        ax.plot(step, risk(point, representation, task) - baseline, ".", color=color, ms=3, alpha=.7)
            ax.axhline(0, color="#222222", linewidth=.85)
            ax.axvline(early, color="#666666", linestyle=":", linewidth=.7)
            ax.axvline(late, color="#666666", linestyle=":", linewidth=.7)
            ax.set(title=f"{task} | {representation}", ylabel="R(t) - R(0)")
            ax.grid(alpha=.15)
            if index == len(tasks) - 1:
                ax.set_xlabel("SGD updates")
    for mode, ratio in itertools.product(modes, ratios):
        axes[0, 0].plot([], [], color=COLORS.get(mode, "#555555"), linestyle=ratio_styles[ratio],
                        label=f"{mode}, r={ratio:g}")
    axes[0, 0].legend(fontsize=7, ncol=min(3, len(modes)))
    fig.suptitle("Risk change from each run's own initialization; all fixed tasks and all available seeds", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"risk_changes.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11.6, 8), constrained_layout=True)
    specs = [("population_loss_for_evaluation_only", "Population loss (evaluation)", False),
             ("encoder_rank", "Encoder numerical rank", False),
             ("encoder_min_nonzero", "Encoder minimum nonzero singular value", True),
             ("projection_min_nonzero", "Projection minimum nonzero singular value", True),
             ("encoder_displacement", "Encoder projector displacement", False),
             ("projection_displacement", "Projection projector displacement", False)]
    for ax, (metric, title, logarithmic) in zip(axes.flat, specs):
        for run in runs:
            xs, ys = [], []
            for row in run.get("history", []):
                if metric.endswith("_min_nonzero"):
                    representation = metric.removesuffix("_min_nonzero")
                    values, rank = row.get(representation + "_singular_values", []), row.get(representation + "_rank", 0)
                    value = values[rank - 1] if rank and len(values) >= rank else None
                else:
                    value = row.get(metric)
                if finite(value) and (not logarithmic or value > 0):
                    xs.append(row["step"])
                    ys.append(value)
            if xs:
                ax.plot(xs, ys, color=COLORS.get(run["mode"], "#555555"),
                        linestyle=ratio_styles[run["ratio"]], alpha=.55, linewidth=.85)
        if logarithmic:
            ax.set_yscale("log")
        ax.axvline(early, color="#666666", linestyle=":", linewidth=.7)
        ax.axvline(late, color="#666666", linestyle=":", linewidth=.7)
        ax.set(title=title, xlabel="SGD updates")
        ax.grid(alpha=.15)
    for mode, ratio in itertools.product(modes, ratios):
        axes[0, 0].plot([], [], color=COLORS.get(mode, "#555555"), linestyle=ratio_styles[ratio],
                        label=f"{mode}, r={ratio:g}")
    axes[0, 0].legend(fontsize=7, ncol=min(3, len(modes)))
    fig.suptitle("Every available run; singular values use each state's reported numerical rank", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"geometry.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_directory(directory, output=None, *, early_step=None, late_step=None,
                        amplitude_threshold=None, refinement_dirs=()):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output is not None else directory / "encoder_rebound_summary"
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    protocol = dict(protocol)
    overrides = {}
    for key, value in (("early_step", early_step), ("late_step", late_step),
                       ("amplitude_threshold", amplitude_threshold)):
        if value is not None and value != protocol.get(key):
            overrides[key] = {"stored": protocol.get(key), "analysis": value}
    protocol["early_step"] = early_step if early_step is not None else protocol.get("early_step", 1000)
    protocol["late_step"] = late_step if late_step is not None else protocol.get("late_step", protocol.get("steps", 4000))
    protocol["amplitude_threshold"] = (amplitude_threshold if amplitude_threshold is not None
                                       else protocol.get("amplitude_threshold", .001))
    early, late, amplitude = (protocol[key] for key in ("early_step", "late_step", "amplitude_threshold"))
    if not (0 < early < late and finite(amplitude) and amplitude >= 0):
        raise ValueError("Require 0 < early_step < late_step and a finite nonnegative amplitude threshold")
    runs = load_runs(directory, protocol)
    tasks = protocol.get("tasks")
    if not tasks:
        initial = next((run["history"][0] for run in runs if run.get("history")), None)
        if initial is None:
            raise ValueError("No task list or readable history available")
        tasks = list(initial["encoder_risks"])
    protocol["tasks"] = tasks
    rows = [analyze_task(run, representation, task, early, late, amplitude)
            for run, representation, task in itertools.product(runs, REPRESENTATIONS, tasks)]
    events = [row for row in rows if row["category"] == "complete_rebound"]
    initial_pairs = initial_comparisons(runs, protocol)
    refinements = refinement_comparisons(runs, protocol, refinement_dirs)
    initial_maxima = {key: max((pair[key] for pair in initial_pairs if finite(pair.get(key))), default=None)
                      for key in ("initial_A_max_abs_difference", "projection_projector_frobenius_difference",
                                  "encoder_projector_frobenius_difference", "encoder_task_risk_max_abs_difference",
                                  "stored_encoder_task_risk_max_abs_difference", "initial_B_max_abs_difference",
                                  "encoder_singular_value_max_abs_difference")}
    groups, family_groups = [], []
    for mode, ratio, representation in itertools.product(protocol["models"], protocol["ratios"], REPRESENTATIONS):
        family = [row for row in rows if (row["mode"], row["ratio"], row["representation"]) == (mode, ratio, representation)]
        for task in tasks:
            group = [row for row in family if row["task"] == task]
            groups.append({"mode": mode, "ratio": ratio, "representation": representation, "task": task,
                           "planned_runs": len(group),
                           "categories": {name: sum(row["category"] == name for row in group) for name in CATEGORIES},
                           "early_improvement": describe([row["early_improvement"] for row in group]),
                           "late_excess": describe([row["late_excess"] for row in group]),
                           "complete_rebound_early_improvement": describe([
                               row["early_improvement"] for row in group if row["category"] == "complete_rebound"]),
                           "complete_rebound_late_excess": describe([
                               row["late_excess"] for row in group if row["category"] == "complete_rebound"]),
                           "complete_rebound_comparisons_inside_T_energy_region": sum(
                               row["category"] == "complete_rebound"
                               and row["early_inside_T_energy_region"] is True
                               and row["late_inside_T_energy_region"] is True for row in group),
                           "complete_rebound_comparison_population_loss_decreases": sum(
                               row["category"] == "complete_rebound"
                               and row["comparison_population_loss_strictly_decreases"] is True for row in group),
                           "rank_threshold_sensitive_runs": [row["seed"] for row in group
                                                             if row["rank_threshold_category_stable"] is False]})
        per_seed = [{"seed": seed,
                     "complete_rebound_tasks": [row["task"] for row in family if row["seed"] == seed
                                                and row["category"] == "complete_rebound"],
                     "missing_or_failed_tasks": [row["task"] for row in family if row["seed"] == seed
                                                 and row["category"] in CATEGORIES[-2:]]}
                    for seed in protocol["seeds"]]
        family_groups.append({"mode": mode, "ratio": ratio, "representation": representation,
                              "planned_seeds": len(per_seed),
                              "seeds_with_any_fixed_task_rebound": sum(bool(item["complete_rebound_tasks"]) for item in per_seed),
                              "seeds_with_all_fixed_tasks_rebound": sum(len(item["complete_rebound_tasks"]) == len(tasks)
                                                                       for item in per_seed),
                              "per_seed": per_seed})
    exploratory = bool(overrides) or protocol.get("phase") not in ("small_validation", "rebound_confirmation")
    summary = {"source_directory": str(directory), "protocol": protocol,
               "analysis_scope": "exploratory_reanalysis" if exploratory else "small_validation",
               "analysis_overrides": overrides,
               "near_threshold_description": "Among complete rebounds only, flag min(I,D)<=1.25*delta. Descriptive flag; the original strict >delta decision is unchanged.",
               "decision_rule": "R(0)-R(early)>amplitude AND R(late)-R(0)>amplitude; exact fixed steps, no minimum selection",
               "planned_run_count": len(runs),
               "run_status_counts": {status: sum(run["status"] == status for run in runs)
                                     for status in sorted({run["status"] for run in runs})},
               "per_task": groups, "fixed_task_family": family_groups,
               "complete_rebound_events": [{key: value for key, value in event.items()
                                              if key != "rank_threshold_checks"} for event in events],
               "event_loss_cross_summary": [{"mode": mode, "representation": representation,
                    "complete_task_events": sum(row["mode"] == mode and row["representation"] == representation for row in events),
                    "events_with_two_segment_population_loss_descent": sum(
                        row["mode"] == mode and row["representation"] == representation
                        and row["comparison_population_loss_strictly_decreases"] is True for row in events)}
                    for mode, representation in itertools.product(protocol["models"], REPRESENTATIONS)],
               "initial_pairing": {"comparisons": initial_pairs, "maximum_differences": initial_maxima,
                    "interpretation": "A matching alone does not imply encoder-space matching; B-space projectors and task risks are compared directly. Parameter scales and singular values can differ. Controls do not identify a unique causal source."},
               "refinement_checks": refinements,
               "geometry": [geometry_summary(run, [row for row in rows
                             if (row["seed"], row["mode"], row["ratio"])
                             == (run["seed"], run["mode"], run["ratio"])]) for run in runs],
               "limitations": ["Small seed counts are descriptive, not population occurrence probabilities.",
                               "Any-task counts refer to the entire fixed task family; per-task counts are also reported.",
                               "Empirical minima are descriptive only and never enter the primary decision.",
                               "Reported risk is exact population linear-readout risk for the known synthetic distribution; SGD uses finite data.",
                               "SGD does not imply strict population-loss descent; initial/early/late losses and whether both comparison segments decrease are reported separately.",
                               "Task-events sharing a seed are correlated; event counts and any-task unions are not independent occurrence probabilities.",
                               "Two-half-step refinements are separate numerical checks, never replacement main runs or additional independent seeds.",
                               "Empirical finite-data linear training is not covered by isotropic population row-space conservation.",
                               "Finite-dataset SGD is not a population-flow theorem validation. Comparisons with view energy Q>=1 are outside the theory's stated Q<1 energy region; early/late view energies and region flags are retained per run."]}
    if exploratory:
        summary["limitations"].append("Comparison steps and amplitude rule were applied after these historical runs; this is exploratory evidence.")
    if not any(row["rank_threshold_checks"] for row in rows):
        summary["limitations"].append("Historical results contain no relative-rank-threshold risk checks; sensitivity is unavailable.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "rank_threshold_checks": json.dumps(row["rank_threshold_checks"], separators=(",", ":"))})
    event_fields = ["seed", "mode", "ratio", "representation", "task", "initial_risk", "early_risk", "late_risk",
                    "early_improvement", "late_excess", "early_margin_above_threshold", "late_margin_above_threshold",
                    "near_threshold_complete_rebound", "initial_population_loss", "early_population_loss",
                    "late_population_loss", "initial_to_early_loss_change", "early_to_late_loss_change",
                    "comparison_population_loss_strictly_decreases", "early_view_energy", "late_view_energy"]
    write_rows(output / "rebound_events.csv", events, event_fields)
    write_rows(output / "initial_pairing.csv", initial_pairs,
               list(initial_pairs[0]) if initial_pairs else ["seed", "left_mode", "right_mode"])
    if refinement_dirs:
        write_rows(output / "refinement_checks.csv", refinements,
                   list(refinements[0]) if refinements else ["seed", "mode", "representation", "task"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Fixed-time complete encoder rebound", "",
             f"Scope: {summary['analysis_scope']}. Planned runs: {len(runs)}; status counts: {summary['run_status_counts']}.", "",
             f"Primary rule: R(0)-R({early}) > {amplitude:g} and R({late})-R(0) > {amplitude:g} for the same task.",
             "Task directions are fixed. A fitted minimum does not enter the primary decision.", "",
             "Primary per-fixed-task results (each column retains the planned seed denominator):", "",
             "| Model | Ratio | Representation | Fixed task | Complete | Improve only | Deteriorate only | Neither | Missing | Failed |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        counts = group["categories"]
        lines.append(f"| {group['mode']} | {group['ratio']:g} | {group['representation']} | {group['task']} | "
                     + " | ".join(str(counts[name]) for name in CATEGORIES) + " |")
    lines += ["", f"Descriptive union across {len(tasks)} fixed tasks; these are not per-task frequencies or uniform task-family guarantees:", "",
              "| Model | Rate ratio | Representation | Any fixed task / planned seeds | All fixed tasks / planned seeds |",
              "|---|---:|---|---:|---:|"]
    for group in family_groups:
        lines.append(f"| {group['mode']} | {group['ratio']:g} | {group['representation']} | "
                     f"{group['seeds_with_any_fixed_task_rebound']}/{group['planned_seeds']} | "
                     f"{group['seeds_with_all_fixed_tasks_rebound']}/{group['planned_seeds']} |")
    checks_available = sum(bool(row["rank_threshold_checks"]) for row in rows)
    checks_sensitive = sum(row["rank_threshold_category_stable"] is False for row in rows)
    lines += ["", f"Rank-threshold checks available for {checks_available}/{len(rows)} task/representation records; "
              f"{checks_sensitive} primary decisions change under a checked threshold.", "",
              "Per-cutoff rank sets and maximum risk differences from the default cutoff over every recorded state/task are in summary.json geometry records.", "",
              "Signed gap distributions and all failed/missing records are retained in summary.json and runs.csv.",
              "Full risk and baseline-subtracted risk plots show every available seed for every fixed task. Geometry plots show the loss, encoder rank, "
              "minimum nonzero singular values, and projector displacement.", ""]
    lines += ["Complete rebound events crossed with two-segment population-loss descent:", "",
              "| Model | Representation | Complete task-events | With L(0)>L(early)>L(late) |",
              "|---|---|---:|---:|"]
    for group in summary["event_loss_cross_summary"]:
        lines.append(f"| {group['mode']} | {group['representation']} | {group['complete_task_events']} | "
                     f"{group['events_with_two_segment_population_loss_descent']} |")
    lines += ["", "These task-events are correlated within a seed. All signed I/D amplitudes are in runs.csv; "
              "every complete event, its three losses, both loss changes and margin flag is in rebound_events.csv.",
              summary["near_threshold_description"], "", "Direct initial-factor comparison (all model pairs within seed/rate):", ""]
    lines.extend(f"- Maximum {key}: {value:.12g}" for key, value in initial_maxima.items() if value is not None)
    lines += ["", summary["initial_pairing"]["interpretation"], ""]
    if refinements:
        checked = sum(row["complete_rebound_changed"] is not None for row in refinements)
        changed = sum(row["complete_rebound_changed"] is True for row in refinements)
        lines += [f"Separate step refinements: {checked} task/representation comparisons; {changed} complete-rebound decisions change. "
                  "Per-run full-curve risk differences and all endpoint amplitudes are retained in refinement_checks.csv. "
                  "They do not modify primary counts.", ""]
    lines.extend("- " + limitation for limitation in summary["limitations"])
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    event_lines = ["# Complete rebound events and objective changes", "",
                   "All complete events, including projections in both controls. Events sharing a seed are correlated.",
                   summary["near_threshold_description"], "",
                   "| Seed | Model | Representation | Task | I | D | L(0) | L(early) | L(late) | Both loss segments descend | Near margin |",
                   "|---:|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in events:
        event_lines.append(f"| {row['seed']} | {row['mode']} | {row['representation']} | {row['task']} | "
                           + " | ".join(f"{row[key]:.9g}" for key in ("early_improvement", "late_excess",
                                 "initial_population_loss", "early_population_loss", "late_population_loss"))
                           + f" | {row['comparison_population_loss_strictly_decreases']} | {row['near_threshold_complete_rebound']} |")
    (output / "rebound_event_loss.md").write_text("\n".join(event_lines) + "\n", encoding="utf-8")
    plot_trajectories(runs, tasks, protocol, output)
    print(json.dumps({"output": str(output), "scope": summary["analysis_scope"],
                      "planned_runs": len(runs), "run_status_counts": summary["run_status_counts"],
                      "encoder_fixed_task_family": [{key: value for key, value in group.items() if key != "per_seed"}
                                                    for group in family_groups if group["representation"] == "encoder"]},
                     ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--early-step", type=int)
    parser.add_argument("--late-step", type=int)
    parser.add_argument("--amplitude-threshold", type=float)
    parser.add_argument("--refinement-dirs", nargs="+", type=Path, default=[])
    args = parser.parse_args()
    summarize_directory(args.directory, args.output, early_step=args.early_step, late_step=args.late_step,
                        amplitude_threshold=args.amplitude_threshold, refinement_dirs=args.refinement_dirs)


if __name__ == "__main__":
    main()
