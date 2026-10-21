"""Summarize every planned texture prediction state, including absent windows.

Uses stdlib only. Streams, tasks and checkpoints are repeated measurements;
seed is the statistical unit. No retrospective refitting of predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def key(row):
    return (row["seed"], row["mode"], row["checkpoint_step"], row["representation"],
            row["ratio"], row["horizon"], row["task_family"])


def pair_order(tasks, score, floor):
    correct = total = 0
    for i, first in enumerate(tasks):
        for second in tasks[i+1:]:
            actual = first["mean_change_absolute"]-second["mean_change_absolute"]
            if abs(actual) <= floor:
                continue
            predicted = first[score]-second[score]
            total += 1
            correct += .5 if abs(predicted) <= floor else float(predicted*actual > 0)
    return {"pair_accuracy": correct/total if total else None, "comparable_pairs": total}


def state_metrics(protocol, predictions, outcomes, statuses):
    pgroup, ogroup = defaultdict(list), defaultdict(list)
    for row in predictions:
        pgroup[key(row)].append(row)
    for row in outcomes:
        ogroup[key(row)].append(row)
    settings = protocol["settings"]
    families = {"predefined_coordinate": 2, "predefined_band": 8}
    if settings["constructed_task"]:
        families["constructed_max_competition"] = 1
    rows = []
    for state in protocol["states"]:
        state_id = (state["seed"], state["mode"], state["step"])
        for rep in ("encoder", "projection"):
            for ratio in settings["ratios"]:
                for horizon in settings["horizons"]:
                    for family, expected in families.items():
                        group_id = (*state_id, rep, ratio, horizon, family)
                        p, o = pgroup[group_id], ogroup[group_id]
                        usable = sum(r["usable_empirical_window"] for r in p)
                        actual = len(o)
                        per_task = []
                        for task in sorted({r["task"] for r in o}):
                            obs = [r for r in o if r["task"] == task and r["same_rank"]]
                            if obs:
                                per_task.append({"task": task, "mean_change_absolute": abs(mean(r["observed_change"] for r in obs)),
                                    "slope_score": abs(obs[0]["predicted_change"]),
                                    "cross_score": obs[0]["forecast_task_cross_bound"],
                                    "displacement_score": obs[0]["forecast_displacement_only_bound"]})
                        slopes = pair_order(per_task, "slope_score", settings["numeric_floor"])
                        cross = pair_order(per_task, "cross_score", settings["numeric_floor"])
                        uniform = pair_order(per_task, "displacement_score", settings["numeric_floor"])
                        usable_observed = [r for r in o if r["usable_empirical_window"]]
                        magnitudes = [r["mean_change_absolute"] for r in per_task]
                        rows.append({"seed": state_id[0], "mode": state_id[1], "checkpoint_step": state_id[2],
                            "representation": rep, "ratio": ratio, "horizon": horizon, "task_family": family,
                            "source_status": statuses[state_id]["status"], "planned_task_slots": expected,
                            "available_predictions": len(p), "nonempty_empirical_windows": usable,
                            "empirical_window_coverage_planned": usable/expected,
                            "strict_certificates": sum(r["strict_certificate_available"] for r in p),
                            "planned_stream_task_observations": expected*settings["future_streams"],
                            "observed_stream_task_count": actual,
                            "interval_hit_rate_available": average([r["empirical_interval_hit"] for r in o]),
                            "interval_hits_all_planned": sum(r["empirical_interval_hit"] for r in o)/(expected*settings["future_streams"]),
                            "mean_interval_width_available": average([r["interval_width"] for r in p]),
                            "mean_width_over_practical_scale": average([r["interval_width"]/settings["practical_change"] for r in p]),
                            "median_interval_width_available": median(r["interval_width"] for r in p) if p else None,
                            "mean_absolute_prediction_error": average([r["absolute_prediction_error"] for r in o]),
                            "zero_change_baseline_mae": average([abs(r["observed_change"]) for r in o]),
                            "usable_direction_accuracy_available": average([r["usable_direction_correct"] for r in usable_observed]),
                            "usable_direction_correct_all_planned": sum(r["usable_direction_correct"] for r in o)/(expected*settings["future_streams"]),
                            "step_direction_persistence_available": average([r["increment_direction_persisted"] for r in usable_observed]),
                            "net_direction_persistence_available": average([r["net_direction_persisted"] for r in usable_observed]),
                            "rank_failure_observations": sum(not r["same_rank"] for r in o),
                            "forecast_motion_bound_coverage": average([r["forecast_displacement_bound_hit"] for r in o]),
                            "forecast_task_bound_coverage": average([r["forecast_task_bound_hit"] for r in o]),
                            "mean_forecast_motion_bound": average([r["forecast_displacement_only_bound"] for r in p]),
                            "mean_forecast_task_bound": average([r["forecast_task_cross_bound"] for r in p]),
                            "mean_posterior_motion_bound": average([r["posterior_displacement_bound"] for r in o]),
                            "mean_posterior_task_bound": average([r["posterior_task_cross_bound"] for r in o]),
                            "predicted_small_tasks": sum(r["predicted_small_change"] for r in p),
                            "small_prediction_accuracy_available": average([r["observed_small_change"] for r in o if r["predicted_small_change"]]),
                            "predicted_large_tasks": sum(r["predicted_large_directional_change"] for r in p),
                            "large_prediction_accuracy_available": average([
                                abs(r["observed_change"]) > settings["practical_change"] and r["usable_direction_correct"]
                                for r in o if r["predicted_large_directional_change"]]),
                            "shared_geometry_task_change_min": min(magnitudes) if magnitudes else None,
                            "shared_geometry_task_change_max": max(magnitudes) if magnitudes else None,
                            "shared_geometry_task_change_spread": max(magnitudes)-min(magnitudes) if magnitudes else None,
                            "slope_task_order_accuracy": slopes["pair_accuracy"],
                            "cross_block_task_order_accuracy": cross["pair_accuracy"],
                            "displacement_only_task_order_accuracy": uniform["pair_accuracy"],
                            "comparable_task_pairs": slopes["comparable_pairs"],
                            "no_window_reasons": dict(Counter(reason for row in p for reason in row["no_window_reasons"]))})
    return rows


def coverage_metrics(state_rows):
    # All three family-specific denominators include every planned state.
    groups = defaultdict(list)
    for row in state_rows:
        groups[(row["mode"], row["checkpoint_step"], row["representation"], row["ratio"],
                row["horizon"], row["task_family"])].append(row)
    out = []
    for group, rows in sorted(groups.items()):
        planned = sum(r["planned_task_slots"] for r in rows)
        available = sum(r["available_predictions"] for r in rows)
        observations = sum(r["observed_stream_task_count"] for r in rows)
        out.append({"mode": group[0], "checkpoint_step": group[1], "representation": group[2],
            "ratio": group[3], "horizon": group[4], "task_family": group[5],
            "planned_states": len(rows), "planned_task_slots": planned, "available_prediction_slots": available,
            "nonempty_empirical_window_slots": sum(r["nonempty_empirical_windows"] for r in rows),
            "empirical_window_coverage_all_planned": sum(r["nonempty_empirical_windows"] for r in rows)/planned,
            "strict_certificate_coverage_all_planned": sum(r["strict_certificates"] for r in rows)/planned,
            "observed_stream_task_count": observations,
            "planned_stream_task_observations": sum(r["planned_stream_task_observations"] for r in rows),
            "descriptive_mean_seed_mae": average([r["mean_absolute_prediction_error"] for r in rows]),
            "descriptive_mean_seed_interval_hit_rate": average([r["interval_hit_rate_available"] for r in rows]),
            "descriptive_mean_seed_width": average([r["mean_interval_width_available"] for r in rows]),
            "descriptive_mean_seed_direction_accuracy": average([r["usable_direction_accuracy_available"] for r in rows]),
            "descriptive_mean_seed_step_persistence": average([r["step_direction_persistence_available"] for r in rows]),
            "descriptive_mean_seed_task_order_accuracy": average([r["slope_task_order_accuracy"] for r in rows]),
            "descriptive_mean_seed_displacement_order_baseline": average([r["displacement_only_task_order_accuracy"] for r in rows]),
            "scope": "within-seed task/stream summaries averaged across seeds; no independent-trial or confidence claim"})
    return out


def window_metrics(predictions, outcomes, protocol):
    pg, og = defaultdict(list), defaultdict(list)
    fields = ("seed", "mode", "checkpoint_step", "representation", "ratio", "task", "task_family")
    for row in predictions:
        pg[tuple(row[f] for f in fields)].append(row)
    for row in outcomes:
        og[tuple(row[f] for f in fields)].append(row)
    rows = []
    for group, predictions_for_task in pg.items():
        horizons = {r["horizon"]: r for r in predictions_for_task}
        eligible = validated = persistent = 0
        forecast_open = outcome_open = persistence_open = True
        for h in protocol["settings"]["horizons"]:
            pred = horizons.get(h)
            observed = [r for r in og[group] if r["horizon"] == h]
            forecast_open = forecast_open and pred is not None and pred["usable_empirical_window"]
            complete = len(observed) == protocol["settings"]["future_streams"]
            outcome_open = outcome_open and forecast_open and complete and all(
                r["usable_direction_correct"] and r["empirical_interval_hit"] for r in observed)
            persistence_open = persistence_open and outcome_open and all(r["increment_direction_persisted"] for r in observed)
            if forecast_open:
                eligible = h
            if outcome_open:
                validated = h
            if persistence_open:
                persistent = h
        rows.append({**dict(zip(fields, group)), "source_empirical_prefix_window": eligible,
                     "all_sampled_streams_validated_prefix_window": validated,
                     "all_sampled_streams_increment_persistence_window": persistent,
                     "strict_direction_window": 0,
                     "exact_invariance_through_horizon": max(horizons) if len(horizons) == len(protocol["settings"]["horizons"]) and all(r["strict_certificate_available"] for r in horizons.values()) else 0,
                     "scope": "descriptive validation on sampled streams, not a guarantee on other SGD streams"})
    return rows


def geometry_metrics(protocol, geometries, statuses):
    grouped = defaultdict(list)
    for row in geometries:
        grouped[(row["seed"], row["mode"], row["checkpoint_step"], row["representation"], row["ratio"], row["horizon"])].append(row)
    rows = []
    for state in protocol["states"]:
        sid = (state["seed"], state["mode"], state["step"])
        for rep in ("encoder", "projection"):
            for ratio in protocol["settings"]["ratios"]:
                for horizon in protocol["settings"]["horizons"]:
                    observed = grouped[(*sid, rep, ratio, horizon)]
                    rows.append({"seed": sid[0], "mode": sid[1], "checkpoint_step": sid[2],
                        "representation": rep, "ratio": ratio, "horizon": horizon,
                        "source_status": statuses[sid]["status"], "observed_streams": len(observed),
                        "planned_streams": protocol["settings"]["future_streams"],
                        "projector_radius_hits_all_planned": sum(r["projector_empirical_radius_hit"] for r in observed)/protocol["settings"]["future_streams"],
                        "normal_radius_hits_all_planned": sum(r["normal_empirical_radius_hit"] for r in observed)/protocol["settings"]["future_streams"],
                        **{f"mean_{field}": average([r[field] for r in observed]) for field in (
                            "projector_forecast_error", "projector_direction_cosine", "normal_forecast_error",
                            "normal_direction_cosine", "observed_projector_displacement", "predicted_projector_displacement",
                            "future_front_tangent_error", "future_head_tangent_error",
                            "future_front_tangent_cosine", "future_head_tangent_cosine",
                            "front_parameter_path_length", "head_parameter_path_length",
                            "total_parameter_block_path_length")}})
    return rows


def summarize(root):
    protocol = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    statuses, predictions, outcomes, geometries, constructions = {}, [], [], [], []
    for state in protocol["states"]:
        sid = (state["seed"], state["mode"], state["step"])
        directory = root/f"seed_{sid[0]}_{sid[1]}_step{sid[2]}"
        status_file = directory/"status.json"
        metadata = {"seed": sid[0], "mode": sid[1], "checkpoint_step": sid[2]}
        try:
            statuses[sid] = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {
                **metadata, "status": "not_completed"}
        except (OSError, json.JSONDecodeError) as error:
            statuses[sid] = {**metadata, "status": "unreadable_status", "reason": str(error)}
        if statuses[sid]["status"] == "complete" and not all((directory/name).is_file() for name in ("predictions.json", "outcomes.json")):
            statuses[sid] = {**metadata, "status": "incomplete_outputs"}
        if (directory/"predictions.json").exists():
            try:
                source = json.loads((directory/"predictions.json").read_text(encoding="utf-8"))
                predictions.extend({**metadata, **r} for r in source["predictions"])
                constructions.append({**metadata, **source["constructed_task"]})
            except (OSError, json.JSONDecodeError) as error:
                statuses[sid] = {**metadata, "status": "unreadable_predictions", "reason": str(error)}
        if (directory/"outcomes.json").exists():
            try:
                future = json.loads((directory/"outcomes.json").read_text(encoding="utf-8"))
                outcomes.extend({**metadata, **r} for r in future["outcomes"])
                geometries.extend({**metadata, **r} for r in future["geometry"])
            except (OSError, json.JSONDecodeError) as error:
                statuses[sid] = {**metadata, "status": "unreadable_outcomes", "reason": str(error)}
    rows = state_metrics(protocol, predictions, outcomes, statuses)
    coverage = coverage_metrics(rows)
    windows = window_metrics(predictions, outcomes, protocol)
    geometry_by_state = geometry_metrics(protocol, geometries, statuses)
    summary = {"phase": protocol["phase"], "planned_states": len(protocol["states"]),
        "state_status_counts": dict(Counter(s["status"] for s in statuses.values())),
        "states": list(statuses.values()), "settings": protocol["settings"],
        "constructed_task_available_states": sum(c["available"] for c in constructions),
        "construction_evaluated_states": len(constructions),
        "construction_not_evaluated_states": len(protocol["states"])-len(constructions),
        "constructed_task_coverage_all_planned": sum(c["available"] for c in constructions)/len(protocol["states"]),
        "constructions": constructions, "coverage": coverage,
        "stochastic_direction_certificate_claim": False,
        "strict_certificate_reason": "only frozen-encoder exact invariance is certified; calibration quantiles do not give uniform future-SGD bounds",
        "max_leakage_identity_residual": max((abs(r["leakage_identity_residual"]) for r in outcomes), default=None),
        "posterior_task_bound_failures": sum(not r["posterior_task_bound_holds"] for r in outcomes if r["same_rank"]),
        "exact_frozen_encoder_invariance_failures": sum(r["exact_invariance_check_passed"] is False for r in outcomes),
        "posterior_scope": "exact bounds given OBSERVED displacement are controls, never forward prediction evidence",
        "aggregation": "source/task/stream records retained; descriptive means first within seed, no p-values or trial-count inference",
        "window_scope": "source empirical prefix windows and sampled-stream validation reported separately"}
    (root/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for filename, records in (("state_metrics.csv", rows), ("coverage.csv", coverage),
                              ("windows.csv", windows), ("geometry_metrics.csv", geometries),
                              ("geometry_by_state.csv", geometry_by_state),
                              ("task_predictions.csv", predictions), ("task_outcomes.csv", outcomes)):
        save_csv(root/filename, records)
    print(json.dumps({"planned_states": summary["planned_states"], "state_status_counts": summary["state_status_counts"],
                      "uniform_stochastic_direction_certificates": False, "summary": str(root/"summary.json")}, indent=2))


def self_check():
    settings = {"constructed_task": True, "ratios": [1.], "horizons": [1, 8], "future_streams": 2, "numeric_floor": 1e-9}
    protocol = {"settings": settings, "states": [{"seed": 0, "mode": "square", "step": 500},
                                                    {"seed": 1, "mode": "square", "step": 500}]}
    statuses = {(0, "square", 500): {"status": "complete"},
                (1, "square", 500): {"status": "missing_checkpoint"}}
    rows = state_metrics(protocol, [], [], statuses)
    groups = coverage_metrics(rows)
    assert len(rows) == 2*2*2*3
    assert all(r["empirical_window_coverage_all_planned"] == 0 and
               r["strict_certificate_coverage_all_planned"] == 0 for r in groups)
    assert all(r["planned_states"] == 2 for r in groups)
    assert pair_order([{"mean_change_absolute": 1., "score": 0.},
                       {"mean_change_absolute": 2., "score": 0.}], "score", 1e-9)["pair_accuracy"] == .5
    print("Summary checks passed: missing/no-window states retained, families separated, tied displacement baseline handled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.input:
        summarize(args.input)
    else:
        parser.error("provide an experiment directory or --self-check")


if __name__ == "__main__":
    main()
