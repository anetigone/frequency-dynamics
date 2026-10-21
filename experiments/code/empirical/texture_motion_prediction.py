"""Source-calibrated geometry/task/window forecasts on saved texture checkpoints.

No initial training, downloads, hashes, or automatic tuning on future outcomes.
SGD error quantiles give empirical prediction intervals, NOT strict certificates.
Use --dry-run to inspect the fixed state matrix, or --self-check for tiny tensors.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import RESULTS, result_path

import torch

from full_empirical import energy_tensors, update
from sgd_state_tools import draw_batch, restore, training_data
from square_images import empirical_loss, texture_tasks


DEFAULTS = {
    "lr": .002, "batch_size": 256, "ratios": [.1, 1., 10.],
    "horizons": [1, 4, 8, 16, 32, 64], "fit_batches": 16,
    "error_batches": 32, "probe_streams": 4, "probe_steps": 8,
    "future_streams": 8, "quantile": .9, "inflation": 2.,
    "numeric_floor": 1e-9, "practical_change": 1e-4,
    "allow_window_extrapolation": False, "constructed_task": True,
    "rng_base": 20261021,
}
SEEDS = {"development": [0, 1], "validation": list(range(2, 12))}
STEPS = {"development": [500, 2000], "validation": [500, 2000, 3500]}
MODES = ["square", "linear", "frozen_front"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def generator_for(settings, seed, step, role, repeat=0):
    # Same future stream across modes/ratios, distinct fit/error/probe/future roles.
    number = (settings["rng_base"] + 1000003*seed + 1009*step
              + 100000000*role + 10000000000*repeat) % (2**63-1)
    return torch.Generator().manual_seed(number)


def row_space(x):
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    tol = max(x.shape)*torch.finfo(x.dtype).eps*float(s.max())
    rank = int((s > tol).sum())
    p = vh[:rank].T @ vh[:rank]
    return p, rank, s


def projector_tangent(x, velocity):
    p, rank, _ = row_space(x)
    if rank != x.shape[0]:
        raise ValueError("Projector derivative requires full row rank")
    inverse = torch.linalg.solve(x@x.T, x).T
    normal = inverse @ velocity @ (torch.eye(x.shape[1], dtype=x.dtype)-p)
    return normal + normal.T


def effective_gradient(model, batch):
    a = model.effective()[0].detach().requires_grad_(True)
    e1, e2 = batch
    loss = empirical_loss(e1@a.T, e2@a.T)
    gradient = torch.autograd.grad(loss, a)[0].detach()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite effective training gradient")
    return gradient


def mean_gradient(model, batches):
    return torch.stack([effective_gradient(model, b) for b in batches]).mean(0)


@torch.no_grad()
def block_fields(model, gradient):
    a, b = model.effective()
    db = (torch.zeros_like(b) if model.mode == "frozen_front" else
          -torch.einsum("cij,cj->ci", model.kernel_gram(), model.weight.T@gradient))
    da_h = model.weight@db
    da_w = -gradient@b.T@b
    return {"encoder": {"front": db, "head": torch.zeros_like(b)},
            "projection": {"front": da_h, "head": da_w}}


def rates(model, ratio, lr):
    return lr*2*model.mobility/(1+ratio), lr*2*ratio*model.mobility/(1+ratio)


@torch.no_grad()
def geometry(model, gradient, ratio, lr):
    a, b = [v.detach() for v in model.effective()]
    fields = block_fields(model, gradient)
    fh, fw = rates(model, ratio, lr)
    result = {}
    for name, x in (("encoder", b), ("projection", a)):
        p, rank, singular = row_space(x)
        if rank != len(x):
            raise ValueError(f"{name} is not full row rank")
        q = torch.eye(x.shape[1], dtype=x.dtype)-p
        zh, zw = fields[name]["front"]*fh, fields[name]["head"]*fw
        dh, dw = projector_tangent(x, zh), projector_tangent(x, zw)
        result[name] = {"X": x, "P": p, "rank": rank, "singular": singular,
                        "D_front": dh, "D_head": dw, "D": dh+dw,
                        "velocity_front": zh, "velocity_head": zw,
                        "normal_front": zh@q, "normal_head": zw@q}
    return result


@torch.no_grad()
def risk_slopes(tangent, tasks):
    return -torch.einsum("ti,ij,tj->t", tasks, tangent, tasks)


def maximum_competition_task(projection, floor):
    """Exact maximum min(|c_h|,|c_W|) for opposite signs when K=1.

    With D_g=u v_g^T+v_g u^T, minimize the norm on segment [-v_h,v_W].
    Its closest-point direction gives the unit rank-one task's largest margin.
    This is selected from source calibration only, never from future task risks.
    """
    if projection["X"].shape[0] != 1:
        return None, {"available": False, "reason": "requires K=1"}
    u = projection["X"][0]/projection["X"][0].norm()
    first, second = -projection["D_front"]@u, projection["D_head"]@u
    difference = second-first
    length = float(difference.square().sum())
    fraction = min(1., max(0., -float(first@difference)/length)) if length else 0.
    closest = first + fraction*difference
    margin = float(closest.norm())
    if margin <= floor:
        return None, {"available": False, "reason": "no resolved competing source tangents",
                      "maximum_margin": margin}
    normal = closest/closest.norm()
    beta = (u+normal)/math.sqrt(2)
    ch = float(risk_slopes(projection["D_front"], beta[None])[0])
    cw = float(risk_slopes(projection["D_head"], beta[None])[0])
    if ch <= floor or cw >= -floor:
        return None, {"available": False, "reason": "competition below numeric floor",
                      "maximum_margin": margin}
    return beta, {"available": True, "selection": "exact K=1 max-min opposing block margin",
                  "maximum_margin": margin, "front_slope": ch, "head_slope": cw,
                  "beta": beta.tolist(), "selected_using_future": False}


def check_source_jvp(model, gradient):
    """Independent parameter-autograd/JVP check of the mobility decomposition."""
    a, _ = model.effective()
    parameters = model.parameters()
    grads = torch.autograd.grad(a, parameters, grad_outputs=gradient)
    if model.mode == "frozen_front":
        vh, vw = torch.zeros_like(model.front), -grads[0]
    else:
        vh, vw = [-g.detach() for g in grads]
    values = (model.front.detach(), model.weight.detach())
    function = lambda front, weight: energy_tensors(model, front, weight)
    jh = torch.autograd.functional.jvp(function, values, (vh, torch.zeros_like(vw)))[1]
    jw = torch.autograd.functional.jvp(function, values, (torch.zeros_like(vh), vw))[1]
    fields = block_fields(model, gradient)
    return {"scope": "source calculus check only; not future predictive evidence",
            "max_absolute_error": max(float((actual-expected).abs().max())
                for actual, expected in ((jh[0], fields["projection"]["front"]),
                                         (jh[1], fields["encoder"]["front"]),
                                         (jw[0], fields["projection"]["head"]),
                                         (jw[1], fields["encoder"]["head"]))) }


def quantile(values, settings):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return settings["inflation"]*torch.quantile(tensor, settings["quantile"], dim=0)


def finite_remainder(model, batch, before, tasks, ratio, settings):
    """Actual SGD map-minus-tangent residual on calibration clones only."""
    candidate = copy.deepcopy(model)
    update(candidate, *batch, settings["lr"], ratio)
    out = {}
    with torch.no_grad():
        a, b = candidate.effective()
        for name, x in (("encoder", b), ("projection", a)):
            p1, rank, _ = row_space(x)
            if rank != before[name]["rank"]:
                raise ValueError("Rank changed on calibration step")
            residual = p1-before[name]["P"]-before[name]["D"]
            out[name] = {"geometry": float(residual.norm()),
                         "task": risk_slopes(residual, tasks).abs(),
                         # A full effective-map remainder also controls its
                         # projection onto a fixed source normal space.
                         "normal": float((x-before[name]["X"]-
                            before[name]["velocity_front"]-before[name]["velocity_head"]).norm())}
    return out


def calibrate(model, data, seed, step, names, families, tasks, settings):
    fit_rng = generator_for(settings, seed, step, 1)
    error_rng = generator_for(settings, seed, step, 2)
    fit_batches = [draw_batch(data, settings["batch_size"], fit_rng)
                   for _ in range(settings["fit_batches"])]
    gradient = mean_gradient(model, fit_batches)
    base = geometry(model, gradient, 1., settings["lr"])
    source_check = check_source_jvp(model, gradient)
    if source_check["max_absolute_error"] > 1e-9:
        raise ValueError("Source mobility/JVP decomposition disagrees")
    construction = {"available": False, "reason": "disabled by protocol"}
    if settings["constructed_task"]:
        beta, construction = maximum_competition_task(base["projection"], settings["numeric_floor"])
        if beta is not None:
            names = names + ["source_maximum_competition"]
            families = families + ["constructed_max_competition"]
            tasks = torch.cat((tasks, beta[None]))
    errors = [effective_gradient(model, draw_batch(data, settings["batch_size"], error_rng))
              for _ in range(settings["error_batches"])]
    calibrated = {}
    for ratio in settings["ratios"]:
        source = geometry(model, gradient, ratio, settings["lr"])
        noise = {rep: {"geometry": [], "normal": [], "task": []} for rep in source}
        for g in errors:
            noisy = geometry(model, g, ratio, settings["lr"])
            for rep in source:
                difference = noisy[rep]["D"]-source[rep]["D"]
                noise[rep]["geometry"].append(float(difference.norm()))
                noise[rep]["task"].append(risk_slopes(difference, tasks).abs())
                noise[rep]["normal"].append(float((noisy[rep]["normal_front"]+
                    noisy[rep]["normal_head"]-source[rep]["normal_front"]-source[rep]["normal_head"]).norm()))
        # Separate short calibration streams explore source-local state drift.
        # At displaced states use the SAME frozen fit objective to remove fresh
        # batch noise from this drift estimate. No future stream is inspected.
        drift = {rep: {"geometry": [], "normal": [], "task": []} for rep in source}
        euler = {rep: {"geometry": [], "normal": [], "task": []} for rep in source}
        for repeat in range(settings["probe_streams"]):
            branch = copy.deepcopy(model)
            probe_rng = generator_for(settings, seed, step, 3, repeat)
            max_drift = {rep: {"geometry": 0., "normal": 0., "task": torch.zeros(len(tasks))}
                         for rep in source}
            max_euler = {rep: {"geometry": 0., "normal": 0., "task": torch.zeros(len(tasks))}
                         for rep in source}
            for j in range(settings["probe_steps"]+1):
                here = geometry(branch, mean_gradient(branch, fit_batches), ratio, settings["lr"])
                batch = draw_batch(data, settings["batch_size"], probe_rng)
                noisy = geometry(branch, effective_gradient(branch, batch), ratio, settings["lr"])
                remainder = finite_remainder(branch, batch, noisy, tasks, ratio, settings)
                for rep in source:
                    if j:
                        difference = here[rep]["D"]-source[rep]["D"]
                        max_drift[rep]["geometry"] = max(max_drift[rep]["geometry"], float(difference.norm())/j)
                        max_drift[rep]["task"] = torch.maximum(max_drift[rep]["task"], risk_slopes(difference, tasks).abs()/j)
                        # Encoder B updates use a fixed SOURCE normal projector.
                        source_q = torch.eye(tasks.shape[1])-source[rep]["P"]
                        normal = (here[rep]["velocity_front"]+here[rep]["velocity_head"])@source_q
                        max_drift[rep]["normal"] = max(max_drift[rep]["normal"],
                            float((normal-source[rep]["normal_front"]-source[rep]["normal_head"]).norm())/j)
                    for key in ("geometry", "normal"):
                        max_euler[rep][key] = max(max_euler[rep][key], remainder[rep][key])
                    max_euler[rep]["task"] = torch.maximum(max_euler[rep]["task"], remainder[rep]["task"])
                if j < settings["probe_steps"]:
                    update(branch, *batch, settings["lr"], ratio)
            for rep in source:
                for key in drift[rep]:
                    drift[rep][key].append(max_drift[rep][key])
                    euler[rep][key].append(max_euler[rep][key])
        calibrated[str(ratio)] = {"mode": model.mode, "geometry": source, "coefficients": {
            rep: {component: {key: quantile(torch.stack(values) if isinstance(values[0], torch.Tensor) else values, settings)
                             for key, values in collection[rep].items()}
                  for component, collection in (("noise", noise), ("drift", drift), ("euler", euler))}
            for rep in source}}
    return names, families, tasks, calibrated, construction, source_check


def make_predictions(calibrated, tasks, names, families, settings):
    records = []
    for ratio, entry in calibrated.items():
        for rep, source in entry["geometry"].items():
            exact_invariance = entry["mode"] == "frozen_front" and rep == "encoder"
            coeff = entry["coefficients"][rep]
            slopes = risk_slopes(source["D"], tasks)
            risks = tasks.square().sum(1)-torch.einsum("ti,ij,tj->t", tasks, source["P"], tasks)
            retained = (tasks@source["P"]).norm(dim=1)
            residual = (tasks-tasks@source["P"]).norm(dim=1)
            cross = retained*residual
            task_norm2 = tasks.square().sum(1)
            for horizon in settings["horizons"]:
                # Calibrated Lipschitz-per-step surrogate; no iid/sqrt(H) claim.
                remainder = (horizon*coeff["noise"]["task"] + .5*horizon*horizon*coeff["drift"]["task"]
                             + horizon*coeff["euler"]["task"])
                p_error = float(horizon*coeff["noise"]["geometry"] + .5*horizon*horizon*coeff["drift"]["geometry"]
                                + horizon*coeff["euler"]["geometry"])
                distance = min(math.sqrt(source["rank"]), (horizon*float(source["D"].norm())+p_error)/math.sqrt(2))
                supported = horizon <= settings["probe_steps"]
                for i, (name, family) in enumerate(zip(names, families)):
                    raw_center = horizon*float(slopes[i])
                    radius = float(remainder[i])
                    # Do not clip this interval into the feasible risk range:
                    # a disjoint/overshooting tangent interval is itself a failure.
                    lower, upper = raw_center-radius, raw_center+radius
                    direction = 1 if lower > settings["numeric_floor"] else -1 if upper < -settings["numeric_floor"] else 0
                    usable = bool(direction and (supported or settings["allow_window_extrapolation"]))
                    uniform_bound = float(task_norm2[i])*min(1., distance)
                    cross_bound = 2*float(cross[i])*distance+float(task_norm2[i])*distance*distance
                    reasons = []
                    if not direction:
                        reasons.append("interval_contains_zero_or_numeric_floor")
                    if not supported and not settings["allow_window_extrapolation"]:
                        reasons.append("outside_calibration_probe_window")
                    if upper < -float(risks[i]) or lower > float(task_norm2[i]-risks[i]):
                        reasons.append("interval_disjoint_feasible_risk_range")
                        usable = False
                    records.append({"ratio": float(ratio), "representation": rep, "horizon": horizon,
                        "task": name, "task_family": family, "source_risk": float(risks[i]),
                        "source_task_cross_block": float(cross[i]), "source_slope": float(slopes[i]),
                        "predicted_change": raw_center, "empirical_lower": lower, "empirical_upper": upper,
                        "interval_width": 2*radius, "empirical_direction": direction,
                        "usable_empirical_window": usable, "no_window_reasons": reasons,
                        "strict_certificate_available": exact_invariance,
                        "strict_certificate_kind": "exact_frozen_encoder_invariance" if exact_invariance else None,
                        "strict_risk_change_interval": [0., 0.] if exact_invariance else None,
                        "strict_certificate_reason": None if exact_invariance else "calibration quantiles are not uniform future-SGD error bounds",
                        "within_calibration_horizon": supported,
                        "forecast_distance_upper_empirical": distance,
                        "forecast_displacement_only_bound": uniform_bound,
                        "forecast_task_cross_bound_raw": cross_bound,
                        "forecast_task_cross_bound": min(uniform_bound, cross_bound),
                        "forecast_projector_error_radius": p_error,
                        "predicted_small_change": min(uniform_bound, cross_bound) < settings["practical_change"],
                        "predicted_large_directional_change": usable and min(abs(lower), abs(upper)) > settings["practical_change"]})
    return records


def serialize_source(calibrated):
    return {ratio: {rep: {key: value.tolist() if isinstance(value, torch.Tensor) else value
                         for key, value in source.items()}
                    for rep, source in entry["geometry"].items()} for ratio, entry in calibrated.items()}


def serialize_coefficients(calibrated):
    return {ratio: {rep: {component: {key: value.tolist() for key, value in values.items()}
                         for component, values in components.items()}
                    for rep, components in entry["coefficients"].items()}
            for ratio, entry in calibrated.items()}


def evaluate_future(model, data, calibrated, tasks, predictions, seed, step, settings):
    lookup = {(r["ratio"], r["representation"], r["horizon"], r["task"]): r for r in predictions}
    outcomes, geometry_rows = [], []
    for repeat in range(settings["future_streams"]):
        rng = generator_for(settings, seed, step, 4, repeat)
        branches = {ratio: copy.deepcopy(model) for ratio in settings["ratios"]}
        path_lengths = {ratio: [0., 0.] for ratio in branches}
        previous = {(ratio, rep): torch.zeros(len(tasks)) for ratio in branches for rep in ("encoder", "projection")}
        first_opposed = {key: torch.zeros(len(tasks), dtype=torch.int64) for key in previous}
        first_net_opposed = {key: torch.zeros(len(tasks), dtype=torch.int64) for key in previous}
        for horizon in range(1, max(settings["horizons"])+1):
            batch = draw_batch(data, settings["batch_size"], rng)
            for ratio, branch in branches.items():
                old_front, old_head = branch.front.detach().clone(), branch.weight.detach().clone()
                future_tangent = None
                if horizon in settings["horizons"]:
                    try:
                        future_tangent = geometry(branch, effective_gradient(branch, batch), ratio, settings["lr"])
                    except ValueError:
                        pass  # Rank loss is recorded below, never omitted.
                update(branch, *batch, settings["lr"], ratio)
                path_lengths[ratio][0] += float((branch.front.detach()-old_front).norm())
                path_lengths[ratio][1] += float((branch.weight.detach()-old_head).norm())
                with torch.no_grad():
                    a, b = branch.effective()
                    for rep, x in (("encoder", b), ("projection", a)):
                        source = calibrated[str(ratio)]["geometry"][rep]
                        p, rank, singular = row_space(x)
                        delta = p-source["P"]
                        same_rank = rank == source["rank"]
                        changes = risk_slopes(delta, tasks)
                        source_slopes = risk_slopes(source["D"], tasks)
                        source_signs = torch.where(source_slopes.abs() > settings["numeric_floor"], source_slopes.sign(), 0.)
                        key = (ratio, rep)
                        increments = changes-previous[key]
                        opposed = increments*source_signs < -settings["numeric_floor"]
                        net_opposed = changes*source_signs < -settings["numeric_floor"]
                        first_opposed[key][opposed & (first_opposed[key] == 0)] = horizon
                        first_net_opposed[key][net_opposed & (first_net_opposed[key] == 0)] = horizon
                        previous[key] = changes.clone()
                        if horizon not in settings["horizons"]:
                            continue
                        q = torch.eye(x.shape[1])-source["P"]
                        d = float(delta.norm())/math.sqrt(2)
                        normal_observed = (x-source["X"])@q
                        normal_predicted = horizon*(source["normal_front"]+source["normal_head"])
                        tangent_prediction = horizon*source["D"]
                        coeff = calibrated[str(ratio)]["coefficients"][rep]
                        geometry_radius = float(horizon*coeff["noise"]["geometry"] +
                            .5*horizon*horizon*coeff["drift"]["geometry"] + horizon*coeff["euler"]["geometry"])
                        normal_radius = float(horizon*coeff["noise"]["normal"] +
                            .5*horizon*horizon*coeff["drift"]["normal"] + horizon*coeff["euler"]["normal"])
                        geometry_rows.append({"repeat": repeat, "ratio": ratio, "horizon": horizon,
                            "representation": rep, "same_rank": same_rank, "rank": rank,
                            "min_singular_value": float(singular.min()),
                            "observed_projector_displacement": float(delta.norm()),
                            "predicted_projector_displacement": float(tangent_prediction.norm()),
                            "projector_forecast_error": float((delta-tangent_prediction).norm()),
                            "projector_direction_cosine": cosine(delta, tangent_prediction),
                            "normal_forecast_error": float((normal_observed-normal_predicted).norm()),
                            "normal_observed_norm": float(normal_observed.norm()),
                            "normal_predicted_norm": float(normal_predicted.norm()),
                            "normal_direction_cosine": cosine(normal_observed, normal_predicted),
                            "front_parameter_path_length": path_lengths[ratio][0],
                            "head_parameter_path_length": path_lengths[ratio][1],
                            "total_parameter_block_path_length": sum(path_lengths[ratio]),
                            "projector_empirical_error_radius": geometry_radius,
                            "normal_empirical_error_radius": normal_radius,
                            "projector_empirical_radius_hit": bool(same_rank and float((delta-tangent_prediction).norm()) <= geometry_radius),
                            "normal_empirical_radius_hit": bool(same_rank and float((normal_observed-normal_predicted).norm()) <= normal_radius),
                            "future_front_tangent_error": float((future_tangent[rep]["D_front"]-source["D_front"]).norm()) if future_tangent else None,
                            "future_head_tangent_error": float((future_tangent[rep]["D_head"]-source["D_head"]).norm()) if future_tangent else None,
                            "future_front_tangent_cosine": cosine(future_tangent[rep]["D_front"], source["D_front"]) if future_tangent else None,
                            "future_head_tangent_cosine": cosine(future_tangent[rep]["D_head"], source["D_head"]) if future_tangent else None,
                            "future_tangents_scope": "diagnostic from held-out batch before this update; never used to revise source predictions"})
                        # Exact finite leakage terms, recorded ONLY as retrospective controls.
                        lost = source["P"]@(torch.eye(x.shape[1])-p)@source["P"]
                        gained = q@p@q
                        offdiagonal = q@p@source["P"]
                        for i, name in enumerate([r["task"] for r in predictions
                                                  if r["ratio"] == ratio and r["representation"] == rep
                                                  and r["horizon"] == settings["horizons"][0]]):
                            prediction = lookup[(ratio, rep, horizon, name)]
                            observed = float(changes[i])
                            norm2 = float(tasks[i].square().sum())
                            posterior_uniform = norm2*min(1., d) if same_rank else None
                            posterior_cross = (min(posterior_uniform,
                                2*prediction["source_task_cross_block"]*d+norm2*d*d) if same_rank else None)
                            cross_term = -2*float(tasks[i]@offdiagonal@tasks[i])
                            diagonal_term = float(tasks[i]@(lost-gained)@tasks[i])
                            covered = prediction["empirical_lower"] <= observed <= prediction["empirical_upper"]
                            actual_sign = 1 if observed > settings["numeric_floor"] else -1 if observed < -settings["numeric_floor"] else 0
                            outcomes.append({**prediction, "repeat": repeat, "observed_change": observed,
                                "same_rank": same_rank, "absolute_prediction_error": abs(observed-prediction["predicted_change"]),
                                "empirical_interval_hit": bool(same_rank and covered),
                                "raw_empirical_interval_hit": covered, "observed_sign": actual_sign,
                                "exact_invariance_check_passed": abs(observed) <= 1e-12 if prediction["strict_certificate_available"] else None,
                                "usable_direction_correct": bool(same_rank and prediction["usable_empirical_window"]
                                                                   and actual_sign == prediction["empirical_direction"]),
                                "source_sign_known": bool(source_signs[i] != 0),
                                "first_increment_opposition": int(first_opposed[key][i]) or None,
                                "first_net_opposition": int(first_net_opposed[key][i]) or None,
                                "increment_direction_persisted": bool(source_signs[i] != 0 and first_opposed[key][i] == 0),
                                "net_direction_persisted": bool(source_signs[i] != 0 and first_net_opposed[key][i] == 0),
                                "observed_small_change": abs(observed) < settings["practical_change"],
                                "observed_distance": d,
                                "forecast_displacement_bound_hit": same_rank and abs(observed) <= prediction["forecast_displacement_only_bound"]+1e-12,
                                "forecast_task_bound_hit": same_rank and abs(observed) <= prediction["forecast_task_cross_bound"]+1e-12,
                                "posterior_displacement_bound": posterior_uniform,
                                "posterior_task_cross_bound": posterior_cross,
                                "posterior_task_bound_holds": same_rank and abs(observed) <= posterior_cross+1e-10,
                                "posterior_cross_term": cross_term, "posterior_diagonal_term": diagonal_term,
                                "leakage_identity_residual": observed-cross_term-diagonal_term})
    return outcomes, geometry_rows


def cosine(first, second):
    denominator = float(first.norm()*second.norm())
    return float((first*second).sum())/denominator if denominator > 1e-30 else None


def run_state(source_root, output, seed, mode, step, phase, settings):
    checkpoint = source_root/f"seed_{seed}_{mode}_r1_stream0"/f"checkpoint_{step}.pt"
    metadata = {"seed": seed, "mode": mode, "checkpoint_step": step, "phase": phase,
                "checkpoint": str(checkpoint.resolve())}
    directory = output/f"seed_{seed}_{mode}_step{step}"
    directory.mkdir()
    if not checkpoint.is_file():
        write_json(directory/"status.json", {**metadata, "status": "missing_checkpoint"})
        return
    try:
        model, saved = restore(checkpoint)
        if (saved["seed"], saved["mode"], saved["step"], saved["ratio"]) != (seed, mode, step, 1.):
            raise ValueError("Checkpoint experimental parameters do not match requested state")
        if not all(torch.isfinite(p).all() for p in (model.front, model.weight)):
            raise FloatingPointError("Nonfinite checkpoint parameters")
        if not math.isfinite(model.mobility) or model.mobility <= 0:
            raise ValueError("Checkpoint mobility must be finite and positive")
        names, tasks = texture_tasks(torch.float64, "cpu")
        families = ["predefined_coordinate"]*2+["predefined_band"]*8
        data = training_data(seed)
        names, families, tasks, calibration, constructed, check = calibrate(
            model, data, seed, step, names, families, tasks, settings)
        predictions = make_predictions(calibration, tasks, names, families, settings)
        write_json(directory/"predictions.json", {**metadata, "settings": settings,
            "task_names": names, "task_families": families, "tasks": tasks.tolist(),
            "constructed_task": constructed, "source_jvp_check": check,
            "geometry_source": serialize_source(calibration),
            "empirical_coefficients": serialize_coefficients(calibration),
            "predictions": predictions,
            "scope": "source-only empirical SGD forecast, saved before future draws; no uniform probability claim",
            "stochastic_direction_certificate_available": False,
            "strict_scope": "only exact frozen-encoder invariance; all fitted stochastic intervals are empirical"})
        # Future RNGs are created only after this source prediction file exists.
        outcomes, geometric = evaluate_future(model, data, calibration, tasks, predictions, seed, step, settings)
        write_json(directory/"outcomes.json", {"outcomes": outcomes, "geometry": geometric})
        write_json(directory/"status.json", {**metadata, "status": "complete"})
    except (ValueError, RuntimeError, FloatingPointError, OSError, KeyError) as error:
        # Retain failed states in the denominator, with the reason and any
        # already-written predictions. Never relabel a failure as zero change.
        write_json(directory/"status.json", {**metadata, "status": "failed",
                                           "reason": f"{type(error).__name__}: {error}"})
    print(f"seed={seed} {mode} step={step}: " + json.loads((directory/"status.json").read_text())["status"], flush=True)


def validate_settings(settings):
    if set(settings) != set(DEFAULTS):
        raise ValueError("Settings must contain exactly the documented prediction settings")
    for key in ("batch_size", "fit_batches", "error_batches", "probe_streams", "probe_steps", "future_streams"):
        if type(settings[key]) is not int or settings[key] < (2 if key in ("batch_size", "error_batches") else 1):
            raise ValueError(f"Invalid {key}")
    if not 0 < settings["quantile"] <= 1 or settings["inflation"] < 1:
        raise ValueError("Invalid calibration quantile/inflation")
    if not all(math.isfinite(settings[k]) for k in ("lr", "quantile", "inflation", "numeric_floor", "practical_change")):
        raise ValueError("Prediction settings must be finite")
    if settings["lr"] <= 0 or settings["numeric_floor"] <= 0 or settings["practical_change"] <= 0:
        raise ValueError("Rates and reporting thresholds must be positive")
    if not settings["horizons"] or any(type(h) is not int or h < 1 for h in settings["horizons"]):
        raise ValueError("Horizons must be positive integers")
    if settings["horizons"] != sorted(set(settings["horizons"])):
        raise ValueError("Horizons must be sorted and unique")
    if not settings["ratios"] or any(not math.isfinite(r) or r <= 0 for r in settings["ratios"]):
        raise ValueError("Ratios must be positive and finite")
    if len(set(settings["ratios"])) != len(settings["ratios"]):
        raise ValueError("Ratios must be unique")
    if type(settings["rng_base"]) is not int or any(type(settings[k]) is not bool for k in ("constructed_task", "allow_window_extrapolation")):
        raise ValueError("Invalid RNG seed or prediction flags")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=SEEDS, default="development")
    parser.add_argument("--settings", type=Path, help="Validation requires a development-selected JSON settings file")
    parser.add_argument("--source", type=Path, help="Override checkpoint directory; selection remains fixed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="List settings/checkpoint presence; no calibration/SGD")
    parser.add_argument("--self-check", action="store_true", help="Tiny synthetic algebra checks only")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if args.self_check:
        self_check()
        return
    if args.phase == "validation" and args.settings is None:
        parser.error("validation requires --settings selected on development states")
    settings = json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else copy.deepcopy(DEFAULTS)
    validate_settings(settings)
    source = args.source or result_path("full_sgd_development_v2" if args.phase == "development" else "full_sgd_main")
    output = args.output or RESULTS/"empirical"/f"task_motion_{args.phase}"
    planned = [{"seed": seed, "mode": mode, "step": step,
                "checkpoint_exists": (source/f"seed_{seed}_{mode}_r1_stream0"/f"checkpoint_{step}.pt").is_file()}
               for seed in SEEDS[args.phase] for mode in MODES for step in STEPS[args.phase]]
    protocol = {"phase": args.phase, "source": str(source.resolve()), "settings": settings,
                "settings_source": str(args.settings.resolve()) if args.settings else "fixed development defaults",
                "states": planned, "predefined_tasks": texture_tasks(torch.float64, "cpu")[0],
                "task_selection": "ten fixed texture tasks; optional source-only K=1 max-min competitor reported separately",
                "separation": "fit/error/probe/future have independent RNG namespaces; no outcome-based settings update",
                "uncertainty": "inflated empirical calibration quantiles, no simultaneous probability guarantee",
                "holdout_scope": "validation seeds/checkpoints excluded from tuning THIS protocol, not unexamined in all previous experiments",
                "strict_certificates": "only exact frozen-encoder invariance; no uniform future-SGD directional certificate",
                "inference_unit": "seed; checkpoints/tasks/streams are correlated repeated measurements"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"protocol.json", protocol)
    write_json(output/"prediction_settings.json", settings)
    for state in planned:
        run_state(source, output, state["seed"], state["mode"], state["step"], args.phase, settings)


def self_check():
    from square_images import SquareImages
    from subspace_mechanism import Factors

    rng = torch.Generator().manual_seed(17)
    base = SquareImages(8, [(1, 0), (0, 1), (1, 1)], [1., 1., 1.], 2,
                        kernel_size=3, backend="fft")
    with torch.no_grad():
        base.kernel.copy_(torch.randn(base.kernel.shape, generator=rng)*.2)
        base.weight.copy_(torch.tensor([[.3, -.7]]))
    checks = []
    for mode in MODES:
        model = Factors(base, mode)
        gradient = torch.tensor([[.2, -.1, .4]])
        check = check_source_jvp(model, gradient)
        assert check["max_absolute_error"] < 1e-12
        source = geometry(model, gradient, 1., .002)
        for rep in source.values():
            p, tangent = rep["P"], rep["D"]
            assert torch.allclose(tangent, tangent.T, atol=1e-12)
            assert float((p@tangent@p).norm()) < 1e-12
        checks.append({"mode": mode, **check})
    # Complete competition criterion in K=1: an actual competing task and a
    # positively proportional exclusion case, with an independently known margin.
    x = torch.tensor([[1., 0., 0.]])
    dh = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    dw = torch.tensor([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
    beta, info = maximum_competition_task({"X": x, "D_front": dh, "D_head": dw}, 1e-12)
    assert beta is not None and abs(info["maximum_margin"]-1/math.sqrt(2)) < 1e-12
    assert maximum_competition_task({"X": x, "D_front": dh, "D_head": 2*dh}, 1e-12)[0] is None
    # Exact task-cross inequality and non-micro projector differential.
    matrix = torch.randn(2, 5, generator=rng)
    velocity = torch.randn(2, 5, generator=rng)*.03
    p0 = row_space(matrix)[0]
    p1 = row_space(matrix+velocity)[0]
    q0 = torch.eye(5)-p0
    tasks = torch.randn(9, 5, generator=rng)
    tasks /= tasks.norm(dim=1, keepdim=True)
    observed = risk_slopes(p1-p0, tasks).abs()
    d = float((p1-p0).norm())/math.sqrt(2)
    cross = (tasks@p0).norm(dim=1)*(tasks@q0).norm(dim=1)
    assert bool((observed <= 2*cross*d+d*d+1e-12).all())
    epsilon = 1e-5
    finite = (row_space(matrix+epsilon*velocity)[0]-row_space(matrix-epsilon*velocity)[0])/(2*epsilon)
    differential_error = float((finite-projector_tangent(matrix, velocity)).abs().max())
    assert differential_error < 1e-8
    # Exercise calibration -> immutable forecasts -> independent tiny future
    # streams without loading saved checkpoints or generating training datasets.
    settings = copy.deepcopy(DEFAULTS)
    settings.update(batch_size=8, fit_batches=2, error_batches=2, probe_streams=1,
                    probe_steps=2, future_streams=2, horizons=[1, 2])
    validate_settings(settings)
    model = Factors(base, "square")
    original_parameters = [p.detach().clone() for p in (model.front, model.weight)]
    data = torch.randn(32, 3, 2, generator=rng)
    names, families, tasks, calibration, _, _ = calibrate(
        model, data, 0, 0, ["toy_0", "toy_1"], ["predefined_coordinate"]*2,
        torch.eye(3)[:2], settings)
    predictions = make_predictions(calibration, tasks, names, families, settings)
    frozen_predictions = json.dumps(predictions, allow_nan=False)
    outcomes, geometric = evaluate_future(model, data, calibration, tasks, predictions, 0, 0, settings)
    assert json.dumps(predictions, allow_nan=False) == frozen_predictions
    assert len(outcomes) == 2*3*2*len(tasks)*2
    assert len(geometric) == 2*3*2*2
    assert all(abs(row["leakage_identity_residual"]) < 1e-12 for row in outcomes)
    assert all(row["posterior_task_bound_holds"] for row in outcomes)
    assert not any(row["strict_certificate_available"] for row in outcomes)
    assert all(torch.equal(before, after) for before, after in zip(original_parameters, (model.front, model.weight)))
    json.dumps({"outcomes": outcomes, "geometry": geometric}, allow_nan=False)
    json.dumps(serialize_source(calibration), allow_nan=False)
    json.dumps(serialize_coefficients(calibration), allow_nan=False)
    frozen = Factors(base, "frozen_front")
    frozen_source = geometry(frozen, torch.tensor([[.2, -.1, .4]]), 1., .002)
    coefficients = {rep: {component: {"geometry": torch.tensor(0.), "normal": torch.tensor(0.),
                                     "task": torch.zeros(2)} for component in ("noise", "drift", "euler")}
                    for rep in frozen_source}
    control = make_predictions({"1.0": {"mode": "frozen_front", "geometry": frozen_source, "coefficients": coefficients}},
                               torch.eye(3)[:2], ["toy_0", "toy_1"], ["predefined_coordinate"]*2, settings)
    assert all(r["strict_certificate_available"] == (r["representation"] == "encoder") for r in control)
    from summarize_texture_motion import state_metrics, coverage_metrics, window_metrics, geometry_metrics
    metadata = {"seed": 0, "mode": "square", "checkpoint_step": 0}
    test_protocol = {"settings": settings, "states": [{"seed": 0, "mode": "square", "step": 0},
                                                         {"seed": 1, "mode": "square", "step": 0}]}
    statuses = {(0, "square", 0): {"status": "complete"}, (1, "square", 0): {"status": "missing_checkpoint"}}
    pp = [{**metadata, **r} for r in predictions]
    oo = [{**metadata, **r} for r in outcomes]
    gg = [{**metadata, **r} for r in geometric]
    summary_rows = state_metrics(test_protocol, pp, oo, statuses)
    summary_coverage = coverage_metrics(summary_rows)
    assert all(r["available_prediction_slots"] == 2 and r["planned_task_slots"] == 4
               for r in summary_coverage if r["task_family"] == "predefined_coordinate")
    assert all(r["available_prediction_slots"] == 0 and r["planned_task_slots"] == 16
               for r in summary_coverage if r["task_family"] == "predefined_band")
    assert all(r["source_empirical_prefix_window"] <= 2 for r in window_metrics(pp, oo, test_protocol))
    assert all(r["observed_streams"] == 0 for r in geometry_metrics(test_protocol, gg, statuses) if r["seed"] == 1)
    print(json.dumps({"source_checks": checks, "projector_difference_error": differential_error,
                      "competition_margin": info["maximum_margin"],
                      "toy_forecast_rows": len(predictions), "toy_future_rows": len(outcomes),
                      "scope": "tiny synthetic tensors only; no saved checkpoints or experimental future streams"}, indent=2))


if __name__ == "__main__":
    main()
