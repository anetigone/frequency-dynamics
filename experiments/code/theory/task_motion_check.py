"""Origin-only task-risk envelopes checked against frozen population profiles.

No trajectory is integrated here. Interval constants are computed from the
published rational initial state; future risks are only read for validation.
Run with the repository's configured Python interpreter from any directory.
"""

from __future__ import annotations

import csv
import json
import math
from fractions import Fraction as F
from pathlib import Path

from interval_core import I, J, S, Z, field, quantities

ROOT = Path(__file__).resolve().parents[2] / "results" / "theory"


def iv(value):
    value = F(value)
    return I.rat(value.numerator, value.denominator)


def initial(c):
    c = F(c)
    rho = F(2, 5)
    return [rho * F(1, 10) / (c*c), rho * F(9, 10) / (c*c)] + [
        rho*rho*c*c*b for b in (F(2), F(1, 5), F(1, 5), F(2, 5))
    ] + [F(1), F(2)]


def physical_field(x):
    w, v, b, c, d, e, g, j = x
    a, z, energy, _ = quantities(x)
    s = iv(F(36, 25)) if isinstance(energy, I) else J([iv(F(36, 25))] + [Z]*(len(energy.a)-1))
    q = s * energy
    gam = (q + 2) / ((q + 1)*(q + 1)*2)
    gain2 = (s*gam - (s - 1))*2
    zero = g*0
    return [(b*a+c*z)*gain2, (d*a+e*z)*gain2,
            4*g*b*w*a*gain2, 4*j*c*w*z*gain2,
            4*g*d*v*a*gain2, 4*j*e*v*z*gain2, zero, zero]


def physical_energy_rate(x):
    _, _, energy, h = quantities(x)
    s = iv(F(36, 25))
    q = s*energy
    gam = (q+2)/((q+1)*(q+1)*2)
    return (s*gam-(s-1))*4*h/energy


def jets(x, order, rhs):
    series = [[v] for v in x]
    for k in range(order):
        velocity = rhs([J(v) for v in series])
        for i in range(len(x)):
            series[i].append(velocity[i].a[k]/(k+1))
    return [J(v) for v in series]


def risk_jet(x, rhs, order=2):
    a, z, energy, _ = quantities(jets(x, order, rhs))
    return (z*z/energy).a


def describe(value):
    return {"lower": value.lo / S, "upper": value.hi / S}


def origin_bound(c, relative_radius, clock):
    """Forward invariant box + interval second derivative, all from t=0."""
    values = initial(c)
    x0 = [iv(v) for v in values]
    # Lower endpoints are exact initial lower endpoints; all active derivatives
    # are positive, so no strict lower-face clearance is needed.
    domain = [I(iv(v).lo, iv(v*(1+relative_radius)).hi) for v in values[:6]] + x0[6:]
    rhs = field if clock == "log_energy" else physical_field
    velocity = rhs(domain)
    assert all(v.lo > 0 for v in velocity[:6])
    # A 10% slack makes the upper-face enclosure strict.
    lifetimes = [F(domain[i].hi-x0[i].hi, velocity[i].hi) for i in range(6)]
    exit_budget = min(lifetimes)*F(9, 10)
    startjet = risk_jet(x0, rhs)
    curvature = risk_jet(domain, rhs)[2]*2
    m = min(abs(startjet[1].lo), abs(startjet[1].hi)) / S
    assert startjet[1].lo*startjet[1].hi > 0
    curvature_abs = max(abs(curvature.lo), abs(curvature.hi))/S
    duration = min(float(exit_budget), m/(2*curvature_abs))
    # Quantize downward to a rational 1e-12 budget before the proof check.
    duration_q = F(math.floor(duration*10**12), 10**12)
    duration_i = iv(duration_q)
    for i in range(6):
        assert (x0[i] + duration_i*velocity[i]).hi < domain[i].hi
    signed = 1 if startjet[1].lo > 0 else -1
    drift = startjet[1] + I(0, duration_i.hi)*curvature
    assert drift.lo > 0 if signed == 1 else drift.hi < 0
    margin_integer = min(abs(startjet[1].lo), abs(startjet[1].hi))
    assert drift.lo*2 >= margin_integer if signed == 1 else -drift.hi*2 >= margin_integer
    envelope = startjet[1]*duration_i + curvature*duration_i*duration_i/2
    return {"clock": clock, "relative_box_radius": str(relative_radius),
            "box_exit_budget": float(exit_budget),
            "initial_derivative": describe(startjet[1]),
            "second_derivative_enclosure": describe(curvature),
            "certified_duration": float(duration_q),
            "derivative_enclosure_to_duration": describe(drift),
            "risk_change_enclosure_at_duration": describe(envelope),
            "direction": signed,
            "minimum_absolute_derivative_margin": m,
            "curvature_absolute_bound": curvature_abs}


def choose(c, clock):
    # Declared in advance; selection sees interval constants, never future CSV.
    candidates = [origin_bound(c, r, clock) for r in (F(1,100), F(1,20), F(1,10), F(1,5))]
    best = max(candidates, key=lambda x: x["certified_duration"])
    return {**best, "candidate_origin_only_constants": candidates}


def validate(c, bound, rows):
    clock = bound["clock"]
    key = "log_energy_progress" if clock == "log_energy" else "physical_time"
    future = [r for r in rows if r["mode"] == "joint" and F(r["c"]) == F(c) and 0 < float(r[key]) <= bound["certified_duration"]]
    d0 = bound["initial_derivative"]
    d2 = bound["second_derivative_enclosure"]
    records = []
    for row in future:
        budget = float(row[key])
        low = d0["lower"]*budget + .5*d2["lower"]*budget*budget
        high = d0["upper"]*budget + .5*d2["upper"]*budget*budget
        change = float(row["risk_unregularized"])-.5
        # CSVs are frozen floating-point RK4 evidence, not interval truth.
        records.append({"budget": budget, "log_energy_progress": float(row["log_energy_progress"]),
                        "physical_time": float(row["physical_time"]),
                        "predicted_risk_change_lower": low,
                        "predicted_risk_change_upper": high,
                        "observed_risk_change": change,
                        "observed_minus_lower": change-low,
                        "upper_minus_observed": high-change,
                        "inside_envelope_with_1e-9_numeric_tolerance": low-1e-9 <= change <= high+1e-9})
    return {"future_samples_in_window": len(records),
            "validation_status": "checked_saved_future" if records else "no_saved_sample_inside_window",
            "all_future_samples_inside": all(r["inside_envelope_with_1e-9_numeric_tolerance"] for r in records),
            "future_samples": records}


def common_rate_budget(predictions, budget=F(1, 10)):
    """Scale equivalence gives a common origin and eta_h+eta_W=1.

    Budget is elapsed GF time under that convention, not parameter path length.
    The interval box and curvature remain origin-only quantities.
    """
    records = []
    for p in predictions:
        c = F(p["c"])
        total = 1/(c*c) + c*c*c*c
        physical = budget/total
        bound = p["physical_time"]
        assert physical <= F(str(bound["certified_duration"]))
        x0 = [iv(v) for v in initial(c)]
        radius = F(bound["relative_box_radius"])
        domain = [I(iv(v).lo, iv(v*(1+radius)).hi) for v in initial(c)[:6]] + x0[6:]
        d0 = risk_jet(x0, physical_field)[1]
        d2 = risk_jet(domain, physical_field)[2]*2
        step = iv(physical)
        change = d0*step + d2*step*step/2
        records.append({"c": str(c), "normalized_encoder_rate": str((1/(c*c))/total),
                        "normalized_head_rate": str((c*c*c*c)/total),
                        "budget": str(budget), "equivalent_scaled_physical_time": float(physical),
                        "risk_change_enclosure": describe(change)})
    # Strict finite-amplitude split under the identical sum-rate convention.
    assert records[0]["risk_change_enclosure"]["lower"] > 0
    assert all(r["risk_change_enclosure"]["upper"] < 0 for r in records[1:])
    return {"definition": "GF elapsed time after mapping all branches to the same unscaled initial parameters and normalizing eta_h+eta_W=1; not path length or gradient norm budget",
            "records": records}


def energy_section_from_physical_box(c, bound, section=F(1, 50)):
    x0 = [iv(v) for v in initial(c)]
    radius = F(bound["relative_box_radius"])
    domain = [I(iv(v).lo, iv(v*(1+radius)).hi) for v in initial(c)[:6]] + x0[6:]
    rate = physical_energy_rate(domain)
    assert rate.lo > 0
    duration = iv(F(str(bound["certified_duration"])))
    lower_progress = (rate*duration).lo
    hit = iv(section)/rate
    reached = hit.hi <= duration.lo
    record = {"physical_energy_rate_enclosure": describe(rate),
              "guaranteed_log_energy_at_end_lower": lower_progress/S,
              "section": str(section), "section_reached_in_tube": reached,
              "first_hitting_time_enclosure": describe(hit)}
    if reached:
        d0 = risk_jet(x0, physical_field)[1]
        d2 = risk_jet(domain, physical_field)[2]*2
        record["risk_change_at_first_section_enclosure"] = describe(d0*hit + d2*hit*hit/2)
    return record


def main():
    # Compute and freeze every prediction before opening the saved trajectories.
    predictions = [{"c": str(c), "initial_state_w_B_g": [str(v) for v in initial(c)],
                    "log_energy": choose(c, "log_energy"),
                    "physical_time": choose(c, "physical_time")}
                   for c in (F(1,2), F(1), F(3,2))]
    common_budget = common_rate_budget(predictions)
    for p in predictions:
        p["energy_section_from_physical_tube"] = energy_section_from_physical_box(F(p["c"]), p["physical_time"])
    with (ROOT/"mechanism_profiles.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for p in predictions:
        for clock in ("log_energy", "physical_time"):
            p[clock]["validation"] = validate(F(p["c"]), p[clock], rows)
        section_prediction = p["energy_section_from_physical_tube"]
        matching = [r for r in rows if r["mode"] == "joint" and F(r["c"]) == F(p["c"]) and F(r["log_energy_progress"]) == F(section_prediction["section"])]
        assert len(matching) == 1
        observed_time = float(matching[0]["physical_time"])
        observed_change = float(matching[0]["risk_unregularized"])-.5
        predicted_time = section_prediction["first_hitting_time_enclosure"]
        observed = {"physical_time": observed_time, "risk_change": observed_change,
                    "hitting_time_inside_prediction": predicted_time["lower"]-1e-9 <= observed_time <= predicted_time["upper"]+1e-9}
        if section_prediction["section_reached_in_tube"]:
            change = section_prediction["risk_change_at_first_section_enclosure"]
            observed["observed_minus_lower"] = observed_change-change["lower"]
            observed["upper_minus_observed"] = change["upper"]-observed_change
            observed["risk_change_inside_prediction"] = change["lower"]-1e-9 <= observed_change <= change["upper"]+1e-9
            assert observed["risk_change_inside_prediction"] and observed["hitting_time_inside_prediction"]
        section_prediction["saved_future_validation"] = observed
    result = {"scope": "Origin-only interval envelopes; saved population curves for validation; no training or ODE integration",
              "validation_input": (ROOT/"mechanism_profiles.csv").as_posix(),
              "state_family": "Matched-function scales of B=[[2,.2],[.2,.4]], w=(.1,.9), rho=.4, g=(1,2); c=1 is the primary certificate center",
              "objective": "Isotropic GUPA, relative augmentation noise .2; parameter rates both one",
              "radii_declared_before_validation": ["1/100", "1/20", "1/10", "1/5"],
              "duration_rule": "min(0.9 box-exit budget, initial derivative margin/(2 curvature bound)); choose largest duration without reading future curves",
              "prediction_validation_separation": "All bounds are computed before opening mechanism_profiles.csv; no future risk or derivative sets constants. This is retrospective checking of an origin-only forward certificate, not a newly preregistered experiment",
              "common_normalized_rate_budget": common_budget,
              "limitations": ["Bounds concern exact rational initial states and population gradient flow, not finite-sample OLS or SGD",
                              "The log-energy clock is a function-energy budget, not equal physical time or equal parameter-update length",
                              "Physical time bounds are reported independently; saved curves are floating-point numerical validation",
                              "No natural-image explanatory claim follows from these controlled windows"],
              "predictions": predictions}
    assert all(p[k]["validation"]["all_future_samples_inside"] for p in predictions for k in ("log_energy", "physical_time"))
    (ROOT/"task_motion_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    flat = []
    for p in predictions:
        for clock in ("log_energy", "physical_time"):
            for r in p[clock]["validation"]["future_samples"]:
                flat.append({"source": result["validation_input"], "stage": "fixed_clock", "c": p["c"],
                             "clock": clock, "budget": r["budget"], "physical_time": r["physical_time"],
                             "log_energy_progress": r["log_energy_progress"],
                             "predicted_lower": r["predicted_risk_change_lower"],
                             "predicted_upper": r["predicted_risk_change_upper"],
                             "observed_change": r["observed_risk_change"],
                             "observed_minus_lower": r["observed_minus_lower"],
                             "upper_minus_observed": r["upper_minus_observed"]})
        sec = p["energy_section_from_physical_tube"]
        if sec["section_reached_in_tube"]:
            observed = sec["saved_future_validation"]
            flat.append({"source": result["validation_input"], "stage": "first_energy_section", "c": p["c"],
                         "clock": "log_energy", "budget": float(F(sec["section"])),
                         "physical_time": observed["physical_time"], "log_energy_progress": float(F(sec["section"])),
                         "predicted_lower": sec["risk_change_at_first_section_enclosure"]["lower"],
                         "predicted_upper": sec["risk_change_at_first_section_enclosure"]["upper"],
                         "observed_change": observed["risk_change"],
                         "observed_minus_lower": observed["observed_minus_lower"],
                         "upper_minus_observed": observed["upper_minus_observed"]})
    with (ROOT/"task_motion_check.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"common_normalized_rate_budget": common_budget,
                      "prediction_summary": [{"c": p["c"], **{clock: {k: p[clock][k] for k in ("certified_duration", "relative_box_radius", "initial_derivative", "risk_change_enclosure_at_duration", "validation")} for clock in ("log_energy", "physical_time")}} for p in predictions]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
