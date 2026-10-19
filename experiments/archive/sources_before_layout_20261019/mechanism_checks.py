"""Mechanism revision, 2026-10-19. Standard library; population computations.

RK4 integrates the actual factor parameter vector using log(clean energy) as
the independent variable. Physical time is integrated as an extra coordinate.
Independent full-batch GD checks use physical parameter updates, not that ODE.
"""
import csv
import json
import math
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def gamma(q):
    return (q + 2) / (2 * (q + 1) ** 2)


def gu(q):
    return -0.5 * math.log(q + 1) - 0.5 * q / (q + 1)


def observables(y, g, s, d, mode="joint"):
    channels = len(g)  # This implementation fixes C=d=2.
    w = y[:channels]
    h = [y[2:4], y[4:6]]
    b = [[g[k] * h[c][k] ** 2 for k in range(2)] for c in range(2)]
    a = [sum(w[c] * b[c][k] for c in range(2)) for k in range(2)]
    v = sum(x * x for x in a)
    q = sum(s[k] * a[k] ** 2 for k in range(2))
    r = [s[k] * gamma(q) - 0.5 * d[k] for k in range(2)]
    fw = [2 * sum(b[c][k] * r[k] * a[k] for k in range(2)) for c in range(2)]
    if mode == "frozen":
        fw = [0, 0]
    fh = [[4 * g[k] * h[c][k] * w[c] * r[k] * a[k]
           for k in range(2)] for c in range(2)]
    field = fw + fh[0] + fh[1]
    da = [sum(fw[c] * b[c][k] + w[c] * 2 * g[k] * h[c][k] * fh[c][k]
              for c in range(2)) for k in range(2)]
    dlogv = 2 * sum(a[k] * da[k] for k in range(2)) / v
    drisk = (2 * a[1] * da[1] * v - a[1] ** 2 * v * dlogv) / v ** 2
    return {"a": a, "clean_energy": v, "view_energy": q,
            "loss": 0.5 * sum(d[k] * a[k] ** 2 for k in range(2)) + gu(q),
            "risk_unregularized": a[1] ** 2 / v,
            "risk_ridge_005": 1 - a[0] ** 2 * (v + 0.1) / (v + 0.05) ** 2,
            "field": field, "dlogv_dt": dlogv,
            "drisk_dlogv": drisk / dlogv,
            "r": r}


def init(c=1, rho=1, g=(1, 2)):
    b = [[2, 0.2], [0.2, 0.4]]
    return [rho * 0.1 / c ** 2, rho * 0.9 / c ** 2] + [
        rho * c * math.sqrt(b[j][k] / g[k]) for j in range(2) for k in range(2)]


def snapshot(y, g, s, d, mode="joint"):
    row = observables(y, g, s, d, mode)
    del row["field"]
    row["physical_time"] = y[6] if len(y) > 6 else 0
    return row


def normalized_field(y, g, s, d, mode="joint"):
    row = observables(y, g, s, d, mode)
    rate = row["dlogv_dt"]
    assert rate > 0, "Energy parameterization invalid: nonpositive energy rate"
    return [x / rate for x in row["field"]] + [1 / rate]


def rk4(y, dt, field):
    k1 = field(y)
    k2 = field([a + dt * b / 2 for a, b in zip(y, k1)])
    k3 = field([a + dt * b / 2 for a, b in zip(y, k2)])
    k4 = field([a + dt * b for a, b in zip(y, k3)])
    return [a + dt * (b + 2 * c + 2 * d + e) / 6
            for a, b, c, d, e in zip(y, k1, k2, k3, k4)]


def profile(c, rho, noise=0, delta=0.002, max_progress=6, mode="joint"):
    g = [1, 2]
    ss = (1 + noise) ** 2
    s, d = [ss, ss], [2 * (ss - 1)] * 2
    y = init(c, rho, g) + [0]
    v0 = observables(y, g, s, d)["clean_energy"]
    full, samples = [], []
    minrow = None
    count = round(max_progress / delta)
    maxenergy_error = 0
    for i in range(count + 1):
        row = snapshot(y, g, s, d, mode)
        row["log_energy_progress"] = i * delta
        maxenergy_error = max(maxenergy_error, abs(math.log(row["clean_energy"] / v0) - i * delta))
        full.append(row)
        if minrow is None or row["risk_unregularized"] < minrow["risk_unregularized"]:
            minrow = row
        if any(abs(i * delta - x) < delta / 4 for x in [0, .3, 1, 1.7, 3, 6]):
            samples.append(row)
        if i < count:
            y = rk4(y, delta, lambda z: normalized_field(z, g, s, d, mode))
    return {"mode": mode, "c": c, "rho": rho, "relative_noise": noise,
            "delta_log_energy": delta, "samples": samples,
            "minimum": minrow, "final": full[-1],
            "max_log_energy_error": maxenergy_error}, full


def actual_gd(eta):
    g, s, d = [1, 2], [1.44, 1.44], [0.88, 0.88]
    y = init(1, 0.4, g)
    start = snapshot(y, g, s, d)
    v0 = start["clean_energy"]
    prev = start["loss"]
    minimum, final = start, None
    maxincrease = 0
    captures = []
    targets = [0.3, 1, 1.7, 3, 6]
    for i in range(1, 500001):
        before = observables(y, g, s, d)
        y = [x + eta * v for x, v in zip(y, before["field"])]
        now = snapshot(y, g, s, d)
        now["physical_time"] = i * eta
        maxincrease = max(maxincrease, now["loss"] - prev)
        prev = now["loss"]
        if now["risk_unregularized"] < minimum["risk_unregularized"]:
            minimum = now
        progress = math.log(now["clean_energy"] / v0)
        if targets and progress >= targets[0]:
            captures.append({"requested_log_progress": targets.pop(0),
                             "step": i, "physical_time": i * eta, **now})
        if progress >= 6:
            final = now
            final["physical_time"] = i * eta
            break
    assert final is not None
    assert maxincrease < 1e-12
    assert minimum["risk_unregularized"] < 0.42
    assert final["risk_unregularized"] > 0.83
    assert final["view_energy"] < 1
    return {"eta": eta, "start": start, "minimum": minimum,
            "final": final, "samples": captures, "max_loss_increase": maxincrease}


def exact_derivatives():
    # d/dzeta with w'=B a, B'_ck=4*g_k*B_ck*w_c*a_k.
    b = [[F(2), F(1, 5)], [F(1, 5), F(2, 5)]]
    w, g, a = [F(1, 10), F(9, 10)], [F(1), F(2)], F(19, 50)
    pd = sum(row[1] ** 2 - row[0] ** 2 for row in b)
    fd = 4 * sum(w[c] ** 2 * (g[1] * b[c][1] - g[0] * b[c][0]) for c in range(2))
    ppd = 8 * sum(w[c] * a * (g[1] * b[c][1] ** 2 - g[0] * b[c][0] ** 2) for c in range(2))
    td = 16 * sum(w[c] ** 3 * a * (g[1] ** 2 * b[c][1] - g[0] ** 2 * b[c][0]) for c in range(2))
    ud = 8 * sum(w[c] * sum(b[c]) * a * (g[1] * b[c][1] - g[0] * b[c][0]) for c in range(2))
    critical6 = -fd / pd
    curvature = critical6 * (ppd + ud) + td
    assert critical6 == F(47, 96)
    assert curvature == F(743489, 125000)
    cstar = float(critical6) ** (1 / 6)
    tangent = []
    for c in [0.5, 1, 1.5]:
        row = snapshot(init(c), [1, 2], [1, 1], [0, 0])
        tangent.append({"c": c, "A0": row["a"], "drisk_dlogv": row["drisk_dlogv"]})
    assert tangent[0]["drisk_dlogv"] > 0
    assert tangent[1]["drisk_dlogv"] < 0
    # Independent directional finite difference of N at cstar.
    def effective_field(y):
        row = observables(y, [1, 2], [1, 1], [0, 0])
        return [x / (2 * gamma(row["view_energy"])) for x in row["field"]]

    def nvalue(y):
        aa = observables(y, [1, 2], [1, 1], [0, 0])["a"]
        dy = effective_field(y)
        ww, hh = y[:2], [y[2:4], y[4:6]]
        da = [sum(dy[c] * g[k] * hh[c][k] ** 2 +
                  ww[c] * 2 * g[k] * hh[c][k] * dy[2 + 2 * c + k]
                  for c in range(2)) for k in range(2)]
        return -aa[1] * da[0] + aa[0] * da[1]

    ystar = init(cstar)
    vec = effective_field(ystar)
    step = 1e-6
    numerical = (nvalue([x + step * v for x, v in zip(ystar, vec)]) -
                 nvalue([x - step * v for x, v in zip(ystar, vec)])) / (2 * step)
    exact = float(a * a * curvature) * cstar ** -4
    assert abs(numerical - exact) < 1e-8
    return {"P22_minus_P11": str(pd), "F22_minus_F11": str(fd),
            "critical_c6": str(critical6), "critical_c": cstar,
            "Pprime22_minus_Pprime11_base": str(ppd),
            "T22_minus_T11": str(td), "U22_minus_U11": str(ud),
            "Nprime_over_a02_cminus4": str(curvature),
            "Nprime_directional_finite_difference": numerical,
            "Nprime_exact_float": exact,
            "initial_energy_section_tangents": tangent}


def gradient_check():
    y = init(1.1, 0.7)
    g, s, d = [1, 2], [1.0, 2.25], [0, 2.5]
    analytic = [-x for x in observables(y, g, s, d)["field"]]
    differences = []
    step = 1e-6
    for j in range(6):
        yp, ym = y[:], y[:]
        yp[j] += step
        ym[j] -= step
        differences.append((observables(yp, g, s, d)["loss"] -
                            observables(ym, g, s, d)["loss"]) / (2 * step))
    err = max(abs(x - y) for x, y in zip(analytic, differences))
    assert err < 1e-8
    return {"max_absolute_error": err, "analytic": analytic, "finite_difference": differences}


def frozen_head():
    rows = []
    for u2 in [0.1, 0.2, 2]:
        u1 = 1 / (5 / 6 + 1 / (4 * u2))
        row = {"u2": u2, "u1": u1}
        for name, w2 in [("frozen", .09), ("joint", -.11 + (u1 + u2) / 2)]:
            e1, e2 = w2 * u1 ** 2, 4 * w2 * u2 ** 2
            v = e1 + e2
            row[name] = {"clean_energy": v, "risk0": e2 / v,
                         "ridge_005": 1 - e1 * (v + .1) / (v + .05) ** 2}
        assert abs(row["joint"]["risk0"] - row["frozen"]["risk0"]) < 1e-14
        rows.append(row)
    assert rows[1]["frozen"]["ridge_005"] < rows[0]["frozen"]["ridge_005"]
    assert rows[2]["frozen"]["ridge_005"] > rows[1]["frozen"]["ridge_005"]
    # Exact crossing in the general formula, for the original initialization.
    g1, g2, u10, u20 = 1, 2, .3, .1
    alpha1, alpha2 = g1 ** 2 * u10, g2 ** 2 * u20
    r0 = g2 * u20 / (g1 * u10)
    scross = (1 - r0) / (alpha2 - r0 * alpha1)
    u1 = u10 / (1 - alpha1 * scross)
    u2 = u20 / (1 - alpha2 * scross)
    vf = 2 * .09 * (g2 * u2) ** 2
    vj = 2 * (.09 + (u1 + u2 - u10 - u20) / 2) * (g2 * u2) ** 2
    return {"states": rows, "crossing": {"s": scross, "u1": u1, "u2": u2,
            "frozen_energy": vf, "joint_energy": vj},
            "iid_complex_gaussian_unconstrained_overtaking_probability":
                (g1 / g2) / (1 + g1 / g2) - (g1 / g2) ** 2 / (1 + (g1 / g2) ** 2)}


def augmentation_switch():
    rows = []
    for eta2 in [0, .4, .5]:
        ss = (1 + eta2) ** 2
        row = snapshot(init(.5), [1, 2], [1, ss], [0, 2 * (ss - 1)])
        rows.append({"relative_noise_band2": eta2, **row})
    assert all(x > 0 for x in rows[-1]["r"])
    assert rows[0]["drisk_dlogv"] > 0 and rows[-1]["drisk_dlogv"] < 0
    threshold = 6269 / 20829
    lo, hi = .4, .5
    for _ in range(60):
        eta2 = (lo + hi) / 2
        ss = (1 + eta2) ** 2
        q = (19 / 50) ** 2 * (1 + ss)
        ratio = ss - (ss - 1) / gamma(q)
        if ratio > threshold:
            lo = eta2
        else:
            hi = eta2
    return {"states": rows, "r2_over_r1_sign_threshold_at_c05": threshold,
            "relative_noise_band2_numerical_root": (lo + hi) / 2,
            "uniform_relative_noise_growth_condition_at_view_cap_1":
                math.sqrt(8 / 5) - 1}


def direct_profile(rho=.4, noise=.2, delta=.002, max_progress=6):
    ss = (1 + noise) ** 2
    v0 = 2 * (.38 * rho ** 3) ** 2
    t, out = 0., []
    for i in range(round(max_progress / delta) + 1):
        ell = i * delta
        v, q = v0 * math.exp(ell), ss * v0 * math.exp(ell)
        if any(abs(ell - x) < delta / 4 for x in [0, .3, 1, 1.7, 3, 6]):
            out.append({"log_energy_progress": ell, "physical_time": t,
                        "clean_energy": v, "view_energy": q,
                        "risk_unregularized": .5,
                        "loss": (ss - 1) * v + gu(q)})
        def dtde(e):
            qe = ss * v0 * math.exp(e)
            return 1 / (4 * (ss * gamma(qe) - (ss - 1)))
        t += delta * (dtde(ell) + 4 * dtde(ell + delta / 2) + dtde(ell + delta)) / 6
    return {"mode": "direct_A", "samples": out, "final": out[-1]}


def local_taylor_check():
    cstar = (47 / 96) ** (1 / 6)
    aa = 288 / 25 * cstar ** 3
    bb = 743489 / 250000 * cstar ** -4
    rows = []
    def effective(y):
        row = observables(y, [1, 2], [1, 1], [0, 0])
        return [x / (2 * gamma(row["clean_energy"])) for x in row["field"]]
    for e in [.001, .0005]:
        pair = []
        for multiplier in [1, 3]:
            zeta = multiplier * aa / bb * e
            y = init(cstar + e)
            for _ in range(200):
                y = rk4(y, zeta / 200, effective)
            pair.append(snapshot(y, [1, 2], [1, 1], [0, 0])["risk_unregularized"])
        assert pair[0] < .5 < pair[1]
        rows.append({"e": e, "risk1": pair[0], "risk2": pair[1],
                     "risk_gap_over_e2": (pair[1] - pair[0]) / e ** 2})
    return {"predicted_limit_gap_over_e2": 2 * aa ** 2 / bb, "rows": rows}


def same_physical_time():
    g, ss = [1, 2], 1.44
    s, d = [ss, ss], [.88, .88]
    rows = []
    for mode, c in [("joint", .5), ("joint", 1), ("joint", 1.5), ("frozen", 1)]:
        y = init(c, .4)
        for _ in range(1000):
            y = rk4(y, .001, lambda z: observables(z, g, s, d, mode)["field"])
        row = snapshot(y, g, s, d, mode)
        assert row["view_energy"] < 1
        rows.append({"mode": mode, "c": c, "physical_time": 1,
                     "clean_energy": row["clean_energy"], "view_energy": row["view_energy"],
                     "risk_unregularized": row["risk_unregularized"]})
    a = [.02432, .02432]
    def direct_field(z):
        gain = ss * gamma(ss * sum(x * x for x in z)) - (ss - 1)
        return [2 * gain * x for x in z]
    for _ in range(1000):
        a = rk4(a, .001, direct_field)
    rows.append({"mode": "direct_A", "physical_time": 1,
                 "clean_energy": sum(x * x for x in a), "risk_unregularized": .5})
    return rows


if __name__ == "__main__":
    result = {"scope": "Population GF/GD; no empirical image data; no literature novelty audit",
              "frozen_head": frozen_head(), "exact_multichannel_derivatives": exact_derivatives(),
              "six_parameter_gradient_check": gradient_check(),
              "augmentation_switch": augmentation_switch(),
              "local_unregularized_U_taylor_check": local_taylor_check(),
              "same_physical_time_controls": same_physical_time()}
    profiles = []
    csvrows = []
    for c in [.5, 1, 1.5]:
        p, full = profile(c, .4, .2)
        assert p["final"]["view_energy"] < 1
        profiles.append(p)
        for row in full[::10]:
            csvrows.append({"mode": "joint", "c": c, **{k: row[k] for k in ["log_energy_progress", "physical_time", "clean_energy", "view_energy", "loss", "risk_unregularized"]}})
    finer, _ = profile(1, .4, .2, .001)
    err = abs(finer["final"]["risk_unregularized"] - profiles[1]["final"]["risk_unregularized"])
    assert err < 1e-9
    assert profiles[1]["minimum"]["risk_unregularized"] < .42
    assert profiles[1]["final"]["risk_unregularized"] > .83
    result["matched_initial_A_energy_profiles"] = profiles
    frozen1, frozenrows = profile(1, .4, .2, mode="frozen")
    frozen05, _ = profile(.5, .4, .2, mode="frozen")
    assert abs(frozen1["final"]["risk_unregularized"] - frozen05["final"]["risk_unregularized"]) < 1e-12
    assert abs(frozen05["final"]["physical_time"] / frozen1["final"]["physical_time"] - .25) < 1e-12
    result["frozen_multichannel_control"] = [frozen1, frozen05]
    for row in frozenrows[::10]:
        csvrows.append({"mode": "frozen", "c": 1, **{k: row[k] for k in ["log_energy_progress", "physical_time", "clean_energy", "view_energy", "loss", "risk_unregularized"]}})
    result["direct_A_control"] = direct_profile()
    result["rk4_step_halving_final_risk_error"] = err
    result["independent_physical_parameter_gd"] = [actual_gd(.01), actual_gd(.005)]
    result["all_checks_passed"] = True
    (ROOT / "mechanism_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "mechanism_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        writer.writeheader()
        writer.writerows(csvrows)
    print(json.dumps({"all_checks_passed": True,
                      "exact_derivatives": result["exact_multichannel_derivatives"],
                      "augmentation_switch_root": result["augmentation_switch"]["relative_noise_band2_numerical_root"],
                      "profiles": [{"c": p["c"], "minimum_risk": p["minimum"]["risk_unregularized"],
                                    "min_at_clean_energy": p["minimum"]["clean_energy"],
                                    "final_risk": p["final"]["risk_unregularized"],
                                    "final_view_energy": p["final"]["view_energy"],
                                    "final_time": p["final"]["physical_time"]} for p in profiles],
                      "rk4_halving_error": err,
                      "gd": [{"eta": r["eta"], "minimum_risk": r["minimum"]["risk_unregularized"],
                              "final_risk": r["final"]["risk_unregularized"],
                              "final_time": r["final"]["physical_time"]} for r in result["independent_physical_parameter_gd"]]}, indent=2))
