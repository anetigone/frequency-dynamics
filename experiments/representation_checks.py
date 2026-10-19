"""v0.3: evaluation location, layer learning rates and robustness scans.

Standard library only. Floating point scans are evidence, not interval proofs.
Run: python representation_checks.py
"""
import itertools
import json
import math
import random
from fractions import Fraction as F
from pathlib import Path

from mechanism_checks import gamma, gu, init, rk4

ROOT = Path(__file__).resolve().parent


class Interval:
    """Exact rational interval operations; no binary floating point endpoints."""
    def __init__(self, lo, hi=None):
        self.lo, self.hi = F(lo), F(lo if hi is None else hi)

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Interval) else -Interval(other))

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        products = [x * y for x in [self.lo, self.hi] for y in [other.lo, other.hi]]
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        assert other.lo > 0 or other.hi < 0
        return self * Interval(1 / other.hi, 1 / other.lo)

    def record(self):
        return {"lower_exact": str(self.lo), "upper_exact": str(self.hi),
                "lower_float": float(self.lo), "upper_float": float(self.hi)}


def initial_box_certificate():
    def box(value):
        return Interval(value * F(99, 100), value * F(101, 100))
    g = [box(F(1)), box(F(2))]
    w = [box(F(1, 25)), box(F(9, 25))]
    b = [[box(F(8, 25)), box(F(4, 125))],
         [box(F(4, 125)), box(F(8, 125))]]
    a = [sum(w[c] * b[c][k] for c in range(2)) for k in range(2)]
    p = [[sum(b[c][j] * b[c][k] for c in range(2)) for k in range(2)] for j in range(2)]
    ff = [4 * g[k] * sum(b[c][k] * w[c] * w[c] for c in range(2)) for k in range(2)]
    k11, k22, k12 = p[0][0] + ff[0], p[1][1] + ff[1], p[0][1]
    n = a[0] * a[1] * (k22 - k11) + k12 * (a[0] * a[0] - a[1] * a[1])
    v = a[0] * a[0] + a[1] * a[1]
    aka = k11 * a[0] * a[0] + 2 * k12 * a[0] * a[1] + k22 * a[1] * a[1]
    derivative = a[0] * a[1] * n / (v * aka)
    det = b[0][0] * b[1][1] - b[0][1] * b[1][0]
    assert derivative.lo >= -F(4, 25) and derivative.hi <= -F(2, 25)
    assert det.lo > 0
    assert (F(36, 25) * v).hi < 1
    return {"relative_box_radius": "1/100", "coordinates": "g1,g2,w1,w2,B11,B12,B21,B22",
            "drisk_dlogv": derivative.record(), "encoder_det": det.record(),
            "scope": "Entire closed initial parameter box; initial derivative only, not a trajectory certificate"}


def state(y, g, channels=2, eta_w=1., eta_h=1.):
    dim = len(g)
    w = y[:channels]
    h = [y[channels + c * dim:channels + (c + 1) * dim]
         for c in range(channels)]
    b = [[g[k] * h[c][k] ** 2 for k in range(dim)] for c in range(channels)]
    a = [sum(w[c] * b[c][k] for c in range(channels)) for k in range(dim)]
    v = sum(x * x for x in a)
    q = 1.44 * v
    gain = 1.44 * gamma(q) - .44
    dw = [eta_w * 2 * gain * sum(b[c][k] * a[k] for k in range(dim))
          for c in range(channels)]
    dh = [eta_h * 4 * gain * g[k] * h[c][k] * w[c] * a[k]
          for c in range(channels) for k in range(dim)]
    da = [sum(dw[c] * b[c][k] + w[c] * 2 * g[k] * h[c][k] * dh[c * dim + k]
              for c in range(channels)) for k in range(dim)]
    rate = 2 * sum(a[k] * da[k] for k in range(dim)) / v
    risk = sum(x * x for x in a[1:]) / v
    row = {"projection_excess_risk": risk, "clean_energy": v,
           "view_energy": q, "loss": .44 * v + gu(q),
           "dlogv_dt": rate, "a": a, "field": dw + dh}
    if channels == dim == 2:
        det = b[0][0] * b[1][1] - b[0][1] * b[1][0]
        # B^T v = e1; explicit inverse, without thresholding a singular matrix.
        if det == 0:
            raise ArithmeticError("Singular encoder")
        probe = [b[1][1] / det, -b[0][1] / det]
        residual = [sum(probe[c] * b[c][k] for c in range(2)) - (k == 0)
                    for k in range(2)]
        row.update(encoder_excess_risk=sum(x * x for x in residual),
                   encoder_det=det)
    return row


def run(y, g, channels=2, step=.01):
    y = y[:] + [0.]
    start = state(y, g, channels)
    captures = {}
    min_abs_det = abs(start.get("encoder_det", 0))
    max_encoder_risk = 0.
    def field(z):
        row = state(z, g, channels)
        assert row["dlogv_dt"] > 0
        return [x / row["dlogv_dt"] for x in row["field"]] + [1 / row["dlogv_dt"]]
    for i in range(round(6 / step) + 1):
        row = state(y, g, channels)
        assert row["view_energy"] < 1
        max_encoder_risk = max(max_encoder_risk, row.get("encoder_excess_risk", 0))
        min_abs_det = min(min_abs_det, abs(row.get("encoder_det", 0)))
        if i in [round(1.7 / step), round(6 / step)]:
            captures[str(round(i * step, 2))] = {k: v for k, v in row.items() if k != "field"}
            captures[str(round(i * step, 2))]["physical_time"] = y[-1]
        if i < round(6 / step):
            y = rk4(y, step, field)
    early, final = captures["1.7"], captures["6.0"]
    return {"initial_risk": start["projection_excess_risk"],
            "improvement": start["projection_excess_risk"] - early["projection_excess_risk"],
            "rebound": final["projection_excess_risk"] - early["projection_excess_risk"],
            "overshoot": final["projection_excess_risk"] - start["projection_excess_risk"],
            "max_encoder_excess_risk": max_encoder_risk if channels == len(g) == 2 else None,
            "min_abs_encoder_det": min_abs_det if channels == len(g) == 2 else None,
            "checkpoints": captures}


def learning_rates():
    rows = []
    for c in [.5, 1., 1.5]:
        scaled = init(c, .4)
        base = init(1, .4)
        ratio_only = base[:]
        max_error = 0.
        for _ in range(1000):
            scaled = rk4(scaled, .001, lambda z: state(z, [1, 2])["field"])
            base = rk4(base, .001, lambda z: state(z, [1, 2], eta_w=c**4, eta_h=c**-2)["field"])
            # eta_h=1, eta_w=c^6: its clock is tau=t/c^2.
            ratio_only = rk4(ratio_only, .001 / c**2,
                             lambda z: state(z, [1, 2], eta_w=c**6)["field"])
            mapped = [x * c**-2 for x in base[:2]] + [x * c for x in base[2:]]
            max_error = max(max_error, max(abs(x - z) for x, z in zip(scaled, mapped)),
                            max(abs(x - z) for x, z in zip(base, ratio_only)))
        assert max_error < 1e-11
        rows.append({"c": c, "relative_learning_rate": c**6,
                     "max_parameter_path_error": max_error})
    return rows


def summarize(rows):
    return {"count": len(rows),
            "min_improvement": min(r["improvement"] for r in rows),
            "min_rebound": min(r["rebound"] for r in rows),
            "min_overshoot": min(r["overshoot"] for r in rows),
            "max_encoder_excess_risk": max((r["max_encoder_excess_risk"] for r in rows
                                           if r["max_encoder_excess_risk"] is not None), default=None)}


def main():
    scale = [{"c": .98 + i * .002, **run(init(.98 + i * .002, .4), [1, 2])}
             for i in range(21)]
    # All vertices of the relative +/-1% box in (g1,g2,w1,w2,B11,B12,B21,B22).
    # Corners do not certify the interior: the ODE outcome need not be monotone.
    def perturbed(offsets):
        g = [v * (1 + offsets[k]) for k, v in enumerate([1., 2.])]
        w = [v * (1 + offsets[2 + k]) for k, v in enumerate([.04, .36])]
        b = [v * .16 * (1 + offsets[4 + k]) for k, v in enumerate([2., .2, .2, .4])]
        h = [math.sqrt(b[k] / g[k % 2]) for k in range(4)]
        return run(w + h, g)
    corners = [perturbed(offsets) for offsets in itertools.product([-.01, .01], repeat=8)]
    rng = random.Random(20261019)
    interior = [perturbed([rng.uniform(-.01, .01) for _ in range(8)]) for _ in range(64)]
    baseline = run(init(1, .4), [1, 2])
    fine = run(init(1, .4), [1, 2], step=.005)
    # Add a third power coordinate and/or a third channel, all with nonzero entries.
    extra = []
    for channels, dim in [(2, 3), (3, 2), (3, 3)]:
        g = [1., 2., 1.5][:dim]
        w = [.04, .36, .005][:channels]
        b = [[.32, .032, .001], [.032, .064, .001], [.001, .001, .001]]
        h = [math.sqrt(b[c][k] / g[k]) for c in range(channels) for k in range(dim)]
        extra.append({"channels": channels, "dimension": dim, **run(w + h, g, channels)})
    result = {"scope": "Projection population risk; floating point, not certified continuum bounds",
              "exact_initial_box_certificate": initial_box_certificate(),
              "layer_learning_rate_equivalence": learning_rates(),
              "baseline": baseline, "scale_grid": scale,
              "relative_1percent_box_corners": summarize(corners),
              "relative_1percent_box_interior_seed_20261019": summarize(interior),
              "extra_power_coordinates_and_channels": extra,
              "step_halving_risk_error": max(abs(baseline["checkpoints"][k]["projection_excess_risk"] -
                                                   fine["checkpoints"][k]["projection_excess_risk"])
                                               for k in baseline["checkpoints"])}
    for rows in [scale, corners, interior, extra]:
        print(json.dumps(summarize(rows)))
        assert min(r["improvement"] for r in rows) > 0
        assert min(r["rebound"] for r in rows) > 0
        assert min(r["overshoot"] for r in rows) > 0
    assert result["step_halving_risk_error"] < 1e-7
    result["all_checks_passed"] = True
    (ROOT / "representation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["scale_grid", "extra_power_coordinates_and_channels"]}, indent=2))


if __name__ == "__main__":
    main()
