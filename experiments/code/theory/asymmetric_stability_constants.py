"""Exact rational checks for the off-symmetry stability proof; no ODE integration.

Every asserted bound uses Fraction interval endpoints. Floats only format output.
Run directly with the workspace's prescribed Python interpreter.
"""
from fractions import Fraction as F
from math import factorial, isqrt
import json
from pathlib import Path


class Interval:
    def __init__(self, lo, hi=None):
        lo, hi = F(lo), F(lo if hi is None else hi)
        # Outward rational rounding keeps transcendental compositions compact.
        scale = 10 ** 50
        self.lo = F(lo.numerator * scale // lo.denominator, scale)
        self.hi = F(-((-hi.numerator * scale) // hi.denominator), scale)

    @staticmethod
    def cast(x):
        return x if isinstance(x, Interval) else Interval(x)

    def __add__(self, other):
        other = self.cast(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -self.cast(other)

    def __rsub__(self, other):
        return self.cast(other) + -self

    def __mul__(self, other):
        other = self.cast(other)
        values = [a * b for a in (self.lo, self.hi)
                  for b in (other.lo, other.hi)]
        return Interval(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.cast(other)
        assert other.lo > 0 or other.hi < 0
        return self * Interval(1 / other.hi, 1 / other.lo)

    def __rtruediv__(self, other):
        return self.cast(other) / self

    def __pow__(self, exponent):
        assert exponent >= 0
        result = Interval(1)
        for _ in range(exponent):
            result = result * self
        return result

    def display(self):
        return [float(self.lo), float(self.hi)]


def sqrt_interval(x):
    x = Interval.cast(x)
    assert x.lo >= 0
    scale = 10 ** 65
    low = isqrt(x.lo.numerator * scale * scale // x.lo.denominator)
    high = isqrt(x.hi.numerator * scale * scale // x.hi.denominator) + 1
    return Interval(F(low, scale), F(high, scale))


def atan_point(x, count=120):
    assert 0 <= x < 1 and count % 2 == 0
    partial = sum(((-1) ** n * x ** (2 * n + 1) / (2 * n + 1)
                   for n in range(count)), F(0))
    return Interval(partial, partial + x ** (2 * count + 1) / (2 * count + 1))


def atan_interval(x):
    x = Interval.cast(x)
    return Interval(atan_point(x.lo).lo, atan_point(x.hi).hi)


def sin_interval(x, count=24):
    x = Interval.cast(x)
    value = sum(((-1) ** n * x ** (2 * n + 1) / factorial(2 * n + 1)
                 for n in range(count)), Interval(0))
    error = max(abs(x.lo), abs(x.hi)) ** (2 * count) / factorial(2 * count)
    return value + Interval(-error, error)


def cos_interval(x, count=24):
    x = Interval.cast(x)
    value = sum(((-1) ** n * x ** (2 * n) / factorial(2 * n)
                 for n in range(count)), Interval(0))
    error = max(abs(x.lo), abs(x.hi)) ** (2 * count - 1) / factorial(2 * count - 1)
    return value + Interval(-error, error)


def log_interval(x, count=160):
    x = Interval.cast(x)
    assert x.lo > 1
    y = (x - 1) / (x + 1)
    assert 0 < y.lo <= y.hi < 1
    value = 2 * sum((y ** (2 * n + 1) / (2 * n + 1)
                     for n in range(count)), Interval(0))
    tail = 2 * y.hi ** (2 * count + 1) / ((2 * count + 1) * (1 - y.hi ** 2))
    return value + Interval(0, tail)


def exp_interval(x, count=80):
    x = F(x)
    assert x >= 0
    value = sum((x ** n / factorial(n) for n in range(count + 1)), F(0))
    ratio = x / (count + 2)
    assert ratio < 1
    tail = x ** (count + 1) / factorial(count + 1) / (1 - ratio)
    return Interval(value, value + tail)


def check():
    e = F(1, 125000)  # Independent absolute perturbation of all six coordinates.
    theta0 = atan_interval(Interval(F(1, 2)))
    sqrt2 = sqrt_interval(2)
    sections = []
    for multiplier, upper in ((F(3, 4), F(203, 1000)),
                              (F(1, 6), F(387, 500))):
        theta = theta0 * multiplier
        tangent = sin_interval(theta) / cos_interval(theta)
        sigma = log_interval(sqrt2 * (1 - tangent) / sqrt_interval(tangent)) / 2
        # Initial D-angle shift <=24e; derivative of 0.5 log(tan D) <=8/5.
        assert sigma.hi + F(8, 5) * 24 * e < upper
        sections.append(sigma)

    E1 = exp_interval(F(2001, 500) * F(203, 1000))
    E2 = exp_interval(F(2001, 500) * F(387, 500))
    assert E1.hi < F(1127, 500)  # 2.254
    assert E2.hi < F(2768, 125)  # 22.144
    assert 33 * e * E2.hi < F(1, 100)
    a1 = 12 + 33 * (E1.hi - 1) / F(2001, 250)
    a2 = 12 + 33 * (E2.hi - 1) / F(2001, 250)
    assert a1 < F(86, 5) and a2 < F(496, 5)
    mix_factor = F(1001, 1000)
    assert 1 / (1 - F(1, 10000)) < mix_factor
    p0 = sqrt2.hi * 12 + 24 + mix_factor * 33
    p1 = sqrt2.hi * a1 + mix_factor * 33 * E1.hi
    p2 = sqrt2.hi * a2 + mix_factor * 33 * E2.hi
    assert p0 < 75 and p1 < 99 and p2 < 872

    delta1 = F(3, 4) * sin_interval(theta0 / 4) * sin_interval(theta0 / 12)
    delta2 = F(3, 4) * sin_interval(theta0 * F(5, 6)) * sin_interval(theta0 / 6)
    assert delta1.lo > F(335, 100000)
    assert delta2.lo > F(2181, 100000)
    assert F(335, 100000) - 174 * e > F(1675, 1000000)
    assert F(2181, 100000) - 947 * e > F(10905, 1000000)

    m0_upper = F(15, 1000) + F(342, 1000) * e + 2 * e ** 2
    assert m0_upper * E2.hi < F(333, 1000)
    u2_upper = (F(333, 1000) + F(5006, 1000000)) / 2
    assert u2_upper < F(17, 100)
    energy_upper = 8 * F(17, 100) * F(333, 1000) ** 2 * F(10001, 10000) ** 2
    assert energy_upper < F(151, 1000)
    view_factor = F(121, 100)  # Fixed nonzero augmentation power eta=1/10.
    q_upper = view_factor * F(151, 1000)
    gain_lower = view_factor*(q_upper+2)/(2*(1+q_upper)**2)-(view_factor-1)
    assert gain_lower > F(29, 40)
    sigma_speed = (F(29, 5) * (F(1, 10) - e) ** 2
                   * (F(15, 1000) - F(342, 1000) * e + 2 * e ** 2)
                   * F(9999, 10000))
    assert sigma_speed > F(869, 1000000)
    assert F(387, 500) / sigma_speed < 900

    return {
        "scope": "Exact rational constant checks; no integration, sampling, or training",
        "absolute_coordinate_radius": float(e),
        "fixed_augmentation_power": .1,
        "gupa_gain_lower": float(gain_lower),
        "reference_section_progress_enclosures": [x.display() for x in sections],
        "amplification_enclosures": [E1.display(), E2.display()],
        "projector_error_coefficients_upper": [float(p0), float(p1), float(p2)],
        "risk_change_error_upper": [float(174 * e), float(947 * e)],
        "retained_risk_gap_lower": [float(F(335, 100000) - 174 * e),
                                    float(F(2181, 100000) - 947 * e)],
        "energy_upper": float(energy_upper),
        "physical_time_upper": float(F(387, 500) / sigma_speed),
        "all_checks_passed": True,
    }


if __name__ == "__main__":
    result = check()
    output = Path(__file__).resolve().parents[2] / "results/theory/asymmetric_stability_constants.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
