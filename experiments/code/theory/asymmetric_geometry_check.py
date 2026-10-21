"""Check the off-symmetry polar identities used by the rebound proof.

Closed states, exact Gaussian moments, and differentiation only: no training,
ODE integration, downloads, or sampling of initialization occurrence rates.
The finite grid checks algebra; the accompanying paper proof establishes the
whole independent initialization neighborhood.
"""

from __future__ import annotations

import json
import itertools
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/results/theory/asymmetric_geometry_check.json"
DTYPE = torch.float64
THETA0 = math.atan(.5)


def svec(matrix):
    return torch.stack((matrix[..., 0, 0], matrix[..., 1, 1],
                        math.sqrt(2) * matrix[..., 0, 1]), dim=-1)


def state(d, angle, m, u, imbalance, head_imbalance):
    angles = torch.tensor((angle + d, angle - d), dtype=DTYPE)
    lengths = torch.sqrt(m * torch.tensor((1 + imbalance, 1 - imbalance), dtype=DTYPE))
    filters = lengths[:, None] * torch.stack((angles.cos(), angles.sin()), dim=1)
    weights = u * torch.tensor((1 + head_imbalance, 1 - head_imbalance), dtype=DTYPE)
    return filters, weights


def polar(filters, weights):
    angles = torch.atan2(filters[:, 1], filters[:, 0])
    v = filters.square().sum(dim=1)
    m, u = v.mean(), weights.mean()
    return torch.stack((angles.mean(), (angles[0] - angles[1])/2,
                        m, u, (v[0] - v[1])/(2*m),
                        (weights[0] - weights[1])/(2*u)))


def clock_field(filters, weights):
    output = torch.einsum("c,ci,cj->ij", weights, filters, filters)
    h_dot = weights[:, None] * (filters @ output)
    w_dot = .5 * torch.einsum("ci,ij,cj->c", filters, output, filters)
    return h_dot, w_dot


def projector(matrix):
    return matrix.T @ torch.linalg.solve(matrix @ matrix.T, matrix)


def projectors(filters, weights):
    b = math.sqrt(2) * svec(filters[:, :, None] * filters[:, None, :])
    return projector(b), projector(weights[None, :] @ b)


def main():
    torch.set_num_threads(1)
    maxima = dict.fromkeys(("physical_gradient", "polar_field", "channel_invariant",
                           "encoder_projector_formula", "projection_projector_formula",
                           "transverse_linearization"), 0.)
    count = 0
    for fraction in (1., .75, 1/6):
        theta = fraction * THETA0
        tau = (.5 - math.tan(theta))/(.01*(1 - math.tan(theta)))
        lp, lm = .02/(1-.02*tau), .01/(1-.01*tau)
        m, u = (lp+lm)/2, math.sqrt(.01+(lp+lm-.03)/4)
        d = math.atan(math.sqrt(math.tan(theta)))
        for angle in (-.17, 0., .05):
            for g in (-.03, 0., .01):
                for z in (-.02, 0., .025):
                    filters, weights = state(d, angle, m, u, g, z)
                    filters.requires_grad_()
                    weights.requires_grad_()
                    h_dot, w_dot = clock_field(filters, weights)
                    _, actual = torch.autograd.functional.jvp(
                        polar, (filters, weights), (h_dot, w_dot))
                    D = 2*d
                    sine, cosine = math.sin(D), math.cos(D)
                    speed = u*u*m*(1-z*z)
                    F = ((1+z*z)*(1+g*g)+4*z*g
                         +(1-z*z)*(1-g*g)*cosine*cosine)
                    expected = torch.tensor((
                        u*u*m*(1-z*z)*g*sine*cosine,
                        -u*u*m*(1-z*z)*sine*cosine,
                        2*u*u*m*m*F,
                        .5*u*m*m*(1+g*g+2*z*g+(1-g*g)*cosine*cosine),
                        speed*2*(1-g*g)/(1-z*z)
                        *(2*z+g*(sine*sine+z*z*(1+cosine*cosine))),
                        speed*(m/(u*u))
                        *(g-z*(1-g*g)*cosine*cosine/(1-z*z))), dtype=DTYPE)
                    maxima["polar_field"] = max(maxima["polar_field"],
                                                  float((actual-expected).abs().max()))
                    invariant_dot = 2*(filters*h_dot).sum(dim=1)-4*weights*w_dot
                    maxima["channel_invariant"] = max(maxima["channel_invariant"],
                                                       float(invariant_dot.abs().max().detach()))
                    output = torch.einsum("c,ci,cj->ij", weights, filters, filters)
                    energy = 2*output.square().sum()
                    for noise in (0., .1):
                        s = (1+noise)**2
                        view_energy = s*energy
                        loss = (s-1)*energy-.5*torch.log1p(view_energy)+.5/(1+view_energy)-.5
                        gh, gw = torch.autograd.grad(loss, (filters, weights), retain_graph=noise == 0.)
                        q = view_energy.detach()
                        multiplier = 8*(s*(q+2)/(2*(1+q).square())-(s-1))
                        discrepancy = max(float((-gh-multiplier*h_dot).abs().max().detach()),
                                          float((-gw-multiplier*w_dot).abs().max().detach()))
                        maxima["physical_gradient"] = max(maxima["physical_gradient"], discrepancy)
                    pb, pa = projectors(filters.detach(), weights.detach())
                    ca, sa = math.cos(angle), math.sin(angle)
                    rotation = torch.tensor(((ca, -sa), (sa, ca)), dtype=DTYPE)
                    normal_matrix = rotation @ torch.diag(torch.tensor(
                        (math.sin(d)**2, -math.cos(d)**2), dtype=DTYPE)) @ rotation.T
                    normal = svec(normal_matrix)
                    normal /= torch.linalg.vector_norm(normal)
                    expected_pb = torch.eye(3, dtype=DTYPE)-normal[:, None]*normal[None, :]
                    mixing = (g+z)/(1+g*z)
                    inner = torch.tensor(((math.cos(d)**2, mixing*math.sin(d)*math.cos(d)),
                                          (mixing*math.sin(d)*math.cos(d), math.sin(d)**2)), dtype=DTYPE)
                    direction = svec(rotation @ inner @ rotation.T)[None, :]
                    maxima["encoder_projector_formula"] = max(
                        maxima["encoder_projector_formula"], float((pb-expected_pb).abs().max()))
                    maxima["projection_projector_formula"] = max(
                        maxima["projection_projector_formula"], float((pa-projector(direction)).abs().max()))
                    count += 1

        # A separate derivative of the exact polar vector field checks its
        # transverse first-order system, including the head damping sign.
        def transverse(v):
            g, z = v[0], v[1]
            return torch.stack((2*(1-g*g)/(1-z*z)
                                *(2*z+g*(math.sin(2*d)**2+z*z*(1+math.cos(2*d)**2))),
                                (m/(u*u))*(g-z*(1-g*g)*math.cos(2*d)**2/(1-z*z))))
        derivative = torch.autograd.functional.jacobian(transverse, torch.zeros(2, dtype=DTYPE))
        expected = torch.tensor(((2*math.sin(2*d)**2, 4),
                                 (m/(u*u), -(m/(u*u))*math.cos(2*d)**2)), dtype=DTYPE)
        maxima["transverse_linearization"] = max(maxima["transverse_linearization"],
                                                   float((derivative-expected).abs().max()))

    d0 = math.atan(math.sqrt(.5))
    exact_improvement = .75*math.sin(THETA0/4)*math.sin(THETA0/12)
    exact_rebound = .75*math.sin(5*THETA0/6)*math.sin(THETA0/6)
    assert all(value < 1e-11 for value in maxima.values()), maxima
    # The proof integrates the growing imbalance instead of multiplying its
    # worst endpoint value by the entire angle travel.
    radius = 8e-6
    cube_maxima = dict.fromkeys(("initial_mean_angle_over_radius", "initial_spread_error_over_radius",
                                "initial_imbalance_over_radius", "initial_encoder_projector_over_radius",
                                "initial_projection_projector_over_radius"), 0.)
    base_h, base_w = state(d0, 0., .015, .1, 0., 0.)
    base_pb, base_pa = projectors(base_h, base_w)
    for signs in itertools.product((-1., 1.), repeat=6):
        increment = radius*torch.tensor(signs, dtype=DTYPE)
        h = base_h+increment[:4].reshape(2, 2)
        w = base_w+increment[4:]
        values = polar(h, w)
        pb, pa = projectors(h, w)
        quantities = (abs(float(values[0]))/radius, abs(float(values[1])-d0)/radius,
                      (abs(float(values[4]))+abs(float(values[5])))/radius,
                      float(torch.linalg.matrix_norm(pb-base_pb, ord=2))/radius,
                      float(torch.linalg.matrix_norm(pa-base_pa, ord=2))/radius)
        for name, value in zip(cube_maxima, quantities):
            cube_maxima[name] = max(cube_maxima[name], value)
    cube_sections = []
    for fraction, sigma_bound, projection_bound in ((.75, .203, 99), (1/6, .774, 872)):
        target_d = math.atan(math.sqrt(math.tan(THETA0*fraction)))
        E = math.exp(4.002*sigma_bound)
        angle_bound = 12+33*(E-1)/8.004
        projector_bound = math.sqrt(2)*angle_bound+33*E/(1-.01**2)
        assert .5*math.log(math.tan(2*(d0+12*radius))/math.tan(2*target_d)) < sigma_bound
        assert projector_bound < projection_bound
        cube_sections.append({"theta_fraction": fraction, "progress_bound": sigma_bound,
                              "imbalance_multiplier": E,
                              "mean_angle_over_radius_bound": angle_bound,
                              "projection_error_over_radius_bound": projector_bound,
                              "rounded_projection_error_over_radius_bound": projection_bound})
    assert cube_maxima["initial_mean_angle_over_radius"] < 12
    assert cube_maxima["initial_spread_error_over_radius"] < 12
    assert cube_maxima["initial_imbalance_over_radius"] < 33
    assert cube_maxima["initial_encoder_projector_over_radius"] < 48
    assert cube_maxima["initial_projection_projector_over_radius"] < 75
    assert exact_improvement-174*radius > .001675
    assert exact_rebound-947*radius > .010905
    cube = {"independent_coordinate_radius": radius,
            "initial_corner_checks": cube_maxima, "comparison_sections": cube_sections,
            "common_improvement_lower_bound": exact_improvement-174*radius,
            "common_rebound_lower_bound": exact_rebound-947*radius,
            "scope": "Corner checks corroborate formulas; they do not prove interior cube bounds"}
    result = {"scope": "Closed-state identity and differentiation checks, no trajectories",
              "states_checked": count, "maximum_absolute_discrepancies": maxima,
              "physical_gradient_checks": 2*count, "noise_powers_checked": [0., .1],
              "coordinate_cube_constant_float_checks": cube,
              "all_checks_passed": True,
              "limitations": ["Finite grids and floating-point calculations check implementation, not the continuous-neighborhood proof",
                              "This check establishes no random-initialization probability or typicality",
                              "Comparison states are first fixed angle sections, not equal physical times"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
