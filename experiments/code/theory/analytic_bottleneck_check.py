"""Numerical verification of the analytic Gaussian bottleneck family.

Evaluates closed-form states only: no training, ODE integration, or downloads.
Checks SVD population risks and autograd in the actual physical h,W metric.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

if "--autograd-worker" not in sys.argv:
    import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "results" / "theory" / "analytic_bottleneck_check.json"
EPSILON = .01
W0 = .1
THETA0 = math.atan(.5)
DELTA1 = .75*math.sin(THETA0/4)*math.sin(THETA0/12)
DELTA2 = .75*math.sin(5*THETA0/6)*math.sin(THETA0/6)
TOLERANCE = 1e-11
AUTOGRAD_PROCESS = None


def svec(m):
    return np.stack((m[..., 0, 0], m[..., 1, 1], math.sqrt(2)*m[..., 0, 1]), axis=-1)


def projector_risk(matrix, tasks):
    _, values, vh = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(matrix.shape)*np.finfo(float).eps*values.max()
    rank = int(np.sum(values > threshold))
    basis = vh[:rank]
    risk = np.sum(tasks*tasks, axis=1)-np.sum((tasks@basis.T)**2, axis=1)
    return risk, rank, values


def axes(angle):
    eplus = np.array((math.cos(angle), math.sin(angle)))
    eminus = np.array((-math.sin(angle), math.cos(angle)))
    rotation = np.column_stack((eplus, eminus))
    eplus_mat = np.outer(eplus, eplus)
    eminus_mat = np.outer(eminus, eminus)
    ecross_mat = (np.outer(eplus, eminus)+np.outer(eminus, eplus))/math.sqrt(2)
    return rotation, (eplus_mat, eminus_mat, ecross_mat)


def closed_state(nu, rate_ratio, theta, rotation):
    tangent = math.tan(theta)
    tau = (.5-tangent)/(EPSILON*(nu[0]-nu[1]*tangent))
    initial_lambda = np.array((2*EPSILON, EPSILON))
    lam = initial_lambda/(1-nu*initial_lambda*tau)
    w_squared = W0*W0+rate_ratio/4*np.sum((lam-initial_lambda)/nu)
    w = math.sqrt(w_squared)
    q_axes = np.array(((math.sqrt(lam[0]/2), math.sqrt(lam[1]/2)),
                       (math.sqrt(lam[0]/2), -math.sqrt(lam[1]/2))))
    q = q_axes@rotation.T
    sigma = rotation@np.diag(nu)@rotation.T
    root = rotation@np.diag(np.sqrt(nu))@rotation.T
    invroot = rotation@np.diag(1/np.sqrt(nu))@rotation.T
    h = q@invroot
    weight = np.array(((w, w),))
    b = math.sqrt(2)*svec(np.einsum("ci,cj->cij", q, q))
    a = weight@b
    return {"tau": tau, "lambda": lam, "w": w, "q": q, "h": h,
            "W": weight, "B": b, "A": a, "Sigma": sigma,
            "Sigma_sqrt": root, "Sigma_invsqrt": invroot}


def physical_autograd(state, noise, rate_ratio):
    """Differentiate population GUPA from physical Gaussian quadratic moments.

    This does not construct B, A, lambda, or use the expected velocity formulas
    in the differentiated loss. Isserlis' identity gives variance 2tr(HΣHΣ).
    """
    # This environment's NumPy SVD and PyTorch load conflicting OpenMP DLLs.
    # Keep the actual autograd computation in a separate process, without
    # permitting duplicate OpenMP runtimes or changing the user's environment.
    global AUTOGRAD_PROCESS
    if AUTOGRAD_PROCESS is None:
        AUTOGRAD_PROCESS = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--autograd-worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    request = {"h": state["h"].tolist(), "W": state["W"].tolist(), "Sigma": state["Sigma"].tolist(),
               "noise": noise, "rate_ratio": rate_ratio}
    AUTOGRAD_PROCESS.stdin.write(json.dumps(request)+"\n")
    AUTOGRAD_PROCESS.stdin.flush()
    line = AUTOGRAD_PROCESS.stdout.readline()
    if not line:
        raise RuntimeError(AUTOGRAD_PROCESS.stderr.read())
    result = json.loads(line)
    result["h_velocity"] = np.array(result["h_velocity"])
    result["W_velocity"] = np.array(result["W_velocity"])
    return result


def run_autograd_worker():
    import torch
    torch.set_num_threads(1)
    for line in sys.stdin:
        request = json.loads(line)
        h = torch.tensor(request["h"], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor(request["W"], dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(request["Sigma"], dtype=torch.float64)
        htotal = torch.einsum("c,ci,cj->ij", weight[0], h, h)
        view_sigma = (1+request["noise"])*sigma
        view_variance = 2*torch.trace(htotal@view_sigma@htotal@view_sigma)
        paired_covariance = 2*torch.trace(htotal@sigma@htotal@sigma)
        difference_second_moment = 2*(view_variance-paired_covariance)
        loss = (.5*difference_second_moment-.5*torch.log1p(view_variance)
                -.5*view_variance/(1+view_variance))
        grad_h, grad_w = torch.autograd.grad(loss, (h, weight))
        result = {"h_velocity": (-grad_h).tolist(),
                  "W_velocity": (-request["rate_ratio"]*grad_w).tolist(),
                  "loss": float(loss.detach()), "view_energy": float(view_variance.detach()),
                  "clean_energy": float(paired_covariance.detach())}
        print(json.dumps(result), flush=True)


def close_autograd_worker():
    if AUTOGRAD_PROCESS is not None:
        AUTOGRAD_PROCESS.stdin.close()
        code = AUTOGRAD_PROCESS.wait(timeout=20)
        if code != 0:
            raise RuntimeError(AUTOGRAD_PROCESS.stderr.read())


def closed_velocity(state, nu, ratio, noise, rotation):
    lam, w = state["lambda"], state["w"]
    v = 2*w*w*np.sum(lam*lam)
    s = (1+noise)**2
    view_energy = s*v
    gamma = (view_energy+2)/(2*(1+view_energy)**2)
    kappa = 2*((s-1)-s*gamma)
    tau_velocity = -8*kappa*w*w  # eta_h=1
    lambda_velocity = tau_velocity*nu*lam*lam
    q_axes = state["q"]@rotation
    q_velocity = (q_axes*(lambda_velocity/(2*lam)))@rotation.T
    h_velocity = q_velocity@state["Sigma_invsqrt"]
    w_velocity = -ratio*kappa*w*np.sum(lam*lam)
    invariant_derivative = 2*w*w_velocity-ratio/4*np.sum(lambda_velocity/nu)
    return {"h_velocity": h_velocity,
            "W_velocity": np.full((1, 2), w_velocity),
            "q_velocity": q_velocity, "invariant_derivative": invariant_derivative,
            "kappa": kappa, "tau_velocity": tau_velocity,
            "clean_energy": v, "view_energy": view_energy}


def covariance_metric_error(sigma, invroot):
    canonical = (np.array(((1., 0.), (0., 0.))),
                 np.array(((0., 0.), (0., 1.))),
                 np.array(((0., 1.), (1., 0.)))/math.sqrt(2))
    # u_i = x^T (Σ^-1/2 E_i Σ^-1/2 / sqrt(2)) x - E[...].
    forms = [invroot@e@invroot/math.sqrt(2) for e in canonical]
    covariance = np.array([[2*np.trace(fi@sigma@fj@sigma) for fj in forms] for fi in forms])
    return float(np.max(np.abs(covariance-np.eye(3))))


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT/"paper"/"figures"
    folder.mkdir(parents=True, exist_ok=True)
    progress = np.linspace(0, 5/6, 401)
    theta = THETA0*(1-progress)
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        for fraction, color, style, label in ((2/3, "#0072B2", "-", r"$\psi=2\theta_0/3$"),
                                               (3/4, "#D55E00", "--", r"$\psi=3\theta_0/4$"),
                                               (5/6, "#009E73", "-.", r"$\psi=5\theta_0/6$")):
            risk = np.sin(theta-fraction*THETA0)**2
            ax.plot(progress, risk, label=label, color=color, linestyle=style, linewidth=1.4)
        for point, label in ((.25, r"$t_1$"), (5/6, r"$t_2$")):
            ax.axvline(point, color=".5", linestyle=":", linewidth=.8)
            ax.text(point-.012, .16, label, transform=ax.get_xaxis_transform(), ha="right", va="bottom", color=".35")
        ax.set(xlabel=r"Subspace progress $1-\theta/\theta_0$", ylabel="Population square risk",
               title="Closed-form risk", xlim=(0, .85), ylim=(-.005, .105))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(.30, 1.0), handlelength=2.3)
        fig.tight_layout(pad=.5)
        paths = [folder/"analytic_bottleneck_family.pdf", folder/"analytic_bottleneck_family.png"]
        fig.savefig(paths[0])
        fig.savefig(paths[1], dpi=240)
        plt.close(fig)
    return [p.as_posix() for p in paths]


def main():
    maxima = {key: 0. for key in ("svd_encoder_risk_error", "svd_projection_risk_error",
                                "h_velocity_absolute_error", "W_velocity_absolute_error",
                                "velocity_relative_error", "gaussian_whitened_covariance_error",
                                "clean_energy_error", "view_energy_error", "w_squared_invariant_derivative_error",
                                "section_angle_error", "identity_metric_mismatch_on_anisotropic_states")}
    minima = {"encoder_improvement": float("inf"), "encoder_overshoot": float("inf"),
              "projection_improvement": float("inf"), "projection_overshoot": float("inf"),
              "gain": float("inf")}
    state_count = 0
    gradient_count = 0
    max_energy = max_view_energy = 0.
    records = []
    psi_values = THETA0*np.array((2/3, 17/24, 3/4, 19/24, 5/6))
    z_values = np.array((-.5, -.25, 0., .25, .5))
    task_parameters = [(float(psi), float(z)) for psi in psi_values for z in z_values]
    for angle in (0., .37):
        rotation, e = axes(angle)
        tasks = np.array([svec(math.sqrt(1-z*z)*(math.cos(psi)*e[0]+math.sin(psi)*e[1])+z*e[2])
                          for psi, z in task_parameters])
        for nu_plus in (.95, 1., 1.05):
            for nu_minus in (.95, 1., 1.05):
                nu = np.array((nu_plus, nu_minus))
                for ratio in (0., .1, 1., 10.):
                    risks_b, risks_a, state_records = [], [], []
                    losses_by_noise = {noise: [] for noise in (0., .1, .2)}
                    for section, theta in enumerate((THETA0, .75*THETA0, THETA0/6)):
                        state = closed_state(nu, ratio, theta, rotation)
                        state_count += 1
                        rb, rank_b, sb = projector_risk(state["B"], tasks)
                        ra, rank_a, sa = projector_risk(state["A"], tasks)
                        assert rank_b == 2 and rank_a == 1
                        expected_b = np.array([(1-z*z)*math.sin(theta-psi)**2 for psi, z in task_parameters])
                        expected_a = expected_b+np.array([z*z for _, z in task_parameters])
                        maxima["svd_encoder_risk_error"] = max(maxima["svd_encoder_risk_error"], float(np.max(np.abs(rb-expected_b))))
                        maxima["svd_projection_risk_error"] = max(maxima["svd_projection_risk_error"], float(np.max(np.abs(ra-expected_a))))
                        maxima["section_angle_error"] = max(maxima["section_angle_error"], abs(math.atan(state["lambda"][1]/state["lambda"][0])-theta))
                        maxima["gaussian_whitened_covariance_error"] = max(maxima["gaussian_whitened_covariance_error"], covariance_metric_error(state["Sigma"], state["Sigma_invsqrt"]))
                        risks_b.append(rb)
                        risks_a.append(ra)
                        v = float(np.sum(state["A"]**2))
                        assert v < .2 and state["lambda"][0] < .33 and state["lambda"][1] < .025 and state["w"]**2 < .87
                        assert state["tau"] < 10/(19*EPSILON)
                        max_energy = max(max_energy, v)
                        for noise in losses_by_noise:
                            actual = physical_autograd(state, noise, ratio)
                            expected = closed_velocity(state, nu, ratio, noise, rotation)
                            gradient_count += 1
                            for block in ("h", "W"):
                                error = float(np.max(np.abs(actual[block+"_velocity"]-expected[block+"_velocity"])))
                                maxima[block+"_velocity_absolute_error"] = max(maxima[block+"_velocity_absolute_error"], error)
                                scale = max(1e-12, float(np.max(np.abs(expected[block+"_velocity"]))))
                                maxima["velocity_relative_error"] = max(maxima["velocity_relative_error"], error/scale)
                            maxima["clean_energy_error"] = max(maxima["clean_energy_error"], abs(actual["clean_energy"]-v))
                            maxima["view_energy_error"] = max(maxima["view_energy_error"], abs(actual["view_energy"]-expected["view_energy"]))
                            maxima["w_squared_invariant_derivative_error"] = max(maxima["w_squared_invariant_derivative_error"], abs(expected["invariant_derivative"]))
                            if nu_plus != 1 or nu_minus != 1:
                                actual_qdot = actual["h_velocity"]@state["Sigma_sqrt"]
                                identity_metric_qdot = -4*expected["kappa"]*state["w"]**2*state["q"]@(state["q"].T@state["q"])
                                maxima["identity_metric_mismatch_on_anisotropic_states"] = max(maxima["identity_metric_mismatch_on_anisotropic_states"], float(np.max(np.abs(actual_qdot-identity_metric_qdot))))
                            assert expected["view_energy"] < 1/3 and -expected["kappa"]/2 > 101/200
                            minima["gain"] = min(minima["gain"], -expected["kappa"]/2)
                            max_view_energy = max(max_view_energy, actual["view_energy"])
                            losses_by_noise[noise].append(actual["loss"])
                        state_records.append({"section": section, "theta": theta, "tau": state["tau"],
                                              "lambda": state["lambda"].tolist(), "w_squared": state["w"]**2,
                                              "clean_energy": v, "encoder_rank": rank_b, "projection_rank": rank_a,
                                              "encoder_singular_values": sb.tolist(), "projection_singular_values": sa.tolist()})
                    for key, risks in (("encoder", risks_b), ("projection", risks_a)):
                        improvement = float(np.min(risks[0]-risks[1]))
                        overshoot = float(np.min(risks[2]-risks[0]))
                        minima[key+"_improvement"] = min(minima[key+"_improvement"], improvement)
                        minima[key+"_overshoot"] = min(minima[key+"_overshoot"], overshoot)
                        assert improvement+TOLERANCE >= DELTA1 and overshoot+TOLERANCE >= DELTA2
                    assert all(values[2] < values[1] < values[0] for values in losses_by_noise.values())
                    records.append({"orientation_radians": angle, "nu_plus": nu_plus, "nu_minus": nu_minus,
                                    "head_to_front_rate_ratio": ratio, "states": state_records,
                                    "minimum_encoder_improvement": float(np.min(risks_b[0]-risks_b[1])),
                                    "minimum_encoder_overshoot": float(np.min(risks_b[2]-risks_b[0])),
                                    "losses_by_noise": {str(k): val for k, val in losses_by_noise.items()}})
    for key, value in maxima.items():
        if key != "identity_metric_mismatch_on_anisotropic_states":
            assert value < TOLERANCE, (key, value)
    assert maxima["identity_metric_mismatch_on_anisotropic_states"] > 1e-6
    figure_paths = make_figure()
    result = {"scope": "Closed-form states only; no training, ODE integration, Monte Carlo sampling, or data downloads",
              "source_theorem": (ROOT/"paper"/"sections"/"bottleneck_family.tex").as_posix(),
              "grid": {"nu_plus": [.95, 1., 1.05], "nu_minus": [.95, 1., 1.05],
                       "head_to_front_rate_ratio": [0., .1, 1., 10.], "front_rate": 1.,
                       "epsilon": EPSILON, "w0": W0, "noise_power": [0., .1, .2],
                       "orientation_radians": [0., .37], "section_theta_fractions": [1., .75, 1/6],
                       "task_psi_fractions": [2/3, 17/24, 3/4, 19/24, 5/6], "task_z": z_values.tolist()},
              "state_count": state_count, "autograd_state_noise_checks": gradient_count,
              "tasks_per_state": len(tasks), "absolute_check_tolerance": TOLERANCE,
              "velocity_comparison": "Actual Euclidean h,W gradients from Gaussian physical quadratic moments, compared with derivatives of closed lambda,w states; float64 autograd",
              "autograd_execution": "Separate process of this same script to avoid this environment's NumPy/PyTorch OpenMP DLL collision; no duplicate-runtime override",
              "covariance_metric_check": "Gaussian quadratic covariance of whitened u by exact fourth-moment identity; rotated Sigma tests the physical nonidentity parameter metric",
              "uniform_gap_constants": {"delta1": DELTA1, "delta2": DELTA2},
              "max_errors": maxima, "minimum_observed_grid_quantities": minima,
              "maximum_clean_energy": max_energy, "maximum_view_energy": max_view_energy,
              "figure_paths": figure_paths,
              "limitations": ["A finite grid verifies implementation and coefficients; the paper proof establishes continuous parameter ranges",
                              "Risk values are exact-population SVD computations, not finite-sample OLS",
                              "The plot depicts an analytic theta path with z=0, not a trained natural-image trajectory"],
              "records": records, "all_checks_passed": True}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state_count", "autograd_state_noise_checks", "uniform_gap_constants", "max_errors", "minimum_observed_grid_quantities", "maximum_clean_energy", "maximum_view_energy", "all_checks_passed")}, indent=2))


if __name__ == "__main__":
    if "--autograd-worker" in sys.argv:
        run_autograd_worker()
    else:
        try:
            main()
        finally:
            close_autograd_worker()
