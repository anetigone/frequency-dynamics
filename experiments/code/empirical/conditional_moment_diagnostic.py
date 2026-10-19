"""Exploratory finite-dataset moment reference for the conditional drift diagnostic.

Exact view moments conditional on the finite clean dataset; a large-batch GUPA
reference, not the exact mean gradient of nonlinear finite-batch GUPA.
"""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import json

import torch

from full_empirical import ROOT, learning_factors
from sgd_state_tools import restore, training_data
from square_images import gupa, population_loss
from subspace_mechanism import projector


def view_moments(data, noise=.2):
    energies = data.square().sum(-1) / 2
    mean = energies.mean(0)
    centered = energies - mean
    covariance = centered.T @ centered / len(energies)
    noise_variance = 2 * noise * mean + noise**2
    s = covariance + torch.diag(noise_variance)
    d = 2 * torch.diag(noise_variance)
    return s, d, covariance


def finite_step(model, conditional):
    with torch.no_grad():
        p0 = projector(model.effective()[1])[0]
    a, _ = model.effective()
    if conditional is None:
        loss = population_loss(a, .2)
    else:
        s, d = conditional
        loss = gupa(a @ s @ a.T, .5 * torch.trace(a @ d @ a.T))
    grads = torch.autograd.grad(loss, model.parameters())
    with torch.no_grad():
        for p, gradient, factor in zip(model.parameters(), grads, learning_factors(model, 1.)):
            p.add_(gradient, alpha=-.002 * factor)
        return projector(model.effective()[1])[0] - p0


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    diagnostic = json.loads((result_path("batch_drift_diagnostic") / "results.json").read_text())
    rows = []
    for seed in [0, 1]:
        data = training_data(seed)
        s, d, covariance = view_moments(data)
        for mode in ["square", "linear", "frozen_front"]:
            path = result_path("full_sgd_development_v2") / f"seed_{seed}_{mode}_r1_stream0" / "checkpoint_2000.pt"
            isotropic, _ = restore(path)
            conditional, _ = restore(path)
            pi = finite_step(isotropic, None)
            pc = finite_step(conditional, (s, d))
            measured = next(r for r in diagnostic["rows"] if r["seed"] == seed and r["mode"] == mode and r["batch_size"] == 16384)
            mean = torch.tensor(measured["mean_delta_projector"])
            rows.append({"seed": seed, "mode": mode, "batch": 16384,
                         "clean_covariance_deviation_from_identity": float((covariance - torch.eye(16)).norm()),
                         "observed_mean_step_norm": float(mean.norm()),
                         "population_reference_norm": float(pi.norm()),
                         "conditional_moment_reference_norm": float(pc.norm()),
                         "error_isotropic_reference": float((mean - pi).norm()),
                         "error_conditional_moment_reference": float((mean - pc).norm()),
                         "rms_sampling_error_of_mean": measured["rms_sampling_error_of_mean"]})
    # Verify the conditional moment formula independently by direct view draws.
    data = training_data(0)
    generator = torch.Generator().manual_seed(1700013)
    clean = data[torch.randint(len(data), (131072,), generator=generator)]
    v1, v2 = [clean + .2**.5 * torch.randn(clean.shape, generator=generator) for _ in range(2)]
    e1, e2 = v1.square().sum(-1) / 2, v2.square().sum(-1) / 2
    z1, z2 = e1 - e1.mean(0), e2 - e2.mean(0)
    s_emp = (z1.T @ z1 + z2.T @ z2) / (2 * (len(clean) - 1))
    d_emp = (e1 - e2).T @ (e1 - e2) / len(clean)
    s, d, _ = view_moments(data)
    checks = {"S_relative_monte_carlo_error": float((s_emp - s).norm() / s.norm()),
              "D_relative_monte_carlo_error": float((d_emp - d).norm() / d.norm())}
    assert max(checks.values()) < .03
    result = {"scope": "post-confirmation exploratory explanation; not a newly held-out controller validation",
              "formula": "S_N=Cov_N(e)+diag(2*eta*mean_N(e)+eta^2); D_N=2*diag(2*eta*mean_N(e)+eta^2)",
              "limitation": "moment plug-in GUPA differs from mean nonlinear minibatch gradient", "checks": checks, "rows": rows}
    output = result_path("batch_drift_diagnostic") / "conditional_moment_reference.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
