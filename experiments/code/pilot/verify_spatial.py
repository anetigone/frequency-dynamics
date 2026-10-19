"""Float64 image-network checks. Numerical evidence, not interval certificates."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from mechanism_checks import init, observables
from square_images import SquareImages, population_loss


def make_model(c=1., rho=.4, phase=0., backend="spatial"):
    model = SquareImages(8, [(1, 0), (0, 2)], [1., 2.], 2, backend=backend)
    y = init(c, rho)
    h = torch.tensor(y[2:], dtype=torch.float64).reshape(2, 2)
    coefficients = torch.stack((h * math.cos(phase), h * math.sin(phase)), -1)
    model.set_spectral(coefficients, torch.tensor([y[:2]], dtype=torch.float64))
    return model


def measurements(model, noise=0.):
    a, b = model.effective()
    energy = a.square().sum()
    risk = a[0, 1].square() / energy
    loss = population_loss(a, noise)
    return a, b, energy, risk, loss


def gradient_check():
    records = []
    for c in [.7, .85, (47 / 96)**(1 / 6), .92, 1., 1.3]:
        for phase in [0., .37]:
            model = make_model(c, phase=phase)
            a, b, v, risk, loss = measurements(model)
            gradients = torch.autograd.grad(loss, (model.kernel, model.weight), retain_graph=True)
            dr = torch.autograd.grad(risk, (model.kernel, model.weight), retain_graph=True)
            dv = torch.autograd.grad(v, (model.kernel, model.weight))
            slope = sum(-(x * y).sum() for x, y in zip(dr, gradients)) / (
                sum(-(x * y).sum() for x, y in zip(dv, gradients)) / v)
            predicted = (235 - 480 * c**6) / (834 + 1300 * c**6)
            assert abs(float(slope.detach()) - predicted) < 2e-12
            spec = model.spectral_coefficients().detach()
            assert (b.detach() - model.g * spec.square().sum(-1)).abs().max() < 2e-15
            if phase == 0:
                analytical = observables(init(c, .4), [1., 2.], [1., 1.], [0., 0.])
                projected = torch.einsum("cij,dpij->cdp", -gradients[0][:, 0], model.basis)
                expected = torch.tensor(analytical["field"][2:]).reshape(2, 2)
                assert (projected[..., 0] - expected).abs().max() < 2e-12
                assert (-gradients[1][0] - torch.tensor(analytical["field"][:2])).abs().max() < 2e-12
            records.append({"c": c, "phase": phase, "predicted_slope": predicted,
                            "image_autograd_slope": float(slope.detach())})
    return records


def forward_check():
    generator = torch.Generator().manual_seed(123)
    model = make_model(phase=.37)
    coefficients = torch.randn(128, 2, 2, dtype=torch.float64, generator=generator)
    images = model.images(coefficients)
    a, b = model.effective()
    q = coefficients.square().sum(-1) / 2 - 1
    centered = model(images) - (model.weight @ b.sum(1))
    forward_error = float((centered - q @ a.T).abs().max().detach())
    assert forward_error < 2e-14
    spatial = model(images)
    spatial_grads = torch.autograd.grad(spatial.square().mean(), (model.kernel, model.weight))
    model.backend = "fft"
    fft = model(images)
    fft_grads = torch.autograd.grad(fft.square().mean(), (model.kernel, model.weight))
    backend_error = float((spatial - fft).abs().max().detach())
    gradient_error = max(float((x - y).abs().max()) for x, y in zip(spatial_grads, fft_grads))
    assert backend_error < 2e-14 and gradient_error < 2e-14
    return {"centered_forward_max_error": forward_error,
            "spatial_fft_output_max_error": backend_error,
            "spatial_fft_gradient_max_error": gradient_error}


def scale_equivalence():
    records = []
    for c in [.7, 1.3]:
        scaled, base = make_model(c), make_model()
        optimizers = [torch.optim.SGD([{"params": [scaled.kernel]}, {"params": [scaled.weight]}], lr=.01),
                      torch.optim.SGD([{"params": [base.kernel], "lr": .01 * c**-2},
                                       {"params": [base.weight], "lr": .01 * c**4}], lr=.01)]
        maximum = 0.
        for _ in range(100):
            for model, optimizer in zip((scaled, base), optimizers):
                optimizer.zero_grad()
                population_loss(model.effective()[0], .2).backward()
                optimizer.step()
            with torch.no_grad():
                error = max(float((scaled.kernel - c * base.kernel).abs().max()),
                            float((scaled.weight - c**-2 * base.weight).abs().max()))
                maximum = max(maximum, error)
        assert maximum < 2e-13
        records.append({"c": c, "steps": 100, "max_parameter_error": maximum})
    return records


def gd_sections(delta, noise):
    """Actual pixel-parameter GD with a recorded positive, variable physical step.

    Step sizes target small changes in log energy. No ODE parameter integration,
    risk gradient, head rescaling or energy constraint projection is used.
    """
    model = make_model()
    with torch.no_grad():
        initial_v = float(measurements(model, noise)[2])
    rows, steps, physical_time = [], 0, 0.
    max_q, last_loss = 0., math.inf
    for target in [0., 1.7, 6.]:
        while True:
            a, b, v, risk, loss = measurements(model, noise)
            ell = math.log(float(v.detach()) / initial_v)
            max_q = max(max_q, (1 + noise)**2 * float(v.detach()))
            assert max_q < 1, "Outside theorem's inactive-constraint regime"
            assert float(loss.detach()) <= last_loss + 1e-12, "Loss did not descend"
            last_loss = float(loss.detach())
            if target - ell < 1e-9:
                rows.append({"ell": ell, "target_ell": target, "risk": float(risk.detach()),
                             "loss": float(loss.detach()), "energy": float(v.detach()),
                             "steps": steps, "physical_time": physical_time})
                break
            gradients = torch.autograd.grad(loss, (model.kernel, model.weight), retain_graph=True)
            dv = torch.autograd.grad(v, (model.kernel, model.weight))
            rate = float((sum(-(x * y).sum() for x, y in zip(dv, gradients)) / v).detach())
            assert rate > 0
            dt = min(delta, .5 * (target - ell)) / rate
            with torch.no_grad():
                model.kernel.add_(gradients[0], alpha=-dt)
                model.weight.add_(gradients[1], alpha=-dt)
            physical_time += dt
            steps += 1
            if steps > 200000:
                raise RuntimeError("Section not reached")
    assert rows[0]["risk"] - rows[1]["risk"] > .04
    assert rows[2]["risk"] - rows[0]["risk"] > .15
    return {"delta_log_energy": delta, "noise": noise, "max_view_energy": max_q, "sections": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=result_path("spatial"))
    parser.add_argument("--step", type=float, default=.004)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Floating-point real image-network verification, not a new certificate",
              "torch": torch.__version__, "device": "cpu", "dtype": "float64",
              "kernel_metric": "orthonormal real half-spectrum; correlation / sqrt(HW/2)",
              "forward": forward_check(), "slopes": gradient_check(),
              "scale_equivalence": scale_equivalence()}
    trajectories = []
    for noise in [0., .2]:
        for delta in [args.step, args.step / 2]:
            print(f"Real spatial GD: noise={noise}, log-energy step={delta}", flush=True)
            trajectories.append(gd_sections(delta, noise))
    from representation_checks import run
    reference = run(init(1., .4), [1., 2.], step=.002)
    result["reference_ode"] = reference
    result["trajectories"] = trajectories
    errors = []
    for i in [0, 2]:
        coarse, fine = trajectories[i:i + 2]
        section_errors = []
        for j, key in [(1, "1.7"), (2, "6.0")]:
            target_risk = reference["checkpoints"][key]["projection_excess_risk"]
            coarse_error = abs(coarse["sections"][j]["risk"] - target_risk)
            fine_error = abs(fine["sections"][j]["risk"] - target_risk)
            assert fine_error < coarse_error and fine_error < .003
            section_errors.append({"ell": float(key), "coarse_error": coarse_error,
                                   "fine_error": fine_error})
        errors.append({"noise": coarse["noise"], "sections": section_errors})
    result["step_halving"] = errors
    (args.output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "initial_slopes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["slopes"][0]))
        writer.writeheader()
        writer.writerows(result["slopes"])
    print(json.dumps({"forward": result["forward"], "step_halving": errors}, indent=2))


if __name__ == "__main__":
    main()
