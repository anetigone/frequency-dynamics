"""Independent image-autograd validation of the subspace geometry implementation."""
import json
from types import SimpleNamespace

import torch

from square_images import population_loss, smooth_risk, texture_tasks
from subspace_mechanism import (Factors, ROOT, geometry_field, loss_gradient,
                                projector, risk_derivative)
from train_textures import initialize
from empirical_subspace_interventions import coefficients, critical


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rows = []
    for outputs, kernel in [(1, 32), (2, 32), (1, 11), (2, 11)]:
        config = SimpleNamespace(dtype="float64", device="cpu", channels=8, outputs=outputs,
                                 kernel_size=kernel, backend="fft", initial_energy=.001,
                                 balance_init=True)
        base = initialize(config, 0)
        tasks = texture_tasks(torch.float64, "cpu")[1]
        for mode in ["square", "linear", "frozen_front"]:
            model = Factors(base, mode)
            calibration = model.calibrate("independently_calibrated")
            a, b = model.effective()
            params = model.parameters()
            loss = population_loss(a, .2)
            parameter_gradient = torch.autograd.grad(loss, params, retain_graph=True)
            gradient_a = torch.autograd.grad(loss, a, retain_graph=True)[0]
            closed_error = float((gradient_a - loss_gradient(a.detach(), .2)).abs().max())
            field = geometry_field(model, .2)
            gr = torch.autograd.grad(smooth_risk(a, tasks[0]), params, retain_graph=True)
            observed = -sum((x * y).sum() for x, y in zip(gr, parameter_gradient))
            predicted = risk_derivative(a.detach(), field["A_front"] + field["A_head"], tasks[:1])[0]
            derivative_error = abs(float(observed) - float(predicted))
            image_gradient_error = 0.
            if mode != "linear":
                image_a, image_b = model.image.effective()
                image_grads = torch.autograd.grad(population_loss(image_a, .2), params)
                image_gradient_error = max(float((x - y).abs().max())
                                           for x, y in zip(image_grads, parameter_gradient))
                assert (image_b - b).abs().max() < 1e-12
            encoder_prediction = risk_derivative(b.detach(), field["B_front"], tasks)
            if mode in ["linear", "frozen_front"]:
                assert encoder_prediction.abs().max() < 1e-10
            if mode == "linear":
                assert risk_derivative(a.detach(), field["A_front"], tasks).abs().max() < 1e-10
                hd, wd, _ = coefficients(model, tasks)
                assert torch.equal(hd, torch.zeros_like(hd))
                assert all(critical(float(h), float(w)) is None for h, w in zip(hd, wd))
            assert max(closed_error, derivative_error, image_gradient_error) < 1e-10
            rows.append({"outputs": outputs, "kernel_size": kernel, "mode": mode,
                         "closed_form_loss_gradient_error": closed_error,
                         "risk_derivative_error": derivative_error,
                         "pixel_image_gradient_error": image_gradient_error,
                         "calibration": calibration})
    output = ROOT / "results" / "subspace_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"configurations_checked": len(rows), "max_image_gradient_error":
                      max(r["pixel_image_gradient_error"] for r in rows),
                      "max_risk_derivative_error": max(r["risk_derivative_error"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
