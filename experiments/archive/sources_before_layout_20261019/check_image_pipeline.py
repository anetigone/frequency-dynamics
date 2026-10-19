"""Scientific implementation checks for multi-frequency images and risks."""
import json
from pathlib import Path

import torch

from square_images import (SquareImages, empirical_loss, excess_risks,
                           population_loss, smooth_risk, texture_spec)


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(314)
    modes, _ = texture_spec()
    errors = []
    for kernel_size in [32, 11]:
        model = SquareImages(32, modes, [1.] * 16, 4, 2, kernel_size=kernel_size)
        with torch.no_grad():
            model.kernel.normal_(std=.1)
            model.weight.normal_(std=.2)
        coefficients = torch.randn(12, 16, 2)
        images = model.images(coefficients)
        a, b = model.effective()
        q = coefficients.square().sum(-1) / 2 - 1
        z = model(images)
        mapping_error = float((z - (q + 1) @ a.T).abs().max().detach())
        spatial_gradient = torch.autograd.grad(z.square().mean(), (model.kernel, model.weight))
        model.backend = "fft"
        zfft = model(images)
        fft_gradient = torch.autograd.grad(zfft.square().mean(), (model.kernel, model.weight))
        backend_error = float((z - zfft).abs().max().detach())
        gradient_error = max(float((x - y).abs().max()) for x, y in zip(spatial_gradient, fft_gradient))
        assert max(mapping_error, backend_error, gradient_error) < 1e-11
        errors.append({"kernel_size": kernel_size, "forward_error": mapping_error,
                       "backend_error": backend_error, "gradient_error": gradient_error})
    a = torch.randn(2, 16, requires_grad=True) / 10
    beta = torch.randn(3, 16)
    risks, _, rank = excess_risks(a.detach(), beta)
    transform = torch.tensor([[2., .3], [-.1, .7]])
    transformed, _, _ = excess_risks(transform @ a.detach(), beta)
    assert rank == 2 and (risks - transformed).abs().max() < 1e-12
    assert abs(float(smooth_risk(a, beta[0]).detach()) - float(risks[0])) < 1e-12
    # Independent Gaussian image coefficients; estimate objective and gradients.
    n, eta = 131072, .2
    clean = torch.randn(n, 16, 2)
    v1 = clean + eta**.5 * torch.randn_like(clean)
    v2 = clean + eta**.5 * torch.randn_like(clean)
    q1, q2 = v1.square().sum(-1) / 2 - 1, v2.square().sum(-1) / 2 - 1
    exact = population_loss(a, eta)
    empirical = empirical_loss(q1 @ a.T, q2 @ a.T)
    exact_grad = torch.autograd.grad(exact, a)[0]
    empirical_grad = torch.autograd.grad(empirical, a)[0]
    relative_error = float((empirical_grad - exact_grad).norm() / exact_grad.norm())
    assert abs(float(empirical.detach() - exact.detach())) < .01 and relative_error < .06
    result = {"multifrequency": errors, "row_space_invariance_max_error":
                      float((risks - transformed).abs().max()),
                      "empirical_gradient_relative_error": relative_error,
                      "empirical_loss": float(empirical.detach()),
                      "population_loss": float(exact.detach())}
    output = Path(__file__).parent / "results" / "spatial" / "pipeline_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
