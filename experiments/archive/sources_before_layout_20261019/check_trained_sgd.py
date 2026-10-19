"""Real image SGD prefixes vs exact-identity SGD at empirically trained states."""
import copy
import json

import torch

from full_empirical import ROOT, learning_factors, update
from sgd_state_tools import restore
from square_images import empirical_loss


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    results = []
    for folder in ["full_sgd_main", "full_sgd_local"]:
        for mode in ["square", "frozen_front"]:
            path = ROOT / "results" / folder / f"seed_2_{mode}_r1_stream0" / "checkpoint_2000.pt"
            fast, _ = restore(path)
            image = copy.deepcopy(fast)
            image.image.backend = "spatial"
            generator = torch.Generator().manual_seed(1400027)
            maximum_loss, maximum_parameter = 0., 0.
            for _ in range(8):
                clean = torch.randn(32, 16, 2, generator=generator)
                c1 = clean + .2**.5 * torch.randn(clean.shape, generator=generator)
                c2 = clean + .2**.5 * torch.randn(clean.shape, generator=generator)
                fast_value = update(fast, c1.square().sum(-1) / 2, c2.square().sum(-1) / 2, .002, 1.)
                loss = empirical_loss(image.image(image.image.images(c1)), image.image(image.image.images(c2)))
                gradients = torch.autograd.grad(loss, image.parameters())
                with torch.no_grad():
                    for parameter, gradient, factor in zip(image.parameters(), gradients, learning_factors(image, 1.)):
                        parameter.add_(gradient, alpha=-.002 * factor)
                maximum_loss = max(maximum_loss, abs(fast_value - float(loss.detach())))
                maximum_parameter = max(maximum_parameter, max(float((a - b).abs().max()) for a, b in
                                        [(fast.front.detach(), image.front.detach()), (fast.weight.detach(), image.weight.detach())]))
            assert maximum_loss < 1e-11 and maximum_parameter < 1e-11
            results.append({"setting": folder, "mode": mode, "checkpoint_step": 2000,
                            "image_backend": "literal spatial conv2d", "batch_size": 32,
                            "real_image_sgd_updates": 8, "max_loss_error": maximum_loss,
                            "max_parameter_error": maximum_parameter})
    (ROOT / "results" / "trained_sgd_image_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
