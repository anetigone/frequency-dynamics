"""Held-out real empirical-SGD branches with shared data and augmentation draws.

Population geometry makes predictions BEFORE drawing empirical batches. Known
task beta is still used: this is not a label-free task-selection method.
"""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from square_images import empirical_loss, excess_risks, texture_tasks
from subspace_mechanism import Factors, ROOT, geometry_field, projector, risk_derivative
from train_textures import initialize


HORIZONS = (1, 8, 64)
CHECKPOINTS = (.75, 1.5, 2.25)


@torch.no_grad()
def coefficients(model, tasks, gradient=None):
    if gradient is None:
        field = geometry_field(model, .2)
    else:
        a, b = model.effective()
        db = (-torch.einsum("cij,cj->ci", model.kernel_gram(), model.weight.T @ gradient)
              if model.mode != "frozen_front" else torch.zeros_like(b))
        field = {"A_front": model.weight @ db, "A_head": -gradient @ b.T @ b}
    a, _ = model.effective()
    front = risk_derivative(a, field["A_front"], tasks)
    if gradient is None and model.mode == "linear":
        # Isotropic population G=H A, so A_front is a left multiple of A.
        # Its row-space risk derivative is EXACTLY zero, for any K. Do not
        # mistake roundoff (~1e-17) for an extremely small positive threshold.
        front = torch.zeros_like(front)
    return front, risk_derivative(a, field["A_head"], tasks), field


def critical(dh, dw):
    return -dh / dw if dh * dw < 0 else None


def checkpoint_run(source, output, seed, target, batch_size, device):
    config = SimpleNamespace(dtype="float64", device="cpu", channels=8, outputs=1,
                             kernel_size=32, backend="fft", initial_energy=.001, balance_init=True)
    image = initialize(config, seed)
    saved = torch.load(source / "checkpoints" / f"seed_{seed}_independently_calibrated_square_r1" /
                       f"ell_{target:g}.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        image.kernel.copy_(saved["front"])
        image.weight.copy_(saved["weight"])
    names, tasks = texture_tasks(torch.float64, "cpu")
    primary = names.index("x_high")
    state_dir = output / f"seed_{seed}_ell_{target:g}"
    state_dir.mkdir(parents=True, exist_ok=True)
    models, predictions, branches = {}, {}, []
    for mode in ("square", "linear", "frozen_front"):
        model = Factors(image, mode)
        model.mobility = 1 / sum(model.traces())
        dh, dw, field = coefficients(model, tasks)
        predicted_critical = critical(float(dh[primary]), float(dw[primary]))
        ratios = set([.1, 1., 10.])
        if predicted_critical is not None:
            ratios.update([predicted_critical / 2, predicted_critical * 2])
        with torch.no_grad():
            a, b = model.effective()
            risk0, _, _ = excess_risks(a, tasks)
            p0 = projector(b)[0]
            rates = [float(2 * (a * (2 * model.mobility / (1 + r) * field["A_front"] +
                                    2 * model.mobility * r / (1 + r) * field["A_head"])).sum() /
                           a.square().sum()) for r in ratios]
        dt = .004 / max(rates)  # common within model, chosen without task outcomes.
        models[mode] = {"model": model, "dh": dh, "dw": dw, "risk0": risk0,
                        "p0": p0, "dt": dt, "ratios": sorted(ratios)}
        predictions[mode] = {"dt": dt, "mobility": model.mobility, "ratios": sorted(ratios),
                             "primary_task": "x_high", "primary_critical_ratio": predicted_critical,
                             "task_coefficients": {name: {"front": float(h), "head": float(w),
                                  "critical_ratio": critical(float(h), float(w))}
                                  for name, h, w in zip(names, dh, dw)}}
        for ratio in sorted(ratios):
            branch_model = copy.deepcopy(model)
            if device != "cpu":
                if mode == "linear":
                    branch_model.front = torch.nn.Parameter(branch_model.front.detach().to(device))
                    branch_model.weight = torch.nn.Parameter(branch_model.weight.detach().to(device))
                else:
                    branch_model.image.to(device)
                    branch_model.front = branch_model.image.kernel
                    branch_model.weight = branch_model.image.weight
            branches.append({"mode": mode, "ratio": ratio, "model": branch_model})
    # This file exists before any empirical data/augmentations are generated.
    (state_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    generator = torch.Generator().manual_seed(410000 + seed * 100 + round(target * 4))
    train = torch.randn(32768, 16, 2, generator=generator)
    outcomes, references = [], {}
    for update in range(1, max(HORIZONS) + 1):
        clean = train[torch.randint(len(train), (batch_size,), generator=generator)]
        c1 = clean + math.sqrt(.2) * torch.randn(clean.shape, generator=generator)
        c2 = clean + math.sqrt(.2) * torch.randn(clean.shape, generator=generator)
        x1, x2 = image.images(c1).to(device), image.images(c2).to(device)
        energy1, energy2 = c1.square().sum(-1) / 2, c2.square().sum(-1) / 2
        if update == 1:
            a = image.effective()[0].detach().requires_grad_(True)
            empirical_g = torch.autograd.grad(empirical_loss(energy1 @ a.T, energy2 @ a.T), a)[0]
            for mode, entry in models.items():
                hd, wd, _ = coefficients(entry["model"], tasks, empirical_g)
                references[mode] = {name: {"front": float(h), "head": float(w),
                                          "critical_ratio": critical(float(h), float(w))}
                                    for name, h, w in zip(names, hd, wd)}
        for branch in branches:
            mode, ratio, model = branch["mode"], branch["ratio"], branch["model"]
            entry = models[mode]
            if mode == "linear":
                a, _ = model.effective()
                z1, z2 = energy1.to(device) @ a.T, energy2.to(device) @ a.T
            else:
                # Actual image convolution/square/GAP, not effective coefficients.
                z1, z2 = model.image(x1), model.image(x2)
            loss = empirical_loss(z1, z2)
            parameters = model.parameters()
            gradients = torch.autograd.grad(loss, parameters)
            fh = 2 * model.mobility / (1 + ratio)
            fw = 2 * model.mobility * ratio / (1 + ratio)
            factors = [fw] if mode == "frozen_front" else [fh, fw]
            with torch.no_grad():
                for parameter, grad, factor in zip(parameters, gradients, factors):
                    parameter.add_(grad, alpha=-entry["dt"] * factor)
            if update in HORIZONS:
                with torch.no_grad():
                    a, b = model.effective()
                    a, b = a.cpu(), b.cpu()
                    risks, _, _ = excess_risks(a, tasks)
                    encoder_drift = float((projector(b)[0] - entry["p0"]).norm())
                    observed = risks - entry["risk0"]
                    predicted = update * entry["dt"] * (fh * entry["dh"] + fw * entry["dw"])
                for j, name in enumerate(names):
                    outcomes.append({"mode": mode, "ratio": ratio, "horizon": update, "task": name,
                                     "predicted_population_change": float(predicted[j]),
                                     "observed_empirical_change": float(observed[j]),
                                     "encoder_projector_displacement": encoder_drift,
                                     "view_energy": 1.44 * float(a.square().sum()),
                                     "dt": entry["dt"]})
    critical_fits = []
    for mode in models:
        for name in names:
            first = [o for o in outcomes if o["mode"] == mode and o["task"] == name and
                     o["horizon"] == 1 and o["ratio"] in (.1, 1., 10.)]
            x = torch.tensor([[1., o["ratio"]] for o in first])
            y = torch.tensor([o["observed_empirical_change"] / o["dt"] * (1 + o["ratio"]) /
                              (2 * models[mode]["model"].mobility) for o in first])
            fit = torch.linalg.lstsq(x, y).solution
            critical_fits.append({"mode": mode, "task": name,
                                  "population_prediction": predictions[mode]["task_coefficients"][name]["critical_ratio"],
                                  "empirical_derivative_reference": references[mode][name]["critical_ratio"],
                                  "one_step_fitted_critical": critical(float(fit[0]), float(fit[1]))})
    result = {"seed": seed, "checkpoint_ell": target, "batch_size": batch_size,
              "predictions": predictions, "outcomes": outcomes, "critical_ratios": critical_fits,
              "scope": "real finite-data empirical SGD, shared minibatches/noise; population oracle predictions"}
    (state_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"seed={seed} ell={target:g} branch_models={len(branches)} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=result_path("subspace_confirm_k1"))
    parser.add_argument("--output", type=Path, default=result_path("subspace_empirical_confirm"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2, 12)))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"seeds": args.seeds, "checkpoints": CHECKPOINTS, "horizons": HORIZONS,
                "device": args.device, "dtype": "float64",
                "batch_size": args.batch_size, "training_set_size": 32768, "noise": .2,
                "primary_task_for_ratio_grid": "x_high", "prediction": "closed-form population geometry; no risk autograd",
                "step_rule": "expected initial log-energy increment <=.004 for all selected ratios",
                "models": ["square", "linear", "frozen_front"],
                "source": str(args.source.resolve()), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "initialization": "same checkpoint B,W in all models; global mobility normalized at r=1; no factor rescaling",
                "inference_unit": "paired seed; checkpoints/tasks/branches are dependent"}
    lock = args.output / "protocol.json"
    normalized = json.loads(json.dumps(protocol))
    if lock.exists() and json.loads(lock.read_text()) != normalized:
        raise ValueError("Existing branch protocol differs")
    lock.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for seed in args.seeds:
        for target in CHECKPOINTS:
            if (args.output / f"seed_{seed}_ell_{target:g}" / "result.json").exists():
                continue
            checkpoint_run(args.source, args.output, seed, target, args.batch_size, args.device)


if __name__ == "__main__":
    main()
