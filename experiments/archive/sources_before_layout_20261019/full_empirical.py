"""End-to-end finite-dataset empirical SGD, independently calibrated factors.

Exact periodic power forward identity; pixel kernels remain trainable. No
population gradients, energy-adaptive steps, rescaling or early energy stops.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from square_images import empirical_loss, excess_risks, population_loss, texture_tasks
from subspace_mechanism import Factors, ROOT, projector
from train_textures import initialize


def build(seed, mode, ratio, kernel=32, outputs=1):
    config = SimpleNamespace(dtype="float64", device="cpu", channels=8, outputs=outputs,
                             kernel_size=kernel, backend="fft", initial_energy=.001, balance_init=True)
    model = Factors(initialize(config, seed), mode)
    calibration = model.calibrate("independently_calibrated")
    th, tw = model.traces()
    model.mobility = 1 / (2 / (1 + ratio) * th + 2 * ratio / (1 + ratio) * tw)
    calibration.update(weighted_initial_trace=1., active_mobility=model.mobility)
    return model, calibration


def learning_factors(model, ratio):
    front, head = 2 * model.mobility / (1 + ratio), 2 * ratio * model.mobility / (1 + ratio)
    return [head] if model.mode == "frozen_front" else [front, head]


def batch_loss(model, energies1, energies2):
    a, _ = model.effective()
    return empirical_loss(energies1 @ a.T, energies2 @ a.T)


def update(model, energies1, energies2, lr, ratio):
    loss = batch_loss(model, energies1, energies2)
    parameters = model.parameters()
    gradients = torch.autograd.grad(loss, parameters)
    with torch.no_grad():
        for p, g, factor in zip(parameters, gradients, learning_factors(model, ratio)):
            p.add_(g, alpha=-lr * factor)
    return float(loss.detach())


def energy_tensors(model, front, weight):
    if model.mode == "linear":
        b = front
    else:
        k = front.shape[-1]
        h = torch.einsum("cij,dpij->cdp", front[:, 0], model.image.basis[..., :k, :k])
        b = model.image.g * h.square().sum(-1)
    return weight @ b, b


@torch.no_grad()
def metrics(a, b, initial, tasks, names, matrices=False):
    pa, sa, ra, _ = projector(a)
    pb, sb, rb, _ = projector(b)
    rp, _, _ = excess_risks(a, tasks)
    re, _, _ = excess_risks(b, tasks)
    energy = float(a.square().sum())
    if not math.isfinite(energy) or energy <= 0:
        raise FloatingPointError("Nonfinite/zero effective energy")
    row = {"ell": math.log(energy / initial["energy"]), "energy": energy,
           "view_energy": 1.44 * energy, "inside_T_energy_region": 1.44 * energy < 1,
           "population_loss_for_evaluation_only": float(population_loss(a, .2)),
           "encoder_displacement": float((pb - initial["pb"]).norm()),
           "projection_displacement": float((pa - initial["pa"]).norm()),
           "encoder_rank": rb, "projection_rank": ra,
           "encoder_singular_values": sb.tolist(), "projection_singular_values": sa.tolist(),
           "encoder_risks": dict(zip(names, re.tolist())), "projection_risks": dict(zip(names, rp.tolist())),
           "encoder_coordinate_risk_sum": float((1 - pb.diag()).sum()),
           "projection_coordinate_risk_sum": float((1 - pa.diag()).sum())}
    assert abs(row["encoder_coordinate_risk_sum"] - (16 - rb)) < 1e-10
    assert abs(row["projection_coordinate_risk_sum"] - (16 - ra)) < 1e-10
    if matrices:
        row.update(encoder_projector=pb.tolist(), projection_projector=pa.tolist())
    return row


@torch.no_grad()
def interpolate_section(model, old_front, old_weight, target, initial, tasks, names):
    new_front, new_weight = model.front.detach(), model.weight.detach()
    low, high = 0., 1.
    for _ in range(36):
        mid = (low + high) / 2
        a, b = energy_tensors(model, old_front.lerp(new_front, mid), old_weight.lerp(new_weight, mid))
        if math.log(float(a.square().sum()) / initial["energy"]) < target:
            low = mid
        else:
            high = mid
    alpha = (low + high) / 2
    a, b = energy_tensors(model, old_front.lerp(new_front, alpha), old_weight.lerp(new_weight, alpha))
    return {"target_ell": target, "fraction_of_update": alpha,
            "evaluation": "linear parameter interpolation inside first upward-crossing SGD update",
            **metrics(a, b, initial, tasks, names, True)}


def save_checkpoint(path, model, step, seed, ratio, kernel, stream):
    torch.save({"front": model.front.detach(), "weight": model.weight.detach(), "step": step,
                "seed": seed, "mode": model.mode, "ratio": ratio, "kernel_size": kernel,
                "mobility": model.mobility, "stream": stream}, path)


def train_seed(args, seed):
    names, tasks = texture_tasks(torch.float64, "cpu")
    generator = torch.Generator().manual_seed(510000 + seed)
    data = torch.randn(32768, 16, 2, generator=generator)
    # stream changes minibatch/augmentation draws, never parameters or dataset.
    draws = torch.Generator().manual_seed(610000 + seed * 100 + args.stream)
    models = []
    for mode in args.models:
        for ratio in args.ratios:
            directory = args.output / f"seed_{seed}_{mode}_r{ratio:g}_stream{args.stream}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "result.json").exists():
                continue
            model, calibration = build(seed, mode, ratio, args.kernel_size)
            with torch.no_grad():
                a, b = model.effective()
                initial = {"energy": float(a.square().sum()), "pa": projector(a)[0], "pb": projector(b)[0]}
                row = metrics(a, b, initial, tasks, names, True)
            models.append({"model": model, "ratio": ratio, "directory": directory, "initial": initial,
                           "initial_A": a.detach().tolist(), "initial_B": b.detach().tolist(),
                           "calibration": calibration, "history": [{"step": 0, **row}], "sections": [], "previous_ell": 0.})
            save_checkpoint(directory / "checkpoint_0.pt", model, 0, seed, ratio, args.kernel_size, args.stream)
    if not models:
        return
    start = time.perf_counter()
    if args.device != "cpu":
        raise ValueError("This exact-identity float64 runner is intentionally CPU; CUDA is used for image checks/branches")
    checkpoint_steps = sorted(set([args.steps // 8, args.steps // 2, args.steps * 7 // 8, args.steps]))
    for step in range(1, args.steps + 1):
        clean = data[torch.randint(len(data), (args.batch_size,), generator=draws)]
        c1 = clean + math.sqrt(.2) * torch.randn(clean.shape, generator=draws)
        c2 = clean + math.sqrt(.2) * torch.randn(clean.shape, generator=draws)
        e1, e2 = c1.square().sum(-1) / 2, c2.square().sum(-1) / 2
        for entry in models:
            model = entry["model"]
            # Keep the pre-update parameters only for reporting crossed sections.
            old_front, old_weight = model.front.detach().clone(), model.weight.detach().clone()
            for _ in range(args.microsteps):
                value = update(model, e1, e2, args.lr / args.microsteps, entry["ratio"])
            with torch.no_grad():
                a, b = model.effective()
                ell = math.log(float(a.square().sum()) / entry["initial"]["energy"])
            for target in [.75, 1.5, 2.25, 3.]:
                if entry["previous_ell"] < target <= ell and not any(s["target_ell"] == target for s in entry["sections"]):
                    section = interpolate_section(model, old_front, old_weight, target, entry["initial"], tasks, names)
                    entry["sections"].append({"crossing_step": step, **section})
            entry["previous_ell"] = ell
            if step % args.record_every == 0 or step in checkpoint_steps:
                row = metrics(a, b, entry["initial"], tasks, names, step in checkpoint_steps)
                entry["history"].append({"step": step, "empirical_batch_loss": value, **row})
            if step in checkpoint_steps:
                save_checkpoint(entry["directory"] / f"checkpoint_{step}.pt", model, step, seed,
                                entry["ratio"], args.kernel_size, args.stream)
        if step % 500 == 0:
            print(f"seed={seed} stream={args.stream} step={step}/{args.steps} models={len(models)}", flush=True)
    for entry in models:
        model = entry.pop("model")
        with torch.no_grad():
            if model.mode != "linear":
                a, b = model.effective()
                image_a, image_b = model.image.effective()
                assert (a - image_a).abs().max() < 1e-10
                assert (b - image_b).abs().max() < 1e-10
        entry.pop("initial")
        directory = entry.pop("directory")
        entry.update(seed=seed, mode=model.mode, stream=args.stream,
                     seconds_for_seed_group=time.perf_counter() - start,
                     training_data_sha256=hashlib.sha256(data.numpy().tobytes()).hexdigest())
        (directory / "result.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    print(f"seed={seed} stream={args.stream} complete seconds={time.perf_counter()-start:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--models", nargs="+", choices=["square", "linear", "frozen_front"], default=["square", "linear", "frozen_front"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[.1, 1., 10.])
    parser.add_argument("--kernel-size", type=int, choices=[32, 11], default=32)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--microsteps", type=int, default=1,
                        help="Development step refinement: reuse a batch across positive substeps")
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=["development", "confirmation", "repeat"], default="development")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "full_sgd_development")
    args = parser.parse_args()
    if args.phase == "confirmation" and (args.seeds != list(range(2, 12)) or args.stream != 0 or args.microsteps != 1):
        raise ValueError("Confirmation requires seeds 2..11, stream 0 and ordinary one-step SGD")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(initialization="independent per-model equal Gram traces, weighted initial total trace=1",
                    primary_common_energy=1.5, primary_fixed_step=args.steps,
                    training_set_size=32768, noise=.2, dtype="float64", initial_energy=.001,
                    stopping="fixed update budget; no rescaling, clipping or energy early stopping",
                    theorem_boundary="log view energy explicitly; observations outside Q<1 do not validate theorem T",
                    checkpoint_steps=sorted(set([args.steps // 8, args.steps // 2, args.steps * 7 // 8, args.steps])),
                    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                   for name in ["full_empirical.py", "subspace_mechanism.py", "square_images.py", "train_textures.py"]})
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing locked protocol differs; choose a new output directory")
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for seed in args.seeds:
        train_seed(args, seed)


if __name__ == "__main__":
    main()
