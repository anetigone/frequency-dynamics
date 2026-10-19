"""Paired periodic-texture pilot: task exchange and prospective interventions.

Examples (from repository root):
  python experiments/code/pilot/train_textures.py --objective population --device cpu --dtype float64
  python experiments/code/pilot/train_textures.py --objective empirical --device cuda --dtype float32

No labels enter pretraining. The population-risk intervention is an oracle
mechanism check, not a label-free model-selection algorithm.
"""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from square_images import (SquareImages, empirical_loss, excess_risks,
                           population_loss, smooth_risk, texture_spec, texture_tasks)


def initialize(args, seed):
    dtype = getattr(torch, args.dtype)
    modes, _ = texture_spec()
    model = SquareImages(32, modes, [1.] * 16, args.channels, args.outputs,
                         kernel_size=args.kernel_size, dtype=dtype, device=args.device,
                         backend=args.backend)
    # CPU generator preserves paired initialization across devices.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if args.kernel_size == 32:
            h = torch.randn(args.channels, 16, 2, dtype=torch.float64, generator=generator) / 4
            w = torch.randn(args.outputs, args.channels, dtype=torch.float64, generator=generator)
            model.set_spectral(h.to(model.kernel), w.to(model.weight))
        else:
            model.kernel.copy_(torch.randn(model.kernel.shape, dtype=torch.float64,
                                           generator=generator).to(model.kernel) / args.kernel_size)
            model.weight.copy_(torch.randn(model.weight.shape, dtype=torch.float64,
                                           generator=generator).to(model.weight))
        a, _ = model.effective()
        model.weight.mul_(math.sqrt(args.initial_energy / float(a.square().sum())))
    model.initialization_scale = 1.
    if getattr(args, "balance_init", False):
        # Label-free, predetermined calibration: equalize traces of the two
        # parameter-Jacobian Gram contributions. Preserve the effective mapping.
        a, _ = model.effective()
        traces = [0., 0.]
        for value in a.reshape(-1):
            derivatives = torch.autograd.grad(value, (model.kernel, model.weight), retain_graph=True)
            for j, derivative in enumerate(derivatives):
                traces[j] += float(derivative.square().sum())
        c = (traces[0] / traces[1])**(1 / 6)
        with torch.no_grad():
            original = a.detach().clone()
            model.kernel.mul_(c)
            model.weight.mul_(c**-2)
            assert torch.allclose(model.effective()[0], original, rtol=2e-5, atol=1e-9)
        model.initialization_scale = c
    return model


@torch.no_grad()
def record(model, tasks, names, step, initial_energy, noise, lr, ratio):
    a, b = model.effective()
    energy = float(a.square().sum())
    risks, values, rank = excess_risks(a, tasks)
    enc_risks, enc_values, enc_rank = excess_risks(b, tasks)
    row = {"step": step, "physical_time": step * lr,
           "ell": math.log(energy / initial_energy), "energy": energy,
           "view_energy": (1 + noise)**2 * energy,
           "population_loss": float(population_loss(a, noise)),
           "ratio": ratio, "projection_rank": rank, "encoder_rank": enc_rank,
           "projection_min_singular": float(values.min()),
           "encoder_min_singular": float(enc_values.min())}
    row.update({"risk_" + name: float(value) for name, value in zip(names, risks)})
    row.update({"encoder_risk_" + name: float(value) for name, value in zip(names, enc_risks)})
    return row


def prospective_intervention(model, tasks, names, noise, ratios, base_lr, prediction_path=None):
    """Write predictions before branch updates; decreasing dt tests local accuracy."""
    # Preserve checkpoint values exactly, then measure tiny risk changes in float64.
    # Otherwise float32 SVD rounding can dominate threshold-bracketing branches.
    if model.weight.dtype != torch.float64 or model.weight.device.type != "cpu":
        model = copy.deepcopy(model).to(device="cpu", dtype=torch.float64)
        tasks = tasks.to(device="cpu", dtype=torch.float64)
    a, _ = model.effective()
    _, _, rank = excess_risks(a.detach(), tasks)
    if rank != len(a):
        return {"status": "skipped_rank_deficient", "rank": rank}
    loss = population_loss(a, noise)
    parameters = (model.kernel, model.weight)
    gl = torch.autograd.grad(loss, parameters, retain_graph=True)
    coefficients = []
    for name, beta in zip(names, tasks):
        gr = torch.autograd.grad(smooth_risk(a, beta), parameters, retain_graph=True)
        dh, dw = [float(-(x * y).sum()) for x, y in zip(gr, gl)]
        coefficients.append((name, dh, dw))
    # Predetermined oracle rule: bracket every positive predicted threshold.
    # No branch outcome is inspected to choose these intervention ratios.
    tested_ratios = set(ratios)
    for _, dh, dw in coefficients:
        if dh * dw < 0:
            critical = -dh / dw
            tested_ratios.update([critical / 2, critical * 2])
    tested_ratios = sorted(tested_ratios)
    predictions = []
    for name, dh, dw in coefficients:
        for ratio in tested_ratios:
            fh, fw = 2 / (1 + ratio), 2 * ratio / (1 + ratio)
            predictions.append({"task": name, "ratio": ratio,
                                "kernel_coefficient": dh, "head_coefficient": dw,
                                "predicted_derivative": fh * dh + fw * dw,
                                "positive_flip_ratio": -dh / dw if dh * dw < 0 else None,
                                "can_flip_with_positive_ratio": dh * dw < 0})
    if prediction_path is not None:
        prediction_path.write_text(json.dumps({"rule": "base ratios plus half and twice every positive analytic threshold",
                                               "predictions": predictions}, indent=2), encoding="utf-8")
    # Only after all predictions have been fixed do we evaluate branch outcomes.
    baseline, _, _ = excess_risks(a.detach(), tasks)
    outcomes = []
    for ratio in tested_ratios:
        for fraction in [1., .5, .25]:
            dt = base_lr * fraction
            branch = copy.deepcopy(model)
            with torch.no_grad():
                branch.kernel.add_(gl[0], alpha=-dt * 2 / (1 + ratio))
                branch.weight.add_(gl[1], alpha=-dt * 2 * ratio / (1 + ratio))
                measured, _, _ = excess_risks(branch.effective()[0], tasks)
            for j, name in enumerate(names):
                prediction = next(p for p in predictions if p["task"] == name and p["ratio"] == ratio)
                observed = float((measured[j] - baseline[j]) / dt)
                outcomes.append({"task": name, "ratio": ratio, "dt": dt,
                                 "observed_derivative": observed,
                                 "predicted_derivative": prediction["predicted_derivative"],
                                 "absolute_derivative_error": abs(observed - prediction["predicted_derivative"])})
    return {"status": "complete", "scope": "oracle task-risk, population-vector-field, one-step branches",
            "evaluation_precision": "float64 CPU, exact promoted copy of checkpoint parameters",
            "predictions": predictions, "outcomes": outcomes}


@torch.no_grad()
def ols_check(model, tasks, names, seed):
    # Independent training/test draws; sample noise is shared across conditions.
    generator = torch.Generator().manual_seed(seed + 70000)
    train = torch.randn(8192, 16, 2, dtype=torch.float64, generator=generator)
    test = torch.randn(32768, 16, 2, dtype=torch.float64, generator=generator)
    qtrain = (train.square().sum(-1) / 2 - 1).to(model.weight)
    qtest = (test.square().sum(-1) / 2 - 1).to(model.weight)
    a, _ = model.effective()
    # Forward equivalence is independently checked on actual images. Using Aq
    # here avoids repeatedly convolving the same large probe datasets.
    xtrain = qtrain @ a.T
    xtest = qtest @ a.T
    xtrain = torch.cat((torch.ones_like(xtrain[:, :1]), xtrain), 1).double()
    xtest = torch.cat((torch.ones_like(xtest[:, :1]), xtest), 1).double()
    ytrain, ytest = (qtrain @ tasks.T).double(), (qtest @ tasks.T).double()
    pop, _, _ = excess_risks(a, tasks)
    rows = []
    # Solve on CPU with SVD; no ridge, including a fitted intercept.
    for n in [256, 1024, 8192]:
        coefficients = torch.linalg.lstsq(xtrain[:n].cpu(), ytrain[:n].cpu(), driver="gelsd").solution
        mse = ((xtest.cpu() @ coefficients - ytest.cpu())**2).mean(0)
        for j, name in enumerate(names):
            rows.append({"labels": n, "task": name, "test_mse": float(mse[j]),
                         "population_risk": float(pop[j])})
    return rows


def run(args, seed, ratio):
    start = time.perf_counter()
    model = initialize(args, seed)
    run_dir = args.output / f"seed_{seed}_ratio_{ratio:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    names, tasks = texture_tasks(model.weight.dtype, model.weight.device)
    with torch.no_grad():
        initial_a = model.effective()[0].clone()
        initial_energy = float(initial_a.square().sum())
    torch.save({"kernel": model.kernel.detach().cpu(), "weight": model.weight.detach().cpu()},
               run_dir / "initial.pt")
    generator = torch.Generator().manual_seed(seed + 10000)
    data = torch.randn(args.train_size, 16, 2, dtype=torch.float64, generator=generator).to(model.weight)
    # Batch indices and augmentation draws are reset for each ratio.
    draws = torch.Generator(device=args.device).manual_seed(seed + 20000)
    optimizer = torch.optim.SGD([{"params": [model.kernel], "lr": args.lr * 2 / (1 + ratio)},
                                {"params": [model.weight], "lr": args.lr * 2 * ratio / (1 + ratio)}], lr=args.lr)
    history, intervention, initial_intervention = [], None, None
    if args.interventions:
        initial_intervention = prospective_intervention(model, tasks, names, args.noise, args.ratios, args.lr,
                                                       run_dir / "initial_predictions.json")
    reason, empirical_value, last_record = "step_budget", None, -1
    for step in range(args.steps + 1):
        with torch.no_grad():
            a, _ = model.effective()
            energy = float(a.square().sum())
            ell = math.log(energy / initial_energy)
            view_energy = (1 + args.noise)**2 * energy
        if not math.isfinite(view_energy):
            raise FloatingPointError(f"Nonfinite trajectory seed={seed}, ratio={ratio}, step={step}")
        if view_energy >= args.energy_cap:
            reason = "energy_cap_reached_no_rescaling"
        elif ell >= args.max_ell:
            reason = "target_energy_reached"
        terminal = reason != "step_budget" or step == args.steps
        if step % args.record_every == 0 or terminal or (args.interventions and intervention is None and ell >= 1.):
            row = record(model, tasks, names, step, initial_energy, args.noise, args.lr, ratio)
            row["empirical_batch_loss"] = empirical_value
            history.append(row)
            last_record = step
        if args.interventions and intervention is None and ell >= 1.:
            intervention = {"step": step, "ell": ell,
                            **prospective_intervention(model, tasks, names, args.noise, args.ratios, args.lr,
                                                       run_dir / "ell_one_predictions.json")}
            torch.save({"kernel": model.kernel.detach().cpu(), "weight": model.weight.detach().cpu(),
                        "step": step, "ell": ell}, run_dir / "ell_one.pt")
        if terminal:
            break
        optimizer.zero_grad(set_to_none=True)
        if args.objective == "population":
            loss = population_loss(model.effective()[0], args.noise)
        else:
            indices = torch.randint(len(data), (args.batch_size,), generator=draws, device=args.device)
            clean = data[indices]
            n1 = torch.randn(clean.shape, generator=draws, dtype=clean.dtype, device=args.device)
            n2 = torch.randn(clean.shape, generator=draws, dtype=clean.dtype, device=args.device)
            z1 = model(model.images(clean + math.sqrt(args.noise) * n1))
            z2 = model(model.images(clean + math.sqrt(args.noise) * n2))
            loss = empirical_loss(z1, z2)
            empirical_value = float(loss.detach())
        loss.backward()
        optimizer.step()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.synchronize()
    with (run_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with torch.no_grad():
        final_a, final_b = model.effective()
    torch.save({"kernel": model.kernel.detach().cpu(), "weight": model.weight.detach().cpu(),
                "A": final_a.cpu(), "B": final_b.cpu(), "initial_A": initial_a.cpu(),
                "modes": model.modes, "g": model.g.cpu()}, run_dir / "final.pt")
    metadata = {"seed": seed, "ratio": ratio, "initial_A": initial_a.cpu().tolist(),
                "initialization_scale": model.initialization_scale,
                "stop_reason": reason, "updates": last_record, "elapsed_seconds": time.perf_counter() - start,
                "initial": history[0], "final": history[-1],
                "initial_intervention": initial_intervention, "ell_one_intervention": intervention,
                "ols": ols_check(model, tasks, names, seed) if args.ols else None,
                "constraint": "stop after first crossing, no projection; inspect final view_energy"}
    (run_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"seed={seed} ratio={ratio:g} steps={last_record} ell={history[-1]['ell']:.3f} "
          f"stop={reason} seconds={metadata['elapsed_seconds']:.1f}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=["population", "empirical"], default="population")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--backend", choices=["fft", "spatial"], default="fft")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--outputs", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--ratios", type=float, nargs="+", default=[.1, 1., 10.])
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--noise", type=float, default=.2)
    parser.add_argument("--initial-energy", type=float, default=.001)
    parser.add_argument("--max-ell", type=float, default=3.)
    parser.add_argument("--energy-cap", type=float, default=1.)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=32768)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--ols", action="store_true")
    parser.add_argument("--balance-init", action="store_true",
                        help="Preserve A0; equalize traces of kernel/head Jacobian Gram contributions without labels")
    parser.add_argument("--output", type=Path, default=result_path("textures_pilot"))
    args = parser.parse_args()
    if not 0 < args.kernel_size <= 32 or not 1 <= args.outputs <= args.channels <= 16:
        parser.error("Require 1 <= outputs <= channels <= 16 and 1 <= kernel size <= 32")
    if any(r <= 0 for r in args.ratios) or args.lr <= 0 or args.noise < 0:
        parser.error("Positive learning rates and nonnegative noise required")
    if args.initial_energy <= 0 or args.max_ell <= 0 or args.steps < 1 or args.record_every < 1:
        parser.error("Positive energy, progress, steps and recording interval required")
    if args.batch_size < 2 or args.train_size < 1:
        parser.error("Batch size >= 2 and nonempty training data required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                    task_semantics="prespecified synthetic frequency-power attributes, not natural classes",
                    learning_rates="eta_h=2*lr/(1+r), eta_W=2*lr*r/(1+r); fixed sum 2*lr",
                    pilot=True)
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runs = []
    for seed in args.seeds:
        paired_initial = None
        for ratio in args.ratios:
            result = run(args, seed, ratio)
            if paired_initial is None:
                paired_initial = result["initial_A"]
            elif result["initial_A"] != paired_initial:
                raise AssertionError("Paired initial mappings differ")
            runs.append({k: result[k] for k in ["seed", "ratio", "stop_reason", "updates", "elapsed_seconds"]})
    (args.output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
