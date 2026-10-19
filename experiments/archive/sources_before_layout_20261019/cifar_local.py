"""CIFAR-10 square-CNN pipeline, throughput test and bounded paired experiments.

Existing standard data only; no download. Labels are used for validation probes,
never pretraining. No batch norm, momentum, weight decay or output normalization.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from audit_cifar import audit
from square_images import empirical_loss


ROOT = Path(__file__).parent


class NaturalSquare(nn.Module):
    def __init__(self, channels=64, outputs=8):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(channels, 1, 11, 11))
        self.weight = nn.Parameter(torch.empty(outputs, channels))

    def features(self, x):
        return F.conv2d(x, self.kernel).square().mean((-2, -1))

    def forward(self, x):
        return self.features(x) @ self.weight.T


def load_data(root, device):
    audit(root)  # Hash verification precedes any dataset deserialization.
    dataset = CIFAR10(str(root), train=True, download=False)
    rgb = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255
    x = (rgb[:, :1] * .2989 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114)
    labels = torch.tensor(dataset.targets)
    generator = torch.Generator().manual_seed(20261019)
    train, val = [], []
    for label in range(10):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        train.append(indices[:4500])
        val.append(indices[4500:])
    train, val = torch.cat(train), torch.cat(val)
    mean, std = x[train].mean(), x[train].std()
    x = ((x - mean) / std).to(device)
    return x, labels, train, val, {"training_mean": float(mean), "training_std": float(std),
                                  "train_size": len(train), "validation_size": len(val),
                                  "test_used": False, "split_seed": 20261019}


def augment(x, generator):
    batch = len(x)
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    offset = torch.randint(9, (batch, 2), generator=generator, device=x.device)
    y = torch.arange(32, device=x.device)[None, :, None] + offset[:, 0, None, None]
    z = torch.arange(32, device=x.device)[None, None, :] + offset[:, 1, None, None]
    cropped = padded[torch.arange(batch, device=x.device)[:, None, None], 0, y, z][:, None]
    flip = torch.rand(batch, 1, 1, 1, generator=generator, device=x.device) < .5
    result = torch.where(flip, cropped.flip(-1), cropped)
    contrast = .8 + .4 * torch.rand(batch, 1, 1, 1, generator=generator, device=x.device)
    return result * contrast + .05 * torch.randn(result.shape, generator=generator, device=x.device)


def initialize(seed, calibration_x, device):
    generator = torch.Generator().manual_seed(seed)
    model = NaturalSquare().to(device)
    with torch.no_grad():
        model.kernel.copy_(torch.randn(model.kernel.shape, generator=generator).to(device) / 11)
        model.weight.copy_(torch.randn(model.weight.shape, generator=generator).to(device) / 8)
        z = model(calibration_x)
        energy = (z - z.mean(0)).square().sum(1).mean()
        model.weight.mul_(math.sqrt(.01 / float(energy)))
    # Unlabelled Hutchinson estimate for centered-output front Jacobian trace.
    z = model(calibration_x)
    z = z - z.mean(0)
    g = torch.Generator(device=device).manual_seed(720000 + seed)
    estimates = []
    for _ in range(16):
        signs = torch.randint(2, z.shape, generator=g, device=device).float() * 2 - 1
        grad = torch.autograd.grad((z * signs).sum() / math.sqrt(len(z)), model.kernel, retain_graph=True)[0]
        estimates.append(float(grad.square().sum()))
    with torch.no_grad():
        r = model.features(calibration_x)
        head = model.weight.shape[0] * float((r - r.mean(0)).square().sum(1).mean())
        front = sum(estimates) / len(estimates)
        c = (front / head)**(1 / 6)
        before = model(calibration_x).clone()
        model.kernel.mul_(c)
        model.weight.mul_(c**-2)
        error = float((model(calibration_x) - before).abs().max())
    return model, {"scale": c, "front_trace_before": front, "head_trace_before": head,
                   "front_trace_after_estimate": front * c**-2, "head_trace_after": head * c**4,
                   "hutchinson_draws": 16, "matched_function_max_error": error,
                   "common_mobility": 1 / (front * c**-2 + head * c**4)}


@torch.no_grad()
def probe(model, data, labels, train, val):
    features = torch.cat([model.features(data[i:i + 512]).cpu() for i in range(0, len(data), 512)])
    projection = features @ model.weight.detach().cpu().T
    output = {}
    for name, matrix in [("encoder", features), ("projection", projection)]:
        # Invertible column centering/scaling aids numerical OLS, no ridge.
        matrix = matrix.double()
        mean, std = matrix[train].mean(0), matrix[train].std(0).clamp_min(1e-12)
        matrix = (matrix - mean) / std
        matrix = torch.cat((torch.ones(len(matrix), 1), matrix), 1)
        targets = F.one_hot(labels[train], 10).double()
        solution = torch.linalg.lstsq(matrix[train], targets, driver="gelsd")
        pred = (matrix[val] @ solution.solution).argmax(1)
        output[name + "_validation_accuracy"] = float((pred == labels[val]).double().mean())
        output[name + "_probe_rank"] = int(solution.rank)
        output[name + "_singular_values"] = solution.singular_values.tolist()
    return output


def train(args, data, labels, train_indices, val_indices, split):
    calibration_indices = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(20261096))[:256]
    cal = data[train_indices[calibration_indices].to(data.device)]
    for seed in args.seeds:
        base, calibration = initialize(seed, cal, args.device)
        for ratio in args.ratios:
            tag = args.output / f"seed_{seed}_r{ratio:g}"
            tag.mkdir(parents=True, exist_ok=True)
            if (tag / "result.json").exists():
                continue
            model = copy.deepcopy(base)
            generator = torch.Generator(device=args.device).manual_seed(820000 + seed)
            mobility = calibration["common_mobility"]
            parameters = (model.kernel, model.weight)
            factors = (2 * mobility / (1 + ratio), 2 * mobility * ratio / (1 + ratio))
            records = []
            energy_checkpoints = []
            probe_steps = sorted(set([0, args.steps // 4, args.steps // 2, args.steps * 3 // 4, args.steps]))
            torch.save(model.state_dict(), tag / "initial.pt")
            if args.evaluate:
                records.append({"step": 0, **probe(model, data, labels, train_indices, val_indices)})
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            timed_start = None
            training_indices = train_indices.to(args.device)
            for step in range(1, args.steps + 1):
                index = training_indices[torch.randint(len(training_indices), (args.batch_size,),
                                                       generator=generator, device=args.device)]
                x1, x2 = augment(data[index], generator), augment(data[index], generator)
                z1, z2 = model(x1), model(x2)
                loss = empirical_loss(z1, z2)
                gradients = torch.autograd.grad(loss, parameters)
                with torch.no_grad():
                    for parameter, grad, factor in zip(parameters, gradients, factors):
                        parameter.add_(grad, alpha=-args.lr * factor)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"CIFAR instability: seed={seed}, r={ratio}, step={step}")
                if step == 20:
                    torch.cuda.synchronize()
                    timed_start = time.perf_counter()
                if step % 10 == 0 or step == args.steps:
                    with torch.no_grad():
                        z = model(cal)
                        energy = float((z - z.mean(0)).square().sum(1).mean())
                    if step % 50 == 0 or step == args.steps:
                        records.append({"step": step, "batch_loss": float(loss.detach()), "clean_calibration_energy": energy,
                                        "ell": math.log(energy / .01)})
                    for target in [.5, 1., 2.]:
                        if math.log(energy / .01) >= target and not any(e["target_ell"] == target for e in energy_checkpoints):
                            torch.save(model.state_dict(), tag / f"energy_{target:g}.pt")
                            energy_checkpoints.append({"target_ell": target, "observed_ell": math.log(energy / .01),
                                                       "step": step, "definition": "first crossing on 10-update measurement grid"})
                if step in probe_steps:
                    torch.save(model.state_dict(), tag / f"step_{step}.pt")
                if step % 500 == 0:
                    print(f"CIFAR seed={seed} r={ratio:g} step={step}/{args.steps}", flush=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            timed = time.perf_counter() - timed_start if timed_start else seconds
            if args.evaluate:
                records.append({"step": args.steps, **probe(model, data, labels, train_indices, val_indices)})
            result = {"seed": seed, "ratio": ratio, "steps": args.steps, "seconds": seconds,
                      "seconds_per_step_after_warmup": timed / max(1, args.steps - 20),
                      "peak_memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
                      "calibration": calibration, "split": split, "history": records,
                      "energy_checkpoints": energy_checkpoints, "probe_steps": probe_steps,
                      "scope": "natural-image extrapolation; no theorem threshold or exact population-risk claim"}
            torch.save(model.state_dict(), tag / "final.pt")
            (tag / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"CIFAR seed={seed} r={ratio:g} complete {seconds:.1f}s "
                  f"{result['seconds_per_step_after_warmup']:.4f}s/step {result['peak_memory_GiB']:.2f}GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.002)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ratios", nargs="+", type=float, default=[.1, 1., 10.])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--phase", choices=["throughput", "bounded_comparison"], default="throughput")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cifar_throughput")
    args = parser.parse_args()
    if args.phase == "bounded_comparison" and (args.seeds != [2, 3, 4, 5, 6] or args.steps != 4000):
        raise ValueError("Locked local comparison: five seeds 2..6, 4000 updates")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    data, labels, train_indices, val_indices, split = load_data(ROOT.parent, args.device)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(model="64x11x11 valid conv, square, GAP, linear 8-d projection",
                    split=split, augmentation="reflect crop +/-4, horizontal flip, contrast .8..1.2, Gaussian .05",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    calibration_selection="256 random unlabelled training images, fixed seed 20261096; not class-sorted prefix",
                    dtype="float32", gpu=torch.cuda.get_device_name(),
                    energy_constraint="none; energy logged, no normalization/rescaling", test_used=False)
    lock = args.output / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise ValueError("Existing CIFAR protocol differs")
    lock.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    train(args, data, labels, train_indices, val_indices, split)


if __name__ == "__main__":
    main()
