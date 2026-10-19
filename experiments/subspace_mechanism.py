"""Matched-function subspace experiments, with a locked held-out-seed protocol.

Pixel kernels are the trainable parameters. The exact Fourier identity evaluates
their periodic-square-GAP output efficiently; image autograd is independently
checked by check_subspace_mechanism.py. This is not direct optimization of B.
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

from square_images import excess_risks, population_loss, texture_tasks
from train_textures import initialize


ROOT = Path(__file__).parent
MODES = ("square", "linear", "frozen_front")
ARMS = ("matched", "independently_calibrated")
TARGETS = tuple(i / 4 for i in range(13))


class Factors:
    def __init__(self, base, mode):
        self.image = copy.deepcopy(base)
        self.mode = mode
        self.weight = self.image.weight
        with torch.no_grad():
            self.initial_a, b = self.image.effective()
            self.initial_a = self.initial_a.clone()
        self.front = (torch.nn.Parameter(b.clone()) if mode == "linear" else self.image.kernel)
        self.initial_parameters = [self.front.detach().clone(), self.weight.detach().clone()]
        if mode == "frozen_front":
            self.front.requires_grad_(False)
        self.scale, self.mobility = 1., 1.

    def effective(self):
        if self.mode == "linear":
            b = self.front
        else:
            k = self.front.shape[-1]
            basis = self.image.basis[..., :k, :k]
            h = torch.einsum("cij,dpij->cdp", self.front[:, 0], basis)
            b = self.image.g * h.square().sum(-1)
        return self.weight @ b, b

    def parameters(self):
        return [p for p in (self.front, self.weight) if p.requires_grad]

    def kernel_gram(self):
        """J_Bc J_Bc^T for each channel, including local-kernel couplings."""
        if self.mode == "linear":
            d = self.front.shape[1]
            return torch.eye(d, dtype=self.front.dtype).expand(len(self.front), d, d)
        k = self.front.shape[-1]
        basis = self.image.basis[..., :k, :k]
        h = torch.einsum("cij,dpij->cdp", self.front[:, 0], basis)
        jacobian = 2 * self.image.g[None, :, None, None] * torch.einsum("cdp,dpij->cdij", h, basis)
        j = jacobian.flatten(2)
        return j @ j.transpose(1, 2)

    @torch.no_grad()
    def traces(self):
        _, b = self.effective()
        head = self.weight.shape[0] * float(b.square().sum())
        front = (0. if self.mode == "frozen_front" else
                 float((self.kernel_gram().diagonal(dim1=-2, dim2=-1).sum(-1) *
                        self.weight.square().sum(0)).sum()))
        return front, head

    def calibrate(self, arm):
        before = self.traces()
        if arm == "independently_calibrated":
            if self.mode != "frozen_front":
                exponent = 1 / 4 if self.mode == "linear" else 1 / 6
                self.scale = (before[0] / before[1])**exponent
                with torch.no_grad():
                    self.front.mul_(self.scale)
                    self.weight.mul_(self.scale**(-1 if self.mode == "linear" else -2))
            after = self.traces()
            self.mobility = 1 / sum(after)
        after = self.traces()
        with torch.no_grad():
            assert torch.allclose(self.effective()[0], self.initial_a, rtol=1e-11, atol=1e-13)
        if arm == "independently_calibrated" and self.mode != "frozen_front":
            assert abs(after[0] / after[1] - 1) < 1e-10
        return {"scale": self.scale, "mobility": self.mobility,
                "trace_front_before": before[0], "trace_head_before": before[1],
                "trace_front_after": after[0], "trace_head_after": after[1],
                "normalized_total_trace": self.mobility * sum(after),
                "frozen_front_balance": "not applicable; normalize only active head" if self.mode == "frozen_front" else None}


@torch.no_grad()
def projector(x):
    _, values, vh = torch.linalg.svd(x, full_matrices=False)
    threshold = max(x.shape) * torch.finfo(x.dtype).eps * values.max()
    rank = int((values > threshold).sum())
    return vh[:rank].T @ vh[:rank], values, rank, float(threshold)


def loss_gradient(a, noise):
    """Closed-form matrix GUPA gradient, independent of parameter autograd."""
    s = (1 + noise)**2
    covariance = s * a @ a.T
    eye = torch.eye(len(a), dtype=a.dtype, device=a.device)
    inverse = torch.linalg.inv(eye + covariance)
    gamma = .5 * (covariance + 2 * eye) @ inverse @ inverse
    return 2 * (s - 1) * a - 2 * s * gamma @ a


@torch.no_grad()
def geometry_field(model, noise):
    a, b = model.effective()
    g = loss_gradient(a, noise)
    dw = -g @ b.T
    db = (-torch.einsum("cij,cj->ci", model.kernel_gram(), model.weight.T @ g)
          if model.mode != "frozen_front" else torch.zeros_like(b))
    return {"A_front": model.weight @ db, "A_head": dw @ b,
            "B_front": db, "B_head": torch.zeros_like(b), "gradient_A": g}


@torch.no_grad()
def risk_derivative(x, dx, tasks):
    # All experimental A and B have full row rank; rank is reported separately.
    solution = torch.linalg.solve(x @ x.T, x @ tasks.T)
    residual = tasks.T - x.T @ solution
    return -2 * (solution * (dx @ residual)).sum(0)


@torch.no_grad()
def record(model, initial, tasks, names, ell, updates, physical_time, noise):
    a, b = model.effective()
    pa, sa, ra, ta = projector(a)
    pb, sb, rb, tb = projector(b)
    risk_a, _, _ = excess_risks(a, tasks)
    risk_b, _, _ = excess_risks(b, tasks)
    result = {"ell": ell, "updates": updates, "physical_time": physical_time,
              "energy": float(a.square().sum()), "loss": float(population_loss(a, noise)),
              "projection_projector_displacement": float((pa - initial["pa"]).norm()),
              "encoder_projector_displacement": float((pb - initial["pb"]).norm()),
              "projection_rank": ra, "encoder_rank": rb,
              "projection_singular_values": sa.tolist(), "encoder_singular_values": sb.tolist(),
              "projection_rank_tolerance": ta, "encoder_rank_tolerance": tb,
              "projection_coordinate_risk_sum": float((1 - pa.diag()).sum()),
              "encoder_coordinate_risk_sum": float((1 - pb.diag()).sum()),
              "projection_coordinate_risks": (1 - pa.diag()).tolist(),
              "encoder_coordinate_risks": (1 - pb.diag()).tolist(),
              "projection_task_risks": dict(zip(names, risk_a.tolist())),
              "encoder_task_risks": dict(zip(names, risk_b.tolist())),
              "max_encoder_task_risk_change": float((risk_b - initial["risk_b"]).abs().max())}
    assert abs(result["projection_coordinate_risk_sum"] - (a.shape[1] - ra)) < 1e-10
    assert abs(result["encoder_coordinate_risk_sum"] - (b.shape[1] - rb)) < 1e-10
    if model.mode in ("linear", "frozen_front"):
        assert result["encoder_projector_displacement"] < 1e-8
    return result


def integrate(base, mode, arm, ratio, step, noise, checkpoint_dir=None):
    model = Factors(base, mode)
    calibration = model.calibrate(arm)
    with torch.no_grad():
        a0, b0 = model.effective()
        a0, b0 = a0.clone(), b0.clone()
        initial = {"pa": projector(a0)[0], "pb": projector(b0)[0],
                   "risk_b": excess_risks(b0, texture_tasks(torch.float64, "cpu")[1])[0]}
        energy0 = float(a0.square().sum())
    names, tasks = texture_tasks(torch.float64, "cpu")
    raw_fh, raw_fw = 2 / (1 + ratio), 2 * ratio / (1 + ratio)
    if arm == "independently_calibrated":
        th, tw = model.traces()
        model.mobility = 1 / (raw_fh * th + raw_fw * tw)
    fh, fw = raw_fh * model.mobility, raw_fw * model.mobility
    calibration["active_ratio_mobility"] = model.mobility
    th, tw = model.traces()
    calibration["weighted_initial_total_trace"] = fh * th + fw * tw
    updates, physical_time, history = 0, 0., []
    last_loss = math.inf
    for target in TARGETS:
        while True:
            a, b = model.effective()
            energy = a.square().sum()
            ell = math.log(float(energy.detach()) / energy0)
            loss = population_loss(a, noise)
            current_loss = float(loss.detach())
            assert current_loss <= last_loss + 1e-10
            assert (1 + noise)**2 * float(energy.detach()) < 1
            last_loss = current_loss
            if target - ell < 1e-9:
                history.append(record(model, initial, tasks, names, ell, updates, physical_time, noise))
                if checkpoint_dir is not None and target in (0., .75, 1.5, 2.25, 3.):
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"front": model.front.detach(), "weight": model.weight.detach(),
                                "target": target, "mode": mode, "arm": arm,
                                "mobility": model.mobility, "scale": model.scale},
                               checkpoint_dir / f"ell_{target:g}.pt")
                break
            params = model.parameters()
            grads = torch.autograd.grad(loss, params, retain_graph=True)
            dv = torch.autograd.grad(energy, params)
            factors = [fw] if mode == "frozen_front" else [fh, fw]
            rate = float((sum(-factor * (g * v).sum() for factor, g, v in zip(factors, grads, dv)) / energy).detach())
            assert rate > 0
            dt = min(step, .5 * (target - ell)) / rate
            with torch.no_grad():
                for parameter, grad, factor in zip(params, grads, factors):
                    parameter.add_(grad, alpha=-dt * factor)
            physical_time += dt
            updates += 1
            if updates > 30000:
                raise RuntimeError("Energy section not reached within update limit")
    with torch.no_grad():
        if mode != "linear":
            spatial_a, spatial_b = model.image.effective()
            fast_a, fast_b = model.effective()
            assert (spatial_a - fast_a).abs().max() < 1e-11
            assert (spatial_b - fast_b).abs().max() < 1e-11
    return {"mode": mode, "arm": arm, "ratio": ratio, "calibration": calibration,
            "initial_A": a0.detach().tolist(), "initial_B": b0.detach().tolist(), "history": history}


def locked_config(args):
    sources = {}
    for name in ["subspace_mechanism.py", "square_images.py", "train_textures.py"]:
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    config = {"phase": args.phase, "seeds": args.seeds, "ratios": [.1, 1., 10.],
              "models": list(MODES), "arms": list(ARMS), "targets": TARGETS,
              "channels": 8, "outputs": args.outputs, "kernel_size": args.kernel_size,
              "log_energy_step": args.step, "noise": .2, "initial_energy": .001,
              "initialization": "square equal-Jacobian-trace base, shared B0,W0 in matched arm",
              "calibration": "each active two-factor model equalizes own trace contributions, then total trace=1",
              "dtype": "float64", "device": "cpu", "source_sha256": sources,
              "algorithm": "real pixel/factor GD, exact periodic-power forward identity, adaptive positive physical step",
              "inference_unit": "paired random seed, never task or checkpoint",
              "primary_endpoint": "encoder projector Frobenius displacement at ell=3",
              "secondary_endpoint": "projection projector displacement and all task risks",
              "intervention_checkpoints": [.75, 1.5, 2.25]}
    if args.phase == "confirmation" and args.seeds != list(range(2, 12)):
        raise ValueError("Locked confirmation uses exactly seeds 2..11")
    lock = args.output / "protocol.json"
    normalized = json.loads(json.dumps(config))
    if lock.exists() and json.loads(lock.read_text()) != normalized:
        raise ValueError("Protocol/source differs from existing lock; use a distinct output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "confirmation", "extension"], default="development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--outputs", type=int, choices=[1, 2], default=1)
    parser.add_argument("--kernel-size", type=int, choices=[32, 11], default=32)
    parser.add_argument("--step", type=float, default=.004)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "subspace_development")
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("Positive step required")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    config = locked_config(args)
    initialization = SimpleNamespace(dtype="float64", device="cpu", channels=8, outputs=args.outputs,
                                     kernel_size=args.kernel_size, backend="fft", initial_energy=.001,
                                     balance_init=True)
    for seed in args.seeds:
        base = initialize(initialization, seed)
        for arm in ARMS:
            for mode in MODES:
                for ratio in config["ratios"]:
                    tag = f"seed_{seed}_{arm}_{mode}_r{ratio:g}"
                    path = args.output / (tag + ".json")
                    if path.exists():
                        continue
                    start = time.perf_counter()
                    checkpoints = (args.output / "checkpoints" / tag
                                   if mode == "square" and arm == "independently_calibrated" and ratio == 1 else None)
                    result = integrate(base, mode, arm, ratio, args.step, .2, checkpoints)
                    result.update(seed=seed, elapsed_seconds=time.perf_counter() - start)
                    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print(f"{tag} enc_drift={result['history'][-1]['encoder_projector_displacement']:.5g} "
                          f"seconds={result['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
