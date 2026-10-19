"""Fixed-step/common-energy validation and fixed-final-step test evaluation.

All checkpoints and training budgets are chosen without test labels. No ridge,
no classifier tuning, no test-based checkpoint selection.
"""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from cifar_local import NaturalSquare, ROOT, load_data, probe


@torch.no_grad()
def test_probe(model, training_data, train_indices, training_labels, test_images, test_labels):
    x = torch.cat([model.features(training_data[i:i + 512]).cpu() for i in range(0, len(training_data), 512)]).double()
    y = torch.cat([model.features(test_images[i:i + 512]).cpu() for i in range(0, len(test_images), 512)]).double()
    weight = model.weight.detach().cpu().double()
    result = {}
    for name, tr, te in [("encoder", x, y), ("projection", x @ weight.T, y @ weight.T)]:
        mean, std = tr[train_indices].mean(0), tr[train_indices].std(0).clamp_min(1e-12)
        tr, te = (tr - mean) / std, (te - mean) / std
        tr = torch.cat((torch.ones(len(tr), 1, dtype=torch.float64), tr), 1)
        te = torch.cat((torch.ones(len(te), 1, dtype=torch.float64), te), 1)
        fit = torch.linalg.lstsq(tr[train_indices], F.one_hot(training_labels[train_indices], 10).double(), driver="gelsd")
        accuracy = float(((te @ fit.solution).argmax(1) == test_labels).double().mean())
        result[name + "_test_accuracy"] = accuracy
        result[name + "_probe_rank"] = int(fit.rank)
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    completed = list(args.directory.glob("seed*/result.json"))
    if len(completed) != len(protocol["seeds"]) * len(protocol["ratios"]):
        raise ValueError("Wait for all prespecified training runs before test evaluation")
    torch.set_num_threads(1)
    data, labels, train, val, split = load_data(ROOT.parent, "cuda")
    dataset = CIFAR10(str(ROOT.parent), train=False, download=False)
    rgb = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255
    test_images = ((rgb[:, :1] * .2989 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114 - split["training_mean"]) /
                   split["training_std"]).cuda()
    test_labels = torch.tensor(dataset.targets)
    rows = []
    cache = {}
    for path in sorted(completed):
        run = json.loads(path.read_text())
        result_path = path.parent / "evaluation.json"
        if result_path.exists():
            rows.append(json.loads(result_path.read_text()))
            continue
        model = NaturalSquare().cuda()
        validation = []
        for step in run["probe_steps"]:
            checkpoint = path.parent / ("initial.pt" if step == 0 else f"step_{step}.pt")
            model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
            key = (run["seed"], 0)
            if step == 0 and key in cache:
                measured = cache[key]
            else:
                measured = probe(model, data, labels, train, val)
                if step == 0:
                    cache[key] = measured
            validation.append({"step": step, **measured})
        energy_rows = []
        for checkpoint in run["energy_checkpoints"]:
            model.load_state_dict(torch.load(path.parent / f"energy_{checkpoint['target_ell']:g}.pt",
                                             map_location="cuda", weights_only=True))
            energy_rows.append({**checkpoint, **probe(model, data, labels, train, val)})
        model.load_state_dict(torch.load(path.parent / "final.pt", map_location="cuda", weights_only=True))
        final_test = test_probe(model, data, train, labels, test_images, test_labels)
        result = {"seed": run["seed"], "ratio": run["ratio"], "fixed_step_validation": validation,
                  "common_energy_validation": energy_rows, "fixed_final_step_test": final_test,
                  "test_selection": "all prespecified ratios at fixed final step, no test-based selection"}
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append(result)
        print(f"CIFAR evaluated seed={run['seed']} r={run['ratio']:g} "
              f"encoder={final_test['encoder_test_accuracy']:.4f} projection={final_test['projection_test_accuracy']:.4f}", flush=True)
    print("Per-run metric evaluation complete. Run summarize_cifar_local.py in a separate process.")



if __name__ == "__main__":
    main()
