"""Re-evaluate saved checkpoints without rerunning or changing pretraining."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from square_images import texture_tasks
from train_textures import initialize, prospective_intervention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = SimpleNamespace(**json.loads((args.directory / "config.json").read_text()))
    for path in sorted(args.directory.glob("seed_*/result.json")):
        result = json.loads(path.read_text())
        model = initialize(config, result["seed"])
        names, tasks = texture_tasks(model.weight.dtype, model.weight.device)
        with torch.no_grad():
            assert model.effective()[0].cpu().tolist() == result["initial_A"]
        result["initial_intervention"] = prospective_intervention(
            model, tasks, names, config.noise, config.ratios, config.lr,
            path.parent / "initial_predictions.json")
        checkpoint = path.parent / "ell_one.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            with torch.no_grad():
                model.kernel.copy_(saved["kernel"])
                model.weight.copy_(saved["weight"])
            result["ell_one_intervention"] = {"step": saved["step"], "ell": saved["ell"],
                **prospective_intervention(model, tasks, names, config.noise, config.ratios, config.lr,
                                           path.parent / "ell_one_predictions.json")}
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(path.parent.name, flush=True)


if __name__ == "__main__":
    main()
