"""Compare complete empirical states to EXISTING population checkpoints only."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import json

import torch

from full_empirical import ROOT, build
from subspace_mechanism import projector


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rows = []
    for empirical_dir, population_dir, kernel in [("full_sgd_main", "subspace_confirm_k1", 32),
                                                 ("full_sgd_local", "subspace_confirm_local", 11)]:
        for seed in range(2, 12):
            run = json.loads((result_path(empirical_dir) / f"seed_{seed}_square_r1_stream0" / "result.json").read_text())
            section = next(row for row in run["sections"] if row["target_ell"] == 1.5)
            old = torch.load(result_path(population_dir) / "checkpoints" /
                             f"seed_{seed}_independently_calibrated_square_r1" / "ell_1.5.pt", weights_only=True)
            model, _ = build(seed, "square", 1., kernel)
            with torch.no_grad():
                a0, b0 = model.effective()
                initial = projector(b0)[0]
                model.front.copy_(old["front"])
                model.weight.copy_(old["weight"])
                a, b = model.effective()
                population_pb = projector(b)[0]
                population_pa = projector(a)[0]
            empirical_pb = torch.tensor(section["encoder_projector"])
            empirical_pa = torch.tensor(section["projection_projector"])
            rows.append({"setting": empirical_dir, "seed": seed, "ratio": 1., "ell": 1.5,
                         "encoder_empirical_minus_population_distance": float((empirical_pb - population_pb).norm()),
                         "projection_empirical_minus_population_distance": float((empirical_pa - population_pa).norm()),
                         "population_encoder_displacement": float((population_pb - initial).norm()),
                         "empirical_encoder_displacement": section["encoder_displacement"]})
    result = {"scope": "read-only comparison to previously computed population trajectories, no new population integration",
              "limitation": "differences combine finite-dataset/minibatch effects and numerical step differences", "rows": rows}
    (result_path("empirical_existing_population_comparison.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for setting in ["full_sgd_main", "full_sgd_local"]:
        group = [r for r in rows if r["setting"] == setting]
        print(setting, {key: sum(r[key] for r in group) / len(group) for key in
                        ["encoder_empirical_minus_population_distance", "projection_empirical_minus_population_distance",
                         "population_encoder_displacement", "empirical_encoder_displacement"]})


if __name__ == "__main__":
    main()
