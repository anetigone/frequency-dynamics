"""Standalone figures and a conservative pilot summary; no label-based selection."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULTS / "pilot")
    args = parser.parse_args()
    output = args.root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    spatial = json.loads((args.root / "spatial" / "verification.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    c = np.linspace(.65, 1.35, 400)
    ax[0].plot(c, (235 - 480 * c**6) / (834 + 1300 * c**6), label="Analytic prediction")
    points = [row for row in spatial["slopes"] if row["phase"] == 0]
    ax[0].scatter([r["c"] for r in points], [r["image_autograd_slope"] for r in points],
                  marker="x", color="black", label="Spatial autograd")
    ax[0].axhline(0, color="grey", linewidth=.7)
    ax[0].axvline((47 / 96)**(1 / 6), color="grey", linestyle=":")
    ax[0].set(xlabel="Matched-function scale c", ylabel="Initial dR / d log V")
    ax[0].legend(fontsize=8)
    for row in spatial["trajectories"]:
        if row["delta_log_energy"] == min(r["delta_log_energy"] for r in spatial["trajectories"]):
            ax[1].plot([s["ell"] for s in row["sections"]], [s["risk"] for s in row["sections"]],
                       "o--", label=f"noise={row['noise']:g}")
    ax[1].set(xlabel="Log-energy section", ylabel="Projection excess risk",
              title="Discrete GD at three sections (not full trajectories)")
    ax[1].legend(fontsize=8)
    fig.savefig(output / "spatial_verification.png", dpi=180)
    plt.close(fig)
    summary = {"scope": "Two-seed pilot, not a robustness or generality claim", "datasets": {}}
    for objective in ["population", "empirical", "balanced_population", "balanced_empirical"]:
        directory = args.root / f"textures_{objective}_pilot"
        paths = sorted(directory.glob("seed_*/result.json"))
        if not paths:
            continue
        runs = [(p, json.loads(p.read_text())) for p in paths]
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        selected = ["x_high", "y_low", "diagonal_low", "single_x_low"]
        for ax, task in zip(axs.flat, selected):
            for path, result in runs:
                with (path.parent / "trajectory.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                ax.plot([float(r["ell"]) for r in rows], [float(r["risk_" + task]) for r in rows],
                        color={.1: "#3576b5", 1.: "#e08c2b", 10.: "#438d56"}[result["ratio"]],
                        linestyle="-" if result["seed"] == 0 else "--",
                        label=f"r={result['ratio']:g}, seed={result['seed']}")
            ax.set(xlabel="Log clean-output energy / initial energy", ylabel="Population excess risk",
                   title=task)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f"{objective.capitalize()} pretraining: paired initialization, task exchange")
        fig.savefig(output / f"textures_{objective}.png", dpi=180)
        plt.close(fig)
        common = 2.
        comparisons, outcomes = [], []
        for path, result in runs:
            with (path.parent / "trajectory.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            x = np.array([float(r["ell"]) for r in rows])
            if x.max() < common:
                continue
            if np.any(np.diff(x) <= 0):
                raise ValueError("Nonmonotone energy: cannot use single-valued section interpolation")
            entry = {"seed": result["seed"], "ratio": result["ratio"], "ell": common}
            all_tasks = [key.removeprefix("risk_") for key in rows[0] if key.startswith("risk_")]
            for task in all_tasks:
                y = np.array([float(r["risk_" + task]) for r in rows])
                entry[task] = float(np.interp(common, x, y))
            comparisons.append(entry)
            check = result["ell_one_intervention"]
            if check and check["status"] == "complete":
                outcomes.extend(check["outcomes"])
        # Show every prespecified task, including results absent from the curves.
        seeds = sorted(set(entry["seed"] for entry in comparisons))
        differences = []
        for seed in seeds:
            by_ratio = {entry["ratio"]: entry for entry in comparisons if entry["seed"] == seed}
            differences.append([by_ratio[10.][task] - by_ratio[.1][task] for task in all_tasks])
        if differences:
            fig, ax = plt.subplots(figsize=(11, 3.4), constrained_layout=True)
            matrix = np.array(differences)
            limit = max(float(np.abs(matrix).max()), 1e-6)
            heat = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_xticks(range(len(all_tasks)), labels=all_tasks, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(seeds)), labels=[f"seed {s}" for s in seeds])
            ax.set_title(f"{objective.replace('_', ' ')}: R(r=10) - R(r=0.1) at log-energy=2; red = harm", fontsize=10)
            fig.colorbar(heat, ax=ax, label="Risk difference")
            fig.savefig(output / f"task_exchange_{objective}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        for dt in sorted(set(o["dt"] for o in outcomes)):
            group = [o for o in outcomes if o["dt"] == dt]
            ax.scatter([o["predicted_derivative"] for o in group],
                       [o["observed_derivative"] for o in group], s=8, alpha=.5, label=f"dt={dt:g}")
        if outcomes:
            lo = min(o["predicted_derivative"] for o in outcomes)
            hi = max(o["predicted_derivative"] for o in outcomes)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=.7)
            ax.legend(fontsize=8)
        ax.set(xlabel="Predicted risk derivative", ylabel="One-step observed derivative",
               title=f"{objective}: population-vector-field intervention")
        fig.savefig(output / f"intervention_{objective}.png", dpi=180)
        plt.close(fig)
        summary["datasets"][objective] = {"fixed_section": comparisons,
                                         "comparison_method": "linear interpolation at ell=2 within recorded common support",
                                         "intervention_note": "oracle population gradients; one-step local check"}
    (output / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
