"""Build the illustrative fixed-task rebound figure from all validation runs.

No training or task selection. The x_low case accompanies a complete ten-task
count table in the paper; its six seeds and three models are all displayed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PAPER.parent / "experiments/results/empirical/encoder_rebound_validation"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7B3294", "#222222")
MODES = (("square", "(a) Square"), ("linear", "(b) Linear"),
         ("frozen_front", "(c) Frozen encoder"))


def build(results, output):
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    seeds = protocol["seeds"]
    if len(seeds) != 6 or protocol["ratios"] != [1.]:
        raise ValueError("This compact figure expects all six validation seeds at r=1")
    early, late = protocol["early_step"], protocol["late_step"]
    threshold = protocol["amplitude_threshold"]
    task = "x_low"
    curves = {}
    for mode, _ in MODES:
        for seed in seeds:
            path = results / f"seed_{seed}_{mode}_r1/result.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("status") != "complete":
                raise ValueError(f"Incomplete planned run: {path}")
            history = run["history"]
            steps = np.asarray([row["step"] for row in history])
            risks = np.asarray([row["encoder_risks"][task] for row in history])
            if not np.all(np.isfinite(risks)) or not {0, early, late}.issubset(steps):
                raise ValueError(f"Missing/nonfinite comparison states: {path}")
            curves[mode, seed] = steps, risks - risks[0]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": .55,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(3, 1, figsize=(3.35, 3.25), sharex=True, sharey=True)
        fig.subplots_adjust(left=.19, right=.98, bottom=.12, top=.86, hspace=.12)
        for ax, (mode, label) in zip(axes, MODES):
            ax.axhspan(-threshold, threshold, color="#ededed", zorder=0)
            ax.axhline(0, color="#666666", linewidth=.55, zorder=1)
            for sign in (-1, 1):
                ax.axhline(sign * threshold, color="#666666", linewidth=.55,
                           linestyle=(0, (3, 2)), zorder=2)
            for step in (early, late):
                ax.axvline(step, color="#888888", linewidth=.6,
                           linestyle=(0, (2, 2)), zorder=1)
            for seed, color in zip(seeds, COLORS):
                steps, delta = curves[mode, seed]
                ax.plot(steps, delta, color=color, linewidth=.95, zorder=3)
                for step in (early, late):
                    index = int(np.flatnonzero(steps == step)[0])
                    ax.plot(step, delta[index], "o", color=color, markersize=2.7,
                            markeredgewidth=0, zorder=4)
            ax.set(xlim=(-30, late + 110), ylim=(-.115, .125), yticks=(-.1, 0, .1))
            ax.text(.025, .86, label, transform=ax.transAxes, fontsize=7.3,
                    ha="left", va="top")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", length=2.3, width=.55, pad=2)
        axes[-1].set(xticks=(0, early, 2000, 3000, late),
                     xticklabels=("0", "1,000", "2,000", "3,000", "4,000"),
                     xlabel="SGD update")
        axes[-1].text(.98, .12, "All six curves overlap", transform=axes[-1].transAxes,
                      ha="right", va="bottom", fontsize=6.4, color="#444444")
        axes[-1].annotate(r"$\pm\delta$, $\delta=10^{-3}$", xy=(560, threshold),
                          xytext=(1450, .04), fontsize=6.4, color="#444444",
                          arrowprops={"arrowstyle": "-", "linewidth": .5,
                                      "color": "#777777"})
        fig.supylabel(r"$R_{\rm enc}(t)-R_{\rm enc}(0)$", x=.018, fontsize=8)
        fig.text(.58, .985, r"Fixed task $x_{\rm low}$ (illustrative)", ha="center",
                 va="top", fontsize=8)
        handles = [Line2D([], [], color=color, linewidth=1.2, label=str(seed))
                   for seed, color in zip(seeds, COLORS)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.58, .955),
                   ncol=6, frameon=False, fontsize=6.6, handlelength=1.15,
                   columnspacing=.65, handletextpad=.3, title="Seed", title_fontsize=6.4)
        output.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            fig.savefig(output / f"encoder_rebound_x_low.{extension}", dpi=300,
                        bbox_inches="tight", pad_inches=.025)
        plt.close(fig)
    print(f"Built illustrative x_low figure from all {len(seeds) * len(MODES)} runs: {output}")


def appendix_figures(results, output):
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    seeds, tasks = protocol["seeds"], protocol["tasks"]
    early, late = protocol["early_step"], protocol["late_step"]
    threshold = protocol["amplitude_threshold"]
    runs = {(mode, seed): json.loads((results / f"seed_{seed}_{mode}_r1/result.json")
                                    .read_text(encoding="utf-8"))
            for mode, _ in MODES for seed in seeds}
    styles = {"square": "-", "linear": "--", "frozen_front": ":"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.titlesize": 8,
                         "axes.linewidth": .55, "pdf.fonttype": 42, "ps.fonttype": 42}):
        for page in range(2):
            selected = tasks[5 * page:5 * (page + 1)]
            fig, axes = plt.subplots(5, 2, figsize=(7., 7.65), sharex=True)
            fig.subplots_adjust(left=.09, right=.985, bottom=.065, top=.87,
                                hspace=.37, wspace=.23)
            for row, task in enumerate(selected):
                for column, representation in enumerate(("encoder", "projection")):
                    ax = axes[row, column]
                    for value in (-threshold, threshold):
                        ax.axhline(value, color="#999999", lw=.5, ls=(0, (3, 2)), zorder=1)
                    ax.axhline(0, color="#999999", lw=.45, zorder=1)
                    for step in (early, late):
                        ax.axvline(step, color="#bbbbbb", lw=.6, ls=(0, (2, 2)), zorder=1)
                    for mode, _ in MODES:
                        for seed, color in zip(seeds, COLORS):
                            history = runs[mode, seed]["history"]
                            steps = np.asarray([point["step"] for point in history])
                            risks = np.asarray([point[representation + "_risks"][task]
                                                for point in history])
                            delta = risks - risks[0]
                            ax.plot(steps, delta, color=color, ls=styles[mode],
                                    lw=.8 if mode == "square" else .65, alpha=.85, zorder=3)
                            for step in (early, late):
                                index = int(np.flatnonzero(steps == step)[0])
                                ax.plot(step, delta[index], ".", color=color, ms=2.2, zorder=4)
                    ax.set(xlim=(-30, late + 110), xticks=(0, 1000, 2000, 3000, 4000),
                           title=f"{task.replace('_', ' ')} | {representation}")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.tick_params(length=2.4, width=.55, pad=2)
                    ax.grid(axis="y", alpha=.1, lw=.5)
                    if column == 0:
                        ax.set_ylabel(r"$R(t)-R(0)$")
                    if row == 4:
                        ax.set_xlabel("SGD update")
            seed_handles = [Line2D([], [], color=color, lw=1.1, label=f"Seed {seed}")
                            for seed, color in zip(seeds, COLORS)]
            mode_handles = [Line2D([], [], color="#444444", ls=styles[mode], lw=1.1,
                                  label=label.removeprefix("(a) ").removeprefix("(b) ").removeprefix("(c) "))
                            for mode, label in MODES]
            fig.legend(handles=seed_handles, loc="upper center", ncol=6,
                       bbox_to_anchor=(.54, .982), fontsize=7, frameon=False,
                       handlelength=1.6, columnspacing=1.35)
            fig.legend(handles=mode_handles, loc="upper center", ncol=3,
                       bbox_to_anchor=(.54, .951), fontsize=7, frameon=False,
                       handlelength=2.4, columnspacing=1.9)
            fig.text(.54, .915, r"Dashed horizontal lines: $\pm10^{-3}$; vertical lines: updates 1,000 and 4,000",
                     fontsize=7, ha="center")
            for extension in ("pdf", "png"):
                fig.savefig(output / f"encoder_rebound_tasks_{page + 1}.{extension}",
                            dpi=220, bbox_inches="tight", pad_inches=.025)
            plt.close(fig)
    print("Built both all-task appendix figures.")


def geometry_figure(results, output):
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    early, late = protocol["early_step"], protocol["late_step"]
    mode_colors = {"square": "#0072B2", "linear": "#D55E00", "frozen_front": "#009E73"}
    runs = [json.loads((results / f"seed_{seed}_{mode}_r1/result.json").read_text(encoding="utf-8"))
            for mode, _ in MODES for seed in protocol["seeds"]]
    specs = (("population_loss_for_evaluation_only", "Population loss (evaluation)", False),
             ("encoder_rank", "Encoder numerical rank", False),
             ("encoder_min_nonzero", "Encoder minimum nonzero singular value", True),
             ("projection_min_nonzero", "Projection minimum nonzero singular value", True),
             ("encoder_displacement", "Encoder projector displacement", False),
             ("projection_displacement", "Projection projector displacement", False))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "axes.titlesize": 8,
                         "axes.linewidth": .55, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(3, 2, figsize=(7., 4.8))
        fig.subplots_adjust(left=.08, right=.99, bottom=.10, top=.90,
                            hspace=.57, wspace=.25)
        for ax, (metric, title, logarithmic) in zip(axes.flat, specs):
            for run in runs:
                xs, ys = [], []
                for point in run["history"]:
                    if metric.endswith("_min_nonzero"):
                        representation = metric.removesuffix("_min_nonzero")
                        rank = point[representation + "_rank"]
                        value = point[representation + "_singular_values"][rank - 1] if rank else None
                    else:
                        value = point[metric]
                    if value is not None and np.isfinite(value) and (not logarithmic or value > 0):
                        xs.append(point["step"])
                        ys.append(value)
                ax.plot(xs, ys, color=mode_colors[run["mode"]], lw=.8, alpha=.7)
            if logarithmic:
                ax.set_yscale("log")
            for step in (early, late):
                ax.axvline(step, color="#aaaaaa", lw=.6, ls=(0, (2, 2)), zorder=1)
            ax.set(xlim=(-30, late + 110), xticks=(0, 1000, 2000, 3000, 4000),
                   xlabel="SGD update", title=title)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2.4, width=.55, pad=2)
            ax.grid(alpha=.12, lw=.5)
        axes[0, 1].set(ylim=(7.8, 8.2), yticks=(7.8, 8., 8.2))
        handles = [Line2D([], [], color=mode_colors[mode], lw=1.2,
                          label=label.removeprefix("(a) ").removeprefix("(b) ").removeprefix("(c) "))
                   for mode, label in MODES]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
                   ncol=3, frameon=False, fontsize=7.5, handlelength=2.4,
                   columnspacing=2.3)
        for extension in ("pdf", "png"):
            fig.savefig(output / f"encoder_rebound_geometry.{extension}", dpi=260,
                        bbox_inches="tight", pad_inches=.025)
        plt.close(fig)
    print(f"Built native 7-inch geometry figure from all {len(runs)} runs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=PAPER / "figures")
    args = parser.parse_args()
    build(args.results.resolve(), args.output.resolve())
    appendix_figures(args.results.resolve(), args.output.resolve())
    geometry_figure(args.results.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
