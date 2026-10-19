"""Complete empirical trajectories: paired tasks, common energy and subspaces."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_subspaces import describe


def endpoint(run, energy=None):
    if energy is None:
        return run["history"][-1]
    return next((row for row in run["sections"] if row["target_ell"] == energy), None)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeat-dirs", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(args.directory.glob("seed*/result.json"))]
    expected = len(protocol["seeds"]) * len(protocol["models"]) * len(protocol["ratios"])
    if len(runs) != expected:
        raise ValueError(f"Only {len(runs)}/{expected} complete")
    lookup = {(r["seed"], r["mode"], r["ratio"]): r for r in runs}
    tasks = list(runs[0]["history"][0]["projection_risks"])
    audits = {"initial_mapping_max_error": 0., "frozen_encoder_max_drift": 0., "rank_changes": [],
              "data_hashes_match_within_seed": True}
    for seed in protocol["seeds"]:
        reference = lookup[seed, protocol["models"][0], protocol["ratios"][0]]
        for run in [r for r in runs if r["seed"] == seed]:
            audits["initial_mapping_max_error"] = max(audits["initial_mapping_max_error"],
                 float(np.abs(np.array(run["initial_A"]) - np.array(reference["initial_A"])).max()))
            assert run["training_data_sha256"] == reference["training_data_sha256"]
            for row in run["history"]:
                if run["mode"] == "frozen_front":
                    audits["frozen_encoder_max_drift"] = max(audits["frozen_encoder_max_drift"], row["encoder_displacement"])
                if row["encoder_rank"] != 8 or row["projection_rank"] != 1:
                    audits["rank_changes"].append([seed, run["mode"], run["ratio"], row["step"]])
    assert audits["initial_mapping_max_error"] < 1e-12
    assert audits["frozen_encoder_max_drift"] < 1e-12
    summary = {"protocol": protocol, "count": len(runs), "audit": audits, "endpoints": [], "paired_task_differences": []}
    for energy in [None, .75, 1.5, 2.25, 3.]:
        for mode in protocol["models"]:
            for ratio in protocol["ratios"]:
                group = [lookup[s, mode, ratio] for s in protocol["seeds"]]
                available = [(r, endpoint(r, energy)) for r in group if endpoint(r, energy) is not None]
                row = {"energy_section": energy, "mode": mode, "ratio": ratio,
                       "missing_seeds": [r["seed"] for r in group if endpoint(r, energy) is None]}
                for metric in ["encoder_displacement", "projection_displacement", "ell"]:
                    row[metric] = describe([e[metric] for _, e in available]) if available else None
                summary["endpoints"].append(row)
            for kind in ["encoder_risks", "projection_risks"]:
                for task in tasks:
                    values = []
                    seeds = []
                    for seed in protocol["seeds"]:
                        lo, hi = endpoint(lookup[seed, mode, .1], energy), endpoint(lookup[seed, mode, 10.], energy)
                        if lo is not None and hi is not None:
                            seeds.append(seed)
                            values.append(hi[kind][task] - lo[kind][task])
                    summary["paired_task_differences"].append({"energy_section": energy, "mode": mode,
                        "representation": kind, "task": task, "seeds": seeds,
                        "r10_minus_r01": describe(values) if values else None,
                        "positive_seed_count": sum(v > 1e-8 for v in values),
                        "negative_seed_count": sum(v < -1e-8 for v in values)})
    repeats = []
    for directory in args.repeat_dirs:
        repeats += [json.loads(p.read_text()) for p in directory.glob("seed*/result.json")]
    if repeats:
        repeat_stats = []
        for mode in ["square", "linear"]:
            per_seed = []
            for seed in protocol["seeds"]:
                group = [lookup[seed, mode, 1.]] + [r for r in repeats if r["seed"] == seed and r["mode"] == mode]
                if len(group) != 3:
                    raise ValueError("Require main plus two independent SGD streams per seed")
                initial = np.array(group[0]["history"][0]["encoder_projector"])
                deltas = np.stack([np.array(r["history"][-1]["encoder_projector"]) - initial for r in group])
                mean = deltas.mean(0)
                cosines = [float(np.vdot(deltas[i], deltas[j]) / (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j])))
                           for i, j in [(0, 1), (0, 2), (1, 2)]]
                per_seed.append({"seed": seed, "mean_pairwise_cosine": float(np.mean(cosines)),
                                 "norm_mean_deltaP": float(np.linalg.norm(mean)),
                                 "mean_squared_norm_deltaP": float((deltas**2).sum((1, 2)).mean())})
            repeat_stats.append({"mode": mode, "n_streams_per_seed": 3, "per_seed": per_seed,
                                 "cosine_across_seeds": describe([r["mean_pairwise_cosine"] for r in per_seed])})
        summary["training_stream_repeatability"] = repeat_stats
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {.1: "#3576b5", 1.: "#e08c2b", 10.: "#438d56"}
    for col, mode in enumerate(["square", "linear"]):
        for run in [r for r in runs if r["mode"] == mode]:
            h = run["history"]
            axes[0, col].plot([r["step"] for r in h], [r["encoder_displacement"] for r in h],
                              color=colors[run["ratio"]], alpha=.45, linewidth=.8)
            axes[1, col].plot([r["ell"] for r in h], [r["projection_risks"]["x_high"] for r in h],
                              color=colors[run["ratio"]], alpha=.45, linewidth=.8)
        axes[0, col].set(title=f"{mode}: complete empirical SGD", xlabel="Updates", ylabel="Encoder projector displacement")
        axes[1, col].set(title="x_high task (all other tasks in JSON)", xlabel="Log clean energy / initial energy", ylabel="Projection population risk")
        for r, color in colors.items():
            axes[0, col].plot([], [], color=color, label=f"r={r:g}")
        axes[0, col].legend(fontsize=8)
    fig.savefig(args.directory / "complete_sgd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
    for ax, kind in zip(axes, ["encoder_risks", "projection_risks"]):
        matrix = []
        for mode in protocol["models"]:
            matrix.append([next(r for r in summary["paired_task_differences"] if r["energy_section"] == 1.5 and
                                r["mode"] == mode and r["representation"] == kind and r["task"] == task)
                           ["r10_minus_r01"]["mean"] for task in tasks])
        values = np.array(matrix)
        bound = max(float(np.abs(values).max()), 1e-8)
        visual = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        ax.set_xticks(range(len(tasks)), tasks, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(protocol["models"])), protocol["models"], fontsize=8)
        ax.set_title(kind.replace("_risks", "") + ": paired mean R(r=10)-R(r=0.1)", fontsize=10)
        fig.colorbar(visual, ax=ax, label="Risk difference")
    fig.suptitle("All prespecified tasks, common log-energy=1.5, ten paired seeds", fontsize=11)
    fig.savefig(args.directory / "all_tasks_common_energy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# 完整有限数据经验 SGD", "", f"完成 {len(runs)} 条固定 {protocol['steps']} 步轨迹；统计单位为 {len(protocol['seeds'])} 个配对种子。", "",
             "| 模型 | 配比 | 固定步数 encoder 位移均值 | ell=1.5 encoder 位移均值 | ell=1.5 缺失种子 |",
             "|---|---:|---:|---:|---|"]
    for mode in protocol["models"]:
        for ratio in protocol["ratios"]:
            fixed = next(r for r in summary["endpoints"] if r["energy_section"] is None and r["mode"] == mode and r["ratio"] == ratio)
            energy = next(r for r in summary["endpoints"] if r["energy_section"] == 1.5 and r["mode"] == mode and r["ratio"] == ratio)
            label = f"{energy['encoder_displacement']['mean']:.5f}" if energy["encoder_displacement"] else "无共同支持"
            lines.append(f"| {mode} | {ratio:g} | {fixed['encoder_displacement']['mean']:.5f} | {label} | {energy['missing_seeds']} |")
    lines += ["", "完整十任务的配对风险差（r=10 减 r=0.1）、所有预设能量截面及缺失情况保存于 summary.json。",
              "能量截面采用首次向上穿越更新内的线性参数插值，只用于评估，不改变实际训练轨迹。",
              "训练不使用总体梯度；无重标定、裁剪或能量早停。后期 Q 可超过定理 T 的约束区间，该部分为外推观察。",
              "经验线性 encoder 的移动不预设为零均值噪声；配合重复流和重复批次诊断分析。"]
    if repeats:
        lines += ["", "固定初始化和有限训练集、独立批次/增强流的末端 encoder 位移方向一致性："]
        for row in summary["training_stream_repeatability"]:
            lines.append(f"- {row['mode']}: 三条流两两余弦的种子均值 {row['cosine_across_seeds']['mean']:.4f}。")
    (args.directory / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
