"""Seed-level summaries, protocol audits and standalone subspace figures."""

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


def describe(values):
    x = np.asarray(values)
    samples = np.random.default_rng(20261019).integers(0, len(x), (10000, len(x)))
    lower, upper = np.quantile(x[samples].mean(1), [.025, .975])
    return {"n_seeds": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "minimum": float(x.min()), "maximum": float(x.max()),
            "seed_bootstrap_mean_ci95": [float(lower), float(upper)]}


def audit_and_summarize(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(directory.glob("seed_*.json"))]
    expected = len(protocol["seeds"]) * len(protocol["arms"]) * len(protocol["models"]) * len(protocol["ratios"])
    if len(runs) != expected:
        raise ValueError(f"{directory}: {len(runs)}/{expected} completed; do not report partial confirmation")
    lookup = {(r["seed"], r["arm"], r["mode"], r["ratio"]): r for r in runs}
    max_a_error, max_b_error, max_frozen_ratio_error = 0., 0., 0.
    for seed in protocol["seeds"]:
        anchor = lookup[seed, "matched", "square", 1.]
        a0, b0 = np.array(anchor["initial_A"]), np.array(anchor["initial_B"])
        for arm in protocol["arms"]:
            for mode in protocol["models"]:
                for ratio in protocol["ratios"]:
                    run = lookup[seed, arm, mode, ratio]
                    max_a_error = max(max_a_error, float(np.abs(np.array(run["initial_A"]) - a0).max()))
                    if arm == "matched":
                        max_b_error = max(max_b_error, float(np.abs(np.array(run["initial_B"]) - b0).max()))
                    if arm == "independently_calibrated":
                        assert abs(run["calibration"]["weighted_initial_total_trace"] - 1) < 1e-10
                    if mode == "frozen_front":
                        reference = lookup[seed, arm, mode, 1.]["history"]
                        for row, ref in zip(run["history"], reference):
                            for name, risk in row["projection_task_risks"].items():
                                max_frozen_ratio_error = max(max_frozen_ratio_error,
                                                             abs(risk - ref["projection_task_risks"][name]))
    assert max_a_error < 1e-12 and max_b_error < 1e-12 and max_frozen_ratio_error < 1e-8
    summaries, paired = [], []
    metrics = ["encoder_projector_displacement", "projection_projector_displacement", "max_encoder_task_risk_change"]
    for arm in protocol["arms"]:
        for mode in protocol["models"]:
            for ratio in protocol["ratios"]:
                group = [lookup[seed, arm, mode, ratio] for seed in protocol["seeds"]]
                entry = {"arm": arm, "mode": mode, "ratio": ratio}
                for metric in metrics:
                    entry[metric] = describe([run["history"][-1][metric] for run in group])
                entry["encoder_rank_range"] = [min(row["encoder_rank"] for run in group for row in run["history"]),
                                                 max(row["encoder_rank"] for run in group for row in run["history"])]
                entry["projection_rank_range"] = [min(row["projection_rank"] for run in group for row in run["history"]),
                                                    max(row["projection_rank"] for run in group for row in run["history"])]
                summaries.append(entry)
        for ratio in protocol["ratios"]:
            differences = [lookup[seed, arm, "square", ratio]["history"][-1]["encoder_projector_displacement"] -
                           lookup[seed, arm, "linear", ratio]["history"][-1]["encoder_projector_displacement"]
                           for seed in protocol["seeds"]]
            paired.append({"arm": arm, "ratio": ratio, "endpoint": "square_minus_linear_encoder_displacement",
                           **describe(differences)})
    return {"protocol": protocol, "completed_runs": len(runs), "summaries": summaries,
            "paired_seed_contrasts": paired, "audit": {"max_initial_A_error": max_a_error,
            "max_matched_initial_B_error": max_b_error, "max_frozen_front_ratio_risk_error": max_frozen_ratio_error}}, runs


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULTS / "population")
    args = parser.parse_args()
    dirs = ["subspace_confirm_k1", "subspace_confirm_k2", "subspace_confirm_local"]
    output = args.root / "subspace_summary"
    output.mkdir(parents=True, exist_ok=True)
    all_results, raw = {}, {}
    for directory in dirs:
        all_results[directory], raw[directory] = audit_and_summarize(args.root / directory)
    (output / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"square": "#3576b5", "linear": "#e08c2b", "frozen_front": "#438d56"}
    for row, directory in enumerate(dirs):
        for col, metric in enumerate(["encoder_projector_displacement", "projection_projector_displacement"]):
            ax = axes[row, col]
            labels = []
            for j, mode in enumerate(["square", "linear", "frozen_front"]):
                for k, ratio in enumerate([.1, 1., 10.]):
                    values = [r["history"][-1][metric] for r in raw[directory]
                              if r["arm"] == "independently_calibrated" and r["mode"] == mode and r["ratio"] == ratio]
                    xpos = j * 3 + k
                    ax.scatter(xpos + np.linspace(-.12, .12, len(values)), values, s=13,
                               color=colors[mode], alpha=.75)
                    ax.plot([xpos - .22, xpos + .22], [np.mean(values)] * 2, color="black", linewidth=1)
                    labels.append(f"{mode.replace('_front', '')}\nr={ratio:g}")
            ax.set_xticks(range(9), labels, fontsize=7)
            ax.set_ylabel("Projector Frobenius displacement")
            ax.set_title(directory.replace("subspace_confirm_", "") + ": " + ("encoder" if col == 0 else "projection"))
            ax.axhline(0, color="grey", linewidth=.6)
    fig.suptitle("Independent per-model calibration; ten paired seeds; endpoint log-energy=3", fontsize=12)
    fig.savefig(output / "subspace_displacements.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# 子空间机制：十种子确认结果", "", "统计单位为种子；各设置共享同一批配对种子，不是额外独立重复。", "",
             "| 设置 | 配比 | 平方 encoder 位移均值 [最小, 最大] | 线性 encoder 位移最大值 | 平方 encoder 任务风险最大变化的种子均值 |", "|---|---:|---:|---:|---:|"]
    for directory in dirs:
        for ratio in [.1, 1., 10.]:
            entries = all_results[directory]["summaries"]
            square = next(e for e in entries if e["arm"] == "independently_calibrated" and e["mode"] == "square" and e["ratio"] == ratio)
            linear = next(e for e in entries if e["arm"] == "independently_calibrated" and e["mode"] == "linear" and e["ratio"] == ratio)
            x = square["encoder_projector_displacement"]
            lines.append(f"| {directory.removeprefix('subspace_confirm_')} | {ratio:g} | {x['mean']:.4f} [{x['minimum']:.4f}, {x['maximum']:.4f}] | {linear['encoder_projector_displacement']['maximum']:.2e} | {square['max_encoder_task_risk_change']['mean']:.4f} |")
    lines += ["", "全部模型、配比、两套校准和配对种子 bootstrap 均保存在 summary.json。", "",
              "初始 A 对齐、匹配组初始 B 对齐、独立校准组初始加权总 Gram 迹为 1、冻结前层的等能量轨迹不依赖配比等审计均通过。",
              "正交坐标任务的总风险恒为 d−rank；任务收益与损失是固定预算的重分配。",
              "线性 encoder 不变属于当前各向同性总体目标的结构性质，不可推广到经验 SGD。",
              "这些结果支持当前模型族的机制区别，不证明自然图像 encoder 退化或平方网络独有的风险回升。"]
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
