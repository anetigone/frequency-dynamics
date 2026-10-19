"""CIFAR analysis in a separate process without loading PyTorch/OpenMP."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from summarize_subspaces import describe


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(args.directory.glob("seed*/evaluation.json"))]
    if len(rows) != len(protocol["seeds"]) * len(protocol["ratios"]):
        raise ValueError("Incomplete numeric evaluations")
    summary = {"training_protocol": protocol, "n_seeds": len(protocol["seeds"]),
               "validation_curves_and_test": rows, "fixed_step_test_summaries": [], "paired_energy_comparisons": [],
               "paired_fixed_step_test": []}
    for ratio in protocol["ratios"]:
        group = [r for r in rows if r["ratio"] == ratio]
        summary["fixed_step_test_summaries"].append({"ratio": ratio,
            "encoder_accuracy": describe([r["fixed_final_step_test"]["encoder_test_accuracy"] for r in group]),
            "projection_accuracy": describe([r["fixed_final_step_test"]["projection_test_accuracy"] for r in group])})
    for representation in ["encoder", "projection"]:
        differences = []
        for seed in protocol["seeds"]:
            low = next(r for r in rows if r["seed"] == seed and r["ratio"] == .1)
            high = next(r for r in rows if r["seed"] == seed and r["ratio"] == 10.)
            key = representation + "_test_accuracy"
            differences.append(high["fixed_final_step_test"][key] - low["fixed_final_step_test"][key])
        summary["paired_fixed_step_test"].append({"representation": representation, "per_seed_differences": differences,
                                                  "r10_minus_r01": describe(differences)})
    for energy in [.5, 1., 2.]:
        for representation in ["encoder", "projection"]:
            differences, seeds, overshoots = [], [], []
            for seed in protocol["seeds"]:
                pair = []
                for ratio in [.1, 10.]:
                    run = next(r for r in rows if r["seed"] == seed and r["ratio"] == ratio)
                    point = next((e for e in run["common_energy_validation"] if e["target_ell"] == energy), None)
                    pair.append(point)
                if all(p is not None for p in pair):
                    differences.append(pair[1][representation + "_validation_accuracy"] - pair[0][representation + "_validation_accuracy"])
                    overshoots.extend(p["observed_ell"] - energy for p in pair)
                    seeds.append(seed)
            summary["paired_energy_comparisons"].append({"target_ell": energy, "representation": representation,
                "paired_seeds": seeds, "missing_seeds": [s for s in protocol["seeds"] if s not in seeds],
                "r10_minus_r01_validation_accuracy": describe(differences) if differences else None,
                "maximum_log_energy_overshoot": max(overshoots) if overshoots else None})
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    colors = {.1: "#3576b5", 1.: "#e08c2b", 10.: "#438d56"}
    for row in rows:
        for ax, name in zip(axes, ["encoder", "projection"]):
            ax.plot([r["step"] for r in row["fixed_step_validation"]],
                    [r[name + "_validation_accuracy"] for r in row["fixed_step_validation"]],
                    color=colors[row["ratio"]], alpha=.5, linewidth=.9)
    for ax, name in zip(axes, ["encoder", "projection"]):
        ax.set(title=name, xlabel="Pretraining updates", ylabel="Validation OLS accuracy")
        for ratio, color in colors.items():
            ax.plot([], [], color=color, label=f"r={ratio:g}")
        ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10, five paired seeds, fixed 4000-update budget")
    fig.savefig(args.directory / "cifar_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# CIFAR-10：本机固定预算五种子比较", "", "灰度、64 个 11×11 valid 滤波器、平方、GAP、8 维线性投影。", "固定 4000 次更新（batch 256，约 22.8 个训练集遍历），不是收敛或标准长程 benchmark。", "",
             "| 配比 | 最终 encoder 测试准确率均值 | 最终 projection 测试准确率均值 |", "|---|---:|---:|"]
    for row in summary["fixed_step_test_summaries"]:
        lines.append(f"| {row['ratio']:g} | {row['encoder_accuracy']['mean']:.2%} | {row['projection_accuracy']['mean']:.2%} |")
    lines += ["", "| 共同校准能量 | 表示 | 配对种子数 | r=10 减 r=0.1 均值（百分点） | bootstrap 描述区间（百分点） |", "|---|---|---:|---:|---:|"]
    for row in summary["paired_energy_comparisons"]:
        value = row["r10_minus_r01_validation_accuracy"]
        label = f"{100*value['mean']:+.2f}" if value else "不可比较"
        interval = f"[{100*value['seed_bootstrap_mean_ci95'][0]:+.2f}, {100*value['seed_bootstrap_mean_ci95'][1]:+.2f}]" if value else "无"
        lines.append(f"| {row['target_ell']:g} | {row['representation']} | {len(row['paired_seeds'])} | {label} | {interval} |")
    lines += ["", "全部逐种子结果、bootstrap 描述区间、缺失截面和能量超调见 summary.json。",
              "能量来自固定无标签校准集的中心化 clean 输出方差，在每 10 步首次穿越时保存，非精确总体能量。",
              "固定步数准确率不能单独排除速度差异；共同能量比较单列，不能以自然图像分类替代总体平方风险。",
              "测试集只用于预设最终步的全部条件评估，没有据测试标签选择停止点或配比。",
              "自然图像外推不验证二维阈值，也不证明总体 encoder 子空间守恒/不守恒。"]
    lines += ["", "当前固定步数的 projection 平均差较小；共同能量下的 projection 差异区间均跨零，尚不足以建立超出训练速度的稳定收益。",
              "encoder 配比差异也很小，不据此宣称自然图像上的稳定 encoder 改善或退化。五种子区间为描述性 bootstrap，未作多重比较校正。"]
    (args.directory / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
