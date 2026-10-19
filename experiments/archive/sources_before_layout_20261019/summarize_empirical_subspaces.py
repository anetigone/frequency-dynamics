"""Held-out empirical branch summaries; random seed is the inference unit."""
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
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).parent / "results" / "subspace_empirical_confirm")
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    paths = sorted(args.directory.glob("seed_*/result.json"))
    expected = len(protocol["seeds"]) * len(protocol["checkpoints"])
    if len(paths) != expected:
        raise ValueError(f"{len(paths)}/{expected} completed; do not summarize partial confirmation")
    runs = [json.loads(p.read_text()) for p in paths]
    summary = {"protocol": protocol, "checkpoint_count": len(runs),
               "scope": "population oracle predictions vs real empirical SGD, conditional on population-trained checkpoints",
               "sign_neutral_band": 1e-9, "sign_grid": [.1, 1., 10.],
               "horizon_summaries": [], "critical_ratio_summaries": []}
    for mode in protocol["models"]:
        for horizon in protocol["horizons"]:
            per_seed = []
            for seed in protocol["seeds"]:
                outcomes = [o for r in runs if r["seed"] == seed for o in r["outcomes"]
                            if o["mode"] == mode and o["horizon"] == horizon and o["ratio"] in [.1, 1., 10.]]
                reliable = [o for o in outcomes if abs(o["predicted_population_change"]) > 1e-9]
                # Each branch appears ten times (tasks); deduplicate geometry.
                branch_drift = {(r["checkpoint_ell"], o["ratio"]): o["encoder_projector_displacement"]
                                for r in runs if r["seed"] == seed for o in r["outcomes"]
                                if o["mode"] == mode and o["horizon"] == horizon and o["ratio"] in [.1, 1., 10.]}
                if mode == "frozen_front":
                    assert max(branch_drift.values()) < 1e-12
                per_seed.append({"seed": seed, "nonneutral_count": len(reliable),
                                 "sign_accuracy": float(np.mean([o["predicted_population_change"] * o["observed_empirical_change"] > 0 for o in reliable])),
                                 "always_decrease_accuracy": float(np.mean([o["observed_empirical_change"] < 0 for o in reliable])),
                                 "always_increase_accuracy": float(np.mean([o["observed_empirical_change"] > 0 for o in reliable])),
                                 "mae_risk_change": float(np.mean([abs(o["predicted_population_change"] - o["observed_empirical_change"]) for o in outcomes])),
                                 "mean_encoder_drift": float(np.mean(list(branch_drift.values())))})
            summary["horizon_summaries"].append({"mode": mode, "horizon": horizon,
                "sign_accuracy": describe([r["sign_accuracy"] for r in per_seed]),
                "always_decrease_accuracy": describe([r["always_decrease_accuracy"] for r in per_seed]),
                "always_increase_accuracy": describe([r["always_increase_accuracy"] for r in per_seed]),
                "mae_risk_change": describe([r["mae_risk_change"] for r in per_seed]),
                "encoder_projector_displacement": describe([r["mean_encoder_drift"] for r in per_seed]),
                "per_seed": per_seed})
        counts = {"both_positive": 0, "population_only_positive": 0,
                  "empirical_only_positive": 0, "neither_positive": 0}
        per_seed = []
        for seed in protocol["seeds"]:
            logs, fitted_logs = [], []
            for result in runs:
                if result["seed"] != seed:
                    continue
                row = next(r for r in result["critical_ratios"] if r["mode"] == mode and r["task"] == "x_high")
                pop, empirical, fitted = (row[key] for key in ["population_prediction", "empirical_derivative_reference", "one_step_fitted_critical"])
                counts["both_positive" if pop is not None and empirical is not None else
                       "population_only_positive" if pop is not None else
                       "empirical_only_positive" if empirical is not None else "neither_positive"] += 1
                if pop is not None and empirical is not None:
                    logs.append(abs(np.log10(pop / empirical)))
                if fitted is not None and empirical is not None:
                    fitted_logs.append(abs(np.log10(fitted / empirical)))
            per_seed.append({"seed": seed, "population_vs_empirical_log10_error": float(np.mean(logs)) if logs else None,
                             "one_step_fit_vs_empirical_log10_error": float(np.mean(fitted_logs)) if fitted_logs else None})
        valid = [r["population_vs_empirical_log10_error"] for r in per_seed if r["population_vs_empirical_log10_error"] is not None]
        summary["critical_ratio_summaries"].append({"mode": mode, "task": "x_high",
            "checkpoint_classification_counts_dependent": counts,
            "conditional_log10_error_across_eligible_seeds": describe(valid) if valid else None,
            "per_seed": per_seed})
    output = args.directory / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for mode in protocol["models"]:
        rows = [r for r in summary["horizon_summaries"] if r["mode"] == mode]
        for ax, metric in zip(axes, ["sign_accuracy", "encoder_projector_displacement"]):
            means = np.array([r[metric]["mean"] for r in rows])
            ci = np.array([r[metric]["seed_bootstrap_mean_ci95"] for r in rows]).T
            ax.errorbar([r["horizon"] for r in rows], means, yerr=np.vstack((means - ci[0], ci[1] - means)),
                        fmt="o-", capsize=3, label=mode)
    square_rows = [r for r in summary["horizon_summaries"] if r["mode"] == "square"]
    axes[0].plot([r["horizon"] for r in square_rows],
                 [r["always_decrease_accuracy"]["mean"] for r in square_rows],
                 ":", color="grey", label="always decrease (square)")
    axes[0].set(xscale="log", xlabel="Real empirical-SGD updates", ylabel="Seed-averaged sign agreement", ylim=(0, 1.03))
    axes[1].set(xscale="log", xlabel="Real empirical-SGD updates", ylabel="Mean encoder projector displacement")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Population-geometry prediction on held-out checkpoints; ten paired seeds", fontsize=11)
    fig.savefig(args.directory / "empirical_interventions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# 未用于开发的 checkpoint：真实经验 SGD 分叉", "", "统计单位是种子（10 个），每个种子的三个 checkpoint、任务、配比和时间窗口均相关。", "",
             "| 模型 | 步数 | 种子平均符号吻合率 | 恒预测降低 | 风险变化预测 MAE | encoder 行空间位移均值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary["horizon_summaries"]:
        lines.append(f"| {row['mode']} | {row['horizon']} | {row['sign_accuracy']['mean']:.1%} | {row['always_decrease_accuracy']['mean']:.1%} | {row['mae_risk_change']['mean']:.5f} | {row['encoder_projector_displacement']['mean']:.5f} |")
    lines += ["", "主任务 x_high 的临界配比比较（每模型 30 个相关 checkpoint，不是 30 个独立重复）：", "",
              "| 模型 | 总体与经验都有正阈值 | 仅总体有 | 仅经验有 | 两者均无 | 条件平均绝对 log10 误差 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["critical_ratio_summaries"]:
        counts = row["checkpoint_classification_counts_dependent"]
        error = row["conditional_log10_error_across_eligible_seeds"]
        label = f"{error['mean']:.3f}（{error['n_seeds']} 个适用种子）" if error else "不适用"
        lines.append(f"| {row['mode']} | {counts['both_positive']} | {counts['population_only_positive']} | {counts['empirical_only_positive']} | {counts['neither_positive']} | {label} |")
    lines += ["", "符号主汇总只用预设配比 0.1/1/10 和全部十个任务；排除预测变化绝对值不超过 1e-9 的数值中性项。",
              "临界配比括号实验另存原始结果，不通过加入阈值附近分叉来提高主汇总的吻合率。",
              "临界配比误差仅在总体和首批经验向量场都存在正阈值时可定义；无正阈值的案例计数完整保留，见 summary.json。",
              "这里预测的是总体向量场，执行的是共享数据/增强的真实经验 SGD；差异包含样本噪声和有限窗口非线性。",
              "线性模型的总体前层方向贡献严格为零，不将其浮点抵消误差计作正临界阈值。经验目标没有这个恒等式。",
              "checkpoint 来自总体目标训练，本实验不等于完整有限数据预训练的十种子验证，也不涉及自然图像。",
              "已知任务 beta 参与预测，仍是任务条件的 oracle 机制检验，不是无需标签的调参算法。"]
    (args.directory / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
