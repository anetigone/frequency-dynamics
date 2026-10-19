"""Sign probabilities and amplitude predictions at genuinely empirical states."""
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
    runs = [json.loads(p.read_text()) for p in sorted(args.directory.glob("seed*/result.json"))]
    if len(runs) != len(protocol["seeds"]) * len(protocol["models"]) * len(protocol["steps"]):
        raise ValueError("Incomplete empirical-state intervention study")
    conditional = []
    thresholds = []
    for run in runs:
        lookup = {}
        for outcome in run["outcomes"]:
            if outcome["ratio"] not in [.1, 1., 10.]:
                continue
            key = (outcome["ratio"], outcome["task"], outcome["horizon"])
            lookup.setdefault(key, []).append(outcome)
        for (ratio, task, horizon), rows in lookup.items():
            values = np.array([r["observed_change"] for r in rows])
            predicted = rows[0]["predicted_change"]
            mean = float(values.mean())
            probability = float(np.mean(values * predicted > 0)) if abs(predicted) > 1e-9 else None
            conditional.append({"seed": run["seed"], "mode": run["mode"], "step": run["training_step"],
                "ratio": ratio, "task": task, "horizon": horizon, "n_batch_streams": len(rows),
                "predicted_change": predicted, "mean_observed_change": mean,
                "observed_change_std": float(values.std(ddof=1)),
                "mean_observed_change_se": float(values.std(ddof=1) / len(rows)**.5),
                "probability_of_predicted_sign": probability,
                "probability_of_decrease": float(np.mean(values < 0)),
                "sign_of_batch_mean_matches": mean * predicted > 0,
                "amplitude_absolute_error": abs(mean - predicted)})
        primary = [row for row in run["thresholds"] if row["task"] == "x_high"]
        pop = primary[0]["population_prediction"]
        empirical = [r["empirical_derivative_reference"] for r in primary]
        errors = [abs(np.log10(pop / e)) for e in empirical if pop is not None and e is not None]
        thresholds.append({"seed": run["seed"], "mode": run["mode"], "step": run["training_step"],
                           "population_positive_threshold": pop,
                           "empirical_positive_threshold_probability": sum(e is not None for e in empirical) / len(empirical),
                           "conditional_mean_log10_error": float(np.mean(errors)) if errors else None,
                           "comparable_replicates": len(errors), "total_replicates": len(empirical)})
    summaries = []
    for mode in protocol["models"]:
        for step in protocol["steps"]:
            for horizon in ([1, 8, 64] if step == 2000 else [1]):
                per_seed = []
                for seed in protocol["seeds"]:
                    group = [r for r in conditional if (r["seed"], r["mode"], r["step"], r["horizon"]) == (seed, mode, step, horizon)]
                    active = [r for r in group if r["probability_of_predicted_sign"] is not None]
                    per_seed.append({"seed": seed,
                        "sign_probability": float(np.mean([r["probability_of_predicted_sign"] for r in active])) if active else None,
                        "constant_decrease_probability": float(np.mean([r["probability_of_decrease"] for r in active])) if active else None,
                        "sign_match_of_replicate_mean": float(np.mean([r["sign_of_batch_mean_matches"] for r in active])) if active else None,
                        "mean_amplitude_error": float(np.mean([r["amplitude_absolute_error"] for r in group]))})
                entry = {"mode": mode, "step": step, "horizon": horizon, "per_seed": per_seed}
                for key in ["sign_probability", "constant_decrease_probability", "sign_match_of_replicate_mean", "mean_amplitude_error"]:
                    values = [r[key] for r in per_seed if r[key] is not None]
                    entry[key] = describe(values) if values else None
                summaries.append(entry)
    result = {"protocol": protocol, "scope": "ten-seed inference; batch repeats conditional on fixed empirical states",
              "conditional_cells": conditional, "seed_summaries": summaries, "threshold_probabilities": thresholds}
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for mode in protocol["models"]:
        rows = [r for r in summaries if r["mode"] == mode and r["horizon"] == 1]
        axes[0].plot([r["step"] for r in rows], [r["sign_probability"]["mean"] if r["sign_probability"] else np.nan for r in rows], "o-", label=mode)
        rows = [r for r in summaries if r["mode"] == mode and r["step"] == 2000]
        axes[1].plot([r["horizon"] for r in rows], [r["mean_amplitude_error"]["mean"] for r in rows], "o-", label=mode)
    axes[0].set(xlabel="Empirical pretraining updates", ylabel="Probability of predicted one-step sign", ylim=(0, 1.03))
    axes[1].set(xlabel="Fork horizon at empirical step 2000", ylabel="Error vs batch-repeat mean risk change", xscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(args.directory / "empirical_state_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# 经验训练 checkpoint 上的重复分叉", "", "32 条独立未来批次/增强流检验单步符号概率；训练中点的前 8 条流继续到 8/64 步。统计单位仍为十个种子。", "",
             "| 模型 | 训练步数 | 分叉窗口 | 符号概率 | 恒预测降低 | 重复均值的符号吻合率 | 幅度 MAE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        def f(key):
            return f"{r[key]['mean']:.1%}" if r[key] else "数值中性"
        lines.append(f"| {r['mode']} | {r['step']} | {r['horizon']} | {f('sign_probability')} | {f('constant_decrease_probability')} | {f('sign_match_of_replicate_mean')} | {r['mean_amplitude_error']['mean']:.6f} |")
    lines += ["", "预设主任务 x_high 的阈值诊断（每格十个训练状态，各有 32 个相关于该状态的条件重复）：", "",
              "| 模型 | 训练步数 | 总体正阈值状态数 | 平均经验正阈值概率 | 条件平均绝对 log10 误差 |", "|---|---:|---:|---:|---:|"]
    for mode in protocol["models"]:
        for step in protocol["steps"]:
            group = [r for r in thresholds if r["mode"] == mode and r["step"] == step]
            errors = [r["conditional_mean_log10_error"] for r in group if r["conditional_mean_log10_error"] is not None]
            error_text = f"{np.mean(errors):.3f}（{len(errors)} 个适用种子）" if errors else "不适用"
            lines.append(f"| {mode} | {step} | {sum(r['population_positive_threshold'] is not None for r in group)} | "
                         f"{np.mean([r['empirical_positive_threshold_probability'] for r in group]):.1%} | {error_text} |")
    lines += ["", "符号与阈值预测分开：每个状态的经验正阈值出现概率、可比较重复数、条件 log10 误差完整保存在 summary.json。",
              "预测使用已知任务 beta 和总体几何，执行使用有限数据经验 SGD。晚期总体梯度弱时，随机批次和有限数据偏移可能主导。",
              "8/64 步更高的符号概率不能解释成线性展开误差更小；幅度误差单列报告。",
              "排除预测变化绝对值不超过 1e-9 的数值中性项；不把大量任务、批次或 checkpoint 当作独立种子。"]
    (args.directory / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
