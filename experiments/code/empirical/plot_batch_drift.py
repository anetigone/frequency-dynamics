"""Finite-step conditional projector means and variance vs batch size."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, result_path, source_path
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = result_path("batch_drift_diagnostic")


def main():
    result = json.loads((ROOT / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for mode, color in [("square", "#3576b5"), ("linear", "#e08c2b")]:
        for seed, style in [(0, "-"), (1, "--")]:
            rows = [r for r in result["rows"] if r["mode"] == mode and r["seed"] == seed]
            x = [r["batch_size"] for r in rows]
            axes[0].plot(x, [r["norm_mean_delta_projector"] for r in rows], "o" + style,
                         color=color, label=f"{mode}, seed={seed}")
            axes[1].plot(x, [r["projector_step_variance_trace"] for r in rows], "o" + style,
                         color=color)
    axes[0].set(xscale="log", yscale="log", xlabel="Batch size", ylabel="Norm of mean projector step")
    axes[1].set(xscale="log", yscale="log", xlabel="Batch size", ylabel="E ||step - mean step||_F^2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.08), ncol=4, fontsize=8)
    fig.suptitle("64 independent batches at fixed empirical states; frozen-front steps are zero", fontsize=10)
    fig.savefig(ROOT / "batch_mean_variance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# 固定经验状态的批次诊断", "", "两个开发种子，各模型在经验训练第 2000 步的状态；每个 batch 大小抽 64 次独立批次/增强。", "",
             "| 种子 | 模型 | batch | ||E Delta P|| | E||Delta P||² | 均值抽样误差 RMS 估计 |", "|---|---|---:|---:|---:|---:|"]
    for r in result["rows"]:
        lines.append(f"| {r['seed']} | {r['mode']} | {r['batch_size']} | {r['norm_mean_delta_projector']:.3e} | {r['mean_squared_norm_delta_projector']:.3e} | {r['rms_sampling_error_of_mean']:.3e} |")
    lines += ["", "均值向量、平均平方位移、方差迹、相对总体有限步更新的偏差均保存在 results.json。",
              "这里的平均针对同一有限训练集的批次与增强，不针对重新抽取训练集；大 batch 下的剩余平均偏移可包含有限数据、非线性矩估计与有限步长效应。",
              "不能将线性模型的全部移动预设为零均值噪声；本诊断也不足以分离上述所有偏移来源。",
              "64 次批次重复是条件诊断，不等于 64 个独立训练种子。"]
    (ROOT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
