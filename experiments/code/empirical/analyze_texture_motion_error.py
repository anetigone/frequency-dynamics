"""Decompose saved frozen-forecast errors across future texture SGD streams.

Read-only on development/validation records. No training, new branches, hashes,
or refitting. Run directly with the workspace's configured Python interpreter.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT/"experiments"/"results"/"empirical"
OUT = RESULTS/"task_motion_error_decomposition"
NOTE = ROOT/"paper"/"notes"/"TEXTURE_MOTION_ERROR_DECOMPOSITION_20261021.md"


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite saved risk or prediction")
    return value


def ratio_or_none(numerator, denominator):
    return numerator/denominator if denominator > 0 else None


def process_source(prediction_path, phase):
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    outcome_path = prediction_path.with_name("outcomes.json")
    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))["outcomes"]
    task_index = {name: i for i, name in enumerate(prediction["task_names"])}
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[(float(row["ratio"]), row["representation"], row["task"], int(row["horizon"]))].append(row)
    source_rows = []
    for p in prediction["predictions"]:
        ratio, rep, task, horizon = float(p["ratio"]), p["representation"], p["task"], int(p["horizon"])
        obs = sorted(grouped[(ratio, rep, task, horizon)], key=lambda r: r["repeat"])
        n = len(obs)
        expected = prediction["settings"]["future_streams"]
        if n != expected or len({r["repeat"] for r in obs}) != expected:
            raise ValueError(f"Incomplete future streams: {prediction_path}, {ratio}, {rep}, {task}, {horizon}")
        center = finite(p["predicted_change"])
        if any(finite(r["predicted_change"]) != center for r in obs):
            raise ValueError("Saved source prediction changed across future streams")
        values = [finite(r["observed_change"]) for r in obs]
        mu = mean(values)
        mean_error = mu-center
        bias2 = mean_error**2
        variance_population = mean((v-mu)**2 for v in values)
        variance_sample = variance_population*n/(n-1)
        mse = mean((v-center)**2 for v in values)
        corrected_bias2 = bias2-variance_sample/n
        coeff = prediction["empirical_coefficients"][str(ratio)][rep]
        ix = task_index[task]
        noise_q = finite(coeff["noise"]["task"][ix])
        drift_q = finite(coeff["drift"]["task"][ix])
        euler_q = finite(coeff["euler"]["task"][ix])
        noise_budget = horizon*noise_q
        drift_budget = .5*horizon*horizon*drift_q
        euler_budget = horizon*euler_q
        radius = noise_budget+drift_budget+euler_budget
        saved_radius = .5*finite(p["interval_width"])
        assert abs(radius-saved_radius) <= 1e-12*max(1., abs(saved_radius))
        row = {"phase": phase, "seed": prediction["seed"], "mode": prediction["mode"],
               "checkpoint_step": prediction["checkpoint_step"], "ratio": ratio,
               "representation": rep, "task": task, "task_family": p["task_family"], "horizon": horizon,
               "streams": n, "rank_failure_streams": sum(not r["same_rank"] for r in obs),
               "frozen_prediction": center, "future_stream_mean": mu,
               "future_mean_minus_prediction": mean_error, "mean_offset_squared": bias2,
               "within_stream_variance_population": variance_population,
               "within_stream_variance_sample": variance_sample,
               "stream_mean_sampling_variance_estimate": variance_sample/n,
               "sample_corrected_mean_bias_squared": corrected_bias2,
               "mse": mse, "mse_identity_residual": mse-bias2-variance_population,
               "sample_corrected_identity_residual": mse-corrected_bias2-variance_sample,
               "raw_mean_offset_fraction_mse": ratio_or_none(bias2, mse),
               "sample_corrected_bias_fraction_mse": ratio_or_none(corrected_bias2, mse),
               "source_slope": finite(p["source_slope"]),
               "source_quantile_noise_surrogate": noise_q,
               "source_quantile_drift_surrogate": drift_q,
               "source_quantile_euler_surrogate": euler_q,
               "horizon_noise_budget": noise_budget, "horizon_drift_budget": drift_budget,
               "horizon_euler_budget": euler_budget, "source_interval_radius": radius,
               "absolute_mean_error_over_source_radius": ratio_or_none(abs(mean_error), radius),
               "empirical_radius_covers_future_mean": abs(mean_error) <= radius+1e-15,
               "future_stream_interval_hit_fraction": mean(r["raw_empirical_interval_hit"] for r in obs),
               "within_calibration_horizon": p["within_calibration_horizon"],
               "source_jvp_error": prediction["source_jvp_check"]["max_absolute_error"],
               "input_prediction": prediction_path.as_posix(), "input_outcomes": outcome_path.as_posix()}
        source_rows.append(row)
    # Exact retrospective split, using paired future first-step means. This is
    # not an attribution to initial calibration error versus a causal mechanism.
    first = {(r["ratio"], r["representation"], r["task"]): r for r in source_rows if r["horizon"] == 1}
    for r in source_rows:
        one = first[(r["ratio"], r["representation"], r["task"])]
        offset = r["horizon"]*one["future_mean_minus_prediction"]
        nonlinear = r["future_stream_mean"]-r["horizon"]*one["future_stream_mean"]
        r["extrapolated_future_first_step_mean_error"] = offset
        r["future_mean_nonlinear_horizon_term"] = nonlinear
        r["squared_first_step_mean_term"] = offset*offset
        r["squared_nonlinear_horizon_term"] = nonlinear*nonlinear
        r["mean_error_terms_cross_product_twice"] = 2*offset*nonlinear
        r["mean_error_split_residual"] = r["future_mean_minus_prediction"]-offset-nonlinear
        r["mean_error_square_split_residual"] = r["mean_offset_squared"]-offset*offset-nonlinear*nonlinear-2*offset*nonlinear
    return source_rows


AVERAGED = ("frozen_prediction", "future_stream_mean", "future_mean_minus_prediction", "mean_offset_squared",
            "within_stream_variance_population", "within_stream_variance_sample", "stream_mean_sampling_variance_estimate",
            "sample_corrected_mean_bias_squared", "mse", "horizon_noise_budget", "horizon_drift_budget",
            "horizon_euler_budget", "source_interval_radius", "future_stream_interval_hit_fraction",
            "extrapolated_future_first_step_mean_error", "future_mean_nonlinear_horizon_term",
            "squared_first_step_mean_term", "squared_nonlinear_horizon_term", "mean_error_terms_cross_product_twice")


def aggregate(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    records = []
    for group, cells in sorted(groups.items()):
        # Average within a source first, then across seed sources. Shared tasks,
        # streams, rates and horizons are never counted as independent trials.
        by_source = defaultdict(list)
        for row in cells:
            by_source[(row["seed"], row["mode"], row["checkpoint_step"])].append(row)
        source_means = [{k: mean(r[k] for r in values) for k in AVERAGED} for values in by_source.values()]
        out = dict(zip(fields, group))
        out.update({"source_count": len(by_source), "seed_count": len({r["seed"] for r in cells}),
                    "source_task_cells": len(cells), "streams_per_cell": sorted({r["streams"] for r in cells}),
                    "rank_failure_streams": sum(r["rank_failure_streams"] for r in cells)})
        out.update({"mean_"+k: mean(r[k] for r in source_means) for k in AVERAGED})
        mse = out["mean_mse"]
        out.update({"rmse": math.sqrt(mse),
                    "root_mean_offset_squared": math.sqrt(out["mean_mean_offset_squared"]),
                    "root_within_stream_variance_population": math.sqrt(out["mean_within_stream_variance_population"]),
                    "mean_offset_share_mse": ratio_or_none(out["mean_mean_offset_squared"], mse),
                    "within_stream_share_mse": ratio_or_none(out["mean_within_stream_variance_population"], mse),
                    "sample_corrected_bias_share_mse": ratio_or_none(out["mean_sample_corrected_mean_bias_squared"], mse),
                    "mean_radius_over_rmse": ratio_or_none(out["mean_source_interval_radius"], math.sqrt(mse)),
                    "noise_surrogate_share_radius_budget": ratio_or_none(out["mean_horizon_noise_budget"], out["mean_source_interval_radius"]),
                    "drift_surrogate_share_radius_budget": ratio_or_none(out["mean_horizon_drift_budget"], out["mean_source_interval_radius"]),
                    "euler_surrogate_share_radius_budget": ratio_or_none(out["mean_horizon_euler_budget"], out["mean_source_interval_radius"]),
                    "negative_corrected_bias_cells": sum(r["sample_corrected_mean_bias_squared"] < 0 for r in cells),
                    "negative_corrected_bias_fraction": mean(r["sample_corrected_mean_bias_squared"] < 0 for r in cells),
                    "source_radius_covers_future_mean_fraction": mean(r["empirical_radius_covers_future_mean"] for r in cells),
                    "scope": "descriptive equal-source averages; tasks/streams are repeated measurements"})
        records.append(out)
    return records


def fmt(value):
    return f"{value:.3g}"


def markdown_table(rows, fields):
    lines = ["| "+" | ".join(title for _, title in fields)+" |", "|"+"---|"*len(fields)]
    for row in rows:
        lines.append("| "+" | ".join(fmt(row[key]) if isinstance(row[key], float) else str(row[key]) for key, _ in fields)+" |")
    return "\n".join(lines)


def main():
    rows = []
    phase_counts = {}
    for phase in ("development", "validation"):
        source_dir = RESULTS/("task_motion_"+phase)
        paths = sorted(source_dir.glob("seed_*/predictions.json"))
        phase_counts[phase] = len(paths)
        for path in paths:
            rows.extend(process_source(path, phase))
    group_fields = ("phase", "mode", "checkpoint_step", "representation", "ratio", "horizon", "task_family")
    grouped = aggregate(rows, group_fields)
    all_ratio_fields = ("phase", "mode", "checkpoint_step", "representation", "horizon", "task_family")
    pooled_ratios = aggregate(rows, all_ratio_fields)
    squared_band = [r for r in grouped if r["phase"] == "validation" and r["mode"] == "square" and r["task_family"] == "predefined_band"]
    max_identity = max(abs(r["mse_identity_residual"]) for r in rows)
    max_corrected = max(abs(r["sample_corrected_identity_residual"]) for r in rows)
    max_split = max(abs(r["mean_error_square_split_residual"]) for r in rows)
    assert max(max_identity, max_corrected, max_split) < 1e-12
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT/"by_source_task.csv", rows)
    write_csv(OUT/"by_group.csv", grouped)
    write_csv(OUT/"square_band_validation.csv", squared_band)
    write_csv(OUT/"ratios_pooled.csv", pooled_ratios)
    summary = {"source_counts": phase_counts, "source_task_horizon_cells": len(rows),
               "input_roots": [(RESULTS/("task_motion_"+p)).as_posix() for p in phase_counts],
               "stream_count_values": sorted({r["streams"] for r in rows}),
               "max_mse_identity_residual": max_identity, "max_corrected_identity_residual": max_corrected,
               "max_signed_mean_error_split_residual": max(abs(r["mean_error_split_residual"]) for r in rows),
               "max_squared_mean_error_split_residual": max_split,
               "rank_failure_streams": sum(r["rank_failure_streams"] for r in rows),
               "maximum_source_jvp_error": max(r["source_jvp_error"] for r in rows),
               "negative_corrected_bias_cells": sum(r["sample_corrected_mean_bias_squared"] < 0 for r in rows),
               "estimands": {"mse": "mean_i (future_change_i - frozen_prediction)^2",
                             "mean_offset_squared": "(mean_i future_change_i - frozen_prediction)^2",
                             "within_stream_variance_population": "mean_i (future_change_i - stream_mean)^2, denominator n",
                             "within_stream_variance_sample": "sum_i (future_change_i - stream_mean)^2/(n-1)",
                             "sample_corrected_mean_bias_squared": "mean_offset_squared - within_stream_variance_sample/n; negative values retained",
                             "exact_identities": ["MSE = mean_offset_squared + within_stream_variance_population",
                                                  "MSE = sample_corrected_mean_bias_squared + within_stream_variance_sample"],
                             "conditional_mean_error_split": "mean_error_H = H*mean_error_1 + (future_mean_H - H*future_mean_1)"},
               "limitations": ["Eight streams estimate a conditional distribution at one fixed source; they do not identify a causal source of error",
                               "The uncorrected mean-offset square includes finite-eight-stream estimation variance",
                               "The sample-corrected bias-square estimator assumes independent identically distributed future streams conditional on the saved source; it can be negative and is noisy with n=8",
                               "Source noise/drift/euler records are inflated quantile error budgets in risk units, not standard deviations, variances, or identified stochastic noise",
                               "Future streams are paired across tasks, ratios and models at each seed/checkpoint; checkpoints are repeated states of the same source training run and are not independent sample sizes",
                               "The first-step/horizon split is retrospective algebra and cannot uniquely separate calibration error, SGD effects, and nonlinear geometric drift"],
               "all_group_summaries": grouped, "ratios_pooled_summaries": pooled_ratios}
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    focus = [r for r in pooled_ratios if r["phase"] == "validation" and r["mode"] == "square" and r["task_family"] == "predefined_band" and r["horizon"] in (1, 8, 64)]
    fields = (("checkpoint_step", "checkpoint"), ("representation", "位置"),
              ("mean_mse", "MSE"), ("mean_mean_offset_squared", "八流均值误差²"),
              ("mean_within_stream_variance_population", "流间总体方差"),
              ("mean_sample_corrected_mean_bias_squared", "修正均值项（原值）"),
              ("sample_corrected_bias_share_mse", "修正均值项/MSE"))
    main_focus = [r for r in focus if r["horizon"] == 8]
    first_focus = [r for r in focus if r["horizon"] == 1]
    late_focus = [r for r in focus if r["horizon"] == 64]
    calibration_focus = [r for r in focus if r["representation"] == "projection" and r["horizon"] in (8, 64)]
    calibration_fields = (("checkpoint_step", "checkpoint"), ("horizon", "H"),
                          ("mean_radius_over_rmse", "平均半径/RMSE"),
                          ("noise_surrogate_share_radius_budget", "noise预算/半径"),
                          ("drift_surrogate_share_radius_budget", "drift预算/半径"),
                          ("euler_surrogate_share_radius_budget", "euler预算/半径"),
                          ("source_radius_covers_future_mean_fraction", "覆盖八流均值比例"))
    text = r"""# 冻结起点预测误差：八条未来流的描述性分解

2026-10-21。仅分析已有 task_motion_development 与 task_motion_validation；不训练、不新增未来分支、不修订预测。

在原协议支持的H=8范围内，square频带任务的冻结预测误差同时含有八流均值偏离和流间离散，不能笼统称作某一种误差主导。下面以H=8为主结果，H=1为首步控制；H=64只作超出校准支持窗的事后诊断。这些描述都不足以把误差唯一归因于几何漂移、梯度估计或随机更新。

## 精确定义与统计限制

对固定 source、ratio、task、representation 和 horizon，设冻结预测为 p，八条未来流结果为 y_i，均值为 m。逐单元保存：

\[
\mathrm{MSE}=\tfrac18\sum_i(y_i-p)^2=(m-p)^2+\tfrac18\sum_i(y_i-m)^2.
\]

这里第二项采用分母8，才能给出精确恒等式。样本方差 s² 采用分母7；无偏修正的条件均值偏差平方估计为 (m-p)²−s²/8，**负值原样保留**。于是也有 MSE = 修正均值偏差平方 + s²。原始均值项会混入八流均值的估计误差，不能直接称作真实系统偏差；修正项在八条流下也很不稳定。

所有汇总先在source内平均任务和指定ratios，再平均source；validation有10个种子，development有2个种子，不混合两个阶段。下表合并三种ratio，但逐ratio与逐task结果完整保留。流、任务、ratio、horizon均为重复或配对测量，不作为新增独立试验数。

## 主结果：Square预定义频带任务，H=8

下面直接报告平方量原值，未先平均带符号误差再平方。三种ratio在source内等权合并，再对10个validation种子等权平均；各表检查点分别报告。所有预测单元均纳入，未按方向窗口是否可用筛选。全部H=1/4/8/16/32/64、逐ratio和逐task结果均在CSV中。

"""+markdown_table(main_focus, fields)+r"""

H=8的修正均值项/MSE在encoder的checkpoint500/2000/3500分别约31.4%/30.6%/39.2%，projection约35.6%/31.8%/52.0%。原始八流均值误差平方占比分别为encoder40.0%/39.2%/46.8%、projection43.6%/40.3%/58.0%。八流均值的采样误差校正有实质影响，不能将原始比例直接写成真实条件偏差占比。

H=8时，checkpoint2000/3500的误差幅度已经比checkpoint500大：encoder的RMSE为9.33e−5/9.52e−4/9.38e−4，projection为1.00e−4/8.40e−4/7.60e−4。它说明支持窗内也有晚期预测精度退化，但八流不足以识别退化的因果来源。

## 首步控制：H=1

"""+markdown_table(first_focus, fields)+r"""

H=1的流间总体方差项占MSE的81.7%–92.8%，而修正后的均值项占比在−6.11%至6.68%之间。checkpoint2000的projection在三个ratio下，H=1的修正均值项占比均为负（约−6.14%、−6.09%、−4.78%）。负值必须原样保留；它是有噪声的条件偏差平方估计，不能解释为无偏或偏差不存在。

## 保存的校准误差能说明什么

source中的 noise、drift、euler 系数分别来自单步梯度偏差的绝对任务误差分位数、短校准分支上冻结fit目标的切向漂移分位数、真实一步更新相对切向近似的余项分位数。它们都乘过固定inflation；总半径为 H q_noise + H² q_drift/2 + H q_euler。这些是**经验误差预算**，量纲为风险变化，不是噪声标准差或方差；不能用其数值占比来声称实际随机噪声占比。

下面给出projection的半径组成；合并ratio方式与上表一致。

"""+markdown_table(calibration_focus, calibration_fields)+r"""

H=8时，校准半径中名为noise的预算占约92%–98%，但实际MSE的流间总体方差占约42%–61%，修正均值项占约31%–52%。这些量描述的对象和量纲不同，不能相互换算。H=8的半径覆盖全部这些八流均值，平均半径约为RMSE的3.6–5.7倍；覆盖充分与区间足够紧是不同问题。H=64项仅供超窗诊断，不是支持窗内的有效性结论。

逐单元结果保留三项预算、未来均值是否落入半径、均值误差/半径以及未来流覆盖率。它们可检查校准区间的松紧，却不能从quantile半径恢复真实梯度噪声分布。原记录没有保留有符号逐批梯度误差样本；单靠这里的三个quantile无法识别真实mean-gradient估计偏差。

另保存精确的事后均值误差拆分：

\[
m_H-p_H=H(m_1-p_1)+(m_H-Hm_1).
\]

第一项把实际八流首步均值偏差线性外推，第二项为未来均值偏离该外推的量。两项可抵消；对应平方和必须加交叉项才能等于均值误差平方。这是使用未来结果的描述性分解，不是新预测，也不能把第二项单独归因为“状态漂移”：非线性、条件随机更新与几何变化都可能参与。

例如checkpoint3500、encoder、H=8的合并ratio结果，首步均值偏差外推项的均方为7.87e−7，非线性horizon项均方为5.91e−7，交叉项为−9.67e−7，最终均值偏差平方只有4.11e−7。该抵消直接说明，不能把某一个正平方量单独称为解释误差的比例。首步均值本身也由同一组八流估计，外推误差含采样误差，不能据此识别初始梯度估计与后续动力学的因果贡献。

## 超出支持窗的诊断：H=64

H=64位于原本8步校准探测窗以外，下表只能描述冻结线性外推在远期的误差结构。

"""+markdown_table(late_focus, fields)+r"""

H=64的修正均值项/MSE在encoder的checkpoint500/2000/3500分别约76.8%/80.6%/88.0%，projection约84.9%/77.8%/83.6%。这不支持将超窗误差笼统视为流间波动，也不证明几何漂移是唯一原因。对ratio=1单独看，encoder约78.2%/80.4%/87.8%，projection约85.0%/77.7%/83.7%；逐ratio数据还显示显著幅度差异，合并表不能替代完整结果。

## 输出与复现

- `experiments/results/empirical/task_motion_error_decomposition/by_source_task.csv`：逐source/ratio/task/位置/H的完整分解、负修正值与校准预算。
- `by_group.csv`：逐phase/model/checkpoint/位置/ratio/H/任务族的等source汇总。
- `square_band_validation.csv`：validation square频带任务的逐ratio完整表。
- `ratios_pooled.csv`：只作描述性展示的ratio合并表。
- `summary.json`：公式、输入路径、直接数值检查与全部汇总。

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/analyze_texture_motion_error.py
```
"""
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(text, encoding="utf-8")
    print(json.dumps({"source_counts": phase_counts, "cells": len(rows), "max_identity_residual": max_identity,
                      "negative_corrected_bias_cells": summary["negative_corrected_bias_cells"],
                      "square_band_focus": [{k: r[k] for k in ("checkpoint_step", "representation", "horizon", "rmse", "mean_offset_share_mse", "sample_corrected_bias_share_mse")} for r in focus]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
