"""One fixed-source feasibility screen for a strict stochastic risk window.

No training, new calibration/future branches, or changes to saved predictions.
The source/task/ratio/horizon are fixed below before reading future outcomes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, variance

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/"experiments/results/empirical/task_motion_development/seed_0_square_step500"
OUT = ROOT/"experiments/results/theory/stochastic_window_feasibility.json"
NOTE = ROOT/"paper/notes/STOCHASTIC_FEASIBILITY_20261021.md"

# Fixed by the task instruction. No selection against saved future outcomes.
TASK = "x_low"
RATIO = 1.
MAX_HORIZON = 8
ALPHA = .05
REPRESENTATIONS = ("encoder", "projection")


def table(rows, fields):
    lines = ["| "+" | ".join(title for _, title in fields)+" |", "|"+"---|"*len(fields)]
    for row in rows:
        lines.append("| "+" | ".join(f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key]) for key, _ in fields)+" |")
    return "\n".join(lines)


def main():
    source = json.loads((SOURCE/"predictions.json").read_text(encoding="utf-8"))
    assert source["seed"] == 0 and source["mode"] == "square" and source["checkpoint_step"] == 500
    task_index = source["task_names"].index(TASK)
    assert source["task_families"][task_index] == "predefined_band"
    task_norm2 = sum(value*value for value in source["tasks"][task_index])
    assert abs(task_norm2-1) < 1e-12
    selected = [r for r in source["predictions"] if r["task"] == TASK and r["ratio"] == RATIO
                and r["representation"] in REPRESENTATIONS and r["horizon"] <= MAX_HORIZON]
    assert sorted({r["horizon"] for r in selected}) == [1, 4, 8]
    # Both representations are reported descriptively. Each boundary is a
    # separate alpha=.05 screen; no joint certificate across the two is issued.
    c = 1.  # Proven from 0<=R<=1, not from an observed maximum increment.
    x_h = math.log(2*MAX_HORIZON/ALPHA)
    finite_h_floor = 2*c*x_h/3
    # For every finite lambda grid: J>=1 and lambda<3/c imply
    # x/lambda > c*log(2/alpha)/3, even if V=0 and all other errors vanish.
    any_grid_floor = c*math.log(2/ALPHA)/3
    oracle_records = []
    for rep in REPRESENTATIONS:
        slope = next(r["source_slope"] for r in selected if r["representation"] == rep)
        for horizon in range(1, MAX_HORIZON+1):
            center = horizon*slope
            oracle_records.append({"representation": rep, "horizon": horizon,
                                   "frozen_source_center": center,
                                   "finite_h_bernstein_zero_variance_floor": finite_h_floor,
                                   "all_finite_lambda_grids_zero_variance_lower_floor": any_grid_floor,
                                   "finite_h_sign_window_empty": abs(center) <= finite_h_floor,
                                   "all_finite_grid_sign_windows_empty_at_c_1": abs(center) <= any_grid_floor})
    assert all(r["finite_h_sign_window_empty"] and r["all_finite_grid_sign_windows_empty_at_c_1"] for r in oracle_records)

    # Open the already saved futures only after source selection and the
    # range-only obstruction have been fixed. They do not calibrate a center.
    outcomes = json.loads((SOURCE/"outcomes.json").read_text(encoding="utf-8"))["outcomes"]
    futures = [r for r in outcomes if r["task"] == TASK and r["ratio"] == RATIO
               and r["representation"] in REPRESENTATIONS and r["horizon"] <= MAX_HORIZON]
    diagnostic_rows = []
    first_variances = {}
    for rep in REPRESENTATIONS:
        first = sorted([r for r in futures if r["representation"] == rep and r["horizon"] == 1], key=lambda r:r["repeat"])
        assert len(first) == 8 and len({r["repeat"] for r in first}) == 8
        first_variances[rep] = variance(r["observed_change"] for r in first)
    for p in selected:
        rep, horizon = p["representation"], p["horizon"]
        observations = sorted([r for r in futures if r["representation"] == rep and r["horizon"] == horizon], key=lambda r:r["repeat"])
        assert len(observations) == 8 and len({r["repeat"] for r in observations}) == 8
        actual = [r["observed_change"] for r in observations]
        coeff = source["empirical_coefficients"][str(RATIO)][rep]
        noise_q = coeff["noise"]["task"][task_index]
        drift_q = coeff["drift"]["task"][task_index]
        euler_q = coeff["euler"]["task"][task_index]
        # These surrogates preserve the old source budgets, but are NOT strict
        # drift/remainder bounds or actual variances. The variance proxy below
        # comes from eight saved exact increments, never from noise_q.
        drift_surrogate = .5*horizon*horizon*drift_q
        euler_surrogate = horizon*euler_q
        variance_proxy = horizon*first_variances[rep]
        stochastic_sqrt_proxy = math.sqrt(2*variance_proxy*x_h)
        proxy_radius = finite_h_floor+stochastic_sqrt_proxy+drift_surrogate+euler_surrogate
        diagnostic_rows.append({"representation": rep, "horizon": horizon,
                                "frozen_prediction": p["predicted_change"],
                                "source_slope": p["source_slope"], "future_streams": 8,
                                "future_mean_change_descriptive_only": mean(actual),
                                "future_sample_variance_descriptive_only": variance(actual),
                                "future_changes_by_repeat": actual,
                                "first_step_exact_increment_sample_variance_proxy": first_variances[rep],
                                "source_noise_quantile_surrogate_not_variance": noise_q,
                                "source_drift_quantile_surrogate": drift_q,
                                "source_euler_quantile_surrogate": euler_q,
                                "horizon_drift_budget_surrogate": drift_surrogate,
                                "horizon_step_budget_surrogate": euler_surrogate,
                                "horizon_variance_proxy_unproved": variance_proxy,
                                "variance_sqrt_term_proxy_unproved": stochastic_sqrt_proxy,
                                "c_1_bernstein_radius_with_empirical_proxies": proxy_radius,
                                "proxy_interval_lower": p["predicted_change"]-proxy_radius,
                                "proxy_interval_upper": p["predicted_change"]+proxy_radius,
                                "old_quantile_interval_lower": p["empirical_lower"],
                                "old_quantile_interval_upper": p["empirical_upper"],
                                "uniform_tube_bound_available": False,
                                "strict_certificate_instantiated": False})
    result = {"selection_fixed_by_instruction": {"phase":"development", "seed":0, "mode":"square", "checkpoint":500,
                                                  "task":TASK,"task_family":"predefined_band","ratio":RATIO,
                                                  "horizons":list(range(1,9)), "saved_horizons":[1,4,8],
                                                  "representations":list(REPRESENTATIONS), "alpha_per_representation":ALPHA,
                                                  "selection_used_future":False},
              "input_predictions":(SOURCE/"predictions.json").as_posix(),
              "input_outcomes":(SOURCE/"outcomes.json").as_posix(),
              "original_algorithm_settings_unchanged":source["settings"],
              "task_norm_squared_float":task_norm2,
              "global_range_argument": "For the unit predefined task, every orthogonal row projector gives 0<=R<=1. Conditional on a pre-step state, D=R_next-R has interval width 1; therefore |D-E[D|past]|<=1. No observed maximum, Gaussian gradient bound or fixed-rank assumption is needed for this range fact.",
              "c_held_fixed":c,"finite_h_log_factor":x_h,
              "finite_h_zero_variance_boundary_floor":finite_h_floor,
              "any_finite_lambda_grid_zero_variance_floor_lower_bound":any_grid_floor,
              "oracle_layer": "Source mean/variance estimation uncertainty=0, future variance=0, mean drift=0, finite-step error=0, no tube penalty, while retaining the justified c=1 increment range. Frozen source slope is used only as the stipulated idealized center, not relabeled as an independently known exact-step mean.",
              "oracle_floor_comparisons":oracle_records,
              "empirical_proxy_diagnostics":diagnostic_rows,
              "evidence_limits": ["This is a feasibility screen for a specified c=1 Bernstein/Freedman boundary, not strict stochastic validation or a proof that prediction is impossible",
                                  "The exact-step source mean and variance upper confidence bound are not stored; source tangent slope is not automatically the mean exact risk increment",
                                  "Eight future streams are used only for descriptive variance and scale comparisons, not to choose/recalibrate source means, tasks, rates or states",
                                  "The n*first-step sample variance proxy is not a proven predictable variance envelope at displaced states",
                                  "Old inflated quantiles are not actual noise variance, uniform drift/step bounds, or a verified rank/parameter tube",
                                  "c=1 is valid globally but loose. If increments were provably deterministic one could instead use c=0; a proved structural tail/local range bound could be much sharper",
                                  "Additional source calibration alone cannot reduce the c=1 boundary floor, and no source sample count is tuned here"],
              "missing_for_strict_instantiation": ["Independent exact-step or batch-tangent calibration with valid mean/variance coverage",
                                                   "A materially tighter justified conditional increment/tail scale, not an observed source maximum",
                                                   "Uniform conditional-mean drift and variance envelope on future states, plus required rank/tube controls if using local geometric formulas",
                                                   "For tangent calibration: a valid finite-step remainder bound; source euler quantiles alone are insufficient"],
              "decision":"Stop this c=1 range-only strict-window route at the fixed source; no new sampling, training, learning-rate tuning, or batch-size tuning",
              "strict_certificate_instantiated":False,
              "new_calibration_or_future_branches":0}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    main_fields = (("representation","位置"),("horizon","H"),("frozen_prediction","冻结预测"),
                   ("future_mean_change_descriptive_only","未来八流均值（描述）"),
                   ("future_sample_variance_descriptive_only","未来八流样本方差（描述）"),
                   ("horizon_drift_budget_surrogate","旧drift预算"),("horizon_step_budget_surrogate","旧step预算"))
    note = r"""# 一次固定source的严格随机窗口可行性筛查

2026-10-21。结论：**停止当前“仅用全局风险范围实例化Bernstein/Freedman严格窗口”的路线。** 在指定source上，即使把所有校准、方差、漂移、离散与tube误差都乐观设为零，只保留可证的全局增量范围尺度c=1，该保守包络仍完全压过冻结预测。继续增加源校准批次不能消除这个固定范围项。本次不新增采样/训练，不调小学习率或增大batch来寻找窗口。

这不是所有严格窗口均不存在的证明，更不是实证认定随机噪声很大；失败的是这一明确的保守范围包络实例化。

## 固定选择与原算法

在读取未来结果前，由任务指令固定：development seed0，square，checkpoint500，ratio=1，task=`x_low`，H<=8，alpha=0.05。保存元数据确认x_low属于预定义频带任务，理论任务范数为1。encoder/projection均报告，不按未来表现挑其中一个。每个位置分别以alpha=.05作可行性比较，没有声称二者联合95%覆盖。未来已保存的8条流只作经验plugin尺度对照，不重新确定预测中心。

原算法保持lr=.002、batch_size=256、16个fit批次、32个error批次、4条8步source probe、8条future流；旧区间采用0.9分位数和inflation=2。这些设置均未改变。

## 可证明的全局范围，以及它为何太粗

对单位任务，无论表示如何移动、秩是否变化，投影算子都给出0<=R<=1。给定当前状态，真实一步增量D=R_next−R落在[−R,1−R]，区间宽度1，因此中心化增量满足|D−E[D|past]|<=1。这里c=1来自风险的全局范围，不来自观察到的最大步长，也不要求Gaussian梯度有界。

采用随机理论note中有限H的保守Bernstein/Freedman包络：

\[
 B_H(v)=\sqrt{2vx}+2cx/3,\qquad x=\log(2H/\alpha).
\]

对H=8、alpha=.05、c=1，即使v=0，半径仍为 **3.84554733**。对任意有限的lambda网格，J>=1、lambda<3/c，即使v=0，其线性边界也满足

\[
 \min_\lambda\frac{\log(2J/\alpha)}{\lambda}
 > \frac{c\log(2/\alpha)}3=1.22962648.
\]

而固定source在全部1<=n<=8的最大冻结预测幅度仅为encoder **5.45943e−5**、projection **3.48344e−5**。所以当c坚持为1时，这两类明确包络均无法排除零，甚至源均值/方差估计不确定性、真实未来方差、漂移、step、tube罚项全部设为零的更乐观oracle层也不行。该比较仅把冻结source slope当作指定理想化中心，并未声称它等于真实exact-step条件均值；后者并未独立校准保存。

如果增量真为确定量，可另取c=0；若能证明更紧的局部增量范围、条件矩母函数或结构化尾界，也可能改变结论。不能将这里的c=1包络失败推广为所有方法或真实可预测性不存在。

## 已有记录的经验plugin对照

下表全部来自预先固定source/task/ratio。未来样本方差是实际risk增量的八流样本方差；它不是旧noise quantile，也不是可直接代入严格定理的variance UCB。

"""+table(diagnostic_rows,main_fields)+r"""

脚本另计算n乘首步样本方差的经验代理，以及旧source的H² q_drift/2和H q_euler预算；保留它们以后c=1项仍使区间包含零。这些代理不构成未来条件方差过程或uniform drift/remainder界。源noise quantile只原样记录，未当作variance使用。没有把观察到的最大增量当作uniform support，也未用未来均值反向替换预测中心。

## 尚缺的严格条件与退出判断

已有source记录只有切向预测与膨胀分位数，不包含独立exact-step均值/variance UCB、经证明的uniform mean-drift和未来variance包络，也没有足够紧的rank/parameter tube支持。全局risk范围虽能给增量有界，却不能代替这些局部条件；Gaussian输入也不能自动提供小的gradient/tail尺度。

本次结论限定为：当前固定source、原算法、c=1的这一保守充分条件没有实用分辨率，因此本轮停止其数值实例化。只有新的解析局部范围/尾尺度与漂移控制，才值得重新考虑；不追加校准批次、未来流或通过调整极小lr/大batch来继续“救”窗口。

输出为`experiments/results/theory/stochastic_window_feasibility.json`，含固定选择、原始设置、全部1到8步oracle比较、经验数值与缺失条件。它是可行性筛查，不是严格随机验证。

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/theory/stochastic_window_feasibility.py
```
"""
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(note,encoding="utf-8")
    print(json.dumps({"finite_h_zero_variance_floor":finite_h_floor,"any_finite_grid_floor_lower":any_grid_floor,
                      "diagnostics":[{k:r[k] for k in ("representation","horizon","frozen_prediction","future_mean_change_descriptive_only","future_sample_variance_descriptive_only")} for r in diagnostic_rows],
                      "decision":result["decision"],"strict_certificate_instantiated":False},indent=2))


if __name__ == "__main__":
    main()
