# 一次固定source的严格随机窗口可行性筛查

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

| 位置 | H | 冻结预测 | 未来八流均值（描述） | 未来八流样本方差（描述） | 旧drift预算 | 旧step预算 |
|---|---|---|---|---|---|---|
| encoder | 1 | 6.82429e-06 | 6.01833e-06 | 2.56609e-11 | 5.39909e-09 | 6.78764e-09 |
| encoder | 4 | 2.72972e-05 | 2.98799e-05 | 4.82558e-11 | 8.63855e-08 | 2.71505e-08 |
| encoder | 8 | 5.45943e-05 | 6.29656e-05 | 2.46586e-10 | 3.45542e-07 | 5.43011e-08 |
| projection | 1 | -4.35431e-06 | 6.34255e-06 | 1.10344e-10 | 1.91405e-08 | 2.80599e-08 |
| projection | 4 | -1.74172e-05 | 1.69178e-06 | 4.72495e-10 | 3.06247e-07 | 1.1224e-07 |
| projection | 8 | -3.48344e-05 | 1.26573e-05 | 5.43049e-10 | 1.22499e-06 | 2.24479e-07 |

脚本另计算n乘首步样本方差的经验代理，以及旧source的H² q_drift/2和H q_euler预算；保留它们以后c=1项仍使区间包含零。这些代理不构成未来条件方差过程或uniform drift/remainder界。源noise quantile只原样记录，未当作variance使用。没有把观察到的最大增量当作uniform support，也未用未来均值反向替换预测中心。

## 尚缺的严格条件与退出判断

已有source记录只有切向预测与膨胀分位数，不包含独立exact-step均值/variance UCB、经证明的uniform mean-drift和未来variance包络，也没有足够紧的rank/parameter tube支持。全局risk范围虽能给增量有界，却不能代替这些局部条件；Gaussian输入也不能自动提供小的gradient/tail尺度。

本次结论限定为：当前固定source、原算法、c=1的这一保守充分条件没有实用分辨率，因此本轮停止其数值实例化。只有新的解析局部范围/尾尺度与漂移控制，才值得重新考虑；不追加校准批次、未来流或通过调整极小lr/大batch来继续“救”窗口。

输出为`experiments/results/theory/stochastic_window_feasibility.json`，含固定选择、原始设置、全部1到8步oracle比较、经验数值与缺失条件。它是可行性筛查，不是严格随机验证。

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/theory/stochastic_window_feasibility.py
```
