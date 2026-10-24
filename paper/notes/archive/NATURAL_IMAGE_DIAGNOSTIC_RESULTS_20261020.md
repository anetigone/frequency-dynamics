# 已回传诊断结果：主方向放大、弱类别学习与 probe 泛化

2026-10-20。分析本地回传的两份 `diagnosis/diagnostic.json`，未启动训练或访问服务器。DTD 包含 r=1、种子 0/1/12 的原始 probe/history，以及 step 0/2000 的 checkpoint 分析；CIFAR 包含五分支、三个种子的 probe/history，本次 checkpoint_changes / checkpoint_heads 为空。以下均为开发诊断，不能把 checkpoint 或分支当作独立种子。

## 当前判断

主要问题更接近：预训练把已有的高方差方向进一步放大，同时重组特征，但没有明显增加类别的线性可读性。DTD 的有限标签 probe 确实有泛化问题，不过它不能解释全部失败，因为训练集类别指标也几乎没有改善。

有效秩接近 1 的原因主要是第一主方向相对增长，不是其余方向集体变成零。将这些结果写成“信息完全塌缩”不准确。也不能写成“只发生全局缩放”：checkpoint 功能诊断显示非标量变化及可分辨的样本空间位移。

## 1. 排查优先级：暂未发现数值求解或记录错配迹象

- 两个 suite 的训练 MSE 均符合包含关系：full encoder OLS 不高于原 projection OLS；没有异常条目。
- DTD checkpoint 重新计算的 full encoder / original projection 四项 probe 指标，与已存 evaluation 的最大差为 1.74e-17。
- loss 与 alignment + 两项 uniformity 的最大差：DTD 2.22e-16，CIFAR 2.78e-17。
- DTD 的设计矩阵秩始终为 encoder 65、projection 9（含截距），即数值满列秩。encoder 设计矩阵条件数均值约 869→2011，projection 38→1254；条件恶化值得保留，但这些结果本身没有显示 double 求解失效。

这些检查不能排除所有实现错误，但不支持把当前现象主要归因于 probe 求解失效或 checkpoint 错配。

## 2. DTD：训练集也没有类别学习增益

r=1、三个种子均值，accuracy 用百分比：

| 指标 | step 0 | step 2000 |
|---|---:|---:|
| encoder train accuracy | 17.8191 | 17.6950 |
| encoder validation accuracy | 8.6525 | 8.5106 |
| encoder train MSE | 0.019622834 | 0.019623530 |
| encoder validation MSE | 0.021706744 | 0.021688780 |
| projection train accuracy | 7.7660 | 7.7660 |
| projection validation accuracy | 6.2411 | 6.2411 |
| projection train MSE | 0.020487464 | 0.020489770 |
| projection validation MSE | 0.020693467 | 0.020690272 |

encoder 的 train→validation MSE gap 约 0.00208，projection 约 0.000206，前者约为后者十倍。泛化误差确实存在，但训练 MSE 也没有改善，说明“只是验证 probe 把真实学习收益掩盖了”不足以解释数据。

## 3. 方差增量几乎全在第一主方向，尾部没有集体消失

以下是训练集 clean 特征协方差的特征值，先逐种子计算再取均值。尾部指除第一特征值外的和，不对应跨时间固定的同一组方向。

| 数据 / 表示 | 第一特征值，初→末 | 尾部特征值总和，初→末 |
|---|---:|---:|
| DTD encoder | 1.7160→86.2559 | 0.15463→0.18756 |
| DTD projection | 0.010219→6.826584 | 0.0008946→0.0010452 |
| CIFAR encoder（分叉后 r=1） | 1.7722→11.5743 | 0.19804→0.24096 |
| CIFAR projection（分叉后 r=1） | 0.035859→0.726572 | 0.0016669→0.0019964 |

DTD encoder 最小特征值均值为 2.153e-6→2.139e-6，projection 为 7.493e-6→8.419e-6。尾部总方差略增、数值秩保持，第一方向则大幅增长。不能从有效秩下降单独推出 tail 信息被删除；OLS 对可逆坐标缩放不变，也解释了谱集中与 accuracy 几乎不动可以同时发生。

按逐种子第一特征值占比取均值，DTD encoder 为 91.93%→99.46%，projection 为 91.89%→99.95%。CIFAR 分叉后的 encoder 为 89.25%→95.59%，projection 为 94.24%→99.14%。这里是各比例的均值，不是均值特征值之比。

## 4. loss 改善来自 uniformity 对方差增长的奖励，alignment 反而增加

DTD r=1、三个种子均值：

| 固定审计项 | 初始 | 2000 步 |
|---|---:|---:|
| alignment | 0.000341 | 0.175722 |
| uniformity logdet | -0.004830 | -0.798034 |
| uniformity trace | -0.004811 | -0.361479 |
| 总 loss | -0.009300 | -0.983790 |

绝对 alignment 没有改善；两项 uniformity 更负，并超过 alignment 的增长。输出尺度同时改变，因此不能仅据绝对 alignment 增大就宣布增强不变性变差，需在相同尺度下另作比较。

CIFAR baseline 分叉也有同样结构：alignment 0.009816→0.177565；uniformity 合计 -0.046236→-0.454106；总 loss -0.036419→-0.276541。两个数据集均兼容“目标主要奖励已有主方向增大”的解释。

DTD seed 0 在 200 个相邻审计区间中有 19 个 loss 上升，另两种子为零。这里是有限批固定审计的经验轨迹，不能套用理论 loss 全程严格下降。上升次数本身也不足以认定训练不稳定或发散。

## 5. checkpoint 功能变化：超出整体缩放，但绝大多数方差可由旧特征预测

DTD step 0→2000，在官方训练图像上拟合变换、在官方验证图像上计算相对平方残差：

| 表示 | 共享标量＋截距残差，三种子范围 | 完整仿射映射残差，三种子范围 |
|---|---:|---:|
| encoder | 25.54%–60.21% | 0.00253%–0.00392% |
| projection | 6.97%–28.34% | 0.06496%–0.17660% |

共享缩放不能充分解释变化；旧特征的仿射组合却能解释训练后 encoder 约 99.997%、projection 约 99.870% 的验证总平方偏差（按各种子残差均值）。这支持“主要在旧特征内重混合和放大”的描述。

但仿射残差按方差加权，会被放大的第一主方向主导；不能据它宣布所有小方差任务方向都没变化。未加方差权重的验证样本空间投影距离均值分别为 encoder 1.1847、projection 0.5803，最大主角度范围分别约 30.46°–40.73°、14.27°–26.68°。空间变化可分辨，且可能在低方差方向上对任务产生影响。也未检查拟合映射可逆，不能宣称表示严格等价。

## 6. DTD 监督 rank-8 head 没有复现 CIFAR 的近乎完全恢复

DTD step 2000：

| 同一个 encoder 的 readout | Train accuracy % | Validation accuracy % | Train MSE | Validation MSE |
|---|---:|---:|---:|---:|
| Full OLS | 17.6950 | 8.5106 | 0.019623530 | 0.021688780 |
| Supervised rank-8 LS | 11.1170 | 6.8794 | 0.019954547 | 0.021180488 |
| 原 projection + OLS | 7.7660 | 6.2411 | 0.020489770 | 0.020690272 |

rank-8 训练 MSE 明显优于原 head，但三个种子的验证 MSE均更差；验证 accuracy 平均仅提高 0.6383 pp，仍低于 full OLS。rank-8 验证 MSE 也高于均匀常数预测的 0.020823902。初始化已经有类似现象。

因此，监督任务方向选择也受到有限样本泛化影响；不能把训练 MSE 的全局最优直接当成最佳验证 head。47 类中心化标签最多 46 个方向，也使 K8 与 CIFAR 的九个标签方向背景不同。

## 7. 下一步修正

当前不优先扩大配比或继续从头跑更多同类轨迹。也不优先把“去掉 contrast”当作修复：削弱 alignment 的增强代价可能让主方向更容易放大，未必带来类别收益。

下一项最有信息量的分析是定位被放大的第一方向到底表示什么。使用已有 DTD baseline 的初末 checkpoint：

1. 从训练集拟合 projection 第一主成分，在验证集比较其分数与图像均值、RMS、标准差、简单梯度能量的关系，确认亮度/对比度/粗纹理统计是否能解释它。不能现在就把它命名为“亮度捷径”。
2. 分开计算这一主方向与其正交剩余部分的类别可读性，以及固定尺度下的两视图一致性。可利用同一主成分分解作现有表示的诊断，不改变原主要结果或官方划分。
3. 如果第一方向主要解释无关图像能量，才有根据设计单因素的输入或目标对照；若它确有类别信号但其余方向没学习，应优先检查目标为何只扩张这一个方向。先定位再干预。

本轮证据已经支持“方差增长与类别风险改善脱钩”的经验描述；还不支持确认具体 nuisance、宣称信息秩塌缩、或提出可泛化的最佳层间学习率。

来源：[DTD 诊断输出](../../../experiments/results/empirical/dtd_pilot/diagnosis/diagnostic.json)、[CIFAR 分叉诊断输出](../../../experiments/results/cifar/checkpoint_branches_development/diagnosis/diagnostic.json)。
