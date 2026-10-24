# DTD DC-free 完整诊断：约束有效，projection 剩余空间出现小幅学习

2026-10-20。分析回传的 `dtd_dc_free/diagnosis/diagnostic.json` 和 `dtd_dc_free/dominant_direction/diagnostic.json`。三开发种子 0/1/12，r=1，2,000 更新。本轮未训练、未改写结果。

## 结论

零均值约束保持正常；DC 并未重新出现。去 DC 后，已有的图像方差/对比度统计成为主要方向，projection 继续增强对这一方向的方差集中。但 projection 剩余空间也出现跨三个种子一致的 MSE、accuracy 和归一化两视图一致性改善，应保留这个正向开发信号。

encoder 的类别风险和准确率未有稳定学习收益，不能把上述 signal 解释成已经学到了显著更好的 encoder。现阶段更接近 head 对已有特征的重选，但尚未通过交叉组合初末两层独立识别。

## 1. 数据与实现核验

- 603 个已保存 audit 时点中，kernel 空间均值最大绝对值不超过 1.48e-9。
- 初末 checkpoint 张量独立计算的空间均值残差为约 3.0e-10–1.4e-9，与 audit 记录相符。
- 三个种子在每十步的 200 个 audit 区间中，loss 都没有上升。该结论限于保存网格，不声称每个 SGD 步严格下降。
- PCA 全表示重算的 probe 与原 evaluation 最大差为 6.94e-18；loss 分解残差不超过 3.47e-18；PC1 与 remainder 的 alignment 可加。

当前未见约束漂移、评估错配或 loss 算术错误迹象。

## 2. DC 被抑制，主方向转为既有的图像方差统计

训练集 clean 特征，三种子均值：

| 指标 | 初始 | 2000 步 |
|---|---:|---:|
| encoder 总方差 | 0.52681 | 0.56541 |
| encoder PC1 方差占比 | 79.56% | 79.28% |
| encoder 谱熵有效秩 | 2.7676 | 2.7694 |
| projection 总方差 | 0.009259 | 0.022383 |
| projection PC1 方差占比 | 75.15% | 86.71% |
| projection 谱熵有效秩 | 2.5165 | 1.7697 |

encoder 总方差约增加 7.33%，projection 约增加 141.75%；encoder 谱占比几乎稳定，projection 更集中。数值秩始终为 encoder 64 / projection 8，不是数值秩塌缩。

PC1 的图像统计预测，训练拟合、验证 R²，末期均值：

| 统计 | encoder | projection |
|---|---:|---:|
| 相对训练均值的 DC 偏移平方 | 0.02943 | 0.02917 |
| 单图像像素方差 | 0.85773 | 0.84462 |
| 单图像像素标准差 | 0.72968 | 0.71681 |
| 梯度能量 | 0.58517 | 0.55808 |

像素方差在初始化已能分别解释约 85.50% / 81.43% 的 PC1 验证变异，并非训练新发现的统计。encoder 初末 PC1 分数相关系数约 0.9996–0.9998，projection 约 0.9925–0.9951。

图像方差对于纹理类别可能有真实意义，不能直接叫作无用捷径。现有结果更适合描述为：目标继续优先强化已经存在、易于在增强下保持的二阶图像统计。不要接着默认去除逐图对比度，否则会变成不断删统计的探索，难以回答原有机制问题。

## 3. 小幅学习信号主要出现在 projection 剩余空间

验证集 OLS，三种子均值：

| 表示 / 分量 | accuracy 初→末 % | MSE 初→末 |
|---|---:|---:|
| encoder all | 8.7766→8.5638 | 0.021823903→0.021821411 |
| encoder PC1 | 3.5106→3.5106 | 0.020737301→0.020737471 |
| encoder remainder | 6.8794→6.7730 | 0.021918771→0.021916086 |
| projection all | 6.2234→6.2057 | 0.020721489→0.020714844 |
| projection PC1 | 3.4752→3.4397 | 0.020737498→0.020737800 |
| projection remainder | 4.6454→4.8936 | 0.020808256→0.020799492 |

projection remainder accuracy 的逐种子增量为 +0.2660/+0.3191/+0.1596 pp；验证 MSE 变化为 -1.274e-5/-8.669e-6/-4.884e-6，全部改善。原始 full projection MSE 在三个种子也改善，但 full accuracy 没有稳定收益。

这是比只看 full accuracy 更细的正向信号，但绝对效果仍小。每个 checkpoint 的 remainder 由其自身训练 PCA 定义，空间随时间变化；不能声称固定同一组方向的风险都变好。validation risk 也不对分量简单可加。末期 full projection accuracy 仍高于 remainder，因此不能说 PC1 对分类必然有害。

## 4. projection 剩余空间的一致性也改善

验证集 remainder 的两视图指标，越低越好：

| 表示 | 指标 | 初始 | 末期 |
|---|---|---:|---:|
| encoder | alignment / 训练 clean 方差 | 0.08962 | 0.08820 |
| encoder | 逐 PC clean 方差白化后平均 alignment | 0.53282 | 0.53234 |
| projection | alignment / 训练 clean 方差 | 0.06677 | 0.06044 |
| projection | 逐 PC clean 方差白化后平均 alignment | 0.11293 | 0.10988 |

projection 的两种 remainder 指标在三个种子中均改善；encoder 的均值改善很小且逐种子不一致。这与 projection remainder 的类别 MSE/accuracy 信号方向一致，比单纯输出尺度放大更有信息量。

当前只用一组共享增强流，且类别任务和一致性改善之间未建立因果关系。不能把这几个指标视为独立重复或机制确认。

## 5. loss 降低仍主要来自 uniformity

固定 audit 均值：alignment 从 0.0004251 到 0.0007482，uniformity logdet 从 -0.0040668 到 -0.0103774，uniformity trace 从 -0.0040563 到 -0.0102906，总 loss 从 -0.0076979 到 -0.0199198。绝对 alignment 增加与输出能量增长同时发生，不等于归一化一致性变差。

训练没有数值异常，目标仍在平稳改善，但目标下降不能单独代表 encoder 类别学习。

## 6. 最有区分力的下一步：初末 encoder/head 交叉组合

继续重训前，可以用现有 checkpoint 评价四个组合：

| 组合 | encoder | head |
|---|---|---|
| 初始 | h0 | W0 |
| 只换 encoder | hT | W0 |
| 只换 head | h0 | WT |
| 实际末期 | hT | WT |

所有组合在原训练标签上重新拟合同一个 OLS，原验证集评价；无网络训练、无新数据划分。如果 h0/WT 已重现末期的小幅收益，将直接支持 head 在已有 encoder 内重新选择的解释；如果必须 hT/WT 才有效，则需要检查层间交互或 encoder 的细小但任务相关变化。四组合不构成训练过程因果分解，非线性交互应保留。

这比立即扩配比或再删除图像对比度更针对稿件的 encoder/projection 区分。若随后延长训练，应固定 r=1、保留三开发种子并报告新预算，检验当前小信号是否扩大，而不是追求任意单个较好终点。

来源：[保存记录诊断](../../../experiments/results/empirical/dtd_dc_free/diagnosis/diagnostic.json)、[主方向诊断](../../../experiments/results/empirical/dtd_dc_free/dominant_direction/diagnostic.json)。
