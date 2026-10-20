# DTD DC-free 从头训练：2,000 步开发结果

2026-10-20。来源：`experiments/results/empirical/dtd_dc_free/summary.json`。三个开发种子 0/1/12，r=1，五个固定 checkpoint，空间零均值约束，未使用测试集。本地目前只有 summary，不能独立核验服务器实际 kernel 残差、校准信息或每步训练记录。

## 判断

目前没有分类 accuracy 收益，也没有明显的 encoder 类别学习增量。projection 则出现小幅但跨种子一致的训练/验证 MSE 改善，应保留为开发信号。DC-free 的输出能量推进显著慢于原 pilot，不能把相同步数视为相同优化进度，因此当前不能断言去 DC 已解决问题，也不能认定它在后续训练中不会起作用。

结果不支持直接扩大 r 网格或宣称自然图像机制确认。

## 1. 相对自身初始化的变化

三种子均值，accuracy 为百分比：

| 指标 | 初始化 | 2000 步 | 变化 |
|---|---:|---:|---:|
| encoder train accuracy | 17.0390 | 17.0567 | +0.0177 pp |
| encoder validation accuracy | 8.7766 | 8.5638 | -0.2128 pp |
| encoder train MSE | 0.019678433 | 0.019678768 | +3.344e-7 |
| encoder validation MSE | 0.021823903 | 0.021821411 | -2.492e-6 |
| projection train accuracy | 8.1028 | 8.0674 | -0.0355 pp |
| projection validation accuracy | 6.2234 | 6.2057 | -0.0177 pp |
| projection train MSE | 0.020518607 | 0.020513125 | -5.482e-6 |
| projection validation MSE | 0.020721489 | 0.020714844 | -6.645e-6 |

encoder validation accuracy 的逐种子变化为 -0.2128/-0.3191/-0.1064 pp；validation MSE 变化为 -2.424e-6/-1.002e-5/+4.973e-6，方向不全一致。encoder train MSE 基本不变，不能说 encoder 学到了明显更好的类别特征。

projection validation accuracy 变化为 0/-0.1596/+0.1064 pp，未有稳定方向。但 validation MSE 变化为 -6.914e-6/-7.366e-6/-5.654e-6，train MSE 也在三个种子均下降。

## 2. projection 的 MSE 小信号是真实保留项

各固定 checkpoint 的 projection 验证 MSE：

| 步数 | 验证 MSE 均值 | 相对初始化 |
|---|---:|---:|
| 0 | 0.020721489 | 0 |
| 500 | 0.020719659 | -1.829e-6 |
| 1000 | 0.020717997 | -3.492e-6 |
| 1500 | 0.020716384 | -5.105e-6 |
| 2000 | 0.020714844 | -6.645e-6 |

每个种子的 train / validation projection MSE 在这五个时点均持续下降，而 accuracy 并未相应改善。这个趋势比只挑一个终点的均值更有结构，但仍是三个开发种子、单一任务的结果。

原始 MSE 相对下降约 0.0321%。均衡常数 MSE 为 46/47²≈0.020823902，平方误差技能分数从 0.4918% 升到 0.5237%，增加 0.0319 个百分点。它相当于初始“优于常数的误差差额”增加约 6.49%，后者的分母很小，不能包装成整体语义能力提高 6.49%。

encoder 验证 MSE 仍高于常数预测，说明先前的有限标签 probe 泛化问题没有解决。

## 3. 去 DC 后推进明显变慢

固定审计的均值：

| 步数 | GUPA loss | clean 输出能量 |
|---|---:|---:|
| 0 | -0.007698 | 0.008350 |
| 500 | -0.009563 | 0.010331 |
| 1000 | -0.011979 | 0.012905 |
| 1500 | -0.015359 | 0.016525 |
| 2000 | -0.019920 | 0.021438 |

能量增长约 2.57 倍，末段仍在增加，已保存时点未呈现明显平台。不能因目标变化小就断言模型未训练，也不能据此保证延长预算会改善类别风险。

原 pilot r=1 的能量从 0.009757 升到 5.980839，约 613 倍；其 500 步能量约 0.023133，已接近 DC-free 2000 步的 0.021438。两组不同初始函数、不同约束向量场，即便总能量相近也不是严格等价进度。这里仅说明相同步数的外推比较存在明显进度差异。

原 pilot 的 projection MSE 终点改善约 3.196e-6，新模式约 6.645e-6；不能只据后一数字较大就宣布新模式优胜，因为初始表示不同，且没有共同状态干预。原模型最终 projection accuracy 约 6.2411%，新模式约 6.2057%，均不支持较大的实际任务收益。

## 4. 下一步选择

不立即扩展学习率配比、seed 或数据集。优先读取这次已经保存的 result / evaluation / checkpoint，核验实际约束保持，检查 DC-free 现在是多个方向共同增长，还是换成另一个图像统计主导。已有入口可直接运行：

```bash
python -B experiments/run.py diagnose_natural_images experiments/results/empirical/dtd_dc_free
python -B experiments/run.py diagnose_dominant_direction experiments/results/empirical/dtd_dc_free \
  --steps 0 2000 --device cuda --feature-batch 128
```

若谱和一致性显示仍在稳定推进，下一笔算力可考虑仅对 r=1 延长一次预算，检验 projection MSE 趋势能否扩大并转化为 accuracy 或 encoder 收益。它必须被定义为新的探索性预算，不能事后把新终点当作原先固定预算的确认结果。当前不能从 summary 单独决定应该加速、延长多少或更换目标。

这次对照已否定“只要训练时去掉 DC，现有预算内就自然出现明显类别增益”的乐观预期；它没有排除更慢的学习、其他统计主导或目标/增强/浅层架构的限制。

本轮复核了均值、样本标准差及相对初态的逐种子差，均与 summary 一致；没有启动新训练或改写结果。

来源：[DC-free summary](../experiments/results/empirical/dtd_dc_free/summary.json)、[原 pilot summary](../experiments/results/empirical/dtd_pilot/summary.json)。
