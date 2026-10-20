# 冻结同一个 encoder，只训练 head

2026-10-20。解决容量诊断中尚未分离的问题：现有 encoder 不变时，8 维投影能否保留更多类别信息？

## 六个来源状态

默认使用原 capacity_development 中三个开发种子 0、1、12，各取 baseline / r=1 的第 0、4000 步。
因此共六个固定 encoder，不再训练任何卷积核。原来的 33 条容量诊断结果不改写。

每个状态比较以下五项，统一使用相同的 45k/5k 训练 / 验证划分：

| 项目 | 用途 |
|---|---|
| 完整 encoder 的无正则 OLS | 未压缩的线性可读性参照 |
| 原始 8 维 projection 的 OLS | 当前 head 保留的信息 |
| 训练平方误差最优的 rank-8 线性投影 | 排除因子化 head 没优化好造成的假象 |
| 固定 encoder，经 CE 训练的 projection 的 OLS | 学到的压缩表示是否改进 |
| 训练得到的 CE 分类器 | 实际分类性能，和重新拟合的 OLS 分开 |

## 训练与数值规则

encoder kernel 设为 requires_grad=False，两个增强视图的 encoder 特征均在 no_grad 中计算。
优化器参数仅包含 projection W 与十类线性分类器；训练前后逐元素检查 kernel 完全相同，并检查缓存特征仍与最终 encoder 一致。

projection 从该 checkpoint 的原始 W 开始，没有重新随机选 head。
分类器使用固定随机初始化，输入标准化统计由同一批 256 张无标签训练图在初始 head 下计算，之后固定。
Adam、W lr=0.0003、分类器 lr=0.001、全局梯度范数裁剪 1.0、4000 更新、batch 256。
增强沿用原 CIFAR 管线；同一种子的初始 / 训练后 encoder 使用相同批次和增强流。
记录训练 loss、裁剪比例和范数，失败仍报错，不自动换优化器或学习率。

默认仅比较最终固定预算，不按验证标签选停。不会推断或评估测试集。

## 最优低秩平方拟合参照

对训练集特征作可逆列标准化并中心化，记设计矩阵为 X，训练 one-hot 目标中心化后为 Y。
若 X=U S V^T，则对 U^T Y 取 rank-8 的截断 SVD，得到训练 Frobenius 平方误差全局最优的低秩预测。
转换回原始特征上的纯线性 8 维 projection；中心化项吸收到最终分类器的 bias 中。
因此没有给 projection 增加非线性或额外输入信息。数值求解与无正则 OLS 一样使用 float64。

这只保证训练平方误差最优，不保证验证平方误差或分类准确率最优；它不是“8 维表示的准确率上限”。
全部求解统计与系数仅使用训练集。验证集用于事后评估，不参与挑选奇异向量或参数。
源 head、低秩 head 和 CE head 都共享同一个 encoder，所以后两者若改善，就直接表明已有 encoder 中存在更好的压缩选择。

## 服务器命令

新增两个脚本到对应目录，即可用直接路径执行，无需修改旧容量实验，也无需进行源码哈希校验。

```bash
# 只预览六个 checkpoint，不读取图像或开始训练
python -B experiments/code/cifar/frozen_encoder_probe.py plan

# 从原来的真实 checkpoint 提取固定特征并训练 head
python -B experiments/code/cifar/frozen_encoder_probe.py train

# 所有六个状态结束后，按配对种子汇总
python -B experiments/code/cifar/summarize_frozen_encoder.py
```

默认 source 为 `experiments/results/cifar/capacity_development/`，输出为 `experiments/results/cifar/frozen_encoder_heads/`。
若原目录不同，训练时指定 `--source`；若输出不同，训练与汇总均指定对应目录。
已存在成功 result.json 的任务直接跳过；未完成任务从同一 source checkpoint 重跑。比较训练参数和数据划分，不检查源码哈希。
只有 summary.json 无法运行此实验，必须在保存了原始 step_0.pt / step_4000.pt 的服务器上执行。

每状态保存：原始 / 学习 head 与完整 encoder 的 OLS、低秩最优 head、训练分类器指标、head_final.pt、reduced_rank_head.pt、训练诊断。
汇总输出 summary.json 和 findings.md；相对原 head 的增益按种子配对计算，不将 checkpoint 数量当作独立种子数量。

## 如何判断

- 最优低秩 head 明显好于原 head：同一 encoder 的信息选择确有改善空间，不能把当前低准确率都归因于 encoder 容量。
- CE head 接近低秩 / 完整 encoder 参照：head 的监督选择可以有效保留已有信息。
- 低秩 head 改善、CE head 不改善：优先排查 head 优化 / CE 目标与 OLS 指标的差异。
- 两种新 head 都不改善：该预算 / 标签样本下未找到更好的压缩；不等于证明不存在，也不能直接推出原总体风险定理失效。

这里的 OLS 与低秩拟合是有限标签诊断，不能代替无正则总体平方风险或扩大认证定理。

实现检查见 [frozen_encoder_checks.json](../results/checks/frozen_encoder_checks.json)。

本机真实数据完整稳定性检查：[frozen_encoder_full_stability.json](../results/checks/frozen_encoder_full_stability.json)。复用旧 CIFAR 五种子实验中的 seed 2 / r=1 初始和第 4000 步 encoder，各训练 head 4000 步，kernel 均逐元素不变、可训练参数均有限。它与新方案的默认 seed 0/1/12 六状态实验分开，不混入正式汇总。
