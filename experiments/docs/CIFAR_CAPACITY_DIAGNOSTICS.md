# CIFAR-10 容量诊断协议（代码就绪，正式实验未启动）

2026-10-19。用于区分当前浅层架构、压缩维数和无监督训练目标的限制。此阶段只作开发诊断，不将结果自动解释成自然图像机制确认。

## 配置

| 组别 | C | K | 训练目标 | 层间配比 |
|---|---:|---:|---|---|
| baseline | 64 | 8 | 原经验 GUPA | 0.1 / 1 / 10 |
| wide | 256 | 8 | 原经验 GUPA | 0.1 / 1 / 10 |
| less_compressed | 64 | 32 | 原经验 GUPA | 0.1 / 1 / 10 |
| supervised_encoder | 64 | 8 | encoder → 十类线性分类器，CE | 不作配比比较 |
| supervised_projection | 64 | 8 | encoder → projection → 十类线性分类器，CE | 不作配比比较 |

默认开发种子 0、1、12。每种子 9 条无监督 + 2 条有监督，共 33 条；每条默认 4000 更新、batch 256。
种子 0、1 曾用于开发，不将这三种子标为独立正式确认。此协议没有扩展到 RGB 或多层网络。

## 数据与配对

复用标准文件 MD5 检查、45k/5k 固定分层划分、训练集灰度标准化、两视图增强。
不做测试集推断或测试准确率计算；数据完整性审计仍会检查原始测试文件。
无监督更新不读取标签。OLS 只在 45k 训练标签上拟合，在同一 5k 验证集上比较。

同一种子所有组共享批次和两次增强的随机流，有监督组也使用两个视图。保存批次索引哈希和增强生成器终态哈希，汇总时核验。
同架构、同种子的各配比从完全相同的模型状态出发；baseline 的两个监督对照也共享这一模型初态。
不同 C/K 的初始函数没有匹配，不能将架构差异直接解释成同初始表示下的干预效应。

每个 C/K 分别作无标签初始能量和 Jacobian Gram 迹校准：中心化输出总方差 0.01，
前层迹用固定 256 个训练样本、16 个 Hutchinson VJP 估计，head 迹精确计算。
平方齐次缩放保持当前初始输出不变；两块等迹后，按原规则调整各配比学习率，使加权总初始更新强度保持一致。
64/8 基线初始化与旧代码的一致性已有实现核验。

## 优化与检查点

无监督组继续普通 SGD，基础 lr=0.002，无 BN、动量、weight decay、裁剪或训练中的输出重标定。
固定更新检查点：0、1000、2000、3000、4000。
共同校准能量 ell=0.5/1/2，每 10 更新检查首次向上穿越，保存实际超调和缺失截面。
能量只用于无监督组内的配比比较；不同 K 的总能量和 GUPA loss 不构成等价训练进度。

有监督组训练同一个卷积平方模型，并附加带 bias 的十类分类器。
分类器输入按初始无标签校准样本的逐列均值 / 标准差标准化，这些统计量训练期间固定。
这是可逆的线性数值缩放，不加入 encoder 的最终 OLS 评估。
有监督优化使用独立 SGD：encoder / projection lr=0.01、分类器 lr=0.05、momentum=0.9，无 weight decay。
监督 encoder 组不更新 projection；监督 projection 组同时更新前层、projection 和分类器。
该优化与无监督梯度尺度不匹配，目的是提供可达性能参照，不能作为同向量场的机制对照或理论容量上限。
如果监督组表现差，仍需排查优化和训练预算，不能直接判定容量不足。

## 评估与输出

训练和评估分开执行，评估要求预设组全部训练完成。所有固定检查点均保留随机初始化基线。

- encoder / projection：训练和验证 OLS 准确率、对十维 one-hot 的逐元素 MSE、真类相对其他最高分的平均 margin，以及十类混淆矩阵。
- OLS：含 intercept、无 ridge，训练列中心化 / 缩放仅改善数值条件；同时保存设计矩阵秩和奇异值。
- 训练特征协方差：全部特征值、总方差、participation ratio、谱熵有效秩。满数值秩不等于谱均匀。
- 有监督组：另外评估训练得到的分类器在 clean 训练 / 验证集上的 CE 和准确率，与 OLS 结果分开。
- 汇总：逐种子值、相对初始化变化、同架构配比差、固定步数架构差、共同能量缺失种子与超调；三个种子仅报告描述性均值和样本标准差。

初始状态、训练参数和代码哈希写入 protocol.json。不同配置 / 源码不能接写到同一锁定目录。
已完成运行和数值评估可以跳过复用；未完成运行从相同初态重跑，不声称恢复优化器中途状态。
数值评估与 NumPy / Matplotlib 汇总分别启动进程，沿用现有 OpenMP 隔离方式。

## 命令

从仓库根目录运行；本机可将 python 替换成 `& H:\anaconda3\envs\py312\python.exe`。

```powershell
# 默认只预览，不读取图像或开始训练
python -B experiments/run.py cifar_capacity plan

# 全部 33 条开发轨迹；本次未执行
python -B experiments/run.py cifar_capacity train
python -B experiments/run.py cifar_capacity evaluate
python -B experiments/run.py summarize_cifar_capacity

# 如先做监督对照，使用单独目录，避免与完整方案的锁混用
python -B experiments/run.py cifar_capacity train --suite supervised --output experiments/results/cifar/capacity_supervised
python -B experiments/run.py cifar_capacity evaluate --output experiments/results/cifar/capacity_supervised
python -B experiments/run.py summarize_cifar_capacity experiments/results/cifar/capacity_supervised

# 仅合成数据实现检查，不训练 CIFAR
python -B experiments/run.py check_cifar_capacity
```

默认输出目录 `experiments/results/cifar/capacity_development/`。每运行保存 `result.json`、`step_*.pt`、达到截面的 `energy_*.pt` 和 `evaluation.json`。
汇总生成 `summary.json`、`findings.md` 和无监督组的 `capacity_validation.png`。
有监督单独组没有无监督曲线，故不生成该图。

当前只完成合成数据检查：[检查记录](../results/checks/cifar_capacity_checks.json)。
覆盖 11 个训练条件、初始化和随机流配对、监督两路径、真实 OLS / 绘图汇总、缺失能量截面与不完整实验拒绝。
CPU 实现检查通过不代表已经验证真实 CIFAR 优化步长、GPU 吞吐或得到新的容量结论。

[实验导航](../README.md) · [此前 CIFAR 结果](../results/cifar/README.md)
