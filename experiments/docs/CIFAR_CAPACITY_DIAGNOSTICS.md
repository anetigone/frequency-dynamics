# CIFAR-10 容量诊断协议（开发阶段）

2026-10-19。用于区分当前浅层架构、压缩维数和无监督训练目标的限制。此阶段只作开发诊断，不将结果自动解释成自然图像机制确认。

2026-10-20 第二次修复：初版监督 SGD 在短程测试通过后仍于服务器第 3397 步发散；此前 500 步检查不足以覆盖 4000 步训练。监督正对照改用 Adam、全局梯度范数裁剪 1.0，encoder / projection lr=0.0003、分类器 lr=0.001。无监督 SGD、随机初始化、数据和增强规则保持原样。不以源码哈希作为训练或评估门槛。

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
有监督优化使用独立 Adam：encoder / projection lr=0.0003、分类器 lr=0.001，标准 beta=(0.9,0.999)，无 weight decay。
每步在更新前对全部活跃监督参数的梯度作全局范数裁剪，阈值 1.0；记录裁剪前范数、是否触发、累计次数和全程最大 loss。该裁剪仅作用于监督正对照，不作用于无监督轨迹，也不改变所评估的模型函数族。
旧 SGD 选项保留用于诊断，但不再是监督对照的默认配置；复现无裁剪的旧配置还需显式关闭裁剪并设置对应学习率。
裁剪可能频繁触发，应报告实际比例，不能声称与原 SGD 轨迹等价。
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

初始状态和训练参数写入 protocol.json。不校验源码哈希；相同数据划分、任务列表和无监督配置可以在原目录接着运行。
监督初版的源码已原样保存为 [历史源码](../archive/cifar_capacity_before_supervised_fix_20261020.py)。
已完成运行和数值评估可以跳过复用；未完成运行从相同初态重跑，不声称恢复优化器中途状态。
数值评估与 NumPy / Matplotlib 汇总分别启动进程，沿用现有 OpenMP 隔离方式。

## 命令

从仓库根目录运行；本机可将 python 替换成 `& H:\anaconda3\envs\py312\python.exe`。

```powershell
# 默认只预览，不读取图像或开始训练
python -B experiments/run.py cifar_capacity plan

# 全部 33 条开发轨迹
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

实现回归检查：[检查记录](../results/checks/cifar_capacity_checks.json)。
覆盖 11 个训练条件、初始化和随机流配对、监督两路径、真实 OLS / 绘图汇总、缺失能量截面、不完整实验拒绝和无源码校验的旧目录续跑。
历史短程检查：[记录](../results/checks/cifar_supervised_fix_realdata.json)。该 SGD 配置后来在服务器第 3397 步发散，不作为完整稳定性的证据。
当前配置已在本机 GPU 上完成三个开发种子各两组监督轨迹，六条均跑满 4000 步，最终参数全部有限。验证期间未读取验证 / 测试指标，见 [完整记录](../results/checks/cifar_supervised_full_4000_adam/README.md)。裁剪比例也已报告。

## 在旧目录直接续跑

最后一条 `less_compressed_unsupervised_r10: 4000/4000` 之后接着执行同种子的 supervised_encoder；初版监督步长在真实数据上发散。
替换服务器上的 `experiments/code/cifar/cifar_capacity.py` 后，使用下面的监督优化参数在原目录继续：

```bash
python -B experiments/run.py cifar_capacity train \
  --supervised-optimizer adam --supervised-grad-clip 1 \
  --supervised-lr 0.0003 --classifier-lr 0.001
```

如果原来用了自定义输出目录或其他训练参数，保留这些参数，但将原命令显式指定的监督优化器 / 步长替换为上面的新配置。默认输出目录不变。

- 有 `result.json` 的运行直接跳过，不重训或复制已有结果。
- 没有 `result.json` 的失败 / 未完成运行，从相同随机初始化重新开始，使用新的监督 Adam / 梯度裁剪配置。
- 不校验源码哈希，评估也不再检查源码哈希。
- 仅比较任务列表、数据划分、步数、batch、无监督学习率、检查点和校准设置，避免误混不同实验配置。
- 若已有成功的监督运行，不允许无记录地把它与不同监督学习率混作一组；当前 seed 0 监督首组失败的情况没有这一限制。
- 首次接写会保留 `protocol_before_resume.json`；`resume_events.json` 记录保留的已完成任务与参数变化，`protocol.json` 记录当前设置。

训练完成后仍使用原目录：

```bash
python -B experiments/run.py cifar_capacity evaluate
python -B experiments/run.py summarize_cifar_capacity
```

发生数值错误时保存 `failure.json`（任务、步数、学习率、参数 / 梯度统计、近期 loss）和 `failure_state.pt`。
失败状态只供诊断，不是可续训的优化器快照；失败任务不写成功 result.json，非有限数值仍直接报错，不自动回滚或尝试其他学习率。监督裁剪是预先设定且记录的优化规则。
