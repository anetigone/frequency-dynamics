# CIFAR 同 checkpoint 配比与速度分叉

2026-10-20。开发诊断，单个 CIFAR 分类任务；不以此直接宣称多任务取舍或无标签配比选择。

## 默认方案

- 来源：服务器 `capacity_development` 的 baseline、r=1、step=1000；开发种子 0/1/12。
- 每个状态复制为五个分支：r=0.1/1/10，以及 r=1 的整体学习率 ×0.5/×2。共 15 条，每条额外训练 1000 更新。
- 继续使用源实验基础学习率 0.002、batch=256、原始初始化的 Jacobian 迹校准 mobility，不在分叉点重新校准。
- 两块实际学习率为 `lr * 2 * mobility * speed / (1+r)` 与其 r 倍。r=1、speed=1 为基准分支；整体快慢同时缩放两块学习率。
- 每 250 步保存固定 checkpoint，包含分叉 step=0。每 50 步以及所有固定 checkpoint 测量固定审计 loss。
- 标签仅在独立 evaluate 阶段拟合/评估 probe；训练分支和审计 loss 不读取标签，不做测试集性能评估。沿用的数据完整性检查会检查标准测试文件。

旧训练使用无动量、无权重衰减的普通 SGD，没有待恢复的优化器状态。旧 checkpoint 不含历史随机数流，本实验**重新设置共享的后续随机流**，不声称恢复原训练轨迹。每种子五分支共享批次及增强；记录全部 batch 索引、部分实际增强视图、终态 RNG 和初始参数的摘要以核验配对。

兼容早期 capacity `result.json` 没有 `optimization` 字段的格式：核验 baseline/unsupervised/r=1 的任务及其在来源 protocol 中的登记后，按历史无监督 SGD 路径读取。协议中的 `supervised_momentum` 仅作用于旧监督对照，不影响无监督来源。若结果明确记录非 SGD 优化器，仍拒绝读取；不需补写或改动旧结果。

新保存的 checkpoint 包含模型、SGD 状态和分支 RNG。当前实现只支持从旧来源重新分叉：成功的分支跳过，未完成分支从来源重新跑；不提供中途恢复。已完成结果和评估按 protocol 校验，不以源码哈希作为运行门槛。

## 服务器命令

在仓库根目录运行。默认 plan 不读取数据、checkpoint 或 GPU。同步实验脚本即可，入口会自动发现脚本。

```bash
python -B experiments/run.py cifar_checkpoint_branches plan
python -B experiments/run.py cifar_checkpoint_branches train
python -B experiments/run.py cifar_checkpoint_branches evaluate
python -B experiments/run.py summarize_checkpoint_branches
```

默认输出为 `experiments/results/cifar/checkpoint_branches_development/`，与原实验分开。
本地容量目录只有 summary，缺少原 checkpoint 时应在服务器执行 train/evaluate。只汇总则不需要 Torch、数据或原服务器权重。

使用已有本地五种子实验中的三个种子验证流程，可显式指定：

```powershell
$branchPython = 'H:/anaconda3/envs/py312/python.exe'
& $branchPython -B experiments/run.py cifar_checkpoint_branches plan --source-kind local --source experiments/results/cifar/cifar_local_five --seeds 2 3 4 --output experiments/results/cifar/checkpoint_branches_local
& $branchPython -B experiments/run.py cifar_checkpoint_branches train --source-kind local --source experiments/results/cifar/cifar_local_five --seeds 2 3 4 --output experiments/results/cifar/checkpoint_branches_local
& $branchPython -B experiments/run.py cifar_checkpoint_branches evaluate --source-kind local --source experiments/results/cifar/cifar_local_five --seeds 2 3 4 --output experiments/results/cifar/checkpoint_branches_local
& $branchPython -B experiments/run.py summarize_checkpoint_branches experiments/results/cifar/checkpoint_branches_local
```

这些种子已经参与 CIFAR 结果分析，此处仍为开发诊断，不是新的确认样本。train/evaluate 参数应保持相同，只有 phase 不同；更换预算、来源、device、匹配设置等需使用新输出目录。

## 固定审计目标与 loss 匹配

从训练 split 用固定无标签随机索引取 512 张图，产生一次固定的两增强视图；五分支/所有种子均复用。审计 RNG 与训练 RNG 分开。保存图像索引及视图摘要，避免评估消耗训练随机流。

审计目标为原经验 GUPA：两视图分别中心化、协方差使用 n−1 分母、alignment 使用未中心化视图差。卷积 float32，审计的小矩阵运算 float64。保存 loss、alignment、两个 uniformity 项、view 协方差全谱与 clean 输出能量；这是固定审计样本目标，不是总体目标或完整训练集 loss。

默认绝对 loss 水平 −0.05/−0.1/−0.2/−0.3/−0.4，可在首次运行前用 `--loss-levels` 修改。仅从分叉初始 loss 之上向下首次穿越才记录：若初始 loss 已低于某水平，不将 step=0 当作训练后的匹配。检查频率由 `--audit-every` 决定；固定评估步也会检查。

穿越时保存权重并评估真实 checkpoint，不做曲线插值。默认 `--loss-tolerance 0.01` 要求双方相对目标的超调及实际 loss 差均不超过该值。未穿越、初始已经低于目标或超调过大，均明确保留为缺失。降低 audit 间隔可减少超调，但不能看完任务结果后无记录地更换标准。

相同 loss 只是标量进度对照，不是方向或协方差谱相同的保证。固定步数、共同 loss、整体学习率快慢分支需要一起检查。

## 输出与解读

- `protocol.json`：冻结的任务列表、参数、源 checkpoint/结果身份、数据划分、固定审计视图与运行环境。
- 每分支 `result.json`：实际学习率、全部审计时间点、固定及 loss 穿越 checkpoint、配对随机流记录。
- 每分支 `evaluation.json`：encoder/projection 的仿射无正则 OLS，训练与验证 accuracy、one-hot MSE、margin、混淆矩阵、设计矩阵秩及奇异值、训练特征协方差谱。
- `summary.json`：固定步数的绝对表现、相对分叉起点、相对同种子基准的变化；共同 loss 下配比分支与基准/慢/快分支的逐种子差异、实际匹配位置、缺失原因。
- `findings.md`：最终固定步数与共同 loss 的摘要表。MSE 差为分支减对照，负值表示改善；accuracy 差同向定义，正文表换成百分点。

统计单位为配对初始化种子，不把五分支、时间点或 loss 水平算作独立样本。三种子只报告逐种子值、均值和样本标准差。有限样本 OLS 不能称为定理的总体风险。

继续/停止判断仍需人工评阅：学习增量是否可信、是否仅有速度效应、差异是否跨种子一致。代码不会根据小样本均值自动宣布成功。若信号弱，暂停扩种子/配比扫描，转向更贴合模型的真实纹理任务。

## 实现检查

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_checkpoint_branches
```

该检查使用合成图像，在 CPU 上完成两种子×五分支的短程 train/evaluate/summarize；核验基准更新与手写原 SGD 完全一致、分叉初态与随机流配对、确定性重跑、固定审计公式、缺失/超调处理、损坏与不完整结果拒绝，以及 local/capacity 两种来源格式。检查记录为 `experiments/results/checks/checkpoint_branch_checks.json`，不当作 CIFAR 机制证据。

本机另已使用真实 CIFAR、seed=2 的 step=1000 来源，在 CUDA 上完成五分支各两步训练、全 45k/5k probe 评估和汇总；配对检查通过。记录为 `experiments/results/checks/checkpoint_branch_real_smoke.json`。这是程序冒烟验证，未运行默认 15 条完整开发实验。
