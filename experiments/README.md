# 实验导航

按阶段查找代码、协议和结果；开发试跑与正式结果分别存放。最新结论先读 [本机收尾报告](docs/LOCAL_CLOSEOUT.md)。

真实纹理下一阶段：[DTD pilot](docs/DTD_PILOT.md)，使用 Voxel51 Hugging Face 原 JPEG 镜像和官方 partition 1。先检查三个配比的学习增量，再独立决定是否运行同 checkpoint 分叉。

| 阶段 | 代码 | 结果与图表 | 协议 / 解释 |
|---|---|---|---|
| 理论认证与二维机制 | [code/theory](code/theory/) | [results/theory](results/theory/README.md) | [主稿附录 A](../paper/notes/THEORY_DRAFT.md) |
| 图像恒等式与纹理 pilot | [code/pilot](code/pilot/) | [results/pilot](results/pilot/README.md) | [图像协议](docs/IMAGE_EXPERIMENTS.md)、[历史试跑记录](docs/PILOT_FINDINGS.md) |
| 总体子空间机制与 K=2 / 局部核 | [code/population](code/population/) | [results/population](results/population/README.md) | [机制确认协议](docs/SUBSPACE_MECHANISM.md) |
| 完整经验 SGD、局部复验与干预 | [code/empirical](code/empirical/) | [results/empirical](results/empirical/README.md) | [完整经验协议](docs/FULL_EMPIRICAL_PROTOCOL.md) |
| CIFAR-10 数据 / 训练 / OLS | [code/cifar](code/cifar/) | [results/cifar](results/cifar/README.md) | [收尾报告](docs/LOCAL_CLOSEOUT.md) |
| 实现核验与结果完整性 | [code/checks](code/checks/) | [results/checks](results/checks/README.md) | [迁移与复现说明](docs/REPOSITORY_LAYOUT.md) |

共享图像模型与目标位于 [code/shared/square_images.py](code/shared/square_images.py)。数据原件保留在仓库根目录的 `cifar-10-batches-py/`。

新增 [CIFAR 容量诊断](docs/CIFAR_CAPACITY_DIAGNOSTICS.md)：原架构两种监督对照、C=256 / K=8 的宽度比较和 C=64 / K=32 的压缩比较。开发训练已开始，初版监督步长发散已复现并修正，恢复命令见协议；`python -B experiments/run.py cifar_capacity` 默认只预览。

下一步 [冻结 encoder 的 head 诊断](docs/FROZEN_ENCODER_DIAGNOSTIC.md)使用原 checkpoint，固定卷积核后只训练 projection / 分类器，并提供同特征上的最优 rank-8 训练平方误差参照。独立输出目录，不改写原容量实验。

新增 [CIFAR 同 checkpoint 分叉](docs/CIFAR_CHECKPOINT_BRANCHES.md)：从 r=1 的 step=1000 状态分出三个配比及整体学习率快慢对照；默认三个开发种子、额外 1000 步，独立汇总固定步数和共同审计 loss 下的任务变化。

## 常用入口

从仓库根目录执行；也可直接执行表中代码目录内的脚本。

```powershell
python -B experiments/run.py --list
python -B experiments/run.py check_arithmetic
python -B experiments/run.py audit_local_closeout
python -B experiments/run.py audit_layout
python -B experiments/run.py summarize_full_sgd experiments/results/empirical/full_sgd_main --repeat-dirs experiments/results/empirical/full_sgd_repeat1 experiments/results/empirical/full_sgd_repeat2
python -B experiments/run.py summarize_cifar_local experiments/results/cifar/cifar_local_five
```

本机解释器为 `H:\anaconda3\envs\py312\python.exe`。理论认证只依赖标准库；图像实验使用 PyTorch，绘图使用 NumPy / Matplotlib。数值评估与绘图汇总分别启动进程。

## 查找规则

- `code/`：当前可运行源码，按研究阶段分组。
- `docs/`：实验协议、历史发现与当前结论。
- `results/`：按同名阶段归类；[总索引](results/README.md)列出主要入口。
- `results/archive/development/`：开发种子、步长试跑和吞吐检查，不作为正式重复。
- `archive/sources_before_layout_20261019/`：整理前的历史源码快照。

实验记录参数与结果。运行入口自动查找源码，不依赖清单文件；详见 [复现说明](docs/REPOSITORY_LAYOUT.md)。原始证书范围与经验结论边界保持不变。
