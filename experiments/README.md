# 实验导航

按阶段查找代码、协议和结果；开发试跑与正式结果分别存放。最新结论先读 [本机收尾报告](docs/LOCAL_CLOSEOUT.md)。

| 阶段 | 代码 | 结果与图表 | 协议 / 解释 |
|---|---|---|---|
| 理论认证与二维机制 | [code/theory](code/theory/) | [results/theory](results/theory/README.md) | [主稿附录 A](../THEORY_DRAFT.md) |
| 图像恒等式与纹理 pilot | [code/pilot](code/pilot/) | [results/pilot](results/pilot/README.md) | [图像协议](docs/IMAGE_EXPERIMENTS.md)、[历史试跑记录](docs/PILOT_FINDINGS.md) |
| 总体子空间机制与 K=2 / 局部核 | [code/population](code/population/) | [results/population](results/population/README.md) | [机制确认协议](docs/SUBSPACE_MECHANISM.md) |
| 完整经验 SGD、局部复验与干预 | [code/empirical](code/empirical/) | [results/empirical](results/empirical/README.md) | [完整经验协议](docs/FULL_EMPIRICAL_PROTOCOL.md) |
| CIFAR-10 数据 / 训练 / OLS | [code/cifar](code/cifar/) | [results/cifar](results/cifar/README.md) | [收尾报告](docs/LOCAL_CLOSEOUT.md) |
| 实现核验与结果完整性 | [code/checks](code/checks/) | [results/checks](results/checks/README.md) | [迁移与复现说明](docs/REPOSITORY_LAYOUT.md) |

共享图像模型与目标位于 [code/shared/square_images.py](code/shared/square_images.py)。数据原件保留在仓库根目录的 `cifar-10-batches-py/`。

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
- `archive/sources_before_layout_20261019/`：整理前的精确源码快照，用于核验旧 protocol 哈希。
- `layout_manifest.json`：旧路径到新路径、原始 / 当前源码哈希及原始结果哈希。

现有正式训练目录保留原始锁定协议。重训应选择新输出目录，不能把整理后的源码接写到旧锁定目录；详见 [复现说明](docs/REPOSITORY_LAYOUT.md)。原始证书范围与经验结论边界保持不变。
