# cifar 代码

[实验导航](../../README.md) · [结果索引](../../results/README.md)

| 脚本 | 用途 |
|---|---|
| [cifar_checkpoint_branches.py](cifar_checkpoint_branches.py) | 从同一 CIFAR checkpoint 作配比与整体快慢分叉；独立训练和验证评估。 |
| [summarize_checkpoint_branches.py](summarize_checkpoint_branches.py) | 核验随机流配对，汇总固定步数及共同审计 loss 的任务变化。 |
| [frozen_encoder_probe.py](frozen_encoder_probe.py) | 固定同一个 encoder，比较原 head、最优 rank-8 平方拟合和监督训练 head。 |
| [summarize_frozen_encoder.py](summarize_frozen_encoder.py) | 按来源状态和配对种子汇总冻结 encoder 对照。 |
| [audit_cifar.py](audit_cifar.py) | Verify standard CIFAR-10 Python batches before deserializing them. |
| [cifar_local.py](cifar_local.py) | CIFAR-10 square-CNN pipeline, throughput test and bounded paired experiments. |
| [cifar_capacity.py](cifar_capacity.py) | 容量诊断：预览、无监督宽度 / 压缩比较、监督对照、训练 / 验证评估。 |
| [summarize_cifar_capacity.py](summarize_cifar_capacity.py) | 容量诊断的配对差、谱与初始化变化汇总；独立进程绘图。 |
| [evaluate_cifar_local.py](evaluate_cifar_local.py) | Fixed-step/common-energy validation and fixed-final-step test evaluation. |
| [summarize_cifar_local.py](summarize_cifar_local.py) | CIFAR analysis in a separate process without loading PyTorch/OpenMP. |
