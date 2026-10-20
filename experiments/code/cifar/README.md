# cifar 代码

[实验导航](../../README.md) · [结果索引](../../results/README.md)

| 脚本 | 用途 |
|---|---|
| [audit_cifar.py](audit_cifar.py) | Verify standard CIFAR-10 Python batches before deserializing them. |
| [cifar_local.py](cifar_local.py) | CIFAR-10 square-CNN pipeline, throughput test and bounded paired experiments. |
| [cifar_capacity.py](cifar_capacity.py) | 容量诊断：预览、无监督宽度 / 压缩比较、监督对照、训练 / 验证评估。 |
| [summarize_cifar_capacity.py](summarize_cifar_capacity.py) | 容量诊断的配对差、谱与初始化变化汇总；独立进程绘图。 |
| [evaluate_cifar_local.py](evaluate_cifar_local.py) | Fixed-step/common-energy validation and fixed-final-step test evaluation. |
| [summarize_cifar_local.py](summarize_cifar_local.py) | CIFAR analysis in a separate process without loading PyTorch/OpenMP. |
