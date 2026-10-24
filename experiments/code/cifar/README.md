# 早期 CIFAR 实验与共享实现

[STATUS](../../../STATUS.md) · [当前代码入口](../README.md)

本目录保留早期 CIFAR 训练、容量比较、checkpoint 分叉和冻结 encoder 诊断。早期开发实验使用 45,000/5,000 的训练/验证划分；当前自然图像矩任务使用官方 50,000/10,000 train/test，入口在 [natural_moment_experiment.py](../empirical/natural_moment_experiment.py)。两套实验的数据划分、checkpoint 元数据和结果应分别理解。

旧文件名仍承载当前代码需要的共享实现：[cifar_local.py](cifar_local.py)提供 square CNN 与 CIFAR 增强，[cifar_capacity.py](cifar_capacity.py)提供初始化和无标签尺度校准，[cifar_checkpoint_branches.py](cifar_checkpoint_branches.py)提供固定图像审计函数。当前 driver 复用这些函数，不调用旧开发划分的数据入口。

历史冻结 encoder 的监督 head 诊断在 [frozen_encoder_probe.py](frozen_encoder_probe.py)；当前官方划分上的同 checkpoint 交叉与 rank-8 反事实在 [natural_head_counterfactual.py](../empirical/natural_head_counterfactual.py)。自然图像运行方式见[当前协议](../../docs/natural.md)与[机制补充实验](../../docs/followup.md)，较早协议保留在 [docs/archive/](../../docs/archive/)。
