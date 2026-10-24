# 历史 CIFAR 开发实验

当前官方 50k/10k 长程、固定窗口及 head 反事实证据见根目录 [STATUS.md](../../../STATUS.md#当前证据在哪里)，不在此分组。这里保留早期划分和容量/冻结 head 诊断，目录保持原位。

五配对种子、三配比、各 4000 更新的有限预算比较；共同能量下 projection 差异不稳定。

历史 [容量诊断代码与协议](../../docs/archive/CIFAR_CAPACITY_DIAGNOSTICS.md)已就绪，服务器的 33 条运行汇总已复制到本地，解读见下表；当前本地只含服务器 summary，未同步其原始 checkpoint。监督数值稳定性与恢复检查位于 checks 分组。

[实验总导航](../../README.md)

[冻结 encoder / head 对照](../../docs/archive/FROZEN_ENCODER_DIAGNOSTIC.md)的六状态汇总已收回，解读见下表。本地保留用户原文件名 summey.json；当前只收到汇总，未同步原始 head 权重。

| 结果目录 / 文件 | 内容入口 |
|---|---|
| [frozen_encoder_heads](frozen_encoder_heads/FROZEN_HEAD_REVIEW.md) | 三种子、六状态冻结 encoder 诊断；[原汇总](frozen_encoder_heads/summey.json)；[比较图](frozen_encoder_heads/frozen_head_review.png) |
| [capacity_development](capacity_development/DIAGNOSTIC_REVIEW.md) | 三开发种子容量诊断解读；[原汇总](capacity_development/summary.json)；[比较图](capacity_development/diagnostic_review.png) |
| [cifar_audit.json](cifar_audit.json) | 保存的证书 / 诊断 / 数值结果 |
| [cifar_local_five](cifar_local_five/) | [findings.md](cifar_local_five/findings.md)；[summary.json](cifar_local_five/summary.json)；[protocol.json](cifar_local_five/protocol.json)；15 份运行结果；[cifar_validation](cifar_local_five/cifar_validation.png) |

对应代码：[code/cifar](../../code/cifar/)。
