# 经验实验结果

当前论文证据及两批下载结果见根目录 [STATUS.md](../../../STATUS.md#当前证据在哪里)。目录内容保持原位。

下表主要保留较早的 full-SGD 辅助实验入口，不是新的自然图像确认批次。从初始化开始的经验训练：主实验 90 条、局部复验 90 条、额外独立流 40 条；分叉状态来自完整经验训练。

[实验总导航](../../README.md)

| 结果目录 / 文件 | 内容入口 |
|---|---|
| [batch_drift_diagnostic](batch_drift_diagnostic/) | [findings.md](batch_drift_diagnostic/findings.md)；[results.json](batch_drift_diagnostic/results.json)；[batch_mean_variance](batch_drift_diagnostic/batch_mean_variance.png) |
| [full_sgd_interventions](full_sgd_interventions/) | [findings.md](full_sgd_interventions/findings.md)；[summary.json](full_sgd_interventions/summary.json)；[protocol.json](full_sgd_interventions/protocol.json)；90 份运行结果；[empirical_state_predictions](full_sgd_interventions/empirical_state_predictions.png) |
| [full_sgd_local](full_sgd_local/) | [findings.md](full_sgd_local/findings.md)；[summary.json](full_sgd_local/summary.json)；[protocol.json](full_sgd_local/protocol.json)；90 份运行结果；[all_tasks_common_energy](full_sgd_local/all_tasks_common_energy.png)；[complete_sgd](full_sgd_local/complete_sgd.png) |
| [full_sgd_main](full_sgd_main/) | [findings.md](full_sgd_main/findings.md)；[summary.json](full_sgd_main/summary.json)；[protocol.json](full_sgd_main/protocol.json)；90 份运行结果；[all_tasks_common_energy](full_sgd_main/all_tasks_common_energy.png)；[complete_sgd](full_sgd_main/complete_sgd.png) |
| [full_sgd_repeat1](full_sgd_repeat1/) | [protocol.json](full_sgd_repeat1/protocol.json)；20 份运行结果 |
| [full_sgd_repeat2](full_sgd_repeat2/) | [protocol.json](full_sgd_repeat2/protocol.json)；20 份运行结果 |

对应代码：[code/empirical](../../code/empirical/)。
