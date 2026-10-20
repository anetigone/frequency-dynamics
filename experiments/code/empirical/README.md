# empirical 代码

[实验导航](../../README.md) · [结果索引](../../results/README.md)

| 脚本 | 用途 |
|---|---|
| [dtd_pilot.py](dtd_pilot.py) | Voxel51 DTD 镜像下载、学习有效性 pilot、47 类评估及可选 checkpoint 分叉。 |
| [batch_drift_diagnostic.py](batch_drift_diagnostic.py) | Conditional mean projector step vs mean squared step; no zero-drift assumption. |
| [compare_empirical_snapshots.py](compare_empirical_snapshots.py) | Compare complete empirical states to EXISTING population checkpoints only. |
| [conditional_moment_diagnostic.py](conditional_moment_diagnostic.py) | Exploratory finite-dataset moment reference for the conditional drift diagnostic. |
| [full_empirical.py](full_empirical.py) | End-to-end finite-dataset empirical SGD, independently calibrated factors. |
| [intervene_full_sgd.py](intervene_full_sgd.py) | Repeated empirical SGD forks from complete empirical-training checkpoints. |
| [plot_batch_drift.py](plot_batch_drift.py) | Finite-step conditional projector means and variance vs batch size. |
| [sgd_state_tools.py](sgd_state_tools.py) | Restore actual complete-empirical-training checkpoints; never population states. |
| [summarize_full_interventions.py](summarize_full_interventions.py) | Sign probabilities and amplitude predictions at genuinely empirical states. |
| [summarize_full_sgd.py](summarize_full_sgd.py) | Complete empirical trajectories: paired tasks, common energy and subspaces. |
