# 图像与纹理实验入口

[STATUS](../../../STATUS.md) · [实验入口](../../README.md) · [代码导航](../README.md)

从仓库根目录运行。以下命令只查看计划；本机 Python 使用 `H:/anaconda3/envs/py312/python.exe`，正式实验按服务器环境指定解释器。

```powershell
python -B experiments/code/empirical/natural_moment_experiment.py plan --dataset cifar10
python -B experiments/code/empirical/natural_rebound_confirmation.py plan
python -B experiments/code/empirical/texture_rebound_sweep.py --stage plan
```

自然图像的共同入口是 [natural_moment_experiment.py](natural_moment_experiment.py)：训练、续跑、固定矩任务和类别读取评估分阶段执行。[natural_rebound_confirmation.py](natural_rebound_confirmation.py)固定确认任务与时间窗口；[natural_metric_intervention.py](natural_metric_intervention.py)从同一 checkpoint 作配对更新度量干预；[natural_head_counterfactual.py](natural_head_counterfactual.py)在冻结 encoder 上比较历史 head 与监督低秩参照。读取和汇总函数分别在 [natural_moment_eval.py](natural_moment_eval.py)、[summarize_natural_moments.py](summarize_natural_moments.py)。

纹理回升使用 [texture_rebound_sweep.py](texture_rebound_sweep.py)组织配对运行与两种回升统计，底层训练在 [encoder_rebound.py](encoder_rebound.py)。短期分叉和局部几何预测使用 [texture_motion_prediction.py](texture_motion_prediction.py)，与完整轨迹回升分别报告。

服务器批量入口仍为 [run_supplemental_server.sh](../../run_supplemental_server.sh) 和 [run_mechanism_followup_server.sh](../../run_mechanism_followup_server.sh)。参数与输出说明见[自然图像协议](../../docs/natural.md)、[纹理协议](../../docs/textures.md)、[机制补充实验](../../docs/followup.md)及[运动预测](../../docs/motion.md)。

较早的 full-SGD 分叉、漂移诊断和 DTD pilot 保留在本目录。其中 [full_empirical.py](full_empirical.py)仍提供纹理训练函数，[dtd_pilot.py](dtd_pilot.py)仍提供 DTD 增强；其他诊断用于解释既有记录。它们不是当前自然图像与回升实验的统一入口，历史说明在 [docs/archive/](../../docs/archive/)。
