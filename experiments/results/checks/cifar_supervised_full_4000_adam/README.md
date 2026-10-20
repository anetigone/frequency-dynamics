# 监督 Adam / 裁剪：完整 4000 步验证

2026-10-20。本机 RTX 4060 Laptop、batch 256；三个开发种子、两种监督路径，共六条完整轨迹。未读取验证 / 测试指标。

| seed | 路径 | 步数 | 最终 CE | 全程最大 CE | 裁剪比例 |
|---|---|---:|---:|---:|---:|
| 0 | supervised_encoder | 4000 | 1.7350 | 2.2978 | 99.38% |
| 0 | supervised_projection | 4000 | 1.7783 | 2.3079 | 98.60% |
| 1 | supervised_encoder | 4000 | 1.8415 | 2.3034 | 98.67% |
| 1 | supervised_projection | 4000 | 1.8750 | 2.2983 | 98.62% |
| 12 | supervised_encoder | 4000 | 1.7486 | 2.2965 | 99.15% |
| 12 | supervised_projection | 4000 | 1.7803 | 2.3062 | 98.58% |

所有更新及最终参数均保持有限。裁剪频繁触发，是监督对照优化规则的一部分，不能解释成原 SGD 的等价轨迹。无监督 SGD 没有加入裁剪。

完整配置见 [protocol.json](protocol.json)，逐条件数据和 checkpoint 保存在各 seed 目录，汇总见 [stability_summary.json](stability_summary.json)。
