# 当前状态

更新：2026-10-24。这里是研究进展和下一步安排的唯一入口；README 只导航，实验协议和详细结果链接到原处。

## 现在到哪一步

本轮补充实验已完成，用户已将结果整合进论文：head 反事实是自然图像正文主表，摘要、引言和讨论已同步更新。当前稿件入口是 [paper/main.tex](paper/main.tex)，正文为 [submission.tex](paper/sections/submission.tex)，附录为 [appendix_contents.tex](paper/sections/appendix_contents.tex)。[编译说明](paper/README.md)与[阅读 PDF](paper/output/pdf/main.pdf)。

目前可支持的研究结论：

- 理论给出平方参数化下的子空间运动及完整 encoder 风险回升构造，保留局部稳定性边界，不声称随机初始化典型性。参见[定理核验](paper/notes/theorem.md)与[文献比较](paper/notes/literature.md)。
- 纹理主报告为 50 个配对种子，square 与 linear 都有回升，不能声称该现象由 square 独占。早期六种子是另一批历史验证，不合并分母。
- CIFAR 三个长程种子是探索；新五种子的固定主任务/窗口确认结果为 1/5，成功实例后期又恢复，支持有限时间现象。不能把旧三个种子合入确认分母，也不能把次要任务搜索替换主结果。
- 固定晚期 encoder 后，原晚期 head、较早 head、训练标签最优八维 head、完整 encoder 的测试准确率分别为 26.89%、29.06%、36.51%、37.26%。97.6% 是监督 head 消除的原 head 相对完整 encoder 的测试 MSE 差距比例，不是信息恢复比例；监督 head 仅作诊断参照。
- 度量干预支持轨迹敏感性，但没有消除回升；不作唯一机制归因或严格等总体目标值结论。

详细复核和汇总图见[本轮结果解读](experiments/results/empirical/mechanism_followup/FOLLOWUP_RESULTS_REVIEW_20261024.md)。当前没有新启动的正式训练任务。

## 当前证据在哪里

保留现有结果目录；两个较长的下载批次名只是容器，不代表仍在进行的工作。下表说明当前使用哪套证据，不要求迁移、重命名或重新下载。

| 证据 | 原位置 / 阅读入口 | 状态与口径 |
| --- | --- | --- |
| 纹理主报告 | [结果目录](experiments/results/empirical/supplemental_results_now/texture_rebound_50_seeds/) | seeds 100–149，三模型共 150 条完整轨迹 |
| CIFAR 长程探索 | [结果目录](experiments/results/empirical/supplemental_results_now/natural_moments_cifar10/) · [解读](experiments/results/empirical/supplemental_results_now/CIFAR_LONG_RESULTS_REVIEW_20261023.md) | seeds 30–32，约 5000 曝光 epoch；较早评估点仍在 |
| 新种子固定窗口确认 | [结果目录](experiments/results/empirical/mechanism_followup/natural_confirmation_cifar10/) · [复核](experiments/results/empirical/mechanism_followup/CONFIRMATION_ANALYSIS_20261024.md) | seeds 40–44，2000 epoch，五个固定评估点，已完成 |
| 配对度量干预 | [结果目录](experiments/results/empirical/mechanism_followup/natural_metric_cifar10/) · [复核](experiments/results/empirical/mechanism_followup/MECHANISM_METRIC_ANALYSIS_20261024.md) | 原三种子的六个起点、十二条分支；不是六个独立种子 |
| 固定 encoder 的 head 反事实 | [结果目录](experiments/results/empirical/mechanism_followup/natural_head_counterfactual_cifar10/) · [复核](experiments/results/empirical/mechanism_followup/HEAD_COUNTERFACTUAL_ANALYSIS_20261024.md) | 原三种子 3000/5000 checkpoint 诊断，不新增预训练重复 |
| DTD 边界结果 | [结果目录](experiments/results/empirical/supplemental_results_now/natural_moments_dtd/) | 200 epoch 的已完成实验；5000 epoch 长程未做，不列为当前待跑任务 |

上述下载包主要是 JSON、CSV 和图表，服务器原 checkpoint 保持原位置。本机路径不替代 JSON 中记录的服务器来源；需要继续服务器旧运行时，明确传入原 source/output。

`results/empirical` 中的六种子验证、旧 full-SGD/运动预测实验，以及 `results/cifar` 的容量、冻结 head 和早期划分实验，均保留原位。它们是历史或辅助证据，不是以上确认实验的新重复；详见[结果导航](experiments/results/README.md)。

## 下一步安排

1. **本轮文档整理已完成。** 工作规范写入 [AGENTS.md](AGENTS.md)，当前协议收在 [experiments/docs](experiments/docs/README.md)，旧协议和笔记归档。结果目录、源码和服务器运行入口本轮保持不变。
2. **下一研究问题是条件可预测性。** 已有分析显示，主任务的 40/43 缺少已观测先改善，42/44 缺少晚期超过初值，只有 41 满足两段；现有五点任意配对仍为 1/5。不同种子运动方向和优化阶段均不同，不能只归因于窗口错位。需要区分“任务/轨迹匹配”与“采样时间”后，再决定具体实验。
3. **候选补充尚未启动。** 优先考虑服务器已有 checkpoint 上更早时段的评估、局部梯度方向重复估计及更细的共同审计进度比较。这些是新的诊断，不改变已完成确认实验的任务、阈值、窗口或分母。暂不默认追加同配置种子、DTD 长程、宽度或数据集大网格。

尚未安排下一轮正式实验，也没有要求本机补训。下一轮确定一个具体诊断目标后，先更新现有协议/入口，避免再生成一套日期提案和新脚本。

## 怎么运行和维护

- [实验入口与命令](experiments/README.md)：已有脚本名称仍有效，默认 plan 不训练。
- [实验协议](experiments/docs/README.md)：操作参数与复现实验方法；已完成协议不代表待执行任务。
- [命名与保留规则](AGENTS.md)：新增文件用稳定短名，现有长路径不追溯改写，不建脚本登记或迁移清单。

阶段完成或下一步改变时只更新本页；历史分析保持原处，不再复制到另一份“最新报告”。
