# 构造之外的完整 encoder 回升：2026-10-22 小规模验证

本轮补代码并实际完成了 6 个新种子 × 3 模型，共 18 条配对完整纹理轨迹，独立重复单位为种子。固定任务和固定时间的 encoder 回升已经出现，适用范围仍是这组高斯初始化、无标签尺度校准和有限数据 SGD 配置。它不说明同一任务族统一回升，也不说明随机初始化下普遍发生。

实验设置见 [协议](../../experiments/docs/ENCODER_REBOUND.md)。新运行开始前保存的参数与任务向量见 [protocol.json](../../experiments/results/empirical/encoder_rebound_validation/protocol.json)。首次验证后，按后续评审补齐事件交叉汇总、初始空间对照核验、针对性步长检查，并将结果接入主文与附录；历史训练结果保留。

## 固定判据与结果

训练沿用已有平台：32×32 周期图像，d=16、C=8、K=1，全支撑像素核，32,768 个有限训练样本，batch=256，lr=0.002，增强功率 0.2，r=1。种子为 20–25；每种子 square、linear、frozen_front 共享数据、批次、增强和初始有效映射 A。训练不读取任务标签或风险。

十任务沿用既有 `texture_tasks`，没有根据新曲线选方向。对每个任务分别要求

\[
I=R(0)-R(1000)>0.001,\qquad D=R(4000)-R(0)>0.001.
\]

两个比较步数和门槛取自旧种子 2–11 的探索性重分析，随后固定用于新种子；不是公开预注册。风险是已知合成分布上的总体最优无正则线性风险，训练则是有限数据上的经验 SGD。

| 固定任务 | square encoder | linear encoder | frozen encoder |
| --- | ---: | ---: | ---: |
| single_x_low | 2/6 | 0/6 | 0/6 |
| single_y_low | 1/6 | 0/6 | 0/6 |
| x_low | 2/6 | 0/6 | 0/6 |
| x_high | 0/6 | 0/6 | 0/6 |
| y_low | 2/6 | 0/6 | 0/6 |
| y_high | 0/6 | 0/6 | 0/6 |
| diagonal_low | 1/6 | 0/6 | 0/6 |
| diagonal_high | 0/6 | 0/6 | 0/6 |
| antidiagonal_low | 1/6 | 0/6 | 0/6 |
| antidiagonal_high | 1/6 | 0/6 | 0/6 |

18 条运行全部完成，没有失败或缺失。每个种子、模型、任务、表示的两段幅度及不回升分类全部保存在 [runs.csv](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/runs.csv)，完整汇总见 [findings.md](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/findings.md)。同一初始化的十个任务相关，不将其当作十个独立重复。

例如固定任务 x_low：种子 21 的 I/D 为 0.013688/0.103034，种子 25 为 0.003482/0.029919；其余四种子未满足这一任务的双门槛。种子 22 的 y_low 改善约为 0.001025，接近门槛，应结合原值阅读，不能把所有计数解释为同等强的效应。

描述性并集：square 有 5/6 个种子至少一个固定任务回升，linear/frozen 均为 0/6；没有任何种子满足全部十任务回升。并集发生率不替代上表的固定任务发生率。

## 数值检查与边界

每 25 步及固定比较点记录风险、损失、秩、奇异值和 projector 位移。全部记录点的 encoder/projection 秩分别为 8/1。square encoder 的最小保留奇异值不低于 0.0182809，最小/最大奇异值比不低于 0.0197198。默认 SVD 阈值和相对阈值 1e-12、1e-10、1e-8 下，所有记录点的秩与风险完全一致，回升计数不变。

种子 20 的三个模型另作同批次两半步检查，独立存于 `experiments/results/checks/encoder_rebound_refinement`，不加入主要分母。所有记录点、任务和两种表示上的最大绝对风险差为 square 0.00129264、linear 0.0000342430、frozen 0.0000543712；该种子全部回升判定不变。此检查不保证其余种子的临界幅度对步长同样稳定。

总体目标仅供评估。全部 square 终点的目标低于初始化，5/6 个种子还满足 L(0)>L(1000)>L(4000)；种子 23 从第 1000 步到第 4000 步略有上升（约 -0.301918 到 -0.301821）。不声称经验 SGD 每步总体目标严格下降。

六条 square 的第 4000 步视图能量 Q 均大于 1，种子 23 在第 1000 步也已大于 1。因此结果不在主定理能量区间内完整复现其所有保证。有限数据与有限 batch 也不满足线性总体守恒的精确各向同性条件；linear 对照的零回升不能写成理论上的严格不动。

本轮没有新增二维定理邻域/正交控制、自然图像属性桥接或深层 SSL 实验。结果支持的具体结论是：在预先固定任务、独立非对称且明确标度校准的初始化下，真实像素核经验训练可以产生 encoder 最优线性风险的完整回升，而非只有 projection 恶化或子空间位移。小样本不足以估计普遍发生概率，仍需要明确保留参数和任务范围。

## 交付

- [训练脚本](../../experiments/code/empirical/encoder_rebound.py)：可直接运行，无脚本登记或自定义哈希要求；保留已有结果，另一次运行使用新输出目录。
- [汇总脚本](../../experiments/code/empirical/summarize_encoder_rebound.py)：保留全部种子与任务，最低点仅作描述分析；改变原协议比较点或门槛的分析标为探索性。
- [绝对风险全轨迹](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/risk_trajectories.png)、[相对初始化的风险变化](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/risk_changes.png)、[损失与几何诊断](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/geometry.png)：均展示全部可用运行；同时提供 PDF。

18 条主要轨迹完成后按预设预算停止，没有按结果追加任务、种子或配比。

## 后续评审所要求的交叉汇总与针对性检查

本节分析沿用原主协议，不更换成功定义。新增 [rebound_events.csv](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/rebound_events.csv) 和 [rebound_event_loss.md](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/rebound_event_loss.md)，逐事件列出 I、D、L(0)、L(1000)、L(4000) 及两阶段目标变化。square encoder 的 10 个完整回升任务—种子事件中，9 个满足 L(0)>L(1000)>L(4000)，分布于种子 20、21、22、25；唯一例外为种子 23 的 antidiagonal_high。这些事件相关，不当作 10 个独立重复。中间记录点仍不保证损失逐步下降。

全部 I/D 原值和相对 0.001 的余量继续保存在 runs.csv。额外的描述性提示把满足回升且某一段幅度不超过 1.25×0.001 的事件标为接近门槛；只有种子 22 的 y_low encoder 事件被标记。1.25 倍提示是后续阅读辅助，不参与原协议成功判定。

从保存的初始 B 独立重算各模型行空间，并比较初始任务风险，结果见 [initial_pairing.csv](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/initial_pairing.csv)。跨全部种子和模型配对，最大 encoder projector Frobenius 差为 8.24223e-15，最大初始任务风险差为 2.44249e-15。因此这组对照确实共享初始 encoder 空间与风险；这个结论来自直接检查，不是由共享 A 推出。原始 B 元素最大差约 0.134566，奇异值尺度最大差约 0.254268，参数化和校准尺度仍不同。控制结果支持架构/参数化的作用，不认定唯一因果来源。

按后续评审，另对强效种子 21 与临界种子 22 执行同批次两半步检查，输出到 `experiments/results/checks/encoder_rebound_targeted_refinement`，三个模型均保留。它们与先前种子 20 的检查统一列入 [refinement_checks.csv](../../experiments/results/empirical/encoder_rebound_validation/encoder_rebound_summary/refinement_checks.csv)。三个种子、三模型、两种表示、全部任务的分类和回升判定均未改变，主结果仍使用原来 18 条普通 SGD 轨迹。

| 检查案例 | 普通一步 I | 两半步 I | 普通一步 D | 两半步 D |
| --- | ---: | ---: | ---: | ---: |
| 种子 21，x_low encoder | 0.013688260 | 0.013707136 | 0.103034269 | 0.102614005 |
| 种子 22，y_low encoder | 0.001024697 | 0.001024928 | 0.004422111 | 0.004413603 |

种子 21、22 的 square 全程最大风险绝对差分别为 0.00479689、0.000808691，因此检查支持所测事件的判定稳定，不支持逐点机器精度一致或任意步长下的保证。新增数值检查不增加独立重复数，也没有改变原比较时间和门槛。
