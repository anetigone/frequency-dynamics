# 相关工作审计与候选贡献边界

更新：2026-10-19。主稿：[THEORY_DRAFT.md](THEORY_DRAFT.md)。本记录合并用户提供的相关工作审计与本轮定点原文复核。结论是**候选贡献仍可推进，尚未通过新颖性验收**，不是排除所有近邻的完整检索报告。

## 阅读范围与证据等级

本轮独立复核重点为 Kunin、Torralba–Weiss、Xue、Gui–Ma–Zhong 和 Chaudhry 的指定段落；另检查 Tian 的两层假设、Jing 的线性动态及 Chen v2 的风险定义。没有逐行验证这些论文的全部证明，没有完成前向引用图。引用次数、全部最新版本与代码链接本轮不统计。

| 文献与版本 | 本轮独立读取的原文位置 | 可确认的重叠与限制 |
|---|---|---|
| [Kunin et al., Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf) | §3、Fig. 2、式 (1)–(2)，引言的扩展范围 | 相同初始乘积、不同层间分配及学习率影响轨迹已有直接先例。其此处使用监督平方损失；不能把本文一般尺度效应写成首次发现。 |
| [Torralba & Weiss, A Theory of Contrastive Learning with Natural Images, arXiv:2607.07470v1](https://arxiv.org/html/2607.07470v1) | App. M 式 (46)–(49)、Thm. M.1、Fig. 24–25；GUPA 定义沿用既有核验 | 已有相同有效表示的不同分解、初始化影响及 loss 下降而识别先升后降。M.1 的冻结正交 head 最优解性质不能替代实际联合 GF 风险轨迹。 |
| [Xue et al., Investigating the Benefits of Projection Head for Representation Learning, arXiv:2403.11391v1, 2024](https://arxiv.org/html/2403.11391v1) | App. A.2 的 balanced 条件；App. D.2、Fig. 8 | good augmentation 下的 post-projection 早停收益已有实验先例；另一设置没有同样收益。不能仅归为收敛极限理论。 |
| [Gui–Ma–Zhong, Unraveling Projection Heads in Contrastive Learning: Insights from Expansion and Shrinkage, arXiv:2306.03335v1, 2023](https://arxiv.org/html/2306.03335v1) | §1.4、§2 实验设置与贡献说明 | 固定 encoder 后研究信号扩张/收缩和下游分类；明确不分析 encoder 训练动态。语义收益与损害不是新的研究对象。 |
| [Faris Chaudhry, The Geometry of Projection Heads: Conditioning, Invariance, and Collapse, arXiv:2605.17180v1, 2026](https://arxiv.org/html/2605.17180v1) | 式 (1)、Thm. 3.1–3.2、§4.1、App. A Assumption 7 | 包含 pullback Hessian 几何、给定轨迹上的 head 存在性构造及快 head 假设。它们不直接保证本文有限层间配比的实际 GF 轨迹。 |
| [Tian, Understanding the Role of Nonlinearity in Training Dynamics of Contrastive Learning, arXiv:2206.01342v3, ICLR 2023](https://arxiv.org/html/2206.01342v3) | §4、Assumption 2 的 fast top-level 与 conditional independence、式 (12)–(15) | 已有非线性动态和模式分化；本轮核到的两层分析带特定假设。未完成全部附录逐定理排除。 |
| [Jing et al., Understanding Dimensional Collapse in Contrastive Self-supervised Learning, arXiv:2110.09348v3, ICLR 2022](https://arxiv.org/pdf/2110.09348v3) | §4–5，§5.2 Lemma 3、Thm. 2–3 的声明及奇异值演化说明 | 两层线性 contrastive 动态、对齐与维数坍缩已有分析。本稿风险变动时标量方差仍正，不将其称作新的 collapse。OpenReview PDF 遇访问验证，改读 arXiv PDF。 |
| [Chen–Zhou–Zhou, Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling, arXiv:2606.26617v2, 2026](https://arxiv.org/html/2606.26617v2) | Problem Setup、Full-Batch Empirical GD、风险定义及 bias/variance 实验说明 | 固定 sketch 后直接训练交互矩阵，其 excess risk 针对评分目标，不是独立语义 probe。有限样本 bias/variance 的 U 形不能与本文总体方向风险混同。 |
| [Tepe, Dynamics of Gradient Flow with Contrastive Learning, MIT thesis, 2024](https://dspace.mit.edu/entities/publication/05dcf88a-4f21-4e3e-8c0d-e9d8f110ce61) | 用户审计及既有记录只覆盖仓储元数据/摘要；本轮未重取全文 | **全文阻断未解除**。不能依据摘要或题名认定已经排除。 |
| [Liao et al., Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution, arXiv:2602.10357v1, 2026](https://arxiv.org/abs/2602.10357v1) | 本轮独立核对 arXiv 作者、题名与摘要；正式会议与全文结论不据此确认 | 摘要涉及不平衡数据下的 Transformer 动态。用户审计报告正式 PDF 遇访问验证；完整结果列为新增待核项。 |

## 可写与不可写的主张

| 主张 | 当前定位 |
|---|---|
| 相同初始函数，不同层间尺度产生不同轨迹 | 已有直接先例，作为背景与对照。 |
| projection 改变几何、特征权重和语义表现 | 已有理论先例，不能独立承载创新。 |
| 预训练损失下降时下游性能回落，或 post-projection 有早停收益 | 已有现象先例。 |
| 实际平方因子化参数训练中的显式语义方向阈值 | 本轮指定阅读范围未见直接等价结论，仍属候选增量。 |
| 在显式连续初值盒内，真实联合 GF 使无正则总体投影风险先改善、后超过初值，并有固定回升幅度 | 定理 T 的核心候选贡献；不据目前阅读范围宣布首次或完成排他性证明。 |

研究问题应表述为：即使排除 probe 正则化与有限标签估计误差，因子化训练诱导的压缩方向变化，仍可能使独立语义任务的最优线性预测风险在预训练损失下降时发生固定幅度回升。匹配初始表示的保证由定理 2 与推论 2.3 承担；定理 T 的完整八维盒本身允许初始有效表示变化。

## 几何对象与目标范围

正文 §4.1 区分参数 Jacobian Gram 更新矩阵与 head 拉回损失 Hessian；两者不因为都具有几何解释就可直接互换。该区分是由各自定义作出的数学比较，不是对 Chaudhry 全部定理的否定。

对当前径向特例，正增长区间上的目标只给参数向量场乘一个正标量，因而同一初值的能量参数化轨迹不依赖具体径向损失。这个结论由本稿方程直接推得，不能据主定理宣称 contrastive 独有机制。不同层间初始化或配比仍可改变几何路径；两种时间等价性不能混淆。一般各向异性增强不自动共享径向路径。

## 尚未解除的验收项

1. 获取并核查 Tepe 全文的可积动态、谱演化及可能的下游风险结果。
2. 核查 Liao 等正式全文的模型、时间尺度、风险定义与定理范围。
3. 对 Kunin 及其引用的非平衡初始化、深矩阵/对角因子化动态文献补齐定理级比较；不能只比较题名与架构。
4. 补齐相关论文的前向引用图，确认是否已有可通过变量替换得到当前方向阈值或固定风险间隔的结果。
5. 验证 projection 评测与真实视觉任务的联系；当前计算机辅助证明不解决这一意义问题。

本轮没有下载批量 PDF、统计引用数或取得 Tepe 全文；没有声称逐行审完全部附录，也没有将当前平方功率模型解释为低频/高频空间邻接机制。以上记录是定点审计，完整新颖性结论仍待上述工作。
