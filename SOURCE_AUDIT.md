# 原文核验与新颖性边界

> 2026-10-19 更新：当前主稿为 [v0.3](THEORY_DRAFT.md)，主张明确限定于 projection 输出，并补上 encoder 风险不变性、尺度—学习率等价与初始参数盒证书。本轮没有重新核验外部文献；下面是 2026-10-18 的来源记录，不能据此确认新增结果的新颖性。

核验日期：2026-10-18。继承的检索材料位于 `../literature_scout/round2-2026-10-18/`；本次仅核对推导直接依赖的定义与若干关键近邻，没有重新做全量 citation census，也没有更新引用数。

## 本次直接阅读的原始来源

| 来源 | 本次阅读范围 | 用法与边界 |
|---|---|---|
| [Torralba & Weiss, A Theory of Contrastive Learning with Natural Images, v1](https://arxiv.org/html/2607.07470v1) | §1 式 (2)–(3)、§3.1 式 (8)、App M.1 定理声明及说明 | 核对完整 GUPA 的 logdet 与 trace 两项，核对有效 Fourier 能量表示；M.1 是冻结正交投影下的最优性质，不能作为两层同训动态保证。 |
| [Xue et al., Investigating the Benefits of Projection Head for Representation Learning, v1](https://arxiv.org/html/2403.11391v1) | 页面与投影头理论相关段落检索；未重审全部证明 | 继承已有逐层权重分析的近邻定位；本文不把尺度守恒单独声明为新贡献。 |
| [Chen, Zhou & Zhou, Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling, v2](https://arxiv.org/html/2606.26617v2) | Problem Setup、bilinear score、经验 GD 与 scaling 讨论段 | 核对其直接训练 bilinear interaction matrix 的设置；尚未逐行审完动态附录，不能写“已经完整排除”。 |
| [Tian, Understanding the Role of Nonlinearity in Training Dynamics of Contrastive Learning, v3](https://arxiv.org/html/2206.01342v3) | 摘要、§1–2 与章节目录 | 已有非线性动态和 global modulation；不能声称“首次分析非线性 CL 的训练动态”。two-layer ReLU 的 receptive-field 设置与本文不同，仍需逐定理比较。 |
| [Tepe, Dynamics of Gradient Flow with Contrastive Learning, MIT thesis, 2024](https://dspace.mit.edu/entities/publication/05dcf88a-4f21-4e3e-8c0d-e9d8f110ce61) | MIT 正式仓储元数据和摘要 | 确认确有 spectral contrastive loss GF 相关学位论文。本次未读取完整 thesis，因此继续列作阻断，不认定已排除。 |

上述链接是本次读到的固定版本或正式仓储页面；本次未重新查询全部论文的最新版 API，所以不把“本次读到版本”写成“截至今日一定最新”。网页核验只用于归属已有结果，本文新推导的真假依赖正文证明与数学复核。

## 与原笔记相比的明确建模选择

1. 两个同分布增强视图替代 clean-anchor/augmentation；正文给出后者的均值不匹配 rank-one 项，不混用两种统计。
2. 输出能量上限替代起点就单位能量归一化，确保冷启动尺度问题有定义。所有证明限定在约束未激活时。
3. 首先研究全支撑周期滤波器；短支撑局部 CNN 需要额外处理 Fourier 参数约束。
4. 首先给总体 GF 的可证明机制和实际低维 GD 核验；不把它们改称一般有限 n、随机多通道主定理。
5. 先用总体 ridge 建立严格风险间隔，再给有限标签的存在性桥梁；尚未得到“少标签”的可用样本界。

## 继续保留的新颖性阻断

- Tepe thesis 全文与 Tian 附录中的可积动态、进入时间或同类耦合是否已经覆盖本文特例。
- Jing 的 dimensional collapse、相关深矩阵分解中的齐次尺度律，以及 GUPA landscape 是否给出等价结果。
- Chen v2 全部动态附录是否含可转化为本模型的分析。
- v0.2 已补上冻结头、最小两通道与一族非零增强的机制比较；v0.3 明确层间尺度等价于有效学习率比例。尚未排除一般齐次因子化文献已给出相同的方向翻转或风险非单调机制；也没有证实视觉 encoder 退化。
- 本次没有完成前向引用图；不得用“未搜索到”推断没有近邻。

因此本目录是可继续推导与讨论的理论初稿，不是通过新颖性验收的投稿方案。

## 2026-10-19 的机制分析进度（非文献复核）

冻结头对照已经完成，并确认旧单通道例子的方向竞争不需要联合训练。v0.2 另给出一般双频反超条件、两通道匹配初始 A 的层尺度符号阈值、无 ridge 风险局部先改善后恶化的证明，以及有限大小非零增强下的结论。较长轨迹与冻结头、直接矩阵的对照已经执行。

因此上文历史记录中“冻结投影、多通道比较尚未完成”已由 v0.2 的局部证明与数值对照推进，但全程解析界仍未完成。这些是本稿数学工作的进展；是否已被相关文献覆盖，仍未重新核验。
