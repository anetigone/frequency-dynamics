# 定理级邻近文献对照（2026-10-21）

本记录回答一个有限问题：所核近邻是否已经证明，实际联合训练的参数化编码器在预训练目标下降时，使独立任务的**最优、无正则、总体线性平方风险**先低于初值、后以固定间隔超过初值？它不是“首创”证明，也不以检索未命中推断已有文献不存在。

当前结论：下面八篇重点全文中，没有定位到同时具备上述训练对象、风险定义和两个有限时风险间隔的直接等价定理。不过，相同初始函数下不同几何路径、projection 前后信息差异、训练中表征表现回落、以及有限时几何过冲和持续时间，分别已有很近的先例。新稿应比较这个精确组合，不能把其中任一宽泛现象当成首创。尤其需要新增 Yang–Saxe–Latham 的直接讨论。

## 核对口径与本稿实际范围

下文页码均为所链接 PDF 的一基物理页码；HTML 用于定位，公式、假设和定理声明以 PDF 对读。核查了相关声明及其直接证明段落，没有逐行复证每篇的全部附录。

比较的风险为 `inf_a E[(Y-a^T r)^2]`，允许重新选择任意线性读出；这与训练目标、固定读出的误差、正则化探针、有限标签分类准确率、跨情境迁移探针和表示条件数均有区别。固定秩下纯可逆缩放不改变该风险，旋转保留子空间才可能改变它。

当前新稿的具体候选增量是：平方 patch 编码器与线性 head 的实际 Euclidean GUPA 流；`K=1<C=2<d=3`；固定秩编码器及投影的总体 OLS 风险；一个共同任务锥上的两个显式非零间隔；以及指定非对称六坐标初始化盒的有限时间延拓。对称族的 ±5% 协方差结果保留对齐、相对协方差增强和白化平衡初始化；非对称盒则固定各向同性数据、增强功率 `.1` 和两学习率为一。两种稳定性**没有联合相乘**，且共同角度截面不是共同物理时间或共同更新预算。

## 八篇重点全文

### 1. Kunin 等：精确联合训练、初始化不平衡和学习率

版本：[2406.06158v2](https://arxiv.org/abs/2406.06158v2)，2024-10-12（v1 为 2024-06-10），NeurIPS 2024；[PDF](https://arxiv.org/pdf/2406.06158v2)。

| 项目 | 原文证据与对照 |
|---|---|
| 训练与假设 | §3，PDF pp4–5，式 (1)–(3)：`f(x)=a w^T x` 的监督平方损失，`a,w` 都进行实际梯度流；白化设计，层学习率允许不同。相同初始乘积、不同层尺度直接见 Fig. 2–3。 |
| 精确结果 | 守恒不平衡量约束路径；App. A.2 给完整解。Thm. 3.1（p6）给插值极限的隐式偏置；Thm. 4.1–4.2（p7）给较宽线性网络的有效梯度预条件结构及相应平衡条件。 |
| 风险与有限间隔 | 核到的定理针对监督训练目标、学习路径和极限偏置，没有将独立任务总体 OLS 的两个固定风险间隔列为结论；也没有本稿的编码器/head 双风险任务锥。 |

不能据此说一般任务风险回升是新现象：**我们的推断**是，已有精确旋转轨迹配合位于中途方向的独立任务，就可能导出探针先改善后恶化；“原文未命名此风险”不足以形成强新颖性。应突出无监督平方瓶颈下的结构族及量化结论，而不是一般层尺度效应。

### 2. Torralba–Weiss：相同平方 CNN / GUPA，已有分类回落实验

版本：[2607.07470v1](https://arxiv.org/abs/2607.07470v1)，2026-07-08；[全文](https://arxiv.org/html/2607.07470v1)。

| 项目 | 原文证据与对照 |
|---|---|
| 损失与架构 | Thm. 1.1（PDF p3）在 Gaussian embedding、无限 batch 条件下联系 GUPA 与 InfoNCE；Thm. 3.2（p6）研究卷积、平方、GAP、线性 head，采用平稳信号与 DFT 渐近条件及指定增强。 |
| 有效参数与最优解 | App. M，p28，式 (46) 写 `W_eff=WH`，指出不同分解对应同一函数。Thm. M.1 要求 `C=K` 且**冻结随机正交 head**，是最优解滤波器结构结论，不是实际联合流的任务风险定理。 |
| 回落先例 | Fig. 24–25（p29）实际平方 CNN 在 CIFAR-10 上 GUPA 损失下降，而投影 KNN 分类表现约在 epoch 100 后回落到 epoch 5000。该指标是经验 KNN 分类，不是最优总体 OLS 风险。 |

这是架构、目标和现象最直接的近邻。不能宣称首次发现此模型下损失下降与分类恶化并存；差别应明确到有限时间解析机制、编码器风险和显式稳定间隔。

### 3. Xue 等：逐层加权及真正非线性联合训练

版本：[2403.11391v1](https://arxiv.org/abs/2403.11391v1)，2024-03-18，ICLR 2024；[PDF](https://arxiv.org/pdf/2403.11391v1)。

| 项目 | 原文证据与对照 |
|---|---|
| 线性结论 | Thm. 3.3–3.5（PDF p4）：谱对比损失下最小参数范数/平衡条件与极限逐层特征权重。Thm. 3.4 的平衡保持是全时段动态结论，不能把全文概括成只研究最优解。 |
| 下游风险 | Thm. 3.6（p5）比较 hard-margin SVM 的样本复杂度指标；对同一支持空间的可逆线性重加权，并不自动给无正则总体 OLS 风险差异。 |
| 非线性实际动态 | Thm. 3.8（p6）：特定对角 BReLU、偏置/权重初始化及完全破坏一个坐标的增强下，联合梯度流使该坐标的 post-projection 特征趋零，而 pre-projection 保持非零。Thm. 4.2（p7）有监督版本。结论是渐近保留/丢弃，不是同一编码器风险的两个有限时固定间隔。 |
| 经验时间非单调 | App. D.2、Fig. 8（pp21–22）已有 post-projection 早停收益；不是时间非单调风险定理。 |

### 4. Tepe：此前全文缺口已解除

版本：Cem Tepe，MIT MEng thesis，2024-05；[官方仓储与持久链接](https://hdl.handle.net/1721.1/157033)，[官方 PDF API](https://dspace.mit.edu/server/api/core/bitstreams/43f70d34-c6f1-4ece-be31-63213af069de/content)。原 `SOURCE_AUDIT.md` 的“未取得全文”状态在本轮不再成立。

| 项目 | 原文证据与对照 |
|---|---|
| 一层及可积条件 | §2.1（p17）、§3.1（pp23–24）：经验谱对比损失、加性零均值增强；输入与增强矩阵可交换，初始 Gram 在同一基下对角，得到标量 logistic 模式方程 (3.3)–(3.6)。 |
| 极限定理 | Thm. 2（pp27–29）假设流收敛，控制极限 Gram 与最小范数解的距离；Cor. 1 处理小初始化极限。 |
| 深线性实际训练 | §4 的平衡分层因子流，Cor. 2，式 (4.1)（p31）给奇异值 ODE；Prop. 1（pp32–33）给方向投影受控后的增长比较；Prop. 2（pp34–35）假设方向未来一直在增强相关矩阵核中，给奇异值幂率衰减。 |
| 边界 | §5（p39）明确未分析奇异向量演化，相关时间尺度需额外假设。核到的结果没有独立任务总体 OLS 的有限时间先降后升间隔。谱幅值衰减不能直接等同于固定秩任务风险增加。 |

### 5. Liao 等：有限训练阶段与不平衡特征，不是独立探针风险

版本：[2602.10357v1](https://arxiv.org/abs/2602.10357v1)，2026-02-10；PDF 标注 ICLR 2026；[PDF](https://arxiv.org/pdf/2602.10357v1)。本轮已读全文相关声明和附录定义，旧“仅摘要待核”缺口已解除。

| 项目 | 原文证据与对照 |
|---|---|
| 训练对象 | §2：简化 Transformer attention + bilateral ReLU MLP；InfoNCE 带第二分支 StopGrad 和 weight decay。不是本稿无正则实际两块 GUPA 流。 |
| 条件与定理 | Assumptions 3.1–3.4（PDF pp6–7）含正交字典、稀疏有界潜变量、不同激活概率和共享潜变量正样本对。Lemmas 3.1–3.2（p7）、Thm. 3.1（p8）分三个有限训练阶段描述神经元特征/噪声；Thm. 3.2（p9）分析 pruning。 |
| 风险定义 | App. B 式 (23) 的 population risk 指期望**预训练对比损失**。主定理给 neuron 分解和预训练误差，不是每个训练状态重拟合最优总体线性探针的两个风险间隔；不能仅凭“population risk”措辞视为相同结果。 |

### 6. Jing 等：联合线性流、对齐和维数坍缩

版本：[2110.09348v3](https://arxiv.org/abs/2110.09348v3)，2022-04-23（初稿 2021-10-18），ICLR 2022；[PDF](https://arxiv.org/pdf/2110.09348v3)。

| 项目 | 原文证据与对照 |
|---|---|
| 训练与动态 | §5.2 Lemma 3（PDF p5）写出非归一化两层线性 InfoNCE 的实际联合梯度流。Thm. 2–3（p6）在正定数据/增强矩阵、非零乘积及相应极限/谱条件下，分析层间奇异向量对齐和对齐后的奇异值动态。 |
| 结果边界 | §4 Thm. 1 及 §5 描述维数坍缩机制；并非独立任务最优总体 OLS 风险的有限间隔。特别是趋零的奇异值在有限时间仍非零时，不能仅由条件数变坏推出无正则总体信息已损失。 |

### 7. Tian：非线性模式动态已有结构与稳定性

版本：[2206.01342v3](https://arxiv.org/abs/2206.01342)，2023-03-03（初稿 2022-06-02），ICLR 2023；[PDF](https://arxiv.org/pdf/2206.01342v3)。

| 项目 | 原文证据与对照 |
|---|---|
| 架构范围 | Assumption 1（PDF p4）的齐次条件允许总体常数因子，正文包含 monomial；不能用“只研究 ReLU、不含平方”排除。权重归一化约束与本稿不同。 |
| 精确/约化动态 | Lemma 4（p7）有两层精确动态式 (12)。Assumption 2 再加均匀权重、大 batch、无增强、快顶层处于平衡、给定类别的局部条件独立等条件。 |
| 结构结论 | Thm. 5–6（p8）证明 global modulation 与模式吸引域扩展；Thm. 3（p5）给模式固定点/扰动邻域条件。它们是模式选择及稳定性结论，不是本稿的任务风险回升间隔；但“非线性参数化会选择不同模式”已有严格先例。 |

### 8. Yang–Saxe–Latham：必须加入的几何过冲与持续时间近邻

正式题名是 **How are linear representations learned? Exact solutions to the dynamics of abstraction**，不是检索时使用的描述性简称。版本：[2607.08843v1](https://arxiv.org/abs/2607.08843v1)，2026-07-09；[PDF](https://arxiv.org/pdf/2607.08843v1)。

| 项目 | 原文证据与对照 |
|---|---|
| 训练/约化假设 | Assumption **1**（PDF p4）使读出在每个时刻都是 ridge-optimal variable-projected readout；Assumption 2 加二因素对称核条件。不是有限两学习率下实际联合 readout 流。App. A.4（pp24–25）用实验比较 joint GD 并承认早期差异。 |
| 真实有限时结论 | Thm. 5（p5）分析初始化尺度控制峰值；Prop. 6（p6）给近完美 abstraction 的 `Theta(log(1/kappa))` 停留时间；Prop. B.2（pp38–41）在指定参数条件下证明有限时 crossing、唯一 overshoot 峰值及之后从上方趋近极限。 |
| 指标与任务 | abstraction 是跨情境概念方向夹角。Prop. 1（p4；证明 pp26–28）联系固定情境训练、另一情境测试的探针分类误差，并有 Gaussian 近似/其他 SNR 固定条件；不是所有情境总体上重拟合的无正则最佳 OLS 风险。 |

因此，“首次证明训练几何发生非单调”“首次给持续时间”都不可写。可比较的窄增量是实际无监督联合参数训练、编码器瓶颈风险本身、统一任务锥与有限风险间隔及初始化邻域。

## 其他筛查及未覆盖部分

- [Gui–Ma–Zhong, 2306.03335v1](https://arxiv.org/html/2306.03335v1) 的 §1.4 明确理论以已训练 encoder 为给定，研究 head 的 expansion/shrinkage 和下游分类；本文也有联合 SimCLR 实验，不能概括为完全不做联合训练。Thm. 5.1–5.2 涉及正则化 logistic/高维 max-margin 分类，风险口径不同。
- [Chaudhry, 2605.17180v1](https://arxiv.org/html/2605.17180v1)，2026-05-16：Thm. 3.1–3.2 为定点/给定路径上 head 的存在性；Thm. 4.1 的 collapse instability 带非零残余梯度、曲率泛化和快 head 条件。不能将存在性/局部 Hessian 结论直接替换成有限层间配比的实际流保证。
- [Defilippis 等, 2509.24882](https://arxiv.org/abs/2509.24882) 的平方网络 feature-learning 风险研究，筛到的非单调主要沿样本量/weight decay 和 double descent；尚未完成所有附录逐定理比对，不能据此排除任何可推导的训练时间结果。
- [Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks, 2610.20678v2](https://arxiv.org/html/2610.20678v2)，2026-10-16 修订：主代理随后成功读取官方 HTML，核对 §2 式 (1)、Thm. 4.1、4.3 及其相邻证明。其模型为 `f_W=<WW^T,(xx^T-I)/sqrt(2)>`，以固定正定 teacher 的总体平方损失训练；定理比较两输入分布下的 teacher 损失历程及弱耦合阶数，不是每个时刻重新选择 decoder 的独立任务风险。Jordan 乘积和真实因子更新已在该文使用，不能将这些工具本身当新颖性。本次未逐行复核其所有可辨识性证明；不以术语 probe 未命中作为排除依据。
- [Neural Quadratic Forms, 2608.13335v1](https://arxiv.org/html/2608.13335v1)，2026-08-13：追加核对官方 HTML 的 Thm. 2、Assumption 2 与 Thm. 7。Thm. 2 给实际 SGD 的 `(M,mu)` 闭合及二阶步长项，Thm. 7 在各向同性结构矩二阶矩和监督平方损失下给矩阵 Riccati 闭式解，并含可交换初始化的模态解。它是本稿二阶矩、因子流和可解性的直接结构近邻；这些工具不应包装成新贡献。当前带独立 trainable head 的平方模型整体是三次参数化，吸收正 head 会改变 Euclidean 度量；本稿的 encoder 任务风险及非对称盒仍需用其自己的推导。该追加只核上述声明与邻近推导，不冒充复证全文。
- 原 `SOURCE_AUDIT.md` 的 Chen–Zhou–Zhou scoring-risk 对照仅沿用为线索，本轮不将其计入八篇新全文审计。

八篇重点文献未定位到与本稿六维非对称初始化盒相同的任务风险稳定定理；这句话只报告阅读结果。Kunin 的连续初值解、Tian 的模式稳定邻域及 Yang 的过冲条件均提示，必须继续核查是否有简单变量替换即可得到我们的某一弱化版本，不能用术语不同充当新颖性。

## 建议采用的贡献措辞

可写成：“我们研究实际联合训练如何改变独立任务的最优总体线性可读性，而不依赖有限标签误差或探针正则化。已有工作刻画了层间不平衡、逐层特征权重、投影前后信息差异和几何过冲；这里给出一个真实平方编码器瓶颈中的解析风险族，以及具有明确间隔和可达性的非对称初始化邻域。”

不建议写：首次发现 head 遗漏信息；首次发现层学习率改变表示；首次证明训练后语义表现回落；首次给表征几何的非单调或持续时间；架构恒等式/OLS 恒等式本身构成主要创新；未检索到等价定理因此已确立首创。

当前最强的可辨识组合是**实际训练对象 + 总体无正则风险 + 编码器本身 + 统一任务族的显式间隔 + 非空的非对称初始化盒**。它仍不证明自然图像预测价值，也不证明初始化具有可观概率。尤其不能把 `8e-6` 邻域写成宽泛鲁棒性。

## 检索与获取边界

遵循本机 `academic-search/SKILL.md`：先读已有 `paper/references.bib`、`SOURCE_AUDIT.md`、`paper/notes/SOURCES.md`；精确追踪已知条目，再用 arXiv/API 和定向 primary 查询扩展。API 两条宽查询分别含 contrastive+dynamics/nonmonotonic/quadratic 和标题 contrastive+dynamics，各筛 20 条，词干/题名召回精度有限；补充查询覆盖 projector、population risk、quadratic encoder、nonmonotonic representation、abstraction overshoot。没有完成前向引用图、Google Scholar 全覆盖或所有 2026-10 新稿核验。

八篇均从 arXiv、会议原文或 MIT 官方仓储取得可读全文。只为 Torralba–Weiss 和 Tepe 各保存一份已确认 OA 的单篇 PDF 及用于阅读的文本；没有批量下载、清单、源码/文件哈希或引用数流程。引用次数不属于任务，未查询；不存在以高引用排序代替定理核对。Tepe 标准下载链接限流后改用同一官方仓储公开 API，不绕过付费或验证。

本记录的“未定位到”仅约束上述版本和阅读范围；所有未读候选均保留为未核，而非负面结论。

## 建议新增 BibTeX（已核主源元数据；未改 references.bib）

```bibtex
@mastersthesis{tepe2024dynamics,
  author = {Tepe, Cem},
  title = {Dynamics of Gradient Flow with Contrastive Learning},
  school = {Massachusetts Institute of Technology},
  year = {2024}, month = may,
  url = {https://hdl.handle.net/1721.1/157033}
}
@article{liao2026imbalanced,
  author = {Liao, Haixu and Zhou, Yating and Zhang, Songyang and Wang, Meng and Zhang, Shuai},
  title = {Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution},
  journal = {arXiv preprint arXiv:2602.10357},
  year = {2026}, note = {Version 1; PDF marked ICLR 2026},
  url = {https://arxiv.org/abs/2602.10357v1}
}
@article{yang2026abstraction,
  author = {Yang, William W. and Saxe, Andrew M. and Latham, Peter E.},
  title = {How are linear representations learned? Exact solutions to the dynamics of abstraction},
  journal = {arXiv preprint arXiv:2607.08843},
  year = {2026}, note = {Version 1},
  url = {https://arxiv.org/abs/2607.08843v1}
}
```
