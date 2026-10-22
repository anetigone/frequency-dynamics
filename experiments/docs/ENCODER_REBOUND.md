# 固定任务下的完整 encoder 回升：小规模验证

2026-10-22。问题是：不采用二维定理的对称初始化，并且不随训练轨迹选择任务时，最优 encoder 线性读取风险是否在固定的两个时间先低于、再高于自身初始值？本轮复用已有纹理平台，补齐风险轨迹与机制对照，不继续扩大分类实验或短期预测框架。

这里的设置根据历史结果的探索性分析确定，在这批新轨迹运行前写下。它是受既有数据启发的小规模验证，不能称为公开预注册，也不能把历史分析重新标成独立确认。

## 现有证据与新验证的分工

`full_sgd_main` 和 `full_sgd_local` 已有各 90 条完整经验训练记录。每条记录包含十个固定任务的 encoder/projection 风险、总体目标、秩与奇异值。原主文主要汇总空间位移，完整回升尚未成为这些结果的主要分析对象。重新汇总这些记录不需要重新训练。

旧任务由 `square_images.texture_tasks` 预先固定；但本次检验所用的第 1,000/4,000 步及幅度阈值 0.001 是在 2026-10-22 的重分析中选择的。因此旧数据给出的是探索性回升证据。所有旧种子、所有任务、未回升情形均保留，不能只展示成功曲线。

新验证使用种子 20–25，与历史纹理完整轨迹的种子 0–11 分开。当前经验结果目录中未发现这些新种子的完整纹理轨迹；“新”限定于本次纹理轨迹验证，不声称这些整数从未用于其他实验。新结果写入单独目录，保留全部历史数据。

## 固定设置

| 项目 | 本轮设置 |
| --- | --- |
| 输入 | 32×32 周期纹理，四方向、两径向频带，每组两个独立共轭频率对 |
| 表示 | d=16 个标准化 clean 功率坐标；C=8 个 encoder 通道，K=1 个 projection 输出 |
| 卷积 | 32×32 全支撑真实像素核；用精确 Fourier 恒等式加速前向，自动微分更新像素核 |
| 数据与增强 | 每种子 32,768 个固定无标签样本；独立双视图 Gaussian 增强，功率 η=0.2 |
| 优化 | float64 CPU，batch=256，普通 SGD，lr=0.002，r=η_W/η_h=1，4,000 次更新 |
| 种子与模型 | 种子 20、21、22、23、24、25；每种子 square、linear、frozen_front，共 18 条轨迹 |
| 记录 | 初始化、第 1,000/4,000 步及每 25 步；风险曲线和数值秩敏感性均覆盖全部记录点 |
| 比较点 | early=1,000，late=4,000；均为固定更新步数，不按风险最低点或能量穿越选取 |
| 幅度门槛 | 两段分别严格大于 δ=0.001，风险以单位标签方差计 |
| 输出 | `experiments/results/empirical/encoder_rebound_validation` |

初始化从原平台的高斯分布独立抽取：每个 full-support 频率的两个实系数为 `N(0,1)/4`，初始 head 元素为独立 `N(0,1)`。随后将有效映射 A 的初始平方范数设为 0.001，按无标签 Jacobian Gram 迹作尺度平衡及初始总 mobility 归一化。这个分布没有施加两滤波器对称关系，但带有明确的能量与尺度校准；不称为所有常规初始化的代表。

三模型由同一种子的基础平方模型构建：linear 直接训练同一功率字典中的线性因子，frozen_front 只训练 head。各模型独立进行迹校准，保留相同的初始 A 与初始 encoder 行空间。它们共享有限训练集、批次与增强随机数。记录初始风险、能量、行空间和校准量，避免把起点差异解释成机制效应。

预训练不使用任务标签、任务风险或总体梯度。不增加动量、weight decay、BN、梯度裁剪或输出归一化；不按曲线结果更换任务、种子、阈值或停止时间。

## 任务、风险与主要报告

固定十任务为 `single_x_low`、`single_y_low`，以及 `x_low`、`x_high`、`y_low`、`y_high`、`diagonal_low`、`diagonal_high`、`antidiagonal_low`、`antidiagonal_high`。前两项各取一个功率坐标，其余为相应两个功率坐标的单位范数和。定义在训练前给出，不依赖任何一条训练轨迹。

合成分布的标准化 clean 功率协方差为 I，标签为固定 β 的线性函数且方差为 1。因此可直接计算

\[
R_{\mathrm{enc}}(t)=1-\beta^\top P_{B(t)}\beta,\qquad
R_{\mathrm{proj}}(t)=1-\beta^\top P_{A(t)}\beta.
\]

这是对已知合成分布的无正则总体最优线性读取风险，不是有限标签 OLS 测试误差。训练仍是有限数据上的经验 SGD；解析总体风险和总体 GUPA 仅供评估，不能反向进入训练。

对每个种子、模型、任务分别报告两段幅度：

\[
I=R(0)-R(1000),\qquad D=R(4000)-R(0).
\]

只有 `I > 0.001` 且 `D > 0.001` 才计为该任务的完整回升。encoder 和 projection 分开统计，不能以 projection 成功替代 encoder 成功。每个固定任务均报告 `成功种子数/6`、六个种子的 I/D 及其描述性幅度汇总，并给出三个模型的配对对照。

十个任务分别是主要报告对象，不在看到新结果后选出一个“主任务”。“每种子十任务中至少一个成功”若列出，只是明确标注的集合并集描述；它不等于某个固定任务的发生率，更不等于定理中的整个任务族统一改善和恶化。任务、checkpoint 和配对模型都不当作额外独立样本，独立重复单位为随机种子。

另外保留三点判定的其他结果：有早期改善但终点未越过初始化、有终点恶化但无达标早期改善、两段均未达标。完整曲线可作描述性最低点分析，但经验最低点不参与本轮主要成功判定，也不作为无偏的改善证据。

风险图同时给出 `R_enc(t)`、`R_proj(t)` 和 `L(t)`。L 显示各记录点的解析总体 GUPA；如展示单批经验目标须单独标注。经验 SGD 不保证每步总体损失严格下降，不能将曲线整体下降改写成定理式逐步单调性。

## 机制与数值检查

每条轨迹报告 encoder/projection 的数值秩、最小保留奇异值、最小/最大奇异值比及相对于初始化的 projector 位移；任务风险变化是同一位移对固定 β 的具体作用。

每个记录点用 SVD 比较四种相对截断：`default` 为 `eps * max(matrix.shape)`（encoder 的 matrix 为 B，projection 为 A），其余记录键为 `1e-12`、`1e-10`、`1e-08`。各阈值下重新计算秩与任务风险，并报告风险差和完整回升判定是否改变。不以默认秩恒定代替敏感性分析；敏感、病态或失败的运行仍留在结果中，不删除后重算成功率。

linear 对照检验相同字典下移除平方参数化后的风险轨迹；frozen_front 对照检验只靠 head selection 能产生多少 projection 变化。有限训练集和有限批次不精确各向同性，所以本轮不要求 linear encoder 严格不动，也不把它的非零运动判为实现失败。frozen_front 的 encoder 风险应在数值误差内恒定。

可对种子 20 作一个同批次两半步的数值细化对照，写入单独目录；它不补充独立种子，不参与 18 条主要轨迹的分母，也不能据其结果修改主要阈值。小尺寸 smoke 只用于核验运行与输出结构，不作为完整回升证据。

本轮不增加定理邻域套件、正交/相关两滤波器控制、局部卷积或自然图像属性实验。八通道有限数据实验也不能验证两滤波器各向同性条件下的正交排除结论。它补的是构造之外的有限经验桥接，不直接复现二维主定理，不证明该机制解释现代深层 SSL 的语义退化。

## 运行与停止

在项目根目录使用指定 Python：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/encoder_rebound.py --output experiments/results/empirical/encoder_rebound_validation
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/summarize_encoder_rebound.py experiments/results/empirical/encoder_rebound_validation
```

独立 smoke 与可选的细化对照：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/encoder_rebound.py --seeds 0 --steps 8 --early-step 2 --late-step 8 --record-every 2 --output experiments/results/checks/encoder_rebound_smoke
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/encoder_rebound.py --seeds 20 --microsteps 2 --output experiments/results/checks/encoder_rebound_refinement
```

每条主要运行固定到 4,000 步；共完成 18 条后停止。即使完整回升少或为零，也不自动增加种子、率比、训练预算或事后任务。运行错误保留种子、模型、已完成更新及原因，其余计划运行继续；缺失与失败都显式报告，主要分母不默默改为成功完成数。若技术故障修复后重跑，保留故障记录并说明实际完成状态。

汇总默认写到输出目录下的 `encoder_rebound_summary`，包含 `runs.csv`、`summary.json`、`findings.md`、`risk_trajectories.png/pdf`、`risk_changes.png/pdf` 和 `geometry.png/pdf`。逐任务成功数与两段幅度是结论基础，图形用于检查完整路径及控制行为。

只有新验证在固定任务和固定时点出现幅度清楚、阈值稳定的 encoder 完整回升，才据此加强主文中的构造外证据。若只能看到位移、局部变化或 projection 回升，按其实际范围报告；不据此扩大 encoder 回升的适用性结论。

本轮已完成的 18 条轨迹、数值检查和结论边界见[结果记录](../../paper/notes/ENCODER_REBOUND_VALIDATION_20261022.md)。

后续评审补充的针对性检查沿用原主要结果，单独比较强效种子 21 和临界种子 22。它们不增加主要分母，不更换比较时间或门槛：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/encoder_rebound.py --seeds 21 22 --microsteps 2 --output experiments/results/checks/encoder_rebound_targeted_refinement
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/summarize_encoder_rebound.py experiments/results/empirical/encoder_rebound_validation --refinement-dirs experiments/results/checks/encoder_rebound_refinement experiments/results/checks/encoder_rebound_targeted_refinement
```

汇总还生成 `rebound_events.csv` / `rebound_event_loss.md`（回升与两阶段目标变化的交叉表）、`initial_pairing.csv`（由初始 B 独立核对空间与风险）和 `refinement_checks.csv`（保留原结果与细化结果）。`runs.csv` 列出全部幅度及相对门槛余量；后续增加的 1.25 倍接近门槛提示只作描述，不改变原回升判据。
