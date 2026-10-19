# 固定 encoder 空间内重选，与 encoder 行空间改变

2026-10-19。本文档区分解析结论、数值确认和仍未覆盖的外推。

## 1. 普通线性因子化的行空间守恒

设功率特征 q 已中心化且 Cov(q)=I，encoder 为 Vq，projection 为 Aq=WVq。
取可微总体目标 L(A)=F(AAᵀ)。当前各向同性增强的矩阵 GUPA 属于这个形式，
K=1 时退化为 phi(||A||²)。记 G=∇_A L=H(A)A，其中 H 对称且沿轨迹连续。
普通两层线性参数梯度流满足

\[
\dot V=-\eta_V W^\top G=-\eta_V W^\top H(A)W V=:C(t)V.
\]

在参数保持有限、C 在有限时间段上连续的条件下，基本解 U 满足
U'=C(t)U、U(0)=I，且 det U(t)=exp(∫tr C) 非零。
因此 V(t)=U(t)V(0)，row V(t)=row V(0)，无须假设 V 为方阵或可逆。
最优无正则总体 encoder probe 风险对每个固定 beta 都不变。

这不限于 K=1：依赖 AAᵀ 的各向同性矩阵目标同样适用。
离散 GD 每步为 V_next=(I−eta_V WᵀH W)V；若该左乘矩阵非奇异，也保持行空间。
经验矩目标、各向异性增强下 G 不一定可写成 H A，不能沿用结论。
这里不声称这个一般线性代数事实本身具有文献新颖性。

还有一个局部干预限制：只更新 V 时，A 的瞬时变化为 −eta_V WWᵀH A，
是 A 的左乘，因此 projection 行空间风险的一阶变化也为零。
在同一参数状态，正学习率配比只能正向缩放 head 造成的初始风险变化，不能翻转其符号。
这是一阶、总体、各向同性结论；不排除有限步交叉项、后续状态变化或经验噪声带来不同结果。

## 2. 平方像素参数产生不同的更新几何

全支撑周期平方卷积有 B_ck=g_k|h_ck|²，A=WB。记 G=∇_A L，像素度量已按正交实 Fourier 基校准。
其真实参数训练诱导

\[
\dot W=-\eta_W G B^\top,\qquad
\dot B_{ck}=-4\eta_h g_k B_{ck}(W^\top G)_{ck}.
\]

后一式通常不是某个公共左乘矩阵作用在 B 上，所以没有上面的普遍行空间守恒。
这不意味着每个初始化、每个任务都必然变化，更不意味着 encoder 风险统一变差。
前层与 head 对 A 的贡献为

\[
\dot A_h=-4\eta_h W[g\odot B\odot(W^\top G)],\qquad
\dot A_W=-\eta_W G B^\top B.
\]

局部核下保留实际像素核的坐标约束。对每个通道定义 J_c=D_kernel B_c、T_c=J_c J_cᵀ，
则 Bdot_c=−eta_h T_c(WᵀG)_c。全支撑时 T_c 为对角矩阵 diag(4g_kB_ck)，
局部核下含有跨频率项。代码显式保留这些项，没有把局部核的频率坐标独立更新。

行满秩 X 的行空间投影 P_X=Xᵀ(XXᵀ)^−1X。对固定任务 beta，令
v=(XXᵀ)^−1X beta、e=(I−P_X)beta，则

\[
\dot R=-2v^\top\dot X e.
\]

因此可用当前有效映射、参数更新几何和明确给定的预训练梯度 G 分别计算前层/head 的风险贡献，
无需对风险本身求自动微分。预测规则在真实小批量分叉前写出。
风险导数仍依赖已知任务 beta，不构成无需标签的调参方法。

## 3. 两种校准回答不同问题

匹配参数组：从共同的平方等迹初始化出发，线性模型使用 V0=B0、同一个 W0。
三种配比 r=0.1/1/10 共用 eta_front:eta_head=1:r、固定原始系数和。

独立校准组：每个模型使用相同的无标签规则，保持 A0，先让两组 Jacobian Gram 迹相等，
再将当前配比下的加权总 Gram 迹校准为 1。平方缩放指数为 c^6，普通线性缩放指数为 c^4。
冻结前层没有两个活跃部分，不强行等迹，只校准活跃 head 的总迹。
独立组的 V0 与 B0 可以相差标量，W0 随之补偿；同一 A0 和 encoder 行空间保持一致。
等迹控制更新强度，并不使几何方向相同。

主要比较在相同 log(V/V0)=3 截面，另记录 0:0.25:3 的全过程截面。
实际算法为正物理步长的像素/因子参数 GD；步长选择只控制能量增量，不用任务风险。
数值确认步长上限为 0.002；开发种子 0、1 已做 0.004→0.002 检查，encoder 位移最大差约 3e-4。

为提高重复效率，训练前向使用真实像素核的精确周期功率恒等式，自动微分更新像素核。
它不直接优化 B。全支撑/局部核及 K=1/2 的梯度已与真实图像卷积自动微分独立核对，
最大差约 6.1e-18；每条非线性训练轨迹末端再与真实图像前向比较。

## 4. 锁定协议的十种子结果

正式种子 2–11 与开发种子分开；三套协议均在开始任何正式训练前写出，并保存核心代码 SHA256。
三套设置为 K=1 全支撑、K=2 全支撑、K=1 11×11 局部核；C=8、d=16。
每套含两校准组×三模型×三配比×十种子，共 180 条；三套共 540 条。
同一随机种子在设置间复用，统计单位始终是十个配对种子，不能称为 540 次独立重复。

独立校准、r=1、ell=3 的种子均值如下：

| 设置 | 平方 encoder 位移 | 线性 encoder 位移 | 平方 projection 位移 | 线性 projection 位移 |
|---|---:|---:|---:|---:|
| K=1，全支撑 | 0.3064 | 1.15e-14 | 0.5161 | 0.6087 |
| K=2，全支撑 | 0.4499 | 1.23e-14 | 0.5597 | 0.5319 |
| K=1，局部核 | 0.3786 | 1.26e-14 | 0.5572 | 0.6196 |

位移为行空间正交投影矩阵差的 Frobenius 范数。所有设置和配比下，
平方模型十个种子的 encoder 均有可分辨位移；线性 encoder 保持舍入误差量级，冻结前层为零。
全部观察截面数值秩保持 C=8、projection 秩保持 K。
不同配比改变了平方 encoder 位移；它不等价于统一的任务风险损害。
完整配对统计、bootstrap 描述区间、两校准组及全部任务见
[结果汇总](results/subspace_summary/findings.md)和 [JSON](results/subspace_summary/summary.json)。

对于完整正交坐标任务，sum_j ||P_X e_j||²=rank(X)。固定秩下总风险恒为 d−rank。
结果说明保留方向和任务偏好的重分配，而不是空间维数减少或所有任务同时变差。

## 5. 与主定理及外推的关系

原主定理 T 的 C=d=2 可逆 encoder 上，总体 encoder 风险不变；
它的 projection 风险回升不需要 encoder 行空间改变。
本轮 C<d 的高维实验提供另一个可区分机制，不能倒过来解释原二维定理为 encoder 退化定理。
数值十种子确认也不替代新的高维区间认证或随机初始化概率定理。

真实经验 SGD 分叉使用 K=1 全支撑总体轨迹在 ell=0.75/1.5/2.25 的预设 checkpoint，
模型间匹配当前 B、W，批次和增强噪声共享，窗口为 1/8/64 步。
预测来自总体几何；实际执行经验 GUPA 图像训练。线性守恒在这里没有保证。
checkpoint 尚不是完整有限数据预训练的确认轨迹，因此迁移证据仍有这一限制。
分叉汇总见 [经验干预结果](results/subspace_empirical_confirm/findings.md)。

平方模型主任务 x_high 在 30 个相关 checkpoint 中有 18 个同时具有总体与首批经验正阈值、
1 个仅总体具有正阈值、11 个两者均无正阈值。条件平均绝对 log10 阈值误差约 0.367，
且只在有可比较正阈值的七个种子上定义；不能把短时符号吻合当成临界配比的高精度预测。
完整结果保留了无可翻转阈值的情况，没有强迫每个 checkpoint 都能翻转。

目前没有 CIFAR-10 或其他自然图像结果；本轮完成的是先区分 encoder 与 projection 的优先任务。

## 6. 复现入口

从仓库根目录执行；本地将 python 替换为 `& H:\anaconda3\envs\py312\python.exe`。
正式目录包含协议和核心源码哈希；参数或核心代码与锁不符时程序会拒绝混入结果。

```powershell
python -B experiments/check_subspace_mechanism.py
python -B experiments/subspace_mechanism.py --phase confirmation --seeds 2 3 4 5 6 7 8 9 10 11 --step 0.002 --output experiments/results/subspace_confirm_k1
python -B experiments/subspace_mechanism.py --phase confirmation --seeds 2 3 4 5 6 7 8 9 10 11 --outputs 2 --step 0.002 --output experiments/results/subspace_confirm_k2
python -B experiments/subspace_mechanism.py --phase confirmation --seeds 2 3 4 5 6 7 8 9 10 11 --kernel-size 11 --step 0.002 --output experiments/results/subspace_confirm_local
python -B experiments/summarize_subspaces.py
python -B experiments/empirical_subspace_interventions.py
python -B experiments/summarize_empirical_subspaces.py
```

`subspace_development`、`subspace_development_fine` 保留早期开发步长对照。
经验分叉早期目录 `subspace_empirical_branches` 保留了浮点抵消产生的线性极小伪阈值，
仅用于调试审计；正式分析只使用已按结构恒等式修正、重新运行的 `subspace_empirical_confirm`。
该修正依据总体线性前层的风险导数严格为零，非按经验分叉优劣挑选结果。
