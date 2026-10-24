# 本机经验训练与自然图像外推协议

2026-10-19。这里是上一轮总体机制确认之后的完整经验阶段；不增加总体轨迹。
训练协议及源码哈希在各正式输出目录的 protocol.json 中保存，参数或源码不符时拒绝混入结果。整理后的路径与历史源码快照见 [目录与复现说明](REPOSITORY_LAYOUT.md)；重训请选择新的输出目录。

## 完整经验主实验

固定 d=16、C=8、K=1，32×32 周期图像、全支撑真实像素卷积核。
三个模型为平方、普通线性功率因子化、冻结前层；r=0.1/1/10，独立校准，种子 2–11。
主实验为 90 条完整轨迹，每条 batch=256、lr=0.002、4000 次普通 SGD 更新。
模型共享 32,768 个有限训练样本、批次、增强随机数；增强功率 eta=0.2。
数据和初始化由固定种子确定；数据哈希记录于每条结果，初始 A 对齐已检查。

独立校准延用旧规则：分别等化两活跃参数块 Jacobian Gram 迹，再将相应配比下的初始加权总迹设为 1。
冻结前层只校准活跃 head，因此它的三种名义配比在这一归一化下更新相同，作为负对照。

预训练不读取标签、任务风险、总体梯度。不使用动量、weight decay、BN、梯度裁剪或输出归一化。
采用精确周期平方网络恒等式计算有限批次前向，自动微分更新真实像素核；不是直接优化 B，亦不是总体 ODE。
已在经验训练第 2000 步的全支撑/局部、平方/冻结状态上，用实际图像卷积连续执行 8 次相同 SGD，
与快速恒等式的参数差最大约 2.3e-16，见 trained_sgd_image_checks.json。

### 固定预算与能量边界

每条运行到固定 4000 步，不按能量或标签风险提前停止。GUPA 为直接定义的经验矩目标，保留完整两项 uniformity。
没有激活输出能量约束，后段 Q 可能超过定理 T 中 Q<1 的区间；这部分只是经验外推。
`inside_T_energy_region` 字段只检查能量条件，不表示高维实验满足二维定理 T 的全部假设。

主固定能量比较预设 ell=1.5，其他报告截面为 0.75、2.25、3。
在首次向上穿越的 SGD 更新中，对更新前后参数作线性插值，求相应能量位置，仅用于评估；实际训练保持原样。
所有主实验和局部复验的条件均到达 ell=1.5，没有删除未达主截面的种子。
所有固定更新步数和固定能量结果分别报告，不混作同一停止规则。

开发种子 0、1 比较普通一步与相同批次的两个半步，总步长/批次序列匹配。
末端任务风险最大差约 2.89e-4、encoder 投影位移最大差约 3.39e-4。
两个半步只用于数值步长诊断，正式训练每批只作一次普通 SGD 更新。

## 条件稳定性与随机波动

在主实验 r=1 的平方/线性模型上，对每个正式种子补两条不同的批次/增强流，
保留同一初始化和有限训练集。连同主流，每种子三条流；额外 40 条轨迹。
末端 Delta P 的两两余弦先在同一种子内平均，再按十个种子汇总。
这些是给定数据和初始化的方向重复性，不是跨初始化共用同一个最优语义方向。

另在开发种子 0、1，第 2000 步的三模型状态上，对 batch=256/1024/16384 各抽 64 次独立批次和增强。
分别保存 E Delta P、E||Delta P||²、方差迹及均值 RMS 抽样误差估计，不预设线性均值为零。
大 batch 从同一有限训练集抽样，不等价于对无限数据总体求平均。

额外条件矩解释为事后探索性分析，不能当作新控制器的独立确认。
若 clean 功率为 e，有限训练集均值为 mu_N、协方差为 Sigma_N，则

\[
S_N=\Sigma_N+\operatorname{diag}(2\eta\mu_N+\eta^2),\qquad
D_N=2\operatorname{diag}(2\eta\mu_N+\eta^2).
\]

这给出对有限数据和增强的精确视图二阶矩；将其代入 GUPA 是大 batch 参考，
不是非线性有限 batch GUPA 的精确平均梯度。Sigma_N 通常不是 I，
线性模型便不再满足总体各向同性守恒的前提，可能存在可重复的平均子空间移动。
独立视图抽样已核验该矩公式，矩阵相对误差约 1.2%。

## 经验 checkpoint 干预

使用完整经验训练 r=1 的第 500/2000/3500 步状态，三模型、十种子，共 90 个状态。
预测由有效映射、解析总体 GUPA 梯度和真实参数 Jacobian Gram 得出，不对风险作自动微分。
预测在抽取未来批次前保存；仍使用已知任务 beta，属于任务条件的 oracle 机制检验。

每状态抽 32 条独立未来批次/增强流，用共享输入比较配比 0.1/1/10。
仅对预设 x_high 任务，若总体正阈值存在，再加阈值一半和两倍的括号点。
单步符号主汇总不包含额外括号点，排除预测绝对变化不超过 1e-9 的数值中性项。
第 2000 步状态的前 8 条流继续到 8/64 步，分别报告概率和幅度误差。

阈值预测单列：正阈值存在概率、条件 log10 误差、无阈值情形都保留。
线性总体前层方向贡献严格为零，代码按恒等式处理，避免浮点抵消造出极小伪阈值。
经验有限 batch 的前层贡献不作这个处理。
统计单位始终为配对种子；批次、任务、窗口和 checkpoint 不作独立样本。

## 局部经验复验

主实验完成并检查后，再运行 11×11、K=1 的同协议十种子 90 条经验轨迹。
局部频率耦合由实际像素核保留，不自由优化 Fourier 功率坐标。
K=2 的总体确认保留；本轮没有再展开 K=2 的完整经验组合。

## CIFAR-10

用户提供的 CIFAR-10 Python 文件共七个，全部 MD5 与 torchvision 标准值一致。
50,000 张训练图、10,000 张测试图，32×32 RGB、各类数量正确；不重新下载或改写原始数据。

模型为灰度输入、64 个 11×11 valid 滤波器、平方、GAP、8 维线性投影。
无 bias、BN、momentum、weight decay 或输出归一化。真实图像卷积训练，float32，确定性 CUDA 算法。
固定分层划分 45k/5k；灰度标准化的均值与标准差只由 45k 计算。
增强为 reflect padding 4 的随机 crop、水平翻转、0.8–1.2 contrast、0.05 Gaussian noise。

校准使用固定随机抽取的 256 张无标签训练图，以 16 次 Hutchinson VJP 估计 centered-output 前层 Gram 迹，
head 迹直接计算，按相同齐次尺度规则平衡，保持初始函数。
早期吞吐开发曾使用类别排序前缀，正式协议前已修正为固定随机抽样；旧吞吐结果仅用于开发审计。
每个种子的三配比共享初始化、批次、增强流，普通 SGD lr=0.002、batch=256。

先完成每配比 200 步吞吐检查，再锁定五种子 2–6、三配比、每条 4000 更新的本机比较。
这约等于 22.8 个训练集遍历，是有限预算外推，不声称收敛或长程标准 benchmark。
本机实测约 0.01 秒/步，峰值约 0.43 GiB；没有使用服务器。

固定步数 checkpoint 为 0/1000/2000/3000/4000；共同能量代理为固定无标签校准集的 centered clean 输出方差。
在每 10 步的测量网格上保存 log(V/V0)=0.5/1/2 的首次穿越，报告实际超调量，非精确总体能量截面。
OLS 拟合 intercept、无 ridge；列标准化为可逆数值变换，不是 probe 正则化。
固定 checkpoint 的验证集曲线与共同能量结果独立保存。
全部预设训练完成后，才在固定最终步统一评估测试集；不按测试标签选停止点或配比。
自然图像分类不是前面总体平方风险的替代，也不直接验证二维 c*。

## 入口与输出

PowerShell 从仓库根目录执行，下列 python 可替换为 `& H:\anaconda3\envs\py312\python.exe`。

```powershell
python -B experiments/code/checks/check_full_empirical.py
python -B experiments/code/empirical/full_empirical.py --phase confirmation --seeds 2 3 4 5 6 7 8 9 10 11 --output experiments/results/empirical/full_sgd_main
python -B experiments/code/empirical/full_empirical.py --phase repeat --seeds 2 3 4 5 6 7 8 9 10 11 --models square linear --ratios 1 --stream 1 --output experiments/results/empirical/full_sgd_repeat1
python -B experiments/code/empirical/full_empirical.py --phase repeat --seeds 2 3 4 5 6 7 8 9 10 11 --models square linear --ratios 1 --stream 2 --output experiments/results/empirical/full_sgd_repeat2
python -B experiments/code/empirical/full_empirical.py --phase confirmation --seeds 2 3 4 5 6 7 8 9 10 11 --kernel-size 11 --output experiments/results/empirical/full_sgd_local
python -B experiments/code/checks/check_trained_sgd.py
python -B experiments/code/empirical/summarize_full_sgd.py experiments/results/empirical/full_sgd_main --repeat-dirs experiments/results/empirical/full_sgd_repeat1 experiments/results/empirical/full_sgd_repeat2
python -B experiments/code/empirical/summarize_full_sgd.py experiments/results/empirical/full_sgd_local
python -B experiments/code/empirical/batch_drift_diagnostic.py
python -B experiments/code/empirical/conditional_moment_diagnostic.py
python -B experiments/code/empirical/plot_batch_drift.py
python -B experiments/code/empirical/intervene_full_sgd.py
python -B experiments/code/empirical/summarize_full_interventions.py experiments/results/empirical/full_sgd_interventions
python -B experiments/code/cifar/audit_cifar.py
python -B experiments/code/cifar/cifar_local.py --phase bounded_comparison --seeds 2 3 4 5 6 --steps 4000 --evaluate --output experiments/results/cifar/cifar_local_five
python -B experiments/code/cifar/evaluate_cifar_local.py experiments/results/cifar/cifar_local_five
python -B experiments/code/cifar/summarize_cifar_local.py experiments/results/cifar/cifar_local_five
```

Windows 环境中的数值评估与 NumPy/Matplotlib 汇总使用两个独立进程，避免混合运行时的 OpenMP 冲突。
逐条件 evaluation.json 可缓存复用，不必重算已完成的测试指标。

- `results/empirical/full_sgd_main`：90 条完整经验主轨迹、配对风险、公共能量、训练 checkpoint。
- `results/full_sgd_repeat1/2`：固定数据与初始化的额外 40 条独立 SGD 流。
- `results/empirical/full_sgd_local`：局部经验 90 条。
- `results/empirical/batch_drift_diagnostic`：条件均值/方差及探索性条件矩解释。
- `results/empirical/full_sgd_interventions`：经验状态重复分叉、概率、幅度和阈值分开汇总。
- `results/cifar/cifar_audit.json`：用户数据的标准文件验证。
- `results/cifar/cifar_local_five`：自然图像固定预算五种子比较及测试评估。

历史总体认证与其计算机辅助证明保持原覆盖范围；新的经验/自然图像结果均不替代定理 T。
