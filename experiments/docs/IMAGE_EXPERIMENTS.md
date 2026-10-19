# 图像实验：实现与试跑协议

2026-10-19。新增图像代码使用 PyTorch；原有区间认证代码保持标准库依赖。
本轮是开发性 pilot，不是正式十种子实验，也不是新的计算机辅助证明。

## 环境与运行

本地已实际运行：Python 3.12.11、PyTorch 2.9.1+cu126、NumPy 2.0.2、
Matplotlib 3.10.6、RTX 4060 Laptop 8 GiB。没有安装或修改环境。

从仓库根目录运行，PowerShell 中可将下面的 `python` 替换为
`& H:\anaconda3\envs\py312\python.exe`。服务器使用其自己的 Python 环境。

```powershell
python -B experiments/code/pilot/verify_spatial.py
python -B experiments/code/checks/check_image_pipeline.py

# 原始随机初始化：总体矩目标与真实图像样本 SGD
python -B experiments/code/pilot/train_textures.py --objective population --device cpu --dtype float64 --interventions --ols --output experiments/results/pilot/textures_population_pilot
python -B experiments/code/pilot/train_textures.py --objective empirical --device cuda --dtype float32 --interventions --ols --output experiments/results/pilot/textures_empirical_pilot

# 保持初始有效映射的无标签层间尺度校准
python -B experiments/code/pilot/train_textures.py --objective population --device cpu --dtype float64 --balance-init --lr 0.01 --interventions --ols --output experiments/results/pilot/textures_balanced_population_pilot
python -B experiments/code/pilot/train_textures.py --objective empirical --device cuda --dtype float32 --balance-init --lr 0.01 --interventions --ols --output experiments/results/pilot/textures_balanced_empirical_pilot

# 匹配有效映射的必要对照；步长减半用于检查离散化误差
python -B experiments/code/pilot/texture_controls.py
python -B experiments/code/pilot/texture_controls.py --step 0.002 --output experiments/results/pilot/texture_controls_fine
python -B experiments/code/pilot/texture_controls.py --balance-init --output experiments/results/pilot/texture_controls_balanced

python -B experiments/code/pilot/plot_image_pilots.py
```

`--seeds`、`--ratios`、`--channels`、`--outputs`、`--kernel-size` 均可配置。
不要在相同输出目录混入不同配置；每个目录的 `config.json` 必须能解释全部结果。
默认种子 0、1 是开发种子。正式重复应锁定协议，使用另十个种子，例如 2–11；
不能把开发结果当作独立确认。

## 参数与空间卷积度量

实际更新的是实数像素卷积核和线性投影权重。前向为周期卷积、逐点平方、GAP、线性投影。
不使用 bias、BatchNorm、weight decay、动量、逐样本输入或输出归一化。

令 N=H×W，实 Fourier 余弦与正弦基的像素欧氏范数均为 1。
周期相关除以 sqrt(N/2)，则 B_ck=g_k(h_cos²+h_sin²)。
这使像素参数度量与稿件中的 Re h、Im h 度量一致，而非差一个未记录的频率重数。
`verify_spatial.py` 使用真实空间 `conv2d`；纹理默认用 FFT 实现同一周期相关，
参数仍然是像素卷积核。两种实现的输出和梯度均有独立比较。

有效 B 通过实际卷积网络对纯频率图像前向计算。不是直接更新 B 或 A 的替代网络。
只有明确命名的 direct、linear 对照更新功率特征上的有效参数。

## 数据与任务

32×32 实周期图像由 16 个独立复 Gaussian 频率对生成；方向为 x、y、对角、反对角，
每个方向各有低频和高频两组，每组两个频率。平坦功率谱 g=1，排除 DC、自共轭频率。
Gaussian 系数通过共轭频率实现实图像，q_k=|x_k^F|²/g_k−1，Cov(q)=I。

每个种子固定 32,768 个无标签训练样本。经验目标以 batch=256、有放回抽样训练；
同一随机种子的不同配比共享初始化、数据、批次索引与增强噪声序列的公共前缀。
两个视图在原始 Gaussian 系数上加入独立噪声，噪声功率 eta=0.2。
图像保持浮点数，不逐图归一化、不裁剪到显示区间。

全部十个任务在运行前定义：两个单坐标标签和八个标准化频带能量和。
标签信号方差均为 1，当前不加标签噪声。这些是人工可控的纹理属性，不能称为自然类别语义。
预训练不读取这些标签。

主模型 C=8、K=1、d=16；全支撑 32×32 卷积。
代码支持 K=2 和局部核；当前多频率实现检查另覆盖 C=4、K=2、11×11 核，
但实现检查不等同于这些设置的训练结论。

## 目标、学习率与停止

总体训练直接使用已知矩的完整矩阵 GUPA（epsilon=1），保留 logdet 和 trace 两项。
经验训练使用实际图像两个视图的输出，分别中心化求无偏样本协方差后平均，
alignment 为半个视图差平方的均值。样本矩代入非线性 GUPA 的梯度不必无偏；
总体与经验训练独立报告，不宣称它们逐步相同。

纹理固定初始 clean 总输出能量 0.001。配比 r=eta_W/eta_h 取 0.1、1、10，
eta_h=2*lr/(1+r)，eta_W=2*lr*r/(1+r)，固定两层学习率之和。
原始初始化 lr=0.001；额外的无标签校准初始化 lr=0.01。

`--balance-init` 在固定 A0 后计算两组参数 Jacobian 的平方范数（Gram 迹），
选择 c^6=trace(M_h)/trace(M_W)，再作 h→ch、W→c^−2 W。
该规则不使用任何任务标签，也不根据训练结果选择 c；它使两组贡献初始等迹。
它改变了初始化分布，故必须与原始随机初始化分开报告。每次的 c 保存在 result.json。

最多 12,000 次更新，首次达到 log(V/V0)≥3 或 view energy≥1 时停止；
没有主动重标定或约束投影。若跨过能量上限，末端记录仅用于说明越界，不纳入约束内结论。
固定截面比较必须在各条件共同可达范围内。图中横轴是真实测量的 log 能量，
汇总表在 ell=2 处线性插值，不冒充精确事件定位或定理截面。

## 评估与干预

投影和 encoder 分别计算已知分布下无正则总体风险，SVD 行空间投影给出保留语义，
记录数值秩和最小奇异值。OLS 使用另行独立抽取的 8,192 个标签训练样本、
32,768 个测试样本，标签子集为 256/1,024/8,192，拟合截距，不加 ridge。
OLS 在等价 Aq 上计算以节省重复卷积；该前向等价已用真实图像独立核验。

干预使用初始化和首次 ell≥1 的 checkpoint。在运行分叉前计算并保存
kernel_coefficient、head_coefficient 和各配比的总体风险方向预测。
除原始三个配比外，按固定规则加入每个正临界配比的一半和两倍。
全部配比依据当前向量场选择，不查看分叉后的风险再选择。
每个配比作 dt、dt/2、dt/4 三种单步分叉。使用原 checkpoint 的 float64 CPU 精确提升副本，
以免 float32 下风险接近 1 时的舍入淹没短时变化。

即便原训练为经验 SGD，这里的分叉也明确使用总体向量场。
这是已知任务总体风险下的 oracle 机制核验，不是无需标签的停止或调参算法；
单步成功不意味着长期干预有效。预测文件先于分叉结果写出。
`recheck_interventions.py DIRECTORY` 可从已保存 checkpoint 重算，不改动预训练轨迹。

## 对照与当前解释边界

`texture_controls.py` 匹配同一 A0，比对：真实平方网络、直接优化 A、普通两层线性因子化、
冻结投影、冻结前层。线性模型输入为同一功率 q，不是原始像素。
采用控制 log 能量增量的正物理步长 GD，到达 ell=1、2、3；与固定步长纹理训练分开报告。
径向总体目标下，直接优化 A 的方向恒定，代码对此作数值断言。

原始初始化 pilot 中，普通线性因子化产生与平方模型接近的任务风险变化。
因此目前不能主张现象为平方网络独有。冻结前层也可能出现损害，必须保留这个结果。
模型对方向的选择具有任务依赖性，但两个种子不足以支持普遍性、稳定性或高概率结论。

## 输出与下一阶段

- `results/spatial/`：梯度/尺度检查、真实参数 GD 截面、步长减半误差。
- `results/textures_*_pilot/`：配置、每条轨迹 CSV、总体风险、OLS、干预预测与结果、checkpoint。
- `results/texture_controls*/`：匹配映射的各类对照及步长减半。
- `results/figures/`：可独立查看的 PNG 与固定截面汇总 JSON。
- `.pt` checkpoint 保存在本地，但由 `.gitignore` 排除；文本输出可审阅。

尚未完成：锁定协议后的十种子确认、K=2/局部核系统训练、长期干预、自然图像外推。
自然图像阶段拟用灰度 CIFAR-10、C=64、11×11 核、K=8/16，从头无标签训练，
分别评估 encoder 和 projection；不能套用二维 c*。
尚未连接服务器，也未启动自然图像下载或训练。

后续进展见 [SUBSPACE_MECHANISM.md](SUBSPACE_MECHANISM.md)：已完成锁定协议后的十种子总体确认、
K=2/局部核与真实经验 SGD 的 1/8/64 步分叉。本文件上面的“尚未完成”记录为第一批 pilot 时的状态；
完整有限数据预训练确认、自然图像与更长期干预仍未完成。
