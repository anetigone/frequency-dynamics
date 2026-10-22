# 自然图像矩任务与长时间尺度补充实验

本轮只补代码、服务器命令和合成小验证；本机不下载真实数据、不启动正式训练。自然图像的补充目标有三项：在固定且有图像含义的二阶矩任务上测量 encoder/projection 可读性；在更长时间尺度比较类别 OLS 和对尺度敏感的 kNN；把类别任务的零变化与任务耦合界作定量比较。全部任务、种子、参数和采样时刻在正式运行前固定，不按曲线选择成功案例。

入口为 `experiments/code/empirical/natural_moment_experiment.py`，阶段为 `plan`、`train`、`evaluate`、`summarize`、`smoke`。不指定阶段时仅输出计划，不访问数据。统一服务器入口为 `experiments/run_supplemental_server.sh`，默认同样只输出计划。沿用服务器既有 Python 环境中的 PyTorch、torchvision、NumPy、Matplotlib；DTD 的准备文件还沿用原图像处理依赖。不增加额外包管理流程。

## 数据与预算

| 项目 | CIFAR-10 | DTD |
| --- | --- | --- |
| 划分 | 官方 train 50,000 / test 10,000 | 官方 partition 1 train 1,880 / validation 1,880 |
| 图像 | 32×32 灰度，训练集全局均值/标准差 | 已准备的 64×64 灰度，原 train-only 归一化 |
| 数据入口 | `--data-root` 下的 `cifar-10-batches-py/`，`download=False` | `--data-root` 下的 `prepared_gray_64.pt` |
| 测试集 | 固定 checkpoint 的测量；不用来选择任务、参数或停止点 | 不下载、不使用 DTD test |
| 新种子 | 30、31、32 | 30、31、32 |
| 总预算 | 5,000 exposure-equivalent epochs | 5,000 exposure-equivalent epochs，可先只跑到 200 |

每一步从官方训练集有放回均匀采样 256 张图像。这里的 epoch 定义为 `step * 256 / train_size`，是累计样本曝光量，不是逐遍无放回遍历数据。CIFAR 5,000 epoch 每种子为 976,563 次更新；DTD 为 36,719 次。新 CIFAR 的官方 50k/10k 结果不与历史 45k/5k 结果合并计算。

模型沿用 64 个 11×11 valid 卷积核、平方激活、全局均值池化、8 维线性 projection。使用既有数据增强和 GUPA 目标、SGD、基准学习率 0.002、块率比 1，无动量、weight decay、梯度裁剪或额外输出归一化。初始化沿用独立高斯参数、初始能量 0.01 及无标签 Jacobian-trace 尺度校准，校准不读取矩任务或类别标签。模型和卷积使用 float32；OLS、几何分解与 kNN 距离使用 float64。float64 拟合不能恢复 float32 特征计算中已经丢失的精度，因此一并保存奇异值与秩截断敏感性。

这套长时间实验用于同架构、同类目标下的时间尺度与读取方式比较。没有逐项复刻 Torralba–Weiss 的全部增强、优化器、初始化和 kNN 配置，不能称为该论文的严格复现。

## 固定矩任务与风险

任务均为与 encoder 相同的平均 valid 11×11 patch 二阶矩的固定线性泛函，不从训练轨迹提取方向：

- `patch_dc_energy`：patch 均值平方的空间平均。
- `patch_mean_square`：patch 像素平方均值的空间平均。
- `patch_contrast`：上述均方能量减去 DC 能量。
- 八个方向/频带能量：x、y、对角、反对角波向量，各取 low/high；每个方向使用固定正弦/余弦滤波对，两个响应平方的和再作空间平均。

频率固定为每 11 像素的 1 或 3 周期，方向名称表示波向量方向。滤波器去 DC、正交化、单位范数，正弦/余弦对能量不依赖相位基的选择。`task_definitions.json` 和源码共同给出任务定义，运行中不新增方向。

每个任务用完整官方训练集的均值和方差归一化。对 encoder 和 projection 分别拟合带截距、无正则的 affine OLS，在完整训练集报告经验最优风险，在官方 held-out 划分报告该训练 decoder 的均方误差。两者都是有限图像样本上的测量，均不是已知自然图像分布的精确总体风险。确定性标签不消除输入分布采样误差，也不保证有限训练集 decoder 在 held-out 上达到总体最优。

每个固定任务保留完整风险轨迹。采样轨迹上的完整回升定义为存在 `0 < t_early < t_late`，使 `R(0)-R(t_early)>0.001` 且 `R(t_late)-R(0)>0.001`；阈值单位为训练目标方差。同时报告最大 drawup（先前最低点至后续点的最大升幅），不要求后续点超过初始化。该轨迹指标依赖预定采样网格和观察时长，是描述性事件统计；它不依赖原来的固定 1,000/4,000 步比较点，也不是连续时间极值的无偏估计。发生率按每个固定任务分别以种子为独立单位，三种子结果仍只是小规模验证，不能把任务或 checkpoint 数当作样本量。

## 运行前局部预测与几何界

初始化时，在独立于正式训练随机流的 16 个无标签 batch 上平均 GUPA 梯度，并分成两个 8-batch half estimate；`prediction.pt` 在训练开始前保存。评估以解析卷积 JVP

`dF = spatial_mean(2 * conv(x,h) * conv(x,dh))`

及最优训练 OLS 风险的 envelope 导数，计算全部固定矩任务的初始风险斜率。它只预报沿估计平均更新方向的局部改善/恶化符号，不预报有限时间回升或转折点。两个 half 的符号不一致、幅度接近零，或初始设计矩阵不满列秩时，方向标为 `unresolved`。保留全部斜率和后续观测；首个记录点与局部斜率同号不等于验证了有限时间机制预测。秩截断敏感性另外逐点报告，应与小幅度事件一起阅读。

每个固定 checkpoint 还在完整训练样本的函数空间计算 encoder/projection 相对初始化的子空间位移、`a0=||Q0 T P0||_F` 和 `osc(T)`。同秩时报告 `2*a0*d + osc(T)*d²`，秩改变时不套用同秩界；同时报告界值/实际风险变化和界值是否小于总目标方差。类别目标还保留原 one-hot 单位的聚合结果，避免把逐类标准化风险与主文 one-hot 均方误差混淆。这是训练经验分布上的函数空间版本，不是 held-out 风险界，也不等同于完整 patch 矩字典上的总体界。若界过宽，就不能据此声称类别任务的零变化得到有力的定量预测。

类别诊断包括同一特征上的 OLS、raw Euclidean kNN、训练坐标标准化 kNN 和训练协方差白化 kNN。固定 `k=20`，全部官方训练样本为候选、全部 held-out 样本为查询；按 query block 计算精确距离，不新增子集划分，也不调 k。白化只保留数值非零协方差支撑。若 raw kNN 回落但白化 kNN/OLS 稳定，结论应限于尺度敏感读取；若 OLS 也回升，再分别检查 encoder/projection、秩和截断敏感性。

同时记录干净训练特征的第一主成分方差占比，以及以 PC1 为单变量读取各固定矩任务的训练 R² 和 held-out MSE。PC1 方向和预测器均只在训练图像上拟合；该读数检验占优方向与 DC、局部对比度或固定频带能量的关联，不用来回头修改任务。PC1 方差集中与子空间旋转需分别阅读，方差集中本身不推出最优线性可读性下降。

## 服务器运行

在仓库根目录同步新增源码后，先指定已有环境和数据位置。脚本不自动准备或下载数据；缺失数据会给出直接错误，原结果保留。`PYTHON_BIN` 是一个可执行文件路径，不是含多个 shell 参数的命令字符串。

```bash
export PYTHON_BIN=/path/to/existing/env/bin/python
export CIFAR_ROOT=/path/to/data_root
export DTD_ROOT=/path/to/dtd_hf
export RESULT_ROOT="$PWD/experiments/results/empirical"
export DEVICE=cuda
bash experiments/run_supplemental_server.sh plan
```

先做 200 epoch 的自然图像矩任务与类别读取测量：

```bash
bash experiments/run_supplemental_server.sh first
```

该阶段顺序训练 CIFAR/DTD 的全部三种子，各自评估到 200 epoch 并汇总。两套协议从一开始就固定总预算 5,000 epoch；`--until-epoch 200` 仅暂停，不把预算或采样网格改成 200。继续 CIFAR 长程与单独的可选 DTD 长程：

```bash
bash experiments/run_supplemental_server.sh long
bash experiments/run_supplemental_server.sh dtd-long
```

同一输出目录恢复 `latest.pt` 中的模型、步数和 minibatch/augmentation RNG。已存在的固定 checkpoint 和评估结果跳过；改变协议内参数时需使用独立输出目录。意外中断后重新执行相同 stage 即可从最近成功保存处继续。失败记录保留，不自动改学习率、替换种子或丢弃失败分母。

合成纹理 50 新种子（100–149，square/linear/frozen_front，原 4,000 步配置）独立启动：

```bash
bash experiments/run_supplemental_server.sh textures
```

纹理 stage 启动完整正式实验；本机只做单独小 smoke。自然图像的合成小验证无需数据：

```bash
bash experiments/run_supplemental_server.sh smoke
```

该 smoke 使用 60 张合成随机图像，检验几步训练、暂停续跑与不中断训练的一致性、矩任务评估和汇总，不读取真实图像。统一 runner 的 `summary` stage 仅用于三套正式结果都已存在后的重新汇总；也可对单套结果直接运行：

```bash
"$PYTHON_BIN" -B experiments/code/empirical/natural_moment_experiment.py summarize \
  --dataset cifar10 --output "$RESULT_ROOT/natural_moments_cifar10"
```

默认采样 epoch 为 0、0.1、0.25、0.5、1、2、3、5、8、10、15、20、30、40、50、60、80、100、125、150、175、200，随后 300、500、750、1000、1500、2000、3000、4000、5000。kNN 使用其中较稀疏的固定网格，减少完整候选搜索成本。每个 kNN 点也保存完整矩任务与 OLS。自定义网格应在新输出目录中运行，不能根据已有风险曲线更换点后仍作为原计划结果。

显存不足时可降低纯评估的 `FEATURE_BATCH` 或 `KNN_BATCH`；它们不改变任务或训练参数。不要以此改动正式 `--batch-size 256`。默认 DTD/CIFAR 均将输入图像放在选定 device，服务器需为完整输入及卷积中间量留出显存。

输出在每套 suite 的 `summary/`：`moment_risks.csv`、`class_readouts.csv`、`knn.csv`、`local_predictions.csv`、`geometry_bounds.csv`、`dominant_statistics.csv`、`trajectory_events.csv`、`task_frequencies.csv`、`rank_sensitivity.csv`、`rank_sensitivity_events.csv`、`coverage.csv`、`audit.csv`、`findings.md` 及曲线图。`geometry_and_dominance.png/pdf` 展示位移和占优方向统计。覆盖表保留缺失种子、失败和部分观察时长；不把较短轨迹与完整轨迹混成同一时长的发生率。纹理结果仍分开放在该 suite 的 `encoder_rebound_summary/`（原固定时点指标）和 `trajectory_summary/`（全轨迹指标），详见 `TEXTURE_REBOUND_50_SEEDS.md`。
