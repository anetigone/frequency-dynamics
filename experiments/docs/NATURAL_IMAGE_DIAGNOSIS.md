# CIFAR / DTD 已有结果诊断

入口：`experiments/code/empirical/diagnose_natural_images.py`。只读已有结果，输出到所给 suite 下的 `diagnosis/`，也可以用 `--output` 指定独立目录。原始 summary、result、evaluation、checkpoint 不改写。没有训练、下载或自定义哈希计算。

## 先运行已有记录分析

在服务器仓库根目录执行：

```bash
python -B experiments/run.py diagnose_natural_images experiments/results/empirical/dtd_pilot
python -B experiments/run.py diagnose_natural_images experiments/results/cifar/checkpoint_branches_development
python -B experiments/run.py diagnose_natural_images experiments/results/cifar/capacity_development
```

这一模式仅需 Python 标准库，不读取图片或导入 Torch。读取各运行目录下的 `result.json`、`evaluation.json`，兼容 branch 的 `rows` 和 capacity 的 `fixed_steps` 结构。有 `summary.json` 时另行展开其中保存的逐种子指标。本地只有 summary 也能运行，但不会补造训练误差、奇异值或 loss 分解。

主要输出：

- `findings.md`：逐运行最终 probe 指标、缺失记录和训练 MSE 包含关系异常。
- `probes.csv`：每种子/步数/表示的训练与验证 MSE、accuracy、相对首次记录变化、常数预测参照、设计矩阵条件数、特征协方差谱和有效秩。
- `audit_history.csv`：完整已保存审计轨迹、alignment / 两项 uniformity、能量增长倍数、协方差谱及主方向方差占比。旧记录未保存的分解字段留空，不能用 batch loss 假充固定审计 loss。
- `summary_values.csv`：原 summary 中可用的逐种子值。
- `diagnostic.json`：上述表格以及可选 checkpoint 结果的完整数据。

常数参照为均衡类别的 `(C-1)/C²`，与当前 CIFAR / DTD 划分相符。`1-MSE/MSE_constant` 只是描述性技能分数。encoder 训练 MSE 不应大于原 projection OLS 的训练 MSE；阈值 1e-9 只用于标出数值检查对象，不阻断结果汇总。验证 MSE 可以反向排序，不能据此声称 projection 创造了新信息。

## 可选：已有 checkpoint 的功能分析

需要 PyTorch、现有图片数据和 suite 的 `protocol.json`。只作推断、SVD 和闭式最小二乘，不运行 SGD。建议先对三个开发种子的 baseline、初末两个状态做分析：

```bash
python -B experiments/run.py diagnose_natural_images experiments/results/empirical/dtd_pilot \
  --checkpoints --branches baseline --seeds 0 1 12 --steps 0 2000 \
  --device cuda --feature-batch 128

python -B experiments/run.py diagnose_natural_images experiments/results/cifar/checkpoint_branches_development \
  --checkpoints --branches baseline --seeds 0 1 12 --steps 0 1000 \
  --device cuda --feature-batch 256
```

`--steps` 指 suite 内的固定 checkpoint 步数；分叉 suite 的 step 0 是分叉起点。参考状态始终为该运行自己的 step 0。可改成所有已保存固定步数；省略 `--steps` 即为全部。省略 `--branches` / `--seeds` 时分析所有运行，capacity 组使用完整 job id 筛选。GPU 只用于卷积推断，闭式拟合和 SVD 在 CPU double 上计算；CIFAR 全部运行可能较慢。

DTD 默认读取 `experiments/data/dtd_hf/prepared_gray_64.pt`，沿用官方 partition 1；不调用 prepare 或 download。CIFAR 默认从仓库根目录的 `cifar-10-batches-py` 读取五个训练批次，精确复用已有 45k/5k split seed 和灰度预处理，不打开测试批次。`--data-root` 可分别指定 prepared tensor 目录或 CIFAR 批次目录/父目录。数据规模、归一化与保存协议不一致时明确报错。不要对其他 CIFAR 划分的实验使用该 loader。

新增输出：

- `checkpoint_changes.csv`：每个 encoder/projection 相对自身 step 0 的共享标量拟合残差、仿射拟合残差、训练/验证样本空间投影距离、数值秩、最大主角度。仿射映射只在原训练图像上拟合，在原验证图像上评价，不新增划分。残差是平方误差除以当前特征相对训练均值的总平方偏差。
- `checkpoint_heads.csv`：同一个 encoder 的全 OLS、监督 rank-K 最小二乘、原 projection OLS 的训练和验证 MSE / accuracy。K 来自当前 checkpoint，不更新模型参数，不用验证标签拟合。

标量残差小支持共同放大解释；标量残差大但仿射残差小支持旧特征重混合解释；仿射残差明显且样本空间位移较大才支持超出可逆坐标变换的变化。它们是功能诊断，不提供自动因果判定。主角度只覆盖较小秩空间，秩变化时必须合看两侧秩与投影距离。小奇异值方向的数值敏感性也要结合 `probes.csv` 判断。

## 本地检查

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_diagnose_natural_images
```

检查覆盖共享缩放、非标量可逆重混合、独立特征空间、47 类均衡目标的精确 rank-8 训练 MSE 下界、两种已存评估结构，以及合成 checkpoint 的完整推断/闭式 head 分析。不下载数据、不训练真实数据。

## 下一步：定位第一主方向

新增独立入口 `diagnose_dominant_direction.py`。使用同一份已准备数据和 checkpoint，默认分析 baseline、种子 0/1/12、初始和最后固定 checkpoint。输出到 `SUITE/dominant_direction/`，不改写前一轮 `diagnosis/`。

服务器执行：

```bash
python -B experiments/run.py diagnose_dominant_direction experiments/results/empirical/dtd_pilot \
  --steps 0 2000 --device cuda --feature-batch 128

# 如需对照 CIFAR，分叉 suite 的 step 0 是分叉起点
python -B experiments/run.py diagnose_dominant_direction experiments/results/cifar/checkpoint_branches_development \
  --steps 0 1000 --device cuda --feature-batch 128
```

脚本从训练集 clean 特征拟合 PCA，不先逐列标准化。encoder / projection 各自分析；PC1 的符号由绝对值最大 loading 的符号确定，不使用图像统计或类别标签选方向。没有新的数据划分或训练。只计算下列预先固定的诊断：

- `nuisance_correlations.csv`：PC1 分数与原灰度像素均值、均值平方、RMS、均方、标准差、方差、水平/竖直相邻像素差的平均平方能量，以及网络标准化输入的均值平方/均方的关系。原灰度从保存的训练集归一化统计还原，不逐图标准化。逐项输出 train/validation Pearson；另在训练集分别拟合单指标和全部指标的仿射预测，在验证集报告相对训练目标均值的预测 R²。全部指标保留，不按验证效果筛选。强相关本身不证明该统计与任务无关。
- `component_probes.csv`：full、PC1、其正交剩余空间、常数预测四组的训练/验证 OLS accuracy 与 one-hot MSE。PCA basis 固定自训练集；剩余部分直接用其余保留 PC 的坐标，避免减去 PC1 后引入数值虚假方向。小方差方向是否有类别信号，以及这些信号能否泛化，应同时看。
- `view_consistency.csv`：沿用该数据集原增强函数，在所有已有训练/验证图像上各生成两个视图。输出 full / PC1 / remainder 的 `0.5*mean(||z1-z2||²)`、除以同 checkpoint 训练集 clean 总方差后的误差、逐 PC 训练方差白化后的平均误差，以及逐 PC 两视图 Pearson 均值。全局输出缩放不改变方差归一化结果；raw alignment 可按 PC1 和 remainder 相加。它与旧固定 audit 的图像集合、随机流不同，不作逐值复现比较。
- `pc1_evolution.csv`：初始与当前 PC1 分数的相关性，以及当前特征在固定初始 PC1 loading 上的方差，避免只比较每个 checkpoint 自身的主方向占比。
- `diagnostic.json`：完整数据，另存训练均值、PC1 loading、保留数值秩与全部 clean PC 方差；`findings.md` 提供读表说明。

两视图使用原增强函数在 CPU 上生成，以 `--view-seed 20261023` 初始化；每个 checkpoint 重置生成器。因此在同样的样本顺序与 `--feature-batch` 下，checkpoint 和分支之间共用后续视图。它是一组描述性增强重复，不能当成独立多流确认。卷积可在 GPU 计算，PCA / OLS 使用 CPU double。无测试图片推断。

优先解读：第一方向与图像能量的关系是否强、该方向是否有独立验证的类别信号、剩余空间的类别信号是否随训练改变，以及消除尺度后的视图一致性是否改善。不能仅凭一个最高相关系数就宣布找到捷径或根因。

本地合成检查：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_dominant_direction
```

覆盖高方差无关方向与低方差语义方向的区分、训练集独占 PCA、归一化一致性的尺度不变性、分量 alignment 可加性、灰度统计还原、完整合成 checkpoint 流程，以及配对增强的可重复性。

## DC 响应消融：保持 checkpoint 不变，去掉滤波器空间均值

同一个脚本新增 `--dc-ablation`，原主方向诊断默认行为保留。先在 DTD 做现有三个开发种子的初末状态：

```bash
python -B experiments/run.py diagnose_dominant_direction experiments/results/empirical/dtd_pilot \
  --dc-ablation --steps 0 2000 --device cuda --feature-batch 128
```

输出默认改为 `dtd_pilot/dc_ablation/`。如果也做 CIFAR，使用同一选项及分叉对应步数：

```bash
python -B experiments/run.py diagnose_dominant_direction experiments/results/cifar/checkpoint_branches_development \
  --dc-ablation --steps 0 1000 --device cuda --feature-batch 128
```

每个 checkpoint 都重新推断两个变体：`original` 与 `zero_dc`。后者仅在内存中计算 `kernel - kernel.mean((-2,-1), keepdim=True)`，head 完全相同，不对 kernel 补偿缩放，不改变输入预处理、增强、数据划分或原文件。两个变体重置到相同 view seed，使用逐样本相同的增强；PCA 和监督 OLS 分别只在各自训练集特征上拟合。这是已有模型的功能消融，不是新的无监督训练结果。

此前的 PC1、类别分量、视图一致性、图像统计拟合表增加 `variant` 列，并新增：

- `ablation_comparisons.csv`：同种子、同 checkpoint 的 `zero_dc - original` accuracy / MSE / 常数技能分数差。优先读 `component=all`，accuracy 差为小数，乘 100 才是百分点；MSE 差为负表示消融后更好。初始与末期都要看，避免把本来就存在的特征差异写成训练收益。每个变体自身重新拟合 PCA，因此不同变体的 PC1 / remainder 不是固定的同一子空间。
- `kernel_statistics.csv`：滤波器空间和的最大绝对值、kernel 平方范数及相对原 kernel 的比例，检查 DC 去除量和范数变化。浮点下空间和允许有舍入残差，不引入哈希检查。
- `view_objective.csv`：projection 两视图的 alignment、两项 uniformity、总 GUPA loss 和协方差全谱。分别在原训练集和验证集图像上算完整集合的 sample covariance，不能与原 mini-batch / 固定小 audit 的 loss 作逐值比较。

loss 同时提供两种评价：`raw` 为实际输出；`original_train_clean_energy` 用一个仅由训练 clean 特征决定的标量，将两视图输出缩放至同 checkpoint 原模型的训练 clean 输出总方差，再计算同一公式。不更新任何参数。此标量只匹配总方差，不匹配协方差谱、方向、初始函数或训练动力学。若消融后 clean 方差为零，则保留 raw / 常数 probe，不伪造可匹配的输出尺度。

判断顺序：先检查 DC 平方与 PC1 的关系、主方向方差占比是否降低，再看 full 类别 probe 与尺度归一化一致性是否保留或改善，最后结合 raw / 同能量 loss 分解。即使消融有效，也不能仅凭一次末期消融证明从头 DC-free 训练一定更好。

本地合成检查：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_dc_response_ablation
```

覆盖 DC 投影的幂等性与范数分解、逐图常数偏移不变性、GUPA 与原训练公式一致、配对原模型结果不变、同能量评价，以及纯 DC 模型被完全消除时的零秩/常数 probe。
