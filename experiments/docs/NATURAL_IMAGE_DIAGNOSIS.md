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
