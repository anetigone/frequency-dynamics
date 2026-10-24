# DTD：从初始化保持零 DC 响应的训练对照

用于检验原 pilot 中的 DC 放大是否妨碍其他类别方向的学习。它不是末期 checkpoint 消融，也不是与原 pilot 初始函数相同的学习率干预。代码不预设去 DC 会改善性能。

## 服务器运行

沿用已有 DTD prepared tensor，不需要重新 download / prepare。在仓库根目录运行：

```bash
# 只预览三个任务，不读取图片或初始化 GPU
python -B experiments/run.py dtd_pilot plan --dc-free

python -B experiments/run.py dtd_pilot train --dc-free
python -B experiments/run.py dtd_pilot evaluate --dc-free
python -B experiments/run.py dtd_pilot summarize --dc-free
```

默认输出 `experiments/results/empirical/dtd_dc_free/`，与旧 `dtd_pilot/` 分开。三次命令都需保留 `--dc-free`；自定义 `--output`、步数或数据路径等参数时也在 train/evaluate 中保持一致。已完成任务复用；未完成任务从保存初态重跑，不作中途优化器续训。

本次同步需要包含四个实现文件：

- `experiments/code/empirical/dtd_pilot.py`
- `experiments/code/cifar/cifar_capacity.py`
- `experiments/code/cifar/cifar_checkpoint_branches.py`
- `experiments/code/cifar/summarize_checkpoint_branches.py`

## 固定配置

- 开发种子 0/1/12，只运行 baseline r=1，共三条轨迹。不增加配比网格，不能与 `--branching` 同用。
- 官方 DTD partition 1：1,880 train / 1,880 validation；不读取测试图像。
- 64×64 灰度输入、C64 / K8、11×11 valid convolution、平方、GAP 和线性 head，无 bias、BN、weight decay、动量、输出归一化。
- 数据、增强、batch=128、基础 lr=0.002、2,000 更新及固定 checkpoint 0/500/1000/1500/2000，均沿用原 pilot。
- 固定 audit 每十步测量 loss 分解、协方差全谱、clean energy，并额外记录 `kernel_max_abs_spatial_mean`。
- 沿用初始校准图像选择、Hutchinson seed、audit seed、训练 future_seed 规则。它们使用独立的局部生成器，校准不会消耗训练随机流。没有新增自定义哈希计算或依赖哈希的运行门槛。

## 约束与重新校准

仍从相同 seed 的高斯 kernel / head 抽样开始，但在能量和 mobility 校准之前，将每个 kernel 在空间维度减去自身均值，记正交投影为 P。

1. 在相同的无标签训练校准图片上，把 centered clean projection 总能量设为 0.01。
2. 前层 Jacobian 迹使用实际约束更新的 `||P VJP||²` 的 Hutchinson 估计；head 迹按原方法精确计算。不能使用旧模型的无约束前层迹。
3. 按原平方齐次缩放平衡两块迹，并重算 common mobility。该缩放在 DC-free 模型内部保留当前函数；它不使新模型函数等于原 pilot。
4. 每步使用普通 SGD：kernel 梯度先减去其空间均值，更新后再对 kernel 作同样投影以清除浮点漂移。head 采用原 SGD 规则。无裁剪或动态重标定。

每个初始状态保存约束类型、投影 VJP 的校准几何、能量、两块迹和 mobility。protocol 记录约束及更新方式；不允许把原无约束初始化当作 DC-free 初态接着运行。最终模型仍是普通 NaturalSquare 的 kernel / weight 张量，因此已有 evaluate 和 checkpoint 诊断脚本可直接读取。

## 结果怎么读

`summary.json` / `findings.md` 同时报告训练与验证指标，以及相对本次运行自身初始化的变化；MSE 为主要终点，accuracy 并列报告。summary 中旧格式的 `minus_fork` 为兼容保留，DC-free 实验应读取更明确的 `minus_initial` 字段。

本实验只有一个 r=1 条件，不声称完成跨分支配对检验；`cross_branch_pairing_applicable=false`，`paired_streams_and_initial_states_valid=null`。没有共同 loss 的分支比较。

先判断 DC-free 的类别训练/验证增量是否出现，再与原 pilot 的增量作背景对照。不同初始表示之间的终点差异不等于学习收益；三个开发种子不构成新的确认集。即使目标降低、谱更分散或增强一致性改善，也不能替代类别风险的变化。

已有诊断入口可继续使用：

```bash
python -B experiments/run.py diagnose_natural_images experiments/results/empirical/dtd_dc_free
python -B experiments/run.py diagnose_dominant_direction experiments/results/empirical/dtd_dc_free \
  --steps 0 2000 --device cuda --feature-batch 128
```

这两条命令在保存完整 result/evaluation/checkpoint 和 prepared 图片的服务器上执行。第一条输出 `dtd_dc_free/diagnosis/`，只读已有 JSON；第二条输出 `dtd_dc_free/dominant_direction/`，读取已有图片和 checkpoint。此处不要加 `--dc-ablation`：需要诊断实际 DC-free 训练状态，而非再次消融。

按以下顺序读：

1. **约束是否保持。** `diagnosis/audit_history.csv` 中的 `kernel_max_abs_spatial_mean` 覆盖原审计时点；`dominant_direction/kernel_statistics.csv` 的 `max_abs_spatial_mean` / `max_abs_spatial_sum` 从选中 checkpoint 张量独立计算。允许浮点舍入残差，均值与系数和相差 kernel 面积这个尺度。
2. **目标为何降低。** `diagnosis/audit_history.csv` 比较 alignment、两项 uniformity、clean energy、top1_variance_share 和完整 view 谱随时间的变化；`probes.csv` 看 clean 特征谱和相对初始化的训练/验证误差。一个方向放大与多个方向共同增长应分开解释。
3. **是否换了主导统计。** `dominant_direction/nuisance_correlations.csv` 比较初末的 DC 平方、方差、梯度能量等预设指标在验证集上的预测 R²。只看相关性不判定因果，也不要跨种子平均可能翻号的带符号 Pearson 来衡量关联强度。
4. **剩余方向有没有学好。** `component_probes.csv` 同看 all / pc1 / remainder 的训练、验证 MSE 与 accuracy；`view_consistency.csv` 同看 alignment_over_clean_train_variance 和 clean_whitened_alignment_per_dimension。若改善只出现在全表示的方差加权指标，不能声称所有方向都更好。

本地若仍只有 summary，第一条只能给已有聚合指标的 fallback，第二条无法完成 checkpoint 分析。回传服务器上两个输出目录的 `diagnostic.json` 即可进一步集中分析。

## 本地合成检查

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_dtd_dc_free
```

检查包括独立重算实际投影梯度 mobility、初始能量、逐图亮度平移不变性、训练中零均值保持、改变标签不改变 SSL 参数轨迹、完成任务复用、不计算哈希，以及 evaluate / summarize / 已有记录诊断的兼容性。全部使用合成张量，不下载或训练真实数据。
