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

## 本地合成检查

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_dtd_dc_free
```

检查包括独立重算实际投影梯度 mobility、初始能量、逐图亮度平移不变性、训练中零均值保持、改变标签不改变 SSL 参数轨迹、完成任务复用、不计算哈希，以及 evaluate / summarize / 已有记录诊断的兼容性。全部使用合成张量，不下载或训练真实数据。
