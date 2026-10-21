# CIFAR 任务风险方向检验

2026-10-21。只检验现有 checkpoint 上的局部两层更新竞争，不继续搜索最佳分类配置。独立入口 `cifar_risk_direction`；不修改旧实验、数据或结果，不下载数据，不计算哈希。默认 `plan` 不访问数据/checkpoint/GPU。

## 固定问题与预算

- 来源：capacity baseline、r=1，开发种子 0/1/12，预设 step 0/1000/4000，全部保留，不按结果挑状态。
- 每状态三组未来随机流。第一步用四个 batch=256 的平均无监督梯度，预测和所有分支使用完全相同的首步梯度；随后为单 minibatch SGD。无动量、无权重衰减。
- 各分支同一个初态、相同批次和实际增强流；不同状态也复用相同 seed/repeat 的流。数据和增强在 CPU float32，模型、目标和风险计算 float64，以降低小差值的数值误差。不是原 CUDA float32 训练的逐位续跑。
- 主终点：projection 的 validation one-hot MSE，相对该来源状态的变化。次终点：accuracy。encoder 不另开终点搜索。
- 观察短程第 1/5/20 更新。共有 27 个状态×随机流作业，每个 5 或 7 个分支。probe 使用完整既有 45k/5k 数据，所以短训练也需要多次全量推断。
- 沿用来源 CIFAR 划分与灰度归一化；不新增划分，不读取测试批次。标签用于训练集 OLS、验证风险导数和临界配比，这是有标签的探索性机制诊断，不是无标签控制器或独立泛化确认。

## 风险导数和分支

给定 clean projection Z，训练集拟合带截距的无正则 OLS，在验证集计算逐元素 one-hot MSE。使用 float64 QR 解，导数通过训练均值/尺度、QR 和 decoder 解传播，包含 decoder 随表示重拟合的变化。不冻结 decoder，不把训练风险包络公式用于验证风险。数值秩不足时明确报错，不用 ridge 偷换风险定义。

设来源 mobility 为 m，基础 lr 为 η，平均训练梯度为 gh/gW。记录：

```
A = <∇h R, -η m gh>
B = <∇W R, -η m gW>
预测首步 ΔR(r,speed) = speed * 2/(1+r) * (A+r B)
```

风险的特征矩阵及 QR 在 CPU 上求导，再按图像块回传特征伴随梯度，避免同时保留全数据卷积图。不重校准来源 mobility。

两参数块分别沿 `-η m g` 用 0.1、0.05 两档中心有限差分核验。默认通过条件为绝对误差 ≤ 1e-10 + 0.02×|解析导数|。所有原值和误差都输出。这只是导数实现检查，不能当成机制发现。

始终运行 r=1、整体速度 0.5/2、r=0.1/10 五个固定分支。如果 A、B 异号，保存正阈值 r*=-A/B；只有有限差分通过、两块绝对贡献均大于 1e-10，且 r*/2 与 2r* 都在预设 [0.001,1000] 内，才增加上下阈值分支。不截断阈值，不静默删除失败或无阈值状态。1e-10 是数值信号底，不是实际效应大小的成功门槛。

每状态/流先保存 `prediction.json`，再产生任何分支结果。所有步的 decoder 都在训练集重新拟合。第 1 步对照真正的一阶预测；第 5/20 步的 `step × predicted_first_step_delta` 只叫初始切线外推，后续梯度已经变化，不能称为沿程积分预测。输出实际跨阈值符号反转及双方是否符合预测，并列报告绝对效应量和 accuracy。

## 服务器运行

从仓库根目录，保持现有 checkpoint 和数据路径：

```bash
python -B experiments/run.py cifar_risk_direction plan
python -B experiments/run.py cifar_risk_direction run --device cuda
```

默认来源 `experiments/results/cifar/capacity_development`，输出 `experiments/results/cifar/risk_direction_development`。主过程完成后自动汇总；中断后同命令可跳过已完成状态/流，未完成作业从来源状态重跑。没有中途优化器恢复。

数据不在仓库根时加 `--data-root /path/to/cifar-10-batches-py`。来源目录不同时加 `--source /path/to/capacity_development`。开始前读取全部所需来源元数据并检查 checkpoint 文件存在；本地目前只有 capacity summary，需要在拥有原 checkpoint 的服务器执行。

旧 local 格式也支持，但这些种子已经看过结果，仍为开发诊断：

```bash
python -B experiments/run.py cifar_risk_direction run \
  --source-kind local --source experiments/results/cifar/cifar_local_five \
  --seeds 2 3 4 --output experiments/results/cifar/risk_direction_local
```

若只做程序试跑，可显式用新目录和小预算；它不替代完整固定方案：

```bash
python -B experiments/run.py cifar_risk_direction run \
  --seeds 0 --source-steps 1000 --repeats 1 --horizons 1 \
  --output experiments/results/cifar/risk_direction_smoke
```

回传整个输出目录后，仅汇总无需原数据/checkpoint：

```bash
python -B experiments/run.py cifar_risk_direction summarize
```

## 输出与收尾判断

- `protocol.json`：本次参数、精度、首步平均梯度和标签用途；只比较直接实验设置，不校验源码指纹。
- 每个 `seed_*_step_*_repeat_*`：`prediction.json` 保存 A/B、阈值、未入选原因、有限差分和更新相对参数范数；`result.json` 保存全部分支终点、同状态基准差、切线误差、符号预测与阈值反转。不复制历史 checkpoint。
- `summary.json`：每种子的阈值覆盖率/有限差分通过率；逐来源步、分支和时长先平均同种子内随机流，再报告种子间均值与样本标准差。保留各单元的种子、流数和缺失覆盖；未完成运行明确标记 incomplete。条件阈值分支的样本数可能小于固定分支，不做无条件优劣比较。
- `findings.md`：MSE 变化简表。完整符号/accuracy/覆盖数据在 JSON 内，不自动宣布成功。

判断顺序：导数是否可靠 → 正阈值覆盖多少状态/流 → 跨阈值是否产生预测方向 → 效应量多大 → 5/20 步是否仍有作用。不能把所有 checkpoint、分支或随机流当作独立种子，也不能只报成功状态。若大多同号、或只有数值上极小的一步效应，收尾当前配置，不继续扩大扫描。是否具有实际任务意义需结合绝对 MSE、accuracy 和跨种子结果，不能以符号正确率单独判断。

本地检查（仅合成图像，无下载或真实训练）：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/run.py check_cifar_risk_direction
```

检查覆盖重拟合验证风险导数与有限差分、QR/SVD OLS 等价、错误的固定 decoder 导数区分、秩不足、分块/完整反传一致、共享梯度流、临界配比各分支条件、完整短分叉、独立首步 SGD 重算及不完整汇总。CPU 检查已通过；未执行真实 CIFAR 完整实验，也未验证服务器 CUDA 的速度。
