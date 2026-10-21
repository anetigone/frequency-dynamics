# 固定起点的纹理几何、任务和窗口预测

本轮代码只从已有完整经验训练 checkpoint 分叉，不重新预训练，不增加数据集、模型、任务划分或数据哈希。当前默认面向全支撑、d=16、C=8、K=1 的既有 checkpoint；局部核可显式传入对应 `--source`，但应另用输出目录报告。

## 固定状态与任务

| 用途 | 种子 | checkpoint 更新步 | 模型 |
|---|---|---|---|
| 开发/选设置 | 0、1 | 500、2000 | square、linear、frozen_front |
| 本轮设置之外的验证 | 2–11 | 500、2000、3500 | 同上 |

开发共 12 个状态，验证共 90 个状态。3500 是本轮开发未使用的 checkpoint；验证种子也不用于本轮调界。它们在旧实验中已有结果，不宣称是从未检查过的全新种子。默认只用原始 r=1、stream=0 的起点；每个状态分叉 r=0.1、1、10，更新率保持原始 mobility 和 lr=0.002。相同学习率和步数不等于相同真实参数移动量，因此另记录两块累计参数路径长度。

固定任务直接调用原有 `texture_tasks`：两个单坐标任务和八个频带/方向任务。汇总按 `predefined_coordinate` 与 `predefined_band` 分开。标签任务已知，任务风险为解析总体最优无正则线性风险；本协议不估计自然图像 OLS 泛化，也不是无标签控制器。

额外的 `source_maximum_competition` 属于单独的 `constructed_max_competition` 家族。它仅由起点校准后的两个投影切向量选择，检验竞争存在性，不计入频带任务覆盖率。K=1 时，写 Dg=u vgᵀ+vg uᵀ，取线段 [−vh,vW] 的最小范数点 c，任务 β=(u+c/||c||)/√2 最大化两个相反块贡献的最小裕量。c=0 或裕量低于数值阈值时明确报告“不可构造”，不删除该状态，也不用未来风险另挑任务。

## 校准和未来流分离

所有随机数都用显式 Generator，按 `(seed, checkpoint, role, repeat)` 分开命名空间：

1. **fit**：16 个起点批次，平均实际有限批次 GUPA 的有效梯度。没有把总体梯度或条件矩 plug-in 当成有限批次平均梯度。
2. **error**：另外 32 个独立起点批次，估计相对于 fit 均值的切向量和任务斜率误差。
3. **probe**：4 条独立、最长 8 步的短校准流，只在源状态的克隆上运行。重放冻结的 fit 批次，在这些局部状态测量参考场漂移；实际 probe 批次测量离散 SGD 更新相对于其切向量的余项。
4. **future**：8 条独立未来流，每条最长 64 步。每状态的 `predictions.json` 写完以后，才创建 future Generator 和生成未来批次。未来批次在所有配比之间配对，在同种子/同 checkpoint 的各模型之间也配对。

这些批次从同一个既有有限训练集有放回独立抽样；不要求图像索引互斥，也不另划分数据。随机流的分离不等于任务标签未知；任务 β 本来就是总体 oracle。

不从未来流估计区间宽度、不按未来结果重新选择任务、配比或窗口。未来步的块切向量仅作为已冻结预测的误差诊断，字段明确标注为事后诊断。默认记录 horizon=1、4、8、16、32、64；8 步以外仍输出外推区间和实测结果，但默认不算作有支持的有效经验窗口。

## 三层预测

### 几何

由实际参数 Jacobian Gram 得到每块有效更新，保留局部核约束。记录编码器法向速度、投影的前层/头部切向量，以及 H 步冻结切向量对未来投影位移的预测。

起点另用参数 autograd + JVP 独立核对分解，这只是微分计算检查。真正的几何外推指标是独立未来流上的累计法向运动误差、投影矩阵误差、方向余弦，以及未来块切向量相对于冻结起点切向量的偏离。不能把起点 JVP 一致算成预测成功。

### 任务

对每个起点、表示、配比和 horizon，同时输出：

- **位移基线**：已知单位任务但不使用其方向，界为 min(1,d)，d=||P₁−P₀||F/√2。
- **任务耦合幅度**：a₀=||(I−P₀)ββᵀP₀||F，界为 min(位移基线, 2a₀d+d²)。
- **方向预测**：冻结起点斜率 q₀=−βᵀD₀β，中心为 Hq₀。

前瞻版本中的 d 由起点切向量和经验几何误差半径得到，因此也是经验预测。另有一组明确标为 `posterior_*` 的对照，把真实未来位移代入精确有限位移不等式，并检查交叉项/对角项分解。这些事后对照用于解释“同一位移下任务变化不同”，不能当成前瞻证据。

汇总在同一状态/表示/配比/horizon 内，先按流平均，再比较预先指定任务的风险变化排序。只用位移的任务排序会全部并列，按 0.5 计分；方向斜率和任务耦合幅度的排序分别报告。构造任务只有一个，不与频带任务拼接计算排序。另报告小幅变化阈值 1e−4 的覆盖度和误差，不能只挑变化最大的任务画图。

### 时间窗与误差

对单步误差 ε、每步状态漂移系数 K 和单步离散余项 e，使用经验包络

\[
\Delta R(H)\in Hq_0\ \pm\left(H\varepsilon+\tfrac12H^2K+He\right).
\]

默认 ε 来自 error 批次绝对误差的 0.9 分位数乘 2；K、e 来自各 probe 流上的最大漂移/余项，再取跨 probe 流 0.9 分位数乘 2。几何矩阵误差也采用同样的形式。所有设置在 `prediction_settings.json` 中明确保存，不从未来误差拟合。

**这些分位数没有对所有未来状态或 SGD 流的统一概率保证。** 固定 fit 批次的漂移不等于真实条件平均梯度的统一 Lipschitz 界；源状态噪声也未证明沿未来路径一致。因此只称经验预测区间，不把 0.9 分位数写成 90% 同时置信保证。使用 Hε 而非声称独立后可缩成 √H，也不能把该经验误差变成严格上界。

区间跨零、低于数值阈值 1e−9、超出默认校准时间范围、或与可行风险变化范围完全不相交，均报告无有效经验方向窗口及具体原因。严格随机方向证书标记为不可用。唯一预先已知的严格对照是冻结编码器的风险恒为零变化；这与一般 SGD 的方向窗口分开报告，也不用于宣称一般理论验证成功。

未来每一步都检查累计变化和单步增量是否反向，两者分别记录。`windows.csv` 给出源状态预测的连续 horizon 前缀、所有抽取未来流同时落入区间的验证前缀，以及逐步增量未反向的前缀。后两者仍是抽取流上的描述，不能升级为其它未来流的保证。禁止跳过中间失败，只报告后面偶然再次同号的 horizon。

## 无窗口与失败状态

`protocol.json` 提前列出全部状态。缺失 checkpoint、读取/秩/数值问题、未完成的分叉都保留状态和原因。输出目录要求为空以保留历史结果，不计算或比较源码/文件哈希，也不建立脚本登记或迁移清单。

汇总的覆盖率分母是所有计划状态 × 家族任务数，不是成功输出数。区间宽度、误差和条件命中率另外给出可用条目数；缺失结果不伪装成零风险变化。坐标任务、八个频带任务、构造任务分开。统计单位为种子，checkpoint、任务、未来流不算独立重复。当前汇总提供描述性种子均值，不自动输出显著性或选择最好 checkpoint。

## 使用

在仓库根目录使用已配置好依赖的 `python`，统一通过 `experiments/run.py` 入口运行。下面是实际分叉与汇总命令；轻量检查不启动真实 checkpoint 分叉。

```bash
# 查看固定状态及文件存在情况；不加载 checkpoint，不生成训练数据。
python -B experiments/run.py texture_motion_prediction --phase development --dry-run

# 少量开发状态：可据此调整经验设置，调整时另用输出目录保留旧结果。
python -B experiments/run.py texture_motion_prediction --phase development --output experiments/results/empirical/task_motion_development
python -B experiments/run.py summarize_texture_motion experiments/results/empirical/task_motion_development

# 确定设置后固定使用该 JSON。验证阶段不自动调参，也不重写开发设置。
python -B experiments/run.py texture_motion_prediction --phase validation --settings experiments/results/empirical/task_motion_development/prediction_settings.json --output experiments/results/empirical/task_motion_validation
python -B experiments/run.py summarize_texture_motion experiments/results/empirical/task_motion_validation
```

默认来源按结果名查找：开发为 `full_sgd_development_v2`，验证为 `full_sgd_main`。兼容当前分组目录、历史开发归档目录和旧版直接位于 `experiments/results/` 下的同名目录；若找到多个同名结果，使用 `--source` 明确选择。服务器上的 checkpoint 放在其它位置时，也通过 `--source` 传入实际目录，不需要修改代码。整个来源目录缺失时，`--dry-run` 仍列出全部计划状态及 `checkpoint_exists=false`；实际运行记录 `missing_checkpoint`，汇总保留这些状态的分母。运行前应先确认 dry-run 中的来源与文件存在情况。

`summary.json`/`coverage.csv` 给全部状态的覆盖与条件误差；`state_metrics.csv` 保留各状态和任务家族；`geometry_by_state.csv` 保留所有状态的几何误差；`task_predictions.csv`/`task_outcomes.csv` 保留逐任务结果；`geometry_metrics.csv` 保留逐流几何；`windows.csv` 对比预测和验证前缀。源状态的原始预测与经验系数保存在每状态 `predictions.json`，不会由汇总程序重拟合。

轻量检查可单独执行，不读取真实 checkpoint：

```bash
python -B experiments/run.py texture_motion_prediction --self-check
python -B experiments/run.py summarize_texture_motion --self-check
```
