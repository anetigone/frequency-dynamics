# 合成纹理配对实验协议

本文是方法和复现命令说明，不是待办清单。当前进展、已完成结果和下一步统一见 [STATUS.md](../../STATUS.md)；本机下载结果的实际位置也在那里说明。

服务器执行正式训练；本机仅做短合成 smoke。原 6 种子验证和历史结果保留，不与本轮新种子合并成独立重复。

入口为 `experiments/code/empirical/texture_rebound_sweep.py`，默认只打印计划，不创建目录或启动训练。默认新种子 100–149，共 50 个独立种子，每种子配对 square、linear、frozen_front，r=1，共 150 条轨迹。沿用原配置：32×32 全支撑像素滤波器、d=16、C=8、K=1、float64 CPU、32,768 个训练样本、batch=256、SGD lr=0.002、增强功率 0.2、4,000 步。每 25 步记录，固定比较点仍为 1,000 和 4,000。每进程限制一个 Torch CPU 线程，不争用自然图像训练的 GPU。

初始化仍为已有高斯抽样加无标签尺度校准：初始 A 平方能量 0.001，Jacobian Gram 迹平衡与 mobility 归一化。校准不读取任务标签、任务向量或任务风险；它读取参数与无标签二阶统计。校准后的分布不能写成未经修改的普通 iid Gaussian 初始化。50 个新种子的发生率只针对这一明确分布和训练设置。

服务器仓库根目录命令（`python` 指服务器已有实验环境）：

```bash
python -B experiments/code/empirical/texture_rebound_sweep.py
python -B experiments/code/empirical/texture_rebound_sweep.py --stage all
```

拆开训练和汇总：

```bash
python -B experiments/code/empirical/texture_rebound_sweep.py --stage train
python -B experiments/code/empirical/texture_rebound_sweep.py --stage summary
```

默认输出 `experiments/results/empirical/texture_rebound_50_seeds`。同目录已有结果逐条保留，完成、失败和中断结果都不自动覆盖或重跑；已有参数与当前参数不同则要求换输出目录。中断轨迹不能靠再次执行自动续跑；如需重跑技术故障，使用新目录并在报告里说明。不同种子范围可用 `--seeds 100 101 ...`，不同正式批次须使用不同 `--output`。

## 三个并列指标

所有任务沿用既有十个固定方向，不根据新曲线挑任务。encoder/projection 分别报告，阈值 δ=0.001，严格大于才计事件。

1. **采样轨迹上的完整回升**：`max_{0<s<t<=T} min(R(0)-R(s), R(t)-R(0)) > δ`。早期和后期时点从整条已保存轨迹定义，且严格按时间排序；两段都必须超过初始化的相应阈值。这不依赖固定的两个比较时刻，但仍依赖事先固定的观测网格和终止预算，不是连续时间极值的无偏估计。
2. **普通 drawup**：`max_{0<=s<t<=T}(R(t)-R(s)) > δ`，同时保留连续幅度。它允许末态仍优于初始化，不能与完整回升混称。
3. **原固定时点判据**：`R(0)-R(1000)>δ` 且 `R(4000)-R(0)>δ`，继续保留，便于与原 6 种子结果比较。

普通 drawup 与完整回升的最优时刻均仅作这一固定采样统计量的定义，不将被选择的风险最低点称为无偏证据。合成风险是已知分布下精确的最优无正则总体线性风险，没有标签估计噪声；训练仍是有限数据 SGD。

## 报告与缺失处理

`encoder_rebound_summary/` 原样使用旧汇总入口，保留固定比较点、完整曲线、损失、几何、初始配对和 SVD 截断敏感性。新增 `trajectory_summary/`：

- `trajectory_events.csv`：每个种子/模型/表示/固定任务的三个指标、幅度、所选早晚时刻及完成状态。
- `task_frequencies.csv`：每个固定任务分别列出事件数、完整种子数、计划种子数、排除种子列表及点态 Wilson 95% 描述区间。
- `coverage.csv`：全部计划运行的状态、缺失记录点、故障信息；不默默改变计划分母。
- `rank_sensitivity_events.csv`：各 SVD 相对截断下重新计算轨迹完整回升，与默认判定对照。
- `seed_task_union.csv`：每个种子十任务的描述性并集，不替代逐任务发生率。
- `summary.json`、`findings.md`：参数、完整原值与解释。

独立重复单位为种子，不把十任务、多个 checkpoint 或两个表示当作额外样本。区间分母为同一完整记录网格、同一固定预算下成功完成的种子，计划分母和被排除种子同时列出。缺失/失败不被当成“没有回升”；部分轨迹发现的事件另行保留，不混入完整预算发生率。Wilson 区间是逐任务、条件于完成的描述区间，不校正数值故障可能导致的选择，也不声称跨十任务的同时覆盖。

## 本机验证

仅执行两个种子的 8 步合成 smoke，共 6 条短轨迹，输出到独立 checks 目录；不作为正式结果：

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/empirical/texture_rebound_sweep.py --stage all --seeds 100 101 --steps 8 --early-step 2 --late-step 8 --record-every 2 --output experiments/results/checks/texture_rebound_sweep_smoke
```

正式结果和完成状态见 STATUS；本协议不维护另一份运行进度。

Smoke 与手工风险序列检查先后顺序、完整回升与普通 drawup 的区别及空分母处理；`checks/texture_rebound_sweep_missing_smoke` 是由短轨迹构造的缺失/故障报告样例（1 complete、1 simulated failure、1 missing），仅核验缺失不被默认为无事件，不是新增训练结果。训练与 NumPy/Matplotlib 汇总在 `--stage all` 中使用两个独立进程，避免本机环境重复加载 OpenMP DLL。
