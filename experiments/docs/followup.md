# 自然图像确认与反事实诊断协议

本文是方法和复现命令说明，不是待办清单。当前进展、已完成结果和下一步统一见 [STATUS.md](../../STATUS.md)；本机下载结果的实际位置也在那里说明。

确认协议使用五个 CIFAR 种子 40–44，跑到 2000 个样本曝光等价 epoch。原 30–32 是探索种子，新五个独立报告，不混成八个确认种子。五个种子只能给出较宽的发生率区间；本轮首先检验预先固定任务/窗口是否能重复，不追求精确估计发生率。

全部沿用官方 CIFAR train 50k/test 10k 及原网络、增强、SGD 和初始化设置。新结果写入独立目录，保留原 5000-epoch 实验。原 `.pt` checkpoint 留在服务器，不需要下载到本机。本机只运行无数据的 plan 和小型合成 smoke。

## 同步与运行

需要在服务器复现时，同步 `experiments/code/empirical/` 的现有源码，以及 `natural_moment_experiment.py` 和 `experiments/run_mechanism_followup_server.sh`。继续保留已有 `natural_moment_eval.py`、`summarize_natural_moments.py` 和原工程的依赖目录；无需下载新数据。

在服务器项目根目录运行：

```bash
# 只打印方案，不读数据或训练。
bash experiments/run_mechanism_followup_server.sh plan

# 全部：先已有 checkpoint 的 head 诊断，再配对干预，再五个新种子。
bash experiments/run_mechanism_followup_server.sh all
```

可独立运行，重新执行训练阶段会从各自的 latest.pt 续跑：

```bash
bash experiments/run_mechanism_followup_server.sh heads
bash experiments/run_mechanism_followup_server.sh metric
bash experiments/run_mechanism_followup_server.sh confirm
bash experiments/run_mechanism_followup_server.sh summary
```

原结果默认来自 `experiments/results/empirical/natural_moments_cifar10`。如服务器位置不同，设置 `SOURCE_ROOT`；`CIFAR_ROOT` 沿用原数据缓存位置：

```bash
PYTHON_BIN=/path/to/python \
CIFAR_ROOT=/path/to/existing/cifar-cache \
SOURCE_ROOT=/path/to/natural_moments_cifar10 \
bash experiments/run_mechanism_followup_server.sh all
```

`DEVICE` 默认 `cuda`，`THREADS` 默认 1，`FEATURE_BATCH` 默认 256。`RESULT_ROOT` 可指定新的结果根目录；更改它时若源实验不在新目录下，应同时设置 `SOURCE_ROOT`。此脚本不会下载 CIFAR。

## 三组输出与主要判据

- `natural_confirmation_cifar10`：五个新种子，只在 0/200/500/750/2000 epoch 提取特征并评估，跳过 kNN。`confirmation_summary/findings.md` 是主要报告；固定 `antidiagonal_high_energy`，分别检查 200→500 和 750→2000。每次都要求同一种子、同一窗口的训练与测试改善、晚期超过初始化四项严格大于 0.001。`summary/` 的全任务轨迹搜索为次要描述。缺失保留为未知；不补抽种子，不按新结果改窗口。
- `natural_metric_cifar10`：原三个种子的 200、750 epoch 各分出普通 SGD 与 pullback 两臂，分别续到 500、2000 epoch。原函数、参数数目、head 更新和未来图像/增强随机流保持配对；预条件分支只改变 front 更新度量，并匹配起点输出 mobility trace。共同进度固定为审计目标相对 source 下降 0.1/0.25/0.5/1.0，保存预定审计网格上的首次穿越；检查实际目标值及 overshoot，不能称为精确等目标比较。未穿越的臂保留缺失。查看根目录 `findings.md`、`paired_differences.csv` 和 `common_progress_pairs.csv`，以区分路径变化和纯速度变化。六个 source 来自三个独立种子，不能当六个独立重复。这里干预的是 rank-one 因子切空间度量，不是完整自然矩空间的白化。
- `natural_head_counterfactual_cifar10`：用已有 3000/5000 epoch checkpoint 的 encoder/head 做 2×2 交叉，每次重新拟合 OLS；加上同八维预算的训练集最优 reduced-rank head。监督最优 head 只是诊断参照，测试集不参与拟合。新增预训练预算为零。

按旧吞吐 0.01110 秒/更新，新五种子训练约 6.02 GPU 小时；度量干预两个配对臂合计约 5.6 小时。总计约 11.6 个旧吞吐 GPU 小时，另加特征评估、保存、校准和新优化器开销。并行任务的实际墙钟时间及服务器当前吞吐会不同。

## 下载结果

下载三个新目录中的 JSON、CSV 和 Markdown 即可，checkpoint 保留服务器。可在项目根目录打包：

```bash
find experiments/results/empirical/natural_confirmation_cifar10 \
     experiments/results/empirical/natural_metric_cifar10 \
     experiments/results/empirical/natural_head_counterfactual_cifar10 \
     -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' \) -print0 \
  | tar --null -T - -czf mechanism_followup_results.tar.gz
```

若修改了 `RESULT_ROOT`，相应替换打包路径。原三种子 5000-epoch 结果及论文文件无需覆盖。
