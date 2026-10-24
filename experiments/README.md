# 实验

先读根目录 [STATUS.md](../STATUS.md)，了解当前证据和下一步；本页只说明入口与运行方式。现有代码、结果目录和服务器 checkpoint 路径保持不变。

## 当前使用的入口

自然图像实验和纹理主实验的代码仍在 [code/empirical](code/empirical/README.md)。两个服务器脚本保留原名称与 stage 参数：

```bash
# 仅打印方案，不读取数据或训练。
bash experiments/run_supplemental_server.sh plan
bash experiments/run_mechanism_followup_server.sh plan
```

也可从项目根目录使用统一入口；它按文件名查找源码，不需要登记：

```bash
python -B experiments/run.py natural_moment_experiment plan
python -B experiments/run.py texture_rebound_sweep --stage plan
python -B experiments/run.py --list
```

`--list` 会列出既有分组中的历史脚本，不代表全部仍是待做实验。实际入口可直接运行，详细选项用 `--help` 查看。

## 协议与结果

- [协议导航](docs/README.md)：自然图像、固定窗口/head/度量诊断、纹理及机制方法。
- [当前证据](../STATUS.md#当前证据在哪里)：两批服务器下载结果的确切路径及各套实验的用途。
- [历史结果导航](results/README.md)：理论、pilot、总体机制、早期经验训练和容量诊断。旧目录保留，不自动合并新旧重复。

运行脚本的默认输出路径与本机下载容器不是同一概念。本机读下载结果时传实际 `--output`/结果路径；服务器复用旧实验时传其原 source/output。不要因为文档整理而改写协议中的来源路径或重新跑一轮。

## 代码分组与环境

[代码导航](code/README.md)说明当前入口和旧依赖。`theory` 是区间与机制核验；`shared` 是共享模型；`pilot`、`population`、`cifar` 及旧 empirical 实现保留其原研究上下文。历史源码快照在 [archive](archive/)，不是当前修改入口。

本机 Python 为 `H:/anaconda3/envs/py312/python.exe`。本机只做小规模合成验证与结果分析；正式训练在服务器，使用官方数据划分。数值训练/评估与绘图汇总分进程。维护规则见 [AGENTS.md](../AGENTS.md)。
