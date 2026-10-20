# 实验目录与运行

源码放在 `experiments/code/` 的 theory、shared、pilot、population、empirical、cifar 和 checks 子目录。
`experiments/run.py` 直接查找脚本名，不需要登记文件或同步清单。

```bash
python -B experiments/run.py --list
python -B experiments/run.py dtd_pilot plan
```

脚本也可直接运行：

```bash
python -B experiments/code/empirical/dtd_pilot.py plan
```

结果在 `experiments/results/` 按研究阶段存放。`result_path()` 直接查找已存在的同名结果，新实验使用明确的 `--output` 路径。
数据缓存放在 `experiments/data/`；历史开发结果在 `results/archive/development/`，历史源码在 `archive/`。

参数和结果文件用于记录实验内容，不要求源码、checkpoint 或结果的文件哈希一致。
`audit_layout` 只检查 Python 语法和文档链接；`audit_local_closeout` 检查已有运行是否完成。

工作纪律见仓库根目录的 `AGENTS.md`。
