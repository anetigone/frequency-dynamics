# 代码与结果

本目录集中存放正文与附录的复现代码和输出。下面的原始认证与机制代码只依赖 Python 标准库。从仓库根目录执行：

```bash
python -B experiments/verify.py
python -B experiments/check_arithmetic.py
python -B experiments/mechanism_checks.py
python -B experiments/representation_checks.py
```

验证器必须启用断言，请勿使用 `-O`。所有默认输出均写入本目录，不依赖运行时的工作目录。认证器从有理数初值重新计算证书，无需读取已有 JSON。

| 代码 | 输出 | 对应结果 |
|---|---|---|
| [verify.py](verify.py)、[interval_core.py](interval_core.py) | [certificate.json](certificate.json) | 主定理 T：±0.001% 连续参数盒上的固定风险间隔、全程轨迹包络及可达性 |
| [check_arithmetic.py](check_arithmetic.py) | 终端检查结果 | 独立 Fraction 算术外包检查，不能替代轨迹证明 |
| [mechanism_checks.py](mechanism_checks.py) | [mechanism_results.json](mechanism_results.json)、[mechanism_profiles.csv](mechanism_profiles.csv) | 冻结头对照、方向阈值、Taylor 曲率及 GF/GD 数值核验 |
| [representation_checks.py](representation_checks.py) | [representation_results.json](representation_results.json) | 学习率等价、±1% 初始导数证书、浮点扰动与附加维度扫描 |

计算机辅助证明见[正文附录 A](../THEORY_DRAFT.md)。±1% 整盒的长轨迹回升尚未认证；普通轨迹扫描与主定理 T 的区间证书分别报告。

新增真实空间卷积核验、可控纹理训练、干预与对照使用 PyTorch，绘图使用 NumPy/Matplotlib。
运行方法、数据定义、学习率与解释边界见 [IMAGE_EXPERIMENTS.md](IMAGE_EXPERIMENTS.md)。
第一批结果位于 `results/`；它们是数值试跑，不替代上面的区间认证。

后续[子空间机制确认](SUBSPACE_MECHANISM.md)已完成独立十种子、两套校准、三种配比、K=2/局部核及真实经验 SGD 分叉。
总体汇总见 [subspace_summary/findings.md](results/subspace_summary/findings.md)，分叉汇总见 [subspace_empirical_confirm/findings.md](results/subspace_empirical_confirm/findings.md)。
