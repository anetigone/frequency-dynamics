# 科学实现检查

[STATUS](../../../STATUS.md) · [代码导航](../README.md)

这里保留用于核对模型恒等式、损失与梯度、线性风险、子空间运动及解析方向导数的小规模检查。当前自然图像实现的合成验证从 [check_natural_moments.py](check_natural_moments.py)开始；它使用随机图像与张量，检查固定矩任务、OLS、运动界和 kNN，不加载公开数据。

```powershell
& H:/anaconda3/envs/py312/python.exe -B experiments/code/checks/check_natural_moments.py
```

自然图像训练及两个 checkpoint 干预脚本也各自提供 `smoke` 阶段，用合成输入检查暂停续跑、配对随机流或低秩读取。入口见 [empirical/README.md](../empirical/README.md)。区间算术原语的独立有理数检查在 [check_arithmetic.py](../theory/check_arithmetic.py)，定理的区间积分核验在 [verify.py](../theory/verify.py)。

其他文件保留较早实验对应的实现检查，各自使用其声明的输入。按修改涉及的科学实现选择检查，不把全部历史脚本作为每次训练或汇总的统一前置流程。
