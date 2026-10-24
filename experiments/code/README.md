# 实验代码

研究状态与结论见[STATUS](../../STATUS.md)，运行说明见[实验入口](../README.md)。代码目录沿用现有路径，按下面的研究对象查找当前入口。

- 自然图像：当前训练、固定窗口确认、度量干预和 head 反事实都在 [empirical/](empirical/README.md)。数据划分与服务器命令见[自然图像协议](../docs/natural.md)和[机制补充实验](../docs/followup.md)。
- 合成纹理：[纹理回升入口](empirical/texture_rebound_sweep.py)复用像素核训练与总体风险计算；[population/](population/README.md)保留总体子空间机制，短期运动预测另见 [empirical/](empirical/README.md)。协议见[纹理实验](../docs/textures.md)和[运动预测](../docs/motion.md)。
- 理论：[theory/](theory/README.md)提供解析机制、区间算术与定理数值核验；[checks/](checks/README.md)提供图像恒等式、风险和导数的实现检查。
- 共享依赖：[shared/](shared/README.md)保存共同图像目标与数据工具。[cifar/](cifar/README.md)、[pilot/](pilot/README.md)及部分较早经验脚本仍提供当前代码调用的模型、增强、初始化和几何函数。

从仓库根目录可以直接运行具体脚本，也可以使用 `python -B experiments/run.py --list` 查找入口。这个入口按现有源码文件自动发现，不要求登记新脚本。
