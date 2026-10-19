# 压缩投影表示中的功率坐标竞争

更新：2026-10-19。当前主稿为 **[THEORY_DRAFT.md](THEORY_DRAFT.md)**。

本稿选择研究保留用于下游的一维 projection 输出。二维 encoder 输出是不同的评测对象：只要其矩阵可逆，最优无正则总体线性 probe 风险恒为噪声底。因此现有定理不能解释丢弃 projection head 后的 encoder 退化。

v0.3 保留原定理 1–4，新增 encoder 风险与有限标签 OLS 的可逆变换不变性、层间尺度与有效学习率比例的精确等价，以及八维相对 ±1% 参数盒内初始风险导数属于 [-0.16,-0.08] 的有理数区间证书。阈值 47/96 是具体构造的相对学习率阈值。

新增轨迹扫描覆盖 21 个尺度点、256 个参数盒顶点、64 个内部点与三个增加坐标/通道的配置；这些数值证据保留常数级回升，但不构成连续参数区域内长轨迹的严格证明。局部 U 形定理仍只有存在性宽度与 O(e²) 风险间隔。

- [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)：本轮审稿意见的处理与未完成项。
- [representation_checks.py](representation_checks.py)：新增评测位置、学习率等价、精确初始参数盒证书与稳健性扫描；运行 `python representation_checks.py`。
- [representation_results.json](representation_results.json)：新增核验的实际输出。
- [mechanism_checks.py](mechanism_checks.py)、[mechanism_results.json](mechanism_results.json)、[mechanism_profiles.csv](mechanism_profiles.csv)：当前主稿依赖的冻结头、精确曲率、参数梯度与总体 GF/GD 核验；运行 `python mechanism_checks.py`。
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md)：历史原文记录与尚未解除的新颖性阻断。

当前结果是全支撑卷积所实现的平方因子化功率坐标竞争。定理不使用频率的空间顺序或邻接关系，尚未证明真实视觉纹理或一般 encoder 的退化。本轮未执行新文献检索或图像实验，未确认新颖性。模型采用两个同分布增强视图的 GUPA 与输出能量上限，证明仅覆盖约束未激活阶段。

工作区采用固定文件名，只保留当前稿件与其复现所需的脚本、结果和来源记录。后续历史版本交由 Git 管理；Python 缓存与虚拟环境已列入 `.gitignore`。
