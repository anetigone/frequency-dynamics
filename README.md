# 平方表示中的任务子空间运动与风险回升

[阅读论文 PDF](paper/output/pdf/main.pdf)：正文 8 页、参考文献 1 页、附录 23 页。暂用已下载的 CVPR 2026 模板，目标年度要求发布后再调整。

核心结果研究实际平方滤波器与头部联合梯度流：编码器在保持秩的同时发生外向运动，使预先指定的一族任务先改善、后超过初始总体无正则线性风险。显式六坐标初始化邻域保留该现象；这是局部构造与稳定性保证，不是随机初始化典型性结论。受控纹理检验早期几何预测及其边界，自然图像结果保留为边界证据。

- [论文源码与编译说明](paper/README.md)：当前单 PDF 入口、章节与附录组织。
- [开发记录](paper/notes/)：理论推导、文献对照、审阅和结果分析；包含此前散落在根目录的记录。
- [实验导航](experiments/README.md)：代码、协议和既有结果。实验数据和结果保留，不随论文版面调整删除。
- [核心定理复核](paper/notes/CORE_THEOREM_AUDIT_20261021.md)与[邻近文献对照](paper/notes/THEOREM_LITERATURE_COMPARISON_20261021.md)：假设、常数及主张范围。

本地 Python 使用 `H:/anaconda3/envs/py312/python.exe`。历史 projection 证书与早期主线保存在 [THEORY_DRAFT](paper/notes/THEORY_DRAFT.md) 和附录中，不与当前 encoder 主定理混作同一结果。
