# 论文稿

当前阅读入口为[合并 PDF](output/pdf/main.pdf)：正文 8 页、参考文献 1 页、附录 23 页，共 32 页。附录与正文共用文献表和内部交叉引用，物理页码连续。正文围绕编码器外向运动、解析回升族、六坐标独立扰动的定量稳定性，以及独立未来流上的经验支持；随机窗口保留为条件性延伸。未新增训练。

- [main.tex](main.tex)：唯一当前编译入口；正文在 [sections/submission.tex](sections/submission.tex)，附录顺序在 [sections/appendix_contents.tex](sections/appendix_contents.tex)。
- [notes/](notes/)：开发说明、理论工作记录、结果分析、文献对照和历史审阅；从工作区根目录移入的早期记录也在这里。
- [technical_report.tex](technical_report.tex)：保留的历史完整工作稿入口，不作为当前交付版本。
- [figures/](figures/) 与 [scripts/](scripts/)：正文及附录使用的图和绘图源码。

暂用用户下载的上一年 [CVPR 2026 author kit](../author-kit-CVPR2026-v1-latex/README.md)。本目录的 `cvpr.sty` 和 `ieeenat_fullname.bst` 与该模板文本相同，未修改模板的边距、字号和行距。正文增加的是已有机制解释、证明思路和实验分析。目标年度 author guide 发布后再调整要求；当前仍为匿名 DRAFT，没有设定注册编号或提交稿件。

从 `paper` 目录编译；首次编译前创建输出目录。无需先编译任何外部补充文档：

```powershell
New-Item -ItemType Directory -Force -Path output/pdf | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
bibtex output/pdf/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
```

核心核验与范围说明见[定理复核](notes/CORE_THEOREM_AUDIT_20261021.md)、[定理级文献对照](notes/THEOREM_LITERATURE_COMPARISON_20261021.md)。非对称结果固定增强 `eta=0.1`、两学习率为 1，初始化盒半径 `8e-6`；比较共同几何截面的首次到达，物理时刻可不同。正文保守改善/回升下界为 `0.001675/0.010905`，不是随机初始化典型性结论。

[固定随机窗口筛查](notes/STOCHASTIC_FEASIBILITY_20261021.md) 未得到当前保守包络的严格实例，已按退出条件停止扩展；[八流误差分解](notes/TEXTURE_MOTION_ERROR_DECOMPOSITION_20261021.md) 保留均值偏离与流间波动，不唯一归因晚期退化。详细历史来源见 [SOURCES](notes/SOURCES.md)。实验数据、结果、章节源码和检查脚本均保留。
