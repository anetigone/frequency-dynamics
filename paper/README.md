# 论文稿

当前阅读入口为[合并 PDF](output/pdf/main.pdf)：正文 8 页、参考文献 1 页、附录 30 页，共 39 页。附录与正文共用文献表和内部交叉引用，物理页码连续。正文围绕编码器外向运动、解析回升族、六坐标独立扰动的定量稳定性，以及固定任务的完整 encoder 回升。纹理主报告使用 50 个配对种子、三个模型的 150 条完整轨迹，同时报告采样路径和固定时点判据；square 与 linear 均有 encoder 完整回升，不作架构排他性结论。原六种子结果和步长检查保留于附录。CIFAR 三种子 5000 epoch 结果展示自然图像固定矩任务的完整 encoder 回升，并区分类别读取与 projection 选择差距。自然图像见证时刻为事后选择，训练经验最优风险与测试误差分别报告。短期预测细节移入附录；随机窗口保留为条件性延伸。

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

纹理主表逐项报告全部十任务的 50 种子结果：采样路径完整 encoder 回升为 square 6–13/50、linear 1–7/50；固定 1000/4000 时点为 square 1–9/50、linear 0–2/50；frozen encoder 均为零。附录 [texture_fifty_details.tex](sections/texture_fifty_details.tex) 补充点态 Wilson 区间、projection 与普通 drawup，三种口径分别报告。固定时点的 square 43 个相关任务事件中 39 个伴随两段总体目标下降，linear 为 3/3；不把这些计数用于采样路径见证的目标判断。全部 square 终点超出定理的 `Q=sV<1` 区间。

[早期六种子验证记录](notes/ENCODER_REBOUND_VALIDATION_20261022.md)保留十任务逐项计数、全部两段幅度、目标下降交叉表与针对性步长检查。早期 linear 零事件仅属于这一六种子批次，不概括新 50 种子结果；两批不合并为新的独立重复。

[固定随机窗口筛查](notes/STOCHASTIC_FEASIBILITY_20261021.md) 未得到当前保守包络的严格实例，已按退出条件停止扩展；[八流误差分解](notes/TEXTURE_MOTION_ERROR_DECOMPOSITION_20261021.md) 保留均值偏离与流间波动，不唯一归因晚期退化。详细历史来源见 [SOURCES](notes/SOURCES.md)。实验数据、结果、章节源码和检查脚本均保留。
