# 论文

当前状态与实验结论见[根目录状态页](../STATUS.md)。论文阅读入口为[合并 PDF](output/pdf/main.pdf)。

当前唯一编译入口是 [main.tex](main.tex)：正文在 [sections/submission.tex](sections/submission.tex)，附录顺序在 [sections/appendix_contents.tex](sections/appendix_contents.tex)。其他章节源码在 [sections/](sections/)；[figures/](figures/) 保存论文用图，[scripts/](scripts/) 保存绘图源码。[technical_report.tex](technical_report.tex) 是保留的历史工作稿入口。

在 `paper` 目录执行：

```powershell
New-Item -ItemType Directory -Force -Path output/pdf | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
bibtex output/pdf/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
```

理论记录保留[核心定理核验](notes/theorem.md)、[定理级文献比较](notes/literature.md)与[随机窗口可行性记录](notes/STOCHASTIC_FEASIBILITY_20261021.md)；其他开发笔记、审阅记录及历史检查 JSON 位于 [notes/archive/](notes/archive/)。
