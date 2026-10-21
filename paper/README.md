# Paper draft

2026-10-21 理论冲刺修订稿。主线改为局部二阶矩中的任务相关子空间运动、可解析的真实编码器瓶颈回升族，以及有限幅度和有效窗口。使用通用双栏 LaTeX，尚未套用目标年份的正式 CVPR 模板或压缩到投稿页数。

本轮结论与成败边界见 [THEORY_SPRINT_20261021.md](THEORY_SPRINT_20261021.md)。新增推导包括结构化 ±5% 协方差范围内的统一风险间隔、线性近守恒、两块竞争判据和已有轨迹上的起点区间预测。没有新增真实数据训练；CIFAR/DTD 主文压缩为一张边界表，原详细诊断保存在附录。

- `main.tex`：编译入口；正文按主题放在 `sections/`。
- `output/pdf/main.pdf`：可阅读的完整初稿，包括参考文献和证明附录。
- `references.bib`：核对过元数据的八条引用。
- `scripts/build_figures.py`：从已有 JSON 汇总生成三张图，不训练模型。
- `SOURCES.md`：数字、证明与实验配置的出处。
- `EDITORIAL_NOTES.md`：首稿定位、待作者处理事项与未完成审计。

从 `paper/` 目录编译：

```powershell
& H:/anaconda3/envs/py312/python.exe -B scripts/build_figures.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
bibtex output/pdf/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
```

稿件保留晚期干预预测退化、线性模型的经验稳定漂移，以及 CIFAR 等能量结果不充分等限制。监督低秩 head 是诊断参照，不是无需标签的干预方法。旧证书、训练代码与实验结果均保留；本轮只增加理论推导、闭式/区间检查和对应的新结果文件。
