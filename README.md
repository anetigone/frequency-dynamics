# 压缩投影表示中的功率坐标竞争

更新：2026-10-19。主稿：[THEORY_DRAFT.md](THEORY_DRAFT.md)。

**主结果为定理 T：** 在显式八维相对 ±0.001% 的连续初值盒上，对径向增强强度 η∈[0,0.2]，无正则总体 projection 风险在相对能量截面 ℓ=1.7 至少改善 0.04，在 ℓ=6 至少超出初值 0.15，两截面回升至少 0.19。认证覆盖整段轨迹、截断与舍入误差、初值扰动、分母正性及 Q<0.689；两截面在有限物理时间可达，训练损失严格下降。

候选贡献集中于真实因子化训练诱导的独立语义任务风险变化。相同初始函数下的尺度敏感性、projection 几何作用及早停现象已有直接先例，正文 §1.1 对照 Kunin、Torralba–Weiss、Xue 等工作。定理 2 解释具体方向阈值，定理 3 的局部 Taylor 展开解释回升如何产生。定理 T 的完整计算机辅助证明置于正文附录 A。

- [THEORY_DRAFT.md](THEORY_DRAFT.md)：主定理、机制解释、数值对照与认证附录。
- [experiments/](experiments)：统一的代码与结果目录，入口说明见 [experiments/README.md](experiments/README.md)。
- [本机实验收尾](experiments/docs/LOCAL_CLOSEOUT.md)与[完整经验协议](experiments/docs/FULL_EMPIRICAL_PROTOCOL.md)：90 条经验主轨迹、40 条独立流、90 条局部经验复验、经验状态重复分叉，以及标准 CIFAR-10 的有限预算五种子比较。早期方向预测有效，晚期退化；自然图像外推证据较弱。
- [图像实验协议](experiments/docs/IMAGE_EXPERIMENTS.md)与[第一批试跑记录](experiments/docs/PILOT_FINDINGS.md)：开发阶段的真实空间卷积核验与两种子 pilot 记录。
- [子空间机制与十种子确认](experiments/docs/SUBSPACE_MECHANISM.md)：此前的全支撑 $K=1/2$ 和局部核共 540 条总体轨迹，解释总体条件下 encoder 子空间改变与固定空间内 projection 重选的区别；不把线性守恒套到经验 SGD。
- [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)：审稿意见处理与当前边界。
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md)：来源记录与尚未解除的新颖性阻断。

从仓库根目录复算：

```bash
python -B experiments/code/theory/verify.py
python -B experiments/code/theory/check_arithmetic.py
python -B experiments/code/theory/mechanism_checks.py
python -B experiments/code/theory/representation_checks.py
```

理论验证的默认结果写入 experiments/results/theory/；区间验证器从有理数初值重新生成 certificate.json，无需已有证书输入。认证依赖启用断言，请勿使用 -O。浮点轨迹及扫描另存对应结果文件，不替代连续参数盒的区间证明。

主定理的评测对象仍是一维 projection 输出。可逆二维 encoder 的最优无正则总体 probe 风险恒为噪声底，主定理不解释丢弃 head 后的 encoder 退化。严格长轨迹认证仅覆盖 ±0.001% 盒；±1% 盒仅有初始导数保证和数值扫描。新增高维纹理实验与十种子确认不扩大该认证范围。本文不声称一般网络、高概率初始化、新颖性或自然图像上的普遍保证；Tepe 全文、Liao 完整结果和前向引用图仍待核查。

实验代码、协议与结果已按阶段整理，参见 [实验导航](experiments/README.md)；开发结果与原始源码快照单独归档。
