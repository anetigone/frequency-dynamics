# 压缩投影表示中的功率坐标竞争

更新：2026-10-19。主稿：[THEORY_DRAFT.md](THEORY_DRAFT.md)。

**主结果为定理 T：** 在显式八维相对 ±0.001% 的连续初值盒上，对径向增强强度 η∈[0,0.2]，无正则总体 projection 风险在相对能量截面 ℓ=1.7 至少改善 0.04，在 ℓ=6 至少超出初值 0.15，两截面回升至少 0.19。认证覆盖整段轨迹、截断与舍入误差、初值扰动、分母正性及 Q<0.689；两截面在有限物理时间可达，训练损失严格下降。

候选贡献集中于真实因子化训练诱导的独立语义任务风险变化。相同初始函数下的尺度敏感性、projection 几何作用及早停现象已有直接先例，正文 §1.1 对照 Kunin、Torralba–Weiss、Xue 等工作。定理 2 解释具体方向阈值，定理 3 的局部 Taylor 展开解释回升如何产生。定理 T 的完整计算机辅助证明置于正文附录 A。

- [THEORY_DRAFT.md](THEORY_DRAFT.md)：主定理、机制解释、数值对照与认证附录。
- [experiments/](experiments/)：统一的代码与结果目录，入口说明见 [experiments/README.md](experiments/README.md)。
- [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)：审稿意见处理与当前边界。
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md)：来源记录与尚未解除的新颖性阻断。

从仓库根目录复算：

```bash
python -B experiments/verify.py
python -B experiments/check_arithmetic.py
python -B experiments/mechanism_checks.py
python -B experiments/representation_checks.py
```

所有默认结果写入 experiments/；区间验证器从有理数初值重新生成 certificate.json，无需已有证书输入。认证依赖启用断言，请勿使用 -O。浮点轨迹及扫描另存对应结果文件，不替代连续参数盒的区间证明。

当前评测对象是一维 projection 输出。可逆二维 encoder 的最优无正则总体 probe 风险恒为噪声底，主定理不解释丢弃 head 后的 encoder 退化。严格长轨迹认证仅覆盖 ±0.001% 盒；±1% 盒仅有初始导数保证和数值扫描。本文不声称一般网络、高概率初始化、新颖性或真实图像结论，定点相关工作审计已更新；Tepe 全文、Liao 完整结果和前向引用图仍待核查，下一阶段并行推进这些核查与视觉验证。

工作区采用固定文件名，历史版本交由 Git 管理。
