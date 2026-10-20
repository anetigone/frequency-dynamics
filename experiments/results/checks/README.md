# 实现与完整性核验

核验结果只说明实现和材料完整性，不扩展理论证书。

[实验总导航](../../README.md)

| 结果目录 / 文件 | 内容入口 |
|---|---|
| [frozen_encoder_full_stability.json](frozen_encoder_full_stability.json) | 两个既有真实 encoder 状态，各 4000 步冻结 head 稳定性检查 |
| [frozen_encoder_checks.json](frozen_encoder_checks.json) | 低秩解、训练标签隔离、精确冻结和汇总检查 |
| [cifar_supervised_full_4000_adam](cifar_supervised_full_4000_adam/README.md) | 新监督 Adam / 裁剪配置：六条本机 GPU 完整 4000 步验证 |
| [cifar_supervised_fix_realdata.json](cifar_supervised_fix_realdata.json) | 历史 SGD 短程检查；随后服务器长程训练发散，不能作为稳定性保证 |
| [cifar_capacity_checks.json](cifar_capacity_checks.json) | 容量诊断的合成数据实现检查；不代表 CIFAR 正式结果 |
| [empirical_existing_population_comparison.json](empirical_existing_population_comparison.json) | 保存的证书 / 诊断 / 数值结果 |
| [full_empirical_checks.json](full_empirical_checks.json) | 保存的证书 / 诊断 / 数值结果 |
| [local_closeout_audit.json](local_closeout_audit.json) | 保存的证书 / 诊断 / 数值结果 |
| [layout_closeout_audit.json](layout_closeout_audit.json) | 整理后的正式运行完整性与历史源码核验 |
| [layout_audit.json](layout_audit.json) | 迁移文件哈希、源码、语法与文档链接核验 |
| [layout_smoke_checks.json](layout_smoke_checks.json) | 17 个入口导入、9 个短训练条件及两类实际汇总 |
| [subspace_checks.json](subspace_checks.json) | 保存的证书 / 诊断 / 数值结果 |
| [trained_sgd_image_checks.json](trained_sgd_image_checks.json) | 保存的证书 / 诊断 / 数值结果 |

对应代码：[code/checks](../../code/checks/)。
