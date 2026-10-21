# 首稿说明

## 写作选择

暂定标题：**What Layerwise Learning Rates Select: Risk Rebound and Subspace Motion in Compressed Representations**。

主线依次为：精确风险对象 → 计算机辅助固定间隔定理 → 匹配初始函数的方向阈值 → encoder / projection 子空间区分 → 完整经验训练与干预失效边界 → CIFAR 容量及冻结 head 诊断。

全文不主张一般尺度律、早停、projection 信息选择或平方网络风险回升现象的独有性。定理 T 与方向阈值使用不同初始化，主文明确分开。二维 projection 定理没有被改写成 encoder 退化定理。

## 后续编辑事项

1. 作者姓名、单位、目标会议模板和版面压缩尚待作者确定；目前使用 Working manuscript。
2. 相关工作基于原有 SOURCE_AUDIT.md，并在 2026-10-20 对引用元数据进行一手来源核对。Tepe 全文、Liao 完整结果、前向引用图及等价定理排除尚未完成；当前不能宣布新颖性审计完成。
3. 主文保留核心认证、总体/经验子空间与晚期预测失效。CIFAR 容量和冻结 head 段落较长，压缩投稿版本时可将容量表移到补充材料，保留冻结 head 核心对照。
4. 服务器容量和冻结 head 只有复制的 summary。初稿明确其证据来源，未声称独立复查原始参数或服务器裁剪率；需归档原 result.json / protocol / 优化日志和必要 checkpoint 后再做最终复现验收。
5. 原始 frozen summary 的文件名为 summey.json，保持不变，构图脚本按真实路径读取。
6. 未新增实验，未将开发种子算入正式确认，也未把相关分叉算作独立重复。
7. 数值图由保存的汇总生成；误差条明确区分十种子 bootstrap 描述区间与三个开发种子的逐点散点。
8. 现有稿件是完整内容首稿，尚非页数合规的投稿定稿。结论、负面证据和来源链优先于压缩。
