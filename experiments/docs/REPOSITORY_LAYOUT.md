# 目录整理与复现

2026-10-19。此次调整只涉及目录、导入路径、结果定位和文档导航；训练更新、损失、统计规则及已保存数值结果保持原样。

## 目录约定

```text
experiments/
  README.md                 按实验阶段导航
  run.py                    按脚本名运行 / 列出入口
  layout.py                 源码和结果位置解析
  layout_manifest.json      迁移映射与 SHA-256
  code/
    theory/                 区间认证、二维机制
    shared/                 真实平方图像模型与共同目标
    pilot/                  图像与纹理开发实验
    population/             总体子空间和总体状态经验分叉
    empirical/              从头到尾的经验训练和诊断
    cifar/                  数据审计、训练、OLS、绘图
    checks/                 实现核验与完整性审计
  docs/                     协议和发现
  results/
    theory/                 证书和二维机制结果
    pilot/                  历史探索结果
    population/             总体确认与对应分叉
    empirical/              完整经验确认与对应分叉
    cifar/                  正式五种子自然图像比较
    checks/                 数值核验和迁移审计
    archive/development/    开发配置和吞吐测试
  archive/
    sources_before_layout_20261019/  原始源码快照
    reorganize_workspace_20261019.py  已执行的一次性迁移记录
```

原始 CIFAR 数据仍在仓库根目录。论文主稿、来源审计和审稿响应仍在根目录。

## 复现旧结果与启动新实验

每个正式结果目录中的 `protocol.json`、`result.json`、权重、图像和数值汇总均保留原内容。
其中记录的旧路径是运行时来源记录；实际新位置由 `layout_manifest.json` 映射。不要把历史来源字段误当成新的运行入口。

旧协议的源码哈希对应 `archive/sources_before_layout_20261019/` 中的原始文件。
当前代码新增了目录引导和路径解析，所以当前哈希另存于迁移清单。
`audit_local_closeout` 同时核验原始快照与当前文件，不能以修改旧协议的方式绕过哈希检查。

当前命令例如：

```powershell
python -B experiments/run.py --list
python -B experiments/run.py full_empirical --phase confirmation --seeds 2 3 4 5 6 7 8 9 10 11 --output experiments/results/empirical/new_confirmation
python -B experiments/run.py summarize_full_sgd experiments/results/empirical/new_confirmation
python -B experiments/run.py intervene_full_sgd --source experiments/results/empirical/new_confirmation --output experiments/results/empirical/new_interventions
```

正式旧目录拒绝不同源码哈希的接写，这是预期保护。历史协议文档中的原配置命令已更新位置；实际重训时更换 `--output`，并相应更换后续 `--source` / 汇总路径。
汇总和已有 checkpoint 核验可直接使用现有归档目录。二维认证的默认输出位于 `results/theory/`；必须启用断言，不使用 `-O`。

## 整理后的检查

`python -B experiments/run.py audit_layout` 检查：

- 原始数值、权重和图片文件 SHA-256；Markdown 允许仅修改导航文字 / 链接。
- 原始源码快照及整理后源码 SHA-256。
- 当前 Python 语法、内部文档链接和登记的迁移目标。

`python -B experiments/run.py audit_local_closeout` 继续检查正式轨迹数量、4000 步终点、CIFAR 评估完整性及配对数据。
迁移检查结果写入 `results/checks/layout_audit.json`，整理后的实验完整性检查写入 `results/checks/layout_closeout_audit.json`，不覆盖历史审计。

整理后还检查了 17 个命令入口，在临时目录运行三模型×三配比的 4 步短训练，并用复制的现有结果运行完整经验 / CIFAR 汇总。临时文件已清理，记录见 [短训练与汇总检查](../results/checks/layout_smoke_checks.json)。
