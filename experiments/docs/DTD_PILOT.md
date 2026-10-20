# DTD：Voxel51 Hugging Face 镜像与平方模型 pilot

2026-10-20。先检查真实纹理上的学习增量，再决定是否执行同 checkpoint 分叉。代码已提供两个阶段，但不自动按验证结果选择或启动下一阶段。

## 数据来源与大小

使用用户指定的 [Voxel51/Describable-Textures-Dataset](https://huggingface.co/datasets/Voxel51/Describable-Textures-Dataset)，固定 revision `e850d7c773fa71940567cb53da4270b89fb3737a`。该仓库包含原 JPEG 图片及 FiftyOne JSON 元数据，页面全量大小约 646 MB。本代码直接读取 JSON/JPEG，不安装或运行 FiftyOne，不使用较大的 Parquet 重打包镜像。

该镜像的 dataset viewer 将全部图片显示为一个 train 集，`samples.json` 没有 train/val 标签。不能直接把 viewer 的 train 用作训练集。代码随附官方 **partition 1** 的 train1/val1/test1 文件名单：

- `experiments/resources/dtd_partition1.json`，含来源 URL、原标签包 SHA-256/MD5 和完整名单。
- 名单来自官方 188,651 字节标签包 `dtd-r1.0.1-labels.tar.gz`，不是完整图片压缩包。
- 已用真实 Voxel51 元数据核对全部 5,640 个图片名字与官方名单一一对应。
- 训练集和验证集各 1,880 张，每类各 40 张；只下载这 3,760 张 JPEG 和两份 JSON。测试图片不下载、不解码、不做性能评估。
- 图片实际下载字节数由命令输出的 `image_bytes` 报告，不能把整个仓库 646 MB 当作本次下载量。

本地缓存默认位于 `experiments/data/dtd_hf/`，已加入 Git 忽略。HF native cache 支持复用与恢复未完成下载；失败时可重复同一 download 命令。镜像 commit、元数据校验、每图摘要和正式划分均记录。数据准备后的 tensor 另存 SHA-256 校验。

下载请求显式设置 `Accept-Encoding: identity`，覆盖 metadata、Hub API 和图片请求，避免某些环境中新版 httpx2 与旧 Brotli 后端的 `process() takes no keyword arguments` 解压兼容错误。仍使用原 HF 缓存及续传，不需卸载或替换服务器依赖；更新代码后重跑 download 即可。

## 安装与命令

现有环境已有 PyTorch、torchvision、NumPy 的前提下，额外只需：

```bash
python -m pip install huggingface_hub pillow
```

不需要 `datasets`、`pyarrow` 或 `fiftyone`。如需设置 `HF_ENDPOINT`，应在启动 Python 前设置；下载沿用 Hugging Face Hub 的环境配置，不会静默改镜像地址。

在仓库根目录运行，先同步新增代码、共享入口更新、`experiments/resources/` 及 `layout_manifest.json`：

```bash
# 不读取数据、不联网、不初始化 GPU
python -B experiments/run.py dtd_pilot plan

# 原 JPEG 和元数据从 Hugging Face 下载；默认 8 并发，可 --download-workers 4 调低
python -B experiments/run.py dtd_pilot download

# 只使用本地 HF cache，生成训练所需张量；这一步不联网
python -B experiments/run.py dtd_pilot prepare

python -B experiments/run.py dtd_pilot train
python -B experiments/run.py dtd_pilot evaluate
python -B experiments/run.py dtd_pilot summarize
```

本机将 python 替换为 `& H:/anaconda3/envs/py312/python.exe`。train/evaluate 只读 prepared tensor，不联网；prepare 缺缓存时明确报错，不回退到官网下载。默认结果目录为 `experiments/results/empirical/dtd_pilot/`。

## 第一阶段配置

- 三个开发种子 0/1/12，r=0.1/1/10，共九条从头训练轨迹。三个配比在各自种子内从同一参数初态出发，训练随机流配对。
- 灰度输入 64×64，64 个 11×11 valid 卷积滤波器，平方、GAP、8 维线性 head；没有 bias、BN、动量、weight decay 或输出归一化。
- 图像预处理：EXIF 纠正后转灰度，保持比例将短边缩放到 72，再中心裁成 64；数据集级均值/标准差只用训练图片计算，不做逐图归一化。
- 两视图增强：反射 padding 8 后随机裁剪、水平翻转、contrast 0.9–1.1、标准差 0.01 的 Gaussian noise。没有预设高频一定无用，也没有增加专门的低通操作。
- 普通 SGD，基础 lr=0.002，batch=128，2,000 次更新，约 136 个训练集遍历。初始 clean 输出能量设为 0.01，沿用无标签 Jacobian 迹校准，层间配比不改变初始函数。
- 固定 checkpoint：0/500/1000/1500/2000。固定无标签审计集 256 张，每 10 步测量 GUPA loss、alignment、两个 uniformity 项、view 协方差全谱及 clean 输出能量。
- 标签仅用于 evaluate：47 类仿射、无正则 OLS，在 1,880 个训练标签上拟合，在独立 1,880 图像验证集上报告 encoder/projection 的 one-hot MSE、accuracy、margin、47×47 混淆矩阵、probe 数值秩/奇异值及特征谱。

初始化、校准、训练与固定 loss 审计均不以任务标签决定更新。数据类别用于核验官方划分和数据完整性。没有默认监督训练参考或现代 backbone；这是单一浅层模型的开发 pilot。

首要阅读 `summary.json` / `findings.md` 中的相对初始化变化、逐种子值及 encoder/projection 差异。MSE 为主要终点，accuracy 并列报告；MSE 差为分支减对照，负值更好。若均没有可信学习增量，先诊断目标/增强/模型适配，不直接扩大配比或种子。

## 第二阶段：同 checkpoint 分叉（另行启动）

默认从第一阶段 r=1、step=500 的状态出发，r=0.1/1/10 加整体学习率 ×0.5/×2，共 15 个分支，继续 1,000 更新，每 250 步评估。沿用来源初始 mobility，不在分叉时重校准。

```bash
python -B experiments/run.py dtd_pilot plan --branching
python -B experiments/run.py dtd_pilot train --branching
python -B experiments/run.py dtd_pilot evaluate --branching
python -B experiments/run.py dtd_pilot summarize --branching
```

输出 `experiments/results/empirical/dtd_branches/`，不改写第一阶段。来源使用保存的完整模型；旧更新为无动量 SGD，无需重建动量。分支使用新的共享后续 RNG，不声称接回原数据顺序。保存的 checkpoint 包含模型、SGD 和分支 RNG；当前实现成功任务跳过、未完成任务从来源重跑，不作中途恢复。

默认共同审计 loss 水平 −0.05/−0.1/−0.2/−0.3/−0.4，首次向下穿越保存，默认超调/两分支 loss 差容差 0.01。保持 10 步测量以减轻 CIFAR 中的超调问题，但仍不能保证完整匹配；初始已低于目标或未到达时保留缺失，不插值。若根据第一阶段无标签 audit 历史制定更合适的预定水平，可用 `--loss-levels`，并在所有 branch train/evaluate 命令中保持一致，避免看过分叉任务结果后改标准。

相同更新次数、共同 loss 和整体速度对照应合看，任何一个标量都不证明路径完全等价。三个开发种子仅作描述性汇总，不把截面/分支当独立重复。一个 DTD 分类任务也不能证明两个任务的偏好取舍。

## 实现与检查

- `code/shared/dtd_hf.py`：Voxel51 数据获取、官方名单映射、原图校验、离线预处理及 tensor 缓存。
- `code/empirical/dtd_pilot.py`：九条学习有效性轨迹、47 类 probe、可选五分支实验。
- 原 CIFAR runner 增加可选 source/augmentation/probe 回调，默认 CIFAR 行为保持；汇总器支持三分支 pilot 与五分支实验。

```bash
python -B experiments/run.py check_dtd_pilot
python -B experiments/run.py check_checkpoint_branches
```

DTD 检查使用本地生成的 JPEG 和镜像 JSON，mock HF 下载调用；覆盖正式名单筛选、禁止测试图片下载、缓存复用、训练集归一化、47 类 probe、pilot/fork 完整流程与损坏输入拒绝。另已检查真实镜像元数据与官方 partition 1 的一致性。未下载真实图片或训练真实 DTD；检查不算机制实验证据。
