# 混合粒度开放集识别评估 (Mixed-Granularity Open-Set Recognition)

在已知类 + 粗/细粒度泛化 + 近邻/远距 OOD 的多粒度设定下，对比多种方法（Zero-Shot CLIP、Linear Probe CLIP、CoOp、CoCoOp、A2Pt、ARPL+CS、DeF、FSMoE、VGG-MLS）的开放集识别能力，核心指标包括 **OSCR**、AUROC、CV-RR / UN-RR / H-Score、IFCR 等。

## 环境要求

- Linux + NVIDIA GPU（建议显存 ≥ 16GB）
- CUDA（11.8 / 12.x 均可，与 PyTorch 版本对应）
- Python 3.8+（推荐 conda 管理环境）

## 快速开始（新机器配置步骤）

### 1. 克隆代码

```bash
git clone https://github.com/KiShiBeRoHann/Mixed-Granularity-OSR-Benchmark.git wty
cd wty
```

> 项目内所有路径均为相对路径（锚定在项目根目录 / 脚本所在目录），克隆到任意位置均可直接运行。

### 2. 创建 conda 环境并安装依赖

```bash
conda create -n wty python=3.8 -y
conda activate wty

# PyTorch（CUDA 版，按你的 CUDA 版本选择，此处以 11.8 为例）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 其余依赖
pip install numpy scikit-learn tqdm matplotlib seaborn pillow nltk

# OpenAI CLIP（代码中 import clip）
pip install git+https://github.com/openai/CLIP.git

# ImageNet 需要用 WordNet 清洗类名
python -c "import nltk; nltk.download('wordnet')"
```

验证安装：

```bash
python -c "import torch, clip, sklearn, nltk; print('OK')"
```

### 3. 准备数据集

所有数据集放在项目根目录的 `datasets/` 下（代码通过相对路径 `./datasets/...` 读取）：

```
datasets/
├── cifar-100-python/        # CIFAR-100 官方包解压后（含 train、test、meta）
├── fgvc-aircraft-2013b/     # FGVC-Aircraft（data/images/ 及各类 txt）
├── imagenet/                # ImageNet-1k（train/ 按类目录 + val/）
├── iNaturalist2021_Mini/    # iNaturalist 2021 Mini（train_mini/ + val/ + train_mini.json）
├── imagenet_tree.txt        # 随仓库：ImageNet 层次树（勿删）
└── inaturalist_2021.txt     # 随仓库：iNaturalist 层次树（勿删）
```

> `datasets/` 下的大文件（图片、压缩包等）已在 `.gitignore` 中排除，clone 后需自行下载并放置；`*.txt` 与 `*.py` 会随仓库提交。

### 4. 生成数据集划分

```bash
python build.py
```

- 为 ImageNet / iNaturalist 生成 `splits/imagenet/L{2,3}/seed_{42..46}.json`、`splits/inaturalist/L{2,3}/...`。
- CIFAR-100 与 Aircraft 的划分会在对应脚本首次运行时自动生成并缓存到 `splits/`，无需手动处理。

### 5. 运行实验

单次运行：

```bash
python train_clip.py --dataset cifar100 --seed 42
python eval_zsclip.py --dataset aircraft --num_layers 2 --include_unseen
```

批量跑 5 个 seed（42~46）：

```bash
bash run.sh train_a2pt.py aircraft          # 标准模式
bash run.sh train_a2pt.py aircraft --include_unseen --num_layers 2
```

脚本统一参数：

| 参数 | 取值 | 说明 |
|---|---|---|
| `--dataset` | `cifar100` / `aircraft` / `imagenet` / `inaturalist` | 数据集 |
| `--seed` | int（默认 42） | 随机种子 |
| `--num_layers` | `2` / `3`（默认 2） | 2 层 = Family→Variant，3 层 = Maker→Family→Variant |
| `--include_unseen` | flag | 训练集是否包含未见变体（否则为纯闭集训练） |
| `--epochs` / `--batch_size` / `--lr` | — | 训练超参 |

## 项目结构

```
.
├── train_clip.py        # Linear Probe CLIP
├── train_coop.py        # CoOp
├── train_cocoop.py      # CoCoOp
├── train_a2pt.py        # A2Pt
├── train_arpl.py        # ARPL+CS（VGG 骨干）
├── train_def.py         # DeF
├── train_fsmoe.py       # FSMoE
├── train_vgg_mls.py     # VGG-MLS
├── eval_zsclip.py       # Zero-Shot CLIP 评估
├── utils.py             # 指标实现（OSCR / CV-RR / UN-RR / H-Score / IFCR 等）
├── build.py             # ImageNet / iNaturalist 层次树划分构建
├── run.sh               # 批量实验入口
├── datasets/            # 数据集与加载器（split_cifar / split_fgvc / split_imin）
└── splits/              # 生成的划分 json
```

## 输出与缓存

| 路径 | 内容 | 说明 |
|---|---|---|
| `models/<dataset>/L<层数>/seed_<seed>/` | 模型权重 `.pth` | **再次运行会自动加载已有权重、跳过训练**（已实现缓存检查） |
| `experiment_logs/` | run.sh 的训练日志 | 按 数据集/方法/模式 分目录 |
| `splits/` | 各数据集划分 json | 首次生成后复用 |

`models/`、`experiment_logs/` 均已加入 `.gitignore`，不会被提交。

## 指标说明

- **OSCR**：开集分类识别曲线面积（`utils.calculate_oscr(pred_k_id, x_k_id, pred_u_ood)`），Global OSCR 用全部已知分数 + 全部 OOD 分数，MG-OSCR 用混合粒度对抗子集（如 coarse_gen vs near_variant）。
- **AUROC**：已知 vs 未知的二分面积。
- **CV-RR / UN-RR / H-Score**：阈值依赖（TPR=95%）的变体拒绝率与综合分。
- **IFCR**：族内混淆率（细粒度类在家族内部的错误占比）。

## 常见问题

- **找不到划分文件**：先运行 `python build.py`（ImageNet / iNaturalist）。
- **找不到数据集**：确认 `datasets/` 下目录名与第 3 节一致（注意 `iNaturalist2021_Mini` 大小写）。
- **模型权重重训**：删除 `models/<dataset>/.../对应权重.pth` 后重新运行即可。
- **nltk 报错**：执行 `python -c "import nltk; nltk.download('wordnet')"`。
