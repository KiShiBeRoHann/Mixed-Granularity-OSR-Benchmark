"""
MixOE on VGG backbone —— Mixture Outlier Exposure (WACV 2023) 的 VGG 版本
==============================================================================
论文: Jingyang Zhang, Nathan Inkawhich, Randolph Linderman, Yiran Chen, Hai Li.
      WACV 2023, pp.5531-5540.  arXiv:2106.03917
      官方代码: github.com/zjysteven/MixOE (train/train_mixoe.py)

【为什么要有这个 VGG 版本】
  先试过的 CLIP 版 (train_mixoe.py) 实测失败, 原因是**结构性冲突**而非调参:
    - CLIP 的 logits = exp(logit_scale) * cos_sim, 而 logit_scale 在 train_clip.py 的
      baseline 阶段被训到了 4.772 (exp=118.17), 把 logits 尺度**锁死**在一个极尖锐的值上;
    - 而 MixOE 的核心是"让置信度从 ID 到 OOD 平滑衰减", 这要求模型能自由输出"半平坦"的分布;
    - 冻结 logit_scale → Mixed Loss 降不到理论下界(lam=0.5 时约 2.1, 实测卡在 3.4), 那点自由度
      只能靠调整 weight 方向来"压平输出", 代价是 ID 判别性被牺牲;
    - 放开 logit_scale → 被 OE 软标签损失直接压塌 (实测 AUROC 掉到 50% 的随机水平)。
  对比之下, 官方 baseline 是 ResNet-50 全量 fine-tune, logits 尺度由**新初始化的 fc 层**决定,
  想压小就压小, 不存在这个耦合。本仓库的 VGG_MLS (train_vgg_mls.py) 恰好也是这种形态:
  vgg16_bn 从零训练 + Linear(512, C), logits 尺度完全自由 —— 所以这里把它当作 MixOE 的 baseline。

【两步流程 (与官方 train.sh 一致)】
  1. baseline: train_vgg_mls.py 训出的 VGG_MLS 权重 (找不到则按同样超参现训)
  2. MixOE   : 从 baseline 出发, 用 ID + 外部 outlier 做 Mixup/CutMix 微调

【MixOE 核心 (照官方 train_mixoe.py)】
      lam     = Beta(alpha, alpha)
      mixup:  mixed_x = lam * x_id + (1 - lam) * x_oe
      cutmix: mixed_x[bbox] = x_oe[bbox]            # 把 outlier patch 贴进 ID 图
      soft_y  = lam * onehot(y_id) + (1 - lam) * (1/C)    # OOD 目标 = 均匀分布
      loss    = CE(logits_id, y_id) + beta * SoftCE(logits_mixed, soft_y)
  论文默认: mixup + alpha=1.0 + beta=5.0;  cutmix + alpha=2.0 + beta=5.0

【outlier 池】
  官方是 WebVision 1.0 并用 INET_SPLITS 显式排除"会被用作评测 OOD 的类"。这里改为
  --oe_root 指向本地 ImageFolder 目录, 启动时**自动做类名互斥核对**, 重叠则拒绝开跑。

用法:
  # 0) 先只做互斥核对
  python train_mixoe_vgg.py --dataset aircraft --seed 42 --oe_root ./datasets/imagenet/train --dry_run

  # 1) 正式跑 (会先加载/现训 VGG_MLS baseline, 再 MixOE 微调)
  python train_mixoe_vgg.py --dataset aircraft --seed 42 --oe_root ./datasets/imagenet/train

  # 批量: bash run.sh train_mixoe_vgg.py aircraft --oe_root ./datasets/imagenet/train
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms

from datasets import get_mixed_granularity_loaders
from utils import evaluate_mixed_metrics, state_dict_of, load_state_dict_into

# 复用 train_mixoe.py 里的 MixOE 组件与 outlier 池 (它们与骨干无关)
from train_mixoe import (
    SoftCE, rand_bbox, OutlierPool, prepare_outlier_pool, _norm_name,
    build_clean_classnames,
)


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_vgg_adaptive(num_classes):
    """与 train_vgg_mls.py 完全一致的骨干定义, 保证能直接加载它的权重。"""
    model = models.vgg16_bn(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
    return model


def build_transforms(dataset):
    """与 train_vgg_mls.py 一致; outlier 池也套这套 train_transform。"""
    if dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return train_transform, test_transform


def main():
    parser = argparse.ArgumentParser(description="MixOE on VGG (WACV 2023)")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    # ---- MixOE 超参 (官方值) ----
    parser.add_argument('--epochs', type=int, default=10, help='MixOE 微调 epoch (论文默认 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='MixOE 微调学习率 (论文默认 1e-3; VGG 没有 logit_scale 耦合, 可直接用)')
    parser.add_argument('--mix_op', type=str, default='mixup', choices=['mixup', 'cutmix'])
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Beta(alpha, alpha) 参数 (论文: mixup=1.0, cutmix=2.0)')
    parser.add_argument('--beta', type=float, default=5.0, help='MixOE 损失权重 (论文默认 5.0)')
    parser.add_argument('--score_mode', type=str, default='mls', choices=['mls', 'msp', 'energy'],
                        help='测试打分: mls(与 VGG_MLS 对齐, 默认) | msp(论文用法) | energy')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='只微调 classifier, 冻结 features (VGG16 有 138M 参数而本仓库数据集偏小, '
                             '过拟合时可开; 默认关闭以贴近官方全量微调)')
    # ---- baseline ----
    parser.add_argument('--baseline_epochs', type=int, default=200,
                        help='找不到 VGG_MLS 权重时, 按 train_vgg_mls.py 的超参现训的 epoch 数')
    # ---- outlier 池 ----
    parser.add_argument('--oe_root', type=str, default='./datasets/imagenet/train',
                        help='外部 outlier 池的 ImageFolder 根目录 (root/<class>/<img>), 只需图像')
    parser.add_argument('--oe_max_classes', type=int, default=100,
                        help='outlier 池最多用多少个类 (0=全部)')
    parser.add_argument('--oe_max_per_class', type=int, default=300,
                        help='每类最多用多少张图 (0=全部)')
    parser.add_argument('--dry_run', action='store_true', help='只做 outlier 池互斥核对, 不训练')
    parser.add_argument('--allow_overlap', action='store_true',
                        help='⚠️ 允许 outlier 池与 ID 类名重叠 (破坏方法前提, 仅调试用)')
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=str, default='2', choices=['2', '3', '2d', '2df', '3fc'], help="划分层级数：'2'=Family->Variant｜'3'=Maker->Family->Variant｜'2d'=从 L3 派生的两层划分（仅 aircraft，用于'层数消融'，产物落在 L2d/）；见 datasets/split_fgvc.py 分支 C")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset, seed=args.seed, batch_size=args.batch_size,
        include_unseen=args.include_unseen, num_workers=4,
        preprocess=None, num_layers=args.num_layers,
    )
    clean_classnames = build_clean_classnames(args.dataset, classnames)
    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_classnames[:3]}")
    print(f"-> Tail Names: {clean_classnames[-3:]}")

    train_transform, test_transform = build_transforms(args.dataset)
    for k, loader in loaders.items():
        if loader is not None and hasattr(loader, 'dataset'):
            loader.dataset.transform = train_transform if 'train' in k else test_transform

    # ---------------- 权重路径（先判断是否需要训练） ----------------
    method_name = "MixOE_VGG"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"{method_name}_{unseen_tag}_weights.pth")

    # ---------------- outlier 池 + 互斥核对（仅训练时需要） ----------------
    if not os.path.exists(model_path):
        if args.dataset == 'imagenet' and os.path.abspath(args.oe_root).startswith(
                os.path.abspath('./datasets/imagenet')):
            raise RuntimeError(
                "❌ imagenet 任务的 ID/OOD 类本身就来自 ImageNet-1K, 不能用它自己当 outlier 池。\n"
                "   请换一个外部数据集, 例如 --oe_root ./datasets/inaturalist/train")

        task_names = {_norm_name(n) for n in clean_classnames}
        kept_dirs, overlaps = prepare_outlier_pool(
            args.oe_root, task_names, num_classes, "MixOE-VGG",
            max_classes=args.oe_max_classes,
            max_per_class=args.oe_max_per_class or None,
            allow_overlap=args.allow_overlap,
        )
        if len(kept_dirs) == 0:
            raise RuntimeError("剔除同名类后 outlier 池为空, 检查 --oe_root 是否选错了数据集")

        oe_dataset = OutlierPool(args.oe_root, kept_dirs, transform=train_transform,
                                 max_per_class=args.oe_max_per_class or None)
        if len(oe_dataset) == 0:
            raise RuntimeError("outlier 池为空, 检查 --oe_root / --oe_max_classes / --oe_max_per_class")
        print(f"  -> outlier 池样本数: {len(oe_dataset)}")

    if args.dry_run:
        print("\n[dry_run] 互斥核对通过, 未做任何训练。")
        return

    # ---------------- 模型 / 路径 ----------------
    method_name = "MixOE_VGG"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"{method_name}_{unseen_tag}_weights.pth")
    baseline_path = os.path.join(model_dir, f"VGG_MLS_{unseen_tag}_weights.pth")

    model = get_vgg_adaptive(num_classes=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"\n[+] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    soft_ce = SoftCE()
    core = model.module if isinstance(model, nn.DataParallel) else model

    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的 MixOE-VGG 权重，直接加载: {model_path}")
        load_state_dict_into(model, torch.load(model_path, map_location=device))
    else:
        # ---------------- 阶段 1: VGG_MLS baseline ----------------
        if os.path.exists(baseline_path):
            print(f"\n[+] 复用 train_vgg_mls.py 的 baseline 权重: {baseline_path}")
            load_state_dict_into(model, torch.load(baseline_path, map_location=device))
        else:
            print(f"\n[!] 未找到 VGG_MLS baseline 权重，先按 train_vgg_mls.py 的设定训一个 "
                  f"(epochs={args.baseline_epochs}, SGD lr=0.1)...")
            opt_base = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            sch_base = optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=args.baseline_epochs)
            model.train()
            for epoch in range(args.baseline_epochs):
                total = 0.0
                for images, targets in loaders['train']:
                    images, targets = images.to(device), targets.to(device)
                    opt_base.zero_grad()
                    loss = criterion(model(images), targets)
                    loss.backward()
                    opt_base.step()
                    total += loss.item()
                sch_base.step()
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"[baseline] Epoch {epoch+1}/{args.baseline_epochs} | "
                          f"Loss {total/max(1,len(loaders['train'])):.4f}")

        # ---------------- 阶段 2: MixOE 混合微调 ----------------
        print(f"\n[!] 开始 MixOE 微调 [{unseen_tag}, {args.mix_op}, alpha={args.alpha}, "
              f"beta={args.beta}, epochs={args.epochs}, lr={args.lr}]...")
        if args.freeze_backbone:
            print("    (--freeze_backbone: 只训 classifier, features 冻结)")
            for p in core.features.parameters():
                p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9,
                              weight_decay=1e-5, nesterov=True)

        train_loader = DataLoader(loaders['train'].dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, drop_last=True)
        oe_loader = DataLoader(oe_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, drop_last=True)
        total_steps = args.epochs * max(1, len(train_loader))
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1e-6 / args.lr + (1 - 1e-6 / args.lr) * 0.5 *
            (1 + np.cos(step / total_steps * np.pi)))

        for epoch in range(args.epochs):
            model.train()
            id_losses = mixed_losses = 0.0
            n_batch = 0
            oe_iter = iter(oe_loader)
            for images, targets in train_loader:
                try:
                    oe_images, _ = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(oe_loader)
                    oe_images, _ = next(oe_iter)

                images, targets = images.to(device), targets.to(device)
                oe_images = oe_images.to(device)
                bs = min(images.size(0), oe_images.size(0))
                images, targets, oe_images = images[:bs], targets[:bs], oe_images[:bs]

                one_hot = torch.zeros(bs, num_classes, device=device)
                one_hot.scatter_(1, targets.view(-1, 1), 1)

                logits = model(images)
                id_loss = criterion(logits, targets)

                lam = np.random.beta(args.alpha, args.alpha)
                if args.mix_op == 'cutmix':
                    mixed_x = images.clone()
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                               (images.size(-1) * images.size(-2)))
                    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = oe_images[:, :, bbx1:bbx2, bby1:bby2]
                else:
                    mixed_x = lam * images + (1 - lam) * oe_images

                oe_y = torch.ones(bs, num_classes, device=device) / num_classes
                soft_labels = lam * one_hot + (1 - lam) * oe_y
                mixed_loss = soft_ce(model(mixed_x), soft_labels)

                loss = id_loss + args.beta * mixed_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                id_losses += id_loss.item()
                mixed_losses += mixed_loss.item()
                n_batch += 1

            print(f"Epoch {epoch+1}/{args.epochs} | ID Loss {id_losses/max(1,n_batch):.4f} | "
                  f"Mixed Loss {mixed_losses/max(1,n_batch):.4f}")

        torch.save(state_dict_of(model), model_path)
        print(f"\n[+] 训练完成！模型权重已保存至: {model_path}")

    # ================= 评估模块 (与 train_vgg_mls.py 同一套口径) =================
    print("\n" + "=" * 50)
    print(f"STARTING STANDALONE {method_name} DEDICATED EVALUATION  (score={args.score_mode})")
    print("=" * 50)
    model.eval()

    def collect_scores(loader):
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device))
                if args.score_mode == 'msp':
                    scores = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
                elif args.score_mode == 'energy':
                    scores = torch.logsumexp(logits.float(), dim=-1)
                else:  # 'mls'
                    scores = logits.max(dim=-1).values.float()
                preds = logits.argmax(dim=-1)
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if isinstance(targets, torch.Tensor):
                    all_targets.extend(targets.cpu().numpy())
                else:
                    all_targets.extend(targets)
        return np.array(all_scores), np.array(all_preds), np.array(all_targets)

    _loader_keys = [
        ('id', 'test_id'),
        ('coarse_gen', 'test_coarse_gen'),
        ('medium_gen', 'test_medium_gen'),
        ('near_ood', 'test_near_ood'),
        ('near_family', 'test_near_family'),
        ('near_variant', 'test_near_variant'),
        ('far_ood', 'test_far_ood'),
    ]
    score_map = {}
    for key, lkey in _loader_keys:
        ldr = loaders.get(lkey)
        score_map[key] = collect_scores(ldr) if (ldr is not None and len(ldr.dataset) > 0) else None

    evaluate_mixed_metrics(
        score_map,
        method_name=method_name,
        mode=('Unseen_In_Train' if getattr(args, 'include_unseen', False) else 'Pure_Closed_Set'),
        tpr_target=0.95,
        save_dir=model_dir,
        extra_meta={'dataset': args.dataset, 'num_layers': args.num_layers,
                    'seed': args.seed, 'eval_view': 'V4'},
    )


if __name__ == "__main__":
    main()
