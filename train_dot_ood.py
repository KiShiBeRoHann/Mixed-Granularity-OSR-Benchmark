"""
DOT-OOD: A Novel Fine-Tuned CLIP-OOD Detection Method with Double Loss Constraint
                Through Optimal Transport Semantic Alignment
========================================================================================
论文: Hengyang Lu, Xin Guo, Shuai Feng, Wenyu Jiang, Yuntao Du, Chang Xia, Chenyou Fan.
      AAAI 2026, Proceedings vol.40, pp.15448-15456.  官方代码: github.com/jncsnlp/DOT-OOD

方法定位: CLIP 微调 + 双重损失约束 (DLC) + 最优传输 (OT) 语义对齐。
  - 全量数据(默认) 或 few-shot(--use_fewshot, 默认每类 16 张) 微调 CLIP 尾部少数 block/proj；
  - 通过文本类别 prompt 的特征做跨模态对齐:
      logits = image_features @ text_features^T / t       (t = 0.01)
  - 三重损失 (Eq. 交叉熵 + 对齐 + 排斥):
      L = CE(logits, y) + alpha * L_ot + beta * L_neg
        * L_ot : Aligned Image-Text Concept Matching Loss
            对每个样本 image/其类文本做 OT 对齐后, 拉近 (1 - align@text)
        * L_neg: Negative Sample Repulsion Loss
            对齐特征与 batch 内其它类图像特征 { abs(align@image_neg) } 求均值(排斥)
  - OT 模块: 把 feature 的维度当作 D 个"传输位置", 用 Sinkhorn-Knopp 计算从图像
    到文本特征分布的最优传输, 得到重加权后的图像特征 (增强表示)。

本脚本仿照仓库内 train_clip.py / eval_zsclip.py 的测评框架:
  - 加载同一套混合粒度 loaders (datasets.get_mixed_granularity_loaders)
  - 输出同一套指标: 全局/分层 Acc、AUROC (strict/near/far)、CV-RR/UN-RR/H-Score、OSCR

⚠️ 设定说明 (相对官方 few-shot 已按仓库习惯调整):
  - 仓库其它方法 (train_clip.py 等) 都是"全量数据 + 50 epoch"。为与它们公平对比,
    本脚本**默认也是全量训练 + 50 epoch**;
  - 若想忠实还原 DOT-OOD 官方设定, 请显式传 `--use_fewshot`(每类抽 --shot=16 张,
    并建议同时 `--epochs 15`, 对应官方 num_epochs=15 / shot=16)。
  - 环境无 POT 库, 因此 Sinkhorn 用纯 PyTorch 复刻 (等价 ot.sinkhorn)。
  - 负样本 = 同一 batch 内其它类别的图像特征 (官方实现, 依赖 batch 内多类混批)。
"""
import argparse
import os
import random

import numpy as np
import torch
import clip
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import get_mixed_granularity_loaders
from utils import evaluate_mixed_metrics, state_dict_of, load_state_dict_into


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# =====================================================================
# 最优传输 (Optimal Transport) 模块 —— 纯 PyTorch 复刻 ot.sinkhorn
# =====================================================================
def sinkhorn(a, b, M, epsilon, numItermax=1000, stopThr=1e-8):
    """
    熵正则化最优传输 (Sinkhorn-Knopp / Cuturi 2013)。
    与 POT 的 ot.sinkhorn(a, b, M, epsilon, numItermax, stopThr) 等价。
    :param a: (n,) 源分布, 和为1
    :param b: (m,) 目标分布, 和为1
    :param M: (n, m) 代价矩阵 (越小越倾向于传输)
    :param epsilon: 熵正则化系数
    :return: (n, m) 最优传输计划 T
    """
    a = a.to(M.device).float()
    b = b.to(M.device).float()
    M = M.to(M.device).float()
    K = torch.exp(-M / epsilon)
    u = torch.ones_like(a)
    Kv = None
    for _ in range(numItermax):
        v = b / (K.T @ u)          # 列缩放: 满足列边际 b
        Kv = K @ v
        u = a / Kv                 # 行缩放: 满足行边际 a
        if torch.max(torch.abs(Kv * u - a)) < stopThr:
            break
    T = u[:, None] * K * v[None, :]
    return T


def optimal_transport(image, text, epsilon):
    """
    官方 DOT-OOD: 把特征向量的每个维度当作一个"传输位置",
    用 Sinkhorn 求图像特征分布到文本特征分布的最优传输, 得到图像对齐特征。
    :param image: (D,) 图像特征 (shape 内部 view 成 (D,1) 与 (1,D) 得 (D,D) 代价)
    :param text:  (D,) 该类的文本特征
    :return: (D,) L2 归一化后的图像对齐特征
    """
    S = 1 - torch.matmul(image.view(-1, 1), text.view(1, -1))   # (D, D) 代价
    dev = S.device
    a = (torch.ones(S.shape[0]) / S.shape[0]).to(dev)
    b = (torch.ones(S.shape[1]) / S.shape[1]).to(dev)

    T = sinkhorn(a, b, S, epsilon, numItermax=1000, stopThr=1e-8)
    image_aligned = torch.matmul(T.T, image)                    # (D,) 重加权特征
    image_aligned = image_aligned / torch.norm(
        image_aligned, p=2, dim=-1, keepdim=True)
    return image_aligned


# =====================================================================
# 类名清洗 (复用 train_clip.py 的逻辑, 保证与其它 CLIP 方法一致)
# =====================================================================
def build_clean_classnames(args, classnames):
    if args.dataset == 'imagenet':
        from nltk.corpus import wordnet as wn
        clean = []
        for name in classnames:
            raw_id = name.split(' ')[0]
            try:
                pos = raw_id[0]
                offset = int(raw_id[1:])
                synset = wn.synset_from_pos_and_offset(pos, offset)
                clean.append(synset.lemmas()[0].name().replace('_', ' '))
            except Exception:
                clean.append(raw_id.replace('_', ' '))
        return clean
    elif args.dataset in ['aircraft', 'inaturalist']:
        # 去掉末尾层级标注括号: "Boeing 767 (Family)" -> "Boeing 767"
        return [name.rsplit(' (', 1)[0].replace('_', ' ') for name in classnames]
    else:
        return [name.replace('_', ' ') for name in classnames]


# =====================================================================
# few-shot 抽取: 按类别随机抽 k 张, 返回 train loader 的 Subset
# =====================================================================
def _extract_labels(train_dataset):
    """
    尽量不解码图像地取回底层 train dataset 的每个样本 label (保序)。
    依次尝试常见属性, 都拿不到再退回用 DataLoader 探测(会解码图像, 较慢)。
    """
    # 1) dataset.data_list = [(..., label), ...]  (aircraft TreeSubDataset / AircraftSubDataset)
    if hasattr(train_dataset, 'data_list'):
        vals = []
        for it in train_dataset.data_list:
            vals.append(int(it[1]))
        return vals
    # 2) Subset 包装情形: 先处理底层再按 indices 取子集
    if isinstance(train_dataset, Subset):
        base = train_dataset.dataset
        base_labels = _extract_labels(base)
        return [base_labels[i] for i in train_dataset.indices]
    # 3) dataset.targets = [label, ...]  (CIFAR CustomSubset)
    if hasattr(train_dataset, 'targets'):
        return [int(x) for x in train_dataset.targets]
    return None


def build_fewshot_subset(train_dataset, num_classes, shot, seed):
    """
    按 label 分组后从每类随机抽 shot 个样本, 返回 torch.utils.data.Subset。
    优先直接读标签(不解码图像), 失败则退回 DataLoader 探测。
    """
    rng = random.Random(seed)
    labels = _extract_labels(train_dataset)
    if labels is not None and len(labels) == len(train_dataset):
        label2idx = {}
        for i, lab in enumerate(labels):
            label2idx.setdefault(int(lab), []).append(i)
        # 统一构造映射, 缺类的类别留空
        for c in range(num_classes):
            label2idx.setdefault(c, [])
    else:
        label2idx = {c: [] for c in range(num_classes)}
        probe = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)
        offset = 0
        for batch in probe:
            _, lb = batch
            lb = lb.numpy() if hasattr(lb, 'numpy') else np.asarray(lb)
            for j, lab in enumerate(lb):
                label2idx.setdefault(int(lab), []).append(offset + j)
            offset += len(lb)

    selected = []
    missing = []
    for c in range(num_classes):
        idx = label2idx.get(c, [])
        if len(idx) == 0:
            missing.append(c)
            continue
        k = min(shot, len(idx))
        selected.extend(rng.sample(idx, k))
    if missing:
        print(f"\n[!] 以下类别在 train loader 中无样本(few-shot会缺类): {missing}")
    # 保持类别内样本聚合, 打乱整体顺序
    rng.shuffle(selected)
    return Subset(train_dataset, selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--shot', type=int, default=16,
                        help='few-shot: 每类训练样本数(仅在 --use_fewshot 时生效)')
    parser.add_argument('--use_fewshot', action='store_true',
                        help='忠实原文的 few-shot 设定: 每类按 --shot 抽样本(配合小 epoch 如 15)。'
                             '默认关闭=全量数据分析训练')
    parser.add_argument('--t', type=float, default=0.01, help='温度 t')
    parser.add_argument('--epsilon', type=float, default=0.3, help='OT 熵正则系数')
    parser.add_argument('--alpha', type=float, default=0.3, help='L_ot 对齐损失权重')
    parser.add_argument('--beta', type=float, default=0.5, help='L_neg 排斥损失权重')
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=str, default='2', choices=['2', '3', '2d', '2df', '3fc'], help="划分层级数：'2'=Family->Variant｜'3'=Maker->Family->Variant｜'2d'=从 L3 派生的两层划分（仅 aircraft，用于'层数消融'，产物落在 L2d/）；见 datasets/split_fgvc.py 分支 C")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = clip.load(args.arch, device=device)
    clip_model.eval()

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset, seed=args.seed, batch_size=args.batch_size,
        include_unseen=args.include_unseen, num_workers=4, preprocess=preprocess,
        num_layers=args.num_layers,
    )
    clean_classnames = build_clean_classnames(args, classnames)
    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_classnames[:3]}")
    print(f"-> Tail Names: {clean_classnames[-3:]}\n")

    # ---------------- 文本类别 prompt 特征 ----------------
    # 官方 prompt_templates[39] = "a photo of a {}." (微调时随文本尾部一起更新)
    prompt_template = "a photo of a {}."
    text_inputs = torch.cat([clip.tokenize(prompt_template.format(c)) for c in clean_classnames]).to(device)

    def compute_text_features():
        text_feat = clip_model.encode_text(text_inputs).to(torch.float32)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-5)
        return text_feat

    # ---------------- 训练数据: 默认全量分析 / --use_fewshot 走 few-shot ----------------
    feat = f"\n[DOT-OOD] 训练数据规模: "
    feat += f"few-shot 每类 {args.shot} 张(忠实原文, 建议 --epochs 15)" if args.use_fewshot else \
            "全量训练集(与仓库其它方法对齐, 50 epoch)"
    print(feat)
    if args.use_fewshot:
        train_dataset = loaders['train'].dataset
        subset = build_fewshot_subset(train_dataset, num_classes, args.shot, args.seed)
        dataloader_train = DataLoader(subset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=4,
                                      drop_last=num_classes > args.batch_size)
        print(f"[few-shot] 训练子集大小: {len(subset)} (每类目标 {args.shot} 张, 共 {num_classes} 类)")
    else:
        dataloader_train = loaders['train']  # 复用原始完整训练 loader(已含 train transform)
        print(f"[full] 使用完整训练集: {len(dataloader_train.dataset)} 样本")

    # ---------------- 冻结 / 微调尾部参数 ----------------
    # 官方只解冻 visual blocks 10,11 / visual.proj / text blocks 10,11 / text_projection
    for param in clip_model.parameters():
        param.requires_grad = False
    tail_names = ("visual.transformer.resblocks.10", "visual.transformer.resblocks.11",
                  "visual.proj", "transformer.resblocks.10", "transformer.resblocks.11",
                  "text_projection")
    for name, param in clip_model.named_parameters():
        if any(k in name for k in tail_names):
            param.requires_grad = True

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in clip_model.parameters())
    print(f"\n[Fine-tune] 可训练参数: {trainable}/{total} ({100*trainable/total:.2f}%)")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, clip_model.parameters()),
                                lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    method_name = "DOT-OOD"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"{method_name}_{unseen_tag}_weights.pth")

    # ---------------- 训练或加载权重 ----------------
    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的模型权重，直接加载: {model_path}")
        load_state_dict_into(clip_model, torch.load(model_path, map_location=device))
    else:
        print(f"\n[!] 开始 {method_name} [{unseen_tag} 设定, "
              f"{'few-shot='+str(args.shot) if args.use_fewshot else 'full-data'}, "
              f"epochs={args.epochs}](CLIP 尾部参数微调)...")
        clip_model.train()
        for epoch in range(args.epochs):
            total_loss = total_ot = total_neg = 0.0
            n_batch = 0
            for images, labels in dataloader_train:
                images, labels = images.to(device), labels.to(device)

                image_input = clip_model.visual(images.type(clip_model.dtype)).to(torch.float32)
                image_features = image_input / (image_input.norm(dim=-1, keepdim=True) + 1e-5)

                # 文本特征每 batch 重算 (文本尾部也在微调, 随模型更新)
                text_features = compute_text_features()

                logits = (image_features @ text_features.T) / args.t

                # ---- L_ot (图文概念匹配对齐) + L_neg (负样本排斥) ----
                loss_ot = torch.zeros((), device=device)
                loss_neg = torch.zeros((), device=device)
                bsz = image_features.shape[0]
                for i in range(bsz):
                    lab = labels[i]
                    align = optimal_transport(image_features[i], text_features[lab], args.epsilon)
                    loss_ot = loss_ot + (1 - align @ text_features[lab])
                    # 批内其它类别图像特征作负样本
                    mask = labels != lab
                    neg_feats = image_features[mask]
                    if neg_feats.shape[0] > 0:
                        loss_neg = loss_neg + torch.sum(torch.abs(align @ neg_feats.T)) / neg_feats.shape[0]
                loss_ot = loss_ot / bsz
                loss_neg = loss_neg / bsz

                loss = criterion(logits, labels) + args.alpha * loss_ot + args.beta * loss_neg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_ot += loss_ot.item()
                total_neg += loss_neg.item()
                n_batch += 1

            if n_batch > 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Loss {total_loss/n_batch:.4f} | "
                      f"L_ot {total_ot/n_batch:.4f} | L_neg {total_neg/n_batch:.4f}")
            scheduler.step()

        torch.save(state_dict_of(clip_model), model_path)
        print(f"\n[+] 训练完成！模型权重已永久保存至: {model_path}")

    # ================= 严谨评估模块 =================
    print("\n" + "=" * 50)
    print(f"STARTING STANDALONE {method_name} EVALUATION")
    print("=" * 50)
    clip_model.eval()

    def collect_scores(loader):
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            text_features = compute_text_features()  # 用最终微调后的文本编码器
            for images, targets in loader:
                images = images.to(device)
                image_features = clip_model.visual(images.type(clip_model.dtype)).to(torch.float32)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-5)
                logits = (image_features @ text_features.T) / args.t
                scores, preds = torch.max(logits, dim=1)  # MLS 打分 (越大越像已知)
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if hasattr(targets, 'numpy'):
                    all_targets.extend(targets.numpy())
                else:
                    all_targets.extend(np.asarray(targets))
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
    )


if __name__ == "__main__":
    main()
