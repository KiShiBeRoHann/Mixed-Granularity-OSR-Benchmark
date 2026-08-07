import json
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from sklearn.metrics import roc_auc_score
from datasets import get_mixed_granularity_loaders
from utils import calculate_cvr_unr,calculate_oscr, state_dict_of, load_state_dict_into

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class LinearProbeCLIP(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype  # 提取 CLIP 原生的精度 (通常是 FP16)
        self.logit_scale = clip_model.logit_scale # 保留原生温度系数
        vis_dim = clip_model.visual.output_dim
        self.weight = nn.Parameter(torch.empty(num_classes, vis_dim, dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, image):
        with torch.no_grad():
            # 1. 在 CLIP 内部使用原生的 FP16 提取特征，不报错不炸显存
            feat = self.image_encoder(image.type(self.dtype))
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-5)
            
            # 2. 特征出炉后，立刻升维转换成 float32，切断 NaN 传染链！
            feat = feat.float()
            
        # 3. 分类头权重归一化与矩阵乘法，完全在安全的 float32 空间下进行
        weight_norm = self.weight / (self.weight.norm(dim=-1, keepdim=True) + 1e-5)
        
        # 确保温度系数在乘法时也是 float32
        logits = self.logit_scale.float().exp() * feat @ weight_norm.t()
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3], help='划分层级数：2层(Family->Variant) 或 3层(Maker->Family->Variant)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset, seed=args.seed, batch_size=args.batch_size, 
        include_unseen=args.include_unseen, num_workers=4, preprocess=preprocess, num_layers=args.num_layers
    )
    clean_classnames = []
    
    if args.dataset == 'imagenet':
        from nltk.corpus import wordnet as wn
        for name in classnames:
            raw_id = name.split(' ')[0] # 强行切掉 (C) 或 (F)
            try:
                pos = raw_id[0]
                offset = int(raw_id[1:])
                synset = wn.synset_from_pos_and_offset(pos, offset)
                clean_classnames.append(synset.lemmas()[0].name().replace('_', ' '))
            except Exception:
                clean_classnames.append(raw_id.replace('_', ' '))
                
    elif args.dataset in ['aircraft', 'inaturalist']:
        # 🚀 Aircraft 和 iNat 2021：保留完整层级类名（如 "Boeing 767 (Family)"），
        # 不能切空格！否则多个 Family 会被切成同一个 "Boeing"，类名重复导致评估错位
        clean_classnames = [name.replace('_', ' ') for name in classnames]
        
    else:
        # 🚀 核心修复：CIFAR-100 绝对不能按空格切分！原样保留！
        clean_classnames = [name.replace('_', ' ') for name in classnames]

    # 🕵️‍♂️ 强力排错：检查最终用于构建 Prompt 的类名是否正确
    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_classnames[:3]}")
    print(f"-> Tail Names: {clean_classnames[-3:]}\n")

    model = LinearProbeCLIP(clip_model=clip_model, num_classes=num_classes).to(device)
    
    # 多卡并行：检测到 2+ 张 GPU 时自动启用 DataParallel（单卡无影响）
    if torch.cuda.device_count() > 1:
        print(f"\n[+] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    method_name = "LinearProbeCLIP"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
        
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"{method_name}_{unseen_tag}_weights.pth")

    # 训练或加载权重逻辑
    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的模型权重，直接加载: {model_path}")
        load_state_dict_into(model, torch.load(model_path, map_location=device))
    else:
        print(f"\n[!] 未发现缓存权重，开始 {method_name} [{unseen_tag} 设定] 的训练...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for images, targets in loaders['train']:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
        torch.save(state_dict_of(model), model_path)
        print(f"\n[+] 训练完成！模型权重已永久保存至: {model_path}")
    
# ================= 严谨评估模块 =================
    print("\n" + "="*50)
    print("STARTING STANDALONE Linear Probe EVALUATION")
    print("="*50)

    def collect_scores(loader):
        model.eval()
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device))
                scores, preds = torch.max(logits, dim=1) # 使用 MLS 算分
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if isinstance(targets, torch.Tensor): all_targets.extend(targets.cpu().numpy())
                else: all_targets.extend(targets)
        return np.array(all_scores), np.array(all_preds), np.array(all_targets)

    id_scores, id_preds, id_targets = collect_scores(loaders['test_id'])
    pos_scores_strict = id_scores 

    def safe_acc(preds, targets): return f"{100 * (preds == targets).mean():.2f}%" if len(targets) > 0 else "N/A"
    def safe_auroc(y_true, y_scores): return roc_auc_score(y_true, y_scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0
    def get_auroc(neg_scores):
        if len(neg_scores) == 0: return "N/A"
        y_true = np.concatenate([np.ones(len(pos_scores_strict)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores_strict, neg_scores])
        return f"{safe_auroc(y_true, y_scores):.2f}%"

    if args.dataset in ['aircraft', 'imagenet', 'inaturalist']:
        c_gen_scores, c_gen_preds, c_gen_targets = collect_scores(loaders['test_coarse_gen'])
        m_gen_scores, m_gen_preds, m_gen_targets = collect_scores(loaders['test_medium_gen'])
        near_fam_scores, _, _ = collect_scores(loaders['test_near_family'])
        near_var_scores, _, _ = collect_scores(loaders['test_near_variant'])
        far_scores, _, _ = collect_scores(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, c_gen_preds, m_gen_preds])
        total_id_targets = np.concatenate([id_targets, c_gen_targets, m_gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0

        print(f"\n{'='*15} Linear Probe Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {safe_acc(c_gen_preds, c_gen_targets)}")
        print(f"Medium Gen Acc (Family): {safe_acc(m_gen_preds, m_gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC:       {get_auroc(np.concatenate([near_fam_scores, near_var_scores, far_scores]))}")
        print(f"Near-Family AUROC (跨族):  {get_auroc(near_fam_scores)}")
        print(f"Near-Variant AUROC (同族): {get_auroc(near_var_scores)}")
        print(f"Strict Far AUROC (隔离):   {get_auroc(far_scores)}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="LinearProbeCLIP")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="LinearProbeCLIP")

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[LinearProbeCLIP] OSCR 综合指标评估")
        
        all_neg = np.concatenate([near_fam_scores, near_var_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        
        # 计算全局正确性布尔数组
        global_correct_mask = (total_id_preds == total_id_targets)
        
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[LinearProbeCLIP] Global OSCR: {global_oscr:.2f}%")
        
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            # 计算混合粒度的正确性布尔数组
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            
            mg_oscr = calculate_oscr(
                pred_k_id=c_gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_var_scores
            )
            print(f"[LinearProbeCLIP] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------

    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = collect_scores(loaders['test_coarse_gen'])
        near_scores, _, _ = collect_scores(loaders['test_near_ood'])
        far_scores, _, _ = collect_scores(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0
        
        print(f"\n{'='*15} Linear Probe Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc: {safe_acc(gen_preds, gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC: {get_auroc(np.concatenate([near_scores, far_scores]))}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="LinearProbeCLIP") 

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[LinearProbeCLIP] OSCR 综合指标评估")
        
        all_neg = np.concatenate([near_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, gen_scores])
        
        # 计算全局正确性布尔数组
        global_correct_mask = (total_id_preds == total_id_targets)
        
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[LinearProbeCLIP] Global OSCR: {global_oscr:.2f}%")
        
        if len(gen_scores) > 0 and len(near_scores) > 0:
            # 计算混合粒度的正确性布尔数组
            mg_correct_mask = (gen_preds == gen_targets)
            
            mg_oscr = calculate_oscr(
                pred_k_id=gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_scores
            )
            print(f"[LinearProbeCLIP] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------

if __name__ == '__main__':
    main()