import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip
from sklearn.metrics import roc_auc_score
from datasets import get_mixed_granularity_loaders
import json
import os

from utils import calculate_cvr_unr, calculate_oscr

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3], help='划分层级数：2层(Family->Variant) 或 3层(Maker->Family->Variant)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Vanilla CLIP ViT-B/32...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        include_unseen=args.include_unseen,
        far_ood_mode='fine',
        preprocess=preprocess,
        num_layers=args.num_layers
    )

    # =========================================================
    # 🎯 智能清洗逻辑：同步我们在训练脚本里的最新版配置
    # =========================================================
    clean_names = []
    
    if args.dataset == 'imagenet':
        from nltk.corpus import wordnet as wn
        for name in classnames:
            raw_id = name.split(' ')[0] # 强行切掉 (C) 或 (F)
            try:
                pos = raw_id[0]
                offset = int(raw_id[1:])
                synset = wn.synset_from_pos_and_offset(pos, offset)
                clean_names.append(synset.lemmas()[0].name().replace('_', ' '))
            except Exception:
                clean_names.append(raw_id.replace('_', ' '))
                
    elif args.dataset in ['aircraft', 'inaturalist']:
        # 🚀 Aircraft 和 iNat 2021：类名去除层级标注括号（"Boeing 767 (Family)" -> "Boeing 767"）
        # 注意不能按空格切（多个 Family 会重复成 "Boeing"），只去末尾括号标注
        clean_names = [name.rsplit(' (', 1)[0].replace('_', ' ') for name in classnames]
        
    else:
        # 🚀 CIFAR-100 专属：绝对不能按空格切分！原样保留！
        clean_names = [name.replace('_', ' ') for name in classnames]

    # 🕵️‍♂️ 强力排错：检查最终用于构建 Prompt 的类名是否正确
    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_names[:3]}")
    print(f"-> Tail Names: {clean_names[-3:]}\n")

    # 构建文本提示
    prompts = [f"a photo of a {name}" for name in clean_names]
    
    # 🕵️‍♂️ 强力排错针：打印前3个和最后3个提示词，确保都是人类可读的英文！
    print(f"\n[Debug] Sample Prompts Head: {prompts[:3]}")
    print(f"[Debug] Sample Prompts Tail: {prompts[-3:]}\n")
    
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-5)

    print("\n" + "="*50)
    print("STARTING STANDALONE ZERO-SHOT CLIP EVALUATION")
    print("="*50)

    def collect_scores(loader):
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                image_features = clip_model.encode_image(images)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-5)
                
                logit_scale = clip_model.logit_scale.exp().item() 
                logits = (image_features @ text_features.t()) * logit_scale
                
                # Zero-Shot CLIP 常用 MLS 算分
                scores, preds = torch.max(logits, dim=1)
                
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if isinstance(targets, torch.Tensor): all_targets.extend(targets.cpu().numpy())
                else: all_targets.extend(targets)
        return np.array(all_scores), np.array(all_preds), np.array(all_targets)

    # 采集分数
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

        print(f"\n{'='*15} Zero-Shot CLIP Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {safe_acc(c_gen_preds, c_gen_targets)}")
        print(f"Medium Gen Acc (Family): {safe_acc(m_gen_preds, m_gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC:       {get_auroc(np.concatenate([near_fam_scores, near_var_scores, far_scores]))}")
        print(f"Near-Family AUROC (跨族):  {get_auroc(near_fam_scores)}")
        print(f"Near-Variant AUROC (同族): {get_auroc(near_var_scores)}")
        print(f"Strict Far AUROC (隔离):   {get_auroc(far_scores)}")
        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="A2Pt")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="A2Pt")

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[ZeroShotCLIP] OSCR 综合指标评估")
        
        all_neg = np.concatenate([near_fam_scores, near_var_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        
        # 计算全局正确性布尔数组
        global_correct_mask = (total_id_preds == total_id_targets)
        
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[ZeroShotCLIP] Global OSCR: {global_oscr:.2f}%")
        
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            # 计算混合粒度的正确性布尔数组
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            
            mg_oscr = calculate_oscr(
                pred_k_id=c_gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_var_scores
            )
            print(f"[ZeroShotCLIP] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------


    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = collect_scores(loaders['test_coarse_gen'])
        near_scores, _, _ = collect_scores(loaders['test_near_ood'])
        far_scores, _, _ = collect_scores(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0
        
        print(f"\n{'='*15} Zero-Shot CLIP Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc: {safe_acc(gen_preds, gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC: {get_auroc(np.concatenate([near_scores, far_scores]))}")
        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="A2Pt") 

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[ZeroShotCLIP] OSCR 综合指标评估")
        
        all_neg = np.concatenate([near_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, gen_scores])
        
        # 计算全局正确性布尔数组
        global_correct_mask = (total_id_preds == total_id_targets)
        
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[ZeroShotCLIP] Global OSCR: {global_oscr:.2f}%")
        
        if len(gen_scores) > 0 and len(near_scores) > 0:
            # 计算混合粒度的正确性布尔数组
            mg_correct_mask = (gen_preds == gen_targets)
            
            mg_oscr = calculate_oscr(
                pred_k_id=gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_scores
            )
            print(f"[ZeroShotCLIP] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------

if __name__ == '__main__':
    main()