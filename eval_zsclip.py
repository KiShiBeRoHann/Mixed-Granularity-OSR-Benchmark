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

from utils import evaluate_mixed_metrics, calculate_cvr_unr, calculate_oscr

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
    # ---------------------------------------------------------
    # 统一多粒度开集评估（合并 gen / near，单次 CVRR-UNRR-H，仅 Global AUROC）
    # ---------------------------------------------------------
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
        method_name="ZeroShotCLIP",
        tpr_target=0.95,
        save_dir=None,  # eval_zsclip 原代码未定义 model_dir；不写盘，避免 NameError
    )

if __name__ == '__main__':
    main()