import json
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import clip

# 导入统一的数据工厂和插件
from datasets import get_mixed_granularity_loaders
from utils import evaluate_mixed_metrics, calculate_oscr, calculate_cvr_unr, state_dict_of, load_state_dict_into, view_save_dir

def set_seed(seed=1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

# =========================================================
# 1. CoOp 基础组件 (原汁原味的 FP16)
# =========================================================
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype

        # 恢复为原生 dtype (FP16)
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"{'X ' * n_ctx} {name}." for name in classnames]
        device = clip_model.token_embedding.weight.device
        tokenized_prompts = clip.tokenize(prompts).to(device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

# =========================================================
# 2. A2Pt 核心模块 (CGA)
# =========================================================
class CGA(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.q_img = nn.Linear(d_model, d_model)
        self.k_img = nn.Linear(d_model, d_model)
        self.v_img = nn.Linear(d_model, d_model)

        self.q_en = nn.Linear(d_model, d_model)
        self.k_txt = nn.Linear(d_model, d_model)
        self.v_en = nn.Linear(d_model, d_model)

    def forward(self, f_img, f_txt):
        B, C = f_img.shape
        N = f_txt.shape[0]

        f_img_exp = f_img.unsqueeze(1).expand(-1, N, -1)

        q_i = self.q_img(f_img_exp)
        k_i = self.k_img(f_img_exp)
        v_i = self.v_img(f_img_exp)
        attn_img = torch.matmul(q_i, k_i.transpose(-2, -1)) / (C ** 0.5)
        M_e = F.softmax(attn_img, dim=-1)
        f_en = torch.matmul(M_e, v_i) + f_img_exp

        q_e = self.q_en(f_en)
        k_t = self.k_txt(f_txt).unsqueeze(0).expand(B, -1, -1)
        v_e = self.v_en(f_en)
        attn_cross = torch.matmul(q_e, k_t.transpose(-2, -1)) / (C ** 0.5)
        M_s = F.softmax(attn_cross, dim=-1)
        
        f_tar = torch.matmul(M_s, v_e) + f_en
        return f_tar, f_img_exp

# =========================================================
# 3. A2Pt 模型整合
# =========================================================
class CustomCLIP_A2Pt(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)

        # 同样和 CLIP 保持一致的精度
        self.cga = CGA(d_model=clip_model.ln_final.weight.shape[0]).to(self.dtype)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        # 加上 1e-5 保险
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-5)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-5)

        f_tar, f_img_exp = self.cga(image_features, text_features)

        f_tar_norm = f_tar / (f_tar.norm(dim=-1, keepdim=True) + 1e-5)
        logits_tar = (f_tar_norm * text_features.unsqueeze(0)).sum(dim=-1) * self.logit_scale.exp()

        if not self.training:
            return logits_tar

        f_ass = f_img_exp - f_tar
        f_ass_norm = f_ass / (f_ass.norm(dim=-1, keepdim=True) + 1e-5)
        logits_ass = (f_ass_norm * text_features.unsqueeze(0)).sum(dim=-1) * self.logit_scale.exp()

        return logits_tar, f_tar_norm, f_ass_norm, logits_ass

def main():
    parser = argparse.ArgumentParser(description="A2Pt Training and Evaluation")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50) # 遵循设定，保持 50 epochs
    parser.add_argument('--include_unseen', action='store_true', help="开启 Oracle 模式 (计算 Seen Acc)")
    parser.add_argument('--num_layers', type=str, default='2', choices=['2', '3', '2d', '2df', '3fc'], help="划分层级数：'2'=Family->Variant｜'3'=Maker->Family->Variant｜'2d'=从 L3 派生的两层划分（仅 aircraft，用于'层数消融'，产物落在 L2d/）；见 datasets/split_fgvc.py 分支 C")
    parser.add_argument('--eval_view', type=str, default='V4', choices=['V1', 'V2', 'V3', 'V4'], help='评估靶场视图(默认 V4)：aircraft/imagenet 的 L3 支持 V1=L2语义/V2=+兄弟族/V3=+粗粒度泛化/V4=全量(imagenet 的 V1/V2 gen 集仅 50~500 张，需看 n_gen_medium)；其它数据集/层级只支持默认 V4')
    args = parser.parse_args()

    set_seed(1)
    g = torch.Generator()
    g.manual_seed(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP ViT-B/32...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for param in clip_model.parameters():
        param.requires_grad = False

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        include_unseen=args.include_unseen,
        far_ood_mode='fine',
        preprocess=preprocess,
        num_layers=args.num_layers,
        eval_view=args.eval_view
    )

    clean_classnames = []
    
    # =========================================================
    # 🎯 智能清洗逻辑：分数据集精确打击
    # =========================================================
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
        # 🚀 Aircraft 和 iNat 2021：类名去除层级标注括号（"Boeing 767 (Family)" -> "Boeing 767"）
        # 注意不能按空格切（多个 Family 会重复成 "Boeing"），只去末尾括号标注
        clean_classnames = [name.rsplit(' (', 1)[0].replace('_', ' ') for name in classnames]
        
    else:
        # 🚀 核心修复：CIFAR-100 绝对不能按空格切分！原样保留！
        clean_classnames = [name.replace('_', ' ') for name in classnames]

    # 🕵️‍♂️ 强力排错：检查最终用于构建 Prompt 的类名是否正确
    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_classnames[:3]}")
    print(f"-> Tail Names: {clean_classnames[-3:]}\n")


    model = CustomCLIP_A2Pt(classnames=clean_classnames, clip_model=clip_model).to(device)
    
    # 多卡并行：检测到 2+ 张 GPU 时自动启用 DataParallel（单卡无影响）
    if torch.cuda.device_count() > 1:
        print(f"\n[+] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)
    # DataParallel 不转发属性访问，内部子模块统一通过 inner 取
    inner = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = optim.SGD([
        {'params': inner.prompt_learner.parameters()},
        {'params': inner.cga.parameters()}
    ], lr=0.002, momentum=0.9)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    method_name = "A2Pt" 
    # 路径严格按照 dataset/seed 分级隔离
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    picture_dir = f"./pictures/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)
        
    # 保留 unseen 设定的权重隔离保护
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"{method_name}_{unseen_tag}_weights.pth")

    print(f"\nStart {method_name} Training for {args.epochs} epochs...")

    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的模型权重，跳过训练，直接加载: {model_path}")
        load_state_dict_into(model, torch.load(model_path, map_location=device))
    else:
        print(f"\n[!] 未发现缓存权重，开始 {method_name} [{unseen_tag} 设定] 的训练...")
        
        for epoch in range(args.epochs):
            model.train()
            inner.image_encoder.eval() 
            
            total_loss = 0
            for images, labels in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                logits_tar, f_tar, f_ass, logits_ass = model(images)
                
                loss_cls = criterion(logits_tar, labels)
                probs_ass = F.log_softmax(logits_ass, dim=1)
                loss_ass = - (1.0 / inner.n_cls) * probs_ass.sum(dim=1).mean()
                loss_div = (f_tar * f_ass).sum(dim=-1).mean()
                
                loss = loss_cls + 21.0 * loss_ass + 1.0 * loss_div
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
        
        torch.save(state_dict_of(model), model_path)
        print(f"\n[+] 训练完成！模型权重已永久保存至: {model_path}")

    # =========================================================
    # 3. 严谨评估模块 (动态兼容双轨制 - A2Pt 专属版)
    # =========================================================
    print("\n" + "="*50)
    print("STARTING STANDALONE A2Pt DEDICATED EVALUATION")
    print("="*50)

    def collect_scores_a2pt(loader):
        model.eval()
        all_scores = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                outputs = model(images)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # 提取主分类头的 logits_tar
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=1)
                max_p, preds = torch.max(probs, dim=1)
                
                all_scores.extend(max_p.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if isinstance(targets, torch.Tensor):
                    all_targets.extend(targets.cpu().numpy())
                else:
                    all_targets.extend(targets)
        return np.array(all_scores), np.array(all_preds), np.array(all_targets)

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
        score_map[key] = collect_scores_a2pt(ldr) if (ldr is not None and len(ldr.dataset) > 0) else None

    evaluate_mixed_metrics(
        score_map,
        method_name="A2Pt",
        mode=('Unseen_In_Train' if getattr(args, 'include_unseen', False) else 'Pure_Closed_Set'),
        tpr_target=0.95,
        save_dir=view_save_dir(model_dir, args.eval_view),
        extra_meta={'dataset': args.dataset, 'num_layers': args.num_layers,
                    'seed': args.seed, 'eval_view': args.eval_view},
    )


    

    
if __name__ == "__main__":
    main()