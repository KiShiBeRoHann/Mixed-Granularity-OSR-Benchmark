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
import random
import torch.backends.cudnn as cudnn

# =========================================================
# 导入统一的数据工厂和插件
# =========================================================
from datasets import get_mixed_granularity_loaders
from utils import calculate_cvr_unr, calculate_oscr, state_dict_of, load_state_dict_into

# =========================================================
# 0. 绝对严谨的随机种子锁死 (确保 Baseline 的完美复现)
# =========================================================
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =========================================================
# 1. CoCoOp 核心组件构建
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
        # prompts: [Batch * N_cls, Seq_Len, Dim]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # tokenized_prompts 是 [N_cls, L]
        # 但是现在的 x 是 [Batch * N_cls, L, D]
        # 我们需要重复 tokenized_prompts 的 EOS 索引以匹配 batch
        N_cls = tokenized_prompts.shape[0]
        B_times_N = x.shape[0]
        B = B_times_N // N_cls
        
        eos_indices = tokenized_prompts.argmax(dim=-1) # [N_cls]
        eos_indices = eos_indices.repeat(B)            # [Batch * N_cls]
        
        x = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection
        return x

class CoCoOpPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        dtype = clip_model.dtype

        # 1. 静态 Context Vectors
        print(f"Initializing generic context with {n_ctx} learnable tokens...")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # 2. Meta-Net (用于生成动态 conditional shift)
        # 将视觉特征映射到 context 维度
        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 16, ctx_dim)
        ).type(dtype)

        # 3. 准备类名和 Token
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"{'X ' * n_ctx} {name}." for name in classnames]
        
        device = clip_model.token_embedding.weight.device
        tokenized_prompts = clip.tokenize(prompts).to(device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # Class Name, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self, image_features):
        # image_features: [B, vis_dim]
        B = image_features.shape[0]
        
        # 静态 ctx: [n_ctx, ctx_dim]
        ctx = self.ctx
        
        # 动态 shift: [B, ctx_dim]
        delta = self.meta_net(image_features)
        
        # 加起来变成动态 ctx: [B, n_ctx, ctx_dim]
        # 公式: v_i(x) = v_i + pi(x)
        dynamic_ctx = ctx.unsqueeze(0) + delta.unsqueeze(1) 
        
        # 扩展到所有类: [B, N_cls, n_ctx, ctx_dim]
        dynamic_ctx = dynamic_ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        # 扩展 prefix 和 suffix 以匹配 Batch
        prefix = self.token_prefix.unsqueeze(0).expand(B, -1, -1, -1) # [B, N_cls, 1, ctx_dim]
        suffix = self.token_suffix.unsqueeze(0).expand(B, -1, -1, -1) # [B, N_cls, *, ctx_dim]

        # 拼接: [B, N_cls, Seq_len, ctx_dim]
        prompts = torch.cat([prefix, dynamic_ctx, suffix], dim=2)
        
        # 展平以便通过 Text Encoder: [B * N_cls, Seq_len, ctx_dim]
        prompts = prompts.view(B * self.n_cls, prompts.shape[2], prompts.shape[3])
        return prompts

class CustomCLIP_CoCoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CoCoOpPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        # 1. 提取图像特征
        image_features = self.image_encoder(image.type(self.dtype))
        
        # 2. 生成动态 Prompts
        prompts = self.prompt_learner(image_features)
        
        # 3. 提取动态文本特征
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        
        # 4. 规范化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 5. 计算相似度
        B = image_features.shape[0]
        N_cls = self.prompt_learner.n_cls
        
        # text_features 是 [B * N_cls, Dim]，需要变回 [B, N_cls, Dim]
        text_features = text_features.view(B, N_cls, -1)
        
        # image_features 是 [B, Dim]，升维变成 [B, 1, Dim]
        image_features = image_features.unsqueeze(1)
        
        # 点积: [B, N_cls]
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_features * text_features).sum(dim=-1)
        
        return logits

# =========================================================
# 2. 主训练与评估逻辑
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="CoCoOp Training and Evaluation")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--include_unseen', action='store_true', help="开启 Oracle 模式 (将未见类加入训练)")
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3], help='划分层级数：2层(Family->Variant) 或 3层(Maker->Family->Variant)')
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

    # 动态获取当前 Seed 下的数据流与类名
    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        include_unseen=args.include_unseen,
        far_ood_mode='fine',
        preprocess=preprocess,
        num_layers=args.num_layers
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


    model = CustomCLIP_CoCoOp(classnames=clean_classnames, clip_model=clip_model).to(device)
    
    # 多卡并行：检测到 2+ 张 GPU 时自动启用 DataParallel（单卡无影响）
    if torch.cuda.device_count() > 1:
        print(f"\n[+] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)
    # DataParallel 不转发属性访问，内部子模块统一通过 inner 取
    inner = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = optim.SGD(inner.prompt_learner.parameters(), lr=0.002, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    method_name = "CoCoOp"
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
            inner.image_encoder.eval() 
            
            total_loss = 0
            for images, labels in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                # 防爆护盾
                torch.nn.utils.clip_grad_norm_(inner.prompt_learner.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
            scheduler.step()
        
        torch.save(state_dict_of(model), model_path)
        print(f"\n[+] 训练完成！模型权重已保存至: {model_path}")

    # =========================================================
    #  严谨评估模块 
    # =========================================================
    print("\n" + "="*50)
    print(f"STARTING STANDALONE {method_name} DEDICATED EVALUATION")
    print("="*50)

    def collect_scores(loader):
        model.eval()
        all_scores = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                logits = model(images)
                
                # CoOp 保持其原生优势，使用 MLS (最大 Logit) 算分
                scores, preds = torch.max(logits, dim=1) 
                
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                if isinstance(targets, torch.Tensor):
                    all_targets.extend(targets.cpu().numpy())
                else:
                    all_targets.extend(targets)
        return np.array(all_scores), np.array(all_preds), np.array(all_targets)

    # 采集基础闭集分数
    id_scores, id_preds, id_targets = collect_scores(loaders['test_id'])
    pos_scores_strict = id_scores 

    # 通用辅助函数
    def safe_acc(preds, targets):
        return f"{100 * (preds == targets).mean():.2f}%" if len(targets) > 0 else "N/A"

    def safe_auroc(y_true, y_scores):
        if len(np.unique(y_true)) < 2: return 0.0
        return roc_auc_score(y_true, y_scores) * 100

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

        c_gen_acc_str = safe_acc(c_gen_preds, c_gen_targets)
        m_gen_acc_str = safe_acc(m_gen_preds, m_gen_targets)

        near_fam_auroc = get_auroc(near_fam_scores)
        near_var_auroc = get_auroc(near_var_scores)
        far_auroc = get_auroc(far_scores)
        
        all_neg = np.concatenate([near_fam_scores, near_var_scores, far_scores])
        global_auroc = get_auroc(all_neg)

        print(f"\n{'='*15} {method_name} Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {c_gen_acc_str}")
        print(f"Medium Gen Acc (Family): {m_gen_acc_str}")
        print("-" * 45)
        print(f"Strict Global AUROC:       {global_auroc}")
        print(f"Near-Family AUROC (跨族):  {near_fam_auroc}")
        print(f"Near-Variant AUROC (同族): {near_var_auroc}")
        print(f"Strict Far AUROC (隔离):   {far_auroc}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="COCOOP")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="COCOOP")

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[COCOOP] OSCR 综合指标评估")
        
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[COCOOP] Global OSCR: {global_oscr:.2f}%")
        
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=c_gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_var_scores
            )
            print(f"[COCOOP] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------


    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = collect_scores(loaders['test_coarse_gen'])
        near_scores, _, _ = collect_scores(loaders['test_near_ood'])
        far_scores, _, _ = collect_scores(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0
        
        gen_acc_str = safe_acc(gen_preds, gen_targets)

        near_auroc = get_auroc(near_scores)
        far_auroc = get_auroc(far_scores)
        
        all_neg = np.concatenate([near_scores, far_scores])
        global_auroc = get_auroc(all_neg)

        status = "Seen" if args.include_unseen else "Unseen"
        
        print(f"\n{'='*15} {method_name} Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc ({status}): {gen_acc_str}")
        print("-" * 45)
        print(f"Strict Global AUROC: {global_auroc}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="COCOOP") 

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[COCOOP] OSCR 综合指标评估")
        
        global_id_scores = np.concatenate([id_scores, gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[COCOOP] Global OSCR: {global_oscr:.2f}%")
        
        if len(gen_scores) > 0 and len(near_scores) > 0:
            mg_correct_mask = (gen_preds == gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_scores
            )
            print(f"[COCOOP] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------

if __name__ == "__main__":
    main()