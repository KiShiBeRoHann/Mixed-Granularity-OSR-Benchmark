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
from utils import calculate_oscr, calculate_cvr_unr, state_dict_of, load_state_dict_into

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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
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

    optimizer = optim.SGD([
        {'params': model.prompt_learner.parameters()},
        {'params': model.cga.parameters()}
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
            model.image_encoder.eval() 
            
            total_loss = 0
            for images, labels in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                logits_tar, f_tar, f_ass, logits_ass = model(images)
                
                loss_cls = criterion(logits_tar, labels)
                probs_ass = F.log_softmax(logits_ass, dim=1)
                loss_ass = - (1.0 / model.n_cls) * probs_ass.sum(dim=1).mean()
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

    # 采集基础闭集分数
    id_scores, id_preds, id_targets = collect_scores_a2pt(loaders['test_id'])
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
        c_gen_scores, c_gen_preds, c_gen_targets = collect_scores_a2pt(loaders['test_coarse_gen'])
        m_gen_scores, m_gen_preds, m_gen_targets = collect_scores_a2pt(loaders['test_medium_gen'])
        near_fam_scores, _, _ = collect_scores_a2pt(loaders['test_near_family'])
        near_var_scores, _, _ = collect_scores_a2pt(loaders['test_near_variant'])
        far_scores, _, _ = collect_scores_a2pt(loaders['test_far_ood'])

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

        print(f"\n{'='*15} A2Pt Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {c_gen_acc_str}")
        print(f"Medium Gen Acc (Family): {m_gen_acc_str}")
        print("-" * 45)
        print(f"Strict Global AUROC:       {global_auroc}")
        print(f"Near-Family AUROC (跨族):  {near_fam_auroc}")
        print(f"Near-Variant AUROC (同族): {near_var_auroc}")
        print(f"Strict Far AUROC (隔离):   {far_auroc}")

        # 计算并打印阈值依赖指标
        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="A2Pt")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="A2Pt")

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"OSCR 综合指标评估")
        
        # 1. Global OSCR
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"Global OSCR : {global_oscr:.2f}%")
        
        # 2. MG-OSCR (混合粒度边界对抗) - 取最容易混淆的 c_gen 和 near_var
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=c_gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_var_scores
            )
            print(f"MG-OSCR     : {mg_oscr:.2f}%")
        # -------------------------------------------------------

        np.save(os.path.join(model_dir, "A2Pt_id_scores.npy"), id_scores)
        np.save(os.path.join(model_dir, "A2Pt_gen_scores.npy"), c_gen_scores)
        np.save(os.path.join(model_dir, "A2Pt_near_ood_scores.npy"), near_var_scores)
    
    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = collect_scores_a2pt(loaders['test_coarse_gen'])
        near_scores, _, _ = collect_scores_a2pt(loaders['test_near_ood'])
        far_scores, _, _ = collect_scores_a2pt(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0
        
        gen_acc_str = safe_acc(gen_preds, gen_targets)

        near_auroc = get_auroc(near_scores)
        far_auroc = get_auroc(far_scores)
        
        all_neg = np.concatenate([near_scores, far_scores])
        global_auroc = get_auroc(all_neg)

        status = "Seen" if args.include_unseen else "Unseen"
        
        print(f"\n{'='*15} A2Pt Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc ({status}): {gen_acc_str}")
        print("-" * 45)
        print(f"Strict Global AUROC: {global_auroc}")
        

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="A2Pt") 
        
        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"OSCR 综合指标评估")
        
        # 1. Global OSCR
        global_id_scores = np.concatenate([id_scores, gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"Global OSCR: {global_oscr:.2f}%")
        
        # 2. MG-OSCR (混合粒度边界对抗)
        if len(gen_scores) > 0 and len(near_scores) > 0:
            mg_correct_mask = (gen_preds == gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_scores
            )
            print(f"MG-OSCR: {mg_oscr:.2f}%")
        # -------------------------------------------------------
        
        np.save(os.path.join(model_dir, "A2Pt_id_scores.npy"), id_scores)
        np.save(os.path.join(model_dir, "A2Pt_gen_scores.npy"), gen_scores)      
        np.save(os.path.join(model_dir, "A2Pt_near_ood_scores.npy"), near_scores) 

    

    
if __name__ == "__main__":
    main()