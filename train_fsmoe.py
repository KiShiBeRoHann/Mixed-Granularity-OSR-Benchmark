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
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn

from datasets import get_mixed_granularity_loaders
from utils import calculate_cvr_unr,calculate_oscr

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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
        
        N_cls = tokenized_prompts.shape[0] 
        B_times_N = x.shape[0]             
        B = B_times_N // N_cls             
        
        eos_indices = tokenized_prompts.argmax(dim=-1) 
        eos_indices = eos_indices.repeat(B)            
        
        x = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection
        return x

class SparseMoE(nn.Module):
    def __init__(self, num_experts, in_dim, out_dim, top_k=5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # ==========================================
        # 🔌 强行将专家层设置为高精度 float32
        # ==========================================
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim // 4, out_dim)
            ).type(torch.float32) for _ in range(num_experts)
        ])
        
        self.router = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 4, num_experts)
        ).type(torch.float32)

    def forward(self, low_level_feats, high_level_feat):
        routing_logits = self.router(high_level_feat) 
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        B = high_level_feat.shape[0]
        D = high_level_feat.shape[1] 
        V_l = torch.zeros(B, D, dtype=torch.float32, device=high_level_feat.device)
        
        for i in range(B):
            for k_idx in range(self.top_k):
                expert_idx = topk_indices[i, k_idx].item()
                weight = topk_weights[i, k_idx]
                f_j = low_level_feats[expert_idx][i].unsqueeze(0)
                expert_out = self.experts[expert_idx](f_j)
                V_l[i] += weight * expert_out.squeeze(0)
                
        return V_l, routing_logits

class CustomCLIP_FSMoE(nn.Module):
    def __init__(self, classnames, clip_model, num_layers=12, prompt_len=8, top_k=5):
        super().__init__()
        self.n_cls = len(classnames)
        self.num_layers = num_layers
        self.dtype = clip_model.dtype
        self.prompt_dim = clip_model.ln_final.weight.shape[0] 
        self.vis_hidden_dim = clip_model.visual.positional_embedding.shape[1] 
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        # ==========================================
        # 🔌 核心修复：将可学习 Prompt 强行设置为 FP32
        # ==========================================
        self.P_l = nn.Parameter(torch.empty(prompt_len, self.prompt_dim, dtype=torch.float32))
        self.P_h = nn.Parameter(torch.empty(prompt_len, self.prompt_dim, dtype=torch.float32))
        nn.init.normal_(self.P_l, std=0.02)
        nn.init.normal_(self.P_h, std=0.02)
        
        self.moe = SparseMoE(num_experts=num_layers-1, in_dim=self.vis_hidden_dim, out_dim=self.prompt_dim, top_k=top_k)
        
        self.high_level_proj = nn.Sequential(
            nn.Linear(self.prompt_dim, self.prompt_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.prompt_dim // 4, self.prompt_dim)
        ).type(torch.float32)
        
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"{'X ' * (2 * prompt_len)} {name}." for name in classnames]
        device = clip_model.token_embedding.weight.device
        self.tokenized_prompts = clip.tokenize(prompts).to(device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(torch.float32)
            
        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1 + 2 * prompt_len:, :])

        # 中间层特征不再在 __init__ 静态注册 hook：
        # 静态闭包 hook 在 DataParallel 深拷贝复制模型时会绑定到原模型实例，
        # 导致 replica 上收集不到特征（low_level_feats 为空）。改为 forward 内动态注册。

    def _visual_forward_with_intermediate(self, x):
        """显式遍历 CLIP 视觉 transformer 并收集中间层特征。
        不依赖 forward hook，避免 DataParallel 多卡下 hook 闭包/设备错位问题。"""
        v = self.image_encoder
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        intermediate = []
        for i, block in enumerate(v.transformer.resblocks):
            x = block(x)
            if i < self.num_layers - 1:
                intermediate.append(x.permute(1, 0, 2)[:, 0, :])
        x = x.permute(1, 0, 2)
        # 兼容不同版本 CLIP 的 ln_post / pooler 调用
        if hasattr(v, 'pooler') and v.pooler is not None:
            x = v.ln_post(x[:, 0], v.pooler)
        else:
            x = v.ln_post(x[:, 0])
        if v.proj is not None:
            x = x @ v.proj
        return x, intermediate

    def forward(self, image):
        # FP16 提取特征（显式 forward 收集中间层特征，兼容 DataParallel 多卡）
        F_L, intermediate = self._visual_forward_with_intermediate(image.type(self.dtype))
        B = F_L.shape[0]
        
        # ==========================================
        # 🔌 将所有视觉特征升维到 FP32，防止运算下溢
        # ==========================================
        F_L_32 = F_L.float()
        low_level_feats_32 = [f.float() for f in intermediate]
        
        V_l, routing_logits = self.moe(low_level_feats_32, F_L_32)
        V_h = self.high_level_proj(F_L_32)
        
        P_l_prime = self.P_l.unsqueeze(0) + V_l.unsqueeze(1) 
        P_h_prime = self.P_h.unsqueeze(0) + V_h.unsqueeze(1) 
        
        prefix = self.token_prefix.unsqueeze(0).expand(B, -1, -1, -1) 
        suffix = self.token_suffix.unsqueeze(0).expand(B, -1, -1, -1)
        
        P_l_exp = P_l_prime.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        P_h_exp = P_h_prime.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        
        prompts = torch.cat([prefix, P_l_exp, P_h_exp, suffix], dim=2)
        prompts = prompts.view(B * self.n_cls, prompts.shape[2], prompts.shape[3])
        
        # 降维回 FP16 喂给文本编码器
        text_features = self.text_encoder(prompts.type(self.dtype), self.tokenized_prompts)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-5)
        text_features = text_features.view(B, self.n_cls, -1)
        
        image_features_norm = F_L / (F_L.norm(dim=-1, keepdim=True) + 1e-5)
        image_features_norm = image_features_norm.unsqueeze(1)
        
        logits = self.logit_scale.exp() * (image_features_norm * text_features).sum(dim=-1)
        
        if self.training:
            return logits, routing_logits
        return logits

def routing_consistency_loss(routing_logits, labels):
    device = routing_logits.device
    B = routing_logits.shape[0]
    routing_norm = F.normalize(routing_logits, p=2, dim=1)
    sim_matrix = torch.matmul(routing_norm, routing_norm.t())
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(B).view(-1, 1).to(device), 0)
    mask = mask * logits_mask
    if mask.sum() == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
    return -mean_log_prob_pos.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3])
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # 🚀 Aircraft 和 iNat 2021 完美合并：直接切掉括号并替换下划线
        # 例如: "Eurybia_divaricata (F)" -> "Eurybia divaricata"
        clean_classnames = [name.split(' ')[0].replace('_', ' ') for name in classnames]
        
    else:
        # 🚀 核心修复：CIFAR-100 绝对不能按空格切分！原样保留！
        clean_classnames = [name.replace('_', ' ') for name in classnames]

    # 🕵️‍♂️ 强力排错：检查最终用于构建 Prompt 的类名是否正确
    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_classnames[:3]}")
    print(f"-> Tail Names: {clean_classnames[-3:]}\n")

    model = CustomCLIP_FSMoE(classnames=clean_classnames, clip_model=clip_model).to(device)
    
    # 多卡并行：检测到 2+ 张 GPU 时自动启用 DataParallel（单卡无影响）
    if torch.cuda.device_count() > 1:
        print(f"\n[+] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)
    
    for name, param in model.named_parameters():
        if "P_l" not in name and "P_h" not in name and "moe" not in name and "high_level_proj" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()


    method_name = "FSMOE"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
        
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"{method_name}_{unseen_tag}_weights.pth")

    # 训练或加载权重逻辑
    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的模型权重，直接加载: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"\n[!] 未发现缓存权重，开始 {method_name} [{unseen_tag} 设定] 的训练...")
        for epoch in range(args.epochs):
            model.train()
            for images, targets in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits, routing_logits = model(images)
                
                loss_ce = criterion(logits, targets)
                loss_route = routing_consistency_loss(routing_logits, targets)
                loss = loss_ce + 0.1 * loss_route 
                
                loss.backward()
                optimizer.step()
    
        torch.save(model.state_dict(), model_path)
        print(f"\n[+] 训练完成！模型权重已永久保存至: {model_path}")

# ================= 严谨评估模块 =================
    print("\n" + "="*50)
    print("STARTING STANDALONE FSMoE EVALUATION")
    print("="*50)

    def collect_scores(loader):
        model.eval()
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device))
                scores, preds = torch.max(logits, dim=1) 
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

        print(f"\n{'='*15} FSMoE Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {safe_acc(c_gen_preds, c_gen_targets)}")
        print(f"Medium Gen Acc (Family): {safe_acc(m_gen_preds, m_gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC:       {get_auroc(np.concatenate([near_fam_scores, near_var_scores, far_scores]))}")
        print(f"Near-Family AUROC (跨族):  {get_auroc(near_fam_scores)}")
        print(f"Near-Variant AUROC (同族): {get_auroc(near_var_scores)}")
        print(f"Strict Far AUROC (隔离):   {get_auroc(far_scores)}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="FSMOE")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="FSMOE")

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[FSMOE] OSCR 综合指标评估")
        
        all_neg = np.concatenate([near_fam_scores, near_var_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[FSMOE] Global OSCR: {global_oscr:.2f}%")
        
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=c_gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_var_scores
            )
            print(f"[FSMOE] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------


    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = collect_scores(loaders['test_coarse_gen'])
        near_scores, _, _ = collect_scores(loaders['test_near_ood'])
        far_scores, _, _ = collect_scores(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0
        
        print(f"\n{'='*15} FSMoE Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc: {safe_acc(gen_preds, gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC: {get_auroc(np.concatenate([near_scores, far_scores]))}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="FSMOE") 

        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"[FSMOE] OSCR 综合指标评估")
        
        all_neg = np.concatenate([near_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"[FSMOE] Global OSCR: {global_oscr:.2f}%")
        
        if len(gen_scores) > 0 and len(near_scores) > 0:
            mg_correct_mask = (gen_preds == gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_scores
            )
            print(f"[FSMOE] MG-OSCR    : {mg_oscr:.2f}%")
        # -------------------------------------------------------
if __name__ == '__main__':
    main()