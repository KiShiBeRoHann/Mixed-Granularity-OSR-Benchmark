import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms

from datasets import get_mixed_granularity_loaders
from utils import calculate_cvr_unr, calculate_oscr, state_dict_of, load_state_dict_into

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_vgg32_backbone():
    model = models.vgg16_bn(pretrained=False)
    # 将特征层与全局自适应池化打包，确保无论输入多大，输出都是 [B, 512, 1, 1]
    backbone = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1))
    )
    return backbone, 512

class ARPL_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone, self.feat_dim = get_vgg32_backbone()
        self.R = nn.Parameter(torch.empty(num_classes, self.feat_dim))
        nn.init.normal_(self.R, std=0.02)
        self.dist_scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x).view(x.size(0), -1)
        feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-5)
        R_norm = self.R / (self.R.norm(dim=-1, keepdim=True) + 1e-5)
        logits = - (feat_norm @ R_norm.t()) * self.dist_scale
        if return_feat: return logits, feat_norm
        return logits

    def compute_logits_from_feat(self, feat_norm):
        R_norm = self.R / (self.R.norm(dim=-1, keepdim=True) + 1e-5)
        return - (feat_norm @ R_norm.t()) * self.dist_scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3], help='划分层级数：2层(Family->Variant) 或 3层(Maker->Family->Variant)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset, seed=args.seed, batch_size=args.batch_size, 
        include_unseen=args.include_unseen, num_workers=4, preprocess=None, num_layers=args.num_layers
    )

    if args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        # Aircraft / ImageNet / iNaturalist 的标准 224 处理
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # 核心修复：强制统一尺寸防堆叠报错
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    for k, loader in loaders.items():
        loader.dataset.transform = train_transform if 'train' in k else test_transform

    model = ARPL_VGG(num_classes=num_classes).to(device)
    
    # 多卡并行：检测到 2+ 张 GPU 时自动启用 DataParallel（单卡无影响）
    if torch.cuda.device_count() > 1:
        print(f"\n[+] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    method_name = "ARPL+CS"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
    unseen_tag = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"ARPL_{unseen_tag}_weights.pth")

    # 训练或加载权重逻辑
    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的模型权重，直接加载: {model_path}")
        load_state_dict_into(model, torch.load(model_path, map_location=device))
    else:
        print(f"\n[!] 未发现缓存权重，开始 {method_name} [{unseen_tag} 设定] 的训练...")
        for epoch in range(args.epochs):
            model.train()
            for images, labels in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                logits, feat = model(images, return_feat=True)
                loss_arpl = criterion(logits, labels)
                
                feat_adv = feat.detach().clone().requires_grad_(True)
                loss_adv_gen = criterion(model.compute_logits_from_feat(feat_adv), labels)
                grad = torch.autograd.grad(loss_adv_gen, feat_adv)[0]
                
                feat_cs = feat_adv + 0.1 * grad.sign()
                feat_cs = feat_cs / (feat_cs.norm(dim=-1, keepdim=True) + 1e-5)
                
                logits_cs = model.compute_logits_from_feat(feat_cs.detach())
                loss_cs = -torch.mean(torch.sum(F.log_softmax(logits_cs, dim=1), dim=1)) / num_classes
                
                loss = loss_arpl + 0.1 * loss_cs
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        torch.save(state_dict_of(model), model_path)
        print(f"\n[+] 训练完成！模型权重已永久保存至: {model_path}")

# ================= 严谨评估模块 =================
    print("\n" + "="*50)
    print(f"STARTING STANDALONE {method_name} EVALUATION")
    print("="*50)

    def collect_scores(loader):
        model.eval()
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device))
                scores, preds = torch.max(logits, dim=1) # ARPL 的 logit 是负距离，max_l 即为最匹配的置信度
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

        print(f"\n{'='*15} {method_name} Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {safe_acc(c_gen_preds, c_gen_targets)}")
        print(f"Medium Gen Acc (Family): {safe_acc(m_gen_preds, m_gen_targets)}")
        print("-" * 45)
        
        # 【修复1】：提前定义 all_neg，供下面复用
        all_neg = np.concatenate([near_fam_scores, near_var_scores, far_scores])
        print(f"Strict Global AUROC:       {get_auroc(all_neg)}")
        print(f"Near-Family AUROC (跨族):  {get_auroc(near_fam_scores)}")
        print(f"Near-Variant AUROC (同族): {get_auroc(near_var_scores)}")
        print(f"Strict Far AUROC (隔离):   {get_auroc(far_scores)}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="ARPL+CS")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="ARPL+CS")       
        
        # ------------------- 增加 OSCR 评估 -------------------
        print("-" * 45)
        print(f"OSCR 综合指标评估")
        
        # 1. Global OSCR
        # 【修复2】：这里应该是 c_gen 和 m_gen，而不是 gen_scores
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=global_correct_mask,
            pred_u_ood=all_neg
        )
        print(f"Global OSCR: {global_oscr:.2f}%")
        
        # 2. MG-OSCR (混合粒度边界对抗)
        # 【修复3】：使用细分数据集专有的变量名 c_gen 和 near_var
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            mg_oscr = calculate_oscr(
                pred_k_id=c_gen_scores,
                x_k_id=mg_correct_mask,
                pred_u_ood=near_var_scores
            )
            print(f"MG-OSCR: {mg_oscr:.2f}%")
        # -------------------------------------------------------

    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = collect_scores(loaders['test_coarse_gen'])
        near_scores, _, _ = collect_scores(loaders['test_near_ood'])
        far_scores, _, _ = collect_scores(loaders['test_far_ood'])

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0
        
        print(f"\n{'='*15} {method_name} Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc: {safe_acc(gen_preds, gen_targets)}")
        print("-" * 45)
        
        # 【修复4】：在打印前显式定义 all_neg
        all_neg = np.concatenate([near_scores, far_scores])
        print(f"Strict Global AUROC: {get_auroc(all_neg)}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="ARPL+CS") 
        
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


        # ---- 保存分数用于层次冲突可视化 ----
        np.save(os.path.join(model_dir, f"{method_name}_id_scores.npy"), id_scores)
        np.save(os.path.join(model_dir, f"{method_name}_gen_scores.npy"), gen_scores)
        np.save(os.path.join(model_dir, f"{method_name}_near_ood_scores.npy"), near_scores)
        np.save(os.path.join(model_dir, f"{method_name}_far_ood_scores.npy"), far_scores)
if __name__ == "__main__":
    main()