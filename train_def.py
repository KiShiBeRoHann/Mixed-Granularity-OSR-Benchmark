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
from utils import calculate_cvr_unr

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        return -(mask * log_prob).sum(1).div(mask_pos_pairs).mean()

def get_etf_target(num_classes, device):
    target = torch.full((num_classes, num_classes), -1.0 / (num_classes - 1), device=device)
    target.fill_diagonal_(1.0)
    return target

class F_DEF_Tracker(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer('class_means', torch.randn(num_classes, feat_dim))
        self.class_means = F.normalize(self.class_means, p=2, dim=1)
    def update_means(self, features, labels):
        with torch.no_grad():
            for c in range(self.num_classes):
                mask_c = (labels == c)
                if mask_c.sum() > 0:
                    self.class_means[c] = self.momentum * self.class_means[c] + (1 - self.momentum) * features[mask_c].mean(dim=0)
            self.class_means = F.normalize(self.class_means, p=2, dim=1)
    def compute_f_def_loss(self, target_etf):
        mu_norm = F.normalize(self.class_means, p=2, dim=1)
        return F.mse_loss(torch.matmul(mu_norm, mu_norm.t()), target_etf)

def compute_c_def_loss(classifier_weight, target_etf):
    w_norm = F.normalize(classifier_weight, p=2, dim=1)
    return F.mse_loss(torch.matmul(w_norm, w_norm.t()), target_etf)

def get_vgg32_backbone():
    model = models.vgg16_bn(pretrained=False)
    backbone = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1))
    )
    return backbone, 512

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim)
        )
    def forward(self, x): return F.normalize(self.head(x), p=2, dim=1)

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
    print(f"Using device: {device}")

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset, seed=args.seed, batch_size=args.batch_size, 
        include_unseen=args.include_unseen, num_workers=4, preprocess=None, num_layers=args.num_layers
    )

    # ================= 智能自适应数据预处理 =================
    if args.dataset == 'cifar100':
        # CIFAR-100 专用 32x32 微缩处理
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
        # Aircraft / ImageNet / iNaturalist 标准 224x224 高清处理
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 高清图标准裁剪
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # 强制统一到 224，保证 eval 时 DataLoader 不炸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 将 transform 动态挂载到数据集上
    for k, loader in loaders.items():
        loader.dataset.transform = train_transform if 'train' in k else test_transform
    # =======================================================

    backbone, feat_dim = get_vgg32_backbone()
    backbone = backbone.to(device)
    proj_head = ProjectionHead(feat_dim, 128).to(device)
    
    target_etf = get_etf_target(num_classes, device)
    f_def_tracker = F_DEF_Tracker(num_classes, 128).to(device)
    supcon_criterion = SupConLoss(temperature=0.1)

    method_name = "DEF"
    model_dir = f"./models/{args.dataset}/L{args.num_layers}/seed_{args.seed}"
    os.makedirs(model_dir, exist_ok=True)
    status_str = "Unseen_In_Train" if args.include_unseen else "Pure_Closed_Set"
    model_path = os.path.join(model_dir, f"DeF_{status_str}_weights.pth")

    # 🚨 核心修改：必须在加载权重前提前初始化 classifier
    classifier = nn.Linear(feat_dim, num_classes, bias=False).to(device)

    print(f"\nStart DeF Training for {args.epochs if hasattr(args, 'epochs') else '200+20'} epochs...")

    if os.path.exists(model_path):
        print(f"\n[+] 发现已保存的模型权重，跳过训练，直接加载: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        backbone.load_state_dict(checkpoint['backbone'])
        classifier.load_state_dict(checkpoint['classifier'])
        
        # 加载完成后务必冻结 backbone 并切换至 eval 模式
        backbone.eval()
        for param in backbone.parameters(): param.requires_grad = False
        classifier.eval()
    else:
        print(f"\n[!] 未发现缓存权重，开始 DeF [{status_str} 设定] 的两阶段训练...")
        
        epochs_s1 = 200
        optimizer_s1 = optim.SGD(list(backbone.parameters()) + list(proj_head.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=epochs_s1)
        
        print(f"\n--- [Stage 1] Training Backbone with SupCon & F-DEF for {epochs_s1} epochs ---")
        for epoch in range(epochs_s1):
            backbone.train(); proj_head.train()
            for images, labels in tqdm(loaders['train'], desc=f"Stage 1 Epoch {epoch+1}/{epochs_s1}", leave=False):
                images, labels = images.to(device), labels.to(device)
                feat = backbone(images).view(images.size(0), -1)
                z = proj_head(feat)
                
                f_def_tracker.update_means(z, labels)
                loss_f_def = f_def_tracker.compute_f_def_loss(target_etf)
                loss_sup = supcon_criterion(z, labels)
                
                alpha = np.random.beta(1.0, 1.0)
                loss = alpha * loss_sup + (1 - alpha) * loss_f_def
                
                optimizer_s1.zero_grad()
                loss.backward()
                optimizer_s1.step()
            scheduler_s1.step()

        backbone.eval()
        for param in backbone.parameters(): param.requires_grad = False

        epochs_s2 = 20
        optimizer_s2 = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=epochs_s2)
        ce_criterion = nn.CrossEntropyLoss()

        print(f"\n--- [Stage 2] Training Classifier with CE & C-DEF for {epochs_s2} epochs ---")
        for epoch in range(epochs_s2):
            classifier.train()
            for images, labels in tqdm(loaders['train'], desc=f"Stage 2 Epoch {epoch+1}/{epochs_s2}", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad(): feat = backbone(images).view(images.size(0), -1)
                
                logits = classifier(feat)
                loss_ce = ce_criterion(logits, labels)
                loss_c_def = compute_c_def_loss(classifier.weight, target_etf)
                
                beta = np.random.beta(1.0, 1.0)
                loss = beta * loss_ce + (1 - beta) * loss_c_def
                
                optimizer_s2.zero_grad()
                loss.backward()
                optimizer_s2.step()
            scheduler_s2.step()

        # 打包保存
        state_to_save = {
            'backbone': backbone.state_dict(),
            'classifier': classifier.state_dict()
        }
        torch.save(state_to_save, model_path)
        print(f"\n[+] DeF 阶段二完成！Backbone 与 Classifier 权重已打包保存至: {model_path}")

    # ================= 严谨评估模块 =================
    print("\n" + "="*50)
    print(f"STARTING STANDALONE {method_name} EVALUATION")
    print("="*50)

    def collect_scores(loader):
        backbone.eval(); classifier.eval()
        all_scores, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for images, targets in loader:
                feat = backbone(images.to(device)).view(images.size(0), -1)
                logits = classifier(feat)
                probs = F.softmax(logits, dim=1) # DEF 测的是最大 Softmax 概率 (MSP)
                scores, preds = torch.max(probs, dim=1)
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
        print(f"Strict Global AUROC:       {get_auroc(np.concatenate([near_fam_scores, near_var_scores, far_scores]))}")
        print(f"Near-Family AUROC (跨族):  {get_auroc(near_fam_scores)}")
        print(f"Near-Variant AUROC (同族): {get_auroc(near_var_scores)}")
        print(f"Strict Far AUROC (隔离):   {get_auroc(far_scores)}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores, near_ood_scores=near_var_scores, tpr_target=0.95, method_name="DEF")
        
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores, near_ood_scores=near_fam_scores, tpr_target=0.95, method_name="DEF")


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
        print(f"Strict Global AUROC: {get_auroc(np.concatenate([near_scores, far_scores]))}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores, near_ood_scores=near_scores, tpr_target=0.95, method_name="DEF") 



if __name__ == "__main__":
    main()