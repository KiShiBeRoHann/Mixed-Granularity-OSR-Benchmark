import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn

# 导入你写好的数据集划分
from datasets.split_cifar import MixedGranularityCIFAR100

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def calculate_cvr_unr(id_scores, coarse_gen_scores, near_ood_scores, tpr_target=0.95, method_name="Method"):
    """
    独立指标计算模块：计算已知类内未见变体拒绝率 (CV-RR) 和 未知近邻拒绝率 (UN-RR)
    :param id_scores: 纯已知类 (test_id) 的预测分数 (用于锚定 TPR 阈值)
    :param coarse_gen_scores: 粗粒度泛化测试集 (test_coarse_gen) 的分数
    :param near_ood_scores: 细粒度近邻对抗测试集 (test_near_variant / test_near_family) 的分数
    """
    if len(id_scores) == 0:
        return "N/A", "N/A"

    # 1. 锚定阈值：保证 tpr_target (默认95%) 的 ID 样本能够存活
    percentile = 100.0 * (1.0 - tpr_target)
    threshold = np.percentile(id_scores, percentile)

    # 2. 计算 CV-RR (误拒率，越低越好)
    cv_rr = "N/A"
    if len(coarse_gen_scores) > 0:
        rejected_cv = np.sum(coarse_gen_scores < threshold)
        cv_rr = (rejected_cv / len(coarse_gen_scores)) * 100.0

    # 3. 计算 UN-RR (有效防守率，越高越好)
    un_rr = "N/A"
    if len(near_ood_scores) > 0:
        rejected_un = np.sum(near_ood_scores < threshold)
        un_rr = (rejected_un / len(near_ood_scores)) * 100.0

    print(f"\n[{method_name}] 阈值依赖指标 @TPR{int(tpr_target*100)} (Threshold: {threshold:.4f})")
    if cv_rr != "N/A":
        print(f"[{method_name}] 已知类内未见变体拒绝率 (CV-RR) [误伤, 越低越好]: {cv_rr:.2f}%")
    if un_rr != "N/A":
        print(f"[{method_name}] 未知近邻拒绝率 (UN-RR) [防御, 越高越好]: {un_rr:.2f}%")

    return cv_rr, un_rr


def evaluate_fine_grained_ood(pos_msp, far_msp, far_preds, far_targets, method_name="Method"):
    """
    计算细粒度 OOD 深度评估指标：WCE 和 MCR
    - pos_msp: ID 样本的最大置信度分数 (1D Array)
    - far_msp: Far OOD 样本的最大置信度分数 (1D Array)
    - far_preds: Far OOD 样本被模型预测成的 ID 类别索引 (1D Array)
    - far_targets: Far OOD 样本的真实细粒度/粗粒度标签 (1D Array)
    """
    print(f"\n[{method_name}] 正在计算远分布 (Far OOD) 深度分析指标...")
    
    # 获取 Far OOD 数据集中包含的所有独立类别
    unique_far_classes = np.unique(far_targets)
    
    if len(unique_far_classes) <= 1:
        print("警告：Far OOD 只有 1 个类别，无法体现细粒度差异。")
        return None, None, None

    # ==========================================
    # 指标 1: 最差子类性能暴露度 (WCE)
    # ==========================================
    subclass_aurocs = []
    for c in unique_far_classes:
        # 提取当前子类的分数
        scores_c = far_msp[far_targets == c]
        if len(scores_c) == 0:
            continue
            
        # 仅用当前 Far OOD 子类与全体 ID 计算 AUROC
        y_true = np.concatenate([np.ones(len(pos_msp)), np.zeros(len(scores_c))])
        y_scores = np.concatenate([pos_msp, scores_c])
        auroc_c = roc_auc_score(y_true, y_scores) * 100
        subclass_aurocs.append(auroc_c)

    wce_min = np.min(subclass_aurocs)
    wce_std = np.std(subclass_aurocs)
    
    print(f"  -> 最差子类 AUROC (WCE-Min): {wce_min:.2f}% (对比 Global AUROC)")
    print(f"  -> 子类 AUROC 标准差 (WCE-Std): {wce_std:.2f}%")

    # ==========================================
    # 指标 2: 映射集中度 (MCR)
    # ==========================================
    mcr_list = []
    for c in unique_far_classes:
        preds_c = far_preds[far_targets == c]
        if len(preds_c) == 0:
            continue
            
        # 统计当前 Far OOD 类被错分到各个 ID 类的次数
        counts = np.bincount(preds_c)
        max_count = np.max(counts)
        total_count = len(preds_c)
        
        # 计算集中度比例
        concentration_ratio = (max_count / total_count) * 100
        mcr_list.append(concentration_ratio)

    mcr_mean = np.mean(mcr_list)
    print(f"  -> 平均映射集中度 (MCR): {mcr_mean:.2f}%")

    return wce_min, wce_std, mcr_mean

def calculate_ifcr(id_targets, id_preds, method_name="Method"):
    """
    独立指标计算模块：计算族内混淆率 (Intra-Family Confusion Rate)
    """
    # 细粒度家族映射 (前5个粗粒度 0-4，后15个细粒度 5-19)
    family_map = {5:0, 6:0, 7:0, 8:1, 9:1, 10:1, 11:2, 12:2, 13:2, 14:3, 15:3, 16:3, 17:4, 18:4, 19:4}

    fine_mask = id_targets >= 5
    fine_targets = id_targets[fine_mask]
    fine_preds = id_preds[fine_mask]

    total_fine_errors = 0
    intra_family_errors = 0
    for t, p in zip(fine_targets, fine_preds):
        if t != p:
            total_fine_errors += 1
            if p >= 5 and family_map.get(t) == family_map.get(p):
                intra_family_errors += 1

    ifcr = (intra_family_errors / total_fine_errors * 100) if total_fine_errors > 0 else 0.0
    print(f"\n[{method_name}] 族内混淆率 (IFCR): {ifcr:.2f}%")
    print(f"[{method_name}] 细粒度错误总数: {total_fine_errors} | 族内坍缩数: {intra_family_errors}")
    
    return ifcr


def visualize_fine_grained(model, device, preprocess, method_name="Method", save_dir="/home/geng/WTY/pictures"):
    """
    独立可视化模块：生成 5x5 超类坍缩热力图与 t-SNE
    """
    print(f"\n正在为 {method_name} 生成细粒度 5x5 坍缩分析图...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    raw_cifar100 = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=preprocess)

    superclasses = {
        "Insects": {
            "raw_classes": [6, 7, 14, 18, 24], 
            "names": ["Bee (ID)", "Beetle (ID)", "Butterfly (ID)", "Caterpillar (OOD)", "Cockroach (OOD)"],
            "pred_map": {11: 0, 12: 1, 13: 2} 
        },
        "Large_Carnivores": {
            "raw_classes": [3, 42, 43, 88, 97], 
            "names": ["Bear (ID)", "Leopard (ID)", "Lion (ID)", "Tiger (OOD)", "Wolf (OOD)"],
            "pred_map": {14: 0, 15: 1, 16: 2}
        }
    }

    model.eval()
    
    for sc_name, sc_info in superclasses.items():
        features_list, labels_list, preds_list = [], [], []
        
        with torch.no_grad():
            for i in range(len(raw_cifar100)):
                img, raw_label = raw_cifar100[i]
                if raw_label in sc_info["raw_classes"]:
                    image = img.unsqueeze(0).to(device)
                    
                    # 兼容不同 Backbone
                    if hasattr(model, 'image_encoder'):
                        feat = model.image_encoder(image.type(model.dtype)) if hasattr(model, 'dtype') else model.image_encoder(image)
                    elif hasattr(model, 'backbone'):
                        feat = model.backbone(image)
                    elif hasattr(model, 'encode_image'): 
                        feat = model.encode_image(image.type(model.dtype)) if hasattr(model, 'dtype') else model.encode_image(image)
                    else:
                        feat = model(image)
                        
                    logits = model(image)
                    pred = logits.argmax(dim=1).item()
                    
                    features_list.append(feat.view(-1).cpu().numpy())
                    labels_list.append(sc_info["raw_classes"].index(raw_label))
                    preds_list.append(pred)

        # ---------------------------
        # 绘制 5x5 热力图
        # ---------------------------
        cm = np.zeros((5, 5), dtype=int)
        for true_idx, pred_val in zip(labels_list, preds_list):
            if pred_val in sc_info["pred_map"]:
                pred_idx = sc_info["pred_map"][pred_val]
                cm[true_idx, pred_idx] += 1

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', 
                    xticklabels=sc_info["names"], yticklabels=sc_info["names"])
        plt.title(f'{sc_name} Intra-Family Collapse ({method_name})')
        plt.xlabel('Predicted Label (OOD columns are zero)')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{method_name}_{sc_name}_5x5_heatmap.png'), dpi=300)
        plt.close()

        # ---------------------------
        # 绘制 5 类对比 t-SNE
        # ---------------------------
        tsne_results = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(np.array(features_list))
        
        plt.figure(figsize=(8, 8))
        
        # 前3个ID使用分类色，后2个OOD使用统一的深灰色 (#7f7f7f)
        colors = ['#1f77b4', '#2ca02c', '#9467bd', '#7f7f7f', '#7f7f7f']
        markers = ['o', 's', '^', '*', 'X'] 
        
        for i in range(5):
            idx = (np.array(labels_list) == i)
            size = 150 if i >= 3 else 70
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                        label=sc_info["names"][i], color=colors[i], 
                        marker=markers[i], alpha=0.7, s=size, edgecolors='white', linewidths=0.5)
        
        plt.title(f'{sc_name} t-SNE Feature Space ({method_name})')
        plt.legend(loc='upper left', prop={'size': 9}, framealpha=0.5, edgecolor='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{method_name}_{sc_name}_tsne.png'), dpi=300)
        plt.close()
        
    print(f"-> {method_name} 的 5x5 细粒度图表已全部生成！\n")


def visualize_cross_family_mapping(model, device, preprocess, classnames, method_name="Method", save_dir="/home/geng/WTY/pictures"):
    """
    方案二：跨族映射热力图 (Cross-Family Mapping Heatmap)
    专门剖析前 3 个 Far OOD 超类 (15个细粒度类) 坍缩到 20 个 ID 类的规律
    """
    print(f"\n正在为 {method_name} 生成 Coarse Far OOD 跨族映射热力图...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    raw_cifar100 = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=preprocess)

    # 宏观超类定义：包含对应的细粒度索引
    target_superclasses = {
        "Natural Scenes\n(5 classes)": [23, 33, 49, 60, 71],
        "Omnivores & Herbivores\n(5 classes)": [15, 16, 17, 31, 38],
        "Medium Mammals\n(5 classes)": [34, 63, 64, 66, 75]
    }
    
    super_names = list(target_superclasses.keys())

    model.eval()
    
    # 构建一个 3 (大超类) x 20 (ID) 的归属计数矩阵
    cm = np.zeros((3, 20), dtype=int)
    
    with torch.no_grad():
        for i in range(len(raw_cifar100)):
            img, raw_label = raw_cifar100[i]
            
            # 遍历检查这张图属于哪个超类
            for row_idx, (sc_name, fine_classes) in enumerate(target_superclasses.items()):
                if raw_label in fine_classes:
                    image = img.unsqueeze(0).to(device)
                    
                    # 万能前向推理
                    logits = model(image)
                    pred_idx = logits.argmax(dim=1).item()
                    
                    # 对应的超类计数 +1
                    cm[row_idx, pred_idx] += 1
                    break # 找到归属就跳出内层循环

    # ==========================================
    # 绘制 3 x 20 扁平热力图
    # ==========================================
    plt.figure(figsize=(16, 5)) # 因为只有 3 行，高度设为 5 即可
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=classnames, yticklabels=super_names,
                linewidths=0.5, linecolor='white')
    
    plt.title(f'Coarse-Grained Semantic Mapping of Far OOD ({method_name})', fontsize=16)
    plt.xlabel('Forced ID Predictions (20 Classes)', fontsize=14)
    plt.ylabel('True Far OOD Superclasses', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{method_name}_Coarse_Mapping.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"-> 粗粒度映射热力图已保存至: {save_path}\n")
    print(f"\n正在为 {method_name} 生成 Far OOD 跨族映射热力图...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载测试集原始数据
    raw_cifar100 = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=preprocess)

    # 我们挑选的前 3 个 Far OOD 超类及其细粒度子类 (共 15 个)
    # 对应的 CIFAR-100 官方标签索引
    target_far_ood = {
        # 1. large natural outdoor scenes
        23: 'cloud', 33: 'forest', 49: 'mountain', 60: 'plain', 71: 'sea',
        # 2. large omnivores and herbivores
        15: 'camel', 16: 'cattle', 17: 'chimpanzee', 31: 'elephant', 38: 'kangaroo',
        # 3. medium-sized mammals
        34: 'fox', 63: 'porcupine', 64: 'possum', 66: 'raccoon', 75: 'skunk'
    }
    
    far_ood_indices = list(target_far_ood.keys())
    far_ood_names = list(target_far_ood.values())

    model.eval()
    
    # 构建一个 15 (Far OOD) x 20 (ID) 的归属计数矩阵
    cm = np.zeros((15, 20), dtype=int)
    
    with torch.no_grad():
        for i in range(len(raw_cifar100)):
            img, raw_label = raw_cifar100[i]
            
            # 如果这张图属于我们关注的 15 个 Far OOD 类
            if raw_label in far_ood_indices:
                image = img.unsqueeze(0).to(device)
                
                # 获取模型分类结果 (被迫在 20 个 ID 里选一个)
                logits = model(image)
                pred_idx = logits.argmax(dim=1).item()
                
                # 找到它在 15 行里的相对行号
                row_idx = far_ood_indices.index(raw_label)
                
                # 矩阵对应位置计数 +1
                cm[row_idx, pred_idx] += 1

    # ==========================================
    # 开始绘制 15 x 20 不对称热力图
    # ==========================================
    plt.figure(figsize=(16, 10)) # 图要画得宽一点，因为横轴有 20 个类
    
    # 画线区分三个超类的边界 (在第 5 行和第 10 行下面画白线)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=classnames, yticklabels=far_ood_names,
                linewidths=0.5, linecolor='white')
    
    # 添加超类分隔辅助线
    plt.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # 增加右侧 Y 轴的超类标注文本
    ax = plt.gca()
    ax.text(20.5, 2.5, 'Natural Scenes', va='center', ha='left', color='red', fontsize=12, fontweight='bold', rotation=270)
    ax.text(20.5, 7.5, 'Omnivores & Herbivores', va='center', ha='left', color='red', fontsize=12, fontweight='bold', rotation=270)
    ax.text(20.5, 12.5, 'Medium Mammals', va='center', ha='left', color='red', fontsize=12, fontweight='bold', rotation=270)

    plt.title(f'Cross-Family Semantic Mapping of Far OOD ({method_name})', fontsize=16)
    plt.xlabel('Forced ID Predictions (20 Classes)', fontsize=14)
    plt.ylabel('True Far OOD Classes (15 Classes from 3 Superclasses)', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{method_name}_Fine_Mapping.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"-> 细粒度映射热力图已保存至: {save_path}\n")