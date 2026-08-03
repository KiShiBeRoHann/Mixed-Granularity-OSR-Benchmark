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


def state_dict_of(model):
    """取模型 state_dict；模型被 DataParallel 包装时自动解包（去掉 module. 前缀），保证保存格式与单卡一致"""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_state_dict_into(model, state_dict):
    """加载 state_dict 到模型；自动兼容 DataParallel 包装的模型"""
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def calculate_cvr_unr(id_scores, coarse_gen_scores, near_ood_scores, tpr_target=0.95, method_name="Method"):
    """
    独立指标计算模块：计算已知类内未见变体拒绝率 (CV-RR) 、 未知近邻拒绝率 (UN-RR) 以及综合 H-Score
    :param id_scores: 纯已知类 (test_id) 的预测分数 (用于锚定 TPR 阈值)
    :param coarse_gen_scores: 粗粒度泛化测试集 (test_coarse_gen) 的分数
    :param near_ood_scores: 细粒度近邻对抗测试集 (test_near_variant / test_near_family) 的分数
    """
    if len(id_scores) == 0:
        return "N/A", "N/A", "N/A"

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

    # 4. 计算综合指标 H-Score (调和平均数，越高越好)
    h_score = "N/A"
    if cv_rr != "N/A" and un_rr != "N/A":
        # 转化 CV-RR 为 CV-AR (接收率)，使其与 UN-RR 保持同向同量级
        cv_ar = 100.0 - cv_rr
        
        # 计算标准调和平均数，防范分母为零的极端情况
        if (cv_ar + un_rr) > 0:
            h_score = (2 * cv_ar * un_rr) / (cv_ar + un_rr)
        else:
            h_score = 0.0

    # 打印输出结果
    print(f"\n[{method_name}] 阈值依赖指标 @TPR{int(tpr_target*100)} (Threshold: {threshold:.4f})")
    if cv_rr != "N/A":
        print(f"[{method_name}] 已知类内未见变体拒绝率 (CV-RR) [误伤, 越低越好]: {cv_rr:.2f}%")
    if un_rr != "N/A":
        print(f"[{method_name}] 未知近邻拒绝率 (UN-RR)       [防御, 越高越好]: {un_rr:.2f}%")
    if h_score != "N/A":
        print(f"[{method_name}] 综合调和得分 (H-Score)        [综合, 越高越好]: {h_score:.2f}%")

    return cv_rr, un_rr, h_score


def calculate_oscr(pred_k_id, x_k_id, pred_u_ood):


    """
    计算开集分类与识别曲线下面积 (OSCR - Open-Set Classification and Recognition Area)
    
    参数:
        pred_k_id: np.ndarray, shape=(N_id,), 已知样本 (ID) 的置信度/分数 (越大的分数越被视为已知)
        x_k_id:    np.ndarray, shape=(N_id,), 已知样本的预测正确性指示矩阵 (布尔数组或0/1数组: 类别预测对为1, 猜错为0)
        pred_u_ood:np.ndarray, shape=(N_ood,), 未知/近邻样本 (OOD) 的置信度/分数
        
    返回:
        oscr_area: float, 百分制下的 OSCR 面积 (0.0 ~ 100.0)
    """
    # 转换为 1D numpy array
    pred_k_id = np.asarray(pred_k_id).flatten()
    x_k_id = np.asarray(x_k_id).flatten().astype(bool)
    pred_u_ood = np.asarray(pred_u_ood).flatten()
    
    if len(pred_k_id) == 0 or len(pred_u_ood) == 0:
        return 0.0

    # 将已知样本置信度与正确性绑定：(置信度, 是否猜对)
    # 按置信度降序排列 (从确信度最高的样本开始扫阈值)
    sort_idx_id = np.argsort(pred_k_id)[::-1]
    sorted_id_scores = pred_k_id[sort_idx_id]
    sorted_correctness = x_k_id[sort_idx_id]
    
    # 针对 OOD 样本，仅按分数降序排列
    sorted_ood_scores = np.sort(pred_u_ood)[::-1]
    
    n_id = len(sorted_id_scores)
    n_ood = len(sorted_ood_scores)
    
    # 构建滑动门限的分数池（结合 ID 与 OOD 的唯一分界点）
    all_scores = np.unique(np.concatenate([sorted_id_scores, sorted_ood_scores]))
    all_scores = np.sort(all_scores)[::-1] # 降序滑动
    
    ccpr_list = []  # Correct Classification Positive Rate (正确分类识别率)
    fpr_list = []   # False Positive Rate (假阳率 / 误识率)
    
    # 指针遍历所有可能阈值
    id_ptr = 0
    ood_ptr = 0
    correct_count = 0
    fp_count = 0
    
    for thresh in all_scores:
        # 统计 ID 中大于等于当前阈值且【分类正确】的样本数
        while id_ptr < n_id and sorted_id_scores[id_ptr] >= thresh:
            if sorted_correctness[id_ptr]:
                correct_count += 1
            id_ptr += 1
            
        # 统计 OOD 中大于等于当前阈值（被错误放进来）的样本数
        while ood_ptr < n_ood and sorted_ood_scores[ood_ptr] >= thresh:
            fp_count += 1
            ood_ptr += 1
            
        ccpr_list.append(correct_count / n_id)
        fpr_list.append(fp_count / n_ood)
        
    # 为了让积分起点终点严密闭合 (FPR 从 0 积到 1)
    if fpr_list[0] > 0.0:
        fpr_list.insert(0, 0.0)
        ccpr_list.insert(0, 0.0)
    if fpr_list[-1] < 1.0:
        fpr_list.append(1.0)
        ccpr_list.append(ccpr_list[-1])
        
    # 使用梯形法则对 FPR-CCPR 曲线计算面积，并转化为百分制
    oscr_area = np.trapz(ccpr_list, fpr_list) * 100.0
    
    return float(oscr_area)

    
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