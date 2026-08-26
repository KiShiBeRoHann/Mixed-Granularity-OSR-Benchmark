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


def evaluate_mixed_metrics(score_map, method_name="Method", tpr_target=0.95, save_dir=None, include_oscr=True):
    """
    统一的多粒度开集评估入口（兼容 2 层 / 3 层划分，无需按数据集区分分支）。

    语义说明（与讨论达成的一致精简方案）：
      * 泛化集(gen)合并：把 coarse_gen 与 medium_gen 的样本合并成同一个"泛化新增"集合，
        对应一个统一的 Gen Acc（不再分 Coarse/Medium 两个打印）。
      * 近邻OOD合并：把 near_ood / near_family / near_variant 合并成同一个"近邻OOD"集合，
        作为 CVRR/UNRR 的 UN 侧与 Global AUROC 的负样本。
      * CVRR/UNRR/H-Score 只计算一次（用合并后的 gen 作为 CV 侧、合并后的 near 作为 UN 侧）。
      * AUROC 只打印一个 Global AUROC（含近邻+远邻全部负样本）。
      * OSCR 保留 Global OSCR（合并 gen 后 "MG-OSCR" 这类依赖细分配对的指标不再有意义，故不输出）。

    :param score_map: dict, 键为 'id'/'coarse_gen'/'medium_gen'/'near_ood'/'near_family'/'near_variant'/'far_ood'，
                      值为 (scores, preds, targets) 三元组或 None（对应 loader 不存在/为空时传 None）。
        scores 越大的分数越视为"已知/已识别"（用于阈值与 AUROC）。
    :param method_name: 打印前缀，例如 "A2Pt" / "CoOp"。
    :param tpr_target: CVRR/UNRR 锚定的 TPR（默认 0.95）。
    :param save_dir: 若提供，则把合并后的 id/gen/near_ood/far_ood 分数保存为 *_{id,gen,near_ood,far_ood}_scores.npy。
                     注意 gen_scores 为 coarse+medium 合并后、near_ood_scores 为 近邻各集合合并后的结果。
    :param include_oscr: 是否输出 Global OSCR（默认 True）。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    def _pick(key):
        v = score_map.get(key)
        if v is None:
            return None
        s, p, t = v
        if s is None or len(s) == 0:
            return None
        return (np.asarray(s), np.asarray(p), np.asarray(t))

    # ---------- ID ----------
    id_scores, id_preds, id_targets = (None, None, None)
    _id = _pick('id')
    if _id is not None:
        id_scores, id_preds, id_targets = _id
    if id_scores is None or len(id_scores) == 0:
        print(f"\n[{method_name}] 评估中止：缺少 ID 分数。")
        return None
    pos_scores_strict = id_scores

    # ---------- 泛化集合并 (coarse + medium) ----------
    gen_parts = []
    for k in ('coarse_gen', 'medium_gen'):
        g = _pick(k)
        if g is not None:
            gen_parts.append(g)
    gen_scores = np.concatenate([g[0] for g in gen_parts]) if gen_parts else np.array([])
    gen_preds   = np.concatenate([g[1] for g in gen_parts]) if gen_parts else np.array([])
    gen_targets = np.concatenate([g[2] for g in gen_parts]) if gen_parts else np.array([])

    # ---------- 近邻OOD合并 (near_ood / near_family / near_variant) ----------
    near_parts = []
    for k in ('near_ood', 'near_family', 'near_variant'):
        n = _pick(k)
        if n is not None:
            near_parts.append(n[0])
    near_scores = np.concatenate(near_parts) if near_parts else np.array([])

    # ---------- 远邻OOD ----------
    _far = _pick('far_ood')
    far_scores = _far[0] if _far is not None else np.array([])

    def safe_acc(preds, targets):
        return f"{100 * (preds == targets).mean():.2f}%" if len(targets) > 0 else "N/A"
    def safe_auroc(y_true, y_scores):
        if len(np.unique(y_true)) < 2:
            return 0.0
        return roc_auc_score(y_true, y_scores) * 100
    def get_auroc(neg_scores):
        if len(neg_scores) == 0:
            return "N/A"
        y_true = np.concatenate([np.ones(len(pos_scores_strict)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores_strict, neg_scores])
        return f"{safe_auroc(y_true, y_scores):.2f}%"

    # ---------- Global Acc ----------
    total_preds = np.concatenate([id_preds, gen_preds]) if len(gen_preds) > 0 else id_preds
    total_targets = np.concatenate([id_targets, gen_targets]) if len(gen_targets) > 0 else id_targets
    global_acc = 100 * (total_preds == total_targets).mean() if len(total_targets) > 0 else 0.0

    # ---------- 输出 ----------
    print(f"\n{'='*15} Results ({method_name}) {'='*15}")
    print(f"Global Acc:  {global_acc:.2f}%")
    print(f"Gen Acc:     {safe_acc(gen_preds, gen_targets)}")
    print("-" * 45)

    all_neg = np.concatenate([near_scores, far_scores]) if len(near_scores) > 0 or len(far_scores) > 0 else np.array([])
    print(f"Strict Global AUROC: {get_auroc(all_neg)}")

    # ---------- 统一 CVRR / UNRR / H-Score（只一次） ----------
    if len(near_scores) > 0:
        calculate_cvr_unr(
            id_scores=id_scores,
            coarse_gen_scores=gen_scores,
            near_ood_scores=near_scores,
            tpr_target=tpr_target,
            method_name=method_name,
        )
    else:
        print(f"\n[{method_name}] 无可用于 CVRR/UNRR 的近邻OOD样本，跳过。")

    # ---------- Global OSCR ----------
    if include_oscr:
        print("-" * 45)
        print(f"[{method_name}] OSCR 综合指标评估")
        global_id_scores = np.concatenate([id_scores, gen_scores]) if len(gen_scores) > 0 else id_scores
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=(total_preds == total_targets),
            pred_u_ood=all_neg if len(all_neg) > 0 else np.array([1.0]),
        )
        print(f"[{method_name}] Global OSCR: {global_oscr:.2f}%")

    # ---------- 可选保存（供下游可视化） ----------
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{method_name}_id_scores.npy"), id_scores)
        np.save(os.path.join(save_dir, f"{method_name}_gen_scores.npy"), gen_scores)
        if len(near_scores) > 0:
            np.save(os.path.join(save_dir, f"{method_name}_near_ood_scores.npy"), near_scores)
        if len(far_scores) > 0:
            np.save(os.path.join(save_dir, f"{method_name}_far_ood_scores.npy"), far_scores)

    return {
        "global_acc": global_acc,
        "gen_acc": 100 * (gen_preds == gen_targets).mean() if len(gen_targets) > 0 else None,
        "global_auroc": get_auroc(all_neg),
        "cv_rr_ar": None,  # 合并后的 CVRR/UNRR/H 由 calculate_cvr_unr 打印
    }


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