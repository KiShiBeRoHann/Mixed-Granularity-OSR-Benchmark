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


def view_save_dir(save_dir, eval_view="V4"):
    """
    评估靶场视图(V1/V2/V3/V4)的产物隔离。

    V1~V4 共用同一份权重与同一份 ID，只有 OOD 靶场构成不同，因此权重路径 model_dir
    **不能**按视图分叉（否则会触发重训）。但评估输出的 *_scores.npy 必须分叉，
    否则非默认视图会覆盖 V4 的历史产物。默认视图 V4 保持写入原目录，行为不变。
    """
    if save_dir is None or eval_view in (None, "V4"):
        return save_dir
    return os.path.join(save_dir, f"view_{eval_view}")


def calculate_cvr_unr(id_scores, coarse_gen_scores, near_ood_scores,
                      coarse_gen_preds=None, coarse_gen_targets=None,
                      tpr_target=0.95, method_name="Method"):
    """
    独立指标计算模块：计算 CV-CR / CV-RR / UN-RR 以及综合 H-Score。

    CV-CR 是“接受且分到正确的已知类头”的比例：
        CV-CR = P(score >= tau 且 pred == target)
    因此它同时惩罚“误拒”和“接受但分错 head”。CV-RR 仅作为误拒率诊断保留。

    :param id_scores: 纯已知类 (test_id) 的预测分数 (用于锚定 TPR 阈值)
    :param coarse_gen_scores: 泛化集 (合并后的 coarse_gen + medium_gen) 的分数
    :param coarse_gen_preds: 泛化集的预测类别（与 coarse_gen_scores 逐样本对齐）
    :param coarse_gen_targets: 泛化集的父类标签（与 coarse_gen_scores 逐样本对齐）
    :param near_ood_scores: 近邻OOD (合并后的 near_ood/near_family/near_variant) 的分数
    :return: (cv_rr, cv_cr, cv_wrong_accepted, un_rr, h_score, threshold)
             —— 前五个为百分制数值（样本为空时返回 "N/A"）；
             threshold 为锚定的分数阈值（无 ID 样本时为 "N/A"）。
    """
    if len(id_scores) == 0:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

    # 1. 锚定阈值：保证 tpr_target (默认95%) 的 ID 样本能够存活
    percentile = 100.0 * (1.0 - tpr_target)
    threshold = np.percentile(id_scores, percentile)

    # 2. CV-RR（误拒率，诊断项）与 CV-CR（接受且分对）
    cv_rr = cv_cr = cv_wrong_accepted = "N/A"
    if len(coarse_gen_scores) > 0:
        scores = np.asarray(coarse_gen_scores)
        accepted = scores >= threshold
        cv_rr = float((~accepted).mean() * 100.0)

        if (coarse_gen_preds is not None and coarse_gen_targets is not None
                and len(coarse_gen_preds) == len(scores)):
            preds = np.asarray(coarse_gen_preds)
            targets = np.asarray(coarse_gen_targets)
            correct = preds == targets
            cv_cr = float((accepted & correct).mean() * 100.0)
            cv_wrong_accepted = float((accepted & ~correct).mean() * 100.0)
        else:
            print(f"[{method_name}] 警告: 缺少 coarse_gen preds/targets，CV-CR 无法计算。")

    # 3. 计算 UN-RR (有效防守率，越高越好)
    un_rr = "N/A"
    if len(near_ood_scores) > 0:
        rejected_un = np.sum(near_ood_scores < threshold)
        un_rr = (rejected_un / len(near_ood_scores)) * 100.0

    # 4. 计算综合指标 H-Score = HM(CV-CR, UN-RR)
    h_score = "N/A"
    if cv_cr != "N/A" and un_rr != "N/A":
        if (cv_cr + un_rr) > 0:
            h_score = (2 * cv_cr * un_rr) / (cv_cr + un_rr)
        else:
            h_score = 0.0

    # 打印输出结果
    print(f"\n[{method_name}] 阈值依赖指标 @TPR{int(tpr_target*100)} (Threshold: {threshold:.4f})")
    if cv_cr != "N/A":
        print(f"[{method_name}] 粗变体正确接受率 (CV-CR)         [接受且分对, 越高越好]: {cv_cr:.2f}%")
    if cv_wrong_accepted != "N/A":
        print(f"[{method_name}] 粗变体接受但分错率 (CV-WrongAccepted) [越低越好]: {cv_wrong_accepted:.2f}%")
    if cv_rr != "N/A":
        print(f"[{method_name}] 粗变体误拒率 (CV-RR)             [诊断, 越低越好]: {cv_rr:.2f}%")
    if un_rr != "N/A":
        print(f"[{method_name}] 未知近邻拒绝率 (UN-RR)           [越高越好]: {un_rr:.2f}%")
    if h_score != "N/A":
        print(f"[{method_name}] 综合调和得分 (H-Score)            [HM(CV-CR,UN-RR), 越高越好]: {h_score:.2f}%")

    return cv_rr, cv_cr, cv_wrong_accepted, un_rr, h_score, threshold


# =====================================================================
# 评估指标落盘（视图对照 / 论文出表用）
# =====================================================================
# 列顺序固定，便于 csv 直读与跨 seed/视图拼表
METRICS_CSV_FIELDS = [
    'dataset', 'num_layers', 'seed', 'eval_view', 'mode', 'method', 'tpr_target', 'threshold',
    'n_id', 'n_gen_coarse', 'n_gen_medium', 'n_gen',
    'n_near_ood', 'n_near_family', 'n_near_variant', 'n_near', 'n_far',
    'global_acc', 'gen_acc', 'global_auroc',
    'cv_cr', 'cv_wrong_accepted', 'cv_rr', 'un_rr', 'h_score', 'global_oscr',
    'gen_sources', 'near_sources',
]


def _to_float(v, ndigits=4):
    """把 'N/A' / numpy 标量归一化成可写 JSON/CSV 的 Python 数值（不可转换返回 None）"""
    if v is None or (isinstance(v, str) and v == "N/A"):
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def write_metrics_record(save_dir, metrics):
    """
    把一次评估的数值结果落盘，供 4 视图对照 / 论文表格直接读取：

      * {method}_metrics.json —— 覆盖写，每个 (seed, 视图, 方法) 一份
      * metrics.csv          —— 同一 (dataset, num_layers, seed, eval_view, method)
                                的记录**覆盖**而非追加，重复跑不会产生重复行

    注意：样本量字段（n_gen / n_near 及分项）必须一并落盘 —— 视图切换会改变
    gen / near 的分母构成（V1/V2 的 n_gen_coarse、n_near_family 为 0），只看
    Acc/AUROC 数值会误读"防御力变化"。
    """
    import csv
    import json

    os.makedirs(save_dir, exist_ok=True)
    method = metrics['method']
    mode = metrics.get('mode', 'Pure_Closed_Set')
    suffix = '' if mode in ('', None, 'Pure_Closed_Set') else f'_{mode}'

    json_path = os.path.join(save_dir, f'{method}{suffix}_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)

    key_fields = ('dataset', 'num_layers', 'seed', 'eval_view', 'mode', 'method')

    def _norm(k, v):
        # 兼容旧记录：没有 mode 字段时视为 Pure_Closed_Set
        if k == 'mode' and str(v) in ('', 'None'):
            return 'Pure_Closed_Set'
        return str(v)

    key = tuple(_norm(k, metrics.get(k, '')) for k in key_fields)
    csv_path = os.path.join(save_dir, 'metrics.csv')

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f)
                    if tuple(_norm(k, r.get(k, '')) for k in key_fields) != key]
    rows.append({k: metrics.get(k) for k in METRICS_CSV_FIELDS})

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in METRICS_CSV_FIELDS})

    print(f"[Metric Record] 已写入 {json_path} 与 {csv_path}")


def evaluate_mixed_metrics(score_map, method_name="Method", tpr_target=0.95, save_dir=None,
                           include_oscr=True, extra_meta=None, mode=None):
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
    :param save_dir: 若提供，则把合并后的 id/gen/near_ood/far_ood 分数保存为 *_{id,gen,near_ood,far_ood}_scores.npy，
                     并写 {method}_metrics.json + metrics.csv（数值指标 + 各集合样本量，供视图对照出表）。
                     注意 gen_scores 为 coarse+medium 合并后、near_ood_scores 为 近邻各集合合并后的结果。
    :param include_oscr: 是否输出 Global OSCR（默认 True）。
    :param extra_meta: 写入 metrics 记录的元数据，如
                       {'dataset': 'aircraft', 'num_layers': 3, 'seed': 42, 'eval_view': 'V1'}。
    :param mode: 'Pure_Closed_Set'（默认）或 'Unseen_In_Train'。
                 后者只用于 Seen Accuracy，跳过 CV-CR/CV-RR/UN-RR/H-Score/AUROC/OSCR，
                 避免 UIT 结果覆盖 PCS 的粒度指标记录。
    :return: dict —— 数值版指标（字段见 METRICS_CSV_FIELDS），可用于下游自动出表。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    meta = dict(extra_meta or {})
    mode = mode or meta.get('mode') or 'Pure_Closed_Set'
    gran_metrics = (mode != 'Unseen_In_Train')

    def _pick(key):
        v = score_map.get(key)
        if v is None:
            return None
        s, p, t = v
        if s is None or len(s) == 0:
            return None
        return (np.asarray(s), np.asarray(p), np.asarray(t))

    _keys = ('id', 'coarse_gen', 'medium_gen', 'near_ood', 'near_family', 'near_variant', 'far_ood')
    picked = {k: _pick(k) for k in _keys}

    def _n(key):
        return int(len(picked[key][0])) if picked[key] is not None else 0

    # ---------- ID ----------
    id_scores, id_preds, id_targets = (None, None, None)
    if picked['id'] is not None:
        id_scores, id_preds, id_targets = picked['id']
    if id_scores is None or len(id_scores) == 0:
        print(f"\n[{method_name}] 评估中止：缺少 ID 分数。")
        return None
    pos_scores_strict = id_scores

    # ---------- 泛化集合并 (coarse + medium) ----------
    gen_keys = ('coarse_gen', 'medium_gen')
    gen_sources = [k for k in gen_keys if picked[k] is not None]
    gen_parts = [picked[k] for k in gen_sources]
    gen_scores = np.concatenate([g[0] for g in gen_parts]) if gen_parts else np.array([])
    gen_preds   = np.concatenate([g[1] for g in gen_parts]) if gen_parts else np.array([])
    gen_targets = np.concatenate([g[2] for g in gen_parts]) if gen_parts else np.array([])

    # ---------- 近邻OOD合并 (near_ood / near_family / near_variant) ----------
    near_keys = ('near_ood', 'near_family', 'near_variant')
    near_sources = [k for k in near_keys if picked[k] is not None]
    near_parts = [picked[k][0] for k in near_sources]
    near_scores = np.concatenate(near_parts) if near_parts else np.array([])

    # ---------- 远邻OOD ----------
    far_scores = picked['far_ood'][0] if picked['far_ood'] is not None else np.array([])

    def safe_acc(preds, targets):
        return f"{100 * (preds == targets).mean():.2f}%" if len(targets) > 0 else "N/A"
    def safe_auroc(y_true, y_scores):
        if len(np.unique(y_true)) < 2:
            return 0.0
        return roc_auc_score(y_true, y_scores) * 100
    def compute_auroc(neg_scores):
        """返回数值 AUROC(%)；负样本为空返回 None"""
        if len(neg_scores) == 0:
            return None
        y_true = np.concatenate([np.ones(len(pos_scores_strict)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores_strict, neg_scores])
        return safe_auroc(y_true, y_scores)
    def get_auroc(neg_scores):
        v = compute_auroc(neg_scores)
        return "N/A" if v is None else f"{v:.2f}%"

    # ---------- Global Acc ----------
    total_preds = np.concatenate([id_preds, gen_preds]) if len(gen_preds) > 0 else id_preds
    total_targets = np.concatenate([id_targets, gen_targets]) if len(gen_targets) > 0 else id_targets
    global_acc = 100 * (total_preds == total_targets).mean() if len(total_targets) > 0 else 0.0

    # ---------- 输出 ----------
    print(f"\n{'='*15} Results ({method_name}) {'='*15}")
    if meta.get('eval_view'):
        print(f"[Eval View] {meta['eval_view']} | gen={','.join(gen_sources) or 'none'}"
              f" | near={','.join(near_sources) or 'none'}"
              f" | n_gen={len(gen_scores)}, n_near={len(near_scores)}, n_far={len(far_scores)}")
    print(f"Global Acc:  {global_acc:.2f}%")
    print(f"Gen Acc:     {safe_acc(gen_preds, gen_targets)}")
    print("-" * 45)

    all_neg = np.concatenate([near_scores, far_scores]) if len(near_scores) > 0 or len(far_scores) > 0 else np.array([])
    global_auroc = None
    if gran_metrics:
        global_auroc = compute_auroc(all_neg)
        print(f"Strict Global AUROC: {get_auroc(all_neg)}")

    # ---------- 统一 CVRR / UNRR / H-Score（只一次） ----------
    cv_rr = cv_cr = cv_wrong_accepted = un_rr = h_score = threshold = None
    if not gran_metrics:
        print(f"\n[{method_name}] Unseen_In_Train 模式：只计算 Global Acc / Gen Acc，跳过粒度指标、AUROC 与 OSCR。")
    elif len(near_scores) > 0:
        cv_rr, cv_cr, cv_wrong_accepted, un_rr, h_score, threshold = calculate_cvr_unr(
            id_scores=id_scores,
            coarse_gen_scores=gen_scores,
            coarse_gen_preds=gen_preds,
            coarse_gen_targets=gen_targets,
            near_ood_scores=near_scores,
            tpr_target=tpr_target,
            method_name=method_name,
        )
    else:
        print(f"\n[{method_name}] 无可用于 CVRR/UNRR 的近邻OOD样本，跳过。")

    # ---------- Global OSCR ----------
    global_oscr = None
    if gran_metrics and include_oscr:
        print("-" * 45)
        print(f"[{method_name}] OSCR 综合指标评估")
        global_id_scores = np.concatenate([id_scores, gen_scores]) if len(gen_scores) > 0 else id_scores
        global_oscr = calculate_oscr(
            pred_k_id=global_id_scores,
            x_k_id=(total_preds == total_targets),
            pred_u_ood=all_neg if len(all_neg) > 0 else np.array([1.0]),
        )
        print(f"[{method_name}] Global OSCR: {global_oscr:.2f}%")

    # ---------- 数值化结果（视图对照出表用） ----------
    metrics = {
        'method': method_name,
        'dataset': meta.get('dataset', ''),
        'num_layers': meta.get('num_layers', ''),
        'seed': meta.get('seed', ''),
        'eval_view': meta.get('eval_view', 'V4'),
        'mode': mode,
        'tpr_target': tpr_target,
        'threshold': _to_float(threshold),
        # --- 各集合样本量（视图切换会改变这些分母）---
        'n_id': int(len(id_scores)),
        'n_gen_coarse': _n('coarse_gen'),
        'n_gen_medium': _n('medium_gen'),
        'n_gen': int(len(gen_scores)),
        'n_near_ood': _n('near_ood'),
        'n_near_family': _n('near_family'),
        'n_near_variant': _n('near_variant'),
        'n_near': int(len(near_scores)),
        'n_far': int(len(far_scores)),
        'gen_sources': ','.join(gen_sources),
        'near_sources': ','.join(near_sources),
        # --- 指标 ---
        'global_acc': _to_float(global_acc),
        'gen_acc': _to_float(100 * (gen_preds == gen_targets).mean()) if len(gen_targets) > 0 else None,
        'global_auroc': _to_float(global_auroc),
        'cv_cr': _to_float(cv_cr),
        'cv_wrong_accepted': _to_float(cv_wrong_accepted),
        'cv_rr': _to_float(cv_rr),
        'un_rr': _to_float(un_rr),
        'h_score': _to_float(h_score),
        'global_oscr': _to_float(global_oscr),
    }

    # ---------- 可选保存（供下游可视化 + 出表） ----------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{method_name}_id_scores.npy"), id_scores)
        np.save(os.path.join(save_dir, f"{method_name}_gen_scores.npy"), gen_scores)
        # 保存逐样本 preds/targets，便于离线重算 CV-CR（无需重跑 forward）
        if len(gen_preds) > 0:
            np.save(os.path.join(save_dir, f"{method_name}_gen_preds.npy"), gen_preds)
        if len(gen_targets) > 0:
            np.save(os.path.join(save_dir, f"{method_name}_gen_targets.npy"), gen_targets)
        if len(near_scores) > 0:
            np.save(os.path.join(save_dir, f"{method_name}_near_ood_scores.npy"), near_scores)
        if len(far_scores) > 0:
            np.save(os.path.join(save_dir, f"{method_name}_far_ood_scores.npy"), far_scores)
        write_metrics_record(save_dir, metrics)

    return metrics


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