"""
TANL (Test-time Activated Negative Labels) for OOD Detection with Vision-Language Models
========================================================================================
论文: "Activation Matters: Test-time Activated Negative Labels for OOD Detection with
      Vision-Language Models" (CVPR 2026, arXiv:2603.25250, 代码 YBZh/OpenOOD-VLM)

方法定位: 完全 zero-shot / training-free 的测试时自适应方法。
  - CLIP 图像/文本编码器全程冻结、无任何可学习参数、无反向传播；
  - 测试时在线维护正/负 FIFO 队列 (Eq.9)，用激活差分 (Eq.8) 从语料库中动态挖掘
    "高激活" 负标签 (Eq.6)，并用激活感知评分函数 S_aa (Eq.15) 打分；
  - 阈值 gamma 由历史分数双峰方差最小化自动确定 (官方 find_best_threshold)。

核心超参（论文默认）:
  M     (--num_neg_labels) = 1000  每 batch 选出的激活负标签数
  g     (--gap)            = 0.2   高置信样本筛选间隔
  L     (--queue_len)      = 300   FIFO 队列长度
  alpha (--alpha)          = 0.95  历史/当前 batch 激活信息融合权重
  step  (--step)           = 1     S_aa 累积步长 (1 = 论文 Eq.15 精确实现)
  prompt                   = "The nice <label>"
  语料库                    = WordNet 名词+形容词 (去除 ID 内重复词), 与论文 140.5K 同量级

本脚本仿照仓库内 train_clip.py / eval_zsclip.py 的测评框架:
  - 加载同一套混合粒度 loaders (datasets.get_mixed_granularity_loaders)
  - 输出同一套指标: 全局/分层 Acc、AUROC (strict/near/far)、CV-RR/UN-RR/H-Score、OSCR

⚠️ TANL 是 transductive 的测试时自适应方法: 队列依赖"混合了 ID 与 OOD"的测试流。
   因此评估时把 test_id + 泛化集 + 近/远 OOD 各 loader 的 batch 顺序按 seed 打乱后
   合成单一测试流（论文用随机种子控制输入序列），流式跑完后按 loader 分组计算指标。
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch
import clip
from sklearn.metrics import roc_auc_score
from datasets import get_mixed_granularity_loaders
from utils import calculate_cvr_unr, calculate_oscr


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# =====================================================================
# 语料库构建 (论文 Appendix A6.5: 从 WordNet 选取名词与形容词, 去除 ID 内重复词)
# =====================================================================
def build_corpus_words(id_classnames, max_words=None):
    """返回负标签候选词表 (不含 ID 类名中的词)。"""
    from nltk.corpus import wordnet as wn
    words = set(wn.all_lemma_names(pos='n')) \
        | set(wn.all_lemma_names(pos='a')) \
        | set(wn.all_lemma_names(pos='s'))
    # 去除与 ID 类名重复的词 (论文: removing duplicate words within the ID set)
    id_words = set(name.replace(' ', '_').lower() for name in id_classnames)
    words = sorted(w for w in words if w.lower() not in id_words)
    if max_words is not None and len(words) > max_words:
        words = words[:max_words]
    return words


def encode_texts(clip_model, texts, device, batch_size=1024):
    """CLIP 文本编码 + L2 归一化 (float32)。"""
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = clip.tokenize(texts[i:i + batch_size]).to(device)
            f = clip_model.encode_text(tokens)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-5)
            feats.append(f.float())
    return torch.cat(feats, dim=0)


def build_noise_image_features(clip_model, device):
    """噪声图像特征: 5 类噪声 x 3 张 = 15 个特征 (官方实现), 用作负队列的初始化。"""
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min())

    gaussian = _norm(torch.randn((3, 3, 224, 224), dtype=torch.float, device=device))
    uniform = torch.rand((3, 3, 224, 224), dtype=torch.float, device=device)
    poisson = _norm(torch.poisson(5.0 * torch.ones((3, 3, 224, 224), device=device)))
    gamma = _norm(torch.distributions.Gamma(2, 1).sample((3, 3, 224, 224)).to(device))
    sp = torch.rand((3, 3, 224, 224), dtype=torch.float, device=device)
    sp[sp >= 0.5] = 1.0
    sp[sp < 0.5] = 0.0
    all_noise = torch.cat([gaussian, uniform, poisson, gamma, sp], dim=0)  # (15,3,224,224)
    with torch.no_grad():
        feats = clip_model.encode_image(all_noise)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-5)
    return feats.float()


# =====================================================================
# TANL 核心 (忠实还原论文 Eq.5-15 与官方 activated_neg_postprocessor_official.py)
# =====================================================================
class TANL:
    def __init__(self, clip_model, text_id, text_corpus, noise_feat,
                 num_neg_labels=1000, gap=0.2, alpha=0.95, queue_len=300,
                 step=1, score_queue_len=20000):
        self.logit_scale = clip_model.logit_scale.exp().item()  # CLIP 默认温度(等价论文 0.01)
        self.text_id = text_id          # (C, D) float32 L2 归一化 ID 文本特征
        self.text_corpus = text_corpus  # (N, D) 语料库负标签文本特征
        self.text_all = torch.cat([text_id, text_corpus], dim=0)  # (C+N, D)
        self.noise_feat = noise_feat    # (15, D)
        self.C = text_id.shape[0]
        self.N = text_corpus.shape[0]
        self.M = num_neg_labels      # Eq.6 中选出的激活负标签数
        self.gap = gap               # Eq.9 中高置信筛选间隔 g
        self.alpha = alpha           # Eq.12-14 历史/batch 融合权重
        self.queue_len = queue_len   # Eq.9 中 FIFO 队列容量 L
        self.step = step             # Eq.15 累积步长
        self.score_queue_len = score_queue_len
        self.reset()

    # ---- Eq.15: 激活感知评分 S_aa(v) = (1/M) * sum_m [ ID softmax sum over (ID + 前m个负标签) ] ----
    @staticmethod
    def activation_aware_score(output, id_num, ood_num, step=1):
        softmax_sums = []
        for i in range(id_num, id_num + ood_num, step):
            softmax_output = output[:, :i + step].softmax(dim=-1)
            softmax_sums.append(softmax_output[:, :id_num].sum(dim=-1))
        return torch.stack(softmax_sums, dim=-1).mean(dim=-1)

    # ---- 官方 find_best_threshold: 双峰方差最小化 (Otsu 式) 自动求阈值 gamma ----
    @staticmethod
    def find_best_threshold(scores):
        thresholds = torch.arange(0, 1, 0.01, device=scores.device, dtype=scores.dtype)
        vals = []
        for th in thresholds:
            mask = scores >= th
            n1 = mask.sum().item()
            n0 = scores.numel() - n1
            if n0 < 2 or n1 < 2:  # 任一分组元素过少时 var 无意义(会返回 NaN)，跳过
                vals.append(float('inf'))
                continue
            var0 = scores[~mask].var(unbiased=False)
            var1 = scores[mask].var(unbiased=False)
            vals.append(((n0 / scores.numel()) * var0 + (n1 / scores.numel()) * var1).item())
        vals = torch.tensor(vals, device=scores.device)
        min_val = torch.min(vals)
        cand = torch.where(vals == min_val)[0]
        if len(cand) == 0:
            return 0.0
        return thresholds[cand[len(cand) // 2]].item()

    # ---- Eq.10: 初始化 FIFO 队列 (正=ID标签特征采样, 负=噪声图像特征) ----
    def reset(self):
        with torch.no_grad():
            # 正队列: ID 文本特征对全部文本的 softmax 激活概率 (取负标签列)
            pos = self.logit_scale * self.text_id @ self.text_all.t()      # (C, C+N)
            pos = torch.softmax(pos, dim=-1)[:, self.C:].float()           # (C, N)
            pos = pos[torch.randperm(pos.size(0))]
            self.score_from_pos = pos[-self.queue_len:] if pos.size(0) > self.queue_len else pos
            # 负队列: 噪声图像特征对全部文本的 softmax 激活概率 (取负标签列)
            neg = self.logit_scale * self.noise_feat @ self.text_all.t()   # (15, C+N)
            neg = torch.softmax(neg, dim=-1)[:, self.C:].float()           # (15, N)
            neg = neg[torch.randperm(neg.size(0))]
            self.score_from_neg = neg[-self.queue_len:] if neg.size(0) > self.queue_len else neg
            # 初始历史分数队列: ID 标签特征行 + 噪声图像特征行的 S_aa
            conf_id = self._saa_score(self.text_id)
            conf_ood = self._saa_score(self.noise_feat)
            self.score_queue = torch.cat([conf_id, conf_ood], dim=0)

    # ---- 用当前队列选择 Top-M 激活负标签并计算 S_aa ----
    def _saa_score(self, feat):
        combined = self.score_from_neg.mean(0) - self.score_from_pos.mean(0)  # Eq.8 差分激活
        top_idx = combined.argsort(descending=True)[:self.M]                  # Eq.6 Top-M
        selected = torch.cat([self.text_id, self.text_corpus[top_idx]], dim=0)  # (C+M, D)
        out = self.logit_scale * feat @ selected.t()                          # (B, C+M)
        return self.activation_aware_score(out, self.C, self.M, self.step)

    # ---- 单 batch 前向 (Algorithm 1) ----
    @torch.no_grad()
    def score_batch(self, image_features):
        # image_features: (B, D) float32 L2 归一化
        # 图像对全部 (ID+语料库) 文本的 softmax 概率 (激活度量用)
        actscore = torch.softmax(self.logit_scale * image_features @ self.text_all.t(), dim=-1).float()
        act_neg = actscore[:, self.C:]                       # (B, N)
        preds = actscore[:, :self.C].argmax(dim=-1)          # zero-shot ID 分类预测

        # 1) 用历史队列选 Top-M 激活负标签, 算 S_aa
        combined = self.score_from_neg.mean(0) - self.score_from_pos.mean(0)
        top_idx = combined.argsort(descending=True)[:self.M]
        selected = torch.cat([self.text_id, self.text_corpus[top_idx]], dim=0)
        out = self.logit_scale * image_features @ selected.t()
        conf = self.activation_aware_score(out, self.C, self.M, self.step).float()  # (B,)

        # 2) 自动阈值: 历史分数双峰方差最小化
        self.score_queue = torch.cat([self.score_queue, conf], dim=0)[-self.score_queue_len:]
        thres = self.find_best_threshold(self.score_queue)
        pos_mask = conf > thres + self.gap * (1.0 - thres)    # Eq.9 高置信 ID
        ood_mask = conf < thres - self.gap * thres            # Eq.9 高置信 OOD

        # 3) batch 自适应激活 (Eq.12-14): 融合历史队列与当前 batch 的激活信息
        if pos_mask.any():
            ins_pos = self.alpha * self.score_from_pos.mean(0) + (1.0 - self.alpha) * act_neg[pos_mask].mean(0)
        else:
            ins_pos = self.score_from_pos.mean(0)
        if ood_mask.any():
            ins_neg = self.alpha * self.score_from_neg.mean(0) + (1.0 - self.alpha) * act_neg[ood_mask].mean(0)
        else:
            ins_neg = self.score_from_neg.mean(0)

        # 4) 用更新后的激活差分重新选负标签, 重算最终 S_aa
        combined = ins_neg - ins_pos
        top_idx = combined.argsort(descending=True)[:self.M]
        selected = torch.cat([self.text_id, self.text_corpus[top_idx]], dim=0)
        out = self.logit_scale * image_features @ selected.t()
        conf = self.activation_aware_score(out, self.C, self.M, self.step).float()

        # 5) FIFO 队列更新 (Eq.9)
        if pos_mask.any():
            self.score_from_pos = torch.cat([self.score_from_pos, act_neg[pos_mask]], dim=0)[-self.queue_len:]
        if ood_mask.any():
            self.score_from_neg = torch.cat([self.score_from_neg, act_neg[ood_mask]], dim=0)[-self.queue_len:]
        return conf, preds


# =====================================================================
# 混合流评估: TANL 是 transductive 方法, 需把 ID+泛化+OOD 合成单一测试流
# =====================================================================
def run_tanl_stream(tanl, clip_model, loaders, device, seed):
    keys = [k for k, l in loaders.items() if l is not None and k != 'train']
    counts = {k: len(loaders[k]) for k in keys}
    # 展开每个 loader 的 batch 序列, 按 seed 打乱 → 模拟论文的随机输入测试流
    # 注意: 某些子集可能为空(len==0, 如 L2 划分下无 test_medium_gen)，空 loader 不进流
    seq = []
    for k in keys:
        if counts[k] > 0:
            seq.extend((k, i) for i in range(counts[k]))
    rng = random.Random(seed)
    rng.shuffle(seq)
    iters = {k: iter(loaders[k]) for k in keys}

    results = {k: {'scores': [], 'preds': [], 'targets': []} for k in keys}
    tanl.reset()
    for k, _ in seq:
        images, targets = next(iters[k])
        images = images.to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(images)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-5)
            conf, preds = tanl.score_batch(feat.float())
        results[k]['scores'].append(conf.cpu().numpy())
        results[k]['preds'].append(preds.cpu().numpy())
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        results[k]['targets'].append(np.asarray(targets))

    out = {}
    for k in keys:
        sc = results[k]['scores']
        pr = results[k]['preds']
        tg = results[k]['targets']
        # 空 loader 返回空数组(而非空 list)，保证后续 np.concatenate / 指标函数兼容
        out[k] = (
            np.concatenate(sc) if len(sc) > 0 else np.array([], dtype=np.float32),
            np.concatenate(pr) if len(pr) > 0 else np.array([], dtype=np.int64),
            np.concatenate(tg) if len(tg) > 0 else np.array([], dtype=np.int64),
        )
    return out


# =====================================================================
# 主流程 (仿照 train_clip.py / eval_zsclip.py 的测评框架)
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'aircraft', 'imagenet', 'inaturalist'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--include_unseen', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3],
                        help='划分层级数：2层(Family->Variant) 或 3层(Maker->Family->Variant)')
    # ---------------- TANL 超参 (论文默认) ----------------
    parser.add_argument('--num_neg_labels', type=int, default=1000, help='M: 每batch激活负标签数 (论文默认1000)')
    parser.add_argument('--gap', type=float, default=0.2, help='g: 高置信样本筛选间隔 (论文默认0.2)')
    parser.add_argument('--alpha', type=float, default=0.95, help='历史/当前batch激活融合权重 (论文默认0.95)')
    parser.add_argument('--queue_len', type=int, default=300, help='L: FIFO队列长度 (论文默认300)')
    parser.add_argument('--step', type=int, default=1, help='S_aa累积步长 (1=论文Eq.15精确实现)')
    parser.add_argument('--ood_number', type=int, default=None,
                        help='语料库负标签词数上限 (默认None=全部WordNet名词+形容词)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    loaders, num_classes, classnames = get_mixed_granularity_loaders(
        dataset_name=args.dataset, seed=args.seed, batch_size=args.batch_size,
        include_unseen=args.include_unseen, far_ood_mode='fine',
        preprocess=preprocess, num_layers=args.num_layers
    )

    # 类名清洗 (与 eval_zsclip.py 完全一致)
    clean_names = []
    if args.dataset == 'imagenet':
        from nltk.corpus import wordnet as wn
        for name in classnames:
            raw_id = name.split(' ')[0]
            try:
                pos = raw_id[0]
                offset = int(raw_id[1:])
                synset = wn.synset_from_pos_and_offset(pos, offset)
                clean_names.append(synset.lemmas()[0].name().replace('_', ' '))
            except Exception:
                clean_names.append(raw_id.replace('_', ' '))
    elif args.dataset in ['aircraft', 'inaturalist']:
        clean_names = [name.rsplit(' (', 1)[0].replace('_', ' ') for name in classnames]
    else:
        clean_names = [name.replace('_', ' ') for name in classnames]

    print(f"\n[Prompt Check] Dataset: {args.dataset.upper()}")
    print(f"-> Head Names: {clean_names[:3]}")
    print(f"-> Tail Names: {clean_names[-3:]}\n")

    # ---------- 文本特征: ID 类 ----------
    id_prompts = [f"The nice {name}." for name in clean_names]
    text_id = encode_texts(clip_model, id_prompts, device)
    print(f"[TANL] ID text features: {text_id.shape}")

    # ---------- 语料库负标签文本特征 (WordNet 名词+形容词, 带磁盘缓存) ----------
    corpus_words = build_corpus_words(clean_names, max_words=args.ood_number)
    print(f"[TANL] Corpus size: {len(corpus_words)} (WordNet nouns+adjectives, ID 去重)")
    cache_dir = './models/tanl_cache'
    os.makedirs(cache_dir, exist_ok=True)
    words_hash = hashlib.sha1('|'.join(corpus_words).encode('utf-8')).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f'corpus_wordnet_{len(corpus_words)}_{words_hash}_ViT-B-32.pt')
    if os.path.exists(cache_path):
        print(f"[TANL] 加载语料库文本特征缓存: {cache_path}")
        text_corpus = torch.load(cache_path, map_location=device, weights_only=True)['feats'].to(device)
    else:
        print(f"[TANL] 编码 {len(corpus_words)} 个语料库词 (首次运行, 结果缓存至 {cache_path}) ...")
        corpus_prompts = [f"The nice {w.replace('_', ' ')}." for w in corpus_words]
        text_corpus = encode_texts(clip_model, corpus_prompts, device)
        torch.save({'words': corpus_words, 'feats': text_corpus.cpu()}, cache_path)
    print(f"[TANL] Corpus text features: {text_corpus.shape}")

    # ---------- 噪声图像特征 (负队列初始化) ----------
    noise_feat = build_noise_image_features(clip_model, device)
    print(f"[TANL] Noise image features: {noise_feat.shape}")

    # ---------- TANL 检测器 ----------
    tanl = TANL(clip_model, text_id=text_id, text_corpus=text_corpus, noise_feat=noise_feat,
                num_neg_labels=args.num_neg_labels, gap=args.gap, alpha=args.alpha,
                queue_len=args.queue_len, step=args.step)
    print(f"[TANL] M={tanl.M}, g={tanl.gap}, alpha={tanl.alpha}, L={tanl.queue_len}, step={tanl.step}")

    print("\n" + "=" * 50)
    print("STARTING TANL (Test-time Activated Negative Labels) EVALUATION")
    print("=" * 50)

    # ---------- 混合流评估 ----------
    scores = run_tanl_stream(tanl, clip_model, loaders, device, seed=args.seed)

    method_name = "TANL"
    id_scores, id_preds, id_targets = scores['test_id']
    pos_scores_strict = id_scores

    def safe_acc(preds, targets):
        return f"{100 * (preds == targets).mean():.2f}%" if len(targets) > 0 else "N/A"

    def safe_auroc(y_true, y_scores):
        return roc_auc_score(y_true, y_scores) * 100 if len(np.unique(y_true)) >= 2 else 0.0

    def get_auroc(neg_scores):
        if len(neg_scores) == 0:
            return "N/A"
        y_true = np.concatenate([np.ones(len(pos_scores_strict)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores_strict, neg_scores])
        return f"{safe_auroc(y_true, y_scores):.2f}%"

    if args.dataset in ['aircraft', 'imagenet', 'inaturalist']:
        c_gen_scores, c_gen_preds, c_gen_targets = scores['test_coarse_gen']
        m_gen_scores, m_gen_preds, m_gen_targets = scores['test_medium_gen']
        near_fam_scores, _, _ = scores['test_near_family']
        near_var_scores, _, _ = scores['test_near_variant']
        far_scores, _, _ = scores['test_far_ood']

        total_id_preds = np.concatenate([id_preds, c_gen_preds, m_gen_preds])
        total_id_targets = np.concatenate([id_targets, c_gen_targets, m_gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0

        print(f"\n{'='*15} TANL Results ({args.dataset.capitalize()}) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Coarse Gen Acc (Maker):  {safe_acc(c_gen_preds, c_gen_targets)}")
        print(f"Medium Gen Acc (Family): {safe_acc(m_gen_preds, m_gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC:       {get_auroc(np.concatenate([near_fam_scores, near_var_scores, far_scores]))}")
        print(f"Near-Family AUROC (跨族):  {get_auroc(near_fam_scores)}")
        print(f"Near-Variant AUROC (同族): {get_auroc(near_var_scores)}")
        print(f"Strict Far AUROC (隔离):   {get_auroc(far_scores)}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=c_gen_scores,
                          near_ood_scores=near_var_scores, tpr_target=0.95, method_name=method_name)
        if len(near_fam_scores) > 0:
            calculate_cvr_unr(id_scores=id_scores,
                              coarse_gen_scores=m_gen_scores if len(m_gen_scores) > 0 else c_gen_scores,
                              near_ood_scores=near_fam_scores, tpr_target=0.95, method_name=method_name)

        print("-" * 45)
        print(f"[{method_name}] OSCR 综合指标评估")
        all_neg = np.concatenate([near_fam_scores, near_var_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, c_gen_scores, m_gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(pred_k_id=global_id_scores, x_k_id=global_correct_mask, pred_u_ood=all_neg)
        print(f"[{method_name}] Global OSCR: {global_oscr:.2f}%")
        if len(c_gen_scores) > 0 and len(near_var_scores) > 0:
            mg_correct_mask = (c_gen_preds == c_gen_targets)
            mg_oscr = calculate_oscr(pred_k_id=c_gen_scores, x_k_id=mg_correct_mask, pred_u_ood=near_var_scores)
            print(f"[{method_name}] MG-OSCR    : {mg_oscr:.2f}%")

    elif args.dataset == 'cifar100':
        gen_scores, gen_preds, gen_targets = scores['test_coarse_gen']
        near_scores, _, _ = scores['test_near_ood']
        far_scores, _, _ = scores['test_far_ood']

        total_id_preds = np.concatenate([id_preds, gen_preds])
        total_id_targets = np.concatenate([id_targets, gen_targets])
        global_acc = 100 * (total_id_preds == total_id_targets).mean() if len(total_id_targets) > 0 else 0.0

        print(f"\n{'='*15} TANL Results (CIFAR-100) {'='*15}")
        print(f"Global Acc:  {global_acc:.2f}%")
        print(f"Gen Acc: {safe_acc(gen_preds, gen_targets)}")
        print("-" * 45)
        print(f"Strict Global AUROC: {get_auroc(np.concatenate([near_scores, far_scores]))}")

        calculate_cvr_unr(id_scores=id_scores, coarse_gen_scores=gen_scores,
                          near_ood_scores=near_scores, tpr_target=0.95, method_name=method_name)

        print("-" * 45)
        print(f"[{method_name}] OSCR 综合指标评估")
        all_neg = np.concatenate([near_scores, far_scores])
        global_id_scores = np.concatenate([id_scores, gen_scores])
        global_correct_mask = (total_id_preds == total_id_targets)
        global_oscr = calculate_oscr(pred_k_id=global_id_scores, x_k_id=global_correct_mask, pred_u_ood=all_neg)
        print(f"[{method_name}] Global OSCR: {global_oscr:.2f}%")
        if len(gen_scores) > 0 and len(near_scores) > 0:
            mg_correct_mask = (gen_preds == gen_targets)
            mg_oscr = calculate_oscr(pred_k_id=gen_scores, x_k_id=mg_correct_mask, pred_u_ood=near_scores)
            print(f"[{method_name}] MG-OSCR    : {mg_oscr:.2f}%")

        # ---- 保存分数用于层次冲突可视化 ----
        np.save(os.path.join(model_dir, f"{method_name}_id_scores.npy"), id_scores)
        np.save(os.path.join(model_dir, f"{method_name}_gen_scores.npy"), gen_scores)
        np.save(os.path.join(model_dir, f"{method_name}_near_ood_scores.npy"), near_scores)
        np.save(os.path.join(model_dir, f"{method_name}_far_ood_scores.npy"), far_scores)

    print("\n" + "=" * 50)
    print("TANL EVALUATION FINISHED")
    print("=" * 50)


if __name__ == '__main__':
    main()
