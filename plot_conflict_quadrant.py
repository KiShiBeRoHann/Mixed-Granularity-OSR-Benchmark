# -*- coding: utf-8 -*-
"""层次冲突象限散点图
横轴 = Coarse Variant 接受率（泛化能力，越高越好）
纵轴 = Fine Variant 接受率（误收程度，越低越好）
理想模型在右下；右上 = 高泛化但高误收（细粒度误收冲突）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, argparse

NPY_MAP = {
    'Zero-Shot CLIP': 'ZeroShotCLIP', 'Linear Probe Clip': 'LinearProbeCLIP',
    'CoOp': 'CoOp', 'CoCoOp': 'CoCoOp', 'A2PT': 'A2Pt', 'FSMoE': 'FSMoE',
    'MLS': 'VGG_MLS', 'ARPL+CS': 'ARPL', 'DEF': 'DeF', 'TANL': 'TANL',
}
BIG = ['Zero-Shot CLIP', 'TANL', 'Linear Probe Clip', 'CoOp', 'CoCoOp', 'A2PT', 'FSMoE']
TRAD = ['MLS', 'ARPL+CS', 'DEF']

def load(base, prefix, tag, seeds):
    arrs = []
    for sd in seeds:
        p = os.path.join(base, f'seed_{sd}', f'{prefix}_{tag}_scores.npy')
        if os.path.exists(p):
            arrs.append(np.load(p))
    return np.concatenate(arrs) if arrs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    thr = args.threshold
    base = f'models/{args.dataset}/L2'
    seeds = [42,43,44,45,46]

    def acc_rate(s):
        return (s > thr).mean() * 100 if s is not None and len(s) else None

    pts = {}
    for m in BIG + TRAD:
        pre = NPY_MAP.get(m)
        if not pre: continue
        g = load(base, pre, 'gen', seeds)
        n = load(base, pre, 'near_ood', seeds)
        if g is None or n is None:
            print(f'  [skip] {m}: 缺 npy'); continue
        pts[m] = (acc_rate(g), acc_rate(n))
        print(f'  {m}: CoarseVariant接受={pts[m][0]:.1f}%  FineVariant接受={pts[m][1]:.1f}%')

    if not pts: print('无数据'); return
    os.makedirs('pictures/hierarchy', exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    ax.text(0.99, 0.97, 'Ideal: high coarse-accept,\nlow fine-accept (no conflict)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color='green')

    for m in BIG:
        if m in pts:
            x, y = pts[m]
            ax.scatter(x, y, marker='o', s=130, color='#1f77b4', edgecolor='black', linewidth=0.8, zorder=3)
            ax.annotate(m.replace('Zero-Shot CLIP','ZS-CLIP').replace('Linear Probe Clip','LP'),
                        (x, y), fontsize=8, xytext=(6, 6), textcoords='offset points')
    for m in TRAD:
        if m in pts:
            x, y = pts[m]
            ax.scatter(x, y, marker='^', s=130, color='#d62728', edgecolor='black', linewidth=0.8, zorder=3)
            ax.annotate(m, (x, y), fontsize=8, xytext=(6, 6), textcoords='offset points')

    ax.set_xlabel('Coarse Variant Acceptance Rate (%)  (generalization, higher is better)', fontsize=11)
    ax.set_ylabel('Fine Variant Acceptance Rate (%)  (mis-acceptance, lower is better)', fontsize=11)
    ax.set_title(f'Hierarchical Conflict: Coarse Generalization vs Fine Acceptance ({args.dataset})', fontsize=12)
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.4)
    from matplotlib.lines import Line2D
    legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='VLM-based'),
              Line2D([0],[0], marker='^', color='w', markerfacecolor='#d62728', markersize=10, label='Traditional')]
    ax.legend(handles=legend, fontsize=10)
    plt.tight_layout()
    fig.savefig(f'pictures/hierarchy/hierarchy_conflict_quadrant_{args.dataset.lower()}.png')
    plt.close(fig)
    print(f'已保存: pictures/hierarchy/hierarchy_conflict_quadrant_{args.dataset.lower()}.png')

if __name__ == '__main__':
    main()
