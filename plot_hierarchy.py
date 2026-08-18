# -*- coding: utf-8 -*-
"""最终版：层次冲突可视化 + ACC-AUROC 散点图
用法: python plot_final.py --dataset cifar100 [--method A2Pt] [--all]
读取 models/{dataset}/L2/seed_*/{Method}_{id,gen,near_ood,far_ood}_scores.npy
固定阈值 THRESHOLD，输出到 pictures/hierarchy/
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, glob, argparse, json

THRESHOLD = 0.5  # 固定阈值

METHOD_MAP = {
    'A2Pt': 'A2Pt', 'CoOp': 'CoOp', 'CoCoOp': 'CoCoOp', 'FSMoE': 'FSMoE',
    'LinearProbeCLIP': 'Linear Probe Clip', 'VGG_MLS': 'MLS',
    'ARPL': 'ARPL+CS', 'DeF': 'DEF', 'ZeroShotCLIP': 'Zero-Shot CLIP', 'TANL': 'TANL',
}
# 表格显示名 -> npy 前缀
NPY_MAP = {
    'Zero-Shot CLIP': 'ZeroShotCLIP', 'Linear Probe Clip': 'LinearProbeCLIP',
    'CoOp': 'CoOp', 'CoCoOp': 'CoCoOp', 'A2PT': 'A2Pt', 'FSMoE': 'FSMoE',
    'MLS': 'VGG_MLS', 'ARPL+CS': 'ARPL', 'DEF': 'DeF', 'TANL': 'TANL',
}
LAYERS = ['ID', 'Coarse', 'Near', 'Far']
LAYER_LABELS = {'ID': 'Known ID', 'Coarse': 'Coarse Variant', 'Near': 'Fine Variant', 'Far': 'Random Category'}
BIG_METHODS = ['Zero-Shot CLIP', 'TANL', 'Linear Probe Clip', 'CoOp', 'CoCoOp', 'A2PT', 'FSMoE']
TRAD_METHODS = ['MLS', 'ARPL+CS', 'DEF']
ALL_METHODS = BIG_METHODS + TRAD_METHODS

def load_method_scores(base_dir, npy_prefix, seeds=[42,43,44,45,46]):
    """跨 seed 合并各层分数"""
    out = {}
    for layer, tag in [('ID','id'), ('Coarse','gen'), ('Near','near_ood'), ('Far','far_ood')]:
        arrs = []
        for sd in seeds:
            p = os.path.join(base_dir, f'seed_{sd}', f'{npy_prefix}_{tag}_scores.npy')
            if os.path.exists(p):
                arrs.append(np.load(p))
        if arrs:
            out[layer] = np.concatenate(arrs)
    return out if out else None

def acceptance_rate(scores):
    if scores is None or len(scores) == 0:
        return None
    return (scores > THRESHOLD).mean() * 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--methods', nargs='*', default=None, help='方法名列表，默认全部')
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    global THRESHOLD
    THRESHOLD = args.threshold

    base_dir = f'models/{args.dataset}/L2'
    methods = args.methods if args.methods else ALL_METHODS

    acc = {}
    for m in methods:
        npy_p = NPY_MAP.get(m)
        if npy_p is None: continue
        d = load_method_scores(base_dir, npy_p)
        if d is None:
            print(f'  [skip] {m}: 无 npy 数据')
            continue
        acc[m] = {l: acceptance_rate(d.get(l)) for l in LAYERS}
        print(f'  {m}: ' + ' '.join(f'{l}={acc[m][l]:.1f}%' if acc[m][l] is not None else f'{l}=N/A' for l in LAYERS))

    if not acc:
        print('没有任何方法的 npy 数据，请先运行实验生成分数文件'); return

    os.makedirs('pictures/hierarchy', exist_ok=True)
    tag = args.dataset.lower()

    # ============ 图1: 层次接受率热力图 ============
    methods_plot = list(acc.keys())
    data = np.array([[acc[m][l] if acc[m][l] is not None else np.nan for l in LAYERS] for m in methods_plot])
    sns.set_theme(style="white", rc={"axes.edgecolor": "black"})
    fig, ax = plt.subplots(figsize=(max(6, len(LAYERS)*1.6), max(4, len(methods_plot)*0.7)), dpi=300)
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r', vmin=0, vmax=100,
                xticklabels=[LAYER_LABELS[l] for l in LAYERS], yticklabels=methods_plot,
                cbar_kws={'label': 'Acceptance Rate (%)'}, ax=ax, annot_kws={'size': 11})
    ax.set_title(f'Hierarchical Acceptance Rate ({args.dataset}, Fixed Thr={THRESHOLD})', fontsize=12)
    plt.tight_layout()
    fig.savefig(f'pictures/hierarchy/hierarchy_heatmap_{tag}.png')
    plt.close(fig)
    print(f'已保存: pictures/hierarchy/hierarchy_heatmap_{tag}.png')

    # ============ 图2: 层次轨迹折线图 ============
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = sns.color_palette('deep', len(methods_plot))
    x = np.arange(len(LAYERS))
    for i, m in enumerate(methods_plot):
        vals = [acc[m][l] if acc[m][l] is not None else np.nan for l in LAYERS]
        ax.plot(x, vals, 'o-', color=colors[i], label=m, linewidth=2, markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS], fontsize=11)
    ax.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax.set_title(f'Hierarchical Trajectory: Coarse Rejection vs Fine Acceptance ({args.dataset})', fontsize=12)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axvspan(-0.4, 0.4, color='green', alpha=0.08)
    ax.axvspan(1.6, 2.4, color='orange', alpha=0.08)
    ax.text(0, 108, 'coarse\nrejection', ha='center', fontsize=8, color='green')
    ax.text(2, 108, 'fine\nacceptance', ha='center', fontsize=8, color='orange')
    plt.tight_layout()
    fig.savefig(f'pictures/hierarchy/hierarchy_trajectory_{tag}.png')
    plt.close(fig)
    print(f'已保存: pictures/hierarchy/hierarchy_trajectory_{tag}.png')

if __name__ == '__main__':
    main()
