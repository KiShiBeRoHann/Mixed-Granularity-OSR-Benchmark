# -*- coding: utf-8 -*-
"""ACC vs AUROC/OSCR 散点图：展示"传统模型 ACC 落后但 AUROC/OSCR 持平"
从日志解析各方法 Global Acc / AUROC / OSCR，跨 5 seed 取平均
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, re, glob, argparse

METHOD_DIR = {
    'Zero-Shot CLIP': 'eval_zsclip', 'TANL': 'train_tanl',
    'Linear Probe Clip': 'train_clip', 'CoOp': 'train_coop', 'CoCoOp': 'train_cocoop',
    'A2PT': 'train_a2pt', 'FSMoE': 'train_fsmoe',
    'MLS': 'train_vgg_mls', 'ARPL+CS': 'train_arpl', 'DEF': 'train_def',
}
BIG = ['Zero-Shot CLIP', 'TANL', 'Linear Probe Clip', 'CoOp', 'CoCoOp', 'A2PT', 'FSMoE']
TRAD = ['MLS', 'ARPL+CS', 'DEF']

def parse_log(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        c = f.read()
    segs = re.split(r'>>> 正在运行 SEED: (\d+)', c)
    res = {}
    for i in range(1, len(segs), 2):
        body = segs[i+1]
        g = re.search(r'Global Acc:\s+([\d.]+)%', body)
        a = re.search(r'Strict Global AUROC:\s*([\d.]+)%', body)
        o = re.search(r'Global OSCR\s*:\s*([\d.]+)%', body)
        res[int(segs[i])] = {'acc': float(g.group(1)) if g else None,
                             'auroc': float(a.group(1)) if a else None,
                             'oscr': float(o.group(1)) if o else None}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--logdir', default=None)
    args = ap.parse_args()

    # 日志目录: cifar100 无 L2 子目录
    base = args.logdir or f'experiment_logs/{args.dataset}'
    if not os.path.exists(os.path.join(base, 'train_a2pt')):
        base = os.path.join(base, 'L2')

    data = {}
    for m, d in METHOD_DIR.items():
        p = os.path.join(base, d, 'standard.log')
        if not os.path.exists(p): continue
        r = parse_log(p)
        vals = [r[s] for s in [42,43,44,45,46] if s in r]
        if not vals: continue
        data[m] = {
            'acc': np.mean([v['acc'] for v in vals if v['acc'] is not None]),
            'auroc': np.mean([v['auroc'] for v in vals if v['auroc'] is not None]),
            'oscr': np.mean([v['oscr'] for v in vals if v['oscr'] is not None]),
        }
        print(f'  {m}: ACC={data[m]["acc"]:.1f} AUROC={data[m]["auroc"]:.1f} OSCR={data[m]["oscr"]:.1f}')

    if not data: print('无数据'); return
    os.makedirs('pictures/hierarchy', exist_ok=True)
    tag = args.dataset.lower()
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})

    for ykey, ylabel in [('auroc', 'Global AUROC (%)'), ('oscr', 'Global OSCR (%)')]:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for m in BIG:
            if m in data:
                ax.scatter(data[m]['acc'], data[m][ykey], marker='o', s=120, color='#1f77b4',
                           edgecolor='black', linewidth=0.8, label=m if m in ('A2PT','FSMoE','CoOp') else '_nolegend_')
                ax.annotate(m.replace('Zero-Shot CLIP','ZS-CLIP').replace('Linear Probe Clip','LP'),
                            (data[m]['acc'], data[m][ykey]), fontsize=8, xytext=(5,5), textcoords='offset points')
        for m in TRAD:
            if m in data:
                ax.scatter(data[m]['acc'], data[m][ykey], marker='^', s=120, color='#d62728',
                           edgecolor='black', linewidth=0.8, label=m if m in ('MLS','ARPL+CS','DEF') else '_nolegend_')
                ax.annotate(m, (data[m]['acc'], data[m][ykey]), fontsize=8, xytext=(5,5), textcoords='offset points')
        ax.set_xlabel('Global Acc (%)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'ACC vs {ylabel.split()[1]} ({args.dataset}): Classification vs Open-Set', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        fig.savefig(f'pictures/hierarchy/acc_vs_{ykey}_{tag}.png')
        plt.close(fig)
        print(f'已保存: pictures/hierarchy/acc_vs_{ykey}_{tag}.png')

if __name__ == '__main__':
    main()
