import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_density_distribution(model_dir, method_name, plot_title=None, save_dir="./pictures/"):
    """
    绘制并保存顶会级的高清 KDE 分数密度分布图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 智能加载我们在 eval 阶段保存的数据包
    try:
        # 优先寻找带有方法前缀的文件 (防止多方法运行时的相互覆盖)
        id_path = os.path.join(model_dir, f"{method_name}_id_scores.npy")
        if not os.path.exists(id_path):
            id_path = os.path.join(model_dir, "id_scores.npy")
            
        gen_path = os.path.join(model_dir, f"{method_name}_gen_scores.npy")
        if not os.path.exists(gen_path):
            gen_path = os.path.join(model_dir, "gen_scores.npy")
            
        near_path = os.path.join(model_dir, f"{method_name}_near_ood_scores.npy")
        if not os.path.exists(near_path):
            near_path = os.path.join(model_dir, "near_ood_scores.npy")

        id_scores = np.load(id_path)
        gen_scores = np.load(gen_path)
        near_ood_scores = np.load(near_path)
    except FileNotFoundError:
        print(f"⚠️ 找不到 {model_dir} 下针对 {method_name} 的 .npy 分数文件！")
        print("请确保在评估代码末尾已正确保存：np.save(...)")
        return

    # 2. 全局样式设置 (符合 CVPR/ICCV 审美)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "grid.linestyle": "--"})
    plt.figure(figsize=(8, 5), dpi=300) # 高清画布
    
    # 3. 计算 95% TPR 的阈值 (在这个阈值之上，包含了 95% 的 ID 样本)
    threshold = np.percentile(id_scores, 5)

    # 4. 绘制叠加密度图 (KDE)
    # 蓝色: 已知类 (ID) - 代表绝对安全圈
    sns.kdeplot(id_scores, fill=True, color="#1f77b4", label="Seen Classes (ID)", linewidth=2, alpha=0.3)
    
    # 绿色: 合法变体 (Coarse Gen) - 测试误伤率 (CV-RR)
    sns.kdeplot(gen_scores, fill=True, color="#2ca02c", label="Unseen Variants (Coarse Gen)", linewidth=2, alpha=0.3)
    
    # 红色: 近邻伪装者 (Near-OOD) - 测试防守率 (UN-RR)
    sns.kdeplot(near_ood_scores, fill=True, color="#d62728", label="Near-OOD (Fake Neighbors)", linewidth=2, alpha=0.3)

    # 5. 画出那条至关重要的业务阈值防线
    plt.axvline(x=threshold, color='black', linestyle='-.', linewidth=2, label=f'TPR@95% Threshold ({threshold:.3f})')
    
    # 标记防线左右两侧的含义
    plt.text(threshold - 0.02, plt.ylim()[1]*0.8, 'Reject\n(Block)', color='red', ha='right', fontweight='bold')
    plt.text(threshold + 0.02, plt.ylim()[1]*0.8, 'Accept\n(Pass)', color='green', ha='left', fontweight='bold')

    # 6. 图表细节打磨
    final_title = plot_title if plot_title else method_name
    plt.title(f"Score Distribution Density - {final_title}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Confidence Score", fontsize=14, fontweight='bold')
    plt.ylabel("Density", fontsize=14, fontweight='bold')
    plt.xlim([0.0, 1.0]) # 根据你模型分数的实际范围调整
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')
    
    # 隐藏上方和右方的边框，让图表更清爽
    sns.despine(top=True, right=True)
    plt.tight_layout()

    # 7. 保存为无损 PDF 矢量图 (极度推荐，放大不失真) 和 PNG 预览图
    pdf_path = os.path.join(save_dir, f"KDE_{method_name}.pdf")
    png_path = os.path.join(save_dir, f"KDE_{method_name}.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    plt.close()
    
    print(f"✅ 完美图表已生成:\n -> {pdf_path}\n -> {png_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="绘制顶会级别的 KDE 分数分布图")
    parser.add_argument('--dataset', type=str, default='inaturalist', help='目标数据集名称 (如: inaturalist, cifar100)')
    parser.add_argument('--layers', type=int, default=2, help='分类的层级数 (如: 2, 3)')
    parser.add_argument('--seed', type=int, default=42, help='使用的随机种子 (如: 42, 44)')
    parser.add_argument('--method', type=str, required=True, help='需要绘制的方法名称 (如: A2Pt, ARPL)')
    parser.add_argument('--title', type=str, default=None, help='画板展示的自定义标题 (可选)')
    parser.add_argument('--model_dir', type=str, default=None, help='强行指定.npy文件所在的目录 (可选)')
    parser.add_argument('--save_dir', type=str, default=None, help='强行指定图片的保存目录 (可选)')
    
    args = parser.parse_args()
    
    # 动态构建对应的路径
    model_dir = args.model_dir if args.model_dir else f"./models/{args.dataset}/L{args.layers}/seed_{args.seed}"
    save_dir = args.save_dir if args.save_dir else f"./pictures/{args.dataset}/L{args.layers}/seed_{args.seed}"
    
    print(f"📊 准备绘制 {args.method} 在 {args.dataset} (Seed {args.seed}) 上的图表...")
    plot_density_distribution(model_dir, method_name=args.method, plot_title=args.title, save_dir=save_dir)