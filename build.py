import os
import json
import random

def parse_lisp_tree(lisp_str):
    tokens = lisp_str.replace('(', ' ( ').replace(')', ' ) ').split()
    def parse(tokens, index=0):
        if index >= len(tokens): return None, index
        token = tokens[index]
        if token == '(':
            node_name = tokens[index + 1]
            children = []
            next_idx = index + 2
            while next_idx < len(tokens) and tokens[next_idx] != ')':
                if tokens[next_idx] == '(':
                    child_tree, next_idx = parse(tokens, next_idx)
                    if child_tree: children.append(child_tree)
                else:
                    children.append({"name": tokens[next_idx], "children": []})
                    next_idx += 1
            return {"name": node_name, "children": children}, next_idx + 1
        return None, index
    tree, _ = parse(tokens)
    return tree

def get_leaves(node):
    if not node["children"]: return [node["name"]]
    leaves = []
    for child in node["children"]: leaves.extend(get_leaves(child))
    return leaves

# 🌟 融合核心 1：将 max_leaves 设为外部可控参数
def collect_all_valid_subtrees(node, min_leaves=2, max_leaves=20):  
    results = []
    leaves = get_leaves(node)
    
    is_l3_ready = len(node["children"]) >= 2 and all(len(get_leaves(c)) >= 2 for c in node["children"] if c["children"])
    
    # 根据传入的 max_leaves 动态过滤
    if min_leaves <= len(leaves) <= max_leaves:
        results.append({
            "name": node["name"],
            "node": node,
            "leaves": set(leaves),
            "is_l3_ready": is_l3_ready,
            "direct_families": [c for c in node["children"] if c["children"] or len(get_leaves(c)) >= 1]
        })
        
    for child in node["children"]:
        if child["children"]:
            results.extend(collect_all_valid_subtrees(child, min_leaves, max_leaves))
    return results

# 🌟 融合核心 2：在主构建函数中接收 max_leaves 和 max_subtrees
def build_mutually_exclusive_split(txt_file, dataset_name, num_layers=2, target_heads=60, seed=42, seen_ratio=0.5, max_leaves=20, max_subtrees=80):
    random.seed(seed)
    
    with open(txt_file, 'r', encoding='utf-8') as f: content = f.read()
    start_idx = content.find('(')
    tree = parse_lisp_tree(content[start_idx:])
    all_global_leaves = set(get_leaves(tree))
    
    # 传入动态的 max_leaves
    all_subtrees = collect_all_valid_subtrees(tree, min_leaves=2, max_leaves=max_leaves)
    if num_layers == 3:
        all_subtrees = [t for t in all_subtrees if t["is_l3_ready"]]
        
    random.shuffle(all_subtrees)
    
    selected_subtrees = []
    occupied_leaves = set()
    
    for candidate in all_subtrees:
        if not candidate["leaves"].isdisjoint(occupied_leaves):
            continue
            
        selected_subtrees.append(candidate)
        occupied_leaves.update(candidate["leaves"])
        
        # 传入动态的 max_subtrees 熔断上限
        if len(selected_subtrees) >= max_subtrees:
            break

    far_ood = sorted(list(all_global_leaves - occupied_leaves))
    
    split_config = {
        "coarse_id": [], "coarse_gen": [], 
        "medium_id": [], "medium_gen": [], "near_family": [], "medium_groups": [],
        "fine_id": [], "near_variant": [],
        "fine_groups": [],
        "far_ood": far_ood
    }
    
    random.shuffle(selected_subtrees)
    current_label = 0

    if num_layers == 2:
        target_coarse = int(target_heads * 0.3) 
        
        coarse_subs = selected_subtrees[:target_coarse]
        fine_subs = selected_subtrees[target_coarse:]
        
        for sub in coarse_subs:
            if current_label >= target_coarse: break 
            variants = sorted(list(sub["leaves"]))
            random.shuffle(variants)
            mid = max(1, len(variants) // 2)
            split_config["coarse_id"].append({"label": current_label, "head_name": f"{sub['name']} (C)", "seen_variants": variants[:mid]})
            split_config["coarse_gen"].append({"mapped_label": current_label, "unseen_variants": variants[mid:]})
            current_label += 1
            
        for sub in fine_subs:
            if current_label >= target_heads: break 
            variants = sorted(list(sub["leaves"]))
            random.shuffle(variants)
            
            seen_count = max(1, int(len(variants) * seen_ratio))
            needed = target_heads - current_label
            actual_seen_count = min(seen_count, needed)
            
            seen_v = variants[:actual_seen_count]
            unseen_v = variants[actual_seen_count:]
            
            group_info = {"family": sub["name"], "heads": [], "near_variants": unseen_v}
            for v in seen_v:
                split_config["fine_id"].append({"label": current_label, "head_name": f"{v} (F)", "seen_variants": [v]})
                group_info["heads"].append(f"L{current_label}:{v}") 
                current_label += 1
                
            split_config["near_variant"].extend(unseen_v)
            split_config["fine_groups"].append(group_info)

    elif num_layers == 3:
        target_coarse = int(target_heads * 0.2) 
        target_medium = int(target_heads * 0.3) 
        
        coarse_subs = selected_subtrees[:target_coarse]
        medium_subs = selected_subtrees[target_coarse : target_coarse + target_medium]
        fine_subs = selected_subtrees[target_coarse + target_medium :]
        
        for sub in coarse_subs:
            if current_label >= target_coarse: break 
            fams = sub["direct_families"]
            random.shuffle(fams)
            seen_fams = fams[:max(1, len(fams)//2)]
            unseen_fams = fams[max(1, len(fams)//2):]
            seen_v, unseen_v = [], []
            for f in seen_fams: seen_v.extend(get_leaves(f))
            for f in unseen_fams: unseen_v.extend(get_leaves(f))
            split_config["coarse_id"].append({"label": current_label, "head_name": f"{sub['name']} (Maker)", "seen_variants": seen_v})
            split_config["coarse_gen"].append({"mapped_label": current_label, "unseen_variants": unseen_v})
            current_label += 1
            
        target_medium_label = target_coarse + target_medium
        for sub in medium_subs:
            if current_label >= target_medium_label: break 
            fams = sub["direct_families"]
            random.shuffle(fams)
            target_fam = fams[0]
            sibling_fams = fams[1:]
            
            group_info = {
                "maker": sub["name"], 
                "target_family": target_fam["name"], 
                "near_families": [sib["name"] for sib in sibling_fams]
            }
            split_config["medium_groups"].append(group_info)
            
            variants = get_leaves(target_fam)
            random.shuffle(variants)
            mid = max(1, len(variants) // 2)
            split_config["medium_id"].append({"label": current_label, "head_name": f"{target_fam['name']} (Fam)", "seen_variants": variants[:mid]})
            split_config["medium_gen"].append({"mapped_label": current_label, "unseen_variants": variants[mid:]})
            current_label += 1
            for sib in sibling_fams: split_config["near_family"].extend(get_leaves(sib))
                
        for sub in fine_subs:
            if current_label >= target_heads: break 
            variants = sorted(list(sub["leaves"]))
            random.shuffle(variants)
            
            seen_count = max(1, int(len(variants) * seen_ratio))
            needed = target_heads - current_label
            actual_seen_count = min(seen_count, needed)
            
            seen_v = variants[:actual_seen_count]
            unseen_v = variants[actual_seen_count:]
            
            group_info = {"family": sub["name"], "heads": [], "near_variants": unseen_v}
            for v in seen_v:
                split_config["fine_id"].append({"label": current_label, "head_name": f"{v} (F)", "seen_variants": [v]})
                group_info["heads"].append(f"L{current_label}:{v}") 
                current_label += 1
                
            split_config["near_variant"].extend(unseen_v)
            split_config["fine_groups"].append(group_info)

    save_dir = f"splits/{dataset_name}/L{num_layers}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seed_{seed}.json")
    with open(save_path, 'w') as f:
        json.dump({"seed": seed, "total_id_classes": current_label, "split_config": split_config}, f, indent=4)


if __name__ == '__main__':
    # 🌟 融合核心 3：通过配置字典，完美实现不同数据集的动态路由！
    dataset_configs = [
        {
            'ds': 'imagenet', 
            'txt': '/home/geng/WTY/datasets/imagenet_tree.txt', 
            'heads': 60, 
            'max_leaves': 20, 
            'max_subtrees': 80
        },
        {
            'ds': 'inaturalist', 
            'txt': '/home/geng/WTY/datasets/inaturalist_2021.txt', 
            'heads': 200, 
            'max_leaves': 500, 
            'max_subtrees': 500  # iNat 需要大量子树备选，设定为 2.5 * heads
        }
    ]
    
    
    for config in dataset_configs:
        txt = config['txt']
        ds = config['ds']
        heads = config['heads']
        m_leaves = config['max_leaves']
        m_subs = config['max_subtrees']
        
        if os.path.exists(txt):
            print(f"\n[{ds.upper()}] 正在构建 (Target Heads: {heads}, Max Leaves: {m_leaves})...")
            for layers in [2, 3]:
                for s in [42, 43, 44, 45, 46]:
                    build_mutually_exclusive_split(
                        txt, ds, 
                        num_layers=layers, 
                        target_heads=heads, 
                        seed=s, 
                        max_leaves=m_leaves, 
                        max_subtrees=m_subs
                    )
                    print(f"  ✅ 成功生成或确认: splits/{ds}/L{layers}/seed_{s}.json")
        else:
            print(f"\n⚠️ 找不到文件: {txt}，已自动跳过 {ds} 的构建。")
            
    print("\n🎉 数据集拓扑配置全部生成/更新完毕！")