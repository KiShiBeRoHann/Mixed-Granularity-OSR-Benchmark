import os
import json
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FGVCAircraftMixedGranularity:
    def __init__(self, root='./datasets/fgvc-aircraft-2013b', seed=42, seen_ratio=0.5, num_layers=2):
        self.root = root
        self.data_dir = os.path.join(root, 'data')
        self.seed = seed
        self.seen_ratio = seen_ratio
        self.num_layers = num_layers  # 2 或 3
        
        self.variant_to_family = {}
        self.variant_to_maker = {}
        self.maker_to_families = {}
        self.family_to_variants = {}
        
        self._build_hierarchy_tree()
        # 【重要修改】：防止两层和三层的 JSON 命名冲突
        self.split_dir = f'splits/aircraft/L{num_layers}'
        os.makedirs(self.split_dir, exist_ok=True)
        self.split_config_path = os.path.join(self.split_dir, f'seed_{self.seed}.json')
        
        self._define_mixed_granularity_splits()

    def _build_hierarchy_tree(self):
        for phase in ['train', 'val', 'test']:
            v_file = os.path.join(self.data_dir, f'images_variant_{phase}.txt')
            f_file = os.path.join(self.data_dir, f'images_family_{phase}.txt')
            m_file = os.path.join(self.data_dir, f'images_manufacturer_{phase}.txt')
            if not (os.path.exists(v_file) and os.path.exists(f_file) and os.path.exists(m_file)): continue
            with open(v_file, 'r') as f_v, open(f_file, 'r') as f_f, open(m_file, 'r') as f_m:
                for line_v, line_f, line_m in zip(f_v, f_f, f_m):
                    parts_v = line_v.strip().split(' ', 1)
                    parts_f = line_f.strip().split(' ', 1)
                    parts_m = line_m.strip().split(' ', 1)
                    if len(parts_v) < 2: continue
                    v_name, f_name, m_name = parts_v[1], parts_f[1], parts_m[1]
                    self.variant_to_family[v_name] = f_name
                    self.variant_to_maker[v_name] = m_name
                    if m_name not in self.maker_to_families: self.maker_to_families[m_name] = set()
                    self.maker_to_families[m_name].add(f_name)
                    if f_name not in self.family_to_variants: self.family_to_variants[f_name] = set()
                    self.family_to_variants[f_name].add(v_name)

        self.maker_to_families = {k: sorted(list(v)) for k, v in self.maker_to_families.items()}
        self.family_to_variants = {k: sorted(list(v)) for k, v in self.family_to_variants.items()}

    def _define_mixed_granularity_splits(self):
        random.seed(self.seed)
        
        self.split_config = {
            "coarse_id": [], "coarse_gen": [], 
            "medium_id": [], "medium_gen": [], "near_family": [], "medium_groups": [],
            "fine_id": [], "near_variant": [], "fine_groups": [],
            "far_ood": []
        }
        current_label = 0
        target_heads = 20  # 🎯 强制锁定 FGVC 的总分类头数量为 20

        # ========================================================
        # ✌️ 分支 A: 统一的两层划分模式 (Family -> Variant)
        # ========================================================
        if self.num_layers == 2:
            all_families = sorted(list(self.family_to_variants.keys()))
            # 严格过滤候选池：家族里必须至少有 2 个型号（过滤掉独生子，确保内斗纯粹）
            valid_families = [f for f in all_families if len(self.family_to_variants[f]) >= 2]
            random.shuffle(valid_families)
            
            coarse_fams = valid_families[:6]
            fine_fams = valid_families[6:]
            used_fams = set(coarse_fams + fine_fams)
            # 剩下的全划入远端隔离 OOD
            far_fams = [f for f in all_families if f not in used_fams]
            
            target_coarse = int(target_heads * 0.3) # 20 的 30% = 6 个粗粒度头

            # 1. 粗粒度层 (Family 标签, 测试型号泛化)
            for fam in coarse_fams:
                if current_label >= target_coarse: break # 🎯 Coarse 额度熔断
                variants = self.family_to_variants[fam]
                random.shuffle(variants)
                split_idx = max(1, len(variants) // 2)
                seen_v = variants[:split_idx]
                unseen_v = variants[split_idx:]
                
                self.split_config["coarse_id"].append({
                    "label": current_label, "head_name": f"{fam} (Family)", "seen_variants": seen_v
                })
                self.split_config["coarse_gen"].append({"mapped_label": current_label, "unseen_variants": unseen_v})
                current_label += 1
                
            # 2. 细粒度层 (Variant 标签, 触发兄弟近邻内斗)
            for fam in fine_fams:
                if current_label >= target_heads: break # 🎯 总额度熔断
                variants = self.family_to_variants[fam]
                random.shuffle(variants)
                
                seen_count = max(1, int(len(variants) * self.seen_ratio))
                
                # 🚀 极限截断：计算剩余的坑位，防止越界
                needed = target_heads - current_label
                actual_seen_count = min(seen_count, needed)
                
                seen_v = variants[:actual_seen_count]
                unseen_v = variants[actual_seen_count:]
                
                # 💡 新增：追踪家族内斗
                group_info = {"family": fam, "heads": [], "near_variants": unseen_v}
                
                for v in seen_v:
                    self.split_config["fine_id"].append({
                        "label": current_label, "head_name": f"{v} (Variant)", "seen_variants": [v]
                    })
                    group_info["heads"].append(f"L{current_label}:{v}")
                    current_label += 1
                    
                self.split_config["near_variant"].extend(unseen_v)
                self.split_config["fine_groups"].append(group_info) 
                
            # 3. 远端隔离层 (Far OOD)
            for fam in far_fams:
                self.split_config["far_ood"].extend(self.family_to_variants[fam])

        # ========================================================
        # 🤟 分支 B: 经典的三层划分模式 (Maker -> Family -> Variant)
        # ========================================================
        elif self.num_layers == 3:
            all_makers = sorted(list(self.maker_to_families.keys()))
            random.shuffle(all_makers)
            
            coarse_pool = [m for m in all_makers if len(self.maker_to_families[m]) >= 2]
            medium_pool = []
            for m in all_makers:
                fams = self.maker_to_families[m]
                if len(fams) >= 2 and any(len(self.family_to_variants[f]) >= 2 for f in fams):
                    medium_pool.append(m)
            fine_pool = [m for m in all_makers if any(len(self.family_to_variants[f]) >= 2 for f in self.maker_to_families[m])]

            coarse_makers = coarse_pool[:4]
            used = set(coarse_makers)
            medium_makers = [m for m in medium_pool if m not in used][:4]
            used.update(medium_makers)
            fine_makers = [m for m in fine_pool if m not in used][:6]
            used.update(fine_makers)
            far_makers = [m for m in all_makers if m not in used]

            # L3 配额：20% Maker (4个头), 30% Family (6个头), 50% Variant (10个头) = 20个
            target_coarse = int(target_heads * 0.2)
            target_medium = target_coarse + int(target_heads * 0.3)

            # --- A. Coarse 级 ---
            for maker in coarse_makers:
                if current_label >= target_coarse: break # 🎯 Coarse 额度熔断
                families = self.maker_to_families[maker]
                random.shuffle(families)
                seen_f = families[:max(1, len(families)//2)]
                unseen_f = families[max(1, len(families)//2):]
                seen_vars, unseen_vars = [], []
                for f in seen_f: seen_vars.extend(self.family_to_variants[f])
                for f in unseen_f: unseen_vars.extend(self.family_to_variants[f])
                
                self.split_config["coarse_id"].append({
                    "label": current_label, "head_name": f"{maker} (Maker)", "seen_variants": seen_vars
                })
                self.split_config["coarse_gen"].append({"mapped_label": current_label, "unseen_variants": unseen_vars})
                current_label += 1

            # --- B. Medium 级 ---
            for maker in medium_makers:
                if current_label >= target_medium: break # 🎯 Medium 额度熔断
                families = self.maker_to_families[maker]
                random.shuffle(families)
                target_f = next(f for f in families if len(self.family_to_variants[f]) >= 2)
                sibling_f = [f for f in families if f != target_f]
                
                # 💡 追踪跨族内斗
                group_info = {
                    "maker": maker, "target_family": target_f, "near_families": sibling_f
                }
                self.split_config["medium_groups"].append(group_info)
                
                variants = self.family_to_variants[target_f]
                random.shuffle(variants)
                seen_v = variants[:max(1, len(variants)//2)]
                unseen_v = variants[max(1, len(variants)//2):]
                
                self.split_config["medium_id"].append({
                    "label": current_label, "head_name": f"{target_f} (Family)", "seen_variants": seen_v
                })
                self.split_config["medium_gen"].append({"mapped_label": current_label, "unseen_variants": unseen_v})
                current_label += 1
                for sib in sibling_f: self.split_config["near_family"].extend(self.family_to_variants[sib])

            # --- C. Fine 级 ---
            for maker in fine_makers:
                if current_label >= target_heads: break # 🎯 总额度熔断
                families = self.maker_to_families[maker]
                for f in families:
                    if current_label >= target_heads: break # 再次检查
                    variants = self.family_to_variants[f]
                    if len(variants) < 2: continue
                    random.shuffle(variants)
                    
                    seen_count = max(1, int(len(variants) * self.seen_ratio))
                    
                    # 🚀 极限截断
                    needed = target_heads - current_label
                    actual_seen_count = min(seen_count, needed) 
                    
                    seen_v = variants[:actual_seen_count]
                    unseen_v = variants[actual_seen_count:]
                    
                    # 💡 追踪家族内斗
                    group_info = {"family": f, "heads": [], "near_variants": unseen_v}
                    
                    for v in seen_v:
                        self.split_config["fine_id"].append({
                            "label": current_label, "head_name": f"{v} (Variant)", "seen_variants": [v]
                        })
                        group_info["heads"].append(f"L{current_label}:{v}")
                        current_label += 1
                        
                    self.split_config["near_variant"].extend(unseen_v)
                    self.split_config["fine_groups"].append(group_info)

            # --- D. Far 级 ---
            for maker in far_makers:
                for f in self.maker_to_families[maker]:
                    self.split_config["far_ood"].extend(self.family_to_variants[f])

        self.total_id_classes = current_label
        with open(self.split_config_path, 'w') as f:
            json.dump({"seed": self.seed, "total_id_classes": self.total_id_classes, "split_config": self.split_config}, f, indent=4)



            
    def get_datasets(self, include_unseen=False):
        train_map, coarse_gen_map, medium_gen_map = {}, {}, {}
        for item in self.split_config["coarse_id"]:
            for v in item["seen_variants"]: train_map[v] = item["label"]
        for item in self.split_config["medium_id"]:
            for v in item["seen_variants"]: train_map[v] = item["label"]
        for item in self.split_config["fine_id"]:
            for v in item["seen_variants"]: train_map[v] = item["label"]
            
        for item in self.split_config["coarse_gen"]:
            for v in item["unseen_variants"]: coarse_gen_map[v] = item["mapped_label"]
        for item in self.split_config["medium_gen"]:
            for v in item["unseen_variants"]: medium_gen_map[v] = item["mapped_label"]

        near_fam_set = set(self.split_config["near_family"])
        near_var_set = set(self.split_config["near_variant"])
        far_ood_set = set(self.split_config["far_ood"])

        train_data, test_id_data = [], []
        test_c_gen, test_m_gen = [], []
        test_near_f, test_near_v, test_far = [], [], []

        for phase in ['train', 'val', 'test']:
            filepath = os.path.join(self.data_dir, f'images_variant_{phase}.txt')
            if not os.path.exists(filepath): continue
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) != 2: continue
                    img_id, v_name = parts[0], parts[1]
                    
                    if phase in ['train', 'val']:
                        if v_name in train_map: 
                            train_data.append((img_id, train_map[v_name]))
                        elif include_unseen:
                            if v_name in coarse_gen_map: train_data.append((img_id, coarse_gen_map[v_name]))
                            elif v_name in medium_gen_map: train_data.append((img_id, medium_gen_map[v_name]))

                    if phase == 'test':
                        if v_name in train_map: test_id_data.append((img_id, train_map[v_name]))
                        elif v_name in coarse_gen_map: test_c_gen.append((img_id, coarse_gen_map[v_name]))
                        elif v_name in medium_gen_map: test_m_gen.append((img_id, medium_gen_map[v_name]))
                        elif v_name in near_fam_set: test_near_f.append((img_id, v_name))
                        elif v_name in near_var_set: test_near_v.append((img_id, v_name))
                        elif v_name in far_ood_set: test_far.append((img_id, v_name))

        return {
            "train": AircraftSubDataset(self.root, train_data),
            "test_id": AircraftSubDataset(self.root, test_id_data),
            "test_coarse_gen": AircraftSubDataset(self.root, test_c_gen),
            "test_medium_gen": AircraftSubDataset(self.root, test_m_gen),
            "test_near_family": AircraftSubDataset(self.root, test_near_f),
            "test_near_variant": AircraftSubDataset(self.root, test_near_v),
            "test_far_ood": AircraftSubDataset(self.root, test_far)
        }

class AircraftSubDataset(Dataset):
    def __init__(self, root, data_list, transform=None):
        self.root = root
        self.data_list = data_list
        self.transform = transform
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        img_id, label = self.data_list[idx]
        image = Image.open(os.path.join(self.root, 'data', 'images', f'{img_id}.jpg')).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label