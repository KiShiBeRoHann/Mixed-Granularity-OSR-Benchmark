import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image

# 锚定脚本所在目录，保证在不同机器上路径都正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TreeSubDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

class TreeBasedMixedGranularity:
    def __init__(self, dataset_name, root, seed=42, seen_ratio=0.5, num_layers=2):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.seed = seed
        self.seen_ratio = seen_ratio
        self.num_layers = num_layers
        
        self.split_dir = f'splits/{self.dataset_name}/L{num_layers}'
        self.split_config_path = os.path.join(self.split_dir, f'seed_{self.seed}.json')
        
        if not os.path.exists(self.split_config_path):
            raise FileNotFoundError(f"找不到划分文件 {self.split_config_path}，请先运行 build.py")
            
        with open(self.split_config_path, 'r') as f:
            config = json.load(f)
            self.split_config = config["split_config"]
            self.total_id_classes = config["total_id_classes"]

        self._scan_image_paths()

    def _scan_image_paths(self):
        """智能扫描: 完美适配 ImageNet (物理文件夹) 和 iNaturalist 2021 (物理文件夹)"""
        self.image_records = {'train': [], 'test': []}
        
        # 模式 A: ImageNet (train/类名/图片)
        if self.dataset_name == 'imagenet':
            for phase in ['train', 'val']:
                phase_dir = os.path.join(self.root, phase)
                if not os.path.exists(phase_dir): continue
                for class_name in os.listdir(phase_dir):
                    class_dir = os.path.join(phase_dir, class_name)
                    if not os.path.isdir(class_dir): continue
                    for img_name in os.listdir(class_dir):
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                        img_path = os.path.join(class_dir, img_name)
                        record_phase = 'test' if phase == 'val' else 'train'
                        self.image_records[record_phase].append((img_path, class_name))
                        
        # 🌟 模式 B: iNaturalist 2021 Mini (train_mini/文件夹名/图片)
        elif self.dataset_name == 'inaturalist':
            # 🚀 核心修复：强行锁定 2021 Mini 的真实物理路径！
            real_inat_root = os.path.join(BASE_DIR, 'iNaturalist2021_Mini')
            
            # 读取官方 JSON，建立：文件夹长名 -> 种(物种短名) 的绝对安全映射
            mapping_file = os.path.join(real_inat_root, 'train_mini.json')
            dir_to_name = {}
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for cat in data['categories']:
                        # 将 JSON 里的名字空格替换为下划线，以匹配树里的格式
                        dir_to_name[cat['image_dir_name']] = cat['name'].replace(' ', '_')
            
            # 2021 版本的两个核心物理目录
            for phase, folder_name in [('train', 'train_mini'), ('test', 'val')]:
                phase_dir = os.path.join(real_inat_root, folder_name)
                
                # 🚨 加强防线：如果真的没找到，直接报错提醒，不静默跳过
                if not os.path.exists(phase_dir): 
                    print(f"⚠️ [致命错误] 找不到物理路径: {phase_dir}！请确认解压是否完成！")
                    continue
                
                print(f"✅ 正在扫描 {folder_name} 物理文件夹...")
                for class_dir_name in os.listdir(phase_dir):
                    class_dir = os.path.join(phase_dir, class_dir_name)
                    if not os.path.isdir(class_dir): continue
                    
                    # 优先通过 JSON 字典映射出物种真名
                    if class_dir_name in dir_to_name:
                        class_name = dir_to_name[class_dir_name]
                    else:
                        # 极端降级：截取文件夹名最后两截作为种名
                        class_name = "_".join(class_dir_name.split('_')[-2:])
                        
                    for img_name in os.listdir(class_dir):
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                        img_path = os.path.join(class_dir, img_name)
                        self.image_records[phase].append((img_path, class_name))

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

        for phase in ['train', 'test']:
            for img_path, v_name in self.image_records[phase]:
                if phase == 'train':
                    if v_name in train_map: train_data.append((img_path, train_map[v_name]))
                    elif include_unseen:
                        if v_name in coarse_gen_map: train_data.append((img_path, coarse_gen_map[v_name]))
                        elif v_name in medium_gen_map: train_data.append((img_path, medium_gen_map[v_name]))
                elif phase == 'test':
                    if v_name in train_map: test_id_data.append((img_path, train_map[v_name]))
                    elif v_name in coarse_gen_map: test_c_gen.append((img_path, coarse_gen_map[v_name]))
                    elif v_name in medium_gen_map: test_m_gen.append((img_path, medium_gen_map[v_name]))
                    elif v_name in near_fam_set: test_near_f.append((img_path, v_name))
                    elif v_name in near_var_set: test_near_v.append((img_path, v_name))
                    elif v_name in far_ood_set: test_far.append((img_path, v_name))

        return {
            "train": TreeSubDataset(train_data),
            "test_id": TreeSubDataset(test_id_data),
            "test_coarse_gen": TreeSubDataset(test_c_gen),
            "test_medium_gen": TreeSubDataset(test_m_gen),
            "test_near_family": TreeSubDataset(test_near_f),
            "test_near_variant": TreeSubDataset(test_near_v),
            "test_far_ood": TreeSubDataset(test_far)
        }