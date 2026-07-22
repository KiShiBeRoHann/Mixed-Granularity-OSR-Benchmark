import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MixedGranularityCIFAR100:
    def __init__(self, root='./data', download=True):
        # 1. 加载并合并 CIFAR-100 (50k Train + 10k Test = 60k Total)
        transform_base = transforms.Compose([transforms.ToTensor()])
        train_raw = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_base)
        test_raw = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_base)
        
        self.data = np.concatenate([train_raw.data, test_raw.data], axis=0)
        self.targets = np.array(train_raw.targets + test_raw.targets)
        
        self.class_to_idx = train_raw.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 默认用 seed=42 初始化分组结构
        pass

    def _define_groups(self, seed=42):
        """动态随机划分粗粒度、细粒度和 Far OOD 的分组结构"""
        import random
        import json
        import os
        split_dir = 'splits/cifar100'
        os.makedirs(split_dir, exist_ok=True)
        split_config_path = os.path.join(split_dir, f'seed_{seed}.json')


        if os.path.exists(split_config_path):
            with open(split_config_path, 'r', encoding='utf-8') as f:
                split_record = json.load(f)
            self.coarse_groups = split_record["coarse_groups"]
            self.fine_groups = split_record["fine_groups"]
            self.far_ood_structure = split_record["far_ood_structure"]
            
            if seed == 46:
                # 🚀 专属通道：只在 seed=46 时锁定你最初的硬编码顺序！
                self.id_classnames = [
                    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
                    "clock", "keyboard", "lamp",
                    "bed", "chair", "couch",
                    "bee", "beetle", "butterfly",
                    "bear", "leopard", "lion",
                    "bridge", "castle", "house"
                ]
                print(f"[Info] 成功读取原版硬编码划分 (仅限 Seed {seed}): {split_config_path}")
            else:
                # 🚀 通用通道：其他所有 seed 根据 JSON 动态还原顺序
                self.id_classnames = []
                
                # 1. 前 5 个粗粒度类：填入超类名
                for k in self.coarse_groups.keys():
                    self.id_classnames.append(k.replace('_', ' '))
                    
                # 2. 后 15 个细粒度类：填入具体的 seen 子类名
                for k in self.fine_groups.keys():
                    for sub in self.fine_groups[k]['seen']:
                        self.id_classnames.append(sub.replace('_', ' '))
                        
                print(f"[Info] 成功动态还原 JSON 划分 (Seed {seed}): {split_config_path}")
                
            return

        all_superclasses = {
            'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
            'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
            'furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
            'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
            'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'kangaroo', 'elephant'],
            'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people': ['baby', 'boy', 'girl', 'man', 'woman'],
            'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
        }

        # 保证洗牌结果可复现
        keys = sorted(list(all_superclasses.keys()))
        random.seed(seed)
        random.shuffle(keys)

        coarse_keys = keys[:5]
        fine_keys = keys[5:10]
        far_keys = keys[10:]

        self.coarse_groups = {}
        self.fine_groups = {}
        self.far_ood_structure = {}
        
        # 核心：用来按标签顺序（0-19）记录当前的真实类名列表
        self.id_classnames = []

        # 构建 Coarse 组 (Labels 0-4)
        for k in coarse_keys:
            subclasses = sorted(all_superclasses[k])
            random.shuffle(subclasses)
            self.coarse_groups[k] = {'seen': subclasses[:3], 'unseen': subclasses[3:]}
            # 粗粒度组的标签名是超类名
            self.id_classnames.append(k.replace('_', ' '))

        # 构建 Fine 组 (Labels 5-19)
        for k in fine_keys:
            subclasses = sorted(all_superclasses[k])
            random.shuffle(subclasses)
            self.fine_groups[k] = {'seen': subclasses[:3], 'ood': subclasses[3:]}
            # 细粒度组的标签名是具体子类名
            for sub in subclasses[:3]:
                self.id_classnames.append(sub.replace('_', ' '))

        # 构建 Far OOD 组 (Labels 20-29)
        for idx, k in enumerate(far_keys):
            self.far_ood_structure[20 + idx] = sorted(all_superclasses[k])


        split_record = {
            "seed": seed,
            "coarse_groups": self.coarse_groups,
            "fine_groups": self.fine_groups,
            "far_ood_structure": {k: all_superclasses[k] for k in far_keys}
        }
        
        split_dir = 'splits/cifar100'
        os.makedirs(split_dir, exist_ok=True)
        split_config_path = os.path.join(split_dir, f'seed_{seed}.json')
        
        # 写入网格化路径
        with open(split_config_path, 'w') as f:
            json.dump(split_record, f, indent=4)
            
        print(f"[Info] CIFAR-100 split saved to {split_config_path}")

        
    def get_datasets(self, far_ood_mode='coarse', include_unseen_in_train=False):
        """
        far_ood_mode: 
          - 'coarse': Far OOD 数据的标签将是 20-29
          - 'fine':   Far OOD 数据的标签将是原始 CIFAR fine_id
        include_unseen_in_train:
          - False: 计算 Unseen ACC (默认模式，用于测试模型的泛化能力)
          - True: 将 coarse_groups 中的 unseen 子类的前 60% 加入训练集，计算 Seen ACC (测算理论上限)
        """
        train_indices, train_labels = [], []
        test_id_indices, test_id_labels = [], []
        test_coarse_gen_indices, test_coarse_gen_labels = [], [] 
        test_near_ood_indices, test_near_ood_labels = [], []     
        test_far_ood_indices, test_far_ood_labels = [], []       
        
        SPLIT_POINT = 360 
        TEST_LEN = 600 - SPLIT_POINT # 240
        current_train_label = 0 
        
        # 固定随机种子，确保无论参数如何，40% 的测试集都是同一批图片
        np.random.seed(42)

        def split_class_data(class_name):
            original_idx = self.class_to_idx[class_name]
            all_indices = np.where(self.targets == original_idx)[0]
            # 这里简单取前后，因为原始 CIFAR 已经打乱过。
            return all_indices[:SPLIT_POINT], all_indices[SPLIT_POINT:]
        
        # ==========================================
        # 1. 处理粗粒度组 (Labels 0-4)
        # ==========================================
        for group_name, content in self.coarse_groups.items():
            assigned_label = current_train_label
            current_train_label += 1
            
            # A. Seen Subclasses (参与训练)
            for class_name in content['seen']:
                tr_idx, te_idx = split_class_data(class_name)
                train_indices.extend(tr_idx); train_labels.extend([assigned_label] * SPLIT_POINT)
                test_id_indices.extend(te_idx); test_id_labels.extend([assigned_label] * TEST_LEN)
            
            # B. Unseen Subclasses (泛化测试)
            for class_name in content['unseen']:
                tr_idx, te_idx = split_class_data(class_name)
                
                # 无论开关如何，后 40% 永远进粗粒度泛化测试集
                test_coarse_gen_indices.extend(te_idx)
                test_coarse_gen_labels.extend([assigned_label] * TEST_LEN)

                # === 核心控制：是否将未见子类加入训练集 ===
                if include_unseen_in_train:
                    train_indices.extend(tr_idx)
                    train_labels.extend([assigned_label] * SPLIT_POINT)

        # ==========================================
        # 2. 处理细粒度组 (Labels 5-19)
        # ==========================================
        for group_name, content in self.fine_groups.items():
            for class_name in content['seen']:
                assigned_label = current_train_label
                current_train_label += 1
                
                tr_idx, te_idx = split_class_data(class_name)
                train_indices.extend(tr_idx); train_labels.extend([assigned_label] * SPLIT_POINT)
                test_id_indices.extend(te_idx); test_id_labels.extend([assigned_label] * TEST_LEN)
            
            for class_name in content['ood']:
                _, te_idx = split_class_data(class_name)
                test_near_ood_indices.extend(te_idx); test_near_ood_labels.extend([-1] * TEST_LEN)
        
        # ==========================================
        # 3. 处理 Far OOD (Labels 20-29 or Fine ID)
        # ==========================================
        for eval_coarse_label, subclasses in self.far_ood_structure.items():
            for class_name in subclasses:
                if class_name not in self.class_to_idx: continue
                _, te_idx = split_class_data(class_name)
                test_far_ood_indices.extend(te_idx)
                if far_ood_mode == 'coarse':
                    test_far_ood_labels.extend([eval_coarse_label] * TEST_LEN)
                else:
                    original_idx = self.class_to_idx[class_name]
                    test_far_ood_labels.extend([original_idx] * TEST_LEN)

        # ==========================================
        # 4. 构建 PyTorch Dataset
        # ==========================================
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        def create_subset(indices, labels, is_train=False):
            class CustomSubset(Dataset):
                def __init__(self, full_data, indices, targets, transform):
                    self.data = full_data[indices]
                    self.targets = targets
                    self.transform = transform
                def __len__(self): return len(self.data)
                def __getitem__(self, idx):
                    img = self.data[idx]
                    target = self.targets[idx]
                    img = torchvision.transforms.ToPILImage()(img)
                    if self.transform:
                        img = self.transform(img)
                    return img, target
            return CustomSubset(self.data, indices, labels, train_transform if is_train else test_transform)

        return {
            "train": create_subset(train_indices, train_labels, is_train=True),
            "test_id": create_subset(test_id_indices, test_id_labels),
            "test_coarse_gen": create_subset(test_coarse_gen_indices, test_coarse_gen_labels), # 这是你需要用来测 Unseen/Seen ACC 的数据集
            "test_near_ood": create_subset(test_near_ood_indices, test_near_ood_labels),
            "test_far_ood": create_subset(test_far_ood_indices, test_far_ood_labels)
        }

if __name__ == "__main__":
    import os
    # 动态获取当前文件同级目录下的 datasets，防止路径写死报错
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, 'datasets/CIFAR100')
    
    processor = MixedGranularityCIFAR100(root=data_root, download=True)
    
    # === 测试 1：正常 OOD 检测模式 (Unseen ACC) ===
    datasets_unseen = processor.get_datasets(far_ood_mode='coarse', include_unseen_in_train=False)
    print("=" * 50)
    print("MODE 1: Normal OOD Mode (Testing Unseen ACC)")
    print("=" * 50)
    print(f"[Train] Size: {len(datasets_unseen['train'])} (Expected: 10800 -> 30个类 * 360)")
    print(f"[Test Coarse Gen] Size: {len(datasets_unseen['test_coarse_gen'])} (Expected: 2400 -> 10个类 * 240)")
    print(f"--> Evaluate 'test_coarse_gen' to get UNSEEN ACC.\n")
    
    # === 测试 2：Oracle 模式 (Seen ACC) ===
    datasets_seen = processor.get_datasets(far_ood_mode='coarse', include_unseen_in_train=True)
    print("=" * 50)
    print("MODE 2: Oracle Mode (Testing Seen ACC)")
    print("=" * 50)
    print(f"[Train] Size: {len(datasets_seen['train'])} (Expected: 14400 -> 增加了10个类 * 360)")
    print(f"[Test Coarse Gen] Size: {len(datasets_seen['test_coarse_gen'])} (Expected: 2400 -> 同上，数据完全一致)")
    print(f"--> Evaluate 'test_coarse_gen' to get SEEN ACC.\n")