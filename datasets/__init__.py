import os
import json
from .split_cifar import MixedGranularityCIFAR100
from .split_fgvc import FGVCAircraftMixedGranularity
from .split_imin import TreeBasedMixedGranularity
from torch.utils.data import DataLoader

def get_mixed_granularity_loaders(dataset_name, seed, batch_size=128, far_ood_mode='fine', include_unseen=False, num_workers=4, preprocess=None, num_layers=2):
    """
    万能数据流工厂入口。追加 num_layers 支持控制两层/三层划分。
    """
    if dataset_name.lower() == 'cifar100':
        # CIFAR-100 固化为两层结构 (Superclass -> Class)
        processor = MixedGranularityCIFAR100(root='./datasets', download=False)
        processor._define_groups(seed=seed)
        datasets = processor.get_datasets(far_ood_mode=far_ood_mode, include_unseen_in_train=include_unseen)
        classnames = processor.id_classnames
        num_classes = len(classnames)

        if preprocess is not None:
            for key in datasets:
                if datasets[key] is not None: datasets[key].transform = preprocess

        loaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test_id': DataLoader(datasets['test_id'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_coarse_gen': DataLoader(datasets['test_coarse_gen'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_near_ood': DataLoader(datasets['test_near_ood'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_far_ood': DataLoader(datasets['test_far_ood'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }

    elif dataset_name.lower() == 'aircraft':
        # 【核心修改】：把外层的 num_layers 完美注入底层处理器
        processor = FGVCAircraftMixedGranularity(root='./datasets/fgvc-aircraft-2013b', seed=seed, seen_ratio=0.5, num_layers=num_layers)
        datasets = processor.get_datasets(include_unseen=include_unseen)
        
        config = processor.split_config
        classnames_dict = {}
        for item in config.get('coarse_id', []): classnames_dict[item['label']] = item['head_name']
        for item in config.get('medium_id', []): classnames_dict[item['label']] = item['head_name']
        for item in config.get('fine_id', []): classnames_dict[item['label']] = item['head_name']
            
        classnames = [classnames_dict[i] for i in range(len(classnames_dict))]
        num_classes = len(classnames)

        if preprocess is not None:
            for key in datasets:
                if datasets[key] is not None: datasets[key].transform = preprocess

        loaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test_id': DataLoader(datasets['test_id'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_coarse_gen': DataLoader(datasets['test_coarse_gen'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_medium_gen': DataLoader(datasets['test_medium_gen'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_near_family': DataLoader(datasets['test_near_family'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_near_variant': DataLoader(datasets['test_near_variant'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_far_ood': DataLoader(datasets['test_far_ood'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
    
    elif dataset_name.lower() in ['imagenet', 'inaturalist']:
        # 对应真实文件夹路径
        root_path = f'./datasets/{dataset_name}'
        if dataset_name.lower() == 'inaturalist':
            root_path = f'./datasets/iNaturalist' # 照顾大小写
            
        processor = TreeBasedMixedGranularity(
            dataset_name=dataset_name, 
            root=root_path, 
            seed=seed, 
            seen_ratio=0.5, 
            num_layers=num_layers
        )
        datasets = processor.get_datasets(include_unseen=include_unseen)
        
        config = processor.split_config
        classnames_dict = {}
        for item in config.get('coarse_id', []): classnames_dict[item['label']] = item['head_name']
        for item in config.get('medium_id', []): classnames_dict[item['label']] = item['head_name']
        for item in config.get('fine_id', []): classnames_dict[item['label']] = item['head_name']
            
        classnames = [classnames_dict[i] for i in range(len(classnames_dict))]
        num_classes = len(classnames)

        if preprocess is not None:
            for key in datasets:
                if datasets[key] is not None: datasets[key].transform = preprocess

        # 完美输出7大 Loader，直接喂给现有的评估代码
        loaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test_id': DataLoader(datasets['test_id'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_coarse_gen': DataLoader(datasets['test_coarse_gen'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_medium_gen': DataLoader(datasets['test_medium_gen'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_near_family': DataLoader(datasets['test_near_family'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_near_variant': DataLoader(datasets['test_near_variant'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test_far_ood': DataLoader(datasets['test_far_ood'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return loaders, num_classes, classnames