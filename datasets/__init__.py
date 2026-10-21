import os
import json
from .split_cifar import MixedGranularityCIFAR100
from .split_fgvc import FGVCAircraftMixedGranularity
from .split_imin import TreeBasedMixedGranularity
from torch.utils.data import DataLoader

def get_mixed_granularity_loaders(dataset_name, seed, batch_size=128, far_ood_mode='fine', include_unseen=False, num_workers=4, preprocess=None, num_layers=2, eval_view='V4'):
    """
    万能数据流工厂入口。追加 num_layers 支持控制两层/三层划分。

    eval_view: 评估靶场视图（V1/V2/V3/V4，默认 V4），用于在同一套 L3 ID / 同一份权重下
               切换 OOD 靶场的构成。支持范围**按数据集硬校验**：
                 * aircraft + L3 : V1 / V2 / V3 / V4
                 * imagenet + L3 : V1 / V2 / V3 / V4
                   ⚠️ imagenet 的 `test_medium_gen` 只有 50~500 张（seed 45 仅 50 张），
                   而 V1/V2 的 gen 集恰好只由 medium_gen 构成 → `Gen Acc`/`CV-RR` 的分母极小，
                   看这两列时必须同时看 metrics.csv 的 `n_gen_medium`，不要单看绝对值。
                 * 其它（含 inaturalist）：仅默认 V4（iNat 的 L3 medium_gen 为空）
               见 datasets/split_fgvc.py / split_imin.py::get_datasets 的说明。
    """
    # =========================================================
    # num_layers 归一化：'2'/'3' 一律转 int（各 splitter 内部按 int 比较）；
    # '2d'（aircraft 专用的"从 L3 派生的两层划分"标记）保持字符串，
    # 以便产物路径自动落到 models/aircraft/L2d/ 与 splits/aircraft/L2d/。
    # =========================================================
    # 允许批量评估脚本通过环境变量关闭 DataLoader worker，避免多进程评估偶发死锁
    num_workers = int(os.environ.get('MG_OSR_NUM_WORKERS', num_workers))

    try:
        num_layers = int(num_layers)
    except (TypeError, ValueError):
        pass

    if str(num_layers) in ('2d', '2df', '3fc') and dataset_name.lower() != 'aircraft':
        raise ValueError(
            f"num_layers='{num_layers}'（从 L3 派生的两层划分）仅对 aircraft 有效，当前 dataset={dataset_name}。"
        )

    # =========================================================
    # eval_view 硬校验：视图只在明确支持的数据集 + L3 下有效
    # =========================================================
    # 不校验会出现两种静默错误（都不会报错，但会污染视图对照表）：
    #   1) num_layers=2 传 V1/V2：get_datasets 里 use_coarse_gen=False 会把 L2 的
    #      Family 级 coarse_gen 整段清空 → Gen Acc/CV-RR 变 N/A、Global Acc 退化成纯 ID 准确率。
    #   2) 数据集分支不接该参数（如 inaturalist）：装载结果与 V4 完全相同，
    #      却会被 run.sh 写进 view_V{n} 日志目录，看起来"做过视图对照"。
    if eval_view not in ('V1', 'V2', 'V3', 'V4'):
        raise ValueError(f"未知 eval_view: {eval_view}（应取 V1/V2/V3/V4）")

    _ds_lower = dataset_name.lower()
    if num_layers != 3:
        _supported = ('V4',)
    elif _ds_lower in ('aircraft', 'imagenet'):
        _supported = ('V1', 'V2', 'V3', 'V4')
    else:
        _supported = ('V4',)

    if eval_view not in _supported:
        raise ValueError(
            f"--eval_view {eval_view} 在 dataset={dataset_name} / num_layers={num_layers} 下不支持；"
            f"支持 {_supported}。不支持时请使用默认视图 V4（其 OOD 靶场与历史结果口径一致）。"
        )

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
        datasets = processor.get_datasets(include_unseen=include_unseen, eval_view=eval_view)
        
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
        datasets = processor.get_datasets(include_unseen=include_unseen, eval_view=eval_view)
        
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