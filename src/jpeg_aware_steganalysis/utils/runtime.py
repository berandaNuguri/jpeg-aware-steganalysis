import os
import random
import shutil
import numpy as np
import pandas as pd
import timm
import yaml 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import jpeg_aware_steganalysis.utils.data_processing.image_processing.custom_transforms as ctransforms
from importlib import import_module
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from jpeg_aware_steganalysis.models.srnet.srnet_color import SRNet_Color
from jpeg_aware_steganalysis.models.zhunet.zhunet import ZhuNet
from jpeg_aware_steganalysis.models.ucnet.ucnet import UCNet
from jpeg_aware_steganalysis.models.penet.penet_fixed import PENet_Fixed
from jpeg_aware_steganalysis.models.penet.penet_arbitrary import PENet_Arbitrary
from jpeg_aware_steganalysis.models.sf3net.sf3net import SF3Net
from jpeg_aware_steganalysis.models.iformer.inception_transformer import iformer_small, iformer_base

from jpeg_aware_steganalysis.utils.data_processing.datasets import DefaultDataset, PairedPatchDataset, PairedYCbCrPatchDataset, PairedDCTPatchDataset, TwoStreamDataset, SF3NetDataset, VisualizeDCTDataset


def estimate_qf(quant_table):
    std_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    best_quality, best_error = None, np.inf
    for quality in range(1, 101):
        scale = 5000/quality if quality<50 else 200-2*quality
        comp = np.floor((std_table*scale+50)/100).clip(1,255)
        err = np.mean(np.abs(comp - quant_table))
        if err < best_error:
            best_error, best_quality = err, quality
    return best_quality

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError
    
def to_float32(image, **kwargs):
    return image.astype(np.float32)

def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)

def build_transforms(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tf_dict = {}

    for phase in ['train', 'test']:
        key = f'{phase}_transforms'
        tf_configs = config.get(key)

        if not tf_configs:
            tf_dict[phase] = None
            continue

        spatial_cfgs = None
        dct_cfgs = None

        if isinstance(tf_configs, dict) and ('spatial' in tf_configs or 'dct' in tf_configs):
            spatial_cfgs = tf_configs.get('spatial') or []
            dct_cfgs = tf_configs.get('dct') or []

        elif isinstance(tf_configs, list) and all(isinstance(x, dict) for x in tf_configs):
            for item in tf_configs:
                if 'spatial' in item:
                    spatial_cfgs = item.get('spatial') or []
                if 'dct' in item:
                    dct_cfgs = item.get('dct') or []

        if spatial_cfgs is None and dct_cfgs is None:
            if not isinstance(tf_configs, list):
                raise ValueError(
                    f"{key} must be a list of transform configs or a dict with 'spatial'/'dct'. "
                    f"Got type={type(tf_configs)}"
                )

            transform_list = []
            for tf_config in tf_configs:
                params = tf_config.copy()
                tf_name = params.pop('name')

                if hasattr(transforms, tf_name):
                    transform_class = getattr(transforms, tf_name)
                    transform_list.append(transform_class(**params))
                elif hasattr(ctransforms, tf_name):
                    transform_class = getattr(ctransforms, tf_name)
                    transform_list.append(transform_class(**params))
                else:
                    raise ValueError(
                        f"Transform '{tf_name}' not found in torchvision.transforms or custom transforms"
                    )

            tf_dict[phase] = transforms.Compose(transform_list) if transform_list else None

        else:
            if spatial_cfgs:
                spatial_list = []
                for cfg in spatial_cfgs:
                    params = cfg.copy()
                    tf_name = params.pop('name')

                    if hasattr(transforms, tf_name):
                        transform_class = getattr(transforms, tf_name)
                        spatial_list.append(transform_class(**params))
                    elif hasattr(ctransforms, tf_name):
                        transform_class = getattr(ctransforms, tf_name)
                        spatial_list.append(transform_class(**params))
                    else:
                        raise ValueError(
                            f"Spatial transform '{tf_name}' not found in torchvision.transforms or custom transforms"
                        )
                spatial_tf = transforms.Compose(spatial_list) if spatial_list else None
            else:
                spatial_tf = None

            if dct_cfgs:
                dct_list = []
                for cfg in dct_cfgs:
                    params = cfg.copy()
                    tf_name = params.pop('name')

                    if hasattr(ctransforms, tf_name):
                        transform_class = getattr(ctransforms, tf_name)
                        dct_list.append(transform_class(**params))
                    elif hasattr(transforms, tf_name):
                        transform_class = getattr(transforms, tf_name)
                        dct_list.append(transform_class(**params))
                    else:
                        raise ValueError(
                            f"DCT transform '{tf_name}' not found in custom transforms or torchvision.transforms"
                        )
                dct_tf = transforms.Compose(dct_list) if dct_list else None
            else:
                dct_tf = None

            tf_dict[phase] = {
                'spatial': spatial_tf,
                'dct': dct_tf,
            }

    dataset_class_map = {
        'DefaultDataset': DefaultDataset,
        'PairedPatchDataset': PairedPatchDataset,
        'PairedYCbCrPatchDataset': PairedYCbCrPatchDataset,
        'PairedDCTPatchDataset': PairedDCTPatchDataset,
        'TwoStreamDataset': TwoStreamDataset,
        'SF3NetDataset': SF3NetDataset,
        'VisualizeDCTDataset': VisualizeDCTDataset
    }
    
    ds_classes = {}
    if 'dataset_class' in config:
        for phase in ['train', 'validation', 'test']:
            class_name = config['dataset_class'].get(phase)
            if class_name in dataset_class_map:
                ds_classes[phase] = dataset_class_map[class_name]
            else:
                raise ValueError(f"Dataset class '{class_name}' not found or defined.")
    else:
        ds_classes = {
            'train': PairedPatchDataset,
            'validation': PairedPatchDataset,
            'test': PairedPatchDataset
        }

    return tf_dict, ds_classes

def debug_collate(batch):
    try:
        return default_collate(batch)
    except Exception as e:
        print("[Collate Error Detected!]")
        for i, sample in enumerate(batch):
            data, qt, label, meta = sample
            print(f"  Sample {i}:")
            print(f"    data shape: {[x.shape for x in data] if hasattr(data, '__iter__') else data.shape}")
            print(f"    qt shape: {qt.shape if hasattr(qt, 'shape') else type(qt)}")
            print(f"    label: {label}")
            print(f"    meta: {meta}")
        raise
    
def load_dataset(args):

    if not args.transform_config or not os.path.exists(args.transform_config):
        raise FileNotFoundError(
            f"Config file not found at {args.transform_config}. Please provide a valid path via --config."
        )
    
    tf, ds_classes = build_transforms(args.transform_config)

    TrainDS = ds_classes['train']
    ValidDS = ds_classes['validation']
    TestDS  = ds_classes['test']

    csv_path_parts = [args.csv_root, args.data]
    
    train_csv = os.path.join(*csv_path_parts, f'{args.stego_method}_train.csv')
    valid_csv = os.path.join(*csv_path_parts, f'{args.stego_method}_valid.csv')
    test_csv  = os.path.join(*csv_path_parts, f'{args.stego_method}_test.csv')

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    test_df  = pd.read_csv(test_csv)

    train_ds = TrainDS(train_df, args.dataset_root, transform=tf['train'], args=args)
    valid_ds = ValidDS(valid_df, args.dataset_root, transform=tf['test'], args=args)
    test_ds  = TestDS(test_df,  args.dataset_root, transform=tf['test'], args=args)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers_per_loader,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs, collate_fn=debug_collate)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **loader_kwargs, collate_fn=debug_collate)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **loader_kwargs, collate_fn=debug_collate)

    return train_loader, valid_loader, test_loader

def load_model(args, device):
    def initWeights(module):
        if type(module) == nn.Conv2d:
            if module.weight is not None and module.weight.requires_grad:
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

        elif type(module) == nn.Linear:
            if module.weight is not None and module.weight.requires_grad:
                nn.init.normal_(module.weight.data, mean=0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

        elif type(module) == nn.GroupNorm:
            if module.weight is not None:
                nn.init.constant_(module.weight.data, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    if 'SRNet_Color' in args.model_name:
        print('Load SRNet_Color using following arguments')

        model = SRNet_Color(num_classes=2)
        model.apply(initWeights)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'ZhuNet' in args.model_name:
        print('Load ZhuNet using following arguments')

        model = ZhuNet()
        model.apply(initWeights)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'UCNet' in args.model_name:
        print('Load UCNet using following arguments')

        model = UCNet(num_classes=2)
        model.apply(initWeights)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'PENet' in args.model_name:
        print(f'Load {args.model_name} using following arguments')
        tokens = args.model_name.split('_')
        if 'Fixed' in tokens:
            typ = 'Fixed'
            ver = tokens[-1] if tokens[-1] != 'Fixed' else 'Default'
        elif 'Arbitrary' in tokens:
            typ = 'Arbitrary'
            ver = tokens[-1] if tokens[-1] != 'Arbitrary' else 'Default'
        else:
            raise ValueError("Unknown PENet model name: {}".format(args.model_name))
        
        if typ == 'Fixed' and ver == 'Default':
            model = PENet_Fixed(dropout=args.dropout)
        elif typ == 'Arbitrary' and ver == 'Default':
            model = PENet_Arbitrary(dropout=args.dropout)
        
        model.apply(initWeights)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    if 'SF3Net' in args.model_name:
        print('Load SF3Net using following arguments')

        model = SF3Net(in_channels=1)
        model.apply(initWeights)
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'iformer' in args.model_name:
        model_size=args.model_name.split('_')[-1]
        print(f'Load iFormer-{model_size} using following arguments')
        if model_size == 'small':
            model = iformer_small(pretrained=True)
        elif model_size == 'base':
            model = iformer_base(pretrained=True)
        
        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'DCTFormer' in args.model_name:
        print(f'Load {args.model_name} using following arguments')

        parts = args.model_name.split('_')
        if len(parts) != 3 or not parts[1].lower().startswith('v'):
            raise ValueError(f"Expected 'DCTFormer_v{{num}}_{{small|base|large}}', got: {args.model_name}")

        ver = parts[1].lower().replace('v', '')
        size = parts[2].lower()
        module_name = f"dctformer_v{ver}"
        full_module = f"jpeg_aware_steganalysis.models.dctformer.{module_name}"

        mod = import_module(full_module)
        builder = getattr(mod, f'build_dctformer_v{ver}')

        model = builder(version=size, num_classes=2)

        if getattr(args, "init_weights", False):
            init_fn = getattr(mod, "initWeights", None)
            if init_fn is not None:
                model.apply(init_fn)

        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'TempNet' in args.model_name:
        print(f'Load {args.model_name} using following arguments')
        module_name = "tempnet" if args.model_name == "TempNet" else args.model_name.lower()
        full_module = f"jpeg_aware_steganalysis.models.tempnet.{module_name}"

        mod = import_module(full_module)
        ModelClass = getattr(mod, args.model_name)
        model = ModelClass()

        if getattr(args, "init_weights", False):
            init_fn = getattr(mod, "initWeights", None)
            if init_fn is not None:
                model.apply(init_fn)

        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    elif 'timm' in args.model_name:
        print(f'Load {args.model_name.split("-")[-1]} using following arguments')
        
        model = timm.create_model(args.model_name.split('-')[-1], pretrained=True, num_classes=2)

        # if 'efficientnet' in args.model_name.lower():
        #     model.conv_stem.stride = (1, 1) 

        if args.pretrained_path is not None:
            print('Loading pretrained model from:', args.pretrained_path)
            state_dict = torch.load(args.pretrained_path)['state_dict']
            model.load_state_dict(state_dict)

    return model.to(device) 

def load_loss(args):
    criterion = {}
    for loss in args.losses:
        if loss == 'CrossEntropy':
            criterion['CE'] = nn.CrossEntropyLoss()
    return criterion
