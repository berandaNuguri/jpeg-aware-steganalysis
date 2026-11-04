from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="libpng warning: eXIf")
warnings.filterwarnings("ignore", message="libpng warning: sBIT")

import argparse
import random
import timm
import wandb
import time
import shutil
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from jpeg_aware_steganalysis.utils.runtime import time_to_str, load_dataset, load_model, load_loss



def save_checkpoint(state, gpus, is_best=False,
                    model_path=f'./ckpt/',
                    model_name=f'checkpoint.pth.tar'):
    new_state_dict = OrderedDict()
    for k, v in state['state_dict'].items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    state['state_dict'] = new_state_dict

    os.makedirs(model_path, exist_ok=True)
    torch.save(state, f'{model_path}/{model_name}')
    if is_best:
        shutil.copyfile(f'{model_path}/{model_name}', f'{model_path}/model_best.pth.tar')


def train(args, train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for (imgs, dct), labels, meta in train_loader:
        B, Pair, PatchPair, C, H, W = imgs.shape
        Bi, Pi, Ppi, Ci, Hi, Wi = dct.shape

        assert (Bi, Pi, Ppi, Ci, Hi, Wi) == (B, Pair, PatchPair, C, H, W)

        N = B * Pair * PatchPair

        imgs   = imgs.view(N, C, H, W).to(device)
        dct    = dct.view(N, C, H, W).to(device)

        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        assert labels.shape[:2] == (B, Pair), f"labels shape {tuple(labels.shape)} must start with (B, Pair)"
        labels = labels.reshape(B * Pair).to(device).long()

        # ---- forward & loss ----
        optimizer.zero_grad()
        logits = model(imgs, dct)
        outputs = logits.view(B * Pair, PatchPair, -1)

        if getattr(args, "aggregate", "mean") == "mean":
            outputs = outputs.mean(dim=1)
        else:
            outputs, _ = outputs.max(dim=1)

        loss = criterion["CE"](outputs, labels)
        loss.backward()
        optimizer.step()

        # ---- 통계 ----
        bs = labels.size(0)  # = B*Pair
        total_loss += loss.item() * bs
        total_samples += bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(1, total_samples)
    accuracy = 100.0 * correct / max(1, total_samples)
    return avg_loss, accuracy

@torch.no_grad()
def validate(args, valid_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for (imgs, dct), labels, meta in valid_loader:
        B, Pair, PatchPair, C, H, W = imgs.shape
        Bi, Pi, Ppi, Ci, Hi, Wi = dct.shape

        assert (Bi, Pi, Ppi, Ci, Hi, Wi) == (B, Pair, PatchPair, C, H, W)

        N = B * Pair * PatchPair

        imgs   = imgs.view(N, C, H, W).to(device)
        dct    = dct.view(N, C, H, W).to(device)
        
        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        assert labels.shape[:2] == (B, Pair), f"labels shape {tuple(labels.shape)} must start with (B, Pair)"
        labels = labels.reshape(B * Pair).to(device).long()

        # ---- forward & loss ----
        logits = model(imgs, dct)
        outputs = logits.view(B * Pair, PatchPair, -1)

        if getattr(args, "aggregate", "mean") == "mean":
            outputs = outputs.mean(dim=1)
        else:
            outputs, _ = outputs.max(dim=1)

        loss = criterion["CE"](outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(1, total_samples)
    accuracy = 100.0 * correct / max(1, total_samples)
    return avg_loss, accuracy


@torch.no_grad()
def test(args, test_loader, model, criterion, device):
    model.eval()
    total_samples = 0
    correct = 0

    for (imgs, dct), labels, meta in test_loader:
        B, Pair, PatchPair, C, H, W = imgs.shape
        Bi, Pi, Ppi, Ci, Hi, Wi = dct.shape

        assert (Bi, Pi, Ppi, Ci, Hi, Wi) == (B, Pair, PatchPair, C, H, W)

        N = B * Pair * PatchPair

        imgs   = imgs.view(N, C, H, W).to(device)
        dct    = dct.view(N, C, H, W).to(device)

        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        assert labels.shape[:2] == (B, Pair), f"labels shape {tuple(labels.shape)} must start with (B, Pair)"
        labels = labels.reshape(B * Pair).to(device).long()

        # ---- forward & loss ----
        logits = model(imgs, dct)
        outputs = logits.view(B * Pair, PatchPair, -1)

        if getattr(args, "aggregate", "mean") == "mean":
            outputs = outputs.mean(dim=1)  # (B*Pair, num_classes)
        else:
            outputs, _ = outputs.max(dim=1)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)  # = B*Pair

    accuracy = 100.0 * correct / max(1, total_samples)
    return accuracy

def wandb_setting(args):
    wandb.init(project=f'{args.wandb_project}', entity=os.environ.get('WANDB_ENTITY') or None)
    code_artifact = wandb.Artifact('Code', type='code', description=args.run_name)
    code_artifact.add_file('./src/jpeg_aware_steganalysis/engine/train.py')
    code_artifact.add_dir('./src/jpeg_aware_steganalysis/utils/', name='utils/')
    code_artifact.add_dir('./src/jpeg_aware_steganalysis/utils/data_processing/', name='data/')
    code_artifact.add_dir('./src/jpeg_aware_steganalysis/engine/', name='engine/')
    code_artifact.add_dir('./src/jpeg_aware_steganalysis/models/', name='models/')
    code_artifact.add_file(args.transform_config, name=args.transform_config)
    wandb.log_artifact(code_artifact)
    for arg in vars(args):
        wandb.config[arg] = getattr(args, arg)
    wandb.run.name = args.run_name


def main(args):
    overall_start_time = time.time()

    if args.run_name == 'auto':
        args.run_name = args.model_name

        if args.pretrained_path:
            args.run_name += "_finetune"

        data_tag = args.data.rstrip("/").replace("/", "_")
        base = f"{data_tag}_{args.stego_method}_{args.aggregate}_{args.patch_size}{args.roi_method}_w_{args.epochs}_{args.batch_size}_{args.optimizer}_{args.lr}_{args.weight_decay}"
        args.run_name += f"_{base}"

        if args.suffix:
            args.run_name += f"_{args.suffix}"

    if args.use_wandb:
        wandb_setting(args)

    result_df = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'train_acc',
        'valid_loss', 'valid_acc',
        'test_acc'
    ])

    model = load_model(args, device)

    if ngpus > 1:
        print(f"Using {ngpus} GPUs for training (logical ids: {list(range(ngpus))})")
        model = nn.DataParallel(model, device_ids=list(range(ngpus))).to(device)
    else:
        print("Using single GPU")
        model = model.to(device)
        
    criterion = load_loss(args)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'MultiStep':
        if args.epochs == 100:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 110, 160], gamma=0.1)
    else:
        scheduler = None

    train_loader, valid_loader, test_loader = load_dataset(args)

    start_time = timer()
    print(f'=================================== Training Info ===================================')
    print(f'Arguments: {args}\n')
    try:
        with open(args.transform_config, 'r', encoding='utf-8') as f:
            config_content = f.read()
            print(f'{config_content}\n')
    except FileNotFoundError:
        print(f"ERROR: Config file not found at '{args.transform_config}'")
    except Exception as e:
        print(f"ERROR: Could not read file '{args.transform_config}'. Reason: {e}")
    print(f'Train Data: {len(train_loader.dataset)*2}')
    print(f'Validation Data: {len(valid_loader.dataset)*2}')
    print(f'Test Data: {len(test_loader.dataset)*2}')
    print(f'=================================== Training Start ===================================')

    best_valid_acc = 0.0
    best_epoch = 0
    best_train_loss = None
    best_train_acc = None
    best_valid_loss = None

    for epoch in range(1, args.epochs + 1):
        if scheduler is not None:
            scheduler.step()

        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, device)
        print(
            f"Epoch [{epoch}/{args.epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f} - "
            f"{time_to_str(timer() - start_time, mode='min')}"
        )
        if args.use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
            }, step=epoch)

        valid_loss, valid_acc = validate(args, valid_loader, model, criterion, device)
        print(
            f"Epoch [{epoch}/{args.epochs}], "
            f"Valid Loss: {valid_loss:.4f}, "
            f"Valid Acc: {valid_acc:.2f} - "
            f"{time_to_str(timer() - start_time, mode='min')}"
        )
        if args.use_wandb:
            wandb.log({
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
            }, step=epoch)

        result_row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'valid_loss': round(valid_loss, 4),
            'valid_acc': round(valid_acc, 2),
            'test_acc': None,
        }
        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)

        is_best = valid_acc >= best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_valid_loss = valid_loss

            if args.use_wandb:
                wandb.log({'best_valid_acc': valid_acc}, step=epoch)

            if args.save_model:
                save_checkpoint(
                    state={
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_valid_acc': best_valid_acc,
                    },
                    gpus=args.gpus,
                    is_best=is_best,
                    model_path=f'{args.ckpt_root}/{args.run_name}/',
                    model_name=f'ep{epoch}.pth.tar'
                )

    print(f'=================================== Training End ===================================')
    print(
        f'Best Valid[{best_epoch}Epoch] - '
        f'Valid Acc: {best_valid_acc:.2f} - '
        f'{time_to_str(timer() - start_time, mode="min")}'
    )

    if args.save_model:
        best_model_path = os.path.join(f'{args.ckpt_root}/{args.run_name}/', f'ep{best_epoch}.pth.tar')
        print(f"Loading best model from {best_model_path} for testing.")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])

        print('=================================== Test Start ===================================')
        test_acc = test(args, test_loader, model, criterion, device)
        print(
            f"Test Acc: {test_acc:.2f} - "
            f"{time_to_str(timer() - start_time, mode='min')}"
        )
        if args.use_wandb:
            wandb.log({'test_acc': test_acc})

        final_row = {
            'epoch': f'Best@{best_epoch}',
            'train_loss': round(best_train_loss, 4),
            'train_acc': round(best_train_acc, 2),
            'valid_loss': round(best_valid_loss, 4),
            'valid_acc': round(best_valid_acc, 2),
            'test_acc': round(test_acc, 2),
        }
        result_df = pd.concat([result_df, pd.DataFrame([final_row])], ignore_index=True)

    os.makedirs(f'{args.ckpt_root}/{args.run_name}/', exist_ok=True)
    result_df.to_csv(f'{args.ckpt_root}/{args.run_name}/log.csv', index=False)
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2025 SCI Steganalysis Parser')
    # Model Parsers
    parser.add_argument('--model_name', type=str, default='UCNet',
                        help='Model name for training '
                             'ex) UCNet, UCNet_Susmap, PENet_Fixed, PENet_Fixed_Susmap, '
                             'PENet_Arbitrary, PENet_Arbitrary_Susmap '
                             'or timm based model using "timm-[model_name]"')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')

    # Training Parsers
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--losses', nargs='+', default=['CrossEntropy'])
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['AdamW', 'SGD'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default=None, choices=['MultiStep'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers_per_loader', type=int, default=4, help='Number of workers per data loader')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')

    # Dataset Parsers
    parser.add_argument('--data', type=str, default='COYO700M', help='Dataset name')
    parser.add_argument('--stego_method', type=str, default='', help='Stego method')
    parser.add_argument('--transform_config', type=str, default='./configs/')
    parser.add_argument('--aggregate', type=str, default='mean', choices=['mean', 'max'], help='Aggregation method for patch outputs')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of each patch')
    parser.add_argument('--roi_method', type=str, default='nzac_topmean', help='Type of ROI algroithm to use \
                        ex) npatch_2, nzac_top_2, nzac_top_1_mean_1, nzac_top_1_bottom_2, wavelet_haar_2, ...')
    parser.add_argument('--with_center', action='store_true', help='Whether to include center patch first')

    # MISC Parsers
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Path to the csv')
    parser.add_argument('--dataset_root', type=str, default=str(_checkout_data_path('')), help='Path to the dataset')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--ckpt_root', type=str, default='./results/', help='Root directory for checkpoint and log file saving')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    parser.add_argument('--run_name', type=str, default='auto', help='Run name for checkpoint saving')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the run name')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='2025_NSRSteganalysis_SecondHalf', help='Wandb project name')

    args = parser.parse_args()
    gpu_list = [g.strip() for g in args.gpus.split(',') if g.strip().isdigit()]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'     # (선택) 고정된 순서 보장
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu_list)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    ngpus = torch.cuda.device_count()
    print(f'Using GPU(s) (physical): {os.environ["CUDA_VISIBLE_DEVICES"]} '
        f'(Visible count: {ngpus})')

    device = torch.device("cuda:0")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mp.set_start_method('spawn', force=True)

    main(args)
