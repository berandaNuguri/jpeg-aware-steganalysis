from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import argparse
import random
import shutil
import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from jpeg_aware_steganalysis.utils.runtime import time_to_str, build_transforms, load_model, load_loss



def save_checkpoint(state, gpus, is_best=False, model_path='./ckpt/', model_name='checkpoint.pth.tar'):
    """Multi-GPU 모델을 포함한 체크포인트를 저장합니다."""
    if len(gpus) > 1 and any('.module' in k for k in state['state_dict'].keys()):
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            name = k.replace('.module.', '')
            new_state_dict[name] = v
        state['state_dict'] = new_state_dict

    os.makedirs(model_path, exist_ok=True)
    torch.save(state, os.path.join(model_path, model_name))
    if is_best:
        shutil.copyfile(os.path.join(model_path, model_name), os.path.join(model_path, 'model_best.pth.tar'))

def test(args, test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    correct_values = []

    with torch.no_grad():
        for data, labels, meta in test_loader:
            # ---- shapes ----
            B, Pair, PatchPair, C, H, W = data.shape

            if labels.dim() == 3 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)                 # (B, Pair)
            assert labels.shape[:2] == (B, Pair), f"labels shape {tuple(labels.shape)} must start with (B, Pair)"
            labels = labels.reshape(B * Pair).to(device).long()   # (B*Pair,)

            # ---- to(device) ----
            data = data.to(device) 
            data = data.reshape(B * Pair * PatchPair, C, H, W)

            logits = model(data)
            outputs = logits.view(B * Pair, PatchPair, -1)
            if args.aggregate == 'mean':
                outputs = outputs.mean(dim=1)
            else:
                outputs = outputs.max(dim=1)[0]

            _, predicted = torch.max(outputs.data, 1)
            batch_correct = predicted == labels
            correct += batch_correct.sum().item()
            total += labels.numel()
            correct_values.extend(batch_correct.detach().cpu().numpy().astype(np.int8).tolist())

    accuracy = 100.0 * correct / max(total, 1)
    return accuracy, np.asarray(correct_values, dtype=np.int8)

def compute_bootstrap_accuracy(correct_values, n_bootstrap=1000, seed=42):
    correct_values = np.asarray(correct_values, dtype=np.float32)
    if correct_values.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    bootstrap_acc = np.empty(n_bootstrap, dtype=np.float32)
    n = correct_values.size
    for i in range(n_bootstrap):
        sampled_idx = rng.integers(0, n, size=n)
        bootstrap_acc[i] = correct_values[sampled_idx].mean() * 100.0

    ddof = 1 if n_bootstrap > 1 else 0
    return float(bootstrap_acc.mean()), float(bootstrap_acc.std(ddof=ddof))

def save_bootstrap_result(args, roi_method, accuracy, correct_values):
    bootstrap_mean, bootstrap_std = compute_bootstrap_accuracy(
        correct_values,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "bootstrap_test_results.csv")
    row = {
        "model_name": args.model_name,
        "checkpoint": args.pretrained_path,
        "data": args.data,
        "data_type": args.data_type,
        "stego_method": args.stego_method,
        "eval_splits": args.eval_splits,
        "roi_method": roi_method,
        "aggregate": args.aggregate,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "n_samples": int(np.asarray(correct_values).size),
        "n_bootstrap": args.bootstrap_iters,
        "accuracy": accuracy,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_std": bootstrap_std,
        "bootstrap_mean_std": f"{bootstrap_mean:.4f} ({bootstrap_std:.4f})",
    }
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    return csv_path, bootstrap_mean, bootstrap_std

def arbitrary_test(args, test_loader, model, device):
    def _to_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    def _safe_int(x):
        try:
            if x is None:
                return None
            if isinstance(x, float) and np.isnan(x):
                return None
            return int(x)
        except Exception:
            return None

    def _get_wh_lists(metadata):
        """
        네 Dataset이 반환하는 meta 포맷 가정:
          PNG: meta = [w_list, h_list, ...]
          JPG: meta = [qf_list, subs_list, w_list, h_list, ...]
        """
        meta = metadata
        if isinstance(meta, (list, tuple)):
            meta = [_to_list(m) for m in meta]
        else:
            return None, None

        dt = str(args.data_type).upper()
        if dt == 'PNG':
            if len(meta) < 2:
                return None, None
            return meta[0], meta[1]
        elif dt == 'JPG':
            if len(meta) < 4:
                return None, None
            return meta[2], meta[3]
        else:
            return None, None

    def _bin_name(min_side):
        if min_side is None:
            return None
        if 256 <= min_side < 1024:
            return "256-1024"
        if 1024 <= min_side < 2048:
            return "1024-2048"
        if 2048 <= min_side < 4096:
            return "2048-4096"
        if min_side >= 4096:
            return "4096+"
        return None

    model.eval()
    print("[ArbitraryTest] Evaluation Start")

    bins = ["256-1024", "1024-2048", "2048-4096", "4096+"]
    bin_correct = {b: 0 for b in bins}
    bin_total   = {b: 0 for b in bins}
    excluded_missing = 0
    excluded_small   = 0

    total_correct = 0
    total_count   = 0

    with torch.no_grad():
        for data, labels, metadata in test_loader:
            # data: (B, Pair, PatchPair, C, H, W)
            B, Pair, PatchPair, C, H, W = data.shape
            N = B * Pair

            # labels shape 정리
            if labels.dim() == 3 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)  # (B, Pair)
            labels = labels.reshape(N).to(device).long()

            data = data.to(device).reshape(N * PatchPair, C, H, W)

            logits = model(data).view(N, PatchPair, -1)
            if getattr(args, 'aggregate', 'mean') == 'mean':
                outputs = logits.mean(dim=1)
            else:
                outputs = logits.max(dim=1)[0]

            preds = outputs.argmax(dim=1) 

            total_correct += (preds == labels).sum().item()
            total_count   += N

            w_list, h_list = _get_wh_lists(metadata)
            if w_list is None or h_list is None:
                excluded_missing += N
                continue

            preds_cpu  = preds.detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()

            idx = 0
            for i in range(B):
                w = _safe_int(w_list[i]) if i < len(w_list) else None
                h = _safe_int(h_list[i]) if i < len(h_list) else None
                ms = None if (w is None or h is None) else max(w, h)

                for _ in range(Pair):
                    if ms is None:
                        excluded_missing += 1
                    else:
                        bn = _bin_name(ms)
                        if bn is None:
                            excluded_small += 1
                        else:
                            bin_total[bn] += 1
                            if preds_cpu[idx] == labels_cpu[idx]:
                                bin_correct[bn] += 1
                    idx += 1

    overall_acc = 100.0 * total_correct / max(total_count, 1)
    print("[ArbitraryTest] Evaluation End")
    print(f"\n[ArbitraryTest] Overall Accuracy: {overall_acc:.2f}% (N={total_count})")

    print("\n[ArbitraryTest] Accuracy by min(width,height) bins:")
    rows = []
    for b in bins:
        tot = bin_total[b]
        acc = (100.0 * bin_correct[b] / tot) if tot > 0 else float("nan")
        print(f"  - {b:10s}: {acc:6.2f}%  (n={tot})")
        rows.append((b, acc, tot))

    if excluded_missing > 0 or excluded_small > 0:
        print(f"\n[ArbitraryTest] Excluded: missing_size={excluded_missing}, min_side<256={excluded_small}")

    figure_dir = f"./figures/{args.pretrained_path.split('/')[-2]}/arbitrary_test/"
    
    os.makedirs(figure_dir, exist_ok=True)

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]

    plt.figure(figsize=(8, 4))
    plt.scatter(xs, ys, s=140)
    plt.title(
        f"{os.path.basename(args.pretrained_path.split('/')[-2])}\n"
        f"Overall: {overall_acc:.2f}% | Acc by min-side bins"
    )
    plt.xlabel("Min(width,height) bin")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    for x, y, n in rows:
        if np.isnan(y):
            plt.text(x, 2, f"n={n}", ha='center', va='bottom', fontsize=9)
        else:
            plt.text(x, y + 2, f"{y:.2f}%\n(n={n})", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'{args.roi_method}_acc_vs_minside_bins.png'))
    plt.close()

    return overall_acc, rows

def worker_init_fn(worker_id):
    """데이터 로더 워커 초기화 함수"""
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf, ds_classes = build_transforms(args.transform_config)
    TestDS = ds_classes['test']

    split_names = [s.strip() for s in args.eval_splits.split("+") if s.strip()]
    if not split_names:
        raise ValueError("eval_splits 에 적어도 하나의 split 이름을 넣어야 합니다. (예: 'test')")

    csv_root = os.path.join(args.csv_root, args.data)
    dfs = []
    for split in split_names:
        csv_path = os.path.join(csv_root, f"{args.stego_method}_{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df["split"] = split  # 필요하면 나중에 split별로 분석할 수 있게 컬럼 추가
        dfs.append(df)
        print(f"[INFO] Loaded {split} CSV: {csv_path} (n={len(df)})")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total evaluation samples from splits {split_names}: {len(full_df)}")

    # 여러 개 roi_method 지원
    roi_methods = args.roi_method if isinstance(args.roi_method, list) else [args.roi_method]

    for roi in roi_methods:
        print(f'\n=================================== Evaluating ROI: {roi} ===================================')

        # args 객체 복사 후 roi_method만 교체
        from copy import deepcopy
        local_args = deepcopy(args)
        local_args.roi_method = roi

        # DataLoader 생성
        test_ds = TestDS(full_df, local_args.dataset_root, transform=tf['test'], args=local_args)
        test_loader = DataLoader(
            test_ds,
            batch_size=local_args.batch_size,
            num_workers=local_args.workers_per_loader,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

        model = load_model(local_args, device)
        if getattr(local_args, "arbitrary_test", False):
            if getattr(local_args, "bootstrap_test", False):
                print("[WARN] --bootstrap_test is applied only to the standard test path, not arbitrary_test.")
            arbitrary_test(local_args, test_loader, model, device)
        else:
            acc, correct_values = test(local_args, test_loader, model, device)
            print(f"{args.data.split('/')[-2]}")
            print(f'{roi} - Accuracy : {acc:.2f}%')
            if getattr(local_args, "bootstrap_test", False):
                csv_path, b_mean, b_std = save_bootstrap_result(local_args, roi, acc, correct_values)
                print(f'{roi} - Bootstrap mean(std): {b_mean:.2f} ({b_std:.2f})%')
                print(f'[SAVE] Bootstrap result: {csv_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2025 NSRSteganlaysis SecondHalf Test Script')

    # --- 필수 인자 그룹 ---
    req_group = parser.add_argument_group('Required Arguments')
    req_group.add_argument('--model_name', type=str, required=True, help='Model name to evaluate.')
    req_group.add_argument('--pretrained', action='store_true', help='Using Pretrained Weight for timm models(default=False)')
    req_group.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model checkpoint.')
    req_group.add_argument('--data', type=str, required=True, help='Dataset name (e.g., NSR_Crawling_2025).')
    req_group.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model.')
    
    # --- 데이터셋 및 경로 인자 그룹 ---
    data_group = parser.add_argument_group('Dataset and Path Arguments')
    data_group.add_argument("--eval_splits",type=str, default="test", help="평가에 사용할 split들. 예: 'test', 'train+valid', 'train+valid+test'")
    data_group.add_argument('--data_type', type=str, default='JPG', choices=['JPG', 'PNG'], help='Type of dataset (JPG or PNG).')
    data_group.add_argument('--dataset_root', type=str, default=str(_checkout_data_path('')), help='Root path to the dataset.')
    data_group.add_argument('--csv_root', type=str, default='./csv/', help='Root path to the CSV files.')
    data_group.add_argument('--transform_config', type=str, default='./configs/')
    data_group.add_argument('--stego_method', type=str, default='', help='Steganography method name.')
    data_group.add_argument('--min_size', type=int, default=0, help='Minimum image size (both width and height) required for evaluation')
    data_group.add_argument('--aggregate', type=str, default='mean', choices=['mean', 'max'], help='Aggregation method for patch outputs')
    data_group.add_argument('--patch_size', type=int, default=256, help='Size of each patch')
    data_group.add_argument('--roi_method', type=str, nargs='+', default=['nzac_top_1_median_1'], help='List of ROI algorithms (e.g., npatch_2 nzac_top_2 wavelet_haar_2)')
    
    # --- 실행 환경 인자 그룹 ---
    run_group = parser.add_argument_group('Execution Arguments')
    run_group.add_argument('--batch_size', type=int, default=8, help='Batch size for testing.')
    run_group.add_argument('--workers_per_loader', type=int, default=4, help='Number of workers per DataLoader.')
    run_group.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use.')
    run_group.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')

    # --- 출력 및 기타 인자 그룹 ---
    misc_group = parser.add_argument_group('Output and Misc Arguments')
    misc_group.add_argument('--arbitrary_test',action='store_true', help='If set, report accuracy by min(width,height) size bins.')
    misc_group.add_argument('--figure_prefix', type=str, default='', help='Prefix for saved figure filenames.')
    misc_group.add_argument('--qf_threshold', type=int, default=None, help='Minimum QF to include in the QF accuracy plot.')
    misc_group.add_argument('--qtable_info_path', type=str, default=str(_checkout_data_path('CC12M/qtable_info.pkl')), help='Path to qtalbe_info.pkl mapping QF→qtable.')
    misc_group.add_argument('--bootstrap_test', action='store_true', help='If set, run 1000 bootstrap resamples over per-sample correctness and save mean/std to CSV.')
    misc_group.add_argument('--bootstrap_iters', type=int, default=1000, help='Number of bootstrap resamples used when --bootstrap_test is set.')
    misc_group.add_argument('--output_dir', type=str, default='./outputs', help='Directory for bootstrap test CSV outputs.')
    args = parser.parse_args()
    if args.bootstrap_test and args.bootstrap_iters < 1:
        raise ValueError("--bootstrap_iters must be >= 1 when --bootstrap_test is set.")

    # GPU 설정
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # 재현성을 위한 시드 설정
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        print(f'Using GPU: {args.gpus}')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Using CPU')
        
    mp.set_start_method('spawn', force=True)

    main(args)
