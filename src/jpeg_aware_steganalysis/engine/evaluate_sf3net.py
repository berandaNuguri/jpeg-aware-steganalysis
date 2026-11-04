from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
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

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

@torch.no_grad()
def test(args, test_loader, model, device):
    model.eval()
    total_samples = 0
    correct = 0

    for (imgs, dct, qtable), labels, meta in test_loader:
        B, Pair, PatchPair, C, H, W = imgs.shape
        Bi, Pi, Ppi, Ci, Hi, Wi = dct.shape
        Bq, Pq, Ppq, Cq, Hq, Wq = qtable.shape

        assert (Bi, Pi, Ppi, Ci, Hi, Wi) == (B, Pair, PatchPair, C, H, W)
        assert (Bq, Pq, Ppq) == (B, Pair, PatchPair)
        assert (Hq, Wq) == (8, 8)

        N = B * Pair * PatchPair

        imgs   = imgs.view(N, C, H, W).to(device)
        dct    = dct.view(N, C, H, W).to(device)
        qtable = qtable.view(N, Cq, Hq, Wq).to(device)
        
        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        assert labels.shape[:2] == (B, Pair), f"labels shape {tuple(labels.shape)} must start with (B, Pair)"
        labels = labels.reshape(B * Pair).to(device).long()

        # ---- forward & loss ----
        logits = model(imgs, dct, qtable)
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
    
def test_plot(args, test_loader, model, device):
    def _safe_int(x):
        try:
            if x is None:
                return None
            if isinstance(x, float) and np.isnan(x):
                return None
            return int(x)
        except Exception:
            return None

    def _as_seq(x):
        # tensor/ndarray -> list, 나머지는 그대로
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    model.eval()
    print("Evaluation Start")

    qf_results, subsampling_results, size_results, correct_results = [], [], [], []

    with torch.no_grad():
        for data, labels, metadata in test_loader:
            # data: (bs0, pair, n_patches, C, H, W)
            bs0, pair, n_patches, C, H, W = data.shape
            N = bs0 * pair  # 플랫된 샘플 수

            # 모델 입력 준비
            data   = data.view(N, n_patches, C, H, W).to(device)
            labels = labels.view(N).to(device)

            flat = data.view(N * n_patches, C, H, W)
            patch_outputs = model(flat).view(N, n_patches, -1)

            if getattr(args, 'aggregate', 'mean') == 'mean':
                outputs = patch_outputs.mean(dim=1)
            else:
                outputs = patch_outputs.max(dim=1)[0]

            predictions = outputs.argmax(dim=1)

            # 정답 기록 (길이 N)
            correct_results.extend((predictions == labels).int().tolist())

            # --- Metadata 처리 ---
            meta = metadata 
            if isinstance(meta, (list, tuple)) and len(meta) > 0 and not isinstance(meta[0], (list, tuple, np.ndarray, torch.Tensor)):
                pass

            meta = [_as_seq(m) for m in meta]

            if str(args.data_type).upper() == 'PNG':
                # 기대: meta = [w_list, h_list, (옵션:qtable...)]
                w_list = meta[0]
                h_list = meta[1]
                for i in range(bs0):
                    w = _safe_int(w_list[i])
                    h = _safe_int(h_list[i])
                    size = None if (w is None or h is None) else max(w, h)
                    if size is None:
                        # 그래도 길이 정합성을 맞추기 위해 pair번 placeholder라도 추가
                        size_results.extend([0] * pair)
                    else:
                        size_results.extend([size] * pair)

            elif str(args.data_type).upper() == 'JPG':
                # 기대: meta = [qf_list, subs_list, w_list, h_list, (옵션:qtable...)]
                qf_list   = meta[0]
                subs_list = meta[1]
                w_list    = meta[2]
                h_list    = meta[3]

                for i in range(bs0):
                    qf   = _safe_int(qf_list[i])
                    subs = subs_list[i]
                    subs = None if subs is None else str(subs)

                    w = _safe_int(w_list[i])
                    h = _safe_int(h_list[i])
                    size = None if (w is None or h is None) else max(w, h)

                    # pair번 복제
                    if qf is not None:
                        qf_results.extend([qf] * pair)
                    else:
                        # 정합성 유지: None이면 채우지 않고 아래에서 컬럼 생성 시 조건부 처리
                        pass

                    if subs is not None:
                        subsampling_results.extend([subs] * pair)

                    if size is not None:
                        size_results.extend([size] * pair)
                    else:
                        size_results.extend([0] * pair)  # placeholder

            else:
                try:
                    w_list = meta[0]; h_list = meta[1]
                    for i in range(bs0):
                        w = _safe_int(w_list[i]); h = _safe_int(h_list[i])
                        size = None if (w is None or h is None) else max(w, h)
                        size_results.extend([size if size is not None else 0] * pair)
                except Exception:
                    size_results.extend([0] * N)

    print("Evaluation End")

    # DataFrame 구성 (길이 모두 N이어야 함)
    df = pd.DataFrame({'size': size_results, 'correct': correct_results})
    # placeholder 0 제거(있다면)
    df = df.copy()
    df.loc[df['size'] <= 0, 'size'] = np.nan

    if len(qf_results) == len(correct_results):
        df['qf'] = qf_results
    if len(subsampling_results) == len(correct_results):
        df['sub'] = subsampling_results

    overall_acc = df['correct'].mean()
    print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

    figure_dir = f'{os.path.dirname(args.pretrained_path)}/figures/'
    os.makedirs(figure_dir, exist_ok=True)

    # --- 1) QF별 정확도 (JPG 전용) ---
    if str(args.data_type).upper() == 'JPG' and 'qf' in df.columns:
        qf_threshold = getattr(args, 'qf_threshold', 0)
        df_qf = df.dropna(subset=['qf']).copy()
        df_qf['qf'] = df_qf['qf'].astype(int)
        df_qf = df_qf[df_qf['qf'] == qf_threshold]
        if not df_qf.empty:
            qf_acc = df_qf['correct'].mean()
            print(f"Accuracy for QF >= {qf_threshold}: {qf_acc * 100:.2f}%")

            perf_qf = df_qf.groupby('qf', observed=True)['correct'].mean()
            plt.figure(figsize=(14, 5))
            plt.scatter(perf_qf.index, perf_qf.values, s=80)
            plt.xticks(sorted(perf_qf.index), rotation=45)
            title = (f'{os.path.basename(args.pretrained_path.split("/")[-2])}\n'
                     f'Overall Acc: {overall_acc * 100:.2f}%, QF>={qf_threshold}: {qf_acc * 100:.2f}%')
            plt.title(title)
            plt.xlabel('Estimated QF')
            plt.ylabel('Accuracy')
            plt.ylim(top=1.05)
            plt.grid(True, linestyle='--', alpha=0.6)
            for q, acc in perf_qf.items():
                plt.text(q, acc + 0.01, f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f'{args.roi_method}_acc_vs_qf.png'))
            plt.close()

    # --- 2) Subsampling별 정확도 (JPG 전용) ---
    if str(args.data_type).upper() == 'JPG' and 'sub' in df.columns:
        df_s = df.dropna(subset=['sub']).copy()
        df_s['sub'] = df_s['sub'].astype(str)
        if not df_s.empty:
            perf_sub = df_s.groupby('sub', observed=True)['correct'].mean()
            plt.figure(figsize=(14, 5))
            plt.scatter(perf_sub.index, perf_sub.values, s=100)
            plt.title(f'{os.path.basename(args.pretrained_path.split("/")[-2])}\n'
                     f'Overall Acc: {overall_acc * 100:.2f}%')
            plt.xlabel('Subsampling')
            plt.ylabel('Accuracy')
            plt.ylim(top=1.05)
            plt.grid(True, linestyle='--', alpha=0.6)
            for sub, acc in perf_sub.items():
                plt.text(sub, acc + 0.01, f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f'{args.roi_method}_acc_vs_sub.png'))
            plt.close()

    # --- 3) 이미지 크기별 정확도 ---
    if 'size' in df.columns:
        df_sizes = df.dropna(subset=['size']).copy()
        if not df_sizes.empty:
            df_sizes['size'] = df_sizes['size'].astype(int)
            bins = [0, 512, 1024, 1536, 2048, np.inf]
            labels_ = ['<512', '512-1024', '1024-1536', '1536-2048', '>2048']
            df_sizes['size_bin'] = pd.cut(df_sizes['size'], bins=bins, labels=labels_, right=False)
            perf_size = df_sizes.groupby('size_bin', observed=True)['correct'].mean()

            plt.figure(figsize=(14, 5))
            plt.scatter(perf_size.index.astype(str), perf_size.values, s=100)
            plt.title(f'{os.path.basename(args.pretrained_path.split("/")[-2])}\n'
                     f'Overall Acc: {overall_acc * 100:.2f}%')
            plt.xlabel('Image Size Bin')
            plt.ylabel('Accuracy')
            plt.ylim(top=1.05)
            plt.grid(True, linestyle='--', alpha=0.6)
            for size_bin, acc in perf_size.items():
                plt.text(str(size_bin), acc + 0.01, f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f'{args.roi_method}_acc_vs_size.png'))
            plt.close()


def worker_init_fn(worker_id):
    """데이터 로더 워커 초기화 함수"""
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf, ds_classes = build_transforms(args.transform_config)
    TestDS = ds_classes['test']

    csv_path_parts = [args.csv_root, args.data]
    test_csv = os.path.join(*csv_path_parts, f'{args.stego_method}_test.csv')
    test_df = pd.read_csv(test_csv)

    # 여러 개 roi_method 지원
    roi_methods = args.roi_method if isinstance(args.roi_method, list) else [args.roi_method]

    for roi in roi_methods:
        print(f'\n=================================== Evaluating ROI: {roi} ===================================')

        # args 객체 복사 후 roi_method만 교체
        from copy import deepcopy
        local_args = deepcopy(args)
        local_args.roi_method = roi

        # --- QT + 크기 필터링 ---
        if local_args.target_qt is not None:
            with open(local_args.qtable_info_path, 'rb') as f:
                stats = pickle.load(f)
            qt_key = f'QT{local_args.target_qt}'
            if qt_key not in stats:
                raise KeyError(f"{local_args.target_qt} not in {local_args.qtable_info_path}")

            target_table = np.array(stats[qt_key]['qtable'][:64])

            def match_qtable_and_size(row):
                img_path = os.path.join(local_args.dataset_root, local_args.data, "Cover", row['image_name'])
                try:
                    with Image.open(img_path) as img:
                        qtables = img.quantization
                        if not qtables or 0 not in qtables:
                            return False
                        ytable = np.array(qtables[0])
                        if not np.array_equal(ytable, target_table):
                            return False
                        w, h = img.size
                        if w < local_args.min_size or h < local_args.min_size:
                            return False
                        return True
                except Exception:
                    return False

            filtered_df = test_df[test_df.apply(match_qtable_and_size, axis=1)]
            print(f"→ Filtered dataset to QT={local_args.target_qt} "
                  f"and size ≥{local_args.min_size}, remaining {len(filtered_df)} cover samples")
        else:
            filtered_df = test_df

        # DataLoader 생성
        test_ds = TestDS(filtered_df, local_args.dataset_root, transform=tf['test'], args=local_args)
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
        acc = test(local_args, test_loader, model, device)
        print(f'{roi} - Accuracy : {acc:.2f}%')
        # if local_args.target_qt is not None:
        #     acc = test(local_args, test_loader, model, device)
        #     print(f'{roi} - Accuracy : {acc:.2f}%')
        # else:
        #     test_plot(local_args, test_loader, model, device)

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
    misc_group.add_argument('--figure_prefix', type=str, default='', help='Prefix for saved figure filenames.')
    misc_group.add_argument('--qf_threshold', type=int, default=None, help='Minimum QF to include in the QF accuracy plot.')
    misc_group.add_argument('--qtable_info_path', type=str, default=str(_checkout_data_path('CC12M/qtable_info.pkl')), help='Path to qtalbe_info.pkl mapping QF→qtable.')
    misc_group.add_argument('--target_qt', type=str, default=None, help='Only evaluate samples matching this key in qtalbe_info.pkl (e.g. QT90_1).')

    args = parser.parse_args()

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