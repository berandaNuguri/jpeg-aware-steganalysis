from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def collect_stego_paths(base_dir, stego_algo, use_ratio=1.0):
    """Stego/{algo}/ 폴더 내 이미지 경로 수집"""
    stego_root = os.path.join(base_dir, 'Stego', stego_algo)
    if not os.path.isdir(stego_root):
        print(f"[Warning] Stego directory not found: {stego_root}")
        return []

    image_paths = glob.glob(os.path.join(stego_root, f'*.png')) + \
                      glob.glob(os.path.join(stego_root, f'*.PNG'))

    random.shuffle(image_paths)
    image_paths = image_paths[:int(len(image_paths) * use_ratio)]
    return [os.path.abspath(p) for p in image_paths]


def extract_metadata(task, min_size=256):
    """Stego 기반으로 Cover 메타데이터 읽기 (PNG용: width/height만)"""
    stego_path, cover_path = task
    try:
        im = Image.open(cover_path)
        arr = np.array(im.convert('RGB'))
        h, w = arr.shape[:2]

        # 너무 작은 이미지/이상 케이스는 제외
        if h < min_size or w < min_size:
            return stego_path, None
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            return stego_path, None
        
        # PNG에선 qt/subsampling이 없으므로 (w,h)만 반환
        return stego_path, (w, h)

    except Exception:
        return stego_path, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV 생성 (Stego 기준, Cover 메타데이터 사용) — PNG용")
    parser.add_argument('--input_dirs', nargs='+', default=[str(_checkout_data_path('COYO_700M/PNG_30000/'))],
                        help="데이터셋 루트 (Cover/Stego 포함)")
    parser.add_argument('--stego_algo', type=str, default='nsF5_0.4', help="Stego 알고리즘 이름 (폴더명)")
    parser.add_argument('--out_base', default='./csv/COYO_700M/PNG_30000/', help="CSV 저장 경로")
    parser.add_argument('--min_size', type=int, default=256)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--use_ratio', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(args.out_base, exist_ok=True)

    print(f"[INFO] Collecting all stego ({args.stego_algo}) image paths...")
    stego_paths = []
    for d in args.input_dirs:
        stego_paths.extend(collect_stego_paths(d, args.stego_algo, use_ratio=args.use_ratio))

    random.shuffle(stego_paths)
    n = len(stego_paths)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        'train': stego_paths[:n_train],
        'valid': stego_paths[n_train : n_train + n_val],
        'test':  stego_paths[n_train + n_val :]
    }
    print(f"[INFO] Total stego images: {n}. Splits: Train({len(splits['train'])}), "
          f"Val({len(splits['valid'])}), Test({len(splits['test'])})")

    # Stego 경로에서 Cover 경로로 매칭
    tasks = []
    for stego_path in stego_paths:
        base_dir = os.path.dirname(os.path.dirname(stego_path))   # .../<dataset>/Stego
        dataset_root = os.path.dirname(base_dir)                  # .../<dataset>
        cover_path = os.path.join(dataset_root, 'Cover', os.path.basename(stego_path))
        if os.path.exists(cover_path):
            tasks.append((stego_path, cover_path))

    metadata = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_metadata, t, args.min_size): t[0] for t in tasks}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting Metadata"):
                stego_path, meta = f.result()
                if meta:
                    metadata[stego_path] = meta

    for split_name in ['train', 'valid', 'test']:
        rows = []
        for stego_path in splits[split_name]:
            if stego_path not in metadata:
                continue
            w, h = metadata[stego_path]
            image_name = os.path.basename(stego_path)
            rows.append([image_name, w, h])

        if not rows:
            print(f"[INFO] No data for {split_name}. Skipping.")
            continue

        df = pd.DataFrame(rows, columns=['image_name', 'width', 'height'])
        csv_filename = f"{args.stego_algo}_{split_name}.csv"
        csv_path = os.path.join(args.out_base, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {split_name} -> {csv_path} ({len(df)} images)")
