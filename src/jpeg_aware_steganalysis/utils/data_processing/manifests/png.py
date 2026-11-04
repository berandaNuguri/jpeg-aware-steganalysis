from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
#!/usr/bin/env python3
import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Allow large and truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def flatten_pairs(pairs):
    """
    Flatten a list of cover-stego pair entries.
    """
    return [item for pair in pairs for item in pair]


def collect_pairs_from_dir(dir_path, stego_algorithms, data_type='png', use_ratio=1.0):
    """
    Collect (cover, stego) pairs from directory structure:
      dir_path/Cover/... and dir_path/Stego/alg/...  
    Returns dict: {alg: [ [ (cover_rel,...), (stego_rel,...) ], ... ] }
    """
    pair_data = {alg: [] for alg in stego_algorithms}
    for alg in stego_algorithms:
        stego_root = os.path.join(dir_path, 'Stego', alg)
        if not os.path.isdir(stego_root):
            print(f"[Warning] Stego 디렉터리가 없습니다: {stego_root}")
            continue

        stego_paths = glob.glob(f'{stego_root}/*.{data_type}')
        random.shuffle(stego_paths)
        stego_paths = stego_paths[:int(len(stego_paths) * use_ratio)]
        embedding_rate = alg.split('_')[1] if '_' in alg else '0'

        for stego_path in tqdm(stego_paths, desc=f'{os.path.basename(dir_path)}::{alg}', unit='img'):
            cover_path = stego_path.replace(os.path.join('Stego', alg), 'Cover')
            if not os.path.exists(cover_path):
                continue

            # 예외처리: grayscale, single-channel, 너무 작으면 스킵
            try:
                im = Image.open(cover_path)
                arr = np.array(im)
                im.close()
            except Exception as e:
                print(f"[Warning] cover 로드 실패: {cover_path} – {e}")
                continue

            # 그레이스케일 혹은 채널 1개
            if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                continue
            # 3채널인데 R=G=B 모두 같으면(흑백)
            if arr.ndim == 3 and np.all(arr[...,0] == arr[...,1]) and np.all(arr[...,1] == arr[...,2]):
                continue
            h, w = arr.shape[:2]
            if h < 256 or w < 256:
                continue

            cover_rel = os.path.join(*cover_path.split(os.sep)[3:])
            stego_rel = os.path.join(*stego_path.split(os.sep)[3:])
            pair_data[alg].append([
                (cover_rel, os.path.basename(cover_rel), 'cover', '0'),
                (stego_rel, os.path.basename(stego_rel), alg.split('_')[0], embedding_rate)
            ])
    return pair_data


def extract_metadata(full_path):
    """
    Open image, validate with PIL and cv2, return (width, height, channels).
    If load fails, return (None, None, None) to skip later.
    """
    # 1) PIL load & metadata
    try:
        im = Image.open(full_path)
        w, h = im.size
        mode_to_channels = {
            '1':1, 'L':1, 'P':1,
            'RGB':3, 'RGBA':4, 'CMYK':4
        }
        channels = mode_to_channels.get(im.mode, len(im.getbands()))
        im.close()
    except Exception as e:
        print(f"[Warning] PIL metadata load failed: {full_path} – {e}")
        return full_path, (None, None, None)

    # 2) cv2 load & convert check
    try:
        img = cv2.imdecode(full_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imread returned None")
        _ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[Warning] cv2 load/convert failed: {full_path} – {e}")
        return full_path, (None, None, None)

    return full_path, (w, h, channels)


if __name__ == '__main__':
    random.seed(42)
    input_dirs       = [str(_checkout_data_path('COYO700M/PNG_50000/'))]
    stego_algorithms = ['LSB_0.4']
    train_ratio, val_ratio, test_ratio = 0.6, 0.1, 0.3
    use_ratio        = 1.0
    out_base         = "./csv/COYO700M/PNG_50000/"
    os.makedirs(out_base, exist_ok=True)

    # 1) collect & split
    all_pd = {
        d: collect_pairs_from_dir(d, stego_algorithms, data_type='png', use_ratio=use_ratio)
        for d in input_dirs
    }
    splits = {alg: {'train': [], 'val': [], 'test': []} for alg in stego_algorithms}
    for alg in stego_algorithms:
        combined = []
        for d in input_dirs:
            combined.extend(all_pd[d][alg])
        random.shuffle(combined)
        n = len(combined)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        splits[alg]['train'] = combined[:n_train]
        splits[alg]['val']   = combined[n_train:n_train+n_val]
        splits[alg]['test']  = combined[n_train+n_val:]

    # 2) metadata paths
    alg = stego_algorithms[0]
    all_pairs = (
        flatten_pairs(splits[alg]['train']) +
        flatten_pairs(splits[alg]['val']) +
        flatten_pairs(splits[alg]['test'])
    )
    full_paths = [
        os.path.join(os.getcwd(), '../../data', rel_path)
        for rel_path, *_ in all_pairs
    ]
    unique_paths = list(set(full_paths))

    # 3) extract metadata
    metadata = {}
    with ProcessPoolExecutor() as exe:
        for path, meta in tqdm(
                exe.map(extract_metadata, unique_paths),
                total=len(unique_paths),
                desc="Meta extract"
        ):
            w, h, ch = meta
            if w is None:
                continue
            metadata[path] = (w, h, ch)

    # 4) save CSVs
    for split_name in ['train', 'val', 'test']:
        rows = []
        for rel_path, name, algo, rate in flatten_pairs(splits[alg][split_name]):
            full = os.path.join(os.getcwd(), '../../data', rel_path)
            w_h_ch = metadata.get(full)
            if w_h_ch is None:
                continue
            w, h, ch = w_h_ch
            rows.append([rel_path, name, algo, rate, w, h, ch])

        if not rows:
            print(f"[INFO] {split_name} 데이터 없음, 스킵")
            continue

        df = pd.DataFrame(rows, columns=[
            'image_path', 'image_name', 'algorithm', 'embedding_rate',
            'width', 'height', 'channels'
        ])
        csv_path = os.path.join(out_base, f"{alg}_{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {split_name} → {csv_path} ({len(df)} rows)")
