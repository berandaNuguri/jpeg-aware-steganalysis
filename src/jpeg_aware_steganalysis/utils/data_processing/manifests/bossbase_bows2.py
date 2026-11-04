from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import jpeglib
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def estimate_qf(quant_table):
    std_table = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ], dtype=np.float32)
    best_quality, best_error = None, np.inf
    for quality in range(1, 101):
        scale = 5000/quality if quality < 50 else 200 - 2*quality
        comp = np.floor((std_table * scale + 50) / 100).clip(1, 255)
        err = np.mean(np.abs(comp - quant_table))
        if err < best_error:
            best_error, best_quality = err, quality
    return best_quality


def collect_stego_paths(base_dir, stego_algo, data_type='jpg', use_ratio=1.0):
    stego_root = os.path.join(base_dir, 'Stego', stego_algo)
    if not os.path.isdir(stego_root):
        print(f"[Warning] Stego directory not found: {stego_root}")
        return []
    if data_type.lower() == 'jpg':
        image_paths = glob.glob(f'{stego_root}/*.jpg') + glob.glob(f'{stego_root}/*.jpeg')
    else:
        image_paths = glob.glob(f'{stego_root}/*.{data_type}')
    random.shuffle(image_paths)
    image_paths = image_paths[:int(len(image_paths) * use_ratio)]
    return [os.path.abspath(p) for p in image_paths]


def extract_metadata(task, min_size=256):
    """Stego 기준으로 Cover 메타데이터 추출"""
    stego_path, cover_path = task
    try:
        im = Image.open(cover_path)
        arr = np.array(im.convert('L'))

        h, w = arr.shape[:2]
        if h < min_size or w < min_size:
            return stego_path, None

        jpeg = jpeglib.read_dct(cover_path)
        qf = estimate_qf(jpeg.qt[0])

        return stego_path, (qf, w, h)

    except Exception:
        return stego_path, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV 생성 (Stego 기준, Cover 메타데이터 사용)")
    parser.add_argument('--input_dirs', nargs='+', default=[str(_checkout_data_path('BOSSbase_JPEG/R256_QF95'))])
    parser.add_argument('--stego_algo', type=str, default='J-UNIWARD_0.4')
    parser.add_argument('--out_base', default='./csv/BOSSbase_BOWS2/R256_QF95')
    parser.add_argument('--min_size', type=int, default=256)
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    parser.add_argument('--use_ratio', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(args.out_base, exist_ok=True)

    print(f"[INFO] Collecting all stego ({args.stego_algo}) image paths...")
    stego_paths = []
    for d in args.input_dirs:
        stego_paths.extend(collect_stego_paths(d, args.stego_algo, use_ratio=args.use_ratio))
     
    bows2_stego_paths = []
    bows2_stego_paths.extend(collect_stego_paths(str(_checkout_data_path('BOWS2_Gray_JPEG/R256_QF95')), args.stego_algo, use_ratio=1.0))
    random.shuffle(bows2_stego_paths)
    
    random.shuffle(stego_paths)
    n = len(stego_paths)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        'train': stego_paths[:n_train],
        'valid': stego_paths[n_train : n_train + n_val],
        'test':  stego_paths[n_train + n_val :]
    }

    splits['train'].extend(bows2_stego_paths)

    print(f"[INFO] Total stego images: {n}. Splits: Train({len(splits['train'])}), Val({len(splits['valid'])}), Test({len(splits['test'])})")

    tasks = []
    for split_name in ['train', 'valid', 'test']:
        for stego_path in splits[split_name]:
            base_dir = os.path.dirname(os.path.dirname(stego_path))  # .../<dataset_root>/Stego/<algo>
            dataset_root = os.path.dirname(base_dir)                 # .../<dataset_root>
            cover_path = os.path.join(dataset_root, 'Cover', os.path.basename(stego_path))
            if os.path.exists(cover_path):
                tasks.append((stego_path, cover_path))

    metadata = {}
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
            qf, w, h = metadata[stego_path]
            image_name = os.path.basename(stego_path)
            rows.append([stego_path.split('/')[3] + '/' + stego_path.split('/')[4], image_name, qf, w, h])

        if not rows:
            print(f"[INFO] No data for {split_name}. Skipping.")
            continue

        df = pd.DataFrame(rows, columns=['bench', 'image_name', 'qf', 'width', 'height'])
        csv_filename = f"{args.stego_algo}_{split_name}.csv"
        csv_path = os.path.join(args.out_base, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {split_name} -> {csv_path} ({len(df)} images)")
