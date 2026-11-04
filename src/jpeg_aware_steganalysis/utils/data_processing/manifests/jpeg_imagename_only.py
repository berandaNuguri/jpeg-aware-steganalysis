from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import jpeglib
from PIL import Image, ImageFile, JpegImagePlugin
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


def collect_stego_paths(base_dir, stego_algo, data_type='jpg',
                        use_ratio=None, max_count=None):
    """
    - use_ratio: 0~1 사이 비율 (float)
    - max_count: 사용할 최대 이미지 수 (int)
      둘 다 None이면 전체 사용.
    """
    stego_root = os.path.join(base_dir, 'Stego', stego_algo)
    if not os.path.isdir(stego_root):
        print(f"[Warning] Stego directory not found: {stego_root}")
        return []

    if data_type.lower() == 'jpg':
        image_paths = glob.glob(f'{stego_root}/*.jpg') + glob.glob(f'{stego_root}/*.jpeg')
    else:
        image_paths = glob.glob(f'{stego_root}/*.{data_type}')

    if not image_paths:
        return []

    random.shuffle(image_paths)

    if max_count is not None:
        # 정수 모드: 최대 max_count 장 사용
        image_paths = image_paths[:max_count]
    elif use_ratio is not None:
        # 비율 모드: len * use_ratio 장 사용
        use_ratio_clamped = max(0.0, min(1.0, float(use_ratio)))
        k = int(len(image_paths) * use_ratio_clamped)
        image_paths = image_paths[:k]

    return [os.path.abspath(p) for p in image_paths]


def extract_metadata(task, min_size=256):
    """Stego 기준으로 Cover 메타데이터 추출 (subsampling은 PIL 기반)"""
    stego_path, cover_path = task
    try:
        im = Image.open(cover_path)
        arr = np.array(im.convert('RGB'))

        h, w = arr.shape[:2]
        if h < min_size or w < min_size:
            return stego_path, None
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            return stego_path, None

        # QF (jpeglib)
        jpeg = jpeglib.read_dct(cover_path)
        qf = estimate_qf(jpeg.qt[0])

        # subsampling (PIL)
        if im.format != "JPEG":
            return stego_path, None

        try:
            code = JpegImagePlugin.get_sampling(im)
        except Exception:
            return stego_path, None

        code_to_label = {
            -1: "gray",
            0: "444",
            1: "422",
            2: "420",
        }
        subs = code_to_label.get(code, f"unknown({code})")

        return stego_path, (qf, subs, w, h)

    except Exception:
        return stego_path, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV 생성 (Stego 기준, Cover 메타데이터 사용)")
    parser.add_argument('--input_dirs', nargs='+', default=[str(_checkout_data_path('RWSteg/Spatial_30000_to_PillowJPG/QT75_1_Subsample_422/'))])
    parser.add_argument('--stego_algo', type=str, default='UERD_0.4')
    parser.add_argument('--out_base', default='./csv/RWSteg/Spatial_30000_to_PillowJPG/QT75_1_Subsample_422/')
    parser.add_argument('--min_size', type=int, default=256)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--use_ratio', type=str, default='1.0', help=("사용할 데이터 양 지정: 소수(예: '0.1', '0.5', '1.0') → 해당 소숫값 비율만큼 사용, 정수(예: '10000') → 최대 N장 랜덤 샘플링"
        )
    )
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(args.out_base, exist_ok=True)

    use_ratio = None
    max_count = None
    use_str = args.use_ratio.strip()

    if any(ch in use_str for ch in ['.', 'e', 'E']):
        try:
            use_ratio = float(use_str)
        except ValueError:
            raise ValueError(f"--use_ratio '{use_str}'를 float로 변환할 수 없습니다.")
        print(f"[INFO] 전체 데이터의 {use_ratio*100}% 사용해 CSV 생성")
    else:
        try:
            max_count = int(use_str)
        except ValueError:
            raise ValueError(f"--use_ratio '{use_str}'를 int로 변환할 수 없습니다.")
        print(f"[INFO] 최대 {max_count}개 데이터 사용해 CSV 생성")

    print(f"[INFO] Collecting all stego ({args.stego_algo}) image paths...")
    stego_paths = []
    for d in args.input_dirs:
        stego_paths.extend(
            collect_stego_paths(
                d,
                args.stego_algo,
                use_ratio=use_ratio,
                max_count=max_count
            )
        )

    if 'BOSSbase_BOWS2' in args.out_base:
        bows2_stego_paths = []
        # BOWS2는 항상 전체 사용 (기존과 동일하게 ratio=1.0으로 고정)
        bows2_stego_paths.extend(
            collect_stego_paths(
                str(_checkout_data_path('BOWS2_Gray_JPEG/R256_QF75')),
                args.stego_algo,
                use_ratio=1.0,
                max_count=None
            )
        )
        random.shuffle(bows2_stego_paths)

    random.shuffle(stego_paths)
    n = len(stego_paths)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        'train': stego_paths[:n_train],
        'valid': stego_paths[n_train: n_train + n_val],
        'test':  stego_paths[n_train + n_val:]
    }
    if 'BOSSbase_BOWS2' in args.out_base:
        splits['train'].extend(bows2_stego_paths)

    print(f"[INFO] Total stego images: {n}. "
          f"Splits: Train({len(splits['train'])}), Val({len(splits['valid'])}), Test({len(splits['test'])})")

    tasks = []
    for stego_path in stego_paths:
        base_dir = os.path.dirname(os.path.dirname(stego_path))  # .../<dataset_root>/Stego/<algo>/file.jpg
        dataset_root = os.path.dirname(base_dir)
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
            qf, subs, w, h = metadata[stego_path]
            image_name = os.path.basename(stego_path)
            rows.append([image_name, qf, subs, w, h])

        if not rows:
            print(f"[INFO] No data for {split_name}. Skipping.")
            continue

        df = pd.DataFrame(rows, columns=['image_name', 'qf', 'subsampling', 'width', 'height'])
        csv_filename = f"{args.stego_algo}_{split_name}.csv"
        csv_path = os.path.join(args.out_base, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {split_name} -> {csv_path} ({len(df)} images)")
