from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import jpeglib
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

subs_map = {
    (1, 1): '4:4:4',
    (1, 2): '4:4:0',
    (2, 1): '4:2:2',
    (2, 2): '4:2:0',
    (4, 1): '4:1:1',
    (4, 2): '4:1:0',
    (1, 4): '4:4:1',
    (2, 4): '4:2:1',
    (4, 4): '4:1:1',
}

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
    for quality in range(1,101):
        scale = 5000/quality if quality<50 else 200-2*quality
        comp = np.floor((std_table*scale+50)/100).clip(1,255)
        err = np.mean(np.abs(comp - quant_table))
        if err < best_error:
            best_error, best_quality = err, quality
    return best_quality

def flatten_pairs(pairs):
    return [item for pair in pairs for item in pair]

def collect_pairs_from_dir(dir_path, stego_algorithms, data_type='jpg', use_ratio=1.0):
    pair_data = {alg: [] for alg in stego_algorithms}
    
    for alg in stego_algorithms:
        stego_root = os.path.join(dir_path, 'Stego', alg)
        if not os.path.isdir(stego_root):
            print(f"[Warning] Stego directory not found for algorithm '{alg}': {stego_root}")
            continue

        if data_type.lower()=='jpg':
            stego_paths = glob.glob(f'{stego_root}/*.jpg') + glob.glob(f'{stego_root}/*.jpeg')
        else:
            stego_paths = glob.glob(f'{stego_root}/*.{data_type}')
        
        random.shuffle(stego_paths)
        stego_paths = stego_paths[:int(len(stego_paths) * use_ratio)]
        
        bpnzac = alg.split('_')[1]

        for stego_path in tqdm(stego_paths, desc=f'{dir_path}::{alg}'):
            cover_path = stego_path.replace(os.path.join('Stego', alg), 'Cover')
            if not os.path.exists(cover_path):
                continue
            
            try:
                im = Image.open(cover_path)
                arr = np.array(im.convert('RGB'))
                if arr.ndim==2 or arr.shape[2]==1: continue
                # if arr.ndim==3 and np.all(arr[...,0]==arr[...,1]) and np.all(arr[...,1]==arr[...,2]): continue
                h,w = arr.shape[:2]
                if h<256 or w<256: continue
            except Exception as e:
                print(e)
                continue
            
            cover_rel = os.path.join(*cover_path.split(os.sep)[3:])
            stego_rel = os.path.join(*stego_path.split(os.sep)[3:])
            
            pair_data[alg].append([
                (cover_rel, f"{os.path.basename(cover_rel)}", 'cover', '0'),
                (stego_rel, f"{os.path.basename(stego_rel)}", alg.split('_')[0], bpnzac)
            ])
            
    return pair_data

def extract_metadata(full_path):
    im = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    jpeg = jpeglib.read_dct(full_path)
    qf = estimate_qf(jpeg.qt[0])
    sp = jpeglib.read_spatial(full_path)
    h0, v0 = sp.samp_factor[0]
    subs = subs_map.get((h0, v0), f'{h0}x{v0}')
    return full_path, (qf, subs, w, h)

if __name__ == '__main__':
    random.seed(42)

    input_dirs       = [str(_checkout_data_path('COYO700M/PNG_30000_to_PillowJPG/QF90_1/'))]
    stego_algorithms = ['nsF5_0.4']
    train_ratio, val_ratio, test_ratio = 0.6, 0.1, 0.3
    use_ratio        = 1
    out_base         = "./csv/COYO700M/PNG_30000_to_PillowJPG_QF90_1/"
    os.makedirs(out_base, exist_ok=True)

    # ==============================================================================
    # (1) QF 필터링 설정 변수
    # ==============================================================================
    qf_filter_enabled = False
    qf_filter_value   = 100
    qf_filter_mode    = 'equal' 
                                # 'equal' : ==
                                # 'ge'    : >= (크거나 같음)
                                # 'le'    : <= (작거나 같음)
                                # 'gt'    : >  (큼)
                                # 'lt'    : <  (작음)
    # ==============================================================================

    # 1) collect & splits
    all_pd = {d: collect_pairs_from_dir(d, stego_algorithms, use_ratio=use_ratio)
              for d in input_dirs}
    splits = {alg: {'train': [], 'val': [], 'test': []}
              for alg in stego_algorithms}
    for alg in stego_algorithms:
        combined = []
        for d in input_dirs:
            combined.extend(all_pd[d][alg])
            
        random.shuffle(combined)
        n       = len(combined)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        splits[alg]['train'] = combined[:n_train]
        splits[alg]['val']   = combined[n_train:n_train+n_val]
        splits[alg]['test']  = combined[n_train+n_val:]

    # 2) 모든 split 대상 경로 수집
    alg = stego_algorithms[0]
    all_pairs = (
        flatten_pairs(splits[alg]['train']) +
        flatten_pairs(splits[alg]['val']) +
        flatten_pairs(splits[alg]['test'])
    )
    full_paths = [os.path.join(os.getcwd(), '../../data', cover)
                  for cover, *_ in all_pairs]
    unique_paths = list(set(full_paths))

    # 3) 메타정보 추출
    metadata = {}
    with ProcessPoolExecutor() as exe:
        for path, meta in tqdm(exe.map(extract_metadata, unique_paths),
                               total=len(unique_paths), desc="Meta extract"):
            metadata[path] = meta

    # 4) split별로 DataFrame 생성 & 저장
    for split_name in ['train', 'val', 'test']:
        rows = []
        for cover, name, algo, rate in flatten_pairs(splits[alg][split_name]):
            full = os.path.join(os.getcwd(), '../../data', cover)
            qf, subs, w, h = metadata[full]
            
            # ==============================================================================
            # (2) QF 값에 따른 필터링 로직
            # ==============================================================================
            if qf_filter_enabled:
                if qf_filter_mode == 'equal' and qf != qf_filter_value:
                    continue
                elif qf_filter_mode == 'ge' and qf < qf_filter_value:
                    continue
                elif qf_filter_mode == 'le' and qf > qf_filter_value:
                    continue
                elif qf_filter_mode == 'gt' and qf <= qf_filter_value:
                    continue
                elif qf_filter_mode == 'lt' and qf >= qf_filter_value:
                    continue
            # ==============================================================================
            
            rows.append([cover, name, algo, rate, qf, subs, w, h])

        if not rows:
            print(f"[INFO] No data left for {split_name} after QF filtering. Skipping CSV creation.")
            continue

        df = pd.DataFrame(rows, columns=[
            'image_path','image_name','algorithm','embedding_rate',
            'qf','subsampling','width','height'
        ])
        
        # CSV 파일 이름에 필터링 조건을 추가 (선택 사항)
        csv_filename = f"{alg}_{split_name}"
        if qf_filter_enabled:
            csv_filename += f"_qf_{qf_filter_mode}_{qf_filter_value}"
        csv_filename += ".csv"
        
        csv_path = os.path.join(out_base, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {split_name} → {csv_path} ({len(df)} rows)")
        
