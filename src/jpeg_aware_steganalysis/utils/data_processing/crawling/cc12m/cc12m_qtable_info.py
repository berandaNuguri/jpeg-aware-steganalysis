import os
import io
import sys
import pickle
import argparse
import requests
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from collections import Counter, defaultdict
from typing import Iterable, Optional, Tuple, Dict, List

# ==============================
# JPEG 유틸
# ==============================
def _is_jpeg_signature(content: bytes) -> bool:
    return content.startswith(b"\xFF\xD8") and content.endswith(b"\xFF\xD9")

def _is_valid_image_pillow(content: bytes, min_size: int = 256) -> bool:
    try:
        bio = io.BytesIO(content)
        img = Image.open(bio)
        img.verify()
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception:
        return False

    arr = np.array(img)
    h, w = arr.shape[:2]
    if h < min_size or w < min_size:
        return False

    # 흑백 제외
    if img.mode in ("L", "LA"):
        return False
    if img.mode == "RGB":
        if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
            return False
    return True

def _is_valid_image_opencv(content: bytes, min_size: int = 256) -> bool:
    try:
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        h, w = img.shape[:2]
        if h < min_size or w < min_size:
            return False
        if img.ndim == 2:
            return False
        if img.ndim == 3:
            c = img.shape[2]
            if c == 2:
                return False
            if c >= 3:
                b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                if np.array_equal(b, g) and np.array_equal(g, r):
                    return False
        return True
    except Exception:
        return False

def _check_jpeglib_safe_bytes_to_tmp(content: bytes, tmp_dir: str = "./tmp_check") -> bool:
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "tmp_check.jpg")
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        code = f"""
import jpeglib
try:
    jpeg = jpeglib.read_dct(r'''{tmp_path}''')
    _ = jpeg.qt[0]
    exit(0)
except Exception:
    exit(1)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        return result.returncode == 0
    except Exception:
        return False

# ==============================
# QF 추정 (표준 Luma 테이블 기준)
# ==============================
def estimate_qf_from_luma(luma_8x8: np.ndarray) -> int:
    std_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99]
    ], dtype=np.float32)
    best_q, best_err = 1, np.inf
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        comp = np.floor((std_table * scale + 50) / 100)
        comp = np.clip(comp, 1, 255)
        err = np.mean(np.abs(comp - luma_8x8))
        if err < best_err:
            best_err, best_q = err, q
    return int(best_q)

# ==============================
# Quantization Table (Y, C) 정규화
# ==============================
def _flatten64(q: Iterable[int]) -> Tuple[int, ...]:
    q = tuple(int(x) for x in q)
    if len(q) != 64:
        raise ValueError(f"Quant table length != 64 (len={len(q)})")
    return q

def canonical_qt_pair_from_pillow_quant(qtables: Dict[int, List[int]]) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    PIL.Image.quantization -> {table_id: [64], ...}
    (Y, C) 64튜플로 정규화:
      - Y: 키 0이 있으면 0, 없으면 가장 작은 키
      - C: 키 1이 있으면 1, 없으면 Y와 다른 테이블 중 첫 번째, 없으면 Y와 동일
    """
    if not qtables:
        return None
    keys = sorted(qtables.keys())
    if not keys:
        return None
    y_key = 0 if 0 in qtables else keys[0]
    y_tbl = _flatten64(qtables[y_key])
    if 1 in qtables:
        c_tbl = _flatten64(qtables[1])
    else:
        c_tbl = None
        for k in keys:
            cand = _flatten64(qtables[k])
            if cand != y_tbl:
                c_tbl = cand
                break
        if c_tbl is None:
            c_tbl = y_tbl
    return y_tbl, c_tbl

def extract_qtables_from_bytes(content: bytes) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...], int]]:
    """
    바이트에서 (Y64, C64, qf_est) 추출. 실패 시 None
    """
    try:
        img = Image.open(io.BytesIO(content))
        qtables = getattr(img, "quantization", None)
        if not qtables:
            return None
        pair = canonical_qt_pair_from_pillow_quant(qtables)
        if pair is None:
            return None
        y_tbl, c_tbl = pair
        y8 = np.array(y_tbl, dtype=np.float32).reshape(8, 8)
        qf = estimate_qf_from_luma(y8)
        return y_tbl, c_tbl, qf
    except Exception:
        return None

# ==============================
# 다운로드 & 처리 워커
# ==============================
def process_url_for_qtable(
    url: str,
    min_size: int = 256,
    timeout: float = 10.0,
    jpeglib_check: bool = False,
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...], int]]:
    """
    URL을 받아 JPEG 검증 후 (Y64, C64, qf_est) 반환. 실패 시 None
    """
    try:
        # 확장자 빠른 필터
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext not in [".jpg", ".jpeg"]:
            return None

        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        content = r.content

        # JPEG 시그니처 및 디코딩 유효성
        if not _is_jpeg_signature(content):
            return None
        if not (_is_valid_image_pillow(content, min_size=min_size) and _is_valid_image_opencv(content, min_size=min_size)):
            return None

        # (옵션) jpeglib 안전검사
        if jpeglib_check and not _check_jpeglib_safe_bytes_to_tmp(content):
            return None

        # QTables 추출
        out = extract_qtables_from_bytes(content)
        return out
    except Exception:
        return None

# ==============================
# 메인 파이프라인: 고유 QTable 수집
# ==============================
def collect_unique_qtables_from_urls(
    tsv_path: str,
    qf_min: int,
    qf_max: int,
    num_workers: int = 32,
    max_urls: Optional[int] = None,
    per_qf_limit: Optional[int] = None,
    min_size: int = 256,
    timeout: float = 10.0,
    jpeglib_check: bool = False,
):
    # URL 로드
    urls = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            urls.append(parts[0])
    if max_urls is not None:
        urls = urls[:max_urls]

    # QF 범위 처리 (1~100로 클램프, 오름차순 정렬)
    qf_min = max(1, min(100, int(qf_min)))
    qf_max = max(1, min(100, int(qf_max)))
    if qf_min > qf_max:
        qf_min, qf_max = qf_max, qf_min
    target_qfs_set = set(range(qf_min, qf_max + 1))

    # 카운터
    pair_counters: Dict[int, Counter] = {qf: Counter() for qf in sorted(target_qfs_set)}
    total_seen = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_url_for_qtable, url, min_size, timeout, jpeglib_check): url for url in urls}

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"스캔 중(QF {qf_min}~{qf_max})"):
            res = fut.result()
            total_seen += 1
            if res is None:
                continue
            y_tbl, c_tbl, qf_est = res
            if qf_est not in target_qfs_set:
                continue

            pair = (tuple(y_tbl), tuple(c_tbl))
            pair_counters[qf_est][pair] += 1

            # per_qf_limit이 모두 충족되면 조기 종료
            if per_qf_limit is not None:
                done_all = True
                for qf in pair_counters:
                    if len(pair_counters[qf]) < per_qf_limit:
                        done_all = False
                        break
                if done_all:
                    break

    return pair_counters, total_seen, (qf_min, qf_max)

def build_ranked_entries_from_counters(pair_counters: Dict[int, Counter]) -> Dict[str, Dict]:
    """
    QF별 Counter를 받아 key='QT{qf}_{rank}' 딕셔너리로 변환.
    정렬: 각 QF 내에서 count 내림차순.
    값: {"qf": int, "qtable": tuple(128), "count": int}
    """
    result = {}
    for qf in sorted(pair_counters.keys()):
        items = pair_counters[qf].most_common()  # list of ((y_tbl, c_tbl), cnt)
        for rank, (pair, cnt) in enumerate(items, start=1):
            y_tbl, c_tbl = pair
            q128 = tuple(y_tbl) + tuple(c_tbl)
            key = f"QT{qf}_{rank}"
            result[key] = {
                "qf": int(qf),
                "qtable": q128,
                "count": int(cnt),
            }
    return result

def save_pickle(obj: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved pickle: {out_path} (items={len(obj)})")

# ==============================
# CLI
# ==============================
def parse_qf_range(s: str) -> Tuple[int, int]:
    """
    '75,100' 또는 '[75,100]' 형식 허용.
    반환: (qf_min, qf_max) (1~100로 클램프, min<=max 보장)
    """
    s = s.strip().strip("[]()")
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("qf_range 형식은 'min,max' 혹은 '[min,max]' 여야 합니다.")
    a, b = int(parts[0]), int(parts[1])
    a = max(1, min(100, a))
    b = max(1, min(100, b))
    if a > b:
        a, b = b, a
    return a, b

def parse_args():
    p = argparse.ArgumentParser(description="URL TSV에서 QF 범위의 고유 QTable(Y,C)을 수집하여 pickle로 저장")
    p.add_argument("--tsv_path", type=str, default="./data/CC12M/cc12m.tsv", help="첫 컬럼이 URL인 TSV 경로")
    p.add_argument("--save_pickle", type=str, default='./qtable_info.pkl', help="저장할 pickle 경로")
    p.add_argument("--qf_range", type=str, default="75,100", help="포함 범위: 'min,max' 또는 '[min,max]' (예: 75,100)")
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--max_urls", type=int, default=1000000, help="최대 처리 URL 수 (None=전체)")
    p.add_argument("--per_qf_limit", type=int, default=None, help="QF별 고유 테이블 최대 수 (없으면 전부 수집)")
    p.add_argument("--min_size", type=int, default=256, help="최소 해상도 필터")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout(sec)")
    p.add_argument("--jpeglib_check", action="store_true", help="jpeglib 안전검사 활성화(속도↓)")
    return p.parse_args()

def main():
    args = parse_args()
    qf_min, qf_max = parse_qf_range(args.qf_range)

    pair_counters, total_seen, (qf_min, qf_max) = collect_unique_qtables_from_urls(
        tsv_path=args.tsv_path,
        qf_min=qf_min,
        qf_max=qf_max,
        num_workers=args.num_workers,
        max_urls=args.max_urls,
        per_qf_limit=args.per_qf_limit,
        min_size=args.min_size,
        timeout=args.timeout,
        jpeglib_check=args.jpeglib_check,
    )

    print("\n=== Summary ===")
    print(f"Total URLs seen: {total_seen}")
    print(f"QF range       : {qf_min}~{qf_max} (inclusive)")
    for qf in sorted(pair_counters.keys()):
        uniq = len(pair_counters[qf])
        cnt_total = sum(pair_counters[qf].values())
        print(f"QF {qf}: unique={uniq}, samples={cnt_total}")

    ranked = build_ranked_entries_from_counters(pair_counters)
    save_pickle(ranked, args.save_pickle)

if __name__ == "__main__":
    main()
