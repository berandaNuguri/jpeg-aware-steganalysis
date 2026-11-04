import os
import io
import sys
import pickle
import argparse
import requests
import numpy as np
import cv2
from PIL import Image, JpegImagePlugin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import subprocess
from collections import Counter
from typing import Iterable, Optional, Tuple, Dict, List

from datasets import load_dataset  # COYO700M 용
import tempfile
import jpeglib

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
    """
    jpeglib이 이 이미지를 안전하게 읽을 수 있는지 확인.
    (NamedTemporaryFile 사용: 멀티스레드에서도 안전하도록 수정)
    """
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=tmp_dir) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        jpeg = jpeglib.read_dct(tmp_path)
        _ = jpeg.qt[0]
        return True
    except Exception:
        return False
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

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
# Color Subsampling 추출
# ==============================
def extract_subsampling_from_bytes(content: bytes) -> Optional[str]:
    """
    Pillow(JPEGImagePlugin.get_sampling)를 사용해 subsampling 정보를 읽고
    4:4:4 / 4:2:2 / 4:2:0 / gray / 기타 패턴으로 문자열 라벨을 반환.
    실패 시 None
    """
    try:
        img = Image.open(io.BytesIO(content))
        img.load()  # 실제 디코딩

        # JPEG가 아니면 패스
        if img.format != "JPEG":
            return None

        # Pillow가 제공하는 sampling 코드 (-1, 0, 1, 2, ...)
        code = JpegImagePlugin.get_sampling(img)

        # 그레이스케일 처리
        if img.mode in ("L", "LA") or code < 0:
            return "gray"

        # Pillow의 sampling 코드 → 우리가 쓰던 레이블로 매핑
        #   0 -> 4:4:4
        #   1 -> 4:2:2
        #   2 -> 4:2:0
        mapping = {
            0: "444",
            1: "422",
            2: "420",
        }
        if code in mapping:
            return mapping[code]

        # 알려지지 않은 패턴은 코드 그대로 남김
        return f"sampling_code_{code}"

    except Exception:
        return None
# ==============================
# 공용: URL 하나 처리 (CC12M / COYO 공용)
# ==============================
def process_url_for_qtable(
    url: str,
    min_size: int = 256,
    timeout: float = 10.0,
    jpeglib_check: bool = False,
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...], int, Optional[str]]]:
    """
    URL을 받아 JPEG 검증 후 (Y64, C64, qf_est, subsampling_label) 반환. 실패 시 None
    subsampling_label 은 "444", "422", "420", "gray", "Y(...)" 등 문자열
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
        qtable_out = extract_qtables_from_bytes(content)
        if qtable_out is None:
            return None
        y_tbl, c_tbl, qf = qtable_out

        # Color Subsampling 추출
        subsamp = extract_subsampling_from_bytes(content)

        return y_tbl, c_tbl, qf, subsamp
    except Exception:
        return None


# ==============================
# CC12M: TSV 기반 수집
# ==============================
def collect_unique_qtables_from_cc12m(
    tsv_path: str,
    qf_min: int,
    qf_max: int,
    num_workers: int = 32,
    max_urls: Optional[int] = None,   # ← 이제 "유효 샘플 목표 개수"로 사용
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

    # 목표 유효 샘플 수
    target_success = max_urls if max_urls is not None else len(urls)

    # QF 범위 처리 (1~100로 클램프, 오름차순 정렬)
    qf_min = max(1, min(100, int(qf_min)))
    qf_max = max(1, min(100, int(qf_max)))
    if qf_min > qf_max:
        qf_min, qf_max = qf_max, qf_min
    target_qfs_set = set(range(qf_min, qf_max + 1))

    pair_counters: Dict[int, Counter] = {qf: Counter() for qf in sorted(target_qfs_set)}
    subsamp_counter: Counter = Counter()

    success = 0          # QTable+Subsampling까지 모두 성공한 이미지 수
    idx = 0              # 다음으로 볼 URL 인덱스
    n_urls = len(urls)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures: Dict = {}

        # 초기 워커 채우기
        def schedule_more():
            nonlocal idx, futures, success
            while (
                len(futures) < num_workers
                and idx < n_urls
                and success < target_success
            ):
                url = urls[idx]
                idx += 1
                fut = executor.submit(
                    process_url_for_qtable, url, min_size, timeout, jpeglib_check
                )
                futures[fut] = url

        schedule_more()

        with tqdm(total=target_success, desc=f"[CC12M] 스캔 중(QF {qf_min}~{qf_max})") as pbar:
            while futures and success < target_success:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    url = futures.pop(fut)
                    res = fut.result()

                    if res is not None:
                        y_tbl, c_tbl, qf_est, subsamp = res

                        # QF 범위 내 + subsampling 성공한 경우만 "유효"로 카운트
                        if qf_est in target_qfs_set and subsamp is not None:
                            pair = (tuple(y_tbl), tuple(c_tbl))
                            pair_counters[qf_est][pair] += 1
                            subsamp_counter[subsamp] += 1

                            success += 1
                            pbar.update(1)

                            if success >= target_success:
                                # 남은 작업들 취소 후 종료
                                for f in futures:
                                    f.cancel()
                                return pair_counters, subsamp_counter, success, (qf_min, qf_max)

                # 더 돌릴 수 있으면 새 작업 스케줄
                schedule_more()

    # 목표만큼 못 채웠어도 현재까지 수집한 값 반환
    return pair_counters, subsamp_counter, success, (qf_min, qf_max)


# ==============================
# COYO700M: streaming 기반 수집
# ==============================
def passes_filters_coyo(example: dict) -> bool:
    """
    COYO 예제 필터: 해상도, 확장자, NSFW 스코어 등
    """
    try:
        w, h = example["width"], example["height"]
        url = example["url"]
        nsfw1 = example["nsfw_score_opennsfw2"]
        nsfw2 = example["nsfw_score_gantman"]
    except KeyError:
        return False

    return (
        w >= 256 and h >= 256 and
        (url.endswith(".jpg") or url.endswith(".jpeg")) and
        nsfw1 < 0.1 and nsfw2 < 0.1
    )


def collect_unique_qtables_from_coyo(
    qf_min: int,
    qf_max: int,
    num_workers: int = 32,
    max_images: int = 500000,   # ← "유효 샘플 목표 개수"
    per_qf_limit: Optional[int] = None,
    min_size: int = 256,
    timeout: float = 10.0,
    jpeglib_check: bool = False,
):
    """
    COYO700M streaming에서
    QTable + Subsampling 추출이 모두 성공하고 QF 범위에 속하는 이미지를
    최대 max_images(유효 샘플 수)까지 수집.
    """
    # QF 범위 처리
    qf_min = max(1, min(100, int(qf_min)))
    qf_max = max(1, min(100, int(qf_max)))
    if qf_min > qf_max:
        qf_min, qf_max = qf_max, qf_min
    target_qfs_set = set(range(qf_min, qf_max + 1))

    pair_counters: Dict[int, Counter] = {qf: Counter() for qf in sorted(target_qfs_set)}
    subsamp_counter: Counter = Counter()

    ds = load_dataset("kakaobrain/coyo-700m", split="train", streaming=True)
    it = iter(ds)

    success = 0   # QTable + Subsampling 모두 성공한 이미지 수

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures: Dict = {}

        def schedule_more():
            nonlocal futures, success
            while len(futures) < num_workers and success < max_images:
                try:
                    ex = next(it)
                except StopIteration:
                    break
                if not passes_filters_coyo(ex):
                    continue
                url = ex["url"]
                fut = executor.submit(
                    process_url_for_qtable, url, min_size, timeout, jpeglib_check
                )
                futures[fut] = url

        schedule_more()

        with tqdm(total=max_images, desc=f"[COYO] 스캔 중(QF {qf_min}~{qf_max})") as pbar:
            while futures and success < max_images:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    url = futures.pop(fut)
                    res = fut.result()

                    if res is not None:
                        y_tbl, c_tbl, qf_est, subsamp = res
                        if qf_est in target_qfs_set and subsamp is not None:
                            pair = (tuple(y_tbl), tuple(c_tbl))
                            pair_counters[qf_est][pair] += 1
                            subsamp_counter[subsamp] += 1

                            success += 1
                            pbar.update(1)

                            # per_qf_limit은 여전히 QTable 기준으로만 제어 (옵션)
                            if per_qf_limit is not None:
                                done_all = True
                                for qf in pair_counters:
                                    if len(pair_counters[qf]) < per_qf_limit:
                                        done_all = False
                                        break
                                if done_all or success >= max_images:
                                    for f in futures:
                                        f.cancel()
                                    return pair_counters, subsamp_counter, success, (qf_min, qf_max)

                schedule_more()

    return pair_counters, subsamp_counter, success, (qf_min, qf_max)

# ==============================
# 공용: Counter → ranked dict → pickle
# ==============================
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


def merge_qtable_counters(
    dst: Dict[int, Counter],
    src: Dict[int, Counter],
) -> Dict[int, Counter]:
    """
    QF별 Counter 딕셔너리 두 개를 합침.
    dst[qf][pair] += src[qf][pair]
    """
    for qf, cnt in src.items():
        if qf not in dst:
            dst[qf] = Counter()
        dst[qf].update(cnt)
    return dst


def merge_subsampling_counters(
    dst: Counter,
    src: Counter,
) -> Counter:
    dst.update(src)
    return dst


def save_subsampling_stats(counter: Counter, out_path: str):
    """
    Subsampling 통계를 pickle로 저장.
    포맷:
    {
        "total": int,
        "counts": { "444": int, "420": int, ... }
    }
    """
    total = int(sum(counter.values()))
    counts = {k: int(v) for k, v in counter.items()}
    obj = {"total": total, "counts": counts}
    save_pickle(obj, out_path)


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
    p = argparse.ArgumentParser(description="CC12M / COYO700M에서 QF 범위의 고유 QTable(Y,C)과 Color Subsampling 통계를 수집하여 pickle로 저장")
    p.add_argument("--mode", type=str, choices=["cc12m", "coyo", "both"], default="both",
                   help="어떤 소스에서 추출할지 선택")
    # CC12M 관련
    p.add_argument("--cc12m_tsv_path", type=str, default="./data/CC12M/cc12m.tsv", help="첫 컬럼이 URL인 TSV 경로")

    p.add_argument("--cc12m_max_urls", type=int, default=1500000,
                   help="CC12M에서 수집할 '유효' 이미지 수(QTable+Subsampling 성공 기준)")
    p.add_argument("--coyo_max_images", type=int, default=1500000,
                   help="COYO700M에서 수집할 '유효' 이미지 수(QTable+Subsampling 성공 기준)")

    # 공통
    p.add_argument("--qf_range", type=str, default="1,100", help="포함 범위: 'min,max' 또는 '[min,max]' (예: 75,100)")
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--per_qf_limit", type=int, default=None, help="QF별 고유 테이블 최대 수 (없으면 전부 수집)")
    p.add_argument("--min_size", type=int, default=256, help="최소 해상도 필터")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout(sec)")
    p.add_argument("--jpeglib_check", action="store_true", help="jpeglib 안전검사 활성화(속도↓)")

    p.add_argument("--save_pickle", type=str, default="./qtable_info_cc12m_coyo.pkl",
                   help="CC12M+COYO 통합 QTable pickle 경로")
    p.add_argument("--save_subsampling", type=str, default="./subsampling_info_cc12m_coyo.pkl",
                   help="CC12M+COYO 통합 Color Subsampling pickle 경로")
    return p.parse_args()


def main():
    args = parse_args()
    qf_min, qf_max = parse_qf_range(args.qf_range)

    combined_counters: Dict[int, Counter] = {}
    combined_subsampling: Counter = Counter()

    # CC12M
    if args.mode in ("cc12m", "both"):
        print("\n===== CC12M 수집 시작 =====")
        cc12m_counters, cc12m_subs, cc12m_valid, (qm1, qM1) = collect_unique_qtables_from_cc12m(
            tsv_path=args.cc12m_tsv_path,
            qf_min=qf_min,
            qf_max=qf_max,
            num_workers=args.num_workers,
            max_urls=args.cc12m_max_urls,
            per_qf_limit=args.per_qf_limit,
            min_size=args.min_size,
            timeout=args.timeout,
            jpeglib_check=args.jpeglib_check,
        )
        print("\n[CC12M] QTable Summary")
        print(f"Total VALID images (QTable+Subsampling): {cc12m_valid}")
        print(f"QF range            : {qm1}~{qM1} (inclusive)")
        for qf in sorted(cc12m_counters.keys()):
            uniq = len(cc12m_counters[qf])
            cnt_total = sum(cc12m_counters[qf].values())
            print(f"QF {qf}: unique={uniq}, samples={cnt_total}")

        print("\n[CC12M] Subsampling Summary")
        total_cc12m_subs = sum(cc12m_subs.values())
        for label, cnt in cc12m_subs.most_common():
            print(f"{label}: {cnt} ({cnt/total_cc12m_subs:.4f})")
        if total_cc12m_subs == 0:
            print("No subsampling info collected from CC12M.")

        combined_counters = merge_qtable_counters(combined_counters, cc12m_counters)
        combined_subsampling = merge_subsampling_counters(combined_subsampling, cc12m_subs)

    # COYO700M
    if args.mode in ("coyo", "both"):
        print("\n===== COYO700M 수집 시작 =====")
        coyo_counters, coyo_subs, coyo_valid, (qm2, qM2) = collect_unique_qtables_from_coyo(
            qf_min=qf_min,
            qf_max=qf_max,
            num_workers=args.num_workers,
            max_images=args.coyo_max_images,
            per_qf_limit=args.per_qf_limit,
            min_size=args.min_size,
            timeout=args.timeout,
            jpeglib_check=args.jpeglib_check,
        )
        print("\n[COYO] QTable Summary")
        print(f"Total VALID images (QTable+Subsampling): {coyo_valid}")
        print(f"QF range            : {qm2}~{qM2} (inclusive)")
        for qf in sorted(coyo_counters.keys()):
            uniq = len(coyo_counters[qf])
            cnt_total = sum(coyo_counters[qf].values())
            print(f"QF {qf}: unique={uniq}, samples={cnt_total}")

        print("\n[COYO] Subsampling Summary")
        total_coyo_subs = sum(coyo_subs.values())
        for label, cnt in coyo_subs.most_common():
            print(f"{label}: {cnt} ({cnt/total_coyo_subs:.4f})")
        if total_coyo_subs == 0:
            print("No subsampling info collected from COYO.")

        combined_counters = merge_qtable_counters(combined_counters, coyo_counters)
        combined_subsampling = merge_subsampling_counters(combined_subsampling, coyo_subs)

    if not combined_counters:
        print("[WARN] 수집된 QTable 통계가 없습니다. pickle을 저장하지 않습니다.")
    else:
        print("\n[Combined] QTable Summary (CC12M + COYO)")
        for qf in sorted(combined_counters.keys()):
            uniq = len(combined_counters[qf])
            cnt_total = sum(combined_counters[qf].values())
            print(f"QF {qf}: unique={uniq}, samples={cnt_total}")

        ranked = build_ranked_entries_from_counters(combined_counters)
        save_pickle(ranked, args.save_pickle)

    if combined_subsampling:
        print("\n[Combined] Subsampling Summary (CC12M + COYO)")
        total_subs = sum(combined_subsampling.values())
        for label, cnt in combined_subsampling.most_common():
            print(f"{label}: {cnt} ({cnt/total_subs:.4f})")

        save_subsampling_stats(combined_subsampling, args.save_subsampling)
    else:
        print("[WARN] 수집된 Subsampling 통계가 없습니다. subsampling pickle을 저장하지 않습니다.")


if __name__ == "__main__":
    main()
