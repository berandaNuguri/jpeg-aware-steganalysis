from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import sys
import argparse
import pickle
from collections import Counter

import numpy as np
import jpeglib


def iter_jpeg_paths(root: str, recursive: bool = True):
    exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(exts):
                    yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(root):
            if fn.endswith(exts):
                yield os.path.join(root, fn)


def flatten_qtable(qt) -> tuple:
    """8x8 또는 64길이 배열을 64 길이 튜플로 변환."""
    arr = np.array(qt, dtype=np.int32).reshape(-1)
    if arr.size != 64:
        raise ValueError(f"QTable size != 64: {arr.size}")
    return tuple(int(x) for x in arr)


def extract_yc_qtables(jpeg) -> tuple:
    """
    jpeglib.read_dct() 결과에서 (Y, C 리스트) 반환.
    - Y: qt[0]
    - C: qt[1], qt[2] 있으면 둘 다 C 리스트에 추가
    """
    qtables = jpeg.qt  # 보통 list of np.ndarray((8,8), dtype=int)
    # if not qtables:
    #     return None, None

    y_tbl = flatten_qtable(qtables[0])

    c_tables = []
    if len(qtables) >= 2:
        c_tables.append(flatten_qtable(qtables[1]))
    if len(qtables) >= 3:
        c_tables.append(flatten_qtable(qtables[2]))

    if not c_tables:
        # 색 테이블이 따로 없으면 Y를 그대로 C로도 사용
        c_tables.append(y_tbl)

    return y_tbl, c_tables  # C는 리스트(여러 개 있을 수 있음)


def scan_folder_for_qtables(root: str, recursive: bool = True):
    """
    폴더 내 JPEG들의 Unique QTable을 수집.
    반환:
      - y_counter: Y QTable -> count
      - c_counter: C QTable -> count
      - pair_counter: (Y, C) 페어 -> count  (C 여러 개이면 각각 페어로 카운트)
      - total_images: 처리한 JPEG 수
    """
    y_counter = Counter()
    c_counter = Counter()
    pair_counter = Counter()
    total_images = 0
    failed = 0

    for path in iter_jpeg_paths(root, recursive=recursive):
        try:
            jpeg = jpeglib.read_dct(path)
        except Exception:
            failed += 1
            continue

        y_tbl, c_tables = extract_yc_qtables(jpeg)
        if y_tbl is None:
            failed += 1
            continue

        total_images += 1
        y_counter[y_tbl] += 1
        for c_tbl in c_tables:
            c_counter[c_tbl] += 1
            pair_counter[(y_tbl, c_tbl)] += 1

    return y_counter, c_counter, pair_counter, total_images, failed


def save_pickle(obj, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved pickle to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="폴더 내 JPEG들의 Unique QTable(Y, C, (Y,C) 페어) 통계 확인"
    )
    parser.add_argument("--root", type=str, default=str(_checkout_data_path('ALASKA2/ALASKA_v2_JPG_256_QF75_COLOR/Cover/')),help="JPEG가 들어 있는 폴더 경로")
    parser.add_argument("--no_recursive", action="store_true",
                        help="하위 폴더는 탐색하지 않음")
    parser.add_argument("--save_pickle", type=str, default=None,
                        help="유니크 QTable 정보를 저장할 pickle 경로 (선택)")
    args = parser.parse_args()

    recursive = not args.no_recursive

    print(f"[INFO] Scanning folder: {args.root} (recursive={recursive})")
    y_counter, c_counter, pair_counter, total_images, failed = scan_folder_for_qtables(
        args.root, recursive=recursive
    )

    print("\n=== Summary ===")
    print(f"Total JPEG images processed : {total_images}")
    print(f"Failed to read / no qtables : {failed}")
    print(f"Unique Y  QTables           : {len(y_counter)}")
    print(f"Unique C  QTables           : {len(c_counter)}")
    print(f"Unique (Y,C) pairs          : {len(pair_counter)}")

    # 상위 몇 개 예시 출력 (선택)
    top_k = 5
    print(f"\nTop {top_k} Y QTables (by count):")
    for i, (qt, cnt) in enumerate(y_counter.most_common(top_k), start=1):
        print(f"  #{i} count={cnt}, first_row={qt[:8]}")

    print(f"\nTop {top_k} C QTables (by count):")
    for i, (qt, cnt) in enumerate(c_counter.most_common(top_k), start=1):
        print(f"  #{i} count={cnt}, first_row={qt[:8]}")

    if args.save_pickle:
        data = {
            "y_counter": y_counter,
            "c_counter": c_counter,
            "pair_counter": pair_counter,
            "total_images": total_images,
            "failed": failed,
        }
        save_pickle(data, args.save_pickle)


if __name__ == "__main__":
    main()
