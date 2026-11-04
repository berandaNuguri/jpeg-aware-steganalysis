from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_dummy_jpeg(path, height=512, width=512, quality=95, seed=42):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to write dummy JPEG: {path}")
    return image


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_method_calls(patch_size, num_rois):
    try:
        from jpeg_aware_steganalysis.utils.data_processing.roi import calibration_roi, inconsist_roi, nzac_roi, random_roi
    except ModuleNotFoundError as exc:
        if exc.name == "jpeglib":
            raise RuntimeError(
                "jpeglib is required for nzAC/Inconsist/Calibration ROI timing. "
                "Run this script in the same Python environment used for train/test."
            ) from exc
        raise

    return {
        "Random": {
            "args": SimpleNamespace(roi_method=f"rand_{num_rois}", patch_size=patch_size),
            "call": lambda args, img, path: random_roi.extract_patch_coords(args, img, residual="random"),
        },
        "nzAC": {
            "args": SimpleNamespace(roi_method=f"nzac_top_{num_rois}", patch_size=patch_size),
            "call": lambda args, img, path: nzac_roi.extract_patch_coords(args, img, str(path)),
        },
        "Inconsist": {
            "args": SimpleNamespace(roi_method=f"inconsist_top_{num_rois}", patch_size=patch_size),
            "call": lambda args, img, path: inconsist_roi.extract_patch_coords(args, img, str(path)),
        },
        "Calibration": {
            "args": SimpleNamespace(roi_method=f"calibration_top_{num_rois}", patch_size=patch_size),
            "call": lambda args, img, path: calibration_roi.extract_patch_coords(args, str(path)),
        },
    }


def time_one_method(name, spec, img, path, warmup, repeats):
    for _ in range(warmup):
        coords = spec["call"](spec["args"], img, path)
        if not coords:
            raise RuntimeError(f"{name} returned no ROI coordinates during warmup.")

    times_ms = []
    last_coords = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_coords = spec["call"](spec["args"], img, path)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "method": name,
        "num_rois": len(last_coords) if last_coords is not None else 0,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "warmup": warmup,
        "repeats": repeats,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark runtime of Random, nzAC, Inconsist, and Calibration ROI extraction."
    )
    parser.add_argument("--image", type=str, default=None, help="Optional JPEG image path. If omitted, a dummy JPEG is generated.")
    parser.add_argument("--height", type=int, default=512, help="Dummy image height.")
    parser.add_argument("--width", type=int, default=512, help="Dummy image width.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality used for the dummy image.")
    parser.add_argument("--patch-size", type=int, default=256, help="ROI patch size.")
    parser.add_argument("--num-rois", type=int, default=1, help="Number of ROI patches to extract per call.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup calls per method.")
    parser.add_argument("--repeats", type=int, default=100, help="Timed calls per method.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dummy image and random ROI.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./outputs/roi_extraction_time_benchmark.csv",
        help="CSV path for benchmark summary.",
    )
    parser.add_argument(
        "--dummy-path",
        type=str,
        default="./outputs/roi_runtime_dummy_512.jpg",
        help="Path used when generating a dummy JPEG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.patch_size % 8 != 0:
        raise ValueError("--patch-size must be divisible by 8.")

    if args.image is None:
        image_path = Path(args.dummy_path)
        make_dummy_jpeg(image_path, height=args.height, width=args.width, quality=args.quality, seed=args.seed)
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    img = load_rgb(image_path)
    h, w = img.shape[:2]
    if h < args.patch_size or w < args.patch_size:
        raise ValueError(f"Image size {w}x{h} is smaller than patch size {args.patch_size}.")

    method_specs = build_method_calls(args.patch_size, args.num_rois)
    rows = []
    for name, spec in method_specs.items():
        row = time_one_method(name, spec, img, image_path, args.warmup, args.repeats)
        row.update(
            {
                "image_path": str(image_path),
                "image_width": w,
                "image_height": h,
                "patch_size": args.patch_size,
            }
        )
        rows.append(row)
        print(
            f"{name:11s} | mean {row['mean_ms']:9.3f} ms | "
            f"std {row['std_ms']:8.3f} | median {row['median_ms']:9.3f} | "
            f"rois {row['num_rois']}"
        )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
