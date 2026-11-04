import os, tempfile
import numpy as np
import cv2
import jpeglib
import random
from typing import List, Tuple

def _read_qtables_sampling(dct_obj):
    qt = getattr(dct_obj, 'qt', None)
    if qt is None:
        qt = getattr(dct_obj, 'quantization_tables', None)

    quant_tbl_no = getattr(dct_obj, 'quant_tbl_no', None)
    if quant_tbl_no is None:
        quant_tbl_no = getattr(dct_obj, 'qtbl_mapping', None)

    samp_factor = getattr(dct_obj, 'samp_factor', None)
    if samp_factor is None:
        samp_factor = getattr(dct_obj, 'sampling_factors', None)

    if qt is None or quant_tbl_no is None or samp_factor is None:
        raise RuntimeError("Failed to read QTables / quant_tbl_no / sampling from jpeglib DCT object.")
    return qt, quant_tbl_no, samp_factor

def _read_rgb_pixels(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        from PIL import Image
        rgb = np.array(Image.open(path).convert('RGB'))
        return rgb

    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif bgr.shape[2] == 4:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    else:
        from PIL import Image
        rgb = np.array(Image.open(path).convert('RGB'))

    return rgb

def _detect_orient(spatial, jpeg):
    H, W = spatial.shape[:2]
    Hb, Wb = jpeg.Y.shape[:2]
    H_vh, W_vh = Hb * 8, Wb * 8
    diff_vh = abs(H - H_vh) + abs(W - W_vh)
    H_hv, W_hv = Wb * 8, Hb * 8
    diff_hv = abs(H - H_hv) + abs(W - W_hv)
    return "vh" if diff_vh <= diff_hv else "hv"


def _calibration(path: str):
    dct1 = jpeglib.read_dct(path)
    qt, quant_tbl_no, samp_factor = _read_qtables_sampling(dct1)

    rgb = _read_rgb_pixels(path)
    tmp_path = None
    try:
        sp2 = jpeglib.from_spatial(rgb)
        sp2.samp_factor = samp_factor
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        sp2.write_spatial(tmp_path, qt=qt, quant_tbl_no=quant_tbl_no)
        dct2 = jpeglib.read_dct(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

    diffs = {}
    for ch in ["Y", "Cb", "Cr"]:
        c1 = getattr(dct1, ch).astype(np.float32)
        c2 = getattr(dct2, ch).astype(np.float32)
        diff = np.abs(c1 - c2)
        diff[..., 0] = 0
        diffs[ch] = diff

    orient = _detect_orient(rgb, dct1)
    return diffs, samp_factor, orient


def extract_patch_coords(args, path) -> List[Tuple[int,int,int,int]]:
    patch_size = getattr(args, 'patch_size', 256)
    tokens = args.roi_method.split('_')
    if len(tokens) < 2:
        raise ValueError("Invalid roi_method format")
    mode_tokens = tokens[1:]

    diffs, samp_factor, orient = _calibration(path)
    H, W, _ = _read_rgb_pixels(path).shape

    stride = max(8, (patch_size//2 // 8) * 8)
    tail_y = ((H - patch_size) // 8) * 8
    tail_x = ((W - patch_size) // 8) * 8

    y0_list = list(range(0, tail_y+1, stride))
    if y0_list[-1] != tail_y: y0_list.append(tail_y)
    x0_list = list(range(0, tail_x+1, stride))
    if x0_list[-1] != tail_x: x0_list.append(tail_x)

    ch2idx = {"Y":0, "Cb":1, "Cr":2}

    def block_range(ch, y0, x0, patch_size):
        c = ch2idx[ch]
        if orient == "vh":
            vY,hY = samp_factor[0]
            vC,hC = samp_factor[c]
        else:
            hY,vY = samp_factor[0]
            hC,vC = samp_factor[c]

        step_h = 8 * (hY // max(1,hC))
        step_v = 8 * (vY // max(1,vC))

        by0 = y0 // step_v
        bx0 = x0 // step_h
        by1 = (y0+patch_size) // step_v
        bx1 = (x0+patch_size) // step_h
        return by0, by1, bx0, bx1

    window_scores, coords = [], []
    for y0 in y0_list:
        for x0 in x0_list:
            score = 0.0
            for ch in ["Y","Cb","Cr"]:
                diff = diffs[ch]
                by0, by1, bx0, bx1 = block_range(ch, y0, x0, patch_size)
                sub = diff[by0:by1, bx0:bx1, :]

                score += float(np.count_nonzero(sub))
            window_scores.append(score)
            coords.append((x0,y0,x0+patch_size,y0+patch_size))

    idx_sorted = np.argsort(window_scores)
    n = len(coords)

    selected, used = [], set()
    need_total, i = 0, 0
    while i < len(mode_tokens):
        keyword = mode_tokens[i]
        k = int(mode_tokens[i+1]); need_total += k

        if keyword == "top":
            cand = [coords[idx_sorted[-1-j]] for j in range(min(k,n))]
        elif keyword == "bottom":
            cand = [coords[idx_sorted[j]] for j in range(min(k,n))]
        elif keyword == "median":
            mid = n//2
            left = max(0, mid-(k//2))
            right = min(n, left+k)
            cand = [coords[idx_sorted[j]] for j in range(left,right)]
        else:
            raise ValueError(f"Unknown ROI keyword: {keyword}")

        for c in cand:
            if c not in used:
                selected.append(c); used.add(c)
        i += 2

    while len(selected) < need_total:
        selected.append(coords[random.choice(idx_sorted)])
    return selected[:need_total]
