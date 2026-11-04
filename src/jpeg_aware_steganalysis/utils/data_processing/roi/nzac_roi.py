import os, cv2, random, numpy as np, jpeglib

def _detect_orient(spatial, jpeg):
    H, W = spatial.shape[:2]
    Hb, Wb = jpeg.Y.shape[:2]

    H_vh, W_vh = Hb * 8, Wb * 8
    diff_vh = abs(H - H_vh) + abs(W - W_vh)

    H_hv, W_hv = Wb * 8, Hb * 8
    diff_hv = abs(H - H_hv) + abs(W - W_hv)

    return "vh" if diff_vh <= diff_hv else "hv"


def extract_patch_coords(args, img, path):
    patch_size = getattr(args, 'patch_size', 256)
    
    tokens = args.roi_method.split('_')
    if len(tokens) < 2:
        raise ValueError(
            f"Invalid structure: {args.roi_method} "
            "should be like 'nzac_top_2' or 'nzac_top_1_median_2'"
        )
    mode_tokens = tokens[1:]

    # 1) Image Shape
    H, W, _ = img.shape

    # 2) Read DCT
    dct_obj = jpeglib.read_dct(path)
    coeffs_list = [dct_obj.Y, dct_obj.Cb, dct_obj.Cr]
    samp_factor = dct_obj.samp_factor  # [(vY,hY),(vCb,hCb),(vCr,hCr)]

    # 3) Detect orientation
    orient = _detect_orient(img, dct_obj)

    # 4) Y-block grid Shape
    arrY = coeffs_list[0]
    hb, wb = arrY.shape[:2]
    block_counts = np.zeros((hb, wb), dtype=int)

    # 5) Calculate nzAC  + Y-grid Mapping
    for ch_idx, arr in enumerate(coeffs_list):  # 0=Y,1=Cb,2=Cr
        ch_hb, ch_wb = arr.shape[:2]

        if orient == "vh":
            vY, hY = samp_factor[0]
            vC, hC = samp_factor[ch_idx]
        else:  # hv
            hY, vY = samp_factor[0]
            hC, vC = samp_factor[ch_idx]

        # Calculate nzAC per block. This is equivalent to flattening each
        # 8x8 block, zeroing the DC coefficient, and counting non-zero ACs.
        ch_counts = np.count_nonzero(arr.reshape(ch_hb, ch_wb, -1)[..., 1:], axis=-1)

        # Map channel block counts to the Y block grid with the same index rule
        # as the original nested-loop implementation.
        y_idx = np.minimum((np.arange(hb) * vC) // vY, ch_hb - 1)
        x_idx = np.minimum((np.arange(wb) * hC) // hY, ch_wb - 1)
        block_counts += ch_counts[y_idx[:, None], x_idx[None, :]]
                
    # 6) Distribute Window (8-align enforced)
    align = 8
    stride = patch_size // 2
    stride = max(align, (stride // align) * align)   # stride도 8의 배수로

    tail_y = max(0, H - patch_size)
    tail_x = max(0, W - patch_size)

    # tail도 8-align (QDCT/DCT block crop에서 264 같은 케이스 방지)
    tail_y8 = (tail_y // align) * align
    tail_x8 = (tail_x // align) * align

    y0_list = sorted({(y0 // align) * align for y0 in range(0, tail_y8 + 1, stride)}) or [0]
    x0_list = sorted({(x0 // align) * align for x0 in range(0, tail_x8 + 1, stride)}) or [0]

    if y0_list[-1] != tail_y8:
        y0_list.append(tail_y8)
    if x0_list[-1] != tail_x8:
        x0_list.append(tail_x8)

    # 7) Scoring (coords are always patch_size, and 8-aligned)
    window_sums, coords = [], []
    ph = patch_size // align
    pw = patch_size // align

    max_y0 = tail_y8
    max_x0 = tail_x8

    for y0 in y0_list:
        for x0 in x0_list:
            # clamp + 8-align (double-safety)
            y0 = int(max(0, min(y0, max_y0)))
            x0 = int(max(0, min(x0, max_x0)))
            y0 = (y0 // align) * align
            x0 = (x0 // align) * align

            bi0, bj0 = y0 // align, x0 // align
            sub = block_counts[bi0:bi0 + ph, bj0:bj0 + pw]
            window_sums.append(int(sub.sum()))

            x2 = x0 + patch_size
            y2 = y0 + patch_size
            coords.append((x0, y0, x2, y2))


    # 8) ROI Selection
    idx_sorted = np.argsort(window_sums)
    n = len(coords)

    selected = []
    i, total = 0, 0
    while i < len(mode_tokens):
        keyword = mode_tokens[i]
        if i + 1 >= len(mode_tokens):
            raise ValueError(f"Invalid ROI method format near {keyword}")
        k = int(mode_tokens[i+1]); total += k

        if keyword == "top":
            sel = [coords[idx_sorted[-j-1]] for j in range(min(k, n))]
        elif keyword == "bottom":
            sel = [coords[idx_sorted[j]] for j in range(min(k, n))]
        elif keyword == "median":
            mid = n // 2
            left = max(0, mid - (k // 2))
            right = min(n, left + k)
            sel = [coords[idx_sorted[j]] for j in range(left, right)]
        else:
            raise ValueError(f"Unknown ROI keyword: {keyword}")

        selected.extend(sel)
        i += 2

    while len(selected) < total:
        selected.append(coords[random.choice(idx_sorted)])

    return selected
