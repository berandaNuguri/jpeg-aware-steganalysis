import os, random, numpy as np, jpeglib

# ----------------------------- helpers -----------------------------
def _map_ywin_to_chwin(xb, yb, pw, ph, src_wb, src_hb, dst_wb, dst_hb):
    cx0 = (xb * dst_wb) // src_wb
    cy0 = (yb * dst_hb) // src_hb

    cx1 = ((xb + pw) * dst_wb + src_wb - 1) // src_wb
    cy1 = ((yb + ph) * dst_hb + src_hb - 1) // src_hb

    cx0 = max(0, min(cx0, dst_wb))
    cy0 = max(0, min(cy0, dst_hb))
    cx1 = max(cx0 + 1, min(cx1, dst_wb))
    cy1 = max(cy0 + 1, min(cy1, dst_hb))

    return cx0, cy0, (cx1 - cx0), (cy1 - cy0)

def _detect_orient(spatial, jpeg):
    H, W = spatial.shape[:2]
    Hb, Wb = jpeg.Y.shape[:2]
    H_vh, W_vh = Hb * 8, Wb * 8
    diff_vh = abs(H - H_vh) + abs(W - W_vh)
    H_hv, W_hv = Wb * 8, Hb * 8
    diff_hv = abs(H - H_hv) + abs(W - W_hv)
    return "vh" if diff_vh <= diff_hv else "hv"

def _get_qtables(dct_obj):
    qt = getattr(dct_obj, 'qt', None)
    if qt is None:
        raise AttributeError("dct_obj.qt not found")

    if isinstance(qt, np.ndarray):
        if qt.ndim == 2 and qt.shape == (8, 8):
            q = qt.astype(np.float32); return [q, q, q]
        if qt.ndim == 3 and qt.shape[1:] == (8, 8):
            n = qt.shape[0]
            if n >= 3: qY, qCb, qCr = qt[0], qt[1], qt[2]
            elif n == 2: qY, qC = qt[0], qt[1]; qCb, qCr = qC, qC
            elif n == 1: qY = qCb = qCr = qt[0]
            else: raise ValueError(f"Unsupported qt shape: {qt.shape}")
            return [qY.astype(np.float32), qCb.astype(np.float32), qCr.astype(np.float32)]
        raise ValueError(f"Unsupported qt shape: {qt.shape}")

    if isinstance(qt, (list, tuple)):
        mats = [np.array(x, dtype=np.float32) for x in qt]
        if len(mats) == 0: raise ValueError("Empty qt list/tuple")
        if len(mats) == 1: return [mats[0], mats[0], mats[0]]
        if len(mats) == 2: return [mats[0], mats[1], mats[1]]
        return [mats[0], mats[1], mats[2]]

    if isinstance(qt, dict):
        qY  = qt.get('Y')  or qt.get('y')  or qt.get('0')
        qCb = qt.get('Cb') or qt.get('cb') or qt.get('1')
        qCr = qt.get('Cr') or qt.get('cr') or qt.get('2')
        if qY is not None and qCb is not None and qCr is not None:
            return [np.array(qY,  dtype=np.float32),
                    np.array(qCb, dtype=np.float32),
                    np.array(qCr, dtype=np.float32)]
        if len(qt) == 0: raise ValueError("Empty qt dict")
        any_q = np.array(next(iter(qt.values())), dtype=np.float32)
        return [any_q, any_q, any_q]

    raise TypeError(f"Unsupported qt type: {type(qt)}")

def _dqdct_channel(qdct_blocks, qtable):
    qtable = qtable.astype(np.float32)
    return qdct_blocks.astype(np.float32) * qtable[None, None, :, :]

def _features_from_dqdct(dqdct_blocks):
    hb, wb = dqdct_blocks.shape[:2]
    feats = np.zeros((hb, wb, 3), dtype=np.float32)
    ac_mask = np.ones((8, 8), dtype=bool); ac_mask[0, 0] = False
    ac = dqdct_blocks[..., ac_mask]                
    abs_ac = np.abs(ac)
    feats[..., 0] = abs_ac.mean(axis=-1)          
    feats[..., 1] = ac.var(axis=-1)                
    feats[..., 2] = (np.count_nonzero(ac, axis=-1) / ac.shape[-1]).astype(np.float32)  
    return feats

# ----------------------------- main ROI -----------------------------
def extract_patch_coords(args, img, path):
    patch_size = int(getattr(args, 'patch_size', 256))
    assert patch_size % 8 == 0, "patch_size must be multiple of 8"
    ph, pw = patch_size // 8, patch_size // 8 
    roi_stride_px = patch_size // 2
    sh = max(1, roi_stride_px // 8)
    sw = sh

    tokens = args.roi_method.split('_')
    if len(tokens) < 2:
        raise ValueError(
            f"Invalid structure: {args.roi_method} "
            "should be like 'dq_top_2' or 'nzac_top_2'; only tokens after the first are used."
        )
    mode_tokens = tokens[1:]

    def _window_inconsistency(feats_hw3, x0, y0, w, h):
        f = feats_hw3[y0:y0+h, x0:x0+w, :]
        mu = f.mean(axis=(0,1), dtype=np.float32)           
        diff = f - mu                                       
        score = np.mean(np.sum(diff * diff, axis=-1))        

        return float(score)

    H, W = img.shape[:2]
    dct_obj = jpeglib.read_dct(path)
    coeffs_list = [dct_obj.Y, dct_obj.Cb, dct_obj.Cr]       
    qtables = _get_qtables(dct_obj)                         
    orient = _detect_orient(img, dct_obj)

    ch_feats = []
    for ch_idx, arr in enumerate(coeffs_list): 
        dqdct = _dqdct_channel(arr, qtables[ch_idx])
        feats = _features_from_dqdct(dqdct)
        ch_feats.append(feats)

    coords = []
    window_scores = []

    hbY, wbY = ch_feats[0].shape[:2]
    hbCb, wbCb = ch_feats[1].shape[:2]
    hbCr, wbCr = ch_feats[2].shape[:2]

    W8 = W // 8
    H8 = H // 8
    if orient == "vh":
        max_xb = min(wbY - pw, W8 - pw)
        max_yb = min(hbY - ph, H8 - ph)
    else:  
        max_xb = min(wbY - pw, H8 - pw) 
        max_yb = min(hbY - ph, W8 - ph)

    for yb in range(0, max_yb + 1, sh):
        for xb in range(0, max_xb + 1, sw):
            # Y Inconsistency
            sY = _window_inconsistency(ch_feats[0], xb, yb, pw, ph)

            # Cb Inconsistency
            cx0, cy0, cw, ch = _map_ywin_to_chwin(xb, yb, pw, ph, wbY, hbY, wbCb, hbCb)
            sCb = _window_inconsistency(ch_feats[1], cx0, cy0, cw, ch)

            # Cr Inconsistency
            rx0, ry0, rw, rh = _map_ywin_to_chwin(xb, yb, pw, ph, wbY, hbY, wbCr, hbCr)
            sCr = _window_inconsistency(ch_feats[2], rx0, ry0, rw, rh)

            score = (sY + sCb + sCr) / 3.0

            if orient == "vh":
                x0 = xb * 8; y0 = yb * 8
                x1 = x0 + pw * 8; y1 = y0 + ph * 8
            else:  # hv
                x0 = yb * 8; y0 = xb * 8
                x1 = x0 + ph * 8; y1 = y0 + pw * 8

            coords.append((x0, y0, x1, y1))
            window_scores.append(score)

    if not coords:
        rx0_max = max(0, (W - patch_size) // 8) * 8
        ry0_max = max(0, (H - patch_size) // 8) * 8
        rx0 = (random.randint(0, rx0_max // 8) * 8) if rx0_max > 0 else 0
        ry0 = (random.randint(0, ry0_max // 8) * 8) if ry0_max > 0 else 0
        return [(rx0, ry0, rx0 + patch_size, ry0 + patch_size)]

    idx_sorted = np.argsort(window_scores) 
    n = len(coords)

    selected = []
    i, total = 0, 0
    while i < len(mode_tokens):
        keyword = mode_tokens[i]
        if i + 1 >= len(mode_tokens):
            raise ValueError(f"Invalid ROI method format near {keyword}")
        k = int(mode_tokens[i + 1]); total += k

        if keyword == "top":
            sel = [coords[idx_sorted[-j - 1]] for j in range(min(k, n))]
        elif keyword == "bottom":
            sel = [coords[idx_sorted[j]] for j in range(min(k, n))]
        elif keyword == "median":
            mid = n // 2
            left = max(0, mid - (k // 2)); right = min(n, left + k)
            sel = [coords[idx_sorted[j]] for j in range(left, right)]
        else:
            raise ValueError(f"Unknown ROI keyword: {keyword}")

        selected.extend(sel); i += 2

    while len(selected) < total:
        selected.append(coords[random.choice(idx_sorted)])

    return selected
