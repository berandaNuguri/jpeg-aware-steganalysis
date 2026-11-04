import time
import numpy as np
import pywt
import torch
import torch.nn.functional as F
import cv2
import torch.nn as nn
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
from torchvision.ops import nms

def probmap(image, mode='haar', device='cuda'):
    H, W, C = image.shape
    wavelet = pywt.Wavelet(mode)
    lo = torch.tensor(wavelet.dec_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(wavelet.dec_hi, dtype=torch.float32, device=device)
    pad_size = len(lo) - 1
    padded = np.pad(image, ((0, pad_size), (0, pad_size), (0, 0)), mode='reflect')
    padded = torch.tensor(padded, dtype=torch.float32, device=device)

    kernel_LH = torch.outer(lo, hi).unsqueeze(0).unsqueeze(0)
    kernel_HL = torch.outer(hi, lo).unsqueeze(0).unsqueeze(0)
    kernel_HH = torch.outer(hi, hi).unsqueeze(0).unsqueeze(0)

    padded_t = padded.permute(2, 0, 1).unsqueeze(0)

    kernel_LH = kernel_LH.repeat(C, 1, 1, 1)
    kernel_HL = kernel_HL.repeat(C, 1, 1, 1)
    kernel_HH = kernel_HH.repeat(C, 1, 1, 1)

    conv_LH = F.conv2d(padded_t, kernel_LH, padding=0, groups=C)
    conv_HL = F.conv2d(padded_t, kernel_HL, padding=0, groups=C)
    conv_HH = F.conv2d(padded_t, kernel_HH, padding=0, groups=C)

    prob_map = torch.abs(conv_LH) + torch.abs(conv_HL) + torch.abs(conv_HH)
    prob_map = prob_map.squeeze(0).permute(1, 2, 0)
    
    return prob_map

def extract_patch_coords(args, img, device='cuda'):
    patch_size  = getattr(args, 'patch_size', 256)
    stride      = patch_size // 2
    with_center = getattr(args, 'with_center', False)
    
    if len(args.roi_method.split('_')) == 3:
        patch_num = int(args.roi_method.split('_')[-1])
        mode = args.roi_method.split('_')[-2]
    else:
        raise ValueError(f"Invalid structure: {args.roi_method} should be in the form of 'wavelet_<mode>'")
    
    # 1) 에너지 맵 계산
    with torch.no_grad():
        prob_map = probmap(img, mode=mode, device=device)
        energy = torch.sum(prob_map, dim=2).squeeze(0).squeeze(0)
    del prob_map
    torch.cuda.empty_cache()

    H, W = energy.shape

    # 2) 모든 후보 윈도우 생성
    y0_list = list(range(0, H - patch_size + 1, stride))
    x0_list = list(range(0, W - patch_size + 1, stride))
    if (H - patch_size) % stride: y0_list.append(H - patch_size)
    if (W - patch_size) % stride: x0_list.append(W - patch_size)

    windows, coords = [], []
    for y0 in y0_list:
        for x0 in x0_list:
            windows.append(energy[y0:y0+patch_size, x0:x0+patch_size].mean())
            coords.append((x0, y0, x0+patch_size, y0+patch_size))
    windows = torch.tensor(windows, device=device)
    order = torch.argsort(windows, descending=True).tolist()

    # 4) 선택
    selected = []
    if with_center:
        cx, cy = W / 2, H / 2
        x1 = max(0, min(cx - patch_size/2, W - patch_size))
        y1 = max(0, min(cy - patch_size/2, H - patch_size))
        selected.append((int(x1), int(y1),
                         int(x1+patch_size), int(y1+patch_size)))

    need = patch_num - len(selected)
    for i in range(need):
        idx = order[i % len(order)]
        selected.append(coords[idx])

    return selected