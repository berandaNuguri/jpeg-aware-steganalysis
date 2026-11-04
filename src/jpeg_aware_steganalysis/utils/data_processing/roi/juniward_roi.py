import numpy as np
import torch
import torch.nn.functional as F
import pywt
from scipy.fftpack import idct
from PIL import Image

def probmap(cover: np.ndarray, qt: dict, mode: str = 'db8', device: str = 'cuda') -> torch.Tensor:
    qt0 = np.array(qt[next(iter(qt))], np.float32).reshape(8,8)
    img_t = torch.from_numpy(cover).to(device).permute(2,0,1).unsqueeze(0)
    H, W = cover.shape[:2]
    wav = pywt.Wavelet(mode)
    lo = torch.tensor(wav.dec_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(wav.dec_hi, dtype=torch.float32, device=device)
    F_LH = lo.unsqueeze(1) @ hi.unsqueeze(0)
    F_HL = hi.unsqueeze(1) @ lo.unsqueeze(0)
    F_HH = hi.unsqueeze(1) @ hi.unsqueeze(0)
    filters = torch.stack([F_LH, F_HL, F_HH], dim=0)
    pad = filters.shape[-1] - 1
    out_h = 8 + 2*pad - (filters.shape[-1] - 1)

    # spatial impact
    spatial_imp = torch.zeros(8,8,8,8,device=device)
    for i in range(8):
        for j in range(8):
            c = np.zeros((8,8),np.float32); c[i,j]=1
            blk = idct(idct(c.T, norm='ortho').T, norm='ortho') * qt0[i,j]
            spatial_imp[i,j] = torch.from_numpy(blk).to(device)

    # wavelet impact
    wavelet_imp = torch.zeros(3,8,8,out_h,out_h,device=device)
    for f in range(3):
        k = filters[f].unsqueeze(0).unsqueeze(0)
        for i in range(8):
            for j in range(8):
                inp = spatial_imp[i,j].unsqueeze(0).unsqueeze(0)
                wavelet_imp[f,i,j] = F.conv2d(inp, k, padding=pad)[0,0]

    # reference coeffs
    half = (out_h-1)//2
    img_pad = F.pad(img_t, (pad+half,)*4, mode='reflect')
    RC = []
    for f in range(3):
        k = filters[f].unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
        RC.append(F.conv2d(img_pad, k, groups=3))

    # rho→p
    rho = torch.zeros(H,W,device=device)
    sgm = 2**-6
    for r in range(H):
        for c in range(W):
            bi, bj = r%8, c%8
            r0, c0 = r+pad, c+pad
            cost = 0.0
            for f in range(3):
                WI = wavelet_imp[f,bi,bj]
                RCsub = RC[f][0,:,r0:r0+out_h,c0:c0+out_h]
                cost += torch.sum(torch.abs(WI).unsqueeze(0)/(torch.abs(RCsub)+sgm))
            rho[r,c] = cost
    p = 1.0/(rho + 1e-12)
    return p / torch.sum(p)

def extract_patch_coords(args, img, path, mu, var):
    patch_num   = getattr(args, 'patch_num', 2)
    patch_size  = getattr(args, 'patch_size', 256)
    stride      = patch_size // 2
    with_center = getattr(args, 'with_center', False)
    
    H, W, _ = img.shape

    y0_list = sorted({(y0 // 8) * 8 for y0 in range(0, H - patch_size + 1, stride)})
    x0_list = sorted({(x0 // 8) * 8 for x0 in range(0, W - patch_size + 1, stride)})
    if not y0_list: y0_list = [0]
    if not x0_list: x0_list = [0]
    tail_y = max(0, H - patch_size)
    tail_x = max(0, W - patch_size)
    if y0_list[-1] != tail_y: y0_list.append(tail_y)
    if x0_list[-1] != tail_x: x0_list.append(tail_x)

    qt0 = np.array(Image.open(path).quantization[0], dtype=np.float32).reshape(8,8)
    qf = estimate_qf(qt0)

    scores = []
    coords = []
    for y0 in y0_list:
        for x0 in x0_list:
            x1 = min(x0 + patch_size, W)
            y1 = min(y0 + patch_size, H)
            f = DCTR(path, coords=(x0, y0, x1, y1), QF=qf)
            score = mahalanobis_diag(f, mu, var)
            scores.append(score)
            coords.append((x0, y0, x1, y1))

    idx_sorted = np.argsort(scores)[::-1]
    selected = [coords[i] for i in idx_sorted[:min(patch_num, len(coords))]]
    selected_scores = [scores[i] for i in idx_sorted[:min(patch_num, len(coords))]]

    if len(selected) < patch_num:
        need = patch_num - len(selected)

        for i in range(need):
            idx = idx_sorted[i % len(idx_sorted)]
            selected.append(coords[idx])
            selected_scores.append(scores[idx])

    return selected, selected_scores