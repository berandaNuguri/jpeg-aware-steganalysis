from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import os
import sys
import argparse
import random
import cv2
import jpeglib
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = str(project_root())  # 프로젝트 루트
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from jpeg_aware_steganalysis.utils.runtime import build_transforms


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)

# ---------------------------
# Robust normalization for visualization
# ---------------------------
def vis_norm(x: torch.Tensor, clip=3.0, eps=1e-12) -> torch.Tensor:
    """
    Robust normalize to [0,1] using median/MAD.
    """
    x = x.detach().float()
    med = x.median()
    mad = (x - med).abs().median()
    z = (x - med) / (1.4826 * mad + eps)
    z = z.clamp(-clip, clip)
    y = (z + clip) / (2 * clip)
    return y.clamp(0.0, 1.0)


def tensor_rgb01_to_uint8_hwc(chw: torch.Tensor) -> np.ndarray:
    """
    chw: [3,H,W] in [0,1] (assumed)
    return: [H,W,3] uint8
    """
    x = chw.detach().float().clamp(0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def tensor_dqdct_to_vis_hwc(chw: torch.Tensor, clip=3.0) -> np.ndarray:
    """
    chw: [C,H,W] (usually C=3)
    return: [H,W,3] float32 in [0,1] for imshow
    """
    x = chw.detach().float()
    if x.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")

    C, H, W = x.shape
    if C >= 3:
        chs = []
        for c in range(3):
            chs.append(vis_norm(x[c].abs(), clip=clip))
        out = torch.stack(chs, dim=0)  # [3,H,W]
        return out.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # if not 3ch, make pseudo-RGB from mean
    m = vis_norm(x.abs().mean(0), clip=clip)  # [H,W]
    out = torch.stack([m, m, m], dim=0)
    return out.permute(1, 2, 0).cpu().numpy().astype(np.float32)


def load_rgb(path: str) -> np.ndarray:
    """
    return RGB uint8 [H,W,3]
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------
# Full-image DQDCT residual using jpeglib
# ---------------------------
def _tile_blocks_to_image(blocks: np.ndarray) -> np.ndarray:
    """
    blocks: [Hb,Wb,8,8] -> [Hb*8, Wb*8]
    """
    if blocks.ndim != 4 or blocks.shape[-2:] != (8, 8):
        raise ValueError(f"Expected blocks [Hb,Wb,8,8], got {blocks.shape}")
    Hb, Wb, _, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(Hb * 8, Wb * 8)


def read_full_dqdct_rgb_like(jpeg_path: str, out_hw=None) -> np.ndarray:
    j = jpeglib.read_dct(jpeg_path)

    # ---- get quant tables ----
    qt = None
    for attr in ["qt", "quant_tables", "quantization", "qtable", "qtables"]:
        if hasattr(j, attr):
            qt = getattr(j, attr)
            break
    if qt is None:
        raise RuntimeError("Cannot find quant tables in jpeglib object. (expected .qt or similar)")

    # normalize qt to list of 2 tables [8,8]
    # many JPEGs use qt[0]=luma, qt[1]=chroma
    if isinstance(qt, dict):
        # common: {0:table0, 1:table1}
        qt0 = np.array(qt.get(0))
        qt1 = np.array(qt.get(1, qt0))
    else:
        qt_list = list(qt)
        qt0 = np.array(qt_list[0])
        qt1 = np.array(qt_list[1]) if len(qt_list) > 1 else np.array(qt_list[0])

    qt0 = qt0.reshape(8, 8)
    qt1 = qt1.reshape(8, 8)

    # ---- get quantized DCT blocks ----
    # Expect j.Y / j.Cb / j.Cr as [Hb,Wb,8,8] or similar
    comps = []
    for name in ["Y", "Cb", "Cr"]:
        if not hasattr(j, name):
            comps.append(None)
        else:
            comps.append(np.array(getattr(j, name)))

    if comps[0] is None:
        raise RuntimeError("jpeglib object has no Y component.")

    # Ensure each comp is [Hb,Wb,8,8] (some libs might give [Hb,Wb,64])
    def ensure_blocks(a):
        a = np.array(a)
        if a.ndim == 3 and a.shape[-1] == 64:
            Hb, Wb, _ = a.shape
            return a.reshape(Hb, Wb, 8, 8)
        if a.ndim == 4 and a.shape[-2:] == (8, 8):
            return a
        raise ValueError(f"Unsupported DCT coeff shape: {a.shape}")

    Yb = ensure_blocks(comps[0])
    Cbb = ensure_blocks(comps[1]) if comps[1] is not None else None
    Crb = ensure_blocks(comps[2]) if comps[2] is not None else None

    # Dequantize: coeff * qtable
    Yd = Yb.astype(np.float32) * qt0[None, None, :, :]
    if Cbb is not None:
        Cbd = Cbb.astype(np.float32) * qt1[None, None, :, :]
    else:
        Cbd = np.zeros_like(Yd, dtype=np.float32)
    if Crb is not None:
        Crd = Crb.astype(np.float32) * qt1[None, None, :, :]
    else:
        Crd = np.zeros_like(Yd, dtype=np.float32)

    # Tile blocks to (H*,W*)
    Yimg  = _tile_blocks_to_image(Yd)
    Cbimg = _tile_blocks_to_image(Cbd)
    Crimg = _tile_blocks_to_image(Crd)

    # Crop to out_hw if given (to match actual pixel size)
    if out_hw is not None:
        H, W = out_hw
        Yimg  = Yimg[:H, :W]
        Cbimg = Cbimg[:H, :W]
        Crimg = Crimg[:H, :W]

    # stack to 3-ch map
    out = np.stack([Yimg, Cbimg, Crimg], axis=-1)  # [H,W,3]
    return out


def dqdct_residual_vis(cover_path: str, stego_path: str, out_hw) -> np.ndarray:
    """
    Compute |DQDCT(stego) - DQDCT(cover)| and return [H,W,3] float32 in [0,1] for imshow.
    """
    # read full dqdct maps
    dc = read_full_dqdct_rgb_like(cover_path, out_hw=out_hw)  # [H,W,3] float
    ds = read_full_dqdct_rgb_like(stego_path, out_hw=out_hw)

    res = np.abs(ds - dc)  # [H,W,3]

    # robust normalize per channel using torch
    t = torch.from_numpy(res).permute(2, 0, 1)  # [3,H,W]
    vis = []
    for c in range(3):
        vis.append(vis_norm(t[c], clip=3.0))
    vis = torch.stack(vis, dim=0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return vis


# ---------------------------
# Loader
# ---------------------------
def build_loader(args):
    tf, ds = build_transforms(args.transform_config)
    TestDS = ds["test"]

    split_names = [s.strip() for s in args.eval_splits.split("+") if s.strip()]
    if not split_names:
        raise ValueError("eval_splits must include at least one split (e.g. 'test').")

    csv_root = os.path.join(args.csv_root, args.data)
    dfs = []
    for split in split_names:
        csv_path = os.path.join(csv_root, f"{args.stego_method}_{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"[INFO] Loaded {split} CSV: {csv_path} (n={len(df)})")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total samples from splits {split_names}: {len(df_all)}")

    ds_test = TestDS(df_all, args.dataset_root, transform=tf["test"], args=args)

    loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        num_workers=args.workers_per_loader,
        shuffle=False,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=(args.workers_per_loader > 0),
    )
    return loader


# ---------------------------
# ROI grid visualization
# ---------------------------
def save_roi_grid(
    out_path: str,
    rgb_full: np.ndarray,          # [H,W,3] uint8
    res_rgb: np.ndarray,           # [H,W,3] uint8 or float in [0,1]
    res_dqdct: np.ndarray,         # [H,W,3] float in [0,1]
    roi_rgbs: list,                # list of [H,W,3] uint8
    roi_dqdcts: list,              # list of [H,W,3] float in [0,1]
    title: str,
):
    R = max(len(roi_rgbs), len(roi_dqdcts))
    ncols = max(3, R if R > 0 else 3)
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2))
    if ncols == 1:
        axes = np.array(axes).reshape(nrows, 1)

    def _imshow(ax, img, t=None):
        ax.imshow(img)
        if t is not None:
            ax.set_title(t, fontsize=9)
        ax.axis("off")

    # --- row 0: full images ---
    _imshow(axes[0, 0], rgb_full, "Full RGB (stego)")
    _imshow(axes[0, 1], res_rgb,  "Residual RGB |S-C|")
    _imshow(axes[0, 2], res_dqdct, "Residual DQDCT |S-C|")
    for c in range(3, ncols):
        axes[0, c].axis("off")

    # --- row 1: ROI RGB ---
    for c in range(ncols):
        if c < len(roi_rgbs):
            _imshow(axes[1, c], roi_rgbs[c], f"ROI RGB #{c}")
        else:
            axes[1, c].axis("off")

    # --- row 2: ROI DQDCT ---
    for c in range(ncols):
        if c < len(roi_dqdcts):
            _imshow(axes[2, c], roi_dqdcts[c], f"ROI DQDCT #{c}")
        else:
            axes[2, c].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] device={device} (model not used)")

    for roi_method in args.roi_method:
        print(f"\n==== ROI: {roi_method} ====")
        args.roi_method = roi_method
        loader = build_loader(args)

        if len(loader) == 0:
            raise RuntimeError("DataLoader is empty.")
        if args.pick_batch < 0 or args.pick_batch >= len(loader):
            raise ValueError(f"--pick_batch out of range: {args.pick_batch} (len={len(loader)})")

        for bi, batch in enumerate(loader):
            if bi != args.pick_batch:
                continue

            # batch = (data, paths)
            if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                raise ValueError("Expected batch=(data, paths).")
            data, paths = batch

            # data = (spatial_pair, dqdct_pair, qdct_pair, qt_pair)
            if not (isinstance(data, (tuple, list)) and len(data) == 4):
                raise ValueError("Expected data=(spatial_pair, dqdct_pair, qdct_pair, qt_pair).")

            spatial_pair, dqdct_pair, _, _ = data

            # expected dims:
            # spatial_pair: [B,2,R,3,H,W] or [B,2,3,H,W]
            # dqdct_pair  : [B,2,R,C,H,W] or [B,2,C,H,W]
            spatial_has_roi = (spatial_pair.dim() == 6)
            dqdct_has_roi   = (dqdct_pair.dim() == 6)

            if spatial_pair.dim() not in (5, 6):
                raise ValueError(f"Unexpected spatial_pair shape: {tuple(spatial_pair.shape)}")
            if dqdct_pair.dim() not in (5, 6):
                raise ValueError(f"Unexpected dqdct_pair shape: {tuple(dqdct_pair.shape)}")

            B = spatial_pair.size(0)
            Pair = spatial_pair.size(1)
            if Pair < 2:
                raise ValueError("Need Pair dimension >=2 (cover, stego).")

            # ROI count
            if spatial_has_roi:
                R = spatial_pair.size(2)
            elif dqdct_has_roi:
                R = dqdct_pair.size(2)
            else:
                R = 1

            # 배치 내 특정 index만 뽑고 싶으면 pick_index 사용
            b_indices = [args.pick_index] if args.pick_index >= 0 else list(range(B))
            for b in b_indices:
                if b < 0 or b >= B:
                    continue

                cover_path = str(paths[0][b])  # may be list-like
                stego_path = str(paths[1][b])

                # ---- 1) full RGB + residual RGB ----
                cover_rgb = load_rgb(cover_path)
                stego_rgb = load_rgb(stego_path)
                H0, W0 = stego_rgb.shape[:2]
                res_rgb = np.abs(stego_rgb.astype(np.int16) - cover_rgb.astype(np.int16)).astype(np.uint8)

                # ---- 2) residual DQDCT (full) ----
                # If not JPEG or jpeglib missing, fallback to zeros
                res_dqdct = np.zeros((H0, W0, 3), dtype=np.float32)
                try:
                    lp = stego_path.lower()
                    if lp.endswith(".jpg") or lp.endswith(".jpeg"):
                        res_dqdct = dqdct_residual_vis(cover_path, stego_path, out_hw=(H0, W0))
                except Exception as e:
                    print(f"[WARN] DQDCT residual failed for {os.path.basename(stego_path)}: {e}")

                # ---- 3) ROI RGB list (use stego pair=1) ----
                roi_rgbs = []
                for r in range(R):
                    if spatial_has_roi:
                        # [B,2,R,3,H,W]
                        roi = spatial_pair[b, 1, r]  # stego
                    else:
                        # [B,2,3,H,W]
                        roi = spatial_pair[b, 1]

                    # ROI tensor -> uint8 HWC
                    roi_rgbs.append(tensor_rgb01_to_uint8_hwc(roi))

                # ---- 4) ROI DQDCT list (use stego pair=1) ----
                roi_dqdcts = []
                for r in range(R):
                    if dqdct_has_roi:
                        # [B,2,R,C,H,W]
                        d = dqdct_pair[b, 1, r].to(device)
                    else:
                        # [B,2,C,H,W]
                        d = dqdct_pair[b, 1].to(device)

                    roi_dqdcts.append(tensor_dqdct_to_vis_hwc(d, clip=3.0))

                # ---- save figure ----
                out_dir = os.path.join(
                    args.vis_out,
                    args.data.replace("/", "_"),
                    f"{args.stego_method}_{roi_method}",
                    f"batch_{bi:05d}",
                )
                os.makedirs(out_dir, exist_ok=True)

                base = os.path.splitext(os.path.basename(stego_path))[0]
                out_png = os.path.join(out_dir, f"{base}_roi_grid.png")

                title = f"{base} | ROI={roi_method} | R={R}"
                save_roi_grid(
                    out_png,
                    rgb_full=stego_rgb,
                    res_rgb=res_rgb,
                    res_dqdct=res_dqdct,
                    roi_rgbs=roi_rgbs,
                    roi_dqdcts=roi_dqdcts,
                    title=title,
                )

            break  # pick_batch만 처리

    print(f"[DONE] {args.vis_out}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str,
                   default="RWSteg/Spatial_30000_to_PillowJPG/QT75_1_from_COYO700M_CC12M")
    p.add_argument("--csv_root", type=str, default="./csv/")
    p.add_argument("--dataset_root", type=str, default=str(_checkout_data_path('')))
    p.add_argument("--stego_method", type=str, default="UERD_0.4")
    p.add_argument("--eval_splits", type=str, default="test")
    p.add_argument("--transform_config", type=str, default="./configs/vis_dct_zeroDC.yaml")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers_per_loader", type=int, default=4)

    p.add_argument("--roi_method", nargs="+", default=["inconsist_top_2"])

    p.add_argument("--vis_out", type=str, default="./figures/")
    p.add_argument("--pick_batch", type=int, default=0, help="0-based batch index")
    p.add_argument("--pick_index", type=int, default=-1, help="index within batch; -1 means all samples in batch")

    p.add_argument("--gpus", type=str, default="5")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gpus is not None and args.gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    if torch.cuda.is_available():
        print(f"Using GPU(s): {args.gpus}")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU")

    run(args)
