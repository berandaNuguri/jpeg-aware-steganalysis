from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import os
import sys
import argparse
import random
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams["font.family"] = "Times New Roman"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = str(project_root())
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from jpeg_aware_steganalysis.utils.runtime import load_model, build_transforms


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)


def save_image_raw(path: str, img: np.ndarray):
    """
    img:
      - [H,W] float32 in [0,1] or uint8
      - [H,W,3] float32 in [0,1] or uint8 (RGB)
    """
    out = img

    if out.dtype != np.uint8:
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)

    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, out)


# ---------------------------
# Robust normalization utilities
# ---------------------------
def vis_stats(x: torch.Tensor, eps=1e-12):
    """
    x: any shape tensor (float)
    return: median, mad (both scalar tensors)
    """
    x = x.detach().float().reshape(-1)
    med = x.median()
    mad = (x - med).abs().median().clamp_min(eps)
    return med, mad


def vis_norm_with_stats(x: torch.Tensor, med: torch.Tensor, mad: torch.Tensor, clip=3.0, eps=1e-12):
    """
    x: tensor
    med/mad: scalar tensors
    return: [0,1] normalized
    """
    x = x.detach().float()
    z = (x - med) / (1.4826 * mad + eps)
    z = z.clamp(-clip, clip)
    y = (z + clip) / (2 * clip)
    return y.clamp(0.0, 1.0)


def vis_norm(x: torch.Tensor, clip=3.0, eps=1e-12) -> torch.Tensor:
    """
    feature마다 개별 median/MAD로 정규화
    """
    med, mad = vis_stats(x, eps=eps)
    return vis_norm_with_stats(x, med, mad, clip=clip, eps=eps)


def tensor_to_imshow(chw: torch.Tensor, is_rgb=False, clip=3.0) -> np.ndarray:
    """
    - is_rgb=True : input은 0~1 텐서라고 가정하고 0~255 uint8로 표시
    - is_rgb=False: robust 표준화(vis_norm)로 톤/대비 통일 (채널 유지)
    """
    if chw.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(chw.shape)}")

    x = chw.detach()
    C, H, W = x.shape

    if is_rgb:
        if C == 3:
            x = x.float().clamp(0.0, 1.0)
            x = (x * 255.0).round().to(torch.uint8)
            return x.permute(1, 2, 0).cpu().numpy()
        else:
            x = x.float().clamp(0.0, 1.0)
            x = (x * 255.0).round().to(torch.uint8)
            return x.mean(0).cpu().numpy()

    x = x.float()
    if C == 3:
        ch0 = vis_norm(x[0].abs(), clip=clip)
        ch1 = vis_norm(x[1].abs(), clip=clip)
        ch2 = vis_norm(x[2].abs(), clip=clip)
        out = torch.stack([ch0, ch1, ch2], dim=0)
        return out.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    m = x.abs().mean(0)
    m = vis_norm(m, clip=clip)
    return m.cpu().numpy().astype(np.float32)


# ---------------------------
# Feature -> 2D map (H,W) for shared normalization
# ---------------------------
def feat_to_map_any(feat: torch.Tensor, out_hw):
    """
    feat -> [H,W] float map (abs 기반)

    - Conv: [1,C,h,w] or [C,h,w] -> abs mean over C -> upsample
    - Token/Linear: [1,T,C] / [T,C] / [1,D] / [D] -> abs -> 1D -> interpolate to W -> repeat to H
    """
    H, W = int(out_hw[0]), int(out_hw[1])

    # batch 제거
    if feat.dim() == 4 and feat.size(0) == 1:
        feat = feat[0]  # [C,h,w]
    if feat.dim() == 3 and feat.size(0) == 1:
        feat = feat[0]  # [T,C] or [C,h,w] 케이스

    # Conv-like: [C,h,w]
    if feat.dim() == 3 and feat.size(0) <= 4096 and feat.size(1) > 1 and feat.size(2) > 1:
        m = feat.detach().float().abs().mean(0)              # [h,w]
        m = m.unsqueeze(0).unsqueeze(0)                      # [1,1,h,w]
        m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        return m

    # Token-like: [T,C]
    if feat.dim() == 2 and feat.size(0) > 1 and feat.size(1) > 1:
        v = feat.detach().float().abs().mean(dim=0)          # [C]
        v = v.unsqueeze(0).unsqueeze(0)                      # [1,1,C]
        v = F.interpolate(v, size=W, mode="linear", align_corners=False)[0, 0]  # [W]
        m = v.unsqueeze(0).repeat(H, 1)                      # [H,W]
        return m

    # Linear-like: [D] or [1,D]
    if feat.dim() == 2 and feat.size(0) == 1:
        v = feat[0].detach().float().abs()                   # [D]
    elif feat.dim() == 1:
        v = feat.detach().float().abs()                      # [D]
    else:
        v = feat.detach().float().abs().reshape(-1)

    v = v.unsqueeze(0).unsqueeze(0)                          # [1,1,D]
    v = F.interpolate(v, size=W, mode="linear", align_corners=False)[0, 0]      # [W]
    m = v.unsqueeze(0).repeat(H, 1)                          # [H,W]
    return m


def map_to_heat_rgb_shared(m: torch.Tensor, med: torch.Tensor, mad: torch.Tensor, cmap_name="coolwarm", clip=3.0):
    """
    m: [H,W] float tensor
    shared med/mad로 [0,1] 정규화 후 colormap 적용
    """
    m01 = vis_norm_with_stats(m, med=med, mad=mad, clip=clip).cpu().numpy()
    cm = plt.get_cmap(cmap_name)
    return cm(m01)[..., :3].astype(np.float32)


class Hook:
    def __init__(self, model: torch.nn.Module, names):
        self.names = names
        self.feats = {}
        self.handles = []
        mods = dict(model.named_modules())
        missing = [n for n in names if n not in mods]
        if missing:
            raise ValueError(f"[HOOK] modules not found: {missing}")
        for n in names:
            self.handles.append(mods[n].register_forward_hook(self._make(n)))

    def _make(self, name):
        def fn(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if torch.is_tensor(out):
                self.feats[name] = out.detach()
        return fn

    def clear(self):
        self.feats.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


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


def list_modules(model, keyword=None):
    for n, _ in model.named_modules():
        if keyword is None or keyword in n:
            print(n)


def _safe_name(s: str) -> str:
    return "".join([c if (c.isalnum() or c in "._-") else "_" for c in s])


def pretty_feature_label(name: str) -> str:
    # 원하는 라벨 매핑
    mapping = {
        "hpf.inter_dct": "Dilated Conv",
        "hpf.intra_dct": "Standard Conv",
    }
    return mapping.get(name, name)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] device={device}")

    model = load_model(args, device)
    model.eval()

    if args.list_modules:
        list_modules(model, keyword=args.filter_modules or None)
        return

    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    if not feature_names:
        raise ValueError("--feature_names must include at least one module name.")

    hook = Hook(model, feature_names)

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

            if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                raise ValueError("Expected batch=(data, paths).")
            data, paths = batch

            if not (isinstance(data, (tuple, list)) and len(data) == 4):
                raise ValueError("Expected data=(spatial_pair, dqdct_pair, qdct_pair, qt_pair).")

            spatial_pair, dqdct_pair, qdct_pair, qt_pair = data

            spatial_has_roi = (spatial_pair.dim() == 6)  # [B,2,R,3,H,W]
            dqdct_has_roi   = (dqdct_pair.dim() == 6)    # [B,2,R,C,H,W]

            if spatial_pair.dim() not in (5, 6):
                raise ValueError(f"Unexpected spatial_pair shape: {tuple(spatial_pair.shape)}")
            if dqdct_pair.dim() not in (5, 6):
                raise ValueError(f"Unexpected dqdct_pair shape: {tuple(dqdct_pair.shape)}")

            B = spatial_pair.size(0)
            if spatial_pair.size(1) < 2 or dqdct_pair.size(1) < 2:
                raise ValueError("Need Pair dimension >=2 (cover, stego).")

            # ROI 개수
            if dqdct_has_roi:
                roi_count = dqdct_pair.size(2)
            elif spatial_has_roi:
                roi_count = spatial_pair.size(2)
            else:
                roi_count = 1

            dqdct_pair = dqdct_pair.to(device, non_blocking=True)

            if dqdct_has_roi:
                _, _, _, C, H, W = dqdct_pair.shape
                R = dqdct_pair.size(2)
                x = dqdct_pair.reshape(B * 2 * R, C, H, W)
                flat_stride = 2 * R
                roi_stride = R
            else:
                _, _, C, H, W = dqdct_pair.shape
                x = dqdct_pair.reshape(B * 2, C, H, W)
                flat_stride = 2
                roi_stride = 1

            hook.clear()
            _ = model(x)

            out_dir = os.path.join(
                args.vis_out,
                args.pretrained_path.split("/")[-2] if args.pretrained_path else "no_ckpt",
                f"{args.stego_method}_{roi_method}",
                f"batch_{bi:05d}",
            )
            os.makedirs(out_dir, exist_ok=True)

            for b in range(B):
                stego_path = str(paths[1][b])
                suptitle = stego_path.split("/")[-1].split(".")[0]

                for roi_i in range(roi_count):
                    # 1) RGB (stego)
                    if spatial_has_roi:
                        rgb_stego = spatial_pair[b, 1, roi_i]
                    else:
                        rgb_stego = spatial_pair[b, 1]
                    col_rgb = tensor_to_imshow(rgb_stego, is_rgb=True)
                    out_hw = (rgb_stego.shape[-2], rgb_stego.shape[-1])

                    # 2) DQDCT residual
                    if dqdct_has_roi:
                        d_cover = dqdct_pair[b, 0, roi_i]
                        d_stego = dqdct_pair[b, 1, roi_i]
                        flat_idx = b * flat_stride + 1 * roi_stride + roi_i
                    else:
                        d_cover = dqdct_pair[b, 0]
                        d_stego = dqdct_pair[b, 1]
                        flat_idx = b * flat_stride + 1

                    col_residual = tensor_to_imshow((d_stego - d_cover).abs())

                    # 3) Feature maps raw
                    feat_maps = []
                    for name in feature_names:
                        feat = hook.feats.get(name, None)
                        if feat is None:
                            raise KeyError(f"Hook did not capture: {name}")

                        if feat.dim() >= 1 and feat.size(0) > flat_idx:
                            feat_b = feat[flat_idx:flat_idx + 1]
                        else:
                            feat_b = feat[:1] if feat.dim() >= 1 else feat

                        m = feat_to_map_any(feat_b, out_hw=out_hw)  # [H,W]
                        feat_maps.append(m)

                    # 4) shared stats (같은 ROI의 모든 feature map 값을 합쳐서)
                    all_vals = torch.cat([m.reshape(-1) for m in feat_maps], dim=0)
                    med, mad = vis_stats(all_vals)

                    # 5) apply shared norm + colormap
                    feat_cols = [
                        map_to_heat_rgb_shared(m, med=med, mad=mad, cmap_name=args.cmap_feat, clip=args.norm_clip)
                        for m in feat_maps
                    ]

                    # --------- compose figure ---------
                    images = [col_rgb, col_residual] + feat_cols
                    titles = ["RGB", "Residual"] + [pretty_feature_label(n) for n in feature_names]
                    ncols = len(images)

                    extra_w = 1.4
                    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4.2 + extra_w, 4.2))
                    if ncols == 1:
                        axes = [axes]

                    for ax, im, t in zip(axes, images, titles):
                        ax.imshow(im)
                        ax.set_title(t, fontsize=args.subplot_title_fs)
                        ax.axis("off")

                    if len(feature_names) > 0:
                        cmap = plt.get_cmap(args.cmap_feat)
                        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
                        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                        sm.set_array([])

                        cbar = fig.colorbar(sm, pad=0.02)

                        cbar.set_ticks([0.0, 1.0])
                        cbar.set_ticklabels(["Low", "High"])
                        cbar.ax.yaxis.set_ticks_position("right")
                        cbar.ax.yaxis.set_label_position("right")
                        cbar.ax.tick_params(labelsize=args.cbar_tick_fs)
                    plt.tight_layout()

                    sample_dir = os.path.join(out_dir, f"b{b:03d}_roi{roi_i}")
                    os.makedirs(sample_dir, exist_ok=True)

                    out_all = os.path.join(sample_dir, "all.png")
                    plt.savefig(out_all, dpi=220, bbox_inches="tight", pad_inches=0.01)
                    plt.close(fig)
                    print(f"[VIS] saved: {out_all}")

                    if args.save_parts:
                        save_image_raw(os.path.join(sample_dir, "rgb.png"), col_rgb)
                        save_image_raw(os.path.join(sample_dir, "residual.png"), col_residual)

                        feat_dir = os.path.join(sample_dir, "features")
                        os.makedirs(feat_dir, exist_ok=True)
                        for name, img in zip(feature_names, feat_cols):
                            safe = _safe_name(name)
                            save_image_raw(os.path.join(feat_dir, f"{safe}.png"), img)

            break  

    hook.close()
    print(f"[DONE] {args.vis_out}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str,
                   default="RWSteg/Spatial_30000_to_PillowJPG/QT75_1_Subsample_420")
    p.add_argument("--csv_root", type=str, default="./csv/")
    p.add_argument("--dataset_root", type=str, default=str(_checkout_data_path('')))
    p.add_argument("--stego_method", type=str, default="UERD_0.4")
    p.add_argument("--eval_splits", type=str, default="test")
    p.add_argument("--transform_config", type=str, default="./configs/vis_dct_zeroDC.yaml")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers_per_loader", type=int, default=4)

    p.add_argument("--roi_method", nargs="+", default=["calibration_top_2"])

    p.add_argument("--model_name", type=str, default="TempNet_v53")
    p.add_argument("--pretrained_path", type=str,
                   default="./results/TempNet_v53_RWSteg_Spatial_30000_to_PillowJPG_QT75_1_Subsample_420_UERD_0.4_mean_256rand_2_w_100_8_AdamW_0.0001_0.0_DQDCTBase_ZeroDC/model_best.pth.tar")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init_weights", action="store_true")

    p.add_argument("--vis_out", type=str, default="./figures/")
    p.add_argument("--pick_batch", type=int, default=0, help="0-based batch index to visualize")

    # ---- feature hook targets ----
    p.add_argument("--feature_names", type=str, default="hpf.inter_dct,hpf.intra_dct")

    # ---- feature colormap / normalization ----
    p.add_argument("--cmap_feat", type=str, default="coolwarm")
    p.add_argument("--norm_clip", type=float, default=3.0,
                   help="shared robust z-score clip (higher -> less saturated)")

    # ---- plot text ----
    p.add_argument("--subplot_title_fs", type=int, default=24)

    # ---- colorbar text ----
    p.add_argument("--cbar_label_fs", type=int, default=17)
    p.add_argument("--cbar_tick_fs", type=int, default=15)

    # ---- save parts ----
    p.add_argument("--save_parts", action="store_true",
                   help="save rgb/residual/features separately too")

    # ---- env ----
    p.add_argument("--gpus", type=str, default="5")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--list_modules", action="store_true")
    p.add_argument("--filter_modules", type=str, default="hpf")

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
