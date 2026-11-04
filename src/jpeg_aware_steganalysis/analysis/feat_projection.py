from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = str(project_root())
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from jpeg_aware_steganalysis.utils.runtime import load_model, build_transforms


def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)


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


def feat_to_vec(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: 모델 내부 feature (배치 포함)
    반환: [N, D] float32
    - Conv: [N,C,h,w] -> abs -> GAP -> [N,C]
    - Token: [N,T,C] -> abs -> mean over T -> [N,C]
    - Linear: [N,D] -> abs -> [N,D]
    - 기타: batch 제외 flatten
    """
    if feat.dim() == 4:
        v = feat.detach().float().abs().mean(dim=(2, 3))  # [N,C]
        return v
    if feat.dim() == 3:
        v = feat.detach().float().abs().mean(dim=1)       # [N,C]
        return v
    if feat.dim() == 2:
        v = feat.detach().float().abs()                   # [N,D]
        return v
    if feat.dim() == 1:
        v = feat.detach().float().abs().unsqueeze(0)      # [1,D]
        return v

    # fallback: batch axis만 남기고 flatten
    x = feat.detach().float().abs()
    N = x.size(0)
    return x.reshape(N, -1)


def run_dr(X: np.ndarray, args) -> np.ndarray:
    """
    X: [N,D] numpy
    return: [N,2]
    """
    method = args.dr_method.lower()
    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_lr,
            init="pca",
            random_state=args.seed,
        )
        return tsne.fit_transform(X)

    if method == "umap":
        try:
            import umap
        except Exception as e:
            print(f"[WARN] UMAP import failed ({e}). Falling back to t-SNE.")
            args.dr_method = "tsne"
            return run_dr(X, args)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
        return reducer.fit_transform(X)

    raise ValueError(f"Unknown --dr_method: {args.dr_method} (use tsne or umap)")


def plot_2d(df: pd.DataFrame, out_png: str):
    """
    df columns: x, y, label (0 cover / 1 stego)
    """
    plt.figure(figsize=(7.2, 6.4))
    for lab, name in [(0, "cover"), (1, "stego")]:
        m = df["label"] == lab
        if m.any():
            plt.scatter(df.loc[m, "x"], df.loc[m, "y"], s=10, alpha=0.4)
    # plt.title(title, fontsize=21)
    plt.axis("off")
    # plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"[SAVE] {out_png}")


def extract_features_for_roi(args, model, hook, loader, roi_method: str, device):
    """
    loader 전체를 끝까지 순회하면서 모든 feature를 수집 (샘플링/중단 없음)
    """
    feats_all = []
    meta_all = []

    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                raise ValueError("Expected batch=(data, paths).")
            data, paths = batch

            if not (isinstance(data, (tuple, list)) and len(data) == 4):
                raise ValueError("Expected data=(spatial_pair, dqdct_pair, qdct_pair, qt_pair).")

            _, dqdct_pair, _, _ = data

            dqdct_has_roi = (dqdct_pair.dim() == 6)  # [B,2,R,C,H,W]
            if dqdct_pair.dim() not in (5, 6):
                raise ValueError(f"Unexpected dqdct_pair shape: {tuple(dqdct_pair.shape)}")

            B = dqdct_pair.size(0)
            Pair = dqdct_pair.size(1)
            if Pair < 2:
                raise ValueError("Need Pair dimension >=2 (cover, stego).")

            dqdct_pair = dqdct_pair.to(device, non_blocking=True)

            if dqdct_has_roi:
                # [B,2,R,C,H,W] -> [B*2*R,C,H,W]
                B0, P0, R, C, H, W = dqdct_pair.shape
                x = dqdct_pair.reshape(B0 * P0 * R, C, H, W)
                flat_stride = P0 * R      # per b
                pair_stride = R           # per pair
                roi_count = R
            else:
                # [B,2,C,H,W] -> [B*2,C,H,W]
                B0, P0, C, H, W = dqdct_pair.shape
                x = dqdct_pair.reshape(B0 * P0, C, H, W)
                flat_stride = P0
                pair_stride = 1
                roi_count = 1

            hook.clear()
            _ = model(x)

            # modules feature -> concat vec [Nflat, Dsum]
            vecs = []
            for name in feature_names:
                feat = hook.feats.get(name, None)
                if feat is None:
                    raise KeyError(f"Hook did not capture: {name}")
                if feat.dim() >= 1 and feat.size(0) == x.size(0):
                    v = feat_to_vec(feat)  # [Nflat, D]
                else:
                    # 극단 케이스 대비
                    v0 = feat_to_vec(feat[:1] if feat.dim() > 0 else feat)
                    v = v0.repeat(x.size(0), 1)
                vecs.append(v)

            V = torch.cat(vecs, dim=1)  # [Nflat, Dsum]
            V = V.detach().cpu().numpy().astype(np.float32)

            cover_paths = paths[0]
            stego_paths = paths[1]

            Nflat = V.shape[0]
            for fi in range(Nflat):
                # fi -> (b, pair, roi)
                if dqdct_has_roi:
                    b = fi // flat_stride
                    rem = fi % flat_stride
                    pair = rem // pair_stride
                    roi_i = rem % pair_stride
                else:
                    b = fi // flat_stride
                    pair = fi % flat_stride
                    roi_i = 0

                label = int(pair)  # cover=0, stego=1
                img_name = str(stego_paths[b]).split("/")[-1]  # 대표로 stego 파일명

                feats_all.append(V[fi])
                meta_all.append({
                    "label": label,
                    "roi": int(roi_i),
                    "batch": int(bi),
                    "img": img_name,
                    "roi_method": roi_method,
                })

    X = np.stack(feats_all, axis=0) if len(feats_all) > 0 else np.zeros((0, 1), np.float32)
    meta_df = pd.DataFrame(meta_all)
    return X, meta_df



def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] device={device}")

    model = load_model(args, device)
    model.eval()

    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    if not feature_names:
        raise ValueError("--feature_names must include at least one module name.")
    hook = Hook(model, feature_names)

    for roi_method in args.roi_method:
        print(f"\n==== ROI: {roi_method} ====")
        args.roi_method = roi_method
        loader = build_loader(args)

        X, meta_df = extract_features_for_roi(args, model, hook, loader, roi_method, device)
        if X.shape[0] == 0:
            print("[WARN] No features collected.")
            continue

        print(f"[FEAT] collected X={X.shape}, meta={len(meta_df)}")

        # 차원축소
        emb = run_dr(X, args)  # [N,2]
        out_df = meta_df.copy()
        out_df["x"] = emb[:, 0]
        out_df["y"] = emb[:, 1]

        out_dir = os.path.join(
            args.vis_out,
            args.pretrained_path.split("/")[-2] if args.pretrained_path else "no_ckpt",
            f"{args.stego_method}_{roi_method}",
            f"{args.dr_method.lower()}",
        )
        os.makedirs(out_dir, exist_ok=True)

        # 저장: csv + png
        # out_csv = os.path.join(out_dir, "embed.csv")
        out_png = os.path.join(out_dir, "embed.png")

        # out_df.to_csv(out_csv, index=False)
        # print(f"[SAVE] {out_csv}")

        # title = f"{args.dr_method.upper()} | {args.stego_method} | ROI={roi_method}\nmodules={args.feature_names}"
        # title = "Global Average Pooling"
        # title = "Spatial Pyramid Pooling"
        plot_2d(out_df, out_png)

    hook.close()
    print(f"[DONE] {args.vis_out}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str,
                   default="RWSteg/Spatial_30000_to_PillowJPG/QT95_1_Subsample_420")
    p.add_argument("--csv_root", type=str, default="./csv/")
    p.add_argument("--dataset_root", type=str, default=str(_checkout_data_path('')))
    p.add_argument("--stego_method", type=str, default="UERD_0.4")
    p.add_argument("--eval_splits", type=str, default="test")
    p.add_argument("--transform_config", type=str, default="./configs/vis_dct_zeroDC.yaml")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers_per_loader", type=int, default=4)

    p.add_argument("--roi_method", nargs="+", default=["calibration_top_2"])

    p.add_argument("--model_name", type=str, default="TempNet_v53")
    p.add_argument("--pretrained_path", type=str, default="./results/TempNet_v53_RWSteg_Spatial_30000_to_PillowJPG_QT95_1_Subsample_420_UERD_0.4_mean_256rand_2_w_100_8_AdamW_0.0001_0.0_DQDCTBase_ZeroDC/model_best.pth.tar")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--init_weights", action="store_true")

    p.add_argument("--vis_out", type=str, default="./figures/")

    # ---- feature hook target modules ----
    p.add_argument("--feature_names", type=str, default="fc1")

    # ---- DR options ----
    p.add_argument("--dr_method", type=str, default="tsne", choices=["tsne", "umap"])

    # t-SNE
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_lr", type=float, default=200.0)

    # UMAP
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", type=str, default="euclidean")

    # env
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
