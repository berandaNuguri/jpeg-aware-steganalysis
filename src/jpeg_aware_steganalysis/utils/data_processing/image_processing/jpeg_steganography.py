from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import argparse
import conseal as cl
import jpeglib
import numpy as np
from glob import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def algorithm_selector(cover, spatial, algorithm, channels, alpha, seed):
    names = ['Y', 'Cb', 'Cr']
    try:
        for ch in channels:
            if ch not in (0, 1, 2):
                print(f"[WARN] Skip invalid channel index: {ch}", file=sys.stderr)
                continue

            ch_name = names[ch]
            y_ch = getattr(cover, ch_name)

            try:
                if algorithm == "J-UNIWARD":
                    qt_index = 0 if ch == 0 else 1
                    stego_ch = cl.juniward.simulate_single_channel(
                        x0=spatial.spatial[..., ch],
                        y0=y_ch,
                        qt=cover.qt[qt_index],
                        alpha=alpha,
                        seed=seed
                    )

                elif algorithm == "nsF5":
                    stego_ch = cl.nsF5.simulate_single_channel(
                        y0=y_ch,
                        alpha=alpha,
                        seed=seed
                    )

                elif algorithm == "UERD":
                    qt_index = 0 if ch == 0 else 1
                    stego_ch = cl.uerd.simulate_single_channel(
                        y0=y_ch,
                        qt=cover.qt[qt_index],
                        alpha=alpha,
                        payload_mode='bpnzAC',
                        seed=seed
                    )

                else:
                    print(f"[ERROR] Unsupported algorithm: {algorithm}. Channel {ch_name} left unchanged.", file=sys.stderr)
                    continue

                setattr(cover, ch_name, stego_ch)

            except Exception as ce:
                pass
                # print(f"[WARN] Embedding failed on channel {ch_name}: {ce}. Keeping original.", file=sys.stderr)

        return cover

    except Exception as e:
        print(f"[ERROR] algorithm_selector fatal: {e}. Returning original cover.", file=sys.stderr)
        return cover

def process_one(args, cover_path):
    basename = os.path.basename(cover_path)
 
    # 1. Read spatial and DCT
    try:
        spatial = jpeglib.read_spatial(cover_path)
        dct_obj = jpeglib.read_dct(cover_path)
    except Exception as e:
        return f"[WARN] {basename}: Failed to read spatial/DCT: {e}"

    # 2. Apply steganography algorithm
    try:
        stego = algorithm_selector(
            dct_obj,
            spatial,
            args.algorithm,
            args.embed_channels,
            args.alpha,
            args.seed
        )
    except Exception as e:
        return f"[ERROR] {basename}: algorithm_selector raised exception: {e}"

    # 3. Write output file
    out_dir = os.path.join(
        args.root_path,
        "Stego",
        f"{args.algorithm}_{args.alpha}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, basename)

    try:
        stego.write_dct(out_path)
    except Exception as e:
        return f"[ERROR] {basename}: Stego write failed; Error: {e}"

    return None


def main(args):
    cover_paths = (
        glob(os.path.join(args.root_path, 'Cover', '*.jpg')) +
        glob(os.path.join(args.root_path, 'Cover', '*.jpeg'))
    )

    if args.num_workers > 1:
        func = partial(process_one, args)
        pool = Pool(processes=args.num_workers)

        for msg in tqdm(pool.imap_unordered(func, cover_paths),
                        total=len(cover_paths),
                        desc=f"Processing images[{args.root_path.split('/')[-2]}]"):
            if msg:
                print(msg)

    else:
        for path in tqdm(cover_paths,
                            desc=f"Processing images[{args.root_path.split('/')[-2]}]",
                            total=len(cover_paths)):
            msg = process_one(args, path)
            if msg:
                print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JPEG Steganography using Conseal"
    )
    parser.add_argument("--root_path", type=str, default=str(_checkout_data_path('RWSteg/Spatial_30000_to_PillowJPG/QT75_1_Subsample_Mixed_from_COYO700M_CC12M/')), help="Path to the cover folder")
    # src/data/RWSteg/Spatial_30000_to_PillowJPG/QT95_Mixed_from_COYO700M_CC12M/
    # src/data/ALASKA2/ALASKA_v2_JPG_256_QF95_COLOR/
    parser.add_argument("--algorithm", type=str, default="UERD", help="Steganography algorithm to use ex) F5, nsF5, EBS, UERD, J-UNIWARD")
    parser.add_argument("--embed_channels", nargs='+', type=int, default=[0, 1, 2], help="Channels to embed in (0 for Y, 1 for Cb, 2 for Cr)")  
    parser.add_argument("--alpha", type=float, default=0.4, help="Payload in bpnzAC")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of parallel workers to use")
    args = parser.parse_args()

    main(args)