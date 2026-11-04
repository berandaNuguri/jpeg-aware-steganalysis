from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import argparse
import numpy as np
import conseal as cl
from PIL import Image
from glob import glob
from tqdm import tqdm

def algorithm_selector(cover, algorithm, channels):
    try:
        """
        permute True = Permute embedding
        permute False = Sequential embedding
        """
        if algorithm == "LSB":
            cover = cl.lsb.simulate(
                cover=cover,
                permute=False,
                alpha=args.alpha,
                seed=args.seed
            )

        else:
            if algorithm == "HUGO":
                for channel in channels:
                    if channel not in [0, 1, 2]:
                        raise ValueError(f"Invalid channel {channel}. Must be 0 (R), 1 (G), or 2 (B).")
                    if channel == 0:
                        cover[..., 0] = cl.hugo.simulate_single_channel(
                            x0=cover[..., 0],
                            alpha=args.alpha,
                            seed=args.seed
                        )
                    elif channel == 1:
                        cover[..., 1] = cl.hugo.simulate_single_channel(
                            x0=cover[..., 1],
                            alpha=args.alpha,
                            seed=args.seed
                        )
                    elif channel == 2:
                        cover[..., 2] = cl.hugo.simulate_single_channel(
                            x0=cover[..., 2],
                            alpha=args.alpha,
                            seed=args.seed
                        )
            elif algorithm == "WOW":
                for channel in channels:
                    if channel not in [0, 1, 2]:
                        raise ValueError(f"Invalid channel {channel}. Must be 0 (R), 1 (G), or 2 (B).")
                    if channel == 0:
                        cover[..., 0] = cl.wow.simulate_single_channel(
                            x0=cover[..., 0],
                            alpha=args.alpha,
                            seed=args.seed
                        )
                    elif channel == 1:
                        cover[..., 1] = cl.wow.simulate_single_channel(
                            x0=cover[..., 1],
                            alpha=args.alpha,
                            seed=args.seed
                        )
                    elif channel == 2:
                        cover[..., 2] = cl.wow.simulate_single_channel(
                            x0=cover[..., 2],
                            alpha=args.alpha,
                            seed=args.seed
                        )
            elif algorithm == "S-UNIWARD":
                for channel in channels:
                    if channel not in [0, 1, 2]:
                        raise ValueError(f"Invalid channel {channel}. Must be 0 (R), 1 (G), or 2 (B).")
                    if channel == 0:
                        cover[..., 0] = cl.suniward.simulate_single_channel(
                            x0=cover[..., 0],
                            alpha=args.alpha,
                            seed=args.seed
                        )
                    elif channel == 1:
                        cover[..., 1] = cl.suniward.simulate_single_channel(
                            x0=cover[..., 1],
                            alpha=args.alpha,
                            seed=args.seed
                        )
                    elif channel == 2:
                        cover[..., 2] = cl.suniward.simulate_single_channel(
                            x0=cover[..., 2],
                            alpha=args.alpha,
                            seed=args.seed
                        )
        return cover
    except Exception as e:
        print(f"Error processing image with {algorithm}: {e}")
        return None

    
def main(args):
    cover_paths = glob(args.root_path + 'Cover' + "/*.jpg") + glob(args.root_path + 'Cover' + "/*.jpeg") + glob(args.root_path + 'Cover' + "/*.png") +  glob(args.root_path + 'Cover' + "/*.tif")

    for cover_path in tqdm(cover_paths, desc="Processing images", total=len(cover_paths)):
        basename = os.path.basename(cover_path)
        cover = np.array(Image.open(cover_path))

        cover = algorithm_selector(cover, args.algorithm, args.embed_channels)
        if cover is None:
            break
        os.makedirs(args.root_path + f"/Stego/conseal_{args.algorithm}_Sequential_{args.alpha}", exist_ok=True)
        Image.fromarray(cover).save(args.root_path + f"/Stego/conseal_{args.algorithm}_Sequential_{args.alpha}/" + basename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG Steganography using Conceal")
    parser.add_argument("--root_path", type=str, default=str(_checkout_data_path('ALASKA2/ALASKA_v2_TIFF_256_COLOR/')), help="Path to the cover folder")
    parser.add_argument("--algorithm", type=str, default="LSB", help="Steganography algorithm to use \
                        ex) LSB, HUGO, WOW, S-UNIWARD, MiPOD, HILL")
    parser.add_argument("--embed_channels", nargs='+', type=int, default=[0, 1, 2], help="Channels to embed in (0 for Y, 1 for Cb, 2 for Cr)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Payload in bpnzAC")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(args)