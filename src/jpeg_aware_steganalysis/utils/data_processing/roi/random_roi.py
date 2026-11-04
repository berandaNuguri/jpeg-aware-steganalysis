import numpy as np
from typing import List, Tuple

def extract_patch_coords(args, img, residual=None) -> List[Tuple[int, int, int, int]]:
    num_patches = int(args.roi_method.split('_')[-1])
    patch_size = getattr(args, 'patch_size', 256)
    if patch_size % 8 != 0:
        raise ValueError(f"patch_size={patch_size} is not divisible by 8.")

    H, W = img.shape[:2]

    Hg = (H // 8) * 8
    Wg = (W // 8) * 8

    if Hg < patch_size or Wg < patch_size:
        raise ValueError(
            f"Valid 8x8 grid area {(Hg, Wg)} is smaller than patch size {patch_size}."
        )

    max_y0 = Hg - patch_size
    max_x0 = Wg - patch_size

    max_y0_aligned = (max_y0 // 8) * 8
    max_x0_aligned = (max_x0 // 8) * 8

    def rand_patch():
        y0 = np.random.randint(0, (max_y0_aligned // 8) + 1) * 8
        x0 = np.random.randint(0, (max_x0_aligned // 8) + 1) * 8
        return (x0, y0, x0 + patch_size, y0 + patch_size)

    if residual is None or (isinstance(residual, str) and residual.lower() == "random"):
        return [rand_patch() for _ in range(num_patches)]

    res = residual[:Hg, :Wg]
    if res.ndim == 3:
        m = np.any(res != 0, axis=-1)
    else:
        m = (res != 0)

    ys, xs = np.where(m)
    if len(xs) == 0 or len(ys) == 0:
        return [rand_patch() for _ in range(num_patches)]

    block_xs = np.unique((xs // 8) * 8)
    block_ys = np.unique((ys // 8) * 8)

    block_xs = block_xs[block_xs <= max_x0_aligned]
    block_ys = block_ys[block_ys <= max_y0_aligned]

    if block_xs.size == 0 or block_ys.size == 0:
        return [rand_patch() for _ in range(num_patches)]

    coords = []
    for _ in range(num_patches):
        x0 = np.random.choice(block_xs)
        y0 = np.random.choice(block_ys)
        coords.append((x0, y0, x0 + patch_size, y0 + patch_size))

    return coords
