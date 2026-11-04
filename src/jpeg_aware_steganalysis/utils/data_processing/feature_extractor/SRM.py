
import numpy as np

filter_class_1 = [
  np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 1],
    [0, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [1, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 1],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 0],
    [1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 0],
    [0, 1, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
  ], dtype=np.float32)
]


filter_class_2 = [
  np.array([
    [1, 0, 0],
    [0, -2, 0],
    [0, 0, 1]
  ], dtype=np.float32),
  np.array([
    [0, 1, 0],
    [0, -2, 0],
    [0, 1, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 1],
    [0, -2, 0],
    [1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [1, -2, 1],
    [0, 0, 0]
  ], dtype=np.float32),
]


filter_class_3 = [
  np.array([
    [-1, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, -1],
    [0, 0, 0, 3, 0],
    [0, 0, -3, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -3, 3, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, -1]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, -3, 0, 0],
    [0, 3, 0, 0, 0],
    [-1, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, 3, -3, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32)
]


filter_edge_3x3 = [
  np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 2, -1],
    [0, -4, 2],
    [0, 2, -1]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [2, -4, 2],
    [-1, 2, -1]
  ], dtype=np.float32),
  np.array([
    [-1, 2, 0],
    [2, -4, 0],
    [-1, 2, 0]
  ], dtype=np.float32),
]

filter_edge_5x5 = [
  np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, -2, 2, -1],
    [0, 0, 8, -6, 2],
    [0, 0, -12, 8, -2],
    [0, 0, 8, -6, 2],
    [0, 0, -2, 2, -1]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
  ], dtype=np.float32),
  np.array([
    [-1, 2, -2, 0, 0],
    [2, -6, 8, 0, 0],
    [-2, 8, -12, 0, 0],
    [2, -6, 8, 0, 0],
    [-1, 2, -2, 0, 0]
  ], dtype=np.float32),
]

square_3x3 = np.array([
  [-1, 2, -1],
  [2, -4, 2],
  [-1, 2, -1]
], dtype=np.float32)

square_5x5 = np.array([
  [-1, 2, -2, 2, -1],
  [2, -6, 8, -6, 2],
  [-2, 8, -12, 8, -2],
  [2, -6, 8, -6, 2],
  [-1, 2, -2, 2, -1]
], dtype=np.float32)


all_hpf_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_edge_3x3 + filter_edge_5x5 + [square_3x3, square_5x5]

hpf_3x3_list = filter_class_1 + filter_class_2 + filter_edge_3x3 + [square_3x3]
hpf_5x5_list = filter_class_3 + filter_edge_5x5 + [square_5x5]

normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
normalized_filter_edge_3x3 = [hpf / 4 for hpf in filter_edge_3x3]
normalized_square_3x3 = square_3x3 / 4
normalized_filter_edge_5x5 = [hpf / 12 for hpf in filter_edge_5x5]
normalized_square_5x5 = square_5x5 / 12

all_normalized_hpf_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + \
 normalized_filter_edge_3x3 + normalized_filter_edge_5x5 + [normalized_square_3x3, normalized_square_5x5]

normalized_hpf_3x3_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_edge_3x3 + [normalized_square_3x3]
normalized_hpf_5x5_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + [normalized_square_5x5]

normalized_3x3_list = normalized_filter_edge_3x3 + [normalized_square_3x3]
normalized_5x5_list = normalized_filter_edge_5x5 + [normalized_square_5x5]

from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

TensorInput = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
ChannelMode = Literal["gray", "luma", "rgb", "per_channel"]


def _pad_to_5x5(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    row_1 = int((5 - kernel.shape[0]) / 2)
    row_2 = int((5 - kernel.shape[0]) - row_1)
    col_1 = int((5 - kernel.shape[1]) / 2)
    col_2 = int((5 - kernel.shape[1]) - col_1)
    return np.pad(kernel, pad_width=((row_1, row_2), (col_1, col_2)), mode="constant")


def build_srm_filter_bank(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the 30 normalized SRM high-pass filters as [30, 1, 5, 5]."""
    filters = [_pad_to_5x5(hpf) for hpf in all_normalized_hpf_list]
    return torch.tensor(np.stack(filters, axis=0), dtype=dtype).view(len(filters), 1, 5, 5)


def image_to_tensor(image: TensorInput) -> torch.Tensor:
    """Convert path/PIL/numpy/tensor input to float tensor [1, C, H, W] in 0-255 range."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.shape[0] in {1, 3}:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] in {1, 3}:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Cannot infer tensor channel dimension from shape {tuple(tensor.shape)}")
        elif tensor.dim() != 4:
            raise ValueError(f"Unsupported tensor shape: {tuple(tensor.shape)}")
        if tensor.numel() > 0 and tensor.max() <= 1.0:
            tensor = tensor * 255.0
        return tensor

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
        array = np.asarray(image)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        return torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    if array.ndim == 3:
        return torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
    raise ValueError(f"Unsupported image array shape: {array.shape}")


def _select_channels(x: torch.Tensor, channel_mode: ChannelMode) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    if channel_mode in {"rgb", "per_channel"}:
        return x
    if x.shape[1] == 1:
        return x
    if x.shape[1] != 3:
        raise ValueError(f"{channel_mode!r} mode expects 1 or 3 channels, got {int(x.shape[1])}")
    if channel_mode == "gray":
        return x.mean(dim=1, keepdim=True)
    if channel_mode == "luma":
        coeff = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * coeff).sum(dim=1, keepdim=True)
    raise ValueError(f"Unsupported channel_mode: {channel_mode}")


@torch.no_grad()
def extract_srm_features(
    image: TensorInput,
    channel_mode: ChannelMode = "luma",
    device: Optional[Union[str, torch.device]] = None,
    return_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract raw SRM residual feature maps.

    Args:
        image: image path, PIL image, numpy array, or torch tensor.
        channel_mode:
            - "luma": RGB -> luminance, output [B, 30, H, W].
            - "gray": RGB channel average, output [B, 30, H, W].
            - "rgb": apply 30 filters to each RGB channel independently,
              output [B, 90, H, W] for RGB input.
        device: torch device. Defaults to CUDA if available.
        return_numpy: return numpy array instead of torch tensor.

    Returns:
        Raw SRM residual feature map without clipping or learned network layers.
        Tensor output stays on the selected compute device. Numpy output is
        moved to CPU before conversion.
    """
    if channel_mode not in {"gray", "luma", "rgb", "per_channel"}:
        raise ValueError(f"Unsupported channel_mode: {channel_mode}")

    x = image_to_tensor(image)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = _select_channels(x.to(device), channel_mode)
    weight = build_srm_filter_bank(dtype=x.dtype).to(device)

    if channel_mode in {"rgb", "per_channel"}:
        batch, channels, height, width = x.shape
        y = F.conv2d(x.reshape(batch * channels, 1, height, width), weight, padding=2)
        y = y.reshape(batch, channels * weight.shape[0], height, width)
    else:
        y = F.conv2d(x, weight, padding=2)

    return y.detach().cpu().numpy() if return_numpy else y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract raw SRM residual feature maps.")
    parser.add_argument("image")
    parser.add_argument("--npy-path", default=None, help="Optional .npy output path.")
    parser.add_argument("--channel-mode", default="luma", choices=["gray", "luma", "rgb"])
    args = parser.parse_args()

    features = extract_srm_features(
        args.image,
        channel_mode=args.channel_mode,
        device=None,
        return_numpy=False,
    )

    print(f"SRM feature shape: {tuple(features.shape)}")
    print(f"SRM feature device: {features.device}")
    if args.npy_path:
        np.save(args.npy_path, features.detach().cpu().numpy())
