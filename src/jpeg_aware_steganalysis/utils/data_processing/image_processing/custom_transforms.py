import torch
from torchvision import transforms

class MinMaxNormalize(object):
    def __init__(self, min, max, out_range=(-1.0, 1.0)):
        self.min = torch.tensor(min).float()
        self.max = torch.tensor(max).float()
        self.out_min, self.out_max = out_range

    def __call__(self, tensor):
        C, H, W = tensor.shape
        out = torch.empty_like(tensor)
        for c in range(C):
            min_c, max_c = self.min[c], self.max[c]
            x = torch.clamp(tensor[c], min_c, max_c)
            x = (x - min_c) / (max_c - min_c)

            out[c] = x * (self.out_max - self.out_min) + self.out_min

        return out


class ZeroDCCoef(object):
    """Set DC coefficient (0,0) of every 8x8 block to zero for each channel."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError("ZeroDCDCT expects a 3D tensor (C, H, W).")

        C, H, W = tensor.shape
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError("ZeroDCDCT requires height and width to be multiples of 8.")

        blocks = tensor.view(C, H // 8, 8, W // 8, 8)
        blocks[:, :, 0, :, 0] = 0.0
        
        return tensor
