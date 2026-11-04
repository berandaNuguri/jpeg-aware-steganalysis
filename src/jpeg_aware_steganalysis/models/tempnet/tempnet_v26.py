import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class DCTAwarePool(nn.Module):
    """
    Non-overlapping block pooling on the block-grid (no padding, no overlap).
    factor=2  => 2x2 DCT blocks -> 1 (H,W halve)
    mode: 'avg' | 'max' | 'lp' (p>1)
    """
    def __init__(self, factor: int = 2, mode: str = 'avg'):
        super().__init__()
        assert factor >= 2, "factor>=2"
        assert mode in ('avg','max','lp')
        self.f = factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        f = self.f

        if (H % f) != 0 or (W % f) != 0:
            raise ValueError(
                f"DCTAwarePool requires H and W divisible by factor={f}. "
                f"Got feature map of size {H}x{W}."
            )
        x = x.view(B, C, H//f, f, W//f, f)

        if self.mode == 'avg':
            y = x.mean(dim=(3,5))
        elif self.mode == 'max':
            y, _ = x.max(dim=5)
            y, _ = y.max(dim=3)
        else:
            raise ValueError(
                f"DCTAwarePool must choose at least one pooling mode (avg or max)."
            )

        return y

class FrequencyAwareConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, dilation=8, bias=False):
        super(FrequencyAwareConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.pool = DCTAwarePool(2, 'avg')

    def forward(self, input):
        output = self.conv(input)
        output = self.pool(output)
        return output

class DCTBlockConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False):
        super(DCTBlockConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.conv(input)
        
        return output


class TempNet_v26(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v26, self).__init__()
        self.stream1_1 = FrequencyAwareConv(3, 64, dilation=8)
        self.stream1_2 = FrequencyAwareConv(64, 128, dilation=4)
        self.stream1_3 = FrequencyAwareConv(128, 256, dilation=2)
        self.stream2_1 = DCTBlockConv(3072, 1536)
        self.stream2_2 = DCTBlockConv(1536, 768)
        self.stream2_3 = DCTBlockConv(768, 384)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(640, 2)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape

        assert H % 8 == 0 and W % 8 == 0, f"H and W must be multiples of 8."
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C*Hb*Wb, 8, 8)
        return x

    def forward_features(self, input):
        stream1_output = self.stream1_1(input)
        stream1_output = self.stream1_2(stream1_output)
        stream1_output = self.stream1_3(stream1_output)
        stream1_output = self.pool(stream1_output).view(stream1_output.size(0), -1)

        stream2_output = self._blockify_8x8(input)
        stream2_output = self.stream2_1(stream2_output)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.stream2_3(stream2_output)
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)

        output = torch.cat([stream1_output, stream2_output], dim=1)

        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)
        return x
