import os
import sys
from numpy import False_
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .srm_kernel_filters import all_normalized_hpf_list

def build_filters():
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in all_normalized_hpf_list:
       row_1 = int((5 - hpf_item.shape[0]) / 2)
       row_2 = int((5 - hpf_item.shape[0]) - row_1)
       col_1 = int((5 - hpf_item.shape[1]) / 2)
       col_2 = int((5 - hpf_item.shape[1]) - col_1)
       hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
       filters.append(hpf_item)
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel((ksize[0], ksize[0]), sigma[k], theta, sigma[k] / 0.56, 0.5, phi[j],
                                          ktype=cv2.CV_32F)
                filters.append(kern)
    return filters

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output

class HPF(nn.Module):
    def __init__(self, in_channels=3):
        super(HPF, self).__init__()

        filt_list = build_filters()
        base_weight = torch.tensor(filt_list, dtype=torch.float32).view(62, 1, 5, 5)

        if in_channels == 1:
            hpf_weight = nn.Parameter(base_weight, requires_grad=False)
            self.hpf = nn.Conv2d(1, 62, kernel_size=5, padding=2, bias=False)
            self.hpf.weight = hpf_weight
            out_channels = 62

        elif in_channels == 3:
            weight_3ch = base_weight.repeat(3, 1, 1, 1)
            hpf_weight = nn.Parameter(weight_3ch, requires_grad=False)

            self.hpf = nn.Conv2d(
                in_channels=3,
                out_channels=62 * 3,
                kernel_size=5,
                padding=2,
                bias=False,
                groups=3
            )
            self.hpf.weight = hpf_weight
            out_channels = 62 * 3

        else:
            raise ValueError("HPF only supports in_channels=1 or 3 for now.")

        self.out_channels = out_channels
        self.tlu = TLU(5.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)
        return output
        
class Stream1_Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stream1_Block1, self).__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

    def forward(self,x):
        out=self.conv(x)
        return out

class Stream1_Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stream1_Block2, self).__init__()
        self.basic=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out

class DCTAwarePool(nn.Module):
    def __init__(self, factor: int = 2, mode: str = 'avg'):
        super().__init__()
        assert mode in ('avg', 'max')
        self.factor = factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0, "Input must be multiple of 8"

        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8)

        if self.mode == 'avg':
            x = x.view(B, C, Hb // self.factor, self.factor, 8,
                       Wb // self.factor, self.factor, 8).mean(dim=(3, 6))
        else:
            x, _ = x.view(B, C, Hb // self.factor, self.factor, 8,
                          Wb // self.factor, self.factor, 8).max(dim=(3, 6))

        x = x.view(B, C, (Hb // self.factor) * 8, (Wb // self.factor) * 8)
        return x

class Stream2_Block1(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation=8):
        super(Stream2_Block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.conv(input)
        return output

class Stream2_Block2(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation=8):
        super(Stream2_Block2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dctpool = DCTAwarePool(2, 'avg')

    def forward(self, input):
        output = self.conv(input)
        output = self.dctpool(output)
        return output

class SpatialFrequencyCrossFiLM(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.mlp_1to2 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2, bias=False)
        )

        self.mlp_2to1 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2, bias=False)
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        assert x2.shape == x1.shape, "x1, x2 must have same shape"

        g1 = x1.mean(dim=(2, 3))
        g2 = x2.mean(dim=(2, 3))

        gb2 = self.mlp_1to2(g1)
        gamma2, beta2 = gb2.chunk(2, dim=1)

        gb1 = self.mlp_2to1(g2)
        gamma1, beta1 = gb1.chunk(2, dim=1)

        gamma1 = torch.sigmoid(gamma1).view(B, C, 1, 1)
        gamma2 = torch.sigmoid(gamma2).view(B, C, 1, 1)
        beta1  = beta1.view(B, C, 1, 1)
        beta2  = beta2.view(B, C, 1, 1)

        x1_out = gamma1 * x1 + beta1
        x2_out = gamma2 * x2 + beta2

        return x1_out, x2_out

class StreamFusion(nn.Module):
    def __init__(self, dim1: int, dim2: int, out_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(dim1, out_dim)
        self.proj2 = nn.Linear(dim2, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        h1 = self.proj1(f1)
        h2 = self.proj2(f2)

        g = self.gate(torch.cat([h1, h2], dim=1))

        fused = g * h1 + (1 - g) * h2

        return self.out(fused)

class TempNet_v59(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v59, self).__init__()
        self.relu = nn.ReLU()
        self.stream1_1 = HPF()
        self.stream1_2 = Stream1_Block1(186,64)
        self.stream1_3 = Stream1_Block2(64,128)

        self.stream2_1 = Stream2_Block1(3,64)
        self.stream2_2 = Stream2_Block2(64,128)

        self.film1 = SpatialFrequencyCrossFiLM(64, 64)
        self.film2 = SpatialFrequencyCrossFiLM(128, 128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = StreamFusion(128, 128, 256)

        self.fc = nn.Linear(256, 2)

    def forward_features(self, spatial, dct):
        stream1_output = self.stream1_1(spatial)

        stream1_output = self.stream1_2(stream1_output)
        stream2_output = self.stream2_1(dct)
        stream1_output, stream2_output = self.film1(stream1_output, stream2_output)

        stream1_output = self.stream1_3(stream1_output)
        stream2_output = self.stream2_2(stream2_output)
        stream1_output, stream2_output = self.film2(stream1_output, stream2_output)

        stream1_output = self.pool(stream1_output).view(stream1_output.size(0), -1)
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)

        output = self.fusion(stream1_output, stream2_output)
        return output

    def forward(self, spatial, dct):
        output = self.forward_features(spatial, dct)
        output = self.fc(output)

        return output
