import os
import sys
from numpy import False_
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.onnx.symbolic_opset9 import index_put

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

class Stream1_Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stream1_Block1, self).__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
        )

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
    def __init__(self, in_channels=3, out_channels=64, bias=False, pool=True):
        super().__init__()
        self.O = out_channels
        self.C = in_channels
        self.pool = pool

        self.conv = nn.Conv2d(
            in_channels * 64,
            out_channels * 64,
            kernel_size=3,
            padding=1,
            groups=64,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels * 64)
        self.act1 = nn.ReLU()

        self.freq_weight = nn.Parameter(torch.empty(out_channels, out_channels, 8, 8))
        nn.init.kaiming_uniform_(self.freq_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='relu')
        self.freq_bias = nn.Parameter(torch.zeros(out_channels, 8, 8)) if bias else None

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

        if self.pool:
            self.dctpool = DCTAwarePool(2, 'avg')

        use_proj = (in_channels != out_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) \
                    if use_proj else None

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0
        Hb, Wb = H // 8, W // 8
        return x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()

    @staticmethod
    def _unblockify_8x8(x):
        B, O, Hb, Wb, U, V = x.shape
        assert U == 8 and V == 8
        return x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, O, Hb * 8, Wb * 8)

    @staticmethod
    def _collapse_freq(x):
        B, C, Hb, Wb, U, V = x.shape
        return x.permute(0, 1, 4, 5, 2, 3).contiguous().view(B, C * 64, Hb, Wb)

    @staticmethod
    def _expand_freq(x, channels):
        B, C64, Hb, Wb = x.shape
        assert C64 == channels * 64
        return x.view(B, channels, 8, 8, Hb, Wb).permute(0, 1, 4, 5, 2, 3).contiguous()

    def forward(self, input):
        identity = self.proj(input)
        if self.pool:
            identity = self.dctpool(identity)

        output = self._blockify_8x8(input)
        output = self._collapse_freq(output)

        output = self.conv(output)
        output = self.bn1(output)
        output = self.act1(output)

        output = self._expand_freq(output, self.O)

        output = torch.einsum('bcijuv,ocuv->boijuv', output, self.freq_weight)
        if self.freq_bias is not None:
            output = output + self.freq_bias.view(1, -1, 1, 1, 8, 8)

        output = self._unblockify_8x8(output)

        if self.pool:
            output = self.dctpool(output)

        output = output + identity
        output = self.bn2(output)
        output = self.act2(output)

        return output

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

class TempNet_v55(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v55, self).__init__()
        self.relu = nn.ReLU()

        self.stream1_1 = Stream1_Block1(3,32)
        self.stream1_2 = Stream1_Block2(32,64)
        self.stream1_3 = Stream1_Block2(64,128)

        self.stream2_1 = Stream2_Block1(3,32, pool=False)
        self.stream2_2 = Stream2_Block1(32,64)
        self.stream2_3 = Stream2_Block1(64,128)

        self.film1 = SpatialFrequencyCrossFiLM(32, 32)
        self.film2 = SpatialFrequencyCrossFiLM(64, 64)
        self.film3 = SpatialFrequencyCrossFiLM(128, 128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = StreamFusion(128, 128, 256)

        self.fc = nn.Linear(256, 2)


    def forward_features(self, spatial, dct):
        stream1_output = self.stream1_1(spatial)
        stream2_output = self.stream2_1(dct)
        stream1_output, stream2_output = self.film1(stream1_output, stream2_output)

        stream1_output = self.stream1_2(stream1_output)
        stream2_output = self.stream2_2(stream2_output)
        stream1_output, stream2_output = self.film2(stream1_output, stream2_output)

        stream1_output = self.stream1_3(stream1_output)
        stream2_output = self.stream2_3(stream2_output)
        stream1_output, stream2_output = self.film3(stream1_output, stream2_output)

        stream1_output = self.pool(stream1_output).view(stream1_output.size(0), -1)
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)

        output = self.fusion(stream1_output, stream2_output)
        return output

    def forward(self, spatial, dct):
        output = self.forward_features(spatial, dct)
        output = self.fc(output)

        return output
