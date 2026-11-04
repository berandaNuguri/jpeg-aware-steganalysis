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

class CrossFiLM(nn.Module):
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

class Stream1_Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stream1_Block1, self).__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
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

class Stream2_Block1(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, bias=False):
        super().__init__()
        self.freq_weight = nn.Parameter(torch.empty(out_channels, in_channels, 8, 8))
        nn.init.kaiming_uniform_(self.freq_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='relu')
        self.freq_bias = nn.Parameter(torch.zeros(out_channels, 8, 8)) if bias else None

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) 

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.dctpool = DCTAwarePool(2, 'avg')

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

    def forward(self, input):
        identity = self.proj(input)
        identity = self.dctpool(identity)

        output = self._blockify_8x8(input)

        output = torch.einsum('bcijuv,ocuv->boijuv', output, self.freq_weight)
        if self.freq_bias is not None:
            output = output + self.freq_bias.view(1, -1, 1, 1, 8, 8)

        output = self._unblockify_8x8(output)
        output = self.dctpool(output)

        output = output + identity
        output = self.bn(output)
        output = self.act(output)

        return output

class Attention(nn.Module):
    def __init__(self, channels=256, act="sigmoid"):
        super().__init__()
        assert act in ("softmax", "sigmoid")
        self.act = act

        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        w = self.proj(x)
        if self.act == "softmax":
            w = torch.softmax(w.view(B, 1, -1), dim=2).view(B, 1, H, W)
        else:
            w = torch.sigmoid(w)
            w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-6)

        y = (x * w).sum(dim=(2, 3))

        return y

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

class TempNet_v54(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v54, self).__init__()
        self.relu = nn.ReLU()

        self.stream1_1 = Stream1_Block1(3,64)
        self.stream1_2 = Stream1_Block1(64,128)
        self.stream1_3 = Stream1_Block2(128,256)

        self.stream2_1 = Stream2_Block1(3,64)
        self.stream2_2 = Stream2_Block1(64,128)
        self.stream2_3 = Stream2_Block1(128,256)

        self.cross1 = CrossFiLM(channels=64, hidden=64)
        self.cross2 = CrossFiLM(channels=128, hidden=128)
        self.cross3 = CrossFiLM(channels=256, hidden=256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = StreamFusion(256, 256, 512)

        self.fc = nn.Linear(512, 2)

    def forward_features(self, input):
        s1 = self.stream1_1(input)
        s2 = self.stream2_1(input) 
        s1, s2 = self.cross1(s1, s2)

        s1 = self.stream1_2(s1)
        s2 = self.stream2_2(s2)
        s1, s2 = self.cross2(s1, s2)

        s1 = self.stream1_3(s1)
        s2 = self.stream2_3(s2)
        s1, s2 = self.cross3(s1, s2)

        stream1_output = self.pool(s1).view(s1.size(0), -1)
        stream2_output = self.pool(s2).view(s2.size(0), -1)

        output = self.fusion(stream1_output, stream2_output)
        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)
        return x
