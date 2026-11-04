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

class Block1(nn.Module):
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
        self.act1 = nn.ReLU(inplace=True)

        self.freq_weight = nn.Parameter(torch.empty(out_channels, out_channels, 8, 8))
        nn.init.kaiming_uniform_(self.freq_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='relu')
        self.freq_bias = nn.Parameter(torch.zeros(out_channels, 8, 8)) if bias else None

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

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
        identity = input

        if self.proj is not None:
            identity = self.proj(identity)

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


class FreqAttentionPool(nn.Module):
    def __init__(self, act: str = "sigmoid"):
        super().__init__()
        assert act in ("softmax", "sigmoid")
        self.act = act

        self.freq_weight = nn.Parameter(torch.zeros(8, 8))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input.shape
        assert H % 8 == 0 and W % 8 == 0, "H,W must be multiple of 8"

        Hb, Wb = H // 8, W // 8

        input = input.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        output = input.mean(dim=(2, 3))

        w = self.freq_weight
        if self.act == "softmax":
            w = torch.softmax(w.view(-1), dim=0).view(1, 1, 8, 8)
        else:
            w = torch.sigmoid(w)
            w = w / (w.sum() + 1e-6)
            w = w.view(1, 1, 8, 8)

        output = (output * w).sum(dim=(2, 3))

        return output


class TempNet_v52(nn.Module):
    def __init__(self):
        super(TempNet_v52, self).__init__()
        self.block1 = Block1(3,64,pool=False)
        self.block2 = Block1(64,128)
        self.block3 = Block1(128,256)
        self.pool = FreqAttentionPool()

        self.fc = nn.Linear(256, 2)

    def forward_features(self, input):
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        output = self.pool(output)

        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)

        return x
