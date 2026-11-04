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
        self.freq_weight = nn.Parameter(torch.empty(out_channels, in_channels, 8, 8))
        nn.init.kaiming_uniform_(self.freq_weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        self.freq_bias = nn.Parameter(torch.zeros(out_channels, 8, 8)) if bias else None
        self.dw_conv = nn.Conv2d(out_channels * 64, out_channels * 64, kernel_size=3, padding=1, groups=out_channels * 64, bias=False)
        self.pw_conv = nn.Conv2d(out_channels * 64, out_channels * 64, kernel_size=1, groups=64, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = pool
        if self.pool:
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

    @staticmethod
    def _collapse_freq(x):
        B, O, Hb, Wb, U, V = x.shape
        return x.permute(0, 1, 4, 5, 2, 3).contiguous().view(B, O * 64, Hb, Wb)

    @staticmethod
    def _expand_freq(x, out_channels):
        B, OC64, Hb, Wb = x.shape
        assert OC64 == out_channels * 64
        return x.view(B, out_channels, 8, 8, Hb, Wb).permute(0, 1, 4, 5, 2, 3).contiguous()

    def forward(self, input):
        input = self._blockify_8x8(input)
        input = torch.einsum('bcijuv,ocuv->boijuv', input, self.freq_weight)
        if self.freq_bias is not None:
            input = input + self.freq_bias.view(1, -1, 1, 1, 8, 8)

        input = self._collapse_freq(input)
        input = self.dw_conv(input)
        input = self.pw_conv(input)
        input = self._expand_freq(input, self.O)
        output = self._unblockify_8x8(input)
        output = self.bn(output)
        output = self.act(output)
        if self.pool:
            output = self.dctpool(output)
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

class TempNet_v49(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v49, self).__init__()
        self.block1 = Block1(3,64,pool=False)
        self.block2 = Block1(64,128)
        self.block3 = Block1(128,256)
        self.attn = Attention(256)

        self.fc = nn.Linear(256, 2)

    def forward_features(self, input):
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        output = self.attn(output)

        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)

        return x
