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

class DCTBlockConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, apply_pool=False, bias=False):
        super(DCTBlockConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.apply_pool = apply_pool
        if self.apply_pool:
            self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)


    def forward(self, input):
        output = self.conv(input)
        if self.apply_pool:
            output = self.pool(output)
        return output


class TempNet_v28(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v28, self).__init__()
        self.stream2_1 = DCTBlockConv(3072, 1536, apply_pool=False)
        self.stream2_2 = DCTBlockConv(1536, 768, apply_pool=True)
        self.stream2_3 = DCTBlockConv(768, 384, apply_pool=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, 2)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape

        assert H % 8 == 0 and W % 8 == 0, f"H and W must be multiples of 8."
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C*Hb*Wb, 8, 8)
        return x

    def forward_features(self, input):
        stream2_output = self._blockify_8x8(input)
        stream2_output = self.stream2_1(stream2_output)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.stream2_3(stream2_output)
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)

        return stream2_output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)
        return x
