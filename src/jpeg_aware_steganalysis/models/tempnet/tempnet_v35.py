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


class DCTFreqConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, bias=False):
        super(DCTFreqConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DCTBlockConv(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, stride=1, bias=False):
        super(DCTBlockConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TempNet_v35(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v35, self).__init__()
        self.stream1_1 = DCTFreqConv(3, 64)
        self.stream1_2 = DCTFreqConv(64, 128)
        
        self.stream2_1 = DCTBlockConv(128, 256)
        self.stream2_2 = DCTBlockConv(256, 512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0, f"H({H}), W({W}) must be multiples of 8"
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        return x

    def forward_features(self, input):
        input = self._blockify_8x8(input)
        B, C, Hb, Wb, _, _ = input.shape
        input = input.view(B * Hb * Wb, C, 8, 8)
        stream1_output = self.stream1_1(input)
        stream1_output = self.stream1_2(stream1_output)
        stream1_output = self.pool(stream1_output)

        stream2_output = stream1_output.view(B, Hb, Wb, -1).permute(0, 3, 1, 2).contiguous()
        stream2_output = self.stream2_1(stream2_output)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.pool(stream2_output)

        output = stream2_output.flatten(1)
        return output

    def forward(self, input):
        output = self.forward_features(input)
        output = self.fc(output)
        return output