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
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 8, 8))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 8, 8))
        else:
            self.bias = None

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # y[b,o,u,v] = sum_c x[b,c,u,v] * W[o,c,u,v] + b[o,u,v]
        y = torch.einsum('bcuv,ocuv->bouv', x, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)

        y = self.bn(y)
        y = self.act(y)
        return y

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
        self.fc = nn.Sequential(
            nn.Linear(dim1 + dim2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)

class TempNet_v47(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v47, self).__init__()
        self.relu = nn.ReLU()

        self.stream1_1 = Stream1_Block1(3,32)
        self.stream1_2 = Stream1_Block1(32,64)
        self.stream1_3 = Stream1_Block2(64,128)

        self.stream2_1 = Stream2_Block1(3,32)
        self.stream2_2 = Stream2_Block1(32,64)
        self.stream2_3 = Stream2_Block1(64,128)
        self.stream2_4 = Attention(128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = StreamFusion(128, 128, 256)

        self.fc = nn.Linear(256, 2)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0, f"H({H}), W({W}) must be multiples of 8"
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        return x

    def forward_features(self, input):
        stream1_output = self.stream1_1(input)
        stream1_output = self.stream1_2(stream1_output)
        stream1_output = self.stream1_3(stream1_output)
        stream1_output = self.pool(stream1_output).view(stream1_output.size(0), -1)

        input = self._blockify_8x8(input)
        B, C, Hb, Wb, _, _ = input.shape
        input = input.view(B * Hb * Wb, C, 8, 8)
        stream2_output = self.stream2_1(input)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.stream2_3(stream2_output)
        stream2_output = self.stream2_4(stream2_output)
        stream2_output = stream2_output.view(B, Hb, Wb, -1).permute(0, 3, 1, 2).contiguous()
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)

        output = self.fusion(stream1_output, stream2_output)
        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)

        return x
