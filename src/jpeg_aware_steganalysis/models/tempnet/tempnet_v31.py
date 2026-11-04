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

class LearnableTopKSelector(nn.Module):
    def __init__(self, num_blocks=1024, factor: int = 2, tau: float = 1.0):
        super().__init__()
        assert factor >= 2
        self.factor = factor
        self.tau = tau
        self.logits = nn.Parameter(torch.empty(num_blocks)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.factor == 0, f"H({H}) must be divisible by factor({self.factor})"
        K = H // self.factor

        idx = torch.topk(self.logits, k=K, dim=0, largest=True, sorted=True).indices
        y_hard = x.index_select(dim=2, index=idx)

        probs_all = torch.softmax(self.logits / self.tau, dim=0)
        probs_kept = probs_all.index_select(dim=0, index=idx)

        y_soft = y_hard * probs_kept.view(1, 1, K, 1)

        y = y_soft + (y_hard - y_soft).detach()
        return y

class DCTBlockConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, stride=1, apply_pool=False, num_blocks=1024, bias=False):
        super(DCTBlockConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), padding=(0,1), bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.apply_pool = apply_pool
        if self.apply_pool:
            self.pool = LearnableTopKSelector(num_blocks=num_blocks, factor=2, tau=1.0)


    def forward(self, input):
        output = self.conv(input)
        if self.apply_pool:
            output = self.pool(output)
        return output


class TempNet_v31(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v31, self).__init__()
        self.cblock1 = DCTBlockConv(3, 64, apply_pool=False, num_blocks=1024)
        self.cblock2 = DCTBlockConv(64, 128, apply_pool=True, num_blocks=1024)
        self.cblock3 = DCTBlockConv(128, 256, apply_pool=True, num_blocks=512)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, 2)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape

        assert H % 8 == 0 and W % 8 == 0, f"H and W must be multiples of 8."
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, Hb*Wb, 8*8)
        return x

    def forward_features(self, input):
        output = self._blockify_8x8(input)
        output = self.cblock1(output)
        output = self.cblock2(output)
        output = self.cblock3(output)
        output = self.pool(output).view(output.size(0), -1)

        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)
        return x
