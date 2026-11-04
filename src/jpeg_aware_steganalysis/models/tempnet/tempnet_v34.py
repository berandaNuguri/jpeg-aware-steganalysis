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
        assert factor >= 2, "factor>=2"
        assert mode in ('avg','max')
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
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, dilation=8, apply_pool=False,bias=False):
        super(FrequencyAwareConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.apply_pool = apply_pool
        if self.apply_pool:
            self.pool = DCTAwarePool(2, 'avg')

    def forward(self, input):
        output = self.conv(input)
        if self.apply_pool:
            output = self.pool(output)
        return output


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

class FreqPosBias(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        grid_u, grid_v = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        freq_energy = 1.0 / (1.0 + grid_u.float()**2 + grid_v.float()**2)
        freq_energy = freq_energy / freq_energy.max()

        freq_energy = freq_energy.flatten()
        self.bias = nn.Parameter(freq_energy.view(1, 1, 1, size * size))

    def forward(self, x):
        assert x.shape[-1] == self.bias.shape[-1], \
            f"FreqPosBias expects last dim={self.bias.shape[-1]}, got {x.shape[-1]}"
        return x + self.bias

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

class TempNet_v34(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v34, self).__init__()
        self.stream1_1 = FrequencyAwareConv(3, 64, dilation=8, apply_pool=False)
        self.stream1_2 = FrequencyAwareConv(64, 128, dilation=8, apply_pool=True)
        self.stream1_3 = FrequencyAwareConv(128, 256, dilation=4, apply_pool=True)
        self.stream1_4 = FrequencyAwareConv(256, 256, dilation=2, apply_pool=True)

        self.dct_bias = FreqPosBias()
        self.stream2_1 = DCTBlockConv(3, 64, apply_pool=False, num_blocks=1024)
        self.stream2_2 = DCTBlockConv(64, 128, apply_pool=True, num_blocks=1024)
        self.stream2_3 = DCTBlockConv(128, 256, apply_pool=True, num_blocks=512)
        self.stream2_4 = DCTBlockConv(256, 256, apply_pool=True, num_blocks=256)
        
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fusion = StreamFusion(256, 256, 512)
        self.fc = nn.Linear(512, 2)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape

        assert H % 8 == 0 and W % 8 == 0, f"H and W must be multiples of 8."
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, Hb*Wb, 8*8)
        return x
    
    def forward_features(self, input):
        stream1_output = self.stream1_1(input)
        stream1_output = self.stream1_2(stream1_output)
        stream1_output = self.stream1_3(stream1_output)
        stream1_output = self.stream1_4(stream1_output)
        stream1_output = self.pool(stream1_output).view(stream1_output.size(0), -1)

        stream2_output = self._blockify_8x8(input)
        stream2_output = self.dct_bias(stream2_output)
        stream2_output = self.stream2_1(stream2_output)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.stream2_3(stream2_output)
        stream2_output = self.stream2_4(stream2_output)
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)
        
        output = self.fusion(stream1_output, stream2_output)

        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)
        return x
