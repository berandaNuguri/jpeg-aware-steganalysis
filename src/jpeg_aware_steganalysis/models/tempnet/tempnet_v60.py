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
        
class Stream1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stream1, self).__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

    def forward(self,x):
        out=self.conv(x)
        return out

class Stream2(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation=8):
        super(Stream2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1, self).__init__()
        self.basic=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()
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

class TempNet_v60(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v60, self).__init__()
        self.relu = nn.ReLU()
        self.hpf = HPF()
        self.stream1 = Stream1(186,32)
        self.stream2 = Stream2(3,32)

        self.conv1 = ConvBlock1(64,128)
        self.conv2 = ConvBlock2(128,256)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, 2)

    def forward_features(self, spatial, dct):
        spatial_output = self.hpf(spatial)
        spatial_output = self.stream1(spatial_output)

        dct_output = self.stream2(dct)

        output = torch.cat([spatial_output, dct_output], dim=1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.pool(output).view(output.size(0), -1)
        return output

    def forward(self, spatial, dct):
        output = self.forward_features(spatial, dct)
        output = self.fc(output)

        return output
