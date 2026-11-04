import os
import sys
import numpy as np
import torch
import torch.nn as nn
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from .dctam import MultiSpectralAttentionLayer

def dct_basis(N=3):
    """N×N DCT basis 생성 -> (N*N, N, N)"""
    basis = []
    for u in range(N):
        for v in range(N):
            alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
            alpha_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
            B = np.zeros((N, N), dtype=np.float32)
            for x in range(N):
                for y in range(N):
                    B[x, y] = (
                        alpha_u * alpha_v *
                        np.cos(((2 * x + 1) * u * np.pi) / (2 * N)) *
                        np.cos(((2 * y + 1) * v * np.pi) / (2 * N))
                    )
            basis.append(B)
    return np.stack(basis, axis=0)


class HPF(nn.Module):
    def __init__(self, in_channels=3, kernel_sizes=[3, 8], bias=False):
        super(HPF, self).__init__()
        
        self.convs = nn.ModuleList()
        self.out_channels = 0

        for k in kernel_sizes:
            dct_kernels = dct_basis(k)           # (k*k, k, k)
            num_basis = dct_kernels.shape[0]     # k*k
            out_channels = in_channels * num_basis
            self.out_channels += out_channels

            # --- stride 결정 ---
            if k == 8:
                stride = 8
                padding = 0      
            else:
                stride = 1
                padding = "same"

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                stride=stride,
                padding=padding,
                bias=bias
            )

            # --- Conv weight 초기화 ---
            with torch.no_grad():
                weight = torch.zeros(out_channels, in_channels, k, k)
                for c in range(in_channels):
                    for i in range(num_basis):
                        weight[c * num_basis + i, c] = torch.from_numpy(dct_kernels[i])
                conv.weight.copy_(weight)

            self.convs.append(conv)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        return torch.cat(features, dim=1)
    
class cblock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(cblock1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )

    def forward(self,x):
        out=self.basic(x)
        return out

class cblock2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(cblock2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

    def forward(self,x):
        out=self.basic(x)
        return out

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_str = math.floor(previous_conv_size[0] / out_pool_size[i])
        w_str = math.floor(previous_conv_size[1] / out_pool_size[i])
        #maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_str, w_str))
        maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_str, w_str))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp


class TempNet_v10(nn.Module):
  def __init__(self, dropout=0.0, dct_h=32, dct_w=32, dct_reduction=16, dct_method='top16'):
    super(TempNet_v10, self).__init__()
    self.relu = nn.ReLU()
    self.output_num = [4,2,1]
    self.group1 = HPF(kernel_sizes=[8])
    # self.dct_attention = MultiSpectralAttentionLayer(
    #     channel=192,
    #     dct_h=dct_h,
    #     dct_w=dct_w,
    #     reduction=dct_reduction,
    #     freq_sel_method=dct_method,
    # )
    self.group2 = cblock1(192,64)
    self.group3 = cblock1(64,128)
    self.group4 = cblock2(128,256)
    self.fc1 = nn.Linear(5376, 1024)
    self.dropout = nn.Dropout(p=dropout)
    self.fc2 = nn.Linear(1024, 2)

  def forward_features(self, input):
    output = input
    output = self.group1(output)
    # output = self.dct_attention(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = spatial_pyramid_pool(output, output.size(0), [int(output.size(2)),int(output.size(3))], [4,2,1])
    output = self.fc1(output)
    output = self.relu(output)
    output = self.dropout(output)
    return output

  def forward(self, input):
    x = self.forward_features(input)
    x = self.fc2(x)
    return x
