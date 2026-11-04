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


class Stream2_Block1(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation=8):
        super(Stream2_Block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.pool = DCTAwarePool(2, 'avg')

    def forward(self, input):
        output = self.conv(input)
        output = self.pool(output)
        return output

class Stream2_Block2(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation=8):
        super(Stream2_Block2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.pool = DCTAwarePool(2, 'avg')

    def forward(self, input):
        output = self.conv(input)
        output = self.pool(output)
        return output

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

class TempNet_v45(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v45, self).__init__()
        self.relu = nn.ReLU()

        self.stream1_1 = Stream1_Block1(3,32)
        self.stream1_2 = Stream1_Block1(32,64)
        self.stream1_3 = Stream1_Block2(64,128)

        self.stream2_1 = Stream2_Block1(3,32,8)
        self.stream2_2 = Stream2_Block1(32,64,8)
        self.stream2_3 = Stream2_Block2(64,128,8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = StreamFusion(128, 128, 256)
        self.fc = nn.Linear(256, 2)

    def forward_features(self, input):
        stream1_output = self.stream1_1(input)
        stream1_output = self.stream1_2(stream1_output)
        stream1_output = self.stream1_3(stream1_output)
        stream1_output = self.pool(stream1_output).view(stream1_output.size(0), -1)

        stream2_output = self.stream2_1(input)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.stream2_3(stream2_output)
        stream2_output = self.pool(stream2_output).view(stream2_output.size(0), -1)
        
        output = self.fusion(stream1_output, stream2_output)
        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)

        return x
