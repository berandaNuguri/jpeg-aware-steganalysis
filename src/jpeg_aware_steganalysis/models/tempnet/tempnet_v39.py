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
    def __init__(self, stride: int = 2, mode: str = 'avg'):
        super().__init__()
        self.stride = stride
        self.mode = mode
        self.unshuffle = nn.PixelUnshuffle(8)
        self.shuffle = nn.PixelShuffle(8)
        if mode == 'avg':
            self.grid_pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        else:
            self.grid_pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if (H % 8) != 0 or (W % 8) != 0:
            raise ValueError(f"DCTGridPool: H,W must be multiples of block_size=8, got {H}x{W}")

        z = self.unshuffle(x)
        z = self.grid_pool(z)
        y = self.shuffle(z)

        return y

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

class TempNet_v39(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v39, self).__init__()
        self.relu = nn.ReLU()

        self.stream1_1 = Stream1_Block1(3,32)
        self.stream1_2 = Stream1_Block1(32,64)
        self.stream1_3 = Stream1_Block2(64,128)

        self.stream2_1 = Stream2_Block1(3,32,8)
        self.stream2_2 = Stream2_Block1(32,64,8)
        self.stream2_3 = Stream2_Block2(64,128,8)

        self.spp_params = [4,2,1]
        self.fc = nn.Linear(2688, 2)

    def forward_features(self, input):
        stream1_output = self.stream1_1(input)
        stream1_output = self.stream1_2(stream1_output)
        stream1_output = self.stream1_3(stream1_output)

        stream2_output = self.stream2_1(input)
        stream2_output = self.stream2_2(stream2_output)
        stream2_output = self.stream2_3(stream2_output)

        output = stream1_output + stream2_output

        output = spatial_pyramid_pool(output, output.size(0), [int(output.size(2)),int(output.size(3))], self.spp_params)
        
        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc(x)

        return x
