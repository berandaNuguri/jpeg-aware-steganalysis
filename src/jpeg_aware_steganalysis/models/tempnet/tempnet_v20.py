import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class HPF(nn.Module):
    def __init__(self, out_channels: int = 1, bias: bool = False):
        super().__init__()
        self.m = int(out_channels)

        self.channel_conv = nn.Conv2d(
            in_channels=3*64,
            out_channels=64*self.m,
            kernel_size=1,
            groups=64,
            bias=bias
        )

    @staticmethod
    def _to_dct(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Hb, Wb = H // 8, W // 8
        x_uv = x.view(B, C, 8, Hb, 8, Wb).permute(0, 1, 2, 4, 3, 5).contiguous()
        return x_uv.view(B, C * 64, Hb, Wb)

    @staticmethod
    def _to_spatial(y_grid: torch.Tensor, m: int, r: int = 8) -> torch.Tensor:
        B, Cm, Hb, Wb = y_grid.shape
        assert Cm % (r * r) == 0 and Cm // (r * r) == m
        y = y_grid.view(B, r, r, m, Hb, Wb)
        y = y.permute(0, 3, 4, 1, 5, 2).contiguous()
        return y.view(B, m, Hb * r, Wb * r)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3 and (H % 8 == 0) and (W % 8 == 0), "Input must be (B,3,H,W) with H,W divisible by 8."
        output = self._to_dct(x)
        output = self.channel_conv(output)
        output = self._to_spatial(output, self.m, r=8)

        return output

class Conv_PDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.5):

        super(Conv_PDC, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,8), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:,:,:,0], self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], self.conv.weight[:,:,:,5], self.conv.weight[:,:,:,6], self.conv.weight[:,:,:,7]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class block1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out

class block2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                Conv_PDC(outchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
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


class TempNet_v20(nn.Module):
  def __init__(self, dropout=0.0):
    super(TempNet_v20, self).__init__()
    self.relu = nn.ReLU()
    self.output_num = [4,2,1]
    self.group1 = HPF(out_channels=64)
    self.group2 = block1(64,64)
    self.group3 = block2(64,128)
    self.group4 = block2(128,256)
    self.fc1 = nn.Linear(5376, 1024)
    self.dropout = nn.Dropout(p=dropout)
    self.fc2 = nn.Linear(1024, 2)

  def forward_features(self, input):
    output = input
    output = self.group1(output)
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
