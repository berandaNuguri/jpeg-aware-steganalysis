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

class HPF(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False):
        super(HPF, self).__init__()
        self.intra_dct = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.inter_dct = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=8, padding=8, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_dct = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        intra = self.intra_dct(input)
        inter = self.inter_dct(input)

        output = torch.cat([intra, inter], dim=1)
        output = self.fuse_dct(output)
        
        return output  

class block1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
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
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
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

class TempNet_v61(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v61, self).__init__()
        self.relu = nn.ReLU()
        self.hpf = HPF()
        self.group1 = block1(64,64)
        self.group2 = block2(64,128)
        self.group3 = block2(128,256)
        self.group4 = block2(256,512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, 2)

    def forward_features(self, input):
        output = input
        output = self.hpf(output)
        output = self.group1(output)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.pool(output).view(output.size(0), -1)

        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        return output

    def forward(self, input):
        x = self.forward_features(input)
        x = self.fc2(x)
        return x
