import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math

from .srm_kernel_filters import all_normalized_hpf_list

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


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


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        filt_list = build_filters()

        hpf_weight = nn.Parameter(torch.Tensor(filt_list).view(62, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 62, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        self.tlu = TLU(5.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)

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


class pdcblock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                # Default Code
                # Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1, stride=2),
                
                Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out

class pdcblock1_v2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock1_v2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                # Default Code
                # Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1, stride=2),
                
                Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                )

    def forward(self,x):
        out=self.basic(x)
        return out

class pdcblock2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock2, self).__init__()
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


class TempNet_v2(nn.Module):
  def __init__(self, dropout=0.0):
    super(TempNet_v2, self).__init__()
    self.relu = nn.ReLU()
    self.output_num = [4,2,1]
    self.group1 = HPF()
    self.group2 = pdcblock1(186,64)
    self.group3 = pdcblock2(64,128)
    self.group4 = pdcblock2(128,256)
    self.dropout = nn.Dropout(p=dropout)
    self.fc1 = nn.Linear(5376, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward_features(self, input):
    output = input
    output_y = output[:, 0, :, :]
    output_u = output[:, 1, :, :] 
    output_v = output[:, 2, :, :] 
    out_y = output_y.unsqueeze(1)
    out_u = output_u.unsqueeze(1)
    out_v = output_v.unsqueeze(1)
    y = self.group1(out_y)
    u = self.group1(out_u)
    v = self.group1(out_v)
    output = torch.cat([y, u, v], dim=1)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    spp = spatial_pyramid_pool(output, output.size(0), [int(output.size(2)),int(output.size(3))], [4,2,1])
    x = self.fc1(spp)
    x = self.relu(x)
    x = self.dropout(x)
    return x

  def forward(self, input):
    x=self.forward_features(input)
    x = self.fc2(x)
    return x

