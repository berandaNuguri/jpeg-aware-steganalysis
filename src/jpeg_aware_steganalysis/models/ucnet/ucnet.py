import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F
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

        self.tlu = TLU(2.0)  # T= 2

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)

        return output  

    
class Type1a(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type1a, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                
                nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
                )

    def forward(self,x):
        out=self.basic(x)
        return out
      
      
class Type1b(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type1b, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(),
                
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
                )

    def forward(self,x):
        out=self.basic(x)

        return out
      
    
class Type2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(),
                
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
            
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=2),
                nn.BatchNorm2d(outchannel),
                )
    def forward(self,x):
        out=self.basic(x)
        out+=self.shortcut(x)
        out=self.relu(out)

        return out
        
        
class Type3(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type3, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, groups=32, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(outchannel),
                )
    def forward(self,x):
        out=self.basic(x)
        out+=self.shortcut(x)
        out=self.relu(out)
        return out
    
        
class UCNet(nn.Module):
  def __init__(self, num_classes=2):
    super(UCNet, self).__init__()
    
    self.pre = HPF()

    self.group1 = Type1a(186,32)
    self.group2 = Type2(32,32)
    self.group3 = Type3(32,64)
    self.group4 = Type2(64,128)
    self.group5 = Type1b(128,256)
    
    # self.avg = nn.AvgPool2d(kernel_size=32, stride=1)
    self.avg = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(1 * 1 * 256, num_classes)

  def forward_features(self, input):
    output = input

    output_c1 = output[:, 0, :, :]
    output_c2 = output[:, 1, :, :] 
    output_c3 = output[:, 2, :, :] 
    out_c1 = output_c1.unsqueeze(1)
    out_c2 = output_c2.unsqueeze(1)
    out_c3 = output_c3.unsqueeze(1)
    c1 = self.pre(out_c1)
    c2 = self.pre(out_c2)
    c3 = self.pre(out_c3)
    output = torch.cat([c1, c2, c3], dim=1)
    
    output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.group5(output)
    
    output = self.avg(output)
    output = output.view(output.size(0), -1)

    return output

  def forward(self, input):
    output = input
    # seperate color channels
    output_c1 = output[:, 0, :, :]
    output_c2 = output[:, 1, :, :] 
    output_c3 = output[:, 2, :, :] 
    out_c1 = output_c1.unsqueeze(1)
    out_c2 = output_c2.unsqueeze(1)
    out_c3 = output_c3.unsqueeze(1)
    c1 = self.pre(out_c1)
    c2 = self.pre(out_c2)
    c3 = self.pre(out_c3)
    output = torch.cat([c1, c2, c3], dim=1)
    
    output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.group5(output)
    
    output = self.avg(output)
    output = output.view(output.size(0), -1)
    output = self.fc1(output)

    return output