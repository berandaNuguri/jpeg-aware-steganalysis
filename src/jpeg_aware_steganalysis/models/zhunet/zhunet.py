from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = str(RESOURCE_ROOT / "zhunet")

SRM_npy1 = np.load(os.path.join(kernel_dir, "SRM3_3.npy"))
SRM_npy2 = np.load(os.path.join(kernel_dir, "SRM5_5.npy"))

class pre_Layer_3_3(nn.Module):
    def __init__(self, stride=1, padding=1):
        super(pre_Layer_3_3, self).__init__()
        self.in_channels = 1
        self.out_channels = 25
        self.kernel_size = (3, 3)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(25, 1, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy1
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

class pre_Layer_5_5(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(pre_Layer_5_5, self).__init__()
        self.in_channels = 1
        self.out_channels = 5
        self.kernel_size = (5, 5)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(5, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(5), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy2
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


# class spp_layer(nn.Module):

#     def __init__(self, batch_size):
#         super(spp_layer, self).__init__()

#         self.batch_size = batch_size

#     def forward(self, x):
#         temp = x
#         a, b, c, d = x.size()

#         x = F.avg_pool2d(x, (c, d), c)

#         spp = x.view(self.batch_size * 2, -1)

#         x = temp
#         x = F.avg_pool2d(x, int(c / 2), int(c / 2))
#         spp = torch.cat((spp, x.view(self.batch_size * 2, -1)), 1)

#         x = temp
#         x = F.avg_pool2d(x, (int(c / 4)), int(c / 4))
#         spp = torch.cat((spp, x.view(self.batch_size * 2, -1)), 1)

#         return spp
class spp_layer(nn.Module):
    def __init__(self, levels=(1, 2, 4)):
        super(spp_layer, self).__init__()
        self.levels = levels

    def forward(self, x):
        B, C, H, W = x.size()
        spp_feats = []

        for l in self.levels:
            pooled = F.adaptive_avg_pool2d(x, output_size=(l, l))
            spp_feats.append(pooled.view(B, -1))

        spp = torch.cat(spp_feats, dim=1)
        return spp

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_global):
        super(Basic_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.use_global = use_global
        self.pool = nn.AvgPool2d(5, 2, 2)

    def forward(self, x):

        if self.use_global:
            x = self.act(self.bn(self.conv(x)))
        else:
            x = self.pool(self.act(self.bn(self.conv(x))))

        return x

class pre_layer(nn.Module):
    def __init__(self):
        super(pre_layer, self).__init__()
        self.conv1 = pre_Layer_3_3()
        self.conv2 = pre_Layer_5_5()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        return torch.cat((x1, x2),1) 

class conv_Layer(nn.Module):
    def __init__(self):
        super(conv_Layer, self).__init__()
        self.conv1 = nn.Conv2d(30, 60, 3, 1, 1,groups=30)
        self.conv1_1 = nn.Conv2d(60, 30, 1)
        self.bn1 = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 60, 3, 1, 1, groups=30)
        self.conv2_1 = nn.Conv2d(60, 30, 1)
        self.bn2 = nn.BatchNorm2d(30)
        self.conv_layer = nn.Sequential(
            Basic_Block(30, 32, 3, 1, 1, False),
            Basic_Block(32, 32, 3, 1, 1, False),
            Basic_Block(32, 64, 3, 1, 1, False),
            Basic_Block(64, 128, 3, 1, 1, True)
        )

        self.spp = spp_layer((1,2,4))
        # self.spp = spp_layer(1)
        self.classfier = nn.Sequential(
            # nn.Linear(2700, 1024),
            nn.Linear(2688, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        temp = x
        x = self.conv1(x).abs()
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x += temp

        x = F.relu(x, inplace=True)
        x = self.conv_layer(x)
        x = self.spp(x)
        x = self.classfier(x)
        return x


class ZhuNet(nn.Module):
    def __init__(self):
        super(ZhuNet, self).__init__()

        self.layer1 = pre_layer()
        self.layer2 = conv_Layer()

    def forward_single(self, x_1ch):
        x = self.layer1(x_1ch)
        x = self.layer2(x)
        return x

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {x.shape}")

        B, C, H, W = x.shape

        if C == 1:
            return self.forward_single(x)

        logits_sum = None
        for c in range(C):
            x_c = x[:, c:c+1, :, :]
            logits_c = self.forward_single(x_c)
            
            if logits_sum is None:
                logits_sum = logits_c
            else:
                logits_sum = logits_sum + logits_c

        logits_mean = logits_sum / float(C)
        return logits_mean