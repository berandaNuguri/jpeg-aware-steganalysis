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

# ---------------------------
# 새로 추가: 대역별 위치 리스트
# ---------------------------
def make_band_index(size=8):
    """
    8x8 내 (u,v) 위치를 대역별로 분리.
    - DC: (0,0)
    - Low: 1 <= u+v <= 3   (DC 제외)
    - Mid: 4 <= u+v <= 6
    - High: else
    return dict: {'dc':[(0,0)], 'low':[(u,v)...], 'mid':..., 'high':...}
    """
    dc, low, mid, high = [], [], [], []
    for u in range(size):
        for v in range(size):
            s = u + v
            if u == 0 and v == 0:
                dc.append((u, v))
            elif 1 <= s <= 3:
                low.append((u, v))
            elif 4 <= s <= 6:
                mid.append((u, v))
            else:
                high.append((u, v))
    return {'dc': dc, 'low': low, 'mid': mid, 'high': high}


# ---------------------------------
# 새로 추가: 주파수 대역별 피처 생성기
# ---------------------------------
class DCTBandProject(nn.Module):
    """
    입력 (B,3,H,W)에서 8x8 블록 그리드(H/8,W/8)로 내려가
    대역별 (u,v) 위치의 계수를 채널로 쌓아 반환.

    out: dict with tensors of shape (B, 3 * n_pos_band, Hb, Wb)
    """
    def __init__(self, size=8):
        super().__init__()
        self.size = size
        self.bands = make_band_index(size)

    def _gather_positions(self, x, pos_list):
        """
        x: (B, 3, H, W), H,W multiple of 8
        pos_list: list[(u,v)]
        returns (B, 3*len(pos_list), Hb, Wb)
        """
        B, C, H, W = x.shape
        assert C == 3, "Expect 3 channels (Y/Cb/Cr) layout."
        Hb, Wb = H // self.size, W // self.size
        outs = []
        for (u, v) in pos_list:
            # 샘플링: 블록 그리드 위치의 (u,v) 계수만 선택
            feat = x[:, :, u::self.size, v::self.size]  # (B,3,Hb,Wb)
            outs.append(feat)
        if len(outs) == 0:
            # 방어 코드: 빈 대역이면 0 채널 반환
            return x.new_zeros((B, 0, Hb, Wb))
        band = torch.cat(outs, dim=1)  # (B, 3*n_pos, Hb, Wb)

        return band

    def forward(self, x):
        bands_out = {}
        for name, pos in self.bands.items():
            bands_out[name] = self._gather_positions(x, pos)
        return bands_out  # {'dc':(B,3*1,Hb,Wb), 'low':(B,3*9,Hb,Wb), ...}


class Conv_PDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.5):
        super(Conv_PDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,8),
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        device = x.device
        tensor_zeros = torch.zeros(C_out, C_in, 1, device=device, dtype=self.conv.weight.dtype)
        conv_weight = torch.cat((
            self.conv.weight[:,:,:,0],
            self.conv.weight[:,:,:,1],
            self.conv.weight[:,:,:,2],
            self.conv.weight[:,:,:,3],
            tensor_zeros,
            self.conv.weight[:,:,:,4],
            self.conv.weight[:,:,:,5],
            self.conv.weight[:,:,:,6],
            self.conv.weight[:,:,:,7]
        ), dim=2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias,
                              stride=self.conv.stride, padding=self.conv.padding)

        if abs(self.theta) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class pdcblock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock1, self).__init__()
        self.basic = nn.Sequential(
            Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)  # /2
        )

    def forward(self, x):
        return self.basic(x)


class pdcblock1_v2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock1_v2, self).__init__()
        self.basic = nn.Sequential(
            Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.basic(x)


class pdcblock2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock2, self).__init__()
        self.basic = nn.Sequential(
            Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            Conv_PDC(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)  # /2
        )

    def forward(self, x):
        return self.basic(x)


class DCONV(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(DCONV, self).__init__()
        self.down = Conv_PDC(in_channels, out_channels, kernel_size = 3,
                             stride = stride, padding = 1)

    def forward(self, x):
        return self.down(x)


class EnhanceMHSA(nn.Module):
    def __init__(self, input_size, channels, d_k, d_v, stride, heads, dropout):
        super(EnhanceMHSA, self).__init__()
        self.dwconv_k = DCONV(channels, channels, stride = stride)
        self.dwconv_v = DCONV(channels, channels, stride = stride)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_k, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2
        self.B = nn.Parameter(torch.Tensor(1, self.heads, input_size ** 2, (input_size // stride) ** 2), requires_grad = True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))

        q = self.fc_q(x_reshape)
        q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

        k = self.dwconv_k(x)
        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous()

        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = attn + self.B
        attn = torch.softmax(attn, dim = -1)

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, h, w)
        result = result + x
        return result


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_str = math.floor(previous_conv_size[0] / out_pool_size[i])
        w_str = math.floor(previous_conv_size[1] / out_pool_size[i])
        maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_str, w_str))
        x = maxpool(previous_conv)
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


# ---------------------------
# 새 네트워크: 대역별 분기 처리
# ---------------------------
class TempNet_v4(nn.Module):
    def __init__(self, dropout=0.0):
        super(TempNet_v4, self).__init__()
        self.proj = DCTBandProject(size=8)  # (B,3,H,W) -> dict of (B,3*npos,H/8,W/8)
        bands = make_band_index(8)
        n_dc   = len(bands['dc'])    # 1
        n_low  = len(bands['low'])   # 9
        n_mid  = len(bands['mid'])   # 18
        n_high = len(bands['high'])  # 36

        # 대역별 branch (입력 채널: 3 * n_pos_band)
        self.branch_dc   = pdcblock1(3 * n_dc,   16)   # 3 -> 16, /2
        self.branch_low  = pdcblock1(3 * n_low,  32)   # 27 -> 16, /2
        self.branch_mid  = pdcblock1(3 * n_mid,  32)   # 54 -> 16, /2
        self.branch_high = pdcblock1(3 * n_high, 32)   # 108 -> 16, /2

        # concat 후 공용 stage
        self.group3 = pdcblock2(16+32 *3, 128)  # /2
        self.group4 = pdcblock2(128, 256)     # /2

        # 입력 256 → 블록그리드 32 → branch에서 /2 → 16 → group3 /2 → 8 → group4 /2 → 4
        self.entrans = EnhanceMHSA(input_size=4, channels=256, d_k=64, d_v=64, stride=2, heads=2, dropout=0.0)

        self.fc1 = nn.Linear(5376, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, input):
        """
        input: (B,3,H,W), H=W=256 (8의 배수)
        """
        bands = self.proj(input)  # dict of (B,3*npos, 32, 32)

        x_dc   = self.branch_dc(bands['dc'])     # (B,32,16,16)
        x_low  = self.branch_low(bands['low'])   # (B,32,16,16)
        x_mid  = self.branch_mid(bands['mid'])   # (B,32,16,16)
        x_high = self.branch_high(bands['high']) # (B,32,16,16)

        x = torch.cat([x_dc, x_low, x_mid, x_high], dim=1)  # (B,128,16,16)
        x = self.group3(x)  # (B,128->128, 8,8)
        x = self.group4(x)  # (B,128->256, 4,4)
        x = self.entrans(x) # (B,256,4,4)

        spp = spatial_pyramid_pool(x, x.size(0), [x.size(2), x.size(3)], [4,2,1])  # 4,2,1 from 4x4 ok
        x = self.fc1(spp)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
