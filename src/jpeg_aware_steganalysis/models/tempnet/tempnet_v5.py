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

# =========================
# Utils: frequency selectors
# =========================
def get_freq_indices(method: str = 'top8'):
    """
    7x7 그리드 기준 저주파(|u|+|v| 작은)부터 선택.
    method: 'top8' | 'top16' | 'top32'
    """
    size = 7
    coords = [(u, v) for u in range(size) for v in range(size)]
    coords.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))  # 저주파 우선

    k = 8
    if '16' in method: k = 16
    elif '32' in method: k = 32
    elif '8' in method: k = 8
    sel = coords[:k]
    mapper_x = [u for (u, v) in sel]
    mapper_y = [v for (u, v) in sel]
    return mapper_x, mapper_y


# =======================================
# Multi-Spectral DCT Layer for DCAM (2D)
# =======================================
class MultiSpectralDCTLayer(nn.Module):
    """
    Generate 2D DCT filters and produce channel descriptors.
    weight: (C, H, W) — 각 채널별 선택 주파수(합성) 2D DCT 패턴
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        """
        x: (N, C, H, W)
        return: (N, C) channel descriptors
        """
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        val = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return val if freq == 0 else val * math.sqrt(2)

    def get_dct_filter(self, H, W, mapper_x, mapper_y, C):
        dct_filter = torch.zeros(C, H, W)
        c_part = C // len(mapper_x)
        for i, (u, v) in enumerate(zip(mapper_x, mapper_y)):
            for y in range(H):
                for x in range(W):
                    dct_filter[i * c_part:(i + 1) * c_part, y, x] = \
                        self.build_filter(y, u, H) * self.build_filter(x, v, W)
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    """
    입력 x: (N, C, H, W)
      1) 필요시 (H,W)->(dct_h,dct_w)로 adaptive pooling
      2) DCT 기저 가중합으로 (N,C) descriptor
      3) MLP(reduction) + Sigmoid → 채널 게이트
    """
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top8'):
        super().__init__()
        self.reduction = reduction
        self.dct_h, self.dct_w = dct_h, dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        # 7×7의 상대 좌표를 현재 (dct_h, dct_w)에 맞춰 스케일
        mapper_x = [ux * (dct_h // 7) for ux in mapper_x]
        mapper_y = [vy * (dct_w // 7) for vy in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        # if h != self.dct_h or w != self.dct_w:
        #     x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x)          # (N, C)
        y = self.fc(y).view(n, c, 1, 1)       # (N, C, 1, 1)
        return x * y


# =========================
# Basic MLP (Transformer FF)
# =========================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# ==============================
# Window partitioning / reverse
# ==============================
def window_partition(x, win_h, win_w):
    """
    x: (B, H, W, C) -> (BnW, M, C), M = win_h*win_w
    """
    B, H, W, C = x.shape
    assert H % win_h == 0 and W % win_w == 0
    x = x.view(B, H // win_h, win_h, W // win_w, win_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()         # (B, nWh, nWw, win_h, win_w, C)
    windows = x.view(-1, win_h * win_w, C)               # (BnW, M, C)
    return windows

def window_reverse(windows, win_h, win_w, H, W, B):
    """
    windows: (BnW, M, C) -> (B, H, W, C)
    """
    nWh = H // win_h; nWw = W // win_w
    x = windows.view(B, nWh, nWw, win_h, win_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# =========================================
# Window MSA with fixed relative bias (RPE)
# =========================================
class WindowMSA(nn.Module):
    """
    창 내부(예: 8x8 토큰)에서만 Self-Attention.
    Position은 learnable이 아닌 고정(relative) 바이어스를 attn logits에 더해 주입.
    """
    def __init__(self, dim, window_size=(8, 8), num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.win_h, self.win_w = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 고정(relative) 위치 바이어스 준비
        coords_h = torch.arange(self.win_h)
        coords_w = torch.arange(self.win_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flat = coords.flatten(1)                                          # (2, M)
        dy = (coords_flat[0][:, None] - coords_flat[0][None, :]).abs().float()   # (M, M)
        dx = (coords_flat[1][:, None] - coords_flat[1][None, :]).abs().float()   # (M, M)

        head_idx = torch.arange(1, num_heads + 1, dtype=torch.float32)
        slopes = 1.0 / head_idx  # 헤드별 기울기 고정

        self.register_buffer("Dy", dy, persistent=False)                          # (M, M)
        self.register_buffer("Dx", dx, persistent=False)                          # (M, M)
        self.register_buffer("slopes_h", slopes, persistent=False)                # (H,)
        self.register_buffer("slopes_w", slopes.clone(), persistent=False)        # (H,)

    def forward(self, x):
        """
        x: (BnW, M, C)   # M = win_h * win_w
        """
        BnW, M, C = x.shape
        qkv = self.qkv(x).reshape(BnW, M, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                # (3, BnW, H, M, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))                                # (BnW, H, M, M)

        # 고정(relative) 위치 바이어스 추가
        bias_h = self.slopes_h[:, None, None] * self.Dy[None, :, :]
        bias_w = self.slopes_w[:, None, None] * self.Dx[None, :, :]
        bias = -(bias_h + bias_w)                                       # (H, M, M)
        attn = attn + bias.unsqueeze(0).to(attn.dtype)                  # (1,H,M,M) broadcast

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BnW, M, C)               # (BnW, M, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==========================================
# DCAM on window tokens (TokenDCAM wrapper)
# ==========================================
class TokenDCAM(nn.Module):
    """
    windows: (BnW, M, C)  [M = win_h * win_w]
    → (BnW, C, win_h, win_w) 변환 후 MultiSpectralAttentionLayer 적용 → 복귀
    """
    def __init__(self, dim, win_h, win_w, reduction=8, freq_sel_method='top16'):
        super().__init__()
        self.win_h, self.win_w = win_h, win_w
        self.att = MultiSpectralAttentionLayer(
            channel=dim, dct_h=win_h, dct_w=win_w,
            reduction=reduction, freq_sel_method=freq_sel_method
        )

    def forward(self, windows):  # (BnW, M, C)
        BnW, M, C = windows.shape
        assert M == self.win_h * self.win_w, "M must equal win_h*win_w"
        x = windows.view(BnW, self.win_h, self.win_w, C).permute(0, 3, 1, 2).contiguous()  # (BnW, C, H, W)
        x = self.att(x)  # DCAM 채널 주파수 게이트
        x = x.permute(0, 2, 3, 1).contiguous().view(BnW, M, C)  # (BnW, M, C)
        return x


# ======================================
# Transformer Window Block with DCAM
# ======================================
class TransformerWindowBlock(nn.Module):
    """
    Pre-LN:
      x -> LN -> WindowMSA -> (+DCAM) -> +res
           -> LN -> MLP -> +res
    """
    def __init__(self, dim, window_size=(8,8), num_heads=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 use_dctam: bool = True, dct_reduction: int = 8, dct_freq_method: str = 'top16'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowMSA(dim, window_size, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

        self.win_h, self.win_w = window_size
        self.use_dctam = use_dctam
        if use_dctam:
            self.dcam = TokenDCAM(dim, self.win_h, self.win_w, reduction=dct_reduction,
                                  freq_sel_method=dct_freq_method)
        else:
            self.dcam = None

    def forward(self, x, Hb, Wb):   # x: (B, Hb, Wb, C)
        B, H, W, C = x.shape
        assert H == Hb and W == Wb

        # (1) Win-MSA
        x_res = x
        x_win = self.norm1(x).view(B, H, W, C)
        windows = window_partition(x_win, self.win_h, self.win_w)       # (BnW, M, C)
        windows = self.attn(windows)                                    # (BnW, M, C)

        # (1.5) DCAM: 윈도우 내부 채널 주파수 어텐션 (옵션)
        if self.dcam is not None:
            windows = self.dcam(windows)                                # (BnW, M, C)

        x = window_reverse(windows, self.win_h, self.win_w, H, W, B)    # (B, H, W, C)
        x = x_res + x                                                   # residual

        # (2) MLP
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


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


class DCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(DCONV, self).__init__()
        self.down = Conv_PDC(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1)

    def forward(self, x):
        result = self.down(x)
        return result
    

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

        # Reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))

        # Get q, k, v
        q = self.fc_q(x_reshape)
        q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, h * w, d_k]

        k = self.dwconv_k(x)
        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_h * k_w, d_k]

        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_h * v_w, d_v]

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = attn + self.B
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, k_h * k_w]

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, h, w)
        result = result + x
        return result


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


class TempNet_v5(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 window_size=(8,8), num_classes=2,
                 use_dctam=True, dct_reduction=8, dct_freq_method='top16'):
        super(TempNet_v5, self).__init__()
        self.relu = nn.ReLU()
        self.output_num = [4,2,1]
        self.group1 = HPF()
        self.group2 = pdcblock1(186,64)
        self.group3 = pdcblock2(64,128)
        self.group4 = pdcblock2(128,256)
        self.entrans = EnhanceMHSA(32, 256, 64, 64, 2, 2, 0.0)
        self.fc1 = nn.Linear(5376, 1024)
        self.fc2 = nn.Linear(1280, 2)

        self.d_model = d_model
        self.win_h, self.win_w = window_size
        assert d_model % nhead == 0

        # 8x8 block embedding (64 -> d)
        self.block_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # fuse Y/Cb/Cr embeddings per block
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(d_model),
        )

        # window transformer layers (+DCAM)
        self.blocks = nn.ModuleList([
            TransformerWindowBlock(d_model, window_size, nhead, mlp_ratio, drop, attn_drop,
                                   use_dctam=use_dctam, dct_reduction=dct_reduction,
                                   dct_freq_method=dct_freq_method)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _blockify_8x8(x):
        """
        x: (B,3,H,W) -> (B,3,Hb*Wb,64), Hb=H/8, Wb=W/8
        """
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, Hb * Wb, 64)
        return x, Hb, Wb
    
    def forward_features(self, rgb, dqdct):
        rgb_output = rgb
        rgb_output_y = rgb_output[:, 0, :, :]
        rgb_output_u = rgb_output[:, 1, :, :]
        rgb_output_v = rgb_output[:, 2, :, :]
        rgb_out_y = rgb_output_y.unsqueeze(1)
        rgb_out_u = rgb_output_u.unsqueeze(1)
        rgb_out_v = rgb_output_v.unsqueeze(1)
        y = self.group1(rgb_out_y)
        u = self.group1(rgb_out_u)
        v = self.group1(rgb_out_v)
        rgb_output = torch.cat([y, u, v], dim=1)

        rgb_output = self.group2(rgb_output)
        rgb_output = self.group3(rgb_output)
        rgb_output = self.group4(rgb_output)
        rgb_output = self.entrans(rgb_output)
        spp = spatial_pyramid_pool(rgb_output, rgb_output.size(0), [int(rgb_output.size(2)),int(rgb_output.size(3))], [4,2,1])
        rgb_output = self.fc1(spp)
        rgb_output = self.relu(rgb_output)

        B, C, H, W = dqdct.shape
        blocks, Hb, Wb = self._blockify_8x8(dqdct)                    # (B,3,N,64), N=Hb*Wb
        N = Hb * Wb

        # per-channel block embedding
        emb = self.block_proj(blocks)                              # (B,3,N,d)
        emb = emb.permute(0, 2, 1, 3).contiguous()                 # (B,N,3,d)
        tokens = self.channel_fuse(emb.view(B, N, 3 * self.d_model))  # (B,N,d)

        dqdct_output = tokens.view(B, Hb, Wb, self.d_model)                  # (B,Hb,Wb,d)

        for blk in self.blocks:
            dqdct_output = blk(dqdct_output, Hb, Wb)                                   # (B,Hb,Wb,d)

        dqdct_output = self.norm(dqdct_output)
        dqdct_output = dqdct_output.mean(dim=(1, 2))                                  # (B,d) global avg pooling

        output = torch.cat([rgb_output, dqdct_output], dim=1)
        return output

    def forward(self, rgb, dqdct):
        x = self.forward_features(rgb, dqdct)
        x = self.fc2(x)
        return x
