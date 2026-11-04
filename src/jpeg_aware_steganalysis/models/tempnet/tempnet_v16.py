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

class CheckerboardConv(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1,
        dilation=1, groups=1, bias=False, theta=0.5,
        pattern=0
    ):
        """
        pattern_type:
            0 → checkerboard mask [[1,0,1],[0,1,0],[1,0,1]]
            1 → inverted mask     [[0,1,0],[1,0,1],[0,1,0]]
        """
        super(CheckerboardConv, self).__init__()
        self.kernel_size = kernel_size
        self.theta = theta
        self.padding = padding

        k = kernel_size
        # --- build checkerboard mask ---
        if pattern == 0:
            checkerboard_mask = torch.tensor(
                [(i + j) % 2 == 0 for i in range(k) for j in range(k)],
                dtype=torch.bool
            )
        elif pattern == 1:
            checkerboard_mask = torch.tensor(
                [(i + j) % 2 == 1 for i in range(k) for j in range(k)],
                dtype=torch.bool
            )
        else:
            raise ValueError(f"Invalid pattern_type {pattern}. Use 0 or 1.")

        self.register_buffer('checkerboard_mask', checkerboard_mask)
        num_masked = int(checkerboard_mask.sum().item())

        self.base_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, num_masked),
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        w_flat = self.base_conv.weight        
        out_ch, in_ch, _, M = w_flat.shape
        k = self.kernel_size

        w_masked = w_flat.view(out_ch, in_ch, M)
        w_full = torch.zeros(out_ch, in_ch, k*k, device=w_flat.device)
        w_full[:, :, self.checkerboard_mask] = w_masked
        w_full = w_full.view(out_ch, in_ch, k, k)

        if abs(self.theta) < 1e-8:
            return F.conv2d(
                x, w_full,
                bias=self.base_conv.bias,
                stride=self.base_conv.stride,
                padding=self.padding,
                groups=self.base_conv.groups
            )

        neigh = torch.zeros_like(w_full)
        neigh[:, :, 1:, :]   += w_full[:, :, :-1, :]
        neigh[:, :, :-1, :]  += w_full[:, :, 1:, :]
        neigh[:, :, :, 1:]   += w_full[:, :, :, :-1]
        neigh[:, :, :, :-1]  += w_full[:, :, :, 1:]

        mask2d = self.checkerboard_mask.view(1, 1, k, k)
        new_kernel = torch.where(
            mask2d,
            w_full,
            - self.theta * neigh
        )

        return F.conv2d(
            x, new_kernel,
            bias=self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.padding,
            groups=self.base_conv.groups
        )

class pdcblock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                CheckerboardConv(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
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
                CheckerboardConv(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                CheckerboardConv(outchannel, outchannel, kernel_size=3, padding=1),
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
        self.down = CheckerboardConv(in_channels, out_channels, kernel_size = 3,
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


class TempNet_v16(nn.Module):
  def __init__(self, dropout=0.0):
    super(TempNet_v16, self).__init__()
    self.relu = nn.ReLU()
    self.output_num = [4,2,1]
    self.group1 = HPF()
    self.group2 = pdcblock1(64,64)
    self.group3 = pdcblock2(64,128)
    self.group4 = pdcblock2(128,256)
    self.entrans = EnhanceMHSA(32, 256, 64, 64, 2, 2, 0.0)
    self.fc1 = nn.Linear(5376, 1024)
    self.dropout = nn.Dropout(p=dropout)
    self.fc2 = nn.Linear(1024, 2)

  def forward_features(self, input):
    output = input
    output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.entrans(output)
    output = spatial_pyramid_pool(output, output.size(0), [int(output.size(2)),int(output.size(3))], [4,2,1])
    output = self.fc1(output)
    output = self.relu(output)
    output = self.dropout(output)
    return output

  def forward(self, input):
    x = self.forward_features(input)
    x = self.fc2(x)
    return x
