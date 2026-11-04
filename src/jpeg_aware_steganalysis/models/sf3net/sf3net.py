from jpeg_aware_steganalysis._paths import project_root, RESOURCE_ROOT
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .Coordatt import CoordAtt
from timm.models.layers import trunc_normal_
from .SwinTransformer import *

SRM_npy = np.load(str(RESOURCE_ROOT / "sf3net" / "SRM_Kernels.npy"))

class L2_nrom(nn.Module):
    def __init__(self,mode='l2'):
        super(L2_nrom, self).__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True)).pow(0.5)
            norm = embedding / (embedding.pow(2).mean(dim=1, keepdim=True)).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x)
            embedding = _x.sum((2,3), keepdim=True)
            norm = embedding / (torch.abs(embedding).mean(dim=1, keepdim=True))
        return norm

class Sepconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sepconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)


    def forward(self, input):

        out1 = self.conv1(input)
        out = self.conv2(out1)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features,
                                           num_input_features,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, num_input_features,
                                           kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, prev_features):
        new_features = self.conv1(self.relu1(self.norm1(prev_features)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock_Add(nn.Module):
    def __init__(self, num_layers, num_input_features):
        super(_DenseBlock_Add, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = init_features
        for name, layer in self.named_children():
            new_features = layer(features)
            features = features + new_features
        return features


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, internal_channels=16):
        super(ResidualBlock, self).__init__()

        # 分支1: 3x3 DW Conv -> 1x1 PW Conv -> PReLU -> 1x1 PW Conv
        self.branch1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, internal_channels, 1),
            nn.BatchNorm2d(internal_channels),
            nn.Conv2d(internal_channels, in_channels, 1),
        )

        # 分支2: 1x3 DW Conv -> 1x1 PW Conv -> PReLU -> 1x1 PW Conv
        self.branch2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, internal_channels, 1),
            nn.BatchNorm2d(internal_channels),
            nn.Conv2d(internal_channels, in_channels, 1),
        )

        # 分支3: 3x1 DW Conv -> 1x1 PW Conv -> PReLU -> 1x1 PW Conv
        self.branch3 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, internal_channels, 1),
            nn.BatchNorm2d(internal_channels),
            nn.Conv2d(internal_channels, in_channels, 1),
        )

    def forward(self, x):
        # 将输入通过三个分支
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # 合并分支输出
        out = out1 + out2 + out3

        # 残差连接
        out += x
        return out


class PreProcess(nn.Module):
    def __init__(self, num_layers=4):
        super(PreProcess, self).__init__()

        # 高通滤波 卷积核权重初始化
        self.srm_filters_weight = nn.Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=False)
        self.srm_filters_weight.data.numpy()[:] = SRM_npy

        self.features = nn.Sequential(OrderedDict([('norm0', nn.BatchNorm2d(30)), ]))
        self.features.add_module('relu0', nn.PReLU())

        block0 = ResidualBlock(30, 16)
        self.features.add_module('denseblock%d' % 0, block0)   # preprocessing
        block = _DenseBlock_Add(num_layers=num_layers, num_input_features=30,)
        self.features.add_module('denseblock%d' % 1, block)    #preprocessing

    def forward(self, input):
        HPF_output = F.conv2d(input, self.srm_filters_weight, stride=1, padding=2)
        output = self.features(HPF_output)
        return output


class Spatial_Stream(nn.Module):
    def __init__(self):
        super(Spatial_Stream, self).__init__()
        #preprocessing+BlockB
        self.preprocess_layer = PreProcess(num_layers=4)

        self.layer1 = nn.Conv2d(30, 32, kernel_size=1, stride=1, bias=False)
        self.layer1_BN = nn.BatchNorm2d(32)
        self.layer1_AC = nn.ReLU(inplace=True)

        #feature extraction
        self.layer2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) #BlockC
        self.layer2_BN = nn.BatchNorm2d(32)
        self.layer2_AC = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#BlockC
        self.layer3_BN = nn.BatchNorm2d(64)
        self.layer3_AC = nn.ReLU(inplace=True)

        self.avgpooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#BlockC
        self.layer4_BN = nn.BatchNorm2d(64)
        self.layer4_AC = nn.ReLU(inplace=True)

        self.layer5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)#BlockC
        self.layer5_BN = nn.BatchNorm2d(128)
        self.layer5_AC = nn.ReLU(inplace=True)

        self.avgpooling3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def tail_forward(self, x):
        preprocess_out = self.preprocess_layer(x)
        layer1_out = self.layer1(preprocess_out)
        layer1_out = self.layer1_BN(layer1_out)
        layer1_out = self.layer1_AC(layer1_out)
        return layer1_out


    def forward(self, x):
        layer2_out = self.layer2(x)
        layer2_out = self.layer2_BN(layer2_out)
        layer2_out = self.layer2_AC(layer2_out)

        layer3_out = self.layer3(layer2_out)
        layer3_out = self.layer3_BN(layer3_out)
        layer3_out = self.layer3_AC(layer3_out)

        avgpooling2 = self.avgpooling2(layer3_out)

        layer4_out = self.layer4(avgpooling2)
        layer4_out = self.layer4_BN(layer4_out)
        layer4_out = self.layer4_AC(layer4_out)

        layer5_out = self.layer5(layer4_out)
        layer5_out = self.layer5_BN(layer5_out)
        layer5_out = self.layer5_AC(layer5_out)

        avgpooling3 = self.avgpooling3(layer5_out)

        return avgpooling3


class SID(nn.Module):
    def __init__(self, patch_size=8, input_dim=30, output_dim=64):
        super(SID, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2_norm = L2_nrom(mode='l2')
        self.l1_norm = L2_nrom(mode='l1')
        self.fclayer = nn.Linear(4 * input_dim, output_dim)


    def sid(self, x):
        x = torch.unsqueeze(x, 3)
        b, c, n, spp_c, h, w = x.shape
        x = x.reshape(b, c, n, spp_c, -1)
        x_min = torch.min(x, dim=4).values
        x_max = torch.max(x, dim=4).values
        x_avg = torch.mean(x, dim=4)
        x_var = torch.var(x, dim=4)
        output = torch.cat([torch.unsqueeze(x_max, 4),
                            torch.unsqueeze(x_min, 4),
                            torch.unsqueeze(x_avg, 4),
                            torch.unsqueeze(x_var, 4)], dim=4)

        return output


    def forward(self, input):
        b, c, h, w = input.shape
        assert (h // self.patch_size or w // self.patch_size) is not True, \
            f"Input image size ({h}*{w}) doesn't match model ({h}*{w})."

        P_h, P_w = self.patch_size, self.patch_size
        N_h, N_w = h // self.patch_size, w // self.patch_size
        # crop
        input = input[:, :, :N_h * P_h, :N_w * P_w]
        x = input.reshape(b, c, P_h, N_h, P_w, N_w).permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(b, c, P_h * P_w, N_h, N_w)
        x = self.sid(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(b, P_h * P_w, -1)
        x = self.fclayer(x)
        return x


class SwTDSN(nn.Module):
    def __init__(self, inpit_dim=32,  num_classes=2, embed_dim=64, depths=[2], num_heads=[4],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, use_checkpoint=False, fused_window_process=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 结构尾部的norm
        self.mlp_ratio = mlp_ratio
        self.patch_embed = SID(patch_size=8, input_dim=inpit_dim)  # four
        num_patches = 8*8
        patches_resolution = [8, 8]
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        b = x.shape[0]
        # x = self.processing(x)
        # print(x.shape)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        norm_x = self.norm(x)  # B L C

        x = self.avgpool(norm_x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return norm_x, x

    def forward(self, x):
        norm_features, x = self.forward_features(x)
        x = self.head(x)
        return norm_features, x


class DCT_Stream(nn.Module):

    def __init__(self, in_channels):
        super(DCT_Stream, self).__init__()

        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dc_layer2_tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.dc_layer0_post = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward_features(self, x, qtable):
        DCTcoef = x
        x0 = self.dc_layer0_dil(DCTcoef)
        x1 = self.dc_layer1_tail(DCTcoef)
        x = torch.cat([x0, x1], dim=1)
        x = self.dc_layer2_tail(x)

        B, C, H, W = x.shape
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)     # [B, 1, 8, 8, 1, 1]

        xq_temp = (x_temp * q_temp)
        x1 = xq_temp.permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        x = torch.cat([x, x1], dim=1)
        x = self.dc_layer0_post(x)

        return x

    def forward(self, x, qtable):
        x = self.forward_features(x, qtable)
        x = self.fc(x)
        return x


class Feature_Fusion(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Feature_Fusion, self).__init__()
        #preprocessing+BlockB
        self.scse = CoordAtt(inchannel, inchannel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1),
            nn.BatchNorm2d(outchannel)
        )
        self.avgpooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.scse(x)
        x = self.conv2(x)
        x = self.avgpooling1(x)
        return x


class SF3Net(nn.Module):
    def __init__(self, in_channels=1):
        super(SF3Net, self).__init__()

        self.in_channels = in_channels
        self.OH = DCT_Stream(1)
        self.SS = Spatial_Stream()
        self.FUSION = Feature_Fusion(64, 32)
        self.SwT = SwTDSN(inpit_dim=128, num_classes=2)

    def _single_chan_forward(self, input_x, input_dct, input_qtable):
        dct_output = self.OH.forward_features(input_dct, input_qtable)
        spatial_output = self.SS.tail_forward(input_x)
        x = torch.cat([spatial_output, dct_output], dim=1)
        x = self.FUSION(x)
        x = self.SS.forward(x)
        _, y_fc = self.SwT.forward(x)

        return y_fc

    def _multi_chan_forward(self, input_x, input_dct, input_qtable):
        spat_list = []
        dct_list  = []

        for c in range(self.in_channels):
            x_c   = input_x[:,  c:c+1,  :, :]           # (B,1,H,W)
            dct_c = input_dct[:, c:c+1,  :, :]          # (B,1,H,W)
            qt_c  = input_qtable[:, c, :, :].unsqueeze(1)  # (B,1,8,8)

            dct_out     = self.OH.forward_features(dct_c, qt_c)  # (B,32,H,W)
            spatial_out = self.SS.tail_forward(x_c)              # (B,32,H,W)

            spat_list.append(spatial_out)
            dct_list.append(dct_out)

        spatial_output = torch.mean(torch.stack(spat_list, dim=0), dim=0)  # (B,32,H,W)
        dct_output     = torch.mean(torch.stack(dct_list,  dim=0), dim=0)  # (B,32,H,W)

        
        x = torch.cat([spatial_output, dct_output], dim=1)  # (B,64,H,W)
        
        x = self.FUSION(x)
        x = self.SS.forward(x)
        _, y_fc = self.SwT.forward(x)
        return y_fc

    def forward(self, input_x, input_dct, input_qtable):
        if self.in_channels == 1:
            y_fc = self._single_chan_forward(input_x, input_dct, input_qtable)
        elif self.in_channels == 3:
            y_fc = self._multi_chan_forward(input_x, input_dct, input_qtable)

        return y_fc

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name) / 1e6


if __name__ == '__main__':
    from torchsummary import summary
    torch.cuda.set_device(7)
    Input = torch.randn(1, 1, 256, 256)
    net = SF3Net().cuda()
    print("param size = %.5fMB", count_parameters_in_MB(net))