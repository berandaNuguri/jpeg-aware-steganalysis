import math
import torch
import torch.nn as nn


def get_freq_indices(method: str):
    supported = {
        'top1','top2','top4','top8','top16','top32',
        'bot1','bot2','bot4','bot8','bot16','bot32',
        'low1','low2','low4','low8','low16','low32',
    }
    if method not in supported:
        raise ValueError(f"Unsupported frequency selection method: {method}")

    num_freq = int(method[3:])

    if 'top' in method:
        mapper_x_all = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        mapper_y_all = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
    elif 'low' in method:
        mapper_x_all = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        mapper_y_all = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
    elif 'bot' in method:
        mapper_x_all = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        mapper_y_all = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
    else:
        raise ValueError(f"Unknown method specifier in {method}")

    mapper_x = mapper_x_all[:num_freq]
    mapper_y = mapper_y_all[:num_freq]
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel: int, dct_h: int, dct_w: int, reduction: int = 16, freq_sel_method: str = 'top16'):
        super().__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [val * (dct_h // 8) for val in mapper_x]
        mapper_y = [val * (dct_w // 8) for val in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        pooled = x
        if h != self.dct_h or w != self.dct_w:
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        y = self.dct_layer(pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height: int, width: int, mapper_x, mapper_y, channel: int):
        super().__init__()
        if len(mapper_x) != len(mapper_y):
            raise ValueError('mapper_x and mapper_y must be the same length')
        if channel % len(mapper_x) != 0:
            raise ValueError('channel must be divisible by the number of frequencies')

        weight = self.get_dct_filter(height, width, mapper_x, mapper_y, channel)
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f'Input must have 4 dimensions, got {x.dim()}')
        x = x * self.weight
        return torch.sum(x, dim=[2, 3])

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    value = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = value
        return dct_filter

    @staticmethod
    def build_filter(pos: int, freq: int, size: int) -> float:
        result = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        if freq == 0:
            return result
        return result * math.sqrt(2)
