import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ===========================
# DCTFormer_v3 with DCAM
# ===========================
class DCTFormer_v3(nn.Module):
    """
    입력: (B,3,H,W) — Y,Cb,Cr 계수맵(예: dqdct 블록을 (H,W)에 쌓은 형태), H,W는 8의 배수
    파이프라인:
      8x8 블록 → (B,3,N,64) → 64→d 선형 → 채널 결합 → (B,Hb,Wb,d)
      [Depth회 반복] Win-MSA → DCAM → MLP
      LN → 글로벌 평균 풀링 → Linear
    """
    def __init__(self, d_model=256, nhead=8, depth=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 window_size=(8,8), num_classes=2,
                 use_dctam=True, dct_reduction=8, dct_freq_method='top16'):
        super().__init__()
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

    def forward_features(self, x):
        """
        x: (B,3,H,W)
        """
        B, C, H, W = x.shape
        blocks, Hb, Wb = self._blockify_8x8(x)                    # (B,3,N,64), N=Hb*Wb
        N = Hb * Wb

        # per-channel block embedding
        emb = self.block_proj(blocks)                              # (B,3,N,d)
        emb = emb.permute(0, 2, 1, 3).contiguous()                 # (B,N,3,d)
        tokens = self.channel_fuse(emb.view(B, N, 3 * self.d_model))  # (B,N,d)

        # grid for window ops
        xg = tokens.view(B, Hb, Wb, self.d_model)                  # (B,Hb,Wb,d)

        # window transformer stack (+DCAM)
        for blk in self.blocks:
            xg = blk(xg, Hb, Wb)                                   # (B,Hb,Wb,d)

        xg = self.norm(xg)
        rep = xg.mean(dim=(1, 2))                                  # (B,d) global avg pooling
        return rep

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)


# ===========================
# Builder
# ===========================
def build_dctformer_v3(version="small", num_classes=2, **kwargs):
    """
    version: "small" | "base" | "large"
    추가 kwargs:
      - use_dctam (bool), dct_reduction (int), dct_freq_method (str), window_size, ...
    """
    if version == "small":
        return DCTFormer_v3(d_model=128, nhead=4, depth=4, mlp_ratio=2.0, num_classes=num_classes, **kwargs)
    elif version == "base":
        return DCTFormer_v3(d_model=256, nhead=8, depth=6, mlp_ratio=4.0, num_classes=num_classes, **kwargs)
    elif version == "large":
        return DCTFormer_v3(d_model=384, nhead=12, depth=8, mlp_ratio=6.0, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown version: {version}")
