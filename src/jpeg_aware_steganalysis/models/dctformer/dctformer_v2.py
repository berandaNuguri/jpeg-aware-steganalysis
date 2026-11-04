import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 기본 MLP --------------------
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

# -------------------- 윈도우 분할/복원 --------------------
def window_partition(x, win_h, win_w):
    # x: (B, H, W, C) -> (BnW, win_h*win_w, C)
    B, H, W, C = x.shape
    assert H % win_h == 0 and W % win_w == 0
    x = x.view(B, H // win_h, win_h, W // win_w, win_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()               # (B, nWh, nWw, win_h, win_w, C)
    windows = x.view(-1, win_h * win_w, C)                     # (BnW, M, C)
    return windows

def window_reverse(windows, win_h, win_w, H, W, B):
    # windows: (BnW, win_h*win_w, C) -> (B, H, W, C)
    nWh = H // win_h; nWw = W // win_w
    x = windows.view(B, nWh, nWw, win_h, win_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# -------------------- 윈도우 MSA --------------------
class WindowMSA(nn.Module):
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

        coords_h = torch.arange(self.win_h)
        coords_w = torch.arange(self.win_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flat = coords.flatten(1)                                          # (2, M)
        dy = (coords_flat[0][:, None] - coords_flat[0][None, :]).abs().float()   # (M, M)
        dx = (coords_flat[1][:, None] - coords_flat[1][None, :]).abs().float()   # (M, M)

        head_idx = torch.arange(1, num_heads + 1, dtype=torch.float32)
        slopes = 1.0 / head_idx                                                  # (H,)

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

        # ---- 고정(relative) 위치 바이어스 추가 ----
        # bias_h: (H, M, M), bias_w: (H, M, M)
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

# -------------------- 윈도우 블록 (Pre-LN) --------------------
class TransformerWindowBlock(nn.Module):
    def __init__(self, dim, window_size=(8,8), num_heads=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowMSA(dim, window_size, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

        self.win_h, self.win_w = window_size

    def forward(self, x, Hb, Wb):
        """
        x: (B, Hb, Wb, C)
        """
        B, H, W, C = x.shape
        assert H == Hb and W == Wb
        # (1) Win-MSA
        x_res = x
        x_win = self.norm1(x).view(B, H, W, C)
        windows = window_partition(x_win, self.win_h, self.win_w)       # (BnW, M, C)
        windows = self.attn(windows)                                    # (BnW, M, C)
        x = window_reverse(windows, self.win_h, self.win_w, H, W, B)    # (B, H, W, C)
        x = x_res + x                                                   # residual

        # (2) MLP
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x

# -------------------- 모델 본체: 8x8 블록 토큰 + 윈도우 Self-Attn --------------------
class DCTFormer_v2(nn.Module):
    """
    - 입력 (B,3,H,W), H,W는 8의 배수 (예: 256→Hb=Wb=32)
    - 블록(8x8) 단위로 토큰화하여 Hb×Wb 그리드 생성
    - 채널별( Y/Cb/Cr ) 64-D -> d 임베딩 후 concat-fuse로 블록당 1토큰
    - 윈도우(8×8 토큰) 단위 Self-Attn에서 '상대 위치 바이어스'로 positional을 어텐션에만 주입
    """
    def __init__(self, d_model=256, nhead=8, depth=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 window_size=(8,8), num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.win_h, self.win_w = window_size
        assert d_model % nhead == 0

        # 채널별 블록 임베딩(64 -> d)
        self.block_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )
        # 채널 결합 (Y,Cb,Cr → concat 후 fuse)
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(d_model),
        )

        # 윈도우 블록 스택
        self.blocks = nn.ModuleList([
            TransformerWindowBlock(d_model, window_size, nhead, mlp_ratio, drop, attn_drop)
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
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0,1,2,4,3,5).contiguous()
        x = x.view(B, C, Hb*Wb, 64)
        return x, Hb, Wb

    def forward_features(self, x):
        """
        x: (B,3,H,W)
        """
        B, C, H, W = x.shape
        blocks, Hb, Wb = self._blockify_8x8(x)                    # (B,3,N,64), N=Hb*Wb
        N = Hb * Wb

        # 채널별 64→d
        emb = self.block_proj(blocks)                              # (B,3,N,d)
        emb = emb.permute(0, 2, 1, 3).contiguous()                 # (B,N,3,d)
        tokens = self.channel_fuse(emb.view(B, N, 3*self.d_model)) # (B,N,d)

        # (B,N,d) -> (B,Hb,Wb,d)
        xg = tokens.view(B, Hb, Wb, self.d_model)

        # 윈도우 단위 self-attention 스택
        for blk in self.blocks:
            xg = blk(xg, Hb, Wb)                                   # (B,Hb,Wb,d)

        xg = self.norm(xg)
        rep = xg.mean(dim=(1,2))                                   # (B,d)  # 글로벌 평균 풀링
        return rep

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)

def build_dctformer_v2(version="small", num_classes=2):
    """
    version: "small" | "base" | "large"
    """
    if version == "small":
        return DCTFormer_v2(d_model=128, nhead=4, depth=4, mlp_ratio=2.0, num_classes=num_classes)
    elif version == "base":
        return DCTFormer_v2(d_model=256, nhead=8, depth=6, mlp_ratio=4.0,num_classes=num_classes)
    elif version == "large":
        return DCTFormer_v2(d_model=384, nhead=12, depth=8, mlp_ratio=6.0, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown version: {version}")
