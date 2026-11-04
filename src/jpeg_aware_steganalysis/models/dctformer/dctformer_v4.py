import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================
# Basic MLP (FFN)
# ================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# =========================================
# Window MSA
# =========================================
class WindowMSA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.win_h, self.win_w = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 고정(relative) 위치 바이어스 미리 계산
        coords_h = torch.arange(self.win_h)
        coords_w = torch.arange(self.win_w)
        yy, xx = torch.meshgrid(coords_h, coords_w, indexing='ij')
        dy = (yy.flatten()[:, None] - yy.flatten()[None, :]).abs().float()
        dx = (xx.flatten()[:, None] - xx.flatten()[None, :]).abs().float()
        head_idx = torch.arange(1, num_heads + 1, dtype=torch.float32)
        slopes = 1.0 / head_idx
        self.register_buffer("Dy", dy, persistent=False)
        self.register_buffer("Dx", dx, persistent=False)
        self.register_buffer("slopes_h", slopes, persistent=False)
        self.register_buffer("slopes_w", slopes.clone(), persistent=False)

    def forward(self, x):  # x: (BnW, M, C)
        BnW, M, C = x.shape
        qkv = self.qkv(x).view(BnW, M, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))

        bias_h = self.slopes_h[:, None, None] * self.Dy[None, :, :]
        bias_w = self.slopes_w[:, None, None] * self.Dx[None, :, :]
        attn = attn + (-(bias_h + bias_w)).unsqueeze(0).to(attn.dtype)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(BnW, M, C)
        return self.proj_drop(self.proj(x))


# ======================================
# Transformer Window Block (no DCAM)
# ======================================
class TransformerWindowBlock(nn.Module):
    def __init__(self, dim, window_size=(8,8), num_heads=8, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowMSA(dim, window_size, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

        self.win_h, self.win_w = window_size

    def forward(self, x_bhwc):   # (B,Hb,Wb,C)
        B, H, W, C = x_bhwc.shape
        assert H % self.win_h == 0 and W % self.win_w == 0
        nWh, nWw = H // self.win_h, W // self.win_w

        # (1) Win-MSA
        x_res = x_bhwc
        x_win = self.norm1(x_bhwc)

        # window partition
        x = x_win.view(B, nWh, self.win_h, nWw, self.win_w, C).permute(0,1,3,2,4,5).contiguous()
        windows = x.view(B * nWh * nWw, self.win_h * self.win_w, C)             # (BnW,M,C)

        windows = self.attn(windows)                                            # (BnW,M,C)

        # window reverse
        x = windows.view(B, nWh, nWw, self.win_h, self.win_w, C).permute(0,1,3,2,4,5).contiguous()
        x = x.view(B, H, W, C)                                                  # (B,Hb,Wb,C)

        x = x + x_res

        # (2) MLP
        x_res = x
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


# ===========================
# DCTFormer_v4
# ===========================
class _ChannelBranch(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 window_size=(8,8)):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.win_h, self.win_w = window_size

        self.block_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )

        # 이 브랜치만의 Transformer 블록 스택
        self.blocks = nn.ModuleList([
            TransformerWindowBlock(d_model, window_size, nhead, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_c):  # x_c: (B, Hb, Wb, 64)  -- 단일 채널의 8x8 벡터
        B, Hb, Wb, _ = x_c.shape

        # (1) 블록 임베딩: (B,Hb,Wb,64) -> (B,Hb,Wb,d)
        xg = self.block_proj(x_c.view(B*Hb*Wb, 64)).view(B, Hb, Wb, self.d_model)

        # (2) 채널 전용 Transformer stack
        for blk in self.blocks:
            xg = blk(xg)                                # (B,Hb,Wb,d)

        # (3) LN + GAP -> (B,d)
        xg = self.norm(xg)
        rep = xg.mean(dim=(1, 2))
        return rep                                      # (B,d)


# ----------------------------------------------------
# DCTFormer_v4
# ----------------------------------------------------
class DCTFormer_v4(nn.Module):
    def __init__(self, d_model=128, nhead=8, depth=6, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, window_size=(8,8), num_classes=2):
        super().__init__()
        self.branch_y  = _ChannelBranch(d_model, nhead, depth, mlp_ratio, drop, attn_drop, window_size)
        self.branch_cb = _ChannelBranch(d_model, nhead, depth, mlp_ratio, drop, attn_drop, window_size)
        self.branch_cr = _ChannelBranch(d_model, nhead, depth, mlp_ratio, drop, attn_drop, window_size)

        self.head = nn.Linear(d_model * 3, num_classes)

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape
        assert C == 3 and H % 8 == 0 and W % 8 == 0
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, Hb, Wb, C, 64)
        return x, Hb, Wb

    def forward(self, x):  # x: (B,3,H,W)
        B, _, H, W = x.shape
        xb, Hb, Wb = self._blockify_8x8(x)          # (B,Hb,Wb,3,64)

        x_y  = xb[:, :, :, 0, :]
        x_cb = xb[:, :, :, 1, :]
        x_cr = xb[:, :, :, 2, :]

        rep_y  = self.branch_y(x_y)                 # (B,d)
        rep_cb = self.branch_cb(x_cb)               # (B,d)
        rep_cr = self.branch_cr(x_cr)               # (B,d)

        rep = torch.cat([rep_y, rep_cb, rep_cr], dim=1)  # (B, 3d)
        logits = self.head(rep)                           # (B, num_classes)
        return logits
        
# ===========================
# Builder
# ===========================
def build_dctformer_v4(version="small", num_classes=2, **kwargs):
    if version == "small":
        return DCTFormer_v4(d_model=64, nhead=4, depth=4, mlp_ratio=2.0, num_classes=num_classes, **kwargs)
    elif version == "base":
        return DCTFormer_v4(d_model=128, nhead=8, depth=6, mlp_ratio=4.0, num_classes=num_classes, **kwargs)
    elif version == "large":
        return DCTFormer_v4(d_model=256, nhead=12, depth=8, mlp_ratio=6.0, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown version: {version}")
