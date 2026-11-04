import torch
import torch.nn as nn
import torch.nn.functional as F

# 그대로 사용
class LearnableAbsPos2D(nn.Module):
    def __init__(self, d_model: int, grid_size=(32, 32)):
        super().__init__()
        gh, gw = grid_size
        self.d_model = d_model
        self.pos = nn.Parameter(torch.zeros(1, d_model, gh, gw))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, Hb: int, Wb: int):
        pos = self.pos
        if pos.shape[-2:] != (Hb, Wb):
            pos = F.interpolate(pos, size=(Hb, Wb), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).contiguous().view(1, Hb * Wb, self.d_model)
        return pos


class DCTFormer(nn.Module):
    """
    Self-Attention만 사용하는 8×8 블록 Transformer
    입력:
      x: (B, 3, H, W)  # Y, Cb, Cr (H,W는 8의 배수)
    처리:
      - 8×8 블록 단위로 분해 → (B, 3, N, 64), N=(H/8)*(W/8)
      - 채널별 64-D를 임베딩 후 채널 결합(learned fuse) → (B, N, d_model)
      - 블록 위치 임베딩 추가(+CLS optional)
      - Transformer Encoder (self-attn only)
      - 분류 헤드
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        p_drop: float = 0.1,
        num_classes: int = 2,
        use_cls: bool = True,
        pos_grid_size=(32, 32),   # 256x256 입력 기준 32x32 블록 격자
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.d_model = d_model
        self.use_cls = use_cls

        # (1) 채널별 블록 임베딩: 64 -> d_model
        self.block_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        # (2) 채널 결합: [Y, Cb, Cr] 임베딩을 concat 후 fuse → d_model
        self.channel_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.LayerNorm(d_model),
        )

        # (3) 위치 임베딩(블록 격자용)
        self.block_pos = LearnableAbsPos2D(d_model, grid_size=pos_grid_size)

        # (4) CLS 토큰 (옵션)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        # (5) Transformer Encoder (Self-Attn 전용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=p_drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

        # (6) Head
        self.head = nn.Linear(d_model, num_classes) if num_classes is not None else None

    @staticmethod
    def _blockify_8x8(x):
        """
        x: (B, 3, H, W) -> (B, 3, N, 64), N=(H/8)*(W/8)
        """
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0, "H and W must be multiples of 8."
        Hb, Wb = H // 8, W // 8
        # (B, C, Hb, 8, Wb, 8) -> (B, C, Hb*Wb, 64)
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, Hb * Wb, 64)
        return x, Hb, Wb

    def forward_features(self, x):
        """
        x: (B,3,H,W)  (YCbCr 권장, H,W는 8의 배수)
        return: (B, d_model)
        """
        B, C, H, W = x.shape
        blocks, Hb, Wb = self._blockify_8x8(x)          # (B,3,N,64)
        N = Hb * Wb

        # 채널별 64→d 임베딩
        # (B,3,N,64) -> (B,N,3,d)
        emb = self.block_proj(blocks)                   # broadcasting over last dim: (B,3,N,64)->(B,3,N,d)
        emb = emb.permute(0, 2, 1, 3).contiguous()      # (B,N,3,d)

        # 채널 결합: concat 후 fuse → (B,N,d)
        tokens = self.channel_fuse(emb.view(B, N, 3 * self.d_model))  # (B,N,d)

        # 위치 임베딩
        pos = self.block_pos(Hb, Wb)                    # (1,N,d)
        tokens = tokens + pos

        # CLS 붙이기 (선택)
        if self.use_cls:
            cls = self.cls_token.expand(B, 1, self.d_model)  # (B,1,d)
            x_seq = torch.cat([cls, tokens], dim=1)          # (B,1+N,d)
        else:
            x_seq = tokens                                   # (B,N,d)

        # Transformer 인코딩 (self-attn only)
        x_seq = self.encoder(x_seq)                          # (B,1+N or N,d)
        x_seq = self.norm(x_seq)

        # 풀링
        if self.use_cls:
            rep = x_seq[:, 0]                                # (B,d)  # CLS
        else:
            rep = x_seq.mean(dim=1)                          # (B,d)  # mean pooling

        return rep

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat) if self.head is not None else feat


def build_dctformer(version="small", num_classes=2):
    """
    version: "small" | "base" | "large"
    """
    if version == "small":
        return DCTFormer(d_model=128, nhead=4, depth=4, num_classes=num_classes, pos_grid_size=(32,32))
    elif version == "base":
        return DCTFormer(d_model=256, nhead=8, depth=6, num_classes=num_classes, pos_grid_size=(32,32))
    elif version == "large":
        return DCTFormer(d_model=384, nhead=12, depth=8, num_classes=num_classes, pos_grid_size=(32,32))
    else:
        raise ValueError(f"Unknown version: {version}")
