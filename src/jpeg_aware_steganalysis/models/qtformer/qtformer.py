import torch
import torch.nn as nn
import torch.nn.functional as F


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


# 8x8 주파수 격자(=QTable)에 대한 위치 인코딩
class LearnableFreqPos2D(nn.Module):
    def __init__(self, d_model: int, grid_size=(8, 8)):
        super().__init__()
        gh, gw = grid_size
        self.d_model = d_model
        self.pos = nn.Parameter(torch.zeros(1, d_model, gh, gw))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self):
        pos = self.pos  # (1,d,8,8)
        pos = pos.permute(0, 2, 3, 1).contiguous().view(1, 64, self.d_model)  # (1,64,d)
        return pos


class BlockToQTAttn(nn.Module):
    """
    쿼리: 블록 시퀀스 (B, N, d)
    키/밸류: QTable 시퀀스 (B, 64, d)
    => 블록별로 서로 다른 QTable-주의 가중치가 생김
    """
    def __init__(self, d_model=64, nhead=4, p_drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=p_drop, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x_blocks, q_tokens, need_attn=False):
        # x_blocks: (B,N,d), q_tokens: (B,64,d)
        y, attn = self.attn(
            query=x_blocks, key=q_tokens, value=q_tokens,
            need_weights=need_attn, average_attn_weights=False
        )  # y: (B,N,d), attn: (B, heads, N, 64) if need_attn
        x = self.ln1(x_blocks + self.drop(y))
        z = self.ffn(x)
        x = self.ln2(x + self.drop(z))
        return x, attn  # x:(B,N,d)


class BlockSelfAttn(nn.Module):
    def __init__(self, d_model=64, nhead=4, p_drop=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=p_drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        return self.encoder(x)  # (B,N,d)


class QTFormer(nn.Module):
    """
    입력:
      input: (B, 3, H, W)   # Y, Cb, Cr 이미지 (H, W는 8의 배수)
      qt:    (B, 2, 8, 8)   # [0]=Y, [1]=Cb/Cr QTable

    처리:
      - 8x8 블록 단위 토큰화 → (B, 3, N, 64), N = (H/8)*(W/8)
      - 각 채널: 블록(쿼리) ↔ QTable 64-토큰(키/밸류) 크로스-어텐션
      - 블록 간 self-attn 후 전역 풀링
      - 채널 결합 → 분류
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        depth: int = 2,
        p_drop: float = 0.1,
        num_classes: int = 2,
        pos_grid_size=(32, 32),
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.d_model = d_model

        # 블록 projection (64 → d)
        self.block_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        # QTable 스칼라(1) → d 임베딩을 64개 위치에 공유 적용
        self.qt_scalar_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        # 위치 인코딩
        self.block_pos = LearnableAbsPos2D(d_model, grid_size=pos_grid_size)   # (Hb,Wb)
        self.freq_pos  = LearnableFreqPos2D(d_model, grid_size=(8, 8))        # (8,8)

        # Cross-Attn 스택
        self.cross_layers = nn.ModuleList(
            [BlockToQTAttn(d_model=d_model, nhead=nhead, p_drop=p_drop) for _ in range(depth)]
        )

        # 블록 간 Self-Attn
        self.self_attn = BlockSelfAttn(d_model=d_model, nhead=nhead, p_drop=p_drop)

        # 채널 결합 및 head
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, num_classes) if num_classes is not None else None

        # 디버깅/시각화를 위한 마지막 cross-attn 맵 저장용(선택)
        self.last_attn_maps = {}

    @staticmethod
    def _blockify_8x8(x):
        B, C, H, W = x.shape

        assert H % 8 == 0 and W % 8 == 0, f"H and W must be multiples of 8."
        Hb, Wb = H // 8, W // 8
        x = x.view(B, C, Hb, 8, Wb, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, Hb * Wb, 64)  # (B,C,N,64)
        return x, Hb, Wb

    def _qt_tokens(self, qtable_flat):
        """
        qtable_flat: (B, 64)
        반환: (B, 64, d)  + 주파수 위치 인코딩
        """
        B = qtable_flat.size(0)
        q = qtable_flat.view(B, 64, 1)                    # (B,64,1)
        q = self.qt_scalar_proj(q)                        # (B,64,d)
        q = q + self.freq_pos()                           # (1,64,d) broadcast
        
        return q

    def _channel_forward(self, block_tokens, qtable_flat, Hb, Wb, tag=None):
        """
        block_tokens: (B, N, 64)  — 해당 채널의 모든 8x8 블록
        qtable_flat : (B, 64)     — 해당 채널의 QTable(64 스칼라)
        """
        # 블록 임베딩 + 블록 위치
        x = self.block_proj(block_tokens)                 # (B,N,d)
        x = x + self.block_pos(Hb, Wb)                    # (1,N,d)

        # QTable을 64 토큰으로
        q_tokens = self._qt_tokens(qtable_flat)           # (B,64,d)

        # 크로스 어텐션 (depth번)
        attn_to_save = None
        for li, layer in enumerate(self.cross_layers):
            need_attn = (li == len(self.cross_layers) - 1)
            x, attn = layer(x, q_tokens, need_attn=need_attn)   # x:(B,N,d)
            if need_attn:
                # attn: (B, heads, N, 64) → 채널/헤드 평균으로 (B,N,64)
                attn_to_save = attn.mean(dim=1)

        # 블록 간 상호작용
        x = self.self_attn(x)                             # (B,N,d)

        # 전역 풀링으로 대표 벡터
        rep = x.mean(dim=1)                               # (B,d)

        if tag is not None and attn_to_save is not None:
            self.last_attn_maps[tag] = attn_to_save       # (B,N,64)

        return rep

    def forward_features(self, input, qt):
        """
        input: (B,3,H,W), qt: (B,2,8,8)  # [0]=Y, [1]=Cb/Cr
        """
        ycbcr = input
        qtable = qt

        blocks, Hb, Wb = self._blockify_8x8(ycbcr)        # (B,3,N,64)
        q_flat = qtable.view(qtable.size(0), 3, 64)       # (B,3,64)

        y_rep  = self._channel_forward(blocks[:, 0], q_flat[:, 0], Hb, Wb, tag="Y")
        cb_rep = self._channel_forward(blocks[:, 1], q_flat[:, 1], Hb, Wb, tag="Cb")
        cr_rep = self._channel_forward(blocks[:, 2], q_flat[:, 1], Hb, Wb, tag="Cr")

        feat = torch.cat([y_rep, cb_rep, cr_rep], dim=1)  # (B, 3*d)
        feat = self.fuse(feat) 
# (B, d)
        return feat

    def forward(self, input, qt):
        feat = self.forward_features(input, qt)
        
        return self.head(feat) if self.head is not None else feat

def build_qtformer(version="small", num_classes=2):
    """
    version: "small" | "base" | "large"
    """
    if version == "small":
        return QTFormer(d_model=64, nhead=4, depth=2, num_classes=num_classes)
    elif version == "base":
        return QTFormer(d_model=128, nhead=8, depth=4, num_classes=num_classes)
    elif version == "large":
        return QTFormer(d_model=256, nhead=8, depth=6, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown QTFormer version: {version}")