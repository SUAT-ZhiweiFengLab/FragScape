"""FragScape v2.0: 支持 2-20 肽的切割预测模型。

与 v1.0 的唯一区别: max_len 从 9 → 20 (或 15)。
其余架构完全相同：
    ESM-2 per-token embeddings (frozen, 480-dim)
    -> learnable positional encoding (1, 20, 480)
    -> 2-layer 4-head self-attention encoder (ffn=1024)
    -> Gated Fusion -> Residual Conv1D x3 (480→128→64→32)
    -> Hybrid Pooling (λ=0.5) -> Dropout -> Linear(32,1)
"""
import torch
import torch.nn as nn

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.components import GatedFusion, ResidualConv1DBlock, HybridPooling


class FragScape(nn.Module):
    """FragScape v2.0 — 支持 2-20 肽。

    Parameters
    ----------
    max_len : int
        Maximum peptide length (default 20).
    """

    def __init__(
        self,
        max_len: int = 12,
        emb_dim: int = 480,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        n_attn_layers: int = 2,
        conv_dims: tuple = (128, 64, 32),
        lam: float = 0.5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim

        self.pos_enc = nn.Parameter(torch.randn(1, max_len, emb_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            emb_dim, n_heads, dim_feedforward=ffn_dim,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        enc_kw = {}
        if hasattr(torch, "__version__") and torch.__version__.startswith("2.0."):
            enc_kw["enable_nested_tensor"] = False
        self.transformer = nn.TransformerEncoder(enc_layer, n_attn_layers, **enc_kw)

        self.protein_proj = nn.Linear(emb_dim, emb_dim)
        self.protein_norm = nn.LayerNorm(emb_dim)
        self.gated_fusion = GatedFusion(emb_dim)

        conv_layers = []
        in_c = emb_dim
        for out_c in conv_dims:
            conv_layers.append(ResidualConv1DBlock(in_c, out_c))
            in_c = out_c
        self.conv_blocks = nn.ModuleList(conv_layers)

        self.pool = HybridPooling(lam=lam)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_dims[-1], 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """x: (B, L_max, 480), mask: (B, L_max) float with 1=valid"""
        # Global protein prior
        if mask is not None:
            m = mask.unsqueeze(-1)
            f_protein = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            f_protein = x.mean(dim=1)
        f_protein = self.protein_norm(self.protein_proj(f_protein))

        x = x + self.pos_enc[:, :x.size(1), :]

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)
        f_peptide = x

        x = self.gated_fusion(f_peptide, f_protein)
        x = x.permute(0, 2, 1)
        for block in self.conv_blocks:
            x = block(x)
        x = self.pool(x, mask)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, mask))