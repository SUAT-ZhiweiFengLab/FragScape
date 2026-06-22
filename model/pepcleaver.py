"""PepCleaver: a deep learning framework for proteolytic cleavage modeling.

Architecture
------------
ESM-2 per-token embeddings (frozen, 480-dim)
    -> learnable positional encoding
    -> 2-layer 4-head self-attention encoder (feed-forward dim 1024)
    -> Gated Fusion (local peptide features + global protein prior)
    -> cascade of residual Conv1D blocks (480 -> 128 -> 64 -> 32)
    -> hybrid average + log-sum-exp pooling (lambda = 0.5)
    -> dropout -> linear classification head

The network outputs a peptide-level proteolytic likelihood score that
reconstructs the continuous cleavage propensity landscape of a parent
protein when applied to all candidate fragments (Multi-Instance Learning).
"""
import torch
import torch.nn as nn

from .components import GatedFusion, ResidualConv1DBlock, HybridPooling


class PepCleaver(nn.Module):
    """PepCleaver proteolytic cleavage model.

    Parameters
    ----------
    max_len : int
        Maximum peptide length (default 9, matching the empirical MEROPS
        cleavage-product distribution).
    emb_dim : int
        ESM-2 embedding dimension (default 480 for esm2_t12_35M_UR50D).
    n_heads : int
        Number of attention heads (default 4).
    ffn_dim : int
        Feed-forward dimension of the transformer layers (default 1024).
    n_attn_layers : int
        Number of transformer encoder layers (default 2).
    conv_dims : tuple
        Output channels of the successive residual Conv1D blocks
        (default (128, 64, 32)).
    lam : float
        Mixing weight for hybrid pooling (default 0.5).
    dropout : float
        Dropout rate before the classification head (default 0.3).
    """

    def __init__(
        self,
        max_len: int = 9,
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

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, emb_dim) * 0.02)

        # Stage 1: self-attention encoder (2 layers, 4 heads, ffn=1024)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim,
            n_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_attn_layers)

        # Global protein prior projection (ESM-2 mean-pooling -> emb_dim)
        self.protein_proj = nn.Linear(emb_dim, emb_dim)
        self.protein_norm = nn.LayerNorm(emb_dim)

        # Stage 2: Gated Fusion
        self.gated_fusion = GatedFusion(emb_dim)

        # Stage 3: residual Conv1D cascade (480 -> 128 -> 64 -> 32)
        conv_layers = []
        in_c = emb_dim
        for out_c in conv_dims:
            conv_layers.append(ResidualConv1DBlock(in_c, out_c))
            in_c = out_c
        self.conv_blocks = nn.ModuleList(conv_layers)

        # Stage 4: hybrid pooling + classification head
        self.pool = HybridPooling(lam=lam)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_dims[-1], 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute peptide-level proteolytic likelihood logits.

        Parameters
        ----------
        x : torch.Tensor
            ESM-2 per-token embeddings, shape ``(B, L, D)``.
        mask : torch.Tensor, optional
            Boolean/float mask, shape ``(B, L)``, with 1 for valid residues
            and 0 for padding. Used for both attention masking and masked
            pooling.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B,)``.
        """
        # Global protein prior: masked mean-pooling of ESM-2 embeddings
        if mask is not None:
            m = mask.unsqueeze(-1)  # (B, L, 1)
            f_protein = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            f_protein = x.mean(dim=1)
        f_protein = self.protein_norm(self.protein_proj(f_protein))

        # Positional encoding
        x = x + self.pos_enc[:, : x.size(1), :]

        # Stage 1: self-attention (padding-aware)
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)
        f_peptide = x  # (B, L, D)

        # Stage 2: Gated Fusion of local peptide features and global prior
        x = self.gated_fusion(f_peptide, f_protein)

        # Stage 3: residual Conv1D blocks  (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1)
        for block in self.conv_blocks:
            x = block(x)

        # Stage 4: hybrid pooling + classification
        x = self.pool(x, mask)  # (B, 32)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Return proteolytic likelihood probabilities (sigmoid)."""
        return torch.sigmoid(self.forward(x, mask))
