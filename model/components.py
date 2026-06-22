"""PepCleaver architectural components.

Implements the modules described in the PepCleaver manuscript:
  * Gated Fusion module
  * Residual Conv1D blocks
  * Hybrid (average + log-sum-exp) pooling
  * Focal loss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Gated Fusion of local peptide features and global protein priors.

    Implements the equation from the manuscript:

        F_fused = g * F_peptide + (1 - g) * F_protein
        g       = sigmoid(W_g * [F_peptide || F_protein] + b_g)

    where ``||`` denotes concatenation along the feature dimension and
    ``*`` denotes the Hadamard (element-wise) product. The gating vector
    ``g`` is computed per residue position, allowing the network to
    dynamically weigh local peptide-level features (captured by the
    self-attention encoder) against global protein-level priors.

    Parameters
    ----------
    dim : int
        Feature dimension of ``F_peptide`` and ``F_protein``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_fc = nn.Linear(dim * 2, dim)

    def forward(self, f_peptide: torch.Tensor, f_protein: torch.Tensor) -> torch.Tensor:
        # f_peptide: (B, L, D)  per-residue local peptide features
        # f_protein: (B, D)     global protein prior (pooled)
        L = f_peptide.size(1)
        f_protein_exp = f_protein.unsqueeze(1).expand(-1, L, -1)
        g = torch.sigmoid(self.gate_fc(torch.cat([f_peptide, f_protein_exp], dim=-1)))
        return g * f_peptide + (1.0 - g) * f_protein_exp


class ResidualConv1DBlock(nn.Module):
    """Residual 1D convolution block.

    Conv1d(kernel=3, stride=1, padding=1) -> BatchNorm -> ReLU + residual.
    A 1x1 convolution is used as the skip connection when the input and
    output channel counts differ.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = torch.relu(self.bn(self.conv(x)))
        return out + residual


class HybridPooling(nn.Module):
    """Hybrid pooling combining average and log-sum-exp pooling.

    Implements the manuscript equation:

        Z = lambda * (1/L) * sum_i x_i  +  (1 - lambda) * log( (1/L) * sum_i exp(x_i) )

    Average pooling captures distributed sequence-level information, while
    log-sum-exp pooling preferentially preserves strong localized
    activation patterns without relying on a hard maximum response.

    Parameters
    ----------
    lam : float
        Mixing weight ``lambda`` balancing average and LSE terms (default 0.5).
    """

    def __init__(self, lam: float = 0.5):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, D, L), mask: (B, L) with 1 for valid positions
        if mask is not None:
            m = mask.unsqueeze(1)  # (B, 1, L)
            L = m.sum(dim=2).clamp(min=1.0)  # (B, 1)
            avg = (x * m).sum(dim=2) / L  # (B, D)
            x_masked = x.masked_fill(m == 0, -1e9)
            lse = torch.logsumexp(x_masked, dim=2) - torch.log(L)  # (B, D)
        else:
            L = x.size(2)
            avg = x.mean(dim=2)
            lse = torch.logsumexp(x, dim=2) - math.log(L)
        return self.lam * avg + (1.0 - self.lam) * lse


class FocalLoss(nn.Module):
    """Focal loss for binary classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The asymmetric weighting (alpha=0.75) heavily penalizes false negatives
    (missed true proteolytic fragments), preserving high sensitivity for
    bioactive candidates embedded in repetitive sequence backgrounds.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default 2.0).
    alpha : float
        Weight for the positive class (default 0.75).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * bce).mean()
