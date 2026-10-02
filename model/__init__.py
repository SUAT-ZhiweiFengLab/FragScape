"""FragScape model components.

Building blocks of the FragScape proteolytic cleavage framework:
- FragScape: the local-context (P20-P20') cleavage model
- GatedFusion: dynamic fusion of local peptide features and global protein priors
- ResidualConv1DBlock: residual 1D convolution block
- HybridPooling: average + log-sum-exp hybrid pooling
- FocalLoss: focal loss for hard-negative optimization
"""
from .fragscape import FragScape
from .components import (
    GatedFusion,
    ResidualConv1DBlock,
    HybridPooling,
    FocalLoss,
)

__all__ = [
    "FragScape",
    "GatedFusion",
    "ResidualConv1DBlock",
    "HybridPooling",
    "FocalLoss",
]
