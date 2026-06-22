"""PepCleaver model components.

Building blocks of the PepCleaver proteolytic cleavage framework:
- GatedFusion: dynamic fusion of local peptide features and global protein priors
- ResidualConv1DBlock: residual 1D convolution block
- HybridPooling: average + log-sum-exp hybrid pooling
- FocalLoss: focal loss for hard-negative optimization
"""
from .pepcleaver import PepCleaver
from .components import (
    GatedFusion,
    ResidualConv1DBlock,
    HybridPooling,
    FocalLoss,
)

__all__ = [
    "PepCleaver",
    "GatedFusion",
    "ResidualConv1DBlock",
    "HybridPooling",
    "FocalLoss",
]
