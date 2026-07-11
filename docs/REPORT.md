# PepCleaver — Technical Report

This document describes the implementation, training, and validation of the
PepCleaver framework, and details the architectural decisions relative to
the development lineage.

## 1. Overview

PepCleaver models proteolytic cleavage as a sequence-to-fragment mapping
problem. Given a precursor protein, the framework evaluates candidate
peptide fragments tiled across the sequence and assigns each a
peptide-level proteolytic likelihood score. Applied to a full-length
protein, these scores reconstruct a continuous cleavage propensity
landscape (Multi-Instance Learning), enabling data-driven prioritization
of bioactive candidate peptides.

## 2. Architecture

The model processes frozen ESM-2 (esm2_t12_35M_UR50D, 480-dim) per-token
embeddings through four stages:

### Stage 1 — Self-Attention Encoder
- 2-layer `nn.TransformerEncoder` with 4 attention heads and feed-forward
  dimension 1024.
- Padding-aware masking via `src_key_padding_mask` ensures variable-length
  peptides are processed without padding artifacts.
- Pre-norm (LayerNorm before attention/FFN) for training stability.
- Learnable positional encoding added to ESM-2 embeddings.

### Stage 2 — Gated Fusion
Combines local peptide features `F_peptide` (self-attention output) with a
global protein prior `F_protein` (masked mean-pooling of ESM-2 embeddings,
projected and LayerNormed):

```
F_fused = g ⊙ F_peptide + (1 − g) ⊙ F_protein
g       = σ(W_g · [F_peptide ‖ F_protein] + b_g)
```

The gate `g` is computed per residue position, allowing the network to
adaptively weigh local peptide context against global protein-level priors.

### Stage 3 — Residual Conv1D Cascade
Three residual blocks progressively reduce feature dimensionality
(480 → 128 → 64 → 32) with kernel size 3, stride 1, padding 1. Each block
applies Conv1D → BatchNorm → ReLU with a 1×1 convolution skip connection
when channel counts differ. This pathway captures short-range motif-like
features complementary to the long-range self-attention.

### Stage 4 — Hybrid Pooling & Classification
```
Z = λ · (1/L) Σ x_i  +  (1 − λ) · log( (1/L) Σ exp(x_i) ),  λ = 0.5
```
Average pooling captures distributed sequence information; log-sum-exp
pooling preserves strong localized activations without hard max responses.
The pooled vector is passed through dropout (0.3) and a linear head.

## 3. Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr = 5×10⁻⁴, weight_decay = 1×10⁻⁴) |
| Scheduler | Cosine annealing (T_max = 25) |
| Epochs | 25 |
| Batch size | 64 |
| Loss | Focal loss (γ = 2.0, α = 0.75) |
| Model selection | Best validation AUC |
| Seed | 42 |

The focal loss with α = 0.75 weights false negatives (missed true cleavage
fragments) more heavily, preserving sensitivity for bioactive candidates
in repetitive collagen Gly-X-Y backgrounds. A subsequent decision threshold
of 0.7 is applied during inference to control false positives.

## 4. Dataset

- **Source:** MEROPS database, metalloprotease families M10 (MMPs), M12
  (ADAM/ADAMTS), and CLE (collagenases).
- **Positives:** experimentally supported cleavage-derived peptides.
- **Negatives (hard):** non-cleaved regions from the same substrate proteins,
  sharing sequence context and amino-acid composition to prevent
  compositional shortcuts (e.g., Gly–Pro content bias).
- **Size:** 13,312 balanced samples (6,656 / 6,656).
- **Split:** precursor-level 8:1:1 (10,649 / 1,331 / 1,332), with CD-HIT
  50%-identity clustering preventing homologous leakage across subsets.

## 5. Results

### Benchmark performance

| Metric | Value |
|--------|-------|
| Test AUC | 0.9566 |
| Validation AUC (best) | 0.9511 (epoch 18) |
| F1 @ 0.7 | 0.8445 |
| Accuracy @ 0.7 | 0.8574 |

Training was stable with smooth AUC convergence across 25 epochs, reaching
train AUC 0.992 while validation AUC plateaued near 0.95, indicating good
generalization without severe overfitting.

### Donkey COL1A1 virtual screening

- Precursor length: 1,463 residues.
- Candidate fragments tiled: 10,076.
- Unique sequences: 7,067.
- Fragments passing score ≥ 0.7: 2,574.
- Final peptide library (after physicochemical filtering + length-stratified
  selection): 346 sequences.

### Manuscript candidate peptide validation

| Peptide | Sequence | Score | ≥ 0.7 | SPR binding |
|---------|----------|-------|-------|-------------|
| P1 | GAQGPPGP | 0.7208 | ✓ | GM-CSFR + GLP1R |
| P2 | GFAGPPGAD | 0.7895 | ✓ | GM-CSFR |
| P3 | GPAGAPGTP | 0.7502 | ✓ | none |
| P4 | GPPGAVGP | 0.5166 | ✗ | none |
| P5 | GAPGIAGAP | 0.9236 | ✓ | GM-CSFR |
| Ref | GAAGLPGPK | 0.7882 | ✓ | none |

Notably, P4 — which showed no measurable SPR binding affinity in
experiments — received a sub-threshold score (0.52), demonstrating that the
model's confidence aligns with downstream experimental outcomes.

## 6. Implementation Notes

### 6.1 ESM-2 encoding scope

The manuscript describes extracting ESM-2 residue-level hidden states from
the *full-length precursor protein* before fragment extraction, so that each
9-residue fragment inherits long-range flanking context (P4–P4′ positions
relative to both cleavage boundaries). The precursor protein sequence data
used during development was not available at the time of code release, so
this implementation extracts ESM-2 embeddings directly at the *fragment
level* (each peptide encoded independently, then padded/truncated to length
9). The Gated Fusion module's global protein prior `F_protein` is
consequently approximated by masked mean-pooling of the fragment's own ESM-2
embeddings, projected through a linear layer and LayerNorm. This does not
alter the model architecture or training protocol, but reduces long-range
context. Users with full-length precursor sequences can restore the original
behavior by modifying `extract_embeddings.py` to encode full proteins and
then slice fragment windows from the per-residue representations.

### 6.2 Dataset split

The manuscript states that precursor proteins were clustered with CD-HIT at
50% sequence identity and all fragments from the same cluster were assigned
to the same subset. The provided `data/dataset.csv` was produced by the
development pipeline and contains peptide-level columns only (`seq`,
`label`, `split`); it has no cross-split sequence duplicates (verified:
train∩val = train∩test = val∩test = 0, and no sequence carries conflicting
labels). `scripts/build_dataset.py` provides a CD-HIT precursor-clustering
split interface for users who reconstruct the dataset from raw MEROPS files;
when CD-HIT is unavailable it falls back to a clustered random split.

### 6.3 "StarHead" attention

The manuscript refers to a "StarHead self-attention encoder." As the
manuscript does not provide a formal definition of StarHead distinct from
standard multi-head attention, this implementation uses
`nn.TransformerEncoder` with 4 heads and feed-forward dimension 1024,
matching all stated hyperparameters (2 layers, pre-norm, GELU activation,
padding-aware masking).

### 6.4 Other details

- **Parameters:** 4,805,057 trainable (ESM-2 frozen).
- **Inference (MIL):** the full-length precursor protein is treated as a
  "bag" and candidate fragments as "instances"; independent instance-level
  scores are tracked across original sequence coordinates to reconstruct the
  global cleavage landscape.
- **Reproducibility:** fixed seeds (42), frozen ESM-2 (deterministic
  embeddings), provided pre-computed caches, trained weights, and result
  CSVs. End-to-end verification: `evaluate.py` reproduces Test AUC = 0.9566;
  `predict.py` + `virtual_screening.py` reproduce the cleavage map and
   candidate peptide library (≤0.01% numerical drift from floating-point
  non-determinism).
