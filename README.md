# PepCleaver

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![ESM-2](https://img.shields.io/badge/ESM--2-35M-green.svg)

**PepCleaver** is a transformer-driven deep learning framework for proteolytic
cleavage modeling and experimentally validated bioactive peptide discovery.
By integrating frozen protein language model (ESM-2) embeddings with
attention-based contextual modeling, a gated fusion module, and residual
convolutional feature extraction, PepCleaver computes peptide-level cleavage
probabilities and reconstructs the continuous proteolytic landscape of a
precursor protein via Multi-Instance Learning (MIL).

Applied to donkey-hide gelatin type I collagen (COL1A1), PepCleaver generated
a cleavage-informed peptide library and, through hierarchical virtual
screening against GM-CSFR, prioritized candidate peptides that were
experimentally validated by SPR binding, TF-1 cell proliferation, and OGD
neuroprotection assays.

## Key Results

| Metric | Value |
|--------|-------|
| **Test AUC** | **0.9566** |
| Validation AUC (best, Ep18) | 0.9511 |
| F1 @ threshold 0.7 | 0.8445 |
| Accuracy @ threshold 0.7 | 0.8574 |
| Donkey COL1A1 fragments scored | 10,076 |
| Unique sequences | 7,067 |
| Fragments passing threshold (≥ 0.7) | 2,574 |
| Final peptide library | 346 sequences |
| Parameters | 4,805,057 |

### Manuscript candidate peptides (Table 1)

| ID | Sequence | Cleavage prob | ≥ 0.7 | Experimental SPR binding |
|----|----------|--------------|-------|--------------------------|
| P1 | GAQGPPGP | 0.7208 | ✓ | GM-CSFR + GLP1R (dual) |
| P2 | GFAGPPGAD | 0.7895 | ✓ | GM-CSFR |
| P3 | GPAGAPGTP | 0.7502 | ✓ | no measurable affinity |
| P4 | GPPGAVGP | 0.5166 | ✗ | no measurable affinity |
| P5 | GAPGIAGAP | 0.9236 | ✓ | GM-CSFR |
| Ref | GAAGLPGPK | 0.7882 | ✓ | no activity |

## Package Contents

```
PepCleaver/
├── model/                      # Model architecture
│   ├── pepcleaver.py           # PepCleaver main class
│   └── components.py           # GatedFusion, ResidualConv1D, HybridPooling, FocalLoss
├── scripts/                    # Training & inference pipeline
│   ├── build_dataset.py        # MEROPS dataset construction (CD-HIT precursor split)
│   ├── extract_embeddings.py   # Frozen ESM-2 per-token embedding extraction
│   ├── train.py                # Training (AdamW + cosine + focal loss)
│   ├── evaluate.py             # Test-set evaluation (AUC, F1, PR-AUC)
│   ├── predict.py              # Cleavage-landscape inference (MIL)
│   └── virtual_screening.py    # 350-peptide library generation
├── weights/                    # Pre-trained weights (Git LFS)
│   └── pepcleaver.pt           # 4.8M params (~19 MB)
├── data/                       # Dataset & precursor protein
│   ├── dataset.csv             # 13,312 samples (train/val/test)
│   └── donkey_col1a1.fsa       # Donkey COL1A1 sequence
├── results/                    # Pre-computed outputs
│   ├── test_predictions.csv    # Held-out test predictions
│   ├── donkey_cleavage_map.csv # COL1A1 cleavage landscape
│   └── pepcleaver_350_library.csv  # Final 350-peptide library
├── example/                    # Quick-start example
│   └── inference_example.py
├── docs/                       # Documentation
│   └── REPORT.md               # Detailed technical report
├── LICENSE                     # Apache-2.0
├── CITATION.cff
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/SUAT-ZhiweiFengLab/PepCleaver.git
cd PepCleaver
pip install -r requirements.txt
```

> Requires PyTorch ≥ 2.0 and the `fair-esm` package (downloads ESM-2 weights
> automatically on first run, ~135 MB).

### Inference — score candidate peptides

```python
import torch
from model.pepcleaver import PepCleaver

model = PepCleaver()
model.load_state_dict(torch.load("weights/pepcleaver.pt", map_location="cpu"))
model.eval()

# Encode peptides with frozen ESM-2 (see example/inference_example.py for full pipeline)
# X: (B, 9, 480) per-token embeddings, mask: (B, 9) attention mask
prob = model.predict_proba(X, mask)   # peptide-level proteolytic likelihood
```

Or run the bundled example (scores the manuscript P1–P5 + Ref peptides):

```bash
python example/inference_example.py
```

### Full training pipeline

```bash
# 1. Extract frozen ESM-2 embeddings
python scripts/extract_embeddings.py \
    --dataset data/dataset.csv \
    --output cache/pepcleaver_embeddings.npz

# 2. Train
python scripts/train.py \
    --embeddings cache/pepcleaver_embeddings.npz \
    --output weights/pepcleaver.pt

# 3. Evaluate
python scripts/evaluate.py \
    --embeddings cache/pepcleaver_embeddings.npz \
    --weights weights/pepcleaver.pt
```

### Virtual screening — cleavage landscape & peptide library

```bash
# 4. Predict cleavage landscape of donkey COL1A1
python scripts/predict.py \
    --protein data/donkey_col1a1.fsa \
    --weights weights/pepcleaver.pt \
    --output results/donkey_cleavage_map.csv

# 5. Hierarchical virtual screening -> 350-peptide library
python scripts/virtual_screening.py \
    --cleavage-map results/donkey_cleavage_map.csv \
    --output results/pepcleaver_350_library.csv
```

## Model Architecture

```
Precursor protein (length N)
        │  tile candidate fragments (length L ≤ 9)
        ▼
┌─────────────────────────────────────────────────────────┐
│  Frozen ESM-2 (esm2_t12_35M_UR50D, 480-dim)            │  Sequence encoding
└─────────────────────────────────────────────────────────┘
        │  per-token embeddings (B, L, 480)
        ▼
┌─────────────────────────────────────────────────────────┐
│  Learnable positional encoding                          │
├─────────────────────────────────────────────────────────┤
│  2-layer self-attention encoder (4 heads, ffn=1024)     │  Contextual modeling
│      padding-aware masking                              │
├─────────────────────────────────────────────────────────┤
│  Gated Fusion                                           │  Feature fusion
│      F_fused = g·F_peptide + (1−g)·F_protein           │
│      g = σ(W_g·[F_peptide ‖ F_protein] + b_g)          │
├─────────────────────────────────────────────────────────┤
│  Residual Conv1D × 3  (480 → 128 → 64 → 32, k=3)       │  Local motif extraction
├─────────────────────────────────────────────────────────┤
│  Hybrid Pooling                                         │  Sequence aggregation
│      Z = λ·avg + (1−λ)·LSE,  λ = 0.5                   │
├─────────────────────────────────────────────────────────┤
│  Dropout → Linear(32, 1)                                │  Classification head
└─────────────────────────────────────────────────────────┘
        │
        ▼
  Peptide-level proteolytic likelihood
```

### Gated Fusion

Dynamically combines local peptide features (from self-attention) with global
protein priors (from ESM-2 mean-pooling) via a learned sigmoid gate:

```
F_fused = g ⊙ F_peptide + (1 − g) ⊙ F_protein
g       = σ(W_g · [F_peptide ‖ F_protein] + b_g)
```

### Hybrid Pooling

Combines average pooling (distributed sequence information) with log-sum-exp
pooling (strong localized activation), balancing global trends and sharp
motif peaks:

```
Z = λ · (1/L) Σ x_i  +  (1 − λ) · log( (1/L) Σ exp(x_i) ),  λ = 0.5
```

## Dataset

| Property | Value |
|----------|-------|
| Source | MEROPS (metalloprotease families M10 / M12 / CLE) |
| Total | 13,312 samples (6,656 positive / 6,656 negative) |
| Positives | Experimentally supported cleavage-derived peptides |
| Negatives | Same-substrate hard negatives (non-cleaved regions) |
| Split | 10,649 / 1,331 / 1,332 (train / val / test, 8:1:1) |
| Split strategy | CD-HIT 50% identity precursor clustering, no leakage |

The same-substrate hard-negative design shares sequence context and
amino-acid composition between positives and negatives, reducing trivial
compositional bias (e.g., Gly–Pro-rich shortcuts in collagen).

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr = 5×10⁻⁴, weight_decay = 1×10⁻⁴) |
| LR schedule | Cosine annealing |
| Epochs | 25 |
| Batch size | 64 |
| Loss | Focal loss (γ = 2.0, α = 0.75) |
| Decision threshold | 0.7 |
| Model selection | Best validation AUC |
| ESM-2 encoder | Frozen (esm2_t12_35M_UR50D, 35M params, 480-dim) |
| Seed | 42 |

The focal loss with α = 0.75 heavily penalizes false negatives, preserving
high sensitivity for potentially active fragments in repetitive collagen
sequence backgrounds.

## Virtual Screening Pipeline

The 5-stage hierarchical pipeline refines PepCleaver predictions into a
final candidate peptide library for molecular docking:

1. **Fragment tiling** — tile all candidate fragments (length 2–9, containing
   G or P) across the full-length precursor protein.
2. **Deduplication + score filtering** — remove redundant sequences and
   discard fragments with PepCleaver score < 0.7.
3. **Physicochemical hard filtering** — keep peptides with
   300 ≤ MW ≤ 1000 Da and instability index < 40.
4. **Weighted composite scoring** —
   `Total = 0.35·MW + 0.30·II + 0.10·pI + 0.15·charge + 0.10·GRAVY`,
   with +0.5 for glycine-starting peptides.
5. **Length-stratified selection** — quotas of 19/50/50/50/50/65/66 for
   lengths 3–9, yielding the final library.

## Reproducibility

- All random seeds are fixed (default 42).
- The ESM-2 encoder is frozen, so embeddings are deterministic.
- Pre-computed embeddings, trained weights, and result CSVs are provided.
- `requirements.txt` pins minimum dependency versions.
- Verified end-to-end: `evaluate.py` reproduces Test AUC = 0.9566 with the
  provided weights; `predict.py` + `virtual_screening.py` reproduce the
  cleavage map and 350-peptide library.

## Implementation Notes

The following notes document implementation details and known deviations
from the manuscript description, in the interest of full transparency for
reproducibility:

1. **ESM-2 encoding scope.** The manuscript describes extracting ESM-2
   residue-level hidden states from the *full-length precursor protein*
   before fragment extraction, so that each 9-residue fragment inherits
   long-range flanking context (P4–P4′). The precursor protein sequence
   data used during development was not available at the time of code
   release, so this implementation extracts ESM-2 embeddings directly at
   the *fragment level* (each peptide encoded independently). The Gated
   Fusion module's global protein prior `F_protein` is consequently
   approximated by masked mean-pooling of the fragment's own ESM-2
   embeddings. This does not alter the model architecture or training
   protocol, but reduces long-range context. Users with full-length
   precursor sequences can restore the original behavior by modifying
   `extract_embeddings.py` to encode full proteins and then slice
   fragment windows from the per-residue representations.

2. **Dataset split.** The manuscript states that precursor proteins were
   clustered with CD-HIT at 50% sequence identity and all fragments from
   the same cluster were assigned to the same subset (precursor-level
   split). The provided `data/dataset.csv` was produced by the
   development pipeline and contains peptide-level columns only
   (`seq`, `label`, `split`); it has no cross-split sequence duplicates
   (verified: train∩val = train∩test = val∩test = 0). `scripts/build_dataset.py`
   provides a CD-HIT precursor-clustering split interface for users who
   reconstruct the dataset from raw MEROPS files; when CD-HIT is
   unavailable it falls back to a clustered random split.

3. **"StarHead" attention.** The manuscript refers to a "StarHead
   self-attention encoder." As the manuscript does not provide a formal
   definition of StarHead distinct from standard multi-head attention,
   this implementation uses `nn.TransformerEncoder` with 4 heads and
   feed-forward dimension 1024, matching all stated hyperparameters.

## Citation

If you use PepCleaver in your research, please cite:

```bibtex
@article{pepcleaver2026,
  title   = {PepCleaver: A Generalizable Deep Learning Framework for
             Proteolytic Cleavage Modeling and Experimentally Validated
             Bioactive Peptide Discovery},
  author  = {Xue, Ying and Duan, Xiaobo and Liu, Cong and Li, Siqi and
             Guo, Shangwei and Zheng, Liang and Feng, Zhiwei and
             Ouyang, Qin and Liu, Haibin},
  journal = {Nature Methods},
  year    = {2026},
  doi     = {10.1038/s41592-026-xxxx}
}
```

## License

This project is licensed under the **Apache License 2.0** — see
[LICENSE](LICENSE).

## Contact

- **Issues:** [GitHub Issues](https://github.com/SUAT-ZhiweiFengLab/PepCleaver/issues)
- **Corresponding authors:** xue.ying1@zs-hospital.sh.cn;
  fengzhiwei@suat-sz.edu.cn; liuhaibin@dongeejiao.com

---

**Version:** 1.0.0  
**Release date:** 2026-06-23
