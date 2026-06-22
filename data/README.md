# Data

This directory contains the data required to train and evaluate PepCleaver.

## `dataset.csv`

The MEROPS-derived proteolytic cleavage dataset used for model training and
benchmarking.

| Property | Value |
|----------|-------|
| Source | MEROPS database (metalloprotease families M10 / M12 / CLE) |
| Total samples | 13,312 (6,656 positive / 6,656 negative) |
| Positive samples | Experimentally supported cleavage-derived peptides |
| Negative samples | Same-substrate hard negatives (non-cleaved regions) |
| Train / Val / Test | 10,649 / 1,331 / 1,332 (8:1:1, precursor-level split) |
| Peptide length | 2–9 residues |
| Split strategy | CD-HIT clustering at 50% sequence identity, no cluster leakage |

**Columns:** `seq` (peptide sequence), `label` (1 = cleavage, 0 = non-cleavage),
`split` (train / val / test).

The dataset is curated from the public
[MEROPS database](https://www.ebi.ac.uk/merops/). To reconstruct it from raw
MEROPS files, see `scripts/build_dataset.py`.

## `donkey_col1a1.fsa`

The full-length donkey-skin type I collagen α1 chain (COL1A1) sequence used as
the precursor protein for the virtual screening case study. COL1A1 features
extensive Gly-X-Y repeating motifs, making it a biologically meaningful
template for generating collagen-derived peptide candidates.
