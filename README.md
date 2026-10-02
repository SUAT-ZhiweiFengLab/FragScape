# FragScape

**FragScape** (formerly *PepCleaver*) is a deep-learning framework for predicting
protease cleavage sites and the bioactive fragments they release from a protein
sequence. It couples the frozen protein language model **ESM-2 (t12, 35M)** with
a lightweight Transformer head that operates on **local sequence context**
(+-20 residues around the candidate fragment) and a gated fusion of
peptide-level and protein-level representations.

> This repository is a **minimal, verified release**: it keeps only the artifacts
> that reproduce a concrete, checkable result. Everything else (older model
> architectures, dataset builders, training code, benchmark dumps) was moved to a
> recoverable trash directory during the FragScape rename.

---

## Verified result

| peptide | weights | context | score |
|---|---|---|---|
| `GFPGTPGLPGVK` (COL2A1) | `weights/fragscape.pt` | +-20 aa | **0.6644** |

`0.6644` is the "balanced point" reported in
`docs/FragScape_v1.0_v2.1_comprehensive_report.md`: the trade-off between overall
test AUC (0.7513) and the confidence assigned to the COL2A1 target.

Known erratum: an earlier, **context-free** path produced `0.5509` for the same
peptide. That value came from a short-peptide-only model that encoded the
fragment without any flanking context (through `scripts/predict_long.py` and a
`col2a1_10_12_predictions.csv` dump generated on 2026-08-03). It is not the
balanced-point result. Both the script and the dump were removed from this
release; see `docs/FragScape_0.664_local_context_status.md` for the full
provenance trail.

---

## Quick start

```bash
# environment (torch 2.6.0+cu124)
/home/jilinan/miniconda3/bin/python -m pip install -r requirements.txt

# score the verified target: the +-20 local-context window is recomputed
# from the COL2A1 precursor with ESM-2 (this variant needs `fair-esm`)
/home/jilinan/miniconda3/bin/python scripts/predict.py \
    --peptides GFPGTPGLPGVK --protein data/human_col2a1.fsa

# several peptides at once
/home/jilinan/miniconda3/bin/python scripts/predict.py \
    --peptides GFPGTPGLPGVK,PEPTIDE2 --protein data/human_col2a1.fsa
```

Expected output:

```
device : cuda
weights: .../weights/fragscape.pt
model  : FragScape(max_len=12)  params=4,806,497
protein: data/human_col2a1.fsa  (1487 aa)

peptide          len          source      prob   flags
--------------------------------------------------------------
GFPGTPGLPGVK      12      esm2(+-20)    0.6644   >=0.6
```

The first run downloads the ESM-2 t12 35M checkpoint (~134 MB) into
`~/.cache/torch/hub/checkpoints/`; every run after that is fully offline.

### Why there is no embedding cache in this repository

Earlier builds shipped a 466 MB pickle of pre-computed embeddings
(`data/local_context_embeddings.pkl`, 20,289 peptides). It is **not** part of
this release — `data/*.pkl` is git-ignored — because it is pure derived data
and dominated the clone size. Nothing is lost:

| | cached vector | ESM-2 recomputed |
|---|---|---|
| max abs difference | — | `2.3e-06` |
| cosine similarity | — | `0.999999999999` |
| probability for `GFPGTPGLPGVK` | `0.664465` | **`0.664465`** |

The cache is only a speed-up. If you have one, `predict.py` prefers it
automatically; otherwise point `--cache /path/to/embeddings.pkl` at it.

---

## Repository layout

```
FragScape/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md
│   └── human_col2a1.fsa              # COL2A1 precursor (for --protein)
├── docs/
│   ├── REPORT.md
│   ├── FragScape_v1.0_v2.1_comprehensive_report.md
│   ├── FragScape_v2_v4_evolution_report.md
│   └── FragScape_0.664_local_context_status.md
├── model/
│   ├── __init__.py
│   ├── components.py                 # GatedFusion, ResidualConv1DBlock, ...
│   └── fragscape.py                  # FragScape transformer head
├── scripts/
│   ├── __init__.py
│   └── predict.py                    # the one supported inference entry point
└── weights/
    └── fragscape.pt                  # balanced-point weights (19 MB)
```

A fresh clone is therefore ~19 MB: only `weights/fragscape.pt` is carried by Git
LFS, and the +-20 embedding cache is regenerated rather than distributed.

---

## Model

`model/fragscape.py` defines a compact Transformer over ESM-2 per-residue
embeddings:

* sinusoidal position encoding (fixed `max_len`, inferred from the checkpoint),
* multi-head self-attention encoder blocks,
* `HybridPooling` (average + log-sum-exp) over residues,
* `GatedFusion` mixing fragment features with a global protein prior,
* a linear head producing a single cleavage logit (sigmoid -> probability).

`scripts/predict.py` reads `max_len` from the checkpoint, builds the model,
loads the state dict strictly (`missing = unexpected = 0`) and reports the
sigmoid probability for each requested peptide.

---

## Citation

See `CITATION.cff`.

## License

Apache License 2.0 — see `LICENSE`.
