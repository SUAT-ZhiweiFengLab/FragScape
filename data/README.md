# Data

Files kept in this directory are exactly those required by the single
reproducible FragScape inference path (`scripts/predict.py`).

## `human_col2a1.fsa`

Full-length human COL2A1 (collagen alpha-1(II) chain, P02458, 1487 aa) precursor
sequence, passed as `--protein` so that the +-20 local-context window of a
candidate fragment can be recomputed with ESM-2 (requires `fair-esm`).

The verified target `GFPGTPGLPGVK` sits at offset 275 (residues 276-287) inside
this 52-residue local-context window:

```
EAGKPGKAGERGPPGPQGARG GFPGTPGLPGVK GHRGYPGLDGAKGEAGAPGV
^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^
  20 residues left      fragment     20 residues right
```

## Not shipped: `local_context_embeddings.pkl`

ESM-2 (t12, 35M) per-residue embeddings for every COL2A1 10-12mer window,
**computed with +-20 residues of local context** (window `P20-P20'`): a pickled
`dict[str, np.ndarray]` with shape `(L, 480)` per peptide (`L` = fragment
length, `480` = ESM-2 t12 hidden size), holding that model's layer-12
representations for the residues of the fragment itself.

It used to live here (468,534,773 B, md5 `620df24dc2e11b46cb350971bc4e8140`).
It is now git-ignored (`data/*.pkl`) because it is pure derived data and made up
the bulk of the clone:

* it is what makes 0.6644 reproducible -- the same peptide scored **without**
  local context gives a different, much lower value -- but
* the identical vectors are produced on demand from `data/human_col2a1.fsa`
  (max |delta| = 2.3e-06, cosine 0.999999999999), yielding the very same
  probability `0.664465`.

The cache is therefore optional. `scripts/predict.py --protein
data/human_col2a1.fsa` recomputes the window per peptide, while a cache placed
here (or passed via `--cache`) is picked up automatically as a speed-up.

## Not included

Training/benchmarking datasets, MEROPS raw dumps, long-peptide dataset builds
and the v1.0 prediction dumps were removed because the model weights they
belong to are no longer part of this release. They are recoverable from the
trash directory created during the FragScape rename.
