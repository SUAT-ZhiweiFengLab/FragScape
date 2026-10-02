#!/usr/bin/env python3
"""FragScape inference: local-context ("balanced-point") cleavage scoring.

This is the single supported inference path in this release, and it reproduces
the balanced-point model documented in
``docs/FragScape_v1.0_v2.1_comprehensive_report.md`` (Sec. 11.4):

    weights/fragscape.pt  +  +-20 residue local context

For the COL2A1 target ``GFPGTPGLPGVK`` it yields 0.6644 (>=0.6, <0.7), the
value the project settled on as the final trade-off between overall test AUC
(0.7513) and the confidence assigned to that target.

Provenance note -- the 0.5509 that is sometimes quoted for the same peptide
came from a *different, context-free* encoder (a short-peptide-only model that
scored the fragment with no flanking residues). That path, its weights and its
prediction dumps were removed from this release; see
``docs/FragScape_0.664_local_context_status.md`` for the full trail.

No embedding cache ships with this release: ``data/*.pkl`` is git-ignored,
because the ~466 MB of pre-computed vectors are pure derived data. Pass
``--protein`` (FASTA) and the +-20 window embedding is recomputed with ESM-2
(needs ``fair-esm``); that reproduces the vectors of the former cache to float
noise and yields the identical 0.664465. An optional ``--cache`` file
(dict: peptide -> (L,480)) is still honoured, and is preferred when present.

Usage
-----
    python scripts/predict.py --peptides GFPGTPGLPGVK \
        --protein data/human_col2a1.fsa
    python scripts/predict.py --peptides PEPTIDE1,PEPTIDE2   # comma-separated
    python scripts/predict.py --peptides GFPGTPGLPGVK \
        --cache data/local_context_embeddings.pkl            # optional speed-up
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from model.fragscape import FragScape

AA = "ACDEFGHIKLMNPQRSTVWY"
CTX = 20  # +-20 residue local context window (P20-P20')


def load_model(weights, device):
    state = torch.load(weights, map_location=device, weights_only=True)
    max_len = None
    for k, v in state.items():
        if "pos_enc" in k:
            max_len = v.shape[1]
    if max_len is None:
        raise RuntimeError(f"cannot infer max_len from {weights}")
    model = FragScape(max_len=max_len).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"architecture mismatch for {weights}: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()
    return model, max_len


def read_fasta(path):
    seq = ""
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += "".join(c for c in line.strip() if c in AA)
    return seq


def compute_context_embedding(pep, protein, esm_model, bc, device):
    """Embed ``pep`` with +-20 residue local context sliced from ``protein``."""
    idx = protein.find(pep)
    if idx < 0:
        raise ValueError(f"peptide {pep} not found in the supplied protein")
    start = max(0, idx - CTX)
    end = min(len(protein), idx + len(pep) + CTX)
    window = protein[start:end]
    left = idx - start
    _, _, tokens = bc([("ctx", window)])
    tokens = tokens.to(device)
    with torch.no_grad():
        rep = esm_model(tokens, repr_layers=[12])["representations"][12]
    frag = rep[0, 1 + left: 1 + left + len(pep), :].cpu().numpy()
    return frag.astype(np.float32)



def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--peptides", default="GFPGTPGLPGVK",
                    help="Comma-separated peptides to score")
    ap.add_argument("--weights",
                    default=os.path.join(BASE, "weights",
                                         "fragscape.pt"))
    ap.add_argument("--cache",
                    default=os.path.join(BASE, "data",
                                         "local_context_embeddings.pkl"),
                    help="Optional pickle of pre-computed embeddings "
                         "{peptide: (L,480)}; not shipped with this release "
                         "(see data/README.md)")
    ap.add_argument("--protein", default=None,
                    help="Optional FASTA; recomputes embeddings not in the cache")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"device : {device}")
    print(f"weights: {args.weights}")

    model, max_len = load_model(args.weights, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model  : FragScape(max_len={max_len})  params={n_params:,}")

    pep_list = [p.strip().upper() for p in args.peptides.split(",") if p.strip()]

    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = pickle.load(open(args.cache, "rb"))
        print(f"cache  : {args.cache}  ({len(cache):,} peptides)")

    esm_model = bc = None
    protein = ""
    if args.protein:
        protein = read_fasta(args.protein)
        print(f"protein: {args.protein}  ({len(protein)} aa)")

    print()
    print(f"{'peptide':<16}{'len':>4}{'source':>16}{'prob':>10}   flags")
    print("-" * 62)
    for pep in pep_list:
        if pep in cache:
            emb = cache[pep]
            src = "cache(localctx)"
        elif protein:
            if esm_model is None:
                import esm
                esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                esm_model = esm_model.to(device).eval()
                bc = alphabet.get_batch_converter()
            try:
                emb = compute_context_embedding(pep, protein, esm_model,
                                                bc, device)
            except ValueError as exc:
                print(f"{pep:<16}{len(pep):>4}{'NOT FOUND':>16}{'-':>10}   {exc}")
                continue
            src = "esm2(+-20)"
        else:
            print(f"{pep:<16}{len(pep):>4}{'MISSING':>16}{'-':>10}   "
                  f"not in cache; pass --protein <FASTA>")
            continue

        length = emb.shape[0]
        x = torch.FloatTensor(emb).unsqueeze(0).to(device)
        m = torch.ones(1, length).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x, m)).item()
        flags = []
        if prob >= 0.7:
            flags.append(">=0.7")
        elif prob >= 0.6:
            flags.append(">=0.6")
        print(f"{pep:<16}{length:>4}{src:>16}{prob:>10.4f}   {' '.join(flags)}")


if __name__ == "__main__":
    main()

