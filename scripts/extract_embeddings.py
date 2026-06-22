#!/usr/bin/env python3
"""Extract frozen ESM-2 (esm2_t12_35M_UR50D) per-token embeddings.

Each peptide is encoded into a ``(MAX_LEN, 480)`` matrix by extracting the
final-hidden-layer residue representations of the frozen ESM-2 encoder and
padding/truncating to ``MAX_LEN`` (default 9).

Usage
-----
    python scripts/extract_embeddings.py \
        --dataset data/dataset.csv \
        --output cache/pepcleaver_embeddings.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"


def build_mask(peps, max_len):
    lens = [min(len("".join(c for c in p if c in AA)), max_len) for p in peps]
    m = np.zeros((len(peps), max_len), dtype=np.float32)
    for i, l in enumerate(lens):
        m[i, :l] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser(description="Extract ESM-2 per-token embeddings")
    ap.add_argument("--dataset", required=True, help="CSV with columns seq,label,split")
    ap.add_argument("--output", required=True, help="Output .npz path")
    ap.add_argument("--max-len", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--repr-layer", type=int, default=12)
    args = ap.parse_args()

    import csv

    with open(args.dataset) as f:
        rows = list(csv.DictReader(f))
    peps = [r["seq"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    splits = [r["split"] for r in rows]
    print(f"Loaded {len(peps)} peptides from {args.dataset}")

    import esm

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.eval()
    bc = alphabet.get_batch_converter()
    emb_dim = 480

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    embs = []
    bs = args.batch_size
    for i in range(0, len(peps), bs):
        batch = ["".join(c for c in p if c in AA) for p in peps[i : i + bs]]
        _, _, tokens = bc([(str(j), p) for j, p in enumerate(batch)])
        with torch.no_grad():
            rep = model(tokens, repr_layers=[args.repr_layer])["representations"][args.repr_layer]
        for j in range(len(batch)):
            nt = min(len(batch[j]), rep.shape[1] - 2)
            tok = rep[j, 1 : nt + 1, :].numpy()
            if nt < args.max_len:
                pad = np.zeros((args.max_len - nt, emb_dim), dtype=np.float32)
                tok = np.vstack([tok, pad])
            else:
                tok = tok[: args.max_len, :]
            embs.append(tok)
        if i % 2000 == 0 and i > 0:
            print(f"  ESM: {i}/{len(peps)}", end="\r", flush=True)
    X = np.array(embs, dtype=np.float32)
    mask = build_mask(peps, args.max_len)
    np.savez(args.output, X=X, labels=labels, splits=np.array(splits), mask=mask)
    print(f"\nSaved embeddings: X={X.shape} mask={mask.shape} -> {args.output}")


if __name__ == "__main__":
    main()
