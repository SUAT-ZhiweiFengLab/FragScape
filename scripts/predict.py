#!/usr/bin/env python3
"""Predict the proteolytic cleavage landscape of a full-length protein.

Applies PepCleaver via Multi-Instance Learning (MIL): the precursor
protein is treated as a "bag" and all candidate peptide fragments tiled
across the sequence are treated as "instances". Each fragment receives an
independent proteolytic likelihood score; tracking these scores across
their original coordinates reconstructs the continuous cleavage
propensity landscape of the parent protein.

Usage
-----
    python scripts/predict.py \
        --protein data/donkey_col1a1.fsa \
        --weights weights/pepcleaver.pt \
        --output results/donkey_cleavage_map.csv \
        --threshold 0.7
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pepcleaver import PepCleaver

AA = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 9
EMB_DIM = 480


def read_fasta(path):
    seq = ""
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += "".join(c for c in line.strip() if c in AA)
    return seq


def build_mask(peps):
    lens = [min(len(p), MAX_LEN) for p in peps]
    m = np.zeros((len(peps), MAX_LEN), dtype=np.float32)
    for i, l in enumerate(lens):
        m[i, :l] = 1.0
    return m


def tile_fragments(seq, min_len=2, max_len=9, require_gp=True):
    """Tile candidate peptide fragments across a precursor sequence."""
    peps, positions = [], []
    for i in range(len(seq)):
        for l in range(min_len, min(max_len + 1, len(seq) - i + 1)):
            w = seq[i : i + l]
            if not require_gp or (w.count("G") + w.count("P") >= 1):
                peps.append(w)
                positions.append(i)
    return peps, positions


def extract_embeddings(peps, batch_converter, esm_model, repr_layer=12):
    embs = []
    bs = 64
    for i in range(0, len(peps), bs):
        batch = ["".join(c for c in p if c in AA) for p in peps[i : i + bs]]
        _, _, tokens = batch_converter([(str(j), p) for j, p in enumerate(batch)])
        with torch.no_grad():
            rep = esm_model(tokens, repr_layers=[repr_layer])["representations"][repr_layer]
        for j in range(len(batch)):
            nt = min(len(batch[j]), rep.shape[1] - 2)
            tok = rep[j, 1 : nt + 1, :].numpy()
            if nt < MAX_LEN:
                pad = np.zeros((MAX_LEN - nt, EMB_DIM), dtype=np.float32)
                tok = np.vstack([tok, pad])
            else:
                tok = tok[:MAX_LEN, :]
            embs.append(tok)
        if i % 5000 == 0 and i > 0:
            print(f"  ESM: {i}/{len(peps)}", end="\r", flush=True)
    return np.array(embs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Predict proteolytic cleavage landscape")
    ap.add_argument("--protein", required=True, help="Input protein FASTA")
    ap.add_argument("--weights", default="weights/pepcleaver.pt")
    ap.add_argument("--output", default="results/donkey_cleavage_map.csv")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=9)
    ap.add_argument("--no-gp-filter", action="store_true", help="Do not require G/P in fragments")
    ap.add_argument("--cache", default=None, help="Optional embedding cache .npy")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    seq = read_fasta(args.protein)
    print(f"Protein length: {len(seq)} residues")

    peps, positions = tile_fragments(
        seq, args.min_len, args.max_len, require_gp=not args.no_gp_filter
    )
    print(f"Candidate fragments: {len(peps)}")

    # Load ESM-2 and extract embeddings
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.eval()
    bc = alphabet.get_batch_converter()

    if args.cache and os.path.exists(args.cache):
        Xd = np.load(args.cache)
        if len(Xd) == len(peps):
            print(f"Using cached embeddings: {Xd.shape}")
        else:
            Xd = extract_embeddings(peps, bc, esm_model)
            np.save(args.cache, Xd)
    else:
        Xd = extract_embeddings(peps, bc, esm_model)
        if args.cache:
            np.save(args.cache, Xd)

    # Predict
    model = PepCleaver().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    mask = build_mask(peps)
    with torch.no_grad():
        prob = torch.sigmoid(
            model(torch.FloatTensor(Xd).to(device), torch.FloatTensor(mask).to(device))
        ).cpu().numpy()

    # Unique sequence statistics
    unique_probs = {}
    for p, pr in zip(peps, prob):
        if p not in unique_probs or pr > unique_probs[p]:
            unique_probs[p] = pr
    n_unique_pass = sum(1 for v in unique_probs.values() if v >= args.threshold)
    print(f"Mean probability: {np.mean(prob):.4f}")
    print(f">= {args.threshold}: {int((prob >= args.threshold).sum())}/{len(prob)} "
          f"({(prob >= args.threshold).mean()*100:.1f}%)")
    print(f"Unique sequences: {len(unique_probs)}, >= {args.threshold}: {n_unique_pass}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "pep", "prob"])
        for p, pos, pr in zip(peps, positions, prob):
            w.writerow([pos, p, f"{pr:.6f}"])
    print(f"Saved cleavage map -> {args.output}")


if __name__ == "__main__":
    main()
