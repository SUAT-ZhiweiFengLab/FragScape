#!/usr/bin/env python3
"""Quick-start example: load PepCleaver and score candidate peptides.

Demonstrates the full inference pipeline:
  1. Load the pretrained PepCleaver model
  2. Encode peptides with frozen ESM-2 (esm2_t12_35M_UR50D)
  3. Predict proteolytic likelihood scores

Usage
-----
    python example/inference_example.py
"""
import os
import sys

# Allow running from repo root or example/ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from model.pepcleaver import PepCleaver

MAX_LEN = 9
EMB_DIM = 480
AA = "ACDEFGHIKLMNPQRSTVWY"


def build_mask(peps):
    lens = [min(len(p), MAX_LEN) for p in peps]
    m = np.zeros((len(peps), MAX_LEN), dtype=np.float32)
    for i, l in enumerate(lens):
        m[i, :l] = 1.0
    return m


def encode_peptides(peps, batch_converter, esm_model, repr_layer=12):
    """Encode peptides into (N, MAX_LEN, EMB_DIM) ESM-2 per-token embeddings."""
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
    return np.array(embs, dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Candidate peptides (manuscript Table 1: P1-P5 + reference peptide)
    candidates = [
        ("P1", "GAQGPPGP"),
        ("P2", "GFAGPPGAD"),
        ("P3", "GPAGAPGTP"),
        ("P4", "GPPGAVGP"),
        ("P5", "GAPGIAGAP"),
        ("Ref", "GAAGLPGPK"),
    ]
    peps = [s for _, s in candidates]
    names = [n for n, _ in candidates]

    # 1. Encode with frozen ESM-2
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.eval()
    bc = alphabet.get_batch_converter()
    X = encode_peptides(peps, bc, esm_model)
    mask = build_mask(peps)

    # 2. Load pretrained PepCleaver
    weights_path = os.path.join(ROOT, "weights", "pepcleaver.pt")
    model = PepCleaver().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Predict proteolytic likelihood
    with torch.no_grad():
        prob = torch.sigmoid(
            model(torch.FloatTensor(X).to(device), torch.FloatTensor(mask).to(device))
        ).cpu().numpy()

    # 4. Report
    print(f"\n{'ID':<5} {'Sequence':<12} {'Cleavage prob':>14}  {'>= 0.7':>7}")
    print("-" * 42)
    for name, seq, p in zip(names, peps, prob):
        flag = "PASS" if p >= 0.7 else ""
        print(f"{name:<5} {seq:<12} {p:>14.4f}  {flag:>7}")


if __name__ == "__main__":
    main()
