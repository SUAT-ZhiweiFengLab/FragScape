#!/usr/bin/env python3
"""Evaluate a trained PepCleaver model on the held-out test set.

Reports AUC, F1, accuracy and the decision-threshold (0.7) statistics
used for downstream peptide library generation.

Usage
-----
    python scripts/evaluate.py \
        --embeddings cache/pepcleaver_embeddings.npz \
        --weights weights/pepcleaver.pt \
        --output results/test_predictions.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pepcleaver import PepCleaver

AA = "ACDEFGHIKLMNPQRSTVWY"


def main():
    ap = argparse.ArgumentParser(description="Evaluate PepCleaver")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--weights", default="weights/pepcleaver.pt")
    ap.add_argument("--output", default="results/test_predictions.csv")
    ap.add_argument("--dataset", default=None,
                    help="Path to dataset.csv (for writing seq column in output). "
                         "Auto-resolved if not provided.")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    data = np.load(args.embeddings)
    X, y, splits, mask = data["X"], data["labels"], data["splits"], data["mask"]
    splits = np.array([str(s) for s in splits], dtype=str)
    te = splits == "test"

    model = PepCleaver().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(
            model(
                torch.FloatTensor(X[te]).to(device),
                torch.FloatTensor(mask[te]).to(device),
            )
        ).cpu().numpy()

    y_te = y[te]
    auc = roc_auc_score(y_te, prob)
    pred = (prob >= args.threshold).astype(int)
    f1 = f1_score(y_te, pred)
    acc = accuracy_score(y_te, pred)
    n_pass = pred.sum()
    print(f"Test AUC: {auc:.4f}")
    print(f"Threshold {args.threshold}: F1={f1:.4f} Acc={acc:.4f} pass-rate={n_pass}/{len(y_te)}={n_pass/len(y_te)*100:.1f}%")

    # PR-AUC (average precision)
    pr_auc = average_precision_score(y_te, prob)
    print(f"PR-AUC (AP): {pr_auc:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "label", "prob", "split"])
        # Re-read sequences to write alongside predictions
        ds = args.dataset
        if ds is None:
            npz_dir = os.path.dirname(os.path.abspath(args.embeddings))
            ds = os.path.join(os.path.dirname(npz_dir), "data", "dataset.csv")
        seqs = None
        if os.path.exists(ds):
            with open(ds) as f2:
                rows = list(csv.DictReader(f2))
            test_rows = [r for r in rows if r["split"] == "test"]
            if len(test_rows) == len(y_te):
                seqs = [r["seq"] for r in test_rows]
        if seqs is None:
            seqs = [""] * len(y_te)
        for s, l, p in zip(seqs, y_te, prob):
            w.writerow([s, int(l), f"{p:.6f}", "test"])
    print(f"Saved predictions -> {args.output}")


if __name__ == "__main__":
    main()
