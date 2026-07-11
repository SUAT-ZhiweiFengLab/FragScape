#!/usr/bin/env python3
"""Train PepCleaver on the MEROPS-derived proteolytic cleavage dataset.

Implements the training protocol from the manuscript:
  * AdamW optimizer (lr=5e-4, weight_decay=1e-4)
  * Cosine annealing learning-rate schedule over 25 epochs
  * Focal loss (gamma=2.0, alpha=0.75)
  * Mini-batch size 64
  * Model selection by best validation AUC

Usage
-----
    python scripts/train.py \
        --embeddings cache/pepcleaver_embeddings.npz \
        --output weights/pepcleaver.pt
"""
import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Allow running both as `python scripts/train.py` and from package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pepcleaver import PepCleaver
from model.components import FocalLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser(description="Train PepCleaver")
    ap.add_argument("--embeddings", required=True, help="Embeddings .npz from extract_embeddings.py")
    ap.add_argument("--output", default="weights/pepcleaver.pt", help="Output model weights")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load embeddings
    data = np.load(args.embeddings)
    X, y, splits, mask = data["X"], data["labels"], data["splits"], data["mask"]
    splits = np.array([str(s) for s in splits], dtype=str)
    tr = splits == "train"; va = splits == "val"; te = splits == "test"
    print(f"Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    X_tr = torch.FloatTensor(X[tr]).to(device)
    y_tr = torch.FloatTensor(y[tr]).to(device)
    m_tr = torch.FloatTensor(mask[tr]).to(device)
    X_va = torch.FloatTensor(X[va]).to(device)
    y_va = torch.FloatTensor(y[va]).to(device)
    m_va = torch.FloatTensor(mask[va]).to(device)
    X_te = torch.FloatTensor(X[te]).to(device)
    y_te = torch.FloatTensor(y[te]).to(device)
    m_te = torch.FloatTensor(mask[te]).to(device)

    model = PepCleaver().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = FocalLoss(gamma=2.0, alpha=0.75)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(TensorDataset(X_tr, y_tr, m_tr), batch_size=args.batch_size, shuffle=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    best_val, best_test, best_ep = 0.0, 0.0, 0
    for ep in range(args.epochs):
        model.train()
        for bx, by, bm in loader:
            opt.zero_grad()
            crit(model(bx, bm), by).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            tr_p = torch.sigmoid(model(X_tr, m_tr)).cpu().numpy()
            va_p = torch.sigmoid(model(X_va, m_va)).cpu().numpy()
            te_p = torch.sigmoid(model(X_te, m_te)).cpu().numpy()
        print(
            f"  Ep{ep+1:2d}: trainAUC={roc_auc_score(y[tr], tr_p):.4f} "
            f"valAUC={roc_auc_score(y[va], va_p):.4f} testAUC={roc_auc_score(y[te], te_p):.4f}"
        )
        sys.stdout.flush()
        va_auc = roc_auc_score(y[va], va_p)
        if va_auc > best_val:
            best_val = va_auc
            best_test = roc_auc_score(y[te], te_p)
            best_ep = ep + 1
            torch.save(model.state_dict(), args.output)

    print(f"\nBest: valAUC={best_val:.4f} testAUC={best_test:.4f} (Ep{best_ep})")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
