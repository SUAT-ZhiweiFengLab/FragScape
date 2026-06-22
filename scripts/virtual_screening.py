#!/usr/bin/env python3
"""Hierarchical virtual screening pipeline for COL1A1-derived peptides.

Refines a PepCleaver-derived peptide library through:
  1. Sequence deduplication
  2. Cleavage-score filtering (threshold 0.7)
  3. Physicochemical hard filtering (300 <= MW <= 1000 Da, instability < 40)
  4. Weighted composite physicochemical scoring
  5. Glycine-starting adjustment (+0.5)
  6. Length-stratified selection (3-9 aa, quotas)

This yields the final COL1A1-derived candidate peptide library for
downstream molecular docking.

Usage
-----
    python scripts/virtual_screening.py \
        --cleavage-map results/donkey_cleavage_map.csv \
        --output results/pepcleaver_350_library.csv
"""
import argparse
import csv
import os
from collections import defaultdict

from Bio.SeqUtils.ProtParam import ProteinAnalysis


def compute_physchem(seq):
    try:
        pa = ProteinAnalysis(seq)
        return {
            "mw": pa.molecular_weight(),
            "ii": pa.instability_index(),
            "pi": pa.isoelectric_point(),
            "charge": pa.charge_at_pH(7.0),
            "gravy": pa.gravy(),
        }
    except Exception:
        return None


def score_subscore(val, low, mid, high, reverse=False):
    """Three-level scoring: 10 / 7 / 3."""
    if reverse:
        if val <= high:
            return 10
        elif val <= mid:
            return 7
        else:
            return 3
    else:
        if low <= val <= high:
            return 10
        elif val < low * 0.8 or val > high * 1.2:
            return 3
        else:
            return 7


def main():
    ap = argparse.ArgumentParser(description="Virtual screening -> 350 peptide library")
    ap.add_argument("--cleavage-map", required=True, help="Output of predict.py")
    ap.add_argument("--output", default="results/pepcleaver_350_library.csv")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--mw-low", type=float, default=300.0)
    ap.add_argument("--mw-high", type=float, default=1000.0)
    ap.add_argument("--ii-max", type=float, default=40.0)
    args = ap.parse_args()

    # Load cleavage map, keep best prob per unique sequence
    unique_probs = {}
    with open(args.cleavage_map) as f:
        for row in csv.DictReader(f):
            pep = row["pep"]
            prob = float(row["prob"])
            if pep not in unique_probs or prob > unique_probs[pep]:
                unique_probs[pep] = prob
    print(f"Loaded {len(unique_probs)} unique fragments")

    # 1+2. Dedup + threshold filtering
    candidates = [(p, pr) for p, pr in unique_probs.items() if pr >= args.threshold]
    print(f"After dedup + >= {args.threshold}: {len(candidates)}")

    # 3. Physicochemical hard filtering
    filtered = []
    for seq, prob in candidates:
        pc = compute_physchem(seq)
        if pc is None:
            continue
        if not (args.mw_low <= pc["mw"] <= args.mw_high):
            continue
        if pc["ii"] >= args.ii_max:
            continue
        filtered.append((seq, prob, pc))
    print(f"After physicochemical hard filter: {len(filtered)}")

    # 4. Weighted composite scoring
    # Total = 0.35*MW + 0.30*II + 0.10*pI + 0.15*charge + 0.10*GRAVY
    scored = []
    for seq, prob, pc in filtered:
        s_mw = score_subscore(pc["mw"], 500, 700, 900)
        s_ii = score_subscore(pc["ii"], 0, 20, 40, reverse=True)
        s_pi = score_subscore(pc["pi"], 5, 7, 9)
        s_charge = score_subscore(abs(pc["charge"]), 0, 1, 2)
        s_gravy = score_subscore(pc["gravy"], -1, 0, 1)
        total = 0.35 * s_mw + 0.30 * s_ii + 0.10 * s_pi + 0.15 * s_charge + 0.10 * s_gravy
        adj = total + (0.5 if seq.startswith("G") else 0.0)
        scored.append((seq, prob, pc, total, adj))

    # 6. Length-stratified selection (quotas: 19,50,50,50,50,65,66 for len 3-9)
    quotas = {3: 19, 4: 50, 5: 50, 6: 50, 7: 50, 8: 65, 9: 66}
    by_len = defaultdict(list)
    for item in scored:
        l = len(item[0])
        if l in quotas:
            by_len[l].append(item)

    final_library = []
    for l in range(3, 10):
        items = sorted(by_len[l], key=lambda x: -x[4])  # adj_score descending
        selected = items[: quotas[l]]
        final_library.extend(selected)
        print(f"  len={l}: {len(items)} candidates -> selected {len(selected)} (quota {quotas[l]})")

    print(f"\nFinal peptide library: {len(final_library)} sequences")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "length", "cleavage_prob", "MW", "II", "pI", "charge", "GRAVY",
                     "total_score", "adj_score"])
        for seq, prob, pc, total, adj in final_library:
            w.writerow([seq, len(seq), f"{prob:.4f}", f"{pc['mw']:.2f}", f"{pc['ii']:.2f}",
                        f"{pc['pi']:.2f}", f"{pc['charge']:.2f}", f"{pc['gravy']:.3f}",
                        f"{total:.3f}", f"{adj:.3f}"])
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
