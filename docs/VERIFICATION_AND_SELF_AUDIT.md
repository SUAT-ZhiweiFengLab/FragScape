# FragScape — Independent Verification & Self-Audit

This document records an **independent, read-only verification** of the FragScape
release. Its purpose is to make the verification methodology, the reproduction of
the released result, and the known limitations explicit and auditable.

No model weights, source files, or released documents were modified during this
verification. The released checkpoint remains byte-for-byte identical
(see §6).

---

## 1. Purpose and scope

We re-verified the single released, checkable result of this repository:

| peptide | weights | context | score |
|---|---|---|---|
| `GFPGTPGLPGVK` (COL2A1) | `weights/fragscape.pt` | +-20 aa | **0.6644** |

The verification covered three questions:

1. **Reproduction** — does the released artifact reproduce the stated score?
2. **Methodology** — was the evaluation protocol chosen so that the reported
   numbers are not artifacts of model/dataset selection?
3. **Corpus integrity** — are the evaluation splits free of the cross-split
   sequence overlap that inflates peptide-level metrics?

The verification was performed entirely offline against the released
checkpoint and a frozen protocol. It did not use the test split for any model
or hyperparameter decision.

---

## 2. Reproduced result

Running the supported entry point on the COL2A1 precursor reproduces the
released score:

```
$ python scripts/predict.py --peptides GFPGTPGLPGVK --protein data/human_col2a1.fsa
...
model  : FragScape(max_len=12)  params=4,806,497
peptide          len          source      prob   flags
--------------------------------------------------------------
GFPGTPGLPGVK      12      esm2(+-20)    0.6644   >=0.6
```

- The checkpoint loads strictly, with **no missing and no unexpected keys**.
- The reported probability, `0.664465`, is stable across runs.
- Recomputing the +-20 local-context embeddings from the precursor with ESM-2
  instead of using a pre-computed cache changes the vector by
  `max abs diff = 2.3e-06` (cosine similarity `0.999999999999`) and does **not**
  change the reported probability (`0.664465` either way).

These facts confirm that the released inference chain is reproducible from the
in-repo artifacts alone.

---

## 3. Verification methodology

To assess whether a reported metric reflects genuine generalization rather than
evaluation artifacts, we followed a frozen protocol:

- **Pre-registered read-out.** The decoding/selection rule is fixed before
  training and is not changed after seeing test results.
- **Validation-only selection.** Models and hyperparameters are selected using
  the **validation** split only. The **test** split is never used for any
  selection decision. This removes the most common source of optimistic bias.
- **Equal budget.** Competing configurations are given an equal tuning budget,
  so differences are not attributable to unequal search effort.
- **Multi-seed statistics.** Results are reported over a fixed set of random
  seeds; a single run is never treated as a result. Paired comparisons use the
  Wilcoxon signed-rank test.
- **Provenance by digest.** Every result used here is tied to the input file's
  MD5 digest, so any claimed number can be traced to the exact corpus and
  weights that produced it.
- **No post-hoc cherry-picking.** Read-outs and components are not selected
  after the fact to favor a preferred outcome.

All conclusions below follow from this protocol.

---

## 4. Corpus-integrity considerations

Peptide-level classification corpora assembled from overlapping precursor
regions are prone to a specific failure mode: the **same short subsequence
(k-mer core) can appear in both the training and the test split**. Because
peptide fragments of the same protein are highly redundant at the k-mer level,
a naive random split can place near-duplicate fragments on both sides of the
partition. A model can then score highly by **memorizing** shared k-mer cores
rather than by learning transferable cleavage signals.

We quantified this effect and confirmed it is severe for peptide-level
random splits:

- On a class-balanced peptide-level split, several models reach near-identical
  scores of about **0.998 from the first training epoch** — a signature of
  leakage rather than learning.
- We therefore adopted an **8-mer union-find leak-safe split**: peptides are
  grouped by shared 8-mer cores (union-find), and whole groups are assigned to
  train/validation/test. The resulting splits have **zero cross-split 8-mer
  overlap** (train-val, train-test, and val-test all = 0).

Under this leak-safe protocol, honest performance on the **same distribution**
(10–12-mer peptides) is approximately **AUC ≈ 0.84**, versus the ~0.998
observed on the leaky split. Under **cross-corpus transfer** (a genuinely
independent external set with a different distribution and labeling
convention), performance falls to approximately **AUC ≈ 0.5**, i.e. close to
chance.

These three numbers — ~0.998 (leaky), ~0.84 (same-distribution, leak-safe),
~0.5 (cross-corpus) — are consistent with one another and should be read
together; quoting only the first would substantially overstate generalization.

---

## 5. Limitations and scope of use

Users of this release should be aware of the following:

- **Mixed peptide lengths.** The evaluation corpus spans lengths 10, 11, and 12
  residues, not 12-mers exclusively. Reported metrics are for the mixed-length
  setting unless a length-restricted subset is explicitly specified.
- **Class-prior imbalance.** The leak-safe setup is not class-balanced across
  splits (the positive rate differs between train/validation/test). Absolute
  probabilities therefore depend on the prior; **ranking metrics (e.g. AUC)**
  are the appropriate comparison, and fixed probability thresholds should be
  interpreted with care.
- **Cross-corpus transfer is weak.** As noted in §4, transfer to independent
  external data is near chance. The released score is intended for the
  within-distribution setting and for relative prioritization, not as a
  cross-database absolute predictor.
- **Single-target score.** `0.6644` is the score for one target peptide under the
  balanced local-context configuration. It is a fragment-compatibility signal,
  not a validated cleavage probability, and it is not a generalization claim.

---

## 6. What was NOT changed

This verification was read-only with respect to released artifacts:

- The released weights are unchanged:
  `weights/fragscape.pt` MD5 = `3e35ccddfadd57ea7e64da2b7032cd56`
  (4,806,497 parameters, `max_len = 12`).
- The released score for `GFPGTPGLPGVK` remains **0.6644**.
- No released document or manuscript file was edited as part of this
  verification.

---

## 7. Reproduction commands

```bash
# 1. Reproduce the released target score
python scripts/predict.py --peptides GFPGTPGLPGVK --protein data/human_col2a1.fsa

# 2. Confirm the weights have not changed
md5sum weights/fragscape.pt
# expected: 3e35ccddfadd57ea7e64da2b7032cd56
```

---

## 8. Research-integrity commitments

For this release and the verification described here, we commit to:

1. Selecting models and thresholds using **validation data only**; never
   selecting on the test split.
2. Reporting metrics under a **leak-safe split** with verified zero cross-split
   k-mer overlap, and disclosing when a number comes from a leaky setting.
3. Reporting **multi-seed** results and paired significance tests rather than
   single-run maxima.
4. Tying every reported number to an input **digest** (MD5) and a reproducible
   command.
5. Reporting **negative and null findings**, and never adding training-observed
   sequences into the test set to inflate scores.
6. Keeping released weights and the associated verified score **frozen**; any
   change is made explicitly and disclosed rather than silently overwritten.
