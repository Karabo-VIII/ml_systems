# MA-DNA → P(oracle-entry): held-out classifier, **per cell** (fit TRAIN+VAL, score UNSEEN)

**Date:** 2026-06-06 · **Script:** `runs/research/ma_dna_oracle_entry_clf.py` (selftest PASS — past-only
falsifier + power + null-label rejection) · **Result JSON:** `ma_dna_oracle_entry_clf_result.json` ·
**Method:** oracle-decomposition step 2 (`docs/ORACLE_DECOMPOSITION_2026_06_06.md`), scored on **UNSEEN only**.

## What was run
- **Label:** oracle-ENTRY bars from the audited perfect-foresight high-capture DP (`oracle_high_capture`;
  entry=open[k], exit=high[j], hold<7d, non-overlap, net taker 0.0024). One row per bar; y=1 at an oracle entry.
- **DNA features (the task's named set, strictly past-only `close.shift(1)`):** **1/2/3-MA distance + slope +
  gap + cross + r(efficiency-ratio)** → `single`(dist20/slope20/dist50) + `two_ma`(gap_10_30/crossstate_10_30/
  gap_20_50) + `ribbon3`(order/compression/ribbon_dist) + `kama_er`(er10/er20/kama_dist). 12 features.
  Cross enters as **one regularized feature** (not a standalone MA-cross trigger — that family is REFUTED).
- **Model:** L2 logistic (C=0.5, balanced), StandardScaler **fit on TRAIN+VAL only**, predict_proba on **UNSEEN**.
- **Per cell** = per (asset, cadence). Grid: {BTC,ETH,SOL,BNB,AVAX,ADA,DOGE,LINK,XRP,PEPE} × {4h,1d,1h,15m};
  {BTC,ETH,PEPE} × {range,dib}. **46 cells, all evaluable.**
- **Metrics per cell:** `AUC` (DNA score vs oracle-entry label, UNSEEN) · `fwd_IC` (Spearman, DNA score vs
  realized forward open→open return over the oracle median hold) · `label_IC` (Spearman, DNA score vs label) ·
  a **shuffled-label null AUC** (20 seeds, permute FIT labels) so AUC has a calibrated baseline.

## Per-cadence result (mean over cells, UNSEEN held-out), ranked by AUC-lift over the shuffled null

| cadence | n cells | mean AUC | mean lift | beats null p95 | mean **fwd-IC** | mean label-IC |
|---|---|---|---|---|---|---|
| **range** (price-range event bars) | 3 | **0.706** | **+0.222** | 3/3 | **+0.002** | +0.252 |
| **15m** | 10 | 0.619 | +0.121 | 10/10 | +0.032 | +0.114 |
| **dib** (dollar-imbalance bars) | 3 | 0.559 | +0.061 | 2/3 | +0.031 | +0.061 |
| **1h** | 10 | 0.522 | +0.023 | 8/10 | −0.005 | +0.028 |
| **4h** | 10 | 0.511 | +0.013 | 4/10 | −0.035 | +0.016 |
| **1d** | 10 | 0.480 | −0.026 | 1/10 | −0.050 | −0.028 |

**Monotonic in bar granularity / event-density:** range ≫ 15m > dib > 1h > 4h > 1d. The finer / more
event-driven the bar, the more the MA-DNA classifies oracle entries. Coarse time bars (4h, 1d) carry
**essentially none** (4h lift +0.013, only 4/10 beat the null p95; 1d lift is *negative* — DNA fit on
2022–2025 anti-generalizes to the 2026 UNSEEN regime, on ~149 bars / ~33 entries, small-sample).

## The decisive caveat — AUC (classification) does NOT convert to forward skill
Even where AUC is highest, the **forward-IC is ≈ 0**:
- range: AUC **0.71** but fwd-IC **+0.002**; 15m: AUC 0.62 / fwd-IC +0.03; dib +0.03; 1h/4h ≈ 0 or negative.

The shuffled-null collapses to ~0.49 everywhere (apparatus sound — not a leak; the selftest also confirms a
null label gives AUC 0.50 and fails p95, and the past-only falsifier passes), so the high range/15m AUC is
**genuine structure** — but its nature is *"we are inside an up-move / cluster."* Event and fine bars pack
densely during big moves, so MA trend-state (positive distance/slope) coincides with oracle entries **by
construction**, which scores high AUC but carries no forward-return *timing* edge. Adding the double-smoothed
`ma_of_ma` group (the `ALL` variant in the JSON) raises AUC further (range 0.71→0.78, 15m 0.62→0.65) yet
leaves fwd-IC ~0 — so the disconnect is **robust to the feature set**, not an artefact of which MA features
are included.

## Thin leads (NOT results — flagged for a deeper, battery-level look, not as alpha)
- **dib** is the only event-bar type with above-null AUC **and** a small *positive* fwd-IC: BTC +0.054,
  PEPE +0.040 (ETH ~0). Consistent with the earlier dib lead. Fragile (2–3 assets), sign-sensitive.
- **SOL 1d** is the single coarse-bar cell that pops (AUC 0.578, fwd-IC +0.153, label-IC +0.113) — but 1d
  is net-negative as a cadence and this is one noisy cell on ~34 entries; treat as noise until reproduced.

## Bottom line
A past-only MA-DNA classifier predicts oracle entries **above the shuffled null only on fine / event bars**
(range AUC 0.71, 15m 0.62, dib 0.56), monotonically in bar fineness; coarse time bars (4h ≈ 0.51, 1d ≈ 0.48)
carry **almost no MA-DNA**. Crucially the AUC is **move-cluster detection, not tradeable forward timing** —
**fwd-IC ≈ 0 even where AUC is highest**, on data the model never saw. This reproduces the prior MA
refutations on UNSEEN-only, per-cell: the MA decomposition is a weak *regime/in-move* descriptor, **not** a
standalone oracle-entry timer.
