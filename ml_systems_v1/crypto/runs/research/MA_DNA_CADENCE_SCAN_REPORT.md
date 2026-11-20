# MA-DNA Cadence Scan — which bar-type/timeframe carries the strongest MA-DNA for oracle entries?

**Date:** 2026-06-06 · **Script:** `runs/research/ma_dna_cadence_scan.py` (selftest PASS; past-only look-ahead
falsifier PASS) · **Result JSON:** `ma_dna_cadence_scan_result.json` · **Method:** oracle-decomposition step 2.

## What was run
- **Labels:** oracle-ENTRY bars from the audited perfect-foresight high-capture DP (`oracle_ceiling_builder.oracle_high_capture`; entry=open[k], exit=high[j], hold<7d, non-overlap, net 0.0024).
- **Features:** MA-decomposition basis ONLY, strictly past-only (`close.shift(1)`), 5 groups + the union:
  `single` (dist/slope) · `two_ma` (gap + cross-**state**) · `ribbon3` (order/compression) · `ma_of_ma` (double-smoothed) · `kama_er` (efficiency-ratio/KAMA) · `ALL`.
  Cross-STATE enters only as one regularized feature — **not** an MA-cross trigger (that family is already REFUTED).
- **Model:** L2 logistic (C=0.5, balanced), fit on TRAIN+VAL, evaluated held-out on OOS+UNSEEN.
- **Control:** shuffled-label (permute FIT labels, refit) × 30 seeds → null AUC dist. Rank key = **held-out AUC − shuffled-mean AUC**.
- **Grid:** assets {BTC, SOL, ETH} × cadences/bar-types {15m, 1h, 4h, 1d, range, dib}. (SOL absent for range/dib; dollar skipped — 2.7–4.1 M bars, granularity-degenerate.)

## Ranking — bar-type/timeframe by mean held-out MA-DNA AUC-lift (ALL group)

| rank | bar-type | mean AUC | mean lift | beats shuffled p95 | mean fwd-IC |
|---|---|---|---|---|---|
| 1 | **range** (price-range event bars) | **0.783** | **+0.300** | 2/2 | **−0.014** |
| 2 | **15m** | 0.630 | +0.131 | 3/3 | +0.014 |
| 3 | **dib** (dollar-imbalance bars) | 0.569 | +0.069 | 2/2 | +0.020 |
| 4 | **1h** | 0.540 | +0.044 | 3/3 | −0.013 |
| 5 | **1d** | 0.508 | +0.009 | 0–1/3 | +0.023 |
| 6 | **4h** | 0.514 | +0.015 | 0/3 | −0.006 |

**Monotonic in bar granularity/event-density:** range ≫ 15m > dib > 1h > 1d ≈ 4h.
The finer / more event-driven the bar, the more MA-structure ↔ oracle-entry classification information.
Coarse time bars (4h, 1d) carry **essentially none** (lift ≈ +0.01, mostly inside the shuffled p95).

**Within the MA decomposition:** `ma_of_ma` (double-smoothed trend) is the consistent leader (best-lift on
15m ×3, range ×2, dib ×1); `two_ma` (gap + cross-state) is the consistent **weakest** group everywhere —
i.e. the gap/cross family that was previously refuted as a trigger also carries the least DNA; smoothing/
trend-persistence carries more.

## The decisive caveat — classification lift ≠ tradeable skill
The AUC-lift is **oracle-entry CLASSIFICATION** lift, and it does **not** convert to forward-return skill on
the very cadences where AUC looks best:
- range: AUC **0.78** but forward-IC **≈ −0.01**; 15m: AUC 0.63 but fwd-IC **≈ +0.01**; 1h/4h: fwd-IC ≈ 0.

The shuffled control collapses to ~0.48–0.49 everywhere (apparatus sound — not a leak), and features are
proven past-only. So the high range/15m AUC is **genuine structure**, but its nature is *"we are inside an
up-move/cluster"* — event bars pack densely during big moves, so MA trend-state coincides with oracle entries
**by construction** — not *timing* with forward edge after cost. This is the same AUC↔capture disconnect the
SOL-4h full-feature falsifier already showed (AUC 0.64 / capture-skill ≈ 0), now shown to be **general and
strongest exactly where AUC is highest (event bars)**.

## One thin lead (NOT a result — needs a deeper, multi-asset look)
**dib (dollar-imbalance) bars** are the only bar-type where MA-DNA shows above-shuffled classification lift AND
a non-trivial **positive forward IC**: `ma_of_ma` fwd-IC = **+0.187 (BTC)**, **+0.171 (ETH)**; `single` similar.
Caveat: only 2 assets (no SOL), and the ALL-group fwd-IC washes to ≈ +0.02 / −0.01, so the signal is
fragile/sign-sensitive. Flag as a candidate to probe with the full firewall battery, not as alpha.

## Bottom line
Strongest MA-DNA *classification* signal = **range bars, then 15m, then dib**; coarsest time bars (4h, 1d)
carry almost none. But the lift is regime/move-cluster detection, not tradeable timing (fwd-IC ≈ 0 on the
high-AUC cadences) — consistent with the prior MA refutations. Only **dib + ma_of_ma** hints at forward edge
(BTC/ETH only) and is the single follow-up worth a deeper look.
