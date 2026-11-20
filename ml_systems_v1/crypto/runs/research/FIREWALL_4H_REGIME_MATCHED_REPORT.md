# -k FALSIFIER (CORE): regime-matched random-entry firewall @4h vs the ER-gated p1 baseline

**Date:** 2026-06-05 (VERIFIED via `date`). **Verdict: REFUTED.** All numbers RWYB-reproduced; no commit/deploy.
**Repro:** `python runs/research/firewall_4h_regime_matched.py`

## What was tested
- **p1 baseline (minimal honest, pre-registered):** EMA8/21 long-only crossover, **ER≥0.40 hard gate**
  (Kaufman Efficiency Ratio, win=20, past-only), `signal_flip_or_filter` exit + max-hold 42 bars (<7d),
  **spot, leverage=1, taker 0.24% round-trip**, no funding. 4h chimera bars.
- **Null:** `strat.firewall` with **`regime_matched=True`** — random entries drawn ONLY from gate-ON
  (ER≥0.40) bars, matched trade count + holding-duration distribution + cost. Isolates **within-gate
  entry TIMING** from the gate's regime SELECTION.
- **Decision surface (task):** baseline **NET per-trade expectancy** vs the regime-matched null
  distribution (null mean/quantiles + baseline percentile rank). Same held-out split the apparatus uses
  everywhere (train_end 2024-05-15 / val 2025-03-15 / oos 2025-12-31 / unseen 2026-05-22).

## Power check FIRST (positive_control@4h) — null is trustworthy
A synthetic 4h series with a **planted within-gate timing edge** (front-loaded up-impulses; cross fires at
move start) run through the SAME pipeline: held-out real per-trade exp **+8.69%** vs null p95 **+5.63%**,
**percentile_rank = 1.0**, compound-firewall **beats_held=True**. The harness **HAS POWER at 4h** — it
detects a genuine within-gate timing edge. (`src/strat/selftest_all.py` independently PASS 4/4.)

## Result — BTC primary (decisive, held-out OOS+UNSEEN, n=48)
| quantity | value |
|---|---|
| baseline NET per-trade exp | **+0.1082%** |
| regime-matched null mean | +0.0681% |
| null p5 / p50 / p95 | −0.4079% / +0.0588% / **+0.5779%** |
| **baseline percentile rank** | **0.566** (≈ the null median; needs >0.95 to pass) |
| beats null p95 | **False** |
| compound-firewall verdict | **BETA-IN-DISGUISE / no timing edge** |

The baseline's positive held-out expectancy (+0.11%) is **the gate's regime selection, not timing** —
random entries among the same ER≥0.40 trending bars do just as well (null mean +0.07%, median +0.06%).

## Breadth — re-test of the 1d 0/69 failure mode @4h
**0/12** liquid assets beat the regime-matched per-trade null on held-out (BTC, ETH, SOL, BNB, XRP, ADA,
DOGE, AVAX, LINK, LTC, DOT, TRX). Most sit BELOW the null median (percentile ranks mostly <0.5; LINK 0.03,
ADA 0.04, LTC 0.06, DOT 0.07). compound_beats_held=False for all 12.

## Config robustness (BTC, pre-registered grid, not fishing)
9 configs {EMA8/21, SMA10/30, EMA5/13} × {ER 0.3, 0.4, 0.5}: **none pass**; max held-out percentile 0.56.
At ER≥0.5 the cross does *worse* than random-among-trending (real −0.18% vs null mean +0.29%).

## Conclusion
The 1d MA-cross failure mode (0/69 beat the firewall) **reproduces at 4h**: the ER-gated MA-cross entry has
**no within-gate timing edge** above a regime-matched random null. PASS criterion not met → **premise
REFUTED** at the new cadence. (Scope note: this is the ENTRY-TIMING firewall on the minimal p1 entry; a
better EXIT/move-capture policy e.g. ATR-trail is orthogonal and untested here — but it operates on a
random-quality entry stream and cannot manufacture entry-timing alpha.) An honest null is a valid outcome.
