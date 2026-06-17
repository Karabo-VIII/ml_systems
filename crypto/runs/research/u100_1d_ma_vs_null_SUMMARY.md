# RWYB — u100 1d: best fixed-config MA crossover vs cost-matched random-entry null

**Date:** 2026-06-05 · **No commit** (read-only analysis) · LONG-ONLY SPOT lev=1
**Reproduce:** `python runs/research/u100_1d_ma_vs_null_rwyb.py` (full output: `u100_1d_ma_vs_null_OUT.txt`)

## Setup (honest, same split + costs for all three legs)
- **Data:** 77 u100 assets with 1d chimera (all 77 load; ≥60 bars each). Via `ChimeraLoader`.
- **Engine:** `wealth_bot.harness.CanonicalHarness` (past-only by API; next-bar-open fills; Pattern S/T/U safe).
- **Cost:** taker round-trip **0.24%** (`TAKER_COST_RT`). Funding OFF (spot). Exit = `signal_flip`. No filter. No max-hold.
- **Windows:** TRAIN<2024-05-15 · VAL<2025-03-15 · OOS<2025-12-31 · **UNSEEN ≤2026-05-22** (the headline held-out).
- **Config selection (no leak):** best of a 10-config MA grid (SMA/EMA pairs) chosen by **IN-SAMPLE (TRAIN+VAL)** equal-weight basket compound. UNSEEN touched once, for reporting only.
- **Baseline (a):** `strat.firewall.random_entry_null` — per asset, same window, same cost, count- and duration-matched random entries.
- **Apparatus validated:** `src/strat/selftest_all.py` → 4/4 PASS incl. **positive control** (`has_power=True, beats_held=True`) — the null is a calibrated comparator (accepts real edges, rejects ghosts), not a reject-everything sieve.

## Selected config
**SMA 10/30** — in-sample (TRAIN+VAL) equal-weight basket = **+2660%** (top of grid; EMA12/26 +1570%, SMA10/50 +1916% next).

## Held-out result (UNSEEN)
| metric | **Candidate MA (SMA10/30)** | **Random-entry null** | Candidate MA (OOS) |
|---|---|---|---|
| n trades (pooled) | 255 | 10,200 (255×40 books) | 357 |
| **per-trade mean (after cost)** | **−2.60% (−259.6 bps)** | **−0.50% (−49.8 bps)** | −0.58% (−58.0 bps) |
| per-trade median | −6.64% | −2.08% | −6.45% |
| per-trade std | 24.4% | 23.5% | 51.8% |
| win rate | 0.231 | 0.417 | 0.283 |
| p90 / max | +10.3% / +242.7% | +18.5% / +306.3% | +18.6% / +902.5% |

**Aggregate compound (UNSEEN):** equal-weight basket **−8.26%** · mean per-asset −8.26% · median −20.08% · **15/77 assets positive**.
**Firewall:** **0 / 77 assets beat their own random-entry null p95.** Mean real UNSEEN compound −8.26% vs mean null p50 −8.46% (statistically indistinguishable). Candidate per-trade mean is **210 bps WORSE** than the null's.

## Reconciliation note
Candidate per-trade mean (−2.60%) is worse than the null's (−0.50%), yet **compound is ~equal** (−8.26% vs −8.46%): the null books carry the same per-asset trade-count + duration and the same ~23% per-trade vol, so **volatility drag** (Jensen, ≈ −½σ² per multiplicative trade) pulls both legs to ≈ −8% regardless of the small mean gap. Numbers are internally consistent (≈3.3 trades/asset × −2.6%/trade ⇒ ≈ −8.4% per asset).

## Verdict
**REFUTED.** A naive fixed-config MA crossover has **no held-out timing edge on u100 1d** — it is **beta-in-disguise and, per-trade, actively worse than random** at taker cost. The +2660%→−8% in-sample→held-out collapse is the overfit signature. Both held-out windows (OOS and UNSEEN) are negative, so it is not a single-window artifact.

## Scope / caveats
- Tests ONE family: vanilla MA crossover, signal_flip exit, **no conditioning gate**. Does NOT refute gated/regime-conditioned avenues (whale-gate, liquidation-cascade, etc.).
- UNSEEN is a single ~5-month regime (Dec-2025→May-2026, choppy/down tape); per-asset samples are small (1–6 UNSEEN trades), but the aggregate 0/77 is decisive and OOS agrees.
- Consistent with project memory: "no verified active alpha at daily/4h" + "naive-harvest = coin-flip (need a signal)." This is a fresh RWYB confirmation for the MA-crossover family.
