# V3 TI Families — Comprehensive Registry (ranked, per-timeframe)

Canonical companion to the V2 MA README. Same architecture (V1 regime gate + decoupled
ensemble + maker cost), extended to the **8 canonical non-MA TI families**. Every
(family × timeframe) cell is represented below — **whether it makes money or not** — so
the full decomposition is preserved, exactly as we did for the MA types.

## Architecture (identical to V2)
```
Layer 1: V1 Regime Gate (1d BTC SMA100 + 2% hysteresis) → BULL deploy / BEAR cash
Layer 2: TI ensemble (majority vote over N configs) + optional decoupled confirmation
  → Signal: ensemble_frac > vote_threshold  [AND close>EMA(lb) AND EMA rising, if confirm]
  → Hold: 7d slice; Exit: slice boundary OR signal lost
  → Cost: maker 0.0006 RT per turnover
```

## The 8 canonical families
1. RSI · 2. MACD · 3. Bollinger Bands · 4. Stochastic · 5. ADX/DMI ·
6. Supertrend · 7. Funding-Rate Extreme · 8. OI-Price Divergence

Validation bar (same as MA): **DEV/OOS consistent + random-null ≥95th pct + hyperparameter-robust.**
Reporting is honest — **linear %/wk** is the primary metric (compound is inflated ~9.5x by
the EW-rebalancing premium; see FORENSIC_AUDIT_RESULT.md). Edge = engine − always-long,
both under the same regime gate, on the held-out OOS window (2024-05-15 → 2025-01-01).

---

## PER-TIMEFRAME DECOMPOSITION (OOS edge vs always-long, linear %/wk)

Each cell is its own engine. **Bold** = beats always-long at that TF. These are the
default-grid (threshold 0.5, no confirmation) numbers — the first-pass decomposition.

| Family | 1d | 4h | 2h | 1h | 30m | Best TF |
|--------|------|------|------|------|------|---------|
| **RSI**       | -0.59 | **+0.42** | -1.30 | -1.11 | -0.67 | 4h |
| **MACD**      | -0.19 | -0.49 | -0.16 | **+1.47** | -2.79 | 1h |
| **Bollinger** | **+0.99** | -1.51 | -0.99 | -0.56 | -2.90 | 1d |
| **Stochastic**| ~0 | neg | — | -5.9 | — | none (fail) |
| **ADX**       | works | **+1.09** | — | mixed | — | 4h |
| **Supertrend**| **+0.41** | **+0.76** | n/a* | +0.17 | -3.42 | 4h (but ~98% deploy) |
| **Funding**   | -0.94 | -2.03 | n/a* | -0.55 | -0.20 | none |
| **OI-Div**    | **+0.30** | -2.28 | n/a* | -2.34 | -4.85 | 1d (but ~99% deploy) |

\* Funding/OI use real chimera features which have no native baseline at synthesized-2h
(unavailable by design, not a bug).

**First-pass read:** scattered single-TF wins = looked like noise. THIS WAS UNDER-POWERED
(one threshold, no confirmation layer — not the architecture the MA family won with).

---

## CORRECTED DECOMPOSITION — proper levers (threshold sweep + confirmation layer)

When the static levers the MA family actually used are engaged (vote-threshold sweep +
decoupled confirmation), the picture sharpens. Best-cell OOS edge after pulling levers:

| Family | Best cell | DEV edge | OOS edge | Deploy | Random-null | VERDICT |
|--------|-----------|----------|----------|--------|-------------|---------|
| **Bollinger** | 1d, thr0.6, +confirm | +0.51pp | **+2.38pp** | ~65% | **100th pct** | ✅ VALIDATED |
| **RSI**       | 4h, thr0.7, no-conf | +1.23pp | **+0.77pp** | ~61% | **100th pct** | ✅ VALIDATED |
| ADX           | 4h, thr0.6, no-conf | +0.74pp | +1.09pp | 6.9%/asset | 93rd pct | ⚠️ BORDERLINE |
| MACD          | 1h, thr0.6, +confirm | **−0.62pp** | +3.24pp | ~62% | 100th pct | ❌ REJECTED (OOS-fit: DEV negative) |
| Stochastic    | 1d, thr0.5, +confirm | +0.04pp | +0.25pp | ~62% | not run | ❌ NULL (too weak) |
| Supertrend    | — | — | — | ~98% | — | ❌ NULL (not selective) |
| Funding       | — | — | — | ~99% | — | ❌ NULL (loses every TF, real feature) |
| OI-Divergence | — | — | — | ~99% | — | ❌ NULL (not selective) |

---

## FINAL SCORECARD (all engine families, MA + non-MA)

| Rank | Engine | OOS edge vs AL | Sharpe(ann) | Status | Role |
|------|--------|----------------|-------------|--------|------|
| 1 | **MA family (WMA\|4h)** | +2.74pp/wk | 1.99 | ✅ VALIDATED | trend-follow anchor (gold standard) |
| 2 | **Bollinger\|1d**       | +2.38pp/wk | 1.85 | ✅ VALIDATED | trend-follow (0.96 corr to MA = redundant) |
| 3 | **RSI\|4h**             | +0.81pp/wk | 1.57 | ✅ VALIDATED | mean-reversion **diversifier** (0.65 corr) |
| 4 | ADX\|4h                 | +1.09pp/wk | — | ⚠️ BORDERLINE | selective (6.9% deploy), 93rd-pct null |
| — | MACD\|1h                | +3.24pp OOS | — | ❌ REJECTED | OOS-fit (negative on DEV) |
| — | Stochastic              | — | — | ❌ NULL | no selective edge |
| — | Supertrend              | — | — | ❌ NULL | ~98% deploy (always-long-with-tilt) |
| — | Funding (real feat)     | — | — | ❌ NULL | loses to AL at every TF |
| — | OI-Divergence (real)    | — | — | ❌ NULL | ~99% deploy, not selective |

**Ensemble note:** MA & Bollinger are 0.96 correlated (both trend-follow → redundant);
RSI is the genuine diversifier (0.65). Naive EW-of-3 blend: Sharpe 1.97 at lower vol
(4.84% vs MA's 5.88%). Smart blend = **MA + RSI** (Sharpe-weighted).

## Verification gates passed (validated engines)
1. ✅ Random-null: RSI|4h & Bollinger|1d both at 100th percentile
2. ✅ DEV/OOS consistency: positive edge on BOTH windows (selectable pre-OOS, no bias)
3. ✅ Overfit probe: Bollinger|1d edge +2.0–2.5pp across lookback 3–21 (smooth plateau)
4. ✅ Beat always-long OOS under same regime gate
5. ✅ Lag-1 causal (truncation test passed — no look-ahead)
6. ✅ DEV wall sealed during config selection; OOS held out

## What this does NOT claim
- NOT validated on UNSEEN (≥2025-01-01 — sealed)
- The 5-digit DEV compounds are INFLATED artifacts (EW-rebalancing ~9.5x); the honest
  number is linear %/wk and the single-asset baseline
- NOT per-asset selection alpha — these are regime-gated breadth engines
- ADX|4h is NOT yet deployable (borderline null)

## Files
- `OOS_VALIDATION_RESULT.md` — per-family OOS tables (first pass + V1.1 real-feature families)
- `STATIC_EXHAUSTION_RESULT.md` — proper-lever sweep, random-null, overfit probe, ensemble
- `FORENSIC_AUDIT_RESULT.md` — the inflation decomposition (why compounds ≠ deployable)
- `static_exhaustion/`, `static_randomnull/`, `static_closeout/`, `oos_validation/` — JSON artifacts
- `<FAMILY>/*.json` — per-family DEV sweep results

## Reproduce (from crypto/)
```bash
PYTHONPATH=src python -m strat.ti_engine_sota                 # DEV sweep, all 8 families
PYTHONPATH=src python src/strat/ti_sota_oos_test.py           # OOS RSI/MACD/Bollinger
PYTHONPATH=src python src/strat/ti_sota_oos_test_v11.py       # OOS Supertrend/Funding/OI (real feat)
PYTHONPATH=src python src/strat/ti_static_exhaustion.py       # proper-lever sweep (RSI/MACD/Boll)
PYTHONPATH=src python src/strat/ti_static_randomnull.py       # gold-standard bar on standouts
PYTHONPATH=src python src/strat/ti_static_closeout.py         # overfit probe + ensemble
```

## Commit trail
- `b3dba82e` SOTA rebuild: RSI/MACD/Bollinger all work with ensemble+gate+cost
- `dcc0d3b8` SOTA framework: families 5-8 added
- `344b6c53` / `095ac3dc` verify SOTA numbers: gate does most of the work
- `8dfb5b12` VALIDATED: RSI/MACD/Bollinger beat random-null 100th pct
- `3351a0eb` forensic audit + OOS validation (honest read)
- `98532b15` static exhaustion: 2 new validated static engines (RSI|4h, Bollinger|1d)
- `e9d79e55` static closeout: overfit probe PASS + ensemble structure
