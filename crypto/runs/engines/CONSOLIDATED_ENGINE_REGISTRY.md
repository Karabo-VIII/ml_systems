# CONSOLIDATED ENGINE REGISTRY — all trading engines/agents (2026-06-24)

> **2026-06-28 leak-fix (regime.shift(1)) — RE-BASELINE.** A same-day 1d-BTC-SMA100 regime
> look-ahead (the V1.1 align_htf_on bug class) was found + falsified in the MA/TI gating layer:
> consumers broadcast the day-D 1d-CLOSE regime onto intraday bars of the SAME day D (regime not
> known until ~23:59). Fixed DRY via a shared `causal_regime(wall_ms) = btc_regime.shift(1)`
> helper (a bar on day D now reads the last CLOSED 1d bar = regime[D-1]). Impact is MODEST
> (~1.7% of OOS bars flip; edge retains ~85%, NOT the V1.1 catastrophe). The capped DEPLOYABLE
> headline (published in CURRENT.md, NOT in this file) re-baselines ~9-15% lower: 1d cap15
> +1.71→**+1.564**/wk, and the cap dial becomes 15=+1.56 / 33=+1.82 / 50=+2.19 (window-luck) /
> EW=+3.47 (window-luck). The OOS edge/wk figures in the tables below are vs-always-long edges
> from the static TI gauntlet and are NOT the capped-book headline; they are not re-baselined
> here. See `runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json` +
> `runs/engines/AUDIT_ORACLE_UPGRADE/wave2_concentration_frontier.json`.

The single ranked view of every trading engine built across all campaigns. Static
phase is now CLOSED and exhausted across **21 TI families**. Honest accounting per the
forensic audit: DEV compounds are inflated ~9.5x by the EW-rebalancing premium; the
**linear %/wk OOS edge vs always-long + random-null percentile** are the real read.

- DEV window: 2020-01-01 .. 2024-05-15  |  OOS (held-out): 2024-05-15 .. 2025-01-01
- Constraints: LONG-ONLY, spot, lev=1, maker 0.0006 RT
- Architecture (all engines): config ENSEMBLE (majority vote) + V1 BTC-SMA100 regime
  GATE + maker COST + decoupled lookback/slice
- VALIDATED bar = DEV/OOS-consistent AND random-null ≥95th pct AND SELECTIVE (deploy<70%)

---

## TIER 1 — VALIDATED & DEPLOYABLE (selective, real timing edge)

| Rank | Engine | TF | OOS edge/wk | DEV edge/wk | Deploy | Null pct | Role |
|------|--------|----|-----------:|-----------:|-------:|---------|------|
| 1 | **MA family (WMA)** | 4h | **+2.74pp** | +4.28% lin | ~35% | 100th | Core wealth engine. Gold standard. |
| 2 | **Bollinger** | 1d | **+2.38pp** | +0.51pp | selective | 100th | Trend-follow (0.96 corr w/ MA — redundant) |
| 3 | **RSI** | 4h | **+0.81pp** | +1.23pp | selective | 100th | The genuine DIVERSIFIER (0.65 corr) |

MA family detail (8 MA types, decoupled lookback): WMA leads; EMA/DEMA/KAMA/VIDYA/SMA
also pass OOS (6/8). TEMA fails, HMA marginal. 4h is the clean sweet spot across all.
Combined V1+V2 (regime gate + 4h trend timing): MaxDD ~-38 to -48%, bear-2022 ≈ -2 to -6%.

Ensemble note: MA + Bollinger are 0.96 correlated (both trend-follow, near-redundant).
**The smart deployable blend is MA + RSI** (RSI is the only real diversifier). Naive
EW/3 ensemble: Sharpe 1.97 at lower vol; MA-alone Sharpe 1.99. Sharpe-weighted MA+RSI
is the open optional refinement.

---

## TIER 2 — GENUINE EDGE but NON-SELECTIVE (long-tilt overlays, not standalone engines)

Real trend-timing edge (DEV/OOS-consistent, random-null 100th pct) but deploy 75-93% —
they refine WHICH longs, not WHETHER to be long. Usable as WEIGHTING overlays on the
long book, NOT as discrete gate engines. (V4 campaign.)

| Engine | TF | OOS edge/wk | DEV edge/wk | Deploy | Null pct |
|--------|----|-----------:|-----------:|-------:|---------|
| CCI | 1d | +2.93pp | +0.96pp | 89% | 100th |
| WILLR | 1d | +2.67pp | +0.79pp | 90% | 100th |
| VWAP | 1d | +2.61pp | +0.84pp | 81% | 100th |
| Keltner | 1d | +2.19pp | +0.60pp | 89% | 76th |
| Donchian | 4h | +2.07pp | +1.22pp | 91% | 100th |
| PSAR | 1d | +1.10pp | +1.04pp | 90% | 100th |
| MFI | 1d | +0.99pp | +0.99pp | 88% | 100th |
| OBV | 1d | +0.48pp | +0.51pp | 96% | 56th |

---

## TIER 3 — BORDERLINE / REJECTED / NULL

| Engine | TF | Status | Why |
|--------|----|--------|----|
| ADX | 4h | BORDERLINE | selective (6.9% deploy!) + DEV/OOS-consistent, but null 93rd (<95) |
| MACD | 1h | REJECTED | big OOS but NEGATIVE DEV edge → OOS-fit, not selectable pre-OOS |
| Stochastic | — | NULL | no selective DEV/OOS-consistent cell |
| Supertrend | 4h | NULL | ~98% deploy (always-long-with-tilt), +0.76pp residual |
| Funding (real norm_funding) | — | NULL | loses to AL every TF even with real feature |
| OI-divergence (real) | — | NULL | beats AL only @1d +0.30pp (noise) |
| FLOW / CMF / AROON / VORTEX | — | NULL | edge flips sign DEV↔OOS (inconsistent) |

---

## CAMPAIGN LINEAGE (where the numbers live)

| Campaign | Families | Artifact dir | Verdict doc |
|----------|----------|--------------|-------------|
| V1 MA drawdown-insurance | 8 MA × {1d,4h} | runs/periods/.../ | verified (halves DD) |
| V2 decoupled MA | 8 MA × 5 TF | runs/engines/V2_subtf_decoupled/ | WMA\|4h +193.5% OOS |
| V3 non-MA SOTA | RSI,MACD,Bollinger,Stoch,ADX,Supertrend,Funding,OI | runs/engines/V3_ti_families/ | STATIC_EXHAUSTION_RESULT.md |
| V4 volume+misc | OBV,VWAP,CMF,MFI,FLOW,Donchian,CCI,WillR,PSAR,Aroon,Vortex,Keltner | runs/engines/V4_ti_extra/ | V4_RESULT.md |

Source code: `ti_engine_sota.py` (+_v11), `ti_engine_extra.py`, `ti_extra_runner.py`,
`ti_static_exhaustion.py`, `ti_sota_oos_test.py`, `ma_wealth_driver.py`.

---

## THE BOTTOM LINE

After 21 TI families through the gold-standard gauntlet, the deployable static base is:
- **3 SELECTIVE validated engines:** MA(WMA)|4h, Bollinger|1d, RSI|4h — deploy MA+RSI.
- **A bench of 8 non-selective long-tilt overlays** (CCI/WILLR/VWAP/Donchian/PSAR/etc.)
  that genuinely add return as a WEIGHTING layer but can't gate on their own.
- The regime gate remains the dominant value lever (consistent with D78 CASH THEOREM).

Static is exhausted to a defensible degree. The next frontier is **Phase 2 dynamic/
adaptive engines** (docs/PHASE2_DYNAMIC_ENGINES_PLAN.md) — and the V4 tilt-overlay bench
gives Phase 2 a concrete target: turn the strongest non-selective tilts (VWAP/WILLR/CCI
at 1d) into selective engines via adaptive thresholds, and improve RSI's diversification.
