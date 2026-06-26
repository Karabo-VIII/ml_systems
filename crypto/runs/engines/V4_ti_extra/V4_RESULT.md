# V4 TI-EXTRA — VOLUME + MISC families (RESULT, 2026-06-24)

Built the two genuine gaps the first 8-family campaign never covered, with the SAME
SOTA rigor (ensemble + V1 BTC-SMA100 gate + maker cost + decoupled lookback/slice)
and the SAME validation gauntlet (DEV sweep → OOS → DEV/OOS consistency → random-null
→ selectivity). Vectorized end-to-end (11-50s/family vs the prior ~400s loop).

## Families built (12 new engines, REAL data — no proxies)
VOLUME (real chimera volume/buy_vol/sell_vol):
  OBV, VWAP, CMF, MFI, FLOW (buy/sell imbalance)
MISC (non-redundant trend/oscillator gaps):
  DONCHIAN, CCI, WILLR, PSAR, AROON, VORTEX, KELTNER

## Levers pulled (the full anti-under-powering protocol)
1. Vote threshold swept {0.5, 0.6, 0.75, 0.9} (selectivity lever)
2. MA-style decoupled confirmation + 7d slice-hold (the architecture MA won with)
3. Multiple timeframes {4h, 1d}
4. Random-null n=100 on DEV; DEV/OOS consistency; deploy<70% selectivity bar

## VERDICT: 0 VALIDATED (no SELECTIVE engine), but genuine non-selective timing edge

The structural result repeats the original non-MA lesson, now PROVEN for volume/misc:
**these indicators are "long while trending/accumulating," which under a bull regime
gate is true 75-93% of the time.** No threshold or confirmation setting pushes deploy
below the 70% selectivity bar. So none clears the MA-family standard (which needs
SELECTIVE + consistent + null≥95th).

BUT several carry a REAL trend-timing edge (DEV/OOS-consistent AND random-null 100th
pct) — they are just not selective:

| Engine (best cell)        | DEV edge | OOS edge | deploy | null pct | profile |
|---------------------------|----------|----------|--------|----------|---------|
| VWAP\|1d (thr0.9)          | +0.84pp  | +2.61pp  | 80.7%  | 100th    | strong, not selective |
| WILLR\|1d (thr0.6, conf)   | +0.79pp  | +2.67pp  | 90.5%  | 100th    | strong, not selective |
| CCI\|1d (thr0.5)           | +0.96pp  | +2.93pp  | 89.0%  | 100th    | strong, not selective |
| DONCHIAN\|4h (thr0.75)     | +1.22pp  | +2.07pp  | 91.2%  | 100th    | strong, not selective |
| PSAR\|1d (thr0.6, conf)    | +1.04pp  | +1.10pp  | 89.6%  | 100th    | strong, not selective |
| MFI\|1d (thr0.5, conf)     | +0.99pp  | +0.99pp  | 87.8%  | 100th    | strong, not selective |
| KELTNER\|1d (thr0.6, conf) | +0.88pp  | +1.65pp  | 83.2%  | 100th    | strong, not selective |
| OBV\|1d                    | +0.51pp  | +0.48pp  | 95.6%  | 56th     | consistent, weak null |
| VORTEX\|4h (thr0.5, conf)  | +0.64pp  | +2.25pp  | 91.2%  | 18th     | OOS-fit (weak null) |
| FLOW / CMF / AROON         | mixed    | flips    | >80%   | varies   | NULL (inconsistent) |

## Reading (honest)
- **"Strong, not selective"** means: as a TILT on a long book under the gate, these add
  real return vs equal-weight always-long (the +2-3pp OOS, 100th-pct null is genuine).
  But they do NOT time ENTRY/EXIT selectively the way the MA family does (35% deploy).
  They're closer to "smart beta long tilts" than to discrete trade engines.
- This is consistent with the CASH THEOREM (D78) + the standing finding that the regime
  gate is the dominant value lever. Volume/trend TIs refine WHICH longs, not WHETHER long.
- The compound numbers (+9K-232K% DEV) remain inflated ~9.5x by EW-rebalancing premium
  (FORENSIC_AUDIT_RESULT.md). Linear/wk + the null percentile are the honest read.

## What this ADDS to the static base
- No new SELECTIVE engine to deploy standalone.
- BUT VWAP|1d, WILLR|1d, CCI|1d are the strongest non-selective long-tilt overlays
  (100th-pct null, big OOS) — candidates for a WEIGHTING scheme on the long book rather
  than a gate. Logged for the optional Sharpe/vol-weighted ensemble refinement.
- The volume family's orthogonality hope did NOT materialize as selectivity; volume
  indicators track the same trend cluster as MA/Bollinger once gated.

## Static phase — now genuinely exhausted across 21 TI families
9 (original) + 12 (V4) = 21 TI families tested at the MA-family standard.
SELECTIVE validated engines remain: **MA family, RSI|4h, Bollinger|1d**.
Non-selective genuine tilts: VWAP/WILLR/CCI/Donchian/PSAR/MFI/Keltner (1d/4h).
Everything else null. Static design space is exhausted to a defensible degree.
