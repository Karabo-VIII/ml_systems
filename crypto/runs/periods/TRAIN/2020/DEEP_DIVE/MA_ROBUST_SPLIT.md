# ROBUST vs NON-ROBUST configs per MA type, ranked by WEALTH (2020)

User /orc 2026-06-14: "separate the robust list from the non-robust (don't throw the list away). And there's
optimising for WEALTH vs Sharpe, right?" -- correct on both. (1) The binding **OBJECTIVE FUNCTION (2026-05-24)
is WEALTH = held-out compound return, NOT Sharpe**; the earlier leaderboards ranked by Sharpe, which surfaced
the thin-time-in OOS-lucky 1h configs. This re-cut ranks by WEALTH. (2) Both groups are KEPT, split by
robustness. Tools: `deep2020_ma_robust_split.py` + `_render.py`. ALL numbers **[VERIFIED-backtest, IN-SAMPLE
2020, base FULL stack, maker, long-only spot lev<=1]**; pooled across 1d/4h/2h/1h. UNSEEN untouched.

DEFINITIONS: **ROBUST** := |drift| <= 10 (delivers in BOTH VAL+OOS; a ~+5-8 positive-drift baseline is expected
from the fixed-EW VAL cash-drag, so the tolerance allows for it). **worst** = min(VAL, OOS) net = compound in
your WORSE window = the wealth-ROBUST metric. NON-ROBUST kept + tagged **LUCKY** (OOS>>VAL, e.g. the Nov-2020
quarter) or **OVERFIT** (VAL>>OOS, fragile).

## THE HEADLINE: WEALTH-leader != SHARPE-leader (for EVERY MA type) [VERIFIED-2020-OOS]
Optimizing for Sharpe leaves wealth on the table. The deployable ROBUST pick per MA by each objective:
| MA | WEALTH-leader (max OOS net, robust) | SHARPE-leader (max Sh, robust) | wealth gap |
|---|---|---|---|
| EMA | 4h EMA(6,16) net 41.1% Sh 2.81 | 2h EMA(8,132,233) Sh 3.65 net 36.4% | +4.7pp |
| SMA | 4h SMA(8,28) net 44.0% Sh 3.19 | 4h SMA(8,132,233) Sh 3.55 net 29.1% | **+14.9pp** |
| WMA | 4h WMA(6,65) net 40.7% Sh 2.93 | 4h WMA(8,132,233) Sh 3.29 net 31.5% | +9.2pp |
| HMA | 2h HMA(5,143) net 43.2% Sh 3.09 | 1h HMA(26,45) Sh 3.39 net 41.2% | +2.0pp |
| DEMA | 1d DEMA(12,39) net 43.1% Sh 3.08 | 1h DEMA(86,170) Sh 3.72 net 37.8% | +5.3pp |
| TEMA | 2h TEMA(12,151) net 39.7% Sh 2.96 | 1h TEMA(73,145) Sh 3.72 net 39.1% | +0.6pp |
| KAMA | 2h KAMA(2,73) net 36.8% Sh 2.73 | 1h KAMA(48,67,94) Sh 4.04 net 34.5% | +2.3pp |
| VIDYA | 4h VIDYA(2,3,4) net 42.8% Sh 3.14 | 2h VIDYA(8,132,233) Sh 4.37 net 24.7% | **+18.1pp** |

The Sharpe-leaders are systematically the slower/more-selective vslow configs (high Sharpe, low time-in, less
wealth); the wealth-leaders are the more-participating mid/fast configs at 4h/2h. **The wealth objective points
to 4h/2h mid configs; the Sharpe objective points to vslow.** They are genuinely different strategies.

## THE DEPLOYABLE PICK: wealth-ROBUST champion per MA (max worst-window = most wealth in the WORSE window)
This is the single most defensible config per MA -- it made the most compound even in its weaker window:
| MA | TF | config | worst (min VAL,OOS) | VAL / OOS | Sharpe | maxDD |
|---|---|---|---|---|---|---|
| EMA | 4h | EMA(5,75,208) | 33.2% | 33.5 / 33.2 | 3.12 | -10.2 |
| **SMA** | 4h | **SMA(8,28)** | **39.0%** | 39.0 / 44.0 | 3.19 | -13.5 |
| WMA | 4h | WMA(22,33) | 36.9% | 36.9 / 40.0 | 2.99 | -13.5 |
| **HMA** | 1h | **HMA(26,45)** | **38.0%** | 38.0 / 41.2 | 3.39 | -6.8 |
| **DEMA** | 4h | **DEMA(18,33)** | **39.3%** | 39.3 / 40.4 | 2.95 | -11.2 |
| TEMA | 1d | TEMA(4,5) | 37.7% | 37.8 / 37.7 | 2.31 | -17.2 |
| KAMA | 2h | KAMA(15,33) | 33.5% | 33.5 / 34.0 | 2.83 | -10.4 |
| VIDYA | 4h | VIDYA(4,5) | 35.1% | 35.1 / 36.1 | 2.68 | -12.8 |

**DEMA(18,33) @4h, SMA(8,28) @4h, and HMA(26,45) @1h are the standouts** -- ~38-40% in BOTH windows. [VERIFIED-2020-OOS] HMA(26,45)
is the rare 1h config that is genuinely robust (drift 3.2, maxDD only -6.8) rather than OOS-lucky. [VERIFIED-2020-OOS]

## THE SPLIT, per MA type (counts pooled across the 4 cadences; ~320 config-instances each)
| MA | ROBUST | NON-ROBUST | robust% |
|---|---|---|---|
| EMA | 185 | 135 | 58% |
| SMA | 191 | 129 | 60% |
| WMA | 179 | 141 | 56% |
| HMA | 152 | 168 | 48% |
| DEMA | 186 | 134 | 58% |
| TEMA | 174 | 146 | 54% |
| KAMA | 188 | 132 | 59% |
| VIDYA | 188 | 132 | 59% |

By cadence the split is stark [VERIFIED-2020-OOS]: **4h is mostly ROBUST (49-74 of 80 per MA), 1h is mostly NON-ROBUST (22-39 of 80)**
-- i.e. a 1h config is more likely OOS-lucky than robust, while a 4h config is more likely to hold up. This is
the single-config confirmation of "4h is the trustworthy cadence" from MA_REMEDY.

## NON-ROBUST is KEPT (not discarded) -- the LUCKY list (OOS>>VAL, big OOS off the Nov quarter)
The non-robust top configs by OOS net are almost all 1h LUCKY (modest VAL ~12-20%, big OOS ~48-60%) [VERIFIED-2020-OOS]. Examples:
1h SMA(8,108) 15.7/59.3 (+44), 1h WMA(37,124) 9.9/57.3 (+47), 1h HMA(4,5) 21.6/60.0 (+38), 1h VIDYA(10,16)
11.9/51.1 (+39). [VERIFIED-2020-OOS] They posted the highest raw OOS net AND Sharpe -- which is exactly why a
naive "top by OOS Sharpe/net" picks them and why they would DISAPPOINT out-of-sample (they didn't earn it in
VAL). They are kept here as the cautionary list, full data in the json.

## WHAT THIS SETTLES
1. **Rank by WEALTH (the mandate), not Sharpe** -- the two objectives pick DIFFERENT configs for every MA type;
   Sharpe-optimizing cost up to ~18pp of compound (VIDYA) / ~15pp (SMA). [VERIFIED-2020-OOS]
2. **WEALTH + ROBUST converges on 4h/2h mid configs** (the wealth-leaders) and the worst-window champions
   (DEMA(18,33), SMA(8,28), HMA(26,45)) are the deployable single-config picks per MA.
3. The non-robust LUCKY list is kept as the "do not be fooled by raw OOS" reference -- high net/Sharpe that did
   NOT hold in VAL.
4. For the ML harness: target the ROBUST wealth-leaders + worst-window champions, NOT the Sharpe/OOS-net leaders.

RWYB: `python -m strat.deep2020_ma_robust_split --cadences 1d,4h` then `--cadences 2h,1h`; `python -m
strat.deep2020_ma_robust_render`. json: `ma_robust_1d_4h.json` + `ma_robust_2h_1h.json` (ALL configs + flag).
