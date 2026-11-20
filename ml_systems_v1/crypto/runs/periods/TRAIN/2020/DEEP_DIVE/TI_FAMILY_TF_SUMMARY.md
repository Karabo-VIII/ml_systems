# TI results SUMMARY -- per indicator family x timeframe (2020, ironed, fixed-EW, long-only)

Summary of the 18-indicator / 6-family sweep (base vs ironed, every config per TI per TF). All numbers
**[VERIFIED-backtest, IN-SAMPLE 2020 OOS, fixed-EW, maker, long-only spot]**. Source: `ti_*.json` via the
`deep2020_ti_*` pipeline. Buy-hold rises with finer TF (rebalancing premium) so JUDGE BY xBH, not raw net.

**BUY-HOLD net% per TF (the benchmark/ceiling):** 1d 47.4 / 4h 47.8 / 2h 50.2 / 1h 51.6 / 30m 53.2 / 15m 54.8.

## (1) BEST IRONED net% (xBH) per family x TF -- the capture
| family | 1d | 4h | 2h | 1h | 30m | 15m |
|---|---|---|---|---|---|---|
| trend | 40 (.85) | 37 (.77) | 32 (.64) | 37 (.72) | 22 (.40) | 4 (.07) |
| momentum | 38 (.80) | 40 (.85) | 37 (.73) | 35 (.68) | 19 (.35) | 15 (.27) |
| breakout | 38 (.80) | 31 (.65) | 28 (.57) | 28 (.54) | 14 (.26) | 3 (.05) |
| mean-rev | 25 (.53) | 16 (.33) | 10 (.21) | 26 (.51) | 17 (.31) | 22 (.40) |
| volume | 56 (1.18)* | 36 (.76) | -- | 40 (.77) | 22 (.40) | 14 (.26) |

*volume@1d 1.18x = OBV(20) which is OOS-LUCKY (non-robust, drift +27). The deployable (ROBUST) ceiling stays
<= ~0.85x across every family x TF -- nothing robust beats buy-hold on net (the drift-beta ceiling holds).

## (2) ROBUST-fraction (share of ironed configs that deliver in BOTH VAL+OOS) -- the trust
| family | 1d | 4h | 2h | 1h | 30m | 15m |
|---|---|---|---|---|---|---|
| trend | 45% | 89% | 63% | 60% | 91% | 92% |
| momentum | 29% | 60% | 85% | 60% | 100% | 100% |
| breakout | 48% | 100% | 73% | 47% | 80% | 93% |
| mean-rev | 93% | 90% | 100% | 73% | 82% | 95% |
| volume | 58% | 58% | -- | 58% | 65% | 94% |

## (3) median ironed maxDD% -- the risk
| family | 1d | 4h | 2h | 1h | 30m | 15m |
|---|---|---|---|---|---|---|
| trend | -11.9 | -7.7 | -5.8 | -5.6 | -7.6 | -10.1 |
| momentum | -12.1 | -9.6 | -6.4 | -5.8 | -5.4 | -6.0 |
| breakout | -10.3 | -7.3 | -5.1 | -6.3 | -6.4 | -10.1 |
| mean-rev | **-2.1** | -5.0 | -7.1 | -4.0 | **-2.9** | **-2.7** |
| volume | -6.4 | -6.1 | -- | -5.3 | -5.6 | -6.2 |

## (4) median ironed Sharpe
| family | 1d | 4h | 2h | 1h | 30m | 15m |
|---|---|---|---|---|---|---|
| trend | 3.17 | 2.88 | 2.98 | 2.89 | 1.14 | **-0.81** |
| momentum | 2.84 | 2.85 | 3.03 | 2.95 | 1.44 | 0.76 |
| breakout | 3.34 | 2.92 | 3.08 | 2.86 | 1.22 | **-1.10** |
| mean-rev | 2.39 | 1.12 | 0.69 | 2.96 | 2.21 | 2.27 |
| volume | 2.30 | 2.90 | -- | 2.86 | 2.22 | 0.87 |

## THE READS
1. **The directional families (trend / momentum / breakout / volume) peak at 1d-1h** (best capture 0.54-0.85x
   BH, Sharpe ~2.9-3.3, shallow maxDD) and **COLLAPSE at 30m/15m** -- cost destroys them: capture falls to
   0.05-0.40x and Sharpe goes NEGATIVE at 15m (trend -0.81, breakout -1.10). The fine-TF whipsaw cost is the
   binding constraint, exactly as on the MAs. **The deployable band for these families is 1d-1h (4h the sweet
   spot for momentum/trend; 1d for breakout/volume).**
2. **Mean-reversion is the mirror image: weak at coarse (0.21-0.53x) but the ONLY family that does NOT collapse
   at fine TF** (15m 0.40x, 30m 0.31x -- it actually does relatively BETTER fine, because MR trades are short so
   fine bars suit them) AND it has the **lowest maxDD of any family (-2 to -7) and the highest robust-fraction
   (73-100%)**. It is the defensive, low-return, most-robust diversifier -- NOT a primary edge in the bull.
3. **Robust-fraction rises at finer TF for most families (80-100% @30m/15m)** -- but that is survivorship: only
   the cost-survivable (mostly near-buy-hold or barely-trading) configs stay positive in both windows, and their
   NET is tiny. High robust-frac at fine TF does NOT mean deployable -- cross-reference with capture (table 1).
4. **The single most trustworthy cells (high robust-frac AND good capture AND shallow DD): 4h breakout
   (100% robust), 4h trend (89%), 2h momentum (85%), 1h mean-rev (73% @0.51x).** The best deployable robust
   config overall is ADX(14,20)@4h (0.77x BH, Sh 3.49, maxDD -6.9) -- see TI_BEST.md.
5. **Verdict (unchanged, now per-cell explicit): no family x TF cell beats buy-hold on net with a ROBUST config**
   -- the iron buys risk-reduction + robustness, not return; the ceiling is the drift-beta. The families differ
   in WHERE they de-risk best: directional @ 1d-1h, mean-reversion @ any-TF-but-low-return + lowest-DD.

Full per-config numbers: `TI_MASTER.md` (92 cells), `TI_BEST.md` (1118 robust ranked), `TI_FAMILIES.md`,
`TI_BREADTH.md` (firewall). RWYB: the `deep2020_ti_*` pipeline.
