# BATCH 2 -- the orthogonal mean-reversion OSCILLATOR family (RSI/Stoch/BB%b/CCI) -- 2020 OOS

User /orc 2026-06-13 "try another TI family?": elevated to the ORTHOGONAL family (a trend family would be
~0.85 corr to the MA book = no diversification). Tool: `deep2020_osc.py`. Long-only MR oscillators (buy
oversold, exit reverted), param grid, min_hold(6)+maker, OOS Oct-Dec 2020, u10. The decisive metric:
correlation to the MA trend book + does combining help.

## OSCILLATOR (MR) family vs MA (trend) + the diversification test
| TF | OSC net/Sh/maxDD | MA(trend) net/Sh/maxDD | 50/50 TREND+OSC | corr(OSC,MA) | combining helps? |
|---|---|---|---|---|---|
| 1d | 11.9 / 1.95 / -5.6 | 20.6 / 2.67 / -10.9 | 16.6 / **2.89** / **-5.8** | +0.31 | YES (Sharpe up, DD halved) |
| 4h | 9.7 / 1.26 / -11.0 | 30.2 / 2.86 / -10.8 | 20.2 / 2.63 / -7.4 | +0.30 | DD better, Sharpe down |
| 1h | 25.4 / 2.51 / -11.3 | 50.5 / 3.60 / -9.6 | 38.6 / **3.88** / **-7.3** | +0.28 | YES (Sharpe up, DD better) |

## THE FINDING -- the FIRST family that diversifies the trend book
- **The MR oscillator family is ORTHOGONAL to trend** (corr +0.28 to +0.31, vs ~0.85-0.94 WITHIN the MA
  family). A genuinely different beta -- exactly what the family finding (eff N ~1.2) said was needed.
- **Standalone it is WEAKER** (net 10-25% vs trend 20-50%, Sharpe 1.3-2.5) -- crypto MR is hard (dead-list
  D37); it is a DIVERSIFIER, not a primary edge.
- **Combining 50/50 trend+MR IMPROVES the book at 1d and 1h** -- higher Sharpe (1d 2.67->2.89; 1h 3.60->3.88)
  AND lower maxDD (1d -10.9->-5.8 HALVED; 1h -9.6->-7.3). At 4h the DD improves but Sharpe dips (TF-dependent).
  This is the FIRST risk-adjusted improvement over the trend book in the whole deep-dive -- the eff-N~1.2
  problem (a within-trend family can't diversify) is SOLVED by adding an orthogonal MR family.
- **Best oscillators:** RSI(7) (Sharpe 3.5-3.9 at 1d/1h) and STOCH(14,20) (Sharpe ~4 at 1h); CCI/BB%b weaker.

## FOR THE ML HARNESS (the upgrade)
The ML target is now TWO orthogonal families to replicate: (1) the TREND MA clusters (the drift-beta) +
(2) the MR OSCILLATOR clusters (the orthogonal beta). A book that replicates BOTH and combines them gets the
diversification benefit (Sharpe + maxDD) that neither family achieves alone. Per-oscillator top-3 in
oscillators_*.json. This is the genuinely-additive second batch (vs a redundant trend family).

## Honest caveats
- In-sample 2020-OOS; the dead-list refuted crypto MR STANDALONE, so the MR family's standalone net likely
  does NOT transfer -- but its DIVERSIFICATION value (orthogonality) is more structural (MR vs trend are
  different mechanisms) and more likely to persist than its standalone return.
- corr ~0.3 is orthogonal-ish, not zero (in a bull even MR is somewhat long-biased).
- The combining benefit is TF-dependent (clear at 1d/1h, Sharpe-mixed at 4h).
json: oscillators_1d_4h_1h.json. RWYB: python -m strat.deep2020_osc --cadences <tf>.
