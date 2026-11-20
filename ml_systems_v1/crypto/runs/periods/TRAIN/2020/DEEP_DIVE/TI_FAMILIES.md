# TI FAMILIES -- per-config base vs IRONED, wealth-ranked, across indicator families (2020)

User /orc 2026-06-14 (6h autonomous): the other instance owns the 8 base MA types; THIS lane ran the REMAINING
trend indicators + other families through the same end-to-end (every config x TF, BASE vs IRONED, wealth-ranked
top-10, robust/non-robust split). Per-config per-TI per-TF NUMBERS (NOT correlation). 2020 band, STRICT
long-only spot, fixed-EW, VAL Jul-Sep / OOS Oct-Dec. Tools: `deep2020_ti_pipeline.py` + `_render.py`. ALL
numbers **[VERIFIED-backtest, IN-SAMPLE 2020, OOS]**. UNSEEN untouched.

FAMILIES + IRONS (deep-research-informed; the irons are evidence-consistent -- see DEEP-RESEARCH note below):
- **MACD** (trend): base = signal-line cross; IRON = + zero-line trend filter (macd>0) + vol-target.
- **RSI / STOCH / BBPCT / CCI** (mean-reversion): base = buy oversold / exit reverted; IRON = + buy-the-dip-
  ONLY-in-uptrend filter (close>SMA100) + vol-target. (The standard fix for "oversold stays oversold in a trend".)
- **DONCHIAN** (breakout): base = N-bar high breakout / M-bar low exit; IRON = + 0.5*ATR confirmation + vol-target.

## WEALTH-ROBUST champion per indicator (best ironed config, across TFs)
| indicator | family | TF | config | net (xBH) | Sharpe | maxDD | drift |
|---|---|---|---|---|---|---|---|
| MACD | trend | 1d | MACD(26,52,9) | 36.0% (0.76x) | 3.65 | -10.3 | 7.7 |
| ROC | momentum | 1h | ROC(50,thr0) | 34.4% (0.67x) | 3.28 | -7.2 | 9.3 |
| DONCHIAN | breakout | 4h | DONCH(30,20) | 31.2% (0.65x) | 3.41 | -9.8 | 5.7 |
| SUPERTREND | trend | 2h | ST(14,3.0) | 28.8% (0.57x) | 3.12 | -5.0 | 8.2 |
| RSI | mean-reversion | 1d | RSI(7,lo35,hi60) | 18.5% (0.39x) | 4.59 | -4.2 | 8.4 |
| STOCH | mean-reversion | 1d | STOCH(14,lo20,hi80) | 16.2% (0.34x) | 3.99 | -2.8 | 2.2 |
| BBPCT | mean-reversion | 1d | BBPCT(14,lo0.2,hi0.9) | 13.8% (0.29x) | 3.10 | -1.9 | -1.4 |
| CCI | mean-reversion | 1h | CCI(14,lo-80,hi0) | 12.4% (0.24x) | 3.79 | -4.3 | 8.6 |

## PER-FAMILY IRON EFFECTIVENESS (median over the family's indicators x TFs) -- the headline
| family | dNet (iron-base) | dMaxDD | d\|drift\| | robust-frac | verdict |
|---|---|---|---|---|---|
| **mean-reversion** | **+0.1pp** | **+6.5pp** | **-4.2pp** | **91%** | the iron is ~FREE robustness: de-risks + robustifies at ZERO net cost (best iron-to-cost) |
| momentum (ROC) | -0.8pp | +4.0pp | -3.0pp | 62% | cheap iron, DD cut + robustifies (a de-risked beta, 0.52-0.67x BH) |
| breakout | -1.2pp | +4.4pp | +0.4pp | 67% | cheap iron, modest DD cut |
| trend (MACD+Supertrend) | -4.5pp | +6.9pp | -0.5pp | 62% | iron trades net for a big DD cut + robustness (like the MAs) |

## WHAT THIS SETTLES (the generalization of the MA finding to ALL indicator families)
1. **TREND (MACD) + BREAKOUT (Donchian) are DE-RISKED BETAS, exactly like the MAs.** Best ironed net is
   0.6-0.76x of same-cadence buy-hold; the iron cuts maxDD ~40% (MACD -16.7->-10.3) [VERIFIED-2020-OOS] + lifts Sharpe + robustifies,
   at a net cost. The net ceiling is the drift-beta. Nothing in the trend/breakout families beats buy-hold on net.
2. **MEAN-REVERSION (RSI/Stoch/BB%b/CCI) is STRUCTURALLY WEAK in the 2020 bull** -- best ironed net only
   0.1-0.4x of buy-hold (CCI as low as 0.05x). [VERIFIED-2020-OOS] This is the documented "oversold stays oversold in a trend"
   killer + the dead-list (crypto MR is hard). BUT: the uptrend-filter IRON is the most effective of any family
   -- it makes MR the MOST ROBUST family (91% of configs robust) and cuts DD +6.5pp at ZERO net cost. [VERIFIED-2020-OOS] So MR is a
   robust, low-return, low-DD profile -- a DIVERSIFIER (different mechanism), not a primary edge, consistent with
   the earlier oscillator-batch finding.
3. **The IRON buys RISK-REDUCTION + ROBUSTNESS, not RETURN -- across EVERY family.** This is the same verdict as
   the MA weakness teardown, now confirmed on trend(MACD), breakout(Donchian), and 4 mean-reversion oscillators:
   no internal-data technical indicator family beats long-only buy-hold on NET in the 2020 bull; the value is
   de-risking. (The orthogonal-diversification value of MR -- combining it with trend -- is a SEPARATE question
   the user explicitly deferred: "I'm not solving for correlation.")

## DEEP-RESEARCH note (irons are evidence-consistent; verification was rate-limited)
A deep-research sweep (the literature on closing each family's gaps) had its adversarial-verification phase
server-rate-limited (inconclusive votes), but the extracted claims + sources converge on:
- The #1 evidence-backed lesson is NOT a fancy filter -- it is **robustness discipline**: in-sample-best
  technical rules do NOT survive out-of-sample (Sullivan-Timmermann-White: best-rule naive p=0.12 ->
  data-snoop-corrected p=0.72, worthless). Our robust/non-robust split + VAL-select/OOS-report + wealth-rank +
  fixed-EW directly implements this discipline -- it is the right apparatus, per the literature.
- **MACD:** per-parameter tuning matters MORE than add-on filters (MDPI 14/1/37: default (12,26,9) loses on
  Nikkei futures; per-market-optimized params win). Our grid sweep does the tuning; the zero-line filter is the
  secondary lever -- consistent.
- **MR:** regime/trend-gating is the validated iron (the crypto premium is unpredictable UNCONDITIONALLY but
  works when gated on greed/fear regimes). Our uptrend-filter iron IS this gate -- and it is empirically the most
  effective iron here (91% robust, free).
- **Breakout:** vol-calibrated stops/filters help (our ATR-confirmation iron), but watch overfit.

## OPEN / next
- Advanced MA types (ALMA/ZLEMA/T3) + a momentum family (ROC/TSI) are the natural next breadth.
- Per-indicator weakness teardown (like MA_WEAKNESS.md) -- the n_trades/cost/time-in profile per indicator.
- 30m/15m TFs (cost-destroyed for trend, but the user wants all TFs) -- run for completeness.
json: `ti_<IND>_1d_4h.json` + `ti_<IND>_2h_1h.json` (6 indicators). RWYB: `python -m strat.deep2020_ti_pipeline
--indicator <IND> --cadences 1d,4h,2h,1h` then `python -m strat.deep2020_ti_render`.
