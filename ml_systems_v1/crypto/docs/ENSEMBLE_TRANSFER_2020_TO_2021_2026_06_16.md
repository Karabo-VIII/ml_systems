# Translating the 2020 selection into 2021: the ROBUST BAND, not the #1 (2026-06-16)

> User /orc 2026-06-16: *"find a way to translate the 2020 results into 2021 ... we didn't just select 1 config,
> but MULTIPLE ones per TI. I suspect this requires extensive research."* Tool:
> [`src/strat/forward_ensemble_2021.py`](../src/strat/forward_ensemble_2021.py). Builds on the 2021 single-#1
> forward test ([FORWARD_TEST_2021](FORWARD_TEST_2021_2026_06_15.md)), which found **rank-transfer ~ 0** -- the
> single 2020-best config does NOT carry to 2021 (ADX(14,20)@4h: 2020-best 36.7% -> 2021-WORST -18%).

## The fix the user pointed at, and the research behind it

The 2020 work selected a **SET of robust configs per TI** (the "working band" = configs positive across
TRAIN+VAL+OOS, |drift|<=10). 13-14 TIs have a multi-config robust band (ADX 6, SUPERTREND 8, ROC 7, DONCHIAN 9,
RSI/STOCH/WILLR 8, BBPCT/CCI/MFI 10, VOLIMB 5, ...). **The band -- not the #1 -- is the regime-invariant object.**

A literature pass (scout brief, cited below) makes the translation method unambiguous:
- The single backtest #1 is **noise out-of-sample** (Bailey/Borwein/Lopez-de-Prado/Zhu, PBO); the stable
  **region/plateau** is the signal (Pardo walk-forward; Zakamulin robust-MA).
- **Equal-weight (1/N) the robust band** is the correct cross-regime translation -- 1/N provably beats any
  performance re-weighting OOS, **especially across a regime shift** (small sample + unstable + OOS-differs-from-IS
  => all three conditions of the forecast-combination puzzle favour 1/N; DeMiguel-Garlappi-Uppal 2009;
  Timmermann 2006; Wang-Hyndman 2022). Re-weighting on 2020-IS re-absorbs the same noise that overfit the configs.
- **Aggregate the return streams** (average), not majority-vote (the configs are homogeneous).
- **Caveat:** the ensemble reduces within-class CONCENTRATION risk; it does NOT rescue the asset-class ceiling
  (long-only TI = de-risked beta). Expect more *robust transfer*, not beating buy-hold.

So: **EW the 2020-robust band per TI (1/N over member daily books)** = PRIMARY; **recency-60d re-weight** = the one
performance-weighting CHALLENGER with crypto evidence (arxiv 2602.11708), tested honestly (must beat EW OOS).

## Result: the band transfers where the #1 collapsed [VERIFIED-2021-FORWARD]

Per-TI 2020-robust EW-ensemble vs the single 2020-#1, forward on the survivorship-clean PIT 2021 universe
(14 TIs with a robust band; CORE = u10 majors, the clean read):

| metric (CORE) | EW-ensemble | single-#1 |
|---|---|---|
| **blow-ups (2021 net < 0)** | **0 / 14** | **2 / 14** (ADX, RSI) |
| **worst-case net** | **+0.5%** | **-12.7%** |
| median net | 9.6% | 15.6% |
| crash-preservation (lost less, May-21) | better at **8 / 14** | -- |

**The poster child -- ADX:** the single #1 ADX(14,20)@4h was the 2020 *best* deployable and the 2021 *worst*
(-12.7% core / -18.4% expand, DD -34%, crash -32%). **Its 6-config band rescues it** to +3.8% core / -6.8% expand,
DD -17%/-19%, crash -15%/-17% -- roughly **half the drawdown** and no catastrophic loss. RSI similarly -1.8% -> +6.9%.

**EXPAND universe** (broader, survivorship-inflated) confirms it: worst-case EW **-6.8% vs #1 -18.4%**, crash-pres
better at 9/14, median EW 20.1 vs #1 17.4. (Honest nuance: on the noisy broad universe 2 bands dip slightly
negative vs the #1's 1, but the worst-loss *magnitude* is always far smaller for the ensemble.)

**Challenger:** recency-60d re-weight beats EW at only 5/14 (core) / 2/14 (expand) -- **1/N EW remains the robust
default**, exactly as the forecast-combination-puzzle literature predicts across a regime shift.

## What this means

**The way to translate the 2020 selection into 2021 is to carry the ROBUST BAND forward as an equal-weight (1/N)
ensemble per TI -- never the single #1.** Doing so:
- **eliminates the catastrophic, unpredictable single-#1 transfer failures** (0 vs 2 blow-ups; worst-case +0.5% vs
  -12.7%) -- you cannot know in advance which 2020-#1 will collapse (ADX looked best, transferred worst), and the
  band is the insurance against that rank-fragility;
- **rescues the exact configs that collapsed** (ADX, RSI) and **preserves better in the crash** with lower DD;
- at the cost of a little peak/median net (the documented robustness trade-off -- robust_ma_runners found the same
  7.9pp rank-fragility tax on 2020; this forward-confirms it across a year);
- and **1/N beats re-weighting** (don't re-optimize the band on 2020 -- the literature and the data agree).

This does **not** break the de-risked-beta ceiling (the ensembles still trail buy-hold in the 2021 bull) -- the
TI class remains a crash-preserving de-risked beta. But it makes the 2020 selection **actually deployable forward**
by removing the rank-instability that the single-#1 forward test exposed. The 2020 -> 2021 translation problem is
SOLVED at the methodology level: **deploy the band, equal-weighted.**

## RWYB
```
python -m strat.forward_ensemble_2021 --selftest          # robust-band load + EW/recency mechanics (PASS)
python -m strat.forward_ensemble_2021                      # core + expand, EW + recency challenger
```
Persists `runs/strat/forward_ensemble_2021_*.json` (per-TI EW/recency/single-#1 + regimes + the transfer verdict).

## Sources (scout brief 2026-06-16)
Bailey-Borwein-Lopez de Prado-Zhu 2016 (PBO/CSCV); DeMiguel-Garlappi-Uppal 2009 (1/N); Timmermann 2006 +
Wang-Hyndman 2022 (forecast-combination puzzle); Pardo 2008 (walk-forward plateau); Zakamulin 2015 (robust MA);
arxiv 2510.23150 2025 (trend-premia redundancy / barbell); arxiv 2602.11708 2025 (crypto adaptive weighting).
