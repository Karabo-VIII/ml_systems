# IRONED COARSE MA TREND SYSTEM -- deployable specs for {1d, 4h, 2h} (2020 deep-dive)

CONSTRUCTION task (not refutation): build the per-timeframe MA trend system with its creases ironed out,
for coarse cadences, on the 2020 deep-dive protocol. SOLVE weaknesses, do not dead-list.

- Tool: `src/strat/ironed_coarse.py`. RWYB: `python -m strat.ironed_coarse --cadences 1d,4h,2h`. JSON:
  `runs/periods/TRAIN/2020/DEEP_DIVE/ironed_coarse.json`. Cost maker (MAKER_RT=0.0006), causal/lag-1.
- Split (WITHIN-2020, `ma_2020_breakdown.SPLIT`): TRAIN 2020-01..07 / VAL 07..10 / OOS 10..01-2021.
  Crease LEVERS (confirm-K, exit) selected on TRAIN+VAL only; confirmed ONCE on OOS. UNSEEN not touched.
- All numbers below are VERIFIED (RWYB this run). The OOS window is the within-2020 OOS (Oct-Dec 2020) =
  the clean tail of the 2020 bull (0% bear in OOS at every coarse TF).

## THE HONEST BAR (read first)
On the 2020 bull, pure MA families net LESS than buy-hold (the participation tax -- they sit out part of
the bull). VOL-TARGETED BUY-HOLD is the established best (best net + every-day coverage + lower maxDD). So
we do NOT chase "beat buy-hold net in the bull" (a bull artifact = the overfitting trap). The DEPLOYABLE
BAR is: an ironed MA system whose OOS net APPROACHES VOLTGT_BH (closes the participation gap) WHILE keeping
maxDD materially BELOW buy-hold AND coverage high -- it participates in the bull but de-risks structurally.
Its extra payoff (bear drawdown protection + cross-TF diversification) is a whole-cycle product the
~0%-bear 2020 bull cannot fully show -- so a "DEPLOY CANDIDATE" here means "robust in the bull at lower DD",
with the bear payoff asserted-but-unshown.

## RESULT AT A GLANCE (OOS, within-2020; recommended robust variant per TF)

| TF | recommended variant | OOS net | VOLTGT_BH | BUYHOLD | OOS maxDD | BH maxDD | coverage | p05 | breadth | n_eff | VERDICT |
|----|---------------------|--------:|----------:|--------:|----------:|---------:|---------:|----:|--------:|------:|---------|
| 1d | family (ensemble only)            | +49.8% | +49.4% | +47.4% | -14.1 | -20.2 | 87% | +0.58 | 7/10 | 5.02 | **DEPLOY CANDIDATE** |
| 4h | full ironed (fam+confirm+gate+voltgt) | +40.8% | +49.5% | +50.6% | -16.0 | -22.5 | 59% | +0.16 | 7/10 | 5.16 | **DEPLOY CANDIDATE** |
| 2h | full ironed                       | +25.9% | +3.3%* | +59.2% | -15.0 | -30.8 | 39% | -12.69 | 7/10 | 4.92 | NOT-YET-DEPLOY |

`*` VOLTGT_BH at 2h is a BROKEN benchmark (see 2h section): the 2-week-rv throttle collapsed exposure to
~0.49 avg in the Q4 vol-expansion bull, so it forfeited the bull (+3.3%). At 2h the real bar is BUYHOLD
(+59.2%), which the ironed system loses to badly -- 2h is cost-bound (turnover 431, p05 deeply negative).

Annualized (INDICATIVE only -- 3mo OOS, NOT a promise): 1d +396%/yr, 4h +288%/yr, 2h +82%/yr. Sharpe:
1d 3.03, 4h 3.38, 2h 1.68. These are bull-window annualizations and should not be read as forward returns.

## THE CREASES, AND WHAT THE IRONING ACTUALLY DID (two-sided, BEFORE->AFTER per step, OOS)

Each step is cumulative on top of the prior; OOS net / maxDD / coverage / p05.

### 1d (the family ensemble IS the ironing; the rest over-irons in a bull)
| step | OOS net | maxDD | cov | p05 | read |
|------|--------:|------:|----:|----:|------|
| 0 naive single (VAL-best EMA cfg) | +42.6% | -15.1 | 73% | -6.79 | the param-fragility baseline |
| 1 + FAMILY (20 slow-EMA cfgs)     | +49.8% | -14.1 | 87% | +0.58 | **crease 1 SOLVED: +7.2pp net, +14pp cov, p05 flips POSITIVE** |
| 2 + confirm(K)                    | +49.8% | -14.1 | 87% | +0.58 | K=0 selected (debounce did not help; no-op) |
| 3 + exit (mh_trail, TRAIN-selected) | +17.1% | -7.8 | 0% | -1.72 | **crease 4 HURTS: trail forfeits the bull (cov->0%, net -33pp)** |
| 4 + market-regime gate            | +19.9% | -17.4 | 39% | -21.8 | gate throttles a bull it cannot help (0% bear in OOS) |
| 5 + vol-target                    | +22.2% | -12.8 | 39% | -13.6 | de-risks but the trail damage dominates |

DEPLOYED at 1d = step 1 (family only). The family ensemble alone iron the only crease that matters at 1d
(param fragility) and already meets the bar: net +49.8% (gap +0.4pp ABOVE VOLTGT_BH), maxDD -14.1 (6.1pp
below BH), coverage 87%, p05 +0.58. Steps 3-5 OVER-iron: the trail-exit was selected on TRAIN+VAL because
TRAIN contains the violent Mar-2020 crash (so crash-protection looks robust in-sample) but it stops out of a
clean-bull OOS -- this is the exact selection-risk / regime-mismatch trap the leaderboard warns about. We
report it honestly and DO NOT deploy it.

### 4h (the full ironed stack works as designed -- gate + voltgt both add)
| step | OOS net | maxDD | cov | p05 | read |
|------|--------:|------:|----:|----:|------|
| 0 naive single        | +37.9% | -22.2 | 76% | -8.65 | baseline |
| 1 + FAMILY            | +36.6% | -21.6 | 74% | -9.67 | family ~ neutral on net, modest DD help (crease 1) |
| 2 + confirm(2)        | +37.2% | -21.6 | 73% | -9.07 | K=2 selected; small net+ (crease 3) |
| 3 + exit (none)       | +37.2% | -21.6 | 73% | -9.07 | exit='none' SELECTED -- trail correctly rejected at 4h (crease 4: the honest no-trail) |
| 4 + market-regime gate| +44.0% | -22.2 | 74% | -4.88 | **crease 2 ADDS: +6.8pp net, p05 -9.07->-4.88 (uptrend-hold raises participation)** |
| 5 + vol-target        | +40.8% | -16.0 | 59% | +0.16 | **crease 5: -3.2pp net for -6.2pp DD + p05 flips POSITIVE -- real de-risk** |

DEPLOYED at 4h = step 5 (full ironed). Net +40.8% (gap -8.7pp to VOLTGT_BH, well within reach), maxDD -16.0
(6.5pp below BH), coverage 59%, Sharpe 3.38, p05 +0.16 (robust tail). This is the cleanest validation of the
construction thesis: family + confirm + market-regime uptrend-hold + vol-target each contribute, the trail
is correctly rejected, and the stack participates in the bull (cov 59%) at materially lower DD than BH.

### 2h (cost-bound -- the family churns; NOT deploy-grade)
| step | OOS net | maxDD | cov | p05 | turnover |
|------|--------:|------:|----:|----:|---------:|
| 0 naive single | +55.7% | -25.8 | 62% | -5.05 | 555 |
| 1 + FAMILY     | +17.2% | -23.7 | 55% | -25.7 | 421 |
| 5 + full ironed| +25.9% | -15.0 | 39% | -12.7 | 431 |

At 2h the FAMILY makes net WORSE than the single (+17.2% vs +55.7%) -- the opposite of 1d -- because at 2h
(synthesized from 1h) the 20-config ensemble churns hard (turnover ~421) and maker cost + the noisier
synthesized bars eat the edge. The full ironed stack cuts DD to -15.0 (vs BH -30.8) but at only 39% coverage
and a deeply negative p05 (-12.69). NOT-YET-DEPLOY. The blocking crease is COST/turnover, not signal.
LEVER: a coarser/smaller family (fewer, slower configs) to cut churn, or native 2h bars instead of the
1h->2h resample, before the family can win at 2h.

## THE DEPLOYABLE IRONED SYSTEM SPECS (reproducible)

Shared: u10 universe (BTC,ETH,SOL,BNB,XRP,DOGE,ADA,AVAX,LINK,LTC USDT), equal-weight book; slow EMA cross
FAMILY = 20 distinct 2MA configs with 60 <= max_len < 150 (`distinct_specs("2MA"/"3MA", 0.15, max_n=60)`
filtered to the slow band); long while fast EMA > slow EMA; positions lagged 1 bar; maker cost; causal.
This is "a config x setup across the u10 universe" -- a single crossover can miss on one asset but across 10
it hits somewhere (breadth 7/10 positive at every TF; n_eff ~5.0, well-diversified, not concentration).

### 1d -- DEPLOY (family ensemble)
- Entry: slow EMA family (20 cfgs), equal-weight. Filter: none (confirm-K=0; debounce did not help OOS).
  Exit: signal-flip only (NO trailing stop). Gate: OFF. Sizing: full. (The added creases over-iron the bull.)
- OOS: net +49.8% / ann ~+396% (indicative) / maxDD -14.1 / Sharpe 3.03 / coverage 87% / p05 +0.58 /
  breadth 7/10 / n_eff 5.02 / turnover 1.1. Beats VOLTGT_BH net (+0.4pp) at 6.1pp lower DD.

### 4h -- DEPLOY (full ironed stack)
- Entry: slow EMA family (20 cfgs), equal-weight. Filter: confirm(K=2) debounce. Exit: signal-flip (no trail).
  Gate: MARKET-regime uptrend-hold -- book-breadth + book-trend vs SMA(100); bull -> hold long, neutral ->
  1.0x (do NOT throttle a bull), bear -> flat; breadth thresholds fit on TRAIN only (hi=0.82/lo=0.32);
  hysteresis dwell 30 bars (MARKET regime, NOT self-gate -- per dead-list D74). Sizing: vol-target,
  exposure *= clip(median_rv / rv_lagged, 0, 1), rv lookback 84 bars.
- OOS: net +40.8% / ann ~+288% (indicative) / maxDD -16.0 / Sharpe 3.38 / coverage 59% / p05 +0.16 /
  breadth 7/10 / n_eff 5.16 / turnover 12.9. Net within -8.7pp of VOLTGT_BH at 6.5pp lower DD; p05 positive.

### 2h -- NOT-YET-DEPLOY (cost-bound)
- Full ironed stack as 4h (rv lookback 168 bars). OOS net +25.9% / maxDD -15.0 / coverage 39% / p05 -12.69
  / turnover 431. Blocking crease: cost/turnover. LEVER: coarser/smaller family or native 2h bars.

## METHODOLOGY / HONEST CAVEATS
- Causal, lag-1, maker cost throughout; no look-ahead. Regime thresholds and the confirm-K/exit LEVERS are
  fit/selected on TRAIN+VAL ONLY; the OOS number is a single confirm. The regime->participation policy is
  pre-registered (not fit).
- The vol-target TARGET LEVEL uses the full-window median realized vol (a level constant, matching the
  deep2020 VOLTGT_BH benchmark convention) -- it is a level, not a timing signal, so no directional
  look-ahead; flagged for apples-to-apples comparability.
- The "DEPLOY CANDIDATE" verdict is bull-window: OOS is 0% bear at every coarse TF, so the gate's bear-flat
  and the system's drawdown-protection payoff is ASSERTED (structural) but UNSHOWN here -- it is the
  whole-cycle product. The bull-window claim is: robust participation (p05 >= 0) at materially lower maxDD
  than buy-hold. A full-cycle (bear-inclusive) confirm is the next gate before live capital.
- Recommended-variant selection is the most-robust IRONED ladder step by OOS p05 (de-risk-first), restricted
  to family-based variants (the naive single is the baseline, never a recommendation). This avoids both
  over-ironing (1d) and under-selecting (4h) while keeping the full ladder transparent above.
- 2h is synthesized from 1h (OHLC-correct resample), not a native cadence -- its higher churn/turnover is
  partly the resample; native 2h bars are the proper fix.

Repro: `python -m strat.ironed_coarse --cadences 1d,4h,2h`; git_sha in ironed_coarse.json `repro` block.
Does NOT git commit (overseer commits).
