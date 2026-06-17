# Can 2020 results translate to 2021? -- the quant solution (2026-06-16)

User /quant: "look at the whole project (both instances) 2020 TI + MA results + the failed 2021 translation;
I need results to translate into 2021, find me a solution." Tool `src/strat/translation_solution_2021.py`;
data `translation_solution_2021_20260616_003129.json` (492 configs x 6 families x cells; 2020-select / 2021-fwd
PIT survivorship-clean; UNSEEN 2025-26 SEALED; STRICT long-only). All numbers [VERIFIED-RWYB, 2021-forward].

## PRE-REGISTRATION
- H0: no 2020-selection metric beats the no-selection ENSEMBLE on 2021-forward; config translation is
  impossible regardless of metric (net-rank ~0.11 is the ceiling).
- H1: a STRUCTURAL 2020 metric (DD-preservation / time-in / stability) translates -- rank-transfer >> 0.11
  AND its 2021-fwd selected-set beats the ensemble AND a planted null does NOT.
- One-sided; asymmetric loss (false-ship a non-translating rule >> false-skip).

## REFEREE CATCH: the auto-verdict "REAL" is a FAMILY-CLASS CONFLATION (corrected here)
The harness reported verdict=REAL on `net_rank_rho=0.503`. That is NOT a config-selection edge -- it is the
cross-FAMILY class signal, and it fails the planted-null gate. Decisive re-derivation:
- **Cross-pooled** Spearman(2020 OOS net, 2021 fwd net) over all 492 configs = **0.503** -- but this pools 6
  families, so it just measures "good FAMILIES in 2020 stay good in 2021" (the beta class), already known.
- **Within-cell** (per family x cad) mean Spearman = **0.30** (median 0.43) but **dispersed -0.37..+0.68**;
  and among the TOP contenders (the deployable "which one" question) it compresses to **~0.11** (the
  forward-test's number) -- you can rank good-vs-bad, you CANNOT pick THE winner.
- **The planted null is dirty:** 2020 Sharpe-rank also scores rho 0.397 and its selected set also "beats" the
  ensemble (51.5%) -- because Sharpe, like net, picks good families. A test where the PLANTED NULL also wins
  is not isolating a real edge. => the "0.503 REAL" is the CLASS effect, not config selection. **ARTIFACT.**

## What ACTUALLY translates (the family-class table)
FAMILY-mean net, 2020 -> 2021 (the genuine translating result): trend 30->82 | breakout 26->67 | momentum
27->55 | MA 26->46 | **volume 17->1.6 (COLLAPSED)** | mean-reversion 6->7 (flat). The FAMILY ranking is
largely preserved (trend/breakout/momentum/MA on top both years; volume + MR at the bottom). This is the one
selection granularity that translates.

## The translation test (select top-41 by metric -> 2021 fwd net / May-crash)
| 2020 selection metric | 2021 fwd net | May-2021 crash | beats ensemble? | what it picks |
|---|---:|---:|:---:|---|
| ENSEMBLE (no selection, all 492) | 41.8% | -10.6% | -- | everything (incl. dead volume/MR) |
| peak 2020 net [FAILED baseline] | 69.3% | -16.2% | yes* | trend/breakout/momentum families |
| 2020 Sharpe [PLANTED NULL] | 51.5% | -11.3% | yes* | also trend/momentum -> *null wins too |
| worst-subperiod floor | 40.1% | -13.6% | no | mixed |
| crash-DD preservation [structural] | 33.6% | **-3.5%** | no | MFI/RSI (best crash, low net) |
| low-turnover [structural] | 6.3% | -1.9% | no | MFI/RSI/slow-SMA (stalls) |
| cross-subperiod stability [structural]| 3.3% | -3.0% | no | MFI/RSI (stalls) |
| low-time-in / de-risk [structural] | **2.7%** | -1.9% | no | MFI/RSI (barely participates) |

`*` "beats ensemble" is a FAMILY-pick effect (peak-net and the Sharpe-NULL both select trend/momentum), NOT a
config-rank edge. H1 is REFUTED: every STRUCTURAL de-risk metric gives the WORST 2021 net (2.7-6.3%) -- it
selects mean-reversion/volume that barely participates in the 2021 mega-bull. De-risk structure translates the
DRAWDOWN (crash -1.9 to -3.5%, the best) but at a catastrophic NET cost in a bull. The de-risk is insurance,
and in a +466% bull the premium eats the return.

## THE SOLUTION (honest, and it IS a solution -- just not the one config-picking hoped for)
1. **Translate at the FAMILY granularity, not the config.** Deploy the families that translate -- TREND +
   BREAKOUT + MOMENTUM (+ MA) -- and DROP the families that don't (volume collapsed 17->1.6; MR flat). This
   is the genuine, reproducible translating selection (the family ranking persisted across a full year).
2. **Within a family, ENSEMBLE the band -- never pick the #1.** Config-rank among the top is ~0.11 (noise);
   the ensemble removes the un-pickable rank bet (the robust_ma_runners finding, now confirmed forward).
3. **Size the de-risk for DRAWDOWN, not net.** Heavy de-risk (low-time-in selection) translates crash-
   preservation (-2%) but kills net (2.7%). The deployable balance = a TREND/BREAKOUT family band-ensemble
   with a LIGHT vol-target/trail (crash insurance) -- 2021 net ~50-69%, May-crash ~-10 to -16% vs core-BH
   crash -58.9%.
4. **The hard truth (rigorously confirmed both years):** NO long-only TI/MA selection beats buy-hold on NET in
   2020 OR 2021 (0/21 both years; the 2021 bull was +466%, the best translating book ~69%). What translates is
   (a) FAMILY-class participation and (b) drawdown-preservation -- a drawdown-preserving beta book, NOT
   bull-beating alpha. Within the long-only constraint that is the honest ceiling.

## Cheapest falsifier
Run the family-selected (trend+breakout+momentum) band-ensemble vs the Sharpe-NULL family-selection forward on
2021: if the NULL matches it (it does -- both pick the same families), then there is NO selection edge beyond
"pick the good families + ensemble + de-risk"; that IS the solution and the config-picking dream is closed.

Charts: `translation_scatter_*.png` (structural-feature vs 2021-fwd, all negative), `translation_metric_bars_*.png`.
