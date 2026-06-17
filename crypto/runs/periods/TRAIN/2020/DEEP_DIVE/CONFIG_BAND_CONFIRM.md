# CONFIG-BAND CONFIRM + PLOT (2020) -- verdict

> **CORRECTION BANNER (2026-06-14, after the cross-harness reconciliation MA_RECONCILIATION.md).**
> The FINE-TF magnitudes in the prose below predate the SKIPNA fix and are INFLATED (the book/buy-hold
> aggregation used `mean(axis=1, skipna=True)`, which reweights to EW-of-present on SOL/AVAX 2020 listing +
> thin-trading gaps -> inflates finer cadences). The aggregation is now `fillna(0.0).mean` (fixed-EW,
> cadence-INVARIANT). The PLOTS + payload json in this folder ARE regenerated (corrected); only this prose
> file's specific fine-TF numbers are pre-fix. **Corrected, cadence-invariant headline (FULL-2020):** u10
> buy-hold 140-157% across all TFs (was 199@1d -> 675@15m); 1d top configs still BEAT buy-hold full-cycle via
> crash cash-avoidance -- WMA(6,9,24) +159.7% / TEMA(8,22,60) +162.7% / HMA(37,38) +155.8% vs buy-hold +140.2%
> (was +269.7/+262.4/+248.3 vs +199.2). ALL 3 VERDICTS HOLD; only the magnitudes came down. The working BAND
> (sign test) + the 1d-OOS numbers + the crash-avoidance MECHANISM are UNAFFECTED. See config_leaderboard.json
> for the corrected per-cell numbers.


**Task:** independently re-derive (RWYB) the per-config 2MA/3MA working-band headline claims from
`config_leaderboard.json`, prove the load-bearing crash-avoidance mechanism, and render presentation-grade plots.

**Constraint honoured throughout:** STRICT LONG-ONLY + spot (held in {0,1}, no short/inverse/long-short anywhere).
2020 BAND ONLY. Causal/lag-1, maker cost (RT=0.0006), ironed sleeve = MA-cross -> 10% trail -> min_hold(12).
The sleeve apparatus is REUSED VERBATIM from `strat.ma_2020_config_leaderboard` (no reinvention); the only new
computation is EXPOSURE (the lagged position series), surfaced by replicating `config_book`'s `pos` lines exactly.

- Re-derivation script: `src/strat/confirm_plot_config_band.py`  (RWYB: `python -m strat.confirm_plot_config_band`)
- Source artifact: `runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json` (git_sha=1fd2802)
- Machine-readable re-derivation: `runs/periods/TRAIN/2020/DEEP_DIVE/config_band_confirm_payload.json`

---

## VERDICT: ALL 3 HEADLINE CLAIMS HOLD (one labelling clarification on the band counts)

| Claim | Status |
|---|---|
| 1. The working BAND is real; VIDYA widest, HMA/TEMA narrowest | **CONFIRMED** (with a counts-framing note) |
| 2. Top 1d configs BEAT no-cost buy-hold full-cycle; mechanism = MARCH-CRASH CASH-AVOIDANCE | **CONFIRMED -- the load-bearing positive is REAL, an exposure-to-cash effect, NOT a ranking artifact** |
| 3. Within-band RANK is regime-transient (median rho 0.57; top-10 overlap 0-8/10) | **CONFIRMED EXACTLY** |

---

## CLAIM 1 -- the BAND is real; VIDYA widest, HMA/TEMA narrowest

Re-ran the live sleeve (`run_cell`) on three 1d cells and cross-checked the band counts against the JSON:

- VIDYA 1d: live (2MA=44, 3MA=35) == JSON (44, 35) [MEASURED, match]
- EMA   1d: live (2MA=59, 3MA=56) == JSON (59, 56) [MEASURED, match]
- TEMA  1d: live (2MA=55, 3MA=57) == JSON (55, 57) [MEASURED, match]

**Band-width per MA-type (avg # in-band across all 6 TFs, out of 120 = 60 2MA + 60 3MA) [MEASURED]:**
VIDYA 110 (widest) > EMA 105 > WMA 97 > SMA 96 > KAMA 93 > DEMA 86 > **HMA 79 > TEMA 76 (narrowest)**.

**CLARIFICATION on the headline numbers (no error, just the grain):**
- "~44-59 of 120 positive across TRAIN&VAL&OOS in robust cells" is a **per-KIND (x/60)** count, not x/120 --
  e.g. VIDYA-1d-2MA=44, EMA-1d-2MA=59. CONFIRMED at that grain.
- "VIDYA has the WIDEST band (56-57/120)" refers to the **15m cell specifically** (VIDYA 15m: 2MA=56, 3MA=56 =
  112/120). CONFIRMED. VIDYA is the standout because it is the *only* MA-type that STAYS wide at fine TF (the
  adaptive-MA finer-TF win): at 15m, HMA collapses to 30/120 and TEMA to 29/120 while VIDYA holds 112/120.
- "HMA/TEMA the NARROWEST (11-19 at 15m)" refers to the **15m 2MA/3MA kind counts**: HMA-15m (2MA=19, 3MA=11),
  TEMA-15m (2MA=18, 3MA=11). CONFIRMED exactly.

So the band exists, is sizeable, and the VIDYA-widest / HMA-TEMA-narrowest ordering holds on both the
avg-across-TF view (chart d) and the specific 15m cells the headline cited.

## CLAIM 2 -- top 1d configs BEAT no-cost buy-hold; mechanism = MARCH-CRASH CASH-AVOIDANCE [LOAD-BEARING]

**Re-derived FULL/OOS net via `config_book` + `_metrics` -- match the JSON to the decimal [MEASURED]:**

| config (1d) | live FULL net | JSON FULL net | live OOS net | JSON OOS net | FULL maxDD | beats buy-hold? |
|---|---:|---:|---:|---:|---:|:---:|
| WMA(6,9,24)  | +269.7% | +269.7% | +33.5% | +33.5% | -23.5% | YES (+70.5pp) |
| TEMA(8,22,60)| +262.4% | +262.4% | +40.9% | +40.9% | -13.5% | YES (+63.2pp) |
| HMA(37,38)   | +248.3% | +248.3% | +41.0% | +41.0% | -18.6% | YES (+49.1pp) |

Equal-weight u10 buy-hold (no cost) FULL-2020 net = **+199.2%**, maxDD = **-60.9%** [MEASURED].
All three named configs reproduce exactly (claim said +270/+262/+248 vs +199 -- CONFIRMED).

**The mechanism is genuine exposure-to-cash, NOT a ranking artifact [MEASURED]:**
The configs win on BOTH return AND drawdown -- impossible for a long-only beta book UNLESS it sidesteps the
crash. I independently computed each config's EXPOSURE (time-in-market 0..1, the lagged position) and confirmed:

| top-5 config (1d) | avg exposure in Feb-Mar crash | min exposure in crash | maxDD FULL | maxDD H1 (the crash half) |
|---|---:|---:|---:|---:|
| WMA(6,9,24)   | 14% | 0% (full cash) | -23.5% | -23.5% |
| WMA(2,19,22)  | 21% | 0% | -20.4% | -20.3% |
| TEMA(8,22,60) | 7%  | 0% | -13.5% | -11.3% |
| WMA(2,8,22)   | 16% | 0% | -24.0% | -24.0% |
| DEMA(8,14,38) | 5%  | 0% | -18.0% | -18.0% |

Crash window = 2020-02-19..03-31 (the COVID "Black Thursday" -50% BTC drop). During it the long-only band sits
**5-21% in-market (hitting 0% = full cash)** while buy-hold stays ~100% exposed and loses **-60.9%** (its H1
maxDD is also -60.9%). The configs cap their H1 drawdown at **-11% to -24%**. **maxDD avoided ~ 37-50pp.**
This is the within-constraint answer to bears: the long-only MA EXIT moves to CASH -- it does NOT (and cannot,
by constraint) short. CONFIRMED as a real exposure-to-cash effect.

## CLAIM 3 -- within-band RANK is regime-transient

Read straight from the leaderboard's own stability blocks (its stated output) [MEASURED]:
- **Median Spearman rho (TRAIN+VAL net vs OOS net) across 48 cells = 0.572** == the JSON `median_spearman_rho`
  field (0.572). Mean rho = 0.526.
- Top-10 overlap ranges **0/10 to 8/10** across cells (the ordering only partially transfers).

CONFIRMED exactly. rho ~0.57 is "some persistence but far from reliable" -> the operating rule "trust the BAND
(the robust set the ensemble rides), not the exact #1" is the right read. (Finer TF transfers somewhat better,
consistent with the leaderboard's per-cell numbers.)

---

## LOOK-AHEAD / FRAMING (stated, not hidden -- no leak found)

- The sleeve is **causal/lag-1**: position at bar t uses held[:t] and close[:t+1] only (verified in the
  `config_book`/`config_exposure_and_equity` code path -- `pos[1:] = w[:-1]`). The trail-stop and min-hold
  overlays are causal. No forward information enters a trade decision.
- The BAND and the per-config RANK are computed **on FULL-2020 = DESCRIPTIVE of what was discovered over the
  year, NOT a forward predictor.** The 3-way-positive band test (TRAIN&VAL&OOS) is an in-sample robustness
  filter, not an out-of-sample forecast. The rank-transience number (Claim 3) is precisely the honest
  quantification of how little the in-sample ordering would carry forward.
- The crash-avoidance (Claim 2) is a causal **mechanism** check -- "does the long-only exit actually go to cash
  when price craters?" -- answered YES, and it is real regardless of ranking, because every config's exposure
  drops to ~0 in the crash independent of where it ranks.
- Data caveats inherited from the leaderboard (unchanged): 2h is synthesised from 1h; SOL/AVAX have only
  2020-H2 history (absent from TRAIN, present VAL/OOS); the book averages over assets present per bar (skipna);
  2020-OOS (Oct-Dec) is a clean bull, so long-only under-participation vs buy-hold there is expected, not a defect.

---

## PLOTS (runs/periods/TRAIN/2020/DEEP_DIVE/charts/)

| file | what it shows |
|---|---|
| `config_top_equity_1d.png`  | Top-5 1d configs' equity vs EW buy-hold, FULL-2020, crash SHADED. The colored lines stay ~flat (in cash) through the shaded crash while buy-hold (black dashed) craters to ~0.7x; configs end +248-270% vs buy-hold +199%, with maxDD ~-13 to -24% vs -61%. The band beats buy-hold full-cycle AND cuts drawdown. |
| `config_top_equity_30m.png` | Same view at 30m. Crash-avoidance is again visible (buy-hold craters to ~0.4x); at 30m the configs UNDER-participate vs the +675% fine-TF buy-hold post-crash -- the expected participation tax, not a defect. |
| `band_ensemble_vs_buyhold.png` | Small-multiple (1d/4h/15m): the BAND ENSEMBLE (equal-weight mean of ALL band members, green) vs the single #1 (orange) vs buy-hold (black). The band-as-a-book is the robust deployable object; the #1 is regime-transient; both de-risk to cash in the crash. |
| `band_exposure_timeline.png` | The band ensemble's average EXPOSURE (time-in-market 0..1) over 2020. Exposure collapses toward 0 in the Feb-Mar crash (avg 17%/23%/38% in-market at 1d/4h/15m vs ~100% for buy-hold), then re-arms for the recovery -- the long-only preservation mechanism made explicit. |
| `band_width_by_matype.png` | Clean bar of avg band size per MA-type across TFs (out of 120). VIDYA widest (110), HMA/TEMA narrowest (79/76). |

(The original leaderboard charts `config_band_heatmap.png` + `rank_stability.png` remain alongside these.)

## BOTTOM LINE

The config-leaderboard headlines reproduce exactly under an independent re-run. The load-bearing positive --
the top 1d long-only MA configs beat no-cost buy-hold full-cycle (+248-270% vs +199%) **because** they exit to
CASH across the March -50% crash (avg 5-21% exposure, 37-50pp drawdown avoided) -- is a verified, causal,
exposure-to-cash mechanism, not a ranking artifact. The band is the robust object; the exact #1 is regime-noise
(median rho 0.57). All within STRICT long-only + spot, 2020 only.
