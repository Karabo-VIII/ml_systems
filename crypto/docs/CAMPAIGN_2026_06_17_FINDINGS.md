# Crypto re-examination campaign — findings ledger (2026-06-17)

Autonomous /orc campaign. VERIFIED anchor 2026-06-17 23:16 +02; target window 7h (→06:16), hard cap 9h (→08:16).
Fresh perspective; both framings (internal-data MOVE-IDENTIFICATION **and** move-RIDING-as-expectancy) held as competing
live hypotheses per user mandate — neither dogmatized. All entries below are RWYB (agent ran the real command/artifact;
load-bearing ones are re-verified by Wave 2's fresh runs).

## §1 — Headline-claim revalidation (Wave 1)

| Claim | Verdict | Fresh evidence | Conditions |
|---|---|---|---|
| Regime-managed beta (`entry_signal_lab`): "UNSEEN −0.3% vs basket −18%" | **CONDITIONAL** (preservation real, number STALE) | fresh UNSEEN −0.9% vs −8.1%; maxDD −2.3% vs −40.2% (37.9pp); held-out p05 −38.8% | drawdown-insurance NOT alpha; per-asset 0/10 beat beta in current bear |
| `daily_engine`/`core_satellite_book` "deployable book, 5/6 gauntlet, fragile p05" | **CONFIRMED** | full-cycle CORE +1969%, Sharpe 1.40, maxDD −48.2%; CAP_70/30 Sharpe 1.64 maxDD −32.6%; held-out p05 −35.99% (disclosed); recent 2024-26 Sharpe 0.76 vs BH 0.62 @ half DD; core/sat corr +0.015 | taker, u10, canonical split |
| Funding-dispersion carry "+7.9% OOS / +10.3% UNSEEN, beta~0" | **CONFIRMED** (but un-deflated) | OOS +7.93%, UNSEEN +10.31%, beta −0.16/−0.04, null p=0.000 | **MARKET-NEUTRAL (violates LO+spot)**; block-bootstrap NOT run → Sharpe~5 autocorr-inflated (Wave-2 gate) |
| "No active alpha at daily/4h LO+spot = de-risked beta" | **CONFIRMED** w/ precise conditions | positive_control PASSES (apparatus has power); all entry families = BETA-IN-DISGUISE on held-out; best OOS_xs 7-11pp(1d)/5-7pp(4h) but rank-unstable, don't reach UNSEEN | taker, u10, 1d+4h only |
| ML refutations (regime→config collapse; meta-labeler AUC 0.52; ML<manual) | **CONFIRMED** | config_selector n_map_differentiates 0-1/4; mover_metalabel OOS 0.521; ML −3.38% vs manual −1.14% | — |
| Grid trading KILLED (D70); long-only funding tilt | **CONFIRMED dead** | grid 0/10 seeds even at p_fill=1; LO funding tilt 0/12 p05 | carry lives in short leg (LO-inaccessible) |
| WM cohort "world-class forecasters" | **CONDITIONAL (projected, not measured)** | V1.1 f41 checkpoints exist; NO held-out compound-objective run; ForecastBundle contract complete but ZERO callers | "world-class" is architectural projection pending compute |

## §2 — Market-model CORRECTIONS (Wave 1 changed these)

1. **Resolution-cliff claim REFUTED.** Derivatives/liquidation/whale/funding/etf families are **0% null on dollar bars**
   (asof-joined + forward-filled at build), not "83-100% null." Only `xex_` (cross-exchange spread) is ~90% null on
   dollar bars — and that's a **join/ffill BUG**, not a resolution limit (the data is ~90% present on 1d). High-EV fix.
2. **Feature count:** 218 features across **23 families** (docs say "12 families / ~27 eff dims" — undercounts families;
   measured effective rank of the norm_ layer ≈ 17, so ~27 is optimistic).
3. **D72 ("mover continuation AUC 0.52, no internal info") was FRAMING-LIMITED → partially falsified.** The 0.52 used a
   composite *trade-economics* label. Decomposed (artifact `mover_continuation_u10_20260613`):
   - **DIRECTION dead:** A1 OOS AUC 0.498 (below null) — confirms D55.
   - **MAGNITUDE alive:** B1 OOS AUC **0.731** (beats shuffled null by 0.17); drivers = `pre_vol`, `run_accel`,
     `dayvol_ratio` (volatility-state, not directional) → vol-clustering, monetizable only via sizing (sign-agnostic).
   - **Directional-CEILING alive:** A2 ("reaches +1.5% further beyond onset") OOS AUC **0.648** (clears the 0.58 spec).
     This is the genuinely-new open door.
   - **But unconditional RIDE economics DEAD at taker:** all 9 mover_ride cells net-negative; the rider *underperforms*
     random-entry-on-same-movers (24bps RT ≈ 1.5-2× the ~0% mean gross). "Meat exists" (oracle ceiling +5.3% median
     remaining at +1.5% trigger @1m → D67 "too late" refuted at 1m), but you can't keep it unconditionally at taker.
4. **TI×TF tier-A is INFLATED.** Band-FILTER mechanism (go-to-cash in bear) is real & structural; but the rolling-PICK
   adds ZERO over random-in-band (T1 p=0.12-0.46); tier-A depends on a (120,30) lookback cherry-picked on the lone 2022
   bear (57% of hyperparam combos negative there); TSI/RSI fragile (1/12 combos positive → drop). 2023 OOS: 6/6 positive
   but 0/6 beat BH (~53% of BH) = de-risked beta again. n_eff≈1 for one bear year (can't detect a +5%/yr bear edge).

## §3 — Embedded-opportunity / missed-insight register (EV-ranked)

1. **[Wave-2A] A2 directional-ceiling (AUC 0.648) + MAKER fills + the move-riding frame.** The strongest realizable form
   of the user's idea: gate confirmed-mover rides by A2, exit by policy, maker p_fill 0.25-0.40 (~12bps, halves the
   hurdle). Untested. The honest "is move-riding a win?" test. Long-only, no shorts, no external data.
2. **[Wave-2B] Funding-dispersion block-bootstrap deflation + dispersion-gate.** Gate the only confirmed edge (is Sharpe~5
   real?), and test a cross-sectional-funding-std gate to rescue the 2023-25 dip (3.2%/yr vs 11.1%).
3. **[Wave-2C] s3 `top_pos_lsr` / `top_trader_ratio` as a conditioner.** On disk (79k rows, 73 assets, 2022-2026), covers
   OOS+UNSEEN, free, NEVER backtested against any surviving strategy. Highest-EV untouched frontier feature.
4. **[Wave-2D] TI band-ENSEMBLE (drop rolling-pick + TSI/RSI) + 2024-H1 second-bear test.** Honest deployable form.
5. **`xex_`→dollar-bar join bug fix** — unlocks a microstructure signal on the primary training resolution.
6. **`xrel_hbr_eta_total_xratio` + `te_in/te_out`** show OOS/UNSEEN IC ~0.13 across BTC/ETH/SOL (flat in OOS, strong in
   UNSEEN). Caveat: IC is a diagnostic only + period-dependent (likely a mid-2025 regime) → robustness probe as a
   *conditioner*, not a standalone signal.
7. **`stbl_z30` (stablecoin-mint z) + Deribit DVOL** as regime gates — both on disk/wired, zero tests.
8. **Dead-feature cleanup:** `xd_funding_spread`(const), `te_*_btc`(std0), 3× `xpct10`(single-value),
   `norm_funding_momentum`(std .079); `bd_` frozen since 2025-05 (stale ffills); LOB only post-2026-01.

## §4 — Re-ranked frontier

- **Wave 2 (dispatched, background):** A (A2+maker move-ride test), B (funding block-bootstrap + dispersion-gate),
  C (s3 top_pos_lsr conditioner), D (TI band-ensemble + 2024-H1 bear).
- **Wave 3 queue:** xex_ join fix; hbr/te conditioner robustness; stbl_z30 + DVOL regime gates; conditional satellite
  sizing (scale satellite when core in regime=chop — free Sharpe from the +0.015 corr); WM f41 held-out run (compute).
- **Dead — do not re-run:** grid; LO funding tilt; mover directional meta-labeler; regime→config ML; raw-direction
  prediction at any internal TF; any short-side framing (binding constraint).

## §5 — Standing caveats (integrity)
- UNSEEN is touched once per candidate; the move-ride test fits TRAIN/VAL, evaluates UNSEEN once.
- IC-0.13 conditioner findings are period-dependent — treat as conditioner hypotheses, not edges, until robustness-probed.
- The beat-random-entry-on-same-movers null is the arbiter for any mover strategy (it is what killed naive riding).
- Maker p_fill ∈ [0.25,0.40], never 0.80; entering mid-move is a chase — model the fill penalty honestly.
