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

## §6 — Wave 2 results (VERIFIED, judged by overseer)

- **2B Funding-dispersion block-bootstrap → SURVIVES (edge is statistically REAL).** Circular block-bootstrap,
  BL swept 5–20 (conservative=20), pre-registered H0 (comp≤0), UNSEEN touched once. OOS p05 **+2.75%** / UNSEEN p05
  **+2.67%**, p(boot>0) 0.994–0.998. The Sharpe~5 was NOT 5×-inflated — the level is ~right (very-low-vol series);
  only the CI was wrong (post-deflation p50 Sharpe ~5.1–5.6). Dispersion-gate (threshold fit on SEL only):
  MARGINAL+ (OOS ann 10.05→11.25%, Sharpe 5.19→6.66; UNSEEN ~flat); does NOT rescue the 2023-25 dip (3.2→2.3% gated).
  n_eff OOS=51 / UNSEEN=34. Cheapest falsifier: the 2023-25 era at 3.2%/yr ⇒ honest forward expectation ~3–10%/yr with
  wide bands, not the 10–27% headline. **VERDICT: statistically real; the binding blocker stays the LO+spot constraint
  (market-neutral perp-short).** Script `src/mining/funding_dispersion_bootstrap.py`.
- **2D TI band-ensemble + real bears → de-risked beta CONFIRMED; PARK as an alpha source.** Worker corrected the spec
  (2024-H1 was +42% BULL, not a bear) and tested the REAL post-2022 bears: 2024-Jun→Oct (−26%) and 2025-Feb→May (−28%) —
  preservation REPLICATES out-of-period (+4 to +14pp). Hyperparam sweep: only 0–25% of lb×step cells positive in 2022 ⇒
  the (120,30) tier-A bear result was cherry-picked (inflation confirmed). Full-cycle under HP-averaging: compound
  +97–673%, p05_bootstrap +28–141% (solidly positive vs risk-free), Sharpe 1.32–1.58 > BH 1.15 — but lags BH compound
  3–20×. **VERDICT: real structural drawdown-insurance (de-risked beta), NOT alpha. Park TI×TF as an alpha source;
  deploy only as the core book's bear-insurance sleeve.** Script `src/strat/ti_wave2d_honest_band.py`.

- **2C S3 top-trader-ratio as a per-trade conditioner → NULL (with a high-value redirect).** Pre-registered on TRAIN
  (extreme-long top-traders → −4.84pp 20d fwd). On the trend book: contrarian direction REFUTED; pro-trend
  (`protrend_z<-1.0`) ekes +1.54pp OOS but raw p=0.068 FAILS Bonferroni(7) and fires on <2% of OOS entries (≈1 blocked
  trade) ⇒ not skill; global_lsr null (p=0.35). **ROOT CAUSE = n_eff mismatch:** the low-freq trend book makes only ~10
  actual entry events/asset in 9-mo OOS, so a per-trade gate is structurally under-powered (the feature discriminates
  arbitrary-bar fwd returns, not the specific entry bars the book sees). **REUSABLE REFEREE LESSON: never test a
  conditioner per-trade on a ~10-trade book — match the conditioner's cadence to a book with adequate n_eff.** REDIRECT
  (next wave): s3 as (a) cross-sectional asset-selector, (b) a modifier inside the daily-rebalanced funding-carry sleeve
  (high n_eff), (c) a global risk-on/off book-sizing overlay — NOT a per-trade skip gate.
  Scripts `src/strat/s3_{conditioner_test,conditioner_4h,gate_sweep,global_lsr_test}.py`.
- **Wave-3 `xex_` fix → DONE + committed (17e5520):** root cause was a registry name-collision (two sources emit the same
  3 cols → polars `_right` dupes), not a join/ffill limit. Removed redundant `cross_exchange_spreads` from
  `sources_to_join`; 0 unique features lost; takes effect on next chimera rebuild (deferred).
- **New correct-as-you-go finding (CDAP):** post-split, several `config/_invariants.yaml` globs point at moved/archived
  paths (`src/strategy/`, `scripts/wealth_bot/`, `src/analysis/`, `config/sleeves/`) → 12 rules report "target missing,
  NOT enforced" = silently vacuous. Queue a repath/retire pass so guards are honestly enforced (no green-washing CRITICALs).

- **Wave-3 CDAP invariant repath/retire → DONE.** 7 silently-vacuous rules fixed: 4 RETIRED with tombstones (gen5
  `src/strategy/sleeves` MA/EMA constraints, `stride_1_predictions`, `claim_contract_v12_import`, per-sleeve
  `lifecycle.yaml` — genuinely gone post-reset), 3 REPATHED to successors. WARN 12→8, vacuous 7→0, 0 CRITICAL, no rule
  broken. **2 GENUINE DRIFTS surfaced by honest re-enabling (NOT green-washed):** (i) `src/strat/oracle_walkforward.py`
  + `train_ma_walkforward.py` LACK DSR/CSCV deflation gating; (ii) `src/wealth_bot/bot/risk_manager.py` has kill-switches
  but NO IC-decay monitor / consecutive-DD halt. Both are pre-promotion apparatus gaps to close before any live ship.

Frontier status: Wave-1 ✅(5) · 2B ✅ · 2C ✅ · 2D ✅ · xex ✅ · CDAP-repath ✅ — all committed; **2A (A2+maker move-ride,
flagship) still running.** Next wave (queued): s3-redirect (cross-sectional/carry/regime) · hbr/te IC-0.13 conditioner
robustness · stbl_z30 + DVOL regime gates · conditional satellite sizing · [new] DSR-gate + decay-monitor wiring.

## §7 — FLAGSHIP VERDICT: move-riding (2A) — the direct test of the user's thesis

**Move-riding in its strongest realizable internal-data form is NET DEAD on UNSEEN — but the kill mechanism is precise
and one execution lever remains.** Harness `src/mining/mover_ride_a2_maker.py`; artifact
`runs/mining/mover_ride_a2_maker_u10_20260618_002718.json`. Design = +1.5% onset (1m, u10, 2021-2026); A2 directional-
ceiling gate (refit on TRAIN, VAL AUC **0.646** — independently reproduces 0.648); **HONEST passive-maker fill** (bid 0.8%
below price → realized p_fill 0.34, so you only keep the ~1/3 of movers that dip back to you); swept exits; B1 sizing; vs
**random-entry-on-the-SAME-movers** null; TRAIN/VAL-select → UNSEEN scored once.
- TRAIN+VAL selected (ft_3_2 | A2 top-20% | B1): beat-null **+0.40%/ev**, breadth 9/10, p05 +0.13% — looked alive in-sample.
- **UNSEEN (n=77, once): beat-null −0.46%/ev; honest seed-averaged ≈ −0.04 to −0.08%/ev (≈0, not positive). GATE FAILS 3/3.**
- **A2's RANKING signal is DURABLE** — UNSEEN gross rises monotonically with gate tightness (top-50% −0.10% → top-20%
  +0.08% → top-10% +0.25%; win 39→48%). The 0.648 AUC was NOT a fit-period artifact.
- **But A2 CALIBRATION decays** (P(continue) 37.7%→30%→26% across TRAIN→VAL→UNSEEN; choppier 2024+ regime), and the
  **kill mechanism is the honest maker fill: it adversely-selects AGAINST the fast-up continuation legs A2 picks** (they
  run away and never fill), while the median ride is −1.3 to −1.6%. The thin, fat-tail-carried positive mean does not
  survive to beat random-entry.
- **Closes Wave-1's open door:** "the meat exists at 1m" = TRUE; "you can KEEP it (long-only, internal data, honest
  cost+fill)" = FALSE. Two un-killed levers: (1) a HYBRID passive+marketable-limit fill that catches runners at some
  slippage (A2 ranking is real enough to justify ONE bounded test → Wave-4A); (2) external whale/basis/orderbook data.
  Do NOT chase the top-10% cell (n=42, inside-noise, post-hoc).

**ANSWER to "isn't that a win?": No — not in its strongest internal-data form.** The signal that *identifies* the move
(A2) is real and durable, but the execution you NEED to afford it (maker, to beat the taker cost that killed naive
riding) preferentially misses the exact legs the signal selects for — **signal real, capture self-defeating at honest
fill.** One execution lever (hybrid fill) and external data remain before final closure.

Wave-4 dispatched: 4A hybrid-fill move-ride closer · 4B s3 cross-sectional/regime (the 2C redirect, n_eff-adequate) ·
4C hbr/te/stbl/DVOL as a regime overlay (n_eff-adequate, with the IC-0.13 robustness probe first).

## §8 — Wave 4 results

- **4C price/flow/vol/stablecoin conditioners (hbr_eta_xratio, te_in/te_out, stbl_z30, DVOL) as regime overlays → ALL NULL.**
  Rigorous: rolling-window IC + regime-split + shuffled-conditioner null + block-bootstrap; n_eff-adequate (daily, n_eff
  300-580, NOT per-trade — the 2C lesson applied); UNSEEN once.
  - **hbr/te: REGIME-ARTIFACT** — the Wave-1 "IC ~0.13" is REFUTED as a robust signal: pre-registered (negative) direction
    held in OOS then SIGN-FLIPPED in UNSEEN; cross-sectional t never p<0.05 held-out. A single-window (mid-2025) effect.
  - **stbl_z30: NULL (macro-regime-conditional)** — IC real per-bar (the IC-as-objective trap) but the book-level return
    differential REVERSES sign bull→bear (OOS +127pp / UNSEEN −27pp). Possible future use: an INPUT to the regime
    classifier, not a standalone gate.
  - **DVOL: STRUCTURALLY UNUSABLE** — non-stationary (vol-compression since 2022) ⇒ a static threshold fires ~100% in
    OOS/UNSEEN = zero discrimination; rolling-normalized still IC≈0 in UNSEEN.
  - **None promotes** — closes the hbr/te/stbl/dvol "embedded opportunity" candidates from §3. Script
    `src/strat/wave4c_regime_overlay.py`. (Apparatus killed its own Wave-1 false positive — good.)

- **4B s3 cross-sectional tilt + global overlay (n_eff-adequate: 5,820 XS decisions) → NOT book-eligible standalone, but
  REAL XS skill with a natural home.** Shuffled-conditioner null + timing-vs-exposure decomposition, UNSEEN once.
  - **Cross-sectional tilt: TIMING SKILL IS REAL** — beats the shuffled-null p=0.000 on BOTH OOS and UNSEEN
    (Bonferroni-passing; genuine info in `top_pos_lsr` for ordering assets) — **but UNDERPERFORMS EW baseline −4.44pp OOS**:
    the tilt sacrifices diversification (eff-N 20→14.6) and the concentration cost outweighs the discrimination in a
    long-only EW basket where EW is already near-optimal.
  - **Global overlay: exposure-reduction only** (+4.71pp vs EW but the shuffled null matches it, p=0.36). Combined: cancels.
  - **NON-DEAD RESIDUAL:** s3's real XS discrimination's natural habitat is the FUNDING-CARRY sleeve (already an
    unequal-weight cross-sectional in/out decision where diversification doesn't auto-resolve the pick) → a `carry+LSR-tilt`
    Wave-5 hypothesis. Reinforces the theme: **internal signals show REAL discrimination but aren't harvestable long-only
    once cost + diversification are honest.** Script `src/strat/s3_wave4b_redirect.py`.

- **4A hybrid-fill move-ride closer → INTERNAL-DATA LONG-ONLY MOVE-RIDING IS DEFINITIVELY CLOSED.** Hybrid fill (passive
  bid + marketable fallback to CATCH the runners — the one lever 2A left), 1024 configs swept, A2 gate, arbiter null with
  identical fill, TRAIN/VAL→UNSEEN once. TRAIN+VAL looked strongly alive (beat-null **+0.78%/ev**, 10/10, stable top-15
  cluster) → **UNSEEN beat-null −0.135%/ev, p05 −0.465, breadth 3/9, GATE FAILS 3/3.** Three converging kills: (1) fill-mix
  degrades in the choppier 2024+ regime (passive 34→21%, so 79% of UNSEEN fills are marketable = taker+slip on exactly the
  legs needing gross); (2) A2 calibration decays (base-rate 37.7→30→26%); (3) the marketable gross (~+0.05%) doesn't cover
  its ~34bps cost; (4) TRVAL→UNSEEN reversal = best-of-1024 overfit. **Complete kill chain: unconditional(taker) →
  A2-passive-maker → A2-hybrid-fill — all dead UNSEEN.** A2's RANKING signal is real but UN-HARVESTABLE long-only at honest
  cost+fill: every affordable execution adversely-selects against the legs the signal picks. Out-of-scope levers: external
  whale/basis/orderbook data; maker-only sub-minute (data not in pipeline). Script `src/mining/mover_ride_hybrid_fill.py`.

## §9 — CAMPAIGN SYNTHESIS (interim, post kill-sweep)

The internal-data, long-only, directional / move-capture frontier is **freshly RE-EARNED as exhausted** — not inherited.
Every candidate this campaign tested shows the SAME shape: **real discrimination, un-harvestable under honest long-only +
cost + diversification.** A2 move-ranking (durable AUC 0.646) — killed by maker adverse-selection. s3 cross-sectional
positioning (real, p=0.000 vs random concentration) — beaten by free EW diversification. hbr/te/stbl/dvol — regime-
artifacts / IC-as-objective traps. TI×TF — de-risked beta. The ONLY held-out-positive edge (funding-dispersion carry,
deflation-survived) is market-neutral ⇒ LO-blocked. **The earned value sits in: (a) the de-risked-beta + carry BOOK
(confirmed: core_satellite ~Sharpe 1.6 / maxDD −33%, satellite corr +0.015); (b) the external-data door (untested, needs
spend); (c) the carry sleeve as the natural high-n_eff home for the real cross-sectional signals (s3-tilt Wave-5).**
Reusable referee lessons banked: match conditioner cadence to n_eff; honest fill model exposes adverse-selection; IC-
significance ≠ harvestable (regime-conditional sign-flips). PIVOT: kill-sweep → CONSTRUCTIVE build of (a).

- **4D WM-as-regime-gate (existing V1.1 f41, inference only) → WM signal REAL but cheap SMA dominates; don't replace.**
  Checkpoint LOADS clean (0 broken keys, 41 feats align, ShIC/IC gate PASS) ⇒ the WM→strat wiring is EXECUTABLE, not broken
  (ForecastBundle's 0 callers = unexecuted, not defective). The WM regime signal has GENUINE transition info (beats shuffled
  null +21.9pp OOS). BUT OOS WM-gate +14.2% vs SMA +2.1% is suspect (maxDD regresses 6.7pp + OOS likely overlaps the WM's
  2026-05 training window); **clean held-out UNSEEN: WM-gate −24.2% vs SMA −6.2% (WM_WORSE −18pp)** — the adaptive threshold
  normalizes away the sustained-bear signal the cheap SMA (price<SMA100 level) catches. **Verdict: do NOT wire WM as a
  regime-gate REPLACEMENT; the cheap SMA wins on the clean held-out. Safe COMPLEMENT = SMA+WM min-scalar (UNSEEN=SMA, OOS
  maxDD −15.2%) — low-priority.** Confirms 'WM deprioritized / diagnostic'. Script `src/strat/wave4d_wm_regime_gate_probe.py`.

## §10 — KILL-SWEEP / VALIDATION PHASE COMPLETE (14 items) → PIVOT to CONSTRUCTIVE + breadth

Every load-bearing claim re-validated, both framings tested, the layers + chimera audited, apparatus weaknesses fixed
(xex_, CDAP), and my own false positives killed. The earned deliverable under the binding LO+spot constraint is the
**de-risked-beta TREND book** (the funding-carry satellite is market-neutral ⇒ LO-blocked, off the table like shorts).
Wave 5 dispatched: **A** build+validate the deployable LO de-risked-beta book through the full gate chain (the capstone) ·
**B** archive deep-dive for genuinely-missed insights (the user flagged the archived section) · **C** alt-bar-type setup
probe (volume/imbalance/range bars — the "sweep all chart-types" axis untouched tonight; all prior work was time/dollar bars).
