# Avenue Execution Specs (FOUNDATION — for the solving phase; conclusions held OPEN)

> **Reading instructions:** each spec below is a RECIPE for the solving phase — not a result, not a verdict.
> The gate chain is universal and stated in full in the preamble. Every "MECHANISM HYPOTHESIS" is a testable
> claim, not a given. "HELD OPEN" means exactly that: no avenue is pre-decided dead or alive.
> Source truth: `docs/AVENUE_MAP_2026_06_04.md` (statuses), `docs/APPARATUS_LOCKDOWN_SPEC_2026_06_04.md`
> (gate implementations), `docs/RETEST_PLAN_2026_06_04.md` (methodology).
> Companion (apparatus home): `src/strat/README.md` (the turnkey gate the tools below now live in).

---

## Preamble: universal gate chain (every avenue, every spec)

Every candidate must traverse this chain IN ORDER. No step may be skipped. No reordering. No touching
held-out before the chain is complete for that candidate.

```
STEP 0  — Phase-0 apparatus check: confirm LD-1 FillModel (taker 0.24% default), LD-2 leak-probe,
            LD-3 DSR manifest, LD-4 random-entry null, LD-5 bear-inclusive holdout are in place.
            Gate: if any LD is missing or BROKEN, STOP and complete APPARATUS_LOCKDOWN_SPEC first.

STEP 1  — Cost-honest backtest (harness.py, FillModel mode=taker, cost_rt=0.0024, p_fill=1.0).
            Also run maker_pessimistic (p_fill=0.30, adverse_selection=0.96) for sensitivity.
            Fail criterion: median compound at taker cost negative on held-out.

STEP 2  — Shift-sensitivity leak-probe (leak_probe.py — use relative_leak_test vs a known-clean twin;
            cadence-robust). Verdict must be PAST_ONLY_OK (ratio < 2.0). Absolute-pp shift verdict is
            ADVISORY-only on coarse bars (see leak_probe docstring CALIBRATION FINDING).

STEP 3  — Cost-matched random-entry null firewall (src/strat/firewall.py). Candidate must beat the null
            p95 per window AND be absolute-positive on held-out. Beats-null is NECESSARY, NOT sufficient.

STEP 4  — Robustness battery (src/strat/battery.py):
            block-bootstrap p05 > 0, jackknife jk2 > 0 AND jk3 > 0, n_eff >= 8 (floor; 15 preferred),
            maxDD < 20% (binding gate; 30% is the published ceiling but 20% is our floor).
            (block-bootstrap p05 is the seed-equivalent for a DETERMINISTIC rule strategy; a
            STOCHASTIC/ML candidate must add a 10-seed outer loop to claim the "10/10 seeds" bar.)

STEP 5  — Benchmark-excess per regime (src/strat/benchmark.py:benchmark_excess): beat the beta-matched
            costless static hold (the asset at the candidate's own time-in-market fraction f) IN EACH
            held-out window including any bear window. Wired into evaluate_candidate as a hard ship gate
            (beats_beta_held). In bull/chop the bar is buy-and-hold; in bear it is capital-preservation.

STEP 6  — DSR/Holm at TRUE family-N: the _sweep_manifest.json sidecar must declare n_variants_tested
            (includes aggregation DoF); n_trials = max(written, manifest); severity="critical" for ship
            claims that fail Holm. Gate file: src/audit/check_dsr_holm.py (VERIFIED in-tree).

STEP 7  — Only after Steps 0-6 pass: decouple -> combine (pre-register pooling weights before
            touching held-out). Sizing, paper, live are downstream of this chain, not part of it.
```

---

## AVENUE-1: Information-driven bar chimeras (dib / runs_tick / runs_volume / range / adaptive_vol)

**AVENUE_MAP status:** dib = PARTIAL (BTC/ETH/PEPE chimera only, 3/379 raw assets); range = PARTIAL (BTC/ETH/PEPE chimera only); runs_tick = UNEXPLORED (chimera EMPTY, 0/379); runs_volume = UNEXPLORED/BROKEN (chimera EMPTY; 77 raw assets, 2023-only calibration); adaptive_vol = UNEXPLORED (chimera EMPTY, 87 raw assets). VERIFIED: raw bar counts from `data/processed/bars/{dib,range,runs_tick,runs_volume,adaptive_vol}/` confirmed 379, 379, 379, 77, 87 respectively; chimera confirmed via `data/processed/chimera/{dib,range}/` (3 assets each), runs_tick/runs_volume/adaptive_vol chimera dirs EMPTY.

### 1. MECHANISM HYPOTHESIS

Information-driven bars (dollar-imbalance, run, range, vol-scaled) sample when something economically meaningful happens — a burst of directional flow, a volatility expansion, a volume event — rather than at arbitrary clock ticks. The hypothesis is that SELECTIVE ENTRY on events detected by an information-driven bar clock creates a time-clock illusion in which the "bar" already encodes informed order-flow or volatility structure that is absent from a uniform time bar. For `dib` specifically: a long sequence of same-side imbalance bars signals directional informed flow (consistent with Easley-O'Hara information asymmetry theory); entering at the start of such a sequence on a big-move asset (e.g. high-vol memecoin) and exiting after a fixed bar count or time-stop should capture the informed drift before it mean-reverts or resolves. For `range` bars: a regime of large range bars signals volatility expansion; range bars offer cleaner trend/breakout reads because each bar represents a uniform price excursion. For `adaptive_vol` (~50 bars/day on average, VERIFIED in AVENUE_MAP): the vol-scaled bar adjusts bar size to realized volatility, creating stationarity in bar-level returns that time bars lack. These mechanisms are theoretically motivated and structurally different from the uniform time-bar substrate where prior mining failed.

### 2. PRE-REGISTERED FALSIFIER

On the first three chimera-enriched assets per bar type (BTC, ETH, PEPE — already materialized for dib and range), run the firewall test: if the cost-matched random-entry null beats the best selective setup on ANY two of {TRAIN, OOS, UNSEEN} across both dib and range, the mechanism is BETA-IN-DISGUISE on the available substrate and the chimera build-out across u100 is suspended. Additional kill trigger: if the lag-1 autocorrelation of dib-bar returns on BTC is not significantly different from the dollar-bar lag-1 (-0.014, VERIFIED in FOUNDATION §17), the bar type adds no new structural information. The runs_volume avenue is separately killed if calibration cannot be fixed to cover 2020-2026 (the 2023-only failure is a data-quality issue, not a mechanism issue).

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
1. make_chimera_bars.py  -- build dib/range/runs_tick/adaptive_vol chimeras across u100
                            (BTC/ETH/PEPE already done; expand to universe)
                            Command template: python src/pipeline/make_chimera_bars.py
                              --assets <u50_symbols> --bar-types dib range runs_tick adaptive_vol
                            CAUTION: runs_volume needs calibration fix before build (2023-only)
2. src/strat/discover.py:discriminate  -- on BTC/ETH/PEPE dib and range first (cheapest, already
                                   chimera-built) identify which features DISCRIMINATE forward returns
                                   on the alt-bar cadence; produce candidate GATE list
3. src/strat/discover.py:scan  -- sweep (asset x TI-config x gate) grid on dib and range BTC/ETH/PEPE;
                          output per-cell verdict; set family_n = total cells swept (manifest sidecar)
4. src/strat/candidate_gate.py:evaluate_candidate  -- pipe surviving cells through full gate chain
5. src/strat/battery.py:evaluate  -- robustness battery on any cell that clears Steps 1-4
6. src/strat/fill_model.py:apply_fill_model  -- sensitivity: taker vs maker_pessimistic on any survivor
```

### 4. DATA NEEDED

- **BUILD BACKLOG (re-runnable inventory: `runs/staging/probe_avenue1_chimera_inventory_2026_06_05.py`, ran 2026-06-05):** dib = 87 raw / 3 chimera (BTC,ETH,PEPE) → **84 to build**; range = 87 / 3 → **84**; runs_tick = 87 / 0 (chimera dir EMPTY) → **87**; adaptive_vol = 87 / 0 → **87**; runs_volume = 77 / 0 → **77** (2023-only, fix calib first). Data-hygiene flag: BOTH `data/processed/bars/runs_vol/` AND `runs_volume/` exist (duplicate/ambiguous — resolve before building). This supersedes the transient counts (G-10).
- dib chimera: BTC/ETH/PEPE MATERIALIZED (VERIFIED); u50 expansion MUST BE BUILT via `make_chimera_bars.py`. VERIFIED builder exists: `src/pipeline/make_chimera_bars.py`. Dry-run proof exists per AVENUE_MAP note (2026-06-05).
- range chimera: same status and build path as dib.
- runs_tick chimera: raw bars exist for 379 assets (VERIFIED) but chimera EMPTY; build via `make_chimera_bars.py --bar-types runs_tick`. No calibration issue flagged.
- runs_volume chimera: raw bars ONLY 77 assets, 2023-only (VERIFIED). Must fix calibration before build; characterize threshold drift and extend to 2020-2026 BEFORE chimera build. Flag as BLOCKED until calibration fixed.
- adaptive_vol chimera: raw exists 87 assets (VERIFIED); chimera EMPTY; build via `make_chimera_bars.py --bar-types adaptive_vol`.
- Chimera enrichment inherits the daily frontier lag (+1d, causal targets) from the dollar-chimera machinery per AVENUE_MAP proof note.
- Bear-inclusive holdout: OOS+UNSEEN must span a bear episode; 2022 bear is in TRAIN; 2026 alt-bear confirmed in UNSEEN (FOUNDATION §10).

### 5. EV RATIONALE

Highest priority avenue for two reasons. First, this is the ONLY structurally unexplored substrate: it was never enriched and never mined under any apparatus — so the dead-list does not apply. Second, the build cost is mechanical and de-risked (dry-run proof; builder exists). The solve-phase cost is chimera build (compute-heavy but known) + scan on 3 existing chimeras first (cheap). If the falsifier triggers on BTC/ETH/PEPE, the universe build is skipped. AVENUE_MAP ranks this #1. SOTA caveats (FOUNDATION §7-E): info-driven bars give marginal stationarity gains and are operationally unstable at high bar counts — the expected uplift is SELECTIVITY (event-clocked entry), not a stationarity miracle. The cost-clearing question for high-cadence alt bars is the central risk.

### 6. EXPECTED FAILURE MODE

The most likely death: bar-level taker cost (0.24%) still exceeds the selective-entry capture distribution on all but the most volatile assets at the finest cadences (FOUNDATION §17 established this for dollar bars on majors). The dib bars at ~200/day would require a per-bar expected return of >0.24% net — equivalent to the same cost-wall that killed 1h/15m time bars. The saving hypothesis is that information-driven bars by construction select high-flow moments where moves are larger; but this must be verified, not assumed. If it fails the firewall, the second-order failure is that dib/range on memecoins (where PEPE showed fuel) is a bid-ask-bounce microstructure trap identical to the fine-dollar finding (FOUNDATION §17). Report both failure modes explicitly if triggered.

---

## AVENUE-2: The 184-feature gate space (s3_smart_vs_retail, hbr_eta_*, liq_*, macro etf_/stbl_)

**AVENUE_MAP status:** UNEXPLORED as standalone GATE avenue. PARTIAL for whale (`wh_*`) on PEPE only. The `liq_delta_z30` concrete lead surfaced by `discriminate` on PEPE at H=4 is a live candidate (same-sign across 4 windows AND beats shuffle-null on UNSEEN). VERIFIED feature columns in 1d chimera: `s3_smart_vs_retail`, `s3_smart_vs_retail_z`, `s3_smart_bullish`, `s3_smart_bearish`, `s3_smart_extreme_long`, `s3_smart_extreme_short`, `hbr_eta_buy`, `hbr_eta_sell`, `hbr_eta_imbalance`, `liq_delta_z30`, `liq_capitulation`, `liq_short_panic`, `liq_long_spike`, `liq_short_spike`, `etf_btc_etf_inflow_shock`, `etf_btc_etf_outflow_shock`, `etf_any_inflow_shock`, `stbl_stable_shock`, `stbl_stable_crash`, `stbl_compound_shock`, `fund_rate_z30`, `fund_sign_flip`. All VERIFIED via ChimeraLoader.

### 1. MECHANISM HYPOTHESIS

If an edge exists in this universe under LO+spot+lev=1, it is most likely to live in an EXOGENOUS CONDITIONING SIGNAL that identifies when a particular asset is about to be driven by informed, concentrated, or structurally-forced flow — rather than in price-TI structure alone. The evidence for this framing: (a) the PEPE whale gate showed a real discriminator (mechanism falsifier passed cleanly in FOUNDATION §20), even though it was PEPE-idiosyncratic and thin; (b) the `s3_smart_vs_retail` composite is the closest analogue at scale — it aggregates top-account long/short ratios vs global ratios into a "smart money vs retail" divergence signal that is economically grounded (informed insiders position before moves). `hbr_eta_imbalance` (Hawkes buy/sell asymmetry) captures self-exciting order-flow where one side has temporarily dominated — a momentum in ARRIVAL RATES rather than prices. `liq_delta_z30` (liquidation delta normalized to 30d z-score) captures forced-exit flow: a large asymmetric liquidation event (shorts being forced out) structurally removes selling pressure and can precede a bounce. `etf_btc_etf_inflow_shock` and `stbl_compound_shock` are macro risk-on signals: institutional buying of ETF products or stablecoin creation signals capital entering the ecosystem that tends to lift systematic exposure. The testable version: a slow structural TI (e.g. norm_ma_distance as the trend frame) filtered by any one of these gates should outperform the bare TI in terms of setup precision — because the gate selects setups where the underlying flow mechanism is active.

### 2. PRE-REGISTERED FALSIFIER

Run `discriminate` on BTC, ETH, PEPE, and DOGE across the full gate family (all s3_/hbr_/liq_/etf_/stbl_/fund_ columns). If fewer than 3 distinct gate features show same-sign-across-4-windows AND beat the shuffle-null base rate (0.125 per feature, so N*0.125 is the chance count), the gate family has no discrimination signal and the avenue is dead before backtest. For `liq_delta_z30` specifically (the live lead): if on a taker-cost backtest with a pre-registered entry rule (e.g. enter when liq_delta_z30 > +1.0 after a norm_ma_distance setup) it does NOT beat the random-entry null on UNSEEN for PEPE, the lead is DEAD (discrimination is not harvestability — already confirmed in the FOUNDATION).

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
1. src/strat/discover.py:discriminate  -- on the full gate family across BTC/ETH/PEPE/DOGE on 1d
                                   (cheapest; chimera materialized). Produces: ranked candidate_gate
                                   list with shuffle-null comparison
2. src/strat/discover.py:scan  -- for each surviving gate: sweep a grid of (gate_threshold x
                          TI_structure x asset) on TRAIN+VAL; log family_n in manifest; top 3/gate
3. src/strat/candidate_gate.py  -- full gate chain on each cell
4. src/strat/battery.py  -- robustness battery on any cell clearing Steps 1-4
5. src/strat/fill_model.py  -- taker + maker_pessimistic sensitivity on any survivor
```

Note on coverage: `dv_dvol_*` is BTC/ETH only; `xex_*` is 5 assets (BTC/ETH/SOL/XRP/DOGE). Do not scan these on alts — they will return NaN. `xrel_liq_long_usd_xpct10` and related cross-sectional relatives are available for all u100 assets. Cover-gap flag: `soc_wiki_views` is nearly empty (FOUNDATION §7-B) — skip entirely.

### 4. DATA NEEDED

1d chimera with the full feature set is MATERIALIZED for the full universe (VERIFIED via ChimeraLoader). No build required. **DATA-READINESS (re-runnable: `runs/staging/probe_gate_data_readiness_2026_06_05.py`, ran 2026-06-05):** across BTC/ETH/PEPE/DOGE 1d, the lead `liq_delta_z30` has 99–100% non-null coverage (fully testable); `hbr_eta_imbalance`, `norm_*`, `wh_whale_net_usd`, `hurst_regime` are ~100% everywhere; `bs_basis_z30` ~95–99%. **BUT `s3_smart_vs_retail` and `s3_top_pos_lsr` are 0% on PEPE (ABSENT) and only ~57% on BTC/ETH, ~63% on DOGE** — the s3 "smart-money" gate (the closest analog to a real mechanism) is NOT testable on PEPE and is thin on majors. Test a gate on an asset ONLY where coverage is high; treat absent/low-coverage cells as non-testable (avoid NaN-driven false reads). Look-ahead flags to honor: `xd_btc_return` and `xd_btc_volatility` are SAME-BAR pass-throughs at intraday cadences (safe on 1d, but must shift +1 if testing on 4h or sub-daily). `bd_`, `te_`, `hbr_`, `lob_`, `mv_` daily-silver cols need a +1d lag in live (available end-of-day only) — the harness handles this in fill-at-next-open mode, which is already its design. Bear-inclusive holdout: the 2026 ALT BEAR is in UNSEEN (VERIFIED FOUNDATION §10); use it.

### 5. EV RATIONALE

Second priority because: (a) data is already materialized (zero build cost, cheapest avenue to start); (b) the `liq_delta_z30` concrete lead from the discriminator is the ONLY extant live candidate from the foundation phase, so it must be followed before any other cell is opened; (c) the `s3_smart_vs_retail` feature is structurally the most defensible gate (aggregates multi-account positioning divergence, economically grounded, cross-sectional z-scored). AVENUE_MAP ranks gate-space second. The cost of a null result here is low — one discrimination pass is cheap.

### 6. EXPECTED FAILURE MODE

The most likely death: discrimination does not survive into harvestability. The PEPE whale gate showed this pattern exactly (real discriminator, failed the firewall on OOS per FOUNDATION §20). The `s3_` and `hbr_` data is perp-native (funded contract data) — there may be an adverse-selection in which the gate fires AFTER the move has already embedded in the price (e.g. `s3_top_pos_lsr_z` is only available end-of-day; the move happens intraday). The `liq_*` data has the same daily availability constraint. Second failure mode: gate signals are asset-specific like the whale gate (PEPE-only, inverts on DOGE — confirmed in D18). Pre-registering cross-asset pooling BEFORE testing is required to avoid the multiple-comparisons ghost of testing each asset independently and reporting the winner.

---

## AVENUE-3: ML meta-labeling on a battery-surviving gate

**AVENUE_MAP status:** UNEXPLORED in defensible role (meta-labeler on a proven gate); ML-as-generator is dead (D9). Archive references: `archive/analysis/meta_labeler.py` (VERIFIED), `src/training/train_meta_labeler.py` (VERIFIED), prior bakeoff archived at `archive/analysis/meta_labeler_bakeoff.py` (VERIFIED). Prior result: AUC 0.495–0.505 = null (FOUNDATION dead-list D9) — but that ran ML as a GENERATOR, not a meta-labeler on a proven first-principles gate.

### 1. MECHANISM HYPOTHESIS

A first-principles gate (e.g. `liq_delta_z30` > threshold triggers a setup) defines a BINARY candidate event: this bar is or is not a setup entry. The gate is parsimonious and avoids overfit by having a single degree of freedom. ML meta-labeling (Lopez de Prado framework) asks a second question: given that the primary gate fires, is this SPECIFIC instance of the gate firing likely to be profitable, given the full feature context? The ML model does not generate trades; it FILTERS false positives from a primary signal that already has edge. The economic logic: not all gate fires are equal — a `liq_delta_z30` spike during a regime where `xd_cross_return_mean` (breadth) is also positive may be more reliable than a spike in a breadth-divergent regime. An LightGBM/XGBoost meta-labeler trained on TRAIN+VAL to predict profitability of GATE-FILTERED entries, using the feature context plus the PURGED k-fold constraint (no temporal leakage), should improve precision without generating new alpha. The ATR-scaled triple-barrier label (TP = 2× ATR, SL = 1× ATR, vertical = N-bar max hold) is the defensible meta-label structure per FOUNDATION §7-E SOTA.

### 2. PRE-REGISTERED FALSIFIER

The meta-labeler AUC on the HELD-OUT set (OOS+UNSEEN, never touched during meta-labeler TRAIN+VAL) must exceed 0.55 (a meaningful lift above null, not merely 0.505). If the held-out AUC is < 0.55, the meta-labeler adds no precision and the avenue is dead — do not proceed to position sizing. Second falsifier: if the meta-labeler reduces n_trades by more than 50% without a corresponding lift in per-trade win rate, it is filtering into a thin-sample regime where the DSR correction kills any apparent lift.

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
0. PRECONDITION: a primary gate must have PASSED Steps 1-5 of the universal gate chain first.
   This avenue is gated on Avenue-2 (or Avenue-1) producing a BATTERY-SURVIVING gate. No survivor
   from Avenue-2 = this avenue is unreachable until one exists.
1. archive/analysis/meta_labeler.py  -- port the archived meta-labeler; add ATR-scaled triple-barrier
   labeling (TP=2x ATR, SL=1x ATR, vertical=N-bar hold); purged-k-fold CV (purge_gap=5 bars)
2. train the meta-labeler on TRAIN+VAL ONLY using the gate-filtered entry points; XGBoost or
   LightGBM with Bayesian hyperparameter search (not grid search — overfits small n)
3. src/strat/candidate_gate.py  -- evaluate the gate+meta-labeler combination as a NEW candidate
   through the FULL gate chain; the DSR family-N must include the meta-labeler hyperparameter DoF
4. src/strat/battery.py  -- robustness battery on the combined system
```

### 4. DATA NEEDED

Requires a battery-surviving gate from Avenue-2 or Avenue-1 (PRECONDITION). The chimera feature set is the feature universe. The meta-labeler must only use TRAIN+VAL features for training; the triple-barrier label computation must use causal data (no future prices in barrier computation except as the look-forward target). `src/training/train_meta_labeler.py` exists (VERIFIED) but is untested on the post-reset apparatus — it must be re-validated on a known gate before use.

### 5. EV RATIONALE

Third priority because it is gated on Avenue-2. If no gate survives the battery, this avenue is unreachable. Conditional on a surviving gate, the meta-label approach has SOTA support (FOUNDATION §7-F: "Meta-labeling filters false positives, does not generate alpha, and can improve precision on a primary signal that already has edge"). The archived meta_labeler.py reduces rebuild cost. The risk is thin: the primary gate sets a precision floor; the meta-labeler can only help or be neutral (if AUC near null, revert to the un-meta-labeled gate).

### 6. EXPECTED FAILURE MODE

The primary failure: the gate-filtered sample is too small (n < 30) for a meaningful ML meta-labeler — at n=11 (the PEPE whale survivor), XGBoost will overfit TRAIN+VAL trivially and fail held-out. The DSR correction at true family-N (hyperparameter combinations × features tested) will kill marginal apparent lift. The archived bakeoff result (AUC 0.495–0.505) was on ML-as-generator without a primary gate — the meta-label version is architecturally different but may still fail if the gate-filtered n is too small. Report n_effective (post-purge) before training.

---

## AVENUE-4: Setup-capture and Benedict-style rotation (cross-asset entry/exit discipline)

**AVENUE_MAP status:** UNEXPLORED systematically. The FOUNDATION framing established this as the user's stated operation style: "enter, leave, profit — chase a setup, capture a meaningful move, cut instantly if wrong, rotate to the next." The Benedict exit layer (hard stop ~2-2.5% per-trade risk cap + time-stop) was posited as method-agnostic. No systematic exploration has been done.

### 1. MECHANISM HYPOTHESIS

At any time in a broad universe (~70-100 assets), a subset is in a setup state: a defined geometric or flow condition that historically precedes a directional move. The economic mechanism is behavioral and structural: assets that have recently undergone a volatility compression (inside-bar, low ATR relative to recent history) or a pullback-to-support within a larger trend often experience a release move driven by latent demand meeting diminishing supply. A Benedict-style rotation exploits this by (a) scanning the universe for setups at each bar, (b) entering the top-K by a pre-registered scoring function, (c) exiting via hard stop (wrong immediately = cut, ~2% per trade) or time-stop (no move in N bars = free the capital), and (d) recycling capital into the next firing setup. The economic advantage over passive buy-and-hold: capital is deployed only during the expected-move window (high capital velocity per FOUNDATION §2), and losses are capped per trade. This is mechanistically different from bar-by-bar discrimination: the unit is a SETUP (multi-bar), not a single prediction.

### 2. PRE-REGISTERED FALSIFIER

Define setup as: `norm_ma_distance` < −0.5 (pullback below slow MA) AND `norm_return_4` > 0 (3-bar reversal beginning) AND `norm_vol_cluster` < 0 (vol compression). Run this pre-registered setup definition on TRAIN+VAL; fire the firewall. If the cost-matched random-entry null beats the setup-selected entries on OOS+UNSEEN in more than 2 of 4 windows, the setup-selection mechanism adds no information over random capital deployment timing (the entries are just beta with a dressed-up filter). Do not optimize the threshold parameters after seeing results.

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
1. src/strat/discover.py:discriminate  -- identify which setup SIGNALS show discrimination
   (cross-sectional breadth of assets in setup state vs forward return)
2. src/strat/battery.py:evaluate_setup_chaser  -- the setup-chaser evaluator (positive expectancy +
   PF + selectivity-vs-flat). Already ported into the battery module.
3. src/strat/discover.py:scan  -- sweep (setup_definition x hold_bars x stop_width); family_n = total
   cells × top-K scoring variants tested; manifest sidecar required
4. src/strat/candidate_gate.py  -- full gate chain on surviving cells
5. src/strat/battery.py  -- robustness; n_eff >= 8 required (setup-based systems can be thin on trades)
```

Critical constraint: the archived `evaluate_setup_chaser` was flagged with the D13 dead-list item (selection leak: gated on UNSEEN > 0 in the archived version). The ported version in `src/strat/battery.py` must select setups on TRAIN+VAL only and test on held-out once — NEVER gate selection on UNSEEN.

### 4. DATA NEEDED

1d chimera across the universe (MATERIALIZED). The setup definition can be constructed from the verified structural cols: `norm_ma_distance`, `norm_return_4`, `norm_return_16`, `norm_vol_cluster`, `norm_efficiency`, `regime_label`. For a sub-daily version (if desired), the 4h chimera is available universe-wide, BUT the `load_panel` sub-daily floor bug must be bypassed — use ChimeraLoader per-asset native loader (`ChimeraLoader().load(sym, '4h')`), never the archived `strat.xsec_momentum.load_panel` which silently floors to daily (FOUNDATION dead-list INFRA BUG, confirmed).

### 5. EV RATIONALE

Fourth priority. The setup-capture approach is the founding user intuition and is architecturally distinct from the failed bar-by-bar mining (different unit of trade). The ported `evaluate_setup_chaser` reduces rebuild cost. However, D13 (selection leak in the archived per-asset chaser) was verified structurally closed; the clean version showed below-chance persistence (0.08–0.15 vs 0.25). This is a real negative signal, but it was produced by the old apparatus (before the LD-1/LD-4 fixes). The falsifier is cheap to run. If it clears the firewall with the fixed apparatus on even one pre-registered setup, the result is more trustworthy than the archived null. Capital velocity is an explicit KPI here: track return-per-deployed-capital-day alongside compound.

### 6. EXPECTED FAILURE MODE

D13 confirmed: the archived setup-chaser scan produced below-chance held-out persistence when measured cleanly (0.08-0.15 vs 0.25 random baseline). The most likely mechanism of failure: a broad universe scan over setup definitions is itself a ~6000-cell multiple-comparisons space (the same point the FOUNDATION oracle made about cell mining). The Holm correction at true family-N will kill marginal apparent survivors. The second failure: thin n_eff (setups are rare; n=11 is too thin for reliable statistics). Pre-register exactly one setup definition per run to control family-N.

---

## AVENUE-5: Cross-sectional / breadth-pooled approaches

**AVENUE_MAP status:** PARTIAL (breadth RSI-bounce satellite was tested in the foundation; it passed OOS firewall but FAILED UNSEEN — verdict: OOS-regime-luck, not robust). Long-only XS-momentum as a universe FILTER is UNEXPLORED on the FIXED native sub-daily loader. D15 (XS-momentum standalone 0/144) is dead-listed but the filter role is architecturally distinct.

### 1. MECHANISM HYPOTHESIS

Cross-sectional momentum (rank assets by past N-day return, long top quintile) in a long-only version concentrates capital in the assets already in established uptrends, which empirically reduces exposure to bear-market laggards. The mechanism is NOT alpha generation — it is a SYSTEMATIC TILT toward assets exhibiting strong momentum within the universe, reducing the draw of weaker assets. Per FOUNDATION §7-E SOTA: Han (2023) showed 85% of long-only XS-momentum specifications are positive in crypto; the best role is a UNIVERSE FILTER (trade setups only in top-quintile recent performers), not a standalone strategy. The testable version: filter all other strategies' entries to only fire on assets that are in the top 25% by 30d volume-weighted return rank. This should improve per-trade win rates by pre-selecting assets in confirmed uptrends — a structural overlay, not a standalone return engine.

### 2. PRE-REGISTERED FALSIFIER

Using the native per-asset ChimeraLoader (NOT the archived `load_panel`), compute 30d-lookback volume-weighted return rank across the full universe at each 1d bar. Filter a pre-registered base strategy's entries to only the top-25% rank. Compare filtered vs unfiltered compound and firewall results on OOS+UNSEEN. If the filtered version does NOT beat the unfiltered version in BOTH compound AND firewall (beats-null on OOS+UNSEEN), the filter adds no value and the avenue is dead. This test requires a valid base strategy as the entry signal source — the filter avenue is gated on another avenue producing a strategy that clears at least Step 3 (firewall) first.

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
0. PRECONDITION: a base strategy from Avenue-1, 2, or 4 that clears at least Step 3 (firewall).
1. ChimeraLoader().load(sym, '1d') per-asset loop  -- compute 30d volume-weighted return rank across
   the universe (this MUST use native loader; the archived load_panel bug silently floors sub-daily)
2. Apply the filter: drop base-strategy entries where the asset is NOT in the top-25% rank on that bar
3. src/strat/candidate_gate.py  -- re-run the filtered candidate through the full gate chain
   family_n = total (asset x rank_threshold x lookback) cells tested in the filter sweep
4. src/strat/battery.py  -- robustness battery; n_eff may drop (filtering reduces trades)
5. pre-register the pooling weight (equal weight across filtered entries) BEFORE touching held-out
```

Note: XS-momentum STANDALONE (D15: 0/144) is on the dead-list. This spec is for the FILTER ROLE only. Do not reopen the standalone version.

### 4. DATA NEEDED

1d chimera (MATERIALIZED). The `norm_return_4` (4-bar), `norm_return_16` (16-bar), and `xd_momentum_rank` (cross-sectional momentum rank) columns are available — `xd_momentum_rank` CONFIRMED present at 100% coverage on BTC + PEPE 1d (re-checked 2026-06-05; still confirm it matches the 30d-vol-weighted spec before relying on its semantics). A pre-computed rank column (`xd_momentum_rank`) already exists in the chimera — use it directly rather than recomputing. Confirm it matches the 30d volume-weighted specification before relying on it; if not, compute from raw closes. Requires a base strategy entry signal as input (PRECONDITION).

### 5. EV RATIONALE

Fifth priority because it is doubly gated (requires both the fixed sub-daily loader AND a base strategy). The cheapest test once a base strategy exists (single overlay, one filter threshold). The SOTA evidence is positive (Han 2023). The breadth satellite (FOUNDATION §18) demonstrated the firewall works and that OOS-positive-only is insufficient — a sobering result but one that clarifies the required bar (OOS+UNSEEN both positive). The `xd_momentum_rank` column being pre-computed in the chimera reduces implementation cost.

### 6. EXPECTED FAILURE MODE

The breadth RSI-bounce satellite already demonstrated the core failure mode: passes OOS (real_exp +1.77% > null p95), fails UNSEEN (real_exp -0.31%, negative). The filter may reduce n_eff below the battery floor (n_eff < 8) by concentrating entries in fewer assets/bars. The ECL tension flagged in FOUNDATION §7-G (literature positive, project negative on the same family) is not fully resolved — the Grobys 2025 "illusion" finding (long-only momentum net-of-cost) may apply. The filter role is architecturally weaker than standalone momentum, so even the filter may not survive cost.

---

## AVENUE-6: Self-improving decay-rotation over a validated sleeve library

**AVENUE_MAP status:** UNEXPLORED. Requires a library of validated sleeves (strategies with known decay patterns). Currently no validated sleeves exist (the PEPE whale-gate provisional survivor does not yet meet the battery gate). This avenue is strictly GATED on having a minimum of 3 battery-surviving sleeves from other avenues.

### 1. MECHANISM HYPOTHESIS

A portfolio of validated, decoupled per-asset strategies (sleeves) each have finite alpha lifetimes — edge decays as market participants arbitrage it out or as the regime changes. A self-improving system monitors each sleeve's rolling out-of-sample performance in near-real-time, detects decay early (below a pre-registered performance floor), and rotates capital to the currently highest-performing non-decayed sleeve. The mechanism is NOT alpha generation within any individual sleeve; it is CAPITAL EFFICIENCY across a library of independently-validated sleeves with non-zero expected decay. The value is operationally: a static portfolio slowly decays to zero; a decay-aware rotating portfolio maintains deployed capital in live edges while cutting dead ones. This is a LIFECYCLE management layer, not a discovery tool.

### 2. PRE-REGISTERED FALSIFIER

Simulate the decay-rotation policy on a synthetic library of 3 sleeves (using TRAIN+VAL for lifecycle simulation, OOS+UNSEEN for holdout). If the rotating portfolio does NOT outperform the equal-weight static portfolio on OOS+UNSEEN (compound + firewall both), the decay-rotation mechanism adds no lifecycle value over passive equal-weighting. This is a structural null: if sleeve decays are uncorrelated and the rotation timing has no information advantage, the rotator just introduces transaction cost.

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
0. PRECONDITION: minimum 3 battery-surviving sleeves from Avenues 1-5. NO sleeve library = this
   avenue is unreachable. Do NOT fabricate test sleeves — must use real validated ones.
1. Locate/port a decay-monitor primitive (rolling Sharpe / hit-rate window, pre-registered floor)
2. Design the rotation policy: (a) select the top-K sleeves by rolling 60-day Sharpe,
   (b) weight equally or by vol-parity (pre-register the rule before touching held-out)
3. src/strat/candidate_gate.py  -- run the rotation policy through the gate chain
   family_n = (rotation_lookback choices) × (K choices) × (floor_threshold choices) tested
4. src/strat/battery.py  -- robustness on the rotation system
```

### 4. DATA NEEDED

Requires 3+ battery-surviving sleeves with per-bar trade logs (input to the decay-monitor rolling window). The rotation simulation must be done on data that the individual sleeves were NOT trained on (OOS+UNSEEN only for the holdout — each sleeve's in-sample learning must not contaminate the rotation simulation).

### 5. EV RATIONALE

Sixth priority — contingent. No validated sleeve library currently exists. The avenue is correctly ordered last among the methodology axes because it amplifies existing edge rather than discovering it. Its EV is a function of the EV of the upstream avenues that produce the library. If Avenues 1-4 produce zero survivors, this avenue is unreachable and irrelevant. If they produce 3+, this becomes a high-value operational layer (prevents the dead-sleeve capital trap).

### 6. EXPECTED FAILURE MODE

The library never materializes (Avenues 1-4 produce fewer than 3 battery-survivors). If the library does exist, the most likely failure is that the decay signals are too slow (Sharpe decay takes months to detect reliably; the rotation fires too late, after the sleeve has already damaged the portfolio) or that the rotation itself introduces costs that offset the lifecycle benefit (each rotation event = taker round-trip at 0.24%).

---

## AVENUE-7: WM (World Model) as simulation trainer / representation ground (AlphaZero analog)

**AVENUE_MAP status:** UNEXPLORED in defensible role. WM-as-signal is DEAD (D14: ~3 OOM below cost). The defensible roles are (a) WM as a FILTER (signal the IC/ShIC of which assets/setups are predictable at this moment), and (b) WM as a SIMULATION TRAINING GROUND for strategy design (analog: AlphaZero-style search in the WM's learned distribution, not in the live market). Multiple WM versions exist in `src/wm/` (VERIFIED: v1.0 through v25, various architectures).

### 1. MECHANISM HYPOTHESIS

A world model trained on historical multi-asset return distributions learns a compressed representation of market states. The representation utility is NOT to directly trade its signal (D14); it is to provide a calibrated simulation environment in which strategy candidates can be rapidly evaluated across many synthetic regimes without burning the held-out data. The analog: AlphaZero does not trade in the chess environment, it uses the environment's rules (a perfect simulator) to search and evaluate policies. Our WM is an IMPERFECT simulator — but a WM that satisfies OOS IC > 0.015 / ShIC > 0.015 (the Filter-tier threshold from CLAUDE.md) at least offers a partially valid representation of the return-generating process. The testable defensible use: (a) given a candidate gate from Avenue-2 or Avenue-1, use the WM's representation of the current state (the latent h_seq vector) to predict whether the gate is in a "reliable" vs "unreliable" regime for THIS asset at THIS time — a meta-labeling role distinct from ML meta-labeling because the features are the WM's compressed latent state, not raw features; (b) use the WM to generate synthetic market sequences for stress-testing strategies across more regime combinations than the 6-year history provides. Role (b) is research-grade and requires the simulator-fidelity problem to be partially solved first (FOUNDATION §7-F: "agents exploit the learned simulator's simplifications").

### 2. PRE-REGISTERED FALSIFIER

For role (a) — WM as regime-quality signal: train a light probe (linear model) on the WM's latent h_seq to predict whether the current bar is in a "gate-reliable" state (defined by the primary gate's per-bar realized Sharpe in a rolling 30-bar window on TRAIN). If the probe's OOS AUC < 0.55, the WM latent does not carry actionable regime-quality information and role (a) is dead. For role (b) — WM as simulation: if the synthetic sequences generated by the WM do not preserve the empirical autocorrelation structure of the real data (KL divergence > 15.0 per CLAUDE.md invariant), the simulator is too distorted for reliable strategy search and role (b) is dead.

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
0. PRECONDITION for role (a): a battery-surviving primary gate (from Avenue-2 or 1).
   PRECONDITION for role (b): WM with validated OOS IC > 0.015 / ShIC > 0.015 (Filter-tier minimum).
1. Load WM checkpoint; extract latent h_seq for each (asset, bar) in the chimera.
   WM validation gate: run the per-version validator to confirm IC and ShIC meet the Filter-tier floor.
2. Role (a): train linear probe on h_seq -> gate_reliable label (TRAIN+VAL); evaluate on OOS+UNSEEN.
   If AUC >= 0.55, add WM-latent filter as an additional gate layer to the primary candidate.
3. src/strat/candidate_gate.py  -- run the augmented candidate (primary gate + WM-filter) through
   the full gate chain; family_n includes the WM checkpoint choices + probe hyperparameter choices.
4. src/strat/battery.py  -- robustness on the augmented system.
```

Note: no new WM training is required if an existing v1.0-v25 checkpoint meets the IC/ShIC floor. The WM training layer is outside the strat discovery scope.

### 4. DATA NEEDED

An existing WM checkpoint with passing IC/ShIC (Filter-tier). Multiple checkpoints exist across `src/wm/v1/` through `src/wm/v25/` (VERIFIED directory structure). The chimera feature set is the WM input. Note: the WM was trained on raw-return targets (`target_return_*` per CLAUDE.md invariant) — the latent representation is a return-conditional embedding, appropriate for the regime-quality probe. Requires the split invariants to match the strat discovery splits exactly.

### 5. EV RATIONALE

Seventh priority. Gated on both a battery-surviving gate AND a validated WM checkpoint. The structural case is weak compared to Avenues 1-4 (the simulator-fidelity problem is unresolved, and FOUNDATION §7-F confirmed "essentially no credible published live-trading wins" for WM-as-policy). The ONLY defensible role is the latent-as-regime-quality-filter, which is cheap to test once the preconditions exist. Role (b) (simulation training ground) is research-grade and belongs in a longer-horizon research track, not the solving phase. If role (a) shows AUC < 0.55 on the first test, close the avenue and record it as a data point, not a direction for further optimization.

### 6. EXPECTED FAILURE MODE

The WM latent is a return-conditional embedding optimized for return prediction, NOT for regime-quality classification of a gate's reliability. The probe will likely fail to find structure because the WM was not trained to distinguish "gate-reliable" from "gate-unreliable" states — the gate is downstream of the WM's loss function. D14 (WM-as-signal, ~3 OOM below cost) reflects that the WM's predictive signal is too weak even at its DIRECT output; using it as a second-order meta-signal is even less likely to carry information. The role (b) failure is structural and already documented: crypto markets are partial-information, non-stationary, with no perfect simulator — self-play exploits the learned simulator's simplifications, not the market's (FOUNDATION §7-F; LiveTradeBench: backtest-top agents did WORSE live).

---

## AVENUE-8: TI × ASSET × REGIME × TIMEFRAME grid (systematic sweep with fixed apparatus)

**AVENUE_MAP status:** PARTIAL — PEPE × MA/EMA at coarse dollar cadence is the prior surviving cell; most of the grid is UNEXPLORED. This is the FOUNDATIONAL discovery unit: every avenue eventually resolves to a specific cell in this grid. The grid is vast; the solving phase must scope it carefully to avoid the multiple-comparisons trap.

### 1. MECHANISM HYPOTHESIS

Within the 8-axis space (chart-type × resolution × instrument × signal × regime × method × approach × exit-policy), there exist cells where a combination of a structural TI (sets the trend frame and entry timing) with an exogenous gate (confirms informed flow is active) produces a setup that clears taker cost with positive expected value in UNSEEN. The hypothesis is NOT that bare TIs have edge (D1 confirmed: standalone price-TI mining produced 0 survivors across ~127 indicators × all cadences). The hypothesis is that the COMBINATION of a TI structure frame with a verified gate (from Avenue-2, e.g. `liq_delta_z30`) produces setups more predictable than either alone, because the gate selects the moments when the TI-defined setup is likely to resolve rather than fail. This is the "gated-TI" architecture from FOUNDATION §2 and AVENUE_MAP Axis 6.

### 2. PRE-REGISTERED FALSIFIER

For a pre-registered cell (e.g. PEPE × SMA(200) pullback × `liq_delta_z30` gate × 1d cadence × taker cost): run the firewall before any parameter variation. If the pre-registered cell fails the firewall on OOS+UNSEEN, the hypothesis that "gated-TI outperforms bare-TI AND random entry" is rejected for this specific cell. Record in the family-N manifest. Do NOT re-run with alternative TI parameters — that re-opens family-N. A pre-registered cell must be specified exactly (TI period, gate threshold, holding policy, stop width) before the backtest is run.

### 3. APPARATUS TOOLS

```
ORDER OF OPERATIONS:
1. Pre-register the full grid spec in a _sweep_manifest.json BEFORE running any cell:
   n_TI_variants × n_gate_variants × n_assets × n_cadences × n_regimes = family_N (upper bound)
2. src/strat/discover.py:discriminate  -- gate pre-selection (only test TI × gate combinations where
   the gate shows discrimination independently; reduces the effective family-N)
3. src/strat/discover.py:scan  -- run the scoped grid (NOT the full 184 × all-TIs × all-assets grid;
   scope to: 3-5 TIs × top-5 gates from discriminator × 10-20 assets × 2 cadences)
4. src/strat/candidate_gate.py  -- full gate chain on each candidate cell
5. src/strat/battery.py  -- robustness on any Step-1-5 survivor
6. src/audit/check_dsr_holm.py  -- DSR at TRUE family-N (the manifest's declared n_variants_tested)
```

Scoping rules to apply BEFORE the grid is opened: (a) TI choices: restrict to 3-5 structural frames (SMA-200 distance, RSI 14-bar, EMA crossover, MACD signal). (b) Gate choices: restrict to the top-3 from discriminate (ranked by same-sign-across-windows AND beats shuffle-null). (c) Asset choices: start with 10 assets including BTC, ETH, PEPE and 7 high-vol alts; expand to u50 only if a survivor is found. (d) Cadence: 1d only in the first sweep; sub-daily only if a 1d survivor is found.

### 4. DATA NEEDED

1d chimera across universe (MATERIALIZED). Gate columns (full list VERIFIED via ChimeraLoader above). The `discriminate` output (gate ranking) is needed as input to step 3 — produces a priority-ordered gate list with shuffle-null comparisons. Sub-daily chimera (4h) is available via `ChimeraLoader().load(sym, '4h')` (use native loader only — NOT `load_panel`). Do not test 4h until a 1d survivor exists.

### 5. EV RATIONALE

Concurrent with Avenue-2 (the grid is the operational expression of Avenue-2). Structured separately because the grid sweep requires an explicit family-N declaration and scoping discipline that is distinct from the single-gate discrimination exercise. The concrete `liq_delta_z30` lead from the foundation's discriminate run makes this actionable: one pre-registered cell can be tested immediately once the apparatus is confirmed (LD-1 through LD-5). Family-N discipline is the central risk — the solving phase must commit to a family-N ceiling before opening ANY cell to avoid the multiple-comparisons ghost that produced the prior +36.8% selection leak.

### 6. EXPECTED FAILURE MODE

The multiple-comparisons ghost. Even with a scoped grid (3 TIs × 3 gates × 10 assets × 1 cadence = 90 cells), the expected number of chance survivors at alpha=0.05 is 4-5 cells. The Holm correction at family-N=90 requires each surviving cell to reach p < 0.00056 to claim significance — far stricter than the nominal 5%. The solving phase MUST declare family-N before running, run Holm, and treat any cell that fails Holm as a non-result. Second failure: the regime conditioning layer adds a multiplicative DoF factor (if regime is a sweep axis, family-N = cells × regimes × regime-detection-method variants) that can easily push family-N into the hundreds.

---

## APPARATUS PRECONDITION — Phase 0 (MUST complete before ANY avenue above)

**This is not an avenue; it is the keystone.** Until Phase 0 is complete, no number from any of the above avenues is trustworthy. (Hardening status reflects the 2026-06-05 apparatus red-audit + fixes — see `docs/APPARATUS_AUDIT_2026_06_05.md`.)

| Lock-down item | File (VERIFIED path) | Status | Blocking condition |
|---|---|---|---|
| LD-1 FillModel (taker 0.24% default) | `src/strat/fill_model.py` | PORTED + adverse-sign fix | Maker calibration PROVISIONAL — recalibrate vs real fills before shipping maker numbers |
| LD-2 Leak probe (relative-twin verdict) | `src/wealth_bot/leak_probe.py` | `relative_leak_test` VALIDATED (cadence-robust) | Use relative_leak_test, NOT the advisory absolute-pp verdict, as the hard gate |
| LD-3 DSR/Holm gate (severity="critical", true family-N) | `src/audit/check_dsr_holm.py` | FIXED 2026-06-05 (ship-claim-fails-Holm ⇒ critical/halt; family-N = max(written, manifest `n_variants_tested`, per-JSON declared)) — staged uncommitted, RWYB `--selftest` passes | Declare `n_variants_tested` in a `_sweep_manifest.json` per sweep so family-N is honest |
| LD-4 Random-entry null firewall | `src/strat/firewall.py` | PORTED + zero-trade-window fix | Add regime-matched-null variant for the solving phase |
| LD-5 Bear-inclusive holdout + native sub-daily loader | ChimeraLoader().load(sym, cadence) — VERIFIED | sub-daily loader is correct; `load_panel` from archive is BROKEN | Never use `strat.xsec_momentum.load_panel` for sub-daily; always use ChimeraLoader |

**Sequencing:** LD-1 + LD-3 + LD-4 are the minimum precondition. LD-2 and LD-5 are parallel. The apparatus must be confirmed PASSING on a known-result test (e.g. the BTC 1d benchmark from FOUNDATION §15 should reproduce its `taker` result: TRAIN ~82% not 135%, OOS negative) before any discovery run opens.

---

## EV-ranked queue summary

| Rank | Avenue | Precondition | Cost to first falsifier test |
|---|---|---|---|
| 1 | AVENUE-1: Info-driven bar chimeras | Phase 0 apparatus | Low (BTC/ETH/PEPE dib+range already chimera-built; scan on 3 assets) |
| 2 | AVENUE-2: Gate space (s3_/hbr_/liq_/etf_/stbl_) | Phase 0 apparatus | Very low (1d chimera materialized; discriminate pass is cheap) |
| 2= | AVENUE-8: TI × ASSET grid (gated-TI, scoped) | Phase 0 + Avenue-2 discriminate output | Low once gate ranking exists |
| 4 | AVENUE-4: Setup-capture / Benedict rotation | Phase 0 apparatus | Medium (D13 leak fix in the ported chaser) |
| 5 | AVENUE-5: XS / breadth filter | Phase 0 + one Avenue 1/2/4 survivor | Low once base strategy exists |
| 6 | AVENUE-3: ML meta-labeling | Phase 0 + one battery-surviving gate | Medium (meta-labeler port + purged CV) |
| 7 | AVENUE-6: Self-improving decay-rotation | Phase 0 + 3+ battery-surviving sleeves | High (library precondition may never materialize) |
| 8 | AVENUE-7: WM as filter / simulation ground | Phase 0 + validated WM IC/ShIC + one surviving gate | High (WM validation is a non-trivial precondition) |

---

## Path dependencies summary

```
Phase 0 (apparatus)
   |
   +-- [cheapest, no precondition] --> AVENUE-2 (gate discriminate) --> AVENUE-8 (gated-TI grid)
   |                                                                          |
   +-- [build chimeras] -----------> AVENUE-1 (info-driven bars)             |
   |                                        |                                 |
   |                                        v                                 v
   |                               Any battery-surviving gate or sleeve
   |                                        |
   +-- [port chaser] --------------> AVENUE-4 (setup-capture)                |
                                            |                                 |
                                     -------+---------------------------+-----+
                                     |                                 |
                                     v                                 v
                               AVENUE-5 (filter)              AVENUE-3 (ML meta-label)
                                                                       |
                                                               3+ sleeves exist?
                                                                       |
                                                               AVENUE-6 (decay-rotation)
                                                               AVENUE-7 (WM filter)
```

---

## Hard constraints (all avenues, non-negotiable)

- LONG-ONLY, SPOT, LEVERAGE = 1. No short positions. No margin. Any backtest that silently enables shorting or leverage is invalid.
- TAKER 0.24% round-trip is the working cost baseline. Any result that only survives at ideal (0.10%) or maker-only cost is rejected.
- Held-out UNSEEN = [2025-12-31, 2026-05-22] (oos_end .. unseen_end; canonical split = `strat.DEFAULT_WINDOWS`) is touched ONCE per candidate, after all spec and hyperparameter choices are finalized. Do not re-touch UNSEEN to iterate.
- The 2026 alt-bear in UNSEEN is a required holdout feature (FOUNDATION §10 confirmed: equal-weight universe -13 to -36% in this period). Any strategy that "passes" only because UNSEEN happened to be a bull is not robust.
- Objective: WEALTH (compound %). Sharpe is a secondary diagnostic, not the optimization target.

---

## Solving-phase caveats (foundation completeness-critic, 2026-06-05)

- **Survivorship bias (G-9):** the universe is currently-listed assets only (≥17 delisted assets purged;
  FOUNDATION §). Every long-only compound % is biased HIGH. Until a delisted-reconstruction quantifies the
  delta, apply a qualitative downward haircut (~10–20pp/yr) to any headline compound, and never treat a
  marginal pass as robust. Quantifying this haircut is itself a pre-solving task.
- **"VERIFIED" data-path claims are point-in-time (G-10):** the bar/chimera inventory counts cited above were
  scanned 2026-06-05 against `data/` (gitignored, regenerable). Treat them as VERIFIED-AT-2026-06-05; the
  solving phase MUST re-verify (`ls data/processed/{bars,chimera}/...`) before relying on a count, and a stale
  chimera will trip the CDAP `chimera_stale_crit` gate (one is currently stale: pepeusdt 4h).
- **Leak probe is vacuous for FILTER-LESS candidates (G-11):** `evaluate_candidate` runs `relative_leak_test`
  only when a `filter_col` is present (the exogenous leak vector); for filter-less / multi-condition setups
  (e.g. AVENUE-4 setup-capture) it returns `PAST_ONLY_OK` WITHOUT measuring. Those avenues must construct a
  bespoke known-clean reference (a guaranteed-past-only twin) and call `relative_leak_test` explicitly, or the
  leak step is unmeasured. The fast/slow columns are past-only by construction (`sma/wma/ema_past_only`).

---

*These specs are EXECUTABLE INSTRUCTIONS for a future authorized solving phase. They are NOT run here. No backtest has been executed. No conclusion about edge existence is drawn. All avenues are held OPEN — the falsifiers will decide, not priors.*
