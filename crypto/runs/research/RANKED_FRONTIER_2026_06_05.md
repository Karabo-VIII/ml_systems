# Ranked Exploration Frontier — unexplored avenues (2026-06-05)

> **Task:** enumerate the unexplored avenues (bar-types, timeframes, instruments, indicators, regimes,
> entry/exit policies) into a ranked frontier — **explicitly NOT treating the archived 4h/daily-LO
> "known-hard" premise as settled** (per the reset rule: prior conclusions are hypotheses to RE-TEST).
> Unit of trading = a SETUP across a multi-candle MOVE; objective = robust held-out compound return;
> constraint = LONG-ONLY + spot + lev=1. Companion: `docs/AVENUE_MAP_2026_06_04.md` (this RE-VERIFIES its
> data-state numbers by RUNNING, and re-positions 4h/daily-LO as a live node rather than a settled-out one).

## A. Verified data state (RWYB — every number below produced from a command this run)

**Chimera (feature-enriched) build counts** — `ls data/processed/chimera/<type>/*.parquet | wc -l`:

| Bar type | chimera assets built | raw bars (distinct assets) | verdict |
|---|---|---|---|
| time 1d | **104** | n/a | built |
| time 4h | **104** (BTC = 13,996 rows, 243 cols) | n/a | built — **re-testable, NOT absent** |
| time 1h | **104** | n/a | built |
| time 30m | **77** | n/a | partial |
| time 15m | **104** | n/a | built |
| dollar | **104** | n/a | built |
| **dib** (dollar-imbalance) | **3** (BTC/ETH/PEPE) | **87** | **substrate unbuilt for ~84 assets** |
| **adaptive_vol** | **0** (EMPTY) | **87** | **fully unbuilt** |
| **runs_tick** | **0** (EMPTY) | **87** (PEPE = 19,628 rows) | **fully unbuilt** |
| **range** | **3** (BTC/ETH/PEPE) | **87** | **substrate unbuilt for ~84 assets** |
| **runs_volume** | **0** (EMPTY) | **77** (2023-only, bad calib) | broken — fix first |

**Sample-density (why info-driven bars matter for "setup across a move")** — PEPE row counts:
`daily 1,119` vs `dib 414,699 (370×)` · `adaptive_vol 355,088 (317×)` · `runs_tick 19,628 (17×)`.
→ event-clocked bars sample by *information arrival*, not time — orders of magnitude more multi-candle
setups to learn from, on a substrate that has **never been mined**.

**Tooling is ready (de-risk):** builder `src/pipeline/make_chimera_bars.py` exists (inherits the audited
look-ahead-safe dollar-chimera machinery). Test gate `src/strat/` — all 6 modules
(`battery / firewall / fill_model / candidate_gate / discover / benchmark`) **import clean this run**.
So the top avenues are *buildable and testable today*; the gap is compute + a signal, not infrastructure.

## B. The ranked frontier (EV = unexplored-ness × substrate size × tractability × mechanism plausibility)

### TIER 1 — highest EV (genuinely unexplored, large substrate, de-risked, mechanism-plausible)

**1. Info-driven-bar chimeras × u100, mined for multi-bar setup-capture** — *bar-type + timeframe axis.*
The single biggest unexplored substrate (VERIFIED: 0–3 of 104 assets enriched; raw exists for 87; 17–370×
the sample density of daily). Directly embodies the "setup across a move" unit (bars are clocked by flow,
not the clock) and **sidesteps the daily/4h single-bar cost-wall entirely** (different substrate, not a
re-mine). De-risked: builder + gate both runnable now.
  - *Within-tier sub-rank:* **dib** (414k rows, informed-flow) > **adaptive_vol** (355k, regime-scaled) >
    **runs_tick** (sparser ~20k, momentum-shift) > **range** (clean breakout but a trend tool; only 3 enriched).
  - *−k falsifier:* enriching ~84 assets is compute-heavy; event bars can carry bid-ask-bounce noise
    (the fine-dollar memecoin cost-trap warning); *selective capture on these bars is untested* — density ≠ edge.

**2. The 184-feature GATE space on the already-built 1d/4h chimeras (gate, NOT bare TI)** — *indicator axis.*
Cheapest high-EV cut: data already built for 104 assets. Market research flagged **liquidation cascades** as
the standout volatility-coincident marker (+25% same-day move, fires 5–6% of days). The one real mechanism
analog (whale-on-PEPE) says *gates*, not bare TIs, are where edges hide. **A concrete lead already exists:**
`liq_delta_z30` (PEPE, H=4) is same-sign across all 4 windows AND beats its own shuffle-null on UNSEEN.
  - *Order to test:* liquidations (`liq_*`) → s3 smart-money (`s3_smart_vs_retail`) → hawkes (`hbr_eta_*`)
    → macro (`etf_*`, `stbl_*`).
  - *−k falsifier:* contemporaneous ≠ predictive (reverse-causality risk on liq spikes); derivatives
    features are thin (present on only 75–96 of 104 assets); **discrimination ≠ harvestability** — must still
    clear cost + firewall + battery.

### TIER 2 — strong, but more uncertain or premise-loaded

**3. Setup-capture / Benedict-style rotation entry-exit policy (the user's stated style)** — *entry/exit axis.*
The "velocity engine" (hard-stop + time-stop + signal-flip exit) is UNEXPLORED systematically and is the
user's stated trading style. Composes with #1/#2 as the harvest layer.
  - *−k falsifier:* worthless without an entry signal — random entry = 47% net-positive coin-flip (market research).

**4. RE-TEST 4h/daily-LO clean under the hardened apparatus — the explicitly-un-settled premise** — *timeframe axis.*
**Per the task, this is NOT settled.** The prior "known-hard" reads came from a *broken* apparatus
(maker-not-taker cost, no-op DSR gate, sub-daily→daily `load_panel` bug) — those are FALSE-POSITIVE
generators, so the negatives are *plausible but never re-confirmed clean*. The data is already built for 104
assets and the `src/strat/` firewall is new → a clean TI×ASSET×REGIME re-run is **cheap**. Either outcome is
valuable: a survivor reopens the corner; a clean death is the strongest confirmation yet (a valid refutation).
  - *−k falsifier:* if it stays dead under the cost-matched random-entry null, the premise is finally *earned*, not inherited.

**5. Per-asset specialist across the full u100 (instrument breadth)** — *instrument axis.*
VERIFIED most of u100 is UNEXPLORED (only PEPE + a few majors touched). Highest value when combined with the
gate space (#2) and info-bars (#1), not as standalone bare-TI (that's in the known-hard family).
  - *−k falsifier:* survivorship (currently-listed only); standalone per-asset TI is the corner #4 calls hard.

### TIER 3 — lower EV now / prerequisite-gated (park with a wake-condition)

**6. ML meta-labeler on a battery-surviving gate** — *method axis.* Defensible only as a *filter on a proven
gate*, not a generator (generator-ML is known-hard). **Prerequisite:** a surviving gate from #1/#2. Wake when ≥1 gate clears.

**7. Regime as a validation-slicing axis + ex-ante regime gating** — *regime axis.* UNEXPLORED systematically;
composes with everything. The ex-ante detection is the hard sub-problem (HMM look-ahead trap).

**8. Self-improving decay-rotation bot / WM-as-trainer** — *method axis.* UNEXPLORED but **prerequisite = a
validated sleeve library that does not yet exist.** Park; wake-condition = ≥3 battery-passing sleeves.

**9. runs_volume + Heikin-Ashi / Renko off-pipeline** — *bar-type axis.* Data-limited/broken (runs_volume is
2023-only with bad calibration; HA/Renko exist only as PEPE sweep files). LOWEST EV until calibration is fixed.

## C. Honest scope notes
- **This is an enumeration + ranking, not an edge.** No avenue here is shown to be profitable; every "EV" is a
  *prior on where to look*, grounded in verified substrate size + mechanism plausibility, not in a backtest.
- **Overlap with the existing map:** `docs/AVENUE_MAP_2026_06_04.md` already enumerates these axes. This doc's
  additions are (a) RWYB re-verification of the build-count + sample-density numbers the ranking rests on, and
  (b) the explicit re-positioning of **4h/daily-LO (#4) as a LIVE re-testable node**, not a settled exclusion.
- **Reproduce the data-state table:** counts via `ls data/processed/chimera/<type>/*.parquet | wc -l`;
  row counts via `polars.read_parquet(...).height`; apparatus via `import strat.{battery,firewall,...}`.
