# Canonical Strategy-Dimension Registry — the other 7 axes, catalogued (2026-06-11)

> User ask (/orc): *"I believe the exit-mechanism dimension has itself been canonically documented, right?
> Including timeframes — 5m, 15m, 30m, 1h, 2h, 4h, 1d? And other static inputs (not really static because the
> list can be extended), and the chart types, say."*
>
> **Honest answer: partially, and asymmetrically.** Until now the only axis with a *dedicated canonical catalog*
> was the **signal axis** — [TI_MASTER_CATALOG](TI_MASTER_CATALOG_2026_06_11.md) +
> [CANONICAL_FACTOR_REGISTRY](CANONICAL_FACTOR_REGISTRY_2026_06_11.md). The other seven decomposition axes
> (chart-type, cadence, instrument, regime, method, approach, **entry/exit policy**) existed only as **one table
> row each** in [MARKET_FRAMEWORK/README.md](MARKET_FRAMEWORK/README.md). This doc closes that asymmetry: it gives
> the **non-signal axes** the same catalog treatment, with each option's params, what's coded, and its dead-list
> tag. Two beliefs were corrected by RWYB while building it (see ⚠️ flags below).

A strategy = a **point in the 8-axis constituent space** ([MARKET_FRAMEWORK/README.md](MARKET_FRAMEWORK/README.md)).
The SIGNAL axis (axis 4) is the factor registry; the other 7 axes are catalogued here. Both are extensible — "static
input" is a misnomer, every list below can grow.

| Axis | This registry | Signal axis (separate) |
|---|---|---|
| 1 Chart / bar-type | §1 | — |
| 2 Resolution / cadence | §2 | — |
| 3 Instrument | §3 | — |
| 4 **Signal / indicator** | → [FACTOR_REGISTRY](CANONICAL_FACTOR_REGISTRY_2026_06_11.md) (A=TI ~110, B=chimera ~218, C=frontier 141) | ✔ catalogued |
| 5 Regime | §4 | — |
| 6 Method | §5 | — |
| 7 Approach / portfolio | §6 | — |
| 8 **Entry / EXIT policy** | §7 (the exit catalog — the primary ask) | — |

Machine-readable mirror: [`config/strategy_dimension_registry.yaml`](../config/strategy_dimension_registry.yaml).

---

## §1 — Chart / bar-type axis
What a "bar" is. Tag: `have` = built on disk (`data/processed/chimera/<type>/` or `bars/<type>/`).

| Bar type | have | look-ahead | note / dead |
|---|---|---|---|
| **time** | ✔ (15m/30m/1h/4h/1d) | S | the default; everything below is alternative-clock |
| **dollar** (coarse ~6676/asset) | ✔ | S | volume-clock; chimera built |
| **dollar-imbalance (dib)** | ✔ bars only (chimera sparse) | S | info-driven; chimera empty/sparse — D-class "raw-only" |
| **runs-tick** | ✔ bars + chimera | S | info-driven; chimera built |
| **runs-volume** | ✔ bars + chimera | S | info-driven; chimera built |
| **range** (constant-range) | ✔ bars + chimera | S | info-driven; chimera built |
| **adaptive-vol** | ✘ | S | proposed; not built |
| **Heikin-Ashi** | ✘ | **R** | smoothed candles **repaint the current bar** — use only confirmed-close |
| **Renko** | ✘ | **R** | brick close lags; brick size = a parameter |

**RWYB note:** on-disk chimera = `{15m,30m,1h,4h,1d,dollar,range,runs_tick,runs_volume}`. Adaptive-vol / HA / Renko
are catalog entries, **not built**. Info-driven-bar chimeras are present but mostly **raw-only / sparse** — the
edge-bearing feature surface there is thin (a Fork-B data-engineering item).

## §2 — Resolution / cadence axis
⚠️ **Belief corrected:** your list was `5m,15m,30m,1h,2h,4h,1d`. **5m and 2h are NOT built** (finest time-bar on
disk is 15m; there is no 2h dir). The canonical *time* set is **{15m,30m,1h,4h,1d}**, plus the alternative clocks.

| Cadence | have | cost verdict | note / dead |
|---|---|---|---|
| **1d** | ✔ | clears at maker | the home cadence; beta+yield floor lives here |
| **4h** | ✔ | clears at maker | the standard sweep cadence (~6 bars/day); trend-beta |
| **2h** | ✘ (resample 1h) | — | **NOT built**; intraday_oracle resampled it ad-hoc → capture NEGATIVE |
| **1h** | ✔ | **cost-walled (HARD)** | D60: 1h MR exp −0.43% OOS; MA-cross whipsaw PF 0.91 |
| **30m** | ✔ | **cost cliff** | gross −89.5% at taker (A7) |
| **15m** | ✔ | **cost cliff** | finest built time-bar; sub-bar cost wall |
| **5m** | ✘ | (worse than 15m) | **NOT built**; finer than the 15m wall → harder, not easier |
| **dollar-coarse** | ✔ | clears | volume-clock; ~event-paced |
| **event-clock / fine-dollar** | partial | — | sub-bar; mostly unexplored |
| **tick** | ✘ | — | Fork-B; the only resolution where the IC-ruling (D13) is genuinely untested |

**Cross-cutting fact (#4 of the 10 lessons):** cost is the binding constraint and it scales with cadence — every
step finer than ~4h taker is a cliff; maker is the #1 lever but p_fill is 0.21–0.40, not 0.80. So "more setups at
5m" is an anti-edge unless maker-routed AND the per-trade gross clears a much higher bar.

## §3 — Instrument axis
| Sub-dimension | scope | note / dead |
|---|---|---|
| **universe** | per-asset over u10 / u50 / u100 | most of u100 unexplored; books pool across the universe |
| **perp vs spot** | both | **constraint: LONG-ONLY + spot + lev=1.0** (binding) — perp only as a data source / bear-short LO-exception (parked, user sign-off) |
| **options** (Deribit BTC/ETH) | ingest absent | the convexity/VRP channel (§A2 of 06_STRATEGY_RESEARCH); Fork-B-deep; D38/D64/D65 |

## §4 — Regime axis
The conditioning layer (the one lever that *did* move the needle — regime-GATING, not per-asset DNA).

| Regime taxonomy | basis | coverage / dead |
|---|---|---|
| **trend** | SMA-200 (close vs SMA) | explored — the gate that preserves the bear (regime-gated book floor) |
| **volatility** | calm / expansion / euphoria (vol terciles, past-only) | largely open; the predictable channel (lesson #2) |
| **crowding** | funding / OI / capitulation z-scores | largely open |
| **macro-liquidity** | stablecoin supply / ETF flows | D69 (macro flows null as timing) |
| **5-state GMM** | empirical (precomputed `regime` family) | available; multi-axis routing open |

**Fact:** per-asset config DNA = **noise** (D62/D73, 1/47 above noise, collapses date-demeaned). Regime is a
**market-state** lever, not an asset-identity one.

## §5 — Method axis
| Method | status / dead |
|---|---|
| **static rules** | explored — the workhorse |
| **dynamic / regime-adaptive** | the live lever (regime-gating); per-asset adaptation dead (D62) |
| **ML as alpha generator** | **DEAD** — AUC ≈ 0.50 (D16/D17) |
| **ML as meta-labeler** on a proven exo-gate | open — the only defensible ML role (lesson #7) |
| **self-improving rotation** | x-sectional momentum = beta+concentration (D68) |
| **world-model** | built (V1.1 IC 0.067), diagnostic-only; not an alpha source post-reset (D13) |

## §6 — Approach / portfolio axis
| Approach | status / dead |
|---|---|
| **per-asset specialist + combine** | concentration failure mode (lesson #6) |
| **cross-sectional / breadth-pooled** | open; x-sec momentum = beta (D68) |
| **regime-gated portfolio** | **the floor** — the one full-cycle survivor (beats buyhold + random null; NOT UNSEEN-positive in the 5mo bear) |
| **setup-chaser book** | the book-not-config robustness frame |
| **oracle-decomposition** | the guiding method (construct oracle → decompose DNA → capture-rate proxy) |

## §7 — Entry / EXIT policy axis (the primary ask — now catalogued)
Two halves. **Lesson:** exit selectivity matters **less than entry selectivity** (lattice note); trailing adds
**~2pp/trade** but does not rescue a non-edge entry; smart exit-timing on daily breakouts is **null** (D61).

### 7a — ENTRY policy
| Entry policy | params | coded | note / dead |
|---|---|---|---|
| MA/EMA cross-up | fast/slow len | ✔ `entry_signal_lab`, `family1_chandelier_trail` | trend-continuation; whipsaws sub-4h |
| breakout (N-bar high / Donchian) | lookback | ✔ `intraday_oracle`, `trend_book_lab` | event-clock breakout |
| RSI/Boll mean-reversion bounce | thresh | ✔ `family_regime_map` | MR class PF 0.84–0.91 net-negative |
| "buy-the-extreme" (liq-spike / oversold) | — | ✔ (refuted) | **ANTI-edge** (D48–D52) — fires mid-cascade; random beats it |
| WM-signal gate | IC>0.015 | ✔ `wm_entry_producer` | diagnostic-only |

### 7b — EXIT mechanism (the dedicated catalog you expected)
| Exit mechanism | params | coded | empirical read / dead |
|---|---|---|---|
| **fixed-horizon / time-stop** (no-skill baseline) | hold H bars | ✔ `oracle_exit_knob` (H∈{2,4,6,10,16,24}), `exit_capture_proxy` | best-hold ~11–12 bars; **leave FIXED** — H is NOT factor-predictable (R² 0.01–0.13, ~0 at 4h); this is the control every smart exit must beat |
| **trailing stop — Chandelier** | ATR mult × rolling-period-high | ✔ `family1_chandelier_trail` | adds ~2pp/trade; **D23** fails on PEPE-MA UNSEEN but **works on breadth-bounce** (entry-dependent) |
| **trailing stop — % off running-high** | trail fraction (e.g. 0.10) | ✔ `naive_ma_trailing` | the "10pct-risk" naive variant |
| **MA-cross exit / signal-flip** | exit MA len | ✔ `exit_capture_proxy`, `family_regime_map` (death-cross) | 1h whipsaw PF 0.91; 4h PF 1.23 = beta |
| **fixed target / take-profit** | R-multiple or % | ✘ | **UNTESTED** — the softest open exit sub-axis (D61 SOFTEST list) |
| **triple-barrier** | up/down/time | ✔ as ML label | **D09** Val AUC 0.472 (below chance) as a *label*; as an *exit policy* untested |
| **managed-RSI exit** | exit RSI thresh | ✔ partial | catalog entry; entry-coupled |
| **regime-conditioned exit** | regime × hold | ✘ | **UNTESTED** — D61 SOFTEST: genuinely open |
| **sub-bar exit (4h decision on a 1d entry)** | finer exit clock | ✘ | **UNTESTED** — D61 SOFTEST |

**The exit verdict (D61, RWYB):** on daily breakouts, smart exit-timing **underperforms a dumb fixed hold**
(fair test Δ −0.037, wins 3/12 alts) — and the earlier "beats-null" was a **hold-length artifact** caught by the
no-skill fixed-hold control. Exit is a *capture-rate* axis (best-vs-worst exit per move, two-sided + held-out via
`exit_capture_proxy`), not a free alpha lever. **Genuinely open exit sub-axes:** take-profit, regime-conditioned,
sub-bar. Tool: `python src/strat/exit_capture_proxy.py` (selftest two-sided).

### 7c — the strat-vs-mechanical exit split (the flagged gap, now a first-class `family` tag)
The exit catalog carries a `family` tag so a discovery run can never silently test only one side
(the user's exact gap: *"strat-based exit vs mechanical exits"*). The split:

| family | meaning | members |
|---|---|---|
| **mechanical** | price/time-driven, no signal | fixed-horizon/time-stop, trailing (Chandelier, %-off-high), take-profit, triple-barrier |
| **strat** | signal/indicator/regime-driven | MA-cross/signal-flip, managed-RSI, regime-conditioned |
| **hybrid** | clock + signal | sub-bar exit |

The `coverage_report` in the discovery contract (below) reads this `family` from the registry — so a
campaign that tests only mechanical exits is **flagged**, not silently incomplete.

## §9 — Config canonicalization + the granularity ladder (search-space discipline)
Not a lattice slice but a **discipline every config-discovery run must apply**: near-identical configs
(**MA(28,29) ≈ MA(27,30) ≈ MA(28,30)**) inflate the search space and the multiple-comparisons count,
so you overfit to a noise-level parameter difference. The archived 9,900-pair MA sweep proved it (top-10
all clustered fast=25–31 / slow=29–33). It was handled *ad-hoc* before — hand-curated sparse grids with
ratio ≥ 2× — never a reusable step. Now formalized in `canonicalize_grid` (keep representatives that are
mutually ≥ `rel_tol` apart; report the honest **effective-N**).

**The granularity ladder (which grouping earns its degrees of freedom — all walk-forward, holdout-sealed):**
per-ASSET = **dead** (config = noise, `regime_dna_lab`) → per-CLUSTER = **not robust** (u10 +5pp did not
replicate at u50) → per-REGIME = **hurts** (adds losing DOWN-regime configs; regime-*gate* instead) →
per-CADENCE = **real + stable** (the one axis whose per-group optimum repeats across folds) →
**POOLED-per-cadence is the honest architecture**. (`MA_GRANULARITY_DISCOVERY` + `PER_CADENCE_CONFIG`.)

## §10 — The discovery preflight contract (how this registry is ENFORCED, not just listed)
[`src/framework/discovery_contract.py`](../src/framework/discovery_contract.py) is the **stage-03
(strat build) preflight** that *consumes* this stage-00 registry and makes coverage + canonicalization
**mechanical**, so the intelligence stops babysitting them:
- `coverage_report(declared, waivers)` — checks a run's declared axis coverage against the registries;
  any registry member that is **neither tested nor waived** is a WARN (a *silent* omission — the thing
  to avoid); an undeclared run FAILs. It does **not** force exhaustive testing — it forces *conscious
  declaration* (you waive what "we've moved from", so nothing is forgotten by accident).
- `canonicalize_grid(configs, rel_tol)` — collapses near-duplicate configs → mutually-separated
  representatives + honest effective-N.
- `preflight(run_decl)` — runs both; the stage-03 entry point. RWYB: `python src/framework/discovery_contract.py --selftest` (5/5) / `--demo` / `--canon "28,29 28,30 50,200"`.

## §8 — Cross-cutting lenses (the +row)
actor (institutional/retail/DeFi/MM), sector (L1/Meme/DeFi/AI…), hold-cadence, capital-velocity — archetype matrix
lives in [CHIMERA_FEATURE_DICTIONARY](CHIMERA_FEATURE_DICTIONARY.md); sector conditioning open.

---

## Honest cross-cutting notes (decision-useful, not just a list)
1. **Every axis here is a CONDITIONING/EXECUTION layer, not an alpha source.** The corpus is unambiguous: no axis
   below the signal axis manufactures edge on its own — they shape, gate, and harvest a setup that must already
   clear the entry+cost bar. The one that moved the needle is **regime-gating** (axis 5/7), and it only *preserves*
   (cuts the bear), it doesn't *generate*.
2. **The two belief-corrections (RWYB):** (a) **5m and 2h are not built** — finest time-bar is 15m, no 2h dir;
   (b) the exit dimension was *not* a dedicated catalog, just 6 words in a lattice row — now §7b.
3. **The genuinely-open cells** (lowest dead-list coverage, where a NEW study should look): §2 the alternative
   clocks (dollar/event/tick — the IC-ruling is untested at tick), §4 vol/crowding multi-axis regime routing,
   §7b take-profit + regime-conditioned + sub-bar exits. These are the strategy-side analogues of the factor
   registry's "underexploited slices."
4. **Look-ahead landmines on these axes:** Heikin-Ashi / Renko **repaint** (§1); full-sample regime/z-score leak
   (§4, G-AUDIT-011); triple-barrier on future returns (§7b). Every option must be causal-at-decision-bar.

Canonical entry point: this doc indexes axes 1/2/3/5/6/7/8; the signal axis is the
[FACTOR_REGISTRY](CANONICAL_FACTOR_REGISTRY_2026_06_11.md). Update both when an option is added — the lists are
extensible by design. Machine-readable: [`config/strategy_dimension_registry.yaml`](../config/strategy_dimension_registry.yaml).
