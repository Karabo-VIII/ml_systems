# Avenue Map — the exploration space (FOUNDATION; conclusions held OPEN)

> **Purpose:** the foundation maps EVERY avenue so the *next* phase (solving) can execute systematically.
> This is NOT a verdict on whether an edge exists. **Status legend:** `EXPLORED` · `PARTIAL` · `UNEXPLORED`
> · `KNOWN-HARD` (the starting premise — 4h/daily-LO). Per ground-zero rule: prior "hard/dead" reads are
> hypotheses to RE-TEST with the fixed apparatus, not facts. A strategy = a point in the 8-axis space below.
> Companion: [FOUNDATION](FOUNDATION_2026_06_04.md) (record), [APPARATUS_SPEC](APPARATUS_LOCKDOWN_SPEC_2026_06_04.md),
> [RETEST_PLAN](RETEST_PLAN_2026_06_04.md) (methodology).

## Axis 1 — Chart / bar type
| Type | What it's good for | Status | Data state |
|---|---|---|---|
| time 1d | regime/trend, low cost | EXPLORED | materialized, 104 assets |
| time 4h | swing | **KNOWN-HARD** (premise: cost-wall + beta) | materialized |
| time 1h / 15m | intraday | KNOWN-HARD (cost-wall, 0 cand @30bps — lightly) | 15m all / 30m 77 |
| dollar (coarse ~6676) | activity-clocked swing | PARTIAL (PEPE only) | materialized |
| dollar (fine ~75s) | micro-moves | PARTIAL (majors efficient per probe) | materialized |
| **dib** (dollar-imbalance) | informed-flow capture | **UNEXPLORED** (flow-surge died; capture-style untested) | raw 87 assets; chimera ONLY BTC/ETH/PEPE |
| **runs_tick** | momentum-shift | **UNEXPLORED** | raw 87; chimera EMPTY |
| **runs_volume** | volume-imbalance | UNEXPLORED/BROKEN (2023-only, bad calib) | raw 77, 2023 |
| **range** | clean trend/breakout (by design) | **UNEXPLORED** | raw 87; chimera ONLY BTC/ETH/PEPE |
| **adaptive_vol** | regime-scaled sampling | **UNEXPLORED** ("orphan compute") | raw 87; chimera EMPTY |
| Heikin-Ashi / Renko | smoothed trend | EXPLORATORY (PEPE only, off-pipeline) | sweep files only |
→ **Biggest unexplored substrate.** Foundation task: build the missing chimeras (dib/runs/adaptive_vol × universe).
> **Data-infra build plan + PROOF (2026-06-05):** the builder `src/pipeline/make_chimera_bars.py --assets <…> --bar-types {dib,runs_tick,runs_volume,range,adaptive_vol}` feature-enriches any bar type, **reusing the AUDITED dollar-chimera machinery (look-ahead-safe by inheritance: +1d frontier lag, causal targets, backward-asof bar-grain).** PROOF-BUILD ran (dry-run): PEPE `runs_tick` → 13,991 bars + 144 frontier cols, **1 ok / 0 fail** → the unexplored substrate is **buildable, de-risked.** Solving-phase job: run it across u100 for dib/range/runs_tick/adaptive_vol (compute-heavy but mechanical; `runs_volume` is 2023-only — fix calibration first). xd_* cross-asset base features are NOT computed per-alt-bar (deferred; daily frontier carries the bulk).

## Axis 2 — Resolution
1d EXPLORED · 4h KNOWN-HARD · 1h/15m cost-wall · dollar-coarse PARTIAL · fine-dollar/event-clock UNEXPLORED (selective capture untested) · tick INFEASIBLE under LO+spot+lev=1 (researched §22) + months-build.

## Axis 3 — Instrument
Per-asset over u10/u50/u100 (the TI×ASSET unit). PARTIAL: PEPE + a few majors tested; **most of u100 UNEXPLORED**. Caveat: survivorship (currently-listed only).

## Axis 4 — Signal / indicator (the 184 chimera features + classic TIs)
| Family | Role | Status |
|---|---|---|
| price-TI (MA/EMA/WMA/RSI/MACD/Bollinger) | STRUCTURE (trend frame) | standalone KNOWN-HARD; gated-structure OPEN |
| whale (`wh_*`, `norm_whale`) | GATE | PARTIAL (PEPE-only so far; non-poolable on memes) |
| s3 smart-money (`s3_smart_vs_retail`) | GATE | **UNEXPLORED** (the closest analog to the one real mechanism) |
| hawkes (`hbr_eta_*`) | GATE (flow asymmetry) | UNEXPLORED |
| liquidations (`liq_capitulation` …) | GATE (event) | UNEXPLORED as a capture trigger |
| basis / funding | GATE + carry | KNOWN-HARD as carry; UNEXPLORED as gate |
| macro (`etf_`, `stbl_`) | GATE (risk-on/off) | UNEXPLORED |
| LOB-proxy / book-depth | GATE (pressure) | reversion KNOWN-HARD; other roles UNEXPLORED |
| cross-asset / `xrel_` / TE | breadth / leader-follower | PARTIAL |
| WM signal | meta-labeler | standalone KNOWN-HARD (~3 OOM<cost); **meta-label-on-proven-gate UNEXPLORED** |

## Axis 5 — Regime
trend(SMA) / vol / breadth — as GATE (PARTIAL) and as a VALIDATION-SLICING axis (UNEXPLORED systematically). Ex-ante detection is the hard sub-problem (HMM look-ahead trap — §7-F).

## Axis 6 — Method
| Method | Defensible role | Status |
|---|---|---|
| static rules | the OG; prove edge simply first | PARTIAL (standalone-TI hard; gated OPEN) |
| dynamic (regime-adaptive) | gate by ex-ante regime | PARTIAL |
| ML | **meta-labeler on a proven gate** (not generator) | UNEXPLORED in defensible role; generator KNOWN-HARD |
| self-improving bot | decay-rotation over a validated sleeve library | UNEXPLORED (needs a library first) |
| WM | filter + simulation training-ground (AlphaZero-analog) | UNEXPLORED in defensible role; analog has simulator-fidelity problem |

## Axis 7 — Approach
per-asset specialist + combine (PARTIAL) · cross-sectional / breadth-pooled (long-only XS KNOWN-HARD; pooled-breadth lightly) · regime-gated portfolio (EXPLORED = the floor) · setup-capture / Benedict-style rotation (UNEXPLORED systematically — the user's stated style).

## Axis 8 — Entry / exit policy
trailing stop · time stop · fixed target · signal-flip · the Benedict exit layer (hard-stop + time-stop = the velocity engine). PARTIAL.

---

## The apparatus (ready to test ANY avenue) — built / staged this run
harness (cost lens via FillModel: taker default + maker sensitivity) · firewall (cost-matched random-entry null — the primary gate) · leak-probe (relative-twin verdict = supervised TODO) · benchmarks (single + portfolio) · block-bootstrap p05 · cost-feasibility · bear-inclusive holdout discipline. **TODO:** rebuild `battery.py` (consolidate these into one importable gate); build missing chimeras.

## Methodology (how to explore each avenue — per RETEST_PLAN)
`mechanism + falsifier (before backtest)` → `cost-honest backtest (taker + maker)` → `shift-leak probe` → `cost-matched random-entry null` → `robustness battery (10-seed p05>0, jk, n_eff, maxDD)` → `benchmark-excess per regime (incl. bear)` → `DSR@family-N` → `decouple → combine (pre-registered weights)`.

## EV-ranked exploration queue (for the NEXT phase — when solving begins)
1. **Build the missing info-driven-bar chimeras** (dib/runs/adaptive_vol × universe) — the genuinely-unexplored substrate; selective-capture-discipline on them.
2. **The TI×ASSET×REGIME grid with the 184 features as GATES** (s3-smart-money, hawkes, liq, macro) — mostly-untested gate space; the one real mechanism (whale-on-PEPE) suggests *gates*, not bare TIs, are where edges hide. **CONCRETE LEAD (surfaced 2026-06-05 by `discriminate` on PEPE, H=4): `liq_delta_z30` is same-sign across all 4 windows AND beats its own shuffle-null on UNSEEN** (whale_net beats-null on UNSEEN but isn't persistent) → a candidate GATE to harvest-test via the scan. (DISCRIMINATION ≠ HARVESTABILITY — it must still clear cost + firewall + battery.)
3. **ML-meta-label** on any §battery-surviving gate.
4. **Cross-sectional / breadth-pooled** with the FIXED sub-daily loader.
5. **self-improving / WM** — only after a validated sleeve library exists.

## Conclusions: HELD OPEN
The 4h/daily-LO corner is KNOWN-HARD (the premise). Everything else is a hypothesis to test with the fixed apparatus. **There is NO global "no-alpha" verdict** — that was a foundation-phase overreach (corrected 2026-06-05). The foundation EQUIPS exploration; it does not pre-empt it.
