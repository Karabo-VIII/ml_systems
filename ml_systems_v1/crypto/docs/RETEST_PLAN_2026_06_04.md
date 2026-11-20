# Ground-Zero Re-Test & Discovery Plan (2026-06-04)

> **Status:** the actionable "approaches we'll use," derived from the research campaign in
> [`FOUNDATION_2026_06_04.md`](FOUNDATION_2026_06_04.md). Ground zero: prior conclusions are verified
> hypotheses, not inherited facts. Constraint: LONG-ONLY + SPOT + LEV=1. Objective: WEALTH under robustness.
> **Read with:** [`APPARATUS_LOCKDOWN_SPEC_2026_06_04.md`](APPARATUS_LOCKDOWN_SPEC_2026_06_04.md) (the instrument).

## What the research established (so the plan isn't re-litigating it)
- **Negatives are trustworthy, positives are not.** The 18-item dead-list is structurally dead (zero re-open, verified); the apparatus bugs were false-POSITIVE generators. So we do NOT re-mine the daily/4h/dollar veins; we DO rebuild trust before believing any positive.
- **The only robust thing found is beta + yield (~13–22% honest forward CAGR), and it is NOT alpha** — buy-and-hold beats it in bulls; its value is drawdown-control.
- **No verified active-trading alpha exists at daily/4h/dollar resolution.** That's the honest-failure surface, not a framing to soften.

## Phase 0 — Fix the apparatus (PRECONDITION, fork-independent, no capital)
Implement [`APPARATUS_LOCKDOWN_SPEC`](APPARATUS_LOCKDOWN_SPEC_2026_06_04.md): LD-1 FillModel (taker default + maker p_fill sensitivity), LD-2 shift-sensitivity leak probe, LD-3 working DSR/family-N gate, LD-4 cost-matched random-ENTRY null (THE gate), LD-5 bear-inclusive holdout + fixed sub-daily loader + pre-registered pooling weights. **Until this is done, no number is trustworthy.** This is the keystone — build it first regardless of which fork is chosen.

## Phase 1 — Re-validate the HONEST BASE (fork-independent; you need it either way)
Establish the trustworthy floor and the benchmark every active strategy must beat:
- Re-run the regime-gated diversified book under the FIXED apparatus: taker cost, bear-inclusive holdout, **β-matched benchmark** (hold-X%-BTC-rest-cash at the strategy's average exposure), 10-seed p05, honest (not optimistic) yield.
- Output: the real forward CAGR (expect ~13–22%, NOT 26%), the honest beta/yield/alpha decomposition, and the **benchmark curve** that gates everything downstream.
- Decision artifact: "is the disciplined-beta floor worth deploying on its own?" — this IS Fork A.

## The fork (A/B/C — your decision; the plan supports all three)
| Fork | What it is | Honest odds | Cost |
|---|---|---|---|
| **A — bank the beta floor** | Deploy the Phase-1 regime-gated beta+yield book as the base. | HIGH it's real; but it's beta, ~13–22%, beaten by buy-hold in bulls. | Low. Ready after Phase 1. |
| **B — fine-resolution capture frontier** | Mine the LEAST-explored existing substrate: information-driven bars (`dib`/`runs_tick`/`runs_volume`/`adaptive_vol` + fine dollar ~75s) with the founding capture-discipline (selective, big-move, time/trail-stop). | **LOW (sharpened 2026-06-05 by the fine-res probe).** Majors are EFFICIENT at fine res (lag-1 ≈ −0.013, no structure) → no-man's-land. Memecoins show structure+fuel (PEPE lag-1 −0.10, 17% bars clear cost in 1 bar) BUT that −0.10 is most likely bid-ask-bounce noise (a cost-trap, not edge; prior D3/D7 = knife-catch). Cost-wall is the central risk (1h/15m died at 30bps). | Med (build missing chimeras). |
| **C — tick / microstructure system** | True sub-second / LOB / MEV-aware stat-arb. | **INFEASIBLE under LO+spot+lev=1 (verified 2026-06-05):** every tick edge needs a forbidden tool (shorts/leverage/MM-status/colocation) or is competed away; NO usable tick data on disk, V20 unbuilt → months-long build. | HIGH + likely a dead end under the constraint. |
| **D — relax the CONSTRAINT (the real lever)** | Long-short / modest perps on EXISTING daily/4h data. | **HIGHEST-EV path to active wealth** (research: ≈2× returns, half DD on same signals); the binding limit was the constraint, not the resolution. Ruin-risk = a risk conversation. | LOW infra; requires the user to relax the LO+spot+lev=1 mandate. |

## Phase 2-B — the fine-resolution capture frontier (if Fork B)
1. **Build the missing chimeras** (currently only BTC/ETH/PEPE have dib/range; runs/adaptive_vol are EMPTY): enrich `dib`/`runs`/`adaptive_vol` across the universe on the fixed pipeline. (SOTA caveat: info-driven bars give only marginal stationarity gains — build for the SELECTIVITY/event-clock benefit, not a stationarity miracle.)
2. **Mine with capture-discipline, not discrimination-mining** (the cell-grid is a verified leak surface): the unit is a SELECTIVE setup that captures a big info-driven move and clears taker cost with margin; few trades, time/trail-stop exit, cut-fast. The cost-clearing rule (move ≫ 0.24%) is the gate.
3. **Honest cost-wall test FIRST:** before any optimization, measure whether ANY event-clocked selective setup clears taker cost on held-out. If it dies like 1h/15m did, STOP and report — don't optimize a sub-cost edge.
4. Every candidate runs the full pipeline (below) on a bear-inclusive holdout with family-N over the entire grid.

## Phase 2-C — the tick/representation path (if Fork C; separate track)
The WM/representation frontier (IC>0.10 plausible only at tick per the corpus + SOTA). New data acquisition + V20 architecture. Off the critical path for this constraint set; flagged as the high-cost, SOTA-supported option for if/when A and B don't satisfy the objective.

## Universal methodology (every candidate, every fork)
`mechanism stated + falsifier (BEFORE backtest)` → `cost-honest backtest (taker + maker sensitivity)` → `shift-sensitivity leak probe` → `cost-matched random-ENTRY null (beats-null necessary, not sufficient)` → `robustness battery (10-seed p05>0, jk2&jk3>0, maxDD<30%, n_eff≥15)` → `benchmark-EXCESS per regime (beat the β-matched base, in EACH regime incl. a bear)` → `DSR/Holm at TRUE family-N incl. aggregation DoF` → only then `sizing` → `paper` → `live`. **Decouple→combine** (find standalone survivors, pool the decoupled ones AFTER; pre-register pooling weights before touching the holdout). **Capital velocity** (return-per-deployed-capital-day + per-calendar-day) is a first-class metric. ML only as a meta-labeler on a proven gate; never a generator. WM gated by OOS IC/ShIC, never a live policy.

## Decision gates (when to switch forks)
- Phase 1 base CAGR robust + acceptable → **Fork A deployable** (bank it; optionally run B as research in parallel).
- Phase 2-B cost-wall test FAILS (no selective setup clears taker cost) → **B is dead; fall back to A**, escalate ambition only via C.
- Phase 2-B yields a battery+DSR+bear-holdout survivor → real active satellite; size it ON TOP of the A base.
- A satisfies neither the user's ambition nor wealth-vs-buy-hold → the honest answer is **C (tick) or a constraint change (leverage/shorts — a separate risk conversation)**, not more daily-bar mining.

## NOT on the path (verified closed — do not spend compute here)
The entire dead-list D1–D18 (standalone TI mining, exo-conditioner discrimination, order-book-flow reversion, funding carry, pairs, vol-climax, breakout, ML-as-generator, RL/PPO, within-cluster relval, TSMOM, per-asset selection-leak, WM-as-signal, XS-momentum standalone & as-filter, flow-surge, regime-cells, DOGE-whale). Re-opening any requires a NEW mechanism + the fixed apparatus, not a re-run.
