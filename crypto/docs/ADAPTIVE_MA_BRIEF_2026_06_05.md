# Adaptive Moving-Average Trading System — Build Brief (2026-06-05)

> **Mission.** Build an **adaptive moving-average** crypto trading system over the **u100** universe that
> dynamically captures market moves and **wins per trade**. "Adaptive" = every rolling window we compute
> **per-asset salient features** (realized volatility, trend strength, dispersion/cluster, etc.) and use them to
> **adapt the moving-average configuration per asset** (MA type, fast/slow lengths, bands, thresholds) so the
> system trades the *current* conditions of each asset.

## BOUNDED GOAL (r2, 2026-06-05 — user mandate, BINDING)
- **Target: 2–5%+ NET (after taker 0.0024) per-MOVE / per-trade expectancy** across the u100. This is the bar —
  per-trade expectancy, held-out, after cost. Not compound headline, not win-rate alone.
- **Hold time: a couple of HOURS to < 7 DAYS max.** This BOUNDS the cadence: 1d bars cannot express
  "couple-hours" holds → **prefer 4h bars** (couple-hours-to-days holds) as the primary cadence; 1h for the short
  end; 1d only reaches the multi-day end. Re-fit the adaptive config on a rolling window (week/month).
- **Anti-overfit is FIRST-CLASS.** The adaptation map has many DOF (feature buckets × MA configs) — it WILL overfit
  if unbounded. Maximally audit: fit the map on TRAIN+VAL only, test UNSEEN; keep the config space SMALL +
  justified; require the edge to survive the random-entry null (src/strat/firewall) AND positive_control's
  two-sided soundness AND multiple seeds/windows. A result that only works in-sample is a REFUTATION.

## Core idea (read carefully — don't anchor on regime)
- Assets trend and move in **every** market — so **regime is an INPUT, not the anchor**. Don't build a
  regime-classifier and stop. The point is to **adapt the MA config to whatever each asset is doing now** so we
  enter and exit its moves effectively, on any random day an opportunity appears.
- **Adaptation loop:** for each asset, over a trailing window (e.g. ~1 week / ~1 month — test both), compute
  rolling features → map them to an MA configuration → trade that config until the next re-fit. The system
  re-adapts as conditions change.
- **Entry:** an MA-cross / MA-band / breakout signal using the *adapted* config. **Exit:** test whether a
  **uniform exit policy** (e.g. opposite-cross, ATR-trail, time-stop) suffices across configs (the user suspects
  the exit may be the same after discovery — verify, don't assume).
- **Objective = per-trade edge across u100 + robust held-out compound return.** Per-bar IC is NOT the target.

## Use the EXISTING apparatus (do not rebuild)
- **Data:** `from src.pipeline.chimera_loader import ChimeraLoader; ChimeraLoader.load(sym, cadence)`. Universe at
  `config/universes/u100.yaml`.
- **Validation apparatus** in `src/strat/`: `firewall.py` (random-entry null / leakage firewall),
  `positive_control.py` (two-sided soundness — must accept a real edge AND reject a ghost/beta), `fill_model.py`
  (costs; **taker cost 0.0024**), `candidate_gate.py`, `battery.py`, `benchmark.py`. Run `src/strat/selftest_all.py`
  to confirm the apparatus is sound before trusting any number.
- **Splits:** honest train / val / **unseen** (the unseen segment is NEVER touched during development). No
  look-ahead: rolling features must use ONLY past data at each point (no full-sample standardization).

## Deliverables
1. **Feature computation** — per-asset rolling features (vol, trend, dispersion/cluster) with NO look-ahead.
2. **Adaptation logic** — features → MA config (documented mapping; can be rule-based or fit on TRAIN/VAL only).
3. **Entry + exit** — entry from the adapted config; exit tested (uniform vs adaptive).
4. **Honest backtest over u100** — per-trade win-rate + held-out compound return, **vs a cost-matched
   random-entry null AND a fixed-config MA baseline** (the adaptation must EARN its keep vs fixed).
5. **A short report** with every number RWYB-reproducible (the exact command that produced it).

## Success criteria (the bar)
A documented, reproducible adaptive-MA system whose **held-out, after-cost, per-trade edge beats both** (a) a
cost-matched random-entry null and (b) a fixed-config MA baseline — with the adaptation mechanism shown to help,
**no look-ahead**, and the result robust (multiple seeds / windows, not a single lucky split).

## −k falsifiers you MUST run (don't skip)
- Is the edge real or an artifact? Check **look-ahead** in the feature/adaptation step, **overfitting** the
  adaptation (does it survive on UNSEEN?), **survivorship** (u100 listing recency), **beta-confound** (is it just
  BTC?), and **cost** (does it survive taker 0.0024?). A refuted hypothesis is a valuable result — log it.

## Rules for the autonomous rigs
- Work ONLY in your assigned scoped dir (`experiments/adaptive_ma/<rig>/`). Keep the repo clean.
- **RWYB**: every claim verified by running real code on real u100 data. Never report an unrun number.
- **Do NOT commit / push / deploy / move capital** — the overseer (Claude) reviews and commits good work.
- Honest about nulls: if the adaptive edge doesn't beat the baselines, SAY SO — that's the finding.
