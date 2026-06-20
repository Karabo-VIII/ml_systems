# CHARTER — binding intent for crypto work (Claude-owned; the user delegated this)

READ THIS FIRST, every cycle. The user does NOT ratify rules — Claude owns this file and updates it as intent
sharpens. It is the <4KB distillation of 35 mandates + 14 feedback files so a fresh instance starts WARM, not cold.

## THE NORTH STAR
Build a **repeatable, dynamic move-CAPTURE engine**: at any point, catch the MOVE on the chart with TIs
(trend / momentum / breakout / continuation — NOT per-candle prediction), across **the whole u50/u100 universe**,
all timeframes, adapting config to regime. The unit of trading is the **SETUP across a MOVE (multiple candles)** —
wait for a confirmed setup, enter, exit by policy. Rank by **WEALTH** (compound net return); also track capture-rate.

## THE BINDING SURFACE (a mandate is NOT attempted until ALL are touched)
- **UNIVERSE = u50/u100, ANY asset.** Never collapse to "BTC only" / "BTC is representative." A single-asset run is
  NOT the universe answer. Floor for a breadth claim: **>=20 distinct symbols** from `config/universes/u100.yaml`.
- **ALL TIMEFRAMES** {1d,4h,2h,1h,30m,15m} — never silently default one cadence; state cadence + why (user flagged 2x).
- **LOOKBACK WINDOW** (7d/14d) is a real executed parameter, not prose — a known prior weakness to fix.
- **USE THE WHOLE PROJECT** — extend existing infra (`src/strat/move_catch_book*.py`, `oracle_capture_lab.py`,
  chimera loader, `src/wealth_bot/harness.py`); do NOT write a fresh single-asset backtest from zero. Mine the
  dead-list (D01–D76) so we BUILD ON prior work and don't repeat killed veins.

## PERMANENT CONSTRAINTS (enablers, not walls)
- **LONG-ONLY + SPOT, lev=1. The short / long-short / market-neutral move is OFF THE TABLE — permanently.** Every
  instance gravitates to it; that is the cognitive SHORTCUT, not a solution. When the bear gap "needs a short," the
  honest answer is **capital-PRESERVATION within long-only** (cash/de-risk), never "enable the short." Only the USER
  may relax a binding constraint; never propose it as the deliverable.
- **Each constraint ENABLES a path.** Decompose the IDEAL achievable *within* the constraint (the oracle = best
  strategy in the ALLOWED space, NOT a perfect predictor), decompose its DNA, reverse-engineer toward it; the gap is
  the honest ceiling. (`scripts/autonomy/resourcefulness.py` — the cognition layer.)
- **IC / per-bar predictability is the WRONG lens, banned as a primary metric.** Judge by move CAPTURE-RATE + WEALTH.

## WAY OF WORKING
- **Claude MANAGES the project and makes the calls.** Deferring a revertible decision to the user is NOT allowed —
  decide, state reasoning, proceed; git is the revert net. Escalate ONLY for genuinely irreversible real-world actions.
- **Decompose the ideal; don't quit at a verdict.** One test refutes ONE FRAMING, not the problem. After ANY
  refutation, switch to BUILDER mode: the most deployable robust system *within* the ceiling. The verdict is the FLOOR.
- **A null is honest ONLY if the full surface was attempted** — >=20 u100 assets AND the window AND existing infra,
  proven by run-output ARTIFACTS (symbol columns in `runs/`, a git SHA), not prose. A null on a 1-asset / 1-TF shadow
  does NOT refute the u100 mandate.
- **Build first, evaluate honestly.** Gates: no-skill fixed-hold baseline, random-entry null, block-bootstrap p05,
  select on TRAIN/VAL not OOS. Discrimination != harvestability. Rank by net WEALTH, not Sharpe.
- **Deliver end-to-end, artifacts where the user looks** (`runs/periods/...`), nothing pending. Plain talk, no preamble.

## THE HONEST FALLBACK (when the surface is genuinely exhausted)
Long-only internal-data alpha may not exist — the standing premise, not a failure. The constructive default is
**capital-preservation within long-only + the external-listing-data frontier (Coinbase/Upbit) driven to DONE** — built
within the constraint, never handed back as "mandate dead."

## SPLIT CONVENTIONS (don't re-derive)
TI/MA capture: 2020=DEV, 2021=IRON, 2022=no-touch BEAR. Engine/WM: Convention B (TRAIN<=2024-05-15 / VAL / OOS /
UNSEEN 2026-05-22) — never touch OOS/UNSEEN in development.
