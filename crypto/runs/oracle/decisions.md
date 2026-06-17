# Oracle-decomposition decisions — the canonical methodology + verdicts (READ-FORWARD before oracle work)

> **Purpose:** the "read before deciding" lane for the oracle-decomposition method. READ THIS FIRST before constructing
> an oracle or decomposing a DNA — the method + its standing verdicts are here so a future instance does not re-derive
> them. Seeded 2026-06-06 from verified findings; append every new oracle decision with evidence.

## The method (decompose-the-ideal)
Treat ROI/per-trade targets as what an **ORACLE** (perfect foresight, WITHIN the constraints) attains; construct the
oracle, **decompose its causal DNA**, diffuse the noise, build a **capture-rate proxy** (cost-free, capital-free L2 KPI
= realized/available move within the signal-valid window), and reverse-engineer a realizable model toward it. The gap
to the oracle is the honest ceiling. Doc: `docs/ORACLE_DECOMPOSITION_2026_06_06.md`.

## Standing decisions (verdicts to honor)
- **The oracle objective is a 3-tier DESIGN VARIABLE** keyed on `min_move_net`: scalp (0% floor) / swing (3-8%) /
  position (15-30%+). SELECTION PROTOCOL: pick the oracle objective whose hold-time satisfies
  `median_hold_bars >= 2 x indicator_lag`, THEN decompose. A "no signal" verdict is only valid on a **lag-matched**
  objective. Spec: `docs/ORACLE_OBJECTIVE_TAXONOMY_2026_06_06.md`.
- **Floor→hold mapping (verified)**: at 4h the fair floor is EMA10→5% / RSI14→8% / MA10·20→10% / MACD→25%; at 1d
  everything slower than a breakout is unreachable within a 7-day hold cap (the position tier needs a `max_hold_days`
  parameter — an apparatus gap).
- **Believing "genuine" requires**: seed-robustness (stochastic models) + OOS→UNSEEN persistence, BOTH. A single
  firewall pass is insufficient (both caught false positives — e.g. BTC-1d GBM looked genuine, then OOS +58% →
  UNSEEN +0.1% = noise).
- **Event-study vs feature-classifier are different framings** — a rare signal (e.g. liquidation cascade) washes out as
  a feature but must also be tested conditionally; run both as the default.
- **Soundness-gate statistic**: MEAN for "collapse-on-average", p95 TAIL for "beat-the-null" — do not mix.

## Current standing finding
No verified active daily/4h long-only entry-timing alpha at the bar level (MA + orderflow + momentum + micro + liq, +
liquidation events + cross-asset lead-lag ALL null held-out); 3 avenues converge on sub-bar / HF. Robust = beta + yield.
The strategy phase inherits this; re-test as HYPOTHESES, not inherited facts.
