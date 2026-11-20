# Trader post-mortems — refuted veins + lessons (READ-FORWARD before any strategy attempt)

> **Purpose:** the "read before deciding" lane for the strategy/discovery layer. READ THIS FIRST before mining any
> setup — do NOT re-pay for a lesson already learned, and do NOT re-mine a vein already REFUTED (cross-ref the
> mechanical dead-catalog: `python scripts/autonomy/hypothesis_register.py dead`). Seeded 2026-06-06 from the verified
> session findings; append every new refutation with its evidence.

## REFUTED veins (do NOT re-mine without a genuinely new framing)
- **MA-DNA entry-timing, ALL cadences** (1d / 4h / 1h / dollar / range bars): **0/14 genuine, 0/74 Holm** held-out. The
  null traces to the **instrument class**, not the cadence or parameters — the realizable capture of a lagging MA is
  bounded by the oracle move's hold-time vs the indicator's response lag. *Do not re-run the MA family hoping a new
  cadence saves it.*
- **SOL / MA / 4h / breakout**: 0/77 beat-null raw; **beta-confounded** (BTC-beta in disguise). Residualize beta before
  claiming idiosyncratic edge — shipping beta as alpha is the canonical crypto false-positive.
- **Per-candle IC framing**: BANNED as a primary objective. IC measures per-bar information, which is **noise** for a
  multi-candle SETUP. Prior instances repeatedly anchored on IC and wrongly concluded "no signal." The unit of trading
  is a **SETUP across a MOVE**; optimize held-out **COMPOUND** return.

## LESSONS (transferable, mechanism-level)
- **The oracle objective (scalp / swing / position) is a load-bearing DESIGN VARIABLE** — fix it (via a per-move net
  floor `min_move_net`) BEFORE concluding an indicator is too-slow/too-fast. A no-floor max-capture oracle silently
  becomes a scalper (~2-bar holds) — an unfair test for a trend instrument.
- **A single firewall pass is NOT enough**: seed-robustness (stochastic models) + OOS→UNSEEN persistence are MANDATORY
  before believing any "genuine" — both caught false positives.
- **A soundness gate must use the statistic matching its question**: MEAN for "collapse-on-average", p95 TAIL for
  "beat-the-null". Mixing them produced false APPARATUS-leak alarms.
- **NEVER declare "impossible" from one narrow attempt** — validate the real numbers (per-day movers, the lag-matched
  oracle ceiling) + re-frame across the breadth axes first (`scripts/autonomy/problem_framing.py`).
