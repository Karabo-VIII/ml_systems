#!/usr/bin/env python3
"""problem_framing.py -- the BREADTH+DEPTH framing engine: turn a single task into the depth AND breadth questions an
expert would ask, with the project's standing lenses always at the back of the mind -- so the loop finds a solution
PATH (possibly different + better than asked) instead of tunnel-visioning the literal request.

WHY (user mandate 2026-06-06): "I give you a task; you ask depth AND breadth questions, solve iter-1, ask newer ones on
that feedback, and by the end find a solution path different+better than what I asked. If I say 'find me an MA strat'
you should ALREADY have jolted out of single-candle trading, out of single-IC, thought like a trader/institution
(setups, their ROI, trade-types), thought about crypto's nature -- all in scope or at the back of the mind. And NEVER
declare 'the objective is impossible' -- validate the numbers first." This mechanizes that so the human stops having
to inject breadth by hand (n+-k alone is a LOCAL neighborhood search = depth-biased; this adds the orthogonal axes).

Two layers:
  LENSES -- always-on priors the user kept having to inject (jolt-regexes flag a task that's already narrow).
  AXES   -- the orthogonal breadth dimensions of the problem; a coverage grid marks each explored / NOT EXPLORED, and
            the loop seeds a forced `diverge` node on the highest-EV NOT-EXPLORED axis (breadth becomes mechanical).
Plus the ANTI-IMPOSSIBLE RAIL: a checklist that MUST pass before any "unreachable/impossible" verdict.

Sources consolidated (not duplicated): MEMORY.md founding framing, docs/MARKET_STRATEGY_ARCHETYPES.md,
src/narrate/crypto_context.py, docs/AVENUE_SPECS_2026_06_05.md. Read-only. No emoji (cp1252).
Usage: python scripts/autonomy/problem_framing.py "find me an MA strat" [--json]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lens:
    key: str
    prior: str                 # the standing consideration to hold
    jolt_if: str = ""          # regex: if the TASK matches, it is already framed narrowly -> jolt
    source: str = ""


@dataclass(frozen=True)
class Axis:
    key: str
    question: str              # the breadth question this axis forces
    options: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# LENSES -- the "back of the mind" priors. jolt_if flags a task ALREADY narrowed on the wrong frame.
LENSES: list[Lens] = [
    Lens("unit_of_trading", "Trade SETUPS across MOVES (multi-candle); enter on a confirmed setup, exit by policy.",
         r"single.?candle|per.?candle|per.?bar|each bar|candle.by.candle", "MEMORY.md founding"),
    Lens("metric", "Optimize held-out COMPOUND return (robust: seed-stable, OOS->UNSEEN). IC/per-bar predictability "
         "is BANNED as a PRIMARY metric -- it survives only as a within-model diagnostic.",
         r"\bIC\b|information coefficient|per.?bar predict|sharpe.only", "MEMORY.md (IC banned)"),
    Lens("actor_mindset", "Think like a TRADER / INSTITUTION: a SETUP, its expected ROI per trade, the trade-TYPE "
         "(scalp/swing/trend/event), capacity, risk-of-ruin -- not a per-candle coin-flip harvester.",
         "", "docs/AVENUE_SPECS + trading-philosophies"),
    Lens("crypto_nature", "Crypto is its OWN market: 24/7, perp FUNDING + basis as first-class positioning, "
         "LIQUIDATION reflexivity (cascades), BTC-beta dominance, venue fragmentation, retail leverage.",
         "", "src/narrate/crypto_context.py"),
    Lens("archetype_fit", "Pick the right MODE for instrument+cadence (PRIMARY = swing + breakout; compose "
         "intraday-momentum / position-trend / event-driven; AVOID scalping + HFT = the per-candle/infra trap; "
         "mean-reversion only in confirmed ranges).", "", "docs/MARKET_STRATEGY_ARCHETYPES.md"),
    Lens("explore_all_dims", "Explore ALL dimensions FRESHLY (chart/bar-type, timeframe, instrument, indicator, "
         "regime, entry/exit policy). Prior 'exhausted / impossible / ceiling ~X%' verdicts are HYPOTHESES to "
         "re-test, NOT inherited facts.", r"exhausted|impossible|unreachable|can.?t be done|ceiling", "MEMORY.md founding"),
    Lens("validate_not_declare", "NEVER conclude 'objective impossible/unreachable' from a single narrow attempt. "
         "FIRST validate the real numbers (per-day movers, the ORACLE ceiling for a lag-matched objective) and "
         "re-frame across the breadth axes. Narrow framing masquerades as impossibility.", "", "user mandate 2026-06-06"),
    Lens("entry_exit_split", "ENTRY signal and EXIT policy are SEPARATE decomposable domains -- don't conflate; a "
         "weak result on one is not a verdict on the other.", "", "MEMORY.md / narrate"),
]

# AXES -- the orthogonal breadth dimensions of a STRATEGY problem (the things forgotten when tunnel-visioning).
STRATEGY_AXES: list[Axis] = [
    Axis("cadence", "Which TIMEFRAME / hold-horizon? (the move's duration, not the bar)",
         ["1m", "5m", "15m", "1h", "4h", "1d", "3d", "1w"]),
    Axis("chart_type", "Which BAR CONSTRUCTION? (information geometry changes with it)",
         ["time", "dollar", "volume", "range", "renko", "tick"]),
    Axis("instrument", "Which INSTRUMENT / scope?", ["spot", "perp", "single-asset", "universe", "cross-section"]),
    Axis("indicator_family", "Which SIGNAL FAMILY? (don't stay in one)",
         ["MA/EMA", "RSI", "MACD", "Bollinger", "orderflow", "liquidation", "funding/basis", "cross-asset lead-lag"]),
    Axis("regime", "Which REGIME conditioning?", ["trend", "mean-rev", "volatile", "compressed", "by-BTC-beta"]),
    Axis("entry_vs_exit", "ENTRY signal, EXIT policy, or both? (separate domains)",
         ["entry-signal", "exit:trailing", "exit:fixed", "exit:vol-scaled", "both"]),
    Axis("sizing", "Which SIZING?", ["fixed", "vol-target", "fractional-Kelly", "regime-scaled", "conviction-weighted"]),
    Axis("cost_model", "Which COST reality?", ["maker", "taker", "p_fill", "slippage", "funding-carry"]),
    Axis("unit_of_trade", "SETUP-across-move (correct) vs single-candle (BANNED)?", ["setup-across-move"]),
    Axis("objective_metric", "Which OBJECTIVE? compound (target) / sharpe (secondary) / IC (diagnostic-only)",
         ["compound-return", "sharpe(secondary)", "IC(diagnostic)"]),
    Axis("actor_lens", "From WHOSE seat? (changes the trade-type + capacity)",
         ["active-trader", "swing-desk", "institution", "market-maker"]),
    Axis("oracle_objective", "Which ORACLE objective (scalp/swing/position) -- pick BEFORE concluding too-slow/fast?",
         ["scalp(0% floor)", "swing(3-8%)", "position(15-30%)"]),
]

ANTI_IMPOSSIBLE_RAIL = [
    "Did you VALIDATE the real numbers (per-day movers >=X%, oracle capture ceiling for a LAG-MATCHED objective)?",
    "Did you RE-FRAME across the breadth axes (other cadence / chart-type / instrument / indicator / actor-lens)?",
    "Did you fix the ORACLE OBJECTIVE (scalp vs swing) BEFORE calling an indicator too-slow/too-fast?",
    "Is this genuine impossibility, or NARROW FRAMING wearing an 'impossible' label? (default: narrow framing)",
    "Did you test ENTRY and EXIT as SEPARATE domains before a combined verdict?",
]


def frame(task: str, explored: list | None = None, axes: list[Axis] | None = None) -> dict:
    axes = axes or STRATEGY_AXES
    explored = set(explored or [])

    # case-INSENSITIVE jolt match on the original task (so "IC", "Single Candle", etc. all fire).
    jolts = [{"lens": L.key, "prior": L.prior, "why": f"task matches /{L.jolt_if}/", "source": L.source}
             for L in LENSES if L.jolt_if and re.search(L.jolt_if, task or "", re.I)]
    back_of_mind = [{"lens": L.key, "prior": L.prior, "source": L.source} for L in LENSES]
    grid = [{"axis": a.key, "question": a.question, "options": a.options,
             "status": "explored" if a.key in explored else "NOT EXPLORED"} for a in axes]
    not_explored = [g for g in grid if g["status"] == "NOT EXPLORED"]

    # seed nodes: n (the literal task) + a -k falsifier + a +k generalization + a forced DIVERGE per top NOT-EXPLORED axis
    seeds = [
        {"id": "n", "kind": "build", "ev": 0.85, "task": f"iter-1: {task} (apply ALL lenses; report numbers, not verdicts)"},
        {"id": "k-", "kind": "verify", "ev": 0.8,
         "task": f"-k FALSIFIER: is the framing of '{task}' sound? (look-ahead, leakage, beta-confound, narrow-frame)"},
        {"id": "k+", "kind": "diverge", "ev": 0.7,
         "task": f"+k GENERALIZATION: the general class of '{task}' -- what is the better-framed version?"},
    ]
    for i, g in enumerate(not_explored[:4]):
        seeds.append({"id": f"breadth-{g['axis']}", "kind": "diverge", "ev": round(0.65 - i * 0.05, 2),
                      "task": f"BREADTH ({g['axis']}): {g['question']} -> try an un-tried option"})

    return {"task": task, "jolts": jolts, "standing_lenses": back_of_mind, "coverage_grid": grid,
            "not_explored": [g["axis"] for g in not_explored], "seed_nodes": seeds,
            "anti_impossible_rail": ANTI_IMPOSSIBLE_RAIL}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    task = args[0] if args else "find me an MA strat"
    rep = frame(task)
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"=== FRAMING: {task!r} ===")
    if rep["jolts"]:
        print("  JOLTS (task is already framed narrowly -- widen NOW):")
        for j in rep["jolts"]:
            print(f"    !! {j['lens']}: {j['prior']}  [{j['why']}]")
    print("  STANDING LENSES (hold all of these):")
    for L in rep["standing_lenses"]:
        print(f"    - {L['lens']}: {L['prior']}")
    print(f"  BREADTH COVERAGE -- NOT EXPLORED: {', '.join(rep['not_explored']) or '(none)'}")
    print("  SEED NODES (depth n+-k + forced breadth diverge):")
    for s in rep["seed_nodes"]:
        print(f"    [{s['kind']:7} ev={s['ev']}] {s['task']}")
    print("  ANTI-IMPOSSIBLE RAIL (ALL must pass before any 'impossible' verdict):")
    for r in rep["anti_impossible_rail"]:
        print(f"    [] {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
