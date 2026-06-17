#!/usr/bin/env python3
"""resourcefulness.py -- the COGNITION-level self-correction layer: how Claude/LLMs predictably FAIL on nuanced/hard
problems, and the resourceful (hustler + scientific) moves that beat each failure. This is the piece the self-evolution
loop was missing -- it reflects on HOW CLAUDE THINKS, not just on the project artifacts.

WHY (user mandate 2026-06-06): "LLMs force framings (they can't get nuance). I want an MA strat; the instance fails on
MA, I add the oracle dimension, it helps a bit, but the framing STILL collapses because the LLM can't differentiate
'capture every candle with the oracle' from 'decompose a whole MOVE, the best adaptive-MA IS the oracle, reverse-
engineer it'. I'd expect the instance to GET THE SPIRIT of the ask and find a way using the provided tools/limits
(adaptive-MA only -> no other TIs) -- to be a HUSTLER, resourceful, robust + scientific. And this very weakness should
be something the self-evolution loop can come up with itself: ask how Claude usually works, how the problem
constrains/enables, refine it to use FULL (unconstrained) capability, work around/through/between to the solution.
Instances make a claim, test it mathematically, and GIVE UP -- meanwhile I have to remind them to decompose the IDEAL
and reverse-engineer it to a working model."

EXTENSIBLE BY DESIGN: when a NEW LLM-failure-mode is observed, add it to FAILURE_MODES -- that IS the self-evolution
loop improving its own cognition (monotonic). Read-only checker. No emoji (cp1252).

Usage:
  python scripts/autonomy/resourcefulness.py check "MA cross shows no signal at 4h, objective looks impossible"
  python scripts/autonomy/resourcefulness.py cognition     # the self-evolution-on-cognition meta-questions
  python scripts/autonomy/resourcefulness.py protocols
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class FailureMode:
    key: str
    symptom: str
    correction: str
    detect: str = ""     # regex on a claim/task that SUGGESTS this mode is in play
    example: str = ""


@dataclass(frozen=True)
class Protocol:
    key: str
    move: str
    worked_example: str = ""


# ---------------------------------------------------------------------------
# HOW CLAUDE/LLMs FAIL on nuanced/hard problems (+ the resourceful correction). EXTEND as new modes are found.
FAILURE_MODES: list[FailureMode] = [
    FailureMode(
        "framing_collapse",
        "Collapsed a NUANCED ask into ONE rigid interpretation (usually the literal-narrow one), then proved THAT fails.",
        "Enumerate >=2 framings; label literal-narrow vs spirit-resourceful; TEST THE RESOURCEFUL ONE FIRST. Same words "
        "can mean very different problems.",
        detect=r"",
        example="'use the oracle' -> read as 'predict every CANDLE with the oracle' (narrow, fails) instead of "
                "'decompose a MOVE; the BEST adaptive-MA that captures it IS the oracle; reverse-engineer it' (resourceful)."),
    FailureMode(
        "premature_give_up",
        "Made a claim, tested it mathematically ONCE, it failed, concluded 'impossible / no signal' and stopped.",
        "One test refutes ONE FRAMING, not the problem. Before giving up: re-frame, decompose the ideal, reverse-engineer, "
        "try the resourceful path. (Composes with the problem_framing anti-impossible rail.)",
        detect=r"no signal|impossible|unreachable|can.?t be done|gave up|not feasible|doesn.?t work|fail(s|ed)?\b|null result"),
    FailureMode(
        "literal_over_spirit",
        "Executed the LITERAL words and missed the INTENT.",
        "State the SPIRIT of the ask in one line; solve THAT with the stated tools/constraints. The user wants the "
        "OUTCOME, not a literal transcription.",
        detect=r"as (you |)asked|literally|exactly what"),
    FailureMode(
        "self_constraining",
        "Treated a constraint as ONLY a limit ('what I can't do'), missing that it also ENABLES a specific path.",
        "For each constraint, ask: what RESOURCEFUL path does THIS constraint open? A constraint narrows the search to a "
        "place where a clever move lives.",
        detect=r"can only|restricted to|limited to|not allowed|constraint",
        example="'adaptive-MA only' -> 'so no other TIs' (limit) MISSES 'so the best adaptive-MA IS my oracle/ceiling to "
                "reverse-engineer toward' (enabler)."),
    FailureMode(
        "constraint_relaxation_shortcut",
        "Stuck WITHIN a hard/binding constraint, gravitated to RELAXING/LIFTING the constraint as 'the answer' -- a "
        "shortcut, not a solution. (The inverse of self_constraining: there the constraint is under-used; here it is "
        "illegitimately discarded.)",
        "When the analysis concludes 'this only works if we drop constraint X', STOP -- that is usually a shortcut, not "
        "a result. Do the HARDER work WITHIN X (decompose the ideal achievable under X; the gap is the honest ceiling). "
        "Only the USER may relax a binding constraint; never propose it as the deliverable.",
        detect=r"if we (just |could |)(drop|relax|lift|remove|enable)|needs? (a |the |)short|long-short|the unlock is|"
               r"only works if|requires (relaxing|lifting|dropping)|exception to",
        example="finer-TF MA on crypto: every instance concluded 'true bear-complementarity needs a SHORT sleeve' and "
                "offered the long-only-exception as THE answer. User: that is the SHORTCUT every instance takes; the real "
                "work is WITHIN long-only (capital-preservation, not shorting). 2026-06-14."),
    FailureMode(
        "tool_underuse",
        "Used a fraction of available capability -- didn't construct the ideal, decompose it, or reverse-engineer.",
        "Refine the problem to use FULL capability: build the IDEAL (the oracle WITHIN the constraints), decompose its "
        "causal DNA, reverse-engineer a realizable model toward it. Work around / through / between the limits."),
    FailureMode(
        "verdict_over_construction",
        "Reached a refutation VERDICT ('de-risked beta / no alpha / rank doesn't transfer / exhausted') and STOPPED -- "
        "defaulted to researcher-writes-a-conclusion mode instead of trader/BUILDER-constructs-a-deployable-system-"
        "within-the-ceiling mode. Symptom: the USER keeps having to SUPPLY the constructive re-frame.",
        "After ANY refutation, do NOT stop at the verdict -- switch to BUILDER mode: given the ceiling is real, what is "
        "the most DEPLOYABLE + ROBUST system within it? GENERATE the constructive framings yourself: the stable OBJECT "
        "(the working BAND/region, not the #1), the SELECTION policy (rolling / walk-forward), REGIME-conditioning "
        "(e.g. faster MAs in chop), the ALL-WEATHER test (bear too, not just the bull where everything looks amazing), "
        "and SHOW THE RAW DATA (price charts) to seed hypotheses. The verdict is the FLOOR of the analysis, not its end.",
        detect=r"de-risked beta|no (alpha|edge)|does(n.?t| not) (translate|transfer)|exhausted|the ceiling|"
               r"nothing beats buy-?hold|consistent with the (ceiling|null)",
        example="MA 2020->2021: instances kept concluding 'config rank doesn't transfer, de-risked beta, done.' The "
                "USER supplied every constructive framing: find the working BAND, ROLLING-pick from it, REGIME-condition "
                "(faster in chop), test ALL-WEATHER (2022 bear) not just the bull, and 'show me the RAW charts to see if "
                "there are trends.' Those framings -- not the verdict -- are where the deployable system lives. "
                "2026-06-16 user: 'I don't understand why claude instances can't think of them.' (They CAN; they default "
                "to verdict-mode and stop. The fix: builder-mode is mandatory AFTER the verdict, not optional.)"),
]
FM_BY_KEY = {f.key: f for f in FAILURE_MODES}

# THE RESOURCEFUL MOVES (hustler + scientific). The default solving strategy for hard/ambiguous asks.
PROTOCOLS: list[Protocol] = [
    Protocol(
        "decompose_the_ideal",
        "Construct the IDEAL achievable WITHIN the constraints (the oracle = best-achievable strategy in the ALLOWED "
        "space, NOT an unconstrained perfect predictor). Decompose its causal DNA. Reverse-engineer a realizable model "
        "that approaches it; the gap to the ideal is your honest ceiling.",
        worked_example="adaptiveMA (constraint: MA only): the oracle is the BEST adaptive-MA config PER MOVE -- decompose "
        "a move, the best MA-overlay that captures it IS the per-move oracle. Reverse-engineer: which (length/type/"
        "adaptivity) params capture each move? Learn observable-state -> params. NOT 'predict every candle'."),
    Protocol(
        "enumerate_framings",
        "Before solving an ambiguous ask, list >=2 framings; label literal-narrow vs spirit-resourceful; default to the "
        "resourceful one that respects the SPIRIT and the CONSTRAINTS."),
    Protocol(
        "constraint_as_enabler",
        "For each stated constraint, name the resourceful PATH it opens, not just what it forbids."),
    Protocol(
        "spirit_first",
        "One-line the SPIRIT/intent of the ask; solve that with the allowed tools before touching the literal wording."),
]

# The self-evolution-ON-COGNITION meta-questions (the user's exact ask -- the loop reflecting on HOW CLAUDE WORKS).
META_QUESTIONS = [
    "How does Claude/an LLM usually FAIL on this KIND of problem? (framing-collapse? give-up-after-one-test? literal-over-spirit?)",
    "How does this problem CONSTRAIN me -- and what does that SPECIFIC constraint ENABLE (the path it opens)?",
    "Refine the problem so I use FULL (unconstrained) capability: construct the IDEAL, decompose it, reverse-engineer toward it.",
    "Am I being a HUSTLER -- working around / through / between the limits to a working model -- or am I about to quit on one refuted framing?",
]


def check(text: str) -> dict:
    t = (text or "")
    flagged = [{"mode": f.key, "symptom": f.symptom, "correction": f.correction, "example": f.example}
               for f in FAILURE_MODES if f.detect and re.search(f.detect, t, re.I)]
    # decompose_the_ideal is ALWAYS the resourceful default to offer on a hard/ambiguous ask
    return {"text": text, "flagged_failure_modes": flagged,
            "resourceful_protocols": [{"key": p.key, "move": p.move, "worked_example": p.worked_example} for p in PROTOCOLS],
            "meta_questions": META_QUESTIONS}


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else "cognition"
    if cmd == "check":
        rep = check(a[1] if len(a) > 1 else "")
        if "--json" in a:
            print(json.dumps(rep, indent=2)); return 0
        print(f"=== resourcefulness check: {rep['text']!r} ===")
        if rep["flagged_failure_modes"]:
            print("  LIKELY FAILURE MODE(S) IN PLAY:")
            for f in rep["flagged_failure_modes"]:
                print(f"    !! {f['mode']}: {f['symptom']}")
                print(f"       -> {f['correction']}")
                if f["example"]:
                    print(f"       e.g. {f['example']}")
        print("  RESOURCEFUL DEFAULT -- decompose the ideal + reverse-engineer:")
        print(f"    {PROTOCOLS[0].move}")
        print(f"    e.g. {PROTOCOLS[0].worked_example}")
        return 0
    if cmd == "protocols":
        for p in PROTOCOLS:
            print(f"- {p.key}: {p.move}")
            if p.worked_example:
                print(f"    e.g. {p.worked_example}")
        return 0
    # cognition (default): the self-evolution-on-cognition meta-questions + the failure-mode registry
    print("=== SELF-EVOLUTION on COGNITION -- ask these BEFORE concluding a hard problem is unsolvable ===")
    for q in META_QUESTIONS:
        print(f"  ? {q}")
    print("=== KNOWN LLM FAILURE MODES (extend this list when a new one is observed = the loop improving its cognition) ===")
    for f in FAILURE_MODES:
        print(f"  - {f.key}: {f.symptom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
