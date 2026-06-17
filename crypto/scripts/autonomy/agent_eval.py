#!/usr/bin/env python3
"""TRAJECTORY-EVAL / OBSERVABILITY / REGRESSION harness for the LangGraph metaop loop.

This is the COMPLEMENTARY eval layer to scripts/autonomy/eval_harness_run.py. That harness scores the
FITNESS (mechanical solve_rate of the WORKER+VERIFIER on a benchmark -- "did the outcome verify"). THIS
harness scores the TRAJECTORY: the PATH the loop took and the CALIBRATION of its judge -- by reading the
JSONL traces the live loop already emits (harness/metaop/graph.py `_trace`). It answers questions the
fitness number cannot:

  - JUDGE CALIBRATION (panel-vs-verdict agreement): when a judge event carries BOTH a final verdict AND an
    LLM 'votes' panel, how often does the LLM-panel-MAJORITY match the loop's final verdict? HONEST SCOPE
    (verified on the real traces): in the current trace schema the MECHANICAL-verifier judges carry NO votes
    panel and the LLM-PANEL judges carry no mechanical result -- the two NEVER co-occur on a node. So this
    metric is NOT "LLM-vs-mechanical-ground-truth"; it is agreement between the naive panel majority and the
    loop's actual decision POLICY (>50% threshold + the H3 unverified-pass downgrade), which produces genuine
    disagreements (e.g. a pass-mode panel downgraded to 'inconclusive'). It is a real calibration signal --
    "is the panel's naive read consistent with the loop's decision rule" -- NOT a check against ground truth.
    TRUE LLM-vs-mechanical-ground-truth calibration requires a one-line INSTRUMENTATION upgrade in
    harness/metaop/graph.py: co-emit BOTH the LLM panel AND the mechanical verify result on the SAME judge
    event (today they are mutually exclusive). Flagged as TODO. The SOTA assessment
    (docs/AUTONOMY_HARNESS_SOTA_ASSESSMENT.md) found observability/eval is the field-wide-MISSING axis; this
    harness computes the panel-vs-verdict number and names the exact gap to the deeper ground-truth one.
  - CONVERGENCE: does open_left trend DOWN across reflect cycles, or sprawl / stall?
  - SELF-IMPROVEMENT (Voyager monotonicity): is every mechanical PASS harvested into the skill library?
  - FAN-OUT / HITL / REPLANNER churn: the orchestration-shape metrics.

It produces (a) a per-trace metric dict, (b) a cross-corpus observability REPORT (leaderboard + the
aggregate judge-calibration + a failure-mode histogram), and (c) a REGRESSION suite pinned to golden
bands DERIVED FROM the real traces (a harness change that degrades trajectory quality exits 2).

RWYB / firewall: --selftest synthesizes a HEALTHY trajectory and a DEGRADED one and ASSERTS the score
DISCRIMINATES (high >> low). An eval that cannot reject a bad trajectory is vacuous. No emoji (cp1252).
Read-only over traces; stdlib only; no look-ahead (each metric is computed from the events as ordered in
the trace, never from a future event informing a past one).

Trace schema (VERIFIED against harness/metaop/graph.py _trace calls):
  traces/<run_id>.jsonl, one JSON object per line: {"t": <unix_ts_float>, "event": <name>, ...data}
  plan        {seeded:int, expert_mode:bool, channel:str}
  dispatch    {ran:[...], parallel:int, experts:{...}, parked_for_approval:[...]}  OR {runnable:0, parked_for_approval:[...]}
  judge       {node, verdict:"pass"|"refuted"|"inconclusive", votes:[...]?, mechanical:bool?, evidence_type?, k?}
  reflect     {cycle:int, adjacent:int, open_left:int, status:str, refuted_left?, done?, stall_cycles?}
  route       {to:str, reason:str, ...}
  replan      {reason, pruned:[...], kept:[...], added:[...]}
  replan_done {replan_count, frontier_after, open_after}
  harvest     {node, verify_cmd}  OR  {node, error}
  record      {node, verdict}  OR  {node, error}
  frame / recall / plan_critique {...}

CLI: --trace <file> | --aggregate <dir> | --regression | --selftest   (default: --aggregate runs/autonomy/traces)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (.../ml_systems)
DEFAULT_TRACE_DIR = ROOT / "runs" / "autonomy" / "traces"
GOLDEN_DIR = ROOT / "tests" / "agent_eval_golden"
GOLDEN_JSON = GOLDEN_DIR / "golden.json"

__contract__ = {
    "kind": "observability_eval_harness",
    "inputs": [
        "JSONL trajectory traces emitted by harness/metaop/graph.py _trace (traces/<run_id>.jsonl)",
        "golden fixtures (real traces copied to tests/agent_eval_golden/) + golden.json pinned bands",
    ],
    "outputs": [
        "per-trace metrics dict (judge calibration, convergence, harvest monotonicity, fan-out, quality)",
        "cross-corpus observability report (leaderboard + AGGREGATE judge-calibration + failure-mode histogram)",
        "regression verdict (exit 2 if any golden metric leaves its pinned band)",
        "two-sided selftest verdict (exit 2 if the eval fails to discriminate healthy vs degraded)",
    ],
    "invariants": [
        "COMPLEMENTARY to eval_harness_run.py: that scores FITNESS (outcome solve_rate); this scores the "
        "TRAJECTORY (path quality + judge calibration). Do not conflate the two.",
        "JUDGE_CALIBRATION = agreement(LLM-majority(votes), final verdict) over judge events that have BOTH; "
        "null when no judge event carries a votes panel (not determinable).",
        "robust to malformed/partial JSONL lines (skip + count n_bad); never raises on a bad trace.",
        "no look-ahead: every metric reads events in trace order; no future event rewrites a past metric.",
        "two-sided: --selftest must ACCEPT a synthetic healthy trajectory AND REJECT a degraded one (margin).",
        "golden bands are DERIVED from the real traces (pinned by --regression --update), never invented.",
        "read-only over traces; stdlib only; no emoji in any output (Windows cp1252).",
    ],
}

# ----------------------------------------------------------------------------------------------------------
# composite trajectory_quality weights (documented; sum of positive weights = 1.0, stall is a penalty).
# Rationale: the four pillars the SOTA checklist names -- convergence (does it close work), calibration
# (does the LLM panel agree with the loop's decision VERDICT -- panel-vs-policy, NOT ground truth; see the
# JUDGE CALIBRATION note in the module docstring), harvest (Voyager monotonicity), solved (terminal reached) --
# plus an explicit STALL penalty so a thrashing run scores BELOW a clean one even if it eventually solves.
# A null calibration (no votes panel at all) is treated as NEUTRAL (0.5) so a mechanical-only run is not
# punished for lacking an LLM panel; a PRESENT-but-disagreeing panel IS punished (that is real miscalibration).
QUALITY_WEIGHTS = {
    "convergence": 0.30,   # open_left trends down / final open_left low
    "calibration": 0.25,   # LLM-majority agrees with verdict (neutral 0.5 when no panel exists)
    "harvest":     0.20,   # mechanical passes are harvested (Voyager monotonicity)
    "solved":      0.25,   # terminal 'solved' reached
}
STALL_PENALTY = 0.30       # subtracted (scaled by stalled-cycle fraction) -- thrashing is worse than idle


# ==========================================================================================================
# 1. TRAJECTORY PARSER
# ==========================================================================================================
def load_trajectory(path):
    """Parse a <run_id>.jsonl trace into a structured trajectory.

    Returns a dict:
      run_id, path, events (ordered, valid only), n_bad (malformed/partial lines skipped),
      cycles (events grouped by reflect-cycle boundary: each reflect event closes a cycle), by_event (counts).
    Robust to malformed/partial lines -- a half-written final line (the live loop appends) is skipped, not fatal.
    """
    p = Path(path)
    events = []
    n_bad = 0
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"run_id": p.stem, "path": str(p), "events": [], "n_bad": 0, "cycles": [],
                "by_event": {}, "read_error": f"{type(e).__name__}: {e}"}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            n_bad += 1
            continue
        if not isinstance(obj, dict) or "event" not in obj:
            n_bad += 1
            continue
        events.append(obj)

    # group events into cycles: a 'reflect' event terminates the cycle that accumulated before it; any
    # trailing events after the last reflect form a final (open) cycle. This mirrors the loop's own cadence
    # (plan/recall -> dispatch -> judge -> reflect). No look-ahead: we only close a cycle when its reflect
    # is reached, never by peeking forward.
    cycles = []
    cur = []
    for ev in events:
        cur.append(ev)
        if ev.get("event") == "reflect":
            cycles.append(cur)
            cur = []
    if cur:
        cycles.append(cur)

    by_event = Counter(ev.get("event") for ev in events)
    return {"run_id": p.stem, "path": str(p), "events": events, "n_bad": n_bad,
            "cycles": cycles, "by_event": dict(by_event)}


def _norm_verdict(v):
    """Normalize a verdict/vote token to lowercase canonical. The live loop emits lowercase
    ('pass'/'refuted'/'inconclusive'); the SOTA prompt referenced 'PASS'/'FAIL' -- accept both."""
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    if v in ("fail", "fa", "reject", "rejected"):
        return "refuted"
    return v


def _majority(votes):
    """LLM-panel majority label (mode). Ties -> the most 'severe' present (refuted > inconclusive > pass)
    is NOT assumed; on an exact tie we return the first-most-common deterministically so the metric is stable."""
    norm = [_norm_verdict(v) for v in votes if _norm_verdict(v) is not None]
    if not norm:
        return None
    return Counter(norm).most_common(1)[0][0]


# ==========================================================================================================
# 2. SCORE A SINGLE TRAJECTORY
# ==========================================================================================================
def score_trajectory(traj):
    """Compute the trajectory metrics dict. Pure function of the parsed trajectory (no I/O, no look-ahead)."""
    events = traj["events"]
    judges = [e for e in events if e.get("event") == "judge"]
    reflects = [e for e in events if e.get("event") == "reflect"]
    dispatches = [e for e in events if e.get("event") == "dispatch"]
    harvests = [e for e in events if e.get("event") == "harvest"]
    replans = [e for e in events if e.get("event") == "replan"]

    # ---- JUDGE ----
    n_judge = len(judges)
    judge_verdicts = [_norm_verdict(e.get("verdict")) for e in judges]
    n_pass = sum(1 for v in judge_verdicts if v == "pass")
    judge_pass_rate = (n_pass / n_judge) if n_judge else None

    # ---- JUDGE CALIBRATION (the SOTA metric) ----
    # over judge events carrying BOTH a final verdict AND a non-empty 'votes' panel: agreement =
    # (LLM-majority(votes) == verdict). Where the verdict was set by a mechanical verify_cmd, this is
    # LLM-vs-ground-truth. null when NO judge event is calibratable.
    n_calibratable = 0
    n_agree = 0
    calib_examples = []
    for e in judges:
        votes = e.get("votes")
        verdict = _norm_verdict(e.get("verdict"))
        if not isinstance(votes, list) or not votes or verdict is None:
            continue
        maj = _majority(votes)
        if maj is None:
            continue
        n_calibratable += 1
        agree = (maj == verdict)
        if agree:
            n_agree += 1
        else:
            calib_examples.append({"node": e.get("node"), "verdict": verdict, "llm_majority": maj,
                                   "mechanical": bool(e.get("mechanical"))})
    judge_calibration = (n_agree / n_calibratable) if n_calibratable else None

    # ---- CONVERGENCE (from reflect events; open_left trend) ----
    open_seq = [int(e.get("open_left")) for e in reflects if isinstance(e.get("open_left"), (int, float))]
    n_cycles = len(reflects)
    final_open_left = open_seq[-1] if open_seq else None
    # monotone-decreasing fraction: of consecutive (prev->next) steps, how many did NOT increase open_left.
    if len(open_seq) >= 2:
        non_increasing = sum(1 for a, b in zip(open_seq, open_seq[1:]) if b <= a)
        open_left_monotone_frac = non_increasing / (len(open_seq) - 1)
        net_open_delta = open_seq[-1] - open_seq[0]
    else:
        open_left_monotone_frac = None
        net_open_delta = 0
    # stalled: any reflect status indicating a stall, OR a stall_cycles counter that climbed, OR a route->replan
    # for a stall/repeated-failure reason. (Checks the explicit new-schema fields AND the older status string.)
    stall_status = any(("stall" in str(e.get("status", "")).lower()) for e in reflects)
    stall_counter = any(isinstance(e.get("stall_cycles"), (int, float)) and e.get("stall_cycles") >= 1
                        for e in reflects)
    stall_routes = sum(1 for e in events if e.get("event") == "route"
                       and any(k in str(e.get("reason", "")).lower() for k in ("stall", "repeated-failure")))
    stalled = bool(stall_status or stall_counter or stall_routes)
    # stalled-cycle fraction (for the quality penalty): cycles whose stall_cycles>=1 / total reflect cycles.
    stalled_cycles = sum(1 for e in reflects
                         if isinstance(e.get("stall_cycles"), (int, float)) and e.get("stall_cycles") >= 1)
    stalled_frac = (stalled_cycles / n_cycles) if n_cycles else 0.0

    # ---- DISPATCH (fan-out width + HITL) ----
    par_vals = [int(e.get("parallel")) for e in dispatches if isinstance(e.get("parallel"), (int, float))]
    n_dispatch = len(dispatches)
    mean_parallel = (sum(par_vals) / len(par_vals)) if par_vals else None
    max_parallel = max(par_vals) if par_vals else None
    n_parked = sum(len(e.get("parked_for_approval") or []) for e in dispatches)

    # ---- SELF-IMPROVEMENT (Voyager monotonicity) ----
    n_adjacent = sum(int(e.get("adjacent")) for e in reflects if isinstance(e.get("adjacent"), (int, float)))
    # a successful harvest carries verify_cmd; a {node,error} harvest is a FAILED harvest (count separately).
    n_harvest = sum(1 for e in harvests if e.get("verify_cmd") and not e.get("error"))
    n_harvest_failed = sum(1 for e in harvests if e.get("error"))
    # MECHANICAL passes are the only ground-truth passes that SHOULD be harvested (H4). harvest_rate =
    # harvests / mechanical-passes (not all passes -- LLM-panel passes are beliefs, not harvest-eligible).
    n_mech_pass = sum(1 for e in judges if _norm_verdict(e.get("verdict")) == "pass" and e.get("mechanical"))
    harvest_denom = max(1, n_mech_pass)
    harvest_rate = n_harvest / harvest_denom
    # monotonicity violation: a mechanical PASS with NO corresponding harvest event. (Only meaningful when
    # the run actually had mechanical passes; LLM-only runs do not harvest and are not violators.)
    harvest_monotonic = (n_mech_pass == 0) or (n_harvest >= n_mech_pass)

    # ---- REPLANNER ----
    n_replan = len(replans)
    replan_churn = sum(len(e.get("pruned") or []) + len(e.get("added") or []) for e in replans)

    # ---- SOLVED + TIME ----
    solved = any(str(e.get("status", "")).lower() == "solved" for e in reflects)
    ts = [e.get("t") for e in events if isinstance(e.get("t"), (int, float))]
    time_span_s = (max(ts) - min(ts)) if len(ts) >= 2 else 0.0

    # ---- COMPOSITE trajectory_quality in [0,1] ----
    # convergence sub-score: reward a low final open_left AND a downward trend. Map final_open_left through a
    # soft saturating function (0 open -> 1.0; many open -> ->0) and average with the monotone fraction.
    if final_open_left is not None:
        conv_close = 1.0 / (1.0 + max(0, final_open_left))          # 0 open -> 1.0 ; 9 open -> 0.1
    else:
        conv_close = 0.0
    conv_trend = open_left_monotone_frac if open_left_monotone_frac is not None else conv_close
    convergence_score = 0.5 * conv_close + 0.5 * conv_trend
    # calibration sub-score: PRESENT panel -> the measured rate; NO panel anywhere -> NEUTRAL 0.5 (don't punish
    # a mechanical-only run for lacking an LLM panel; do punish a present-but-disagreeing panel).
    calibration_score = judge_calibration if judge_calibration is not None else 0.5
    # harvest sub-score: clamp harvest_rate to [0,1]; a run with no mechanical passes is NEUTRAL 0.5 (nothing
    # to harvest is not a self-improvement failure).
    if n_mech_pass == 0:
        harvest_score = 0.5
    else:
        harvest_score = min(1.0, harvest_rate)
    solved_score = 1.0 if solved else 0.0

    quality = (QUALITY_WEIGHTS["convergence"] * convergence_score
               + QUALITY_WEIGHTS["calibration"] * calibration_score
               + QUALITY_WEIGHTS["harvest"] * harvest_score
               + QUALITY_WEIGHTS["solved"] * solved_score)
    quality -= STALL_PENALTY * stalled_frac
    if stalled and stalled_frac == 0.0:   # stall signalled only via route/status (older schema, no counter)
        quality -= STALL_PENALTY * 0.5
    trajectory_quality = round(max(0.0, min(1.0, quality)), 4)

    # failure-mode flags (for the aggregate histogram)
    failure_modes = []
    if stalled:
        failure_modes.append("stalled")
    if not solved:
        failure_modes.append("unsolved")
    if n_mech_pass > 0 and not harvest_monotonic:
        failure_modes.append("harvest_violation")
    if judge_calibration is not None and judge_calibration < 0.5:
        failure_modes.append("judge_miscalibrated")
    if open_left_monotone_frac is not None and open_left_monotone_frac < 0.5:
        failure_modes.append("non_converging")

    return {
        "run_id": traj["run_id"],
        "n_bad_lines": traj["n_bad"],
        "n_events": len(events),
        # judge
        "n_judge": n_judge,
        "judge_pass_rate": _r(judge_pass_rate),
        "judge_calibration": _r(judge_calibration),
        "n_calibratable": n_calibratable,
        "calib_disagreements": calib_examples[:5],
        # convergence
        "n_cycles": n_cycles,
        "final_open_left": final_open_left,
        "open_left_monotone_frac": _r(open_left_monotone_frac),
        "net_open_delta": net_open_delta,
        "stalled": stalled,
        "stalled_frac": _r(stalled_frac),
        # dispatch
        "n_dispatch": n_dispatch,
        "mean_parallel": _r(mean_parallel),
        "max_parallel": max_parallel,
        "n_parked": n_parked,
        # self-improvement
        "n_adjacent": n_adjacent,
        "n_harvest": n_harvest,
        "n_harvest_failed": n_harvest_failed,
        "n_mech_pass": n_mech_pass,
        "harvest_rate": _r(harvest_rate),
        "harvest_monotonic": harvest_monotonic,
        # replanner
        "n_replan": n_replan,
        "replan_churn": replan_churn,
        # terminal
        "solved": solved,
        "time_span_s": _r(time_span_s),
        # composite
        "trajectory_quality": trajectory_quality,
        "failure_modes": failure_modes,
    }


def _r(x, nd=4):
    """Round a float; pass through None/non-floats."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)):
        return round(float(x), nd)
    return x


# ==========================================================================================================
# 3. AGGREGATE -- the observability report across all traces
# ==========================================================================================================
def aggregate(trace_dir):
    """Score every <run_id>.jsonl in trace_dir and build the cross-corpus observability report."""
    d = Path(trace_dir)
    files = sorted(d.glob("*.jsonl")) if d.is_dir() else []
    rows = []
    n_dropped_empty = 0   # traces with zero valid events (all-malformed / empty) -- surfaced, not silently hidden
    total_n_bad = 0       # total malformed/partial JSONL lines skipped across the corpus
    for f in files:
        traj = load_trajectory(f)
        total_n_bad += traj.get("n_bad", 0)
        if not traj["events"]:
            n_dropped_empty += 1   # nothing to score, but COUNT it (a 100%-corrupt trace must not vanish silently)
            continue
        rows.append(score_trajectory(traj))

    n = len(rows)
    report = {"trace_dir": str(d), "n_traces": n, "n_files": len(files),
              "n_dropped_empty": n_dropped_empty, "total_n_bad_lines": total_n_bad, "rows": rows}
    if n == 0:
        report["empty"] = True
        return report

    # AGGREGATE JUDGE-CALIBRATION (the field-missing number): pool calibratable judge events across ALL
    # traces (weight by event count, not by trace -- a 1-judge trace should not equal a 50-judge trace).
    tot_calibratable = sum(r["n_calibratable"] for r in rows)
    # recover agreement counts from per-trace rate*count (exact: rate was n_agree/n_calibratable).
    tot_agree = sum(round((r["judge_calibration"] or 0.0) * r["n_calibratable"]) for r in rows)
    agg_calibration = (tot_agree / tot_calibratable) if tot_calibratable else None

    # metric distributions (min / median / mean / max) for the headline metrics.
    def _dist(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        if not vals:
            return None
        vals_sorted = sorted(vals)
        return {"min": _r(vals_sorted[0]), "median": _r(_median(vals_sorted)),
                "mean": _r(sum(vals) / len(vals)), "max": _r(vals_sorted[-1]), "n": len(vals)}

    distributions = {k: _dist(k) for k in
                     ("trajectory_quality", "judge_calibration", "judge_pass_rate",
                      "open_left_monotone_frac", "harvest_rate", "n_cycles", "mean_parallel")}

    # leaderboard: sorted by trajectory_quality desc.
    leaderboard = sorted(rows, key=lambda r: r["trajectory_quality"], reverse=True)
    lb = [{"run_id": r["run_id"], "trajectory_quality": r["trajectory_quality"],
           "judge_calibration": r["judge_calibration"], "solved": r["solved"],
           "n_cycles": r["n_cycles"], "failure_modes": r["failure_modes"]} for r in leaderboard]

    # failure-mode histogram.
    fm_hist = Counter()
    for r in rows:
        for fm in r["failure_modes"]:
            fm_hist[fm] += 1
    n_solved = sum(1 for r in rows if r["solved"])
    n_calibrated_traces = sum(1 for r in rows if r["judge_calibration"] is not None)

    report.update({
        "aggregate_judge_calibration": _r(agg_calibration),
        "aggregate_calibratable_events": tot_calibratable,
        "n_traces_with_calibration": n_calibrated_traces,
        "n_solved": n_solved,
        "solved_rate": _r(n_solved / n),
        "distributions": distributions,
        "leaderboard": lb,
        "failure_mode_histogram": dict(fm_hist.most_common()),
    })
    return report


def _median(sorted_vals):
    k = len(sorted_vals)
    if k == 0:
        return None
    mid = k // 2
    if k % 2:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def print_report(report):
    """Human-readable observability report (no emoji)."""
    print("=" * 88)
    print("METAOP TRAJECTORY-EVAL / OBSERVABILITY REPORT  (complement to eval_harness_run fitness)")
    print("=" * 88)
    print(f"  trace_dir : {report['trace_dir']}")
    print(f"  n_traces  : {report['n_traces']}"
          + (f"  (of {report['n_files']} files; {report['n_dropped_empty']} dropped-empty/all-bad, "
             f"{report['total_n_bad_lines']} malformed lines skipped)" if 'n_files' in report else ""))
    if report.get("empty"):
        print("  (no scorable traces found)")
        return
    ac = report["aggregate_judge_calibration"]
    print(f"  AGGREGATE JUDGE-CALIBRATION : {ac if ac is not None else 'null (no votes panels)'}"
          f"   over {report['aggregate_calibratable_events']} calibratable judge events"
          f"  ({report['n_traces_with_calibration']} traces had a panel)")
    print(f"  solved_rate : {report['solved_rate']}  ({report['n_solved']}/{report['n_traces']})")
    print("-" * 88)
    print("  DISTRIBUTIONS (min / median / mean / max):")
    for k, dv in report["distributions"].items():
        if dv:
            print(f"    {k:26s} {dv['min']!s:>8} / {dv['median']!s:>8} / {dv['mean']!s:>8} / {dv['max']!s:>8}"
                  f"   (n={dv['n']})")
    print("-" * 88)
    print("  FAILURE-MODE HISTOGRAM:")
    if report["failure_mode_histogram"]:
        for fm, c in report["failure_mode_histogram"].items():
            print(f"    {fm:22s} {c}")
    else:
        print("    (none)")
    print("-" * 88)
    lb = report["leaderboard"]
    print("  LEADERBOARD (top 5 by trajectory_quality):")
    for r in lb[:5]:
        print(f"    {r['trajectory_quality']:.4f}  {r['run_id'][:46]:46s}  cal={r['judge_calibration']}"
              f"  solved={r['solved']}  cyc={r['n_cycles']}  {','.join(r['failure_modes'])}")
    print("  BOTTOM 5:")
    for r in lb[-5:]:
        print(f"    {r['trajectory_quality']:.4f}  {r['run_id'][:46]:46s}  cal={r['judge_calibration']}"
              f"  solved={r['solved']}  cyc={r['n_cycles']}  {','.join(r['failure_modes'])}")
    print("=" * 88)


# ==========================================================================================================
# 4. REGRESSION SUITE -- golden fixtures + pinned bands derived from REAL traces
# ==========================================================================================================
# Which REAL traces become golden fixtures (chosen for schema coverage):
#   chess_validate-1780780054 : real multi-cycle, multi-vote panels, verdict!=majority (calibration < 1.0)
#   proofdeliver-1780878429   : real run with REPLANS + STALLS + new-schema reflect (refuted_left/stall_cycles)
#   meta-ma-1780698974        : real long sprawling run (open_left climbs -> non-converging, all single-vote)
#   ama2-expert-1780695470    : real run with a runnable:0 dispatch + a refuted + a pass (mixed)
#   dr-1780685625             : real SOLVED run (6 calibrated judges, multi-vote panels) -> covers the
#                               solved/converged path so the regression also catches a solved-weight drift
GOLDEN_SOURCE_TRACES = [
    "chess_validate-1780780054.jsonl",
    "proofdeliver-1780878429.jsonl",
    "meta-ma-1780698974.jsonl",
    "ama2-expert-1780695470.jsonl",
    "dr-1780685625.jsonl",
]
# metrics pinned for regression + the +/- tolerance band (absolute). Counts pinned exactly (tol 0);
# floats given a small band so a benign refactor (rounding) does not false-trip, but a real degradation does.
PINNED_METRICS = {
    "trajectory_quality":      0.02,
    "judge_calibration":       0.001,   # exact ratio of integer counts -> tiny band
    "n_calibratable":          0,
    "n_judge":                 0,
    "n_cycles":                0,
    "final_open_left":         0,
    "open_left_monotone_frac": 0.001,
    "n_harvest":               0,
    "n_replan":                0,
    "solved":                  0,        # bool: exact
}


def build_golden(trace_dir=DEFAULT_TRACE_DIR):
    """Copy the chosen REAL traces into tests/agent_eval_golden/ and pin their CURRENT metrics as the golden
    bands. The bands are DERIVED from the real traces (the harness scoring them now), never hand-invented."""
    src = Path(trace_dir)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden = {"_doc": "golden bands DERIVED from real metaop traces; regenerate via "
                       "`agent_eval.py --regression --update`. Each fixture pins the metrics in PINNED_METRICS "
                       "with the per-metric tolerance band. A harness change that moves any metric outside its "
                       "band is a trajectory-quality REGRESSION (exit 2).",
              "weights": QUALITY_WEIGHTS, "stall_penalty": STALL_PENALTY,
              "tolerances": PINNED_METRICS, "fixtures": {}}
    missing = []
    for name in GOLDEN_SOURCE_TRACES:
        s = src / name
        if not s.exists():
            missing.append(name)
            continue
        shutil.copy2(s, GOLDEN_DIR / name)
        m = score_trajectory(load_trajectory(GOLDEN_DIR / name))
        golden["fixtures"][name] = {k: m[k] for k in PINNED_METRICS if k in m}
    GOLDEN_JSON.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    return golden, missing


def regression(current_dir=None, update=False):
    """Re-score the golden fixtures and flag any pinned metric outside its band.

    Returns (ok: bool, findings: list). `current_dir` overrides where the fixtures are read from (default:
    the pinned tests/agent_eval_golden copies -- a FROZEN, in-repo corpus so the regression is hermetic and
    does not depend on the mutable live runs/autonomy/traces). exit 2 on any out-of-band metric."""
    if update:
        golden, missing = build_golden()
        if missing:
            return False, [f"cannot build golden -- source trace(s) missing: {missing}"]
        return True, [f"golden rebuilt: {len(golden['fixtures'])} fixtures pinned -> {GOLDEN_JSON}"]

    if not GOLDEN_JSON.exists():
        # first run: bootstrap the golden from the real traces, then it is frozen in-repo.
        golden, missing = build_golden()
        if missing:
            return False, [f"golden bootstrap failed -- missing source trace(s): {missing}"]
        findings = [f"golden bootstrapped ({len(golden['fixtures'])} fixtures) from real traces -> {GOLDEN_JSON}"]
    else:
        golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
        findings = []

    read_root = Path(current_dir) if current_dir else GOLDEN_DIR
    tolerances = golden.get("tolerances", PINNED_METRICS)
    ok = True
    for name, pinned in golden["fixtures"].items():
        fpath = read_root / name
        if not fpath.exists():
            ok = False
            findings.append(f"REGRESSION: fixture {name} not found under {read_root}")
            continue
        cur = score_trajectory(load_trajectory(fpath))
        for metric, expected in pinned.items():
            actual = cur.get(metric)
            tol = tolerances.get(metric, 0)
            if not _within_band(actual, expected, tol):
                ok = False
                findings.append(f"REGRESSION [{name}] {metric}: expected {expected} +/-{tol}, got {actual}")
    if ok and not findings:
        findings.append(f"all {len(golden['fixtures'])} golden fixtures within band")
    elif ok:
        findings.append(f"all {len(golden['fixtures'])} golden fixtures within band")
    return ok, findings


def _within_band(actual, expected, tol):
    """True if actual is within +/- tol of expected. None/bool handled exactly (tol ignored)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if expected is None or actual is None:
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= float(tol) + 1e-9
    return actual == expected


# ==========================================================================================================
# 5. TWO-SIDED SELFTEST -- the firewall (must ACCEPT a healthy trajectory AND REJECT a degraded one)
# ==========================================================================================================
def _synth_healthy(path):
    """A HEALTHY trajectory: open_left converges 5->0 (monotone down), judge calibrated (votes match
    verdict), mechanical passes ARE harvested, terminal 'solved'. Should score HIGH."""
    base = 1_780_000_000.0
    lines = []
    def emit(off, ev, **d):
        lines.append({"t": base + off, "event": ev, **d})

    emit(0, "plan", seeded=5, expert_mode=True, channel="expert")
    open_seq = [4, 3, 2, 1, 0]   # 5 nodes -> converging to 0
    for i, ol in enumerate(open_seq, start=1):
        node = f"h{i}"
        emit(10 * i, "dispatch", ran=[node], parallel=min(i, 2), experts={}, parked_for_approval=[])
        # mechanical pass + a matching LLM panel (calibrated) + a harvest for the ground-truth pass.
        emit(10 * i + 1, "harvest", node=node, verify_cmd="python -c \"assert True\"")
        emit(10 * i + 2, "judge", node=node, verdict="pass", votes=["pass", "pass", "pass"],
             mechanical=True, exit=0, evidence_type="mechanical")
        status = "solved" if ol == 0 else "running"
        emit(10 * i + 3, "reflect", cycle=i, adjacent=0, open_left=ol, refuted_left=0,
             status=status, done=i, stall_cycles=0)
    _write_jsonl(path, lines)


def _synth_degraded(path):
    """A DEGRADED trajectory: open_left FLAT/climbing (no convergence), judge MISCALIBRATED (LLM majority
    DISAGREES with the verdict every time), ZERO harvest of its passes, repeated STALL/replan, never solved.
    Should score LOW."""
    base = 1_780_500_000.0
    lines = []
    def emit(off, ev, **d):
        lines.append({"t": base + off, "event": ev, **d})

    emit(0, "plan", seeded=3, expert_mode=False, channel="plain")
    open_flat = [6, 7, 7, 8]   # climbing -> non-converging
    for i, ol in enumerate(open_flat, start=1):
        node = f"d{i}"
        emit(10 * i, "dispatch", ran=[node], parallel=1, experts={}, parked_for_approval=[])
        # verdict says 'refuted' but the LLM panel said 'pass' x3 -> majority DISAGREES (miscalibrated).
        # NO harvest event despite (we also mark mechanical to make it a harvest-eligible pass that was NOT
        # harvested in the pass cases below) -- here verdict refuted so calibration is the main signal.
        emit(10 * i + 1, "judge", node=node, verdict="refuted", votes=["pass", "pass", "pass"])
        emit(10 * i + 2, "reflect", cycle=i, adjacent=2, open_left=ol, refuted_left=1,
             status="running", done=0, stall_cycles=i)   # stall_cycles climbs, done never advances
        emit(10 * i + 3, "route", to="replan",
             reason="repeated-failure: node terminally refuted and no open node progressed", replan_count=i - 1)
        emit(10 * i + 4, "replan", reason="repeated-failure", pruned=[node], kept=[], added=[f"d{i}b"])
    # Also a mechanical PASS that is NOT harvested -> harvest monotonicity violation.
    emit(100, "dispatch", ran=["dpass"], parallel=1, experts={}, parked_for_approval=[])
    emit(101, "judge", node="dpass", verdict="pass", mechanical=True, exit=0, evidence_type="mechanical")
    emit(102, "reflect", cycle=len(open_flat) + 1, adjacent=2, open_left=9, refuted_left=1,
         status="running", done=0, stall_cycles=len(open_flat) + 1)   # still never solved
    _write_jsonl(path, lines)


def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for o in lines:
            fh.write(json.dumps(o) + "\n")


def selftest(margin=0.30):
    """Two-sided firewall. Synthesize healthy + degraded trajectories; assert the harness DISCRIMINATES.

    Asserts: quality(healthy) > quality(degraded) by >= margin; healthy calibration==1.0; degraded
    calibration==0.0; healthy solved & converging; degraded stalled & not solved & harvest violation; the
    malformed-line parser skips junk. Returns (ok, detail)."""
    detail = {}
    tmp = Path(tempfile.mkdtemp(prefix="agent_eval_selftest_"))
    try:
        hp = tmp / "healthy.jsonl"
        dp = tmp / "degraded.jsonl"
        _synth_healthy(hp)
        _synth_degraded(dp)
        # inject a malformed + a partial line into the degraded trace to exercise the robust parser.
        with open(dp, "a", encoding="utf-8") as fh:
            fh.write("{not valid json at all\n")
            fh.write('{"t": 1.0, "event": "judge", "node": "trunc"')   # truncated, no newline
        h = score_trajectory(load_trajectory(hp))
        d = score_trajectory(load_trajectory(dp))
        detail["healthy"] = h
        detail["degraded"] = d

        checks = []
        def chk(name, cond):
            checks.append((name, bool(cond)))

        chk("healthy_quality_high (>=0.75)", h["trajectory_quality"] >= 0.75)
        chk("degraded_quality_low (<=0.45)", d["trajectory_quality"] <= 0.45)
        chk(f"discriminates (high-low >= {margin})",
            (h["trajectory_quality"] - d["trajectory_quality"]) >= margin)
        chk("healthy_calibration==1.0", h["judge_calibration"] == 1.0)
        chk("degraded_calibration==0.0", d["judge_calibration"] == 0.0)
        chk("healthy_solved", h["solved"] is True)
        chk("degraded_not_solved", d["solved"] is False)
        chk("healthy_converges (monotone_frac==1.0)", h["open_left_monotone_frac"] == 1.0)
        chk("degraded_stalled", d["stalled"] is True)
        chk("degraded_harvest_violation", ("harvest_violation" in d["failure_modes"]))
        chk("healthy_harvest_ok", h["harvest_monotonic"] is True)
        chk("parser_skipped_malformed (n_bad>=2)", load_trajectory(dp)["n_bad"] >= 2)

        detail["checks"] = checks
        detail["margin_observed"] = round(h["trajectory_quality"] - d["trajectory_quality"], 4)
        ok = all(c for _, c in checks)
        return ok, detail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================================================
# CLI
# ==========================================================================================================
def main():
    ap = argparse.ArgumentParser(
        description="metaop TRAJECTORY-EVAL / observability / regression harness "
                    "(complement to eval_harness_run.py fitness)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--trace", metavar="FILE", help="score a single <run_id>.jsonl trace and print its metrics")
    g.add_argument("--aggregate", metavar="DIR", nargs="?", const=str(DEFAULT_TRACE_DIR),
                   help=f"observability report across all traces in DIR (default: {DEFAULT_TRACE_DIR})")
    g.add_argument("--regression", action="store_true",
                   help="re-score golden fixtures vs pinned bands; exit 2 on any out-of-band metric")
    g.add_argument("--selftest", action="store_true",
                   help="two-sided firewall: synth healthy+degraded, assert the eval discriminates")
    ap.add_argument("--update", action="store_true",
                    help="(with --regression) rebuild the golden bands from the real traces")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of the text report")
    args = ap.parse_args()

    if args.trace:
        traj = load_trajectory(args.trace)
        m = score_trajectory(traj)
        print(json.dumps(m, indent=2))
        return 0

    if args.regression:
        ok, findings = regression(update=args.update)
        print("=" * 78)
        print("METAOP TRAJECTORY-EVAL REGRESSION" + ("  (--update: rebuild golden)" if args.update else ""))
        print("=" * 78)
        for f in findings:
            print(f"  {f}")
        print("-" * 78)
        print(f"  RESULT: {'PASS' if ok else 'REGRESSION DETECTED'}")
        return 0 if ok else 2

    if args.selftest:
        ok, detail = selftest()
        print("=" * 78)
        print("METAOP TRAJECTORY-EVAL SELFTEST (two-sided firewall)")
        print("=" * 78)
        h, d = detail["healthy"], detail["degraded"]
        print(f"  healthy  : quality={h['trajectory_quality']}  calib={h['judge_calibration']}  "
              f"solved={h['solved']}  monotone={h['open_left_monotone_frac']}  harvest_ok={h['harvest_monotonic']}")
        print(f"  degraded : quality={d['trajectory_quality']}  calib={d['judge_calibration']}  "
              f"solved={d['solved']}  stalled={d['stalled']}  fmodes={d['failure_modes']}")
        print(f"  margin (high-low) = {detail['margin_observed']}")
        print("-" * 78)
        for name, passed in detail["checks"]:
            print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        print("-" * 78)
        print(f"  RESULT: {'PASS (eval discriminates)' if ok else 'FAIL (eval is vacuous)'}")
        return 0 if ok else 2

    # default: aggregate over the live trace dir
    target = args.aggregate or str(DEFAULT_TRACE_DIR)
    report = aggregate(target)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
