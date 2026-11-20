#!/usr/bin/env python3
"""RWYB test harness for the GLOBAL anti-stuck Stop-hook fix (audit 2026-06-07 -- closes all 7 coupled paths).

It drives the Stop hook (.claude/hooks/autonomy_loop.py) inside a SANDBOX ROOT so the live runs/autonomy state is
never touched. For each crafted scenario it asserts the NEW behavior:
  P0  silent-death          -> a live tracked job + open window => WAIT-MODE block (NOT allow_stop, NOT spin)
  P1a stall false-trip      -> an above-floor OPEN node + a live worker => keep going (no premature release)
  P1b spent++ defeats gate  -> incrementing budget.spent must NOT reset the stall marker (marker excludes spent)
  P2  corrupt progress      -> corrupt loop_progress.json => does NOT silently reset to stall 0 (no silent spin)
  P3b tracked-job visibility-> frontier exhausted + live job + open window => WAIT-MODE (not silent death)
  P4  unbounded arm         -> autonomous=true with no/garbage envelope_end => SAFE default window, expires->stop
  REGRESSION normal loop    -> a real open node + progress => BLOCKS (keeps going), no spurious release

The hook resolves ROOT as .claude/hooks -> 3x dirname. We mirror that layout in a tempdir, drop the hook + a
track_job.py shim under the right paths, and run the hook with stdin = {"stop_hook_active": false}.

Decision encoding (what the hook prints on stdout):
  - allow_stop()      -> exit 0, NO stdout JSON with "decision":"block"  (the loop ENDS)
  - block(reason)     -> stdout = {"decision":"block","reason":...}        (the loop CONTINUES)
We classify each run as one of: BLOCK_WAIT (block whose reason starts with the WAIT-MODE banner), BLOCK_WORK
(block with the normal keep-going / overseer-cycle banner), or STOP (allow_stop / no block).

Run: python scripts/autonomy/test_loop_control.py    (exit 0 = all pass). No emoji (cp1252).
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REAL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We test the STAGED new hook if present (it can't be installed into .claude/ in some envs), else the live one.
STAGED_HOOK = os.path.join(REAL_ROOT, "runs", "staging", "autonomy_loop.new.py")
LIVE_HOOK = os.path.join(REAL_ROOT, ".claude", "hooks", "autonomy_loop.py")
TRACK_JOB = os.path.join(REAL_ROOT, "scripts", "autonomy", "track_job.py")
PROC_LIVENESS = os.path.join(REAL_ROOT, "scripts", "autonomy", "proc_liveness.py")  # G-J: track_job's liveness dep


def _hook_source():
    # Prefer the staged candidate (the version under test); fall back to the installed hook.
    if os.path.exists(STAGED_HOOK):
        return STAGED_HOOK, "staged"
    return LIVE_HOOK, "live"


def _build_sandbox(tmp, jobs=None, frontier=None, progress=None, auto_mode=None, switch=False, env=None):
    """Lay out a sandbox ROOT mirroring the real tree, with only what the hook reads.
    jobs: list of (id, pid) lock files to drop. progress/frontier/auto_mode: dict|None|"__CORRUPT__"."""
    hooks_dir = os.path.join(tmp, ".claude", "hooks")
    sa_dir = os.path.join(tmp, "scripts", "autonomy")
    aut_dir = os.path.join(tmp, "runs", "autonomy")
    locks_dir = os.path.join(aut_dir, "locks")
    claude_dir = os.path.join(tmp, ".claude")
    for d in (hooks_dir, sa_dir, locks_dir, claude_dir):
        os.makedirs(d, exist_ok=True)
    # the hook + its sibling deps (track_job is imported; ensure_watcher import is best-effort and may be absent)
    src, _ = _hook_source()
    shutil.copy(src, os.path.join(hooks_dir, "autonomy_loop.py"))
    shutil.copy(TRACK_JOB, os.path.join(sa_dir, "track_job.py"))
    shutil.copy(PROC_LIVENESS, os.path.join(sa_dir, "proc_liveness.py"))  # G-J: track_job imports it (create-time liveness)
    # locks
    for (jid, pid) in (jobs or []):
        with open(os.path.join(locks_dir, f"{jid}.lock"), "w", encoding="utf-8") as fh:
            json.dump({"pid": pid, "thread": jid, "kind": "job"}, fh)
    # frontier
    if frontier is not None:
        with open(os.path.join(aut_dir, "frontier.json"), "w", encoding="utf-8") as fh:
            json.dump(frontier, fh)
    # progress
    pp = os.path.join(aut_dir, "loop_progress.json")
    if progress == "__CORRUPT__":
        with open(pp, "w", encoding="utf-8") as fh:
            fh.write("{ this is not valid json :::")
    elif progress is not None:
        with open(pp, "w", encoding="utf-8") as fh:
            json.dump(progress, fh)
    # autonomous_mode.json
    if auto_mode is not None:
        with open(os.path.join(claude_dir, "autonomous_mode.json"), "w", encoding="utf-8") as fh:
            json.dump(auto_mode, fh)
    # AUTONOMY_ON switch
    if switch:
        open(os.path.join(aut_dir, "AUTONOMY_ON"), "w").close()
    return os.path.join(hooks_dir, "autonomy_loop.py")


def _run(hook_path, stdin_obj=None):
    if stdin_obj is None:
        stdin_obj = {"stop_hook_active": False}
    r = subprocess.run([sys.executable, hook_path], input=json.dumps(stdin_obj),
                       capture_output=True, text=True, timeout=30,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    return r


def _classify(r):
    out = (r.stdout or "").strip()
    decision = None
    # block() prints a single JSON line; allow_stop prints nothing (or only diagnostic [autonomy] lines).
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                if d.get("decision") == "block":
                    decision = d
                    break
            except Exception:
                pass
    if decision is None:
        return "STOP", out
    reason = decision.get("reason", "")
    if reason.startswith("[AUTONOMY WAIT-MODE"):
        return "BLOCK_WAIT", reason
    return "BLOCK_WORK", reason


# a future + a past timestamp in the plain envelope format
FUTURE = (datetime.datetime.now() + datetime.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
PAST = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
LIVE_PID = os.getpid()        # this test process -- guaranteed alive
DEAD_PID = 999999             # not a real PID -> not alive

OPEN_NODE_FRONTIER = {
    "objective": "obj", "value_floor": 0.3, "budget": {"spent": 1, "max_cycles": 40},
    "overseer": {"acceptance_test": "verified"},
    "nodes": [{"id": "n1", "ev": 0.9, "kind": "build", "status": "open"}],
}
EXHAUSTED_FRONTIER = {
    "objective": "obj", "value_floor": 0.3, "budget": {"spent": 5, "max_cycles": 40},
    "overseer": {"acceptance_test": "verified"},
    "nodes": [{"id": "n1", "ev": 0.9, "kind": "build", "status": "done"}],
}
IN_PROGRESS_FRONTIER = {  # the overseer is ACTIVELY working n1 (status in_progress) -- NOT done, NOT a black hole
    "objective": "obj", "value_floor": 0.3, "budget": {"spent": 2, "max_cycles": 40},
    "overseer": {"acceptance_test": "verified"},
    "nodes": [{"id": "n1", "ev": 0.9, "kind": "build", "status": "in_progress"}],
}


def main():
    src, kind = _hook_source()
    print(f"[test] hook under test = {kind}: {src}\n")
    results = []

    def case(name, expect, **kw):
        tmp = tempfile.mkdtemp(prefix="loopctl_")
        try:
            hook = _build_sandbox(tmp, **kw)
            r = _run(hook)
            got, detail = _classify(r)
            ok = got == expect
            results.append(ok)
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} expect={expect}")
            if not ok:
                print(f"        stdout: {(r.stdout or '').strip()[:300]}")
                print(f"        stderr: {(r.stderr or '').strip()[:200]}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- P0 SILENT-DEATH: live tracked job + open window + NO open frontier node => WAIT-MODE (not stop, not spin)
    case("P0 silent-death: live job + open window + exhausted frontier => WAIT-MODE",
         "BLOCK_WAIT",
         jobs=[("train", LIVE_PID)], frontier=EXHAUSTED_FRONTIER,
         auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"},
         progress={"marker": "x", "stall": 0})

    # --- P0b: live tracked job + open window + NO frontier file at all => envelope-fallback WAIT-MODE
    case("P0b: live job + open window + no frontier => WAIT-MODE",
         "BLOCK_WAIT",
         jobs=[("train", LIVE_PID)],
         auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"})

    # --- P1a STALL FALSE-TRIP: above-floor OPEN node + live worker + a marker already stalled => keep WORKING
    #     (the stall gate must EXEMPT a legit in-flight build; never release while real open work + a worker exist)
    stalled_progress = {"marker": "0|n1", "stall": 99,
                        "idle_since": time.time() - 99999}  # very idle, would trip if not exempt
    case("P1a stall false-trip: open node + live worker (idle long) => keep WORKING (exempt)",
         "BLOCK_WORK",
         jobs=[("train", LIVE_PID)], frontier=OPEN_NODE_FRONTIER,
         auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"},
         progress=stalled_progress)

    # --- P1b SPENT++ DEFEATS GATE: marker must EXCLUDE budget.spent. We assert the persisted marker after a run
    #     has NO spent component, and that two runs differing ONLY in spent keep the SAME marker (so stall grows).
    tmp = tempfile.mkdtemp(prefix="loopctl_p1b_")
    try:
        # run 1: spent=1, no progress baseline
        fr1 = json.loads(json.dumps(EXHAUSTED_FRONTIER)); fr1["budget"]["spent"] = 1
        hook = _build_sandbox(tmp, jobs=[("train", LIVE_PID)], frontier=fr1,
                              auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"},
                              progress={"marker": None, "stall": 0})
        _run(hook)
        m1 = json.load(open(os.path.join(tmp, "runs", "autonomy", "loop_progress.json")))
        # run 2: ONLY spent changes (1 -> 2). Real progress (done-count + open-set) identical.
        fr2 = json.loads(json.dumps(EXHAUSTED_FRONTIER)); fr2["budget"]["spent"] = 2
        with open(os.path.join(tmp, "runs", "autonomy", "frontier.json"), "w") as fh:
            json.dump(fr2, fh)
        _run(hook)
        m2 = json.load(open(os.path.join(tmp, "runs", "autonomy", "loop_progress.json")))
        marker_excludes_spent = ("|1|" not in str(m1.get("marker")) and "|2|" not in str(m2.get("marker")))
        marker_stable = (m1.get("marker") == m2.get("marker"))
        stall_grew = int(m2.get("stall", 0)) > int(m1.get("stall", -1))
        ok = marker_excludes_spent and marker_stable and stall_grew
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] P1b spent++ no longer defeats gate: "
              f"marker_excludes_spent={marker_excludes_spent} marker_stable={marker_stable} "
              f"stall_grew({m1.get('stall')}->{m2.get('stall')})={stall_grew}")
        if not ok:
            print(f"        m1={m1} m2={m2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- P2 CORRUPT PROGRESS: a corrupt loop_progress.json must NOT silently reset stall to 0. We craft a stalled
    #     state with NO live job + NO open node + corrupt progress, and a frontier whose marker would be stable.
    #     The conservative prior (stall=LIMIT) means the gate is ELIGIBLE; with no job/window it RELEASES (STOP)
    #     and emits a visible warning -- the key assertion is the warning is printed (path made visible).
    tmp = tempfile.mkdtemp(prefix="loopctl_p2_")
    try:
        # no envelope, switch ON (frontier loop), exhausted frontier, corrupt progress, NO live job
        hook = _build_sandbox(tmp, jobs=[("dead", DEAD_PID)], frontier=EXHAUSTED_FRONTIER,
                              switch=True, progress="__CORRUPT__")
        r = _run(hook)
        warned = "unreadable/corrupt" in (r.stdout or "")
        # and: it must NOT have written a fresh {stall:0} that hides the corruption -- assert stall preserved >=1
        try:
            after = json.load(open(os.path.join(tmp, "runs", "autonomy", "loop_progress.json")))
            stall_not_zeroed = int(after.get("stall", 0)) >= 1
        except Exception:
            stall_not_zeroed = True  # if it stayed unreadable, it certainly wasn't silently reset to 0
        ok = warned and stall_not_zeroed
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] P2 corrupt progress no silent reset: "
              f"warned={warned} stall_not_zeroed={stall_not_zeroed}")
        if not ok:
            print(f"        stdout: {(r.stdout or '').strip()[:300]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- P3b TRACKED-JOB VISIBILITY: a DEAD lock must NOT keep the loop alive (reaped), but a LIVE lock must.
    #     dead job + open window + exhausted frontier => nothing to wait for => STOP (envelope still active but no
    #     live job AND no open node -> the envelope fallback would normally block; assert dead lock doesn't WAIT).
    case("P3b dead lock is reaped (not treated as live) => no spurious WAIT-MODE",
         "BLOCK_WORK",   # envelope open, no live job -> falls to the envelope-fallback keep-going (NOT wait-mode)
         jobs=[("dead", DEAD_PID)],
         auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"})

    # --- P4 UNBOUNDED ARM (no envelope_end): must NOT loop forever -- a SAFE default window applies. Fresh anchor
    #     => still within the window => keeps going (BLOCK), but bounded. (Expiry tested separately below.)
    case("P4 unbounded arm (no envelope_end): bounded keep-going within safe window",
         "BLOCK_WORK",
         auto_mode={"autonomous": True, "mandate": "m"})  # no envelope_end at all

    case("P4 garbage envelope_end: bounded keep-going within safe window (not forever)",
         "BLOCK_WORK",
         auto_mode={"autonomous": True, "envelope_end": "not-a-timestamp", "mandate": "m"})

    # --- P4 EXPIRY: unbounded arm whose default-window anchor is already in the past => RELEASE (STOP)
    old_anchor = (datetime.datetime.now() - datetime.timedelta(hours=99)).strftime("%Y-%m-%d %H:%M:%S")
    case("P4 unbounded arm past safe-window => RELEASE (no infinite loop)",
         "STOP",
         auto_mode={"autonomous": True, "mandate": "m"},
         progress={"marker": None, "stall": 0, "default_window_anchor": old_anchor})

    # --- REGRESSION: a NORMAL working loop (real open above-floor node, progressing) must keep going (BLOCK_WORK)
    case("REGRESSION normal loop: open node, envelope live => keeps going (BLOCK)",
         "BLOCK_WORK",
         frontier=OPEN_NODE_FRONTIER,
         auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"},
         progress={"marker": "0|n1", "stall": 0, "idle_since": time.time()})

    # --- REGRESSION: nothing armed => STOP (the hook must not trap a non-autonomous session)
    case("REGRESSION not armed => STOP",
         "STOP",
         frontier=OPEN_NODE_FRONTIER)

    # --- REGRESSION: stop_hook_active true => STOP (never double-block in one turn)
    tmp = tempfile.mkdtemp(prefix="loopctl_dbl_")
    try:
        hook = _build_sandbox(tmp, frontier=OPEN_NODE_FRONTIER,
                              auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"})
        r = _run(hook, {"stop_hook_active": True})
        got, _ = _classify(r)
        ok = got == "STOP"
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] REGRESSION stop_hook_active => STOP: got={got}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- REGRESSION: expired envelope + no frontier => STOP
    case("REGRESSION expired envelope => STOP",
         "STOP",
         auto_mode={"autonomous": True, "envelope_end": PAST, "mandate": "m"})

    # === SWITCH-ONLY ARMING (AUTONOMY_ON, the /orc default path) -- the audit-found gap this fix closes. ===
    # The switch arms WITHOUT .claude/autonomous_mode.json (launch_autonomy DELETES it), so env_active=False. The P0
    # silent-death protection + the enforceable time-bound must STILL hold via the unified loop_active concept +
    # the P4 SAFE default window. No auto_mode at all in these cases (switch=True, auto_mode=None).

    # SW1: AUTONOMY_ON + NO autonomous_mode.json + tracked LIVE job + within default window + exhausted frontier
    #      => WAIT-MODE (NOT allow_stop). This is the exact silent-death bug under /orc: previously env_active=False
    #      meant every WAIT-MODE check was skipped and the loop allow_stopped while the tracked job ran -> silent
    #      mid-window death. With loop_active = env_active OR switch_on, it now WAITs. Fresh anchor => inside window.
    case("SW1 switch-only: live job + within safe window + exhausted frontier => WAIT-MODE (no silent death)",
         "BLOCK_WAIT",
         jobs=[("train", LIVE_PID)], frontier=EXHAUSTED_FRONTIER, switch=True,
         progress={"marker": "1|", "stall": 0})

    # SW1b: AUTONOMY_ON + live job + within safe window + NO frontier file at all => switch-fallback WAIT-MODE
    case("SW1b switch-only: live job + within safe window + no frontier => WAIT-MODE",
         "BLOCK_WAIT",
         jobs=[("train", LIVE_PID)], switch=True)

    # SW2: AUTONOMY_ON + tracked job + PAST the default window => STOP. The enforceable time-bound (P4 SAFE window)
    #      MUST win even with a live job -- an expired/elapsed window means RELEASE (the bound is the clock).
    sw_old_anchor = (datetime.datetime.now() - datetime.timedelta(hours=99)).strftime("%Y-%m-%d %H:%M:%S")
    case("SW2 switch-only: live job but PAST safe window => STOP (time-bound wins)",
         "STOP",
         jobs=[("train", LIVE_PID)], switch=True,
         progress={"marker": None, "stall": 0, "default_window_anchor": sw_old_anchor})

    # SW3: AUTONOMY_ON + NO tracked job + no open node (exhausted frontier) + within window => STOP (no eternal
    #      block). The switch-only contract is frontier-DRIVEN; with nothing to wait for it releases cleanly.
    case("SW3 switch-only: no job + exhausted frontier (within window) => STOP (no eternal block)",
         "STOP",
         frontier=EXHAUSTED_FRONTIER, switch=True,
         progress={"marker": "1|", "stall": 0})

    # SW3b: AUTONOMY_ON + no job + NO frontier (within window) => STOP (frontier-driven, nothing to do)
    case("SW3b switch-only: no job + no frontier (within window) => STOP",
         "STOP",
         switch=True)

    # SW4: AUTONOMY_ON + open above-floor frontier node (within window) => BLOCK_WORK (keep working the frontier).
    #      Confirms the switch-only path still drives the normal keep-going loop, not just WAIT/STOP.
    case("SW4 switch-only: open frontier node (within window) => keeps working (BLOCK)",
         "BLOCK_WORK",
         frontier=OPEN_NODE_FRONTIER, switch=True,
         progress={"marker": "0|n1", "stall": 0, "idle_since": time.time()})

    # --- 2026-06-07 in_progress BLACK-HOLE fix: a node the overseer is ACTIVELY working (status in_progress) must
    #     be treated as ACTIVE work (== open), not silently swallowed as "exhausted/done" (false-done).
    # IP1 (envelope arm): in_progress node + envelope live + no job => KEEP WORKING (re-surface it), not stop.
    case("IP1 in_progress node + envelope live => keeps working (not false-done)",
         "BLOCK_WORK",
         frontier=IN_PROGRESS_FRONTIER,
         auto_mode={"autonomous": True, "envelope_end": FUTURE, "mandate": "m"},
         progress={"marker": "0|n1", "stall": 0, "idle_since": time.time()})

    # IP2 (switch arm): in_progress node within the safe window => KEEP WORKING (frontier NOT exhausted).
    #     Pre-fix this returned STOP (in_progress counted neither done nor open => false-exhausted black hole).
    case("IP2 switch-only: in_progress node within window => keeps working (not false-exhausted)",
         "BLOCK_WORK",
         frontier=IN_PROGRESS_FRONTIER, switch=True,
         progress={"marker": "0|n1", "stall": 0, "idle_since": time.time()})

    print()
    npass = sum(1 for x in results if x)
    print(f"{'ALL PASS' if all(results) else '*** SOME FAILED ***'} ({npass}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
