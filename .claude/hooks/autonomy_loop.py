#!/usr/bin/env python3
"""Stop hook -- the MECHANICAL keep-going engine of the autonomy framework.

Fires when Claude finishes a response. If autonomy is ACTIVE and there is still work to do, it returns
{"decision":"block","reason": <next>} so the harness feeds that back as the next instruction INSTEAD of
letting Claude stop. This is the thing prose ("keep working") cannot do -- the model cannot talk itself out
of a mechanical block. It is THE fix for Claude's "give a summary, then stop" weakness: a summary is NOT a
stop condition while the autonomous window is open. (Same mechanism /loop uses.)

TWO WAYS TO BE ACTIVE (either arms the loop):
  A. runs/autonomy/AUTONOMY_ON exists           -> frontier-driven loop (explicit objective + EV nodes)
  B. .claude/autonomous_mode.json autonomous=true AND now < envelope_end  -> TIMED loop (run to the clock)
     This is the connective tissue the user asked for: setting "autonomous mode" now mechanically prevents
     the halt-after-summary until the allocated window closes or the objective is verified SOLVED.

SAFETY (this hook controls whether the session can ever stop -- it is fail-OPEN and fenced):
  1. stop_hook_active == true   -> allow stop (we already blocked once this turn; never double-block this turn)
  2. neither switch active       -> allow stop (no AUTONOMY_ON, and autonomous_mode OFF/expired)
  3. envelope expired            -> allow stop (the TIMED budget IS the clock; auto-off at envelope_end)
  4. frontier budget spent / hard ceiling -> allow stop (spent >= max_cycles, or HARD_CEIL backstop)
  5. frontier empty/below floor AND no live envelope AND no tracked live job -> allow stop
  6. any error / missing state  -> allow stop (fail-open: a broken hook must NEVER trap the session)

THE GLOBAL ANTI-STUCK CONTRACT (2026-06-07 audit close -- 7 coupled paths). "Release the spin" is DECOUPLED from
"end the session". Whenever the loop would otherwise release WHILE the envelope is live and a TRACKED long job is
still alive (e.g. a detached training run), the hook does NOT allow_stop -- it BLOCKS with a bounded WAIT-MODE
instruction (one health check, then end the turn; do NOT spin/manufacture-work/grow-frontier). This guarantees
re-engageability without silent death and without busywork. allow_stop only when: the envelope expired, OR
(no tracked job alive AND no open above-floor frontier node).

Returns control text, not work -- the model still does the judgment each cycle. No emoji (Windows cp1252).
"""
import datetime
import time
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .claude/hooks -> parent root
CRYPTO = os.path.join(ROOT, "crypto")  # crypto sub-project root after the 3-way split (runs/scripts live here)
FRONTIER = os.path.join(CRYPTO, "runs", "autonomy", "frontier.json")
SWITCH = os.path.join(CRYPTO, "runs", "autonomy", "AUTONOMY_ON")
AUTO_MODE = os.path.join(ROOT, ".claude", "autonomous_mode.json")  # arming flag STAYS at the shared root
PROGRESS = os.path.join(CRYPTO, "runs", "autonomy", "loop_progress.json")
HARD_CEIL = 500  # absolute backstop on cycles, independent of frontier.budget
# P4: a loop armed WITHOUT an enforceable envelope (AUTONOMY_ON alone, or an unparseable envelope_end) must NOT
# run forever -- it falls back to this SAFE default window, anchored when the fallback first fires (persisted).
DEFAULT_MAX_WINDOW_HOURS = 6.0
STALL_LIMIT = 3            # consecutive UNCHANGED-progress cycles before the stall gate is *eligible* to release
STALL_IDLE_SECONDS = 1800  # P1a: ...but RELEASE only if wall-clock idle since the last real progress >= this. A
                           # legit multi-cycle build (dispatch->judge per node) ticks the hook faster than it
                           # advances the marker, so a raw 3-count would false-trip it -- gate on idle TIME instead.
# 2026-06-07 audit FIX (in_progress black-hole): a node the overseer is ACTIVELY working is marked
# status="in_progress". Treat it as ACTIVE (== open) everywhere the gate reasons about "work remaining", "next
# node", and the stall marker. Without this, in_progress was a BLACK HOLE -- counted neither done nor open -- so
# unfinished work read as "frontier exhausted/done" (false-done) and was never re-surfaced. ACTIVE closes that gap.
ACTIVE_STATUSES = ("open", "in_progress")

sys.path.insert(0, os.path.join(CRYPTO, "scripts", "autonomy"))

# UNSKIPPABLE 60s-WATCHER GATE (user mandate 2026-06-06): every loop continuation re-ensures the watcher is alive.
try:
    from ensure_watcher import ensure as _ensure_watcher
    _ensure_watcher()
except Exception:
    pass

# P0/P3b: the tracked-live-job check. A detached long job (training/backtest/metaop) registers a lock via
# scripts/autonomy/track_job.py; this lets the Stop hook SEE it and WAIT instead of silently dying mid-window.
try:
    from track_job import alive_jobs as _alive_jobs
except Exception:
    def _alive_jobs():  # fail-safe: if the tracker import breaks, report no tracked jobs (never trap on it)
        return []


def allow_stop():
    sys.exit(0)


def envelope_state():
    """Read .claude/autonomous_mode.json. Returns (active: bool, end: str|None, mandate: str, bounded: bool).
    active = autonomous flag true AND (no enforceable expiry yet reached). `bounded` is False when armed without an
    enforceable envelope (no envelope_end, or unparseable) -- the caller then applies the P4 SAFE default window.
    Fail-closed to inactive."""
    try:
        with open(AUTO_MODE, encoding="utf-8") as fh:
            d = json.load(fh)
        if not d.get("autonomous"):
            return (False, None, "", True)
        end = d.get("envelope_end")
        if end:
            end_dt = _parse_envelope(end)  # plain "%Y-%m-%d %H:%M:%S" OR ISO. None = unparseable.
            if end_dt is None:
                # P4: a garbage/unparseable envelope_end must NOT mean "run forever". Treat as UNBOUNDED -> the
                # caller enforces the SAFE default window instead of trusting the string.
                return (True, end, d.get("mandate", ""), False)
            if datetime.datetime.now() >= end_dt:  # expired -> auto-off
                return (False, end, "", True)
            return (True, end, d.get("mandate", ""), True)
        # autonomous=true with NO envelope_end at all -> UNBOUNDED -> P4 safe default window applies.
        return (True, None, d.get("mandate", ""), False)
    except Exception:
        return (False, None, "", True)


def _parse_envelope(end):
    """Parse envelope_end tolerant of BOTH the plain format and ISO (T separator, tz offset). Returns a naive
    local datetime, or None if genuinely unparseable. FIX 2026-06-06: an ISO timestamp used to raise in strptime
    -> the whole read fell to `except` -> autonomous mode SILENTLY DISARMED (W3 silent-off bug)."""
    if not end:
        return None
    try:
        return datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        pass
    try:
        return datetime.datetime.fromisoformat(end).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _load_progress():
    """Read loop_progress.json. P2 FIX: on an UNREADABLE/corrupt file do NOT silently reset to a fresh
    {marker:None, stall:0} -- that path used to silently zero the stall counter every cycle so the gate could
    NEVER trip (silent infinite spin). Instead return a conservative prior (stall preserved high, marker poisoned
    so it cannot match) and a flag so the caller emits a visible no-emoji warning. Returns (state, corrupt)."""
    try:
        with open(PROGRESS, encoding="utf-8") as fh:
            st = json.load(fh)
        if not isinstance(st, dict):
            raise ValueError("progress not a dict")
        # normalize
        st.setdefault("marker", None)
        st.setdefault("stall", 0)
        st.setdefault("idle_since", None)
        st.setdefault("default_window_anchor", None)
        return st, False
    except FileNotFoundError:
        # genuinely absent (first cycle) is NOT corruption -> fresh state is correct here
        return {"marker": None, "stall": 0, "idle_since": None, "default_window_anchor": None}, False
    except Exception:
        # corrupt/unreadable -> conservative prior: keep stall at the LIMIT so a real stall still releases, but
        # poison the marker so a working loop's next real progress resets it cleanly. NEVER silently spin.
        return {"marker": "__CORRUPT__", "stall": STALL_LIMIT, "idle_since": None,
                "default_window_anchor": None, "_corrupt": True}, True


def _save_progress(st):
    # H2 (2026-06-09): atomic write. PROGRESS is read by every live watcher (14 stacked at one point) while the
    # hook rewrites it each tick; a bare in-place json.dump can hand a reader a torn file (the read side has a
    # __CORRUPT__ fallback, but atomicity prevents the corrupt state entirely). tmp+os.replace = reader sees
    # old-or-new, never torn. Unique tmp (pid) so concurrent hook ticks never clobber one tmp.
    try:
        tmp = "%s.%d.tmp" % (PROGRESS, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
            fh.flush()
            os.fsync(fh.fileno())
        for _ in range(50):
            try:
                os.replace(tmp, PROGRESS)
                return
            except OSError:
                time.sleep(0.01)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass


def block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def _wait_mode_block(jobs, env_end, source):
    """P0: bounded WAIT-MODE. A tracked long job is alive and the window is open -- do ONE health check, then END
    the turn. Do NOT spin, do NOT manufacture work, do NOT grow the frontier. Re-engagement is the NEXT cycle."""
    jl = "; ".join(f"{j['id']}(pid {j['pid']})" for j in jobs[:6])
    block(
        "[AUTONOMY WAIT-MODE -- a tracked long job is ALIVE; the loop must WAIT EFFICIENTLY, not spin or "
        "manufacture work; you are the OVERSEER (Tier-0)]\n"
        f"TRACKED LIVE JOB(S): {jl}\n"
        f"WINDOW: open until {env_end or '(safe default window)'} ({source}).\n"
        "DO EXACTLY ONE health check this turn (e.g. read the job's latest log line / strength_curve / checkpoint "
        "mtime, or `python scripts/autonomy/track_job.py list`), state in one line whether it is HEALTHY or "
        "needs intervention, and -- if it needs intervention -- take the SINGLE corrective action (it is "
        "git-revertible; you make the call). Then STOP your response. Do NOT add frontier nodes, do NOT invent "
        "filler tasks, do NOT re-dispatch the running job. The job + the envelope continue independently; the "
        "NEXT Stop cycle re-checks. This is the anti-silent-death guarantee: the session stays re-engageable "
        "while the long job runs, without busywork."
    )


def main():
    # 1. anti-infinite-loop: if we already blocked this turn, let it stop
    try:
        data = json.load(sys.stdin)
    except Exception:
        return allow_stop()
    if data.get("stop_hook_active"):
        return allow_stop()

    # explicit kill switch
    if os.environ.get("AUTONOMY_LOOP", "").lower() == "off":
        return allow_stop()

    # 2 + 3. master switch. AUTHORITY: .claude/autonomous_mode.json (one place to arm/disarm -- W3 in
    # docs/SYSTEM_TOPOLOGY.md). AUTONOMY_ON is a LEGACY fallback that arms ONLY when autonomous_mode.json is
    # ABSENT/unreadable -- so setting autonomous:false there truly disarms even if a stale AUTONOMY_ON exists.
    env_active, env_end, env_mandate, env_bounded = envelope_state()
    switch_on = os.path.exists(SWITCH) and not os.path.exists(AUTO_MODE)  # legacy fallback only, non-authoritative
    if not switch_on and not env_active:
        return allow_stop()

    # progress state (shared by the stall gate + the P4 default-window anchor). Read ONCE; persist ONCE at the end
    # of the gate. P2: corrupt reads return a conservative prior + a visible warning instead of a silent reset.
    st, corrupt = _load_progress()
    if corrupt:
        print("[autonomy] WARNING: runs/autonomy/loop_progress.json was unreadable/corrupt -- preserving a "
              "conservative stall prior (NOT resetting to 0) so the anti-stuck gate stays live. Investigate the "
              "progress file.")

    # UNIFIED LOOP-ACTIVE CONCEPT (2026-06-07 audit close -- the AUTONOMY_ON arming-path gap). The P0 silent-death
    # protection must hold under BOTH arming paths: the envelope (.claude/autonomous_mode.json) AND the switch
    # (runs/autonomy/AUTONOMY_ON, which /orc uses by default and which DELETES autonomous_mode.json so env_active is
    # False). Previously every WAIT-MODE / release decision gated on `env_active AND live_jobs`, so a tracked long job
    # launched under a switch-only /orc run was NOT protected -> silent mid-window death reopened. FIX: treat the loop
    # as ACTIVE-AND-BOUNDED under switch_on too. switch_on always lacks an enforceable envelope_end, so the P4 SAFE
    # default window provides its bound (the same machinery the unbounded-envelope case uses). `loop_active` is the
    # one concept all downstream checks use; envelope-EXPIRY still wins (an expired envelope yields env_active=False
    # AND switch_on=False -- they are mutually exclusive since switch_on requires autonomous_mode.json ABSENT -- so an
    # expired envelope correctly STOPs, never silently converting to a switch-bounded loop).
    loop_active = env_active or switch_on

    # P4: enforce a SAFE bounded window whenever the loop is armed WITHOUT an enforceable envelope -- i.e. AUTONOMY_ON
    # alone (switch_on), OR autonomous=true with a missing/unparseable envelope_end (env_active and not env_bounded).
    # Without this an unbounded arm loops forever. We anchor the window the FIRST time the fallback fires and persist
    # it; once it elapses, RELEASE. This is what BOUNDS the switch-only /orc path (the auditor's P4-bounds-switch_on
    # claim is realized here: switch_on enters this block and gets a real deadline + env_end string for WAIT-MODE).
    default_window_source = None
    unbounded_arm = (env_active and not env_bounded) or (switch_on and not env_active)
    if unbounded_arm:
        anchor = st.get("default_window_anchor")
        now = datetime.datetime.now()
        if not anchor:
            st["default_window_anchor"] = now.strftime("%Y-%m-%d %H:%M:%S")
            _save_progress(st)
            anchor = st["default_window_anchor"]
        anchor_dt = _parse_envelope(anchor) or now
        deadline = anchor_dt + datetime.timedelta(hours=DEFAULT_MAX_WINDOW_HOURS)
        default_window_source = f"SAFE default {DEFAULT_MAX_WINDOW_HOURS:g}h window (no enforceable envelope_end)"
        if now >= deadline:
            print("[autonomy] SAFE-WINDOW EXPIRED: armed without an enforceable envelope_end; the "
                  f"{DEFAULT_MAX_WINDOW_HOURS:g}h default window (anchored {anchor}) has elapsed -- RELEASING. "
                  "Re-arm with a valid envelope_end to continue. (P4: refuse to loop forever on an unbounded arm.)")
            # treat as expired: downgrade BOTH arming flags so every downstream check sees an inactive loop.
            env_active = False
            switch_on = False
            loop_active = False
            return allow_stop()
        else:
            env_end = deadline.strftime("%Y-%m-%d %H:%M:%S") + f" [{default_window_source}]"

    # P0: discover tracked LIVE jobs once (used by every release decision below).
    try:
        live_jobs = _alive_jobs()
    except Exception:
        live_jobs = []

    # Try the frontier (it may not exist under a pure-timed run). Fail-open on read error.
    f = None
    if os.path.exists(FRONTIER):
        try:
            with open(FRONTIER, encoding="utf-8") as fh:
                f = json.load(fh)
        except Exception:
            f = None

    # === GLOBAL MANDATORY ANTI-STUCK GATE (user mandate 2026-06-07: "loop stuck -- resolve globally as a mandatory
    # gate"). The loop must WAIT EFFICIENTLY, never SPIN. The marker = REAL PROGRESS ONLY (done-count + sorted open
    # node-ids); budget.spent is DELIBERATELY EXCLUDED (P1b: the loop instruction increments spent every cycle, so
    # including it made the marker change every cycle and the gate could NEVER trip). Stall is measured in
    # wall-clock IDLE time (P1a), not a raw cycle count, so a legit multi-cycle build that dispatches+judges per
    # node does not false-trip. CRUCIAL (P0): a stall RELEASE never silently ends the session -- if the envelope is
    # live AND a tracked job is alive, it converts to WAIT-MODE (block) instead of allow_stop. ===
    open_ids_for_exempt = []
    if f is not None:
        try:
            _nodes = f.get("nodes", []) or []
            _floor_m = float(f.get("value_floor", 0.0))
            _done = sum(1 for n in _nodes if n.get("status") == "done")
            _open_ids = sorted(n.get("id", "") for n in _nodes if n.get("status") in ACTIVE_STATUSES)
            _open_above = [n for n in _nodes if n.get("status") in ACTIVE_STATUSES and float(n.get("ev", 0)) >= _floor_m]
            open_ids_for_exempt = [n.get("id", "") for n in _open_above]
            _marker = f"{_done}|{','.join(_open_ids)}"  # P1b: NO spent -- real progress only
            now_ts = datetime.datetime.now().timestamp()
            if st.get("marker") == _marker:
                st["stall"] = int(st.get("stall", 0)) + 1
                if not st.get("idle_since"):
                    st["idle_since"] = now_ts
            elif corrupt:
                # P2: the progress file was unreadable -- we CANNOT trust "marker changed = real progress" (the prior
                # marker is unknown/poisoned). Adopt the recomputed marker but PRESERVE the conservative stall prior
                # instead of zeroing it, so a corrupt file can never silently reset the anti-stuck gate to 0.
                st["marker"] = _marker
                st["stall"] = max(int(st.get("stall", 0)), STALL_LIMIT)
                if not st.get("idle_since"):
                    st["idle_since"] = now_ts - STALL_IDLE_SECONDS  # conservative: treat as already-idle
            else:
                st["marker"] = _marker
                st["stall"] = 0
                st["idle_since"] = now_ts
            _save_progress(st)

            idle_secs = now_ts - float(st.get("idle_since") or now_ts)
            # P1a EXEMPTION: an above-floor OPEN node + a tracked worker alive = a legit in-flight build. Do NOT let
            # the stall gate trip; the per-node dispatch/judge pattern legitimately ticks the hook without changing
            # the marker for a while. Wall-clock idle still bounds it (below) so a TRULY hung build still releases.
            legit_build_in_flight = bool(_open_above) and bool(live_jobs)
            if (st["stall"] >= STALL_LIMIT and idle_secs >= STALL_IDLE_SECONDS and not legit_build_in_flight):
                # Eligible to release the SPIN. But P0: do not silently die mid-window with a tracked job alive.
                # loop_active = env_active OR (switch_on within the bounded default window) -- protects BOTH arms.
                if loop_active and live_jobs:
                    return _wait_mode_block(live_jobs, env_end, "stall-gate -> wait-mode")
                print(f"[autonomy] STALL GATE (mandatory anti-stuck): no REAL frontier progress for {st['stall']} "
                      f"cycles / {int(idle_secs)}s idle (done-count + open-set unchanged; spent excluded) -- the "
                      "loop is spin-polling, not advancing. RELEASING. Re-engage on a real EVENT (a tracked "
                      "worker/bg-task completion, or a user message); any detached job + the envelope continue.")
                return allow_stop()
        except Exception:
            pass

    if f is not None:
        budget = f.get("budget", {}) or {}
        spent = int(budget.get("spent", 0))
        max_cycles = int(budget.get("max_cycles", 0))
        # 4. budget / hard ceiling -- but a live window (envelope OR switch-bounded) + a tracked live job still wants
        # re-engageability; only allow_stop when the loop is not active.
        if (max_cycles and spent >= max_cycles) or spent >= HARD_CEIL:
            if loop_active and live_jobs:
                return _wait_mode_block(live_jobs, env_end, "budget-spent -> wait-mode")
            if not loop_active:
                return allow_stop()
        else:
            floor = float(f.get("value_floor", 0.0))
            nodes = f.get("nodes", []) or []
            open_nodes = [n for n in nodes if n.get("status") in ACTIVE_STATUSES and float(n.get("ev", 0)) >= floor]
            if open_nodes:
                nxt = max(open_nodes, key=lambda n: float(n.get("ev", 0)))
                objective = f.get("objective", "(objective unset)")
                overseer = f.get("overseer", {}) or {}
                acceptance = overseer.get("acceptance_test", f.get("success_criteria", "(acceptance_test unset)"))
                return block(
                    "[AUTONOMY LOOP -- mechanical keep-going; you are the OVERSEER (Tier-0, .claude/skills/_common/OVERSEER.md); do NOT stop]\n"
                    f"OBJECTIVE (re-anchor every cycle -- am I still solving THIS?): {objective}\n"
                    f"ACCEPTANCE TEST (DONE only when this is VERIFIED by you, not asserted by a worker): {acceptance}\n"
                    f"NEXT FRONTIER NODE (EV={nxt.get('ev')}, id={nxt.get('id')}, kind={nxt.get('kind')}): {nxt.get('task')}\n"
                    "OVERSEER CYCLE: (a) re-state the objective in one line + confirm this node serves it (drift guard); "
                    "(b) DISPATCH execution -- you do NOT do primary building/running yourself; route by kind (build->single "
                    "worker, verify->adversarial panel, diverge->scout fan-out, decide->BULL/BEAR/NULL) via the Agent/Workflow "
                    "tools; only tiny lookups you may do inline; (c) JUDGE the return against the ACCEPTANCE TEST adversarially "
                    "(RWYB -- verify claims against artifacts; refuse false victory / drift-to-proxy / single-path narrowness); "
                    "(d) CORRECT-AS-YOU-GO: if this cycle exposes a weakness in the apparatus/brain/framework, fix it NOW (you "
                    "are authorized -- git is the revert net) and log it; (e) UPDATE runs/autonomy/frontier.json: mark this node "
                    "done|refuted|blocked WITH evidence, append overseer.fulfillment_ledger (verdict + git SHA/run-output + a "
                    "real `date`), PUSH new neighbor nodes (always a -k falsifier + a +k generalization -- breadth guard), "
                    "increment budget.spent, re-rank by ev; (f) WRITE-FORWARD any learning to memory. If you DISPATCH a long "
                    "background job (training/backtest/metaop), REGISTER it first: `python scripts/autonomy/track_job.py add "
                    "<id> --pid <PID> --cmd <desc>` (or launch via `track_job.py run`) so the watcher + this Stop hook can SEE "
                    "it and WAIT instead of silently dying mid-window. You MAKE THE CALLS (you are the user's proxy) + COMMIT "
                    "(git is the revert net) -- escalate ONLY for a genuinely irreversible real-world action. When the "
                    "acceptance_test is VERIFIED (SOLVED), say so and the loop releases. Then STOP your response -- this hook "
                    "hands you the next node."
                )
            # frontier exhausted (no open above-floor node). P0: do NOT silently die if a tracked job is alive in a
            # live window (envelope OR switch-bounded) -- WAIT instead. Release only when nothing is left to wait for.
            if loop_active and live_jobs:
                return _wait_mode_block(live_jobs, env_end, "frontier-exhausted -> wait-mode")
            if not loop_active:
                return allow_stop()

    # 5a/Switch-only fallback (P0 silent-death close for the /orc AUTONOMY_ON path): the switch is armed but the
    # frontier is absent/empty/spent. The switch-only contract is frontier-DRIVEN (it does not manufacture work from
    # an empty frontier the way the timed envelope does), so with NO live job we RELEASE (test case 3: no eternal
    # block). BUT if a TRACKED long job is alive inside the bounded window, we must NOT silently die mid-run -- we
    # WAIT-MODE (the exact bug this fix closes). switch_on here is already bounded by the P4 SAFE default window.
    if switch_on and not env_active:
        if live_jobs:
            return _wait_mode_block(live_jobs, env_end, "switch-fallback -> wait-mode")
        return allow_stop()

    # 5/Envelope fallback: a TIMED window is open but the frontier is absent/empty/spent. A SUMMARY IS NOT A STOP.
    if env_active:
        # P0: even here, if a tracked long job is alive, prefer bounded WAIT-MODE over open-ended "keep going" so we
        # neither spin nor silently die -- one health check per cycle while the job runs.
        if live_jobs:
            return _wait_mode_block(live_jobs, env_end, "envelope-fallback -> wait-mode")
        return block(
            "[AUTONOMOUS MODE ACTIVE -- mechanical keep-going; a summary is NOT a stop condition; do NOT stop]\n"
            f"The autonomous window is OPEN until {env_end}. You are the OVERSEER (Tier-0, .claude/skills/_common/OVERSEER.md): "
            "you summarized, but the mandate is not fulfilled and the clock is still running -- keep going.\n"
            f"MANDATE: {env_mandate}\n"
            "DO NOW: (a) re-anchor the objective in one line; (b) if there is no runs/autonomy/frontier.json, CREATE one from "
            "scripts/autonomy/frontier.template.json (fill objective + success_criteria + acceptance_test + seed the n+-k nodes) "
            "so the loop is frontier-driven; (c) take the next concrete value-producing action via DISPATCH to a worker (you "
            "oversee, you do not execute), JUDGE it with RWYB, and COMMIT it (git is the revert net) -- and if you dispatch a "
            "long background job, REGISTER it via `python scripts/autonomy/track_job.py add <id> --pid <PID>` so the loop waits "
            "on it instead of dying; (d) CORRECT-AS-YOU-GO: fix any weakness you find in the apparatus/brain/framework right now "
            "(meta-authorized); (e) WRITE-FORWARD the learning. Honest-stop is allowed ONLY if the objective is genuinely "
            "VERIFIED-SOLVED (then set autonomous=false to release) or you are blocked on a genuinely irreversible real-world "
            "action. Never idle-stop while the window is open. Then STOP your response -- this hook hands you the next cycle."
        )

    return allow_stop()


if __name__ == "__main__":
    main()
