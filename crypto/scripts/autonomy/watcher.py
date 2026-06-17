#!/usr/bin/env python3
"""ABSOLUTE 1-min liveness watcher (resumable, bounded lifetime).

Role (per the orchestrator mandate "1m loop and watcher is absolute"):
  - every 60s: auto-discover loop locks, check PROCESS liveness, append a timestamped CHECK-IN line to
    runs/autonomy/watcher.log (the durable trail the orchestrator reads on each wake).
  - it does NOT auto-relaunch: relaunch/pivot decisions belong to the orchestrator (a refuted vein should be
    pivoted, not blindly restarted). If a loop dies, the watcher EXITS EARLY with reason=loop_dead so the
    orchestrator wakes promptly to decide.
  - self-evolution windows (03:00 / 06:00 / 09:00 SAST): if a window is open and not yet marked done in the
    ledger-state, EXIT EARLY with reason=evolution_due so the orchestrator runs the 30-min cycle on time.
  - bounded lifetime (default 22 ticks ~= 22 min) then EXIT reason=tick_budget -> gives the orchestrator a
    natural ~20-min active-monitoring cadence (this harness re-invokes the orchestrator when a bg task exits).

Resumable: holds no in-memory state that matters; the log + watcher_state.json are on disk. Relaunch continues.
No emoji (cp1252).

Usage: python scripts/autonomy/watcher.py [--max-ticks 22] [--threads sol-ma meta-ma]
"""
from __future__ import annotations
import argparse
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCKS = os.path.join(ROOT, "runs", "autonomy", "locks")
LOG = os.path.join(ROOT, "runs", "autonomy", "watcher.log")
STATE = os.path.join(ROOT, "runs", "autonomy", "watcher_state.json")
LEARN = os.path.join(ROOT, "runs", "autonomy", "learnings")
MAP = os.path.join(ROOT, "experiments", "adaptive_ma", "PER_CADENCE_MAP.json")
EVOLUTION_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]  # every 3h from 00:00 SAST -- the project-wide loop-3 cadence
STALL_TICKS = 8  # if no loop PROGRESS (checkpoint mtime + lane counts) for this many 60s ticks -> wake overseer
META_TICKS = 5   # wake the overseer every N ticks for the 60s-cadence META dual-view pass (project + loop tasks)
CKPT = os.path.join(ROOT, "runs", "autonomy")  # metaop_<thread>.db live here
SINGLETON = os.path.join(ROOT, "runs", "autonomy", "watcher.singleton.lock")  # the ONE live-watcher lock
SINGLETON_STALE_S = 150  # a singleton heartbeat older than this (> 2 ticks) => the owner crashed; claim is free


def _progress_sig(threads):
    """A signature of loop PROGRESS: per-thread checkpoint-db mtime + lane counts. Unchanged across ticks = stalled
    (a loop that is ALIVE but HUNG -- the silent-hang case the lock-PID liveness check cannot see)."""
    sig = []
    for t in threads:
        db = os.path.join(CKPT, f"metaop_{t}.db")
        try:
            sig.append(round(os.path.getmtime(db), 1))
        except OSError:
            sig.append(0)
    return tuple(sig) + tuple(sorted(_lane_counts().items()))


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from proc_liveness import alive as _alive  # G-J 2026-06-07: the ONE create-time-aware liveness check (no PID reuse)


def _discover():
    try:
        return sorted(f[:-5] for f in os.listdir(LOCKS) if f.endswith(".lock"))
    except FileNotFoundError:
        return []


def _liveness(threads):
    alive, dead = [], []
    for t in threads:
        try:
            _d = json.load(open(os.path.join(LOCKS, f"{t}.lock"), encoding="utf-8"))
            (alive if _alive(_d.get("pid"), _d.get("created")) else dead).append(t)
        except Exception:
            dead.append(t)
    return alive, dead


def _lane_counts():
    out = {}
    try:
        for f in os.listdir(LEARN):
            if f.endswith(".jsonl"):
                with open(os.path.join(LEARN, f), encoding="utf-8") as fh:
                    out[f[:-6]] = sum(1 for _ in fh)
    except FileNotFoundError:
        pass
    return out


def _load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {"evolution_done": {}}  # {"2026-06-06T03": true}


def _evolution_due(state):
    now = dt.datetime.now()
    for h in EVOLUTION_HOURS:
        if now.hour == h:  # inside the hour the cycle opens
            key = now.strftime(f"%Y-%m-%dT{h:02d}")
            if not state.get("evolution_done", {}).get(key):
                return key
    return None


def _append(line):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _singleton_owner_alive():
    """Return the (pid, created) of a LIVE, FRESH-heartbeat watcher that already owns the singleton lock, or None.
    'Fresh' = heartbeat within SINGLETON_STALE_S (a crashed owner leaves a stale beat -> the lock is free to claim).
    This is what stops every /orc launch from starting a NEW immortal self-respawning watcher lineage (the 2026-06-09
    leak: 14 watchers from 7 launches). Never raises -> on any error the lock is treated as free (fail-open to spawn)."""
    try:
        d = json.load(open(SINGLETON, encoding="utf-8"))
    except Exception:
        return None
    pid = d.get("pid")
    if not pid or pid == os.getpid():
        return None
    try:
        if (time.time() - float(d.get("heartbeat", 0))) > SINGLETON_STALE_S:
            return None  # stale -> owner crashed -> free
    except (TypeError, ValueError):
        return None
    return (pid, d.get("created")) if _alive(pid, d.get("created")) else None


def _claim_singleton():
    """Write THIS process as the singleton owner + refresh the heartbeat (called at startup and every tick). Atomic
    (tmp+replace) so a concurrent reader never sees a torn lock. Best-effort; never raises."""
    try:
        rec = {"pid": os.getpid(), "created": None, "heartbeat": round(time.time(), 1),
               "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:  # stamp our own create-time so a future PID-reuse can't masquerade as us (proc_liveness contract)
            import proc_liveness as _pl
            rec["created"] = _pl.created_at(os.getpid()) if hasattr(_pl, "created_at") else None
        except Exception:
            pass
        tmp = "%s.%d.tmp" % (SINGLETON, os.getpid())
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        for _ in range(20):
            try:
                os.replace(tmp, SINGLETON); return
            except OSError:
                time.sleep(0.01)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ticks", type=int, default=22)
    ap.add_argument("--threads", nargs="*", default=None)
    # --respawn: set by the self-respawn handoff. Such a child SKIPS the singleton CHECK (its dying parent still holds
    # a fresh heartbeat) but still CLAIMS the lock -> the lineage stays singular. A FRESH launch (no --respawn) runs the
    # check and EXITS if a live watcher already owns the lock (closes the 14-watchers-from-7-launches leak).
    ap.add_argument("--respawn", action="store_true", help="internal: self-respawn handoff (bypass singleton check)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if not a.respawn:
        owner = _singleton_owner_alive()
        if owner:
            _append(f"[{dt.datetime.now():%H:%M:%S}] EXIT reason=duplicate (singleton already held by pid={owner[0]}) "
                    "-- not stacking another watcher lineage")
            print(f"EXIT duplicate (watcher pid={owner[0]} already live)"); return 0
    _claim_singleton()
    started = dt.datetime.now().strftime("%H:%M:%S")
    _append(f"=== WATCHER START {dt.datetime.now():%Y-%m-%d %H:%M:%S} max_ticks={a.max_ticks} respawn={a.respawn} ===")
    prev_sig, stall = None, 0
    for tick in range(a.max_ticks):
        _claim_singleton()  # refresh heartbeat each tick so a concurrent fresh launch sees us as the live owner -> exits
        state = _load_state()
        threads = a.threads or _discover()
        alive, dead = _liveness(threads)
        lanes = _lane_counts()
        ts = dt.datetime.now().strftime("%H:%M:%S")
        lane_str = " ".join(f"{k}:{v}" for k, v in sorted(lanes.items()))
        _append(f"[{ts}] tick {tick+1}/{a.max_ticks} | alive={alive} dead={dead} | "
                f"map={'Y' if os.path.exists(MAP) else 'N'} | lanes: {lane_str}")
        # 3h-cadence EVOLUTION SIGNAL: write a flag + mark the window handled, but DO NOT die. Exiting on evolution_due
        # was a self-kill that kept the 'absolute' watcher DOWN every evolution hour (00/03/.../21) -- the same class as
        # the meta_due self-kill. The deep evolution pass is the overseer's job at its cadence (it polls the flag); the
        # watcher's job is to STAY ALIVE. Only loop_dead/loop_stalled/tick_budget are legitimate exits. (root fix 2026-06-06)
        ev = _evolution_due(state)
        if ev:
            _append(f"[{ts}] evolution_due signal window={ev} -- flag written + window marked; watcher CONTINUES (no self-kill)")
            try:
                with open(os.path.join(os.path.dirname(LOG), "evolution_due.flag"), "w", encoding="utf-8") as _f:
                    _f.write(ev)
                state.setdefault("evolution_done", {})[ev] = True
                with open(STATE, "w", encoding="utf-8") as _sf:
                    json.dump(state, _sf)
            except Exception:
                pass
        # loop_dead SIGNAL: write a flag the overseer polls, but DO NOT die. Exiting on loop_dead was the recurring
        # cause of "the watcher stopped" -- during a long ATTENDED turn no hook fires to respawn it, so it stayed dead.
        # The watcher's job is to STAY ALIVE and MONITOR; relaunch decisions are the overseer's (it reads the flag).
        # Only tick_budget exits (and that respawns). The watcher is now TRULY absolute. (root fix 2026-06-06)
        if dead:
            _append(f"[{ts}] loop_dead signal dead={dead} -- flag written; watcher CONTINUES (no self-kill)")
            try:
                with open(os.path.join(os.path.dirname(LOG), "loop_dead.flag"), "w", encoding="utf-8") as _f:
                    _f.write(",".join(dead))
            except Exception:
                pass
        # STALL SIGNAL: a loop ALIVE but making no PROGRESS (checkpoint+lanes frozen) for STALL_TICKS = hung. Flag it
        # for the overseer; do NOT die. Reset the counter so it fires once per stall episode, not every tick.
        if alive:
            sig = _progress_sig(alive)
            stall = stall + 1 if sig == prev_sig else 0
            prev_sig = sig
            if stall >= STALL_TICKS:
                _append(f"[{ts}] loop_stalled signal (no progress {stall} ticks) alive={alive} -- flag written; watcher CONTINUES")
                try:
                    with open(os.path.join(os.path.dirname(LOG), "loop_stalled.flag"), "w", encoding="utf-8") as _f:
                        _f.write(",".join(alive))
                except Exception:
                    pass
                stall = 0
        # 60s-cadence META dual-view SIGNAL: write a flag the overseer polls, but DO NOT die. Exiting every META_TICKS
        # was self-inflicting ~5-min watcher deaths (undermining the "absolute" 60s watcher). The overseer reads the
        # flag at its own cadence; the genuine wakes (loop_dead / loop_stalled / evolution_due) still exit. (fix 2026-06-06)
        if (tick + 1) % META_TICKS == 0:
            _append(f"[{ts}] meta_due signal (tick {tick+1}) -- flag written; watcher CONTINUES (no self-kill)")
            try:
                with open(os.path.join(os.path.dirname(LOG), "meta_due.flag"), "w", encoding="utf-8") as _f:
                    _f.write(ts)
            except Exception:
                pass
        if tick < a.max_ticks - 1:
            time.sleep(60)
    # tick budget exhausted -> RESPAWN a fresh watcher (truly ABSOLUTE: never die on budget; only exit to wake the
    # overseer for ACTIONABLE events -- loop_dead / evolution_due). Fresh process = bounded memory + clean state.
    # (This is why the "absolute" watcher silently went down twice on 2026-06-06: it exited on tick_budget and
    # nobody respawned it. Self-respawn closes that.)
    if a.max_ticks < 5:  # smoke-test / one-shot run -> do NOT respawn (a low budget would tight-loop)
        _append(f"[{dt.datetime.now():%H:%M:%S}] EXIT reason=tick_budget (no respawn; max_ticks<5)")
        print("EXIT tick_budget"); return 0
    _append(f"[{dt.datetime.now():%H:%M:%S}] tick_budget -> RESPAWN fresh watcher (absolute; started {started})")
    try:
        if os.name == "nt":
            # WINDOWLESS respawn: CREATE_NO_WINDOW (no console flash / focus steal) + CREATE_NEW_PROCESS_GROUP. Was
            # DETACHED_PROCESS, which can still flash a console; CREATE_NO_WINDOW is the no-window flag (and the two
            # are mutually exclusive). Use pythonw.exe too so there is no console subsystem at all.
            kw = {"creationflags": 0x08000000 | 0x00000200}  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        else:
            kw = {"start_new_session": True}
        exe = sys.executable
        if os.name == "nt":  # pythonw.exe -> no console window
            exe = exe.lower().replace("python.exe", "pythonw.exe")
            if not os.path.exists(exe):
                exe = sys.executable
        # carry --respawn so the child BYPASSES the singleton check (our heartbeat is still fresh as we exit) but
        # still CLAIMS the lock -> the lineage stays SINGULAR across respawns. De-dupe sys.argv so --respawn isn't
        # appended twice across successive respawns.
        child_argv = [x for x in sys.argv[1:] if x != "--respawn"] + ["--respawn"]
        subprocess.Popen([exe, os.path.abspath(__file__)] + child_argv,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, **kw)
    except Exception as e:
        _append(f"respawn FAILED: {e} -- watcher down, overseer must relaunch")
    print("EXIT tick_budget (respawned)"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
