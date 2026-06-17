"""Operator MANAGER -- launch / monitor / resume / approve the awake loop. This is "call + manage it as META".

The graph (graph.py) is the persistent META that never sleeps. THIS is the supervisory surface a human (or a
present Claude instance) uses to: launch a run, watch its observability trace, resume it across processes, and
approve human-gated (irreversible) actions it parked. Durable state = SqliteSaver. No emoji (Windows cp1252).

  python -m metaop.manager launch  --objective "..." --budget 8 --parallel 2 [--durable --thread t1]
  python -m metaop.manager status  --thread t1
  python -m metaop.manager resume  --thread t1 --budget 16
  python -m metaop.manager approve --thread t1 --node a3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from .brain import make_brain
from .champion import apply_champion  # U1: install the evolved/dspy champion planner prompt onto the live brain
from .graph import build, ROOT, TRACE_DIR

def _db_path(thread: str):
    # PER-THREAD checkpoint DB -> concurrent rigs (different threads) never contend on one SQLite file
    return ROOT / "runs" / "autonomy" / f"metaop_{thread}.db"
LOCKS = ROOT / "runs" / "autonomy" / "locks"


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # metaop -> scripts/autonomy
from proc_liveness import alive as _pid_alive, create_time as _create_time  # G-J 2026-06-07: ONE liveness check


def _acquire_lease(thread: str):
    """THREAD LEASE (closes W6), race-proof via ATOMIC O_EXCL create -- the OS lets only ONE process create the
    lock, so two simultaneous launches can't both win (the earlier TOCTOU check-then-write let both through)."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    lf = LOCKS / f"{thread}.lock"
    payload = json.dumps({"pid": os.getpid(), "ts": int(time.time()), "thread": thread,
                          "created": _create_time(os.getpid())}).encode("utf-8")  # G-J: create-time -> reuse-safe
    for _ in range(2):
        try:
            fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomic: fails if it already exists
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            return True, "acquired"
        except FileExistsError:
            try:
                info = json.loads(lf.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            if _pid_alive(info.get("pid", -1), info.get("created")):
                return False, f"thread '{thread}' already owned by live PID {info.get('pid')} (since {info.get('ts')})"
            try:
                lf.unlink()  # stale (dead owner) -> reclaim and retry the atomic create
            except FileNotFoundError:
                pass
    return False, f"thread '{thread}' lock contended -- try again"


def _release_lease(thread: str):
    try:
        (LOCKS / f"{thread}.lock").unlink()
    except FileNotFoundError:
        pass


def _kill_tree(pid: int) -> bool:
    """Kill a process AND its descendants (the metaop python + its claude.exe workers). Clean stop, no orphans."""
    import subprocess
    try:
        if os.name == "nt":
            r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                               creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            return r.returncode == 0
        import signal
        os.kill(int(pid), signal.SIGTERM)
        return True
    except Exception:
        return False


def _find_metaop_pids(thread: str) -> list:
    """Find live metaop launcher PIDs for a thread (belt-and-suspenders if the lock is missing)."""
    import subprocess
    if os.name != "nt":
        return []
    ps = (r"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          r"Where-Object { $_.CommandLine -like '*metaop*' -and $_.CommandLine -like '*--thread " + thread + r"*' } | "
          r"Select-Object -ExpandProperty ProcessId")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


def _reap_stale_locks() -> int:
    """Remove lock files whose owner PID is dead (self-healing after a hard kill). Returns count reaped."""
    n = 0
    if not LOCKS.exists():
        return 0
    for lf in LOCKS.glob("*.lock"):
        try:
            info = json.loads(lf.read_text(encoding="utf-8"))
            if not _pid_alive(info.get("pid", -1), info.get("created")):
                lf.unlink(); n += 1
        except Exception:
            try:
                lf.unlink(); n += 1
            except Exception:
                pass
    return n


def _init(args, run_id):
    return {"objective": args.objective, "success_criteria": getattr(args, "success", "all nodes verified + neighborhood exhausted"),
            "frontier": [], "ledger": [], "budget": args.budget, "cycle": 0, "status": "running",
            "parallel": args.parallel, "run_id": run_id, "awaiting_approval": []}


def _stream(app, stream_input, cfg):
    last = None
    for step in app.stream(stream_input, cfg, stream_mode="values"):
        last = step
    return last


def _report(last, thread):
    fr = last["frontier"]
    print("=== TERMINATED ===")
    print(f"  status   : {last['status']}   cycles: {last['cycle']}")
    print(f"  frontier : {dict(Counter(n.get('status') for n in fr))}  ({len(fr)} nodes incl. dynamically-generated)")
    print(f"  ledger   : {len(last['ledger'])} lessons (project-evolution memory)")
    if last.get("awaiting_approval"):
        print(f"  AWAITING APPROVAL (HITL): {last['awaiting_approval']}  -> `manager approve --node <id>`")
    print(f"  thread   : '{thread}'  trace: runs/autonomy/traces/{last['run_id']}.jsonl")


def _checkpointer(durable, thread="t1"):
    if not durable:
        from langgraph.checkpoint.memory import MemorySaver
        return None, MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    db = _db_path(thread)
    db.parent.mkdir(parents=True, exist_ok=True)
    cm = SqliteSaver.from_conn_string(str(db))
    return cm, cm.__enter__()


def launch(args):
    _reap_stale_locks()  # self-heal: clear locks left by hard-killed prior runs before acquiring
    ok, msg = _acquire_lease(args.thread)
    if not ok:
        print(f"REFUSED (thread lease / W6): {msg}")
        print(f"  another run owns thread '{args.thread}'. Use a different --thread, or stop that run first.")
        return 2
    brain = make_brain(args.backend)
    champ = apply_champion(brain)  # U1: GATED install -- only when champion.json exists AND best > baseline
    run_id = f"{args.thread}-{int(time.time())}"
    expert_mode = (args.mode == "expert")
    channel = args.learnings_channel or args.mode  # default: separate lane per mode; same name -> pooled meta-lessons
    cm, cp = _checkpointer(args.durable, args.thread)
    try:
        app = build(brain, parallel=args.parallel, judges=args.judges, taper=args.taper, checkpointer=cp,
                    expert_mode=expert_mode, channel=channel,
                    fill_window=getattr(args, "fill_window", False))
        cfg = {"configurable": {"thread_id": args.thread}}
        print(f"=== METAOP awake loop  brain={brain.name}  mode={args.mode}  channel={channel}  "
              f"parallel={args.parallel}  budget={args.budget}  durable={args.durable} ===")
        print(f"    champion-planner: {'APPLIED' if champ.get('applied') else 'baseline'} ({champ.get('reason')})")
        last = _stream(app, _init(args, run_id), cfg)
        _report(last, args.thread)
    finally:
        if cm:
            cm.__exit__(None, None, None)
        _release_lease(args.thread)
    return 0


def resume(args):
    brain = make_brain(args.backend)
    apply_champion(brain)  # U1: a resumed run gets the same gated champion-planner install as a fresh launch
    expert_mode = (args.mode == "expert")
    channel = args.learnings_channel or args.mode
    cm, cp = _checkpointer(True, args.thread)
    try:
        app = build(brain, parallel=args.parallel, judges=args.judges, taper=args.taper, checkpointer=cp,
                    expert_mode=expert_mode, channel=channel,
                    fill_window=getattr(args, "fill_window", False))
        cfg = {"configurable": {"thread_id": args.thread}}
        snap = app.get_state(cfg)
        if not (snap and snap.values):
            print(f"no durable checkpoint for thread '{args.thread}'"); return 1
        base = dict(snap.values); base["budget"] = args.budget; base["status"] = "running"
        print(f"=== METAOP RESUME thread '{args.thread}': cycle={base['cycle']}, frontier={len(base['frontier'])}, ledger={len(base['ledger'])} -> continuing ===")
        rcfg = {"configurable": {"thread_id": args.thread + "-r"}}
        last = _stream(app, base, rcfg)
        _report(last, args.thread + "-r")
    finally:
        cm.__exit__(None, None, None)
    return 0


def approve(args):
    """HITL: release a parked irreversible node so the next run can execute it."""
    cm, cp = _checkpointer(True, args.thread)
    try:
        app = build(make_brain("mock"), checkpointer=cp)
        cfg = {"configurable": {"thread_id": args.thread}}
        snap = app.get_state(cfg)
        if not (snap and snap.values):
            print("no checkpoint"); return 1
        fr = [dict(n) for n in snap.values["frontier"]]
        found = False
        for n in fr:
            if n["id"] == args.node and n.get("status") == "awaiting_approval":
                n["status"] = "open"; n["approved"] = True; n["irreversible"] = False; found = True
        if not found:
            print(f"node '{args.node}' not found awaiting approval"); return 1
        app.update_state(cfg, {"frontier": fr})
        print(f"approved node '{args.node}' on thread '{args.thread}' -> resume to execute it")
    finally:
        cm.__exit__(None, None, None)
    return 0


def status(args):
    # read the latest trace for the thread + the durable checkpoint
    traces = sorted(TRACE_DIR.glob(f"{args.thread}-*.jsonl"))
    print(f"=== METAOP status: thread '{args.thread}' ===")
    if traces:
        lines = traces[-1].read_text(encoding="utf-8").strip().splitlines()
        ev = Counter(json.loads(l)["event"] for l in lines if l)
        print(f"  trace {traces[-1].name}: {dict(ev)}  ({len(lines)} events)")
        for l in lines[-6:]:
            print("   ", l[:160])
    else:
        print("  no trace yet")
    try:
        cm, cp = _checkpointer(True, args.thread)
        app = build(make_brain("mock"), checkpointer=cp)
        snap = app.get_state({"configurable": {"thread_id": args.thread}})
        if snap and snap.values:
            v = snap.values
            print(f"  checkpoint: cycle={v.get('cycle')} status={v.get('status')} "
                  f"frontier={dict(Counter(n.get('status') for n in v.get('frontier', [])))} awaiting={v.get('awaiting_approval')}")
        cm.__exit__(None, None, None)
    except Exception as e:
        print(f"  (checkpoint read: {e})")
    return 0


def stop(args):
    """Clean stop = kill the run's process TREE (python + its claude.exe workers) + release the lease + reap.
    The durable checkpoint is PRESERVED, so the work can be resumed."""
    lf = LOCKS / f"{args.thread}.lock"
    pid = None
    if lf.exists():
        try:
            _info = json.loads(lf.read_text(encoding="utf-8"))
            pid = _info.get("pid")
            # G-J: only target the lock PID if it is GENUINELY the tracked process (create-time match) -- never
            # kill a RECYCLED pid that now belongs to an unrelated process.
            if pid is not None and not _pid_alive(pid, _info.get("created")):
                pid = None
        except Exception:
            pid = None
    killed = []
    targets = ([pid] if pid else []) + _find_metaop_pids(args.thread)
    for p in list(dict.fromkeys(t for t in targets if t)):  # unique, preserve order
        if _pid_alive(p) and _kill_tree(p):
            killed.append(p)
    _release_lease(args.thread)
    reaped = _reap_stale_locks()
    print(f"=== METAOP stop: thread '{args.thread}' ===")
    print(f"  killed process tree(s): {killed or 'nothing was running'}")
    print(f"  lease released; {reaped} stale lock(s) reaped; durable checkpoint PRESERVED (resume with `resume --thread {args.thread}`).")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="metaop.manager")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--objective", default="characterize the opportunity surface")
        p.add_argument("--success", default="all nodes verified + neighborhood exhausted")
        p.add_argument("--budget", type=int, default=8)
        p.add_argument("--parallel", type=int, default=2)
        p.add_argument("--judges", type=int, default=3)
        p.add_argument("--taper", type=int, default=3)
        p.add_argument("--backend", default="auto",
                       choices=["auto", "sdk", "mock", "cli", "api", "persistent", "ollama", "cascade", "litellm"])
        p.add_argument("--mode", default="plain", choices=["plain", "expert"],
                       help="execution variant: plain (generic workers) or expert (attach .claude/agents specialists)")
        p.add_argument("--learnings-channel", default=None,
                       help="learnings lane (default = mode -> SEPARATE per variant; set both modes to one name to POOL)")
        p.add_argument("--thread", default="t1")
        p.add_argument("--durable", action="store_true")
        p.add_argument("--fill-window", dest="fill_window", action="store_true",
                       help="NO-IDLE-STOP: complete the planned frontier then DRAIN-REPLAN the next adjacent work "
                            "(n+-k) until the budget is spent OR the brain returns no new work -- use the whole "
                            "allocated window. Off = complete-then-stop (default). ON for timed/autonomous runs.")
    sp = sub.add_parser("status"); sp.add_argument("--thread", default="t1")
    ap_ = sub.add_parser("approve"); ap_.add_argument("--thread", default="t1"); ap_.add_argument("--node", required=True)
    st = sub.add_parser("stop"); st.add_argument("--thread", default="t1")
    lr = sub.add_parser("learnings")  # inspect the project-wide compounding memory
    args = ap.parse_args()
    if args.cmd == "launch":
        return launch(args)
    if args.cmd == "resume":
        return resume(args)
    if args.cmd == "status":
        return status(args)
    if args.cmd == "approve":
        return approve(args)
    if args.cmd == "stop":
        return stop(args)
    if args.cmd == "learnings":
        from . import learnings as _lr
        s = _lr.stats()
        print(f"=== METAOP learnings: {s['total_lessons']} lessons across channels {s['per_channel']} ===")
        for ch in s["channels"]:
            print(f"  -- channel '{ch}' --")
            for r in _lr.recent(20, ch):
                print(f"    [{r.get('thread')}] {r.get('lesson')}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
