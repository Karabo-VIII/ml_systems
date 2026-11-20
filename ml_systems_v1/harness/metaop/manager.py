"""Harness MANAGER -- launch / status / resume / approve / stop the awake loop. The supervisory surface.

The graph (graph.py) is the persistent loop; THIS is what a human (or a present model instance) uses to: launch a
run, watch its observability trace, resume it across processes (durable SqliteSaver), approve human-gated
(irreversible) parked actions, and stop it cleanly. Project-agnostic: all state lives under the harness WORKSPACE.

  python -m harness.metaop.manager launch --objective "..." --budget 8 --backend mock [--durable --thread t1]
  python -m harness.metaop.manager status --thread t1
  python -m harness.metaop.manager resume --thread t1 --budget 16
  python -m harness.metaop.manager approve --thread t1 --node a3
  python -m harness.metaop.manager stop   --thread t1
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from .brain import make_brain
from .champion import apply_champion  # U1: install the evolved/dspy champion planner prompt onto the live brain
from .graph import build
from .config import workspace_root, trace_dir, checkpoint_db


def _locks_dir(workspace=None):
    d = workspace_root(workspace) / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire_lease(thread: str, workspace=None):
    """THREAD LEASE, race-proof via ATOMIC O_EXCL create -- only ONE process can create the lock."""
    lf = _locks_dir(workspace) / f"{thread}.lock"
    payload = json.dumps({"pid": os.getpid(), "ts": int(time.time()), "thread": thread}).encode("utf-8")
    for _ in range(2):
        try:
            fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
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
            if _pid_alive(info.get("pid", -1)):
                return False, f"thread '{thread}' already owned by live PID {info.get('pid')} (since {info.get('ts')})"
            try:
                lf.unlink()
            except FileNotFoundError:
                pass
    return False, f"thread '{thread}' lock contended -- try again"


def _release_lease(thread: str, workspace=None):
    try:
        (_locks_dir(workspace) / f"{thread}.lock").unlink()
    except FileNotFoundError:
        pass


def _reap_stale_locks(workspace=None) -> int:
    n = 0
    d = _locks_dir(workspace)
    for lf in d.glob("*.lock"):
        try:
            info = json.loads(lf.read_text(encoding="utf-8"))
            if not _pid_alive(info.get("pid", -1)):
                lf.unlink(); n += 1
        except Exception:
            try:
                lf.unlink(); n += 1
            except Exception:
                pass
    return n


def _kill_tree(pid: int) -> bool:
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


def _init(args, run_id):
    return {"objective": args.objective,
            "success_criteria": getattr(args, "success", "all nodes verified + neighborhood exhausted"),
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
    print(f"  thread   : '{thread}'  trace: {trace_dir()}/{last['run_id']}.jsonl")


def _checkpointer(durable, thread="t1", workspace=None):
    if not durable:
        from langgraph.checkpoint.memory import MemorySaver
        return None, MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    db = checkpoint_db(workspace, thread)
    db.parent.mkdir(parents=True, exist_ok=True)
    cm = SqliteSaver.from_conn_string(str(db))
    return cm, cm.__enter__()


def _build_kwargs(args):
    expert_mode = (args.mode == "expert")
    channel = args.learnings_channel or args.mode
    kw = dict(parallel=args.parallel, judges=args.judges, taper=args.taper, expert_mode=expert_mode,
              channel=channel, cwd=getattr(args, "cwd", None),
              persona_dir=getattr(args, "persona_dir", None),
              fill_window=getattr(args, "fill_window", False))
    # SKILLS + LOCALISED CONTEXT (optional): wire the skill SELECTOR + project-context pack onto the existing
    # recaller/framer host hooks (metaop/skills.py) -- the planner then receives the top-k relevant SKILL.md bodies
    # (payload["recall"]) + the project context (payload["framing"]). No graph change; off unless flags are passed.
    skills_dir = getattr(args, "skills_dir", None)
    context = getattr(args, "context", None)
    if skills_dir or context:
        try:
            from .skills import skills_recaller, context_framer, skill_harvester
            if skills_dir:
                kw["recaller"] = skills_recaller(skills_dir, k=getattr(args, "skills_k", 3))
                if getattr(args, "harvest", False):  # SELF-AUGMENT: grow the skill library from verified builds
                    kw["harvester"] = skill_harvester(skills_dir)
            if context:
                kw["framer"] = context_framer([c.strip() for c in str(context).split(",") if c.strip()])
        except Exception as e:
            print(f"  (skills/context wiring skipped: {type(e).__name__}: {e})")
    return kw, channel


def launch(args):
    _reap_stale_locks()
    ok, msg = _acquire_lease(args.thread)
    if not ok:
        print(f"REFUSED (thread lease): {msg}")
        print(f"  another run owns thread '{args.thread}'. Use a different --thread, or stop that run first.")
        return 2
    brain = make_brain(args.backend, domain=getattr(args, "domain", None) or "a software engine-builder project", cwd=getattr(args, "cwd", None))
    champ = apply_champion(brain)  # U1: GATED install -- only when champion.json exists AND best > baseline
    run_id = f"{args.thread}-{int(time.time())}"
    bkw, channel = _build_kwargs(args)
    cm, cp = _checkpointer(args.durable, args.thread)
    try:
        app = build(brain, checkpointer=cp, **bkw)
        cfg = {"configurable": {"thread_id": args.thread}}
        print(f"=== HARNESS awake loop  brain={brain.name}  mode={args.mode}  channel={channel}  "
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
    brain = make_brain(args.backend, domain=getattr(args, "domain", None) or "a software engine-builder project", cwd=getattr(args, "cwd", None))
    apply_champion(brain)  # U1: a resumed run gets the same gated champion-planner install as a fresh launch
    bkw, _ = _build_kwargs(args)
    cm, cp = _checkpointer(True, args.thread)
    try:
        app = build(brain, checkpointer=cp, **bkw)
        cfg = {"configurable": {"thread_id": args.thread}}
        snap = app.get_state(cfg)
        if not (snap and snap.values):
            print(f"no durable checkpoint for thread '{args.thread}'"); return 1
        base = dict(snap.values); base["budget"] = args.budget; base["status"] = "running"
        print(f"=== HARNESS RESUME thread '{args.thread}': cycle={base['cycle']}, frontier={len(base['frontier'])}, "
              f"ledger={len(base['ledger'])} -> continuing ===")
        rcfg = {"configurable": {"thread_id": args.thread + "-r"}}
        last = _stream(app, base, rcfg)
        _report(last, args.thread + "-r")
    finally:
        cm.__exit__(None, None, None)
    return 0


def approve(args):
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
    traces = sorted(trace_dir().glob(f"{args.thread}-*.jsonl"))
    print(f"=== HARNESS status: thread '{args.thread}' ===")
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
                  f"frontier={dict(Counter(n.get('status') for n in v.get('frontier', [])))} "
                  f"awaiting={v.get('awaiting_approval')}")
        cm.__exit__(None, None, None)
    except Exception as e:
        print(f"  (checkpoint read: {e})")
    return 0


def stop(args):
    lf = _locks_dir() / f"{args.thread}.lock"
    pid = None
    if lf.exists():
        try:
            pid = json.loads(lf.read_text(encoding="utf-8")).get("pid")
        except Exception:
            pid = None
    killed = []
    if pid and _pid_alive(pid) and _kill_tree(pid):
        killed.append(pid)
    _release_lease(args.thread)
    reaped = _reap_stale_locks()
    print(f"=== HARNESS stop: thread '{args.thread}' ===")
    print(f"  killed process tree(s): {killed or 'nothing was running'}")
    print(f"  lease released; {reaped} stale lock(s) reaped; durable checkpoint PRESERVED "
          f"(resume with `resume --thread {args.thread}`).")
    return 0


def evolve(args):
    """SELF-AUGMENTATION entry ('ask the harness to evolve itself'): optimize the PLANNER prompt against the honest
    solve-rate fitness (eval_harness, cannot be faked) and, on a real improvement over baseline, INSTALL it as the
    champion (champion.json). The next `launch`/`resume` auto-applies it via apply_champion. Elitism floor: a worse
    candidate is never written."""
    from . import evolve as _evo
    from .champion import write_champion
    brain = make_brain(args.backend, domain=getattr(args, "domain", None) or "a software engine-builder project", cwd=getattr(args, "cwd", None))
    print(f"=== HARNESS evolve  brain={brain.name}  generations={args.generations}  pop={args.pop_size} ===")
    try:
        res = _evo.evolve_planner(brain, generations=args.generations, pop_size=args.pop_size,
                                  eval_limit=args.eval_limit, budget=args.budget)
    except Exception as e:
        print(f"  evolve FAILED: {type(e).__name__}: {e}")
        return 1
    seed_f, best_f = res.get("seed_fitness", 0.0), res.get("best_fitness", 0.0)
    print(f"  planner solve_rate: seed {seed_f:.3f}  ->  best {best_f:.3f}  ({res.get('evaluations', 0)} evals)")
    if best_f > seed_f and res.get("best"):
        p = write_champion(str(res["best"]), best_f, seed_f, source="manager.evolve")
        print(f"  IMPROVEMENT -> champion installed at {p}  (next launch/resume applies it)")
    else:
        print("  no improvement over baseline -> champion NOT updated (elitism floor held)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="harness.metaop.manager")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--objective", default="characterize the problem surface")
        p.add_argument("--success", default="all nodes verified + neighborhood exhausted")
        p.add_argument("--budget", type=int, default=8)
        p.add_argument("--parallel", type=int, default=2)
        p.add_argument("--judges", type=int, default=3)
        p.add_argument("--taper", type=int, default=3)
        p.add_argument("--backend", default="mock",
                       choices=["auto", "sdk", "mock", "cli", "api", "ollama", "cascade", "litellm", "composite"])
        p.add_argument("--mode", default="plain", choices=["plain", "expert"])
        p.add_argument("--learnings-channel", default=None)
        p.add_argument("--thread", default="t1")
        p.add_argument("--durable", action="store_true")
        p.add_argument("--fill-window", dest="fill_window", action="store_true",
                       help="NO-IDLE-STOP: when the planned frontier completes with budget remaining, DRAIN-REPLAN "
                            "the next adjacent work (n+-k) instead of ending -- use the whole allocated window. "
                            "Stops on budget spent OR the brain returning no new work twice. Off = complete-then-stop "
                            "(default). Turn ON for timed/autonomous runs.")
        p.add_argument("--domain", default=None, help="task-flavor injected into the brain prompts")
        p.add_argument("--cwd", default=None, help="build cwd where workers run tools (default: current dir)")
        p.add_argument("--persona-dir", default=None, help="dir of <alias>.md personas for expert mode")
        p.add_argument("--skills-dir", default=None,
                       help="dir of SKILL.md skills (<name>/SKILL.md or <name>.md). The top-k relevant skills are "
                            "selected per objective + injected at plan time (progressive disclosure). The Claude-Code "
                            "skills equivalent for ANY model -- see metaop/skills.py.")
        p.add_argument("--skills-k", type=int, default=3, help="how many relevant skills to inject (default 3)")
        p.add_argument("--context", default=None,
                       help="comma-separated project-context files (CLAUDE.md equivalent) injected at plan time")
        p.add_argument("--harvest", action="store_true",
                       help="SELF-AUGMENT: when a build node passes the MECHANICAL verifier, author a SKILL.md for it "
                            "into --skills-dir, so the harness GROWS its own skill library from what it proves. "
                            "Requires --skills-dir.")
    sp = sub.add_parser("status"); sp.add_argument("--thread", default="t1")
    ap_ = sub.add_parser("approve"); ap_.add_argument("--thread", default="t1"); ap_.add_argument("--node", required=True)
    st = sub.add_parser("stop"); st.add_argument("--thread", default="t1")
    sub.add_parser("learnings")
    # SELF-EVOLUTION entry: improve the PLANNER prompt against the honest solve-rate fitness + install it as champion.
    ev = sub.add_parser("evolve", help="self-augment: evolve the planner prompt (-> champion.json; next launch uses it)")
    ev.add_argument("--backend", default="mock",
                    choices=["auto", "sdk", "mock", "cli", "api", "ollama", "cascade", "litellm", "composite"])
    ev.add_argument("--generations", type=int, default=2)
    ev.add_argument("--pop-size", dest="pop_size", type=int, default=3)
    ev.add_argument("--eval-limit", dest="eval_limit", type=int, default=4)
    ev.add_argument("--budget", type=int, default=4)
    ev.add_argument("--domain", default=None)
    # CONTINUOUS self-evolution daemon: keep evolving the planner hands-off (start once, it runs itself).
    im = sub.add_parser("improve", help="continuous self-evolution daemon: keep improving the planner, hands-off")
    im.add_argument("--backend", default="mock",
                    choices=["auto", "sdk", "mock", "cli", "api", "ollama", "cascade", "litellm", "composite"])
    im.add_argument("--rounds", type=int, default=10, help="how many rounds (0 = until --max-minutes)")
    im.add_argument("--max-minutes", dest="max_minutes", type=float, default=0.0,
                    help="wall-clock bound; run continuously until this elapses (0 = use --rounds)")
    im.add_argument("--generations", type=int, default=2)
    im.add_argument("--pop-size", dest="pop_size", type=int, default=3)
    im.add_argument("--eval-limit", dest="eval_limit", type=int, default=4)
    im.add_argument("--domain", default=None)
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
    if args.cmd == "evolve":
        return evolve(args)
    if args.cmd == "improve":
        from .selfimprove import self_improve
        print(f"=== HARNESS self-improve daemon  backend={args.backend}  "
              f"{'rounds=' + str(args.rounds) if args.rounds > 0 else 'max_minutes=' + str(args.max_minutes)} ===")
        self_improve(rounds=args.rounds, backend=args.backend, generations=args.generations,
                     pop_size=args.pop_size, eval_limit=args.eval_limit, max_minutes=args.max_minutes,
                     domain=getattr(args, "domain", None))
        return 0
    if args.cmd == "learnings":
        from . import learnings as _lr
        s = _lr.stats()
        print(f"=== HARNESS learnings: {s['total_lessons']} lessons across channels {s['per_channel']} ===")
        for ch in s["channels"]:
            print(f"  -- channel '{ch}' --")
            for r in _lr.recent(20, ch):
                print(f"    [{r.get('thread')}] {r.get('lesson')}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
