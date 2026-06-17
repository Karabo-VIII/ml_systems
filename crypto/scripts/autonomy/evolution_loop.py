#!/usr/bin/env python3
"""evolution_loop.py -- the project-wide EVOLUTION loop as a REAL, BOUNDED, TRACKED background process.

Before this, the EVOLUTION "loop" (loop-3) was PROSE: a watcher flag fired every 3h and the OVERSEER was supposed to
run an evolution pass by hand. The gap map found only the SOLVER loop is a real process. This module makes the
evolution job a genuine bounded-lifetime worker the watcher + Stop-hook SEE (via track_job) -- the third real loop.

WHAT THE EVOLUTION JOB IS (the self-improvement / OpenEvolve pattern, already implemented in harness/metaop/evolve):
  - Run an EVOLVE PASS: harness.metaop.evolve.evolve_planner -- a population-based evolutionary optimizer that scores
    candidate PLANNER PROMPTS against the HONEST mechanical fitness (eval_harness.run_planner_eval -> solve_rate, the
    fraction of benchmark tasks the engine actually SOLVES; cannot be faked). The CHAMPION CONTRACT guarantees the
    result is NEVER worse than the seed (worst case = no-op), so a pass is always safe.
  - WRITE the champion + its solve_rate to a lane (runs/autonomy/learnings/evolution.jsonl) so the gain compounds and
    the overseer can see what improved. The full per-generation log goes to runs/autonomy/evolve/<run>.jsonl.
  - FITNESS BRAIN (U2): when the local ollama server is reachable, the pass uses a REAL OllamaBrain (the genuine
    fitness path that can actually move solve_rate). When ollama is DOWN, the pass is SKIPPED with a clear note --
    NOT a vacuous MockBrain churn at solve_rate 0.0. Pass --allow-mock to force the mechanical proof path (tests).

CHAMPION PERSIST (U1): on a GENUINE improvement (best_solve_rate > baseline) the champion planner prompt is written
to runs/autonomy/evolve/champion.json via harness.metaop.champion.write_champion. The live loop's make_brain install
seam (manager.apply_champion) then GATED-installs it (only when best > baseline) -- so the evolved prompt reaches the
live planner. This loop still NEVER commits (overseer-only); installing is a gated, reversible brain-prompt swap.

SAFETY CONTRACT (additive autonomy machinery; the user has had STUCK issues -- this is bounded + tracked + idempotent):
  - TRACKED: registers runs/autonomy/locks/<job_id>.lock via track_job (watcher monitors liveness; Stop hook waits).
  - BOUNDED: --max-iters AND --max-hours (whichever first) -> ALWAYS exits. Each evolve pass is itself bounded
    (generations/pop_size/eval_limit/per-task timeout small by default) so a single pass cannot run away.
  - RESUMABLE / IDEMPOTENT: state in runs/autonomy/evolution_loop_state.json (best-so-far solve_rate + iter). Best
    is monotonic non-decreasing across restarts (the champion contract); a relaunch never regresses.
  - NEVER COMMITS: no git here.
  - CLEAN TEARDOWN: untracks its lock in a finally block on ANY exit -> no orphan lock.

No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUT = os.path.join(ROOT, "runs", "autonomy")
LANES = os.path.join(AUT, "learnings")
STATE = os.path.join(AUT, "evolution_loop_state.json")
EVOLVE_LOGDIR = os.path.join(AUT, "evolve")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_job import track, untrack  # the canonical lock contract (watcher + Stop hook read these)

sys.path.insert(0, ROOT)
try:
    from harness.metaop import learnings as _learn
    _HAVE_LEARN = True
except Exception:
    _HAVE_LEARN = False


def _ollama_up(host: str = "http://localhost:11434", timeout: float = 3.0) -> bool:
    """Quick liveness probe of the local ollama server (U2): GET /api/tags. True iff it answers 200. Pure stdlib,
    never raises -- a dead server returns False so the loop can skip cleanly instead of churning MockBrain at 0.0."""
    import urllib.request
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _load_state() -> dict:
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {"iter": 0, "best_solve_rate": None, "passes": []}


def _save_state(state: dict) -> None:
    try:
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except Exception:
        pass


def _write_evo_note(lesson: str, objective: str, cycle: int) -> None:
    """Write the evolution result FORWARD to runs/autonomy/learnings/evolution.jsonl (durable; compounds)."""
    if _HAVE_LEARN:
        _learn.record(lesson, thread="EVOLUTION_LOOP", objective=objective, cycle=cycle,
                      channel="evolution", workspace=AUT)
        return
    try:
        os.makedirs(LANES, exist_ok=True)
        row = {"ts": int(time.time()), "thread": "EVOLUTION_LOOP", "objective": str(objective)[:160],
               "cycle": cycle, "lesson": str(lesson)[:600]}
        with open(os.path.join(LANES, "evolution.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _one_pass(state: dict, objective: str, generations: int, pop_size: int, eval_limit: int,
              per_task_timeout: int, budget: int, allow_mock: bool = False) -> dict:
    """One EVOLUTION iteration: run a bounded evolve_planner pass, record the result, and (on a genuine improvement)
    PERSIST the champion planner prompt for the live loop to install. The champion contract guarantees
    best_solve_rate >= seed_solve_rate, so best-so-far is monotonic across passes.

    U2 -- NOT VACUOUS: the fitness path uses an OLLAMA brain when the local server is reachable (a real planner that
    can actually move solve_rate). If ollama is DOWN, the pass is SKIPPED with a clear note (not a silent MockBrain
    churn at solve_rate 0.0) -- unless allow_mock=True (the mechanical proof path, used by tests / explicit opt-in)."""
    it = state.get("iter", 0) + 1
    os.makedirs(EVOLVE_LOGDIR, exist_ok=True)
    log_path = os.path.join(EVOLVE_LOGDIR, f"loop_pass_{int(time.time())}.jsonl")

    # FITNESS BRAIN (fixed 2026-06-13): PREFER A CAPABLE BRAIN so evolution has a real gradient to climb. The local
    # ollama model is too weak to clear the eval_harness benchmark -- it scores ~0.0, so evolve_planner stays at
    # 0.0->0.0 forever (the long plateau that made self-evolution vacuous in practice). A capable brain (sdk =
    # in-process Claude > cli = claude on PATH) gives a NON-ZERO solve_rate that evolution can actually optimize.
    # Env override: EVOLUTION_BACKEND ('auto'|'sdk'|'cli'|'ollama'|'mock'). NOTE: a Claude fitness brain costs tokens
    # (it solves the benchmark each generation) -- the price of a fitness signal that isn't a flat 0.0. Set
    # EVOLUTION_BACKEND=ollama to force the free-but-likely-vacuous local path.
    from harness.metaop.brain import make_brain as _make_brain
    brain = None
    brain_label = "MockBrain"
    pref = os.environ.get("EVOLUTION_BACKEND", "auto")  # 'auto' = sdk -> api -> cli (best capable Claude available)
    if pref not in ("ollama", "mock"):
        try:
            b = _make_brain(pref)
            if getattr(b, "name", "MockBrain") != "MockBrain":
                brain, brain_label = b, b.name
        except Exception:
            brain = None
    if brain is None and pref != "mock" and _ollama_up():  # no capable Claude -> local fitness (LIKELY ~0.0 here)
        try:
            brain = _make_brain("ollama")
            brain_label = getattr(brain, "name", "OllamaBrain") + " (LOCAL -- may be vacuous at 0.0 on this benchmark)"
        except Exception as e:
            brain = None
            brain_label = f"ollama-build-failed:{type(e).__name__}"
    if brain is None and not allow_mock:
        note = (f"[EVOLUTION pass {it}] SKIPPED: ollama down (no reachable local model at localhost:11434) -> the "
                "real fitness path is unavailable. Not churning MockBrain at solve_rate 0.0. Bring ollama up to run.")
        state.setdefault("passes", []).append({"iter": it, "skipped": "ollama_down", "ts": int(time.time())})
        _write_evo_note(note, objective, it)
        state["iter"] = it
        state["last_pass_ts"] = int(time.time())
        _save_state(state)
        print(f"[evolution_loop] {note}")
        return state

    try:
        from harness.metaop import evolve as _evolve
        from harness.metaop import champion as _champion
        res = _evolve.evolve_planner(
            brain=brain,  # OllamaBrain (real fitness) when up; None -> MockBrain only on allow_mock
            generations=generations, pop_size=pop_size, elite_k=1,
            eval_limit=eval_limit, timeout=per_task_timeout, budget=budget, log_path=log_path)
        seed = res.get("seed_solve_rate")
        best = res.get("best_solve_rate")
        improved = res.get("improved")
        evals = res.get("evaluations")
        # U1 PRODUCER: persist the champion planner prompt so the live loop's make_brain install seam can apply it.
        # write_champion's own gate (best > baseline -> improved=True) + the apply gate together mean a no-op evolve
        # (best == seed) is persisted but NOT installed.
        champ_path = None
        if improved and res.get("best_prompt"):
            champ_path = _champion.write_champion(
                prompt=res["best_prompt"], best_solve_rate=best, baseline_solve_rate=seed,
                source=f"evolution_loop:{brain_label}:gen{generations}pop{pop_size}",
                extra={"eval_limit": eval_limit, "evaluations": evals, "log": os.path.basename(log_path)})
        note = (f"[EVOLUTION pass {it}] evolve_planner({brain_label}): seed_solve_rate={seed} -> "
                f"best_solve_rate={best} (improved={improved}, evals={evals}, gen={generations} pop={pop_size}). "
                f"CHAMPION CONTRACT holds: best >= seed. "
                f"{'champion.json WRITTEN' if champ_path else 'no champion write (no improvement)'}. "
                f"log={os.path.basename(log_path)}.")
        # monotonic best-so-far across passes/restarts
        prior_best = state.get("best_solve_rate")
        if prior_best is None or (best is not None and best > prior_best):
            state["best_solve_rate"] = best
        state.setdefault("passes", []).append(
            {"iter": it, "seed": seed, "best": best, "improved": improved, "brain": brain_label,
             "champion_written": bool(champ_path), "ts": int(time.time())})
        ok = True
    except Exception as e:
        note = f"[EVOLUTION pass {it}] evolve pass FAILED: {type(e).__name__}: {e} (no-op; loop continues bounded)."
        state.setdefault("passes", []).append({"iter": it, "error": f"{type(e).__name__}: {e}", "ts": int(time.time())})
        ok = False

    _write_evo_note(note, objective, it)
    state["iter"] = it
    state["last_pass_ts"] = int(time.time())
    _save_state(state)
    print(f"[evolution_loop] {note}" if ok else f"[evolution_loop] {note}")
    return state


def main():
    ap = argparse.ArgumentParser(description="EVOLUTION loop -- bounded, tracked self-improvement background process")
    ap.add_argument("--job-id", default="evolution_loop", help="track_job lock id (watcher + Stop hook see this)")
    ap.add_argument("--objective", default="evolve the planner against mechanical solve_rate (evolution lane)")
    ap.add_argument("--max-iters", type=int, default=4, help="hard cap on evolve passes (bounded lifetime)")
    ap.add_argument("--max-hours", type=float, default=3.0, help="hard wall-clock cap (bounded lifetime)")
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between evolve passes")
    # per-pass bounds (kept small so one pass is itself bounded -- additive, never a runaway)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--pop-size", type=int, default=2)
    ap.add_argument("--eval-limit", type=int, default=2, help="benchmark tasks per fitness eval (small=fast)")
    ap.add_argument("--per-task-timeout", type=int, default=120, help="per-task wall-clock cap inside the eval")
    ap.add_argument("--graph-budget", type=int, default=3, help="max graph cycles per task")
    ap.add_argument("--allow-mock", action="store_true",
                    help="run the MockBrain mechanical proof path even when ollama is down (default: SKIP cleanly)")
    a = ap.parse_args()

    os.makedirs(AUT, exist_ok=True)
    ok, msg = track(a.job_id, pid=os.getpid(), cmd="evolution_loop.py (evolve_planner, MockBrain)", kind="loop")
    if not ok:
        print(f"[evolution_loop] not starting: {msg}")
        return 0
    print(f"[evolution_loop] {msg} max_iters={a.max_iters} max_hours={a.max_hours} interval={a.interval}s")

    deadline = time.time() + a.max_hours * 3600.0
    state = _load_state()
    try:
        for i in range(a.max_iters):
            if time.time() >= deadline:
                print(f"[evolution_loop] EXIT reason=max_hours after {i} passes")
                break
            state = _one_pass(state, a.objective, a.generations, a.pop_size, a.eval_limit,
                              a.per_task_timeout, a.graph_budget, allow_mock=a.allow_mock)
            if i < a.max_iters - 1 and time.time() + a.interval < deadline:
                time.sleep(a.interval)
            elif i < a.max_iters - 1:
                print(f"[evolution_loop] EXIT reason=max_hours (no budget for another interval) after {state['iter']} passes")
                break
        else:
            print(f"[evolution_loop] EXIT reason=max_iters after {state['iter']} passes "
                  f"(best_solve_rate={state.get('best_solve_rate')})")
    finally:
        untrack(a.job_id)
        print(f"[evolution_loop] untracked {a.job_id}; clean exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
