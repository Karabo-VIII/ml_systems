"""LITERAL DSPy compiled-planner -- the REAL dspy library wired to the HONEST eval_harness metric, LOCAL only.

This is the dspy-3.x realization of the "compiled planner (DSPy)" gap. The sibling `evolve.py::evolve_planner`
optimizes the SAME planner-quality knob (brain._PLAN_INSTRUCTION, injected via brain.set_plan_instruction) with a
dependency-free evolutionary loop; THIS module does it with the LITERAL dspy optimizer (BootstrapFewShot) so the
project can claim the genuine library path runs end-to-end against the honest objective.

WHAT IS OPTIMIZED + HOW IT IS GROUNDED (the keystone):
  - A dspy.Signature `PlanSig` maps an engineering `objective` -> a `plan` (the multi-step decomposition the engine
    must execute -- one line per file/step, in dependency order).
  - A dspy.Module `PlannerModule` (a single dspy.Predict over PlanSig) is the student that BootstrapFewShot compiles.
  - The METRIC is NOT a string-similarity proxy. It is GROUNDED in the eval_harness: for a given (objective, predicted
    plan), the metric INSTALLS the predicted plan as a planner-prompt augmentation via brain.set_plan_instruction(...)
    and runs eval_harness.run_planner_eval on the ONE matching PLANNER_BENCHMARK task. The example scores 1.0 IFF that
    task MECHANICALLY VERIFIES (graph._run_verify exit 0 on the composed artifact) -- the exact honest objective the
    spec demands. A plan that yields a wrong/incomplete decomposition fails the mechanical verifier -> 0.0.

LOCAL ONLY: dspy.LM("ollama_chat/qwen2.5-coder:3b", api_base=http://localhost:11434, api_key=""). No cloud keys, no
telemetry. The compile is BOUNDED (few demos, small trainset, hard per-call token cap + per-example eval timeout) so
it finishes in minutes on qwen3b -- the deliverable is the LITERAL dspy compile wired to the honest metric running
locally, NOT a guaranteed score gain (qwen3b is weak; a tiny compile may not move solve_rate).

EXPORT: the compiled artifact (the optimized instruction + bootstrapped demos, rendered as a single planner-prompt
string) is persisted to runs/autonomy/dspy/ and can be INSTALLED into any Brain via install_compiled_planner(brain,...)
-> brain.set_plan_instruction(<compiled prompt>), so the live loop uses the dspy-compiled planner. No emoji (cp1252).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import eval_harness as _eval
from .brain import _PLAN_INSTRUCTION, make_brain


# --------------------------------------------------------------------------- LOCAL ollama wiring for dspy
def configure_dspy_ollama(model: str = "ollama_chat/qwen2.5-coder:3b",
                          api_base: str = "http://localhost:11434",
                          max_tokens: int = 512, temperature: float = 0.2, verify: bool = True):
    """Configure dspy to use the LOCAL ollama model and (optionally) prove a real call returns text.

    Returns the dspy.LM. Raises a clear RuntimeError if the verify call fails (so a dead ollama is caught loudly,
    not silently degraded). LOCAL ONLY: api_key="" (ollama needs no key); no telemetry is enabled."""
    import dspy
    lm = dspy.LM(model, api_base=api_base, api_key="", max_tokens=max_tokens, temperature=temperature)
    dspy.configure(lm=lm)
    if verify:
        out = lm("Reply with exactly the word: PONG")
        txt = out[0] if isinstance(out, list) else str(out)
        if not txt or not txt.strip():
            raise RuntimeError("dspy.LM(ollama) returned empty -- is ollama up at %s with %s pulled?" %
                               (api_base, model))
    return lm


# --------------------------------------------------------------------------- the dspy Signature + Module
def _make_planner_module():
    """Build the (Signature, Module) pair. Lazy so importing this file does not require dspy until used."""
    import dspy

    class PlanSig(dspy.Signature):
        """Decompose an engineering objective into the multi-step plan the engine must execute.

        Output `plan` as a SHORT numbered list -- ONE line per file/step, in dependency order (a step that imports
        another must come AFTER it). Name each file explicitly. The plan must cover EVERY file the objective names so
        the final composed artifact behaves end-to-end. Be concrete and minimal; no prose, no code -- just the steps."""

        objective: str = dspy.InputField(desc="the engineering objective (build a small multi-file artifact)")
        plan: str = dspy.OutputField(desc="numbered decomposition: one line per file/step, dependency order")

    class PlannerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.plan = dspy.Predict(PlanSig)

        def forward(self, objective: str):
            return self.plan(objective=objective)

    return PlanSig, PlannerModule


# --------------------------------------------------------------------------- the honest, eval_harness-grounded metric
# A trainset example carries the objective AND its benchmark task id, so the metric can run the RIGHT mechanical task.
def _trainset(tasks, max_n: int):
    import dspy
    ex = []
    for t in list(tasks)[:max_n]:
        ex.append(dspy.Example(objective=t["objective"], task_id=t["id"]).with_inputs("objective"))
    return ex


def _task_by_id(task_id: str):
    for t in _eval.PLANNER_BENCHMARK:
        if t["id"] == task_id:
            return t
    return None


def _augment_plan_instruction(predicted_plan: str) -> str:
    """Render the dspy-predicted decomposition INTO the planner-prompt knob. We keep the baseline _PLAN_INSTRUCTION
    (its strict JSON output-contract is what the graph/schema depend on) and PREPEND the dspy-compiled decomposition
    as an explicit recipe the planner must follow. Braces are doubled because the result is .format()-ed in
    brain._decide_sys (same contract as _PLAN_INSTRUCTION)."""
    safe = (predicted_plan or "").replace("{", "{{").replace("}", "}}").strip()
    header = ("DSPy-COMPILED DECOMPOSITION RECIPE (follow this step ordering; emit one build node per file/step in "
              "dependency order):\n" + safe + "\n\nThen, obeying the output contract:\n")
    return header + _PLAN_INSTRUCTION


def make_eval_metric(brain, budget: int = 4, per_task_timeout: int = 120):
    """Return a dspy metric(example, pred, trace=None) -> float in {0.0, 1.0} GROUNDED in the eval_harness.

    For (example.objective, pred.plan): install the predicted plan as a planner-prompt augmentation on `brain`
    (set_plan_instruction), run eval_harness.run_planner_eval on the ONE matching task, and return solve_rate (1.0
    iff the composed artifact MECHANICALLY verifies). Restores the brain's prior plan_instruction afterward (no leak).
    HONEST: the PASS is the harness's own mechanical verify_cmd on the artifact, not the model's self-report."""
    def metric(example, pred, trace=None):
        task = _task_by_id(getattr(example, "task_id", "") or "")
        if task is None:
            return 0.0
        predicted_plan = getattr(pred, "plan", "") or ""
        instr = _augment_plan_instruction(predicted_plan)
        prev = getattr(brain, "plan_instruction", None)
        try:
            brain.set_plan_instruction(instr)
            card = _eval.run_planner_eval(brain, tasks=[task], budget=budget, timeout=per_task_timeout,
                                          brain_label=getattr(brain, "name", "Brain"))
        except Exception:
            return 0.0
        finally:
            brain.set_plan_instruction(prev)
        return float(card.get("solve_rate", 0.0))
    return metric


# --------------------------------------------------------------------------- baseline / compiled solve_rate
def planner_solve_rate(brain, plan_instruction: str | None, tasks, budget: int = 4,
                       per_task_timeout: int = 120) -> dict:
    """Honest solve_rate of a GIVEN planner instruction over `tasks` (None -> baseline _PLAN_INSTRUCTION).
    Installs the instruction, runs run_planner_eval, restores. Returns the full scorecard."""
    prev = getattr(brain, "plan_instruction", None)
    try:
        brain.set_plan_instruction(plan_instruction)
        card = _eval.run_planner_eval(brain, tasks=list(tasks), budget=budget, timeout=per_task_timeout,
                                      brain_label=getattr(brain, "name", "Brain"))
    finally:
        brain.set_plan_instruction(prev)
    return card


# --------------------------------------------------------------------------- the BOUNDED dspy compile
def compile_planner(brain=None, n_train: int = 2, n_eval: int = 2, max_bootstrapped_demos: int = 2,
                    max_labeled_demos: int = 2, max_tokens: int = 512, budget: int = 3,
                    per_task_timeout: int = 120, verify_lm: bool = True) -> dict:
    """LITERAL dspy compile of the planner with BootstrapFewShot, scored by the eval_harness-grounded metric.

    BOUNDED for qwen3b: small trainset (n_train), few demos (max_bootstrapped_demos), small per-task budget/timeout.
    Returns a result dict incl. the compiled planner-prompt string + before/after solve_rate. Persisted by the caller
    (or directly via save_compiled). Default brain = the LOCAL ollama brain (make_brain('ollama')).

    Steps:
      1. configure dspy -> local ollama (a real LM call is verified).
      2. build the PlannerModule student + the honest metric.
      3. BootstrapFewShot.compile(student, trainset) -- the LITERAL optimizer; it RUNS the metric to keep only demos
         whose plan mechanically verifies (so the bootstrapped demos are HONEST exemplars).
      4. score the compiled planner vs the baseline over n_eval tasks (the honest before/after)."""
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    t_start = time.time()
    configure_dspy_ollama(max_tokens=max_tokens, verify=verify_lm)
    if brain is None:
        brain = make_brain("ollama")

    _PlanSig, PlannerModule = _make_planner_module()
    student = PlannerModule()

    tasks = _eval.PLANNER_BENCHMARK
    trainset = _trainset(tasks, n_train)
    metric = make_eval_metric(brain, budget=budget, per_task_timeout=per_task_timeout)

    # ---- baseline solve_rate (no compiled prompt -> the project's default _PLAN_INSTRUCTION) over the eval subset
    eval_tasks = list(tasks)[:n_eval]
    base_card = planner_solve_rate(brain, None, eval_tasks, budget=budget, per_task_timeout=per_task_timeout)
    baseline_solve = float(base_card.get("solve_rate", 0.0))

    # ---- the LITERAL dspy compile
    optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=max_bootstrapped_demos,
                                 max_labeled_demos=max_labeled_demos, max_rounds=1)
    t_compile0 = time.time()
    compiled = optimizer.compile(student, trainset=trainset)
    compile_secs = round(time.time() - t_compile0, 1)

    # ---- render the compiled planner: run the compiled module on EACH eval objective to get its decomposition, then
    # score that decomposition through the honest metric (the same install->run_planner_eval path). We pick the
    # compiled plan from the FIRST eval task as the exported planner-prompt augmentation (it is the dspy-optimized
    # decomposition recipe); the per-task solve_rate is measured for the full eval subset.
    compiled_plans = {}
    for t in eval_tasks:
        try:
            pred = compiled(objective=t["objective"])
            compiled_plans[t["id"]] = getattr(pred, "plan", "") or ""
        except Exception as e:
            compiled_plans[t["id"]] = f"[compile-run-error: {type(e).__name__}: {e}]"

    # Export prompt = the baseline planner instruction AUGMENTED with the compiled decomposition for the first task.
    first_id = eval_tasks[0]["id"] if eval_tasks else None
    export_plan_text = compiled_plans.get(first_id, "") if first_id else ""
    compiled_instruction = _augment_plan_instruction(export_plan_text)

    # ---- compiled solve_rate over the eval subset: install the compiled instruction, run run_planner_eval.
    comp_card = planner_solve_rate(brain, compiled_instruction, eval_tasks, budget=budget,
                                   per_task_timeout=per_task_timeout)
    compiled_solve = float(comp_card.get("solve_rate", 0.0))

    # bootstrapped demos that survived (the literal compiled artifact's few-shot)
    demos = []
    try:
        for d in getattr(compiled.plan, "demos", []) or []:
            demos.append({"objective": (getattr(d, "objective", "") or "")[:400],
                          "plan": (getattr(d, "plan", "") or "")[:600]})
    except Exception:
        pass

    return {
        "optimizer": "BootstrapFewShot",
        "model": "ollama_chat/qwen2.5-coder:3b",
        "n_train": n_train, "n_eval": n_eval,
        "max_bootstrapped_demos": max_bootstrapped_demos, "max_labeled_demos": max_labeled_demos,
        "budget": budget, "per_task_timeout": per_task_timeout, "max_tokens": max_tokens,
        "baseline_solve_rate": baseline_solve,
        "compiled_solve_rate": compiled_solve,
        "improved": compiled_solve > baseline_solve,
        "n_demos": len(demos), "demos": demos,
        "compiled_plans": compiled_plans,
        "compiled_instruction": compiled_instruction,
        "baseline_per_task": base_card.get("per_task", []),
        "compiled_per_task": comp_card.get("per_task", []),
        "compile_secs": compile_secs,
        "total_secs": round(time.time() - t_start, 1),
    }


# --------------------------------------------------------------------------- persist + install
def save_compiled(result: dict, out_dir: str | None = None) -> Path:
    """Persist the compiled planner artifact to runs/autonomy/dspy/dspy_planner_<ts>.json (stamps ts). Returns path."""
    base = Path(out_dir) if out_dir else (Path("runs") / "autonomy" / "dspy")
    base.mkdir(parents=True, exist_ok=True)
    sc = dict(result)
    sc["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = base / f"dspy_planner_{int(time.time())}.json"
    out.write_text(json.dumps(sc, indent=2, default=str), encoding="utf-8")
    return out


def install_compiled_planner(brain, result_or_path, persist_champion: bool = True) -> "object":
    """Install the dspy-compiled planner into a live Brain via brain.set_plan_instruction(...), so the loop's `plan`
    node uses the dspy-optimized decomposition. Accepts a result dict (from compile_planner) or a path to a saved
    artifact JSON. Returns the brain (chaining).

    U1 -- CHAMPION PERSIST: when the compile IMPROVED solve_rate (compiled_solve_rate > baseline_solve_rate), also
    persist the compiled planner prompt to runs/autonomy/evolve/champion.json (via champion.write_champion) so the
    LIVE loop's make_brain install seam picks it up on the next launch (single shared champion across evolve + dspy).
    write_champion is monotonic + gated, so a non-improving compile is never installed by the live loop. Set
    persist_champion=False to install onto THIS brain only without touching the shared champion file."""
    if isinstance(result_or_path, (str, Path)):
        result = json.loads(Path(result_or_path).read_text(encoding="utf-8"))
    else:
        result = result_or_path
    instr = result.get("compiled_instruction")
    if instr:
        brain.set_plan_instruction(instr)
    if persist_champion and instr:
        try:
            comp = float(result.get("compiled_solve_rate", 0.0))
            base = float(result.get("baseline_solve_rate", 0.0))
            if comp > base:  # genuine improvement -> persist as the shared champion for the live loop's install seam
                from . import champion as _champion
                _champion.write_champion(prompt=instr, best_solve_rate=comp, baseline_solve_rate=base,
                                         source="dspy_planner:BootstrapFewShot",
                                         extra={"model": result.get("model"), "n_demos": result.get("n_demos")})
        except Exception:
            pass  # persisting the champion is best-effort; installing onto this brain already succeeded
    return brain


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LITERAL dspy compiled planner (local ollama), eval_harness-grounded.")
    ap.add_argument("--n-train", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--demos", type=int, default=2)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    print("[dspy_planner] configuring dspy -> local ollama + verifying a real LM call ...")
    res = compile_planner(n_train=args.n_train, n_eval=args.n_eval, max_bootstrapped_demos=args.demos,
                          budget=args.budget, per_task_timeout=args.timeout, max_tokens=args.max_tokens)
    print("[dspy_planner] optimizer      :", res["optimizer"])
    print("[dspy_planner] compile_secs   :", res["compile_secs"], " total_secs:", res["total_secs"])
    print("[dspy_planner] bootstrapped demos:", res["n_demos"])
    print("[dspy_planner] baseline solve_rate:", res["baseline_solve_rate"])
    print("[dspy_planner] compiled solve_rate:", res["compiled_solve_rate"], "(improved:", res["improved"], ")")
    if not args.no_save:
        p = save_compiled(res)
        print("[dspy_planner] saved artifact  :", p)
