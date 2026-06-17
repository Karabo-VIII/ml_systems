"""Harness EVOLVE -- a dependency-free EVOLUTIONARY optimizer (the AlphaEvolve / OpenEvolve pattern), LOCAL.

This is the SELF-EVOLUTION engine: it optimizes an artifact (a string -- here the PLANNER PROMPT) against an
HONEST, MECHANICAL fitness function (eval_harness.run_planner_eval -> solve_rate, the fraction of benchmark tasks
the engine actually SOLVES; cannot be faked -- see eval_harness.py). It IS the "compiled planner" goal that DSPy
would serve, but with ZERO external deps (no dspy, no litellm-required) -- pure stdlib + the harness brain.

WHY THIS CLOSES THE SELF-EVOLUTION GAP:
  - The loop COMPOUNDS: each generation scores a population of candidate planner prompts against the real harness,
    keeps the elite(s), and asks the BRAIN to propose DIRECTED mutations (given the current prompt + its solve_rate
    + which benchmark tasks failed). Experience -> a better artifact, measured mechanically. That is self-improvement
    on an honest objective, not a self-report.
  - The CHAMPION CONTRACT (elitism floor): evolve NEVER returns a candidate with fitness < the seed's. The optimized
    planner is provably never worse than the baseline (the worst case is "no change", which is a safe no-op).
  - DETERMINISTIC given a seed index: mutation is varied by (generation, index), not wall-clock randomness, so a run
    is reproducible. Fitness is CACHED by candidate hash (it is expensive -- a full harness eval per candidate).

TWO LAYERS (mirrors the AlphaEvolve split):
  1. `evolve(seed, fitness_fn, mutate_fn, ...)` -- the GENERIC core. Knows nothing about planners; works on ANY
     candidate type with a fitness_fn (candidate -> float, higher=better) and a mutate_fn (elite, context -> child).
  2. `evolve_planner(brain, ...)` -- the DEFAULT application: seed = graph/brain `_PLAN_INSTRUCTION`; fitness =
     run_planner_eval with that prompt INJECTED (via brain.set_plan_instruction -- the minimal graph-side hook) over
     a small task subset; mutate = ask the brain for an IMPROVED planner prompt given the failures (directed, not
     random). Returns the best planner prompt + its solve_rate vs the seed's.

No emoji (Windows cp1252). Append-only JSONL log per generation under runs/autonomy/evolve/<run>.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import brain as _brain
from . import eval_harness as _eval


# --------------------------------------------------------------------------- helpers
def _cand_hash(candidate) -> str:
    """Stable content hash of a candidate (used to CACHE its expensive fitness). Strings hash directly; anything
    else is JSON-canonicalized first so structurally-equal candidates share a cache slot."""
    if isinstance(candidate, str):
        blob = candidate.encode("utf-8", "replace")
    else:
        blob = json.dumps(candidate, sort_keys=True, default=str).encode("utf-8", "replace")
    return hashlib.sha1(blob).hexdigest()[:16]


def _append_log(log_path, record: dict) -> None:
    """Append one JSON record to log_path (best-effort -- a logging failure must NEVER break the optimizer)."""
    if not log_path:
        return
    try:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- the GENERIC evolutionary core
def evolve(seed, fitness_fn, mutate_fn, generations: int = 2, pop_size: int = 3, elite_k: int = 1,
           log_path=None) -> dict:
    """Population-based evolutionary search over an arbitrary candidate space (AlphaEvolve/OpenEvolve pattern).

    Args:
      seed        : the starting candidate (e.g. the current planner prompt). Generation 0's population is the seed
                    plus pop_size-1 mutations of it.
      fitness_fn  : candidate -> float (HIGHER is better). EXPENSIVE -> results are CACHED by candidate hash, so the
                    same candidate is never re-scored. A fitness that raises is treated as -inf (that candidate dies).
      mutate_fn   : (parent_candidate, context) -> child_candidate. `context` is a dict the application can use to
                    make a DIRECTED mutation: {generation, index, parent_fitness, best_fitness, history}. Determinism
                    is the application's responsibility (vary by generation+index, not wall-clock) -- the core passes
                    those in so a mutate_fn can be reproducible.
      generations : number of generations to evolve (>=0; 0 just scores the seed population).
      pop_size    : population size per generation (>=1).
      elite_k     : how many top candidates survive unchanged into the next generation (1 <= elite_k <= pop_size).
      log_path    : optional JSONL file; one record per generation is appended (best-so-far, population fitnesses).

    Returns: {"best": <champion candidate>, "best_fitness": float, "history": [per-gen records],
              "seed_fitness": float, "evaluations": int}.

    CHAMPION CONTRACT (elitism floor): the returned best is the highest-fitness candidate EVER scored, and it is
    seeded by the original `seed`, so best_fitness >= seed_fitness ALWAYS (the optimizer never regresses below the
    baseline). best-so-far is monotonically non-decreasing across generations."""
    pop_size = max(1, int(pop_size))
    elite_k = max(1, min(int(elite_k), pop_size))
    generations = max(0, int(generations))

    cache: dict = {}            # candidate_hash -> fitness (the expensive call is paid ONCE per distinct candidate)
    evaluations = 0

    def score(cand) -> float:
        nonlocal evaluations
        h = _cand_hash(cand)
        if h in cache:
            return cache[h]
        try:
            f = float(fitness_fn(cand))
        except Exception:
            f = float("-inf")       # a candidate whose fitness errors simply dies -- never breaks the run
        cache[h] = f
        evaluations += 1
        return f

    # --- generation 0: the seed + (pop_size-1) directed mutations of it -------------------------------------------
    seed_fitness = score(seed)
    best = seed
    best_fitness = seed_fitness        # CHAMPION starts at the seed -> the floor can never drop below it

    population = [seed]
    for idx in range(1, pop_size):
        ctx = {"generation": 0, "index": idx, "parent_fitness": seed_fitness,
               "best_fitness": best_fitness, "history": []}
        try:
            child = mutate_fn(seed, ctx)
        except Exception:
            child = seed               # a failed mutation degrades to the parent (never crashes the gen)
        population.append(child)

    history: list = []

    def run_generation(gen: int, pop: list) -> list:
        nonlocal best, best_fitness
        scored = [(c, score(c)) for c in pop]
        scored.sort(key=lambda cf: cf[1], reverse=True)       # best first
        # update the monotonic champion (>= so ties never replace the incumbent -> stable, and seed stays champion
        # on a tie, satisfying "never return a WORSE champion than the seed").
        if scored[0][1] > best_fitness:
            best, best_fitness = scored[0][0], scored[0][1]
        rec = {"generation": gen, "best_fitness": best_fitness, "best_hash": _cand_hash(best),
               "pop_fitness": [round(f, 4) if f != float("-inf") else None for _, f in scored],
               "pop_size": len(pop), "evaluations": evaluations, "ts": int(time.time())}
        history.append(rec)
        _append_log(log_path, rec)
        return scored

    # score generation 0
    scored = run_generation(0, population)

    # --- generations 1..G: keep elites, fill the rest with DIRECTED mutations of the elites -----------------------
    for gen in range(1, generations + 1):
        elites = [c for c, _f in scored[:elite_k]]
        next_pop = list(elites)                                # elites survive UNCHANGED (the floor is preserved)
        idx = 0
        while len(next_pop) < pop_size:
            parent = elites[idx % len(elites)]
            # parent is drawn from `scored` so identity-match is guaranteed; the default guards against any edge.
            parent_fit = next((f for c, f in scored if c is parent), best_fitness)
            ctx = {"generation": gen, "index": idx, "parent_fitness": parent_fit,
                   "best_fitness": best_fitness, "history": list(history)}
            try:
                child = mutate_fn(parent, ctx)
            except Exception:
                child = parent
            next_pop.append(child)
            idx += 1
        scored = run_generation(gen, next_pop)

    return {"best": best, "best_fitness": best_fitness, "seed_fitness": seed_fitness,
            "history": history, "evaluations": evaluations}


# --------------------------------------------------------------------------- the DEFAULT application: planner prompt
# The mutation prompt asked of the BRAIN. It is a META-task: improve a PLANNER PROMPT given its measured solve_rate
# and which benchmark tasks failed. The brain returns the improved prompt as a JSON string field so it is robustly
# parseable (weak local models trail prose; _extract_json tolerates it). We keep the brace convention: the planner
# prompt is later .format()-ed for {domain}, so literal braces must stay DOUBLED -- we instruct the brain to do so.
_MUTATE_PROMPT_SYS = (
    "You are optimizing a PLANNER PROMPT for an autonomous build-and-verify engine. The planner prompt instructs a "
    "model to DECOMPOSE an objective into a small frontier of build/verify/diverge nodes; the engine then executes "
    "the plan and a MECHANICAL verifier scores the final artifact (solve_rate = fraction of benchmark tasks solved). "
    "Your job: given the CURRENT planner prompt, its measured solve_rate, and WHICH tasks failed, return an IMPROVED "
    "planner prompt that should raise solve_rate. Keep it a drop-in REPLACEMENT (same JSON output-contract on the "
    "first line: a frontier of nodes with id/task/ev/kind/status). Make the decomposition guidance sharper for the "
    "kinds of tasks that FAILED (e.g. multi-file build order, importing one module from another, running a CLI). "
    "PRESERVE all literal curly braces as DOUBLED braces ({{ }}) -- the prompt is later .format()-ed. Respond with "
    "EXACTLY one JSON object: {{\"plan_instruction\": \"<the full improved planner prompt as a single string>\"}} "
    "and NOTHING else."
)


def _mutate_planner(brain, parent_prompt: str, context: dict) -> str:
    """DIRECTED mutation: ask the brain for an improved planner prompt given the parent prompt + its solve_rate +
    the failed-task ids carried in `context`. On any failure (brain error / no usable string / unchanged) -> return
    the PARENT (a no-op mutation, never a regression). Determinism: the brain is low-temp; we also append the
    (generation,index) so repeated mutations in the same gen differ slightly."""
    fails = context.get("failed_tasks", [])
    parent_fit = context.get("parent_fitness")
    gen, idx = context.get("generation", 0), context.get("index", 0)
    payload = {
        "current_planner_prompt": parent_prompt,
        "measured_solve_rate": parent_fit,
        "failed_task_ids": fails,
        "note": (f"variant {gen}.{idx}: propose a DISTINCT improvement from any prior variant; focus the "
                 "decomposition guidance on the failed task types."),
    }
    try:
        # use a NON-overridden brain call: a transient plain brain so the mutation request is NOT itself filtered
        # through the candidate planner prompt (we are reasoning ABOUT prompts, not planning a build).
        out = brain.decide("evolve_mutation", payload, persona=_MUTATE_PROMPT_SYS)
        cand = out.get("plan_instruction") if isinstance(out, dict) else None
        if isinstance(cand, str) and cand.strip() and cand.strip() != parent_prompt.strip():
            return cand.strip()
    except Exception:
        pass
    return parent_prompt  # no usable mutation -> the parent survives (the elitism floor is never violated)


def _planner_fitness(brain, prompt: str, tasks, eval_limit: int, timeout: int, budget: int,
                     fail_sink: dict) -> float:
    """Fitness of a candidate PLANNER PROMPT = run_planner_eval with that prompt INJECTED over a SMALL task subset
    -> solve_rate in [0,1]. The injection is the minimal graph-side hook: brain.set_plan_instruction(prompt) makes
    the graph's `plan` node use THIS prompt (default None -> the baseline _PLAN_INSTRUCTION; see brain._decide_sys).
    Records the failed-task ids into fail_sink[prompt_hash] so the mutation step can do a DIRECTED improvement."""
    subset = list(tasks)[:max(1, eval_limit)]
    # INJECT the candidate planner prompt; restore afterward so the brain is left as found (no side effects leak).
    prev = getattr(brain, "plan_instruction", None)
    try:
        brain.set_plan_instruction(prompt)
        card = _eval.run_planner_eval(brain, tasks=subset, budget=budget, timeout=timeout,
                                      brain_label=getattr(brain, "name", "Brain"))
    finally:
        brain.set_plan_instruction(prev)
    fail_sink[_cand_hash(prompt)] = [r["id"] for r in card.get("per_task", []) if not r.get("passed")]
    return float(card.get("solve_rate", 0.0))


def evolve_planner(brain=None, generations: int = 2, pop_size: int = 3, elite_k: int = 1, eval_limit: int = 4,
                   timeout: int = 240, budget: int = 4, log_path=None, seed_prompt: str | None = None,
                   tasks=None) -> dict:
    """Optimize the PLANNER PROMPT (graph/brain `_PLAN_INSTRUCTION`) against the HONEST planner-mode solve_rate.

    This is the "compiled planner" deliverable, dependency-free (no dspy): it evolves the single planner-quality knob
    against the mechanical eval harness and returns a planner prompt that is NEVER worse than the baseline.

    Args:
      brain       : the Brain used BOTH to plan (fitness) AND to propose mutations. Default None -> a MockBrain
                    (mechanical proof path: no LLM, fast). Pass an OllamaBrain/CliBrain/... for a real run.
      generations : evolutionary generations (default 2).
      pop_size    : candidates per generation (default 3).
      elite_k     : elites carried unchanged (default 1).
      eval_limit  : how many PLANNER_BENCHMARK tasks each fitness eval runs (SMALL -> fast; the fitness is expensive).
      timeout     : per-task wall-clock cap (seconds) inside run_planner_eval.
      budget      : max graph cycles per task inside run_planner_eval.
      log_path    : JSONL generation log (default runs/autonomy/evolve/planner_<ts>.jsonl under cwd).
      seed_prompt : the starting planner prompt (default = brain._PLAN_INSTRUCTION, the current baseline).
      tasks       : the benchmark to optimize against (default = eval_harness.PLANNER_BENCHMARK).

    Returns: {"best_prompt", "best_solve_rate", "seed_solve_rate", "improved": bool, "history", "evaluations",
              "log_path", "eval_limit", "generations", "pop_size"}. best_solve_rate >= seed_solve_rate ALWAYS.
    """
    if brain is None:
        brain = _brain.MockBrain()
    seed = seed_prompt if seed_prompt is not None else _brain._PLAN_INSTRUCTION
    tasks = list(tasks if tasks is not None else _eval.PLANNER_BENCHMARK)
    if log_path is None:
        log_path = Path("runs") / "autonomy" / "evolve" / f"planner_{int(time.time())}.jsonl"

    fail_sink: dict = {}

    def fitness_fn(prompt: str) -> float:
        return _planner_fitness(brain, prompt, tasks, eval_limit, timeout, budget, fail_sink)

    def mutate_fn(parent_prompt: str, context: dict) -> str:
        ctx = dict(context)
        ctx["failed_tasks"] = fail_sink.get(_cand_hash(parent_prompt), [])  # DIRECTED: tell the brain what failed
        return _mutate_planner(brain, parent_prompt, ctx)

    result = evolve(seed, fitness_fn, mutate_fn, generations=generations, pop_size=pop_size,
                    elite_k=elite_k, log_path=log_path)

    return {"best_prompt": result["best"], "best_solve_rate": result["best_fitness"],
            "seed_solve_rate": result["seed_fitness"],
            "improved": result["best_fitness"] > result["seed_fitness"],
            "history": result["history"], "evaluations": result["evaluations"],
            "log_path": str(log_path), "eval_limit": eval_limit, "generations": generations,
            "pop_size": pop_size, "best_prompt_changed": result["best"].strip() != seed.strip()}


if __name__ == "__main__":
    # MECHANICAL self-proof of the GENERIC core on a TOY fitness (no harness, no LLM): evolve a string toward a
    # target by char-match fraction. Shows population scored, elites kept, best monotonically non-decreasing.
    import string as _string

    TARGET = "evolve"
    ALPHABET = _string.ascii_lowercase

    def toy_fitness(s: str) -> float:
        return sum(1 for a, b in zip(s, TARGET) if a == b) / len(TARGET)

    def toy_mutate(parent: str, ctx: dict) -> str:
        # DETERMINISTIC mutation: flip ONE position chosen by (generation,index) toward a deterministic letter.
        gen, idx = ctx.get("generation", 0), ctx.get("index", 0)
        pos = (gen * 7 + idx * 3) % len(TARGET)
        repl = TARGET[pos] if (gen + idx) % 2 == 0 else ALPHABET[(gen + idx) % 26]
        chars = list(parent.ljust(len(TARGET), "a")[:len(TARGET)])
        chars[pos] = repl
        return "".join(chars)

    seed = "aaaaaa"
    res = evolve(seed, toy_fitness, toy_mutate, generations=3, pop_size=4, elite_k=1)
    print("TOY evolve -- seed_fitness:", res["seed_fitness"], "best_fitness:", res["best_fitness"],
          "best:", repr(res["best"]))
    bests = [h["best_fitness"] for h in res["history"]]
    print("best-so-far per generation:", bests)
    print("monotonic non-decreasing:", all(b2 >= b1 for b1, b2 in zip(bests, bests[1:])))
