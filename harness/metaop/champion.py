"""CHAMPION INSTALL SEAM -- persist the evolved/dspy-optimized planner prompt + install it on the LIVE brain.

This closes the U1 gap: evolve_planner / compile_planner PRODUCE a better planner prompt, but nothing INSTALLED it
into the live loop's brain. The brain seam already exists (Brain.set_plan_instruction); this module is the missing
PERSISTENCE + GATED-APPLY layer between "an evolution run found a better prompt" and "the next solver run USES it".

CONTRACT (the only honest gate):
  - The champion is APPLIED to a live brain ONLY when champion.json exists AND it improves the GATE METRIC. The gate
    metric is the HELD-OUT test solve_rate when the record carries one (test_solve_rate vs baseline_test_solve_rate),
    else the train best_solve_rate (backward-compat). Gating on the held-out split -- the very thing evolve.py now
    scores the champion against -- means a champion is installed ONLY when it provably GENERALIZES beyond the tasks
    it was tuned on, not merely fits them (the 2026-07-03 audit fix: the old gate compared train-vs-train, so the
    elitism floor guaranteed nothing about generalization). best == baseline is NOT an improvement -> NOT applied.
  - write_champion(...) is the ONE writer: both evolution_loop (evolve_planner) and dspy_planner
    (install_compiled_planner) persist through it, so the format is single-sourced.

PATH: <harness_workspace>/evolve/champion.json -- under the CONFIGURABLE harness workspace, NOT any host repo (the
2026-07-03 audit fix: the old default wrote into the parent crypto mono-repo, contradicting config.py's standalone-
purity claim). Overridable via the HARNESS_CHAMPION_PATH env var or an explicit path arg. No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# the required fields a champion record carries (the directive's contract)
REQUIRED_FIELDS = ("prompt", "best_solve_rate", "baseline_solve_rate", "source")


def champion_path(path: str | os.PathLike | None = None) -> Path:
    """The canonical champion.json path. Override via arg or HARNESS_CHAMPION_PATH; default =
    <harness_workspace>/evolve/champion.json (the CONFIGURABLE workspace, never a host repo)."""
    if path:
        return Path(path)
    env = os.environ.get("HARNESS_CHAMPION_PATH")
    if env:
        return Path(env)
    from .config import workspace_root
    return workspace_root() / "evolve" / "champion.json"


def _gate_metric(rec: dict):
    """(candidate, baseline, which) the install-gate + monotonic floor compare on. Prefers the HELD-OUT test
    solve_rate when the record carries both test fields (generalization), else the train best/baseline solve_rate."""
    if rec.get("test_solve_rate") is not None and rec.get("baseline_test_solve_rate") is not None:
        try:
            return float(rec["test_solve_rate"]), float(rec["baseline_test_solve_rate"]), "test"
        except (TypeError, ValueError):
            pass
    try:
        return float(rec.get("best_solve_rate", 0.0)), float(rec.get("baseline_solve_rate", 0.0)), "train"
    except (TypeError, ValueError):
        return 0.0, 0.0, "train"


def read_champion(path: str | os.PathLike | None = None) -> dict | None:
    """Load champion.json -> dict, or None if it does not exist / is unreadable (never raises)."""
    p = champion_path(path)
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_champion(prompt: str, best_solve_rate: float, baseline_solve_rate: float, source: str,
                   path: str | os.PathLike | None = None, extra: dict | None = None,
                   test_solve_rate: float | None = None, baseline_test_solve_rate: float | None = None) -> Path | None:
    """Persist the champion planner prompt. The ONE writer used by evolution_loop + dspy_planner.

    Writes atomically (tmp + replace) so a concurrent reader never sees a half-file. Returns the path written, or
    None on failure (best-effort -- a persist failure must never break the producing loop). MONOTONIC on the GATE
    METRIC (held-out test solve_rate when supplied, else train best): a worse later run can't clobber a better
    earlier champion, so the champion-contract floor -- now a GENERALIZATION floor -- extends across runs.

    test_solve_rate / baseline_test_solve_rate: the HELD-OUT champion vs baseline solve_rate (from evolve_planner).
    When supplied, they become the install-gate + monotonic metric (generalization), so a champion is only ever
    installed when it beat the baseline on tasks it was NOT tuned on."""
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    try:
        best = float(best_solve_rate)
        base = float(baseline_solve_rate)
    except (TypeError, ValueError):
        return None
    p = champion_path(path)
    rec = {
        "prompt": prompt,
        "best_solve_rate": best,
        "baseline_solve_rate": base,
        "source": str(source),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": int(time.time()),
    }
    if test_solve_rate is not None and baseline_test_solve_rate is not None:
        try:
            rec["test_solve_rate"] = float(test_solve_rate)
            rec["baseline_test_solve_rate"] = float(baseline_test_solve_rate)
        except (TypeError, ValueError):
            pass
    cand_metric, base_metric, _which = _gate_metric(rec)
    rec["improved"] = cand_metric > base_metric
    # MONOTONIC guard on the GATE METRIC: keep the better incumbent if it already beats this candidate.
    prior = read_champion(path)
    if prior is not None:
        prior_metric, _pbase, _pw = _gate_metric(prior)
        if prior_metric >= cand_metric:
            return p  # incumbent generalizes at least as well -> leave it (no regression, no churn)
    if extra:
        rec.update(extra)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        os.replace(str(tmp), str(p))  # atomic on the same volume
        return p
    except Exception:
        return None


def is_improvement(rec: dict | None) -> bool:
    """True iff the record is a GENUINE improvement on the GATE METRIC: the HELD-OUT test solve_rate when present
    (generalization), else the train best_solve_rate -- strictly greater than its baseline. This is the apply-gate;
    best == baseline (evolve's safe no-op worst case) is NOT applied."""
    if not isinstance(rec, dict):
        return False
    cand, base, _which = _gate_metric(rec)
    return cand > base


def apply_champion(brain, path: str | os.PathLike | None = None) -> dict:
    """THE INSTALL SEAM. If a champion.json exists AND it is a genuine improvement (best > baseline), install its
    prompt onto `brain` via brain.set_plan_instruction(...) so the loop's `plan` node uses the evolved/dspy prompt.

    Returns a small status dict {applied, reason, best_solve_rate, baseline_solve_rate, source} for observability.
    NEVER raises (a malformed/absent champion leaves the brain at its baseline _PLAN_INSTRUCTION). Duck-typed: a brain
    without set_plan_instruction is left untouched with applied=False."""
    rec = read_champion(path)
    if rec is None:
        return {"applied": False, "reason": "no champion.json"}
    if not is_improvement(rec):
        cand, base, which = _gate_metric(rec)
        return {"applied": False, "reason": f"no gain on the {which} gate metric ({cand} <= {base}) -> no-op, not applied",
                "gate_metric": which, "candidate": cand, "baseline": base,
                "best_solve_rate": rec.get("best_solve_rate"), "test_solve_rate": rec.get("test_solve_rate")}
    prompt = rec.get("prompt")
    setter = getattr(brain, "set_plan_instruction", None)
    if not callable(setter) or not isinstance(prompt, str) or not prompt.strip():
        return {"applied": False, "reason": "brain has no set_plan_instruction or empty prompt"}
    try:
        setter(prompt)
    except Exception as e:
        return {"applied": False, "reason": f"set_plan_instruction raised: {type(e).__name__}"}
    cand, base, which = _gate_metric(rec)
    return {"applied": True, "reason": f"champion installed ({which} gate metric {cand} > {base})",
            "gate_metric": which, "candidate": cand, "baseline": base,
            "best_solve_rate": rec.get("best_solve_rate"), "test_solve_rate": rec.get("test_solve_rate"),
            "source": rec.get("source")}
