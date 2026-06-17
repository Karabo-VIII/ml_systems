"""CHAMPION INSTALL SEAM -- persist the evolved/dspy-optimized planner prompt + install it on the LIVE brain.

This closes the U1 gap: evolve_planner / compile_planner PRODUCE a better planner prompt, but nothing INSTALLED it
into the live loop's brain. The brain seam already exists (Brain.set_plan_instruction); this module is the missing
PERSISTENCE + GATED-APPLY layer between "an evolution run found a better prompt" and "the next solver run USES it".

CONTRACT (the only honest gate):
  - The champion is APPLIED to a live brain ONLY when champion.json exists AND best_solve_rate > baseline_solve_rate.
    best == baseline (the evolve champion-contract worst case = no-op) is NOT an improvement -> NOT applied. This is
    the same mechanical-fitness gate evolve.py / dspy_planner.py measure (run_planner_eval -> solve_rate), so a
    champion is only ever installed when it provably beat the baseline on the honest objective.
  - write_champion(...) is the ONE writer: both evolution_loop (evolve_planner) and dspy_planner
    (install_compiled_planner) persist through it, so the format is single-sourced.

PATH: <repo_root>/runs/autonomy/evolve/champion.json (the live crypto loop's workspace). Resolved relative to this
file's repo root so it is stable regardless of cwd; overridable via the HARNESS_CHAMPION_PATH env var or an explicit
path arg (keeps the harness reusable without pinning it to crypto). No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# repo root = .../ml_systems (harness/metaop/champion.py -> parents[2])
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHAMPION = _REPO_ROOT / "runs" / "autonomy" / "evolve" / "champion.json"

# the required fields a champion record carries (the directive's contract)
REQUIRED_FIELDS = ("prompt", "best_solve_rate", "baseline_solve_rate", "source")


def champion_path(path: str | os.PathLike | None = None) -> Path:
    """The canonical champion.json path. Override via arg or HARNESS_CHAMPION_PATH; default = the crypto live loop's
    runs/autonomy/evolve/champion.json."""
    if path:
        return Path(path)
    env = os.environ.get("HARNESS_CHAMPION_PATH")
    return Path(env) if env else _DEFAULT_CHAMPION


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
                   path: str | os.PathLike | None = None, extra: dict | None = None) -> Path | None:
    """Persist the champion planner prompt. The ONE writer used by evolution_loop + dspy_planner.

    Writes atomically (tmp + replace) so a concurrent reader never sees a half-file. Returns the path written, or
    None on failure (best-effort -- a persist failure must never break the producing loop). MONOTONIC: only writes
    when the new best_solve_rate is >= any existing champion's best_solve_rate, so a worse later run can't clobber a
    better earlier champion (the champion-contract floor extends across runs)."""
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    try:
        best = float(best_solve_rate)
        base = float(baseline_solve_rate)
    except (TypeError, ValueError):
        return None
    p = champion_path(path)
    # MONOTONIC guard: keep the better incumbent if it already beats this candidate.
    prior = read_champion(path)
    if prior is not None:
        try:
            if float(prior.get("best_solve_rate", -1.0)) >= best:
                return p  # incumbent is at least as good -> leave it (no regression, no churn)
        except (TypeError, ValueError):
            pass
    rec = {
        "prompt": prompt,
        "best_solve_rate": best,
        "baseline_solve_rate": base,
        "source": str(source),
        "improved": best > base,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": int(time.time()),
    }
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
    """True iff the record is a GENUINE improvement: best_solve_rate strictly > baseline_solve_rate. This is the
    apply-gate -- best == baseline (evolve's safe no-op worst case) is NOT applied."""
    if not isinstance(rec, dict):
        return False
    try:
        return float(rec.get("best_solve_rate")) > float(rec.get("baseline_solve_rate"))
    except (TypeError, ValueError):
        return False


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
        return {"applied": False, "reason": "best_solve_rate <= baseline_solve_rate (no-op, not applied)",
                "best_solve_rate": rec.get("best_solve_rate"), "baseline_solve_rate": rec.get("baseline_solve_rate")}
    prompt = rec.get("prompt")
    setter = getattr(brain, "set_plan_instruction", None)
    if not callable(setter) or not isinstance(prompt, str) or not prompt.strip():
        return {"applied": False, "reason": "brain has no set_plan_instruction or empty prompt"}
    try:
        setter(prompt)
    except Exception as e:
        return {"applied": False, "reason": f"set_plan_instruction raised: {type(e).__name__}"}
    return {"applied": True, "reason": "champion installed (best > baseline)",
            "best_solve_rate": rec.get("best_solve_rate"), "baseline_solve_rate": rec.get("baseline_solve_rate"),
            "source": rec.get("source")}
