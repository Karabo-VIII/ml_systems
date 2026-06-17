"""lifecycle_state.py -- G1 governance: strategy lifecycle state machine.

Per docs/STRATEGIC_OBJECTIVES.md §7-G1.

Reads config/lifecycle_registry.yaml; provides API for:
  - querying current state of any pillar
  - proposing state transitions (with gate checks)
  - emitting transition signals to runs/lifecycle/transitions.parquet
  - listing pillars by state (e.g., "all LIVE_PROMOTED" for meta-allocator)

USAGE
-----
```python
from audit.lifecycle_state import LifecycleRegistry

reg = LifecycleRegistry()
state = reg.state_of("REGIME_ROUTER_STRICT")
live_promoted = reg.pillars_in_state("LIVE_PROMOTED")

# Propose transition based on v3 metrics
ok, reason = reg.can_transition("REGIME_ROUTER_STRICT", "PAPER",
                                  {"sharpe": 1.2, "dd_pct": -8.0})
```

CONTRACT
--------
- Reads `config/lifecycle_registry.yaml` (truth) at construction
- Transitions write to `runs/lifecycle/transitions.parquet` + update yaml
- Gates evaluated against transition_rules below
- LIVE_PROMOTED transitions require manual confirmation flag
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_YAML = ROOT / "config" / "lifecycle_registry.yaml"
TRANSITIONS_DIR = ROOT / "runs" / "lifecycle"
TRANSITIONS_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "lifecycle_state_machine",
    "owner": "audit/governance",
    "outputs": "runs/lifecycle/transitions.parquet (audit trail)",
    "invariants": [
        "single source of truth at config/lifecycle_registry.yaml",
        "transitions evaluated against transition_rules below",
        "LIVE_PROMOTED requires manual --confirm flag (not auto-transitionable)",
    ],
}


STATES = ("BIRTH", "PAPER", "LIVE_PROBATION", "LIVE_PROMOTED",
           "DECAY_WATCH", "SUNSET", "ARCHIVED")


# Transition rules: gate from -> to requires these conditions
# Each rule: dict of (metric, op, threshold) tuples; ALL must pass
TRANSITION_RULES = {
    ("BIRTH", "PAPER"): {
        # Need at least one v3 window completed
        "n_v3_windows": (">=", 1),
    },
    ("PAPER", "LIVE_PROBATION"): {
        "sharpe": (">=", 0.5),
        "dd_pct": (">=", -10.0),
    },
    ("LIVE_PROBATION", "LIVE_PROMOTED"): {
        "n_v3_windows": (">=", 3),
        "sharpe_min_fold": (">=", 0.3),
        "pbo": ("<=", 0.30),
        # NOTE: LIVE_PROMOTED also requires manual --confirm; gate alone not enough
    },
    ("LIVE_PROMOTED", "DECAY_WATCH"): {
        # ANY of these fail → automatic DECAY_WATCH (OR-rule, not AND)
        "_or_rule": True,
        "sharpe_30d_vs_validated_ratio": ("<", 0.5),
        "ic_30d_drop_pct": (">", 30.0),
    },
    ("DECAY_WATCH", "SUNSET"): {
        "sharpe_60d": ("<", 0.3),
        "dd_60d_pct": ("<", -15.0),
    },
    ("SUNSET", "ARCHIVED"): {
        "days_in_sunset": (">=", 90),
    },
}


def _now() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


# ============================================================================
# Registry
# ============================================================================

class LifecycleRegistry:
    """Lifecycle registry reader + transition manager."""

    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = Path(yaml_path) if yaml_path else REGISTRY_YAML
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.yaml_path.exists():
            self._data = {"version": 1, "pillars": {}, "filters": {}}
            return
        with open(self.yaml_path, "r", encoding="utf-8") as fh:
            self._data = yaml.safe_load(fh) or {}

    def save(self) -> None:
        self._data["last_updated"] = _now()
        with open(self.yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=False)

    def state_of(self, pillar: str) -> Optional[str]:
        for key in ("pillars", "filters"):
            d = self._data.get(key, {}) or {}
            if pillar in d:
                return d[pillar].get("state")
        return None

    def pillars_in_state(self, state: str, kind: str = "pillars") -> List[str]:
        d = self._data.get(kind, {}) or {}
        return [name for name, info in d.items() if info.get("state") == state]

    def can_transition(self, pillar: str, target_state: str,
                         metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if pillar can move to target_state given supplied metrics."""
        current = self.state_of(pillar)
        if current is None:
            return False, f"pillar '{pillar}' not in registry"
        if target_state not in STATES:
            return False, f"unknown target state '{target_state}'"

        key = (current, target_state)
        if key not in TRANSITION_RULES:
            return False, f"no rule for {current} -> {target_state}"

        rules = dict(TRANSITION_RULES[key])  # copy
        is_or = rules.pop("_or_rule", False)

        failures: List[str] = []
        passes: List[str] = []
        for metric, (op, thresh) in rules.items():
            v = metrics.get(metric)
            if v is None:
                failures.append(f"{metric} not provided")
                continue
            if op == ">=" and v >= thresh:
                passes.append(f"{metric}={v}>={thresh}")
            elif op == "<=" and v <= thresh:
                passes.append(f"{metric}={v}<={thresh}")
            elif op == ">" and v > thresh:
                passes.append(f"{metric}={v}>{thresh}")
            elif op == "<" and v < thresh:
                passes.append(f"{metric}={v}<{thresh}")
            else:
                failures.append(f"{metric}={v} fails {op}{thresh}")

        if is_or:
            # OR-rule: any pass triggers
            if passes:
                return True, f"OR-rule fired: {passes[0]}"
            return False, f"OR-rule: no triggers ({'; '.join(failures)})"

        # AND-rule: all must pass
        if not failures:
            return True, f"all gates pass: {'; '.join(passes)}"
        return False, "; ".join(failures)

    def transition(self, pillar: str, target_state: str,
                    metrics: Dict[str, Any], force: bool = False) -> Tuple[bool, str]:
        """Execute transition if gates pass (or if force=True).

        LIVE_PROMOTED requires force=True (manual confirm).
        """
        if target_state == "LIVE_PROMOTED" and not force:
            return False, "LIVE_PROMOTED requires force=True (manual confirm)"

        ok, reason = self.can_transition(pillar, target_state, metrics) if not force \
                        else (True, "FORCED")
        if not ok:
            return False, reason

        prev_state = self.state_of(pillar)
        # Update yaml
        for key in ("pillars", "filters"):
            d = self._data.get(key, {}) or {}
            if pillar in d:
                d[pillar]["state"] = target_state
                d[pillar]["state_since"] = _now()
                d[pillar]["_last_transition_reason"] = reason
                break
        self.save()

        # Log transition
        try:
            import pandas as pd
            tlog = TRANSITIONS_DIR / "transitions.parquet"
            row = {
                "ts": _now(),
                "pillar": pillar,
                "from_state": prev_state,
                "to_state": target_state,
                "reason": reason,
                "metrics_json": str(metrics),
            }
            if tlog.exists():
                df = pd.read_parquet(tlog)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_parquet(tlog, index=False)
        except Exception:
            pass
        return True, reason


# ============================================================================
# CLI smoke test
# ============================================================================

def _smoke() -> int:
    reg = LifecycleRegistry()
    print(f"[SMOKE] Loaded {len(reg._data.get('pillars', {}))} pillars + "
          f"{len(reg._data.get('filters', {}))} filters from {reg.yaml_path}")
    for s in STATES:
        in_state = reg.pillars_in_state(s)
        print(f"  {s:<18s}: {len(in_state):>2d}  {in_state[:3]}{'...' if len(in_state)>3 else ''}")

    print("\n[SMOKE] Transition probe: REGIME_ROUTER_STRICT BIRTH -> PAPER")
    ok, reason = reg.can_transition("REGIME_ROUTER_STRICT", "PAPER",
                                      {"n_v3_windows": 0})
    print(f"  with 0 windows: ok={ok} reason={reason}")
    ok, reason = reg.can_transition("REGIME_ROUTER_STRICT", "PAPER",
                                      {"n_v3_windows": 2})
    print(f"  with 2 windows: ok={ok} reason={reason}")

    print("\n[SMOKE] LIVE_PROMOTED requires force:")
    ok, reason = reg.can_transition("CONSERVATIVE_3D", "LIVE_PROBATION",
                                      {"sharpe": 1.5, "dd_pct": -5.0})
    print(f"  PAPER->LIVE_PROBATION (auto): ok={ok} reason={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
