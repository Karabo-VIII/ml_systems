"""registry.py -- class-aware reader/writer for the two-leaderboard agent registry.

Two physically separate leaderboards (doc SS1.5) -- a critical anti-pattern guard:
  - runs/registry/forecasters.json : F champions, keyed by held-out IC + ShIC (TRUTH).
  - runs/registry/agents.json      : A1/A2/A1H champions, keyed by held-out COMPOUND
                                     + a required 'class' field (MONEY).
  - runs/registry/champion.json    : the single deployable; reads ONLY from agents.json.

This helper REJECTS an agent write without a valid `class in {A1, A2, A1H}` -- the
mechanical guard that stops the V16/V17 mis-file (an A1 backbone landing in the
forecaster zoo) from recurring at the registry layer.

No emoji (Windows cp1252). ASCII only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

__contract__ = {
    "kind": "agent_registry",
    "inputs": ["champion record dicts (agents must carry class in {A1,A2,A1H})"],
    "outputs": ["validated reads/writes to runs/registry/{forecasters,agents,champion}.json"],
    "invariants": [
        "agents.json writes REJECTED without class in {A1, A2, A1H}",
        "champion reads ONLY from agents.json (never forecasters.json)",
        "forecasters and agents leaderboards stay physically separate",
    ],
}

VALID_CLASSES = {"A1", "A2", "A1H"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = _REPO_ROOT / "runs" / "registry"
FORECASTERS_JSON = REGISTRY_DIR / "forecasters.json"
AGENTS_JSON = REGISTRY_DIR / "agents.json"
CHAMPION_JSON = REGISTRY_DIR / "champion.json"


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"champions": [], "_note": ""}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)  # atomic on the same filesystem


def list_forecasters() -> List[Dict[str, Any]]:
    return _load(FORECASTERS_JSON).get("champions", [])


def list_agents(cls: str | None = None) -> List[Dict[str, Any]]:
    champs = _load(AGENTS_JSON).get("champions", [])
    if cls is None:
        return champs
    if cls not in VALID_CLASSES:
        raise ValueError(f"unknown class {cls!r}; expected one of {sorted(VALID_CLASSES)}")
    return [c for c in champs if c.get("class") == cls]


def register_agent(record: Dict[str, Any]) -> None:
    """Append an A1/A2/A1H champion record. REJECTS an untagged write."""
    cls = record.get("class")
    if cls not in VALID_CLASSES:
        raise ValueError(
            f"agent registry write REJECTED: 'class' must be one of {sorted(VALID_CLASSES)}, "
            f"got {cls!r}. (Untagged agents are the V16/V17 mis-file class -- not allowed.)"
        )
    if "held_out_compound" not in record:
        raise ValueError("agent registry write REJECTED: missing 'held_out_compound' (the agent MONEY key).")
    data = _load(AGENTS_JSON)
    data.setdefault("champions", []).append(record)
    _save(AGENTS_JSON, data)


def get_deployable_champion() -> Dict[str, Any] | None:
    """The single deployable champion -- reads ONLY from agents.json (never forecasters)."""
    champs = _load(CHAMPION_JSON).get("champions", [])
    return champs[-1] if champs else None


if __name__ == "__main__":
    # Smoke: the leaderboards load and the class-tag guard rejects an untagged write.
    print("forecasters:", len(list_forecasters()), "agents:", len(list_agents()))
    try:
        register_agent({"agent_id": "x", "held_out_compound": 0.1})  # no class -> reject
        raise SystemExit("FAIL: untagged agent write was accepted")
    except ValueError:
        print("[registry] class-tag guard rejects untagged write -- OK")
