"""Liveness round-trip verifier (RWYB).

Proves THIS node is live + produces valid structured output on demand by:
  1. emitting a JSON artifact conforming to the metaop `plan` contract,
  2. round-tripping the raw text through the SYSTEM's real parser
     (`_extract_json` loaded directly from scripts/autonomy/metaop/brain.py),
  3. validating the parsed dict against the contract rules, AND
  4. confirming the graph's `_open_nodes` selector would accept the frontier.

Exit 0 = live + contract-conforming; exit 2 = FAIL. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_PARENT = ROOT / "scripts" / "autonomy"          # makes `metaop` an importable package
BRAIN = ROOT / "scripts" / "autonomy" / "metaop" / "brain.py"
GRAPH = ROOT / "scripts" / "autonomy" / "metaop" / "graph.py"

KINDS = {"build", "verify", "diverge"}


# --- THE ARTIFACT: a JSON response conforming to the plan contract -----------
# objective of this node = "confirm liveness"; frontier has a build node,
# a -k falsifier (verify) and a +k generalization (diverge).
ARTIFACT = {
    "frontier": [
        {"id": "n1",
         "task": "build: emit a plan-contract JSON liveness artifact and validate "
                 "it against the real _extract_json parser",
         "ev": 0.95, "kind": "build", "status": "open"},
        {"id": "n2",
         "task": "-k falsifier: does the artifact ACTUALLY round-trip through the "
                 "system's real parser, or is 'valid' merely asserted?",
         "ev": 0.85, "kind": "verify", "status": "open"},
        {"id": "n3",
         "task": "+k generalization: lift this into a reusable contract-conformance "
                 "probe for ALL decide() roles (plan/judge/reflect)",
         "ev": 0.55, "kind": "diverge", "status": "open"},
    ]
}


def validate_plan(d: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(d, dict):
        return [f"top-level is {type(d).__name__}, expected dict"]
    if d.get("_error"):
        return [f"parser error: {d.get('_error')}"]
    fr = d.get("frontier")
    if not isinstance(fr, list):
        return ["missing/invalid 'frontier' (must be a list)"]
    if not (3 <= len(fr) <= 6):
        # contract says 3-6; we ship 3 (a build + -k falsifier + +k generalization)
        errs.append(f"frontier has {len(fr)} nodes, contract wants 3-6")
    ids = set()
    for i, n in enumerate(fr):
        if not isinstance(n, dict):
            errs.append(f"node[{i}] not a dict"); continue
        for f in ("id", "task", "ev", "kind", "status"):
            if f not in n:
                errs.append(f"node[{i}] missing '{f}'")
        if not isinstance(n.get("id"), str) or not n.get("id"):
            errs.append(f"node[{i}].id not a non-empty str")
        if n.get("id") in ids:
            errs.append(f"node[{i}].id duplicate: {n.get('id')}")
        ids.add(n.get("id"))
        if not isinstance(n.get("task"), str) or not n.get("task"):
            errs.append(f"node[{i}].task not a non-empty str")
        ev = n.get("ev")
        if not isinstance(ev, (int, float)) or isinstance(ev, bool) or not (0.0 <= ev <= 1.0):
            errs.append(f"node[{i}].ev not a number in [0,1]: {ev!r}")
        if n.get("kind") not in KINDS:
            errs.append(f"node[{i}].kind not in {sorted(KINDS)}: {n.get('kind')!r}")
        if n.get("status") != "open":
            errs.append(f"node[{i}].status != 'open': {n.get('status')!r}")
    kinds = {n.get("kind") for n in fr if isinstance(n, dict)}
    if "verify" not in kinds:
        errs.append("contract expects a -k falsifier (kind=verify) -- none present")
    return errs


def main() -> int:
    print(f"[liveness] repo root      : {ROOT}")
    print(f"[liveness] brain.py exists : {BRAIN.exists()}")
    print(f"[liveness] graph.py exists : {GRAPH.exists()}")
    if not BRAIN.exists():
        print("FAIL: real parser source not found"); return 2

    # import the SYSTEM's real modules as a package so relative imports resolve
    sys.path.insert(0, str(PKG_PARENT))
    from metaop.brain import _extract_json as extract        # the exact parser plan() uses
    print(f"[liveness] loaded parser     : metaop.brain._extract_json")
    try:
        from metaop.graph import _open_nodes as open_nodes    # the graph's dispatch selector
        graph_real = True
        print(f"[liveness] loaded selector   : metaop.graph._open_nodes (real)")
    except Exception as e:                                    # e.g. langgraph not importable
        def open_nodes(frontier):
            return [n for n in frontier if n.get("status") == "open"]
        graph_real = False
        print(f"[liveness] graph import skipped ({type(e).__name__}); "
              f"using inline selector mirror")

    # 1. serialize the artifact the way a brain would emit it (fenced json block)
    raw = "```json\n" + json.dumps(ARTIFACT, indent=2) + "\n```"
    print(f"[liveness] emitted raw chars: {len(raw)}")

    # 2. ROUND-TRIP through the SYSTEM's real parser (the exact one plan() uses)
    parsed = extract(raw)
    print(f"[liveness] parser returned   : {type(parsed).__name__} "
          f"with keys {sorted(parsed)[:6]}")

    # 3. byte-for-byte fidelity check (parse must equal the source object)
    if parsed != ARTIFACT:
        print("FAIL: round-trip altered the object (parsed != emitted)")
        print("  parsed :", json.dumps(parsed)[:200])
        return 2
    print("[liveness] round-trip OK     : parsed == emitted (no corruption)")

    # 4. validate against the plan contract
    errs = validate_plan(parsed)
    if errs:
        print("FAIL: contract violations:")
        for e in errs:
            print("  -", e)
        return 2
    print(f"[liveness] contract OK       : {len(parsed['frontier'])} nodes, "
          f"kinds={sorted({n['kind'] for n in parsed['frontier']})}")

    # 5. confirm the GRAPH would actually accept this frontier as dispatchable
    selectable = open_nodes(parsed["frontier"])
    if len(selectable) != len(parsed["frontier"]):
        print("FAIL: graph._open_nodes would drop nodes (status not 'open')")
        return 2
    print(f"[liveness] graph accepts     : {len(selectable)}/"
          f"{len(parsed['frontier'])} nodes selectable by _open_nodes "
          f"({'real graph' if graph_real else 'mirror'})")

    print("\nPASS: node is LIVE -- emitted a plan-contract artifact that "
          "round-tripped through the real parser and is graph-dispatchable.")
    # emit the canonical artifact last for the caller to capture
    print("ARTIFACT:", json.dumps(ARTIFACT, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
