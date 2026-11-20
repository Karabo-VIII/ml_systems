"""
CDAP Contract Loader — Axis 1 of the Contract-Driven Audit Protocol.

Each component (pipeline stage / strategy module / model trainer / ...)
declares a top-of-file `__contract__` dict:

    __contract__ = {
        "kind": "pipeline_stage" | "strategy" | "model" | "simulator" | "feature" | ...,
        "inputs": {
            "args":        ["--universe {u10|u50|u100}", "--workers"],
            "upstream":    ["data/raw/<SYM>/aggTrades/*.parquet"],
            "config_keys": ["data.start_date"],
        },
        "outputs": {
            "files":       "data/processed/hawkes/daily/hawkes_branching_daily_*.parquet",
            "columns":     ["date", "asset", "eta_total", ...],
            "row_count":   "n_assets * n_days",
            "value_ranges": {"eta_total": [0.0, 0.99]},
        },
        "invariants": {
            "asset_set_eq":   "downstream:frontier_consolidate",
            "atomic_write":   True,
            "purge_gap":      ">=400",
        },
        "rationale": "...",
    }

This module:
    - Discovers files with __contract__ (via AST grep — no execution).
    - Validates that consumer input contracts match producer output contracts.
    - Surfaces drift as findings (compatible with check_invariants.py format).

Use:
    from src.audit.contract_loader import load_all_contracts, validate_contracts
    contracts = load_all_contracts()
    findings = validate_contracts(contracts)
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Contract:
    file:       str
    kind:       str = "unknown"
    stage:      str = ""        # e.g. "hawkes_branching", "chimera_v51"
    module:     str = ""        # for non-stage components (e.g. "WalkForwardSplitter")
    inputs:     Dict[str, Any] = field(default_factory=dict)
    outputs:    Dict[str, Any] = field(default_factory=dict)
    invariants: Dict[str, Any] = field(default_factory=dict)
    rationale:  str = ""


def _extract_contract_dict(source: str) -> Optional[dict]:
    """Parse the source for a top-level `__contract__ = {...}` assignment.

    Returns the dict if found and parseable, None otherwise. Uses AST so
    we do NOT execute the module (avoids import-time side effects).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__contract__":
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
    return None


def load_contract_for_file(path: Path) -> Optional[Contract]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    if "__contract__" not in text:
        return None
    d = _extract_contract_dict(text)
    if not isinstance(d, dict):
        return None
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return Contract(
        file=rel,
        kind=str(d.get("kind", "unknown")),
        stage=str(d.get("stage") or ""),
        module=str(d.get("module") or ""),
        inputs=d.get("inputs") or {},
        outputs=d.get("outputs") or {},
        invariants=d.get("invariants") or {},
        rationale=str(d.get("rationale") or ""),
    )


def load_all_contracts(roots: Optional[List[Path]] = None) -> List[Contract]:
    """Discover every Python file under `roots` (default: src/) that declares
    a `__contract__` dict and parse it.
    """
    if roots is None:
        roots = [PROJECT_ROOT / "src"]
    contracts: List[Contract] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts or "archive" in p.parts:
                continue
            c = load_contract_for_file(p)
            if c is not None:
                contracts.append(c)
    return contracts


def summary_table(contracts: List[Contract]) -> str:
    """Pretty-print a per-file contract summary."""
    lines = ["+--- CDAP Contracts -----+--------------+----------------------+",
             "| file                    | kind         | invariants           |",
             "+-------------------------+--------------+----------------------+"]
    for c in sorted(contracts, key=lambda x: x.file):
        inv_keys = ",".join(sorted(c.invariants.keys()))[:20]
        f_disp = c.file.replace("src/", "")[:24]
        lines.append(f"| {f_disp:<24} | {c.kind:<12} | {inv_keys:<20} |")
    lines.append("+-------------------------+--------------+----------------------+")
    return "\n".join(lines)


def validate_contracts(contracts: List[Contract]) -> List[dict]:
    """Cross-contract validation. Currently checks:
        - For producers declaring `outputs.files`, no two contracts may
          claim conflicting glob patterns.
        - For invariants of form 'asset_set_eq: downstream:<other>', the
          referenced contract must exist.
    """
    findings: List[dict] = []

    # Map output globs -> producers
    glob_owner: Dict[str, str] = {}
    for c in contracts:
        outs = c.outputs.get("files")
        if isinstance(outs, str):
            outs = [outs]
        if not outs:
            continue
        for g in outs:
            if g in glob_owner and glob_owner[g] != c.file:
                findings.append({
                    "severity": "warn", "category": "contract",
                    "name": "duplicate_output_glob", "file": c.file,
                    "detail": f"output glob {g!r} also claimed by {glob_owner[g]}",
                })
            else:
                glob_owner[g] = c.file

    # Cross-reference asset_set_eq invariants
    by_kind: Dict[str, Contract] = {}
    for c in contracts:
        by_kind.setdefault(c.kind, c)

    for c in contracts:
        for inv_name, inv_val in (c.invariants or {}).items():
            if isinstance(inv_val, str) and inv_val.startswith("downstream:"):
                target = inv_val.split(":", 1)[1]
                # Match if target appears in any other contract's stage / kind / module / file.
                hit = any(
                    target == oc.stage
                    or target == oc.kind
                    or target == oc.module
                    or target in oc.file
                    for oc in contracts
                )
                if not hit:
                    findings.append({
                        "severity": "warn", "category": "contract",
                        "name": "missing_downstream_target", "file": c.file,
                        "detail": f"invariant {inv_name}={inv_val!r} references non-existent contract",
                    })

    return findings


# ─── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list all discovered contracts")
    ap.add_argument("--validate", action="store_true", help="run cross-contract validation")
    args = ap.parse_args()

    contracts = load_all_contracts()
    print(f"Discovered {len(contracts)} contracts under src/")

    if args.list or not args.validate:
        print(summary_table(contracts))

    if args.validate:
        findings = validate_contracts(contracts)
        print(f"\nValidation findings: {len(findings)}")
        for f in findings:
            print(f"  [{f['severity'].upper()}] {f['category']}::{f['name']} "
                  f"[{f.get('file','-')}]: {f['detail']}")
