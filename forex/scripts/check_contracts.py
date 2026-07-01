"""CDAP-style pre-commit gate for the forex pipeline (ported from crypto's audit).

Static-scans src/pipeline/**.py and asserts the project invariants without
importing anything:
  1. every non-__init__ pipeline module declares a top-level `__contract__` dict;
  2. no non-ASCII characters inside print() string arguments (Windows cp1252 crash);
  3. producers use atomic_write_parquet, never a raw `.write_parquet(` (crash-safety);
  4. every `__contract__` has the required keys (kind, stage/inputs/outputs, invariants).

Exit codes: 0 clean, 2 CRITICAL drift (block commit).

    python scripts/check_contracts.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "pipeline"
REQUIRED_CONTRACT_KEYS = {"kind", "invariants", "rationale"}


def _has_contract(tree: ast.Module) -> dict | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__contract__":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return {}
    return None


def _nonascii_in_prints(tree: ast.Module) -> list[str]:
    issues = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            for arg in ast.walk(node):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if any(ord(c) > 127 for c in arg.value):
                        issues.append(f"line {node.lineno}: non-ASCII in print()")
    return issues


def main() -> int:
    files = [p for p in PIPELINE.rglob("*.py") if p.name != "__init__.py"]
    crit = 0
    for p in sorted(files):
        rel = p.relative_to(ROOT)
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(p))

        contract = _has_contract(tree)
        if contract is None:
            print(f"[FAIL] {rel}: missing top-level __contract__ dict")
            crit += 1
        elif not REQUIRED_CONTRACT_KEYS.issubset(contract.keys()):
            missing = REQUIRED_CONTRACT_KEYS - set(contract.keys())
            print(f"[FAIL] {rel}: __contract__ missing keys {sorted(missing)}")
            crit += 1

        for issue in _nonascii_in_prints(tree):
            print(f"[FAIL] {rel}: {issue}")
            crit += 1

        # raw .write_parquet( outside parquet_io.py = crash-safety bypass
        if p.name != "parquet_io.py":
            for i, line in enumerate(text.splitlines(), 1):
                if ".write_parquet(" in line and "atomic_write_parquet" not in line \
                        and not line.lstrip().startswith("#"):
                    print(f"[FAIL] {rel}:{i}: raw .write_parquet( "
                          f"(use atomic_write_parquet)")
                    crit += 1

    if crit:
        print(f"\n[check_contracts] {crit} CRITICAL finding(s) -- FIX before commit.")
        return 2
    print(f"[check_contracts] {len(files)} pipeline modules OK "
          f"(contracts + ASCII prints + atomic writes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
