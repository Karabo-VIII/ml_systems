"""
CDAP — Wealth-Bot Audit-JSON Claim Contract Checker
====================================================

Scans every `runs/audit/**/data/*.json` (or a configured subset) and
verifies that any SHIP-tier claim block conforms to the canonical
claim contract defined in `src/wealth_bot/framework/claim_contract.py`.

Triggers a CDAP FAIL (exit 2) when:
  - An audit JSON declares `ship_claim` or has fields suggesting a
    SHIP-tier candidate (verdict=SHIP_CANDIDATE, all_4_positive=True
    + UNSEEN compound >= 50%) but missing required claim-contract fields.
  - top_3_pct_of_compound > 70% AT n<30 AND mechanism_falsifier_check
    is unverified.
  - passes_strict_gate=True declared but internal gates fail.

Exit codes:
    0  - all SHIP-tier claims pass contract validation
    1  - WARN findings only (e.g. missing field on a non-SHIP claim)
    2  - CRIT (BLOCK COMMIT): SHIP-tier claim missing required fields
         OR mechanism_falsifier_check fails AT high top-trade-pct.

Usage:
    python src/audit/check_wealth_bot_claims.py
    python src/audit/check_wealth_bot_claims.py --json    # machine-readable
    python src/audit/check_wealth_bot_claims.py --paths "runs/audit/SUBSET/**/data/*.json"

Wired into `check_invariants.py` run_audit() so CDAP runs both crawlers.

Provenance: 2026-05-25 INST-A RED-team audit of P4_route_basis_pos_only
revealed the mechanism claim ("filter strips top-tail-dependent trades")
was empirically FALSE. The filter kept ABC_AND's top 3 trades and
dropped diversifying ones. Pre-commit would have caught this had the
contract been enforced. This module is that enforcement.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to path so we can import claim_contract
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from wealth_bot.framework.claim_contract import (
        validate_claim_block, REQUIRED_FIELDS, CONTRACT_VERSION,
    )
except ImportError as e:
    print(f"[check_wealth_bot_claims] cannot import claim_contract: {e}", file=sys.stderr)
    sys.exit(2)


# Default scan paths
DEFAULT_SCAN_PATHS = [
    "runs/audit/**/data/*.json",
]

# JSON-key candidates that hold the ship-claim block. Scripts may use
# either of these keys for backward compatibility.
SHIP_CLAIM_KEYS = ["ship_claim", "claim", "ship_candidate_block"]


def _is_ship_tier_candidate(audit_data: dict) -> bool:
    """Heuristic: does this audit JSON declare a SHIP-tier claim?

    Triggers:
      - Has a `ship_claim` block (explicit declaration)
      - has `verdict: "SHIP" | "SHIP_CANDIDATE" | "SHIPPED"` at top level
      - has nested `all_4_positive: True` AND UNSEEN compound >= 50%
        AT root level (heuristic catch for legacy audit JSONs)
    """
    if any(k in audit_data for k in SHIP_CLAIM_KEYS):
        return True

    verdict = audit_data.get("verdict")
    if isinstance(verdict, str) and verdict.upper() in ("SHIP", "SHIPPED", "SHIP_CANDIDATE"):
        return True

    # Heuristic: scan top-level for the all_4_positive + UNSEEN >= 50% pattern
    if audit_data.get("all_4_positive") is True:
        unseen = audit_data.get("UNSEEN", {})
        if isinstance(unseen, dict):
            comp = unseen.get("compound_pct", 0.0)
            if isinstance(comp, (int, float)) and comp >= 50.0:
                return True

    return False


def _find_ship_claim_block(audit_data: dict) -> dict | None:
    """Extract the ship_claim block from an audit JSON, if present."""
    for key in SHIP_CLAIM_KEYS:
        if key in audit_data and isinstance(audit_data[key], dict):
            return audit_data[key]
    return None


def scan_audit_json(path: Path) -> list[dict]:
    """Validate one audit JSON. Returns list of finding dicts."""
    findings: list[dict] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        findings.append({
            "severity": "warn",
            "file": str(path.relative_to(PROJECT_ROOT)),
            "name": "json_decode_error",
            "detail": str(e),
        })
        return findings
    except (OSError, UnicodeDecodeError) as e:
        findings.append({
            "severity": "warn",
            "file": str(path.relative_to(PROJECT_ROOT)),
            "name": "read_error",
            "detail": str(e),
        })
        return findings

    if not isinstance(data, dict):
        return findings  # not a dict-shaped audit JSON; ignore (could be a list of trades, etc.)

    if not _is_ship_tier_candidate(data):
        return findings  # not a SHIP-tier candidate; contract doesn't apply

    block = _find_ship_claim_block(data)
    if block is None:
        # SHIP-tier claim declared (e.g. all_4_positive + UNSEEN>=50%) but
        # no explicit ship_claim block. WARN — legacy audit JSON. Future audits
        # should use build_ship_claim_block().
        findings.append({
            "severity": "warn",
            "file": str(path.relative_to(PROJECT_ROOT)),
            "name": "ship_tier_without_explicit_claim_block",
            "detail": (
                "SHIP-tier heuristics fired but no `ship_claim` block found. "
                "Recommend retroactive `build_ship_claim_block()` call to fill required fields."
            ),
        })
        return findings

    # Explicit ship_claim block found — validate
    errors = validate_claim_block(block)
    if errors:
        for e in errors:
            findings.append({
                "severity": "critical",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "name": "claim_contract_violation",
                "detail": e,
            })
    return findings


def run_audit(scan_paths: list[str] | None = None) -> tuple[list[dict], int]:
    """Scan audit JSONs across the configured paths. Return findings + exit code."""
    paths_to_scan = scan_paths or DEFAULT_SCAN_PATHS
    all_findings: list[dict] = []
    n_files_scanned = 0

    for pattern in paths_to_scan:
        for fp in glob.glob(str(PROJECT_ROOT / pattern), recursive=True):
            n_files_scanned += 1
            all_findings.extend(scan_audit_json(Path(fp)))

    n_critical = sum(1 for f in all_findings if f["severity"] == "critical")
    n_warn = sum(1 for f in all_findings if f["severity"] == "warn")

    if n_critical > 0:
        exit_code = 2
    elif n_warn > 0:
        exit_code = 1
    else:
        exit_code = 0

    return all_findings, exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description="CDAP wealth-bot audit-JSON claim-contract checker")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON output")
    ap.add_argument("--paths", nargs="*", default=None,
                    help="glob paths to scan (default: runs/audit/**/data/*.json)")
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    args = ap.parse_args()

    findings, exit_code = run_audit(args.paths)

    if args.json:
        print(json.dumps({"findings": findings, "exit_code": exit_code}, indent=2))
        return exit_code

    if exit_code == 0 and not args.quiet:
        print("[check_wealth_bot_claims] OK - all SHIP-tier claims pass contract validation")
        return 0

    n_crit = sum(1 for f in findings if f["severity"] == "critical")
    n_warn = sum(1 for f in findings if f["severity"] == "warn")
    header_severity = "CRIT" if n_crit > 0 else "WARN"
    print(f"[check_wealth_bot_claims] {header_severity} - {n_crit} CRIT, {n_warn} WARN findings")

    for f in findings:
        prefix = {"critical": "FAIL", "warn": "WARN", "info": "INFO"}.get(f["severity"], "?")
        loc = f"  [{f['file']}]" if f.get("file") else ""
        print(f"  {prefix} {f['name']}{loc}: {f['detail']}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
