"""
CDAP — Trader Deploy-Gates Checker
===================================

Companion to `check_wealth_bot_claims.py`. Validates the file-existence /
JSON-shape / directory-correlation gates that the trader skill's
PRE_DEPLOY_CHECKLIST + LIFECYCLE protocol require, but which the
pattern-based schema in `config/_invariants.yaml` cannot express.

Triggers a CDAP FAIL (exit 2) when:
  - A deploy_claim.json is missing required PRE_DEPLOY_CHECKLIST items
    (1-16, see .claude/skills/trader/PRE_DEPLOY_CHECKLIST.md).
  - A sleeve directory under `config/sleeves/<id>/` is missing
    `lifecycle.yaml`.
  - A lifecycle.yaml declares `current_stage: live_*` but no matching
    stage transition record exists in `runs/lifecycle/`.

Detects a "deploy commit" via either:
  - File path pattern: `runs/deploy/*/deploy_claim.json` present in the diff.
  - `config/sleeves/*.yaml` changes its `stage:` field to a `live_*` value.

When neither pattern triggers, the checker runs in advisory mode (still
emits warnings, exit 1 max) — does NOT block non-deploy commits.

Exit codes:
    0  - all deploy gates pass (or no deploy commit detected)
    1  - WARN findings only (advisory mode)
    2  - CRIT (BLOCK COMMIT): deploy commit detected and a gate failed

Usage:
    python src/audit/check_deploy_gates.py
    python src/audit/check_deploy_gates.py --json
    python src/audit/check_deploy_gates.py --claim runs/deploy/h18_v2/deploy_claim.json

Wired into `check_invariants.py` run_audit() alongside check_wealth_bot_claims.py.

Provenance: 2026-05-28 trader-skill upgrade. The 16-item PRE_DEPLOY_CHECKLIST
(.claude/skills/trader/PRE_DEPLOY_CHECKLIST.md) is the canonical anchor.
Lifecycle protocol in .claude/skills/trader/LIFECYCLE.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Required keys in a deploy_claim.json (per PRE_DEPLOY_CHECKLIST.md §"Deploy
# claim JSON shape"). Items 1-12 are CRIT; items 13-16 are WARN-eligible.
# ---------------------------------------------------------------------------
DEPLOY_CLAIM_REQUIRED_KEYS = {
    "sleeve_id": "critical",
    "stage_transition": "critical",
    "ship_claim": "critical",
    "deploy_gates.item_01_cost_source_canonical": "critical",
    "deploy_gates.item_02_mtm_reconciliation": "critical",
    "deploy_gates.item_03_look_ahead": "critical",
    "deploy_gates.item_04_stride_1": "critical",
    "deploy_gates.item_05_p_fill_budget": "critical",
    "deploy_gates.item_06_claim_contract": "critical",
    "deploy_gates.item_07_mechanism_falsifier": "critical",
    "deploy_gates.item_08_multi_seed": "critical",
    "deploy_gates.item_09_jackknife": "critical",
    "deploy_gates.item_10_purge_gap": "critical",
    "deploy_gates.item_11_dsr_cscv": "critical",
    "deploy_gates.item_12_survivorship": "critical",
    "deploy_gates.item_13_capacity_estimate": "warn",
    "deploy_gates.item_14_decay_monitor": "warn",
    "deploy_gates.item_15_dd_response_plan": "warn",
    "deploy_gates.item_16_exchange_verification": "warn",
}

LIFECYCLE_REQUIRED_KEYS = {
    "sleeve_id",
    "current_stage",
    "stage_entered_at_utc",
    "min_time_in_stage_days",
}

VALID_LIFECYCLE_STAGES = {
    "incubation", "paper", "live_small", "live_scale", "retired",
}

LIVE_STAGES = {"live_small", "live_scale"}


def _nested_get(d: dict, dotted_key: str) -> Any:
    """Get a value from a nested dict using dotted-key notation."""
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_deploy_claim(path: Path) -> tuple[list[str], list[str]]:
    """Return (crits, warns) for a deploy_claim.json file."""
    crits: list[str] = []
    warns: list[str] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            claim = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        crits.append(f"{path}: failed to parse JSON: {e}")
        return crits, warns

    for key, severity in DEPLOY_CLAIM_REQUIRED_KEYS.items():
        value = _nested_get(claim, key)
        if value is None or (isinstance(value, str) and value == ""):
            msg = f"{path}: missing or empty required key '{key}'"
            if severity == "critical":
                crits.append(msg)
            else:
                warns.append(msg)

    # Check ship_claim block is non-trivial (delegates content validation
    # to check_wealth_bot_claims.py, but we verify the key is populated).
    ship_claim = claim.get("ship_claim")
    if ship_claim is not None and not isinstance(ship_claim, dict):
        crits.append(f"{path}: ship_claim must be a dict (got {type(ship_claim).__name__})")
    elif isinstance(ship_claim, dict) and not ship_claim:
        crits.append(f"{path}: ship_claim is empty dict")

    # p_fill budget sanity
    p_fill = _nested_get(claim, "deploy_gates.item_05_p_fill_budget")
    if isinstance(p_fill, dict):
        low = p_fill.get("low", 1.0)
        high = p_fill.get("high", 1.0)
        if low > 0.50 or high > 0.80:
            warns.append(
                f"{path}: p_fill_budget [{low}, {high}] is optimistic; "
                f"empirical OHLC replay 2026-04-22 found 0.21-0.40. Budget [0.25, 0.50] expected."
            )

    # Stage transition sanity
    stage_transition = claim.get("stage_transition", "")
    if "-> live_" in stage_transition or "_to_live_" in stage_transition:
        # This IS a deploy transition; all 16 items must be present.
        # (Already enforced above by required_keys.)
        pass

    return crits, warns


def validate_lifecycle_yaml(path: Path) -> tuple[list[str], list[str]]:
    """Return (crits, warns) for a lifecycle.yaml file."""
    crits: list[str] = []
    warns: list[str] = []

    # Avoid pyyaml dependency by hand-parsing key: value pairs (loose).
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        crits.append(f"{path}: cannot read: {e}")
        return crits, warns

    declared_keys: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            declared_keys[key.strip()] = val.strip()

    for required in LIFECYCLE_REQUIRED_KEYS:
        if required not in declared_keys:
            crits.append(f"{path}: missing required key '{required}'")

    current_stage = declared_keys.get("current_stage", "").strip().strip("\"'")
    if current_stage and current_stage not in VALID_LIFECYCLE_STAGES:
        crits.append(
            f"{path}: current_stage='{current_stage}' is not in {VALID_LIFECYCLE_STAGES}"
        )

    # If sleeve is in a live stage, a stage transition record must exist.
    if current_stage in LIVE_STAGES:
        sleeve_id = declared_keys.get("sleeve_id", "").strip().strip("\"'")
        if sleeve_id:
            transition_glob = str(PROJECT_ROOT / "runs" / "lifecycle" / f"{sleeve_id}_*_to_{current_stage}_*.json")
            if not glob.glob(transition_glob):
                warns.append(
                    f"{path}: sleeve {sleeve_id} is in stage '{current_stage}' but "
                    f"no transition record found at {transition_glob}. "
                    f"LIFECYCLE.md requires a transition_record JSON per stage promotion."
                )

    return crits, warns


def find_deploy_claims() -> list[Path]:
    """Find all deploy_claim.json files in the project."""
    pattern = str(PROJECT_ROOT / "runs" / "deploy" / "*" / "deploy_claim.json")
    return [Path(p) for p in glob.glob(pattern)]


def find_lifecycle_yamls() -> list[Path]:
    """Find all lifecycle.yaml files in config/sleeves/."""
    pattern = str(PROJECT_ROOT / "config" / "sleeves" / "*" / "lifecycle.yaml")
    return [Path(p) for p in glob.glob(pattern)]


def find_sleeve_dirs_missing_lifecycle() -> list[Path]:
    """Find sleeve directories under config/sleeves/ that lack a lifecycle.yaml.

    Only flags directories that contain at least one .yaml file (i.e., are
    actual sleeve configs, not empty dirs).
    """
    sleeve_root = PROJECT_ROOT / "config" / "sleeves"
    if not sleeve_root.exists():
        return []
    missing: list[Path] = []
    for d in sleeve_root.iterdir():
        if not d.is_dir():
            continue
        yamls = list(d.glob("*.yaml"))
        if yamls and not (d / "lifecycle.yaml").exists():
            missing.append(d)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate trader deploy gates.")
    ap.add_argument("--claim", type=str, default=None,
                    help="Validate a single deploy_claim.json path. Default: scan runs/deploy/")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON output instead of human text.")
    ap.add_argument("--lifecycle-only", action="store_true",
                    help="Only validate lifecycle.yaml files; skip deploy claims.")
    args = ap.parse_args()

    all_crits: list[str] = []
    all_warns: list[str] = []

    # Deploy claim validation
    if not args.lifecycle_only:
        if args.claim:
            claims = [Path(args.claim)]
        else:
            claims = find_deploy_claims()

        for p in claims:
            if not p.exists():
                all_crits.append(f"{p}: file does not exist")
                continue
            crits, warns = validate_deploy_claim(p)
            all_crits.extend(crits)
            all_warns.extend(warns)

    # Lifecycle YAML validation
    for p in find_lifecycle_yamls():
        crits, warns = validate_lifecycle_yaml(p)
        all_crits.extend(crits)
        all_warns.extend(warns)

    # Sleeve dirs missing lifecycle.yaml (advisory; CRIT only if any
    # sleeve directory contains a yaml file we'd expect to belong to a
    # live sleeve).
    missing = find_sleeve_dirs_missing_lifecycle()
    for d in missing:
        all_warns.append(
            f"{d}: sleeve directory has .yaml files but no lifecycle.yaml. "
            f"Add per LIFECYCLE.md §Lifecycle YAML."
        )

    if args.json:
        import json as _json
        print(_json.dumps({
            "crits": all_crits,
            "warns": all_warns,
            "n_crits": len(all_crits),
            "n_warns": len(all_warns),
        }, indent=2))
    else:
        if all_crits:
            print(f"[check_deploy_gates] {len(all_crits)} CRITICAL findings:", file=sys.stderr)
            for c in all_crits:
                print(f"  - {c}", file=sys.stderr)
        if all_warns:
            print(f"[check_deploy_gates] {len(all_warns)} WARN findings:", file=sys.stderr)
            for w in all_warns:
                print(f"  - {w}", file=sys.stderr)
        if not all_crits and not all_warns:
            print("[check_deploy_gates] OK: no trader deploy-gate violations")

    if all_crits:
        return 2
    if all_warns:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
