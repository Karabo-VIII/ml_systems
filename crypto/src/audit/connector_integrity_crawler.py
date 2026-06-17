"""Connector-Integrity Crawler — meta-crawler for canonical-class drift.

Catches the bug class that CDAP's existing checkers miss:
  A1. data_source_canonical    — trainer/strategy reads wrong-version data
  A2. cost_model_canonical     — simulator uses no/wrong cost model
  A3. constraint_propagation   — LO/lev only enforced at blend layer not sim
  A4. yaml_vs_v3_drift         — yaml aggregate_metrics_* inflated vs v3
  A5. wm_trainer_coverage      — new trainer ships without CDAP invariant coverage

Provenance: 2026-05-17 — paper_trade_replay_v3 audit found 5 connector gaps
that escaped CDAP. User mandate: "crawlers too primitive to catch real errors".

Usage:
    python src/audit/connector_integrity_crawler.py             # full audit
    python src/audit/connector_integrity_crawler.py --json      # machine-readable
    python src/audit/connector_integrity_crawler.py --axis A1   # single axis

Exit codes:
    0  - clean (no critical findings)
    1  - WARN findings only (non-blocking)
    2  - CRITICAL drift (BLOCK COMMIT)

Wired into CDAP via _invariants.yaml::connector_integrity block (subprocess
pattern; same as perf_anti_pattern_crawler).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONNECTOR_INVARIANTS_PATH = PROJECT_ROOT / "config" / "_connector_invariants.yaml"


@dataclass
class Finding:
    axis: str
    severity: str
    name: str
    file: str = ""
    line: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "severity": self.severity,
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
        }

    def fmt(self) -> str:
        prefix = {"critical": "FAIL", "warn": "WARN", "info": "INFO"}.get(self.severity, "INFO")
        loc = f"  [{self.file}:{self.line}]" if self.file else ""
        return f"  {prefix:4s}  {self.axis}::{self.name}{loc}\n         {self.detail}"


def _expand_glob(pattern: str, exclude: Optional[set] = None) -> list[str]:
    exclude = exclude or set()
    p = (PROJECT_ROOT / pattern).as_posix()
    out = []
    for hit in glob.glob(p, recursive=True):
        rel = Path(hit).relative_to(PROJECT_ROOT).as_posix()
        if rel in exclude:
            continue
        out.append(rel)
    return sorted(set(out))


def _read(path: str) -> str:
    try:
        with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _find_line(text: str, pattern: re.Pattern) -> int:
    for ln_num, ln in enumerate(text.splitlines(), 1):
        if pattern.search(ln):
            return ln_num
    return 0


# ─────────────────────────────────────────────────────────────────────
# Axis A1: data_source_canonical
# ─────────────────────────────────────────────────────────────────────
def check_a1_data_source(spec: dict) -> list[Finding]:
    findings = []
    for rule in spec or []:
        name = rule["name"]
        sev = rule.get("severity", "warn")
        files = []
        for pat in rule.get("files", []):
            files.extend(_expand_glob(pat))
        exclude = set()
        for ex_pat in rule.get("exclude_files", []):
            for f in _expand_glob(ex_pat):
                exclude.add(f)
        files = [f for f in sorted(set(files)) if f not in exclude]

        forbidden_pat = rule.get("forbidden_pattern")
        if forbidden_pat:
            forbidden_re = re.compile(forbidden_pat)
            for f in files:
                text = _read(f)
                for ln_num, ln in enumerate(text.splitlines(), 1):
                    # Skip comment lines
                    if ln.lstrip().startswith("#"):
                        continue
                    if forbidden_re.search(ln):
                        findings.append(Finding(
                            axis="A1_DATA_SOURCE",
                            severity=sev,
                            name=name,
                            file=f,
                            line=ln_num,
                            detail=f"forbidden pattern matched: {forbidden_pat!r} at line: `{ln.strip()[:120]}`",
                        ))
    return findings


# ─────────────────────────────────────────────────────────────────────
# Axis A2: cost_model_canonical
# ─────────────────────────────────────────────────────────────────────
def check_a2_cost_model(spec: dict) -> list[Finding]:
    findings = []
    for rule in spec or []:
        name = rule["name"]
        sev = rule.get("severity", "warn")
        files = []
        for pat in rule.get("files", []):
            files.extend(_expand_glob(pat))
        files = sorted(set(files))

        required_any = rule.get("required_any_pattern")
        forbidden_hardcoded = rule.get("forbidden_hardcoded_cost_pattern")
        explicit_pattern = rule.get("pattern")

        if required_any:
            req_re = re.compile(required_any)
            for f in files:
                text = _read(f)
                if not req_re.search(text):
                    findings.append(Finding(
                        axis="A2_COST_MODEL",
                        severity=sev,
                        name=name,
                        file=f,
                        line=0,
                        detail=f"required cost-model import absent — must reference one of: {required_any}",
                    ))

        if forbidden_hardcoded:
            forb_re = re.compile(forbidden_hardcoded)
            for f in files:
                text = _read(f)
                for ln_num, ln in enumerate(text.splitlines(), 1):
                    if ln.lstrip().startswith("#"):
                        continue
                    if forb_re.search(ln):
                        findings.append(Finding(
                            axis="A2_COST_MODEL",
                            severity="critical",
                            name=f"{name}_hardcoded",
                            file=f,
                            line=ln_num,
                            detail=f"hardcoded cost constant: `{ln.strip()[:120]}`",
                        ))

        # Soft check: explicit pattern (for p_fill calibration range)
        if explicit_pattern:
            exp_re = re.compile(explicit_pattern)
            for f in files:
                text = _read(f)
                # Find any p_fill mention
                if "p_fill" in text and "MakerCostConfig" in text:
                    if not exp_re.search(text):
                        # Look for explicit out-of-range p_fill
                        bad_re = re.compile(r'p_fill\s*=\s*0\.(8\d|9\d|0\d)')
                        m = bad_re.search(text)
                        if m:
                            ln_num = _find_line(text, bad_re)
                            findings.append(Finding(
                                axis="A2_COST_MODEL",
                                severity=sev,
                                name=f"{name}_out_of_range",
                                file=f,
                                line=ln_num,
                                detail=f"p_fill outside calibrated [0.21, 0.50] range: `{m.group(0)}`",
                            ))
    return findings


# ─────────────────────────────────────────────────────────────────────
# Axis A3: constraint_propagation
# ─────────────────────────────────────────────────────────────────────
def check_a3_constraint_propagation(spec: dict) -> list[Finding]:
    findings = []
    for rule in spec or []:
        name = rule["name"]
        sev = rule.get("severity", "warn")
        files = []
        for pat in rule.get("files", []):
            files.extend(_expand_glob(pat))
        files = sorted(set(files))

        # Conditional invariant: if triggered, require pattern
        trigger_pat = rule.get("presence_triggered_by")
        required_pat = rule.get("required_pattern_if_present")
        if trigger_pat and required_pat:
            trig_re = re.compile(trigger_pat)
            req_re = re.compile(required_pat)
            for f in files:
                text = _read(f)
                if trig_re.search(text) and not req_re.search(text):
                    ln_num = _find_line(text, trig_re)
                    findings.append(Finding(
                        axis="A3_CONSTRAINT_PROP",
                        severity=sev,
                        name=name,
                        file=f,
                        line=ln_num,
                        detail=(
                            f"triggered by `{trigger_pat}` at line {ln_num}, but "
                            f"required guard absent: {required_pat}"
                        ),
                    ))

        # Direct required pattern
        required_only = rule.get("required_pattern")
        if required_only and not trigger_pat:
            req_re = re.compile(required_only)
            for f in files:
                text = _read(f)
                if not req_re.search(text):
                    findings.append(Finding(
                        axis="A3_CONSTRAINT_PROP",
                        severity=sev,
                        name=name,
                        file=f,
                        line=0,
                        detail=f"required guard absent: {required_only}",
                    ))
    return findings


# ─────────────────────────────────────────────────────────────────────
# Axis A4: yaml_vs_v3_drift
# ─────────────────────────────────────────────────────────────────────
def check_a4_yaml_v3_drift(spec: dict) -> list[Finding]:
    """Compare yaml aggregate_metrics_* claims to v3 paper_trade_replay reality."""
    findings = []
    if not spec:
        return findings
    try:
        import yaml as yamllib
    except ImportError:
        return findings

    threshold = spec.get("threshold_inflation_ratio", 2.0)
    flip_critical = spec.get("threshold_sign_flip", True)
    v3_dir = PROJECT_ROOT / spec.get("v3_log_dir", "logs/strat_audit")
    blends_yaml = PROJECT_ROOT / spec.get("blends_yaml", "config/production_blends.yaml")
    severity = spec.get("severity", "warn")

    if not blends_yaml.exists() or not v3_dir.exists():
        return findings

    try:
        with open(blends_yaml, "r", encoding="utf-8") as f:
            yml = yamllib.safe_load(f) or {}
    except Exception:
        return findings

    root_key = next((k for k in yml if k.startswith("production_blends")), None)
    if not root_key:
        return findings
    blends = yml[root_key] or {}

    # For each blend with v3 logs, compare yaml claim to v3 reality
    for blend_name, blend_def in blends.items():
        if not isinstance(blend_def, dict):
            continue

        # Extract yaml-claimed total_pnl_pct (look for aggregate_metrics_* or claims)
        yaml_claim = None
        for k, v in blend_def.items():
            if k.startswith("aggregate_metrics") and isinstance(v, dict):
                for nested_k in ("total_pnl_pct", "compound_return_pct", "return_pct"):
                    if nested_k in v:
                        try:
                            yaml_claim = float(v[nested_k])
                            break
                        except (TypeError, ValueError):
                            continue
                if yaml_claim is not None:
                    break

        if yaml_claim is None:
            continue   # no claim to verify

        # Find v3 logs for this blend (4-month aggregate)
        v3_total = 0.0
        v3_factor = 1.0
        v3_found = 0
        for month_window in ("20260101_20260131", "20260201_20260228",
                              "20260301_20260331", "20260401_20260430"):
            for f in v3_dir.glob(f"paper_trade_replay_v3_{blend_name}_*_{month_window}.json"):
                try:
                    with open(f, "r") as fh:
                        data = json.load(fh)
                    pnl = data.get("total_pnl_pct")
                    if pnl is not None:
                        v3_total += float(pnl)
                        v3_factor *= (1.0 + float(pnl) / 100.0)
                        v3_found += 1
                        break   # one log per month is enough
                except Exception:
                    continue

        if v3_found == 0:
            continue   # no v3 evidence to compare

        v3_comp = (v3_factor - 1.0) * 100.0

        # Compute inflation ratio
        if abs(v3_comp) < 0.01:
            ratio_str = "v3≈0"
            severe = "warn"
            flagged = abs(yaml_claim) > 1.0
        elif (yaml_claim > 0) != (v3_comp > 0):
            ratio_str = "SIGN-FLIPPED"
            severe = "critical" if flip_critical else "warn"
            flagged = True
        else:
            ratio = yaml_claim / v3_comp if v3_comp != 0 else float("inf")
            ratio_str = f"{abs(ratio):.2f}x"
            severe = "critical" if abs(ratio) > threshold * 2 else severity
            flagged = abs(ratio) > threshold

        if flagged:
            findings.append(Finding(
                axis="A4_YAML_VS_V3",
                severity=severe,
                name=f"yaml_inflation:{blend_name}",
                file="config/production_blends.yaml",
                line=0,
                detail=(
                    f"yaml claim={yaml_claim:+.2f}% vs v3 reality (4mo comp)={v3_comp:+.2f}%  "
                    f"inflation={ratio_str}  (threshold={threshold}x); v3_logs={v3_found}/4"
                ),
            ))

    return findings


# ─────────────────────────────────────────────────────────────────────
# Axis A5: wm_trainer_coverage
# ─────────────────────────────────────────────────────────────────────
def check_a5_wm_trainer_coverage(spec: dict) -> list[Finding]:
    """Detect WM trainers that aren't in the CDAP invariant lists."""
    findings = []
    if not spec:
        return findings
    try:
        import yaml as yamllib
    except ImportError:
        return findings

    sev = spec.get("severity", "warn")
    active_glob = spec.get("active_trainers_glob", "src/wm/v*/v*_training/train_world_model.py")
    excluded_glob = spec.get("excluded_archived_glob", "src/wm/v*/archive/**/train_world_model.py")
    must_be_in = spec.get("must_be_in_invariants", [])

    active_trainers = set(_expand_glob(active_glob))
    archived = set(_expand_glob(excluded_glob))
    active_trainers -= archived

    # Read main _invariants.yaml
    inv_path = PROJECT_ROOT / "config" / "_invariants.yaml"
    if not inv_path.exists():
        return findings
    try:
        with open(inv_path, "r", encoding="utf-8") as f:
            inv = yamllib.safe_load(f) or {}
    except Exception:
        return findings

    # For each invariant name, find which trainers it covers
    def get_covered(invariant_name: str) -> set:
        """Walk all top-level keys, find rules with matching name, collect their `files` lists."""
        covered = set()
        for top_key, top_val in inv.items():
            if isinstance(top_val, list):
                for rule in top_val:
                    if isinstance(rule, dict) and rule.get("name") == invariant_name:
                        for pat in rule.get("files", []):
                            for f in _expand_glob(pat):
                                covered.add(f)
            elif isinstance(top_val, dict):
                for pat in top_val.get("files", []) or []:
                    for f in _expand_glob(pat):
                        covered.add(f)
        return covered

    for invariant_name in must_be_in:
        covered = get_covered(invariant_name)
        missing = active_trainers - covered
        if missing:
            for trainer in sorted(missing):
                findings.append(Finding(
                    axis="A5_WM_COVERAGE",
                    severity=sev,
                    name=f"missing_from_invariant:{invariant_name}",
                    file=trainer,
                    line=0,
                    detail=(
                        f"trainer not covered by CDAP invariant `{invariant_name}` — "
                        f"add it to the rule's `files` list"
                    ),
                ))
    return findings


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def run_crawler(axis_filter: Optional[str] = None) -> tuple[list[Finding], int]:
    try:
        import yaml as yamllib
    except ImportError:
        print("[connector_integrity] PyYAML not available", file=sys.stderr)
        return [], 2

    if not CONNECTOR_INVARIANTS_PATH.exists():
        print(f"[connector_integrity] missing {CONNECTOR_INVARIANTS_PATH}", file=sys.stderr)
        return [], 2

    with open(CONNECTOR_INVARIANTS_PATH, "r", encoding="utf-8") as f:
        spec = yamllib.safe_load(f) or {}

    findings = []
    if not axis_filter or axis_filter == "A1":
        findings.extend(check_a1_data_source(spec.get("data_source_canonical", [])))
    if not axis_filter or axis_filter == "A2":
        findings.extend(check_a2_cost_model(spec.get("cost_model_canonical", [])))
    if not axis_filter or axis_filter == "A3":
        findings.extend(check_a3_constraint_propagation(spec.get("constraint_propagation", [])))
    if not axis_filter or axis_filter == "A4":
        findings.extend(check_a4_yaml_v3_drift(spec.get("yaml_vs_v3_drift", {})))
    if not axis_filter or axis_filter == "A5":
        findings.extend(check_a5_wm_trainer_coverage(spec.get("wm_trainer_coverage", {})))

    n_critical = sum(1 for f in findings if f.severity == "critical")
    n_warn = sum(1 for f in findings if f.severity == "warn")
    exit_code = 2 if n_critical else (1 if n_warn else 0)
    return findings, exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=["A1", "A2", "A3", "A4", "A5"],
                    help="filter to a single axis")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    args = ap.parse_args()

    findings, exit_code = run_crawler(args.axis)

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
        return exit_code

    print("=" * 72)
    print("CONNECTOR-INTEGRITY CRAWLER")
    print(f"  registry: {CONNECTOR_INVARIANTS_PATH.relative_to(PROJECT_ROOT)}")
    if args.axis:
        print(f"  axis filter: {args.axis}")
    print("=" * 72)

    by_axis = {}
    for f in findings:
        by_axis.setdefault(f.axis, []).append(f)

    for axis in sorted(by_axis):
        axis_findings = by_axis[axis]
        print(f"\n[{axis}]  ({len(axis_findings)} findings)")
        for f in axis_findings:
            if args.quiet and f.severity not in ("critical", "warn"):
                continue
            print(f.fmt())

    n_crit = sum(1 for f in findings if f.severity == "critical")
    n_warn = sum(1 for f in findings if f.severity == "warn")
    n_info = sum(1 for f in findings if f.severity == "info")
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_crit} CRITICAL · {n_warn} WARN · {n_info} INFO")
    print(f"EXIT CODE: {exit_code}  ({'BLOCK COMMIT' if exit_code == 2 else 'WARN ONLY' if exit_code == 1 else 'CLEAN'})")
    print("=" * 72)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
