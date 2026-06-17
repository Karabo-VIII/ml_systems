"""
CDAP Invariant Checker
======================

Loads `config/_invariants.yaml` and validates the current tree against
every declared invariant. Mandatory pre-commit gate per
docs/DOUBLE_AUDIT_PROTOCOL.md.

Exit codes:
    0  - clean (all invariants hold)
    1  - WARN findings only (non-blocking; surfaced in stderr)
    2  - CRITICAL drift (BLOCK COMMIT)

Usage:
    python src/audit/check_invariants.py                # full audit
    python src/audit/check_invariants.py --quiet        # only failures
    python src/audit/check_invariants.py --json         # machine-readable

Categories audited (each with own checker):
    cross_version_constants  - identical literal across multiple files
    walk_forward             - purge-gap presence
    cost_model               - p_fill / SPOT_COST consistency
    simulator                - MtM-bug regression guard
    leakage                  - look-ahead / inline-gitignore guards
    dag                      - run_pipeline stage-order rules
    cli_universe_support     - multi-asset scripts accept --universe
    cli_coverage_report      - multi-asset scripts emit coverage
    atomic_write             - silver/gold producers use atomic-tmp-rename
    windows_safety           - no emoji in py prints (cp1252)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INVARIANTS_PATH = PROJECT_ROOT / "config" / "_invariants.yaml"


# ──────────────────────────────────────────────────────────────────────
# Findings model
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    severity: str       # "critical" | "warn" | "info"
    category: str
    name: str
    file: Optional[str] = None
    detail: str = ""

    def fmt(self) -> str:
        prefix = {"critical": "FAIL", "warn": "WARN", "info": "INFO"}[self.severity]
        loc = f"  [{self.file}]" if self.file else ""
        return f"  {prefix:4s}  {self.category}::{self.name}{loc}\n         {self.detail}"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _expand_glob(pattern: str, exclude: Optional[set] = None) -> list:
    """Resolve a glob pattern to project-relative paths."""
    exclude = exclude or set()
    p = (PROJECT_ROOT / pattern).as_posix()
    matches = []
    for hit in glob.glob(p, recursive=True):
        rel = Path(hit).relative_to(PROJECT_ROOT).as_posix()
        if rel in exclude:
            continue
        matches.append(rel)
    return sorted(set(matches))


def _read(path: str) -> str:
    try:
        with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────
# Category checkers
# ──────────────────────────────────────────────────────────────────────
_CONST_RE_TEMPLATE = r"^\s*{name}\s*[:=]\s*([^\s#]+)"


def check_cross_version_constants(rules: list, archived_globs: Optional[list] = None) -> list:
    """For each rule: every matching file must define the same literal value.

    Excludes files matching `archived_globs` (top-level YAML key) AND
    rule-level `exclude_files`. Both serve the same purpose — opt out
    intentionally-archived versions from cross-version drift checks.
    """
    findings = []
    archived_globs = archived_globs or []
    archived_files: set = set()
    for pat in archived_globs:
        for f in _expand_glob(pat):
            archived_files.add(f)

    for rule in rules:
        name = rule["name"]
        expected = rule["value"]
        files: list = []
        for pat in rule.get("files", []):
            files.extend(_expand_glob(pat))
        # Per-rule exclusions (legacy support)
        excluded: set = set(archived_files)
        for ex_pat in rule.get("exclude_files", []):
            for f in _expand_glob(ex_pat):
                excluded.add(f)
        files = [f for f in sorted(set(files)) if f not in excluded]

        if not files:
            findings.append(Finding(
                severity="warn", category="cross_version_constants", name=name,
                detail=f"no files match patterns: {rule.get('files', [])}",
            ))
            continue

        regex = re.compile(_CONST_RE_TEMPLATE.format(name=re.escape(name)), re.M)
        per_file_value = {}
        for f in files:
            text = _read(f)
            m = regex.search(text)
            if not m:
                continue            # constant not defined in this file (OK)
            raw_val = m.group(1).strip().rstrip(",")
            # Try eval-as-literal for ints/floats/booleans/lists
            try:
                val = eval(raw_val, {"__builtins__": {}}, {})
            except Exception:
                val = raw_val
            per_file_value[f] = val

        if not per_file_value:
            continue                # constant not present anywhere; nothing to drift-check

        bad = {f: v for f, v in per_file_value.items() if v != expected}
        if bad:
            for f, v in bad.items():
                findings.append(Finding(
                    severity=rule.get("severity", "critical"),
                    category="cross_version_constants", name=name, file=f,
                    detail=f"value={v!r} expected={expected!r}",
                ))
    return findings


def check_walk_forward(rules: list) -> list:
    findings = []
    for rule in rules:
        name = rule["name"]
        pattern = re.compile(rule["pattern"], re.M)
        must_match = rule.get("must_match", True)
        matched_any = False
        for pat in rule.get("files", []):
            for f in _expand_glob(pat):
                matched_any = True
                text = _read(f)
                hit = bool(pattern.search(text))
                if hit != must_match:
                    findings.append(Finding(
                        severity=rule.get("severity", "warn"),
                        category="walk_forward", name=name, file=f,
                        detail=("missing required pattern" if must_match
                                else "forbidden pattern present"),
                    ))
        if must_match and not matched_any:
            # must_match rule over a glob that matches no files = silent no-op.
            findings.append(Finding(
                severity="warn",
                category="walk_forward", name=name, file="<none>",
                detail=(f"invariant target file(s) missing -- rule NOT enforced: "
                        f"{rule.get('files', [])}"),
            ))
    return findings


def check_pattern_must_not_contain(rules: list, category: str) -> list:
    """Generic: file MUST NOT contain pattern. LINE-BY-LINE — skips comment lines.

    Compiles each rule's pattern WITHOUT re.M; tests each non-comment line
    independently. This avoids cross-line matches where `[^#]*` would
    otherwise span newlines.
    """
    findings = []
    for rule in rules:
        name = rule["name"]
        pat_str = rule.get("pattern_must_not_contain")
        if pat_str is None:
            continue
        pattern = re.compile(pat_str)
        for fp in rule.get("files", []):
            for f in _expand_glob(fp):
                text = _read(f)
                hit = False
                for ln in text.splitlines():
                    stripped = ln.lstrip()
                    if stripped.startswith("#"):
                        continue        # comment line — historical-bug docs OK
                    if pattern.search(ln):
                        hit = True
                        break
                if hit:
                    findings.append(Finding(
                        severity=rule.get("severity", "critical"),
                        category=category, name=name, file=f,
                        detail=f"forbidden pattern matched: {pat_str}",
                    ))
    return findings


def check_pattern_must_contain(rules: list, category: str) -> list:
    """Generic: file MUST contain pattern."""
    findings = []
    for rule in rules:
        name = rule["name"]
        pat_str = rule.get("pattern")
        if pat_str is None:
            continue
        pattern = re.compile(pat_str, re.M)
        matched_any = False
        for fp in rule.get("files", []):
            for f in _expand_glob(fp):
                matched_any = True
                text = _read(f)
                if not pattern.search(text):
                    findings.append(Finding(
                        severity=rule.get("severity", "warn"),
                        category=category, name=name, file=f,
                        detail=f"required pattern absent: {pat_str}",
                    ))
        if not matched_any:
            # A must-contain rule whose file glob matches NOTHING silently
            # no-ops -> false safety assurance (the gate "passes" a check it
            # never ran). Surface it. Emitted at WARN (NOT the rule's own
            # severity) so a stale path can't escalate to exit-2 and block
            # every commit; repointing the path restores real enforcement.
            findings.append(Finding(
                severity="warn",
                category=category, name=name, file="<none>",
                detail=(f"invariant target file(s) missing -- rule NOT enforced "
                        f"(declared severity={rule.get('severity', 'warn')}): "
                        f"{rule.get('files', [])}"),
            ))
    return findings


def check_report_claims_staged() -> list:
    """Honest-reporting gate (#1 recurring user correction): flag UNTAGGED performance numbers in STAGED report *.md
    artifacts. Reuses src/audit/check_report_claims.scan. WARN-level -- advisory at commit, surfaces inflation without
    blocking (the user's most-repeated steer -- 'no lies, no inflated numbers')."""
    import os as _os
    import subprocess
    import sys as _sys
    findings: list = []
    try:
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from check_report_claims import scan as _scan
    except Exception as e:
        return [Finding("info", "report_claims", "loader", detail=f"check_report_claims not importable: {e}")]
    try:
        out = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout
        files = [f for f in out.splitlines()
                 if f.endswith(".md") and ("report" in f.lower() or f.startswith("runs/") or f.startswith("docs/"))]
    except Exception:
        files = []
    for f in files:
        if not _os.path.exists(f):
            continue
        try:
            for _w, ln, num, _line in _scan(open(f, encoding="utf-8", errors="replace").read(), f):
                findings.append(Finding("warn", "report_claims", "untagged_perf_number", file=f,
                    detail=(f"line {ln}: performance number [{num}] lacks a claim-tag (VERIFIED/REPORTED/INFERRED) or "
                            "reconciliation -- the honest-reporting gate (#1 recurring correction). Tag it or reconcile.")))
        except Exception:
            continue
    return findings


def check_framework_store() -> list:
    """Solutioning-pipeline store-accuracy gate: every recorded artifact path/ref resolves + manifests are well-formed
    (the anti-disparity guarantee for workspaces/). WARN-level -- surfaces a drifted store without blocking. Runs the
    fast `doctor` (NOT selftest, which would re-invoke CDAP -> recursion). Skips cleanly if the framework isn't present."""
    import os as _os
    import sys as _sys
    try:
        _sys.path.insert(0, _os.path.join(PROJECT_ROOT, "src"))
        from framework.pipeline import doctor as _doctor
    except Exception as e:
        return [Finding("info", "framework_store", "loader", detail=f"framework.pipeline not importable: {e}")]
    findings: list = []
    try:
        rep, n = _doctor()
        if n:
            for line in rep.splitlines():
                s = line.strip()
                if s.startswith(("MISSING", "DANGLING-REF", "MALFORMED")):
                    findings.append(Finding("warn", "framework_store", "store_problem", detail=s))
    except Exception as e:
        findings.append(Finding("info", "framework_store", "doctor_error", detail=str(e)))
    return findings


def check_dag(rules: list) -> list:
    """Run-pipeline DAG ordering rules (string-grep over run_pipeline.py)."""
    findings = []
    rp = _read("src/pipeline/run_pipeline.py")
    if not rp:
        findings.append(Finding(
            severity="critical", category="dag", name="run_pipeline_missing",
            detail="src/pipeline/run_pipeline.py not readable",
        ))
        return findings

    for rule in rules:
        name = rule["name"]
        rtxt = rule["rule"]

        if name == "chimera_legacy_runs_after_fetch_only":
            # Verify: depends_on=['fetch_binance'] in chimera_legacy stage
            # Look for the chimera_legacy block
            block = re.search(
                r'key="chimera_legacy".*?depends_on=\[([^\]]*)\]',
                rp, re.S,
            )
            if not block:
                findings.append(Finding(
                    severity=rule.get("severity", "critical"),
                    category="dag", name=name, file="src/pipeline/run_pipeline.py",
                    detail="chimera_legacy stage block not found",
                ))
                continue
            deps_raw = block.group(1)
            if "fetch_binance" not in deps_raw:
                findings.append(Finding(
                    severity=rule.get("severity", "critical"),
                    category="dag", name=name, file="src/pipeline/run_pipeline.py",
                    detail=f"chimera_legacy.depends_on doesn't include fetch_binance: {deps_raw!r}",
                ))
            # Forbidden deps: hawkes / panels / frontier
            for forbidden in ("hawkes_branching", "build_panels", "frontier_consolidate"):
                if forbidden in deps_raw:
                    findings.append(Finding(
                        severity="critical", category="dag", name=name,
                        file="src/pipeline/run_pipeline.py",
                        detail=f"chimera_legacy.depends_on includes forbidden {forbidden!r}",
                    ))

        elif name == "chimera_v51_depends_on_v50_and_frontier":
            block = re.search(
                r'key="chimera_v51".*?depends_on=\[([^\]]*)\]',
                rp, re.S,
            )
            if not block:
                continue
            deps = block.group(1)
            for required in ("chimera_legacy", "frontier_consolidate"):
                if required not in deps:
                    findings.append(Finding(
                        severity="critical", category="dag", name=name,
                        file="src/pipeline/run_pipeline.py",
                        detail=f"chimera_v51 missing required dep {required!r}",
                    ))

        elif name == "parse_tiers_chimera_legacy_position":
            # parse_tiers('all') return list — chimera_legacy should be index 1
            # Find the return list inside parse_tiers
            block = re.search(
                r'def parse_tiers.*?return\s*\[([^\]]*)\]',
                rp, re.S,
            )
            if not block:
                continue
            items = [s.strip().strip('"') for s in block.group(1).split(",") if s.strip()]
            try:
                idx = items.index("chimera_legacy")
                if idx != 1:
                    findings.append(Finding(
                        severity=rule.get("severity", "warn"),
                        category="dag", name=name, file="src/pipeline/run_pipeline.py",
                        detail=f"chimera_legacy at index {idx}, expected 1 (after fetch_binance)",
                    ))
            except ValueError:
                findings.append(Finding(
                    severity=rule.get("severity", "warn"),
                    category="dag", name=name, file="src/pipeline/run_pipeline.py",
                    detail="chimera_legacy not in parse_tiers('all')",
                ))
    return findings


def check_cli_flag_required(spec: dict, category: str) -> list:
    """Verify each file in `files` accepts each flag in `required_flags`
    via its `--help` output. Skip files in `exclude_files` and `pending_files`.
    """
    findings = []
    flags = spec.get("required_flags", [])
    files = spec.get("files", [])
    exclude = set(spec.get("exclude_files", []))
    pending = set(spec.get("pending_files", []))
    severity = spec.get("severity", "warn")

    for f in files:
        if f in exclude:
            continue
        if f in pending:
            findings.append(Finding(
                severity="info", category=category, name="pending_patch", file=f,
                detail="patch deferred (per pipeline validation doc)",
            ))
            continue

        # Try `python <f> --help`
        try:
            out = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / f), "--help"],
                capture_output=True, text=True, timeout=20,
                cwd=str(PROJECT_ROOT),
            )
            help_text = (out.stdout or "") + (out.stderr or "")
        except Exception as e:
            findings.append(Finding(
                severity="warn", category=category, name="help_failed", file=f,
                detail=f"could not run --help: {type(e).__name__}: {e}",
            ))
            continue

        for flag in flags:
            # Match flag root only (e.g. "--universe" matches "--universe X")
            flag_root = flag.split()[0] if " " in flag else flag
            if flag_root not in help_text:
                findings.append(Finding(
                    severity=severity, category=category, name="missing_flag", file=f,
                    detail=f"--help does not advertise required flag: {flag_root}",
                ))

    return findings


def check_pattern_simple(spec: dict, category: str) -> list:
    """One-shot 'required_pattern' check across multiple files."""
    findings = []
    pattern = re.compile(spec.get("required_pattern", ""), re.M)
    files = spec.get("files", [])
    pending = set(spec.get("pending_files", []))
    severity = spec.get("severity", "warn")

    for f in files:
        if f in pending:
            findings.append(Finding(
                severity="info", category=category, name="pending_patch", file=f,
                detail="patch deferred",
            ))
            continue
        text = _read(f)
        if not pattern.search(text):
            findings.append(Finding(
                severity=severity, category=category, name="pattern_absent", file=f,
                detail=f"required pattern absent: {pattern.pattern}",
            ))
    return findings


def check_atomic_write(spec: dict) -> list:
    """All required_patterns must each appear at least once in each file."""
    findings = []
    patterns = [re.compile(p, re.M) for p in spec.get("required_patterns", [])]
    files = spec.get("files", [])
    severity = spec.get("severity", "warn")
    for f in files:
        text = _read(f)
        for pat in patterns:
            if not pat.search(text):
                findings.append(Finding(
                    severity=severity, category="atomic_write", name="missing_pattern",
                    file=f,
                    detail=f"required pattern absent: {pat.pattern}",
                ))
    return findings


def check_windows_safety(rules: list) -> list:
    """Detect emoji-range chars in print/log lines.

    Uses simple codepoint ranges instead of regex to avoid issues with
    YAML-parsed unicode escapes. Emoji range = U+1F300..U+1F9FF + dingbats.
    """
    findings = []
    for rule in rules:
        name = rule["name"]
        for fp in rule.get("files", []):
            for f in _expand_glob(fp):
                try:
                    text = _read(f)
                except Exception:
                    continue
                for ln_num, ln in enumerate(text.splitlines(), 1):
                    if not ("print" in ln or "log" in ln.lower()):
                        continue
                    if any(0x1F300 <= ord(c) <= 0x1FAFF or 0x2700 <= ord(c) <= 0x27BF
                           for c in ln):
                        findings.append(Finding(
                            severity=rule.get("severity", "warn"),
                            category="windows_safety", name=name, file=f,
                            detail=f"line {ln_num}: emoji in print/log statement",
                        ))
                        break
    return findings


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def check_lifecycle_consistency(spec: dict, category: str = "lifecycle") -> list:
    """B1.4 (R32): warn if production_blends references SUNSET/ARCHIVED pillars.

    Reads:
      - config/production_blends.yaml (blends + their sleeve names)
      - config/lifecycle_registry.yaml (pillar lifecycle states)
    Emits WARN for any blend whose name is SUNSET/ARCHIVED in registry but
    still appears in production_blends.yaml.

    Spec keys (in _invariants.yaml::lifecycle):
      blends_yaml: path-relative to repo root (default: config/production_blends.yaml)
      registry_yaml: path-relative (default: config/lifecycle_registry.yaml)
      enabled: bool (default: True)
    """
    findings: list = []
    if not spec or not spec.get("enabled", True):
        return findings
    try:
        import yaml
    except ImportError:
        return findings

    blends_path = PROJECT_ROOT / spec.get("blends_yaml", "config/production_blends.yaml")
    registry_path = PROJECT_ROOT / spec.get("registry_yaml", "config/lifecycle_registry.yaml")
    if not blends_path.exists() or not registry_path.exists():
        return findings

    try:
        with open(blends_path, "r", encoding="utf-8") as fh:
            blends_data = yaml.safe_load(fh) or {}
        with open(registry_path, "r", encoding="utf-8") as fh:
            reg_data = yaml.safe_load(fh) or {}
    except Exception:
        return findings

    pillars = reg_data.get("pillars", {}) or {}
    sunset_names = {n for n, info in pillars.items()
                       if info.get("state") in ("SUNSET", "ARCHIVED")}

    root_key = next((k for k in blends_data if k.startswith("production_blends")), None)
    if not root_key:
        return findings

    for blend_name in blends_data[root_key].keys():
        if blend_name in sunset_names:
            findings.append(Finding(
                severity="warn",
                category=category,
                name=f"sunset_blend_still_in_yaml:{blend_name}",
                file=str(blends_path.relative_to(PROJECT_ROOT)),
                detail=f"blend '{blend_name}' is {pillars[blend_name].get('state')} "
                          f"in lifecycle_registry but still present in production_blends.yaml",
            ))
    return findings


def check_perf_anti_patterns(spec: dict) -> list:
    """Run perf_anti_pattern_crawler.py as subprocess + map its findings to CDAP.

    Wires the AST-based hot-loop detector (born 2026-05-16 from the
    hawkes/lob_proxy/range_bars episode) into pre-commit. The crawler's
    own exit code policy is:
      rc=2 = CRITICAL findings present
      rc=1 = HIGH findings only
      rc=0 = clean / LOW only

    CDAP translates each crawler finding into a Finding, severity from
    the per-axis classification (critical_axes / warn_axes in spec).
    """
    if not spec.get("enabled", False):
        return []
    crawler = spec.get("crawler", "src/audit/perf_anti_pattern_crawler.py")
    scan_root = spec.get("scan_root", "src/pipeline")
    critical_axes = set(spec.get("critical_axes", []))
    warn_axes = set(spec.get("warn_axes", []))
    timeout = int(spec.get("timeout_seconds", 30))

    findings: list = []
    crawler_path = PROJECT_ROOT / crawler
    if not crawler_path.exists():
        findings.append(Finding(
            severity="warn", category="perf_anti_patterns",
            name="crawler_missing",
            detail=f"crawler script not found at {crawler}; skipping",
        ))
        return findings

    try:
        out = subprocess.run(
            [sys.executable, str(crawler_path), "--root", scan_root, "--json"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        findings.append(Finding(
            severity="warn", category="perf_anti_patterns",
            name="crawler_timeout",
            detail=f"crawler exceeded {timeout}s; skipping (re-run manually to inspect)",
        ))
        return findings
    except Exception as e:
        findings.append(Finding(
            severity="warn", category="perf_anti_patterns",
            name="crawler_error",
            detail=f"crawler invocation failed: {type(e).__name__}: {e}",
        ))
        return findings

    # Parse crawler JSON output. Empty list = clean.
    try:
        raw = json.loads(out.stdout or "[]")
    except json.JSONDecodeError as e:
        findings.append(Finding(
            severity="warn", category="perf_anti_patterns",
            name="crawler_output_unparseable",
            detail=f"crawler stdout not valid JSON: {e}; stderr={out.stderr[:200]}",
        ))
        return findings

    # Map each crawler finding to a CDAP Finding. Axis-based severity routing
    # is the authoritative split: CDAP critical iff axis in critical_axes.
    for f in raw:
        axis = f.get("axis", "")
        if axis in critical_axes:
            sev = "critical"
        elif axis in warn_axes:
            sev = "warn"
        else:
            sev = "info"
        file_str = f.get("file", "")
        line = f.get("line", 0)
        fn = f.get("function_name", "")
        detail = (
            f"{axis} at line {line}"
            + (f" in `{fn}()`" if fn else "")
            + f" -- {f.get('suggestion', '')[:200]}"
        )
        findings.append(Finding(
            severity=sev, category="perf_anti_patterns",
            name=axis.lower(), file=file_str, detail=detail,
        ))
    return findings


def check_connector_integrity(spec: dict) -> list:
    """Run connector_integrity_crawler.py as subprocess + map its findings to CDAP.

    Catches the bug class that existing CDAP checkers miss:
      A1 data_source_canonical   (e.g., anti_fragile v50 default)
      A2 cost_model_canonical    (simulator missing cost model)
      A3 constraint_propagation  (LO/lev only at blend, not sim)
      A4 yaml_vs_v3_drift        (yaml inflated vs v3 reality)
      A5 wm_trainer_coverage     (trainer not in CDAP invariant lists)

    Crawler exit code policy:
      rc=2 = CRITICAL findings present
      rc=1 = WARN only
      rc=0 = clean
    """
    if not spec.get("enabled", False):
        return []
    crawler = spec.get("crawler", "src/audit/connector_integrity_crawler.py")
    critical_axes = set(spec.get("critical_axes", []))
    warn_axes = set(spec.get("warn_axes", []))
    timeout = int(spec.get("timeout_seconds", 60))

    findings: list = []
    crawler_path = PROJECT_ROOT / crawler
    if not crawler_path.exists():
        findings.append(Finding(
            severity="warn", category="connector_integrity",
            name="crawler_missing",
            detail=f"crawler script not found at {crawler}; skipping",
        ))
        return findings

    try:
        out = subprocess.run(
            [sys.executable, str(crawler_path), "--json"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        findings.append(Finding(
            severity="warn", category="connector_integrity",
            name="crawler_timeout",
            detail=f"crawler exceeded {timeout}s; skipping",
        ))
        return findings
    except Exception as e:
        findings.append(Finding(
            severity="warn", category="connector_integrity",
            name="crawler_error",
            detail=f"crawler invocation failed: {type(e).__name__}: {e}",
        ))
        return findings

    try:
        raw = json.loads(out.stdout or "[]")
    except json.JSONDecodeError as e:
        findings.append(Finding(
            severity="warn", category="connector_integrity",
            name="crawler_output_unparseable",
            detail=f"crawler stdout not valid JSON: {e}",
        ))
        return findings

    # Map each crawler finding to a CDAP Finding.
    for f in raw:
        axis = f.get("axis", "")
        # Crawler axis severity overrides invariants.yaml mapping if specified
        crawler_sev = f.get("severity", "warn")
        if axis in critical_axes:
            sev = "critical"
        elif axis in warn_axes:
            sev = "warn"
        else:
            sev = crawler_sev
        file_str = f.get("file", "")
        line = f.get("line", 0)
        detail = f"{axis}::{f.get('name', '')}"
        if file_str:
            detail += f"  at line {line}" if line else ""
        detail += f"  -- {f.get('detail', '')[:200]}"
        findings.append(Finding(
            severity=sev, category="connector_integrity",
            name=axis.lower(), file=file_str, detail=detail,
        ))
    return findings


def _strip_comments(text: str) -> list:
    """Return (line_no, code_part) for each non-blank line, with the trailing
    '#'-comment removed (naive: ignores '#' inside strings, acceptable for the
    agent-taxonomy pattern checks which operate on assignment/call syntax)."""
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            continue
        # drop trailing comment (naive)
        code = ln.split("#", 1)[0] if "#" in ln else ln
        if code.strip():
            out.append((i, code))
    return out


def check_agent_taxonomy(spec: dict) -> list:
    """The 4 agent-layer invariants (doc SS1.8). Each is TWO-SIDED and carries an
    empty-glob guard (a glob matching nothing surfaces a WARN, never silent-pass).

    (a) forecaster_frozen_in_agents        -- no F.train()/F-params-in-optimizer under src/agents/**
    (b) no_predicted_return_as_realized_reward -- no critic/reward target tracing to decoded return_logits
    (c) agent_class_declared               -- every agent-logic module declares __class_tag__ in {A1,A2,A1H}
    (d) v16_v17_not_in_wm                  -- src/wm/v16 + v17 contain only MOVED.md (no .py)
    """
    findings: list = []
    if not spec:
        return findings

    # ---- (a) forecaster_frozen_in_agents ----
    rule = spec.get("forecaster_frozen_in_agents")
    if rule:
        sev = rule.get("severity", "critical")
        pats = [re.compile(p) for p in rule.get("forbidden_patterns", [])]
        files: list = []
        for g in rule.get("agent_globs", []):
            files.extend(_expand_glob(g))
        files = sorted(set(files))
        if not files:
            findings.append(Finding("warn", "agent_taxonomy", "forecaster_frozen_in_agents",
                file="<none>", detail=f"no files match agent_globs={rule.get('agent_globs')} -- rule NOT enforced"))
        else:
            for f in files:
                for ln_no, code in _strip_comments(_read(f)):
                    for pat in pats:
                        if pat.search(code):
                            findings.append(Finding(sev, "agent_taxonomy", "forecaster_frozen_in_agents",
                                file=f, detail=f"line {ln_no}: forecaster un-frozen in an agent: {code.strip()[:140]}"))
                            break

    # ---- (b) no_predicted_return_as_realized_reward ----
    rule = spec.get("no_predicted_return_as_realized_reward")
    if rule:
        sev = rule.get("severity", "critical")
        pats = [re.compile(p) for p in rule.get("forbidden_patterns", [])]
        exempt = tuple(rule.get("dream_exempt_prefixes", []))
        files = []
        for g in rule.get("agent_globs", []):
            files.extend(_expand_glob(g))
        files = sorted(set(files))
        if not files:
            findings.append(Finding("warn", "agent_taxonomy", "no_predicted_return_as_realized_reward",
                file="<none>", detail=f"no files match agent_globs={rule.get('agent_globs')} -- rule NOT enforced"))
        else:
            for f in files:
                base = Path(f).name
                if exempt and base.startswith(exempt):
                    continue  # explicit dream_* module -- predicted-return reward allowed (gated elsewhere)
                for ln_no, code in _strip_comments(_read(f)):
                    for pat in pats:
                        if pat.search(code):
                            findings.append(Finding(sev, "agent_taxonomy", "no_predicted_return_as_realized_reward",
                                file=f, detail=f"line {ln_no}: reward/critic target traces to a decoded F prediction "
                                               f"(use a realized target_return_*): {code.strip()[:140]}"))
                            break

    # ---- (c) agent_class_declared ----
    rule = spec.get("agent_class_declared")
    if rule:
        sev = rule.get("severity", "critical")
        valid = set(rule.get("valid_classes", ["A1", "A2", "A1H"]))
        tag_re = re.compile(r'^\s*__class_tag__\s*=\s*[\'"]([^\'"]+)[\'"]', re.M)
        files = []
        for g in rule.get("agent_logic_globs", []):
            files.extend(_expand_glob(g))
        files = sorted(set(files))
        if not files:
            findings.append(Finding("warn", "agent_taxonomy", "agent_class_declared",
                file="<none>", detail=f"no files match agent_logic_globs={rule.get('agent_logic_globs')} -- rule NOT enforced"))
        else:
            for f in files:
                m = tag_re.search(_read(f))
                if not m:
                    findings.append(Finding(sev, "agent_taxonomy", "agent_class_declared",
                        file=f, detail="agent-logic module missing module-level __class_tag__"))
                elif m.group(1) not in valid:
                    findings.append(Finding(sev, "agent_taxonomy", "agent_class_declared",
                        file=f, detail=f"__class_tag__={m.group(1)!r} not in {sorted(valid)}"))

    # ---- (d) v16_v17_not_in_wm ----
    rule = spec.get("v16_v17_not_in_wm")
    if rule:
        sev = rule.get("severity", "critical")
        ext = rule.get("forbidden_extension", ".py")
        any_dir_present = False
        for d in rule.get("tombstone_dirs", []):
            dpath = PROJECT_ROOT / d
            if dpath.exists():
                any_dir_present = True
                offenders = _expand_glob(f"{d}/**/*{ext}")
                for f in offenders:
                    findings.append(Finding(sev, "agent_taxonomy", "v16_v17_not_in_wm",
                        file=f, detail=f"model code ({ext}) resurfaced in the forecaster zoo {d}/ "
                                       f"-- V16/V17 are A1 backbones under src/agents/, not forecasters"))
        if not rule.get("tombstone_dirs"):
            findings.append(Finding("warn", "agent_taxonomy", "v16_v17_not_in_wm",
                file="<none>", detail="no tombstone_dirs declared -- rule NOT enforced"))
        elif not any_dir_present:
            # The dirs being entirely ABSENT is fine (they could have been removed
            # outright); the invariant is satisfied. Emit INFO, not WARN, so it
            # doesn't bump exit code -- absence is a valid post-migration state.
            findings.append(Finding("info", "agent_taxonomy", "v16_v17_not_in_wm",
                file="<none>", detail=f"tombstone dirs absent ({rule.get('tombstone_dirs')}) -- no .py present (OK)"))

    return findings


def run_audit() -> tuple:
    """Run all category checkers; return (findings, exit_code)."""
    try:
        import yaml
    except ImportError:
        print("[CDAP] PyYAML not available; cannot load _invariants.yaml", file=sys.stderr)
        return [], 2

    if not INVARIANTS_PATH.exists():
        print(f"[CDAP] missing {INVARIANTS_PATH}", file=sys.stderr)
        return [], 2

    # Force UTF-8 — _invariants.yaml may contain non-ASCII (box-drawing,
    # arrows in rationale prose). Default Windows encoding is cp1252 which
    # crashes on U+2500. Match the read pattern used for source files in
    # _read() above.
    with open(INVARIANTS_PATH, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    findings: list = []

    findings.extend(check_cross_version_constants(
        spec.get("cross_version_constants", []),
        archived_globs=spec.get("archived_version_globs", []),
    ))
    findings.extend(check_walk_forward(spec.get("walk_forward", [])))
    findings.extend(check_walk_forward(spec.get("train_world_model", [])))
    findings.extend(check_walk_forward(spec.get("universe_yamls", [])))
    findings.extend(check_walk_forward(spec.get("browser_directive", [])))
    findings.extend(check_walk_forward(spec.get("run_pipeline_cmds", [])))
    findings.extend(check_pattern_must_not_contain(spec.get("simulator", []), "simulator"))
    findings.extend(check_pattern_must_not_contain(spec.get("layer_isolation", []), "layer_isolation"))
    findings.extend(check_pattern_must_not_contain(spec.get("leakage", []), "leakage"))
    findings.extend(check_pattern_must_contain(spec.get("required_patterns", []), "required_patterns"))
    # 2026-06-06 bitemporal/as-of (knowable-at) look-ahead defense. Advisory
    # (warn) must-contain rules over the apparatus files only; check_pattern_must_contain
    # WARNs (never escalates to critical) if a target file is missing, so the
    # rule can't false-positive across the repo nor block every commit on a path move.
    findings.extend(check_pattern_must_contain(spec.get("bitemporal_asof", []), "bitemporal_asof"))
    findings.extend(check_report_claims_staged())  # honest-reporting gate (#1 correction): untagged perf numbers in staged reports
    findings.extend(check_framework_store())  # solutioning-pipeline store-accuracy gate (workspaces/ paths+refs resolve)

    # 2026-05-28 trader-skill upgrade: 3 new sections wired into CDAP.
    # Each section has mixed rule types (some `pattern` + must_match, some
    # `pattern_must_not_contain`, some `must_exist`). Split by rule-kind
    # before dispatch so each checker sees only the rules it understands.
    for section in ("trader_deploy_gates", "trader_risk_sizing", "trader_lifecycle"):
        rules = spec.get(section, []) or []
        pattern_rules = [r for r in rules if "pattern" in r and "pattern_must_not_contain" not in r]
        not_contain_rules = [r for r in rules if "pattern_must_not_contain" in r]
        if pattern_rules:
            findings.extend(check_walk_forward(pattern_rules))
        if not_contain_rules:
            findings.extend(check_pattern_must_not_contain(not_contain_rules, section))
        # `must_exist`-style rules (e.g. trader_mtm_reconciliation_probe) are
        # validated separately by check_deploy_gates.py; here we only flag
        # them if the file is missing.
        for r in rules:
            if r.get("must_exist") and r.get("files"):
                for fp in r["files"]:
                    if not (PROJECT_ROOT / fp).exists():
                        findings.append(Finding(
                            severity=r.get("severity", "warn"),
                            category=section, name=r.get("name", "must_exist"),
                            file=fp,
                            detail=(r.get("rationale") or "must_exist file is missing").strip(),
                        ))
    findings.extend(check_dag(spec.get("dag", [])))
    findings.extend(check_cli_flag_required(
        spec.get("cli_universe_support", {}), "cli_universe_support"))
    findings.extend(check_cli_flag_required(
        spec.get("cli_force_support", {}), "cli_force_support"))
    findings.extend(check_pattern_simple(
        spec.get("cli_coverage_report", {}), "cli_coverage_report"))
    findings.extend(check_atomic_write(spec.get("atomic_write", {})))
    findings.extend(check_windows_safety(spec.get("windows_safety", [])))
    # R32 B1.4: lifecycle registry vs production_blends.yaml consistency
    findings.extend(check_lifecycle_consistency(spec.get("lifecycle", {})))
    # 2026-06-11 Phase 0: the 4 agent-layer taxonomy invariants (doc SS1.8).
    # F-frozen-in-agents, no-predicted-return-as-reward, agent-class-declared,
    # v16/v17-not-in-wm. Each two-sided + empty-glob-guarded.
    findings.extend(check_agent_taxonomy(spec.get("agent_taxonomy", {})))
    # 2026-05-16: perf anti-pattern crawler -- AST-based hot-loop detector.
    findings.extend(check_perf_anti_patterns(spec.get("perf_anti_patterns", {})))
    # 2026-05-17: connector-integrity meta-crawler -- catches canonical-class
    # drift, cost-model bypass, constraint-propagation gaps, yaml-vs-v3 drift,
    # and WM trainer coverage gaps. Five axes A1-A5; A1/A2 block commit.
    findings.extend(check_connector_integrity(spec.get("connector_integrity", {})))

    # 2026-05-25: wealth-bot SHIP-tier claim contract checker. Enforces
    # canonical fields (jackknife, top_3_pct, mechanism_falsifier_check,
    # sample-size discipline vs stressed compound) on every audit JSON
    # that declares a SHIP-tier candidate. Provenance: P4_route_basis_pos_only
    # 2026-05-25 mechanism-claim false. See check_wealth_bot_claims.py.
    # 2026-05-25 trust-stack item #4: ImportError must ESCALATE, not silent-skip.
    # The 7-expert audit (2026-05-25 19:35 SAST) flagged the prior silent-fail
    # fallback as a CRIT single-point-of-failure. If the wealth_bot_claims module
    # cannot be imported, the whole stack collapses without warning.
    try:
        from audit import check_wealth_bot_claims  # type: ignore
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        try:
            import audit.check_wealth_bot_claims as check_wealth_bot_claims  # type: ignore
        except ImportError as e:
            findings.append(Finding(
                severity="critical",
                category="wealth_bot_claim_contract",
                name="claim_contract_module_unavailable",
                file="src/audit/check_wealth_bot_claims.py",
                detail=(
                    f"Cannot import wealth_bot claim-contract checker: {e}. "
                    "Trust stack is degraded; commit halted to prevent silent bypass. "
                    "Fix: ensure src/ is on sys.path and src/wealth_bot/framework/claim_contract.py imports cleanly."
                ),
            ))
            check_wealth_bot_claims = None  # type: ignore

    if check_wealth_bot_claims is not None:
        wb_findings_dicts, _ = check_wealth_bot_claims.run_audit()
        for fd in wb_findings_dicts:
            findings.append(Finding(
                severity=fd["severity"],
                category="wealth_bot_claim_contract",
                name=fd["name"],
                file=fd.get("file"),
                detail=fd["detail"],
            ))

    # 2026-05-25 trust-stack item #8: Layer 7 — chimera liveness +
    # column-parity + content-hash gate. Pipeline-expert finding:
    # chimera 56.9h stale + 3-alias drift + missing SHA in repro.
    try:
        from audit import check_chimera_liveness  # type: ignore
    except ImportError:
        try:
            import audit.check_chimera_liveness as check_chimera_liveness  # type: ignore
        except ImportError as e:
            findings.append(Finding(
                severity="critical",
                category="chimera_liveness",
                name="chimera_liveness_module_unavailable",
                file="src/audit/check_chimera_liveness.py",
                detail=f"Cannot import chimera liveness checker: {e}",
            ))
            check_chimera_liveness = None  # type: ignore
    if check_chimera_liveness is not None:
        cl_findings, _ = check_chimera_liveness.run_audit()
        for fd in cl_findings:
            findings.append(Finding(
                severity=fd["severity"],
                category="chimera_liveness",
                name=fd["name"],
                file=fd.get("file"),
                detail=fd["detail"],
            ))

    # 2026-05-25 trust-stack item #9: Layer 8 — Holm-corrected DSR sweep gate.
    # Researcher-expert finding: at N=122 variants tested vs n=141 days,
    # MinBTL breached 15-30x. Compute family-wise corrected DSR per Bailey 2014.
    try:
        from audit import check_dsr_holm  # type: ignore
    except ImportError:
        try:
            import audit.check_dsr_holm as check_dsr_holm  # type: ignore
        except ImportError as e:
            findings.append(Finding(
                severity="critical",
                category="dsr_holm",
                name="dsr_holm_module_unavailable",
                file="src/audit/check_dsr_holm.py",
                detail=f"Cannot import Holm-corrected DSR checker: {e}",
            ))
            check_dsr_holm = None  # type: ignore
    if check_dsr_holm is not None:
        dsr_findings, _ = check_dsr_holm.run_audit()
        for fd in dsr_findings:
            findings.append(Finding(
                severity=fd["severity"],
                category="dsr_holm",
                name=fd["name"],
                file=fd.get("file"),
                detail=fd["detail"],
            ))

    # 2026-06-05 preflight gap: Layer 9 -- STRAT APPARATUS regression gate. The measurement layer (src/strat:
    # battery/firewall/positive_control/dsr) is sound now but otherwise unguarded; a regression there makes every
    # strategy verdict untrustworthy (the 2026-06-04-reset failure mode). Runs src/strat/selftest_all.py.
    try:
        from audit import check_strat_apparatus  # type: ignore
    except ImportError:
        try:
            import audit.check_strat_apparatus as check_strat_apparatus  # type: ignore
        except ImportError as e:
            findings.append(Finding(
                severity="warn",
                category="strat_apparatus",
                name="strat_apparatus_module_unavailable",
                file="src/audit/check_strat_apparatus.py",
                detail=f"Cannot import strat apparatus checker: {e}",
            ))
            check_strat_apparatus = None  # type: ignore
    if check_strat_apparatus is not None:
        sa_findings, _ = check_strat_apparatus.run_audit()
        for fd in sa_findings:
            findings.append(Finding(
                severity=fd["severity"],
                category="strat_apparatus",
                name=fd["name"],
                file=fd.get("file"),
                detail=fd["detail"],
            ))

    # 2026-06-07 G-F: wire the MANDATORY-GATE meta-verifier INTO CDAP so it is actually DISPATCHED. Previously
    # scripts/mandatory_gate.py was a standalone CLI never called by the pre-commit path -> itself an undispatched
    # gate (the irony the audit flagged). It verifies mandatory gates can't be silently skipped (bypass env-vars,
    # declared-but-undispatched invariant sections, missing pre-commit wiring). FAITHFUL severity: a mandatory_gate
    # CRITICAL (a mandatory gate IS skippable) BLOCKS here too ("what's mandatory is never skipped"); a WARN (drift,
    # e.g. the dead-guard rot tracked by G-E) -> WARN. Guarded import: a missing/broken module degrades to ONE WARN,
    # never crashes CDAP and never silently no-ops.
    try:
        _scripts_dir = str(PROJECT_ROOT / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import mandatory_gate as _mandatory_gate  # scripts/mandatory_gate.py
        for _sev, _msg in _mandatory_gate.check():
            findings.append(Finding(
                severity=("critical" if _sev == "critical" else "warn"),
                category="mandatory_gate", name="mandatory_gate_enforced",
                file="config/mandatory_gates.yaml", detail=str(_msg)[:600],
            ))
    except Exception as _e:
        findings.append(Finding(
            severity="warn", category="mandatory_gate", name="mandatory_gate_unavailable",
            file="scripts/mandatory_gate.py",
            detail=f"mandatory-gate meta-verifier did not run: {type(_e).__name__}: {_e}",
        ))

    n_critical = sum(1 for f in findings if f.severity == "critical")
    n_warn = sum(1 for f in findings if f.severity == "warn")
    if n_critical > 0:
        exit_code = 2
    elif n_warn > 0:
        exit_code = 1
    else:
        exit_code = 0

    return findings, exit_code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="only print failures + summary")
    ap.add_argument("--json",  action="store_true", help="machine-readable output")
    args = ap.parse_args()

    findings, exit_code = run_audit()

    if args.json:
        out = {
            "exit_code": exit_code,
            "n_critical": sum(1 for f in findings if f.severity == "critical"),
            "n_warn":     sum(1 for f in findings if f.severity == "warn"),
            "n_info":     sum(1 for f in findings if f.severity == "info"),
            "findings": [f.__dict__ for f in findings],
        }
        print(json.dumps(out, indent=2))
        return exit_code

    print("=" * 72)
    print("CDAP INVARIANT AUDIT")
    print(f"  registry: {INVARIANTS_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72)

    by_cat: dict = {}
    for f in findings:
        by_cat.setdefault(f.category, []).append(f)

    for cat in sorted(by_cat):
        cat_findings = by_cat[cat]
        if args.quiet and not any(x.severity in ("critical", "warn") for x in cat_findings):
            continue
        print(f"\n[{cat}]  ({len(cat_findings)} findings)")
        for f in cat_findings:
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
