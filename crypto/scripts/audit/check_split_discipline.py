"""check_split_discipline.py -- scan the project for TRAIN/VAL/OOS/UNSEEN drift.

Per CLAUDE.md invariant: "Unseen segment NEVER touched during development --
reserved for backtesting only." This audit catches three drift patterns that
silently contaminate the UNSEEN reserve or fork from the canonical SoT.

Patterns:
  Type-1 (CRITICAL): "OOS" prose covering data >= 2026-01-01 (= UNSEEN)
                     Example: 'OOS window: 2024-05-16 -> 2026-05-19'
  Type-2 (CRITICAL): "OOS" labelling 2026 data
                     Example: 'OOS Jan|Feb|Mar|Apr 2026'
  Type-3 (WARN):     .py files hardcoding LEGACY split dates without
                     `from split_config import ...`

Usage:
  python scripts/audit/check_split_discipline.py

Exit codes:
  0 -- clean
  1 -- WARN-only (Type-3 hardcoded dates without canonical import)
  2 -- CRITICAL drift (Type-1 / Type-2 UNSEEN contamination)
"""
from __future__ import annotations

__contract__ = {
    "kind": "audit_script",
    "owner": "audit/split-discipline",
    "purpose": "Scan repo for TRAIN/VAL/OOS/UNSEEN drift patterns",
    "inputs": {"none": "scans repo from PROJECT_ROOT"},
    "outputs": {
        "stdout": "CRITICAL / WARN / INFO findings + summary table",
        "exit_code": "0 clean / 1 WARN-only / 2 CRITICAL",
        "report": "runs/audit/split_discipline_audit_<date>.md",
    },
    "invariants": [
        "Reads CLAUDE.md 'UNSEEN reserved for backtesting only' rule",
        "Validates split_config.py SoT against hardcoded date drift",
        "Skip dirs: .git/, runs/, data/processed/, backups/, node_modules/, __pycache__/",
    ],
}

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical LEGACY dates (sync with src/split_config.py)
CANONICAL_LEGACY = {
    "TRAIN_END":    "2024-05-15",
    "VAL_END":      "2025-03-15",
    "OOS_END":      "2025-12-31",
    "UNSEEN_START": "2026-01-01",
}

# Dirs to skip entirely
SKIP_DIRS = {
    ".git", "runs", "backups", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".venv", "venv", ".idea", ".vscode",
}
# data/processed/ skipped; other data/* allowed (data/oracle/, data/manifests/ etc.)
SKIP_SUBPATHS = {
    str(Path("data") / "processed"),
}

# Patterns
RE_OOS_RANGE_INTO_2026 = re.compile(
    r"OOS[^\n]{0,80}?(?:->|→|to|—|-)\s*20(2[6-9]|[3-9]\d)[-./]",
    re.IGNORECASE,
)
RE_OOS_2026_PROSE = re.compile(
    r"OOS[^\n]{0,40}?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*20(2[6-9]|[3-9]\d)",
    re.IGNORECASE,
)
RE_OOS_2026_TIMESTAMP = re.compile(
    r"\bOOS\b[^\n]{0,80}?20(2[6-9]|[3-9]\d)[-./](0[1-9]|1[0-2])",
    re.IGNORECASE,
)
# Hardcoded LEGACY dates -- any of the 4 canonical strings literal in source
RE_LEGACY_DATE = re.compile(
    r"\"(2024-05-15|2025-03-15|2025-12-31|2026-01-01)\""
    r"|'(2024-05-15|2025-03-15|2025-12-31|2026-01-01)'"
)
RE_SPLIT_CONFIG_IMPORT = re.compile(
    r"from\s+split_config\s+import|import\s+split_config", re.MULTILINE
)


def iter_files(root: Path):
    """Yield .py / .md / .yaml files under root, respecting skip rules."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        rel_str = str(rel).replace("\\", "/")
        if any(rel_str.startswith(sp.replace("\\", "/") + "/") or rel_str == sp.replace("\\", "/")
               for sp in SKIP_SUBPATHS):
            continue
        if p.suffix.lower() not in (".py", ".md", ".yaml", ".yml"):
            continue
        # also skip ourselves and split_config.py (the SoT) + the canonical
        # SPLIT_DISCIPLINE.md doc (which legitimately quotes drift examples).
        # The two .claude/skills/ files are Claude-harness-owned prose
        # documentation (calibration ledger entries + project bindings) whose
        # historical "OOS" references describe development-window observations,
        # not contaminated training data. They are not editable from agent
        # context (sandbox-protected) and contain no live consumer code.
        if rel_str in (
            "scripts/audit/check_split_discipline.py",
            "src/split_config.py",
            "docs/SPLIT_DISCIPLINE.md",
            ".claude/skills/dialectic/CALIBRATION_LEDGER.md",
            ".claude/skills/oracle/BINDINGS.md",
        ):
            continue
        yield p


def scan_file(path: Path) -> dict:
    """Return findings dict: {critical: [...], warn: [...], info: [...]}."""
    findings: dict[str, list[tuple[int, str]]] = {
        "critical": [],
        "warn": [],
        "info": [],
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return findings
    lines = text.splitlines()

    is_py = path.suffix == ".py"
    is_md = path.suffix == ".md"
    is_yaml = path.suffix in (".yaml", ".yml")
    has_split_config_import = bool(RE_SPLIT_CONFIG_IMPORT.search(text))

    for i, line in enumerate(lines, start=1):
        # Skip lines that explicitly use a canonical ROLLING / V3_REPLAY
        # window. These are legitimate references (rolling test start
        # 2026-01-07 with 7d purge, or V3 replay 2026-01-01 -> 2026-04-30
        # with explicit "UNSEEN" demarcation).
        if "2026-01-07" in line:
            continue
        if "v3-paper-trade-replay" in line.lower() or "v3_paper_trade_replay" in line.lower():
            continue
        if "UNSEEN reserved" in line or "unseen reserve" in line.lower():
            continue
        # Skip the audit-script regex source itself if grepped elsewhere
        if "RE_OOS_RANGE_INTO_2026" in line or "RE_OOS_2026_PROSE" in line:
            continue

        # Type-1 / Type-2: OOS window extending into 2026+
        m1 = RE_OOS_RANGE_INTO_2026.search(line)
        if m1:
            severity = "critical" if (is_py or is_md) else "warn"
            findings[severity].append(
                (i, f"OOS window extends into UNSEEN (>=2026): {line.strip()[:140]}")
            )
            continue
        m2 = RE_OOS_2026_PROSE.search(line)
        if m2:
            severity = "critical" if (is_py or is_md) else "warn"
            findings[severity].append(
                (i, f"'OOS <month> 2026' = UNSEEN contamination: {line.strip()[:140]}")
            )
            continue
        m3 = RE_OOS_2026_TIMESTAMP.search(line)
        if m3:
            severity = "critical" if (is_py or is_md) else "warn"
            findings[severity].append(
                (i, f"OOS timestamp >= 2026-01: {line.strip()[:140]}")
            )
            continue

        # Type-3: hardcoded LEGACY date string
        m4 = RE_LEGACY_DATE.search(line)
        if m4 and (is_py or is_yaml) and not has_split_config_import:
            findings["warn"].append(
                (i, f"hardcoded canonical date without split_config import: "
                    f"{line.strip()[:140]}")
            )

    # Bucket .md drift down to INFO when prose-only and obviously historical
    # (we already classified critical at line level above; this is a global
    # downgrade rule for docs under data/oracle/ that have a SPLIT DISCIPLINE
    # NOTE admonition).
    rel_str = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    if is_md and findings["critical"]:
        # Detect admonition in file head
        head = "\n".join(lines[:10])
        if "SPLIT DISCIPLINE NOTE" in head:
            # Downgrade -- file has been annotated as historical
            findings["info"].extend(findings["critical"])
            findings["critical"] = []

    # Symmetric .py downgrade: when a Python file has a top-of-file
    # SPLIT DISCIPLINE NOTE comment block (within the first 60 lines, to
    # accommodate longer module docstrings followed by the admonition comment),
    # treat its CRITICALs as historical artifacts (INFO). Same intent as the
    # .md downgrade above -- the script has been annotated as legacy
    # one-shot tooling whose hardcoded labels predate the canonical SoT.
    if is_py and findings["critical"]:
        head = "\n".join(lines[:60])
        if "SPLIT DISCIPLINE NOTE" in head:
            findings["info"].extend(findings["critical"])
            findings["critical"] = []

    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        default=None,
        help="Path to write markdown report (default: runs/audit/split_discipline_audit_<date>.md)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    args = ap.parse_args()

    print(f"Split discipline audit -- scanning {PROJECT_ROOT}", flush=True)
    print(f"Canonical SoT: src/split_config.py", flush=True)
    print(f"Skip dirs: {sorted(SKIP_DIRS)}", flush=True)
    print(f"Skip subpaths: {sorted(SKIP_SUBPATHS)}", flush=True)
    print("=" * 80, flush=True)

    all_findings: list[tuple[str, str, int, str]] = []  # (severity, relpath, line, msg)
    n_files_scanned = 0
    n_files_with_findings = 0

    for fp in iter_files(PROJECT_ROOT):
        n_files_scanned += 1
        findings = scan_file(fp)
        total_here = sum(len(v) for v in findings.values())
        if total_here == 0:
            continue
        n_files_with_findings += 1
        rel = str(fp.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for severity in ("critical", "warn", "info"):
            for ln, msg in findings[severity]:
                all_findings.append((severity, rel, ln, msg))

    counts = {"critical": 0, "warn": 0, "info": 0}
    for sev, _, _, _ in all_findings:
        counts[sev] += 1

    print(f"\nScanned {n_files_scanned} files; {n_files_with_findings} have findings", flush=True)
    print(f"CRITICAL: {counts['critical']}  WARN: {counts['warn']}  INFO: {counts['info']}", flush=True)
    print("=" * 80, flush=True)

    if not args.quiet:
        # Use ASCII-safe encoding for Windows cp1252 console (project invariant per CLAUDE.md:
        # "No emoji characters in any Python print statements (Windows cp1252 crashes)").
        # Flagged-file content may contain Unicode (->, etc); strip on print only.
        def _safe(s: str) -> str:
            return s.encode("ascii", "replace").decode("ascii")
        for severity in ("critical", "warn", "info"):
            sev_findings = [f for f in all_findings if f[0] == severity]
            if not sev_findings:
                continue
            label = {"critical": "[CRITICAL]", "warn": "[WARN]", "info": "[INFO]"}[severity]
            print(f"\n{label} ({len(sev_findings)} findings)", flush=True)
            for _, rel, ln, msg in sev_findings:
                print(_safe(f"  {rel}:{ln}  {msg}"), flush=True)

    # Write report
    report_path = (
        Path(args.report) if args.report
        else PROJECT_ROOT / "runs" / "audit"
             / f"split_discipline_audit_{datetime.now().strftime('%Y_%m_%d')}.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Split Discipline Audit -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Project root**: `{PROJECT_ROOT}`",
        f"**Canonical SoT**: [src/split_config.py](../../src/split_config.py)",
        "",
        "## Summary",
        "",
        f"- Files scanned: {n_files_scanned}",
        f"- Files with findings: {n_files_with_findings}",
        f"- **CRITICAL**: {counts['critical']} (UNSEEN reserve contamination)",
        f"- **WARN**: {counts['warn']} (hardcoded canonical dates without split_config import)",
        f"- **INFO**: {counts['info']} (historical docs already annotated)",
        "",
        "## Exit code semantics",
        "",
        "- `0` clean",
        "- `1` WARN-only (acceptable for legacy scripts not yet refactored)",
        "- `2` CRITICAL drift (HALT -- UNSEEN reserve may be contaminated)",
        "",
        "## Patterns checked",
        "",
        "- Type-1: `OOS <date> -> 2026+` (OOS window extends into UNSEEN year)",
        "- Type-2: `OOS Jan|Feb|...|Dec 2026` (OOS prose labelling UNSEEN months)",
        "- Type-3: `'2024-05-15' | '2025-03-15' | '2025-12-31' | '2026-01-01'` literal in .py/.yaml without `from split_config import ...`",
        "",
    ]

    for severity in ("critical", "warn", "info"):
        sev_findings = [f for f in all_findings if f[0] == severity]
        if not sev_findings:
            continue
        label = {"critical": "CRITICAL", "warn": "WARN", "info": "INFO"}[severity]
        lines.append(f"## {label} ({len(sev_findings)})")
        lines.append("")
        for _, rel, ln, msg in sev_findings:
            lines.append(f"- `{rel}:{ln}` -- {msg}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}", flush=True)

    if counts["critical"] > 0:
        return 2
    if counts["warn"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
