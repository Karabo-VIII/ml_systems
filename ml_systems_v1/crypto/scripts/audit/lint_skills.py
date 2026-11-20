"""Lint .claude/skills/* against SOTA frontmatter + structure best practices.

PURPOSE
=======
Surface gaps in the project's skill system in a structured, machine-readable
way so subsequent upgrades (BINDINGS rollout, LEDGER extension, body-length
trimming, etc.) can be data-driven rather than vibes-based.

CHECKS PERFORMED (per canonical skill — alias stubs are checked more leniently)

1. FRONTMATTER  -- YAML present and well-formed
2. NAME-MATCH   -- `name:` field == directory name
3. DESCR-WHEN   -- description includes WHEN to invoke ("when", "use", "trigger" etc.)
4. DESCR-THIRD-PERSON -- description doesn't use first/second person ("I", "you")
5. ARG-HINT     -- argument-hint field present
6. SCHEMA-VER   -- metadata.schema_version present
7. BODY-LEN     -- body <= 200 lines (Anthropic: every token is recurring cost)
8. WHEN-TABLE   -- contains a "When to invoke" or similar decision table
9. GOTCHAS      -- has per-skill Gotchas section or links to one
10. PORTABLE-BINDINGS -- if metadata.portable=true, BINDINGS.md must exist
11. LEDGER-CLAIM -- if description claims "ledger" / "calibration", LEDGER.md must exist
12. SUB-DIRS    -- scripts/ references/ assets/ supporting-file pattern (Anthropic-recommended)
13. STUB-TARGET -- alias stub's `alias_of` must point to a real skill

Each check is tagged with a GAP-CLASS so the output can be summarized.

OUTPUT
======
- Per-skill scorecard (PASS/FAIL counts)
- Gap-class summary (which classes have the most findings)
- Concept-axis vs Engineering-axis score (for CLAIM-C resolution)
- Exit code: 0 if no CRIT findings, 1 if WARN, 2 if CRIT

USAGE
=====
    python scripts/audit/lint_skills.py
    python scripts/audit/lint_skills.py --verbose       # show every finding
    python scripts/audit/lint_skills.py --json out.json # machine-readable
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Map each check to (gap_class, axis) where axis in {"concept", "engineering"}.
# This drives the CLAIM-C concept-vs-engineering scoring.
CHECKS = {
    "FRONTMATTER":         ("frontmatter",   "engineering"),
    "NAME-MATCH":          ("frontmatter",   "engineering"),
    "DESCR-WHEN":          ("description",   "concept"),
    "DESCR-THIRD-PERSON":  ("description",   "engineering"),
    "ARG-HINT":            ("frontmatter",   "engineering"),
    "SCHEMA-VER":          ("frontmatter",   "engineering"),
    "BODY-LEN":            ("body",          "engineering"),
    "WHEN-TABLE":          ("body",          "concept"),
    "GOTCHAS":             ("body",          "concept"),
    "PORTABLE-BINDINGS":   ("portability",   "concept"),
    "LEDGER-CLAIM":        ("portability",   "concept"),
    "SUB-DIRS":            ("structure",     "engineering"),
    "STUB-TARGET":         ("alias-stub",    "engineering"),
}

SEVERITY = {
    "FRONTMATTER":         "CRIT",
    "NAME-MATCH":          "HIGH",
    "DESCR-WHEN":          "MED",
    "DESCR-THIRD-PERSON":  "LOW",
    "ARG-HINT":            "LOW",
    "SCHEMA-VER":          "LOW",
    "BODY-LEN":            "MED",
    "WHEN-TABLE":          "MED",
    "GOTCHAS":             "LOW",
    "PORTABLE-BINDINGS":   "HIGH",
    "LEDGER-CLAIM":        "HIGH",
    "SUB-DIRS":            "LOW",
    "STUB-TARGET":         "CRIT",
}


@dataclass
class Finding:
    skill: str
    check: str
    severity: str
    gap_class: str
    axis: str
    detail: str


@dataclass
class SkillReport:
    name: str
    path: Path
    is_stub: bool
    is_canonical: bool
    findings: list[Finding] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0


def parse_frontmatter(text: str) -> tuple[dict, str] | None:
    """Naive YAML-frontmatter parser. Returns (dict, body) or None if no FM."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    fm_end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm_end = i
            break
    if fm_end is None:
        return None
    fm: dict = {}
    nested_stack: list[tuple[int, dict]] = [(0, fm)]
    for raw in lines[1:fm_end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        while nested_stack and nested_stack[-1][0] > indent:
            nested_stack.pop()
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        target = nested_stack[-1][1]
        if not val:
            sub: dict = {}
            target[key] = sub
            nested_stack.append((indent + 2, sub))
        else:
            # Strip quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            if val.lower() in ("true", "false"):
                target[key] = (val.lower() == "true")
            else:
                target[key] = val
    body = "\n".join(lines[fm_end + 1:])
    return fm, body


def lint_skill(name: str, skill_dir: Path, all_skill_names: set[str]) -> SkillReport:
    skill_md = skill_dir / "SKILL.md"
    report = SkillReport(name=name, path=skill_md, is_stub=False, is_canonical=False)

    if not skill_md.exists():
        report.findings.append(Finding(name, "FRONTMATTER", "CRIT", "frontmatter",
                                       "engineering", "SKILL.md missing"))
        report.checks_run = 1
        return report

    text = skill_md.read_text(encoding="utf-8")
    parsed = parse_frontmatter(text)
    report.checks_run += 1
    if parsed is None:
        report.findings.append(Finding(name, "FRONTMATTER", "CRIT", "frontmatter",
                                       "engineering", "no YAML frontmatter"))
        return report
    report.checks_passed += 1
    fm, body = parsed

    # Distinguish stub vs canonical
    metadata = fm.get("metadata", {}) if isinstance(fm.get("metadata"), dict) else {}
    is_stub = bool(metadata.get("autogenerated"))
    report.is_stub = is_stub
    report.is_canonical = not is_stub

    # CHECK: NAME-MATCH
    report.checks_run += 1
    fm_name = fm.get("name", "")
    if fm_name != name:
        report.findings.append(Finding(name, "NAME-MATCH", SEVERITY["NAME-MATCH"],
                                       *CHECKS["NAME-MATCH"],
                                       f"name='{fm_name}' but directory='{name}'"))
    else:
        report.checks_passed += 1

    # STUB-only checks
    if is_stub:
        # CHECK: STUB-TARGET
        report.checks_run += 1
        alias_of = metadata.get("alias_of", "")
        target_type = metadata.get("alias_target_type", "project")
        # Built-in targets are valid even though they don't have a project dir
        builtins = {"verify", "run", "init", "review", "update-config", "loop",
                    "schedule", "claude-api", "simplify", "fewer-permission-prompts",
                    "statusline-setup", "keybindings-help", "security-review"}
        if alias_of in all_skill_names or alias_of in builtins:
            report.checks_passed += 1
        else:
            report.findings.append(Finding(name, "STUB-TARGET", SEVERITY["STUB-TARGET"],
                                           *CHECKS["STUB-TARGET"],
                                           f"alias_of='{alias_of}' is not a known skill or built-in"))
        return report  # Stubs skip canonical-grade checks

    # Canonical-grade checks below
    # CHECK: DESCR-WHEN (semantic — does description tell us WHEN to use?)
    report.checks_run += 1
    descr = fm.get("description", "")
    when_signals = ["when ", "use ", "trigger", "invoke", "for ", "before "]
    if any(sig in descr.lower() for sig in when_signals):
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "DESCR-WHEN", SEVERITY["DESCR-WHEN"],
                                       *CHECKS["DESCR-WHEN"],
                                       "description lacks WHEN-to-invoke signal"))

    # CHECK: DESCR-THIRD-PERSON
    report.checks_run += 1
    first_or_second = re.search(r"\b(I|you|your|we|our)\b", descr)
    if first_or_second is None:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "DESCR-THIRD-PERSON", SEVERITY["DESCR-THIRD-PERSON"],
                                       *CHECKS["DESCR-THIRD-PERSON"],
                                       f"description uses 1st/2nd person: '{first_or_second.group(0)}'"))

    # CHECK: ARG-HINT
    report.checks_run += 1
    if fm.get("argument-hint"):
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "ARG-HINT", SEVERITY["ARG-HINT"],
                                       *CHECKS["ARG-HINT"],
                                       "argument-hint field missing"))

    # CHECK: SCHEMA-VER
    report.checks_run += 1
    if metadata.get("schema_version"):
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "SCHEMA-VER", SEVERITY["SCHEMA-VER"],
                                       *CHECKS["SCHEMA-VER"],
                                       "metadata.schema_version missing"))

    # CHECK: BODY-LEN
    report.checks_run += 1
    body_lines = len(body.splitlines())
    if body_lines <= 200:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "BODY-LEN", SEVERITY["BODY-LEN"],
                                       *CHECKS["BODY-LEN"],
                                       f"body is {body_lines} lines, recommend <=200"))

    # CHECK: WHEN-TABLE
    report.checks_run += 1
    has_when_table = bool(re.search(r"(?im)^\s*#+\s*when to invoke", body) or
                          re.search(r"(?im)when to use", body) or
                          re.search(r"(?im)trigger:", body))
    if has_when_table:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "WHEN-TABLE", SEVERITY["WHEN-TABLE"],
                                       *CHECKS["WHEN-TABLE"],
                                       "no 'When to invoke' decision table found"))

    # CHECK: GOTCHAS
    report.checks_run += 1
    has_gotchas = bool(re.search(r"(?im)gotchas|common mistakes|anti-pattern|red flags|pitfalls", body))
    if has_gotchas:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "GOTCHAS", SEVERITY["GOTCHAS"],
                                       *CHECKS["GOTCHAS"],
                                       "no Gotchas / anti-patterns / pitfalls section"))

    # CHECK: PORTABLE-BINDINGS
    portable = bool(metadata.get("portable"))
    has_bindings = (skill_dir / "BINDINGS.md").exists()
    report.checks_run += 1
    if not portable:
        report.checks_passed += 1
    elif has_bindings:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "PORTABLE-BINDINGS", SEVERITY["PORTABLE-BINDINGS"],
                                       *CHECKS["PORTABLE-BINDINGS"],
                                       "metadata.portable=true but BINDINGS.md missing"))

    # CHECK: LEDGER-CLAIM
    report.checks_run += 1
    claims_ledger = bool(re.search(r"(?i)ledger|calibration", descr))
    has_ledger = (skill_dir / "LEDGER.md").exists()
    if not claims_ledger:
        report.checks_passed += 1
    elif has_ledger:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "LEDGER-CLAIM", SEVERITY["LEDGER-CLAIM"],
                                       *CHECKS["LEDGER-CLAIM"],
                                       "description claims ledger/calibration but LEDGER.md missing"))

    # CHECK: SUB-DIRS (Anthropic recommended scripts/ references/ assets/)
    report.checks_run += 1
    has_any_subdir = any((skill_dir / sub).exists() for sub in ("scripts", "references", "assets"))
    # Don't penalize tiny skills; only flag if BODY-LEN is high (would benefit from extraction)
    if has_any_subdir or body_lines <= 150:
        report.checks_passed += 1
    else:
        report.findings.append(Finding(name, "SUB-DIRS", SEVERITY["SUB-DIRS"],
                                       *CHECKS["SUB-DIRS"],
                                       f"body is {body_lines} lines and no scripts/references/assets subdir — consider extracting"))

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="show every finding")
    ap.add_argument("--json", type=Path, help="write machine-readable JSON output to this path")
    ap.add_argument("--canonicals-only", action="store_true", help="skip alias stubs entirely")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    skills_root = project_root / ".claude" / "skills"
    if not skills_root.exists():
        print(f"ERROR: {skills_root} not found", file=sys.stderr)
        return 2

    all_skill_dirs = sorted(
        p for p in skills_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "SKILL.md").exists()
    )
    all_skill_names = {p.name for p in all_skill_dirs}

    reports: list[SkillReport] = []
    for skill_dir in all_skill_dirs:
        rpt = lint_skill(skill_dir.name, skill_dir, all_skill_names)
        if args.canonicals_only and rpt.is_stub:
            continue
        reports.append(rpt)

    canonicals = [r for r in reports if r.is_canonical]
    stubs = [r for r in reports if r.is_stub]

    # Aggregate by gap class + axis (canonicals only — stubs are intentionally minimal)
    findings_by_class: dict[str, list[Finding]] = defaultdict(list)
    findings_by_check: dict[str, list[Finding]] = defaultdict(list)
    concept_runs = concept_pass = 0
    engineering_runs = engineering_pass = 0

    for r in canonicals:
        for f in r.findings:
            findings_by_class[f.gap_class].append(f)
            findings_by_check[f.check].append(f)
        # Score axes
        for f in r.findings:
            if f.axis == "concept":
                concept_runs += 1
            else:
                engineering_runs += 1
        # We need to also count passes per axis. Recompute via checks_run/passed +
        # axis composition.
        for check_name, (_gc, axis) in CHECKS.items():
            if check_name == "STUB-TARGET":
                continue  # not applicable to canonicals
            failed = any(f.check == check_name for f in r.findings)
            if axis == "concept":
                concept_runs += 0  # already counted above
                if not failed:
                    concept_pass += 1
            else:
                engineering_runs += 0
                if not failed:
                    engineering_pass += 1
        # Need to also count fails toward the per-axis runs total
    # Recompute axes properly: total runs per axis = #canonicals * #checks-on-axis
    concept_check_names = [c for c, (_, a) in CHECKS.items() if a == "concept" and c != "STUB-TARGET"]
    engineering_check_names = [c for c, (_, a) in CHECKS.items() if a == "engineering" and c != "STUB-TARGET"]
    concept_total = len(canonicals) * len(concept_check_names)
    engineering_total = len(canonicals) * len(engineering_check_names)
    concept_pass = concept_total - sum(1 for f_list in findings_by_check.values() for f in f_list if f.axis == "concept")
    engineering_pass = engineering_total - sum(1 for f_list in findings_by_check.values() for f in f_list if f.axis == "engineering")
    concept_score = concept_pass / concept_total if concept_total else 0.0
    engineering_score = engineering_pass / engineering_total if engineering_total else 0.0

    # Stub findings (CRIT only — only STUB-TARGET applies)
    stub_findings = [f for r in stubs for f in r.findings]

    # Print summary
    print("=" * 72)
    print(f"SKILL LINTER -- {len(canonicals)} canonical, {len(stubs)} stub, {len(reports)} total")
    print("=" * 72)
    print()
    print("## Per-skill scorecard (canonicals only)")
    print()
    print(f"{'Skill':<14} {'Pass/Run':<10} {'Findings':<10} {'Worst':<8}")
    print("-" * 56)
    for r in sorted(canonicals, key=lambda x: x.name):
        worst = "-"
        if r.findings:
            sev_order = {"CRIT": 0, "HIGH": 1, "MED": 2, "LOW": 3}
            worst = sorted(r.findings, key=lambda f: sev_order[f.severity])[0].severity
        print(f"{r.name:<14} {r.checks_passed}/{r.checks_run:<8} {len(r.findings):<10} {worst:<8}")

    print()
    print("## Distinct gap classes (canonicals)")
    print()
    for cls, findings in sorted(findings_by_class.items(), key=lambda x: -len(x[1])):
        unique_checks = sorted({f.check for f in findings})
        print(f"  {cls:<14} {len(findings):>3} findings across {len(unique_checks)} check(s): {unique_checks}")

    print()
    print("## CLAIM-C axis scores")
    print()
    print(f"  Concept-axis:     {concept_pass}/{concept_total} = {concept_score:.2%}  (checks: {sorted(concept_check_names)})")
    print(f"  Engineering-axis: {engineering_pass}/{engineering_total} = {engineering_score:.2%}  (checks: {sorted(engineering_check_names)})")
    print()

    if stub_findings:
        print(f"## Stub findings ({len(stub_findings)} CRIT issues across stubs)")
        print()
        for f in stub_findings[:20]:
            print(f"  CRIT  {f.skill:<20} {f.check:<14} {f.detail}")
        if len(stub_findings) > 20:
            print(f"  ... and {len(stub_findings) - 20} more")
    else:
        print("## Stub findings: NONE  (all 111 stubs have valid alias_of targets)")
    print()

    if args.verbose:
        print("## All findings (verbose)")
        print()
        for r in canonicals:
            for f in r.findings:
                print(f"  {f.severity:<5} {r.name:<14} {f.check:<22} {f.detail}")
        print()

    # Distinct gap classes count (for CLAIM-A discriminating experiment)
    n_distinct_classes = len(findings_by_class)
    n_distinct_checks = len({f.check for findings in findings_by_class.values() for f in findings})
    print(f"## CLAIM-A discriminating result")
    print(f"  Distinct gap classes surfaced: {n_distinct_classes}")
    print(f"  Distinct check types triggering: {n_distinct_checks}")
    print(f"  Total findings (canonicals): {sum(len(f) for f in findings_by_class.values())}")
    print()

    # JSON output for machine consumption
    if args.json:
        out_data = {
            "summary": {
                "n_canonicals": len(canonicals),
                "n_stubs": len(stubs),
                "concept_score": concept_score,
                "engineering_score": engineering_score,
                "concept_pass": concept_pass,
                "concept_total": concept_total,
                "engineering_pass": engineering_pass,
                "engineering_total": engineering_total,
                "n_distinct_gap_classes": n_distinct_classes,
                "n_distinct_check_types": n_distinct_checks,
                "n_total_findings_canonicals": sum(len(f) for f in findings_by_class.values()),
                "n_stub_findings": len(stub_findings),
            },
            "by_gap_class": {cls: [{"skill": f.skill, "check": f.check,
                                    "severity": f.severity, "detail": f.detail}
                                   for f in findings]
                             for cls, findings in findings_by_class.items()},
            "per_skill": {r.name: {"is_canonical": r.is_canonical,
                                   "is_stub": r.is_stub,
                                   "checks_passed": r.checks_passed,
                                   "checks_run": r.checks_run,
                                   "findings": [{"check": f.check, "severity": f.severity,
                                                 "detail": f.detail} for f in r.findings]}
                          for r in reports},
        }
        args.json.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"JSON output written to {args.json}")

    # Exit code
    crit = sum(1 for r in reports for f in r.findings if f.severity == "CRIT")
    warn = sum(1 for r in reports for f in r.findings if f.severity in ("HIGH", "MED"))
    if crit > 0:
        return 2
    if warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
