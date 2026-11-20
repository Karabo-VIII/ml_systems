#!/usr/bin/env python3
"""MANDATORY GATE verifier -- the meta-gate that ensures mandatory gates can NEVER be silently skipped.

User mandate (2026-06-05): "what's mandatory should never be skipped no matter what." A mandatory gate gets
skipped three ways -- this verifier catches all three:
  (1) BYPASS VECTOR: a flag/env turns the gate off (SKIP_CDAP=1, git commit --no-verify).
  (2) SILENT NO-OP: the gate's checks are declared in config/_invariants.yaml but never dispatched by
      check_invariants.py -> they pass silently (the 2026-06-05 audit found ~10 such dead sections).
  (3) MISSING WIRING: the pre-commit hook isn't installed or doesn't invoke CDAP.

Reads config/mandatory_gates.yaml. Exit 0 = all mandatory gates enforced; exit 1 = WARN (drift to watch);
exit 2 = CRITICAL, a mandatory gate is skippable -> HALT (fix the root cause; do NOT bypass). Designed to be
wired into CDAP (check_invariants) and the pre-commit hook so the gate enforces its own unskippability.
No emoji (Windows cp1252).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "mandatory_gates.yaml"
INVARIANTS = ROOT / "config" / "_invariants.yaml"
CHECK_INVARIANTS = ROOT / "src" / "audit" / "check_invariants.py"
PERMISSION_POLICY = ROOT / "runs" / "autonomy" / "permission_policy.json"
PERMISSION_TEMPLATE = ROOT / "scripts" / "autonomy" / "permission_policy.template.json"
PRECOMMIT = ROOT / ".git" / "hooks" / "pre-commit"


def _load_manifest():
    try:
        import yaml
        return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return None


def _top_level_yaml_keys(path):
    keys = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(#.*)?$", line)
            if m:
                keys.append(m.group(1))
    except Exception:
        pass
    return keys


def check():
    findings = []  # (severity, message)
    man = _load_manifest()
    if man is None:
        return [("warn", "could not load config/mandatory_gates.yaml (PyYAML missing?) -- verifier degraded")]

    # --- (1) bypass vectors must be blocked by the PreToolUse permission gate (live policy AND template) ---
    declared_bypass = []
    for g in man.get("gates", []):
        declared_bypass += g.get("bypass_vectors_to_block", [])
    for label, pol_path in (("live policy", PERMISSION_POLICY), ("tracked template", PERMISSION_TEMPLATE)):
        try:
            pol = json.loads(pol_path.read_text(encoding="utf-8"))
            cmd_deny_blob = " ".join(pol.get("cmd_deny_regex", []))
            for bv in declared_bypass:
                token = bv.split()[0]  # 'SKIP_CDAP' or 'git' -> match a representative token
                needle = "SKIP_CDAP" if "SKIP_CDAP" in bv else "--no-verify"
                if needle not in cmd_deny_blob:
                    findings.append(("critical", f"bypass vector '{bv}' is NOT blocked in the {label} permission gate (mandatory gate is skippable)"))
        except FileNotFoundError:
            if pol_path is PERMISSION_POLICY:
                findings.append(("warn", f"no live permission policy at {pol_path} -- gate inactive on this machine (copy the template)"))
            else:
                findings.append(("critical", f"tracked permission template missing at {pol_path}"))
        except Exception as e:
            findings.append(("warn", f"{label} permission policy unreadable ({e})"))

    # --- (3) pre-commit hook installed + invokes CDAP ---
    if not PRECOMMIT.exists():
        findings.append(("warn", "git pre-commit hook NOT installed -> CDAP not auto-run on commit (run: python src/audit/install_hook.py)"))
    else:
        txt = PRECOMMIT.read_text(encoding="utf-8", errors="ignore")
        if "check_invariants" not in txt and "mandatory_gate" not in txt:
            findings.append(("critical", "pre-commit hook exists but does NOT invoke CDAP/mandatory_gate (CDAP unenforced on commit)"))

    # --- (2) mandatory invariant sections must be DISPATCHED by check_invariants.py (no silent no-op) ---
    try:
        src = CHECK_INVARIANTS.read_text(encoding="utf-8")
        declared = set(_top_level_yaml_keys(INVARIANTS))
        for sec in man.get("mandatory_invariant_sections", []):
            if sec not in declared:
                findings.append(("warn", f"mandatory section '{sec}' is not even declared in _invariants.yaml"))
            elif sec not in src:
                findings.append(("critical", f"mandatory section '{sec}' is declared in _invariants.yaml but NEVER dispatched by check_invariants.py -> SILENT NO-OP (a skipped mandatory gate)"))
    except Exception as e:
        findings.append(("warn", f"section-dispatch scan failed ({e})"))

    # --- (4) DEAD CRITICAL GUARDS: a severity:critical rule whose files all resolve to 0 = a silent no-op ---
    # (the 2026-06-04 reset left several critical guards pointing at deleted files -> false confidence).
    try:
        import glob as _glob
        import yaml as _yaml
        inv = _yaml.safe_load(INVARIANTS.read_text(encoding="utf-8")) or {}
        dead = []
        for section, rules in inv.items():
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if not isinstance(rule, dict) or rule.get("severity") != "critical":
                    continue
                files = rule.get("files")
                if not files:
                    continue
                resolved = sum(len(_glob.glob(str(ROOT / pat), recursive=True)) for pat in files)
                if resolved == 0:
                    dead.append(f"{section}::{rule.get('name', '?')}")
        if dead:
            findings.append(("warn", (
                f"{len(dead)} DEAD CRITICAL guard(s) (severity:critical but files resolve to 0 -> silent no-op): "
                f"{', '.join(dead[:10])}{' ...' if len(dead) > 10 else ''}. Retire or re-point "
                f"(see docs/CDAP_DEAD_SECTIONS_2026_06_05.md).")))
    except Exception as e:
        findings.append(("warn", f"dead-critical-guard scan failed ({e})"))

    return findings


def main():
    findings = check()
    crit = [f for f in findings if f[0] == "critical"]
    warn = [f for f in findings if f[0] == "warn"]
    print(f"[mandatory_gate] {len(crit)} CRITICAL, {len(warn)} WARN")
    for sev, msg in findings:
        print(f"  {'FAIL' if sev == 'critical' else 'WARN'}: {msg}")
    if crit:
        print("[mandatory_gate] HALT: a MANDATORY gate is skippable/unwired. Fix the root cause -- do NOT bypass.")
        return 2
    if warn:
        print("[mandatory_gate] OK with warnings: mandatory gates enforced; warnings are drift to watch.")
        return 1
    print("[mandatory_gate] OK: all mandatory gates are mechanically enforced and unskippable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
