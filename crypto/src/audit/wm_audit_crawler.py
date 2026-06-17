"""wm_audit_crawler.py -- world-model layer comprehensive audit.

User mandate 2026-05-16 (post-pipeline overhaul). Sister to
pipeline_audit_crawler / pipeline_staleness_crawler / doc_audit_crawler.
Audits every world-model version under src/wm/v*/ against the CLAUDE.md
"Indisputable Operating Lens" + the WM_HEADLINE_UPGRADE_PLAN.

SEVEN AUDIT AXES
================
  1. CROSS-VERSION INVARIANTS DRIFT  : settings.py constants that MUST be
                                         identical (BIN_MIN/MAX, NUM_BINS,
                                         ACTIVE_HORIZONS, WM_BATCH_SIZE, etc.)
  2. CC-H* HEADLINE COMPONENT WIRING : per version, which of the shared
                                         Headline components (CC-H1..H7) is
                                         actually wired in train_world_model.py
  3. TRAINER INVARIANTS              : strict=False on load_state_dict,
                                         shic_decline_count persisted,
                                         n_features in checkpoint, collision
                                         guard, 6-tuple load_latest return
  4. IRON-CLAD ANTI-MEMO             : at least one of RSSM/VIB/ATME/JEPA/
                                         time-discriminator wired
  5. HEADLINE PLAN COVERAGE          : every active version present in
                                         WM_HEADLINE_UPGRADE_PLAN_2026_04_30
  6. CKPT vs SETTINGS STALENESS      : checkpoint mtime older than
                                         settings.py = ckpt may load against
                                         drifted schema
  7. FIX LOG PRESENCE                : memory/fix_logs/v{N}_{M}.md exists

OUTPUT
------
runs/audit/wm_audit_<DATE>.md  -- per-version findings + remediation queue
runs/audit/wm_audit_<DATE>.json -- machine-readable mirror

INVOKE
------
    python src/audit/wm_audit_crawler.py
    python src/audit/wm_audit_crawler.py --version v1_1
    python src/audit/wm_audit_crawler.py --axes invariants,wiring
"""
from __future__ import annotations

__contract__ = {
    "kind": "wm_audit_crawler",
    "owner": "audit/wm",
    "outputs": [
        "runs/audit/wm_audit_<DATE>.md",
        "runs/audit/wm_audit_<DATE>.json",
    ],
    "invariants": [
        "non-invasive: reads code + ckpt mtimes only; never instantiates models",
        "every finding includes version + file:line + remediation prescription",
        "complements pipeline_audit_crawler + check_invariants CDAP",
    ],
}

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WM_BASE = PROJECT_ROOT / "src" / "wm"
SHARED_DIR = WM_BASE / "_shared"
MODELS_DIR = PROJECT_ROOT / "models"
FIX_LOG_DIR = PROJECT_ROOT / "memory" / "fix_logs"
DOCS_DIR = PROJECT_ROOT / "docs"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical cross-version invariants (from CLAUDE.md table).
CANONICAL_INVARIANTS: dict[str, str] = {
    "WM_STEPS_PER_EPOCH": "2000",
    "DIVERSITY_STEPS_PER_EPOCH": "2000",
    "DIRECT_RETURN_WEIGHT": "3.0",
    "WM_BATCH_SIZE": "32",
    "BIN_MIN": "-1.0",
    "BIN_MAX": "1.0",
    "NUM_BINS": "255",
    "TWOHOT_FOCAL_GAMMA": "0.0",
}
CANONICAL_LIST: dict[str, list] = {
    "ACTIVE_HORIZONS": [1, 4, 16, 64],
}

# CC-H component name patterns. Per WM_HEADLINE_UPGRADE_PLAN_2026_04_30 §0
# the shared module lives at src/wm/_shared/headline_components.py.
# Tightened to require either (a) import of the headline_components symbol or
# (b) class-instantiation call, not loose substrings (which match comments).
CC_COMPONENT_PATTERNS: dict[str, list[str]] = {
    "CC-H1 multi-resolution": [
        r"from\s+.*headline_components\s+import\s+.*MultiResolution",
        r"MultiResolutionEncoder\s*\(",
    ],
    "CC-H2 linear-attn": [
        r"from\s+.*headline_components\s+import\s+.*LinearAttention",
        r"LinearAttention\s*\(",
        r"\bHyenaOperator\s*\(",
        r"\bPerformerBlock\s*\(",
    ],
    "CC-H3 cross-asset": [
        r"from\s+.*headline_components\s+import\s+.*CrossAsset",
        r"CrossAssetAttention\s*\(",
    ],
    "CC-H5 quantile-head": [
        r"from\s+.*headline_components\s+import\s+.*Quantile",
        r"QuantileHead\s*\(",
        r"def\s+pinball_loss\b",
        r"USE_QUANTILE_LOSS\s*=\s*True",
    ],
    "CC-H6 regime-cond": [
        r"from\s+.*headline_components\s+import\s+.*Regime",
        r"RegimeConditional\s*\(",
        r"REGIME_HEADS\s*=\s*True",
    ],
    "CC-H7 dream-rollout": [
        r"from\s+.*headline_components\s+import\s+.*Dream",
        r"DreamRolloutLoss\s*\(",
        r"dream_step\s*\(.*\).*loss",
        r"DREAM_LOSS_WEIGHT\s*=",
    ],
}

# Trainer-invariant patterns (CLAUDE.md "Code Change Verification" #11).
TRAINER_INVARIANT_PATTERNS: dict[str, str] = {
    "strict_false_on_load":     r"load_state_dict\([^)]*strict\s*=\s*False",
    "shic_decline_count":        r"shic_decline_count",
    "n_features_in_ckpt":        r"\bn_features\b",
    "load_latest_collision":     r"collision|n_features_in_ckpt",
}

# Iron-clad anti-memo mechanism signatures.
ANTI_MEMO_PATTERNS: dict[str, list[str]] = {
    "RSSM":          [r"\bRSSM\b", r"recurrent_state_space", r"rssm"],
    "VIB":           [r"\bVIB\b", r"VariationalInformationBottleneck"],
    "ATME":          [r"TEMPORAL_CTX_DROP", r"ATME", r"atme"],
    "JEPA":          [r"\bJEPA\b", r"joint_embedding"],
    "TimeDisc":      [r"TimeDiscriminator", r"adversarial.*discriminator"],
    "XD":            [r"XD_DROPOUT", r"xd_split"],
}

HEADLINE_PLAN = DOCS_DIR / "WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md"


# ===========================================================================
# Version discovery
# ===========================================================================

def discover_versions() -> list[dict]:
    """Walk src/wm/v*/ and identify each version's settings + trainer files.

    A "version" can be either:
    - A top-level dir with direct files (e.g. v15/patchtst_encoder.py)
    - A sub-dir inside a top-level (e.g. v1/v1_0_training/, v1/v1_1_training/)
    Each *_training subdir is its own version (V1.0 != V1.1 != V1.4 != V1.6).
    """
    versions: list[dict] = []
    if not WM_BASE.exists():
        return versions
    for vdir in sorted(WM_BASE.iterdir()):
        if not vdir.is_dir() or not vdir.name.startswith("v"):
            continue
        if vdir.name in ("_shared", "_archive", "__pycache__"):
            continue
        # Enumerate ALL *_training subdirs as separate sub-versions
        training_subs = sorted(vdir.glob("*_training"))
        if training_subs:
            for sub in training_subs:
                # Derive sub-version name from dir name (v1_0_training -> v1_0)
                sub_name = sub.name.replace("_training", "")
                info: dict = {"name": sub_name, "dir": vdir,
                               "settings": None, "trainer": None,
                               "wm_file": None, "search_root": sub}
                cand_settings = sub / "settings.py"
                if cand_settings.exists():
                    info["settings"] = cand_settings
                cand_trainer = sub / "train_world_model.py"
                if cand_trainer.exists():
                    info["trainer"] = cand_trainer
                cand_wm = sub / "world_model.py"
                if cand_wm.exists():
                    info["wm_file"] = cand_wm
                versions.append(info)
        else:
            # No *_training subdirs - top-level dir is the version
            info = {"name": vdir.name, "dir": vdir, "settings": None,
                     "trainer": None, "wm_file": None, "search_root": vdir}
            for fname, key in (("settings.py", "settings"),
                                 ("train_world_model.py", "trainer"),
                                 ("world_model.py", "wm_file")):
                if (vdir / fname).exists():
                    info[key] = vdir / fname
            versions.append(info)
    return versions


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


# ===========================================================================
# Axis 1: Cross-version invariants drift
# ===========================================================================

def audit_invariants_drift(versions: list[dict]) -> list[dict]:
    """For each canonical constant, scan all settings.py and flag any
    version where the value differs from the canonical."""
    findings: list[dict] = []
    for ver in versions:
        if not ver["settings"]:
            continue
        body = _read(ver["settings"])
        for const, expected in CANONICAL_INVARIANTS.items():
            # Match "CONST = value" or "CONST: type = value"
            m = re.search(rf"^{const}\s*(?::[^=]+)?=\s*([^\s#\n]+)",
                            body, re.MULTILINE)
            if not m:
                findings.append({
                    "version": ver["name"], "axis": "invariants",
                    "severity": "MEDIUM",
                    "kind": "MISSING_INVARIANT",
                    "constant": const,
                    "expected": expected,
                    "file": str(ver["settings"].relative_to(PROJECT_ROOT)),
                    "remediation": f"Add `{const} = {expected}` to settings.py",
                })
                continue
            found = m.group(1).strip().rstrip(",")
            if found != expected:
                findings.append({
                    "version": ver["name"], "axis": "invariants",
                    "severity": "HIGH",
                    "kind": "DRIFTED_INVARIANT",
                    "constant": const,
                    "expected": expected, "found": found,
                    "file": str(ver["settings"].relative_to(PROJECT_ROOT)),
                    "remediation": (f"Set `{const} = {expected}` in "
                                     f"{ver['settings'].name}"),
                })
        # List-valued invariants
        for const, expected in CANONICAL_LIST.items():
            m = re.search(rf"{const}\s*=\s*([\[(][^\]\)]+[\])])", body)
            if not m:
                findings.append({
                    "version": ver["name"], "axis": "invariants",
                    "severity": "MEDIUM",
                    "kind": "MISSING_INVARIANT",
                    "constant": const,
                    "expected": str(expected),
                    "file": str(ver["settings"].relative_to(PROJECT_ROOT)),
                    "remediation": f"Add `{const} = {expected}` to settings.py",
                })
                continue
            found = m.group(1)
            found_nums = sorted(int(n) for n in re.findall(r"\d+", found))
            if found_nums != sorted(expected):
                findings.append({
                    "version": ver["name"], "axis": "invariants",
                    "severity": "HIGH",
                    "kind": "DRIFTED_INVARIANT",
                    "constant": const,
                    "expected": str(expected), "found": found,
                    "file": str(ver["settings"].relative_to(PROJECT_ROOT)),
                    "remediation": (f"Set `{const} = {expected}` in "
                                     f"{ver['settings'].name}"),
                })
    return findings


# ===========================================================================
# Axis 2: CC-H* Headline component wiring
# ===========================================================================

def audit_cch_wiring(versions: list[dict]) -> list[dict]:
    """For each version's trainer + world_model + settings, which CC-H
    components are referenced? Per-version wiring matrix; not all components
    apply to all versions but the inventory is the missing observability."""
    findings: list[dict] = []
    shared_components_present = SHARED_DIR.joinpath("headline_components.py").exists()
    for ver in versions:
        if not ver["trainer"] and not ver["wm_file"]:
            continue
        scan_text = ""
        if ver["trainer"]:
            scan_text += _read(ver["trainer"])
        if ver["wm_file"]:
            scan_text += "\n" + _read(ver["wm_file"])
        if ver["settings"]:
            scan_text += "\n" + _read(ver["settings"])
        wired: list[str] = []
        missing: list[str] = []
        for comp, patterns in CC_COMPONENT_PATTERNS.items():
            if any(re.search(p, scan_text, re.IGNORECASE) for p in patterns):
                wired.append(comp)
            else:
                missing.append(comp)
        findings.append({
            "version": ver["name"], "axis": "cch_wiring",
            "severity": "INFO",
            "kind": "WIRING_MATRIX",
            "wired": wired, "missing": missing,
            "shared_module_present": shared_components_present,
            "remediation": (
                f"To wire missing CC-H components, import from "
                f"`src/wm/_shared/headline_components.py` "
                f"(present={shared_components_present})"),
        })
    return findings


# ===========================================================================
# Axis 3: Trainer invariants (CLAUDE.md Code Change Verification #11)
# ===========================================================================

def audit_trainer_invariants(versions: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for ver in versions:
        if not ver["trainer"]:
            continue
        body = _read(ver["trainer"])
        for name, pat in TRAINER_INVARIANT_PATTERNS.items():
            if not re.search(pat, body):
                # collision-guard is informational unless trainer has load_latest
                sev = ("MEDIUM" if name == "load_latest_collision"
                         and "load_latest" not in body
                         else "HIGH")
                findings.append({
                    "version": ver["name"], "axis": "trainer_invariants",
                    "severity": sev, "kind": "MISSING_TRAINER_INVARIANT",
                    "missing": name,
                    "pattern": pat,
                    "file": str(ver["trainer"].relative_to(PROJECT_ROOT)),
                    "remediation": (
                        f"Per CLAUDE.md Code Change Verification #11, add "
                        f"`{name}` pattern to {ver['trainer'].name}"),
                })
    return findings


# ===========================================================================
# Axis 4: Iron-clad anti-memo
# ===========================================================================

def audit_anti_memo(versions: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for ver in versions:
        if not (ver["wm_file"] or ver["settings"]):
            continue
        scan = ""
        if ver["wm_file"]:
            scan += _read(ver["wm_file"])
        if ver["settings"]:
            scan += "\n" + _read(ver["settings"])
        present: list[str] = []
        for mech, patterns in ANTI_MEMO_PATTERNS.items():
            if any(re.search(p, scan) for p in patterns):
                present.append(mech)
        if not present:
            findings.append({
                "version": ver["name"], "axis": "anti_memo",
                "severity": "HIGH", "kind": "NO_ANTI_MEMO",
                "file": (str(ver["wm_file"].relative_to(PROJECT_ROOT))
                          if ver["wm_file"] else ""),
                "remediation": ("Add at least one anti-memo mechanism "
                                 "(RSSM/VIB/ATME/JEPA/TimeDisc/XD)"),
            })
        else:
            findings.append({
                "version": ver["name"], "axis": "anti_memo",
                "severity": "INFO", "kind": "ANTI_MEMO_PRESENT",
                "mechanisms": present,
            })
    return findings


# ===========================================================================
# Axis 5: Headline plan coverage
# ===========================================================================

def audit_plan_coverage(versions: list[dict]) -> list[dict]:
    findings: list[dict] = []
    plan_text = _read(HEADLINE_PLAN) if HEADLINE_PLAN.exists() else ""
    if not plan_text:
        findings.append({
            "version": "_global", "axis": "plan_coverage",
            "severity": "HIGH", "kind": "PLAN_DOC_MISSING",
            "file": str(HEADLINE_PLAN.relative_to(PROJECT_ROOT)),
            "remediation": "Restore WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md",
        })
        return findings
    for ver in versions:
        # Skip pure stubs (no settings)
        if not ver["settings"]:
            continue
        name = ver["name"]
        # Per CLAUDE.md, every active version needs a Headline plan.
        # Look for the version name as a section heading (## N. VN -- ...)
        # or as a bullet in the plan.
        pretty = name.replace("v", "V").replace("_", ".")
        # Match V1, V1.0, V1.1, V21, etc.
        plan_present = bool(
            re.search(rf"##\s+\d+\.\s+{re.escape(pretty)}\b", plan_text,
                       re.IGNORECASE)
            or re.search(rf"##\s+\d+\.\s+{re.escape(name.upper())}\b",
                          plan_text)
        )
        if not plan_present:
            findings.append({
                "version": name, "axis": "plan_coverage",
                "severity": "MEDIUM", "kind": "NO_HEADLINE_PLAN",
                "remediation": (
                    f"Add a per-{pretty} section to "
                    f"WM_HEADLINE_UPGRADE_PLAN with H1..HN upgrade ladder "
                    f"(per CLAUDE.md D10 rule)"),
            })
    return findings


# ===========================================================================
# Axis 6: Ckpt vs settings staleness
# ===========================================================================

def audit_ckpt_staleness(versions: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for ver in versions:
        if not ver["settings"]:
            continue
        # Look in models/ for ckpts matching the version name pattern
        s_mtime = ver["settings"].stat().st_mtime
        # Common patterns: models/<vN>/, models/<vN>_*, models/wm_<vN>_*
        for pat in (f"{ver['name']}", f"{ver['name']}_*", f"wm_{ver['name']}*"):
            for p in MODELS_DIR.glob(pat):
                if not p.is_dir():
                    continue
                ckpts = list(p.glob("*.pt")) + list(p.glob("*.ckpt"))
                for c in ckpts:
                    c_mtime = c.stat().st_mtime
                    if c_mtime < s_mtime - 3600:  # ckpt > 1h older than settings
                        age_days = (s_mtime - c_mtime) / 86400
                        findings.append({
                            "version": ver["name"], "axis": "ckpt_staleness",
                            "severity": "MEDIUM" if age_days < 14 else "HIGH",
                            "kind": "STALE_CKPT_VS_SETTINGS",
                            "ckpt": str(c.relative_to(PROJECT_ROOT)),
                            "ckpt_age_days_behind_settings":
                                round(age_days, 1),
                            "remediation": (
                                f"settings.py drifted ~{age_days:.0f}d since "
                                f"ckpt; verify load_state_dict(strict=False) "
                                f"is in trainer, or retrain"),
                        })
    return findings


# ===========================================================================
# Axis 7: Fix log presence
# ===========================================================================

def audit_fix_log_presence(versions: list[dict]) -> list[dict]:
    findings: list[dict] = []
    if not FIX_LOG_DIR.exists():
        return [{
            "version": "_global", "axis": "fix_logs",
            "severity": "MEDIUM", "kind": "FIX_LOG_DIR_MISSING",
            "remediation": "Create memory/fix_logs/ + INDEX.md",
        }]
    for ver in versions:
        # Settings-bearing version should have a fix log
        if not ver["settings"]:
            continue
        # Fix logs use vN_M.md naming (v1_0.md, v1_1.md, v22.md...)
        candidates = [
            FIX_LOG_DIR / f"{ver['name']}.md",
            FIX_LOG_DIR / f"{ver['name'].replace('_', '.')}.md",
        ]
        if not any(c.exists() for c in candidates):
            findings.append({
                "version": ver["name"], "axis": "fix_logs",
                "severity": "LOW", "kind": "NO_FIX_LOG",
                "remediation": (
                    f"Create memory/fix_logs/{ver['name']}.md once a bug is "
                    f"fixed (per CLAUDE.md Code Change Verification #9)"),
            })
    return findings


# ===========================================================================
# Report rendering
# ===========================================================================

def render_md(findings: list[dict], versions: list[dict]) -> str:
    today = dt.date.today().isoformat()
    sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    by_sev: dict[str, list[dict]] = defaultdict(list)
    by_axis: dict[str, int] = defaultdict(int)
    for f in findings:
        by_sev[f.get("severity", "INFO")].append(f)
        by_axis[f.get("axis", "?")] += 1
    out: list[str] = []
    out.append(f"# WM-Layer Audit -- {today}\n")
    out.append(f"Auditor: `src/audit/wm_audit_crawler.py`")
    out.append(f"Companion: `runs/audit/wm_audit_{today}.json`\n")
    out.append("## Summary\n")
    out.append(f"- **Versions audited**: {len(versions)} "
                 f"({sum(1 for v in versions if v['settings'])} with settings.py)")
    out.append("- **Findings by severity**:")
    for sev in ("HIGH", "MEDIUM", "LOW", "INFO"):
        out.append(f"  - {sev}: {len(by_sev[sev])}")
    out.append("- **Findings by axis**:")
    for axis, n in sorted(by_axis.items()):
        out.append(f"  - {axis}: {n}")
    out.append("")
    # CC-H wiring matrix table
    wiring = [f for f in findings if f.get("axis") == "cch_wiring"]
    if wiring:
        all_comps = list(CC_COMPONENT_PATTERNS.keys())
        out.append("## CC-H Headline Component Wiring Matrix\n")
        out.append("| Version | " + " | ".join(all_comps) + " |")
        out.append("|" + "---|" * (len(all_comps) + 1))
        for f in sorted(wiring, key=lambda x: x["version"]):
            row = [f["version"]]
            for c in all_comps:
                row.append("YES" if c in f["wired"] else ".")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    # Non-wiring findings
    out.append("## Findings (HIGH / MEDIUM / LOW)\n")
    for sev in ("HIGH", "MEDIUM", "LOW"):
        bucket = [f for f in by_sev[sev] if f["axis"] != "cch_wiring"]
        if not bucket:
            continue
        out.append(f"### {sev} ({len(bucket)})\n")
        for f in sorted(bucket,
                          key=lambda x: (x.get("version", ""),
                                           x.get("axis", ""))):
            ver = f.get("version", "?")
            axis = f.get("axis", "?")
            kind = f.get("kind", "?")
            file = f.get("file", "")
            file_s = f" `{file}`" if file else ""
            out.append(f"- **{ver}** [{axis}] `{kind}`{file_s}")
            for key in ("constant", "expected", "found", "missing", "ckpt",
                          "ckpt_age_days_behind_settings"):
                if key in f:
                    out.append(f"    - {key}: `{f[key]}`")
            if "remediation" in f:
                out.append(f"    - Remediation: {f['remediation']}")
        out.append("")
    return "\n".join(out)


# ===========================================================================
# Main
# ===========================================================================

def audit_component_drift(versions: list[dict]) -> list[dict]:
    """Consume the components_registry drift detector.

    Defined as a separate axis so registry-level findings (which span the
    whole tree, not per-version) appear under their own column in the
    summary report.
    """
    findings: list[dict] = []
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "wm" / "_shared"))
        from components_registry import audit_all_components
        registry_findings = audit_all_components()
        for f in registry_findings:
            # Tag axis + try to attribute to a version when possible
            vf = f.get("version_file", "")
            ver = "_global"
            for v in versions:
                if v["name"] in vf:
                    ver = v["name"]
                    break
            findings.append({
                **f, "axis": "components_drift",
                "version": ver,
            })
    except ImportError:
        findings.append({
            "version": "_global", "axis": "components_drift",
            "severity": "LOW", "kind": "REGISTRY_IMPORT_FAIL",
            "remediation": "Verify src/wm/_shared/components_registry.py importable",
        })
    return findings


AXES = {
    "invariants":     audit_invariants_drift,
    "wiring":         audit_cch_wiring,
    "trainer":        audit_trainer_invariants,
    "anti_memo":      audit_anti_memo,
    "plan_coverage":  audit_plan_coverage,
    "ckpt":           audit_ckpt_staleness,
    "fix_logs":       audit_fix_log_presence,
    "components":     audit_component_drift,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", help="Audit only this version (e.g. v1_1)")
    ap.add_argument("--axes", help="Comma-separated axes to run "
                                      "(default: all)")
    args = ap.parse_args()
    versions = discover_versions()
    if args.version:
        versions = [v for v in versions if v["name"] == args.version]
        if not versions:
            print(f"[wm_audit] no version named {args.version}", flush=True)
            return 2
    axes_to_run = (args.axes.split(",") if args.axes
                     else list(AXES.keys()))
    findings: list[dict] = []
    for axis_name in axes_to_run:
        if axis_name not in AXES:
            print(f"[wm_audit] unknown axis: {axis_name}", flush=True)
            return 2
        sub = AXES[axis_name](versions)
        findings.extend(sub)
    today = dt.date.today().isoformat()
    md_path = OUT_DIR / f"wm_audit_{today}.md"
    json_path = OUT_DIR / f"wm_audit_{today}.json"
    md_path.write_text(render_md(findings, versions), encoding="utf-8")
    json_path.write_text(
        json.dumps({"date": today, "n_versions": len(versions),
                     "findings": findings}, indent=2),
        encoding="utf-8")
    crit = sum(1 for f in findings if f.get("severity") == "HIGH")
    print(f"[wm_audit] {len(versions)} versions audited, "
          f"{len(findings)} findings ({crit} HIGH). "
          f"Report: {md_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
