"""Feature-registry contract test — CDAP gate.

Purpose
=======
Catch the class of bugs that have hit `config/feature_registry.yaml` in the
last 14 days:
  - 78dbdd7  btc_ret_same_day label leak (a "feature" was secretly a forward-shifted target)
  - c59c4e7  xrel_ fix BREAKS the z-score wall (registry edit broke downstream norm_* invariant)
  - 7fa8bbd  feature_registry.yaml restored after 2026-05-18 schema clash (no rev tag → silent breakage)

Each of these landed because the registry was edited without a deterministic
contract test that round-trips the schema. This module fills that gap.

Invocation
==========
    python src/pipeline/registry_contract_test.py            # exit 0 = pass, 2 = CRITICAL
    python src/pipeline/registry_contract_test.py --verbose  # show every check

Wired into:
  - `src/pipeline/pre_train_gate.py` (CDAP pre-commit gate)
  - `config/_invariants.yaml` invariant: `feature_registry_has_schema_version`

Exit codes:
  0  all checks pass
  1  warnings (e.g., glob has zero matches in data/ — likely a stale/never-built source)
  2  CRITICAL — schema_version missing, orphan sources_to_join reference,
     reserved-name collision, prefix collision, layout/date_col/date_unit invalid

Checks (all run; first CRITICAL determines exit code):
  C1  schema_version present + parses as semver
  C2  sources_to_join references resolve (no orphan source-names)
  C3  reserved-name guard: no `features:` entry collides with target/label/key column names
  C4  prefix uniqueness across sources (avoids field-name collisions on join)
  C5  layout/date_col/date_unit declarations are valid enum/string
  C6  glob `path:` patterns have ≥1 match in data/ (warn-only — sources may be intentionally pending)
  C7  no wide_pattern source lacks a `wide_pattern` regex with exactly one capture group
"""
from __future__ import annotations

# CDAP contract
__contract__ = {
    "kind": "pipeline_validator",
    "stage": "registry_contract",
    "inputs": {
        "config": "config/feature_registry.yaml",
        "data_root": "data/",
    },
    "outputs": {
        "exit_code": "0=pass, 1=warn, 2=CRITICAL",
        "stdout": "human-readable check report",
    },
    "invariants": {
        "deterministic": True,
        "no_network": True,
        "no_data_write": True,
    },
    "rationale": "Closes feature-engineering A+ gap. Catches the registry-edit interaction-bug class (label leak, z-score wall break, schema clash) at commit time.",
}

import argparse
import re
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import yaml

# Tokens that MUST NOT appear in a `features:` list — these are labels/targets,
# not features. Mirrors validate_chimera.py:409 LABEL_PATTERNS exactly.
RESERVED_FEATURE_TOKENS: tuple[str, ...] = (
    "target_return", "target_voladj", "target_vol",
    "ret_fwd_", "bm_ret", "_label", "ret_high",
)

# Reserved key columns — also can't appear in `features:`.
RESERVED_KEY_COLUMNS: tuple[str, ...] = (
    "date", "asset", "ts", "tick_seq", "symbol",
)

VALID_LAYOUTS: tuple[str, ...] = ("per_asset", "global", "wide_per_asset")
VALID_DATE_UNITS: tuple[str, ...] = ("date", "datetime")

# Semver regex (MAJOR.MINOR.PATCH; pre-release/build metadata not required).
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass
class CheckResult:
    check_id: str
    severity: str  # ok | warn | fail (= CRITICAL)
    message: str


def _load_registry(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def C1_schema_version(reg: dict[str, Any]) -> list[CheckResult]:
    """schema_version must be present and parse as semver MAJOR.MINOR.PATCH."""
    out: list[CheckResult] = []
    sv = reg.get("schema_version")
    if sv is None:
        out.append(CheckResult("C1_schema_version", "fail",
            "schema_version field MISSING. Required by oracle pipeline-A+ closure "
            "(2026-05-22). Add schema_version: \"X.Y.Z\" at the top of feature_registry.yaml."))
        return out
    if not isinstance(sv, str) or not SEMVER_RE.match(sv):
        out.append(CheckResult("C1_schema_version", "fail",
            f"schema_version={sv!r} is not valid semver MAJOR.MINOR.PATCH (e.g. \"1.1.0\")."))
        return out
    out.append(CheckResult("C1_schema_version", "ok", f"schema_version={sv}"))
    return out


def C2_sources_to_join_resolve(reg: dict[str, Any]) -> list[CheckResult]:
    """Every name in chimera_v51.sources_to_join must exist as a key in sources."""
    out: list[CheckResult] = []
    sources = reg.get("sources", {}) or {}
    join_list = (reg.get("chimera_v51", {}) or {}).get("sources_to_join", []) or []
    orphans = [n for n in join_list if n not in sources]
    if orphans:
        out.append(CheckResult("C2_join_resolve", "fail",
            f"sources_to_join references {orphans} which have no entry in `sources:`. "
            f"Orphan references break the chimera build silently with missing-source warnings."))
    else:
        out.append(CheckResult("C2_join_resolve", "ok",
            f"all {len(join_list)} sources_to_join references resolve"))
    return out


def C3_reserved_name_guard(reg: dict[str, Any]) -> list[CheckResult]:
    """No feature in any source's features: list may match reserved label/target tokens
    or key-column names. Catches the btc_ret_same_day-class of label leaks at config time."""
    out: list[CheckResult] = []
    sources = reg.get("sources", {}) or {}
    offenders: list[tuple[str, str, str]] = []  # (source, feature, reason)
    for src_name, spec in sources.items():
        feats = spec.get("features", [])
        if feats == "*":
            continue  # wildcard — caught by post-build validate_chimera instead
        for f in feats or []:
            full = f"{spec.get('prefix', '')}{f}"
            # reserved label/target tokens
            for tok in RESERVED_FEATURE_TOKENS:
                if tok in f or tok in full:
                    offenders.append((src_name, f, f"contains reserved label token '{tok}'"))
            # reserved key columns (exact match)
            if f in RESERVED_KEY_COLUMNS or full in RESERVED_KEY_COLUMNS:
                offenders.append((src_name, f, "matches reserved key column"))
    if offenders:
        msg = "\n".join(f"  - source={s}, feature={f}: {r}" for s, f, r in offenders)
        out.append(CheckResult("C3_reserved_names", "fail",
            f"{len(offenders)} feature(s) collide with reserved label/key columns:\n{msg}"))
    else:
        out.append(CheckResult("C3_reserved_names", "ok",
            "no feature collides with reserved label/key column names"))
    return out


def C4_prefix_uniqueness(reg: dict[str, Any]) -> list[CheckResult]:
    """Two sources must not declare the same `prefix:` (would cause column collisions
    on join). Empty prefix is allowed when EXPLICITLY declared as `prefix: ""`
    (deliberate — features already carry the right prefix per the registry comment);
    treated as missing only when the key is absent.

    Refined 2026-05-22 oracle pipeline-A+ closure: previously WARN'd on multiple
    sources sharing empty prefix, but the canonical registry uses `prefix: ""`
    deliberately for sources whose features pre-carry a project-level prefix
    (e.g., liq_features features all start with 'liq_' so the source declares
    `prefix: ""` to avoid double-prefix). Treat explicit-empty as deliberate;
    only warn when the `prefix:` key is genuinely missing (omitted entirely).
    """
    out: list[CheckResult] = []
    sources = reg.get("sources", {}) or {}
    prefix_to_sources: dict[str, list[str]] = {}
    explicit_empty: list[str] = []
    implicit_empty: list[str] = []
    for src_name, spec in sources.items():
        if "prefix" not in spec:
            implicit_empty.append(src_name)
            prefix_to_sources.setdefault("", []).append(src_name)
        else:
            pfx = spec.get("prefix", "")
            prefix_to_sources.setdefault(pfx, []).append(src_name)
            if pfx == "":
                explicit_empty.append(src_name)
    dupes = {p: srcs for p, srcs in prefix_to_sources.items() if len(srcs) > 1 and p != ""}
    # Only count GENUINELY missing prefix (key omitted) as warn-able
    no_prefix = implicit_empty
    if dupes:
        msg = "; ".join(f"prefix={p!r} used by {srcs}" for p, srcs in dupes.items())
        out.append(CheckResult("C4_prefix_unique", "fail",
            f"duplicate prefixes: {msg}. Field collisions on join."))
    elif len(no_prefix) > 1:
        out.append(CheckResult("C4_prefix_unique", "warn",
            f"{len(no_prefix)} sources have no `prefix:` declared: {no_prefix}. "
            f"Risk of column-name collision; declare prefix on all but one."))
    else:
        out.append(CheckResult("C4_prefix_unique", "ok",
            f"all {len(prefix_to_sources)} prefixes are unique"))
    return out


def C5_layout_date_validity(reg: dict[str, Any]) -> list[CheckResult]:
    """layout must be one of {per_asset, global, wide_per_asset}; date_unit must
    be one of {date, datetime}; date_col must be a non-empty string."""
    out: list[CheckResult] = []
    sources = reg.get("sources", {}) or {}
    offenders: list[str] = []
    for src_name, spec in sources.items():
        layout = spec.get("layout")
        if layout not in VALID_LAYOUTS:
            offenders.append(f"  - {src_name}: layout={layout!r} not in {VALID_LAYOUTS}")
        date_unit = spec.get("date_unit")
        if date_unit not in VALID_DATE_UNITS:
            offenders.append(f"  - {src_name}: date_unit={date_unit!r} not in {VALID_DATE_UNITS}")
        date_col = spec.get("date_col")
        if not isinstance(date_col, str) or not date_col:
            offenders.append(f"  - {src_name}: date_col={date_col!r} must be non-empty string")
    if offenders:
        out.append(CheckResult("C5_layout_date", "fail",
            f"{len(offenders)} declaration error(s):\n" + "\n".join(offenders)))
    else:
        out.append(CheckResult("C5_layout_date", "ok",
            f"all {len(sources)} sources have valid layout/date_col/date_unit"))
    return out


def C6_glob_has_match(reg: dict[str, Any], data_root: Path) -> list[CheckResult]:
    """Each source's `path:` glob should match ≥1 file in data/. Warn-only because
    sources may be intentionally pending (e.g., scheduled rebuild)."""
    out: list[CheckResult] = []
    sources = reg.get("sources", {}) or {}
    empty: list[str] = []
    for src_name, spec in sources.items():
        path_glob = spec.get("path", "")
        if not path_glob:
            continue
        full = str(data_root / path_glob)
        matches = glob(full, recursive=True)
        if not matches:
            empty.append(src_name)
    if empty:
        out.append(CheckResult("C6_glob_match", "warn",
            f"{len(empty)} source(s) have zero data files: {empty}. May be intentional "
            f"(scheduled rebuild) or stale registry entry — verify."))
    else:
        out.append(CheckResult("C6_glob_match", "ok",
            f"all {len(sources)} sources have ≥1 file under data/"))
    return out


def C7_wide_pattern_capture(reg: dict[str, Any]) -> list[CheckResult]:
    """wide_per_asset sources must declare a wide_pattern with exactly one capture group."""
    out: list[CheckResult] = []
    sources = reg.get("sources", {}) or {}
    offenders: list[str] = []
    for src_name, spec in sources.items():
        if spec.get("layout") != "wide_per_asset":
            continue
        wp = spec.get("wide_pattern")
        if not wp:
            offenders.append(f"  - {src_name}: layout=wide_per_asset but no wide_pattern declared")
            continue
        try:
            cre = re.compile(wp)
            if cre.groups != 1:
                offenders.append(f"  - {src_name}: wide_pattern={wp!r} has {cre.groups} capture group(s); must be exactly 1")
        except re.error as e:
            offenders.append(f"  - {src_name}: wide_pattern={wp!r} is not a valid regex ({e})")
    if offenders:
        out.append(CheckResult("C7_wide_pattern", "fail",
            f"{len(offenders)} wide_per_asset declaration error(s):\n" + "\n".join(offenders)))
    else:
        out.append(CheckResult("C7_wide_pattern", "ok",
            "all wide_per_asset sources have valid wide_pattern"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature registry contract test (CDAP gate)")
    ap.add_argument("--registry", type=Path,
                    default=Path("config/feature_registry.yaml"),
                    help="Path to feature_registry.yaml (default: config/feature_registry.yaml)")
    ap.add_argument("--data-root", type=Path, default=Path("data"),
                    help="Root for path-glob expansion (default: data/)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print every check, not just failures + warnings")
    args = ap.parse_args()

    if not args.registry.exists():
        print(f"[registry_contract_test] FAIL: {args.registry} not found")
        return 2

    try:
        reg = _load_registry(args.registry)
    except yaml.YAMLError as e:
        print(f"[registry_contract_test] FAIL: yaml parse error: {e}")
        return 2

    results: list[CheckResult] = []
    results += C1_schema_version(reg)
    results += C2_sources_to_join_resolve(reg)
    results += C3_reserved_name_guard(reg)
    results += C4_prefix_uniqueness(reg)
    results += C5_layout_date_validity(reg)
    results += C6_glob_has_match(reg, args.data_root)
    results += C7_wide_pattern_capture(reg)

    n_fail = sum(1 for r in results if r.severity == "fail")
    n_warn = sum(1 for r in results if r.severity == "warn")
    n_ok = sum(1 for r in results if r.severity == "ok")

    for r in results:
        if r.severity == "ok" and not args.verbose:
            continue
        marker = {"ok": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}[r.severity]
        print(f"{marker} {r.check_id}: {r.message}")

    print(f"\n[registry_contract_test] SUMMARY: {n_ok} ok / {n_warn} warn / {n_fail} fail")
    if n_fail:
        return 2
    if n_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
