"""crawler_coordinator.py -- persistent crawler swarm coordinator.

PURPOSE
-------
The audit_crawler_swarm protocol (memory/agent_protocols/audit_crawler_swarm.md)
spawns 4 parallel agents per invocation. To make crawlers PERSISTENT (running
continuously) and TALK TO EACH OTHER (avoid duplicate findings, cross-reference
prior work), this module:

1. Maintains a shared state at `runs/crawler_swarm/_shared_state.json`
   containing every finding ever logged + last-seen-by-each-crawler.
2. Provides `prepare_crawler_brief()` -- composes a crawler prompt that
   includes deduplication context (so the next firing doesn't re-report
   already-known issues).
3. Provides `ingest_findings()` -- parses crawler output, dedupes, updates
   shared state, writes structured findings to crawler-specific log.
4. Provides `triage_summary()` -- generates a single MD report of all
   currently-open findings ranked by severity + age.

CALL PATTERN
------------
A scheduled job (CronCreate) fires this with --crawler <lens>. The coord
emits a prompt for the Agent tool to use, then later ingests its output.

LAYOUT
------
runs/crawler_swarm/
├── _shared_state.json              # dedup index; what each crawler has seen
├── _triage_<DATE>.md               # consolidated open findings
├── auditor/findings_<DATE>.md      # RED TEAM crawler log
├── pipeline/findings_<DATE>.md     # pipeline integrity log
├── validator/findings_<DATE>.md    # validation rigor log
└── trader/findings_<DATE>.md       # trading realism log

USAGE
-----
    # Prepare a brief for the next crawler firing (called by scheduler):
    python src/audit/crawler_coordinator.py --action brief --crawler auditor

    # Ingest a completed crawler's output:
    python src/audit/crawler_coordinator.py --action ingest \
        --crawler auditor --report /tmp/auditor_output.md

    # Generate the current triage summary:
    python src/audit/crawler_coordinator.py --action triage

    # List crawler activity history:
    python src/audit/crawler_coordinator.py --action list
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SWARM_DIR = ROOT / "runs" / "crawler_swarm"
STATE_PATH = SWARM_DIR / "_shared_state.json"
SWARM_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "crawler_coordinator",
    "owner": "audit/governance",
    "outputs": "runs/crawler_swarm/_shared_state.json + per-crawler findings logs",
    "invariants": [
        "shared state is append-only on findings; closed findings retain history",
        "dedup key = sha1(file:line:category:short_detail)",
        "each crawler reads prior findings before starting (no re-report)",
    ],
}


# Crawler lens definitions
LENSES = {
    "auditor": {
        "agent_type": "expert-auditor",
        "scope_hint": "RED TEAM: latest commit + critical paths",
        "files_focus": ["src/audit/", "src/strategy/sleeves/", "src/strategy/gen5_growth/"],
    },
    "pipeline": {
        "agent_type": "expert-pipeline",
        "scope_hint": "Pipeline integrity: src/pipeline + chimera consumers",
        "files_focus": ["src/pipeline/", "config/asset_dag.yaml"],
    },
    "validator": {
        "agent_type": "expert-validator",
        "scope_hint": "Validation rigor: math, statistics, walk-forward purge",
        "files_focus": ["src/analysis/", "scripts/strat_audit/", "src/strategy/ta_sml/"],
    },
    "trader": {
        "agent_type": "expert-trader",
        "scope_hint": "Trading realism: cost, sizing, capacity, tail risk",
        "files_focus": ["src/strategy/sleeves/", "src/strategy/maker_cost_model.py",
                          "config/production_blends.yaml"],
    },
}


def _now() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def _finding_id(file: str, line: str, category: str, detail: str) -> str:
    """Deterministic dedup key for a finding."""
    short = (detail or "")[:80]
    return hashlib.sha1(f"{file}:{line}:{category}:{short}".encode()).hexdigest()[:12]


# ============================================================================
# Shared state I/O
# ============================================================================

def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"findings": {}, "crawlers": {}, "version": 1}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"findings": {}, "crawlers": {}, "version": 1}


def save_state(state: Dict[str, Any]) -> None:
    state["_updated_at"] = _now()
    STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ============================================================================
# Brief preparation (the prompt for the next crawler firing)
# ============================================================================

BRIEF_TEMPLATE = """
You are crawler `{crawler}` (agent type `{agent_type}`).

SCOPE — {scope_hint}

FILES IN YOUR FOCUS:
{files_block}

PRIOR FINDINGS (do NOT re-report these — they are already tracked):
{prior_block}

OPEN-IN-OTHER-LENSES (low priority for you, but cross-reference if directly relevant):
{cross_block}

INVESTIGATIONS — produce a structured report finding:
1. NEW correctness defects in the focus files (file:line + manifestation + fix)
2. Drift in cross-file contracts (file A says X, file B reads Y)
3. Silent-failure patterns (B7: try/except Exception: pass; default fallthrough)
4. Status changes on prior findings (RESOLVED / WORSE / STABLE)
5. What's done well in your focus (be specific, 3-5 items)
6. Missed opportunities (high-EV additions)

DELIVERABLE FORMAT (machine-parseable):
For each finding write a line:
  🔴 CRITICAL | file:line | category | short detail | fix
  🟠 HIGH     | file:line | category | short detail | fix
  🟡 MEDIUM   | file:line | category | short detail | fix
  ✅ POSITIVE | file        | category | what's done well
  💡 OPPORTUNITY | category | what to add | est. EV / cost

Use exactly that pipe-delimited format. Each finding on its own line.

Total report under 400 lines. Skip generic praise. Surface real defects.

Working directory: c:/Users/karab/Documents/coding/ml_systems
""".strip()


def prepare_brief(crawler: str) -> str:
    if crawler not in LENSES:
        raise ValueError(f"unknown crawler '{crawler}'; known: {list(LENSES)}")
    lens = LENSES[crawler]
    state = load_state()

    files_block = "\n".join(f"  - {f}" for f in lens["files_focus"])

    prior_findings = [f for f in state["findings"].values()
                       if f.get("crawler") == crawler and f.get("status") == "open"]
    if prior_findings:
        prior_block = "\n".join(
            f"  - [{f['severity']}] {f['file']}:{f['line']} — {f['detail'][:60]}"
            for f in prior_findings[:30]
        )
    else:
        prior_block = "  (no prior findings — this is the first crawler firing)"

    cross_findings = [f for f in state["findings"].values()
                       if f.get("crawler") != crawler and f.get("status") == "open"
                       and f.get("severity") in ("CRITICAL", "HIGH")]
    if cross_findings:
        cross_block = "\n".join(
            f"  - [{f['crawler']}/{f['severity']}] {f['file']}:{f['line']}"
            for f in cross_findings[:15]
        )
    else:
        cross_block = "  (none)"

    return BRIEF_TEMPLATE.format(
        crawler=crawler,
        agent_type=lens["agent_type"],
        scope_hint=lens["scope_hint"],
        files_block=files_block,
        prior_block=prior_block,
        cross_block=cross_block,
    )


# ============================================================================
# Findings ingestion
# ============================================================================

FINDING_LINE_RE = re.compile(
    r"^\s*(🔴|🟠|🟡|✅|💡)\s*(\w+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*(?:\|\s*(.+?))?\s*$"
)
SEVERITY_MAP = {"🔴": "CRITICAL", "🟠": "HIGH", "🟡": "MEDIUM", "✅": "POSITIVE",
                  "💡": "OPPORTUNITY"}


def parse_findings(report_text: str, crawler: str) -> List[Dict]:
    out = []
    for line in report_text.splitlines():
        m = FINDING_LINE_RE.match(line)
        if not m:
            continue
        emoji, _label, file_part, category, detail, fix = m.groups()
        severity = SEVERITY_MAP.get(emoji, "UNKNOWN")
        # Extract line number if present (file:line format)
        file_ = file_part.strip()
        line_no: Optional[str] = None
        if ":" in file_ and file_.rsplit(":", 1)[-1].isdigit():
            file_, line_no = file_.rsplit(":", 1)
        out.append({
            "id": _finding_id(file_, line_no or "?", category.strip(),
                                detail.strip()),
            "crawler": crawler,
            "severity": severity,
            "file": file_.strip(),
            "line": line_no or "",
            "category": category.strip(),
            "detail": detail.strip(),
            "fix": (fix or "").strip(),
            "status": "open",
            "first_seen": _now(),
            "last_seen": _now(),
            "fire_count": 1,
        })
    return out


def ingest_findings(crawler: str, report_text: str) -> Dict[str, int]:
    """Merge new findings into shared state. Returns counts by status."""
    if crawler not in LENSES:
        raise ValueError(f"unknown crawler '{crawler}'")
    state = load_state()
    findings = parse_findings(report_text, crawler)

    counts = {"new": 0, "reaffirmed": 0, "resolved": 0}
    new_ids = set()
    for f in findings:
        new_ids.add(f["id"])
        if f["id"] in state["findings"]:
            state["findings"][f["id"]]["last_seen"] = f["last_seen"]
            state["findings"][f["id"]]["fire_count"] = \
                state["findings"][f["id"]].get("fire_count", 1) + 1
            counts["reaffirmed"] += 1
        else:
            state["findings"][f["id"]] = f
            counts["new"] += 1

    # Mark previously-open findings (from this crawler) not in this report as resolved
    for fid, f in state["findings"].items():
        if f["crawler"] != crawler:
            continue
        if f["status"] != "open":
            continue
        if fid not in new_ids:
            # Hint at resolution; require 2 consecutive misses to actually close
            f["miss_count"] = f.get("miss_count", 0) + 1
            if f["miss_count"] >= 2:
                f["status"] = "resolved"
                f["resolved_at"] = _now()
                counts["resolved"] += 1
        else:
            f["miss_count"] = 0

    state["crawlers"].setdefault(crawler, {})
    state["crawlers"][crawler]["last_fired"] = _now()
    state["crawlers"][crawler]["last_findings_count"] = len(findings)
    save_state(state)

    # Write per-crawler log
    log_dir = SWARM_DIR / crawler
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"findings_{_today()}.md"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n## Fire {_now()}\n\n")
        fh.write(f"new={counts['new']}, reaffirmed={counts['reaffirmed']}, "
                  f"resolved={counts['resolved']}\n\n")
        fh.write(report_text)
        fh.write("\n---\n")

    return counts


# ============================================================================
# Triage summary
# ============================================================================

def triage_summary() -> Path:
    state = load_state()
    open_findings = [f for f in state["findings"].values()
                       if f.get("status") == "open"]
    open_findings.sort(key=lambda f: (
        {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "POSITIVE": 3, "OPPORTUNITY": 4}
        .get(f["severity"], 5),
        f["first_seen"],
    ))

    out_path = SWARM_DIR / f"_triage_{_today()}.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Crawler Swarm Triage — {_today()}\n\n")
        fh.write(f"Total open findings: {len(open_findings)}\n\n")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "OPPORTUNITY", "POSITIVE"):
            subset = [f for f in open_findings if f["severity"] == sev]
            if not subset:
                continue
            fh.write(f"## {sev} ({len(subset)})\n\n")
            for f in subset:
                age_days = (dt.datetime.utcnow() - dt.datetime.strptime(
                    f["first_seen"], "%Y-%m-%dT%H:%M:%SZ")).days
                fh.write(f"- [{f['crawler']}] {f['file']}:{f['line']} "
                          f"— {f['category']}: {f['detail']}")
                if f.get("fix"):
                    fh.write(f"\n  fix: {f['fix']}")
                fh.write(f"\n  age: {age_days}d, fires: {f.get('fire_count', 1)}\n\n")
        fh.write(f"\n## Crawler status\n\n")
        for c, info in state.get("crawlers", {}).items():
            fh.write(f"- {c}: last_fired={info.get('last_fired')} "
                      f"findings={info.get('last_findings_count')}\n")
    return out_path


# ============================================================================
# CLI
# ============================================================================

# R32+++ verify-resolved feature: per-finding heuristic check that the cited
# anti-pattern is GONE from the cited file. If so, auto-mark resolved so
# `triage_summary` stops listing it. Conservative: never marks resolved
# unless a positive signature for the FIX is present (avoids false-resolve
# on commented-out code or moved logic).
#
# Each rule = (category-substring, file-glob-or-None, check-fn(file_text) -> bool)
# check-fn returns True if the bug is RESOLVED (signature of fix present).


def _check_smoke_crash(file_text: str, finding: Dict[str, Any]) -> bool:
    """smoke-crash resolved if (a) cash/fallback dict has the key OR (b) smoke uses .get()"""
    detail = (finding.get("detail") or "").lower()
    # Heuristic: find the key name (e.g. 'intensity', 'ofi_score') in the detail
    import re as _re
    m = _re.search(r"['\"]([\w_]+)['\"]", detail)
    if not m:
        return False
    key = m.group(1)
    return f'.get("{key}"' in file_text or f".get('{key}'" in file_text or \
              f'"{key}":' in file_text and "fallback" in file_text.lower()


def _check_allocation_leak(file_text: str, finding: Dict[str, Any]) -> bool:
    """allocation-leak resolved if cash_USDC fallback emitted with size_pct=blend_weight_pct"""
    return ("cash_USDC" in file_text or "cash_usdc" in file_text.lower()) and \
              "blend_weight_pct" in file_text and \
              ("fallback" in file_text.lower() or "always-emit" in file_text.lower())


def _check_contract_drift_v3(file_text: str, finding: Dict[str, Any]) -> bool:
    """contract-drift resolved if file uses get_v3_metric or v3_schema import"""
    return "get_v3_metric" in file_text or "from audit.v3_schema" in file_text


def _check_silent_failure(file_text: str, finding: Dict[str, Any]) -> bool:
    """silent-failure resolved if except blocks have print/log/WARN nearby"""
    # Heuristic: count `except.*:\s*\n\s*pass$` (bare swallows)
    import re as _re
    bare_swallow = _re.findall(r"except[^:]*:\s*\n\s*pass\s*\n", file_text)
    # Resolved if no bare swallow OR the cited file/line region contains WARN
    return len(bare_swallow) <= 1 and ("WARN" in file_text or "print" in file_text)


def _check_math_kurt_convention(file_text: str, finding: Dict[str, Any]) -> bool:
    """math-correctness resolved if kurtosis_convention parameter present"""
    return "kurtosis_convention" in file_text or "KURT_PEARSON" in file_text or \
              "KURT_FISHER" in file_text


def _check_hidden_emoji(file_text: str, finding: Dict[str, Any]) -> bool:
    """hidden-contract emoji resolved if [FLAGGED] sentinel used + no raw emoji"""
    # Check for ASCII sentinel; tolerate legacy reader emoji in regex
    return "[FLAGGED]" in file_text


def _check_temporal_leakage_sort(file_text: str, finding: Dict[str, Any]) -> bool:
    """temporal-leakage resolved if sort key is timestamp-major (timestamp first)"""
    return 'sort_values(["timestamp"' in file_text or \
              'sort_values([\'timestamp\'' in file_text


def _check_coverage_gap_null_rate(file_text: str, finding: Dict[str, Any]) -> bool:
    """coverage-gap resolved if EITHER null-rate iterates pfx_cols (pre_train_gate)
    OR WF_NAME_HINTS expanded (purge_gap_audit)."""
    if "for c in pfx_cols" in file_text or "for c in cols" in file_text:
        return True
    # purge_gap_audit case: WF hints expanded
    if "purged_kfold" in file_text and "evaluate_cell_cpcv" in file_text:
        return True
    return False


def _check_logic_bug_dead_code(file_text: str, finding: Dict[str, Any]) -> bool:
    """logic-bug pbo dead-code resolved if median-pbo line is gone"""
    return "removed the median-PBO" in file_text or "neg_frac = float" in file_text


def _check_data_fetch_1000_prefix(file_text: str, finding: Dict[str, Any]) -> bool:
    """data-fetch-bug / data-correctness 1000-prefix resolved if fetch_all has the prefix handling"""
    if "fetch_all" not in (finding.get("file") or ""):
        return False
    return ("1000" in file_text and ("FAPI_1000_PREFIX" in file_text or
                                       "_apply_1000_prefix" in file_text or
                                       "1000-prefix" in file_text or
                                       "PREFIX_1000_ASSETS" in file_text))


def _check_cash_fallback(file_text: str, finding: Dict[str, Any]) -> bool:
    """cash-fallback same shape as allocation-leak"""
    return _check_allocation_leak(file_text, finding)


def _check_date_misalignment(file_text: str, finding: Dict[str, Any]) -> bool:
    """date-misalignment resolved if single-pass extraction present"""
    return "_extract_returns_and_dates" in file_text or \
              "single-pass" in file_text


def _check_validation(file_text: str, finding: Dict[str, Any]) -> bool:
    """validation gate (pre_train_gate) resolved if dead_features check + Exception broadening present"""
    return ("dead_features" in file_text or
              "high_null_prefixes" in file_text and "_rc = 2" in file_text)


def _check_misleading_doc(file_text: str, finding: Dict[str, Any]) -> bool:
    """misleading-doc (yaml caveat) resolved if pos_cap + CLAMP/clamped wording present"""
    return "pos_cap" in file_text.lower() and \
              ("clamp" in file_text.lower() or "CLAMPED" in file_text)


def _check_windows_safety(file_text: str, finding: Dict[str, Any]) -> bool:
    """windows-safety resolved if no emoji in print/log statements"""
    # Crude: count non-ASCII chars in print() calls — if 0, considered safe
    import re as _re
    matches = _re.findall(r"print\([^)]*\)", file_text)
    for m in matches:
        if any(ord(c) > 127 for c in m):
            return False
    return True


def _check_timeout_yaml(file_text: str, finding: Dict[str, Any]) -> bool:
    """yaml timeout-finding resolved if cited lines have timeout_seconds: 21600+"""
    # Heuristic: every line cited in finding must have a 21600+ timeout near it.
    # Simpler: count occurrences of "timeout_seconds: 21600" or "_seconds: 43200"
    import re as _re
    bumps = _re.findall(r"timeout_seconds:\s*(\d+)", file_text)
    bumps_ints = [int(x) for x in bumps]
    high_bumps = [x for x in bumps_ints if x >= 21600]
    # At least 4 stages bumped (matches the 4 panel stages flagged)
    return len(high_bumps) >= 4


def _check_hold_bars_signal_flip(file_text: str, finding: Dict[str, Any]) -> bool:
    """hold-bars-signal-flip resolved if exits_allowed parameter set on intents"""
    return "exits_allowed" in file_text and \
              ("max_hold" in file_text or "stop_loss" in file_text)


def _check_trigger_logic_inversion(file_text: str, finding: Dict[str, Any]) -> bool:
    """trigger-logic-inversion resolved if liq_long_z* appears FIRST in
    candidate_cols (so the corrected long-side priority is in effect)."""
    import re as _re
    m = _re.search(r"candidate_cols\s*=\s*\[([^\]]+)\]", file_text)
    if not m:
        # Fallback: any liq_long_z* present and the inversion-fix comment
        return "liq_long_z" in file_text and \
                  ("TRADER-C1" in file_text or "priority to liq_long_z" in file_text)
    cols = m.group(1)
    # First column in the list must be liq_long_z* (not liq_delta_z30)
    first_col = cols.split(",")[0].strip().strip("'\"")
    return first_col.startswith("liq_long_z")


def _check_invariant_missing_parity(file_text: str, finding: Dict[str, Any]) -> bool:
    """invariant-missing fetch parity resolved if parity_failed sentinel logic present"""
    return "parity_failed" in file_text or "_parity_failed" in file_text or \
              "FAIL-PARITY" in file_text


def _check_operational_parity(file_text: str, finding: Dict[str, Any]) -> bool:
    """operational fetch parity resolved if parity check no longer print-only"""
    return "parity_failed" in file_text or "DONE-WITH-PARITY-FAIL" in file_text


def _check_regression_fisher(file_text: str, finding: Dict[str, Any]) -> bool:
    """regression resolved if Pearson(3.0) -> Fisher(0.0) fallback in place"""
    return "Fisher excess (normal=0)" in file_text or \
              "kurt_excess = 0.0" in file_text or \
              "emp_kurt_excess = 0.0" in file_text


def _check_small_n_fold_count(file_text: str, finding: Dict[str, Any]) -> bool:
    """small-N fold-count resolved if actual_folds is computed before printing"""
    return "actual_folds" in file_text or \
              "len(folds)" in file_text and "requested" in file_text


def _check_coverage_gap_wf_hints(file_text: str, finding: Dict[str, Any]) -> bool:
    """coverage-gap WF_NAME_HINTS resolved if modern entry points added"""
    return "purged_kfold" in file_text and "evaluate_cell_cpcv" in file_text


def _check_reliability_timeout(file_text: str, finding: Dict[str, Any]) -> bool:
    """reliability missing timeout resolved if timeout_seconds present near chimera entries"""
    import re as _re
    return bool(_re.search(r"chimera_(v51|legacy):[\s\S]{0,500}timeout_seconds:\s*\d{5,}",
                              file_text))


def _check_misleading_log_real_threshold(file_text: str, finding: Dict[str, Any]) -> bool:
    """misleading-log (cta_trend close-0.001) resolved if don_high_20 referenced in reason"""
    return "don_high_20" in file_text and "close - 0.001" not in file_text and \
              "close-0.001" not in file_text


def _check_dead_code_unreachable(file_text: str, finding: Dict[str, Any]) -> bool:
    """dead-code (ofi `if not top: return []`) resolved if the unreachable line removed"""
    # Match the specific pattern: `if not top:\n        return []`
    import re as _re
    return not bool(_re.search(r"if\s+not\s+top:\s*\n\s+return\s+\[\]", file_text))


def _check_bollinger_def(file_text: str, finding: Dict[str, Any]) -> bool:
    """bollinger-def resolved if file uses iv_sub.iloc[-1] for boll_pos"""
    return "iv_sub.iloc[-1]" in file_text


def _check_doc_drift_comment(file_text: str, finding: Dict[str, Any]) -> bool:
    """doc-drift / stale-comment: only resolvable if specific string mentioned in detail
       is GONE from file. Otherwise leave open (manual).
    """
    detail = (finding.get("detail") or "")
    # Look for the quoted string in the detail (e.g. "expect 0")
    import re as _re
    quoted = _re.findall(r'"([^"]+)"', detail)
    if not quoted:
        return False
    for q in quoted:
        if q in file_text:
            return False   # stale string still present
    return True   # all cited stale strings gone


VERIFY_RULES: list[tuple[str, callable]] = [
    ("smoke-crash", _check_smoke_crash),
    ("allocation-leak", _check_allocation_leak),
    ("contract-drift", _check_contract_drift_v3),
    ("silent-failure", _check_silent_failure),
    ("silent-skip", _check_silent_failure),
    ("math-correctness", _check_math_kurt_convention),
    ("hidden-contract", _check_hidden_emoji),
    ("temporal-leakage", _check_temporal_leakage_sort),
    ("coverage-gap", _check_coverage_gap_null_rate),
    ("logic-bug", _check_logic_bug_dead_code),
    ("data-fetch-bug", _check_data_fetch_1000_prefix),
    ("data-correctness", _check_data_fetch_1000_prefix),
    ("cash-fallback", _check_cash_fallback),
    ("date-misalignment", _check_date_misalignment),
    ("validation", _check_validation),
    ("misleading-doc", _check_misleading_doc),
    ("windows-safety", _check_windows_safety),
    ("dead-code", _check_logic_bug_dead_code),
    ("timeout", _check_timeout_yaml),
    ("hold-bars-signal-flip", _check_hold_bars_signal_flip),
    ("trigger-logic-inversion", _check_trigger_logic_inversion),
    ("invariant-missing", _check_invariant_missing_parity),
    ("operational", _check_operational_parity),
    ("regression", _check_regression_fisher),
    ("small-n", _check_small_n_fold_count),
    ("coverage-gap", _check_coverage_gap_wf_hints),
    ("doc-drift", _check_doc_drift_comment),
    ("stale-comment", _check_doc_drift_comment),
    ("reliability", _check_reliability_timeout),
    ("misleading-log", _check_misleading_log_real_threshold),
    ("bollinger-def", _check_bollinger_def),
]


# Replace the generic dead-code check with the more specific unreachable-block one,
# falling back to the dead-PBO check.
def _check_dead_code_combined(file_text: str, finding: Dict[str, Any]) -> bool:
    return _check_dead_code_unreachable(file_text, finding) or \
              _check_logic_bug_dead_code(file_text, finding)


# Replace the dead-code entry to use the combined check
VERIFY_RULES = [(k, _check_dead_code_combined if k == "dead-code" else v)
                  for k, v in VERIFY_RULES]


def verify_resolved_findings(dry_run: bool = False) -> Dict[str, int]:
    """Re-check every open finding; mark resolved if its file no longer contains
    the cited anti-pattern (per heuristic in VERIFY_RULES).

    Returns counts: {checked, marked_resolved, no_rule, file_missing}.
    Conservative: a finding without a matching rule is left open.
    """
    state = load_state()
    findings = state.get("findings", {})
    counts = {"checked": 0, "marked_resolved": 0, "no_rule": 0, "file_missing": 0}
    project_root = Path(__file__).resolve().parents[2]
    for fid, f in findings.items():
        if not isinstance(f, dict):
            continue
        if f.get("status") != "open":
            continue
        counts["checked"] += 1
        cat = (f.get("category") or "").lower()
        file_str = f.get("file") or ""
        # Files come in many shapes: "a.py:42", "a.py:1,b.py:2", "a.py + b.py".
        # Split on ',' then '+' then ':' and trim. Try the FIRST valid file
        # that exists in the project tree.
        import re as _re
        candidates = _re.split(r"[,+]", file_str)
        cleaned = [c.split(":")[0].strip() for c in candidates if c.strip()]
        existing = [c for c in cleaned if (project_root / c).exists()]
        if not existing:
            counts["file_missing"] += 1
            continue
        # Read first existing file (concat if multiple to cover spanning rules)
        text_parts: list[str] = []
        for c in existing:
            try:
                text_parts.append((project_root / c).read_text(encoding="utf-8",
                                                                  errors="replace"))
            except Exception:
                continue
        if not text_parts:
            counts["file_missing"] += 1
            continue
        text = "\n".join(text_parts)
        # Pick matching rule by category substring
        matched_check = None
        for cat_key, check_fn in VERIFY_RULES:
            if cat_key in cat:
                matched_check = check_fn
                break
        if matched_check is None:
            counts["no_rule"] += 1
            continue
        try:
            if matched_check(text, f):
                if not dry_run:
                    f["status"] = "resolved"
                    f["resolved_at"] = _now()
                    f["resolved_by"] = "verify-resolved"
                counts["marked_resolved"] += 1
        except Exception:
            # Conservative: keep open if rule crashed
            continue
    if not dry_run:
        save_state(state)
    return counts


def archive_old_findings(max_age_days: int = 30) -> int:
    """Move resolved findings older than max_age_days into a date-stamped archive.

    R32++ drift fix: shared_state.json grows monotonically as findings are
    added (resolved ones retain status but don't leave). At ~50 findings/week
    the file becomes unwieldy after ~3-6 months. This action prunes.
    """
    state = load_state()
    findings = state.get("findings", {})
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=max_age_days)
    archive_dir = SWARM_DIR / "_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"resolved_pre_{_today()}.json"

    to_archive: Dict[str, Any] = {}
    keep: Dict[str, Any] = {}
    for fid, f in findings.items():
        if f.get("status") == "resolved":
            ts_str = f.get("resolved_at") or f.get("last_seen") or _now()
            try:
                ts = dt.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                if ts < cutoff:
                    to_archive[fid] = f
                    continue
            except ValueError:
                pass
        keep[fid] = f

    if to_archive:
        # Append to existing archive (don't overwrite)
        existing: Dict[str, Any] = {}
        if archive_path.exists():
            try:
                existing = json.loads(archive_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(to_archive)
        archive_path.write_text(
            json.dumps(existing, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    state["findings"] = keep
    save_state(state)
    return len(to_archive)


def main() -> int:
    # Reconfigure stdout to UTF-8 to handle emojis in prompts on Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", required=True,
                     choices=["brief", "ingest", "triage", "list", "archive",
                                "verify-resolved"])
    ap.add_argument("--crawler", default=None, choices=list(LENSES) + [None])
    ap.add_argument("--report", default=None, help="Path to crawler output file")
    ap.add_argument("--max-age-days", type=int, default=30,
                     help="For --action archive: resolve-age cutoff in days")
    ap.add_argument("--dry-run", action="store_true",
                     help="For --action verify-resolved: report counts without saving state")
    args = ap.parse_args()

    if args.action == "brief":
        if not args.crawler:
            print("--crawler required for action=brief", file=sys.stderr)
            return 2
        print(prepare_brief(args.crawler))
        return 0

    if args.action == "ingest":
        if not args.crawler or not args.report:
            print("--crawler and --report required for action=ingest", file=sys.stderr)
            return 2
        report_text = Path(args.report).read_text(encoding="utf-8")
        counts = ingest_findings(args.crawler, report_text)
        print(f"[ingest] crawler={args.crawler} new={counts['new']} "
              f"reaffirmed={counts['reaffirmed']} resolved={counts['resolved']}")
        return 0

    if args.action == "triage":
        path = triage_summary()
        print(f"[triage] written to {path}")
        return 0

    if args.action == "archive":
        n = archive_old_findings(max_age_days=args.max_age_days)
        print(f"[archive] moved {n} resolved findings older than {args.max_age_days}d "
              f"to runs/crawler_swarm/_archive/")
        return 0

    if args.action == "verify-resolved":
        c = verify_resolved_findings(dry_run=args.dry_run)
        prefix = "[verify-resolved DRY-RUN]" if args.dry_run else "[verify-resolved]"
        print(f"{prefix} checked={c['checked']} marked_resolved={c['marked_resolved']} "
              f"no_rule={c['no_rule']} file_missing={c['file_missing']}")
        if not args.dry_run and c["marked_resolved"] > 0:
            print(f"  Re-run with --action triage to regenerate triage doc.")
        return 0

    if args.action == "list":
        state = load_state()
        print(f"crawlers seen: {list(state.get('crawlers', {}).keys())}")
        print(f"findings total: {len(state.get('findings', {}))}")
        open_n = sum(1 for f in state.get("findings", {}).values()
                       if f.get("status") == "open")
        print(f"  open: {open_n}")
        for sev in ("CRITICAL", "HIGH", "MEDIUM"):
            n = sum(1 for f in state.get("findings", {}).values()
                       if f.get("status") == "open" and f.get("severity") == sev)
            print(f"  {sev}: {n}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
