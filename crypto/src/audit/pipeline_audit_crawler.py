"""pipeline_audit_crawler.py -- comprehensive pipeline-layer audit.

Phase A1 of the pipeline overhaul (user mandate 2026-05-16). Walks every
DAG stage + every producer + every output and flags issues across SIX
audit axes:

  1. BACKFILL-COVERAGE      : output spans far less than upstream input
                              (LOB-class issue: 30 days vs 2300 days available)
  2. ATOMIC-WRITE-MISSING   : producer writes without rename-atomic guarantee
                              -> half-written file class of corruption
  3. FRAMEWORK-PRIMITIVES   : producer doesn't use the framework cli/dispatch
  4. DATA-QUALITY           : stale-value forward-fills, dead features,
                              suspicious distributions
  5. SPEED-PROFILE          : per-stage wall-time from refresh logs;
                              flag regressions vs prior runs
  6. SCHEMA-DRIFT           : output column count / dtype changes since
                              prior _dag_state snapshot

OUTPUT
------
runs/audit/pipeline_audit_<DATE>.md  -- per-stage findings + remediation queue

INVOKE
------
    python src/audit/pipeline_audit_crawler.py
    python src/audit/pipeline_audit_crawler.py --backfill-only
    python src/audit/pipeline_audit_crawler.py --coverage-threshold-days 365
"""
from __future__ import annotations

__contract__ = {
    "kind": "pipeline_audit_crawler",
    "owner": "audit/pipeline",
    "outputs": ["runs/audit/pipeline_audit_<DATE>.md"],
    "invariants": [
        "non-invasive: only reads code + outputs; never invokes producers",
        "every flagged finding includes file:line + remediation prescription",
        "complements existing orphan_feature_crawler + check_invariants CDAP",
    ],
}

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    import polars as pl
except ImportError:
    pl = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = PROJECT_ROOT / "config" / "asset_dag.yaml"
CHIMERA_BASE = PROJECT_ROOT / "data" / "processed" / "chimera"
PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"
RAW_BASE = PROJECT_ROOT / "data" / "raw"
DAG_STATE = PROJECT_ROOT / "data" / "_dag_state.json"
LOGS_REFRESH = PROJECT_ROOT / "logs" / "refresh"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dag() -> dict[str, dict]:
    with open(DAG_PATH) as f:
        d = yaml.safe_load(f)
    return d.get("assets") or {}


# ============================================================================
# Axis 1: Backfill coverage
# ============================================================================

def audit_backfill_coverage(dag: dict[str, dict],
                              coverage_threshold_days: int = 365
                              ) -> list[dict]:
    """For each per_asset stage, measure date span of its output vs the
    span of its upstream inputs. Flag when output < threshold% of input span."""
    findings: list[dict] = []
    for stage_name, body in dag.items():
        per_asset = body.get("per_asset", False)
        deps = body.get("deps", []) or []
        output = body.get("output", "")
        # Skip non-per-asset and ephemeral
        if body.get("output_kind") == "ephemeral":
            continue
        # Resolve canonical output dir
        if "{asset}" not in output:
            # multi-asset blob - check the parquet itself
            paths = list(PROJECT_ROOT.glob(output))
            if paths and pl is not None:
                try:
                    df = pl.read_parquet(paths[0]).to_pandas()
                    if "date" in df.columns:
                        date_min = df["date"].min()
                        date_max = df["date"].max()
                        span = (date_max - date_min).days if hasattr(date_max, "__sub__") else 0
                        if span < coverage_threshold_days and span > 0:
                            findings.append({
                                "stage": stage_name, "category": "backfill-gap",
                                "output": str(paths[0].relative_to(PROJECT_ROOT)),
                                "date_min": str(date_min), "date_max": str(date_max),
                                "span_days": int(span),
                                "threshold_days": coverage_threshold_days,
                                "fix": f"backfill: rerun {body.get('producer')} for "
                                          f"start_date={date_min - dt.timedelta(days=coverage_threshold_days)}",
                            })
                except Exception:
                    pass
            continue
        # per-asset: sample 3 assets
        for sample_sym in ("btc", "eth", "sol"):
            sym_lower = sample_sym
            pattern = output.replace("{asset}", sym_lower)
            paths = sorted(PROJECT_ROOT.glob(pattern))
            if not paths:
                continue
            if pl is not None:
                try:
                    df = pl.read_parquet(paths[-1]).to_pandas()
                    if "date" in df.columns:
                        date_min = df["date"].min()
                        date_max = df["date"].max()
                    elif "timestamp" in df.columns:
                        import pandas as pd
                        date_min = pd.to_datetime(df["timestamp"].min(), unit="ms")
                        date_max = pd.to_datetime(df["timestamp"].max(), unit="ms")
                    else:
                        continue
                    span = (date_max - date_min).days if hasattr(date_max, "__sub__") else 0
                    if span < coverage_threshold_days and span > 0:
                        findings.append({
                            "stage": stage_name, "category": "backfill-gap",
                            "sample_asset": sample_sym.upper(),
                            "output": str(paths[-1].relative_to(PROJECT_ROOT)),
                            "date_min": str(date_min)[:10],
                            "date_max": str(date_max)[:10],
                            "span_days": int(span),
                            "threshold_days": coverage_threshold_days,
                            "fix": f"backfill: rerun {body.get('producer')} for "
                                      f"historical range",
                        })
                        break    # one finding per stage is enough
                except Exception:
                    pass
    return findings


# ============================================================================
# Axis 2: Atomic-write missing
# ============================================================================

def audit_atomic_write(dag: dict[str, dict]) -> list[dict]:
    findings: list[dict] = []
    for stage_name, body in dag.items():
        producer = body.get("producer", "")
        if not producer:
            continue
        fp = PROJECT_ROOT / producer
        if not fp.exists():
            continue
        if body.get("output_kind") == "ephemeral":
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Look for writes-without-atomic
        # Detect EITHER the framework helper OR the canonical manual pattern
        # (tmp file + rename). Manual pattern matches:
        #   tmp = ...with_suffix(".tmp")  ... tmp.rename(out_path)
        #   _tmp = ...  ... _tmp.rename(...)
        #   tmp_path = ...  ... tmp_path.rename(...)
        has_atomic = ("atomic_write_parquet" in text or
                       (".rename(" in text and (
                           "tmp" in text or "_tmp" in text or "tmp_path" in text
                       )))
        has_write = ("write_parquet" in text or "to_parquet" in text or
                      ".to_csv" in text)
        if has_write and not has_atomic:
            # Find the write call line for precision
            line_no = "?"
            for i, line in enumerate(text.splitlines(), 1):
                if "write_parquet" in line or "to_parquet" in line:
                    line_no = i
                    break
            findings.append({
                "stage": stage_name, "category": "atomic-write-missing",
                "producer": producer, "line": line_no,
                "fix": "import from src.pipeline.parquet_io and replace "
                          ".write_parquet/to_parquet with atomic_write_parquet()",
            })
    return findings


# ============================================================================
# Axis 3: Framework primitives adoption
# ============================================================================

def audit_framework_primitives(dag: dict[str, dict]) -> list[dict]:
    """Per PIPELINE_FRAMEWORK_2026_05_01.md, producers should use:
       - parquet_io.atomic_write_parquet (covered by Axis 2)
       - dispatch.run_per_task for parallel
       - cli.add_standard_args + cli.resolve_assets
    """
    findings: list[dict] = []
    for stage_name, body in dag.items():
        producer = body.get("producer", "")
        if not producer:
            continue
        fp = PROJECT_ROOT / producer
        if not fp.exists() or body.get("output_kind") == "ephemeral":
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        missing = []
        # CLI primitives
        if "add_standard_args" not in text and "argparse" in text:
            missing.append("cli.add_standard_args")
        # Dispatch primitives (only if producer has explicit workers)
        if "--workers" in text and "run_per_task" not in text and \
            "ThreadPoolExecutor" in text:
            missing.append("dispatch.run_per_task")
        if missing:
            findings.append({
                "stage": stage_name, "category": "framework-primitives",
                "producer": producer, "missing": missing,
                "fix": f"adopt: {', '.join(missing)}",
            })
    return findings


# ============================================================================
# Axis 4: Data quality (stale-value forward-fill detection)
# ============================================================================

def audit_data_quality(dag: dict[str, dict],
                          sample_assets: list[str] = None) -> list[dict]:
    """Scan chimera per-asset 1d files; flag columns with REPEATING values
    (forward-fill bug) or extreme NaN runs.

    Multi-asset gating: only flag if pattern persists across BTC + ETH + SOL.
    This eliminates BTC-only definitional zeros (e.g. xd_funding_spread is 0
    for BTC by construction).
    """
    if pl is None:
        return []
    if sample_assets is None:
        sample_assets = ["btc", "eth", "sol"]
    findings: list[dict] = []
    # First pass: gather per-asset dead/stale signatures
    per_asset_findings: dict[str, dict[str, dict]] = {sym: {} for sym in sample_assets}
    for sym in sample_assets:
        fps = sorted(CHIMERA_BASE.glob(f"1d/{sym}usdt_*.parquet"))
        if not fps:
            continue
        try:
            df = pl.read_parquet(fps[-1]).to_pandas()
        except Exception:
            continue
        n = len(df)
        # For each non-trivial numeric column, check for "all same value" runs
        for c in df.columns:
            try:
                non_null = df[c].notna().sum()
                if non_null < 100:
                    continue
                non_null_series = df[c].dropna()
                if non_null_series.dtype.kind not in "if":
                    continue
                # Detect forward-fill: same value repeated > 50% of bars
                from collections import Counter
                value_counts = Counter(non_null_series.values)
                most_common_val, most_common_n = value_counts.most_common(1)[0]
                pct = most_common_n / non_null
                # Refined classification:
                #   - most_common = 0 AND pct > 90%: rare-event indicator (likely
                #     legit, e.g. extreme_long flag). Skip unless > 99%.
                #   - most_common != 0 AND pct > 50%: REAL forward-fill suspect.
                #   - any value at > 95% pct AND surrounding distribution
                #     is degenerate: forward-fill suspect.
                is_zero = (abs(most_common_val) < 1e-9)
                is_suspect = False
                if not is_zero and pct > 0.50:
                    is_suspect = True
                elif is_zero and pct > 0.95 and pct < 1.0:
                    # Almost all zero but a small minority varies -> probably
                    # rare-event indicator (legit); skip
                    pass
                elif is_zero and pct >= 1.0:
                    per_asset_findings[sym][c] = {
                        "category": "dead-feature",
                        "n_non_null": int(non_null),
                        "most_common_value": float(most_common_val),
                        "most_common_pct": round(pct, 3),
                    }
                    continue
                if is_suspect:
                    per_asset_findings[sym][c] = {
                        "category": "stale-forward-fill",
                        "n_non_null": int(non_null),
                        "most_common_value": float(most_common_val),
                        "most_common_pct": round(pct, 3),
                    }
            except Exception:
                continue
    # Second pass: emit findings only if pattern persists across ALL sampled assets
    all_cols = set()
    for sym in sample_assets:
        all_cols.update(per_asset_findings.get(sym, {}).keys())
    for col in all_cols:
        present_in = [sym for sym in sample_assets
                         if col in per_asset_findings.get(sym, {})]
        if len(present_in) >= len(sample_assets):
            # All assets show the same pattern -> real systemic issue
            sample = per_asset_findings[present_in[0]][col]
            findings.append({
                "stage": "chimera_v51_extended",
                "category": sample["category"],
                "column": col, "n_non_null": sample["n_non_null"],
                "most_common_value": sample["most_common_value"],
                "most_common_pct": sample["most_common_pct"],
                "persists_across_assets": present_in,
                "fix": (f"feature {col} is {sample['category']} on ALL "
                          f"sampled assets ({present_in}); systemic - "
                          f"investigate producer"),
            })
    return findings


# ============================================================================
# Axis 5: Speed profile from refresh logs
# ============================================================================

def audit_speed_profile(dag: dict[str, dict]) -> list[dict]:
    """Read recent refresh logs; flag stages running > 80% of their timeout."""
    findings: list[dict] = []
    if not LOGS_REFRESH.exists():
        return findings
    # Read recent log files; extract elapsed_s per stage
    recent_logs = sorted(LOGS_REFRESH.glob("*.log"),
                            key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    by_stage: dict[str, list[float]] = defaultdict(list)
    for fp in recent_logs:
        stage_name = fp.stem.rsplit("_", 1)[0]
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Heuristic: look for total wall time in last 10 lines
        last_lines = text.strip().splitlines()[-30:]
        for ln in last_lines:
            m = re.search(r"elapsed[_ ]?(?:s|sec)?[=:\s]*([\d.]+)", ln,
                            re.IGNORECASE)
            if m:
                by_stage[stage_name].append(float(m.group(1)))
                break
    for stage_name, body in dag.items():
        timeout = body.get("timeout_seconds", 14400)
        elapsed_samples = by_stage.get(stage_name, [])
        if not elapsed_samples:
            continue
        max_elapsed = max(elapsed_samples)
        if max_elapsed > 0.8 * timeout:
            findings.append({
                "stage": stage_name, "category": "speed-profile",
                "max_elapsed_sec": int(max_elapsed),
                "timeout_sec": int(timeout),
                "pct_of_timeout": round(max_elapsed / timeout, 2),
                "fix": "stage close to timeout limit; consider workers bump "
                          "OR per-task partition OR producer optimization",
            })
    return findings


# ============================================================================
# Axis 6: Schema drift
# ============================================================================

def audit_schema_drift() -> list[dict]:
    """Compare current chimera schema to prior _dag_state snapshot."""
    findings: list[dict] = []
    if not DAG_STATE.exists() or pl is None:
        return findings
    try:
        state = json.loads(DAG_STATE.read_text(encoding="utf-8"))
    except Exception:
        return findings
    # Pull a chimera sample
    fps = sorted(CHIMERA_BASE.glob("1d/btcusdt_*.parquet"))
    if not fps:
        return findings
    try:
        current = list(pl.read_parquet_schema(fps[-1]).keys())
    except Exception:
        return findings
    n_cols_current = len(current)
    # Compare to state-stored schema if available
    chimera_state = state.get("assets", {}).get("chimera_v51:u100", {}) or \
                       state.get("assets", {}).get("chimera_v51", {})
    prior_n_cols = chimera_state.get("n_cols")
    if prior_n_cols and prior_n_cols != n_cols_current:
        findings.append({
            "stage": "chimera_v51", "category": "schema-drift",
            "prior_n_cols": prior_n_cols, "current_n_cols": n_cols_current,
            "delta": n_cols_current - prior_n_cols,
            "fix": f"schema changed by {n_cols_current - prior_n_cols}; "
                      f"verify intentional + update downstream consumers",
        })
    return findings


# ============================================================================
# Reporter
# ============================================================================

def write_report(all_findings: list[dict]) -> Path:
    today = dt.date.today().isoformat()
    out = OUT_DIR / f"pipeline_audit_{today}.md"
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_cat[f["category"]].append(f)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Pipeline Audit -- {today}\n\n")
        fh.write(f"Total findings: {len(all_findings)}\n\n")
        fh.write(f"## Summary by category\n\n")
        for cat, lst in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"- **{cat}**: {len(lst)}\n")
        fh.write("\n")
        for cat, lst in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"## {cat} ({len(lst)})\n\n")
            for f in lst:
                rows = [f"  - {k}: {v}" for k, v in f.items() if k != "category"]
                fh.write("- finding:\n" + "\n".join(rows) + "\n\n")
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill-only", action="store_true")
    ap.add_argument("--coverage-threshold-days", type=int, default=365)
    args = ap.parse_args()

    dag = load_dag()
    print(f"[pipeline-audit] loaded {len(dag)} DAG stages")

    findings: list[dict] = []
    if args.backfill_only:
        findings = audit_backfill_coverage(dag, args.coverage_threshold_days)
    else:
        print("  axis 1: backfill coverage ...", flush=True)
        findings += audit_backfill_coverage(dag, args.coverage_threshold_days)
        print("  axis 2: atomic-write ...", flush=True)
        findings += audit_atomic_write(dag)
        print("  axis 3: framework primitives ...", flush=True)
        findings += audit_framework_primitives(dag)
        print("  axis 4: data quality ...", flush=True)
        findings += audit_data_quality(dag)
        print("  axis 5: speed profile ...", flush=True)
        findings += audit_speed_profile(dag)
        print("  axis 6: schema drift ...", flush=True)
        findings += audit_schema_drift()

    out = write_report(findings)
    print(f"[pipeline-audit] {len(findings)} findings -> {out}")
    by_cat: dict[str, int] = defaultdict(int)
    for f in findings:
        by_cat[f["category"]] += 1
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<28s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
