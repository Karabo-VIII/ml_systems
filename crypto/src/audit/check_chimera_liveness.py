"""
CDAP Layer 7 — Chimera Liveness + Column-Parity + Content-Hash Gate
====================================================================

Per pipeline-expert audit 2026-05-25 19:35 SAST, the trust stack validates
audit JSONs but does NOT validate the underlying chimera data. Gaps caught:

  1. Chimera STALENESS: parquet last bar was 2026-05-22 (56.9h stale at
     audit time). Live bot would trade on data 3 days old.
  2. Column-alias DRIFT: pipeline column `wh_whale_net_usd`, ingest
     `whale_net_usd`, strategy filter_kind `whale_net>0` — three names,
     no runtime check that they resolve consistently.
  3. Content-hash MISSING: repro block records `chimera_mtime_utc`
     (filesystem mtime) but not SHA-256 of the parquet bytes. Bit-exact
     replay not verifiable.

This module enforces all three at commit-time. Exit codes:
  0 — chimera fresh + columns resolve + (SHA absent OR matches manifest)
  1 — chimera stale beyond WARN threshold OR SHA absent (WARN only)
  2 — chimera stale beyond CRIT threshold OR column-alias resolution fails

Wired into `check_invariants.py:run_audit()` alongside the wealth_bot
claim contract checker (Layers 1-6).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configurable thresholds (tuned for 4h cadence + multi-instance dev cadence)
CHIMERA_WARN_STALE_HOURS = 24       # 4h cadence × 6 bars = ~1 day
CHIMERA_CRIT_STALE_HOURS = 96       # 4 days → live bot would be flying blind

# Canonical column-alias mapping. Adding a NEW filter_kind requires
# documenting its column mapping here.
CANONICAL_FILTER_COLUMN_MAP = {
    "whale_net>0":               "wh_whale_net_usd",
    "whale_net>30d_median":      "wh_whale_net_usd",
    "whale_net>60d_median":      "wh_whale_net_usd",
    "btc_tape>0":                "te_btc_imb",
    "short_liq_z>0":             "liq_short_z30",
    "long_liq_z<0":              "liq_long_z30",
    "basis_z<0":                 "bs_basis_z30",
    "btc_ret<0":                 "xd_btc_return",
    "tape_imb>0":                "hbr_eta_imbalance",
    "hbr_eta_buy>0":             "hbr_eta_buy",
    "bd_imb>med":                "bd_imbalance_l1",
    "fund_low":                  "fund_rate_mean",
    "lob_kyle_low":              "lob_bgf_kyle_lambda_mean",
}


def _find_latest_chimera(cadence: str = "4h", asset: str = "pepeusdt") -> Path | None:
    chimera_dir = PROJECT_ROOT / "data" / "processed" / "chimera" / cadence
    if not chimera_dir.exists():
        return None
    candidates = sorted(chimera_dir.glob(f"{asset}_v51_chimera_{cadence}_*.parquet"))
    return candidates[-1] if candidates else None


def _load_chimera_metadata(parquet_path: Path) -> dict:
    """Read just the parquet metadata + last timestamp (no full load).

    Falls back gracefully if polars/pyarrow not available; reports the file
    mtime + size + columns. CDAP runs on the lightest path possible to keep
    pre-commit fast.
    """
    try:
        import polars as pl
        # Read only the timestamp column to find the last bar
        df_ts = pl.read_parquet(parquet_path, columns=["timestamp"])
        last_ts_ms = int(df_ts["timestamp"].max())
        last_dt = datetime.datetime.fromtimestamp(last_ts_ms / 1000, tz=datetime.timezone.utc)
        # Get column list from schema
        schema = pl.read_parquet_schema(parquet_path)
        cols = list(schema.keys())
        return {
            "path": str(parquet_path),
            "last_bar_utc": last_dt.isoformat(),
            "last_bar_ts_ms": last_ts_ms,
            "n_columns": len(cols),
            "columns_subset": cols[:5] + (["..."] if len(cols) > 5 else []),
            "all_columns": cols,
            "mtime_utc": datetime.datetime.fromtimestamp(
                parquet_path.stat().st_mtime, tz=datetime.timezone.utc
            ).isoformat(),
            "size_bytes": parquet_path.stat().st_size,
        }
    except Exception as e:
        return {
            "path": str(parquet_path),
            "error": f"could not read chimera metadata: {e}",
            "mtime_utc": datetime.datetime.fromtimestamp(
                parquet_path.stat().st_mtime, tz=datetime.timezone.utc
            ).isoformat() if parquet_path.exists() else None,
        }


def _check_staleness(metadata: dict, now_utc: datetime.datetime) -> list[dict]:
    findings = []
    last_ts_iso = metadata.get("last_bar_utc")
    if not last_ts_iso:
        findings.append({
            "severity": "warn",
            "name": "chimera_last_bar_unknown",
            "detail": f"Could not parse last_bar_utc from {metadata.get('path')}",
        })
        return findings
    last_dt = datetime.datetime.fromisoformat(last_ts_iso)
    hours_stale = (now_utc - last_dt).total_seconds() / 3600.0
    if hours_stale > CHIMERA_CRIT_STALE_HOURS:
        # Staleness is a DATA/DEPLOY concern, not CODE-correctness: blocking code COMMITS on it trains
        # wholesale SKIP_CDAP bypass (brain-audit 2026-06-05 doc 13; CDAP was permanently-red on stale 4h data).
        # Commit-time = WARN; the hard CRIT gate fires only on the deploy-preflight path (CDAP_DEPLOY=1).
        # column-alias + sha-mismatch stay CRITICAL regardless (real-capital). (Fx3, 2-skill consensus 2026-06-05.)
        _deploy = os.environ.get("CDAP_DEPLOY") == "1"
        findings.append({
            "severity": "critical" if _deploy else "warn",
            "name": "chimera_stale_crit",
            "file": metadata.get("path"),
            "detail": (
                f"Chimera last bar {last_ts_iso} is {hours_stale:.1f}h stale "
                f"(>{CHIMERA_CRIT_STALE_HOURS}h CRIT threshold). "
                "Live bot would trade on stale data. Rebuild chimera before deploy."
                + ("" if _deploy else " [commit-time WARN; set CDAP_DEPLOY=1 for the deploy-preflight CRIT gate]")
            ),
        })
    elif hours_stale > CHIMERA_WARN_STALE_HOURS:
        findings.append({
            "severity": "warn",
            "name": "chimera_stale_warn",
            "file": metadata.get("path"),
            "detail": (
                f"Chimera last bar {last_ts_iso} is {hours_stale:.1f}h stale "
                f"(>{CHIMERA_WARN_STALE_HOURS}h WARN). Verify before deploy."
            ),
        })
    return findings


def _check_column_aliases(metadata: dict) -> list[dict]:
    """Verify every canonical filter_kind resolves to an existing column."""
    findings = []
    cols = set(metadata.get("all_columns", []))
    if not cols:
        return findings  # graceful degrade if metadata couldn't be read
    for filter_kind, expected_col in CANONICAL_FILTER_COLUMN_MAP.items():
        if expected_col not in cols:
            findings.append({
                "severity": "critical",
                "name": "column_alias_resolution_failure",
                "file": metadata.get("path"),
                "detail": (
                    f"filter_kind '{filter_kind}' expects column '{expected_col}' "
                    f"but it is missing from the chimera. The strategy will silently "
                    f"fail or read zero-filled column. "
                    f"Fix: rebuild chimera with feature or update CANONICAL_FILTER_COLUMN_MAP."
                ),
            })
    return findings


def _check_content_hash(parquet_path: Path, manifest_path: Path | None = None) -> list[dict]:
    """If a manifest with chimera_sha256 exists, verify the parquet matches.

    Currently the project doesn't pin SHA — emit WARN to nudge toward
    pinned-hash repro blocks per pipeline-expert finding.
    """
    findings = []
    if manifest_path is None:
        manifest_path = PROJECT_ROOT / "data" / "manifests" / f"v51_{parquet_path.stem.split('_')[0].upper()}.json"
    if not manifest_path.exists():
        findings.append({
            "severity": "warn",
            "name": "chimera_content_hash_missing",
            "file": str(parquet_path),
            "detail": (
                f"No manifest at {manifest_path}. Cannot verify bit-exact replay. "
                "Per pipeline-expert finding: every audit's repro block should include "
                "sha256(chimera) for bit-exact reproducibility."
            ),
        })
        return findings
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest.get("chimera_sha256") or manifest.get("v51_output_sha256")
        if not expected_sha:
            findings.append({
                "severity": "warn",
                "name": "chimera_manifest_lacks_sha",
                "file": str(manifest_path),
                "detail": "Manifest exists but does not include 'chimera_sha256' field. Update producers.",
            })
            return findings
        # Verify
        h = hashlib.sha256()
        with open(parquet_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
        if actual_sha != expected_sha:
            findings.append({
                "severity": "critical",
                "name": "chimera_sha_mismatch",
                "file": str(parquet_path),
                "detail": (
                    f"Chimera SHA mismatch: actual {actual_sha[:12]}... vs manifest {expected_sha[:12]}... "
                    "Data drift or silent rebuild. HALT commit."
                ),
            })
    except Exception as e:
        findings.append({
            "severity": "warn",
            "name": "chimera_sha_check_error",
            "detail": str(e),
        })
    return findings


def run_audit(asset: str = "pepeusdt", cadence: str = "4h") -> tuple[list[dict], int]:
    """Run chimera liveness + column-parity + content-hash checks."""
    parquet = _find_latest_chimera(cadence=cadence, asset=asset)
    if parquet is None:
        return [{
            "severity": "warn",
            "name": "no_chimera_found",
            "detail": f"No chimera parquet for {asset} {cadence} cadence under data/processed/chimera/{cadence}/",
        }], 1
    metadata = _load_chimera_metadata(parquet)
    findings: list[dict] = []
    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    findings.extend(_check_staleness(metadata, now_utc))
    findings.extend(_check_column_aliases(metadata))
    findings.extend(_check_content_hash(parquet))

    n_crit = sum(1 for f in findings if f["severity"] == "critical")
    n_warn = sum(1 for f in findings if f["severity"] == "warn")
    if n_crit > 0:
        return findings, 2
    if n_warn > 0:
        return findings, 1
    return findings, 0


def main() -> int:
    findings, exit_code = run_audit()
    if exit_code == 0:
        print("[check_chimera_liveness] OK - chimera fresh, columns resolve, SHA matches")
        return 0
    severity_label = "CRIT" if exit_code == 2 else "WARN"
    print(f"[check_chimera_liveness] {severity_label} - {sum(1 for f in findings if f['severity'] == 'critical')} CRIT, "
          f"{sum(1 for f in findings if f['severity'] == 'warn')} WARN")
    for f in findings:
        prefix = {"critical": "FAIL", "warn": "WARN"}.get(f["severity"], "?")
        loc = f"  [{f['file']}]" if f.get("file") else ""
        print(f"  {prefix} {f['name']}{loc}: {f['detail']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
