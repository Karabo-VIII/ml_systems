"""feature_health_audit.py -- end-to-end audit of every feature column in every chimera file.

Tier 2 audit (2026-05-21): pre_train_gate surfaced 2 dead norm_* features +
8 high-null prefix families on BTC. This script generalizes that check to all
87 assets, breaks down null/dead/wrong-magnitude per prefix family, AND traces
upstream panels to find WHERE the data is dropping.

OUTPUT:
  runs/audit/FEATURE_HEALTH_2026_05_21/
    01_per_asset_summary.csv        -- 87 rows × dead/null/wrong counts
    02_prefix_family_panel.csv      -- 87 assets × 14 prefix families null%
    03_upstream_panel_status.csv    -- panel file existence + last_date + null_rate
    04_dead_features_by_asset.csv   -- per (asset, feature) for std<1e-6
    05_oi_raw_audit.csv             -- raw OI data freshness per asset
    REPORT.md                       -- synthesis
"""
from __future__ import annotations

__contract__ = {
    "kind": "audit_script",
    "stage": "feature_health_audit",
    "outputs": {"dir": "runs/audit/FEATURE_HEALTH_2026_05_21/"},
}

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_DOLLAR = ROOT / "data" / "processed" / "chimera" / "dollar"
RAW_DIR = ROOT / "data" / "raw"
PANELS_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR = ROOT / "runs" / "audit" / f"FEATURE_HEALTH_{datetime.now().strftime('%Y_%m_%d')}"

# Prefix families to audit (matches pre_train_gate's chimera_v51_extended)
PREFIX_FAMILIES = [
    "norm_", "target_", "xd_", "xrel_", "te_", "rv_", "hbr_",
    "lob_", "bd_", "liq_", "wh_", "stbl_", "etf_", "soc_",
    "fp_", "mv_", "dv_", "xex_", "s3_", "hbr_",
]

# Upstream panels to check
PANEL_FILES = {
    "te_panel":          "te_panel_*.parquet",
    "etf_flows_panel":   "btc_etf_flows.parquet",   # also eth_etf_flows
    "multi_venue":       "multi_venue_features.parquet",
    "basis":             "basis_features_long.parquet",
    "liq_daily":         "liq_daily_approx.parquet",
    "liq_features":      "liq_features_long.parquet",
    "whale":             "whale_activity_daily.parquet",
    "rv_jump":           "rv_jump_panel_*.parquet",
    "s3_metrics":        "s3_metrics_panel.parquet",
    "s3_features":       "s3_features_long.parquet",
    "spot_klines":       "spot_klines_daily.parquet",
    "lob_proxy_daily":   "lob_proxy_daily.parquet",
    "stbl_supply":       "stbl_supply_daily.parquet",
    "soc_wiki":          "soc_wiki_*.parquet",
    "book_depth":        "book_depth_profile_daily.parquet",
}

HAWKES_DIR = ROOT / "data" / "processed" / "hawkes" / "daily"
FUNDING_DIR_TEMPLATE = "{sym}/funding"   # data/raw/<SYM>USDT/funding/*.parquet


def _per_asset_chimera_stats(path: Path) -> dict:
    """For one v51 chimera dollar file, compute per-column null rate + std."""
    sym = path.name.split("_")[0].upper().replace("USDT", "")
    try:
        # Read schema first to know columns
        schema = pl.read_parquet_schema(path)
        df = pl.read_parquet(path)
    except Exception as e:
        return {"asset": sym, "error": f"{type(e).__name__}: {e}"}

    n_rows = len(df)
    if n_rows == 0:
        return {"asset": sym, "n_rows": 0, "error": "empty"}

    result = {"asset": sym, "n_rows": n_rows, "n_cols": len(df.columns)}

    # Per-prefix null rate
    for pref in PREFIX_FAMILIES:
        cols = [c for c in df.columns if c.startswith(pref)]
        if not cols:
            result[f"{pref}n_cols"] = 0
            result[f"{pref}avg_null_rate"] = None
            continue
        result[f"{pref}n_cols"] = len(cols)
        nulls = [df[c].null_count() / n_rows for c in cols]
        result[f"{pref}avg_null_rate"] = float(np.mean(nulls))

    # Dead-feature scan (std<1e-6) on numeric cols
    dead = []
    near_dead = []
    for c in df.columns:
        if c in ("timestamp", "bar_id", "open", "high", "low", "close",
                 "volume", "regime_label", "hurst_regime"):
            continue
        # Only check numeric dtypes
        dtype = df.schema[c]
        if not (dtype.is_numeric() or dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)):
            continue
        try:
            s = df[c].drop_nulls()
            if len(s) == 0:
                continue
            std = s.std()
            if std is not None and std < 1e-6:
                dead.append(c)
            elif std is not None and std < 1e-3:
                near_dead.append(c)
        except Exception:
            pass
    result["dead_features"] = dead
    result["n_dead"] = len(dead)
    result["near_dead_features"] = near_dead
    result["n_near_dead"] = len(near_dead)
    return result


def _panel_status(panel_name: str, glob_pattern: str) -> dict:
    """Check if panel exists, when last modified, last data date if has 'date' col."""
    files = sorted(PANELS_DIR.glob(glob_pattern))
    info = {"panel": panel_name, "pattern": glob_pattern}
    if not files:
        # Try hawkes dir
        files = sorted(HAWKES_DIR.glob(glob_pattern)) if HAWKES_DIR.exists() else []
    if not files:
        info["status"] = "MISSING"
        return info
    f = files[-1]
    info["latest_file"] = f.name
    info["mtime"] = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    info["size_mb"] = f.stat().st_size / 1024 / 1024
    try:
        schema = pl.read_parquet_schema(f)
        info["n_cols"] = len(schema)
        if "date" in schema:
            d = pl.read_parquet(f, columns=["date"]).to_pandas()
            d["date"] = pd.to_datetime(d["date"])
            info["last_date"] = d["date"].max().strftime("%Y-%m-%d")
            info["n_rows"] = len(d)
        else:
            info["last_date"] = "no_date_col"
        info["status"] = "OK"
    except Exception as e:
        info["status"] = f"READ_FAIL: {type(e).__name__}"
    return info


def _oi_raw_status(sym: str) -> dict:
    """Check raw OI data freshness for one asset (data/raw/<sym>USDT/oi/)."""
    p = RAW_DIR / f"{sym}USDT" / "oi"
    info = {"asset": sym}
    if not p.exists():
        # Try funding/openInterestHist
        p = RAW_DIR / f"{sym}USDT" / "openInterestHist"
        if not p.exists():
            info["status"] = "MISSING_DIR"
            return info
    files = sorted(p.glob("*.parquet"))
    if not files:
        info["status"] = "EMPTY"
        return info
    info["n_files"] = len(files)
    info["latest_file"] = files[-1].name
    info["latest_mtime"] = datetime.fromtimestamp(
        files[-1].stat().st_mtime).strftime("%Y-%m-%d")
    info["status"] = "OK"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Subset of assets (root names; default: all 87)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[audit] output dir: {OUT_DIR.relative_to(ROOT)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Phase A: per-asset chimera scan -----
    print(f"\n[A] Scanning chimera v51 dollar files for null rates + dead features...")
    files = sorted(CHIMERA_DOLLAR.glob("*_v51_chimera_*.parquet"))
    if args.assets:
        wanted = {a.upper() for a in args.assets}
        files = [f for f in files
                  if f.name.split("_")[0].upper().replace("USDT", "") in wanted]
    print(f"  {len(files)} chimera files")

    asset_records = []
    dead_records = []
    for i, f in enumerate(files, 1):
        if args.dry_run and i > 3:
            break
        if i % 10 == 0 or i <= 3:
            print(f"  [{i}/{len(files)}] {f.name}")
        r = _per_asset_chimera_stats(f)
        asset_records.append(r)
        for d in r.get("dead_features", []):
            dead_records.append({"asset": r["asset"], "feature": d, "status": "dead_std<1e-6"})
        for nd in r.get("near_dead_features", []):
            dead_records.append({"asset": r["asset"], "feature": nd, "status": "near_dead_std<1e-3"})

    # ----- Phase B: prefix-family null panel -----
    print(f"\n[B] Building prefix-family null-rate panel...")
    panel_rows = []
    for r in asset_records:
        row = {"asset": r["asset"], "n_rows": r.get("n_rows", 0),
               "n_cols": r.get("n_cols", 0)}
        for pref in PREFIX_FAMILIES:
            row[pref] = r.get(f"{pref}avg_null_rate")
            row[f"{pref}n"] = r.get(f"{pref}n_cols", 0)
        panel_rows.append(row)
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(OUT_DIR / "02_prefix_family_panel.csv", index=False)
    print(f"  wrote {OUT_DIR / '02_prefix_family_panel.csv'} ({len(panel_df)} rows)")

    # ----- Phase A summary CSV -----
    summary_rows = []
    for r in asset_records:
        summary_rows.append({
            "asset": r["asset"],
            "n_rows": r.get("n_rows"),
            "n_cols": r.get("n_cols"),
            "n_dead": r.get("n_dead", 0),
            "n_near_dead": r.get("n_near_dead", 0),
            "dead_sample": ", ".join(r.get("dead_features", [])[:5]),
            "error": r.get("error", ""),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "01_per_asset_summary.csv", index=False)
    print(f"  wrote {OUT_DIR / '01_per_asset_summary.csv'}")

    pd.DataFrame(dead_records).to_csv(OUT_DIR / "04_dead_features_by_asset.csv", index=False)
    print(f"  wrote {OUT_DIR / '04_dead_features_by_asset.csv'} ({len(dead_records)} rows)")

    # ----- Phase C: upstream panel status -----
    print(f"\n[C] Auditing upstream panels...")
    panel_status = []
    for panel_name, pattern in PANEL_FILES.items():
        info = _panel_status(panel_name, pattern)
        panel_status.append(info)
        print(f"  {panel_name:<24} {info.get('status'):<10} {info.get('last_date','-'):<12} {info.get('latest_file','')}")
    pd.DataFrame(panel_status).to_csv(OUT_DIR / "03_upstream_panel_status.csv", index=False)
    print(f"  wrote {OUT_DIR / '03_upstream_panel_status.csv'}")

    # ----- Phase D: OI raw audit -----
    print(f"\n[D] Auditing raw OI data per asset...")
    oi_rows = []
    sample = args.assets if args.assets else [r["asset"] for r in asset_records]
    for sym in sample[:15]:  # cap to 15 for speed
        oi_rows.append(_oi_raw_status(sym))
    pd.DataFrame(oi_rows).to_csv(OUT_DIR / "05_oi_raw_audit.csv", index=False)
    print(f"  wrote {OUT_DIR / '05_oi_raw_audit.csv'}")

    # ----- Phase E: synthesis REPORT.md -----
    print(f"\n[E] Building REPORT.md synthesis...")

    # Cross-asset prefix family heatmap (which prefixes are mostly-null on most assets)
    pref_high_null = {}
    for pref in PREFIX_FAMILIES:
        col = pref
        if col in panel_df.columns:
            nulls = panel_df[col].dropna()
            if len(nulls) == 0: continue
            pct_assets_high_null = (nulls > 0.95).sum() / len(nulls) * 100
            avg_null = nulls.mean() * 100 if len(nulls) > 0 else 0
            pref_high_null[pref] = {
                "pct_assets_>95%null": pct_assets_high_null,
                "avg_null_pct": avg_null,
                "n_cols_per_asset": panel_df[f"{pref}n"].mean()
            }

    # Sort by severity
    pref_sorted = sorted(pref_high_null.items(), key=lambda x: -x[1]["pct_assets_>95%null"])

    # Dead-feature analysis
    dead_df = pd.DataFrame(dead_records)
    if len(dead_df) > 0:
        dead_by_feat = dead_df.groupby("feature").size().sort_values(ascending=False)
        dead_by_asset = dead_df.groupby("asset").size().sort_values(ascending=False)
    else:
        dead_by_feat = dead_by_asset = pd.Series([], dtype=int)

    lines = [
        f"# Feature Health Audit — {datetime.now().strftime('%Y-%m-%d')}\n",
        f"**Scope**: {len(asset_records)} chimera v51 dollar files",
        f"**Prefix families audited**: {len(PREFIX_FAMILIES)}",
        f"**Upstream panels checked**: {len(PANEL_FILES)}",
        "",
        "## 🔴 Headline findings",
        "",
        f"### A. Prefix families with ≥95% null on most assets (sorted by severity)",
        "",
        "| Prefix | % of assets >95% null | Avg null % | Cols/asset |",
        "|---|---:|---:|---:|",
    ]
    for pref, stats in pref_sorted:
        lines.append(f"| `{pref}` | {stats['pct_assets_>95%null']:.0f}% | "
                     f"{stats['avg_null_pct']:.1f}% | {stats['n_cols_per_asset']:.1f} |")

    lines.extend([
        "",
        "### B. Dead features (std < 1e-6) by feature name",
        "",
        "| Feature | N assets dead | Note |",
        "|---|---:|---|",
    ])
    for feat, n in dead_by_feat.head(20).items():
        lines.append(f"| `{feat}` | {n} | "
                     f"{'UNIVERSE-WIDE' if n >= len(asset_records) * 0.9 else 'partial'} |")

    lines.extend([
        "",
        "### C. Per-asset dead-feature counts (top 20)",
        "",
        "| Asset | N dead features |",
        "|---|---:|",
    ])
    for asset, n in dead_by_asset.head(20).items():
        lines.append(f"| `{asset}` | {n} |")

    lines.extend([
        "",
        "### D. Upstream panel status",
        "",
        "| Panel | Status | Last date | mtime | n_cols |",
        "|---|---|---|---|---:|",
    ])
    for p in panel_status:
        lines.append(f"| `{p['panel']}` | {p.get('status','?')} | "
                     f"{p.get('last_date','-')} | {p.get('mtime','-')} | "
                     f"{p.get('n_cols','-')} |")

    lines.extend([
        "",
        "## Diagnosis",
        "",
        "### High-null prefix families = upstream panel NOT joining into chimera",
        "",
        "When a prefix family is 100% null across most assets, it means:",
        "1. The upstream panel may exist but is NOT being JOINED into chimera_v51 (frontier_consolidator failure mode), OR",
        "2. The upstream panel never produced data for those assets (legitimate gap), OR",
        "3. The chimera_v51 schema declares the columns but skips the actual values (silent NaN).",
        "",
        "Cross-reference with Section D (upstream panel status) to disambiguate.",
        "",
        "### Dead features = source data constant or missing",
        "",
        "When a `norm_*` feature has std<1e-6 across many assets, it usually means:",
        "1. The raw source for that feature is constant (e.g., OI not updating)",
        "2. The z-score computation is degenerate (denominator zero), OR",
        "3. The fill_null behavior is replacing all values with a constant.",
        "",
        "For BTC's `norm_oi_change` / `norm_oi_price_divergence` flagged by pre_train_gate,",
        "Section E (raw OI audit) shows whether the raw OI data is current per asset.",
        "",
        "## CSV outputs",
        "",
        "- `01_per_asset_summary.csv` — per-asset row counts + dead counts",
        "- `02_prefix_family_panel.csv` — 87 assets × 14+ prefix-family null rates",
        "- `03_upstream_panel_status.csv` — panel existence + freshness + cols",
        "- `04_dead_features_by_asset.csv` — every (asset, feature) where std<1e-6",
        "- `05_oi_raw_audit.csv` — raw OI data freshness per asset",
    ])

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Audit complete. Open: {OUT_DIR.relative_to(ROOT)}/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
