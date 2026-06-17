"""Spot ETF flow ingestion — Farside Investors public table (no API key).

Scrapes the daily ETF flow table from:
    https://farside.co.uk/bitcoin-etf-flow-all-data/
    https://farside.co.uk/ethereum-etf-flow-all-data/

Per-issuer daily net flow in USD millions. Covers:
    BTC ETFs (11 issuers): IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW,
                           GBTC, BTC  (Grayscale mini), + 'Total' column
    ETH ETFs (9 issuers): ETHA, FETH, ETHW, CETH, ETHV, QETH, EZET, ETHE,
                          ETH (Grayscale mini), + 'Total' column

ETFs launched January 2024 (BTC) and July 2024 (ETH). Full daily history since.

Hypothesis: ETF net inflows/outflows are leading indicator for BTC/ETH spot price.
When IBIT (BlackRock) alone takes in $500M+ in a day, that capital hits spot BTC
within 0-24h, creating measurable price pressure. Documented by J.P. Morgan,
Galaxy, K33 Research — major 2024-2026 signal.

Output:
    data/frontier/etf/btc_etf_flows.parquet
    data/frontier/etf/eth_etf_flows.parquet
        columns: date, IBIT, FBTC, ..., Total_USDm

Rate limit: Farside is a low-traffic site. 1 request/minute safe. We pull once.
"""
from __future__ import annotations
import os

import io
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src"))
from pipeline.ingest._manifest import MissingManifest
from pipeline.parquet_io import atomic_write_parquet

OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Derived ETF flow-feature file that the chimera actually consumes (registry
# source `etf_flows`). 2026-05-29: the derivation step (etf_flow_signals.py) was
# orphaned in src/_archive when src/frontier was archived, so the raw farside
# flows were fetched but etf_flow_features.parquet (the chimera input) went ~5wk
# stale. Derivation is now folded into this producer.
FEAT_PATH = ROOT / "data" / "raw_external" / "farside" / "etf_flow_features.parquet"

# Manifest: per-asset (btc / eth), key = "page" (whole-page fetch unit).
# Farside has one URL per asset; the only manifest-worthy failure is a
# persistent HTTP error or parse failure on that URL.
_MANIFEST_ROOT = OUT_DIR / "etf_flows_manifests"
_mm = MissingManifest(_MANIFEST_ROOT, recheck_stale_days=7)  # retry after 7d (site usually up)

# Sentinel key used for the single-page-per-asset fetch.
_PAGE_KEY = "page"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

URLS = {
    "btc": "https://farside.co.uk/bitcoin-etf-flow-all-data/",
    "eth": "https://farside.co.uk/ethereum-etf-flow-all-data/",
}


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("etf", phase, message, **kw)


def _fetch(url: str, retries: int = 3) -> str:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(2 ** i)
    raise RuntimeError(f"failed to fetch {url}")


def _parse_etf_table(html: str) -> pd.DataFrame:
    """Extract the <table class='etf'>...</table> and parse with pandas.read_html."""
    # Locate the etf-class table
    m = re.search(r'<table class="etf">.*?</table>', html, re.DOTALL)
    if not m:
        raise ValueError("etf table not found")
    table_html = m.group(0)
    tables = pd.read_html(io.StringIO(table_html))
    if not tables:
        raise ValueError("pandas read_html returned no tables")
    df = tables[0]
    # Normalize headers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(c) for c in col if c).strip() for col in df.columns.values]
    # Find date column
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    # Clean: drop summary rows (Total, Avg, etc.)
    df = df[df["date"].astype(str).str.match(r"^\d{2}\s+\w{3}\s+\d{4}$", na=False)]
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Numeric conversion (dashes/blanks -> 0)
    for c in df.columns:
        if c == "date":
            continue
        s = df[c].astype(str)
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace(r"^\s*-\s*$", "0", regex=True)
        s = s.str.replace(r"^\s*$", "0", regex=True)
        s = s.str.replace(r"\((.+)\)", r"-\1", regex=True)  # parentheses = negative
        df[c] = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return df


def fetch_and_save(asset: str, recheck_missing: bool = False) -> pd.DataFrame:
    url = URLS[asset]
    # confirmed_missing: skip if the Farside page was unreachable recently
    # and recheck has not been requested.
    if not recheck_missing and _mm.is_known_missing(asset, _PAGE_KEY):
        raise RuntimeError(
            f"etf {asset}: confirmed-missing (Farside page unreachable); "
            "use --recheck-missing to retry"
        )
    _pl("BUILD", f"{asset}: fetching {url}...")
    try:
        html = _fetch(url)
    except Exception as e:
        _mm.mark_missing(asset, _PAGE_KEY)
        raise
    df = _parse_etf_table(html)
    # Successful parse: clear any stale manifest entry.
    _mm.unmark_missing(asset, _PAGE_KEY)
    _pl("BUILD", f"{asset}: parsed {len(df)} rows, {len(df.columns)-1} issuers")
    print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  columns: {[c for c in df.columns if c != 'date']}")

    out = OUT_DIR / f"{asset}_etf_flows.parquet"
    # G-AUDIT-020: atomic-tmp-rename + column-name verify (RED TEAM contract)
    _tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(_tmp, index=False)
    import pyarrow.parquet as _pq
    _written = set(_pq.read_schema(_tmp).names)
    if "date" not in _written:
        _tmp.unlink(missing_ok=True)
        raise ValueError(f"etf {asset}: missing 'date' column")
    if out.exists():
        out.unlink()
    os.replace(str(_tmp), str(out))  # atomic overwrite (Windows-safe)
    print(f"  saved: {out}")

    # Print tail for sanity
    print(f"\n[{asset}] tail:")
    print(df.tail(5).to_string(index=False))
    return df


def _build_etf_one(df: pd.DataFrame, prefix: str, ibit_col: str) -> pd.DataFrame:
    """Derive ETF flow features for one chain (btc_etf / eth_etf). z-scores use
    shift(1) before the rolling window (no look-ahead)."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    total_col = "Total" if "Total" in df.columns else [c for c in df.columns if "Total" in c][0]
    df[f"{prefix}_total_usdm"] = df[total_col]
    df[f"{prefix}_ibit_usdm"] = df[ibit_col] if ibit_col in df.columns else np.nan
    for src_col, z_col in [(f"{prefix}_total_usdm", f"{prefix}_total_z30"),
                            (f"{prefix}_ibit_usdm", f"{prefix}_ibit_z30")]:
        rm = df[src_col].shift(1).rolling(30, min_periods=10).mean()
        rs = df[src_col].shift(1).rolling(30, min_periods=10).std()
        df[z_col] = (df[src_col] - rm) / rs.replace(0, np.nan)
    df[f"{prefix}_total_7d"] = df[f"{prefix}_total_usdm"].rolling(7, min_periods=3).sum()
    df[f"{prefix}_total_14d"] = df[f"{prefix}_total_usdm"].rolling(14, min_periods=5).sum()
    rm7 = df[f"{prefix}_total_7d"].shift(1).rolling(30, min_periods=10).mean()
    rs7 = df[f"{prefix}_total_7d"].shift(1).rolling(30, min_periods=10).std()
    df[f"{prefix}_total_7d_z"] = (df[f"{prefix}_total_7d"] - rm7) / rs7.replace(0, np.nan)
    df[f"{prefix}_inflow_shock"] = (df[f"{prefix}_total_z30"] > 2.0).astype(int)
    df[f"{prefix}_outflow_shock"] = (df[f"{prefix}_total_z30"] < -2.0).astype(int)
    df[f"{prefix}_mega_inflow"] = (df[f"{prefix}_total_usdm"] > 500).astype(int)
    df[f"{prefix}_mega_outflow"] = (df[f"{prefix}_total_usdm"] < -500).astype(int)
    df[f"{prefix}_consistent_inflow_7d"] = (df[f"{prefix}_total_7d"] > 1000).astype(int)
    df[f"{prefix}_consistent_outflow_7d"] = (df[f"{prefix}_total_7d"] < -1000).astype(int)
    keep = ["date"] + [c for c in df.columns if c.startswith(prefix + "_")]
    return df[keep]


def derive_and_save_features() -> int:
    """Build etf_flow_features.parquet (chimera source `etf_flows`) from the raw
    btc/eth ETF flow panels. Returns row count, or -1 if raw inputs missing."""
    btc_in, eth_in = OUT_DIR / "btc_etf_flows.parquet", OUT_DIR / "eth_etf_flows.parquet"
    if not btc_in.exists() or not eth_in.exists():
        print(f"[etf_feat] WARN raw inputs missing ({btc_in.name}/{eth_in.name}); "
              f"skipping feature derivation", flush=True)
        return -1
    btc, eth = pd.read_parquet(btc_in), pd.read_parquet(eth_in)
    eth_bk = next((c for c in eth.columns if "Blackrock" in c and "ETHA" in c), "ETHA")
    btc_f = _build_etf_one(btc, "btc_etf", "IBIT")
    eth_f = _build_etf_one(eth, "eth_etf", eth_bk)
    merged = btc_f.merge(eth_f, on="date", how="outer").sort_values("date").reset_index(drop=True)
    if "btc_etf_inflow_shock" in merged and "eth_etf_inflow_shock" in merged:
        merged["any_inflow_shock"] = ((merged["btc_etf_inflow_shock"].fillna(0) == 1) |
                                       (merged["eth_etf_inflow_shock"].fillna(0) == 1)).astype(int)
        merged["both_inflow_shock"] = ((merged["btc_etf_inflow_shock"].fillna(0) == 1) &
                                        (merged["eth_etf_inflow_shock"].fillna(0) == 1)).astype(int)
    atomic_write_parquet(merged, FEAT_PATH, required_cols={"date", "btc_etf_total_z30"})
    print(f"[etf_feat] saved: {FEAT_PATH} ({len(merged)} rows, {len(merged.columns)} cols)", flush=True)
    return len(merged)


def main():
    # G-AUDIT-027: parse_args (strict) instead of parse_known_args. Previously
    # silently accepted any flag the orchestrator passed (--universe, --asset,
    # ...) -- safe today (DAG declares neither for this stage) but a brittle
    # default if the DAG metadata changes.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Delete existing ETF panels and rebuild from Farside.")
    ap.add_argument("--recheck-missing", action="store_true",
                    help="Bypass the confirmed_missing manifest and re-attempt "
                         "the Farside fetch even if it was recently unreachable.")
    args = ap.parse_args()

    if args.force:
        n_deleted = 0
        for a in ["btc", "eth"]:
            p = OUT_DIR / f"{a}_etf_flows.parquet"
            if p.exists():
                p.unlink()
                n_deleted += 1
        print(f"[etf] [force] deleted {n_deleted} prior ETF panels; "
              f"rebuilding from Farside", flush=True)

    n_ok = 0
    failures: list[tuple[str, str]] = []
    for a in ["btc", "eth"]:
        try:
            fetch_and_save(a, recheck_missing=args.recheck_missing)
            n_ok += 1
            time.sleep(3)
        except Exception as e:
            _pl("FAIL", f"{a}: FAILED: {e}")
            failures.append((a, str(e)))
    print(f"\n[etf] ingest done: {n_ok} ok / {len(failures)} fail")
    if n_ok == 0:
        # All assets failed -> orchestrator must see a non-zero exit so the
        # downstream stage doesn't silently use stale ETF flows.
        for a, err in failures:
            print(f"  FAIL: {a}: {err}")
        sys.exit(2)
    # Derive the chimera-consumed feature file from the raw flows.
    n_feat = derive_and_save_features()
    if n_feat < 0:
        # raw fetched but feature derivation could not run -> chimera input stale.
        print("[etf] ERROR: feature derivation skipped (missing raw inputs)", flush=True)
        sys.exit(2)
    if failures:
        # Partial failure: warn (rc=1) so the orchestrator can flag it.
        sys.exit(1)


if __name__ == "__main__":
    main()
