"""Deribit DVOL index ingest (free public API, no key) -- canonical pipeline producer.

DVOL = Deribit's implied-volatility index (crypto VIX), computed from the full
BTC/ETH options chain. Daily resolution: [ts_ms, open, high, low, close].

Output (registry source `dvol`, feature_registry.yaml):
    data/raw_external/deribit/dvol_daily.parquet
        columns: date, asset, dvol_open, dvol_high, dvol_low, dvol_close
        coverage: BTC + ETH only (Deribit options only exist for these)

Provenance: restored 2026-05-29 from src/_archive/frontier/ingest/deribit_dvol.py
(orphaned when src/frontier was archived; chimera `dv_*` features had gone ~5wk
stale because nothing in the refresh DAG fetched it). Now writes the registry
path, uses atomic_write_parquet, and carries the canonical CLI so refresh.py can
orchestrate it.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# Path bootstrap: refresh.py runs this as a direct script, so src/ is not on
# sys.path. Mirror the canonical producer pattern (etf_flows.py) before importing
# the pipeline framework. Fixes ModuleNotFoundError on the 2026-05-30 refresh run.
import sys
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "external_ingest",
    "inputs": ["deribit.com /api/v2/public/get_volatility_index_data (BTC, ETH)"],
    "outputs": ["data/raw_external/deribit/dvol_daily.parquet"],
    "invariants": [
        "schema: date, asset, dvol_open, dvol_high, dvol_low, dvol_close",
        "coverage: BTC+ETH only (by design; Deribit options limited to these)",
        "atomic_write_via_parquet_io",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = PROJECT_ROOT / "data" / "raw_external" / "deribit" / "dvol_daily.parquet"
UA = "v4-pipeline/1.0 (research)"


def fetch_dvol(currency: str, start_ms: int, end_ms: int) -> list[list]:
    url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?"
           + urllib.parse.urlencode({
               "currency": currency,
               "start_timestamp": start_ms,
               "end_timestamp": end_ms,
               "resolution": "1D",
           }))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data.get("result", {}).get("data", [])
    except Exception as e:
        print(f"  [{currency}] failed: {e}", flush=True)
        return []


def build_panel(start: str = "2021-01-01") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    frames = []
    for currency in ["BTC", "ETH"]:
        rows: list[list] = []
        cur = start_ms
        while cur < end_ms:
            chunk_end = min(cur + int(1.5 * 365 * 86400 * 1000), end_ms)  # 1.5y chunks
            batch = fetch_dvol(currency, cur, chunk_end)
            if not batch:
                break
            rows.extend(batch)
            cur = batch[-1][0] + 86400 * 1000
            time.sleep(0.5)
        if not rows:
            print(f"[dvol] {currency}: no data", flush=True)
            continue
        df = pd.DataFrame(rows, columns=["ts_ms", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])
        df["date"] = pd.to_datetime(
            df["ts_ms"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
        df["asset"] = currency
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        print(f"[dvol] {currency}: {len(df)} daily rows, "
              f"{df['date'].min().date()} -> {df['date'].max().date()}", flush=True)
        frames.append(df[["date", "asset", "dvol_open", "dvol_high", "dvol_low", "dvol_close"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Fetch Deribit DVOL -> raw_external/deribit/")
    ap.add_argument("--start", default="2021-01-01", help="History start (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true", help="No-op (always full refetch); accepted for refresh.py")
    ap.add_argument("--workers", type=int, default=1, help="No-op (single API source); accepted for refresh.py")
    ap.add_argument("--dry-run", action="store_true", help="Fetch + report, do not write")
    args = ap.parse_args()
    panel = build_panel(start=args.start)
    if panel.empty:
        print("[dvol] ERROR: no data fetched (Deribit unreachable?); output unchanged", flush=True)
        raise SystemExit(2)
    if args.dry_run:
        print(f"[dvol] DRY-RUN: would write {len(panel)} rows to {OUT_PATH}", flush=True)
        return
    atomic_write_parquet(panel, OUT_PATH, required_cols={"date", "asset", "dvol_close"})
    print(f"[dvol] saved: {OUT_PATH} ({len(panel)} rows)", flush=True)


if __name__ == "__main__":
    main()
