"""DeFiLlama stablecoin supply ingest + flow-feature derivation -- canonical producer.

FREE, no API key. Two-step in one module:
  1. FETCH raw daily stablecoin market cap (aggregate + USDT/USDC/USDe/DAI) from
     stablecoins.llama.fi  -> raw_external/defillama/stable_flows_daily.parquet
  2. DERIVE flow features (deltas, 30d z-scores, mint/crash shock flags) the chimera
     consumes -> raw_external/defillama/stable_flow_features.parquet (registry source
     `stable_flow`, prefix stbl_).

Hypothesis (Griffin & Shams 2020): stablecoin issuance precedes BTC rallies; a
mint z-score > +2 is the regime signal.

Provenance: restored 2026-05-29 by merging the two archived steps
(src/_archive/frontier/ingest/defillama_stable_flows.py +
src/_archive/frontier/features/stable_flow_signals.py), which were orphaned when
src/frontier was archived (chimera `stbl_*` features went ~5wk stale). z-score uses
shift(1) before the rolling window (no look-ahead).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
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
    "inputs": ["stablecoins.llama.fi /stablecoincharts/all + /stablecoin/{id}"],
    "outputs": [
        "data/raw_external/defillama/stable_flows_daily.parquet (raw)",
        "data/raw_external/defillama/stable_flow_features.parquet (derived; chimera source)",
    ],
    "invariants": [
        "global layout (date only, no asset)",
        "30d z-score uses shift(1) before rolling (no look-ahead)",
        "atomic_write_via_parquet_io",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "raw_external" / "defillama"
RAW_PATH = OUT_DIR / "stable_flows_daily.parquet"
FEAT_PATH = OUT_DIR / "stable_flow_features.parquet"

BASE = "https://stablecoins.llama.fi"
STABLE_IDS = {"usdt": 1, "usdc": 2, "usde": 146, "dai": 5}
UA = "v4-pipeline/1.0 (free-tier; research)"


def _fetch_json(url: str, retries: int = 3) -> Any:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(1.5 ** i)
    raise RuntimeError(f"fetch failed: {url}: {last_err}")


def _parse_chart(js: list[dict], col: str) -> pd.DataFrame:
    rows = []
    for r in js:
        try:
            ts = int(r.get("date", 0))
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        circ = None
        c = r.get("circulating")
        if isinstance(c, dict):
            circ = c.get("peggedUSD")
        if circ is None and isinstance(r.get("totalCirculatingUSD"), dict):
            circ = r["totalCirculatingUSD"].get("peggedUSD")
        if circ is None and isinstance(r.get("totalCirculating"), dict):
            circ = r["totalCirculating"].get("peggedUSD")
        if circ is None:
            continue
        rows.append({"date": pd.to_datetime(ts, unit="s").normalize(), col: float(circ)})
    return pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def fetch_raw() -> pd.DataFrame:
    print("[defillama] fetching aggregate...", flush=True)
    merged = _parse_chart(_fetch_json(f"{BASE}/stablecoincharts/all"), "total_usd")
    for name, sid in STABLE_IDS.items():
        time.sleep(0.5)
        try:
            j = _fetch_json(f"{BASE}/stablecoin/{sid}")
            tokens = j.get("tokens") if isinstance(j, dict) else j
            if tokens is None:
                continue
            merged = merged.merge(_parse_chart(tokens, f"{name}_usd"), on="date", how="outer")
        except Exception as e:
            print(f"[defillama]   {name}: FAILED ({e}); continuing", flush=True)
    return merged.sort_values("date").reset_index(drop=True).ffill().bfill()


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    for col in ["total_usd", "usdt_usd", "usdc_usd", "usde_usd", "dai_usd"]:
        if col not in df.columns:
            continue
        base = col.replace("_usd", "")
        df[f"{base}_delta_1d_usd"] = df[col].diff()
        df[f"{base}_delta_1d_pct"] = df[col].pct_change()
        df[f"{base}_delta_7d_pct"] = df[col].pct_change(7)
        df[f"{base}_delta_30d_pct"] = df[col].pct_change(30)
        delta = df[f"{base}_delta_1d_usd"]
        rmean = delta.shift(1).rolling(30, min_periods=10).mean()  # shift(1): no look-ahead
        rstd = delta.shift(1).rolling(30, min_periods=10).std()
        df[f"{base}_zscore_30d"] = (delta - rmean) / rstd.replace(0, np.nan)
        df[f"{base}_delta_1d_logusd"] = np.sign(delta) * np.log1p(np.abs(delta))
    if "total_zscore_30d" in df.columns:
        df["stable_shock"] = (df["total_zscore_30d"] > 2.0).astype(int)
        df["stable_crash"] = (df["total_zscore_30d"] < -2.0).astype(int)
        df["stable_shock_strong"] = (df["total_zscore_30d"] > 3.0).astype(int)
        df["stable_pos_regime"] = (df["total_delta_7d_pct"] > 0.005).astype(int)
    if "usdt_zscore_30d" in df.columns:
        df["usdt_shock"] = (df["usdt_zscore_30d"] > 2.0).astype(int)
        df["usdt_shock_strong"] = (df["usdt_zscore_30d"] > 3.0).astype(int)
    if "stable_shock" in df.columns and "usdt_shock" in df.columns:
        df["compound_shock"] = ((df["stable_shock"] == 1) & (df["usdt_shock"] == 1)).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description="Fetch DeFiLlama stablecoin flows + derive -> raw_external/defillama/")
    ap.add_argument("--force", action="store_true", help="No-op (always full refetch)")
    ap.add_argument("--workers", type=int, default=1, help="No-op; accepted for refresh.py")
    ap.add_argument("--dry-run", action="store_true", help="Fetch + derive + report, do not write")
    args = ap.parse_args()
    raw = fetch_raw()
    if raw.empty:
        print("[defillama] ERROR: no data fetched; output unchanged", flush=True)
        raise SystemExit(2)
    feats = derive_features(raw)
    if args.dry_run:
        print(f"[defillama] DRY-RUN: raw {len(raw)} rows, features {len(feats)} rows x "
              f"{len(feats.columns)} cols", flush=True)
        return
    atomic_write_parquet(raw, RAW_PATH, required_cols={"date", "total_usd"})
    atomic_write_parquet(feats, FEAT_PATH, required_cols={"date", "total_zscore_30d"})
    print(f"[defillama] saved raw={RAW_PATH.name} + features={FEAT_PATH.name} "
          f"({raw['date'].min().date()} -> {raw['date'].max().date()})", flush=True)


if __name__ == "__main__":
    main()
