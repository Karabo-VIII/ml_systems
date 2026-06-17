"""DeFiLlama stablecoin supply ingestion — FREE, no API key.

Endpoints:
    GET https://stablecoins.llama.fi/stablecoincharts/all
        -> aggregate daily stablecoin market cap (USD)
    GET https://stablecoins.llama.fi/stablecoin/{id}
        -> per-stablecoin historical supply
        ids: USDT=1, USDC=2, USDe=146, DAI=5

Produces:
    data/frontier/defillama/stable_flows_daily.parquet
        date (datetime), total_usd, usdt_usd, usdc_usd, usde_usd, dai_usd

Rate limit: ~1 req/sec safe. We make 4 requests total (one-shot pull).
Cadence: refresh daily — run before backtest.

Hypothesis (Griffin & Shams 2020): stablecoin issuance precedes BTC rallies.
Large USDT mints >$500M in 24h have ~2x base-rate of +3% BTC moves in next
72h. Aggregate daily issuance z-score > +2 is the stronger signal — filters
noise and captures regime changes when demand for USD on-ramp spikes.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "stable_flows_daily.parquet"

BASE = "https://stablecoins.llama.fi"
STABLE_IDS = {
    "usdt": 1,
    "usdc": 2,
    "usde": 146,
    "dai": 5,
}

UA = "ml_systems-frontier/1.0 (free-tier; research)"


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
    """Parse both aggregate (`totalCirculatingUSD`) and per-stable (`circulating`) rows."""
    rows = []
    for r in js:
        try:
            ts = int(r.get("date", 0))
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        # per-stablecoin: 'circulating.peggedUSD'
        circ = None
        c = r.get("circulating")
        if isinstance(c, dict):
            circ = c.get("peggedUSD")
        # aggregate: 'totalCirculatingUSD.peggedUSD' (preferred) or 'totalCirculating.peggedUSD'
        if circ is None:
            tcu = r.get("totalCirculatingUSD")
            if isinstance(tcu, dict):
                circ = tcu.get("peggedUSD")
        if circ is None:
            tc = r.get("totalCirculating")
            if isinstance(tc, dict):
                circ = tc.get("peggedUSD")
        if circ is None:
            continue
        rows.append({"date": pd.to_datetime(ts, unit="s").normalize(), col: float(circ)})
    return pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def fetch_all() -> pd.DataFrame:
    print("[defillama] fetching aggregate...", flush=True)
    agg_js = _fetch_json(f"{BASE}/stablecoincharts/all")
    agg = _parse_chart(agg_js, "total_usd")
    print(f"[defillama]   aggregate: {len(agg)} days, latest ${agg['total_usd'].iloc[-1]/1e9:.1f}B")

    merged = agg
    for name, sid in STABLE_IDS.items():
        print(f"[defillama] fetching {name} (id={sid})...", flush=True)
        time.sleep(0.5)
        try:
            j = _fetch_json(f"{BASE}/stablecoin/{sid}")
            tokens = j.get("tokens") if isinstance(j, dict) else j
            if tokens is None:
                print(f"[defillama]   {name}: no tokens field, skipping")
                continue
            df = _parse_chart(tokens, f"{name}_usd")
            merged = merged.merge(df, on="date", how="outer")
            latest = df[f"{name}_usd"].iloc[-1] if len(df) else 0.0
            print(f"[defillama]   {name}: {len(df)} days, latest ${latest/1e9:.1f}B")
        except Exception as e:
            print(f"[defillama]   {name}: FAILED ({e}), continuing")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill().bfill()
    return merged


def main():
    df = fetch_all()
    df.to_parquet(OUT_PATH, index=False)
    print(f"[defillama] saved: {OUT_PATH}")
    print(f"[defillama] range: {df['date'].min().date()} -> {df['date'].max().date()} ({len(df)} days)")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
