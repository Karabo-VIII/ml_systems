#!/usr/bin/env python3
"""MARKET RESEARCH r3 -- is the opportunity PERSISTENT or regime-bound?

Re-derives movers/day (>=5% raw daily move) split by YEAR and by BTC REGIME (bull = BTC trailing-30d return > 0,
bear = < 0). If the opportunity only exists in bull markets, the 'case' is fragile. If it persists in bear/chop,
it is durable. Run: python scripts/research/regime_variation.py
No emoji (Windows cp1252).
"""
import datetime
import glob
import json
import os
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THR = 0.05


def _files():
    return sorted(glob.glob(os.path.join(ROOT, "data", "processed", "chimera", "1d", "*.parquet")))


def _asset(p):
    return os.path.basename(p).split("_v51_chimera_")[0].upper()


def _close_by_date(path):
    try:
        df = pl.read_parquet(path, columns=["timestamp", "close"]).drop_nulls().sort("timestamp")
    except Exception:
        return {}
    c = df["close"].to_numpy().astype(float)
    ts = df["timestamp"].to_numpy()
    return {datetime.datetime.utcfromtimestamp(int(ts[i]) / 1000).date().isoformat(): float(c[i]) for i in range(len(c))}


def main():
    files = _files()
    btc_path = next((f for f in files if _asset(f) == "BTCUSDT"), None)
    btc_close = _close_by_date(btc_path)
    btc_dates = sorted(btc_close)
    # BTC trailing-30d return -> regime per date
    regime = {}
    for i, d in enumerate(btc_dates):
        if i >= 30:
            prev = btc_close[btc_dates[i - 30]]
            regime[d] = "bull" if (btc_close[d] - prev) / prev > 0 else "bear"

    # per-date abs daily returns across assets
    daily_abs = defaultdict(dict)
    for f in files:
        a = _asset(f)
        cbd = _close_by_date(f)
        dates = sorted(cbd)
        for i in range(1, len(dates)):
            d0, d1 = dates[i - 1], dates[i]
            if cbd[d0] > 0:
                daily_abs[d1][a] = abs(cbd[d1] - cbd[d0]) / cbd[d0]

    by_year = defaultdict(list)
    by_regime = defaultdict(list)
    for d, assets in daily_abs.items():
        if len(assets) < 10:
            continue
        movers = sum(1 for v in assets.values() if v >= THR)
        by_year[d[:4]].append(movers)
        if d in regime:
            by_regime[regime[d]].append(movers)

    out = {"threshold": THR, "metric": "assets moving >=5% per day (raw)",
           "by_year": {y: {"avg_movers": round(float(np.mean(v)), 1), "n_days": len(v)} for y, v in sorted(by_year.items())},
           "by_btc_regime": {r: {"avg_movers": round(float(np.mean(v)), 1), "n_days": len(v),
                                 "pct_days_>=1_mover": round(float(np.mean([1 if m >= 1 else 0 for m in v])), 3)}
                             for r, v in by_regime.items()}}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, "runs", "research", "regime_variation.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
