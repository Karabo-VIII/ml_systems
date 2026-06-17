#!/usr/bin/env python3
"""MARKET RESEARCH r3b -- the LONG-ONLY skew: how much of 'movers/day' is UP vs DOWN?

Our constraint is LO+spot+lev=1 -> only UP-moves are directly harvestable. The symmetric |move| count
over-states the long-only opportunity, especially in bear markets. This splits movers/day into UP (ret>=+t)
vs DOWN (ret<=-t), by BTC regime + year. Run: python scripts/research/long_only_skew.py
No emoji (Windows cp1252).
"""
import datetime
import glob
import os
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = 0.05


def _files():
    return sorted(glob.glob(os.path.join(ROOT, "data", "processed", "chimera", "1d", "*.parquet")))


def _asset(p):
    return os.path.basename(p).split("_v51_chimera_")[0].upper()


def _close_by_date(path):
    try:
        df = pl.read_parquet(path, columns=["timestamp", "close"]).drop_nulls().sort("timestamp")
    except Exception:
        return {}
    c = df["close"].to_numpy().astype(float); ts = df["timestamp"].to_numpy()
    return {datetime.datetime.utcfromtimestamp(int(ts[i]) / 1000).date().isoformat(): float(c[i]) for i in range(len(c))}


def main():
    files = _files()
    btc = _close_by_date(next(f for f in files if _asset(f) == "BTCUSDT"))
    bdates = sorted(btc)
    regime = {d: ("bull" if (btc[d] - btc[bdates[i - 30]]) / btc[bdates[i - 30]] > 0 else "bear")
              for i, d in enumerate(bdates) if i >= 30}

    up_by, dn_by = defaultdict(dict), defaultdict(dict)
    for f in files:
        a = _asset(f); cbd = _close_by_date(f); ds = sorted(cbd)
        for i in range(1, len(ds)):
            if cbd[ds[i - 1]] > 0:
                r = (cbd[ds[i]] - cbd[ds[i - 1]]) / cbd[ds[i - 1]]
                up_by[ds[i]][a] = r
    # per-date up/down counts
    rows = []
    for d, assets in up_by.items():
        if len(assets) < 10:
            continue
        up = sum(1 for r in assets.values() if r >= T)
        dn = sum(1 for r in assets.values() if r <= -T)
        rows.append((d, up, dn))

    def summ(filt):
        ups = [u for d, u, dn in rows if filt(d)]
        dns = [dn for d, u, dn in rows if filt(d)]
        if not ups:
            return None
        return {"avg_up": round(float(np.mean(ups)), 1), "avg_down": round(float(np.mean(dns)), 1),
                "up_share": round(float(np.mean(ups) / (np.mean(ups) + np.mean(dns) + 1e-9)), 3), "n_days": len(ups)}

    print(f"LONG-ONLY skew at +/-{T*100:.0f}% daily (UP = harvestable for LO+spot):")
    print("  ALL     :", summ(lambda d: True))
    print("  BULL    :", summ(lambda d: regime.get(d) == "bull"))
    print("  BEAR    :", summ(lambda d: regime.get(d) == "bear"))
    for y in ["2021", "2022", "2023", "2024", "2025", "2026"]:
        print(f"  {y}    :", summ(lambda d, y=y: d.startswith(y)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
