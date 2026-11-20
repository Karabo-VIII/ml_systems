#!/usr/bin/env python3
"""MARKET RESEARCH -- WHEN opportunity appears (clustering) + long-only ASYMMETRY (characterization).

(1) Clustering: is the daily 'movers >=5%' count BURSTY (clustered in time) or uniform? lag-1 autocorrelation of
    the universe mover-count series + P(high-opportunity day | prev day high). Descriptive market structure.
(2) Asymmetry: across all asset-days, are big DOWN-moves bigger than big UP-moves (crypto crash skew)? per-asset
    P95(up-day return) vs P95(|down-day return|). Relevant to a LONG-ONLY lens.
Both are descriptive (no forward trading signal). Run: python scripts/research/timing_asymmetry.py
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


def main():
    by_date = defaultdict(dict)
    up95, dn95 = [], []
    for f in _files():
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).drop_nulls().sort("timestamp")
        except Exception:
            continue
        c = df["close"].to_numpy().astype(float); ts = df["timestamp"].to_numpy()
        if len(c) < 50:
            continue
        r = (c[1:] - c[:-1]) / c[:-1]
        ups = r[r > 0]; dns = r[r < 0]
        if len(ups) > 20 and len(dns) > 20:
            up95.append(float(np.percentile(ups, 95)))
            dn95.append(float(np.percentile(np.abs(dns), 95)))
        for i in range(1, len(c)):
            d = datetime.datetime.utcfromtimestamp(int(ts[i]) / 1000).date().isoformat()
            by_date[d][_asset(f)] = float(r[i - 1])

    # (1) clustering: daily mover-count series (>=10 assets present)
    dates = sorted(d for d, a in by_date.items() if len(a) >= 10)
    counts = np.array([sum(1 for v in by_date[d].values() if abs(v) >= T) for d in dates], dtype=float)
    ac1 = float(np.corrcoef(counts[:-1], counts[1:])[0, 1])
    hi = counts >= np.median(counts)
    p_hi_given_hi = float(np.mean(hi[1:][hi[:-1]]))  # P(high today | high yesterday)
    base_hi = float(np.mean(hi))

    # (2) asymmetry
    up_med = float(np.median(up95)); dn_med = float(np.median(dn95))
    print("CLUSTERING (when opportunity appears):")
    print(f"  lag-1 autocorr of daily mover-count = {ac1:.3f}  (>0 => bursty/clustered)")
    print(f"  P(high-opportunity day | prev high) = {p_hi_given_hi:.3f}  vs base rate {base_hi:.3f}")
    print(f"  n_days = {len(dates)}")
    print("ASYMMETRY (long-only relevant):")
    print(f"  median P95 UP-day move   = {up_med:.4f}")
    print(f"  median P95 DOWN-day move = {dn_med:.4f}")
    print(f"  down/up tail ratio = {dn_med/up_med:.3f}  (>1 => down-tails bigger, crash skew)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
