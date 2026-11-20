"""Alpha turn-011: A7 — Liquidation cascade 3-7d mean-reversion, U50 balanced.

Hypothesis: after a large liquidation cascade, forced selling exhausts in 2-4
days, then 3-7d forward return turns positive as real buyers return. Prior
1d bounce was dead (Family B capitulation, 11% hit); this tests longer
horizon.

Setup (U50 balanced per constitution):
  - Per-asset daily liq_total_usd series
  - Trigger: day's liq > 3x trailing 20-day mean (cascade spike)
  - Entry: 2 days AFTER cascade day (allow forced-selling to complete)
  - Horizons: 3d, 5d, 7d post-entry
  - Forward return from daily close; D1 long-only spot
  - 20bps round-trip cost
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.strategy.universe import UNIVERSE_50_LIQUID

OUT = ROOT / "logs" / "frontier" / "a7_liq_cascade" / "a7_liq_cascade_U50.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

CASCADE_LOOKBACK = 20
CASCADE_MULT = 3.0
DELAY_DAYS = 2     # enter 2d post-spike
COST_PCT = 0.0010


def load_liq() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "frontier" / "liquidations" / "liq_daily_approx.parquet")


def load_daily_price(asset: str) -> pd.DataFrame | None:
    cache = ROOT / "logs" / "frontier" / "cycle_gate" / f"{asset.lower()}usdt_daily_klines.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return None


def probe(asset: str, liq: pd.DataFrame) -> dict:
    liq_a = liq[liq["asset"] == asset].copy().sort_values("date").reset_index(drop=True)
    if len(liq_a) < 50:
        return {"asset": asset, "error": f"thin_liq:{len(liq_a)}"}
    px = load_daily_price(asset)
    if px is None:
        return {"asset": asset, "error": "no_daily_price"}

    liq_a["liq_ma20"] = liq_a["liq_total_usd"].rolling(CASCADE_LOOKBACK, min_periods=10).mean()
    liq_a["spike"] = liq_a["liq_total_usd"] > (CASCADE_MULT * liq_a["liq_ma20"])
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    liq_a["date"] = pd.to_datetime(liq_a["date"]).dt.normalize()
    df = liq_a.merge(px[["date", "close"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        return {"asset": asset, "error": f"thin_joined:{len(df)}"}

    # Entry 2d after spike day
    df["spike_lag2"] = df["spike"].shift(DELAY_DAYS).fillna(False).astype(bool)
    # Forward returns from entry day
    for h in (3, 5, 7):
        df[f"fwd_{h}d"] = df["close"].shift(-h) / df["close"] - 1.0

    trig = df[df["spike_lag2"]].copy()
    if len(trig) < 5:
        return {"asset": asset, "error": f"no_triggers:{len(trig)}"}

    out: dict = {"asset": asset, "n_triggers": int(len(trig))}
    for h in (3, 5, 7):
        col = f"fwd_{h}d"
        arr = (trig[col] - 2 * COST_PCT).dropna().values
        n = len(arr)
        if n < 5:
            continue
        mean = float(arr.mean())
        std = float(arr.std())
        t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        hit = float((arr > 0).mean())
        out[f"h{h}d"] = {
            "n": int(n), "mean_pct": mean * 100,
            "t_stat": float(t), "hit_rate": hit,
        }
    return out


def main() -> None:
    liq = load_liq()
    assets_with_liq = set(liq["asset"].unique())
    results = []
    ok = 0
    for asset in UNIVERSE_50_LIQUID:
        if asset not in assets_with_liq:
            # try with slightly different naming conventions
            alt_candidates = [asset, asset.lower(), asset.upper()]
            if not any(a in assets_with_liq for a in alt_candidates):
                results.append({"asset": asset, "error": "no_liq_data"})
                print(f"  {asset:>10s}: ERROR=no_liq_data")
                continue
        r = probe(asset, liq)
        results.append(r)
        if "error" in r:
            print(f"  {asset:>10s}: ERROR={r['error']}")
            continue
        ok += 1
        print(f"  {asset:>10s}: n_trig={r['n_triggers']:3d}", end="")
        for h in (3, 5, 7):
            if f"h{h}d" in r:
                e = r[f"h{h}d"]
                star = "*" if e["t_stat"] > 2 and e["mean_pct"] > 0 else " "
                print(f"  h{h}:{e['mean_pct']:+.2f}%(t={e['t_stat']:+.2f}){star}", end="")
        print()

    candidates = []
    for r in results:
        if "error" in r:
            continue
        for h in (3, 5, 7):
            if f"h{h}d" in r and r[f"h{h}d"]["t_stat"] > 2 and r[f"h{h}d"]["mean_pct"] > 0 and r[f"h{h}d"]["hit_rate"] > 0.5:
                candidates.append({"asset": r["asset"], "h": h, **r[f"h{h}d"]})

    print(f"\n=== SUMMARY: {ok}/{len(UNIVERSE_50_LIQUID)} U50 assets with data ===")
    print(f"Candidates (t>2 & mean>0 & hit>0.5): {len(candidates)}")
    for c in sorted(candidates, key=lambda x: -x["t_stat"]):
        print(f"  {c['asset']:<10s} h{c['h']}d  n={c['n']:3d}  mean={c['mean_pct']:+.3f}%  "
              f"t={c['t_stat']:+.2f}  hit={c['hit_rate']:.3f}")

    with open(OUT, "w") as f:
        json.dump({"results": results, "candidates": candidates}, f, indent=2)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
