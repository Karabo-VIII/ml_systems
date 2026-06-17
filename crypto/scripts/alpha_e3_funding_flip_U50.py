"""Alpha turn-011: E3 — Funding-flip event trade probe, U50 balanced.

Trigger: per-asset perp funding rate flips from POSITIVE to NEGATIVE at a
settlement. D1 compliant: spot-only, BUY on flip-to-negative (shorts pay longs),
ignore flip-to-positive (would need short).

Hypothesis: the flip moment is a capitulation signal (retail shorts crowded,
longs getting paid, prices often rally 8-24h post-flip as shorts cover).

Setup (U50 balanced per new constitution rule):
  - For each U50 asset, compute daily funding rate sign
  - Trigger: today's fund < 0 AND yesterday's fund >= 0 (flip-to-negative)
  - Forward return: spot price tomorrow-ish (use next-day return as proxy
    since we only have daily resolution in funding_panel_daily.parquet)
  - Horizons: 1d, 3d, 5d, 10d
  - Report per-asset t-stat, hit rate, n_trades
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

OUT = ROOT / "logs" / "frontier" / "e3_funding_flip" / "e3_funding_flip_U50.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

COST_PCT = 0.0010  # 10 bps per side


def load_funding() -> pd.DataFrame:
    p = ROOT / "data" / "frontier" / "funding" / "funding_panel_daily.parquet"
    return pd.read_parquet(p)


def load_daily_price(asset: str) -> pd.DataFrame | None:
    """Load daily BTCUSDT-style klines from the cycle_gate cache."""
    cache = ROOT / "logs" / "frontier" / "cycle_gate" / f"{asset.lower()}usdt_daily_klines.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    # Some U50 assets aren't in the cycle_gate cache (only 10 majors were fetched).
    # We'd need to fetch them; for MVP, skip.
    return None


def probe(asset: str, fund_col: str, fund_df: pd.DataFrame) -> dict:
    px = load_daily_price(asset)
    if px is None:
        return {"asset": asset, "error": "no_daily_price"}

    fund_df = fund_df[["date", fund_col]].copy()
    fund_df = fund_df.dropna(subset=[fund_col]).reset_index(drop=True)
    fund_df["fund_prev"] = fund_df[fund_col].shift(1)
    fund_df["flip_neg"] = (fund_df[fund_col] < 0) & (fund_df["fund_prev"] >= 0)

    # Join with daily price
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    fund_df["date"] = pd.to_datetime(fund_df["date"]).dt.normalize()
    df = fund_df.merge(px[["date", "close"]], on="date", how="inner").sort_values("date")
    if len(df) < 100:
        return {"asset": asset, "error": f"thin_joined:{len(df)}"}

    # Forward returns
    for h in (1, 3, 5, 10):
        df[f"fwd_{h}d"] = df["close"].shift(-h) / df["close"] - 1.0

    trig = df[df["flip_neg"]].copy()
    if len(trig) < 5:
        return {"asset": asset, "error": f"no_triggers:{len(trig)}"}

    out: dict = {"asset": asset, "n_triggers": int(len(trig))}
    for h in (1, 3, 5, 10):
        col = f"fwd_{h}d"
        arr = (trig[col] - 2 * COST_PCT).dropna().values
        n = len(arr)
        if n < 5:
            continue
        mean = float(arr.mean())
        std = float(arr.std())
        t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        hit = float((arr > 0).mean())
        out[f"h{h}"] = {
            "n": int(n), "mean_pct": mean * 100, "t_stat": float(t),
            "hit_rate": hit,
        }
    return out


def main() -> None:
    fund = load_funding()
    results = []
    asset_to_col = {s: f"{s.lower()}_fund" for s in UNIVERSE_50_LIQUID}
    missing_col = []
    ok_count = 0

    for asset, col in asset_to_col.items():
        if col not in fund.columns:
            missing_col.append(asset)
            results.append({"asset": asset, "error": "no_fund_column"})
            continue
        r = probe(asset, col, fund)
        results.append(r)
        if "error" in r:
            print(f"  {asset:>10s}: ERROR={r['error']}")
            continue
        ok_count += 1
        best_h = max([h for h in (1, 3, 5, 10) if f"h{h}" in r],
                     key=lambda h: r[f"h{h}"]["t_stat"], default=None)
        print(f"  {asset:>10s}: n_trig={r['n_triggers']:3d}", end="")
        for h in (1, 3, 5, 10):
            if f"h{h}" in r:
                e = r[f"h{h}"]
                star = "*" if e["t_stat"] > 2 and e["mean_pct"] > 0 else " "
                print(f"  h{h}:{e['mean_pct']:+.2f}%(t={e['t_stat']:+.2f}){star}", end="")
        print()

    # Aggregate: candidates where ANY horizon has t>2 + mean>0 + hit>0.5
    candidates = []
    for r in results:
        if "error" in r:
            continue
        for h in (1, 3, 5, 10):
            if f"h{h}" in r and r[f"h{h}"]["t_stat"] > 2 and r[f"h{h}"]["mean_pct"] > 0 and r[f"h{h}"]["hit_rate"] > 0.5:
                candidates.append({"asset": r["asset"], "h": h, **r[f"h{h}"]})

    print(f"\n=== SUMMARY: {ok_count}/{len(UNIVERSE_50_LIQUID)} U50 assets with data ===")
    print(f"Candidates (t>2 & mean>0 & hit>0.5 at any horizon): {len(candidates)}")
    if candidates:
        for c in sorted(candidates, key=lambda x: -x["t_stat"]):
            print(f"  {c['asset']:<10s} h{c['h']:<2d}  n={c['n']:3d}  mean={c['mean_pct']:+.3f}%  "
                  f"t={c['t_stat']:+.2f}  hit={c['hit_rate']:.3f}")

    with open(OUT, "w") as f:
        json.dump({"results": results, "candidates": candidates,
                   "missing_fund_col": missing_col}, f, indent=2)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
