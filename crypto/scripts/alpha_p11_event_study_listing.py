"""Alpha turn-018: p11 Phase 1 event-study -- LISTING category.

Uses cached announcements (logs/frontier/announcements/listing_recent.parquet,
100% dated post-Bravo-turn-017-fix) and U50 daily kline cache
(logs/frontier/cycle_gate/).

For each listing announcement:
  - Extract affected token(s)
  - Look up daily close at ann_date, ann_date+1d, +3d, +7d, +14d
  - Compute raw forward returns net of 20bps round-trip cost

Categorize trades by TRAIN / VAL / OOS chronological split for paranoid-OOS
validation. Also run shuffle control (random entry dates on same universe).

Goal: is listing announcement a directional signal at 1d/3d/7d/14d windows
for tokens AFTER they already trade on Binance (p8 owns h1; this is day-scale
follow-through)?

NB: p8 listing h1 is already proven; this is the complementary PRE-trading
signal: Binance announces it will list X, and the question is whether X's
price responds (often not affected since X is not yet tradeable on Binance
until listing day). For now we test tokens that ARE in U50 (i.e., listings
of tokens that later became majors).
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

OUT = ROOT / "logs" / "frontier" / "p11_event_study" / "listing_event_study.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

COST_PCT = 0.0010  # 10 bps per side


def era(ts: pd.Timestamp) -> str:
    if ts < pd.Timestamp("2024-01-01"):
        return "TRAIN"
    if ts < pd.Timestamp("2025-01-01"):
        return "VAL"
    return "OOS"


def load_price(asset: str) -> pd.DataFrame | None:
    p = ROOT / "logs" / "frontier" / "cycle_gate" / f"{asset.lower()}usdt_daily_klines.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    ann = pd.read_parquet(ROOT / "logs" / "frontier" / "announcements" / "listing_recent.parquet")
    ann = ann[ann["release_ms"] > 0].copy()
    ann["release_ts"] = pd.to_datetime(ann["release_ms"], unit="ms")
    ann["ann_date"] = ann["release_ts"].dt.normalize()
    print(f"[DATA] listings: {len(ann)} with dates (range {ann['release_ts'].min().date()} -> {ann['release_ts'].max().date()})")

    # Pre-load price panels for U50
    prices: dict[str, pd.DataFrame] = {}
    for a in UNIVERSE_50_LIQUID:
        df = load_price(a)
        if df is not None:
            prices[a] = df

    # Build event list: for each ann, for each token in tokens column,
    # if token in U50 with available prices, emit a test-trade.
    events = []
    for _, row in ann.iterrows():
        tokens = row["tokens"] if isinstance(row["tokens"], (list, np.ndarray)) else []
        ann_date = row["ann_date"]
        for tok in tokens:
            if tok not in prices:
                continue
            df = prices[tok]
            # Find the first bar on/after ann_date
            mask = df["date"] >= ann_date
            if not mask.any():
                continue
            entry_idx = df[mask].index[0]
            # Ensure we have forward bars
            if entry_idx + 14 >= len(df):
                continue
            entry_price = float(df.at[entry_idx, "close"])
            forward_rets = {}
            for h in (1, 3, 7, 14):
                if entry_idx + h < len(df):
                    px = float(df.at[entry_idx + h, "close"])
                    forward_rets[f"fwd_{h}d"] = (px / entry_price) - 1.0 - 2 * COST_PCT
                else:
                    forward_rets[f"fwd_{h}d"] = np.nan
            events.append({
                "ann_date": ann_date,
                "token": tok,
                "era": era(ann_date),
                "entry_price": entry_price,
                **forward_rets,
            })

    if not events:
        print("[EMPTY] no events with U50-token overlap. Most listed tokens are non-U50.")
        with open(OUT, "w") as f:
            json.dump({"events": [], "summary": {}}, f, default=str)
        return

    ev = pd.DataFrame(events)
    print(f"[EVENTS] {len(ev)} test-trades across {ev['token'].nunique()} U50 tokens")
    print(f"  per-era counts: {ev['era'].value_counts().to_dict()}")
    print()
    print(f"  tokens hit: {sorted(ev['token'].unique())}")

    # Aggregate per era + horizon
    print()
    print(f"{'era':<8} {'hor':<6} {'n':>4} {'mean%':>8} {'std%':>8} {'t':>6} {'hit':>6}")
    out: dict = {"events": int(len(ev)), "tokens": sorted(ev["token"].unique().tolist())}
    for e in ("TRAIN", "VAL", "OOS", "ALL"):
        sub = ev if e == "ALL" else ev[ev["era"] == e]
        out[e] = {"n": int(len(sub))}
        if len(sub) < 3:
            continue
        for h in (1, 3, 7, 14):
            arr = sub[f"fwd_{h}d"].dropna().values
            if len(arr) < 3:
                continue
            mean = float(arr.mean())
            std = float(arr.std())
            t = mean / (std / np.sqrt(len(arr))) if std > 0 else 0.0
            hit = float((arr > 0).mean())
            out[e][f"h{h}d"] = {"n": int(len(arr)), "mean_pct": mean * 100,
                                "t_stat": t, "hit_rate": hit}
            print(f"{e:<8} h{h}d  {len(arr):>4d} {mean*100:>+7.3f} {std*100:>7.3f} {t:>+6.2f} {hit:>6.3f}")

    # Shuffle control
    rng = np.random.default_rng(42)
    print()
    print("[SHUFFLE CONTROL] same tokens, random dates -- fwd_7d")
    for token in ev["token"].unique()[:10]:
        df = prices[token]
        if len(df) < 30:
            continue
        # Random dates across the token's history
        n_samples = 100
        idxs = rng.integers(0, len(df) - 14, size=n_samples)
        rets = []
        for i in idxs:
            rets.append(df.at[i + 7, "close"] / df.at[i, "close"] - 1.0 - 2 * COST_PCT)
        shuf_mean = float(np.mean(rets))
        real_token = ev[ev["token"] == token]["fwd_7d"].dropna()
        real_mean = float(real_token.mean()) if len(real_token) > 0 else float("nan")
        print(f"  {token:<10s}  real_fwd_7d={real_mean*100:+.2f}% (n={len(real_token)})  shuffle_fwd_7d={shuf_mean*100:+.2f}%")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
