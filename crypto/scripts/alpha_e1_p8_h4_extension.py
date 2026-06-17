"""Alpha turn-011: E1 — P8 listing h4 sub-horizon extension.

P8 listing h1-momentum is already OOS-validated + shipped. This extends the
signal to h4 sub-horizon (4h post-listing) to see if a second trade can be
stacked on the same listing event.

Hypothesis: the listing-pop dynamic compresses into h1, mean-reverts in h4-h24,
then stabilizes. If h4 return has a systematic sign (positive OR negative), a
h1 -> h4 sequence strategy could capture two trades per event.

Setup:
  - Read logs/p1_listing_wf_results.csv (405 events, 2024-2026)
  - Compute per-event h4_minus_h1 = ret_4h - ret_1h (the h1-to-h4 leg)
  - Split by chronological era (TRAIN/VAL/OOS) matching p1_listing_oos.py
  - Compute mean, t-stat, hit rate on h4-h1 leg
  - Test as standalone strategy: enter at h1 peak (long if h1 > 0), exit at h4

Also tests "long-always-at-h1" and "hold-and-ride-to-h4" vs "flip-to-short-at-
h1-exit". D1: spot only, no short. So effectively we test:
  - Stack 1: BUY at t0 (listing), SELL at h1  (existing P8)
  - Stack 2: BUY at h4 (if h4 mean-reversion is up), SELL at h24
  OR
  - Combined: BUY at t0, hold to h4 (skip h1 exit)

U50 note: we don't gate by U50 here because listings ARE the universe for
this strategy; all 405 events are eligible. This is a single-event-family
strategy per the U50 exception.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "frontier" / "e1_p8_h4" / "e1_h4_extension.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

TAKER_RT = 0.0016
MAKER_RT = 0.0004


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "logs" / "p1_listing_wf_results.csv")
    df["date"] = pd.to_datetime(df["onboard_date"])
    for h in (1, 4, 24, 48, 72, 168):
        df[f"ret_{h}h_gross"] = pd.to_numeric(df[f"ret_{h}h_gross"], errors="coerce")
    return df


def era(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2025-01-01"):
        return "TRAIN 2024"
    if date < pd.Timestamp("2025-10-01"):
        return "VAL 2025 H1-H3"
    return "OOS 2025Q4-26"


def analyze(df: pd.DataFrame, cost: float = MAKER_RT) -> dict:
    df = df.copy()
    df["era"] = df["date"].apply(era)
    df = df[df["ret_1h_gross"].notna() & df["ret_4h_gross"].notna()].reset_index(drop=True)
    df["h1_to_h4_raw"] = df["ret_4h_gross"] - df["ret_1h_gross"]
    df["t0_to_h4_net"] = df["ret_4h_gross"] - cost
    df["t0_to_h1_net"] = df["ret_1h_gross"] - cost
    # Stack: exit at h1 (net r1 - cost), re-enter for h4 leg (net h1_to_h4 - cost)
    df["stack_net"] = df["t0_to_h1_net"] + (df["h1_to_h4_raw"] - cost)
    # Combined (single entry, hold to h4): one cost round-trip
    df["combined_t0_to_h4_net"] = df["ret_4h_gross"] - cost

    strategies = ["t0_to_h1_net", "combined_t0_to_h4_net", "stack_net"]
    out = {"cost_bps": cost * 10000, "n_total": int(len(df))}
    for e in ["TRAIN 2024", "VAL 2025 H1-H3", "OOS 2025Q4-26", "ALL"]:
        sub = df if e == "ALL" else df[df["era"] == e]
        out[e] = {"n": int(len(sub))}
        if len(sub) < 5:
            continue
        for s in strategies:
            arr = sub[s].dropna().values
            n = len(arr)
            if n < 5:
                continue
            mean = arr.mean()
            std = arr.std()
            t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
            hit = float((arr > 0).mean())
            out[e][s] = {
                "n": int(n),
                "mean_pct": float(mean * 100),
                "t_stat": float(t),
                "hit_rate": hit,
            }
    return out


def main() -> None:
    df = load()
    res_maker = analyze(df, cost=MAKER_RT)
    res_taker = analyze(df, cost=TAKER_RT)

    import json
    with open(OUT, "w") as f:
        json.dump({"maker": res_maker, "taker": res_taker}, f, indent=2)

    # Pretty print key findings
    print("=" * 90)
    print("E1 P8 h4-extension analysis")
    print("=" * 90)
    for label, res in [("MAKER 4bps RT", res_maker), ("TAKER 16bps RT", res_taker)]:
        print(f"\n[{label}]  n_total={res['n_total']}")
        for era_name in ["TRAIN 2024", "VAL 2025 H1-H3", "OOS 2025Q4-26", "ALL"]:
            e = res[era_name]
            if "t0_to_h1_net" not in e:
                continue
            print(f"  {era_name} (n={e['n']}):")
            for strat in ("t0_to_h1_net", "combined_t0_to_h4_net", "stack_net"):
                if strat not in e:
                    continue
                s = e[strat]
                sig = "  *SHIP*" if s["t_stat"] > 2 and s["mean_pct"] > 0 else ""
                print(f"    {strat:<26} n={s['n']:3d}  mean={s['mean_pct']:+.3f}%  "
                      f"t={s['t_stat']:+.2f}  hit={s['hit_rate']:.3f}{sig}")
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
