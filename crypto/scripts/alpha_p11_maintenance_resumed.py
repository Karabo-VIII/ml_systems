"""Alpha turn-024: p11 maintenance-resumed event-study via Bravo's
re-fetched cat-157 cache + harness with fixed shuffle null.

Filter: title r'(Has Completed|Opens Deposits and Withdrawals|Resumes Trading)'
Hypothesis (per p11 scoping): pent-up demand on wallet re-enable -> +2-5%
mean-reversion bullish at 1-6h.
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.frontier.utils.event_study import run_event_study

CACHE = ROOT / "logs" / "frontier" / "announcements" / "maintenance_recent.parquet"
OUT = ROOT / "logs" / "frontier" / "p11_event_study" / "maintenance_resumed.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not CACHE.exists():
        print(f"[ERR] cache not found at {CACHE}")
        return
    df = pd.read_parquet(CACHE)
    df = df[df["release_ms"] > 0].copy()
    df["release_ts"] = pd.to_datetime(df["release_ms"], unit="ms")
    print(f"[CACHE] {len(df)} dated rows, {df['release_ts'].min()} -> {df['release_ts'].max()}")

    # Bravo's filter for resumed events
    pat = r"(Has Completed|Opens Deposits and Withdrawals|Resumes Trading)"
    mask = df["title"].str.contains(pat, regex=True, case=False, na=False)
    resumed = df[mask].copy()
    print(f"[FILTER] resumed events: {len(resumed)} / {len(df)} match '{pat}'")
    if len(resumed) < 5:
        print("[ABORT] too few resumed events for stable t-stat. Recommend p11 wind-down trigger.")
        with open(OUT, "w") as f:
            json.dump({"n": int(len(resumed)), "status": "thin"}, f)
        return

    # Explode to (symbol, event_date)
    rows = []
    for _, r in resumed.iterrows():
        tokens = r["tokens"]
        if not isinstance(tokens, (list, tuple, np.ndarray)):
            continue
        for tok in tokens:
            if not tok or not isinstance(tok, str):
                continue
            rows.append({
                "event_date": r["release_ts"],
                "symbol": tok,
                "category": "maintenance_resumed",
                "title": str(r["title"])[:120],
            })
    if not rows:
        print("[EMPTY] no events with extractable tokens")
        return
    events = pd.DataFrame(rows).drop_duplicates(["symbol", "event_date"]).reset_index(drop=True)
    print(f"[EVENTS] {len(events)} (symbol, event_date) pairs across {events['symbol'].nunique()} symbols")

    # Run event-study with fixed harness
    try:
        results = run_event_study(
            events_df=events,
            horizons_h=[1, 3, 6, 12, 24],
            interval="1h",
            use_spot=True,
            entry_lag_min=10,  # 10 min after announcement
            cost_rt_pct=0.0020,
            splits={
                "TRAIN": ("2020-01-01", "2023-12-31"),
                "VAL":   ("2024-01-01", "2024-12-31"),
                "OOS":   ("2025-01-01", "2026-12-31"),
            },
            shuffle_null_n=20,
        )
    except Exception as e:
        print(f"[ERR] harness failed: {e}")
        import traceback; traceback.print_exc()
        return

    print()
    print(f"=== RESULTS PER SPLIT ===")
    for split in ("TRAIN", "VAL", "OOS"):
        s = results.get("per_split_horizon", {}).get(split, {})
        print(f"\n[{split}]")
        for h in (1, 3, 6, 12, 24):
            r = s.get(h, {})
            if r.get("n", 0) < 3:
                print(f"  h{h}h: thin")
                continue
            sig = " *SHIP*" if (r.get("t_stat",0) > 2 and r.get("mean_pct",0) > 0 and r.get("hit_rate",0) > 0.5) else ""
            print(f"  h{h}h: n={r['n']:3d}  mean={r.get('mean_pct',0):+.3f}%  "
                  f"t={r.get('t_stat',0):+.2f}  hit={r.get('hit_rate',0):.3f}{sig}")

    # Null comparison
    null = results.get("null", {}) or {}
    print(f"\n=== FIXED SHUFFLE NULL ===")
    for h in (1, 3, 6, 12, 24):
        n = null.get(h, {})
        if isinstance(n, dict) and "null_t_p95" in n:
            print(f"  h{h}h null: mean_t={n.get('null_t_mean',0):+.2f}  "
                  f"p5={n.get('null_t_p5',0):+.2f}  p95={n.get('null_t_p95',0):+.2f}")

    # Save
    flat = {"n_events": int(len(events))}
    for split, hdict in results.get("per_split_horizon", {}).items():
        flat[split] = {h: {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in r.items()}
                        for h, r in hdict.items()}
    flat["null"] = {str(h): n for h, n in (null or {}).items()}
    with open(OUT, "w") as f:
        json.dump(flat, f, indent=2, default=str)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
