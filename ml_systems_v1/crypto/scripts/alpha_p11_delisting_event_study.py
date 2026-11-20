"""Alpha turn-020: p11 Phase 1 event-study on DELISTING category.

Uses Bravo's turn-019 harness + body-enriched delisting cache (60/60 with
bodies + 134 tokens extracted in turn-020 re-enrichment).

Hypothesis (per p11 scoping doc):
  - Delisting announcement triggers forced-seller flow (holders exit before
    spot pair is removed).
  - Price typically dips -15 to -50% over 0-7d following announcement.
  - Rebound: once forced-selling exhausts (day 2-4), mean-reversion to +10
    to +25% over subsequent 3-7d as buyers return.
  - Caveat: Binance spot kline data stops when pair is delisted. Horizons
    must stay within the announcement -> effective-delisting window
    (typically 5-7 days). Use <=120h.

Constraint: D1 spot-only. We can only play the REBOUND leg (long after the
dip), not the dip itself.

Event-study:
  - For each delisting announcement with extracted tokens, enter at t0+2d
    (after initial selloff), horizons 24/48/72/120h forward
  - Shuffle control + TRAIN/VAL/OOS split
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.frontier.utils.event_study import run_event_study

ANN_PATH = ROOT / "logs" / "frontier" / "announcements" / "delisting_recent.parquet"
OUT = ROOT / "logs" / "frontier" / "p11_event_study" / "delisting_rebound.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(ANN_PATH)
    df = df[df["release_ms"] > 0].copy()
    df["release_ts"] = pd.to_datetime(df["release_ms"], unit="ms")
    # Explode to one row per (announcement, token)
    rows = []
    for _, r in df.iterrows():
        tokens = r["tokens"]
        if not isinstance(tokens, (list, tuple, np.ndarray)):
            continue
        for tok in tokens:
            if not tok or not isinstance(tok, str):
                continue
            rows.append({
                "event_date": r["release_ts"],
                "symbol": tok,
                "category": "delisting",
                "title": str(r["title"])[:120],
            })
    if not rows:
        print("[EMPTY] no events")
        return
    events = pd.DataFrame(rows)
    # Filter to unique (symbol, event_date) to avoid double-counting
    events = events.drop_duplicates(subset=["symbol", "event_date"]).reset_index(drop=True)
    print(f"[EVENTS] {len(events)} unique (symbol, event_date) pairs")
    print(f"  unique symbols: {events['symbol'].nunique()}")
    print(f"  date range: {events['event_date'].min()} -> {events['event_date'].max()}")
    print()

    # D1 constraint: spot only, long-only
    # Entry at t0+2d = 2880 minutes after announcement
    # Horizons constrained to <=120h (5d, within pre-delisting window)
    try:
        results = run_event_study(
            events_df=events,
            horizons_h=[24, 48, 72, 120],
            interval="1h",
            use_spot=True,
            entry_lag_min=2880,  # t0 + 2d
            cost_rt_pct=0.0020,  # 20 bps round-trip
            splits={
                "TRAIN": ("2020-01-01", "2023-12-31"),
                "VAL":   ("2024-01-01", "2024-12-31"),
                "OOS":   ("2025-01-01", "2026-12-31"),
            },
            shuffle_null_n=10,
        )
    except Exception as e:
        print(f"[ERR] harness run failed: {e}")
        import traceback; traceback.print_exc()
        return

    print("\n=== RESULTS PER SPLIT / HORIZON ===")
    for split_name in ("TRAIN", "VAL", "OOS"):
        split = results.get("per_split_horizon", {}).get(split_name, {})
        print(f"\n[{split_name}]")
        for h in (24, 48, 72, 120):
            r = split.get(h)
            if not r or r.get("n", 0) < 3:
                print(f"  h{h}h: thin")
                continue
            sig = " *SHIP*" if (r.get("t_stat",0) > 2 and r.get("mean_pct",0) > 0 and r.get("hit_rate",0) > 0.5) else ""
            print(f"  h{h}h: n={r['n']:3d}  mean={r.get('mean_pct',0):+.3f}%  "
                  f"t={r.get('t_stat',0):+.2f}  hit={r.get('hit_rate',0):.3f}{sig}")

    if "null" in results:
        print("\n=== SHUFFLE NULL ===")
        for h, n in (results["null"] or {}).items():
            if isinstance(n, dict):
                print(f"  h{h}h  {n}")

    # Save summary
    import json
    flat = {
        "n_events": int(len(events)),
        "per_split_horizon": {
            s: {h: {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in r.items()}
                for h, r in hdict.items()}
            for s, hdict in results.get("per_split_horizon", {}).items()
        },
    }
    with open(OUT, "w") as f:
        json.dump(flat, f, indent=2, default=str)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
