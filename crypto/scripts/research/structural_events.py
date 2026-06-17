#!/usr/bin/env python3
"""MARKET RESEARCH g1 -- the EVENT-DRIVEN opportunity surface (structural avenues).

CHARACTERIZATION ONLY (NOT edge-mining): for each structural avenue the chimera exposes (funding, liquidations,
basis, whale flow, tape imbalance), how OFTEN does it fire an extreme event (|z|>=2), on how many assets is it
even AVAILABLE, and -- descriptively -- is the SAME-DAY |move| elevated when it fires? The same-day comparison is
CONTEMPORANEOUS (coincidence, a marker of activity), NOT a forward-predictive test. It answers 'do these avenues
have material density worth investigating later?', not 'do they predict returns'. Run:
python scripts/research/structural_events.py
No emoji (Windows cp1252).
"""
import glob
import json
import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-normalized / z-scored event features -> |value| >= 2 ~ a 2-sigma event.
EVENTS = {
    "funding_extreme": "norm_funding",
    "liq_short_spike": "liq_short_z30",
    "liq_long_spike": "liq_long_z30",
    "basis_dislocation": "bs_basis_z30",
    "whale_flow_extreme": "norm_whale",
    "tape_imbalance_extreme": "norm_flow_imbalance",
}
Z = 2.0


def _files():
    return sorted(glob.glob(os.path.join(ROOT, "data", "processed", "chimera", "1d", "*.parquet")))


def main():
    files = _files()
    results = {}
    for name, col in EVENTS.items():
        avail_assets = 0
        freqs = []          # per-asset event frequency
        contemp_ratio = []  # per-asset mean|move| on event-days / mean|move| off-event-days
        for f in files:
            try:
                df = pl.read_parquet(f, columns=["close", col]).drop_nulls()
            except Exception:
                continue
            if len(df) < 100:
                continue
            v = df[col].to_numpy().astype(float)
            c = df["close"].to_numpy().astype(float)
            if np.nanstd(v) == 0:
                continue
            avail_assets += 1
            ret = np.abs(np.concatenate([[0.0], (c[1:] - c[:-1]) / np.where(c[:-1] == 0, np.nan, c[:-1])]))
            ev = np.abs(v) >= Z
            if ev.sum() < 5:
                freqs.append(float(ev.mean()))
                continue
            freqs.append(float(ev.mean()))
            on = np.nanmean(ret[ev]); off = np.nanmean(ret[~ev])
            if off and off > 0 and np.isfinite(on) and np.isfinite(off):
                contemp_ratio.append(on / off)
        results[name] = {
            "feature": col,
            "available_on_assets": avail_assets,
            "median_event_freq": round(float(np.median(freqs)), 4) if freqs else None,
            "univ_avg_event_freq": round(float(np.mean(freqs)), 4) if freqs else None,
            "contemp_move_ratio_event_vs_quiet": round(float(np.median(contemp_ratio)), 2) if contemp_ratio else None,
            "n_assets_for_ratio": len(contemp_ratio),
        }
    out = {"threshold_z": Z, "note": "contemp ratio is SAME-DAY |move| on event vs quiet days (coincidence, NOT forward prediction)", "events": results}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, "runs", "research", "structural_events.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
