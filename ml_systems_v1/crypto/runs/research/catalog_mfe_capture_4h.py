"""catalog_mfe_capture_4h.py -- INDEPENDENT 4h capture cross-check using the LITERALLY-REQUESTED catalog.

The OVERSEER task names capturable_4h_catalog.parquet as the oracle. That file is a binary win-LABEL
catalog, BUT it also carries the continuous `max_gain_12bar` column = max favorable excursion (MFE) over
the next 12 (4h) bars = a legitimate PER-BAR capture oracle, INDEPENDENT of the DP high-capture oracle
the main grid uses. This script answers the same question with that independent oracle + the catalog's
OWN split column, so the 4h refutation is shown robust to the oracle definition (not an artefact of the DP).

CAPTURE (per asset, held-out = oos+unseen rows):  for each MA cross-up entry bar t, realized =
close[min(t+h,exit)]/close[t]-1-cost (h = bars to bearish cross, capped at 12 to match the 12-bar MFE
window); oracle = max_gain_12bar[t]. aggregate capture = sum(realized) / sum(max_gain_12bar) over the
held-out MA entries. NULL = regime-matched random entries drawn ONLY from bullish (fast>slow) bars, count
matched per asset, hold-durations sampled from the MA's own held-out hold distribution, same cost, same
capture formula -> p50/p95 capture. EDGE requires capture_MA > capture_null_p95.

RWYB: reads only the catalog parquet; writes one JSON. No look-ahead (MA past-only; entry at t uses
close<=t; the only optimism is close-fill, applied IDENTICALLY to MA and null so the comparison is fair).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COST_RT = 0.0024
HELD = ("oos", "unseen")
MA_CONFIGS = [(10, 30), (20, 50)]
HOLD_CAP = 12            # match the 12-bar MFE oracle window
N_BOOKS = 300


def sma_past_only(x, w):
    """rolling mean using values up to and including t, NaN during warmup (past-only; t is known at close t)."""
    s = pl.Series(x).rolling_mean(window_size=w, min_periods=w)
    return s.to_numpy()


def ma_entries(close, fast, slow):
    sf = sma_past_only(close, fast)
    ss = sma_past_only(close, slow)
    bull = (sf > ss) & np.isfinite(sf) & np.isfinite(ss)
    bull_prev = np.concatenate([[False], bull[:-1]])
    x_up = bull & ~bull_prev          # cross-up bars
    x_dn = (~bull) & bull_prev         # cross-down bars (exit)
    return x_up, x_dn, bull


def realized_capture(close, mgain, entry_idx, x_dn, n):
    """sum of realized net & sum of oracle MFE over the given entry bars (exit = first cross-down<=12 else 12)."""
    real_sum, orc_sum = 0.0, 0.0
    for t in entry_idx:
        # exit = first cross-down after t within HOLD_CAP, else t+HOLD_CAP
        h = HOLD_CAP
        for k in range(1, HOLD_CAP + 1):
            if t + k >= n:
                h = max(1, (n - 1) - t)
                break
            if x_dn[t + k]:
                h = k
                break
        xf = min(t + h, n - 1)
        if xf <= t:
            continue
        realized = close[xf] / close[t] - 1.0 - COST_RT
        orc = mgain[t]
        if orc <= 1e-9:        # only count bars where the oracle had positive opportunity
            continue
        real_sum += realized
        orc_sum += orc
    return real_sum, orc_sum


def null_capture(close, mgain, x_dn, bull, held_mask, n_entries, durs, n_books, seed=7):
    """regime-matched random-entry null capture distribution (entries among bullish & held-out bars)."""
    rng = np.random.default_rng(seed)
    n = len(close)
    eligible = np.array([i for i in range(1, n - 2) if bull[i] and held_mask[i]])
    if len(eligible) == 0 or n_entries == 0 or len(durs) == 0:
        return None
    caps = np.empty(n_books)
    for b in range(n_books):
        ents = rng.choice(eligible, size=n_entries, replace=True)
        ds = rng.choice(durs, size=n_entries, replace=True)
        rs, os_ = 0.0, 0.0
        for t, d in zip(ents, ds):
            xf = min(t + int(d), n - 1)
            if xf <= t:
                continue
            orc = mgain[t]
            if orc <= 1e-9:
                continue
            rs += close[xf] / close[t] - 1.0 - COST_RT
            os_ += orc
        caps[b] = rs / os_ if os_ > 1e-9 else 0.0
    return caps


def main():
    df = pl.read_parquet(ROOT / "data" / "processed" / "capturable_4h_catalog.parquet")
    assets = df["asset"].unique().to_list()
    print(f"[catalog-MFE 4h] {len(assets)} assets, held-out = {HELD}, hold_cap={HOLD_CAP} bars, cost={COST_RT}")
    print(f"{'asset':8} {'ma':>7} {'nE':>4} | {'capMA':>7} {'cap_p50':>8} {'cap_p95':>8} {'beat95':>6}")
    print("-" * 64)

    rows = []
    for a in assets:
        sub = df.filter(pl.col("asset") == a).sort("ts")
        close = sub["close"].to_numpy().astype(float)
        mgain = sub["max_gain_12bar"].to_numpy().astype(float)
        split = sub["split"].to_numpy()
        n = len(close)
        if n < 200:
            continue
        held_mask = np.isin(split, HELD)
        for (f, sl) in MA_CONFIGS:
            x_up, x_dn, bull = ma_entries(close, f, sl)
            ent = np.where(x_up & held_mask)[0]
            ent = ent[(ent > 0) & (ent < n - 2)]
            if len(ent) < 5:
                continue
            # held-out MA hold durations (for the null)
            durs = []
            for t in ent:
                h = HOLD_CAP
                for k in range(1, HOLD_CAP + 1):
                    if t + k >= n:
                        h = max(1, (n - 1) - t); break
                    if x_dn[t + k]:
                        h = k; break
                durs.append(h)
            durs = np.array(durs)
            rs, os_ = realized_capture(close, mgain, ent, x_dn, n)
            if os_ <= 1e-9:
                continue
            capMA = rs / os_
            caps = null_capture(close, mgain, x_dn, bull, held_mask, len(ent), durs, N_BOOKS)
            if caps is None:
                continue
            p50, p95 = float(np.percentile(caps, 50)), float(np.percentile(caps, 95))
            beat = capMA > p95
            rows.append({"asset": a, "ma": f"{f}/{sl}", "n_entries": int(len(ent)),
                         "capture_MA": capMA, "capture_null_p50": p50, "capture_null_p95": p95,
                         "beats_null_p95": bool(beat)})
            print(f"{a:8} {f}/{sl:<4} {len(ent):>4} | {capMA:>7.3f} {p50:>8.3f} {p95:>8.3f} {str(beat):>6}")

    caps = [r["capture_MA"] for r in rows]
    edges = [r["capture_MA"] - r["capture_null_p50"] for r in rows]
    n_beat = sum(r["beats_null_p95"] for r in rows)
    summ = {
        "n_cells": len(rows), "median_capture_MA": float(np.median(caps)) if caps else None,
        "median_edge_vs_null_p50": float(np.median(edges)) if edges else None,
        "n_beats_null_p95": n_beat, "frac_beats_p95": (n_beat / len(rows)) if rows else None,
        "expected_false_positive_rate": 0.05,
    }
    print("=" * 64)
    print(f"[catalog-MFE 4h] cells={summ['n_cells']} medianCapMA={summ['median_capture_MA']:.4f} "
          f"medianEdge_p50={summ['median_edge_vs_null_p50']:.4f} beats_p95={n_beat}/{summ['n_cells']} "
          f"(expected by chance @5% = {0.05*summ['n_cells']:.1f})")
    out = {"task": "independent 4h capture cross-check vs catalog MFE oracle (max_gain_12bar)",
           "oracle": "capturable_4h_catalog.max_gain_12bar (max-favorable-excursion, 12x4h-bar window)",
           "cost_rt": COST_RT, "hold_cap_bars": HOLD_CAP, "n_books": N_BOOKS,
           "summary": summ, "cells": rows}
    outp = ROOT / "runs" / "research" / "catalog_mfe_capture_4h_result.json"
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[OK] wrote {outp}")


if __name__ == "__main__":
    main()
