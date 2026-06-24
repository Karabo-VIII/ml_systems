"""src/strat/ma_val_forward.py -- TASK A: VAL / HELD-OUT FORWARD breakdown of the 8 MA engines.

WHY (meta directive 2026-06-20): the 8 MA engines SELECTED their working band on 2020 (DEV/TRAIN).
"VAL results" = does that 2020-selected DE-RISKED-BETA property HOLD FORWARD on data it did NOT
select on? This module breaks the SAME verified driver run (ma_wealth_driver) into FORWARD sub-windows
and recompounds each leg WITHIN each window, per (engine, forward-window, tf).

FORWARD WINDOWS (all strictly < DEV_END = 2024-05-15 -- NO OOS/UNSEEN touch, the DEV wall stays):
  - 2021              : the "IRON" forward-validation year (bull).
  - 2022              : the crypto bear.
  - 2023-01-01..2024-05-14 : the DEV tail (recovery + chop).

PER (engine, window) at tf in {1d, 4h}, window=14d:
  engine_compound, hodl_compound, vs_hodl_pp, upmkt_timing_alpha_pp (engine vs always-long-index IN
  that window -- here the WHOLE window is the "up-market" slice; we report engine-vs-alwayslong as the
  timing control IN the window), maxdd_engine vs maxdd_hodl.

NO NEW BACKTEST: this REUSES ma_wealth_driver's exact panel construction + the COMMITTED capture infra
(select_band_trainval TRAIN+VAL sub-band OOS-excluded, make_hold_provider ensemble, fleet_lab DEV wall,
the 2h synth-from-1h recipe). It only adds a date-window SLICER that recompounds each leg within a window.

WITHIN-WINDOW SEMANTICS (the honest forward read):
  ENGINE / ALWAYS-LONG : both are rebalanced weight books -> a sub-window product is well-defined. We
                         restrict the per-bar net stream to the window's bars and recompound (same
                         MtM-no-double-count accounting as the driver's _window_compound).
  HODL                 : RE-ANCHORED per window -- each asset is (re)bought at its FIRST available bar
                         INSIDE the window (or at window open if already listed), equal 1/N share, held
                         to window end, weights drift, no rebalance. This is the honest "what if I just
                         bought-and-held over THIS window" -- the correct within-window buy-hold baseline.
  maxDD                : recomputed on the equity curve RESTRICTED to the window's bars (within-window
                         peak-to-trough), re-based to 1.0 at window open.

Strictly causal + DEV-walled (inherited from the driver): weights lag 1 bar, maker 0.0006 RT, panel
capped < DEV_END before any compute, selection = TRAIN+VAL sub-band (OOS never read).

RWYB: python -m strat.ma_val_forward --ma-type DEMA --tf 1d
No emoji (cp1252). No git commit (meta reviews).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl                                              # noqa: E402  DEV_END, BARS_PER_DAY
import strat.ma_wealth_driver as wd                                       # noqa: E402  THE verified driver
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402

OUTDIR = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH"
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]

# FORWARD windows: [start_inclusive, end_exclusive). ALL strictly < DEV_END (2024-05-15). The DEV-tail
# end is 2024-05-15 (exclusive) which == DEV_END, so the panel (already < DEV_END) never crosses it.
FORWARD_WINDOWS = [
    ("2021",        "2021-01-01", "2022-01-01"),
    ("2022",        "2022-01-01", "2023-01-01"),
    ("2023_24tail", "2023-01-01", "2024-05-15"),   # 2023-01-01 .. 2024-05-14 inclusive (DEV tail)
]


def _maxdd_window(net_stream: np.ndarray, mask: np.ndarray) -> float:
    """Within-window maxDD: recompound the masked net stream from 1.0 and take peak-to-trough min."""
    sub = net_stream[mask]
    if len(sub) == 0:
        return float("nan")
    eq = np.cumprod(1.0 + sub)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def _hodl_window(cp: pd.DataFrame, listed: pd.DataFrame, mask: np.ndarray, cost_rt: float):
    """TRUE buy-and-hold RE-ANCHORED to a window. Each asset is (re)bought at its FIRST available bar
    INSIDE the window (price finite, listed), equal 1/N_total dollar share (N_total = #assets that ever
    list within the window), held to window end, weights DRIFT, no rebalance, one round-trip cost/name.
    Returns (compound_pct, maxdd_pct, n_entered). Mirrors wd._true_hodl but restricted to the window."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return float("nan"), float("nan"), 0
    cols = list(cp.columns)
    prices = cp.to_numpy()
    lst = listed.to_numpy()
    w0, w1 = idx[0], idx[-1] + 1                               # window bar span [w0, w1)
    # first listed bar of each name INSIDE the window
    first_in = {}
    for j, c in enumerate(cols):
        listed_in = np.where((lst[w0:w1, j] > 0) & np.isfinite(prices[w0:w1, j]) & (prices[w0:w1, j] > 0))[0]
        first_in[c] = (w0 + listed_in[0]) if len(listed_in) else None
    entered = [c for c in cols if first_in[c] is not None]
    n_total = len(entered)
    if n_total == 0:
        return float("nan"), float("nan"), 0
    alloc = 1.0 / n_total
    nrows = w1 - w0
    value = np.zeros(nrows)
    bought_cum = np.zeros(nrows)
    for c in entered:
        j = cols.index(c)
        fi = first_in[c]                                       # absolute bar index
        entry_px = prices[fi, j]
        units = alloc / entry_px
        px = pd.Series(prices[w0:w1, j]).ffill().to_numpy()
        rel = fi - w0
        held = units * px
        held[:rel] = 0.0
        held[~np.isfinite(held)] = 0.0
        value += held
        bought_cum[rel:] += 1.0
    cash = alloc * (n_total - bought_cum)
    equity_gross = value + cash                                # starts at 1.0 (all cash), drifts in
    total_cost = n_total * alloc * cost_rt                     # one round-trip per entered name
    eq = equity_gross * (1.0 - total_cost)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    comp = float((eq[-1] - 1.0) * 100) if len(eq) else 0.0
    return comp, maxdd, n_total


def run_forward(ma_type: str, tf: str, window: str = "14d", *, cost_rt: float = wd.MAKER_RT) -> dict:
    """Build the SAME driver panel + weights, then recompound each leg within each FORWARD window."""
    window_days = int(str(window).rstrip("d"))
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)

    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        return {"error": f"sub-band < 2 for {ma_type}|{tf}", "selection": sel_meta}
    provider = make_hold_provider(sub_band, ma_type)

    closes, holds = {}, {}
    for sym in wd._u100_symbols():
        rec = wd._load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        hh = provider(sym, o, h, l, c).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]
    if not closes:
        return {"error": f"no assets loaded for {ma_type}|{tf}|{window}"}

    cp = pd.DataFrame(closes).sort_index()
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)
    assert cp.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION (panel index >= DEV_END)"

    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    w_eng = held.div(n_listed, axis=0).fillna(0.0)            # ENGINE: fixed-N EW per HELD name
    w_along = listed.div(n_listed, axis=0).fillna(0.0)        # ALWAYS-LONG: daily-rebalanced EW index

    # per-bar net streams (lag-1, maker cost) -- needed for within-window maxDD
    def _net(W):
        Wl = W.shift(1).fillna(0.0)
        gross = (Wl * R).sum(axis=1)
        turn = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
        return (gross - turn * (cost_rt / 2.0)).to_numpy()
    eng_net = _net(w_eng)
    along_net = _net(w_along)

    bidx = cp.index
    rows = {}
    for name, start, end in FORWARD_WINDOWS:
        mask = np.asarray((bidx >= pd.Timestamp(start)) & (bidx < pd.Timestamp(end)))
        n_bars = int(mask.sum())
        if n_bars == 0:
            rows[name] = {"window": name, "range": [start, end], "n_bars": 0, "note": "no bars"}
            continue
        eng_c = wd._window_compound(w_eng, R, cost_rt, mask)
        along_c = wd._window_compound(w_along, R, cost_rt, mask)
        hodl_c, hodl_dd, n_entered = _hodl_window(cp, listed, mask, cost_rt)
        eng_dd = _maxdd_window(eng_net, mask)
        along_dd = _maxdd_window(along_net, mask)
        vs_hodl = round(eng_c - hodl_c, 2) if np.isfinite(eng_c) and np.isfinite(hodl_c) else None
        # upmkt timing alpha IN-window = engine vs always-long index over the window (timing control)
        upmkt = round(eng_c - along_c, 2) if np.isfinite(eng_c) and np.isfinite(along_c) else None
        rows[name] = {
            "window": name, "range": [start, end], "n_bars": n_bars,
            "engine_compound_pct": round(eng_c, 2) if np.isfinite(eng_c) else None,
            "hodl_compound_pct": round(hodl_c, 2) if np.isfinite(hodl_c) else None,
            "alwayslong_rebal_compound_pct": round(along_c, 2) if np.isfinite(along_c) else None,
            "vs_hodl_pp": vs_hodl,
            "upmkt_timing_alpha_pp": upmkt,                  # engine - alwayslong IN the window
            "maxdd_engine_pct": round(eng_dd, 1) if np.isfinite(eng_dd) else None,
            "maxdd_hodl_pct": round(hodl_dd, 1) if np.isfinite(hodl_dd) else None,
            "maxdd_alwayslong_pct": round(along_dd, 1) if np.isfinite(along_dd) else None,
            "n_entered": n_entered,
            "beats_hodl": bool(vs_hodl is not None and vs_hodl > 0),
            "upmkt_alpha_nonpos": bool(upmkt is not None and upmkt <= 0),
        }
    out = {
        "ma_type": ma_type, "tf": tf, "window": window, "cost_rt": cost_rt,
        "selection_rule": sel_meta.get("selection_rule"),
        "n_sub_band": len(sub_band),
        "n_assets": int(cp.shape[1]),
        "panel_range": [str(cp.index.min()), str(cp.index.max())],
        "split": f"DEV forward sub-windows, all < {fl.DEV_END}; OOS+UNSEEN SEALED",
        "basis": ("SAME driver panel+weights (ma_wealth_driver), per-window recompound. ENGINE/ALWAYS-LONG = "
                  "masked net stream recompounded. HODL = RE-ANCHORED buy-and-hold within the window (buy each "
                  "name once at its first in-window bar, 1/N share, drift, no rebal). maxDD = within-window "
                  "peak-to-trough rebased to window open. lag-1, maker 0.0006, DEV wall, TRAIN+VAL sub-band."),
        "windows": rows,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_val_forward")
    ap.add_argument("--ma-type", default="DEMA")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--window", default="14d")
    ap.add_argument("--all", action="store_true", help="run all 8 MA types x {1d,4h}, write artifacts")
    a = ap.parse_args()
    if a.all:
        for mt in MA_TYPES:
            rec = {"ma_type": mt, "window": a.window, "by_tf": {}}
            for tf in ["1d", "4h"]:
                r = run_forward(mt, tf, a.window)
                rec["by_tf"][tf] = r
                print(f"[{mt}|{tf}] " + " ".join(
                    f"{w}:vsHODL={v.get('vs_hodl_pp')}pp,upalpha={v.get('upmkt_timing_alpha_pp')}pp"
                    for w, v in r.get("windows", {}).items() if isinstance(v, dict) and "vs_hodl_pp" in v))
            outp = OUTDIR / f"val_forward_{mt}.json"
            outp.write_text(json.dumps(rec, indent=1, default=str))
            print(f"  -> wrote {outp}")
        return 0
    r = run_forward(a.ma_type, a.tf, a.window)
    print(json.dumps(r, indent=1, default=str))
    return 2 if "error" in r else 0


if __name__ == "__main__":
    sys.exit(main())
