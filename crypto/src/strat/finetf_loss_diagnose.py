"""src/strat/finetf_loss_diagnose.py -- V1.1 FINE-TF loss-attribution DIAGNOSTIC (2020 DEV-only).

WHY (meta /orc 2026-06-21): the V1 MA drawdown-insurance engine is binding-scope 1d/4h; it DEGRADES through
2h/1h/30m and BREAKS at 15m. The V1 doc asserts "churn eats the edge" but never DECOMPOSED *why* the fine TFs
lose on 2020. This module runs the BASELINE engine (band_off exit = the engine's own held_cross ensemble, NO
smart exit) on 2020 u100 for {2h,1h,30m,15m} x {EMA,HMA,TEMA,KAMA} (+ {1d,4h} as the working reference) and
QUANTIFIES the loss attribution per (engine, TF):
    - trades_per_week            (CHURN)
    - avg_hold_bars / avg_hold_days  (hold length)
    - roundtrip_cost_drag_pct_of_gross  (cost as a share of gross return)
    - whipsaw_rate_K{2,3}        (fraction of entries reversed/exited within K bars)
    - capture_rate_5pct          (fraction of >=5% moves the engine was long for at the move start)
    - frac_7d_slices_net_profitable  (fraction of rolling 7d slices net-positive)
    plus engine_compound, hodl_compound, maxdd, avg_gross (exposure), gross vs net.

SPLIT DISCIPLINE (BINDING -- this wave = 2020 DEV ONLY):
  * SELECT the working sub-band on the within-2020 TRAIN^VAL split = the committed config_leaderboard convention
    (select_band_trainval reads stored TRAIN.net>0 AND VAL.net>0; the leaderboard split is TRAIN=2020-H1,
    VAL=2020-Q3, OOS=2020-Q4, FULL=2020 -- all < 2021-01-01; OOS field never read in selection).
  * LOAD only 2020 data: WALL_2020 = 2021-01-01. We do NOT reuse fleet_lab.DEV_END (2024-05-15) for the data
    wall -- this wave never loads 2021 (reserved for the next wave's IRON) nor 2022+ (sealed OOS). We ASSERT
    max loaded bar < 2021-01-01.

REUSE (charter: use the whole project; do NOT hand-roll wealth or re-derive a band):
  - band = select_band_trainval(type, tf)  -> fixed-EW ENSEMBLE via make_hold_provider (NEVER ranked[0]).
  - accounting mirrors ma_wealth_driver._compound (MtM-no-double-count, lag-1, maker 0.0006 RT) -- the SAME
    causal weight-book replay, just walled at 2021-01-01 and instrumented to expose the held mask + trades so we
    can attribute the loss. The 2h synth-from-1h recipe is copied byte-for-byte from ma_wealth_driver._load_asset.

PRIOR-NULL CITATION: exits were a NULL on DAILY breakouts (dead-list -- the daily exit-axis null; closest D-IDs:
D67 causal mover-CAPTURE per-move<0, and the exit-axis NULL memory project-exit-axis-null-2026-06-08, daily
breakouts). BUT the fine-TF whipsaw regime is a DIFFERENT mechanism (cost/churn reduction at 2h-15m where the
engine over-trades), genuinely untested -- this is a DIAGNOSIS of that regime, not a re-run of the daily null.

RWYB: python -m strat.finetf_loss_diagnose --ma-type EMA --tf 1h    (one cell, prints the attribution)
Full: python -m strat.finetf_loss_diagnose --all                    (4 engines x 6 TFs -> diagnose_2020.json)
No emoji (cp1252). No git commit (overseer commits). Writes incrementally so a stall is salvageable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl                                       # noqa: E402  BARS_PER_DAY only
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402  committed infra
from mining.family_regime_map import _norm_sym                     # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                  # noqa: E402

# ---- BINDING 2020-only wall (NOT fleet_lab.DEV_END=2024-05-15). This wave never loads 2021+/2022+ ----
WALL_2020 = "2021-01-01"
MAKER_RT = 0.0006                                                  # maker round-trip cost (same as ma_wealth_driver)
_2H_MS = 7_200_000
_FREQ = {"1d": "D", "4h": "4h", "2h": "2h", "1h": "h", "30m": "30min", "15m": "15min"}
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
ENGINES = ["EMA", "HMA", "TEMA", "KAMA"]                           # the 4 the meta asked for
MOVE_THRESH = 0.05                                                 # a "5%+ move" for capture-rate


def _u100_symbols() -> list[str]:
    spec = yaml.safe_load(open(CRYPTO / "config" / "universes" / "u100.yaml"))
    if "assets" in spec:
        return [a["symbol"] for a in spec["assets"]]
    u50 = yaml.safe_load(open(CRYPTO / "config" / "universes" / "u50.yaml"))
    syms = [a["symbol"] for a in u50["assets"]] + [a["symbol"] for a in spec.get("extra_assets", [])]
    excl = set(spec.get("excluded_assets") or [])
    return [s for s in dict.fromkeys(syms) if s not in excl]


def _load_asset_2020(sym: str, tf: str):
    """Load one asset's 2020-walled OHLC at `tf` (2h synth-from-1h, causal). EXACT copy of
    ma_wealth_driver._load_asset but with the 2020-only wall. Returns (dates,o,h,l,c) or None."""
    load_cad = "1h" if tf == "2h" else tf
    wall_ms = int(pd.Timestamp(WALL_2020).value // 10 ** 6)
    try:
        df = ChimeraLoader().load(_norm_sym(sym), cadence=load_cad, features=["open", "high", "low", "close"])
    except Exception:
        return None
    if tf == "2h":
        ts1h = df["timestamp"].to_numpy()
        b = (ts1h // _2H_MS).astype("int64")
        pdf = pd.DataFrame({"open": df["open"].to_numpy().astype(float),
                            "high": df["high"].to_numpy().astype(float),
                            "low": df["low"].to_numpy().astype(float),
                            "close": df["close"].to_numpy().astype(float), "__b": b})
        g = pdf.groupby("__b", sort=True).agg(open=("open", "first"), high=("high", "max"),
                                              low=("low", "min"), close=("close", "last"))
        ts = (g.index.to_numpy() * _2H_MS).astype("int64")
        o = g["open"].to_numpy(); h = g["high"].to_numpy(); l = g["low"].to_numpy(); c = g["close"].to_numpy()
    else:
        ts = df["timestamp"].to_numpy()
        o = df["open"].to_numpy().astype(float); h = df["high"].to_numpy().astype(float)
        l = df["low"].to_numpy().astype(float); c = df["close"].to_numpy().astype(float)
    keep = ts < wall_ms                                            # 2020-ONLY WALL
    if keep.sum() < 31:
        return None
    dates = pd.to_datetime(ts[keep], unit="ms").floor(_FREQ.get(tf, "D"))
    return dates, o[keep], h[keep], l[keep], c[keep]


def _trade_segments(hold_arr: np.ndarray):
    """List of (entry_idx, exit_idx) for each contiguous held==1 run. exit_idx = first bar AFTER the run
    (so duration = exit_idx - entry_idx bars held). hold_arr is 0/1 the engine LONG state per bar (pre-lag)."""
    h = np.asarray(hold_arr).astype(int)
    if h.sum() == 0:
        return []
    d = np.diff(np.concatenate([[0], h, [0]]))
    entries = np.where(d == 1)[0]
    exits = np.where(d == -1)[0]
    return list(zip(entries.tolist(), exits.tolist()))


def run_cell(ma_type: str, tf: str, *, verbose: bool = False) -> dict:
    """ONE diagnostic cell: BASELINE engine (band_off exit = held_cross ensemble) on 2020 u100.
    Returns the full loss-attribution record. Causal MtM (lag-1, maker), 2020-walled, band frozen on 2020 TRAIN^VAL."""
    t0 = time.time()
    bpd = fl.BARS_PER_DAY[tf]

    # band = committed fixed-EW ensemble, selected on the within-2020 TRAIN^VAL split (OOS field never read)
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        return {"cell": f"{ma_type}|{tf}", "error": f"sub-band < 2 after TRAIN+VAL selection", "selection": sel_meta}
    provider = make_hold_provider(sub_band, ma_type)

    closes, holds = {}, {}
    per_asset_trade_stats = []                                     # (n_trades, sum_dur, whip2, whip3) per asset
    for sym in _u100_symbols():
        rec = _load_asset_2020(sym, tf)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        hh = provider(sym, o, h, l, c).astype(float)              # engine LONG state per bar (band_off exit)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]

    if not closes:
        return {"cell": f"{ma_type}|{tf}", "error": "no assets loaded"}
    cp = pd.DataFrame(closes).sort_index()
    _max_rows = max((s.notna().sum() for s in closes.values()), default=0)
    assert len(cp) <= 1.5 * _max_rows + 5, f"date-alignment regression: {len(cp)} >> {_max_rows}"
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)

    # WALL ASSERT (the binding split-discipline check)
    assert cp.index.max() < pd.Timestamp(WALL_2020), f"WALL VIOLATION: max bar {cp.index.max()} >= {WALL_2020}"

    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed                                            # long-while-held AND listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)

    # ENGINE weight book = fixed-N EW (1/N_listed per HELD name; cash else) -- identical to ma_wealth_driver LEG-1
    w_eng = held.div(n_listed, axis=0).fillna(0.0)

    # ---- MtM-no-double-count replay (lag-1, maker). NET stream + GROSS (pre-cost) stream ----
    Wl = w_eng.shift(1).fillna(0.0)
    gross_ret = (Wl * R).sum(axis=1)                              # per-bar GROSS portfolio return (pre-cost)
    turnover = (w_eng - w_eng.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost_stream = turnover * (MAKER_RT / 2.0)                     # per-bar cost
    net_stream = (gross_ret - cost_stream).to_numpy()
    eq = np.cumprod(1.0 + net_stream)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    eng_compound = float((eq[-1] - 1.0) * 100) if len(eq) else 0.0
    gross_compound = float((np.cumprod(1.0 + gross_ret.to_numpy())[-1] - 1.0) * 100)
    total_gross = float(gross_ret.sum())                          # sum of per-bar gross (additive proxy)
    total_cost = float(cost_stream.sum())
    avg_gross_exposure = float(w_eng.sum(axis=1).mean())          # avg fraction of capital deployed (phi)

    # ---- TRUE buy-and-hold on the SAME 2020 panel (units/MtM, drift-with-price, no rebalance) ----
    cols = list(cp.columns); n_total = len(cols)
    alloc = 1.0 / n_total if n_total else 0.0
    prices = cp.to_numpy(); nrows = prices.shape[0]
    first_idx = {c: (int(np.argmax(listed[c].to_numpy() > 0)) if listed[c].to_numpy().any() else None) for c in cols}
    value = np.zeros(nrows)
    for j, c in enumerate(cols):
        fi = first_idx[c]
        if fi is None:
            continue
        entry_px = prices[fi, j]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        px = pd.Series(prices[:, j]).ffill().to_numpy()
        hv = (alloc / entry_px) * px
        hv[:fi] = 0.0; hv[~np.isfinite(hv)] = 0.0
        value += hv
    bought_cum = np.zeros(nrows)
    for c in cols:
        fi = first_idx[c]
        if fi is not None:
            bought_cum[fi:] += 1.0
    cash = alloc * (n_total - bought_cum)
    n_entered = int(sum(1 for c in cols if first_idx[c] is not None))
    hodl_eq = (value + cash) * (1.0 - n_entered * alloc * MAKER_RT)
    hodl_compound = float((hodl_eq[-1] - 1.0) * 100) if len(hodl_eq) else 0.0
    hpk = np.maximum.accumulate(hodl_eq)
    hodl_maxdd = float(((hodl_eq - hpk) / hpk).min() * 100) if len(hodl_eq) else 0.0

    # ---- CHURN / HOLD-LENGTH / WHIPSAW (per asset, on the engine's pre-lag LONG state) ----
    n_bars = len(cp)
    weeks = n_bars / (7.0 * bpd)
    tot_trades = 0; tot_dur = 0; tot_whip2 = 0; tot_whip3 = 0; durs_all = []
    for sym in closes:
        harr = held[sym].to_numpy()
        segs = _trade_segments(harr)
        if not segs:
            continue
        n_tr = len(segs)
        dur = [ex - en for en, ex in segs]
        whip2 = sum(1 for du in dur if du <= 2)
        whip3 = sum(1 for du in dur if du <= 3)
        tot_trades += n_tr; tot_dur += sum(dur); tot_whip2 += whip2; tot_whip3 += whip3
        durs_all.extend(dur)
        per_asset_trade_stats.append((sym, n_tr, float(np.mean(dur)), whip2, whip3))
    avg_hold_bars = float(np.mean(durs_all)) if durs_all else 0.0
    median_hold_bars = float(np.median(durs_all)) if durs_all else 0.0
    n_assets_active = len(per_asset_trade_stats)
    trades_per_week_per_asset = (tot_trades / max(1, n_assets_active)) / max(1e-9, weeks)
    whip2_rate = tot_whip2 / max(1, tot_trades)
    whip3_rate = tot_whip3 / max(1, tot_trades)

    # ---- ROUND-TRIP COST DRAG as % of gross ----
    # cost as a fraction of |gross|: how much of the gross return the round-trips ate. Report both the
    # net-vs-gross compound gap (pp) and cost/|total_gross| (additive).
    cost_drag_pp = round(gross_compound - eng_compound, 2)        # compound pp lost to cost (gross -> net)
    cost_frac_of_gross = (total_cost / abs(total_gross) * 100) if abs(total_gross) > 1e-9 else None

    # ---- CAPTURE-RATE of 5%+ moves ----
    # A "move" = a forward 1-week (7-calendar-day = 7*bpd bars) return >= MOVE_THRESH on a listed asset, sampled
    # at every bar. We were LONG for the move iff the engine held that asset at the move START bar (lag-1 causal:
    # the position we carry INTO bar t is w_eng.shift(1)). capture = fraction of 5%+ moves we were long at start.
    horizon = 7 * bpd
    held_lag = held.shift(1).fillna(0.0)                         # the position carried INTO each bar (causal)
    fwd = cp.shift(-horizon) / cp - 1.0                          # forward 7d return per asset
    move_mask = (fwd >= MOVE_THRESH) & listed.astype(bool) & cp.notna()
    n_moves = int(move_mask.to_numpy().sum())
    captured = int((move_mask & (held_lag > 0)).to_numpy().sum())
    capture_rate_5pct = (captured / n_moves) if n_moves else None

    # ---- FRACTION OF ROLLING 7d SLICES NET-PROFITABLE ----
    # roll the NET portfolio stream into non-overlapping 7d windows; fraction with positive compound.
    win = 7 * bpd
    slice_rois = []
    ns = net_stream
    for st in range(0, len(ns) - win + 1, win):
        seg = ns[st:st + win]
        slice_rois.append(float(np.prod(1.0 + seg) - 1.0))
    frac_7d_profitable = (float(np.mean(np.array(slice_rois) > 0)) if slice_rois else None)
    n_7d_slices = len(slice_rois)

    runtime = round(time.time() - t0, 1)
    out = {
        "cell": f"{ma_type}|{tf}",
        "ma_type": ma_type, "tf": tf, "bars_per_day": bpd,
        "n_assets": int(cp.shape[1]), "n_assets_active": n_assets_active,
        "n_bars": n_bars, "weeks_span": round(weeks, 1),
        "data_range": [str(cp.index.min()), str(cp.index.max())],
        "n_sub_band": len(sub_band), "selection_rule": sel_meta.get("selection_rule"),
        # --- wealth context ---
        "engine_compound_pct": round(eng_compound, 2),
        "gross_compound_pct": round(gross_compound, 2),
        "hodl_compound_pct": round(hodl_compound, 2),
        "vs_hodl_pp": round(eng_compound - hodl_compound, 2),
        "maxdd_engine_pct": round(maxdd, 1),
        "maxdd_hodl_pct": round(hodl_maxdd, 1),
        "avg_gross_exposure": round(avg_gross_exposure, 3),
        # --- THE LOSS-ATTRIBUTION TABLE ---
        "trades_per_week_per_asset": round(trades_per_week_per_asset, 3),
        "total_trades": tot_trades,
        "avg_hold_bars": round(avg_hold_bars, 2),
        "median_hold_bars": round(median_hold_bars, 2),
        "avg_hold_days": round(avg_hold_bars / bpd, 3),
        "roundtrip_cost_drag_pp": cost_drag_pp,
        "cost_frac_of_gross_pct": (round(cost_frac_of_gross, 2) if cost_frac_of_gross is not None else None),
        "total_gross_additive": round(total_gross, 4),
        "total_cost_additive": round(total_cost, 4),
        "whipsaw_rate_K2": round(whip2_rate, 4),
        "whipsaw_rate_K3": round(whip3_rate, 4),
        "capture_rate_5pct": (round(capture_rate_5pct, 4) if capture_rate_5pct is not None else None),
        "n_5pct_moves": n_moves, "n_5pct_captured": captured,
        "frac_7d_slices_net_profitable": (round(frac_7d_profitable, 4) if frac_7d_profitable is not None else None),
        "n_7d_slices": n_7d_slices,
        "cost_rt": MAKER_RT, "cost_kind": "maker",
        "split": f"2020 DEV only (band on TRAIN^VAL within-2020; wall < {WALL_2020})",
        "runtime_s": runtime,
    }
    if verbose:
        print(f"[{out['cell']}] eng {out['engine_compound_pct']}% / gross {out['gross_compound_pct']}% / "
              f"HODL {out['hodl_compound_pct']}%  maxDD {out['maxdd_engine_pct']}%  phi {out['avg_gross_exposure']}")
        print(f"   churn {out['trades_per_week_per_asset']} tr/wk/asset  hold {out['avg_hold_bars']}b "
              f"({out['avg_hold_days']}d)  cost-drag {out['roundtrip_cost_drag_pp']}pp "
              f"({out['cost_frac_of_gross_pct']}% of gross)")
        print(f"   whip K2 {out['whipsaw_rate_K2']}  K3 {out['whipsaw_rate_K3']}  "
              f"capt-5% {out['capture_rate_5pct']} ({out['n_5pct_captured']}/{out['n_5pct_moves']})  "
              f"7d-profitable {out['frac_7d_slices_net_profitable']}  ({runtime}s)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.finetf_loss_diagnose")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--all", action="store_true", help="4 engines x 6 TFs -> diagnose_2020.json (incremental)")
    a = ap.parse_args()
    outdir = CRYPTO / "runs" / "engines" / "V1.1_finetf_exits"
    outdir.mkdir(parents=True, exist_ok=True)
    if not a.all:
        r = run_cell(a.ma_type, a.tf, verbose=True)
        print(json.dumps(r, indent=1, default=str))
        return 0 if "error" not in r else 2

    out_path = outdir / "diagnose_2020.json"
    cells = {}
    # resume if a partial file exists (salvageable)
    if out_path.exists():
        try:
            cells = json.load(open(out_path)).get("cells", {})
        except Exception:
            cells = {}
    for tf in TFS:
        for ma_type in ENGINES:
            key = f"{ma_type}|{tf}"
            if key in cells and "error" not in cells[key]:
                print(f"[skip] {key} (already done)"); continue
            print(f"[run] {key} ...", flush=True)
            try:
                cells[key] = run_cell(ma_type, tf, verbose=True)
            except Exception as e:
                cells[key] = {"cell": key, "error": str(e)[:300]}
                print(f"[ERR] {key}: {e}")
            # WRITE INCREMENTALLY after every cell (a stall is salvageable)
            json.dump({"meta": {"generated": time.strftime("%Y%m%d_%H%M%S"), "wall": WALL_2020,
                                "engines": ENGINES, "tfs": TFS, "cost_rt": MAKER_RT,
                                "move_thresh": MOVE_THRESH},
                       "cells": cells}, open(out_path, "w"), indent=1, default=str)
    print(f"\n[json] {out_path}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
