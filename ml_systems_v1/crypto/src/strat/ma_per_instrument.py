"""src/strat/ma_per_instrument.py -- PER-INSTRUMENT view of the decoupled MA configs x exit x timeframe.

WHY (user /orc 2026-06-11): *"tell me about the strats: best, worst, average holding time, timeframe,
etc per instrument. And then tell me the data narrative: what do you observe on the winners, losers."*

Complements the book-level ma_exit_tf_sweep (which pools u10 into one portfolio). This runs each
distinct MA config SINGLE-ASSET, so we can see -- per instrument -- which (config, exit, timeframe)
wins/loses, the average holding time, and the best timeframe. Reuses the validated engine pieces:
holding_state (signal), apply_trail_stop (mechanical exit), per_asset_trades (round-trip trades).

For each (asset, family-config, exit, cadence): hold-state on the FULL series (MA warmup), optional
trailing-stop overlay, extract round-trip trades, keep those that ENTER inside the window. Per cell we
record compound return, n_trades, avg hold (bars + wall-clock), win-rate. Then we aggregate per asset.
Descriptive (no verdicts). Oldest month default (2020-01-07..2020-02-07); 1d is excluded by default
(slow MAs are warmup-starved that early -> no trades), cadences = 4h/1h/30m/15m.

RWYB:  python -m strat.ma_per_instrument            # u10, oldest month, 8 distinct configs/family
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT, MAKER_RT
from strat.portfolio_replay_per_asset import per_asset_trades
from strat.replay_distinct_grid import distinct_specs
from pipeline.chimera_loader import ChimeraLoader
from mining.family_regime_map import _norm_sym

OUT = ROOT.parent / "runs" / "strat"
FLOOR = {"1d": "D", "4h": "4h", "1h": "h", "30m": "30min", "15m": "15min"}
TF_HOURS = {"1d": 24, "4h": 4, "1h": 1, "30m": 0.5, "15m": 0.25}
EXITS = [("signalflip", 0.0), ("trail5", 0.05), ("trail10", 0.10)]


def _panel(sym, cadence):
    df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence, features=["open", "high", "low", "close"])
    idx = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor(FLOOR[cadence])
    sub = pd.DataFrame({"o": df["open"].to_numpy().astype(float), "h": df["high"].to_numpy().astype(float),
                        "l": df["low"].to_numpy().astype(float), "c": df["close"].to_numpy().astype(float)},
                       index=idx)
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()
    ms = (sub.index.asi8 // 10**6).astype("int64")
    return sub["o"].to_numpy(), sub["h"].to_numpy(), sub["l"].to_numpy(), sub["c"].to_numpy(), ms


def run(universe="u10", cadences=("4h", "1h", "30m", "15m"), start="2020-01-07", end="2020-02-07",
        max_configs=8, maker=False):
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    syms = [a["symbol"] for a in spec["assets"]]
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    cost = MAKER_RT if maker else TAKER_RT
    # inject the distinct configs so holding_state can resolve them by name
    specs = {}
    for fam in ("2MA", "3MA"):
        specs.update(distinct_specs(fam, 0.15, max_n=max_configs))
    PR.STRATS.update(specs)

    cells = []
    for cad in cadences:
        for sym in syms:
            try:
                o, h, l, c, ms = _panel(sym, cad)
            except Exception:
                continue
            # keep history UP TO window-end only (warmup preserved; post-window bars are unused and
            # would loop 2020->2026 -- ~223k bars at 15m). Correct for any window; tiny for the oldest.
            keep = ms < e_ms
            o, h, l, c, ms = o[keep], h[keep], l[keep], c[keep], ms[keep]
            if (ms >= s_ms).sum() < 5:               # asset has ~no data in the window
                continue
            for name, (fam, _p) in specs.items():
                held0 = holding_state(name, o, h, l, c).astype(np.int8)
                for ex_name, trail in EXITS:
                    held = held0
                    if trail > 0:
                        held = apply_trail_stop(held0.copy(), c, trail)[0].astype(np.int8)
                    trades = per_asset_trades(o, c, held, ms, cost)
                    wt = [t for t in trades if s_ms <= t["entry_ms"] < e_ms]
                    if not wt:
                        continue
                    rets = np.array([t["ret"] for t in wt])
                    holds = np.array([t["hold"] for t in wt])
                    comp = float(np.prod(1 + rets) - 1)
                    cells.append({"asset": sym, "cadence": cad, "family": fam, "config": name,
                                  "exit": ex_name, "compound_pct": round(comp * 100, 2),
                                  "n_trades": len(wt), "avg_hold_bars": round(float(holds.mean()), 1),
                                  "avg_hold_hours": round(float(holds.mean()) * TF_HOURS[cad], 1),
                                  "win_rate": round(float((rets > 0).mean()), 2)})
    return {"window": f"{start}..{end}", "universe": universe, "max_configs": max_configs,
            "n_cells": len(cells), "cells": cells}


def per_instrument(cells):
    """Aggregate the flat cells into a per-asset summary."""
    out = {}
    by_asset = {}
    for r in cells:
        by_asset.setdefault(r["asset"], []).append(r)
    for asset, rs in by_asset.items():
        rs_sorted = sorted(rs, key=lambda x: x["compound_pct"], reverse=True)
        best, worst = rs_sorted[0], rs_sorted[-1]
        all_holds = [r["avg_hold_bars"] for r in rs]
        # best timeframe by mean compound across its cells
        tf_mean = {}
        for r in rs:
            tf_mean.setdefault(r["cadence"], []).append(r["compound_pct"])
        tf_rank = sorted(((np.mean(v), k) for k, v in tf_mean.items()), reverse=True)
        out[asset] = {
            "best": f"{best['config']}/{best['exit']}/{best['cadence']} = {best['compound_pct']}% "
                    f"({best['n_trades']}tr, hold {best['avg_hold_hours']}h)",
            "worst": f"{worst['config']}/{worst['exit']}/{worst['cadence']} = {worst['compound_pct']}%",
            "best_tf": f"{tf_rank[0][1]} (mean {tf_rank[0][0]:.1f}%)",
            "worst_tf": f"{tf_rank[-1][1]} (mean {tf_rank[-1][0]:.1f}%)",
            "avg_hold_bars": round(float(np.mean(all_holds)), 1),
            "mean_compound_pct": round(float(np.mean([r["compound_pct"] for r in rs])), 2),
            "n_cells": len(rs),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_per_instrument")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadences", default="4h,1h,30m,15m")
    ap.add_argument("--start", default="2020-01-07")
    ap.add_argument("--end", default="2020-02-07")
    ap.add_argument("--max-configs", type=int, default=8)
    ap.add_argument("--maker", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.universe, tuple(a.cadences.split(",")), a.start, a.end, a.max_configs, a.maker)
    pin = per_instrument(res["cells"])

    print(f"## MA PER-INSTRUMENT -- {a.universe} -- window {res['window']} -- {res['n_cells']} cells "
          f"({a.max_configs} distinct configs/family x {len(EXITS)} exits x {a.cadences})")
    print(f"\n{'asset':9} {'mean%':>7} {'best (config/exit/tf)':38} {'best_tf':16} {'avgHold':>8}")
    for asset, s in sorted(pin.items(), key=lambda kv: kv[1]["mean_compound_pct"], reverse=True):
        print(f"{asset:9} {s['mean_compound_pct']:>7} {s['best'][:38]:38} {s['best_tf']:16} "
              f"{s['avg_hold_bars']:>6}b")
    # overall winners/losers (top + bottom cells across everything)
    allc = sorted(res["cells"], key=lambda x: x["compound_pct"], reverse=True)
    print("\nTOP 8 cells (winners):")
    for r in allc[:8]:
        print(f"  {r['asset']:9} {r['config']:14} {r['exit']:10} {r['cadence']:4} "
              f"{r['compound_pct']:>7}%  {r['n_trades']}tr win{r['win_rate']} hold{r['avg_hold_hours']}h")
    print("BOTTOM 6 cells (losers):")
    for r in allc[-6:]:
        print(f"  {r['asset']:9} {r['config']:14} {r['exit']:10} {r['cadence']:4} "
              f"{r['compound_pct']:>7}%  {r['n_trades']}tr win{r['win_rate']} hold{r['avg_hold_hours']}h")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"ma_per_instrument_{a.universe}_{stamp}.json"
    json.dump({"summary": pin, **res}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
