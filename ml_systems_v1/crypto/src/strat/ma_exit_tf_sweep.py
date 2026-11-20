"""src/strat/ma_exit_tf_sweep.py -- replay the DECOUPLED (distinct) MA configs x EXIT mechanism x TIMEFRAME.

WHY (user /orc 2026-06-11): *"replay the decoupled MA configs 2MA and 3MA, with different exit
mechanisms in different timeframes and report on the results of the performance. Pick the oldest month
we have in our books."* This is the analysis stack put to work on MA: the distinct-config decomposition
(canonicalize_grid) feeds the portfolio_replay engine (validated), swept over exit x cadence, reported.

THE GRID (per cadence):
  family : 2MA (distinct fast,slow pairs)  +  3MA (distinct a,b,c triples)   -- from the deduped grid
  exit   : signal-flip (MA cross reverses; the STRAT exit, trail=0)
           trail-5% / trail-10% (high-water trailing stop; the MECHANICAL exit, --trail-stop)
  cadence: 1d / 4h / 1h / 30m / 15m
Each cell = the family's distinct configs run as ONE risk-budgeted book over the window, with that exit.

OLDEST MONTH: our data starts 2020-01-07; default window = 2020-01-07 -> 2020-02-07 (a calendar month).
HONEST CAVEAT (reported): a 1-month DAILY window gives slow MAs no warmup (data starts at the window),
so 1d MA is signal-starved in the oldest month; intraday MAs warm up within days. Annualized CAGR /
Sortino are unstable on a 1-month window -- the headline metrics are window-final % + maxDD + Sharpe.

Reuses: framework.discovery_contract.canonicalize_grid (decomposition), portfolio_replay.run (engine),
portfolio_analysis.analyze (metrics). Descriptive performance report -- not a verdict machine.

RWYB:
  python -m strat.ma_exit_tf_sweep --cadence 4h                       # one cadence (6 cells)
  python -m strat.ma_exit_tf_sweep --all-cadences --start 2020-01-07 --end 2020-02-07
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_analysis import analyze
from strat.replay_distinct_grid import distinct_specs

OUT = ROOT.parent / "runs" / "strat"
EXITS = [("signalflip", 0.0), ("trail5", 0.05), ("trail10", 0.10)]
FAMILIES = ["2MA", "3MA"]
ALL_CADENCES = ["1d", "4h", "1h", "30m", "15m"]


def run_cell(family, trail, cadence, universe, window, max_configs, vol_target, max_per_name, maker):
    specs = distinct_specs(family, 0.15, max_n=max_configs)
    PR.STRATS.update(specs)
    cost = PR.MAKER_RT if maker else PR.TAKER_RT
    r = PR.run(universe, cadence, list(specs), window, cost, False, vol_target, max_per_name,
               trail_stop=trail)
    if "error" in r:
        return {"error": r["error"]}
    a = analyze(r, cadence)
    return {k: a.get(k) for k in ("final_pct", "CAGR_pct", "ann_vol_pct", "sharpe", "sortino",
            "calmar", "maxdd_pct", "maxdd_duration_days", "exposure_avg_gross", "avg_turnover",
            "roll3d_pos", "n_bars")}


def sweep_cadence(cadence, universe, window, max_configs, vol_target, max_per_name, maker):
    rows = []
    for family in FAMILIES:
        for exit_name, trail in EXITS:
            cell = run_cell(family, trail, cadence, universe, window, max_configs, vol_target,
                            max_per_name, maker)
            rows.append({"cadence": cadence, "family": family, "exit": exit_name, **cell})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_exit_tf_sweep")
    ap.add_argument("--cadence", default="4h")
    ap.add_argument("--all-cadences", action="store_true")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--start", default="2020-01-07")
    ap.add_argument("--end", default="2020-02-07")
    ap.add_argument("--max-configs", type=int, default=8)
    ap.add_argument("--vol-target", type=float, default=0.02)
    ap.add_argument("--max-per-name", type=float, default=0.15)
    ap.add_argument("--maker", action="store_true")
    a = ap.parse_args(argv)
    PR.WIN["CUSTOM"] = (a.start, a.end)
    window = "CUSTOM"
    cadences = ALL_CADENCES if a.all_cadences else [a.cadence]

    all_rows = []
    for cad in cadences:
        all_rows.extend(sweep_cadence(cad, a.universe, window, a.max_configs, a.vol_target,
                                      a.max_per_name, a.maker))

    print(f"## MA EXIT x TIMEFRAME SWEEP -- {a.universe} -- window {a.start}..{a.end} -- "
          f"{'maker' if a.maker else 'taker'} -- {a.max_configs} distinct configs/family")
    print(f"   {'cadence':8} {'family':6} {'exit':10} {'final%':>9} {'maxDD%':>8} {'Sharpe':>7} "
          f"{'expo':>5} {'turn':>6} {'bars':>5}")
    def _f(x, n=1):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)
    for r in all_rows:
        if r.get("error"):
            print(f"   {r['cadence']:8} {r['family']:6} {r['exit']:10}  {r['error']}"); continue
        print(f"   {r['cadence']:8} {r['family']:6} {r['exit']:10} {_f(r['final_pct']):>9} "
              f"{_f(r['maxdd_pct']):>8} {_f(r['sharpe'],2):>7} {_f(r['exposure_avg_gross'],2):>5} "
              f"{_f(r['avg_turnover'],3):>6} {str(r['n_bars']):>5}")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "all" if a.all_cadences else a.cadence
    p = OUT / f"ma_exit_tf_sweep_{a.universe}_{tag}_{stamp}.json"
    json.dump({"spec": vars(a), "rows": all_rows}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"   [persisted] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
