"""src/strat/portfolio_analysis.py -- SOTA portfolio ANALYSIS surface, paired with strat/config inputs.

WHY (user /orc 2026-06-11): *"recheck portfolio, upgrade it to SOTA, and we will be using it for
analysis. Pair it with strat inputs / config inputs."*

WHAT THIS IS. The ANALYSIS layer on top of the portfolio_replay engine (src/strat/portfolio_replay.py,
built+validated+trail-stopped by the parallel instance -- its validity suite proves MtM self-
consistency / cost-recon / determinism). This file does NOT edit that engine (two agents editing one
file risks losing edits); it COMPOSES it as the kernel and adds the two things the analysis use-case
needs:
  1. STRAT/CONFIG-INPUT PAIRING -- drive the engine from the distinct-config grid
     (config/distinct_strategy_grid.yaml, the 852 deduped specs) OR any strat/config spec file OR
     inline names. The engine's STRATS dict is otherwise a hardcoded 12 names; this opens it to the
     whole decomposed config universe as first-class input.
  2. SOTA ANALYSIS OUTPUT -- richer descriptive metrics computed from the engine's own net/equity
     series: CAGR, ann-vol, Sharpe, Sortino, Calmar, maxDD + drawdown DURATION, exposure, turnover,
     rolling 1d/3d/7d ROI (median + positive-rate), and the equal-weight buy&hold anchor. Plus
     per-strategy hold-bar attribution. DESCRIPTIVE analysis (this is the analysis tool, not a verdict
     machine) -- it reports what the book did; it does not rank-and-refute.

Two modes:
  --portfolio (default) : run the selected strat/config SET as ONE risk-budgeted book -> full analysis.
  --per-config          : run each selected config as its own 1-strategy book -> per-config attribution
                          table (no value-judgement; just the descriptive panel).

All replay honesty is inherited from the engine (causal, weights lagged 1 bar, MtM-no-double-count,
taker/maker cost, optional trailing-stop). Fill realism: taker is solid; maker uses the (provisional)
calibration -- flagged, per the MakerCostModel invariant.

RWYB:
  python -m strat.portfolio_analysis --grid --family 2MA --max-configs 12 --window ALL   # portfolio
  python -m strat.portfolio_analysis --strategies "ema_50_100,donch20" --window OOS
  python -m strat.portfolio_analysis --strategies-file config/distinct_strategy_grid.yaml --family 3MA --per-config
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR                       # the engine (kernel) -- composed, not edited
from strat.replay_distinct_grid import distinct_specs     # the decomposition -> spec builder
from framework.discovery_contract import canonicalize_grid  # noqa: F401 (kept for parity / future)

GRID_YAML = ROOT.parent / "config" / "distinct_strategy_grid.yaml"
OUT = ROOT.parent / "runs" / "strat"


# ---------------------------------------------------------------------------------------------
# STRAT/CONFIG-INPUT PAIRING -- resolve specs from grid / file / inline, inject into the engine
# ---------------------------------------------------------------------------------------------
def _grid_to_specs(doc: dict) -> dict:
    """distinct_strategy_grid.yaml {name:{family,**params}} -> PR-form {name:(family,params)}."""
    out = {}
    for name, d in (doc.get("strategies") or {}).items():
        d = dict(d)
        fam = d.pop("family")
        out[name] = (fam, d)
    return out


def resolve_inputs(args) -> dict:
    """The unified strat/config INPUT surface. Returns {name:(family,params)} and injects into
    PR.STRATS so the engine can replay them by name."""
    specs = {}
    if args.strategies_file:
        doc = yaml.safe_load(Path(args.strategies_file).read_text(encoding="utf-8"))
        specs.update(_grid_to_specs(doc) if "strategies" in doc else doc)
    if args.grid:
        specs.update(_grid_to_specs(yaml.safe_load(GRID_YAML.read_text(encoding="utf-8"))))
    if args.family:                                        # filter by family (e.g. only 2MA)
        specs = {n: v for n, v in specs.items() if v[0] == args.family}
        if not specs:                                      # family not in the loaded set -> generate it
            specs = distinct_specs(args.family, args.rel_tol, max_n=args.max_configs)
    if args.max_configs and len(specs) > args.max_configs:  # evenly sample to keep tractable
        names = sorted(specs)
        idx = [round(i * (len(names) - 1) / (args.max_configs - 1)) for i in range(args.max_configs)]
        specs = {names[i]: specs[names[i]] for i in sorted(set(idx))}
    if args.strategies:                                    # inline names (from STRATS or loaded grid)
        inline = [s.strip() for s in args.strategies.split(",") if s.strip()]
        # keep inline names that are either already-known (PR.STRATS) or resolvable in specs
        specs.update({n: specs[n] for n in inline if n in specs})
        for n in inline:
            if n in PR.STRATS:
                specs[n] = PR.STRATS[n]
    PR.STRATS.update(specs)                                # <-- pairing: configs become engine inputs
    return specs


# ---------------------------------------------------------------------------------------------
# SOTA ANALYSIS METRICS (descriptive) computed from the engine's net/equity series
# ---------------------------------------------------------------------------------------------
def analyze(r: dict, cadence: str) -> dict:
    """Richer descriptive metrics from the engine's returned _net / _equity. Adds Sortino, Calmar,
    drawdown DURATION, exposure, rolling 1d/3d/7d to what the engine already reports."""
    net = np.asarray(r.get("_net", []), dtype=float)
    eq = np.asarray(r.get("_equity", []), dtype=float)
    ann = PR.ANN.get(cadence, 365)
    if net.size < 5 or eq.size < 5:
        return {"note": "series too short for analysis"}
    downside = net[net < 0]
    sortino = float(net.mean() / (downside.std() + 1e-12) * np.sqrt(ann)) if downside.size else None
    peak = np.maximum.accumulate(eq)
    under = eq < peak * (1 - 1e-9)
    # longest underwater run (in bars) -> convert to days
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    cagr = r.get("ann_pct")
    calmar = float(cagr / abs(r["maxdd_pct"])) if r.get("maxdd_pct") else None

    def roll(h):
        if eq.size <= h:
            return (None, None)
        rr = eq[h:] / eq[:-h] - 1
        return (round(float(np.median(rr) * 100), 3), round(float((rr > 0).mean()), 3))

    r1m, r1p = roll(1); r3m, r3p = roll({"1d": 3, "4h": 18, "1h": 72, "30m": 144, "15m": 288}.get(cadence, 3))
    r7m, r7p = roll({"1d": 7, "4h": 42, "1h": 168, "30m": 336, "15m": 672}.get(cadence, 7))
    return {
        "CAGR_pct": cagr, "ann_vol_pct": round(float(net.std() * np.sqrt(ann) * 100), 1),
        "sharpe": r.get("sharpe"), "sortino": round(sortino, 2) if sortino is not None else None,
        "calmar": round(calmar, 2) if calmar is not None else None,
        "maxdd_pct": r.get("maxdd_pct"), "maxdd_duration_days": round(longest / (ann / 365), 1),
        "exposure_avg_gross": r.get("avg_gross"), "avg_turnover": r.get("avg_daily_turnover"),
        "roll1d_median_pct": r1m, "roll1d_pos": r1p,
        "roll3d_median_pct": r3m, "roll3d_pos": r3p,
        "roll7d_median_pct": r7m, "roll7d_pos": r7p,
        "final_pct": r.get("final_pct"), "n_bars": r.get("n_bars"),
        "per_strat_hold_bars": r.get("per_strat_hold_bars"),
    }


def _bh_anchor(r: dict) -> dict:
    """Equal-weight buy&hold of the SAME window/universe from the engine's persisted return panel
    (the descriptive sanity anchor -- not a verdict)."""
    rp = r.get("_ret_panel_window")
    if rp is None or getattr(rp, "empty", True):
        return {}
    bh = (1 + rp.mean(axis=1)).cumprod()
    return {"bh_final_pct": round(float((bh.iloc[-1] - 1) * 100), 1)}


def run_portfolio(specs, args) -> dict:
    cost = PR.MAKER_RT if args.maker else PR.TAKER_RT
    names = list(specs)
    r = PR.run(args.universe, args.cadence, names, args.window, cost, args.spine,
               args.vol_target, args.max_per_name, trail_stop=args.trail_stop)
    if "error" in r:
        return {"error": r["error"]}
    a = analyze(r, args.cadence); a.update(_bh_anchor(r))
    return {"mode": "portfolio", "n_strategies": len(names), "strategies": names, "analysis": a}


def run_per_config(specs, args) -> dict:
    cost = PR.MAKER_RT if args.maker else PR.TAKER_RT
    rows = []
    for name in specs:
        r = PR.run(args.universe, args.cadence, [name], args.window, cost, args.spine,
                   args.vol_target, args.max_per_name, trail_stop=args.trail_stop)
        if "error" in r:
            continue
        a = analyze(r, args.cadence)
        rows.append({"strategy": name, **{k: a.get(k) for k in
                     ("final_pct", "CAGR_pct", "sharpe", "sortino", "calmar", "maxdd_pct",
                      "maxdd_duration_days", "roll3d_pos")}})
    return {"mode": "per_config", "n_configs": len(rows), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.portfolio_analysis")
    # strat/config INPUT surface (the pairing)
    ap.add_argument("--grid", action="store_true", help="load config/distinct_strategy_grid.yaml")
    ap.add_argument("--strategies-file", default=None, help="a YAML of strat/config specs")
    ap.add_argument("--strategies", default=None, help="inline names (from STRATS or the loaded grid)")
    ap.add_argument("--family", default=None, help="filter/generate a family: 2MA|3MA|ROC|TREND|DONCH|RSI|BOLL")
    ap.add_argument("--max-configs", type=int, default=12)
    ap.add_argument("--rel-tol", type=float, default=0.15)
    # engine knobs (passthrough)
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--window", default="ALL")
    ap.add_argument("--start", default=None, help="custom window start YYYY-MM-DD (overrides --window)")
    ap.add_argument("--end", default=None, help="custom window end YYYY-MM-DD (exclusive)")
    ap.add_argument("--maker", action="store_true")
    ap.add_argument("--spine", action="store_true")
    ap.add_argument("--vol-target", type=float, default=0.02)
    ap.add_argument("--max-per-name", type=float, default=0.15)
    ap.add_argument("--trail-stop", type=float, default=0.0)
    ap.add_argument("--per-config", action="store_true", help="per-config attribution instead of one book")
    a = ap.parse_args(argv)
    if a.start or a.end:                                   # custom window -> engine CUSTOM split
        PR.WIN["CUSTOM"] = (a.start, a.end); a.window = "CUSTOM"

    if not (a.grid or a.strategies_file or a.strategies or a.family):
        a.strategies = "ema_50_100,donch20,rsi_30_50"      # sensible default set
    specs = resolve_inputs(a)
    if not specs:
        print("no strat/config inputs resolved; pass --grid / --strategies-file / --strategies / --family")
        return 2

    out = run_per_config(specs, a) if a.per_config else run_portfolio(specs, a)
    hdr = (f"## PORTFOLIO ANALYSIS -- {a.universe} {a.cadence} window={a.window} "
           f"-- {'maker' if a.maker else 'taker'} -- sizing={'spine' if a.spine else 'inverse-vol'}"
           f"{' -- trail '+str(a.trail_stop) if a.trail_stop else ''}")
    print(hdr)
    if out.get("error"):
        print(f"   {out['error']}"); return 0
    if out["mode"] == "portfolio":
        an = out["analysis"]
        print(f"   strat/config inputs: {out['n_strategies']} -> one risk-budgeted book")
        bh = f"  B&H {an['bh_final_pct']}%" if an.get("bh_final_pct") is not None else ""
        print(f"   final {an.get('final_pct')}%  CAGR {an.get('CAGR_pct')}%  vol {an.get('ann_vol_pct')}%{bh}")
        print(f"   Sharpe {an.get('sharpe')}  Sortino {an.get('sortino')}  Calmar {an.get('calmar')}  "
              f"maxDD {an.get('maxdd_pct')}% ({an.get('maxdd_duration_days')}d underwater)")
        print(f"   exposure {an.get('exposure_avg_gross')}  turnover {an.get('avg_turnover')}")
        print(f"   roll ROI pos-rate: 1d {an.get('roll1d_pos')}  3d {an.get('roll3d_pos')}  7d {an.get('roll7d_pos')}"
              f"  | median 3d {an.get('roll3d_median_pct')}%")
    else:
        print(f"   {'strategy':18} {'final%':>8} {'CAGR%':>7} {'Shrp':>5} {'Sort':>5} {'Calm':>5} {'maxDD%':>7} {'DD_d':>5}")
        def _f(x, n=1):
            return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)
        for r in out["rows"]:
            print(f"   {r['strategy']:18} {_f(r['final_pct']):>8} {_f(r['CAGR_pct']):>7} "
                  f"{_f(r['sharpe'],2):>5} {_f(r['sortino'],2):>5} {_f(r['calmar'],2):>5} "
                  f"{_f(r['maxdd_pct']):>7} {_f(r['maxdd_duration_days']):>5}")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"portfolio_analysis_{a.universe}_{a.window}_{out['mode']}_{stamp}.json"
    json.dump({"spec": vars(a), "out": out}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"   [persisted] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
