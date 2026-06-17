"""src/strat/replay_distinct_grid.py -- BRIDGE: distinct-config decomposition -> portfolio_replay.

WHY (user /orc 2026-06-11): (1) execute the next-value item -- emit the DISTINCT configs (from
canonicalize_grid) as a CONSUMABLE search-grid artifact the discovery/replay engine can load; (2)
"plug into" the portfolio_replay engine the other instance built (src/strat/portfolio_replay.py).

HOW IT PLUGS IN (non-invasively -- does NOT edit portfolio_replay.py, which may be mid-edit):
  - generates the DISTINCT (near-dup-free) config set per family via canonicalize_grid,
  - formats them as portfolio_replay strategy specs  name -> (family, params),
  - INJECTS them into portfolio_replay.STRATS at runtime (module global; holding_state reads it),
  - calls portfolio_replay.run(...) per distinct config -> a ranked replay LEADERBOARD on a window,
  - AND emits config/distinct_strategy_grid.yaml = the loadable grid (the consumable artifact).

So: decomposition (the search space) -> portfolio_replay (paper-trade each) -> ranked findings. The
deduped grid means the leaderboard searches ~306 REAL alternatives, not thousands of noise-twins.

Honest scope: 2MA/3MA/DONCH/ROC/TREND map cleanly (lengths are the params). RSI(lo,hi)/BOLL(n,k) carry
threshold/mult axes my length-decomposition does not cover -- emitted with a small standard param set
and flagged. Distinct != tradeable (dead-list: pooled-per-cadence wins; per-asset dead). All replay
honesty (causal, lagged weights, MtM-no-double-count, cost) is inherited from portfolio_replay.

RWYB:
  python -m strat.replay_distinct_grid --emit-grid                                  # write the artifact
  python -m strat.replay_distinct_grid --family 2MA --max-configs 24 --window OOS   # leaderboard
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.discovery_contract import canonicalize_grid          # the dedup callable
from strat import ti_config_decompose as TD                          # reuse the raw-grid generators
import strat.portfolio_replay as PR                                  # the engine we plug into

GRID_OUT = ROOT.parent / "config" / "distinct_strategy_grid.yaml"
RUN_OUT = ROOT.parent / "runs" / "strat"


def _sample(seq, max_n):
    """Evenly sample <= max_n items from an ordered list (keeps endpoints)."""
    if max_n <= 0 or len(seq) <= max_n:
        return list(seq)
    idx = [round(i * (len(seq) - 1) / (max_n - 1)) for i in range(max_n)]
    return [seq[i] for i in sorted(set(idx))]


def distinct_specs(family: str, rel_tol: float = 0.15, ma_type: str = "EMA", max_n: int = 0) -> dict:
    """Return {name: (PR_family, params)} for the DISTINCT configs of a family, in portfolio_replay form."""
    specs = {}
    if family == "2MA":
        reps = canonicalize_grid(TD._ma_cross(), rel_tol).representatives
        for (f, s) in _sample(reps, max_n):
            f, s = int(f), int(s)
            specs[f"{ma_type.lower()}_{f}_{s}"] = ("2MA", dict(type=ma_type, fast=f, slow=s))
    elif family == "3MA":
        reps = canonicalize_grid(TD._ma_triple(), rel_tol).representatives
        for (a, b, c) in _sample(reps, max_n):
            a, b, c = int(a), int(b), int(c)
            specs[f"{ma_type.lower()}_{a}_{b}_{c}"] = ("3MA", dict(type=ma_type, fast=a, mid=b, slow=c))
    elif family in ("ROC", "TREND", "DONCH"):
        reps = canonicalize_grid(TD._single_length(), rel_tol).representatives
        for (L,) in _sample(reps, max_n):
            L = int(L)
            if family == "ROC":
                specs[f"roc{L}"] = ("ROC", dict(n=L))
            elif family == "TREND":
                specs[f"trend{L}"] = ("TREND", dict(n=L))
            else:
                specs[f"donch{L}"] = ("DONCH", dict(n=L, exit_n=max(2, L // 2)))
    elif family == "RSI":
        # threshold axis (NOT length) -- a small standard canonical set, flagged in the artifact
        for lo, hi in [(20, 50), (30, 50), (30, 60), (25, 55), (35, 65)]:
            specs[f"rsi_{lo}_{hi}"] = ("RSI", dict(lo=lo, hi=hi))
    elif family == "BOLL":
        for n in _sample([int(L) for (L,) in canonicalize_grid(TD._single_length(), 0.25).representatives], 8):
            for k in (2, 3):
                specs[f"boll{n}_{k}"] = ("BOLL", dict(n=n, k=k))
    else:
        raise ValueError(family)
    return specs


def emit_grid(rel_tol: float = 0.15) -> dict:
    """Write config/distinct_strategy_grid.yaml -- the loadable, deduped search grid (the artifact)."""
    fams = ["2MA", "3MA", "ROC", "TREND", "DONCH", "RSI", "BOLL"]
    grid = {}
    counts = {}
    for fam in fams:
        # 3MA full set is large; emit it at a coarser tol to keep the file consumable
        rt = 0.25 if fam == "3MA" else rel_tol
        s = distinct_specs(fam, rt)
        counts[fam] = len(s)
        for name, (f, p) in s.items():
            grid[name] = {"family": f, **p}
    doc = {
        "meta": {
            "built": "2026-06-11", "rel_tol": rel_tol,
            "source": "canonicalize_grid over each family's param sweep (near-dup eliminated)",
            "consumed_by": "src/strat/portfolio_replay.py (inject into STRATS) + any config sweep",
            "counts": counts, "total": len(grid),
            "caveats": ["RSI/BOLL carry threshold/mult axes (not length) -- standard set, not full sweep",
                        "distinct != tradeable: pooled-per-cadence wins, per-asset dead (dead-list)",
                        "timeframe-AGNOSTIC: same set at every cadence; selection is per-cadence"],
        },
        "strategies": dict(sorted(grid.items())),
    }
    GRID_OUT.write_text(yaml.safe_dump(doc, sort_keys=False, width=100), encoding="utf-8")
    return {"path": str(GRID_OUT), "counts": counts, "total": len(grid)}


def replay_leaderboard(family="2MA", rel_tol=0.15, max_configs=24, universe="u10", cadence="1d",
                       window="OOS", maker=False, spine=False, vol_target=0.02, max_per_name=0.15) -> dict:
    """Inject distinct configs into portfolio_replay.STRATS and replay each as a single-strategy
    portfolio over `window`; rank by final %. Reuses the engine (no edits to portfolio_replay.py)."""
    specs = distinct_specs(family, rel_tol, max_n=max_configs)
    PR.STRATS.update(specs)                                  # <-- the plug-in: inject into the engine
    cost = PR.MAKER_RT if maker else PR.TAKER_RT
    rows = []
    for name in specs:
        r = PR.run(universe, cadence, [name], window, cost, spine, vol_target, max_per_name)
        if "error" in r:
            continue
        rows.append({"strategy": name, "final_pct": r["final_pct"], "ann_pct": r["ann_pct"],
                     "maxdd_pct": r["maxdd_pct"], "sharpe": r["sharpe"],
                     "roll3d_pos": r["roll3d_pos_rate"], "n_bars": r["n_bars"]})
    rows.sort(key=lambda x: x["final_pct"], reverse=True)
    return {"family": family, "rel_tol": rel_tol, "n_configs": len(specs), "universe": universe,
            "cadence": cadence, "window": window, "cost": "maker" if maker else "taker",
            "sizing": "spine" if spine else "inverse-vol", "leaderboard": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.replay_distinct_grid")
    ap.add_argument("--emit-grid", action="store_true", help="write config/distinct_strategy_grid.yaml")
    ap.add_argument("--family", default="2MA", help="2MA|3MA|ROC|TREND|DONCH|RSI|BOLL")
    ap.add_argument("--rel-tol", type=float, default=0.15)
    ap.add_argument("--max-configs", type=int, default=24, help="cap the leaderboard (evenly sampled)")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--window", default="OOS")
    ap.add_argument("--maker", action="store_true")
    ap.add_argument("--spine", action="store_true")
    ap.add_argument("--no-leaderboard", action="store_true")
    a = ap.parse_args(argv)

    if a.emit_grid:
        g = emit_grid(a.rel_tol)
        print(f"## DISTINCT STRATEGY GRID emitted -> {g['path']}")
        print(f"   counts per family: {g['counts']}  TOTAL {g['total']} distinct specs")
        if a.no_leaderboard:
            return 0

    lb = replay_leaderboard(a.family, a.rel_tol, a.max_configs, a.universe, a.cadence, a.window,
                            a.maker, a.spine)
    print(f"\n## REPLAY LEADERBOARD over the DISTINCT {lb['family']} grid "
          f"({lb['n_configs']} configs) -- {lb['universe']} {lb['cadence']} window={lb['window']} "
          f"-- {lb['cost']} -- sizing={lb['sizing']}")
    print(f"   {'rank':>4} {'strategy':18} {'final%':>8} {'ann%':>7} {'maxDD%':>7} {'Sharpe':>7} {'3dpos':>6}")
    for i, r in enumerate(lb["leaderboard"][:20], 1):
        print(f"   {i:>4} {r['strategy']:18} {r['final_pct']:>8.1f} {r['ann_pct']:>7.1f} "
              f"{r['maxdd_pct']:>7.1f} {r['sharpe']:>7.2f} {str(r['roll3d_pos']):>6}")
    if not lb["leaderboard"]:
        print("   (no valid runs -- window too short or data missing)")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUN_OUT / f"replay_distinct_{lb['family']}_{lb['universe']}_{lb['window']}_{stamp}.json"
    json.dump(lb, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"   [persisted] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
