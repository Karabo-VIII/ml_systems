"""src/strat/deep2020_clusters.py -- per-MA-type per-TF strategy CLUSTERS + TOP-K (the ML-harness target).

User /orc 2026-06-13: "what works in CLUSTER/statistical form (a family of clusters of strats that work),
per MA type SEPARATELY (EMA/SMA/.../VIDYA), top-K per MA per timeframe, weaknesses solved (min-hold = FULL
stack); then an ML harness will replicate the best-in-class of each."

For each (timeframe, MA type) over the full distinct config grid (2MA+3MA, ALL speeds), with the FULL stack
(trail10 + min_hold12 + maker = weaknesses fixed), graded on the 2020 OOS (Oct-Dec):
  CLUSTERS  -- group configs by {structure 2MA/3MA} x {speed: fast<20 / mid20-60 / slow60-150 / vslow>=150}
               and report each cluster's STATISTICAL profile: n, mean+-std OOS net, mean Sharpe, mean maxDD,
               mean time-in, medoid config. The winning cluster = the family that works.
  TOP-K     -- the best K configs per (MA, TF) by OOS Sharpe (the ML target: best-in-class to replicate).
  BEHAVIOR  -- mean within-cluster return correlation (confirms the eff-N~1.2 'tight cluster' structure).
Exports clusters.json (the ML-harness input). RWYB: python -m strat.deep2020_clusters --cadences <tf>.
No emoji (cp1252).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}


def _speed(periods):
    m = max(periods)
    return "fast<20" if m < 20 else "mid20-60" if m < 60 else "slow60-150" if m < 150 else "vslow150+"


def _config_books(cfgs, ma_type, cad):
    """per config: OOS daily return series (FULL stack book = mean across assets)."""
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    per_cfg = {n: [] for n in cfgs}
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        idx = pd.to_datetime(ms2[win], unit="ms")
        uniq = sorted({p for n in cfgs for p in _nums(n)}); cache = {p: _MA[ma_type](c2, p) for p in uniq}
        for name in cfgs:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
            per_cfg[name].append(pd.Series(net, index=idx))
    books = {}
    for name, cols in per_cfg.items():
        if cols:
            b4 = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
            d = b4.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
            books[name] = d
    return books


def _name(cfg, ma_type):
    return f"{ma_type}({','.join(str(p) for p in _nums(cfg))})"     # MA-type + periods (ema_ prefix is just a label)


def _metrics(d, ann=None):
    oos = d[d.index >= pd.Timestamp(SPLIT)].to_numpy()
    if len(oos) < 5:
        return None
    eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    # series is DAILY-resampled -> annualize with sqrt(365) for ALL cadences (comparable across TF)
    return {"net": round(float((eq[-1] - 1) * 100), 1), "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(dd, 1)}


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    allcfg = list(ma_cfg)
    print(f"CLUSTERS: {len(allcfg)} configs (all speeds) x {len(MA_TYPES)} MA x {len(CADENCES)} TF; OOS 2020\n")

    export = {}
    for cad in CADENCES:
        for mt in MA_TYPES:
            books = _config_books(allcfg, mt, cad)
            recs = []
            oosidx = None
            for name, d in books.items():
                m = _metrics(d)
                if m is None:
                    continue
                nums = _nums(name); struct = "2MA" if len(nums) == 2 else "3MA"
                recs.append({"cfg": _name(name, mt), "struct": struct, "speed": _speed(nums), "cluster": f"{struct}-{_speed(nums)}",
                             "oos": d[d.index >= pd.Timestamp(SPLIT)], **m})
            if not recs:
                continue
            # cluster stats
            clusters = {}
            for r in recs:
                clusters.setdefault(r["cluster"], []).append(r)
            cstats = {}
            for cl, rs in clusters.items():
                nets = np.array([x["net"] for x in rs]); shs = np.array([x["sharpe"] for x in rs])
                # within-cluster mean correlation (tightness)
                if len(rs) > 1:
                    M = pd.concat([x["oos"].rename(i) for i, x in enumerate(rs)], axis=1).fillna(0.0).corr().to_numpy()
                    mc = float((M.sum() - len(rs)) / (len(rs) * (len(rs) - 1)))
                else:
                    mc = 1.0
                medoid = max(rs, key=lambda x: x["sharpe"])["cfg"]
                cstats[cl] = {"n": len(rs), "mean_net": round(float(nets.mean()), 1), "std_net": round(float(nets.std()), 1),
                              "mean_sharpe": round(float(shs.mean()), 2), "mean_maxdd": round(float(np.mean([x["maxdd"] for x in rs])), 1),
                              "within_corr": round(mc, 2), "medoid": medoid}
            top = sorted(recs, key=lambda x: -x["sharpe"])[:5]
            export[f"{cad}|{mt}"] = {"clusters": cstats,
                                     "top5": [{"cfg": t["cfg"], "cluster": t["cluster"], "net": t["net"], "sharpe": t["sharpe"], "maxdd": t["maxdd"]} for t in top]}
            best_cl = max(cstats, key=lambda c: cstats[c]["mean_sharpe"])
            print(f"## {cad} {mt}: best CLUSTER = {best_cl} (mean Sharpe {cstats[best_cl]['mean_sharpe']}, "
                  f"mean net {cstats[best_cl]['mean_net']}%+-{cstats[best_cl]['std_net']}, within-corr {cstats[best_cl]['within_corr']}, n={cstats[best_cl]['n']})")
            print(f"   top5: " + " | ".join(f"{t['cfg']}(Sh{t['sharpe']},{t['net']}%)" for t in top))

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump(export, open(op / f"clusters_{jt}.json", "w"), indent=1, default=str)
    print(f"\n[json] {op / f'clusters_{jt}.json'}  (the ML-harness target: clusters + top-5 per MA per TF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
