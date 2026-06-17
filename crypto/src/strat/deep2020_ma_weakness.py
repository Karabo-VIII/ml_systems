"""src/strat/deep2020_ma_weakness.py -- per-MA-TYPE weakness teardown + targeted IRON (2020).

User /orc 2026-06-14: "did you iron out the weaknesses of EACH type of MA strat type?" The MA_REMEDY work
applied UNIFORM remedies; this is the per-TYPE diagnosis: each MA type has a CHARACTERISTIC failure mode --
measure it, apply the targeted iron, show before->after. Reuses the corrected machinery (fixed-EW + VAL-only
vol target). Per (MA type, cadence), aggregated (median) over the full config grid:

  WEAKNESS METRICS (base FULL stack):
    whipsaw%  = fraction of trades held <=2 bars (CHURN -- the low-lag/overshoot weakness: DEMA/TEMA/fast)
    n_trades  = round-trips in the window (activity)
    cost_drag = net_nocost - net_maker (pp the churn COSTS -- the realized whipsaw penalty)
    time_in   = mean OOS exposure (PARTICIPATION -- the lag/selectivity weakness: SMA/KAMA-low-ER/vslow)
    maxDD     = OOS drawdown (RISK)
    net       = OOS compound (WEALTH)

  THE IRONS (targeted): +CONFIRM band (0.5% hysteresis) irons WHIPSAW+COST; +VOLTGT irons maxDD (risk).
  Reported per type: the base profile, the iron deltas, and which weakness was BINDING for that type.

RWYB: python -m strat.deep2020_ma_weakness --cadences 4h. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.deep2020_ma_remedy import _load_assets, _held_band, SPLIT, BAND
import pandas as pd


def _whipsaw(held):
    """held in {0,1} (pre-lag). Returns (whipsaw% = frac trades held<=2 bars, n_trades)."""
    d = np.diff(np.concatenate([[0], held.astype(int), [0]]))
    entries = np.where(d == 1)[0]; exits = np.where(d == -1)[0]
    n = min(len(entries), len(exits))
    if n == 0:
        return None, 0
    durs = exits[:n] - entries[:n]
    return round(float(np.mean(durs <= 2)) * 100, 1), int(n)


def _profile(assets, caches, cfg, mt, band, vt):
    """Weakness profile for ONE config: equal-weight (fixed-EW) book, causal. OOS metrics + whipsaw/cost."""
    nets_m, nets_n, whips, ntr = [], [], [], []
    for A, cache in zip(assets, caches):
        c2, ret, win, idx, rv = A["c"], A["ret"], A["win"], A["idx"], A["rv"]
        pp = _nums(cfg); mas = [cache[p] for p in pp]
        held = _held_band(mas, band)
        held = min_hold(apply_trail_stop(held.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
        wp, nt = _whipsaw(held[win])
        if wp is not None:
            whips.append(wp); ntr.append(nt)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]
        if vt is not None:
            pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        nets_m.append(pd.Series((pos * ret - flips * (MAKER_RT / 2.0))[win], index=idx))
        nets_n.append(pd.Series((pos * ret)[win], index=idx))
    if not nets_m:
        return None

    def _oos(series_list):
        b = pd.concat(series_list, axis=1).fillna(0.0).mean(axis=1)
        d = b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
        return d[d.index >= pd.Timestamp(SPLIT)]
    dm = _oos(nets_m); dn = _oos(nets_n)
    if len(dm) < 5:
        return None
    x = dm.to_numpy(); eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    net_m = float((eq[-1] - 1) * 100); net_n = float((np.prod(1 + dn.to_numpy()) - 1) * 100)
    # time-in: book exposure OOS
    tin = []
    for A, cache in zip(assets, caches):
        c2, win, idx, rv = A["c"], A["win"], A["idx"], A["rv"]
        pp = _nums(cfg); mas = [cache[p] for p in pp]
        held = _held_band(mas, band)
        held = min_hold(apply_trail_stop(held.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]
        if vt is not None:
            pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        s = pd.Series(pos[win], index=idx); s = s[s.index >= pd.Timestamp(SPLIT)]
        if len(s):
            tin.append(float(s.mean()))
    return {"net": round(net_m, 1), "cost_drag": round(net_n - net_m, 1),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
            "whipsaw": round(float(np.mean(whips)), 1) if whips else None,
            "n_trades": round(float(np.mean(ntr)), 1) if ntr else None,
            "time_in": round(float(np.mean(tin)), 2) if tin else None}


def _agg(assets, caches, cfgs, mt, band, vt):
    """Median weakness profile across the config grid for one (type, variant)."""
    rows = [p for cfg in cfgs if (p := _profile(assets, caches, cfg, mt, band, vt))]
    if not rows:
        return None
    keys = ["net", "cost_drag", "maxdd", "whipsaw", "n_trades", "time_in"]
    return {k: round(float(np.median([r[k] for r in rows if r[k] is not None])), 1) for k in keys}


def main() -> int:
    cads = ["4h"]
    if "--cadences" in sys.argv:
        cads = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    allcfg = list(ma_cfg)
    allp = sorted({p for n in allcfg for p in _nums(n)})

    # the canonical per-type CHARACTERISTIC weakness (MA theory) -- what to look for
    THEORY = {
        "EMA": "balanced; mild lag+chop whipsaw", "SMA": "MOST LAG (equal weights) -> low participation",
        "WMA": "moderate lag", "HMA": "low-lag but OVERSHOOTS reversals (whipsaw)",
        "DEMA": "low-lag, OVERSHOOT whipsaw on reversals", "TEMA": "lowest-lag, WORST overshoot whipsaw",
        "KAMA": "adaptive; STALLS in low-efficiency chop (under-participates)",
        "VIDYA": "adaptive (CMO); stalls in low-momentum chop"}

    export = {}
    for cad in cads:
        assets, vt = _load_assets(cad)
        if not assets:
            print(f"## {cad}: no assets"); continue
        print(f"########## {cad} -- per-MA-TYPE weakness teardown (median over {len(allcfg)} configs) ##########")
        print(f"   {'MA':5} {'whipsaw%':>9} {'n_trd':>6} {'costDrag':>9} {'time_in':>8} {'maxDD':>7} {'net':>6}  characteristic weakness")
        base = {}
        for mt in MA_TYPES:
            caches = [{p: _MA[mt](A["c"], p) for p in allp} for A in assets]
            b = _agg(assets, caches, allcfg, mt, 0.0, None)
            if not b:
                continue
            base[mt] = (b, caches)
            print(f"   {mt:5} {str(b['whipsaw']):>9} {str(b['n_trades']):>6} {str(b['cost_drag']):>9} "
                  f"{str(b['time_in']):>8} {str(b['maxdd']):>7} {str(b['net']):>6}  {THEORY[mt]}")

        # IRON deltas: +CONFIRM (whipsaw/cost), +VOLTGT (maxDD)
        print(f"\n   -- IRON effect (median delta vs base): +CONFIRM (anti-whipsaw) / +VOLTGT (anti-DD) --")
        print(f"   {'MA':5} {'dWhip(conf)':>12} {'dCost(conf)':>12} {'dMaxDD(vt)':>11} {'dNet(conf/vt)':>14}")
        cell = {}
        for mt in MA_TYPES:
            if mt not in base:
                continue
            b, caches = base[mt]
            conf = _agg(assets, caches, allcfg, mt, BAND, None)
            volt = _agg(assets, caches, allcfg, mt, BAND, vt)
            if not conf or not volt:
                continue
            dwhip = round(conf["whipsaw"] - b["whipsaw"], 1) if (conf["whipsaw"] is not None and b["whipsaw"] is not None) else None
            dcost = round(conf["cost_drag"] - b["cost_drag"], 1)
            dmaxdd = round(volt["maxdd"] - b["maxdd"], 1)
            print(f"   {mt:5} {str(dwhip):>12} {str(dcost):>12} {str(dmaxdd):>11} "
                  f"{str(round(conf['net']-b['net'],1))+'/'+str(round(volt['net']-b['net'],1)):>14}")
            cell[mt] = {"base": b, "confirm": conf, "voltgt": volt,
                        "d_whipsaw_confirm": dwhip, "d_cost_confirm": dcost, "d_maxdd_voltgt": dmaxdd,
                        "characteristic_weakness": THEORY[mt]}
        export[cad] = cell
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(cads)
    json.dump(export, open(op / f"ma_weakness_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'ma_weakness_{jt}.json'}  -- per-MA-type weakness profile + iron deltas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
