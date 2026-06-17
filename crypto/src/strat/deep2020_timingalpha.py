"""src/strat/deep2020_timingalpha.py -- BLOCK F: is there ANY timing skill, or just exposure? (rigorous)

The grand synthesis claims a long-only MA strat is a DE-RISKED buy-hold with no timing alpha. This tests it
rigorously. For each (instrument, TF) it asks: does the MA family go long during BETTER-THAN-AVERAGE bars?
  ret_per_exposure = sum(pos*ret) / sum(pos)   -- mean return on the bars we are long (per unit exposure)
  buyhold_per_bar  = mean(ret)                 -- the unconditional mean (what 100% exposure earns/bar)
  timing_alpha     = ret_per_exposure - buyhold_per_bar
NULL (the key): SHUFFLE the position series (preserve time-in-market, destroy the timing) 300x -> the null
distribution of ret_per_exposure. p = frac(null >= actual). If actual >> null -> the MA TIMES well (long
during good bars = skill). If actual ~= null -> NO timing skill; the strat is random positioning of the
same exposure = de-risked buy-hold. RWYB: python -m strat.deep2020_timingalpha --cadences <tf>. No emoji.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01")
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
NPERM = 300


def _ta(sym, cad, slow, seed=7):
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    e = int(np.searchsorted(ms, w1)); s0 = max(0, int(np.searchsorted(ms, w0)) - WARMUP)
    c2, ms2 = c[s0:e], ms[s0:e]
    if len(c2) < 60:
        return None
    win = ms2 >= w0
    if win.sum() < 50:
        return None
    ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
    uniq = sorted({p for n in slow for p in _nums(n)}); cache = {p: _MA["EMA"](c2, p) for p in uniq}
    poss = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
        held = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]; poss.append(pos)
    fpos = np.mean(poss, axis=0)[win]; r = ret[win]
    if fpos.sum() < 5:
        return None
    actual = float(np.sum(fpos * r) / np.sum(fpos))
    bh = float(np.mean(r))
    rng = np.random.default_rng(seed)
    null = np.empty(NPERM)
    for i in range(NPERM):
        sp = rng.permutation(fpos)
        null[i] = np.sum(sp * r) / np.sum(sp)
    p = float(np.mean(null >= actual))
    return {"ret_per_exposure_bp": round(actual * 1e4, 2), "buyhold_per_bar_bp": round(bh * 1e4, 2),
            "timing_alpha_bp": round((actual - bh) * 1e4, 2), "null_mean_bp": round(float(np.mean(null)) * 1e4, 2),
            "p_value": round(p, 3), "time_in": round(float(np.mean(fpos)), 3)}


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK F timing-alpha (permutation null, {NPERM}x): {len(CADENCES)} TF; 2020 H2\n")

    out = {}
    for cad in CADENCES:
        print(f"########## {cad} -- ret/exposure vs buyhold/bar (bp); timing_alpha; permutation p (p<0.05 = real skill) ##########")
        print(f"   {'sym':9} {'ret/exp_bp':>11} {'BH/bar_bp':>10} {'alpha_bp':>9} {'null_bp':>8} {'p':>6} {'time_in':>8}")
        rows = []
        for sym in SYMS:
            d = _ta(sym, cad, slow)
            if d is None:
                continue
            out[(cad, sym)] = d; rows.append(d)
            sig = "*" if d["p_value"] < 0.05 else (" " if d["p_value"] > 0.2 else ".")
            print(f"   {sym.replace('USDT',''):9} {d['ret_per_exposure_bp']:>11} {d['buyhold_per_bar_bp']:>10} "
                  f"{d['timing_alpha_bp']:>9} {d['null_mean_bp']:>8} {d['p_value']:>5}{sig} {d['time_in']:>8}")
        if rows:
            def a(k):
                return float(np.mean([r[k] for r in rows]))
            nsig = sum(1 for r in rows if r["p_value"] < 0.05)
            print(f"   {'MEAN':9} {a('ret_per_exposure_bp'):>11.1f} {a('buyhold_per_bar_bp'):>10.1f} "
                  f"{a('timing_alpha_bp'):>9.1f} {a('null_mean_bp'):>8.1f} {a('p_value'):>6.2f} {a('time_in'):>8.2f}  "
                  f"[{nsig}/{len(rows)} instruments p<0.05]")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump({f"{c}|{s}": d for (c, s), d in out.items()}, open(op / f"timingalpha_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'timingalpha_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
