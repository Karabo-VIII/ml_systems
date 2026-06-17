"""src/strat/deep2020_stats.py -- BLOCK D: the STATISTICAL / MATHEMATICAL structure of 2020.

User /orc 2026-06-13: "there is a lot to learn still (statistically and mathematically speaking)." Per
(instrument, timeframe) on 2020 H2 it computes:
  CONCENTRATION  -- what % of the total up-return comes from the top 5/10/20% of up-bars (the 'few big
                    bars' phenomenon -- if the bull is concentrated, MISSING those bars is catastrophic);
  BIG-BAR CAPTURE-- the MA family's avg position DURING the top-decile up-bars vs its avg position overall
                    (does the lagging MA cross CATCH or MISS the biggest moves? -- the core participation Q);
  TREND PERSIST  -- Hurst (variance-ratio), lag-1 autocorrelation, longest up-run (how trending was 2020);
  TRADE MATH     -- from the EMA family: win rate, avg-win/avg-loss (payoff), expectancy, median hold-bars.
RWYB: python -m strat.deep2020_stats --cadences <tf>. No emoji (cp1252).
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
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01")
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def _hurst_vr(r, k=10):
    r = r[np.isfinite(r)]
    if len(r) < k * 4:
        return None
    v1 = np.var(r)
    rk = np.add.reduceat(r, np.arange(0, len(r) - len(r) % k, k))
    vk = np.var(rk)
    if v1 <= 0 or vk <= 0:
        return None
    vr = vk / (k * v1)
    return float(0.5 + 0.5 * np.log(vr) / np.log(k))     # >0.5 trending, <0.5 mean-reverting


def _max_run(sign):
    best = cur = 0
    for s in sign:
        cur = cur + 1 if s > 0 else 0
        best = max(best, cur)
    return best


def _stats(sym, cad, slow):
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
    logret = np.zeros(len(c2)); logret[1:] = np.log(c2[1:] / c2[:-1])
    r = logret[win]
    up = r[r > 0]
    total_up = up.sum()
    ups = np.sort(up)[::-1]
    def topk(p):
        k = max(1, int(len(ups) * p)); return float(ups[:k].sum() / total_up) if total_up > 0 else None
    # MA family position
    uniq = sorted({p for n in slow for p in _nums(n)}); cache = {p: _MA["EMA"](c2, p) for p in uniq}
    poss = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
        h1 = apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8)
        held = min_hold(h1, 12).astype(np.float64)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]; poss.append(pos)
    fpos = np.mean(poss, axis=0)[win]
    # big-bar capture: avg position during the top-decile up-bars
    thr = np.quantile(r[r > 0], 0.90) if (r > 0).any() else np.inf
    big = r >= thr
    big_cap = float(np.mean(fpos[big])) if big.any() else None
    # trade stats from ONE representative slow config (ema_62_89 if present else first)
    rep = next((n for n in slow if _nums(n) == [62, 89]), slow[0])
    pp = _nums(rep); mas = [cache[p] for p in pp]
    h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
    held = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.int8)
    d = np.diff(np.concatenate([[0], held, [0]])); starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    trs = [(np.exp(logret[s:e2].sum()) - 1) for s, e2 in zip(starts, ends) if e2 <= len(c2)]
    holds = [e2 - s for s, e2 in zip(starts, ends)]
    trs = np.array([t for t in trs if np.isfinite(t)])
    wins = trs[trs > 0]; losses = trs[trs < 0]
    payoff = float(np.mean(wins) / abs(np.mean(losses))) if len(wins) and len(losses) else None
    return {"conc_top5": topk(0.05), "conc_top10": topk(0.10), "conc_top20": topk(0.20),
            "bigbar_capture": None if big_cap is None else round(big_cap, 3),
            "avg_position": round(float(np.mean(fpos)), 3),
            "hurst": _hurst_vr(r), "ac1": round(float(np.corrcoef(r[:-1], r[1:])[0, 1]), 3) if len(r) > 5 else None,
            "max_up_run": _max_run(np.sign(r)),
            "win_rate": round(float(len(wins) / len(trs)), 2) if len(trs) else None,
            "payoff": None if payoff is None else round(payoff, 2),
            "expectancy_pct": round(float(np.mean(trs)) * 100, 2) if len(trs) else None,
            "n_trades": len(trs), "median_hold": int(np.median(holds)) if holds else None}


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK D stats: {len(slow)} configs x {len(CADENCES)} TF; 2020 H2\n")

    out = {}
    for cad in CADENCES:
        print(f"########## {cad} -- concentration / big-bar capture / persistence / trade math (per instrument) ##########")
        print(f"   {'sym':9} {'top5%':>6} {'top10%':>7} {'bigcap':>7} {'avgpos':>7} {'hurst':>6} {'ac1':>6} {'uprun':>6} "
              f"{'win%':>5} {'payoff':>7} {'exp%':>6} {'hold':>5}")
        rows = []
        for sym in SYMS:
            d = _stats(sym, cad, slow)
            if d is None:
                continue
            out[(cad, sym)] = d; rows.append(d)
            print(f"   {sym.replace('USDT',''):9} {str(round(d['conc_top5'],2) if d['conc_top5'] else '-'):>6} "
                  f"{str(round(d['conc_top10'],2) if d['conc_top10'] else '-'):>7} {str(d['bigbar_capture']):>7} "
                  f"{str(d['avg_position']):>7} {str(round(d['hurst'],2) if d['hurst'] else '-'):>6} {str(d['ac1']):>6} "
                  f"{str(d['max_up_run']):>6} {str(d['win_rate']):>5} {str(d['payoff']):>7} {str(d['expectancy_pct']):>6} {str(d['median_hold']):>5}")
        if rows:
            def a(k):
                v = [r[k] for r in rows if r[k] is not None]; return float(np.mean(v)) if v else float("nan")
            print(f"   {'MEAN':9} {a('conc_top5'):>6.2f} {a('conc_top10'):>7.2f} {a('bigbar_capture'):>7.2f} "
                  f"{a('avg_position'):>7.2f} {a('hurst'):>6.2f} {a('ac1'):>6.3f} {a('max_up_run'):>6.0f} "
                  f"{a('win_rate'):>5.2f} {a('payoff'):>7.2f} {a('expectancy_pct'):>6.2f} {a('median_hold'):>5.0f}")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump({f"{c}|{s}": d for (c, s), d in out.items()}, open(op / f"stats_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'stats_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
