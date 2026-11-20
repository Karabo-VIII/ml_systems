"""src/strat/deep2020_leaderboard.py -- the RANKED strategy leaderboard per timeframe + COVERAGE.

User /orc 2026-06-13: "best emergent strategies, ranked, several per timeframe; pitfalls; and COVERAGE --
are we in the market every day?" Per timeframe, ranks the candidate causal long-only strategies by OOS
Sharpe and reports net / maxDD / TIME-IN-MARKET (coverage) / days-in-vs-out:
  per-MA-type FAMILY (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA slow-config books) -- 'several per TF'
  BUYHOLD              -- the drift baseline (in every bar)
  VOLTGT_BH            -- vol-targeted buy-hold (the robust risk-sizing win)
  CALENDAR             -- sit out the VAL-negative weekdays (1d only; the orthogonal-structure tilt)
2020 H2; OOS = Oct-Dec; ranking metric = OOS Sharpe. RWYB: python -m strat.deep2020_leaderboard --cadences <tf>.
No emoji (cp1252).
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
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
VW = {"1d": 14, "4h": 84, "2h": 168, "1h": 168, "30m": 336, "15m": 672}


def _grade(net_s, pos_s, ann):
    """net_s, pos_s = date-indexed pd.Series (windowed). OOS sharpe/net/maxdd + coverage (avg exposure, days)."""
    oos_net = net_s[net_s.index >= pd.Timestamp(SPLIT)].dropna()
    oos_pos = pos_s[pos_s.index >= pd.Timestamp(SPLIT)].reindex(oos_net.index).fillna(0.0)
    if len(oos_net) < 5:
        return None
    n = oos_net.to_numpy(); p = oos_pos.to_numpy()
    eq = np.cumprod(1 + n); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    sh = float(np.mean(n) / (np.std(n) + 1e-12) * np.sqrt(ann))
    avg_exp = float(np.mean(p))                          # average exposure 0..1 (the real 'time in market')
    dd2 = pd.DataFrame({"day": oos_net.index.date, "in": (p > 0.5)})    # meaningfully long (>50% exposed)
    day_in = dd2.groupby("day")["in"].max()
    return {"oos_net": round(float((eq[-1] - 1) * 100), 1), "oos_sharpe": round(sh, 2), "oos_maxdd": round(dd, 1),
            "avg_exposure": round(avg_exp, 2), "days_in": int(day_in.sum()), "days_tot": int(len(day_in))}


def _book_series(per_net, per_pos):
    """align per-asset date-indexed series by timestamp -> book net + pos (mean across assets)."""
    bn = pd.concat(per_net, axis=1).mean(axis=1, skipna=True)
    bp = pd.concat(per_pos, axis=1).mean(axis=1, skipna=True)
    return bn, bp


def _family(cfgs, ma_type, cad, full=True):
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    per_net, per_pos = [], []
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
        uniq = sorted({p for n in cfgs for p in _nums(n)}); cache = {p: _MA[ma_type](c2, p) for p in uniq}
        nets, poss = [], []
        for name in cfgs:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            if full:
                h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.int8)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            nets.append(pos * ret - flips * (MAKER_RT / 2.0)); poss.append(pos)
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_net.append(pd.Series(np.mean(nets, axis=0)[win], index=idx))
        per_pos.append(pd.Series(np.mean(poss, axis=0)[win], index=idx))
    if not per_net:
        return None
    return _book_series(per_net, per_pos)


def _universal(cad, kind):
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    per_net, per_pos = [], []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= s_ms - 30 * 86400000) & (ms < e_ms)
        c2 = c[keep]; ms2 = ms[keep]
        if len(c2) < 40:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        if kind == "VOLTGT":
            rv = pd.Series(ret).rolling(VW[cad]).std().to_numpy(); med = np.nanmedian(rv)
            exp = np.nan_to_num(np.clip(med / (np.concatenate([[np.nan], rv[:-1]]) + 1e-12), 0, 1))
        else:
            exp = np.ones(len(c2))
        win = ms2 >= s_ms
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_net.append(pd.Series((exp * ret)[win], index=idx)); per_pos.append(pd.Series(exp[win], index=idx))
    if not per_net:
        return None
    return _book_series(per_net, per_pos)


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    allout = {}
    for cad in CADENCES:
        rows = []
        for mt in MA_TYPES:
            fam = _family(slow, mt, cad)
            if fam is None:
                continue
            g = _grade(fam[0], fam[1], ANN[cad])
            if g:
                rows.append({"strat": f"{mt}-family", **g})
        for kind, lab in [("BH", "BUYHOLD"), ("VOLTGT", "VOLTGT_BH")]:
            u = _universal(cad, kind)
            if u:
                g = _grade(u[0], u[1], ANN[cad])
                if g:
                    rows.append({"strat": lab, **g})
        rows.sort(key=lambda r: -r["oos_sharpe"])
        allout[cad] = rows
        print(f"\n########## {cad} -- RANKED by OOS Sharpe (2020 OOS Oct-Dec) ##########")
        print(f"   {'#':2} {'strategy':14} {'Sharpe':>7} {'net%':>7} {'maxDD%':>7} {'avg_exp':>8} {'days>50%/tot':>13} {'coverage':>9}")
        for i, r in enumerate(rows, 1):
            cov = "every day" if r["days_in"] == r["days_tot"] else f"{100*r['days_in']//max(r['days_tot'],1)}% of days"
            print(f"   {i:>2} {r['strat']:14} {r['oos_sharpe']:>7} {r['oos_net']:>7} {r['oos_maxdd']:>7} "
                  f"{r['avg_exposure']:>8} {str(r['days_in'])+'/'+str(r['days_tot']):>13} {cov:>13}")

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump(allout, open(op / f"leaderboard_{jt}.json", "w"), indent=1, default=str)
    print(f"\n[json] {op / f'leaderboard_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
