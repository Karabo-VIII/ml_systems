"""src/strat/mr_diversify.py -- does an ORTHOGONAL-BETA archetype (mean-reversion) diversify the trend ceiling?

WHY: the MA+breakout ensemble failed to diversify because both are SAME beta (long-only crypto trend). The
textbook escape from a beta-tail ceiling is an ORTHOGONAL beta. Mean-reversion (buy oversold dips, sell at
the mean) is a DIFFERENT beta (contrarian / short-vol), so a TREND+MR ensemble is the diversification that
*could* lift the held-out p05 where trend+trend didn't. HONEST PRIOR: the dead-list (D37 crypto trends not
reverts; D48/D49 buy-the-extreme is anti-edge) says MR is likely individually NEGATIVE on crypto -- but the
DIVERSIFICATION value (anti-correlation with the trend book) is a SEPARATE question this tests directly.

Long-only MR (no short): enter when causal z = (close-SMA(N))/std(N) < -entry_k (oversold), exit when
z > -exit_k (reverted). Compare: MR standalone / breakout standalone / TREND+MR ensemble, with the
breakout-vs-MR book CORRELATION + ensemble OOS-heldout p05 vs breakout-alone (-21). 4h, u10, UNSEEN sealed.
RWYB: python -m strat.mr_diversify. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.scorecard import score_book
from strat.breakout_arc import donchian_held, _configs as _bo_configs

WARMUP = 600
PERIODS = {"Jun2022_bear": ("2022-06-01", "2022-07-01"),
           "VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}
MR_GRID = [(n, ek, xk) for n in (10, 20, 40) for ek in (1.5, 2.0, 2.5) for xk in (0.5,)]


def _zscore(c, n):
    s = pd.Series(c)
    m = s.rolling(n, min_periods=n).mean().shift(1)
    sd = s.rolling(n, min_periods=n).std().shift(1)          # causal: prior-bar stats
    return ((s - m) / (sd + 1e-12)).to_numpy()


def mr_held(c, n, ek, xk):
    """long-only mean-reversion: enter z<-ek (oversold), exit z>-xk (reverted toward mean)."""
    z = _zscore(c, n)
    held = np.zeros(len(c), dtype=np.int8); cur = 0
    for i in range(len(c)):
        if np.isnan(z[i]):
            cur = 0
        elif cur == 0 and z[i] < -ek:
            cur = 1
        elif cur == 1 and z[i] > -xk:
            cur = 0
        held[i] = cur
    return held


def _mr_full(c, n, ek, xk):
    h = mr_held(c, n, ek, xk).astype(np.int8)
    h = min_hold(h, 6).astype(np.float64)                    # shorter min-hold (MR holds are brief)
    return h


def _bo_full(c, n, m):
    h = donchian_held(c, n, m).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    return min_hold(h, 12).astype(np.float64)


def cells(which, start, end, date_index=False):
    bo_cfg, bo_slow = _bo_configs()
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, cell_roi, series = [], [], {}
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, "4h")
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c2) < 30:
            continue
        wm = ms2 >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        ws = []
        if which in ("bo", "ens"):
            ws += [_bo_full(c2, *bo_cfg[name]) for name in bo_slow]
        if which in ("mr", "ens"):
            ws += [_mr_full(c2, n, ek, xk) for (n, ek, xk) in MR_GRID]
        for w in ws:
            pos = np.zeros(len(c2)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))
            per_cell.append(net[wm]); cell_roi.append(float(np.cumprod(1 + net[wm])[-1] - 1) * 100)
            if date_index:
                series.setdefault(sym, []).append(pd.Series(net[wm], index=pd.to_datetime(ms2[wm], unit="ms")))
    return per_cell, cell_roi, series


def book_daily(which):
    _, _, series = cells(which, "2018-01-01", "2025-12-31", date_index=True)
    allc = [x for lst in series.values() for x in lst]
    if not allc:
        return None
    b4 = pd.concat(allc, axis=1).mean(axis=1, skipna=True)
    return b4.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()


def mt(per_cell):
    if not per_cell:
        return {}
    m = min(len(x) for x in per_cell); bk = np.mean([x[:m] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}


def main() -> int:
    print(f"MR diversify: {len(MR_GRID)} MR configs vs breakout; does orthogonal-beta diversify the ceiling?\n")
    print(f"   {'book':10}" + "".join(f"{p:>16}" for p in PERIODS) + f"{'OOS %pos':>10}")
    rows = {}
    for which, label in [("mr", "MR_only"), ("bo", "BREAKOUT"), ("ens", "BO+MR_ens")]:
        line = f"   {label:10}"
        for plabel, (s, e) in PERIODS.items():
            pc, _, _ = cells(which, s, e)
            rows[(label, plabel)] = mt(pc)
            line += f"{(str(rows[(label,plabel)].get('roi'))+'/'+str(rows[(label,plabel)].get('maxdd'))):>16}" if rows[(label, plabel)] else f"{'--':>16}"
        _, orois, _ = cells(which, *PERIODS["OOS"])
        line += f"{(str(round(100*float(np.mean(np.array(orois)>0))) if orois else '?')+'%'):>10}"
        print(line)

    # the diversification test: correlation + ensemble p05 vs breakout-alone
    bo_d = book_daily("bo"); mr_d = book_daily("mr"); ens_d = book_daily("ens")
    print("\n[DIVERSIFICATION TEST]")
    if bo_d is not None and mr_d is not None:
        j = pd.concat([bo_d.rename("bo"), mr_d.rename("mr")], axis=1).dropna()
        corr = float(j["bo"].corr(j["mr"]))
        print(f"   corr(breakout daily, MR daily) = {corr:+.3f}  "
              f"({'orthogonal/anti -> diversification POSSIBLE' if corr < 0.3 else 'too correlated -> no diversification'})")
    sc = {}
    for d, label in [(bo_d, "BREAKOUT"), (mr_d, "MR_only"), (ens_d, "BO+MR_ens")]:
        if d is None:
            continue
        card = score_book(label, d)
        oosp = card["per_split"].get("OOS", {}); hb = card["heldout_block_bootstrap"]
        sc[label] = {"oos_compound": oosp.get("compound_pct"), "oos_sharpe": oosp.get("sharpe"),
                     "heldout_p05": hb.get("p05"), "unseen_n": card["per_split"].get("UNSEEN", {}).get("n", 0)}
        print(f"   {label:10} OOS compound {str(oosp.get('compound_pct')):>7}%  Sharpe {str(oosp.get('sharpe')):>5}  "
              f"OOS-heldout p05 {str(hb.get('p05')):>7}  (UNSEEN n={sc[label]['unseen_n']})")
    bo_p05 = sc.get("BREAKOUT", {}).get("heldout_p05"); ens_p05 = sc.get("BO+MR_ens", {}).get("heldout_p05")
    if bo_p05 is not None and ens_p05 is not None:
        better = ens_p05 > bo_p05
        crossed = ens_p05 > 0
        print(f"\n   VERDICT: ensemble p05 {ens_p05} vs breakout-alone {bo_p05} -> "
              f"{'CLEARS p05>0 -- robust!' if crossed else ('IMPROVES tail (diversifies) but still <0' if better else 'does NOT improve -- ceiling holds')}")
    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "mr_diversify.json"
    json.dump({"rows": {f"{k[0]}|{k[1]}": v for k, v in rows.items()}, "scorecard": sc}, open(out, "w"), indent=1, default=str)
    print(f"[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
