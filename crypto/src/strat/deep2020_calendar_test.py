"""src/strat/deep2020_calendar_test.py -- BLOCK N: does the calendar structure PERSIST + is it TRADABLE?

Block M found significant calendar structure (weekend + US-hours drift) -- the one orthogonal-to-beta signal.
This stress-tests it: (1) PERSISTENCE -- split 2020 H2 into halves (Jul-Sep vs Oct-Dec); do the day-of-week
means AGREE in sign across halves? (a real effect persists; noise flips). (2) TRADABILITY -- does a
calendar-timed long-only strategy (long only on the historically-positive days/hours, flat otherwise) beat
buy-hold / equal-weight, net of cost? Honest grade. RWYB: python -m strat.deep2020_calendar_test. No emoji.
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

from strat.deep2020_seasonality import _pooled_returns, DOW
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TAKER = 0.0024


def _half_means(s, keyfn, lo, hi):
    sub = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))]
    return sub.groupby(sub.index.map(keyfn)).mean() * 1e4


def main() -> int:
    out = {}
    # ---- (1) PERSISTENCE: DOW means in the two halves ----
    s = _pooled_returns("1d")
    h1 = _half_means(s, lambda t: t.dayofweek, "2020-07-01", SPLIT)
    h2 = _half_means(s, lambda t: t.dayofweek, SPLIT, "2021-01-01")
    print("BLOCK N calendar persistence + tradability\n")
    print("## (1) DAY-OF-WEEK mean (bp/day) -- Jul-Sep vs Oct-Dec (sign agreement = persistence):")
    print(f"   {'day':5} {'Jul-Sep':>9} {'Oct-Dec':>9} {'agree?':>7}")
    agree = 0
    for k in range(7):
        a = float(h1.get(k, np.nan)); b = float(h2.get(k, np.nan))
        ag = np.isfinite(a) and np.isfinite(b) and np.sign(a) == np.sign(b)
        agree += int(ag)
        print(f"   {DOW[k]:5} {a:>9.1f} {b:>9.1f} {('YES' if ag else 'no'):>7}")
    rho = float(np.corrcoef([h1.get(k, 0) for k in range(7)], [h2.get(k, 0) for k in range(7)])[0, 1])
    print(f"   -> {agree}/7 days agree in sign; cross-half DOW-profile corr = {rho:+.2f}")
    out["dow_sign_agree"] = f"{agree}/7"; out["dow_halfcorr"] = round(rho, 2)

    # ---- (2) TRADABILITY: weekend-long + good-hours-long strategies vs buy-hold ----
    # build the per-bar equal-weight book at 1d and 1h
    def book(cad):
        w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
        cols = []
        for sym in SYMS:
            try:
                o, h, l, c, ms = _panel(sym, cad)
            except Exception:
                continue
            keep = (ms >= w0) & (ms < w1); c2 = c[keep]
            if len(c2) < 20:
                continue
            r = np.zeros(len(c2)); r[1:] = c2[1:] / c2[:-1] - 1
            cols.append(pd.Series(r, index=pd.to_datetime(ms[keep], unit="ms")))
        return pd.concat(cols, axis=1).mean(axis=1).dropna()

    b1 = book("1d"); b1h = book("1h")
    # weekend strategy: long Sat+Sun only (1d)
    wk = b1.index.dayofweek >= 5
    wk_ret = b1.where(wk, 0.0)
    # cost: enter/exit each weekend ~ 2 trades/week
    turn1 = np.abs(np.diff(np.concatenate([[0], wk.astype(int)]))).sum()
    wk_net = float(np.prod(1 + wk_ret.to_numpy()) - 1) * 100 - turn1 * (TAKER / 2) * 100
    bh1 = float(np.prod(1 + b1.to_numpy()) - 1) * 100
    # good-hours strategy: long only hours with positive full-sample t>1 (from Block M: 0,1,6,7,13,15,21,23)
    goodhrs = {0, 1, 6, 7, 13, 15, 21, 23}
    gh = b1h.index.hour.isin(list(goodhrs))
    gh_ret = b1h.where(gh, 0.0)
    turnh = np.abs(np.diff(np.concatenate([[0], gh.astype(int)]))).sum()
    gh_net = float(np.prod(1 + gh_ret.to_numpy()) - 1) * 100 - turnh * (TAKER / 2) * 100
    bh1h = float(np.prod(1 + b1h.to_numpy()) - 1) * 100
    print("\n## (2) TRADABILITY (net of taker cost):")
    print(f"   weekend-long (Sat+Sun, 1d): net {wk_net:.1f}%  vs buy-hold {bh1:.1f}%  (time-in {wk.mean()*100:.0f}%)")
    print(f"   good-hours-long (8 UTC hrs, 1h): net {gh_net:.1f}%  vs buy-hold {bh1h:.1f}%  (time-in {gh.mean()*100:.0f}%)")
    # risk-adjusted per-unit-time: return per fraction-of-time-invested
    print(f"   return-per-time-in: weekend {wk_net/max(wk.mean(),1e-9):.0f}% vs BH {bh1:.0f}%; "
          f"good-hours {gh_net/max(gh.mean(),1e-9):.0f}% vs BH {bh1h:.0f}%")
    out["weekend_net"] = round(wk_net, 1); out["weekend_bh"] = round(bh1, 1)
    out["goodhours_net"] = round(gh_net, 1); out["goodhours_bh"] = round(bh1h, 1)

    persists = agree >= 5 and rho > 0.3
    print(f"\n## VERDICT: persistence {agree}/7 sign-agree, corr {rho:+.2f} -> "
          f"{'PERSISTS within 2020' if persists else 'WEAK/does-not-persist across halves (likely period-specific)'}. "
          f"Tradable lift per-time-in: weekend {'YES' if wk_net/max(wk.mean(),1e-9) > bh1 else 'no'}.")
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op / "calendar_test.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'calendar_test.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
