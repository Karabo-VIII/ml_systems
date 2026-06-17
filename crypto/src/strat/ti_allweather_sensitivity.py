"""src/strat/ti_allweather_sensitivity.py -- robustness of the TIER-A all-weather candidates to the ROLLING
hyperparameters (lookback, step). Addresses the #1 caveat: the rolling-pick lookback(120d)/step(30d) were
in-sample-tuned -- does the bull->bear all-weather result (esp. 2022-bear flat-or-positive) HOLD across a grid,
or is it a single-point artifact?

USER /orc 2026-06-16 (3h, end-to-end candidates): the candidate register flagged 6 TIER-A all-weather TIs
(MACD/TSI/KELTNER/MFI/RSI/PSAR). This sweeps lookback in {60,90,120,180} x step in {15,30,60} per candidate and
reports the per-year net distribution -> a candidate is HYPERPARAMETER-ROBUST iff its 2022-bear net stays
flat-or-mild (>= -10%) AND both bulls stay positive across the WHOLE grid (not just the tuned 120/30 point).

Reuses ti_band_rolling._ti_series (the per-config daily-net builder) + a parametrized rolling. NO look-ahead.
Long-only spot, fixed-EW, maker, 4h. No emoji. RWYB: python -m strat.ti_allweather_sensitivity
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ti_band_rolling import _ti_series, _net, YEARS                    # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
TIER_A = ["MACD", "TSI", "KELTNER", "MFI", "RSI", "PSAR"]
LOOKBACKS = [60, 90, 120, 180]
STEPS = [15, 30, 60]


def _rolling_pick(series_df, lookback, step):
    """Walk-forward rolling-pick (recent-best trailing-positive config), parametrized lookback/step. No look-ahead."""
    idx = series_df.index; cfgs = list(series_df.columns)
    t = idx.min() + pd.Timedelta(days=lookback)
    pieces = []
    while t < idx.max():
        nxt = t + pd.Timedelta(days=step)
        look = series_df[(idx >= t - pd.Timedelta(days=lookback)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        ln = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(cfgs, ln) if v > 0] or [cfgs[int(np.argmax(ln))]]
        best = max(band, key=lambda c: ln[cfgs.index(c)])
        seg = fwd[best].dropna()
        if len(seg):
            pieces.append(seg)
        t = nxt
    return pd.concat(pieces).sort_index() if pieces else None


def _year_net(daily, yk):
    lo, hi = YEARS[yk]
    return round(_net(daily[(daily.index >= pd.Timestamp(lo)) & (daily.index < pd.Timestamp(hi))]), 1)


def main(argv=None) -> int:
    tf = "4h"
    print(f"## TIER-A all-weather SENSITIVITY @ {tf}: lookback {LOOKBACKS} x step {STEPS} per candidate\n")
    out = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for ax_i, ti in enumerate(TIER_A):
        sdf, _bh = _ti_series(ti, tf)
        if sdf is None:
            print(f"   {ti}: no series"); continue
        bear = np.full((len(LOOKBACKS), len(STEPS)), np.nan)
        cell = {}
        for i, lb in enumerate(LOOKBACKS):
            for j, st in enumerate(STEPS):
                rp = _rolling_pick(sdf, lb, st)
                if rp is None:
                    continue
                n20 = _year_net(rp, "2020_bull"); n21 = _year_net(rp, "2021_mixed"); n22 = _year_net(rp, "2022_bear")
                bear[i, j] = n22
                cell[f"{lb}/{st}"] = {"2020": n20, "2021": n21, "2022": n22}
        bvals = bear[np.isfinite(bear)]
        bulls_ok = all(c["2020"] > 0 and c["2021"] > 0 for c in cell.values())
        robust = bool(bvals.size and bvals.min() >= -10 and bulls_ok)        # bear stays mild across the WHOLE grid
        out[ti] = {"bear_min": round(float(bvals.min()), 1) if bvals.size else None,
                   "bear_med": round(float(np.median(bvals)), 1) if bvals.size else None,
                   "bear_max": round(float(bvals.max()), 1) if bvals.size else None,
                   "bulls_all_positive": bulls_ok, "hyperparam_robust": robust, "grid": cell}
        print(f"   {ti:8} | 2022-bear net across grid: min {out[ti]['bear_min']} / med {out[ti]['bear_med']} / "
              f"max {out[ti]['bear_max']} | bulls all+ {bulls_ok} -> {'ROBUST' if robust else 'fragile'}")
        ax = axes[ax_i // 3][ax_i % 3]
        im = ax.imshow(bear, cmap="RdYlGn", vmin=-30, vmax=10, aspect="auto")
        ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([f"s{s}" for s in STEPS], fontsize=8)
        ax.set_yticks(range(len(LOOKBACKS))); ax.set_yticklabels([f"L{l}" for l in LOOKBACKS], fontsize=8)
        for i in range(len(LOOKBACKS)):
            for j in range(len(STEPS)):
                if np.isfinite(bear[i, j]):
                    ax.text(j, i, f"{bear[i,j]:.0f}", ha="center", va="center", fontsize=8,
                            color="black")
        ax.set_title(f"{ti} 2022-bear net ({'ROBUST' if robust else 'fragile'})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"TIER-A candidate ROBUSTNESS @ {tf}: 2022-BEAR net % across rolling lookback (L, rows) x step "
                 f"(s, cols). GREEN/near-0 everywhere = the all-weather result is NOT a hyperparameter artifact.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "charts" / "tier_a_sensitivity_4h.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    robust_set = [ti for ti, v in out.items() if v["hyperparam_robust"]]
    print(f"\n   HYPERPARAM-ROBUST tier-A candidates (bear>=-10 + bulls+ across the FULL grid): {robust_set}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = OUT / f"tier_a_sensitivity_{stamp}.json"
    json.dump({"repro": {"git_sha": sha, "lookbacks": LOOKBACKS, "steps": STEPS, "tf": tf},
               "robust_set": robust_set, "results": out}, open(jp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"   [chart] {p}\n   [json] {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
