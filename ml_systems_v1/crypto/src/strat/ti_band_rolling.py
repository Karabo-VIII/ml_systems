"""src/strat/ti_band_rolling.py -- generalize the WORKING-BAND + ROLLING-SELECTION + ALL-WEATHER test to ALL
TI families (not just MA). The same builder-mode treatment, per TI type, end-to-end 2020/2021/2022.

USER /orc 2026-06-16 (3h): "after MA, expand to ALL other TI families -- actually RUN them, test the hypothesis,
NO shortcuts. End-to-end 2020-21 report where candidates per TI family AND per TI type start emerging."

For each TI (INDICATORS registry: trend MACD/VORTEX/ADX/SUPERTREND/PSAR, momentum TSI/ROC, breakout DONCHIAN/
KELTNER, mean-reversion RSI/STOCH/BBPCT/CCI/WILLR, volume OBV/MFI/VOLIMB/CMF): build every config's DAILY net
over 2020-2022 (ironed sleeve, fixed-EW u10), then the SAME walk-forward ROLLING-FROM-BAND selection as the MA
harness -- rolling-pick (recent-best trailing-positive config) vs band-ENSEMBLE (EW the rolling band) vs
STATIC-#1 (in-sample-peeked, the naive baseline) vs BUY-HOLD -- and the ALL-WEATHER per-year table
(2020 bull / 2021 mixed / 2022 BEAR). The deployable CANDIDATE per TI = the rolling-from-band book IF it
participates the bulls AND preserves the bear (the all-weather bar), ranked per family.

Reuses deep2020_ti_pipeline.{INDICATORS, load_ohlc, load_ohlcv, _book} (the EXACT ironed sleeve) + the rolling
machinery pattern from working_band_rolling. NO look-ahead (rolling band+pick trailing-only; static-#1 peeked).
Long-only spot, fixed-EW, maker, UNSEEN sealed. No emoji.

RWYB: python -m strat.ti_band_rolling --tfs 4h --families trend  (then all)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.deep2020_ti_pipeline as TI                                      # noqa: E402
from strat.deep2020_ti_pipeline import INDICATORS                           # noqa: E402
from strat.portfolio_replay import MAKER_RT, TAKER_RT                        # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True); CHARTS.mkdir(parents=True, exist_ok=True)
SPAN = ("2020-01-01", "2023-01-01")
YEARS = {"2020_bull": ("2020-01-01", "2021-01-01"), "2021_mixed": ("2021-01-01", "2022-01-01"),
         "2022_bear": ("2022-01-01", "2023-01-01")}
LOOKBACK_D = 120
STEP_D = 30


def _net(s):
    s = s.dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0


def _maxdd(s):
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy()); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _per_year(daily):
    out = {}
    for yk, (lo, hi) in YEARS.items():
        s = daily[(daily.index >= pd.Timestamp(lo)) & (daily.index < pd.Timestamp(hi))]
        out[yk] = {"net": round(_net(s), 1), "maxdd": round(_maxdd(s), 1)}
    return out


def _ti_series(ti_key, tf):
    """DataFrame [date x cfg-name] of DAILY net for every config of `ti_key` (ironed sleeve), over SPAN.
    Plus the buy-hold daily series (same universe)."""
    TI.WIN = SPAN; TI.SPLIT = "2022-10-01"                                   # span the 3 years; SPLIT only sets vt/tin
    ind = INDICATORS[ti_key]
    loader = TI.load_ohlcv if ind.get("loader") == "ohlcv" else TI.load_ohlc
    assets, vt = loader(tf)
    if not assets or len(assets) < 5:
        return None, None
    mh = ind.get("minhold", 12)
    cols = {}
    for p in ind["grid"]():
        r = TI._book(assets, ind["iron"], p, vt, mh)
        if r is None:
            continue
        daily = r[0]
        if daily is not None and len(daily) > 50:
            cols[ind["name"](p)] = daily
    if not cols:
        return None, None
    # buy-hold daily over the same assets/window
    import pandas as _pd
    bh_cells = []
    for A in assets:
        ret, win, idx = A["ret"], A["win"], A["idx"]
        bh_cells.append(_pd.Series(ret[win], index=idx))
    bh = _pd.concat(bh_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
    bh_daily = bh.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return pd.DataFrame(cols).sort_index(), bh_daily


def _rolling(series_df, mode="pick"):
    """Walk-forward rolling-from-band (NO look-ahead): band = trailing-LOOKBACK-positive configs; pick = the
    recent-best; ensemble = EW the band. Trade strictly forward over the next STEP days."""
    idx = series_df.index
    cfgs = list(series_df.columns)
    start = idx.min() + pd.Timedelta(days=LOOKBACK_D)
    pieces, picks = [], []
    t = start
    while t < idx.max():
        nxt = t + pd.Timedelta(days=STEP_D)
        look = series_df[(idx >= t - pd.Timedelta(days=LOOKBACK_D)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(cfgs, look_net) if v > 0]
        if not band:
            band = [cfgs[int(np.argmax(look_net))]]
        if mode == "pick":
            best = max(band, key=lambda c: look_net[cfgs.index(c)])
            seg = fwd[best].dropna(); picks.append(best)
        else:
            seg = fwd[band].mean(axis=1).dropna(); picks.append(f"EW{len(band)}")
        if len(seg):
            pieces.append(seg)
        t = nxt
    if not pieces:
        return None, picks
    return pd.concat(pieces).sort_index(), picks


def _static_1(series_df):
    full = (np.prod(1 + series_df.fillna(0.0).to_numpy(), axis=0) - 1)
    best = series_df.columns[int(np.argmax(full))]
    return series_df[best].dropna(), best


def run_ti(ti_key, tf):
    sdf, bh = _ti_series(ti_key, tf)
    if sdf is None:
        return None
    rp, picks = _rolling(sdf, "pick")
    re_, _ = _rolling(sdf, "ensemble")
    s1, s1cfg = _static_1(sdf)
    return {"ti": ti_key, "family": INDICATORS[ti_key]["family"], "n_configs": sdf.shape[1],
            "rolling_pick": _per_year(rp) if rp is not None else {},
            "band_ensemble": _per_year(re_) if re_ is not None else {},
            "static_1": _per_year(s1), "static_1_cfg": s1cfg,
            "buyhold": _per_year(bh), "n_distinct_picks": len(set(picks)),
            "_rp_daily": rp, "_re_daily": re_, "_bh_daily": bh}


# =====================================================================================================
# CANDIDATE EMERGENCE: per TI, is the rolling-from-band book an all-weather CANDIDATE?
# =====================================================================================================
def _is_candidate(rec):
    """All-weather bar: participate BOTH bulls (2020 & 2021 net > 0) AND preserve the bear (2022 net > buy-hold
    2022 net by a clear margin, i.e. it loses materially less). Returns (is_cand, score, detail)."""
    rp = rec["rolling_pick"]; bh = rec["buyhold"]
    n20 = rp.get("2020_bull", {}).get("net"); n21 = rp.get("2021_mixed", {}).get("net")
    n22 = rp.get("2022_bear", {}).get("net"); dd22 = rp.get("2022_bear", {}).get("maxdd")
    bh22 = bh.get("2022_bear", {}).get("net")
    if None in (n20, n21, n22, bh22):
        return False, None, "incomplete"
    participates = n20 > 0 and n21 > 0
    preserves = n22 > bh22 + 10                                              # loses >=10pp LESS than buy-hold in the bear
    # all-weather score: bull participation (capped) + bear preservation (the scarce thing) - drawdown
    score = round(min(n20, 200) * 0.1 + min(n21, 400) * 0.1 + (n22 - bh22) * 1.0 + (dd22 or 0) * 0.2, 1)
    return bool(participates and preserves), score, f"20:{n20} 21:{n21} 22:{n22} (BH22 {bh22})"


def build_report(allres, tf):
    rows = []
    for ti, rec in allres.items():
        if rec is None:
            continue
        cand, score, detail = _is_candidate(rec)
        rows.append({"ti": ti, "family": rec["family"], "candidate": cand, "score": score, "detail": detail,
                     "rp": rec["rolling_pick"], "bh": rec["buyhold"], "n_picks": rec["n_distinct_picks"]})
    rows.sort(key=lambda r: (r["score"] is not None, r["score"] or -1e9), reverse=True)
    return rows


def chart_allweather(allres, tf):
    """Per-TI all-weather net bars (2020/2021/2022 rolling-pick) vs buy-hold 2022 line -- candidates pop out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [(ti, r) for ti, r in allres.items() if r is not None]
    if not recs:
        return None
    recs.sort(key=lambda kv: kv[1]["rolling_pick"].get("2022_bear", {}).get("net", -1e9), reverse=True)
    tis = [ti for ti, _ in recs]
    n20 = [r["rolling_pick"].get("2020_bull", {}).get("net", 0) for _, r in recs]
    n21 = [r["rolling_pick"].get("2021_mixed", {}).get("net", 0) for _, r in recs]
    n22 = [r["rolling_pick"].get("2022_bear", {}).get("net", 0) for _, r in recs]
    bh22 = np.median([r["buyhold"].get("2022_bear", {}).get("net", 0) for _, r in recs])
    fig, ax = plt.subplots(figsize=(max(10, len(tis) * 0.8), 6))
    x = np.arange(len(tis)); w = 0.26
    ax.bar(x - w, n20, w, label="2020 bull", color="#2ca02c", alpha=0.85)
    ax.bar(x, n21, w, label="2021 mixed", color="#1f77b4", alpha=0.85)
    ax.bar(x + w, n22, w, label="2022 BEAR", color="#d62728", alpha=0.85)
    ax.axhline(bh22, color="black", ls="--", lw=1.2, label=f"buy-hold 2022 ({bh22:.0f}%)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tis, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("rolling-from-band net %"); ax.legend(fontsize=8)
    ax.set_title(f"ALL-WEATHER rolling-from-band per TI @ {tf} (sorted by 2022-bear net). A CANDIDATE participates "
                 f"the bulls AND sits ABOVE the buy-hold-2022 dashed line (preserves the bear).", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = CHARTS / f"ti_allweather_{tf}.png"
    fig.savefig(p, dpi=110); plt.close(fig); print(f"   [chart] {p}"); return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ti_band_rolling")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--families", default="all")
    a = ap.parse_args(argv)
    fams = None if a.families == "all" else set(a.families.split(","))
    tis = [k for k, v in INDICATORS.items() if fams is None or v["family"] in fams]
    allout = {}
    for tf in [t.strip() for t in a.tfs.split(",") if t.strip()]:
        print(f"\n================= {tf} =================")
        allres = {}
        for ti in tis:
            rec = run_ti(ti, tf)
            allres[ti] = rec
            if rec:
                rp = rec["rolling_pick"]; bh = rec["buyhold"]
                print(f"   {rec['family']:14} {ti:10} | rolling-pick "
                      f"[{rp.get('2020_bull',{}).get('net')},{rp.get('2021_mixed',{}).get('net')},"
                      f"{rp.get('2022_bear',{}).get('net')}] vs BH "
                      f"[{bh.get('2020_bull',{}).get('net')},{bh.get('2021_mixed',{}).get('net')},"
                      f"{bh.get('2022_bear',{}).get('net')}]")
        rows = build_report(allres, tf)
        cands = [r for r in rows if r["candidate"]]
        print(f"\n   === CANDIDATES @ {tf} (participate bulls + preserve 2022 bear vs buy-hold): {len(cands)} ===")
        for r in cands:
            print(f"   [CAND] {r['family']:14} {r['ti']:10} score {r['score']} | {r['detail']}")
        chart_allweather(allres, tf)
        slim = {ti: {k: v for k, v in r.items() if not k.startswith("_")} for ti, r in allres.items() if r}
        allout[tf] = {"results": slim, "report_rows": [{k: v for k, v in r.items()} for r in rows],
                      "candidates": [r["ti"] for r in cands]}
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"ti_band_rolling_{stamp}.json"
    json.dump({"repro": {"git_sha": sha, "span": SPAN, "lookback_d": LOOKBACK_D, "step_d": STEP_D,
                         "cost_maker": MAKER_RT, "cost_taker": TAKER_RT}, "by_tf": allout},
              open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
