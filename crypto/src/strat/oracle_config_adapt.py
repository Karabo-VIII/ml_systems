"""src/strat/oracle_config_adapt.py -- THE MISSING LINK: dynamically adapt the TI-oracle config week-by-week.

THE PROBLEM (user, 2026-06-10): the per-week oracle config does NOT persist -- last week's best config loses money
next week (naive walk-forward FAILS; confirmed by the other instance + the config-selection~random prior). The ask:
"how, at any point, do we adjust the config week by week to capture next week's moves?" Hypothesis (user): the answer
is MATHEMATICAL -- if we decompose what DETERMINES an oracle config, we can adapt it.

THE THESIS (decompose-the-ideal): the oracle config is a FUNCTION of the window's observable STATE. The naive test
fails because it carries the CONFIG forward (which doesn't persist); the right thing is to carry the STATE forward
(trendiness / volatility), which DOES persist (vol clusters; Hurst|ret|~0.8), and map STATE -> config. The decompose
already hinted it: trending windows -> price>MA / slower-MA / loose exit; choppy windows -> mechanical exit (cross+time)
/ faster MA. The mathematical "trendiness" knob is the KAUFMAN EFFICIENCY RATIO (|net move| / sum|bar moves| = how
straight the path is = a trendline-fit-quality proxy).

THE TEST (rigorous, held-out):
  Roll weekly windows. Per week compute STATE (efficiency-ratio, realized vol, swing count, autocorr) + each config's
  captured ROI of the 2-10% moves + the best config. Then on a TRAIN span learn the mapping STATE->best-config (bucket
  by efficiency-ratio tercile -> the bucket's historically-best config). On a HELD-OUT span, apply at week w: read
  state_w (PAST-ONLY) -> pick the config its bucket says -> APPLY to week w+1 -> measure realized capture. Compare to:
    NAIVE   (last week's argmax config),  FIXED (one global-best config),  RANDOM (mean over configs),
    ADAPTIVE (state-conditioned),         ORACLE (next week's actual best = the ceiling).
  The question: does ADAPTIVE beat NAIVE / FIXED / RANDOM on held-out next-week capture? + does STATE persist w->w+1?

LONG-ONLY, GROSS capture (the config ceiling); cost is a separate later layer. No emoji (cp1252). Run:
  python src/strat/oracle_config_adapt.py --asset BTC --cadence 1d --window-days 7
"""
from __future__ import annotations
import argparse, sys, warnings, json
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from strat.entry_signal_lab import load_ohlc, _zigzag_pivots   # FIXED loader + zigzag


# --- the curated CONFIG set: spans the trend<->chop axis (interpretable, so the mapping is readable) ---------
def _sma(c, n): return pd.Series(c).rolling(n, min_periods=max(2, n // 2)).mean().to_numpy()
def _ema(c, n): return pd.Series(c).ewm(span=n, adjust=False).mean().to_numpy()
def _hma(c, n):
    import pandas_ta as ta
    out = ta.hma(pd.Series(c), length=n)
    return out.to_numpy(float) if out is not None else np.full(len(c), np.nan)

def _pos_price_above(close, ma_fn, n): return (close > ma_fn(close, n)).astype(float)
def _pos_cross(close, ma_fn, f, s): return (ma_fn(close, f) > ma_fn(close, s)).astype(float)
def _pos_cross_time(close, ma_fn, f, s, hold):
    cross = ma_fn(close, f) > ma_fn(close, s); n = len(close); pos = np.zeros(n); i = 1
    while i < n:
        if cross[i] and not cross[i - 1]:
            j = i
            while j < n and (j - i) < hold: pos[j] = 1.0; j += 1
            i = j
        else: i += 1
    return pos

# CONFIGS keyed by name; each -> a function(close)->position. Spanning fast/slow x price>MA/cross/cross+time.
CONFIGS = {
    "fastMA_hma5":   lambda c: _pos_price_above(c, _hma, 5),     # fast low-lag, trend-ride
    "fastMA_sma8":   lambda c: _pos_price_above(c, _sma, 8),
    "slowMA_sma20":  lambda c: _pos_price_above(c, _sma, 20),    # slower, smoother trend
    "slowMA_sma50":  lambda c: _pos_price_above(c, _sma, 50),
    "cross_ema10_30": lambda c: _pos_cross(c, _ema, 10, 30),
    "crossT_sma5_20": lambda c: _pos_cross_time(c, _sma, 5, 20, 10),   # mechanical time-exit (chop)
    "crossT_sma8_21": lambda c: _pos_cross_time(c, _sma, 8, 21, 8),
    "crossT_hma5_20": lambda c: _pos_cross_time(c, _hma, 5, 20, 12),
}
CONFIG_NAMES = list(CONFIGS.keys())


# --- window STATE (the mathematical DNA) + price-oracle ------------------------------------------------------
def efficiency_ratio(close):
    """Kaufman ER over the window: |net move| / sum|bar moves|. 1=pure trend(straight line), ~0=pure chop. THE knob."""
    if len(close) < 3: return np.nan
    net = abs(close[-1] - close[0]); path = np.sum(np.abs(np.diff(close)))
    return net / path if path > 1e-12 else np.nan

def window_state(close):
    ret = np.diff(close) / close[:-1]
    er = efficiency_ratio(close)
    vol = float(np.std(ret)) if len(ret) > 1 else np.nan
    piv = _zigzag_pivots(close, thr=0.02); nsw = sum(1 for k in range(len(piv) - 1) if piv[k + 1][1] > piv[k][1])
    ac = float(pd.Series(ret).autocorr(lag=1)) if len(ret) > 3 else np.nan
    return {"er": er, "vol": vol, "n_swings": nsw, "autocorr": ac}

def price_oracle(close, lo=0.02, hi=0.10):
    piv = _zigzag_pivots(close, thr=lo); comp = 1.0
    for k in range(len(piv) - 1):
        v0, v1 = piv[k][1], piv[k + 1][1]
        if v1 > v0:
            r = v1 / v0 - 1.0
            comp *= (1 + min(r, hi)) if r >= lo else 1.0
    return float((comp - 1.0) * 100)

def config_roi(close_full, pos_full, a, b):
    """GROSS captured ROI of a config (its position) within window [a,b]."""
    ret = np.zeros(len(close_full)); ret[1:] = close_full[1:] / close_full[:-1] - 1.0
    pl = np.roll(pos_full, 1); pl[0] = 0.0
    return float(np.prod(1.0 + pl[a:b + 1] * ret[a:b + 1]) - 1.0) * 100


# --- per-week records ----------------------------------------------------------------------------------------
def weekly_records(asset, cadence, window_days=7, lo=0.02, hi=0.10, max_weeks=600):
    df = load_ohlc(asset if asset.endswith("USDT") else asset + "USDT", cadence)
    if df is None or len(df) < 60: return None
    dates = pd.to_datetime(df["date"]).to_numpy(); close = df["close"].to_numpy(float)
    pos_full = {nm: fn(close) for nm, fn in CONFIGS.items()}                     # configs on the FULL series (warmup)
    recs = []; win = np.timedelta64(window_days, "D"); wstart = dates[0]; guard = 0
    while wstart < dates[-1] and guard < max_weeks:
        wend = wstart + win; idx = np.where((dates >= wstart) & (dates < wend))[0]; wstart = wend; guard += 1
        if idx.size < 5: continue
        a, b = idx[0], idx[-1]
        porc = price_oracle(close[a:b + 1], lo, hi)
        if porc <= 0.5: continue                                                # only weeks with a real move
        rois = {nm: config_roi(close, pos_full[nm], a, b) for nm in CONFIG_NAMES}
        best = max(rois, key=rois.get)
        recs.append({"start": str(pd.Timestamp(dates[a]).date()), "a": int(a), "b": int(b),
                     "price_oracle": porc, "state": window_state(close[a:b + 1]),
                     "rois": rois, "best": best, "best_roi": rois[best]})
    return recs


# --- the adaptive test ---------------------------------------------------------------------------------------
def adapt_test(recs, train_frac=0.6):
    """Learn STATE->config on TRAIN weeks (efficiency-ratio terciles -> bucket-best config), test on held-out."""
    n = len(recs)
    if n < 20: return None
    ntr = int(n * train_frac); train, test = recs[:ntr], recs[ntr:]
    # STATE PERSISTENCE (does er/vol carry week->week?)
    er = np.array([r["state"]["er"] for r in recs], float); vol = np.array([r["state"]["vol"] for r in recs], float)
    def _ac(x):
        x = x[np.isfinite(x)]
        return float(pd.Series(x).autocorr(lag=1)) if len(x) > 3 else np.nan
    persist = {"er_autocorr": _ac(er), "vol_autocorr": _ac(vol)}
    fixed_best = max({nm: float(np.mean([r["rois"][nm] for r in train])) for nm in CONFIG_NAMES}.items(),
                     key=lambda kv: kv[1])[0]
    # FIXED + NAIVE + RANDOM + ORACLE baselines on TEST
    base = {m: [] for m in ("naive", "fixed", "random", "oracle")}
    for i in range(len(test) - 1):
        w, nxt = test[i], test[i + 1]
        base["naive"].append(nxt["rois"][w["best"]])
        base["fixed"].append(nxt["rois"][fixed_best])
        base["random"].append(float(np.mean([nxt["rois"][nm] for nm in CONFIG_NAMES])))
        base["oracle"].append(nxt["best_roi"])
    # ADAPTIVE -- try EACH state var as the bucketing signal (the secret-sauce search): which PREDICTABLE state,
    # if any, lets state_w pick a config that beats fixed on week w+1?
    STATE_VARS = ["er", "vol", "n_swings", "autocorr"]
    adaptive = {}
    for sv in STATE_VARS:
        tr_v = np.array([r["state"][sv] for r in train], float)
        if not np.isfinite(tr_v).sum() > 10: continue
        q1, q2 = np.nanpercentile(tr_v[np.isfinite(tr_v)], [33, 66])
        def bk(x): return "hi" if (np.isfinite(x) and x >= q2) else ("lo" if (np.isfinite(x) and x < q1) else "mid")
        bbest = {}
        for b in ("lo", "mid", "hi"):
            rows = [r for r in train if bk(r["state"][sv]) == b]
            if not rows: continue
            mr = {nm: float(np.mean([r["rois"][nm] for r in rows])) for nm in CONFIG_NAMES}
            bbest[b] = max(mr, key=mr.get)
        picks = []
        for i in range(len(test) - 1):
            w, nxt = test[i], test[i + 1]
            cfg = bbest.get(bk(w["state"][sv]), fixed_best)
            picks.append(nxt["rois"][cfg])
        picks = np.array(picks); fx = np.array(base["fixed"])
        adaptive[sv] = {"mean_roi": round(float(np.mean(picks)), 3),
                        "beats_fixed_frac": round(float(np.mean(picks > fx)), 2),
                        "mean_edge_vs_fixed": round(float(np.mean(picks - fx)), 3),
                        "map": bbest}
    # WEEK-SELECTION diagnostic (the reframe): does THIS-week state predict NEXT-week CAPTURABILITY? If yes, the
    # adaptive lever is WHEN-to-trade / sizing (not which config). Corr(state_w, best_roi_{w+1}) + a hi/lo-state split.
    sv_all = {sv: np.array([r["state"][sv] for r in recs], float) for sv in STATE_VARS}
    next_best = np.array([recs[i + 1]["best_roi"] for i in range(len(recs) - 1)], float)
    next_porc = np.array([recs[i + 1]["price_oracle"] for i in range(len(recs) - 1)], float)
    fixed_pos = np.zeros(len(CONFIG_NAMES))  # placeholder
    next_fixedroi = np.array([recs[i + 1]["rois"][fixed_best] for i in range(len(recs) - 1)], float)
    week_sel = {}
    for sv in STATE_VARS:
        x = sv_all[sv][:-1]; m = np.isfinite(x) & np.isfinite(next_fixedroi)
        if m.sum() < 20: continue
        c_best = float(np.corrcoef(x[m], next_best[m])[0, 1])
        c_fix = float(np.corrcoef(x[m], next_fixedroi[m])[0, 1])
        # top-tercile-state weeks: is next-week fixed-config ROI higher than bottom-tercile?
        hi = next_fixedroi[m][x[m] >= np.nanpercentile(x[m], 66)]
        lo = next_fixedroi[m][x[m] < np.nanpercentile(x[m], 33)]
        week_sel[sv] = {"corr_state_vs_next_best": round(c_best, 3), "corr_state_vs_next_fixedroi": round(c_fix, 3),
                        "next_fixedroi_hiState": round(float(np.mean(hi)), 3) if len(hi) else None,
                        "next_fixedroi_loState": round(float(np.mean(lo)), 3) if len(lo) else None}
    out = {"n_weeks": n, "n_test_steps": len(test) - 1, "persistence": persist, "fixed_best": fixed_best,
           "baseline_mean_roi": {m: round(float(np.mean(v)), 3) for m, v in base.items() if v},
           "adaptive_by_state": adaptive, "week_selection": week_sel}
    return out


def participation_backtest(recs, config_name="fastMA_sma8", min_hist=20, cost_rt=0.0024, trades_per_wk=2.0):
    """ROLLING vol-timed participation (fully causal). Each week w: read vol_w, set NEXT-week weight from the
    EXPANDING-median vol threshold (past-only). Apply a FIXED config (config is interchangeable). Compound weekly.
    Schemes: UNIFORM(always in) / VOL_GATE(in iff vol_w>=past-median) / VOL_RANK(weight=past-percentile of vol_w).
    Reports GROSS + NET (cost_rt*trades_per_wk per participating week). The cash-out of the framework's Stage 3."""
    vols = np.array([r["state"]["vol"] for r in recs], float)
    schemes = {"uniform": [], "vol_gate": [], "vol_rank": []}
    inmkt = {"uniform": [], "vol_gate": [], "vol_rank": []}
    for w in range(min_hist, len(recs) - 1):
        past = vols[:w + 1]; past = past[np.isfinite(past)]
        if len(past) < 5 or not np.isfinite(vols[w]): continue
        thr = float(np.median(past)); rank = float(np.mean(past < vols[w]))     # past-only percentile in [0,1]
        nxt = recs[w + 1]["rois"].get(config_name, 0.0) / 100.0
        for sc, wt in (("uniform", 1.0), ("vol_gate", 1.0 if vols[w] >= thr else 0.0), ("vol_rank", rank)):
            net = wt * nxt - wt * cost_rt * trades_per_wk
            schemes[sc].append(net); inmkt[sc].append(wt)
    out = {}
    for sc, rets in schemes.items():
        if not rets: continue
        r = np.array(rets); eq = np.cumprod(1 + r); comp = (eq[-1] - 1) * 100
        nyr = len(r) / 52.0; ann = (eq[-1] ** (1 / nyr) - 1) * 100 if eq[-1] > 0 else -100.0
        dd = float(np.min((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq))) * 100
        wk_in = float(np.mean(inmkt[sc]))
        # gross (no cost) for reference
        rg = np.array([rr + (im * cost_rt * trades_per_wk) for rr, im in zip(rets, inmkt[sc])])
        compg = (np.cumprod(1 + rg)[-1] - 1) * 100
        out[sc] = {"compound_net_pct": round(comp, 1), "ann_net_pct": round(ann, 1), "compound_gross_pct": round(compg, 1),
                   "max_dd_pct": round(dd, 1), "avg_weight": round(wk_in, 2), "n_weeks": len(r),
                   "ret_per_week_in_mkt_net": round(float(np.sum(r) / max(np.sum(inmkt[sc]), 1e-9)) * 100, 3)}
    return out


def portfolio_rotation(assets, cadence, window_days=7, config_name="fastMA_sma8", topk=3, cost_rt=0.0024, trades_per_wk=2.0):
    """The 'config is not the answer -> participation is' dimension, at PORTFOLIO level. Each week rank the pool by
    vol-state (high vol = high next-week capturability), deploy the fixed config across the TOP-K names (equal weight),
    roll weekly. Always deployed (no cash-drag) but always in the most-capturable names. Compare vs equal-weight-ALL
    (uniform) + cash. Net of cost. The cash-out of Stage-3 the per-asset participation pointed to."""
    recs = {}
    for sym in assets:
        r = weekly_records(sym, cadence, window_days)
        if r:
            keyed = {}
            for x in r:
                ic = pd.Timestamp(x["start"]).isocalendar(); keyed[f"{ic[0]}-W{ic[1]:02d}"] = x
            recs[sym] = keyed
    if len(recs) < 2: return None
    weeks = sorted(set().union(*[set(d.keys()) for d in recs.values()]))
    rng = np.random.default_rng(13)
    rot, uni, rnd = [], [], []
    for i in range(len(weeks) - 1):
        wk, nxt = weeks[i], weeks[i + 1]
        avail = [s for s in recs if wk in recs[s] and nxt in recs[s]]
        if len(avail) < topk: continue
        vols = {s: recs[s][wk]["state"]["vol"] for s in avail if np.isfinite(recs[s][wk]["state"]["vol"])}
        if len(vols) < topk: continue
        top = sorted(vols, key=vols.get, reverse=True)[:topk]
        rpick = list(rng.choice(list(vols.keys()), size=topk, replace=False))   # RANDOM-K null (same diversification)
        nxt_roi = {s: recs[s][nxt]["rois"].get(config_name, 0.0) / 100.0 for s in avail}
        c = cost_rt * trades_per_wk
        rot.append(float(np.mean([nxt_roi[s] for s in top])) - c)
        uni.append(float(np.mean([nxt_roi[s] for s in avail])) - c)
        rnd.append(float(np.mean([nxt_roi[s] for s in rpick])) - c)
    if len(rot) < 20: return None
    def stats(r):
        r = np.array(r); eq = np.cumprod(1 + r); comp = (eq[-1] - 1) * 100
        nyr = len(r) / 52.0; ann = (eq[-1] ** (1 / nyr) - 1) * 100 if eq[-1] > 0 else -100.0
        dd = float(np.min((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq))) * 100
        return {"ann_pct": round(ann, 1), "comp_pct": round(comp, 1), "maxdd_pct": round(dd, 1),
                "mean_wk_pct": round(float(np.mean(r)) * 100, 3), "sharpe_wk": round(float(np.mean(r) / (np.std(r) + 1e-9)), 3)}
    h = len(rot) // 2
    return {"n_weeks": len(rot), "topk": topk, "pool": list(recs.keys()),
            "vol_rotation": stats(rot), "equal_weight_all": stats(uni), "random_rotation": stats(rnd),
            "vol_beats_random_frac": round(float(np.mean(np.array(rot) > np.array(rnd))), 2),
            "vol_edge_vs_random_pp": round(float(np.mean(np.array(rot) - np.array(rnd))) * 100, 3),
            "OOS_2nd_half": {"vol_rotation": stats(rot[h:]), "equal_weight_all": stats(uni[h:]), "random_rotation": stats(rnd[h:])}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.oracle_config_adapt")
    ap.add_argument("--asset", default="BTC"); ap.add_argument("--cadence", default="1d")
    ap.add_argument("--assets", default=None, help="comma list (overrides --asset)")
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--participation", action="store_true", help="rolling vol-timed participation backtest")
    ap.add_argument("--rotation", action="store_true", help="portfolio vol-rotation backtest")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--config-name", default="fastMA_sma8")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    assets = a.assets.split(",") if a.assets else [a.asset]
    allout = {}
    if a.rotation:
        for cad in a.cadence.split(","):
            pr = portfolio_rotation(assets, cad, a.window_days, a.config_name, a.topk)
            if not pr: print(f"   {cad}: insufficient"); continue
            r, u = pr["vol_rotation"], pr["equal_weight_all"]
            print(f"## VOL-ROTATION {cad} (top{pr['topk']} of {len(pr['pool'])}, fixed {a.config_name}, {pr['n_weeks']} wks)")
            print(f"   vol_rotation : ann={r['ann_pct']}% comp={r['comp_pct']}% maxDD={r['maxdd_pct']}% wk={r['mean_wk_pct']}% sharpe_wk={r['sharpe_wk']}")
            print(f"   equal_wt_all : ann={u['ann_pct']}% comp={u['comp_pct']}% maxDD={u['maxdd_pct']}% wk={u['mean_wk_pct']}% sharpe_wk={u['sharpe_wk']}")
            rd = pr["random_rotation"]; o = pr["OOS_2nd_half"]
            print(f"   random_rot   : ann={rd['ann_pct']}% comp={rd['comp_pct']}% wk={rd['mean_wk_pct']}%  | vol beats random {pr['vol_beats_random_frac']:.0%} of wks, edge {pr['vol_edge_vs_random_pp']:+.3f}pp/wk")
            print(f"   OOS 2nd-half : vol_rot ann={o['vol_rotation']['ann_pct']}%  equal_wt ann={o['equal_weight_all']['ann_pct']}%  random ann={o['random_rotation']['ann_pct']}%")
        return 0
    if a.participation:
        print(f"## ROLLING VOL-TIMED PARTICIPATION (fixed config={a.config_name}, causal expanding-median vol gate)")
        print(f"   {'asset':5} {'scheme':9} {'ann_net%':>8} {'comp_net%':>9} {'comp_gross%':>11} {'maxDD%':>7} {'avgWt':>5} {'ret/wk-in%':>10}")
        for sym in assets:
            recs = weekly_records(sym, a.cadence, a.window_days)
            if not recs: print(f"   {sym}: no data"); continue
            pb = participation_backtest(recs, a.config_name)
            allout[sym] = pb
            for sc in ("uniform", "vol_gate", "vol_rank"):
                d = pb.get(sc, {})
                if d: print(f"   {sym:5} {sc:9} {d['ann_net_pct']:>8} {d['compound_net_pct']:>9} {d['compound_gross_pct']:>11} {d['max_dd_pct']:>7} {d['avg_weight']:>5} {d['ret_per_week_in_mkt_net']:>10}")
        if a.json and allout:
            import subprocess
            sha = subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
            outdir = ROOT.parent/"runs"/"mining"; outdir.mkdir(parents=True,exist_ok=True)
            path = outdir/f"participation_{'-'.join(assets)}_{a.cadence}_w{a.window_days}.json"
            json.dump({"repro":{"command":"python "+" ".join(sys.argv),"git_sha":sha},"results":allout}, open(path,"w",encoding="utf-8"), indent=2, default=str)
            print(f"[persisted] {path}")
        return 0
    for sym in assets:
        recs = weekly_records(sym, a.cadence, a.window_days)
        if not recs: print(f"{sym}: no data"); continue
        out = adapt_test(recs)
        if not out: print(f"{sym}: too few weeks ({len(recs)})"); continue
        allout[sym] = out
        b = out["baseline_mean_roi"]
        print(f"## {sym} {a.cadence} {a.window_days}d -- {out['n_weeks']} weeks, {out['n_test_steps']} held-out steps")
        print(f"   STATE persistence (autocorr w->w+1): er={out['persistence']['er_autocorr']:.2f}  vol={out['persistence']['vol_autocorr']:.2f}")
        print(f"   baselines next-week mean ROI:  NAIVE={b.get('naive')}  FIXED={b.get('fixed')}  RANDOM={b.get('random')}  ORACLE(ceiling)={b.get('oracle')}")
        print(f"   ADAPTIVE by state-var (mean_roi | beats_fixed%% | edge_vs_fixed):")
        for sv, d in out["adaptive_by_state"].items():
            flag = "  <-- BEATS fixed" if d["beats_fixed_frac"] > 0.5 and d["mean_edge_vs_fixed"] > 0 else ""
            print(f"      {sv:9} {d['mean_roi']:>6} | {d['beats_fixed_frac']:>4.0%} | {d['mean_edge_vs_fixed']:>+6}{flag}")
        print(f"   WEEK-SELECTION (does state_w predict next-week CAPTURABILITY? corr + hi/lo-state next-roi):")
        for sv, d in out["week_selection"].items():
            flag = "  <-- PREDICTS" if abs(d["corr_state_vs_next_fixedroi"]) > 0.15 else ""
            print(f"      {sv:9} corr(->next_fixedROI)={d['corr_state_vs_next_fixedroi']:>+6}  hiState={d['next_fixedroi_hiState']} loState={d['next_fixedroi_loState']}{flag}")
    if a.json and allout:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        outdir = ROOT.parent / "runs" / "mining"; outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"config_adapt_{'-'.join(assets)}_{a.cadence}_w{a.window_days}.json"
        json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha,
                             "note": "deterministic; re-run to regenerate"}, "results": allout},
                  open(path, "w", encoding="utf-8"), indent=2, default=str)
        print(f"[persisted] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
