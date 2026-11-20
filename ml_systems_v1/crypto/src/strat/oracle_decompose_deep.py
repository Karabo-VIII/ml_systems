"""src/strat/oracle_decompose_deep.py -- DEEP oracle-config decomposition: quantify the per-week factors, fit a
PSEUDO-FORMULA for the MA config, across assets AND timeframes. (User 2026-06-10: the 1D single-var-tercile pass was
shallow; this is the depth + breadth.)

THE METHOD (decompose-the-ideal, quantifiable):
  1. The oracle config is a VECTOR OF KNOBS, not "which of 8". The primary CONTINUOUS knob is the MA PERIOD; the
     categorical knobs are STRUCTURE (price>MA / cross / mechanical-exit) and MA-TYPE. We solve the period knob as a
     real formula and the structure knob as a separation.
  2. Per (asset, cadence, week) compute a RICH FACTOR set (quantifiable numbers): efficiency-ratio (trendiness),
     realized vol, vol-of-vol, lag-1 autocorr, Hurst (R/S), range/ATR, dominant-cycle (bars-per-swing), swing
     asymmetry, return skew, net drift.
  3. The ORACLE knob each week = the period (and structure) that maximized capture of the 2-10% moves (hindsight).
  4. QUANTIFY factor -> knob: correlation table + a least-squares PSEUDO-FORMULA  best_period ~ b0 + sum b_i*factor_i
     with R2; + which factor SEPARATES the winning structure. Report per (asset, cadence) -> the MAP (it is NOT
     expected to be universal; the variation IS the finding).
  5. ADAPTABILITY: fit the formula on TRAIN weeks, apply PAST-ONLY to set next-week's period, test vs FIXED on
     held-out. Across assets x cadences.

GROSS capture; long-only; no emoji (cp1252). Run:
  python src/strat/oracle_decompose_deep.py --assets BTC,ETH,SOL --cadences 1d,4h,1h --window-days 7
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
from strat.entry_signal_lab import load_ohlc, _zigzag_pivots

PERIODS = [3, 5, 8, 13, 21, 34, 55]                 # the period knob (continuous-ish, log-spaced)
def _sma(c, n): return pd.Series(c).rolling(n, min_periods=max(2, n // 2)).mean().to_numpy()
def _ema(c, n): return pd.Series(c).ewm(span=n, adjust=False).mean().to_numpy()


# --- rich per-week FACTORS (quantifiable, computed on the window's bars) -------------------------------------
def _hurst(x):
    """R/S Hurst exponent estimate. ~0.5 random, >0.5 trending/persistent, <0.5 mean-reverting."""
    x = np.asarray(x, float); n = len(x)
    if n < 16: return np.nan
    lags = [2, 4, 8, min(16, n // 2)]
    rs = []
    for lag in lags:
        if lag < 2 or lag >= n: continue
        segs = n // lag
        vals = []
        for s in range(segs):
            seg = x[s * lag:(s + 1) * lag]
            if len(seg) < 2: continue
            z = seg - seg.mean(); cz = np.cumsum(z)
            R = cz.max() - cz.min(); S = seg.std()
            if S > 1e-12: vals.append(R / S)
        if vals: rs.append((np.log(lag), np.log(np.mean(vals))))
    if len(rs) < 2: return np.nan
    a = np.array(rs); return float(np.polyfit(a[:, 0], a[:, 1], 1)[0])

def rich_factors(close, high, low):
    ret = np.diff(close) / close[:-1]
    er = abs(close[-1] - close[0]) / (np.sum(np.abs(np.diff(close))) + 1e-12)
    vol = float(np.std(ret)) if len(ret) > 1 else np.nan
    vov = float(np.std(np.abs(ret))) if len(ret) > 2 else np.nan
    ac = float(pd.Series(ret).autocorr(lag=1)) if len(ret) > 3 else np.nan
    hurst = _hurst(close)
    rng = (np.max(high) - np.min(low)) / (np.mean(close) + 1e-12)
    piv = _zigzag_pivots(close, thr=0.02)
    ups = [piv[k + 1][1] / piv[k][1] - 1 for k in range(len(piv) - 1) if piv[k + 1][1] > piv[k][1]]
    downs = [1 - piv[k + 1][1] / piv[k][1] for k in range(len(piv) - 1) if piv[k + 1][1] < piv[k][1]]
    n_sw = len(piv) - 1
    dom_cycle = (len(close) / n_sw) if n_sw > 0 else len(close)        # bars per swing = the move timescale
    swing_asym = (np.mean(ups) - np.mean(downs)) if ups and downs else 0.0
    skew = float(pd.Series(ret).skew()) if len(ret) > 3 else np.nan
    drift = float(close[-1] / close[0] - 1)
    return {"er": er, "vol": vol, "vov": vov, "autocorr": ac, "hurst": hurst, "range": rng,
            "dom_cycle": dom_cycle, "swing_asym": swing_asym, "skew": skew, "drift": drift}

FACTOR_KEYS = ["er", "vol", "vov", "autocorr", "hurst", "range", "dom_cycle", "swing_asym", "skew", "drift"]


# --- the oracle KNOBS per week (best period + best structure) ------------------------------------------------
def _price_above(close, ma_fn, n): return (close > ma_fn(close, n)).astype(float)
def _cross(close, ma_fn, f, s): return (ma_fn(close, f) > ma_fn(close, s)).astype(float)

def precompute_positions(close, ma_fn=_sma):
    """Precompute LAGGED positions ONCE per series (the speed fix). Returns (ret, pos_by_period, cross_pos_list)."""
    ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1.0
    pos_by_period = {}
    for n in PERIODS:
        pl = np.roll(_price_above(close, ma_fn, n), 1); pl[0] = 0.0; pos_by_period[n] = pl
    cross_pos = []
    for f, s in [(5, 20), (8, 21), (13, 34)]:
        pl = np.roll(_cross(close, ma_fn, f, s), 1); pl[0] = 0.0; cross_pos.append(pl)
    return ret, pos_by_period, cross_pos

def week_knobs(ret, pos_by_period, cross_pos, a, b):
    def roi(pl): return float(np.prod(1.0 + pl[a:b + 1] * ret[a:b + 1]) - 1.0)
    pr = {n: roi(pos_by_period[n]) for n in PERIODS}
    best_period = max(pr, key=pr.get); best_period_roi = pr[best_period]
    cr = max(roi(p) for p in cross_pos)
    structure = "trend" if best_period_roi >= cr else "cross"
    return {"best_period": best_period, "best_period_roi": best_period_roi, "structure": structure,
            "best_period_log": float(np.log(best_period)), "roi_by_p": {n: pr[n] for n in PERIODS}}


# --- decompose one (asset, cadence) --------------------------------------------------------------------------
def decompose(asset, cadence, bars=28, step=14, lo=0.02, max_weeks=900):
    df = load_ohlc(asset if asset.endswith("USDT") else asset + "USDT", cadence)
    if df is None or len(df) < 120: return None
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    ret, pos_by_period, cross_pos = precompute_positions(close)
    F, periods, logp, structs = [], [], [], []
    starts = range(60, len(close) - bars, step); guard = 0
    for w0 in starts:
        guard += 1
        if guard > max_weeks: break
        a, b = w0, w0 + bars - 1
        # only weeks with a real 2-10% move
        piv = _zigzag_pivots(close[a:b + 1], thr=lo)
        if not any(piv[k + 1][1] > piv[k][1] and (piv[k + 1][1] / piv[k][1] - 1) >= lo for k in range(len(piv) - 1)):
            continue
        fac = rich_factors(close[a:b + 1], high[a:b + 1], low[a:b + 1])
        kn = week_knobs(ret, pos_by_period, cross_pos, a, b)
        F.append(fac); periods.append(kn["best_period"]); logp.append(kn["best_period_log"]); structs.append(kn["structure"])
    if len(F) < 25: return None
    X = pd.DataFrame(F); yp = np.array(logp)                # fit LOG-period (multiplicative knob)
    # 1) factor -> best_period correlations
    corrs = {k: round(float(X[k].corr(pd.Series(yp))), 3) for k in FACTOR_KEYS if X[k].notna().sum() > 20}
    # 2) least-squares PSEUDO-FORMULA: log_period ~ b0 + sum b_i*z(factor_i); report R2 + standardized coefs
    use = [k for k in FACTOR_KEYS if X[k].notna().sum() > len(X) * 0.8]
    Z = X[use].copy()
    for k in use: Z[k] = (Z[k] - Z[k].mean()) / (Z[k].std() + 1e-9)
    Z = Z.fillna(0.0); A = np.column_stack([np.ones(len(Z)), Z.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, yp, rcond=None)
    pred = A @ coef; ss_res = np.sum((yp - pred) ** 2); ss_tot = np.sum((yp - yp.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    formula = {"intercept_logp": round(float(coef[0]), 3), "intercept_period": round(float(np.exp(coef[0])), 1),
               "std_coefs": {k: round(float(c), 3) for k, c in zip(use, coef[1:])}, "R2": round(r2, 3)}
    # 3) structure separation: which factor best separates trend vs cross weeks (mean diff in std units)
    st = np.array(structs); sep = {}
    for k in FACTOR_KEYS:
        v = X[k].to_numpy(float); m = np.isfinite(v)
        if m.sum() < 20: continue
        tr = v[m & (st == "trend")]; cx = v[m & (st == "cross")]
        if len(tr) > 5 and len(cx) > 5:
            sep[k] = round(float((np.mean(tr) - np.mean(cx)) / (np.std(v[m]) + 1e-9)), 3)
    return {"n_weeks": len(F), "period_mean": round(float(np.mean(periods)), 1),
            "period_std": round(float(np.std(periods)), 1), "pct_trend_structure": round(float(np.mean(st == "trend")), 2),
            "factor_period_corr": corrs, "pseudo_formula": formula, "structure_separation": sep}


# --- adaptability: does the fitted period-formula beat FIXED on held-out next-week? ---------------------------
def adaptability(asset, cadence, bars=28, step=14, lo=0.02, train_frac=0.6):
    df = load_ohlc(asset if asset.endswith("USDT") else asset + "USDT", cadence)
    if df is None or len(df) < 120: return None
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    ret, pos_by_period, cross_pos = precompute_positions(close)
    recs = []
    for w0 in range(60, len(close) - bars, step):
        a, b = w0, w0 + bars - 1
        piv = _zigzag_pivots(close[a:b + 1], thr=lo)
        if not any(piv[k + 1][1] > piv[k][1] and (piv[k + 1][1] / piv[k][1] - 1) >= lo for k in range(len(piv) - 1)): continue
        fac = rich_factors(close[a:b + 1], high[a:b + 1], low[a:b + 1])
        kn = week_knobs(ret, pos_by_period, cross_pos, a, b)
        recs.append({"fac": fac, "roi_by_p": kn["roi_by_p"], "best_period": kn["best_period"], "logp": kn["best_period_log"]})
    if len(recs) < 30: return None
    ntr = int(len(recs) * train_frac); tr, te = recs[:ntr], recs[ntr:]
    use = [k for k in FACTOR_KEYS]
    Xtr = pd.DataFrame([r["fac"] for r in tr])[use]; mu = Xtr.mean(); sd = Xtr.std() + 1e-9
    Ztr = ((Xtr - mu) / sd).fillna(0.0); ytr = np.array([r["logp"] for r in tr])
    A = np.column_stack([np.ones(len(Ztr)), Ztr.to_numpy()]); coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    fixed_period = int(round(np.exp(np.mean(ytr))))            # the train-mean period (the FIXED baseline)
    fixed_period = min(PERIODS, key=lambda p: abs(p - fixed_period))
    ad_roi, fx_roi, or_roi = [], [], []
    for i in range(len(te) - 1):
        r, nxt = te[i], te[i + 1]
        z = ((pd.Series(r["fac"])[use] - mu) / sd).fillna(0.0).to_numpy()
        pred_logp = float(coef[0] + z @ coef[1:]); pred_p = min(PERIODS, key=lambda p: abs(np.log(p) - pred_logp))
        ad_roi.append(nxt["roi_by_p"][pred_p]); fx_roi.append(nxt["roi_by_p"][fixed_period])
        or_roi.append(max(nxt["roi_by_p"].values()))
    adp, fxp, orp = np.array(ad_roi), np.array(fx_roi), np.array(or_roi)
    return {"n_test": len(adp), "fixed_period": fixed_period,
            "adaptive_mean_roi": round(float(np.mean(adp)) * 100, 3), "fixed_mean_roi": round(float(np.mean(fxp)) * 100, 3),
            "oracle_mean_roi": round(float(np.mean(orp)) * 100, 3),
            "adaptive_beats_fixed_frac": round(float(np.mean(adp > fxp)), 2),
            "adaptive_edge_vs_fixed_pp": round(float(np.mean(adp - fxp)) * 100, 3)}


def adaptability_predictive(asset, cadence, bars=28, step=14, lo=0.02, train_frac=0.6):
    """HONEST predictive adaptability: predict NEXT window's best period from THIS window's factors (factor_w ->
    period_{w+1}). Linear vs GBM (the ML-headroom). Realize next-window roi at the predicted period vs FIXED vs ORACLE."""
    df = load_ohlc(asset if asset.endswith("USDT") else asset + "USDT", cadence)
    if df is None or len(df) < 120: return None
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    ret, pos_by_period, cross_pos = precompute_positions(close)
    recs = []
    for w0 in range(60, len(close) - bars, step):
        a, b = w0, w0 + bars - 1
        piv = _zigzag_pivots(close[a:b + 1], thr=lo)
        if not any(piv[k + 1][1] > piv[k][1] and (piv[k + 1][1] / piv[k][1] - 1) >= lo for k in range(len(piv) - 1)): continue
        kn = week_knobs(ret, pos_by_period, cross_pos, a, b)
        recs.append({"fac": rich_factors(close[a:b + 1], high[a:b + 1], low[a:b + 1]),
                     "roi_by_p": kn["roi_by_p"], "logp": kn["best_period_log"]})
    if len(recs) < 40: return None
    use = FACTOR_KEYS
    # PAIRS: X = factor_w, y = logp_{w+1}, realize on roi_by_p_{w+1}
    Xf = pd.DataFrame([r["fac"] for r in recs[:-1]])[use]
    y = np.array([recs[i + 1]["logp"] for i in range(len(recs) - 1)])
    nxt_roi = [recs[i + 1]["roi_by_p"] for i in range(len(recs) - 1)]
    ntr = int(len(Xf) * train_frac)
    mu = Xf.iloc[:ntr].mean(); sd = Xf.iloc[:ntr].std() + 1e-9
    Z = ((Xf - mu) / sd).fillna(0.0).to_numpy()
    Atr = np.column_stack([np.ones(ntr), Z[:ntr]]); coef, *_ = np.linalg.lstsq(Atr, y[:ntr], rcond=None)
    fixed_p = min(PERIODS, key=lambda pp: abs(pp - int(round(np.exp(np.mean(y[:ntr]))))))
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8)
        gb.fit(Z[:ntr], y[:ntr]); has_gb = True
    except Exception: has_gb = False
    lin, gbm, fx, orc = [], [], [], []
    for i in range(ntr, len(Xf)):
        z = Z[i]
        pl = float(coef[0] + z @ coef[1:]); pp = min(PERIODS, key=lambda q: abs(np.log(q) - pl))
        lin.append(nxt_roi[i][pp])
        if has_gb:
            pg = float(gb.predict(z.reshape(1, -1))[0]); pgp = min(PERIODS, key=lambda q: abs(np.log(q) - pg))
            gbm.append(nxt_roi[i][pgp])
        fx.append(nxt_roi[i][fixed_p]); orc.append(max(nxt_roi[i].values()))
    lin, fx, orc = np.array(lin), np.array(fx), np.array(orc)
    out = {"n_test": len(lin), "fixed_period": fixed_p,
           "linear_mean_roi_pct": round(float(np.mean(lin)) * 100, 3), "fixed_mean_roi_pct": round(float(np.mean(fx)) * 100, 3),
           "oracle_mean_roi_pct": round(float(np.mean(orc)) * 100, 3),
           "linear_beats_fixed_frac": round(float(np.mean(lin > fx)), 2),
           "linear_edge_vs_fixed_pp": round(float(np.mean(lin - fx)) * 100, 3)}
    if has_gb:
        gbm = np.array(gbm)
        out.update({"gbm_mean_roi_pct": round(float(np.mean(gbm)) * 100, 3),
                    "gbm_beats_fixed_frac": round(float(np.mean(gbm > fx)), 2),
                    "gbm_edge_vs_fixed_pp": round(float(np.mean(gbm - fx)) * 100, 3),
                    "gbm_beats_linear_pp": round(float(np.mean(gbm - lin)) * 100, 3)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.oracle_decompose_deep")
    ap.add_argument("--assets", default="BTC,ETH,SOL"); ap.add_argument("--cadences", default="1d,4h,1h")
    ap.add_argument("--bars", type=int, default=28); ap.add_argument("--step", type=int, default=14); ap.add_argument("--predictive", action="store_true", help="honest predictive adaptability (factor_w->period_w+1) + GBM ML-headroom")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv); allout = {}
    for sym in a.assets.split(","):
        for cad in a.cadences.split(","):
            d = decompose(sym, cad, a.bars, a.step)
            if not d: print(f"## {sym} {cad}: insufficient weeks"); continue
            ad = adaptability(sym, cad, a.bars, a.step)
            if a.predictive:
                pad = adaptability_predictive(sym, cad, a.bars, a.step)
                if pad:
                    g = f"  GBM={pad.get('gbm_mean_roi_pct')}% (edge_vs_fix {pad.get('gbm_edge_vs_fixed_pp')}pp, vs_lin {pad.get('gbm_beats_linear_pp')}pp)" if 'gbm_mean_roi_pct' in pad else ""
                    print(f"   PREDICTIVE (factor_w->period_w+1): linear={pad['linear_mean_roi_pct']}% (edge_vs_fix {pad['linear_edge_vs_fixed_pp']}pp) fixed={pad['fixed_mean_roi_pct']}% oracle={pad['oracle_mean_roi_pct']}%{g}")
            allout[f"{sym}_{cad}"] = {"decompose": d, "adaptability": ad}
            f = d["pseudo_formula"]
            topc = sorted(d["factor_period_corr"].items(), key=lambda kv: -abs(kv[1]))[:4]
            tops = sorted(d["structure_separation"].items(), key=lambda kv: -abs(kv[1]))[:3]
            print(f"## {sym} {cad} -- {d['n_weeks']} weeks | best-period mean={d['period_mean']} std={d['period_std']} | {d['pct_trend_structure']:.0%} trend-structure")
            print(f"   factor->log(period) corr (top): " + "  ".join(f"{k}={v:+.2f}" for k, v in topc))
            print(f"   PSEUDO-FORMULA log(period) R2={f['R2']:.2f}  base_period={f['intercept_period']}  top std-coefs: " +
                  "  ".join(f"{k}={c:+.2f}" for k, c in sorted(f['std_coefs'].items(), key=lambda kv: -abs(kv[1]))[:4]))
            print(f"   structure(trend vs cross) separated by: " + "  ".join(f"{k}={v:+.2f}" for k, v in tops))
            if ad:
                print(f"   ADAPTABILITY (period-formula vs fixed, held-out): adaptive={ad['adaptive_mean_roi']}%  fixed={ad['fixed_mean_roi']}%  "
                      f"oracle={ad['oracle_mean_roi']}%  beats_fixed={ad['adaptive_beats_fixed_frac']:.0%}  edge={ad['adaptive_edge_vs_fixed_pp']:+.2f}pp")
    if a.json and allout:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        outdir = ROOT.parent / "runs" / "mining"; outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"oracle_decompose_deep_{a.assets.replace(',', '-')}_{a.cadences.replace(',', '-')}_b{a.bars}.json"
        json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha}, "results": allout},
                  open(path, "w", encoding="utf-8"), indent=2, default=str)
        print(f"[persisted] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
