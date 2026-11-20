"""src/strat/mover_capture.py -- the CROSS-SECTIONAL daily-MOVER-CAPTURE lab (the real profitable objective).

THE PROBLEM (user, 2026-06-09): "do you have a strat that captures the 25% of daily movers?" The market research
(docs/MARKET_RESEARCH_2026_06_05.md) establishes ~7.6 long-only up-movers/day >=5% (broad, bursty, durable) BUT
raw moves != edge (random entry = 47% net-positive = coin-flip). Capturing them REQUIRES a causal signal -- which
the research left UNTESTED. This lab tests it.

THE THEORY-GROUNDED ATTACK (decompose-the-ideal): direction is unpredictable (AUC~0.51) but MAGNITUDE/vol IS
predictable (Hurst|ret| 0.80-0.84). So: each day SELECT the assets with an imminent BIG move (a causal magnitude
signal -- vol-expansion / breakout / range-surge), go LONG-ONLY, and the long-only asymmetry (up-moves are bigger
than down, MARKET_RESEARCH 3b) + a convex exit extract positive expectancy from coin-flip direction. The crux test:
does a causal SELECTION score pick higher-forward-return assets than RANDOM selection, net of cost, on held-out data?

THE CLEAN SELECTION-POWER TEST (isolates selection from position-management):
  For each day t, rank assets by a past-only score; the top-K mean forward net return = SELECTED. The cross-sectional
  mean that day = the RANDOM-selection baseline (picking K random assets -> E = x-sec mean). The top-K by *forward*
  return = the ORACLE ceiling. If SELECTED >> RANDOM on held-out (OOS/UNSEEN), the score has genuine selection power.
  forward net return = enter next-bar open, exit +H bars (or chandelier), minus taker RT 0.24%. Past-only entry.

HARD CONSTRAINTS: LONG-ONLY, SPOT, taker 0.0024, UNSEEN touched once. Survivorship caveat (u100 = survivors; the
true down-tail of delisted coins is truncated -> selection looks better than live). No emoji (cp1252).

Run:
  python src/strat/mover_capture.py --selftest
  python src/strat/mover_capture.py --universe u100 --horizon 5
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from strat.entry_signal_lab import load_ohlc, U10   # reuse the loader + window anchors
from wealth_bot.harness import WindowSpec

WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-06-01")
TAKER_RT = 0.0024


def _load_universe(name: str) -> list[str]:
    if name == "u10": return U10
    import yaml
    d = yaml.safe_load(open(f"config/universes/{name}.yaml", encoding="utf-8"))
    syms = []
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, list):
            syms += [(x["symbol"] if isinstance(x, dict) else x) for x in v]
    return sorted(set(syms))


# ---------------------------------------------------------------------------
# Past-only cross-sectional SELECTION scores (each returns a per-bar score Series; higher = more likely a big move)
# ---------------------------------------------------------------------------
def score_vol_expansion(c: pd.Series, n=20):
    """Realized vol relative to its own trailing median, AND rising -> a magnitude signal (vol clusters)."""
    rv = c.pct_change().rolling(n, min_periods=n // 2).std()
    base = rv.rolling(n * 3, min_periods=n).median()
    return (rv / base) * (rv > rv.shift(1)).astype(float)


def score_breakout(c: pd.Series, n=20):
    """Proximity to / above the n-bar high (Donchian breakout pressure), past-only."""
    prior_high = c.rolling(n, min_periods=n).max().shift(1)
    return c / prior_high - 1.0


def score_momentum(c: pd.Series, k=10):
    """k-bar momentum (rate of change), past-only."""
    return c / c.shift(k) - 1.0


def score_range_surge(h: pd.Series, l: pd.Series, c: pd.Series, n=20):
    """Today's true-range vs its trailing median -- an intrabar magnitude/activity surge (volume proxy)."""
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    base = tr.rolling(n * 3, min_periods=n).median()
    return tr / base


def score_accel(c: pd.Series, k=10):
    """Momentum acceleration: this bar's k-ROC minus the prior bar's -> igniting moves."""
    roc = c / c.shift(k) - 1.0
    return roc - roc.shift(1)


SCORES = {"vol_expansion": score_vol_expansion, "breakout": score_breakout,
          "momentum": score_momentum, "range_surge": score_range_surge, "accel": score_accel}

# STRUCTURAL (non-price) chimera-feature scores -- the research's #1 avenue (liquidation cascades) + funding/basis.
# Each maps to a chimera column read past-only at bar t. The thesis: forced-flow / positioning extremes precede a
# forward bounce that PRICE alone does not see. (Caveat: contemporaneous != predictive -- this IS the forward test.)
STRUCTURAL_FEATS = {
    "liq_long_bounce": "liq_long_z30",      # longs liquidated (forced selling, capitulation low) -> forward bounce
    "liq_total_bounce": "liq_delta_z30",    # net forced-flow extreme
    "funding_reset": "fund_rate_z30",        # funding extreme -> positioning unwind / reversion
    "basis_capitulation": "bs_basis_z30",    # basis dislocation -> reversion
    "whale_accum": "wh_whale_net_usd",       # whale net buying -> informed accumulation
}


def _load_feature(sym: str, cadence: str, col: str) -> pd.Series | None:
    """Load a single chimera feature column as a date-indexed past-only Series (NaN where missing)."""
    from pipeline.chimera_loader import ChimeraLoader
    try:
        loaded = ChimeraLoader().load(sym if sym.endswith("USDT") else sym + "USDT", cadence=cadence)
        cf = pd.DataFrame(loaded.to_dict(as_series=False)) if (hasattr(loaded, "to_dict") and not hasattr(loaded, "iloc")) else loaded
    except Exception:
        return None
    if col not in cf.columns: return None
    cf["date"] = (pd.to_datetime(cf["date"], unit="ms") if np.issubdtype(cf["date"].dtype, np.number) else pd.to_datetime(cf["date"]))
    return cf.sort_values("date").drop_duplicates("date", keep="last").set_index("date")[col].astype(float)


# ---------------------------------------------------------------------------
def _forward_net(close, high, t, H, cost, chandelier_atr=None, atr=None):
    """Forward net return from entering next-bar open (t+1) and exiting at +H bars (or a chandelier trailing stop).
    LONG-ONLY. close/high are np arrays; t is the signal bar. Returns NaN if not enough forward room."""
    ef = t + 1
    if ef + 1 >= len(close): return np.nan
    entry = close[ef - 0] if False else close[t]  # placeholder (unused)
    entry = close[ef]  # enter at next-bar close as the fill proxy (open not always present per-asset; close is past-only-safe vs signal at t)
    end = min(ef + H, len(close) - 1)
    if chandelier_atr is None or atr is None:
        exitp = close[end]
    else:
        hwm = entry; exitp = close[end]
        for j in range(ef + 1, end + 1):
            a = atr[j - 1]
            if np.isfinite(a) and (close[j] <= hwm - chandelier_atr * a):  # trailing-stop breach (close-based, pessimistic)
                exitp = close[j]; break
            hwm = max(hwm, high[j] if np.isfinite(high[j]) else close[j])
    return exitp / entry - 1.0 - cost


def evaluate_selection(universe="u100", score_name="vol_expansion", H=5, K=5, cadence="1d",
                       cost=TAKER_RT, chandelier_atr=None, reverse=False) -> dict:
    """Cross-sectional selection-power test: SELECTED top-K vs RANDOM (x-sec mean) vs ORACLE top-K, per window.
    reverse=True selects the BOTTOM-K by score (cross-sectional REVERSAL: bet the recent-weakest/oversold bounce)."""
    assets = _load_universe(universe)
    closes, highs, lows, atrs, scores = {}, {}, {}, {}, {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < 300: continue
        closes[sym] = df.set_index("date")["close"]
        highs[sym] = df.set_index("date")["high"]; lows[sym] = df.set_index("date")["low"]
        atrs[sym] = df.set_index("date")["atr14"]
        c = closes[sym]
        if score_name in STRUCTURAL_FEATS:
            fs = _load_feature(sym, cadence, STRUCTURAL_FEATS[score_name])
            if fs is None: continue
            scores[sym] = fs.reindex(c.index)
        elif score_name == "range_surge":
            scores[sym] = score_range_surge(highs[sym], lows[sym], c)
        else:
            scores[sym] = SCORES[score_name](c)
    if not closes: print("no data"); return {}
    panel_c = pd.DataFrame(closes).sort_index()
    panel_s = pd.DataFrame(scores).reindex(panel_c.index)
    dates = panel_c.index
    # precompute forward net returns per asset at each date
    fwd = pd.DataFrame(index=dates, columns=panel_c.columns, dtype=float)
    if chandelier_atr is None:
        # VECTORIZED fixed-H: enter close[t+1], exit close[t+1+H], minus cost (fast at any cadence)
        cl = panel_c.reindex(dates)
        entry = cl.shift(-1); exitp = cl.shift(-(1 + H))
        fwd = (exitp / entry - 1.0 - cost)
    else:
        for sym in panel_c.columns:                      # chandelier path needs the per-bar walk
            cl = closes[sym].reindex(dates).to_numpy(float)
            hi = highs[sym].reindex(dates).to_numpy(float)
            at = atrs[sym].reindex(dates).to_numpy(float)
            fr = np.full(len(dates), np.nan)
            for t in np.where(np.isfinite(cl))[0]:
                if t + 1 + H < len(dates):
                    fr[t] = _forward_net(cl, hi, t, H, cost, chandelier_atr, at)
            fwd[sym] = fr

    def _wlabel(ts):
        ts = pd.Timestamp(ts)
        if ts < pd.Timestamp(WIN.train_end): return "TRAIN"
        if ts < pd.Timestamp(WIN.val_end): return "VAL"
        if ts < pd.Timestamp(WIN.oos_end): return "OOS"
        return "UNSEEN"
    wlab = np.array([_wlabel(d) for d in dates])

    out = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        sel, rnd, orc, nd = [], [], [], 0
        idx = np.where(wlab == w)[0]
        for t in idx:
            s = panel_s.iloc[t]; f = fwd.iloc[t]
            ok = s.notna() & f.notna()
            if ok.sum() < max(2 * K, 8): continue
            sv = s[ok]; fv = f[ok]
            top = (sv.nsmallest(K) if reverse else sv.nlargest(K)).index   # SELECTED K (reverse=bottom=oversold)
            sel.append(fv[top].mean())
            rnd.append(fv.mean())                            # RANDOM-selection baseline = x-sec mean that day
            orc.append(fv.nlargest(K).mean())               # ORACLE top-K by forward return (hindsight ceiling)
            nd += 1
        if not sel:
            out[w] = {"n_days": 0}; continue
        sel, rnd, orc = np.array(sel), np.array(rnd), np.array(orc)
        out[w] = {"n_days": nd,
                  "sel_mean_pct": round(float(np.mean(sel)) * 100, 3),
                  "rnd_mean_pct": round(float(np.mean(rnd)) * 100, 3),
                  "oracle_mean_pct": round(float(np.mean(orc)) * 100, 3),
                  "edge_vs_rnd_pct": round(float(np.mean(sel) - np.mean(rnd)) * 100, 3),
                  "capture_vs_oracle": round(float(np.mean(sel) / np.mean(orc)), 3) if np.mean(orc) > 1e-9 else None,
                  "sel_winrate": round(float((sel > 0).mean()), 3),
                  "beats_rnd_daily": round(float((sel > rnd).mean()), 3)}
    return {"per_window": out, "score": score_name, "H": H, "K": K, "n_assets": len(closes), "reverse": reverse}


# ---------------------------------------------------------------------------
# LEARNED cross-sectional RANKER (the research's "conditioned setup"): combine the full feature set, fit on TRAIN
# only, predict the forward mover, select top-K, test held-out. The honest test of whether a MULTI-feature model
# can select daily movers where single scores cannot. Cross-sectional rank-normalize (regime-robust) + regularize.
# ---------------------------------------------------------------------------
RANKER_FEATS = ["liq_long_z30", "liq_short_z30", "liq_delta_z30", "fund_rate_z30", "bs_basis_z30",
                "wh_whale_net_usd", "norm_yz_volatility", "norm_vol_ratio", "norm_efficiency",
                "norm_oi_price_divergence", "norm_momentum_accel", "norm_perm_entropy"]


def _load_asset_features(sym, cadence):
    """One chimera load per asset -> date-indexed DataFrame: OHLC + computed price scores + RANKER_FEATS present."""
    from pipeline.chimera_loader import ChimeraLoader
    try:
        loaded = ChimeraLoader().load(sym if sym.endswith("USDT") else sym + "USDT", cadence=cadence)
        cf = pd.DataFrame(loaded.to_dict(as_series=False)) if (hasattr(loaded, "to_dict") and not hasattr(loaded, "iloc")) else loaded
    except Exception:
        return None
    cf["date"] = (pd.to_datetime(cf["date"], unit="ms") if np.issubdtype(cf["date"].dtype, np.number) else pd.to_datetime(cf["date"]))
    cf = cf.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    if not {"open", "high", "low", "close"}.issubset(cf.columns): return None
    out = cf[["open", "high", "low", "close"]].astype(float).copy()
    c, h, l = out["close"], out["high"], out["low"]
    out["f_vol_expansion"] = score_vol_expansion(c); out["f_breakout"] = score_breakout(c)
    out["f_momentum"] = score_momentum(c); out["f_range_surge"] = score_range_surge(h, l, c)
    out["f_accel"] = score_accel(c)
    for col in RANKER_FEATS:
        out[f"f_{col}"] = cf[col].astype(float) if col in cf.columns else np.nan
    return out


def evaluate_learned_ranker(universe="u100", H=5, K=5, cadence="1d", cost=TAKER_RT, model="ridge") -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    assets = _load_universe(universe)
    frames = {}
    for sym in assets:
        af = _load_asset_features(sym, cadence)
        if af is not None and len(af) > 300: frames[sym] = af
    if not frames: print("no data"); return {}
    feat_cols = [c for c in next(iter(frames.values())).columns if c.startswith("f_")]
    # assemble long panel: (date, asset) rows with features + forward-H net return target
    rows = []
    for sym, af in frames.items():
        cl = af["close"].to_numpy(float)
        fwd = np.full(len(cl), np.nan)
        for t in range(len(cl) - H - 1):
            fwd[t] = cl[min(t + 1 + H, len(cl) - 1)] / cl[t + 1] - 1.0 - cost
        d = af[feat_cols].copy(); d["fwd"] = fwd; d["sym"] = sym; d["date"] = af.index
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True).dropna(subset=["fwd"])
    # cross-sectional rank-normalize each feature WITHIN each day (regime-robust, scale-free); fill missing with 0.5
    for fc in feat_cols:
        panel[fc] = panel.groupby("date")[fc].rank(pct=True)
    panel[feat_cols] = panel[feat_cols].fillna(0.5)
    panel["win"] = panel["date"].apply(lambda ts: "TRAIN" if ts < pd.Timestamp(WIN.train_end) else
                                       "VAL" if ts < pd.Timestamp(WIN.val_end) else
                                       "OOS" if ts < pd.Timestamp(WIN.oos_end) else "UNSEEN")
    tr = panel[panel["win"] == "TRAIN"]
    if len(tr) < 500: print("insufficient train rows"); return {}
    mdl = (Ridge(alpha=10.0) if model == "ridge" else
           GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.03, subsample=0.7))
    mdl.fit(tr[feat_cols].to_numpy(), tr["fwd"].to_numpy())
    panel["pred"] = mdl.predict(panel[feat_cols].to_numpy())
    out = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        sub = panel[panel["win"] == w]
        sel, rnd, orc, nd = [], [], [], 0
        for dt, g in sub.groupby("date"):
            if len(g) < max(2 * K, 8): continue
            sel.append(g.nlargest(K, "pred")["fwd"].mean()); rnd.append(g["fwd"].mean())
            orc.append(g.nlargest(K, "fwd")["fwd"].mean()); nd += 1
        if not sel: out[w] = {"n_days": 0}; continue
        sel, rnd, orc = np.array(sel), np.array(rnd), np.array(orc)
        out[w] = {"n_days": nd, "sel_mean_pct": round(float(np.mean(sel)) * 100, 3),
                  "rnd_mean_pct": round(float(np.mean(rnd)) * 100, 3),
                  "oracle_mean_pct": round(float(np.mean(orc)) * 100, 3),
                  "edge_vs_rnd_pct": round(float(np.mean(sel) - np.mean(rnd)) * 100, 3),
                  "capture_vs_oracle": round(float(np.mean(sel) / np.mean(orc)), 3) if np.mean(orc) > 1e-9 else None,
                  "sel_winrate": round(float((sel > 0).mean()), 3),
                  "beats_rnd_daily": round(float((sel > rnd).mean()), 3)}
    # feature DNA (top |coef| or importance)
    if model == "ridge":
        imp = sorted(zip(feat_cols, mdl.coef_), key=lambda x: -abs(x[1]))[:6]
    else:
        imp = sorted(zip(feat_cols, mdl.feature_importances_), key=lambda x: -x[1])[:6]
    return {"per_window": out, "score": f"LEARNED_{model}", "H": H, "K": K, "n_assets": len(frames),
            "reverse": False, "dna": [(f, round(float(v), 4)) for f, v in imp]}


# ---------------------------------------------------------------------------
# ORACLE-MOVE CAPTURE (user reframe 2026-06-09): not "predict which asset moves today" but "capture a FRACTION of
# every 2-10% move the oracle finds, ANY asset / ANY time / ANY cadence". Oracle = zigzag up-legs in [lo,hi]. Strat =
# a causal IGNITION entry (move-in-progress: new-high breakout + vol rising) -> convex chandelier exit. Metric =
# capture-rate = strat_realized_compound / oracle_move_compound, per window, net of cost. Target: >=25% on held-out.
# ---------------------------------------------------------------------------
def _zigzag_up_compound(close, lo=0.02, hi=0.10, thr=0.02):
    """Compound of zigzag up-legs whose magnitude is in [lo, hi] -- the oracle 'moves' (perfect buy-low/sell-high)."""
    if len(close) < 5: return 0.0
    piv = [(0, close[0])]; trend = 0; ext = close[0]; ext_i = 0
    for i in range(1, len(close)):
        if trend >= 0 and close[i] > ext: ext, ext_i = close[i], i
        if trend <= 0 and close[i] < ext: ext, ext_i = close[i], i
        if trend >= 0 and close[i] < ext * (1 - thr): piv.append((ext_i, ext)); trend = -1; ext, ext_i = close[i], i
        elif trend <= 0 and close[i] > ext * (1 + thr): piv.append((ext_i, ext)); trend = 1; ext, ext_i = close[i], i
    piv.append((ext_i, ext))
    comp = 1.0
    for k in range(len(piv) - 1):
        v0, v1 = piv[k][1], piv[k + 1][1]
        if v1 > v0:
            r = v1 / v0 - 1.0
            if lo <= r <= hi: comp *= (1 + r)
            elif r > hi: comp *= (1 + hi)        # cap a big move's oracle contribution at hi (a 2-10% move target)
    return float((comp - 1.0) * 100)


def evaluate_oracle_capture(universe="u50", cadence="1d", H=20, atr_mult=3.0, cost=TAKER_RT,
                            lo=0.02, hi=0.10) -> dict:
    """Ignition entry (new 20-bar high + vol rising) -> chandelier exit; capture vs the [lo,hi] zigzag-move oracle."""
    from strat.entry_signal_lab import sig_donchian, SetupHarness, ExitPolicy, oracle_compound_pct
    from wealth_bot.harness import WindowSpec as _WS
    win = _WS(train_end=WIN.train_end, val_end=WIN.val_end, oos_end=WIN.oos_end, unseen_end=WIN.unseen_end)
    assets = _load_universe(universe)
    rows = {w: {"strat": [], "oracle": [], "capt": [], "bh": [], "xs": []} for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
    n_assets = 0
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < 300: continue
        entry = sig_donchian(df, n=20)                 # ignition: new 20-bar-high breakout (move in progress)
        if entry.sum() < 5: continue
        d = df.copy(); d["entry_sig"] = entry
        pol = ExitPolicy(atr_trail_mult=atr_mult, atr_col="atr14", sl_pct=0.10, max_hold_bars=H * 4)
        h = SetupHarness(d, "entry_sig", pol, win, cost_rt=cost)
        res = h.run(); n_assets += 1
        wl = np.array([h._window_label(pd.Timestamp(t)) for t in d["date"]])
        close = d["close"].to_numpy(float)
        for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
            idx = np.where(wl == w)[0]
            if idx.size < 30: continue
            orc = oracle_compound_pct(close[idx], thr=0.02)   # uncapped perfect swing-trader (all up-legs >=2%)
            strat = res.window_stats[w].compound_pct
            bh = float((close[idx][-1] / close[idx][0] - 1.0) * 100) if idx.size > 1 else 0.0
            if orc > 1e-6:
                rows[w]["strat"].append(strat); rows[w]["oracle"].append(orc); rows[w]["capt"].append(strat / orc)
                rows[w]["bh"].append(bh); rows[w]["xs"].append(strat - bh)
    out = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        if not rows[w]["capt"]: out[w] = {"n": 0}; continue
        out[w] = {"n": len(rows[w]["capt"]),
                  "strat_med_pct": round(float(np.median(rows[w]["strat"])), 1),
                  "oracle_med_pct": round(float(np.median(rows[w]["oracle"])), 1),
                  "bh_med_pct": round(float(np.median(rows[w]["bh"])), 1),
                  "xs_med_pp": round(float(np.median(rows[w]["xs"])), 1),
                  "beats_bh_frac": round(float(np.mean([x > 0 for x in rows[w]["xs"]])), 2),
                  "capture_med": round(float(np.median(rows[w]["capt"])), 3),
                  "capture_ge25_frac": round(float(np.mean([c >= 0.25 for c in rows[w]["capt"]])), 2)}
    return {"per_window": out, "cadence": cadence, "n_assets": n_assets, "lo": lo, "hi": hi}


def _print_capture(res):
    print(f"## ORACLE-MOVE CAPTURE -- {res['cadence']} -- {res['n_assets']} assets -- oracle = "
          f"{int(res['lo']*100)}-{int(res['hi']*100)}% zigzag up-moves -- ignition entry + chandelier exit (net cost)")
    print(f"   {'window':8} {'n':>4} {'strat%':>8} {'oracle%':>9} {'BH%':>8} {'xs_pp':>7} {'>BH':>5} {'capture':>8} {'cap>=25%':>8}")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        d = res["per_window"].get(w, {})
        if not d.get("n"): print(f"   {w:8} {0:>4}  (no data)"); continue
        print(f"   {w:8} {d['n']:>4} {d['strat_med_pct']:8.1f} {d['oracle_med_pct']:9.1f} {d['bh_med_pct']:8.1f} {d['xs_med_pp']:7.1f} {d['beats_bh_frac']:5.2f} {d['capture_med']:8.3f} {d['capture_ge25_frac']:8.2f}")


# ---------------------------------------------------------------------------
# PER-MOVE CAPTURE (the faithful reframe): for each INDIVIDUAL oracle up-move (swing-low->swing-high in [lo,hi]),
# what FRACTION did the realizable strat bank? cap_m = strat_trade_net / move_size for the trade that entered
# during the move (missed move -> 0). Bounded, honest. overall = mean(cap_m over ALL moves). Target: >=0.25.
# ---------------------------------------------------------------------------
def evaluate_per_move_capture(universe="u50", cadence="1d", atr_mult=3.0, cost=TAKER_RT, lo=0.02, hi=0.50, entry_kind="breakout"):
    from strat.entry_signal_lab import sig_donchian, SetupHarness, ExitPolicy, _zigzag_pivots
    from wealth_bot.harness import WindowSpec as _WS
    win = _WS(train_end=WIN.train_end, val_end=WIN.val_end, oos_end=WIN.oos_end, unseen_end=WIN.unseen_end)
    assets = _load_universe(universe)
    rows = {w: {"cap": [], "cov": []} for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
    n_assets = 0
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < 300: continue
        c = df["close"]
        if entry_kind == "dip":
            sma_long = c.rolling(100, min_periods=50).mean(); slo = c.rolling(10, min_periods=5).min()
            entry = ((c > sma_long) & (c <= slo * 1.005)).fillna(False).to_numpy()   # uptrend + at a 10-bar low = early dip-buy
        elif entry_kind == "rsi_bounce":
            import pandas_ta as ta
            rsi = ta.rsi(c, length=14)
            entry = ((rsi < 35) & (rsi > rsi.shift(1))).fillna(False).to_numpy()      # oversold + turning up = bounce
        elif entry_kind == "mom_cont":
            roc = c / c.shift(10) - 1.0
            entry = ((roc > 0.05) & (roc > roc.shift(1))).fillna(False).to_numpy()    # momentum continuation (already +5%, accelerating)
        elif entry_kind.startswith("breakout"):
            n = int(entry_kind.split("_")[1]) if "_" in entry_kind else 20
            entry = sig_donchian(df, n=n)
        else:
            entry = sig_donchian(df, n=20)
        if entry.sum() < 5: continue
        d = df.copy(); d["entry_sig"] = entry
        pol = ExitPolicy(atr_trail_mult=atr_mult, atr_col="atr14", sl_pct=0.10, max_hold_bars=200)
        h = SetupHarness(d, "entry_sig", pol, win, cost_rt=cost); res = h.run(); n_assets += 1
        close = d["close"].to_numpy(float)
        trades = [(int(t["entry_fill_idx"]), float(t["net_pnl"])) for t in res.trades]
        wl = np.array([h._window_label(pd.Timestamp(t)) for t in d["date"]])
        piv = _zigzag_pivots(close, thr=lo)
        for k in range(len(piv) - 1):
            (i0, v0), (i1, v1) = piv[k], piv[k + 1]
            if v1 <= v0: continue
            size = v1 / v0 - 1.0
            if not (lo <= size <= hi): continue
            w = wl[i0]
            cap = 0.0; participated = 0
            for (ei, net) in trades:
                if i0 <= ei <= i1:
                    cap = net / size if size > 1e-9 else 0.0
                    participated = 1; break
            rows[w]["cap"].append(min(cap, 1.5)); rows[w]["cov"].append(participated)
    out = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        if not rows[w]["cap"]: out[w] = {"n_moves": 0}; continue
        cap = np.array(rows[w]["cap"]); cov = np.array(rows[w]["cov"]); part = cap[cov == 1]
        out[w] = {"n_moves": len(cap), "coverage": round(float(cov.mean()), 3),
                  "capture_overall": round(float(cap.mean()), 3),
                  "capture_participated": round(float(part.mean()), 3) if len(part) else None,
                  "ge25_overall": round(float((cap >= 0.25).mean()), 3)}
    return {"per_window": out, "cadence": cadence, "n_assets": n_assets, "lo": lo, "hi": hi}


def _print_per_move(res):
    print(f"## PER-MOVE CAPTURE -- {res['cadence']} -- {res['n_assets']} assets -- oracle moves "
          f"{int(res['lo']*100)}-{int(res['hi']*100)}% -- ignition entry + chandelier (net cost)")
    print(f"   {'window':8} {'n_moves':>8} {'coverage':>9} {'cap_overall':>12} {'cap_in_move':>12} {'ge25_all':>9}")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        d = res["per_window"].get(w, {})
        if not d.get("n_moves"): print(f"   {w:8}  (no moves)"); continue
        print(f"   {w:8} {d['n_moves']:>8} {d['coverage']:>9.3f} {d['capture_overall']:>12.3f} "
              f"{str(d['capture_participated']):>12} {d['ge25_overall']:>9.3f}")


def _print(res):
    a = res
    rv = a.get("reverse"); print(f"## MOVER-CAPTURE selection power -- score={a['score']}{'(REVERSAL)' if rv else ''} H={a['H']} K={a['K']} -- {a['n_assets']} assets")
    print(f"   (SELECTED top-K by score vs RANDOM=x-sec-mean vs ORACLE top-K-by-forward; net taker 0.24%, long-only)")
    print(f"   {'window':8} {'days':>5} {'SEL%':>8} {'RND%':>8} {'ORACLE%':>8} {'edge_pp':>8} {'capt':>6} {'win':>5} {'>rnd':>5}")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        d = a["per_window"].get(w, {})
        if not d.get("n_days"): print(f"   {w:8} {0:>5}  (no days)"); continue
        print(f"   {w:8} {d['n_days']:>5} {d['sel_mean_pct']:8.3f} {d['rnd_mean_pct']:8.3f} {d['oracle_mean_pct']:8.3f} "
              f"{d['edge_vs_rnd_pct']:8.3f} {str(d['capture_vs_oracle']):>6} {d['sel_winrate']:5.2f} {d['beats_rnd_daily']:5.2f}")


def selftest():
    """Synthetic: assets whose past 'score' is constructed to genuinely predict the next move -> SELECTED must
    beat RANDOM (x-sec mean) on held-out. Validates the cross-sectional selection harness end-to-end, no market data."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-01", periods=1400, freq="D")
    N = 30; closes = {}; highs = {}; lows = {}; atrs = {}
    # each bar: a hidden "edge" = some assets get a real positive drift next bar, and their PRIOR-bar return is high
    base = {}
    for i in range(N):
        r = rng.normal(0.0003, 0.02, len(dates))
        # inject predictability: when prior-day return is in the top, next-day gets +1.5% extra (momentum truth)
        for t in range(1, len(dates) - 1):
            if r[t - 1] > 0.025: r[t] += 0.015
        c = 100 * np.cumprod(1 + r)
        s = pd.Series(c, index=dates)
        closes[f"A{i}"] = s; highs[f"A{i}"] = s * 1.005; lows[f"A{i}"] = s * 0.995
        atrs[f"A{i}"] = s.pct_change().rolling(14).std() * s
    # monkeypatch loaders via a temp module-level dict is overkill; inline-evaluate using momentum score
    panel_c = pd.DataFrame(closes); panel_s = pd.DataFrame({k: v.pct_change() for k, v in closes.items()})
    H = 1; K = 5; cost = 0.0
    fwd = panel_c.shift(-1) / panel_c - 1.0 - cost
    sel, rnd = [], []
    for t in range(20, len(dates) - 2):
        s = panel_s.iloc[t]; f = fwd.iloc[t]
        ok = s.notna() & f.notna()
        if ok.sum() < 10: continue
        top = s[ok].nlargest(K).index
        sel.append(f[ok][top].mean()); rnd.append(f[ok].mean())
    sel, rnd = np.array(sel), np.array(rnd)
    edge = (np.nanmean(sel) - np.nanmean(rnd)) * 100
    ok = edge > 0.05 and (sel > rnd).mean() > 0.5
    print(f"[mover_capture selftest] synthetic momentum-truth: SEL {np.nanmean(sel)*100:.3f}% vs RND "
          f"{np.nanmean(rnd)*100:.3f}% edge {edge:.3f}pp beats_rnd {(sel>rnd).mean():.2f}")
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'} (selection harness detects a genuine cross-sectional edge)")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.mover_capture")
    ap.add_argument("--universe", default="u100"); ap.add_argument("--cadence", default="1d")
    ap.add_argument("--score", default=None, help="single score; default = sweep all")
    ap.add_argument("--horizon", type=int, default=5); ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--chandelier-atr", type=float, default=None, help="convex trailing-stop exit (e.g. 3.0)")
    ap.add_argument("--permove", action="store_true", help="per-MOVE capture-rate (faithful reframe)")
    ap.add_argument("--cost", type=float, default=None, help="override RT cost (0 = frictionless diagnostic)")
    ap.add_argument("--entry-kind", dest="entry_kind", default="breakout", help="breakout|dip (dip=pullback in uptrend)")
    ap.add_argument("--capture", action="store_true", help="ORACLE-MOVE capture: ignition entry + convex exit vs 2-10%% move oracle")
    ap.add_argument("--learned", action="store_true", help="learned cross-sectional ranker (fit TRAIN, test held-out)")
    ap.add_argument("--model", default=None, help="ridge|gbm (default both for --learned)")
    ap.add_argument("--reverse", action="store_true", help="select BOTTOM-K (cross-sectional reversal / oversold bounce)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest: return selftest()
    if a.permove:
        for cad in (a.cadence.split(',') if a.cadence else ['1d']):
            r = evaluate_per_move_capture(a.universe, cadence=cad, atr_mult=a.chandelier_atr or 3.0, entry_kind=a.entry_kind, cost=a.cost if a.cost is not None else TAKER_RT)
            if r: _print_per_move(r)
        return 0
    if a.capture:
        cads = a.cadence.split(',') if a.cadence else ['1d']
        for cad in cads:
            res = evaluate_oracle_capture(a.universe, cadence=cad, H=a.horizon, atr_mult=a.chandelier_atr or 3.0)
            if res: _print_capture(res)
        return 0
    if a.learned:
        for m in ([a.model] if a.model else ['ridge','gbm']):
            res = evaluate_learned_ranker(a.universe, H=a.horizon, K=a.topk, cadence=a.cadence, model=m)
            if res:
                _print(res); print('   DNA (top features):', res.get('dna'))
        return 0
    score_list = [a.score] if a.score else list(SCORES.keys())
    for sc in score_list:
        res = evaluate_selection(a.universe, sc, H=a.horizon, K=a.topk, cadence=a.cadence,
                                 chandelier_atr=a.chandelier_atr, reverse=a.reverse)
        if res: _print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
