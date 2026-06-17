"""src/strat/entry_signal_lab.py -- the ENTRY-SIGNAL x CAPTURE-RATE laboratory.

THE PROBLEM (user, 2026-06-09, de-literalized): "there might be a question of capture rate + entry signal.
Use SOTA trading/quant knowledge." So: given we capture multi-candle UP-MOVES with a FIXED mechanical exit
(a volatility-scaled trailing stop = chandelier) and FIXED size, what causal, past-only ENTRY signal
maximizes (a) held-out CAPTURE-RATE vs the realizable oracle and (b) drawdown-adjusted return vs buy&hold?

THE SOTA SPINE (why this can work despite "direction is unpredictable", AUC~0.51, IC~=0):
  A trailing-stop trend entry is a CONVEXITY harvester -- bounded downside (stop), open-ended upside
  (ride the trend). Managed-futures (Carver/AHL/Winton) monetize payoff convexity, NOT direction
  prediction. So the entry's job is not "predict up" but "select moments whose FORWARD move-size
  distribution has a fat enough right tail to beat whipsaw + cost". And our own market model says
  MAGNITUDE/vol IS predictable (Hurst|ret| 0.80-0.84) even when direction is not -> the strongest entry
  family is TREND CONFIRMED BY VOL-EXPANSION (catch the fat right tail), with the classic MA-stack /
  Donchian breakout as trend baselines. MEAN-REVERSION is excluded by design (dead-list D53
  "continuation dominates"; reversal is the anti-edge; + the user's "stubborn market" intuition).

REUSE (thin layer -- builds NOTHING the harness already has):
  - strat.setup_harness.SetupHarness  : arbitrary past-only bool entry -> policy exit, next-bar fills,
                                        leak-safe trailing stop. THE simulator.
  - strat.setup_harness.ExitPolicy    : the FIXED chandelier exit (atr_trail_mult + atr_col + backstop sl).
  - strat.firewall.random_entry_null  : cost-matched random-entry null (membership_matched = isolate
                                        TRIGGER timing from MOVE selection).
  - strat.benchmark.benchmark_excess  : vs beta-matched costless passive hold (the buy&hold RIGOR GUARD).
  - wealth_bot.harness.WindowSpec     : the 50/20/20/10 train/val/oos/unseen split.

HONEST CONTRACT: fit/rank on TRAIN+VAL only; UNSEEN touched once at verdict. capture-rate uses a
zigzag realizable-oracle (perfect long entries at CONFIRMED swing lows -> swing highs) = an UPPER BOUND,
labelled as such. LONG-ONLY / SPOT / LEV=1 / TAKER 0.24% honest cost. No emoji (cp1252).

Run:
  python src/strat/entry_signal_lab.py --selftest
  python src/strat/entry_signal_lab.py --asset BTCUSDT --cadence 1d --family ma_stack
  python src/strat/entry_signal_lab.py --grid --cadence 1d --kfwd 0   # grid the u10
"""
from __future__ import annotations
import argparse, sys, json, warnings
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)   # cosmetic bool-fillna downcast noise

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from strat.setup_harness import SetupHarness, ExitPolicy            # noqa: E402
from strat.firewall import random_entry_null                       # noqa: E402
from strat.benchmark import benchmark_excess                       # noqa: E402
from wealth_bot.harness import WindowSpec                          # noqa: E402

U10 = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","LTCUSDT"]
# the standard 50/20/20/10 split anchors (UNSEEN reserved -- touched once at verdict)
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-06-01")
TAKER_RT = 0.0024


# ---------------------------------------------------------------------------
# data: chimera -> pandas OHLC + a PAST-ONLY ATR column (for the chandelier exit)
# ---------------------------------------------------------------------------
def load_ohlc(sym: str, cadence: str, atr_len: int = 14, resample_days: int | None = None) -> pd.DataFrame | None:
    from pipeline.chimera_loader import ChimeraLoader
    try:
        loaded = ChimeraLoader().load(sym if sym.endswith("USDT") else sym + "USDT", cadence=cadence)
    except Exception:
        return None
    df = pd.DataFrame(loaded.to_dict(as_series=False)) if (hasattr(loaded, "to_dict") and not hasattr(loaded, "iloc")) else loaded
    df["date"] = (pd.to_datetime(df["date"], unit="ms") if np.issubdtype(df["date"].dtype, np.number)
                  else pd.to_datetime(df["date"]))
    # CHIMERA QUIRK (fix 2026-06-10): sub-daily bars store a DAY-TRUNCATED `date` (all 6 4h bars of a day share
    # 00:00) -> a naive drop_duplicates("date") silently collapses them to 1/day (= daily). Reconstruct a unique
    # intra-day timestamp from the day-date + within-day position (chimera is pre-sorted chronologically), so finer
    # cadences keep all their bars and sort stably. (Was: sort_values+drop_duplicates -> collapsed 4h/1h/15m to 1d.)
    df = df[["date", "open", "high", "low", "close"]].reset_index(drop=True)   # preserve chimera chronological order
    _cad_min = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720}.get(cadence)
    if _cad_min is not None:
        _off = df.groupby("date", sort=False).cumcount()                       # 0,1,.. within each identical day-date
        df["date"] = df["date"] + pd.to_timedelta(_off * _cad_min, unit="m")
    df = df.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)  # unique+monotonic
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    if resample_days and resample_days > 1:   # coarser bars (the user's 3d/7d move axis): O=first H=max L=min C=last
        g = df.set_index("date").resample(f"{resample_days}D")
        df = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                           "low": g["low"].min(), "close": g["close"].last()}).dropna().reset_index()
    # PAST-ONLY ATR: true range uses high, low, prev-close (all known at bar close); rolling mean
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(atr_len, min_periods=atr_len // 2).mean()
    return df


# ---------------------------------------------------------------------------
# ENTRY-SIGNAL LIBRARY -- each returns a past-only boolean np.ndarray (True = setup confirmed at bar close).
# All use only data up to bar t (rolling/ewm/shift). Trend + breakout + momentum families only.
# ---------------------------------------------------------------------------
def _sma(c, n): return c.rolling(n, min_periods=max(2, n // 2)).mean()
def _ema(c, n): return c.ewm(span=n, adjust=False).mean()


def sig_ma_stack(df, long=200, fast=20, mid=50):
    """User's seed: price ABOVE the long-MA (regime up) AND fast-EMA above mid-EMA (shorter MAs confirm)."""
    c = df["close"]
    regime = c > _sma(c, long)
    confirm = _ema(c, fast) > _ema(c, mid)
    return (regime & confirm).fillna(False).to_numpy()


def sig_ma_reclaim(df, long=200, confirm_bars=2):
    """Regime TURN-UP: close crosses back above the long-MA and stays above for confirm_bars (a fresh trend)."""
    c = df["close"]; above = c > _sma(c, long)
    held = above.rolling(confirm_bars, min_periods=confirm_bars).sum() >= confirm_bars
    fresh = held & (~above.shift(confirm_bars).fillna(False))   # was below confirm_bars ago
    return fresh.fillna(False).to_numpy()


def sig_donchian(df, n=20):
    """Turtle/Donchian breakout: close makes a new n-bar high (PRIOR window, shift(1) => past-only)."""
    c = df["close"]
    prior_high = c.rolling(n, min_periods=n).max().shift(1)
    return (c > prior_high).fillna(False).to_numpy()


def sig_momentum(df, k=20):
    """Momentum persistence: k-bar return positive AND accelerating (this bar's ROC > prior bar's ROC)."""
    c = df["close"]; roc = c / c.shift(k) - 1.0
    return ((roc > 0) & (roc > roc.shift(1))).fillna(False).to_numpy()


def sig_vol_expansion_breakout(df, n=20, vol_n=20, exp_q=0.6):
    """SOTA top-prior: a Donchian breakout CONFIRMED by realized-vol EXPANSION (catch the fat right tail).
    breakout = new n-bar high; vol-expansion = current rolling realized vol above its own exp_q quantile
    over the recent window (vol is rising into the move). Both past-only."""
    c = df["close"]
    prior_high = c.rolling(n, min_periods=n).max().shift(1)
    brk = c > prior_high
    ret = c.pct_change()
    rv = ret.rolling(vol_n, min_periods=vol_n // 2).std()
    rv_thr = rv.rolling(vol_n * 3, min_periods=vol_n).quantile(exp_q)
    expanding = (rv > rv_thr) & (rv > rv.shift(1))
    return (brk & expanding).fillna(False).to_numpy()


SIGNALS = {
    "ma_stack": sig_ma_stack, "ma_reclaim": sig_ma_reclaim, "donchian": sig_donchian,
    "momentum": sig_momentum, "vol_exp_breakout": sig_vol_expansion_breakout,
}


# ---------------------------------------------------------------------------
# CAPTURE-RATE oracle: zigzag realizable-perfect long capture (UPPER BOUND, labelled).
# ---------------------------------------------------------------------------
def _zigzag_pivots(p, thr):
    """Confirmed swing pivots via separate running hi/lo (FIXED 2026-06-10: the prior version dragged a single
    `ext` both ways while trend==0, so a reversal never fired -> it silently collapsed to [start, end] = buy&hold.
    Now hi and lo are tracked independently; a thr drop from the running hi confirms a HIGH pivot (trend down),
    a thr rise from the running lo confirms a LOW pivot (trend up); both reset on each confirmed pivot)."""
    n = len(p)
    piv = [(0, p[0])]
    if n < 2: return piv
    hi = lo = p[0]; hi_i = lo_i = 0; trend = 0
    for i in range(1, n):
        if p[i] > hi: hi, hi_i = p[i], i
        if p[i] < lo: lo, lo_i = p[i], i
        if trend >= 0 and p[i] <= hi * (1 - thr):          # reversal down -> the running hi was a swing HIGH
            piv.append((hi_i, hi)); trend = -1; hi, hi_i = p[i], i; lo, lo_i = p[i], i
        elif trend <= 0 and p[i] >= lo * (1 + thr):        # reversal up -> the running lo was a swing LOW
            piv.append((lo_i, lo)); trend = 1; hi, hi_i = p[i], i; lo, lo_i = p[i], i
    piv.append((hi_i, hi) if trend >= 0 else (lo_i, lo))   # final leg's extreme
    return piv


def oracle_compound_pct(close: np.ndarray, thr=0.05) -> float:
    """Realizable perfect long capture: compound of every confirmed swing-low -> next swing-high up-leg
    (costless, perfect entries/exits). An UPPER BOUND on what any long-only entry could capture."""
    if len(close) < 5: return 0.0
    piv = _zigzag_pivots(close, thr)
    comp = 1.0
    for k in range(len(piv) - 1):
        (_, v0), (_, v1) = piv[k], piv[k + 1]
        if v1 > v0:  # an up-leg
            comp *= (v1 / v0)
    return float((comp - 1.0) * 100)


# ---------------------------------------------------------------------------
# FIXED exit (the chandelier) -- held CONSTANT so all performance diff is attributable to the ENTRY.
# ---------------------------------------------------------------------------
def chandelier(atr_mult=3.0, hard_sl=0.10, max_hold=90) -> ExitPolicy:
    """Volatility-scaled trailing stop (chandelier) + a hard initial stop backstop + a long max-hold cap.
    This IS the user's 'by the time you enter you're just waiting for your stop' -- mechanical, fixed."""
    return ExitPolicy(atr_trail_mult=atr_mult, atr_col="atr14", sl_pct=hard_sl, max_hold_bars=max_hold)


# ---------------------------------------------------------------------------
# evaluate ONE (asset, cadence, signal, params) -> per-window compound + capture + buy&hold excess
# ---------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, entry: np.ndarray, policy: ExitPolicy, cost=TAKER_RT,
             zz_thr=0.05) -> dict:
    df = df.copy(); df["entry_sig"] = entry
    h = SetupHarness(df, "entry_sig", policy, WIN, cost_rt=cost)
    res = h.run()
    # capture-rate per window = strategy compound / realizable-oracle compound (coarse, window-level)
    wlab = np.array([h._window_label(pd.Timestamp(t)) for t in df["date"]])
    close = df["close"].to_numpy(float)
    out = {}
    for w in SetupHarness.WINDOWS:
        s = res.window_stats[w]
        idx = np.where(wlab == w)[0]
        orc = oracle_compound_pct(close[idx], zz_thr) if idx.size > 5 else 0.0
        out[w] = {"compound_pct": round(s.compound_pct, 2), "n": s.n_trades,
                  "dd_pct": round(s.max_dd_pct, 2), "win": round(s.win_rate, 3),
                  "oracle_pct": round(orc, 1),
                  "capture": round(s.compound_pct / orc, 3) if orc > 1e-9 else None}
    return {"per_window": out, "all4": res.all_4_positive, "harness": h, "res": res}


def _held_score(ev: dict) -> float:
    """rank key for grid: held-out (OOS+UNSEEN-proxy: use OOS only during search; UNSEEN reserved)."""
    pw = ev["per_window"]
    return (pw["TRAIN"]["compound_pct"] + pw["VAL"]["compound_pct"]) / 2.0


# ---------------------------------------------------------------------------
# GRID SEARCH (fit/rank on TRAIN+VAL; report a ranked table). UNSEEN NOT used here.
# ---------------------------------------------------------------------------
def grid_for_family(family: str):
    if family == "ma_stack":
        return [dict(long=L, fast=f, mid=m) for L in (100, 150, 200) for f in (10, 20) for m in (30, 50) if f < m]
    if family == "ma_reclaim":
        return [dict(long=L, confirm_bars=b) for L in (100, 150, 200) for b in (1, 2, 3)]
    if family == "donchian":
        return [dict(n=n) for n in (10, 20, 30, 55)]
    if family == "momentum":
        return [dict(k=k) for k in (10, 20, 30, 60)]
    if family == "vol_exp_breakout":
        return [dict(n=n, vol_n=v, exp_q=q) for n in (20, 30) for v in (14, 20) for q in (0.5, 0.7)]
    return []


def grid_search(cadence: str, families=None, assets=None, atr_mult=3.0, cost=TAKER_RT, zz_thr=0.05,
                rank="oos_excess"):
    """Rank entry-signal configs across the u10. rank='oos_excess' = the RIGOR-GUARD (median OOS excess
    vs beta-matched buy&hold) -- the honest, beta-robust objective. rank='trainval' = raw compound (beta-
    confounded; kept only for contrast). UNSEEN is NOT touched here."""
    families = families or list(SIGNALS.keys())
    assets = assets or U10
    policy = chandelier(atr_mult=atr_mult)
    rows = []
    for fam in families:
        for params in grid_for_family(fam):
            agg = {"TRAIN": [], "VAL": [], "OOS": [], "capt_OOS": [], "xs_OOS": [], "bb_OOS": [], "n_OOS": 0}
            for sym in assets:
                df = load_ohlc(sym, cadence)
                if df is None or len(df) < 300: continue
                entry = SIGNALS[fam](df, **params)
                if entry.sum() < 5: continue
                ev = evaluate(df, entry, policy, cost=cost, zz_thr=zz_thr)
                pw = ev["per_window"]
                agg["TRAIN"].append(pw["TRAIN"]["compound_pct"]); agg["VAL"].append(pw["VAL"]["compound_pct"])
                agg["OOS"].append(pw["OOS"]["compound_pct"]); agg["n_OOS"] += pw["OOS"]["n"]
                if pw["OOS"]["capture"] is not None: agg["capt_OOS"].append(pw["OOS"]["capture"])
                bm = benchmark_excess(ev["harness"])["per_window"]["OOS"]
                if bm["excess_pp"] is not None:
                    agg["xs_OOS"].append(bm["excess_pp"]); agg["bb_OOS"].append(bool(bm["beats_beta"]))
            if not agg["TRAIN"]: continue
            rows.append({"family": fam, "params": params,
                         "TRAIN_med": float(np.median(agg["TRAIN"])), "VAL_med": float(np.median(agg["VAL"])),
                         "OOS_med": float(np.median(agg["OOS"])),
                         "OOS_xs_med": float(np.median(agg["xs_OOS"])) if agg["xs_OOS"] else None,
                         "OOS_beatsbeta_frac": round(float(np.mean(agg["bb_OOS"])), 2) if agg["bb_OOS"] else None,
                         "OOS_capt_med": float(np.median(agg["capt_OOS"])) if agg["capt_OOS"] else None,
                         "trainval": (float(np.median(agg["TRAIN"])) + float(np.median(agg["VAL"]))) / 2.0,
                         "n_OOS": agg["n_OOS"]})
    if rank == "trainval":
        rows.sort(key=lambda r: -r["trainval"])
    else:  # oos_excess: the rigor-guard -- median OOS excess vs buy&hold
        rows.sort(key=lambda r: -(r["OOS_xs_med"] if r["OOS_xs_med"] is not None else -1e9))
    return rows


# ---------------------------------------------------------------------------
# DEEP honest gate on ONE config: per-asset benchmark-vs-buy&hold + membership-matched firewall.
# This is the RIGOR-GUARD surface (beta-excess + real-timing), not raw compound. OOS is the working
# held-out surface during iteration; UNSEEN is reserved for the single final verdict.
# ---------------------------------------------------------------------------
def deep_eval(family: str, params: dict, cadence: str, assets=None, atr_mult=3.0,
              cost=TAKER_RT, n_books=200, look_unseen=False) -> dict:
    assets = assets or U10
    policy = chandelier(atr_mult=atr_mult)
    held = ["OOS", "UNSEEN"] if look_unseen else ["OOS"]
    per = {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < 300: continue
        entry = SIGNALS[family](df, **params)
        if entry.sum() < 5: continue
        ev = evaluate(df, entry, policy, cost=cost)
        h = ev["harness"]
        bm = benchmark_excess(h)
        fw = random_entry_null(h, n_books=n_books, seed=7, membership_matched=True)
        pw = ev["per_window"]
        per[sym] = {
            "OOS_comp": pw["OOS"]["compound_pct"], "OOS_capt": pw["OOS"]["capture"],
            "OOS_excess_pp": bm["per_window"]["OOS"]["excess_pp"],
            "OOS_buyhold": bm["per_window"]["OOS"]["buyhold_pct"],
            "OOS_beats_beta": bm["per_window"]["OOS"]["beats_beta"],
            "OOS_beats_null": fw["per_window"]["OOS"].get("beats_null"),
            "UNSEEN_comp": pw["UNSEEN"]["compound_pct"], "UNSEEN_capt": pw["UNSEEN"]["capture"],
            "UNSEEN_excess_pp": bm["per_window"]["UNSEEN"]["excess_pp"],
            "UNSEEN_beats_beta": bm["per_window"]["UNSEEN"]["beats_beta"],
            "UNSEEN_beats_null": fw["per_window"]["UNSEEN"].get("beats_null"),
        }
    n = len(per)
    def _frac(key):
        vals = [v[key] for v in per.values() if v.get(key) is not None]
        return round(sum(bool(x) for x in vals) / len(vals), 2) if vals else None
    def _med(key):
        vals = [v[key] for v in per.values() if v.get(key) is not None]
        return round(float(np.median(vals)), 2) if vals else None
    agg = {
        "family": family, "params": params, "cadence": cadence, "n_assets": n,
        "OOS_beats_beta_frac": _frac("OOS_beats_beta"), "OOS_beats_null_frac": _frac("OOS_beats_null"),
        "OOS_excess_med_pp": _med("OOS_excess_pp"), "OOS_capt_med": _med("OOS_capt"),
    }
    if look_unseen:
        agg.update({"UNSEEN_beats_beta_frac": _frac("UNSEEN_beats_beta"),
                    "UNSEEN_beats_null_frac": _frac("UNSEEN_beats_null"),
                    "UNSEEN_excess_med_pp": _med("UNSEEN_excess_pp"), "UNSEEN_capt_med": _med("UNSEEN_capt")})
    return {"per_asset": per, "agg": agg}


def _print_deep(d: dict, look_unseen=False):
    a = d["agg"]
    print(f"## DEEP GATE -- {a['family']} {a['params']} -- {a['cadence']} -- n_assets={a['n_assets']}")
    print(f"   {'asset':9} {'OOS_cmp%':>8} {'OOS_BH%':>8} {'OOS_xs_pp':>9} {'>beta':>6} {'>null':>6} {'OOS_capt':>8}")
    for sym, v in d["per_asset"].items():
        print(f"   {sym:9} {v['OOS_comp']:8.1f} {v['OOS_buyhold']:8.1f} {str(v['OOS_excess_pp']):>9} "
              f"{str(v['OOS_beats_beta']):>6} {str(v['OOS_beats_null']):>6} {str(v['OOS_capt']):>8}")
    print(f"   AGG: beats_beta(OOS)={a['OOS_beats_beta_frac']}  beats_null(OOS)={a['OOS_beats_null_frac']}  "
          f"excess_med={a['OOS_excess_med_pp']}pp  capt_med={a['OOS_capt_med']}")
    if look_unseen:
        print(f"   UNSEEN(verdict): beats_beta={a['UNSEEN_beats_beta_frac']}  beats_null={a['UNSEEN_beats_null_frac']}  "
              f"excess_med={a['UNSEEN_excess_med_pp']}pp  capt_med={a['UNSEEN_capt_med']}")


# ---------------------------------------------------------------------------
# CANONICAL REGIME-TIMING rule (Faber/GTAA): the verifiable form of the user's "price>long-MA" idea.
# LONG when close>SMA(long), FLAT when below. Entry=cross-above; exit=regime-end (close<MA) + chandelier
# backstop. Evaluated vs buy&hold on COMPOUND and DRAWDOWN (the user's real bar) -- this trades enough to
# verdict honestly on UNSEEN (unlike the rare ma_reclaim).
# ---------------------------------------------------------------------------
def _maxdd_pct(close: np.ndarray) -> float:
    if len(close) < 2: return 0.0
    eq = close / close[0]; peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def evaluate_regime(df: pd.DataFrame, long: int, atr_mult=3.0, cost=TAKER_RT, zz_thr=0.05) -> dict:
    df = df.copy(); c = df["close"]; sma = _sma(c, long)
    above = (c > sma)
    df["entry_sig"] = (above & ~above.shift(1).fillna(False)).fillna(False).to_numpy()
    df["regime_exit"] = (~above).fillna(True).astype(bool).to_numpy()
    policy = ExitPolicy(exit_signal_col="regime_exit", atr_trail_mult=atr_mult, atr_col="atr14")
    h = SetupHarness(df, "entry_sig", policy, WIN, cost_rt=cost)
    res = h.run()
    wlab = np.array([h._window_label(pd.Timestamp(t)) for t in df["date"]])
    close = c.to_numpy(float)
    bm = benchmark_excess(h)["per_window"]
    out = {}
    for w in SetupHarness.WINDOWS:
        s = res.window_stats[w]; idx = np.where(wlab == w)[0]
        orc = oracle_compound_pct(close[idx], zz_thr) if idx.size > 5 else 0.0
        out[w] = {"comp": round(s.compound_pct, 2), "n": s.n_trades, "dd": round(s.max_dd_pct, 2),
                  "bh": bm[w]["buyhold_pct"], "bh_dd": round(_maxdd_pct(close[idx]), 2) if idx.size > 1 else None,
                  "xs": bm[w]["excess_pp"], "beats_beta": bm[w]["beats_beta"],
                  "capt": round(s.compound_pct / orc, 3) if orc > 1e-9 else None}
    # FULL-CYCLE (all windows chained): the decision-relevant terminal-wealth + drawdown comparison.
    # Strategy equity = chained net_pnl of every trade; an equity curve sampled per-trade. Buy&hold = full close path.
    rets = np.array([t["net_pnl"] for t in res.trades]) if res.trades else np.array([0.0])
    eq = np.cumprod(1.0 + rets); strat_comp = float((eq[-1] - 1.0) * 100)
    peak = np.maximum.accumulate(eq); strat_dd = float(((eq - peak) / peak).min() * 100)
    bh_comp = float((close[-1] / close[0] - 1.0) * 100)
    full = {"strat_comp": round(strat_comp, 1), "strat_dd": round(strat_dd, 1),
            "bh_comp": round(bh_comp, 1), "bh_dd": round(_maxdd_pct(close), 1), "n_trades": len(res.trades),
            "strat_calmar": round(strat_comp / abs(strat_dd), 2) if strat_dd < -1e-6 else None,
            "bh_calmar": round(bh_comp / abs(_maxdd_pct(close)), 2) if _maxdd_pct(close) < -1e-6 else None}
    return {"per_window": out, "harness": h, "full_cycle": full}


def run_regime(long: int, resample_days=None, assets=None, atr_mult=3.0, cadence="1d"):
    assets = assets or U10
    per = {}; full = {}
    for sym in assets:
        df = load_ohlc(sym, cadence, resample_days=resample_days)
        if df is None or len(df) < max(60, long + 20): continue
        ev = evaluate_regime(df, long, atr_mult=atr_mult)
        per[sym] = ev["per_window"]; full[sym] = ev["full_cycle"]
    def agg(win, key):
        return [v[win][key] for v in per.values() if v[win].get(key) is not None]
    tag = f"{resample_days}D-bars" if resample_days else cadence
    print(f"## REGIME-TIMING (Faber/GTAA: long>SMA{long}, flat below) -- u10 -- {tag} -- chandelier backstop atr={atr_mult}")
    # FULL-CYCLE headline (the decision-relevant terminal-wealth + drawdown comparison over ALL history)
    print(f"   -- FULL CYCLE (all history): strat vs buy&hold terminal wealth + max drawdown + Calmar --")
    print(f"   {'asset':9} {'STRAT%':>9} {'S_dd%':>7} {'S_calmar':>8} {'BH%':>9} {'BH_dd%':>7} {'BH_calmar':>9} {'nT':>4}")
    for sym, fc in full.items():
        print(f"   {sym:9} {fc['strat_comp']:9.0f} {fc['strat_dd']:7.1f} {str(fc['strat_calmar']):>8} "
              f"{fc['bh_comp']:9.0f} {fc['bh_dd']:7.1f} {str(fc['bh_calmar']):>9} {fc['n_trades']:>4}")
    sc = [f["strat_comp"] for f in full.values()]; sd = [f["strat_dd"] for f in full.values()]
    bc = [f["bh_comp"] for f in full.values()]; bd = [f["bh_dd"] for f in full.values()]
    scal = [f["strat_calmar"] for f in full.values() if f["strat_calmar"] is not None]
    bcal = [f["bh_calmar"] for f in full.values() if f["bh_calmar"] is not None]
    wins = sum(s > b for s, b in zip(sc, bc))
    cwins = sum(s > b for s, b in zip(scal, bcal)) if scal and bcal else None
    print(f"   AGG FULL: strat_comp_med={round(float(np.median(sc)),0)}%  bh_comp_med={round(float(np.median(bc)),0)}%  "
          f"strat_dd_med={round(float(np.median(sd)),1)}%  bh_dd_med={round(float(np.median(bd)),1)}%")
    print(f"   AGG FULL: strat_calmar_med={round(float(np.median(scal)),2) if scal else None}  "
          f"bh_calmar_med={round(float(np.median(bcal)),2) if bcal else None}  "
          f"compound_wins={wins}/{len(sc)}  calmar_wins={cwins}/{len(scal) if scal else 0}")
    print(f"   -- held-out windows (OOS working surface | UNSEEN verdict) --")
    print(f"   {'asset':9} {'OOS_cmp%':>8} {'OOS_dd%':>7} {'OOS_BH%':>8} {'OOS_xs':>7} {'>b':>3}  ||  "
          f"{'UNS_cmp%':>8} {'UNS_BH%':>8} {'UNS_xs':>7} {'>b':>3}")
    for sym, v in per.items():
        o, u = v["OOS"], v["UNSEEN"]
        print(f"   {sym:9} {o['comp']:8.1f} {o['dd']:7.1f} {o['bh']:8.1f} {str(o['xs']):>7} "
              f"{str(o['beats_beta'])[0]:>3}  ||  {u['comp']:8.1f} {u['bh']:8.1f} {str(u['xs']):>7} {str(u['beats_beta'])[0]:>3}")
    for win in ("OOS", "UNSEEN"):
        bb = agg(win, "beats_beta"); xs = agg(win, "xs"); comp = agg(win, "comp"); dd = agg(win, "dd")
        fb = round(np.mean([bool(x) for x in bb]), 2) if bb else None
        print(f"   AGG {win:6}: beats_beta={fb}  excess_med={round(float(np.median(xs)),2) if xs else None}pp  "
              f"comp_med={round(float(np.median(comp)),1) if comp else None}%  strat_dd_med={round(float(np.median(dd)),1) if dd else None}%")
    return {"per_window": per, "full_cycle": full}


# ---------------------------------------------------------------------------
# CYCLE-2 FRONTIER: REGIME-ADAPTIVE exit. Loosen the trail when trend-strength is high (let winners run),
# tighten in chop (protect). Trend-strength = Kaufman Efficiency Ratio (causal). The decompose-the-ideal
# answer to the fixed-exit dilemma (loose-in-trend vs tight-in-chop, not knowable ahead). Built leak-safe by
# baking the per-bar width into a past-only 'adaptive_atr' column = atr * width_factor, with atr_trail_mult=1.
# ---------------------------------------------------------------------------
def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman ER in [0,1]: |net move over n| / sum(|bar moves| over n). 1=pure trend, 0=pure chop. Past-only."""
    net = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n, min_periods=n // 2).sum()
    return (net / vol.replace(0, np.nan)).clip(0, 1)


def evaluate_regime_adaptive(df: pd.DataFrame, long: int, er_n=20, tight_mult=2.0, loose_mult=14.0,
                             cost=TAKER_RT, zz_thr=0.05) -> dict:
    """Same regime entry/exit as evaluate_regime, but the chandelier trail WIDTH adapts to the Efficiency
    Ratio: width_factor = tight + (loose-tight)*ER -> wide trail in trends, tight in chop. adaptive_atr =
    atr14 * width_factor (past-only), used with atr_trail_mult=1.0 so the effective trail is the adaptive one."""
    df = df.copy(); c = df["close"]; sma = _sma(c, long)
    above = (c > sma)
    df["entry_sig"] = (above & ~above.shift(1).fillna(False)).fillna(False).to_numpy()
    df["regime_exit"] = (~above).fillna(True).astype(bool).to_numpy()
    er = efficiency_ratio(c, er_n)
    width = (tight_mult + (loose_mult - tight_mult) * er).fillna(tight_mult)
    df["adaptive_atr"] = (df["atr14"] * width).to_numpy()
    policy = ExitPolicy(exit_signal_col="regime_exit", atr_trail_mult=1.0, atr_col="adaptive_atr")
    h = SetupHarness(df, "entry_sig", policy, WIN, cost_rt=cost)
    res = h.run()
    wlab = np.array([h._window_label(pd.Timestamp(t)) for t in df["date"]])
    close = c.to_numpy(float)
    bm = benchmark_excess(h)["per_window"]
    out = {}
    for w in SetupHarness.WINDOWS:
        s = res.window_stats[w]; idx = np.where(wlab == w)[0]
        out[w] = {"comp": round(s.compound_pct, 2), "n": s.n_trades, "dd": round(s.max_dd_pct, 2),
                  "bh": bm[w]["buyhold_pct"], "xs": bm[w]["excess_pp"], "beats_beta": bm[w]["beats_beta"]}
    rets = np.array([t["net_pnl"] for t in res.trades]) if res.trades else np.array([0.0])
    eq = np.cumprod(1.0 + rets); strat_comp = float((eq[-1] - 1.0) * 100)
    peak = np.maximum.accumulate(eq); strat_dd = float(((eq - peak) / peak).min() * 100)
    full = {"strat_comp": round(strat_comp, 1), "strat_dd": round(strat_dd, 1),
            "strat_calmar": round(strat_comp / abs(strat_dd), 2) if strat_dd < -1e-6 else None}
    return {"per_window": out, "full_cycle": full, "harness": h}


def run_adaptive_compare(long=150, er_n=20, tight=2.0, loose=14.0, assets=None, cadence="1d"):
    """Side-by-side: fixed-tight (atr=3) vs fixed-loose (atr=15) vs ADAPTIVE(ER) regime exit. The test:
    does the adaptive exit get the chop-protection of tight AND the trend-capture of loose -> beat BOTH
    on the held-out OOS? If yes, the regime is causally detectable enough to resolve the exit dilemma."""
    assets = assets or U10
    rows = []
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + 40: continue
        t = evaluate_regime(df, long, atr_mult=3.0); l = evaluate_regime(df, long, atr_mult=15.0)
        a = evaluate_regime_adaptive(df, long, er_n=er_n, tight_mult=tight, loose_mult=loose)
        rows.append({"sym": sym,
                     "t_full": t["full_cycle"]["strat_comp"], "t_oos_xs": t["per_window"]["OOS"]["xs"], "t_oos_bb": t["per_window"]["OOS"]["beats_beta"],
                     "l_full": l["full_cycle"]["strat_comp"], "l_oos_xs": l["per_window"]["OOS"]["xs"], "l_oos_bb": l["per_window"]["OOS"]["beats_beta"],
                     "a_full": a["full_cycle"]["strat_comp"], "a_oos_xs": a["per_window"]["OOS"]["xs"], "a_oos_bb": a["per_window"]["OOS"]["beats_beta"],
                     "a_cal": a["full_cycle"]["strat_calmar"], "a_oos_cmp": a["per_window"]["OOS"]["comp"],
                     "bh_oos": t["per_window"]["OOS"]["bh"]})
    print(f"## REGIME-ADAPTIVE EXIT compare -- u10 -- {cadence} -- SMA{long} regime, ER{er_n} trail in [{tight},{loose}]xATR")
    print(f"   {'asset':9} | {'TIGHT full%':>11} {'oosXS':>6} | {'LOOSE full%':>11} {'oosXS':>6} | {'ADAPT full%':>11} {'oosXS':>6} {'oosCmp%':>7} {'>b':>3}")
    for r in rows:
        print(f"   {r['sym']:9} | {r['t_full']:11.0f} {str(r['t_oos_xs']):>6} | {r['l_full']:11.0f} {str(r['l_oos_xs']):>6} | "
              f"{r['a_full']:11.0f} {str(r['a_oos_xs']):>6} {r['a_oos_cmp']:7.1f} {str(r['a_oos_bb'])[0]:>3}")
    def med(k):
        v = [r[k] for r in rows if r[k] is not None]; return round(float(np.median(v)), 1) if v else None
    def bbfrac(k):
        v = [r[k] for r in rows if r[k] is not None]; return round(np.mean([bool(x) for x in v]), 2) if v else None
    print(f"   MED full%:   tight={med('t_full')}  loose={med('l_full')}  adaptive={med('a_full')}")
    print(f"   MED OOS_xs:  tight={med('t_oos_xs')}  loose={med('l_oos_xs')}  adaptive={med('a_oos_xs')}")
    print(f"   OOS beats_beta frac:  tight={bbfrac('t_oos_bb')}  loose={bbfrac('l_oos_bb')}  adaptive={bbfrac('a_oos_bb')}")
    return rows


# ---------------------------------------------------------------------------
# CYCLE-3: REGIME-EXIT DNA. The cycle showed the EXIT is the lever and the failure mode is WHIPSAW (exiting
# on a brief shakeout below the MA, then missing the recovery). Question: at the cross-below-MA bar, is there
# a CAUSAL feature that separates a REAL decline (good exit) from a whipsaw (bad exit)? If yes -> a conditional
# exit beats both always-exit and buy&hold. Sample-rich (every cross-below event across u10 x years).
# ---------------------------------------------------------------------------
def regime_exit_dna(long=150, er_n=20, fwd_k=30, cadence="1d", assets=None) -> dict:
    assets = assets or U10
    rows = []
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + fwd_k + 40: continue
        c = df["close"]; close = c.to_numpy(float); n = len(close)
        sma = _sma(c, long).to_numpy(); er = efficiency_ratio(c, er_n).to_numpy()
        slope = (sma / np.roll(sma, 20) - 1.0)                       # prior MA slope (past-only)
        ret20 = (close / np.roll(close, 20) - 1.0)
        vol20 = pd.Series(close).pct_change().rolling(20, min_periods=10).std().to_numpy()
        hi60 = pd.Series(close).rolling(60, min_periods=30).max().to_numpy()
        dist_hi = close / hi60 - 1.0
        above = close > sma
        for t in range(max(long, 60), n - fwd_k - 1):
            if above[t - 1] and not above[t]:                        # cross BELOW the MA at t = an exit event
                fwd = close[t:t + fwd_k + 1]
                fwd_min = fwd.min(); fwd_end = fwd[-1]
                # exit_quality: how much the exit SAVED = -(worst forward drawdown captured by being out).
                # good exit = price kept falling (we avoided it); bad exit (whipsaw) = it recovered.
                exit_quality = -(fwd_min / close[t] - 1.0) * 100      # high = price fell a lot after exit = good exit
                whipsaw = (fwd_end / close[t] - 1.0) > 0.03            # ended >3% ABOVE exit = whipsaw (bad)
                rows.append({"sym": sym, "t": t, "exit_quality": exit_quality, "whipsaw": bool(whipsaw),
                             "er": er[t], "dist_below_ma": (close[t] / sma[t] - 1.0) * 100,
                             "ma_slope": slope[t] * 100, "ret20": ret20[t] * 100,
                             "vol20": vol20[t] * 100, "dist_from_high": dist_hi[t] * 100})
    if not rows:
        print("no exit events"); return {}
    d = pd.DataFrame(rows)
    # label good (top tercile exit_quality = real declines) vs bad (bottom tercile = whipsaws/shallow)
    q_hi, q_lo = d["exit_quality"].quantile(0.66), d["exit_quality"].quantile(0.34)
    good = d[d["exit_quality"] >= q_hi]; bad = d[d["exit_quality"] <= q_lo]
    feats = ["er", "dist_below_ma", "ma_slope", "ret20", "vol20", "dist_from_high"]
    print(f"## REGIME-EXIT DNA -- u10 -- {cadence} -- SMA{long} -- {len(d)} exit events  (fwd {fwd_k} bars)")
    print(f"   whipsaw rate (exit then recover >3%): {d['whipsaw'].mean():.1%}  "
          f"median exit_quality(=fwd drawdown avoided): {d['exit_quality'].median():.1f}%")
    print(f"   feature separation -- GOOD exits (real declines) vs BAD (whipsaws), median + normalized gap:")
    print(f"   {'feature':16} {'GOOD_med':>9} {'BAD_med':>9} {'norm_gap':>9}")
    sep = {}
    for f in feats:
        gm, bm = good[f].median(), bad[f].median()
        pooled = d[f].std() + 1e-9
        gap = (gm - bm) / pooled                                     # standardized separation (effect size)
        sep[f] = gap
        print(f"   {f:16} {gm:9.2f} {bm:9.2f} {gap:9.2f}")
    best = max(sep, key=lambda k: abs(sep[k]))
    print(f"   strongest separator: {best} (norm_gap={sep[best]:.2f})  "
          f"-- |gap|>~0.5 = a usable conditioner; <0.3 = weak/none")
    return {"df": d, "sep": sep, "best": best}


# ---------------------------------------------------------------------------
# CYCLE-3b: CONDITIONAL exit built from the DNA. Suppress low-conviction (whipsaw) exits: only step out on a
# cross-below-MA when the break looks like a REAL decline (high vol OR steep prior MA-slope), else HOLD THROUGH
# (with a chandelier backstop so a true crash still stops you out). Test: does it beat always-exit + buy&hold OOS?
# All conditions are past-only. Thresholds = rolling-median (vol) and >0 (slope) -- no fit, no per-window peek.
# ---------------------------------------------------------------------------
def evaluate_regime_conditional(df: pd.DataFrame, long: int, er_n=20, backstop_atr=8.0,
                                cost=TAKER_RT) -> dict:
    df = df.copy(); c = df["close"]; sma = _sma(c, long)
    above = (c > sma)
    vol20 = c.pct_change().rolling(20, min_periods=10).std()
    vol_med = vol20.rolling(150, min_periods=60).median()           # past-only rolling median vol
    ma_slope = sma / sma.shift(20) - 1.0
    real_decline = (vol20 > vol_med) | (ma_slope < -0.0)            # high-vol break OR MA already rolling over
    df["entry_sig"] = (above & ~above.shift(1).fillna(False)).fillna(False).to_numpy()
    # exit only on a CONFIRMED real-decline break below the MA; otherwise hold (chandelier backstop catches crashes)
    df["regime_exit"] = ((~above) & real_decline).fillna(True).astype(bool).to_numpy()
    policy = ExitPolicy(exit_signal_col="regime_exit", atr_trail_mult=backstop_atr, atr_col="atr14")
    h = SetupHarness(df, "entry_sig", policy, WIN, cost_rt=cost)
    res = h.run()
    wlab = np.array([h._window_label(pd.Timestamp(t)) for t in df["date"]])
    close = c.to_numpy(float); bm = benchmark_excess(h)["per_window"]
    out = {}
    for w in SetupHarness.WINDOWS:
        s = res.window_stats[w]
        out[w] = {"comp": round(s.compound_pct, 2), "n": s.n_trades, "dd": round(s.max_dd_pct, 2),
                  "bh": bm[w]["buyhold_pct"], "xs": bm[w]["excess_pp"], "beats_beta": bm[w]["beats_beta"]}
    rets = np.array([t["net_pnl"] for t in res.trades]) if res.trades else np.array([0.0])
    eq = np.cumprod(1.0 + rets); sc = float((eq[-1] - 1.0) * 100)
    peak = np.maximum.accumulate(eq); sd = float(((eq - peak) / peak).min() * 100)
    return {"per_window": out, "full_cycle": {"strat_comp": round(sc, 1), "strat_dd": round(sd, 1),
            "strat_calmar": round(sc / abs(sd), 2) if sd < -1e-6 else None}}


def run_conditional_compare(long=150, er_n=20, backstop=8.0, assets=None, cadence="1d"):
    assets = assets or U10
    rows = []
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + 60: continue
        t = evaluate_regime(df, long, atr_mult=3.0); l = evaluate_regime(df, long, atr_mult=15.0)
        cnd = evaluate_regime_conditional(df, long, er_n=er_n, backstop_atr=backstop)
        rows.append({"sym": sym, "bh_full": round(t["full_cycle"]["bh_comp"], 0),
                     "t_full": t["full_cycle"]["strat_comp"], "t_xs": t["per_window"]["OOS"]["xs"], "t_bb": t["per_window"]["OOS"]["beats_beta"],
                     "l_full": l["full_cycle"]["strat_comp"], "l_xs": l["per_window"]["OOS"]["xs"], "l_bb": l["per_window"]["OOS"]["beats_beta"],
                     "c_full": cnd["full_cycle"]["strat_comp"], "c_dd": cnd["full_cycle"]["strat_dd"], "c_cal": cnd["full_cycle"]["strat_calmar"],
                     "c_xs": cnd["per_window"]["OOS"]["xs"], "c_bb": cnd["per_window"]["OOS"]["beats_beta"], "c_cmp": cnd["per_window"]["OOS"]["comp"],
                     "bh_cal": t["full_cycle"]["bh_calmar"]})
    print(f"## CONDITIONAL EXIT (DNA-built: hold through low-conviction breaks) -- u10 -- {cadence} -- SMA{long}, backstop atr={backstop}")
    print(f"   {'asset':9} | {'BH full%':>9} {'BHcal':>6} | {'COND full%':>10} {'Ccal':>6} {'C_dd%':>7} {'oosXS':>6} {'>b':>3}")
    for r in rows:
        print(f"   {r['sym']:9} | {r['bh_full']:9.0f} {str(r['bh_cal']):>6} | {r['c_full']:10.0f} {str(r['c_cal']):>6} "
              f"{r['c_dd']:7.1f} {str(r['c_xs']):>6} {str(r['c_bb'])[0]:>3}")
    def med(k):
        v = [r[k] for r in rows if r[k] is not None]; return round(float(np.median(v)), 1) if v else None
    def frac(k):
        v = [r[k] for r in rows if r[k] is not None]; return round(np.mean([bool(x) for x in v]), 2) if v else None
    cwin = sum(r["c_full"] > r["bh_full"] for r in rows)
    ccalwin = sum((r["c_cal"] or -9) > (r["bh_cal"] or 9) for r in rows)
    print(f"   MED full%: BH={med('bh_full')}  tight={med('t_full')}  loose={med('l_full')}  COND={med('c_full')}")
    print(f"   MED OOS_xs: tight={med('t_xs')}  loose={med('l_xs')}  COND={med('c_xs')}")
    print(f"   OOS beats_beta: tight={frac('t_bb')}  loose={frac('l_bb')}  COND={frac('c_bb')}   "
          f"|| COND full-cycle: compound_wins_vs_BH={cwin}/{len(rows)}  calmar_wins_vs_BH={ccalwin}/{len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# CYCLE-4: RICH chimera-feature exit DNA. The user's "decompose the surrounding chimera info" applied to the
# EXIT decision (the one place real signal appeared). Decompose good-vs-whipsaw exits against the crypto-native
# families (liquidation, funding, basis, whale, vol-state, OI). If one beats the computed vol/slope separators
# (|gap|>~0.6) it is a stronger conditioner. All chimera features are computed at bar close = past-only at the event.
# ---------------------------------------------------------------------------
RICH_FEATS = ["liq_capitulation", "liq_delta_z30", "liq_short_panic", "liq_long_z30",
              "fund_rate_z30", "fund_sign_flip", "fund_extreme_long_count",
              "bs_basis_panic", "bs_basis_bear_shock", "bs_basis_z30",
              "wh_whale_net_usd", "wh_whale_sell_usd", "norm_yz_volatility", "norm_vol_ratio",
              "norm_perm_entropy", "norm_oi_price_divergence", "norm_efficiency"]


def regime_exit_dna_rich(long=150, fwd_k=30, cadence="1d", assets=None) -> dict:
    from pipeline.chimera_loader import ChimeraLoader
    assets = assets or U10
    rows = []
    for sym in assets:
        try:
            loaded = ChimeraLoader().load(sym, cadence=cadence)
            cf = pd.DataFrame(loaded.to_dict(as_series=False)) if (hasattr(loaded, "to_dict") and not hasattr(loaded, "iloc")) else loaded
        except Exception:
            continue
        cf["date"] = (pd.to_datetime(cf["date"], unit="ms") if np.issubdtype(cf["date"].dtype, np.number) else pd.to_datetime(cf["date"]))
        cf = cf.sort_values("date").reset_index(drop=True)
        close = cf["close"].to_numpy(float); n = len(close)
        if n < long + fwd_k + 40: continue
        sma = _sma(cf["close"], long).to_numpy()
        above = close > sma
        avail = [f for f in RICH_FEATS if f in cf.columns]
        comp_vol = cf["close"].pct_change().rolling(20, min_periods=10).std().to_numpy()
        comp_slope = (sma / np.roll(sma, 20) - 1.0)
        for t in range(max(long, 60), n - fwd_k - 1):
            if above[t - 1] and not above[t]:
                fwd = close[t:t + fwd_k + 1]
                rec = {"sym": sym, "exit_quality": -(fwd.min() / close[t] - 1.0) * 100,
                       "comp_vol20": comp_vol[t], "comp_ma_slope": comp_slope[t] * 100}
                for f in avail:
                    v = cf[f].iloc[t]
                    rec[f] = float(v) if pd.notna(v) else np.nan
                rows.append(rec)
    if not rows:
        print("no exit events / no chimera"); return {}
    d = pd.DataFrame(rows)
    q_hi, q_lo = d["exit_quality"].quantile(0.66), d["exit_quality"].quantile(0.34)
    good = d[d["exit_quality"] >= q_hi]; bad = d[d["exit_quality"] <= q_lo]
    feats = ["comp_vol20", "comp_ma_slope"] + [f for f in RICH_FEATS if f in d.columns]
    seps = []
    for f in feats:
        g = good[f].dropna(); b = bad[f].dropna()
        if len(g) < 20 or len(b) < 20: continue
        pooled = d[f].std() + 1e-9
        gap = (g.median() - b.median()) / pooled
        seps.append((f, gap, g.median(), b.median(), int(d[f].notna().sum())))
    seps.sort(key=lambda x: -abs(x[1]))
    print(f"## RICH EXIT DNA (chimera) -- u10 -- {cadence} -- SMA{long} -- {len(d)} exit events")
    print(f"   ranked causal separators of GOOD(real-decline) vs BAD(whipsaw) exits:")
    print(f"   {'feature':24} {'norm_gap':>9} {'GOOD_med':>10} {'BAD_med':>10} {'n':>6}")
    for f, gap, gm, bm, nn in seps[:14]:
        print(f"   {f:24} {gap:9.2f} {gm:10.3f} {bm:10.3f} {nn:>6}")
    strong = [s for s in seps if abs(s[1]) >= 0.6 and not s[0].startswith("comp_")]
    print(f"   crypto-native features beating the computed vol/slope baseline (|gap|>=0.6): "
          f"{[s[0] for s in strong] if strong else 'NONE -- computed vol/slope remain the best separators'}")
    return {"df": d, "seps": seps}


# ---------------------------------------------------------------------------
# CYCLE-5: PORTFOLIO lens. Trend-following is a PORTFOLIO strategy (Carver/AHL) -- the diversification across
# trend signals + bounded drawdown is where managed-futures earns its Calmar, NOT per-asset. Equal-weight u10
# book, each asset regime-timed (DNA-conditional exit), flat=cash, vs an equal-weight buy&hold basket. The
# correct unit of evaluation. Position applied NEXT bar (shift) -- no look-ahead; taker cost on each flip.
# ---------------------------------------------------------------------------
def _regime_position(close: np.ndarray, sma: np.ndarray, vol20: np.ndarray, vol_med: np.ndarray,
                     slope: np.ndarray, conditional: bool) -> np.ndarray:
    """Stateful long/flat regime position: ON at cross-above SMA; OFF at cross-below (conditional: only on a
    confirmed real-decline break = high-vol OR MA rolling over; else hold through). Returns 0/1 per bar (the
    decision AT bar close; the caller shifts it +1 for the next-bar fill)."""
    n = len(close); pos = np.zeros(n); on = False
    for t in range(1, n):
        if not on and close[t] > sma[t] and close[t - 1] <= sma[t - 1]:
            on = True
        elif on and close[t] < sma[t]:
            real = (vol20[t] > vol_med[t]) or (slope[t] < 0) if conditional else True
            if real: on = False
        pos[t] = 1.0 if on else 0.0
    return pos


def portfolio_backtest(long=150, conditional=True, cost=TAKER_RT, cadence="1d", assets=None,
                       vol_filter=False, vol_q=0.90) -> dict:
    assets = assets or U10
    # build a wide date-aligned close panel
    series = {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + 60: continue
        series[sym] = df.set_index("date")["close"]
    if not series: print("no data"); return {}
    panel = pd.DataFrame(series).sort_index()
    dates = panel.index
    strat_ret = pd.DataFrame(index=dates); bh_ret = pd.DataFrame(index=dates)
    for sym in panel.columns:
        c = panel[sym].dropna()
        close = c.to_numpy(float)
        sma = c.rolling(long, min_periods=long // 2).mean().to_numpy()
        vol20 = c.pct_change().rolling(20, min_periods=10).std().to_numpy()
        vol_med = pd.Series(vol20).rolling(150, min_periods=60).median().to_numpy()
        slope = sma / np.roll(sma, 20) - 1.0
        pos = _regime_position(close, sma, vol20, vol_med, slope, conditional)
        if vol_filter:   # ADJACENT TEST: risk-off when realized vol is in a spike regime (vol > rolling q-pctile)
            vthr = pd.Series(vol20).rolling(250, min_periods=120).quantile(vol_q).to_numpy()
            spike = np.where(np.isfinite(vthr), vol20 > vthr, False)
            pos = pos * (~spike)
        r = np.zeros(len(close)); r[1:] = close[1:] / close[:-1] - 1.0
        pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0                       # NEXT-bar fill (no look-ahead)
        flips = np.abs(np.diff(np.concatenate([[0.0], pos_lag])))         # position changes
        sret = pos_lag * r - flips * cost
        strat_ret[sym] = pd.Series(sret, index=c.index)
        bh_ret[sym] = pd.Series(r, index=c.index)
    # equal-weight portfolio: mean across available assets per bar (flat asset contributes 0 = cash drag)
    port_s = strat_ret.mean(axis=1, skipna=True).fillna(0.0)
    port_b = bh_ret.mean(axis=1, skipna=True).fillna(0.0)

    def stats(ret, lo=None, hi=None):
        r = ret.copy()
        if lo is not None: r = r[(r.index >= lo) & (r.index < hi)]
        eq = (1.0 + r).cumprod()
        if len(eq) == 0: return {"comp": 0.0, "dd": 0.0, "calmar": None}
        comp = float((eq.iloc[-1] - 1.0) * 100)
        peak = eq.cummax(); dd = float(((eq - peak) / peak).min() * 100)
        return {"comp": round(comp, 1), "dd": round(dd, 1), "calmar": round(comp / abs(dd), 2) if dd < -1e-6 else None}

    bounds = {"FULL": (None, None),
              "OOS": (pd.Timestamp(WIN.val_end), pd.Timestamp(WIN.oos_end)),
              "UNSEEN": (pd.Timestamp(WIN.oos_end), pd.Timestamp(WIN.unseen_end))}
    print(f"## PORTFOLIO regime-managed beta (equal-weight {len(panel.columns)} assets) vs buy&hold basket -- {cadence} -- SMA{long}, "
          f"{'DNA-conditional' if conditional else 'always'}-exit")
    print(f"   {'window':8} {'STRAT%':>8} {'S_dd%':>7} {'S_cal':>6} | {'BHbskt%':>8} {'B_dd%':>7} {'B_cal':>6}")
    out = {}
    for w, (lo, hi) in bounds.items():
        s = stats(port_s, lo, hi); b = stats(port_b, lo, hi); out[w] = {"strat": s, "bh": b}
        print(f"   {w:8} {s['comp']:8.1f} {s['dd']:7.1f} {str(s['calmar']):>6} | {b['comp']:8.1f} {b['dd']:7.1f} {str(b['calmar']):>6}")
    f = out["FULL"]
    print(f"   READ: full-cycle strat {f['strat']['comp']}% @ dd {f['strat']['dd']}% (Calmar {f['strat']['calmar']}) "
          f"vs basket {f['bh']['comp']}% @ dd {f['bh']['dd']}% (Calmar {f['bh']['calmar']})")
    return out


# ---------------------------------------------------------------------------
# CYCLE-7: the PORTFOLIO RIGOR GATE. Is the book's edge real regime-TIMING, or just a lower-exposure artifact
# (in cash ~35% -> mechanically less DD)? Exposure-matched null: circularly PHASE-SHIFT each asset's regime
# position by a random offset -- preserves in-fraction + block structure + flip-count EXACTLY, decorrelates
# timing from the price path. If the real regime book beats the phase-shifted null on Calmar, the timing is real.
# ---------------------------------------------------------------------------
def portfolio_timing_null(long=150, conditional=False, cost=TAKER_RT, cadence="1d", assets=None,
                          n_seeds=200) -> dict:
    assets = assets or U10
    series = {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + 60: continue
        series[sym] = df.set_index("date")["close"]
    if not series: print("no data"); return {}
    panel = pd.DataFrame(series).sort_index()
    # precompute per-asset: aligned returns + the regime position (on the asset's own valid index)
    rets = {}; positions = {}; idxs = {}
    for sym in panel.columns:
        c = panel[sym].dropna(); close = c.to_numpy(float)
        sma = c.rolling(long, min_periods=long // 2).mean().to_numpy()
        vol20 = c.pct_change().rolling(20, min_periods=10).std().to_numpy()
        vol_med = pd.Series(vol20).rolling(150, min_periods=60).median().to_numpy()
        slope = sma / np.roll(sma, 20) - 1.0
        pos = _regime_position(close, sma, vol20, vol_med, slope, conditional)
        r = np.zeros(len(close)); r[1:] = close[1:] / close[:-1] - 1.0
        rets[sym] = (c.index, r); positions[sym] = pos; idxs[sym] = c.index

    def book_calmar(pos_map):
        sret = pd.DataFrame(index=panel.index)
        for sym in panel.columns:
            pos = pos_map[sym]; idx, r = rets[sym]
            pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
            flips = np.abs(np.diff(np.concatenate([[0.0], pos_lag])))
            sret[sym] = pd.Series(pos_lag * r - flips * cost, index=idx)
        port = sret.mean(axis=1, skipna=True).fillna(0.0)
        eq = (1.0 + port).cumprod()
        comp = float((eq.iloc[-1] - 1.0) * 100); peak = eq.cummax()
        dd = float(((eq - peak) / peak).min() * 100)
        return (comp / abs(dd)) if dd < -1e-6 else 0.0, comp, dd

    real_cal, real_comp, real_dd = book_calmar(positions)
    rng = np.random.default_rng(7); null_cals = []
    for s in range(n_seeds):
        shifted = {}
        for sym in panel.columns:
            p = positions[sym]; off = int(rng.integers(long, len(p) - long)) if len(p) > 2 * long else 1
            shifted[sym] = np.roll(p, off)                      # phase-shift: same exposure+blocks, decorrelated
        c, _, _ = book_calmar(shifted); null_cals.append(c)
    null_cals = np.array(null_cals)
    p50, p95 = float(np.percentile(null_cals, 50)), float(np.percentile(null_cals, 95))
    beats = real_cal > p95
    print(f"## PORTFOLIO TIMING NULL -- {len(panel.columns)} assets -- {cadence} -- SMA{long} -- {n_seeds} phase-shift seeds")
    print(f"   REAL regime book: Calmar={real_cal:.2f}  (comp={real_comp:.0f}% dd={real_dd:.1f}%)")
    print(f"   exposure-matched NULL (phase-shifted timing): Calmar p50={p50:.2f}  p95={p95:.2f}")
    print(f"   VERDICT: real {'BEATS' if beats else 'does NOT beat'} the exposure-matched null at p95 "
          f"-> regime timing {'ADDS value beyond just lower exposure' if beats else 'is ~just a lower-exposure (de-risking) effect'}")
    return {"real_calmar": real_cal, "null_p50": p50, "null_p95": p95, "beats": beats}


# ---------------------------------------------------------------------------
# CYCLE-8: per-YEAR decomposition -- WHEN does the regime book win vs lose? (the mechanism: bear/volatile-year
# capital-preservation vs bull-year cash-drag). Reuses the portfolio return construction, groups by calendar year.
# ---------------------------------------------------------------------------
def portfolio_by_year(long=120, conditional=False, cost=TAKER_RT, cadence="1d", assets=None) -> dict:
    assets = assets or U10
    series = {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + 60: continue
        series[sym] = df.set_index("date")["close"]
    if not series: print("no data"); return {}
    panel = pd.DataFrame(series).sort_index()
    strat_ret = pd.DataFrame(index=panel.index); bh_ret = pd.DataFrame(index=panel.index)
    for sym in panel.columns:
        c = panel[sym].dropna(); close = c.to_numpy(float)
        sma = c.rolling(long, min_periods=long // 2).mean().to_numpy()
        vol20 = c.pct_change().rolling(20, min_periods=10).std().to_numpy()
        vol_med = pd.Series(vol20).rolling(150, min_periods=60).median().to_numpy()
        slope = sma / np.roll(sma, 20) - 1.0
        pos = _regime_position(close, sma, vol20, vol_med, slope, conditional)
        r = np.zeros(len(close)); r[1:] = close[1:] / close[:-1] - 1.0
        pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
        flips = np.abs(np.diff(np.concatenate([[0.0], pos_lag])))
        strat_ret[sym] = pd.Series(pos_lag * r - flips * cost, index=c.index)
        bh_ret[sym] = pd.Series(r, index=c.index)
    port_s = strat_ret.mean(axis=1, skipna=True).fillna(0.0)
    port_b = bh_ret.mean(axis=1, skipna=True).fillna(0.0)
    exposure = (strat_ret.notna() & (strat_ret != 0)).mean(axis=1)   # rough fraction of book deployed

    def yr_stat(r):
        eq = (1.0 + r).cumprod(); comp = (eq.iloc[-1] - 1.0) * 100 if len(eq) else 0.0
        peak = eq.cummax(); dd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
        return comp, dd
    print(f"## PORTFOLIO by-YEAR -- {len(panel.columns)} assets -- {cadence} -- SMA{long}, "
          f"{'cond' if conditional else 'always'}-exit  (when does the regime book win vs the basket?)")
    print(f"   {'year':6} {'STRAT%':>8} {'S_dd%':>7} | {'BSKT%':>8} {'B_dd%':>7} | {'edge_pp':>8} {'avg_expo':>8}")
    years = sorted(set(port_s.index.year))
    wins = 0; n = 0
    for y in years:
        ms = port_s[port_s.index.year == y]; mb = port_b[port_b.index.year == y]
        if len(ms) < 30: continue
        sc, sd = yr_stat(ms); bc, bd = yr_stat(mb); expo = exposure[exposure.index.year == y].mean()
        edge = sc - bc; wins += int(edge > 0); n += 1
        flag = "WIN " if edge > 0 else "lose"
        print(f"   {y:6} {sc:8.1f} {sd:7.1f} | {bc:8.1f} {bd:7.1f} | {edge:8.1f} {expo:8.2f}  {flag}")
    print(f"   -> book beats basket on annual compound in {wins}/{n} years (the rest = bull-year cash-drag give-up)")
    return {"port_s": port_s, "port_b": port_b}


# ---------------------------------------------------------------------------
# CYCLE-9: MARKET-BREADTH regime gate. Instead of each asset timing on its OWN MA, gate the whole book on
# market BREADTH = fraction of the universe above its MA (the classic risk-on/off regime signal). Variants:
# (A) pure breadth: hold ALL assets when breadth>thr; (B) combined: hold asset i when (own price>MA) AND breadth-on.
# Tests whether a market-regime view beats the per-asset MA book. All past-only (breadth uses only bar-t closes).
# ---------------------------------------------------------------------------
def portfolio_breadth_gate(long=120, breadth_thr=0.5, cost=TAKER_RT, cadence="1d", assets=None) -> dict:
    assets = assets or U10
    series = {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < long + 60: continue
        series[sym] = df.set_index("date")["close"]
    if not series: print("no data"); return {}
    panel = pd.DataFrame(series).sort_index()
    above = pd.DataFrame(index=panel.index); rmat = pd.DataFrame(index=panel.index)
    for sym in panel.columns:
        c = panel[sym]; sma = c.rolling(long, min_periods=long // 2).mean()
        above[sym] = (c > sma)
        rmat[sym] = c.pct_change()
    breadth = above.mean(axis=1, skipna=True)               # fraction of universe above its MA (past-only)
    risk_on = (breadth > breadth_thr)

    listed = panel.notna()                                   # asset trading at bar t (else excluded from EW mean)
    def book(pos_df):
        sret = pd.DataFrame(index=panel.index)
        for sym in panel.columns:
            pos = pos_df[sym].shift(1).fillna(False).astype(float)   # next-bar fill
            flips = pos.diff().abs().fillna(0.0)
            sret[sym] = (pos * rmat[sym].fillna(0.0) - flips * cost).where(listed[sym])  # NaN pre-listing
        port = sret.mean(axis=1, skipna=True).fillna(0.0)
        eq = (1.0 + port).cumprod(); comp = float((eq.iloc[-1] - 1.0) * 100)
        peak = eq.cummax(); dd = float(((eq - peak) / peak).min() * 100)
        return comp, dd, (comp / abs(dd) if dd < -1e-6 else None)

    base_pos = above.copy()
    a_pos = pd.DataFrame({s: risk_on for s in panel.columns}, index=panel.index)
    b_pos = above & risk_on.values[:, None]
    bc, bd, bcal = book(base_pos); ac, ad, acal = book(a_pos); cc, cd, ccal = book(b_pos)
    bh = rmat.mean(axis=1, skipna=True).fillna(0.0)
    bhe = (1.0 + bh).cumprod(); bhc = float((bhe.iloc[-1] - 1) * 100)
    bhd = float(((bhe - bhe.cummax()) / bhe.cummax()).min() * 100)
    print(f"## MARKET-BREADTH regime gate -- {len(panel.columns)} assets -- {cadence} -- SMA{long}, breadth_thr={breadth_thr}")
    print(f"   {'variant':28} {'comp%':>9} {'dd%':>7} {'Calmar':>7}")
    print(f"   {'buy&hold basket':28} {bhc:9.0f} {bhd:7.1f} {str(round(bhc/abs(bhd),2) if bhd<-1e-6 else None):>7}")
    print(f"   {'per-asset MA (baseline)':28} {bc:9.0f} {bd:7.1f} {str(round(bcal,2)):>7}")
    print(f"   {'A: pure market-breadth':28} {ac:9.0f} {ad:7.1f} {str(round(acal,2) if acal else None):>7}")
    print(f"   {'B: own-MA AND breadth-on':28} {cc:9.0f} {cd:7.1f} {str(round(ccal,2) if ccal else None):>7}")
    best = max([("baseline", bcal), ("A_breadth", acal or 0), ("B_combined", ccal or 0)], key=lambda x: x[1])
    helps = best[0] != "baseline" and best[1] > bcal
    print(f"   -> best Calmar: {best[0]} ({best[1]:.2f})  [breadth gate {'HELPS' if helps else 'does NOT beat per-asset MA'}]")
    return {"baseline": bcal, "A": acal, "B": ccal, "best": best[0]}


# ---------------------------------------------------------------------------
# CYCLE-10: PBO/CSCV deflation of the MA-length parameter search (success-criterion rigor). The wealth edge is
# MA-sensitive (SMA100/150 win, 200 loses) -> is "pick the best MA" overfit? Build the book return series for a
# GRID of MA lengths as the candidate family, run pbo_cscv (combinatorially-symmetric IS/OOS): PBO ~ probability
# the in-sample-best config is an OOS under-performer. PBO < 0.10 = the selection generalizes.
# ---------------------------------------------------------------------------
def _book_return_series(long, conditional, cost, cadence, panel_cache) -> pd.Series:
    panel = panel_cache
    sret = pd.DataFrame(index=panel.index); listed = panel.notna()
    for sym in panel.columns:
        c = panel[sym].dropna(); close = c.to_numpy(float)
        sma = c.rolling(long, min_periods=long // 2).mean().to_numpy()
        vol20 = c.pct_change().rolling(20, min_periods=10).std().to_numpy()
        vol_med = pd.Series(vol20).rolling(150, min_periods=60).median().to_numpy()
        slope = sma / np.roll(sma, 20) - 1.0
        pos = _regime_position(close, sma, vol20, vol_med, slope, conditional)
        r = np.zeros(len(close)); r[1:] = close[1:] / close[:-1] - 1.0
        pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
        flips = np.abs(np.diff(np.concatenate([[0.0], pos_lag])))
        sret[sym] = pd.Series(pos_lag * r - flips * cost, index=c.index)
    return sret.where(listed).mean(axis=1, skipna=True).fillna(0.0)


def portfolio_pbo(longs=(80, 100, 120, 150, 180, 200), conditional=False, cost=TAKER_RT,
                  cadence="1d", assets=None, S=16) -> dict:
    try:
        from .pbo_cscv import pbo_cscv
    except ImportError:
        from strat.pbo_cscv import pbo_cscv
    assets = assets or U10
    series = {}
    for sym in assets:
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < max(longs) + 60: continue
        series[sym] = df.set_index("date")["close"]
    panel = pd.DataFrame(series).sort_index()
    cols = {f"SMA{L}": _book_return_series(L, conditional, cost, cadence, panel) for L in longs}
    R = pd.DataFrame(cols).fillna(0.0)
    res = pbo_cscv(R.to_numpy(), S=S)
    print(f"## PBO/CSCV deflation -- {len(panel.columns)} assets -- {cadence} -- MA grid {list(longs)} -- S={S} blocks")
    print(f"   candidate Sharpes (full-sample): " + "  ".join(f"{c}={R[c].mean()/ (R[c].std()+1e-9)*np.sqrt(252):.2f}" for c in R.columns))
    pbo = res.get("pbo", res.get("PBO"))
    print(f"   PBO = {pbo:.3f}   (~0.5 skill-less selection | <0.10 generalizes | >0.5 overfit)")
    for k in ("prob_oos_loss", "oos_below_median_rate", "n_splits", "logits_summary"):
        if k in res: print(f"   {k}: {res[k]}")
    verdict = ("the MA-length selection GENERALIZES (low overfit)" if pbo is not None and pbo < 0.10 else
               "the MA-length selection is PARAMETER-SENSITIVE / not clearly generalizing" if pbo is not None and pbo < 0.5 else
               "the MA-length selection is SKILL-LESS/overfit (picking the best MA in-sample does NOT predict OOS)")
    print(f"   VERDICT: {verdict}")
    return res


# ---------------------------------------------------------------------------
def selftest():
    """Synthetic: a trending series with vol-expansion breakouts -> the breakout entry must (a) produce
    trades, (b) beat a random-entry null on held-out. Validates the lab end-to-end, no market data."""
    rng = np.random.default_rng(5); n = 1600
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rets = rng.normal(0.0003, 0.012, n); run_left = 0
    for t in range(1, n):  # inject occasional sustained up-runs preceded by a breakout
        if run_left > 0: rets[t] += 0.012; run_left -= 1
        elif rng.random() < 0.02: run_left = 12
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})
    h_, l_, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h_ - l_), (h_ - pc).abs(), (l_ - pc).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=7).mean()
    policy = chandelier(atr_mult=3.0)
    entry = sig_donchian(df, n=20)
    ev = evaluate(df, entry, policy)
    fw = random_entry_null(ev["harness"], n_books=200, seed=7)
    has_trades = entry.sum() >= 5 and ev["per_window"]["TRAIN"]["n"] > 0
    orc_ok = ev["per_window"]["TRAIN"]["oracle_pct"] > 0
    print("[entry_signal_lab selftest]")
    print(f"  donchian breakout trades fired: {int(entry.sum())}")
    for w in SetupHarness.WINDOWS:
        p = ev["per_window"][w]
        print(f"  {w:6} comp={p['compound_pct']:+8.2f}%  n={p['n']:<3} cap={p['capture']} oracle={p['oracle_pct']}%")
    print(f"  firewall verdict: {fw['verdict']}")
    ok = has_trades and orc_ok
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'} (has_trades={has_trades}, oracle>0={orc_ok})")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.entry_signal_lab")
    ap.add_argument("--asset"); ap.add_argument("--cadence", default="1d")
    ap.add_argument("--family", default="ma_stack", choices=list(SIGNALS.keys()))
    ap.add_argument("--atr-mult", type=float, default=3.0)
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--families", default=None, help="comma list for --grid")
    ap.add_argument("--deep", action="store_true", help="deep honest gate (benchmark+firewall) on --family --params")
    ap.add_argument("--params", default="{}", help="JSON params dict for --deep")
    ap.add_argument("--unseen", action="store_true", help="include the UNSEEN verdict look in --deep")
    ap.add_argument("--rank", default="oos_excess", choices=["oos_excess", "trainval"], help="grid rank key")
    ap.add_argument("--regime", action="store_true", help="canonical Faber/GTAA regime-timing rule vs buy&hold")
    ap.add_argument("--long", type=int, default=150, help="long-MA length for --regime")
    ap.add_argument("--resample-days", type=int, default=None, help="resample 1d into N-day bars (3,7) for --regime")
    ap.add_argument("--adaptive", action="store_true", help="regime-adaptive exit compare (tight vs loose vs ER-adaptive)")
    ap.add_argument("--er-n", type=int, default=20, help="efficiency-ratio window for --adaptive")
    ap.add_argument("--tight", type=float, default=2.0, help="tight-end atr-mult for adaptive trail")
    ap.add_argument("--loose", type=float, default=14.0, help="loose-end atr-mult for adaptive trail")
    ap.add_argument("--dna", action="store_true", help="regime-exit DNA: causal feature separation good vs whipsaw exits")
    ap.add_argument("--conditional", action="store_true", help="DNA-built conditional exit vs always-exit + buy&hold")
    ap.add_argument("--dna-rich", dest="dna_rich", action="store_true", help="rich chimera-feature exit DNA")
    ap.add_argument("--portfolio", action="store_true", help="PORTFOLIO regime-managed beta vs buy&hold basket")
    ap.add_argument("--always-exit", dest="always_exit", action="store_true", help="portfolio: always-exit (not DNA-conditional)")
    ap.add_argument("--universe", default="u10", help="portfolio universe (u10/u50/u100)")
    ap.add_argument("--timing-null", dest="timing_null", action="store_true", help="exposure-matched phase-shift null on the portfolio")
    ap.add_argument("--by-year", dest="by_year", action="store_true", help="per-calendar-year portfolio win/lose decomposition")
    ap.add_argument("--pbo", action="store_true", help="PBO/CSCV deflation of the MA-length parameter search")
    ap.add_argument("--vol-filter", dest="vol_filter", action="store_true", help="adjacent test: risk-off on vol-spike regime")
    ap.add_argument("--breadth-gate", dest="breadth_gate", action="store_true", help="market-breadth regime gate vs per-asset MA")
    ap.add_argument("--breadth-thr", dest="breadth_thr", type=float, default=0.5, help="breadth risk-on threshold")
    a = ap.parse_args(argv)
    if a.selftest: return selftest()
    if a.regime:
        run_regime(a.long, resample_days=a.resample_days, atr_mult=a.atr_mult, cadence=a.cadence)
        return 0
    if a.adaptive:
        run_adaptive_compare(long=a.long, er_n=a.er_n, tight=a.tight, loose=a.loose, cadence=a.cadence)
        return 0
    if a.dna:
        regime_exit_dna(long=a.long, er_n=a.er_n, cadence=a.cadence)
        return 0
    if a.conditional:
        run_conditional_compare(long=a.long, er_n=a.er_n, backstop=a.loose, cadence=a.cadence)
        return 0
    if a.dna_rich:
        regime_exit_dna_rich(long=a.long, cadence=a.cadence)
        return 0
    if a.pbo:
        assets = U10
        if a.universe and a.universe != 'u10':
            import yaml
            d = yaml.safe_load(open(f'config/universes/{a.universe}.yaml', encoding='utf-8'))
            syms=[]
            for k,v in (d.items() if isinstance(d,dict) else []):
                if isinstance(v,list): syms+=[(x['symbol'] if isinstance(x,dict) else x) for x in v]
            assets=sorted(set(syms))
        portfolio_pbo(conditional=not a.always_exit, cadence=a.cadence, assets=assets)
        return 0
    if a.breadth_gate:
        assets = U10
        if a.universe and a.universe != 'u10':
            import yaml
            d = yaml.safe_load(open(f'config/universes/{a.universe}.yaml', encoding='utf-8'))
            syms=[]
            for k,v in (d.items() if isinstance(d,dict) else []):
                if isinstance(v,list): syms+=[(x['symbol'] if isinstance(x,dict) else x) for x in v]
            assets=sorted(set(syms))
        portfolio_breadth_gate(long=a.long, breadth_thr=a.breadth_thr, cadence=a.cadence, assets=assets)
        return 0
    if a.by_year:
        assets = U10
        if a.universe and a.universe != 'u10':
            import yaml
            d = yaml.safe_load(open(f'config/universes/{a.universe}.yaml', encoding='utf-8'))
            syms=[]
            for k,v in (d.items() if isinstance(d,dict) else []):
                if isinstance(v,list): syms+=[(x['symbol'] if isinstance(x,dict) else x) for x in v]
            assets=sorted(set(syms))
        portfolio_by_year(long=a.long, conditional=not a.always_exit, cadence=a.cadence, assets=assets)
        return 0
    if a.portfolio:
        assets = U10
        if a.universe and a.universe != "u10":
            import yaml
            d = yaml.safe_load(open(f"config/universes/{a.universe}.yaml", encoding="utf-8"))
            syms = []
            for k, v in (d.items() if isinstance(d, dict) else []):
                if isinstance(v, list):
                    syms += [(x["symbol"] if isinstance(x, dict) else x) for x in v]
            assets = sorted(set(syms))
            print(f"[universe {a.universe}] {len(assets)} assets")
        if a.timing_null:
            portfolio_timing_null(long=a.long, conditional=not a.always_exit, cadence=a.cadence, assets=assets)
        else:
            portfolio_backtest(long=a.long, conditional=not a.always_exit, cadence=a.cadence, assets=assets, vol_filter=a.vol_filter)
        return 0
    if a.deep:
        params = json.loads(a.params)
        d = deep_eval(a.family, params, a.cadence, atr_mult=a.atr_mult, look_unseen=a.unseen)
        _print_deep(d, look_unseen=a.unseen)
        return 0
    if a.grid:
        fams = a.families.split(",") if a.families else None
        rows = grid_search(a.cadence, families=fams, atr_mult=a.atr_mult, rank=a.rank)
        print(f"## ENTRY-SIGNAL GRID -- u10 -- {a.cadence} -- chandelier(atr={a.atr_mult}) -- ranked by {a.rank} "
              "(oos_excess = median OOS excess vs buy&hold = the RIGOR GUARD)")
        print(f"   {'family':16} {'params':30} {'TRAIN%':>8} {'OOS%':>7} {'OOS_xs':>7} {'>beta':>6} {'OOS_capt':>8} {'n_OOS':>6}")
        for r in rows[:25]:
            print(f"   {r['family']:16} {str(r['params']):30} {r['TRAIN_med']:8.0f} {r['OOS_med']:7.1f} "
                  f"{str(r['OOS_xs_med']):>7} {str(r['OOS_beatsbeta_frac']):>6} {str(r['OOS_capt_med']):>8} {r['n_OOS']:>6}")
        return 0
    # single asset+family quick read
    df = load_ohlc(a.asset or "BTCUSDT", a.cadence)
    if df is None: print("no data"); return 1
    entry = SIGNALS[a.family](df)
    ev = evaluate(df, entry, chandelier(atr_mult=a.atr_mult))
    print(f"## {a.asset or 'BTCUSDT'} {a.cadence} {a.family} -- chandelier(atr={a.atr_mult})")
    for w in SetupHarness.WINDOWS:
        p = ev["per_window"][w]
        print(f"  {w:6} comp={p['compound_pct']:+8.2f}%  n={p['n']:<3} cap={p['capture']} oracle={p['oracle_pct']}% dd={p['dd_pct']}%")
    bm = benchmark_excess(ev["harness"])
    print(f"  benchmark beats_beta_held={bm['beats_beta_held']}  bear_preserved={bm['bear_preserved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
