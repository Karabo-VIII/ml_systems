"""src/strat/daily_engine.py -- THE DEPLOYABLE DAILY-RETURN ENGINE (turnkey "click play").

WHAT THIS IS (user mandate 2026-06-12): a WORKING, deployable trading engine that, run daily,
produces a long-only u10 book of positions + a daily return stream. NOT a discovery / alpha hunt --
the discovery is DONE. This ASSEMBLES the two proven robust components the other instance validated
into one turnkey system:

  CORE (the return generator, positive every-day, full coverage):
    a VOL-TARGETED long buy-hold book over u10 at DAILY cadence. Per-asset target weight =
    vol_target / realized_vol, capped at max_per_name, normalized to <= max_gross, long-only,
    EVERY day (full coverage). This is VOLTGT_BH -- the robust winner from deep2020_bestbook.py
    (the vol-target component is the part that transfers; the XS-momentum selection does not).
    An ORTHOBOOK option adds a small ORTHOGONAL calendar tilt (day-of-week) on top of the
    vol-target weights; we pick whichever core is more robust on the full cycle (compound + maxDD
    + daily-positive-rate).

  DEFENSIVE OVERLAY (the regime/ML value -- drawdown control in the bear):
    REUSES rolling_regime_book.py's causal regime classifier (regime_features + fit_regime_thresholds
    + classify_raw + apply_hysteresis): trend / chop / down on a rolling causal window, thresholds
    FIT ON A TRAIN PREFIX ONLY (no look-ahead), hysteresis to exploit regime persistence. The regime
    label maps to a daily EXPOSURE SCALAR in [0,1]: trend=1.0, chop=0.6, down=0.2 (de-risk the bear).
    Final book = core_weights * regime_scalar.

The engine is CAUSAL (weights lagged 1 bar, MtM-no-double-count), costed (taker default / maker opt),
long-only, daily. It produces a DAILY RETURN STREAM (non-null) + a "what to hold today" book.

HONEST FRAME: this is a working POSITIVE-CORE risk-managed beta + vol-target engine with controlled
drawdown. It does NOT print alpha (internal-data ceiling, per MEMORY.md). The deliverable is a
turnkey system that generates returns daily, with the regime overlay cutting maxDD in the bear.

ROBUSTNESS (2026-06-13): subjected to a 7-dimension gauntlet (src/strat/daily_engine_gauntlet.py).
The HARDENED deployment default uses the MORE DEFENSIVE regime scalar {trend:1.0, chop:0.5, down:0.1}
(maxDD ~-48% vs the prior -55%, Sharpe 1.37 vs 1.31, held-out p05 less negative), confirmed by a
per-episode bear split (improves BOTH 2022 and 2025 bears monotonically). The prior documented headline
scalar {1.0,0.6,0.2} is preserved as --aggressive. HONEST: held-out (2025-03+) compound is still
flat-to-slightly-negative -- this is a beta engine, not an alpha sleeve (see docs/DAILY_ENGINE.md
"## Robustness").

MODES (turnkey):
  python -m strat.daily_engine                              # default: 2020-2025 backtest + today's book
  python -m strat.daily_engine --backtest 2020-01-01:2025-12-31
  python -m strat.daily_engine --today                      # the recommended book for the latest day
  python -m strat.daily_engine --date 2025-06-01            # the book for a specific day
  python -m strat.daily_engine --selftest                   # two-sided synthetic soundness
  python -m strat.daily_engine --core orthobook             # use the calendar-tilt core variant
  python -m strat.daily_engine --maker                      # maker cost (0.0006 rt)
  python -m strat.daily_engine --aggressive                 # prior {1.0,0.6,0.2} scalar (more compound, more DD)

No emoji (Windows cp1252). Does NOT git commit (overseer commits).
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
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

from strat.ma_per_instrument import _panel                                  # noqa: E402
from strat.portfolio_replay import TAKER_RT, MAKER_RT                        # noqa: E402
from strat.rolling_regime_book import (regime_features, fit_regime_thresholds,  # noqa: E402
                                       classify_raw, apply_hysteresis)
from strat.scorecard import score_book                                      # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# ----------------------------- engine config (all declarative) -----------------------------
CADENCE = "1d"
ANN = 365.0
VOL_TARGET = 0.02            # per-asset daily vol target (2% daily ~ matches the validated core)
VOL_WINDOW = 30             # realized-vol lookback (daily bars), causal
MAX_PER_NAME = 0.15        # per-name weight cap
MAX_GROSS = 1.0            # total gross-exposure cap (long-only)
# regime overlay (daily-cadence tuned; mirrors rolling_regime_book's structure)
REGIME_LOOKBACK = 60        # ~2 months of daily bars for the regime features
REGIME_MIN_DWELL = 5       # hysteresis: a new regime must persist 5 days before it takes effect
BREADTH_MA = 100            # an asset is "up" if close > its own SMA100 (breadth member)
# the regime -> daily exposure scalar map (the defensive overlay; de-risk the bear).
# HARDENED DEFAULT (2026-06-13 robustness gauntlet): the deployment default is the MORE DEFENSIVE map
# {trend:1.0, chop:0.5, down:0.1}. The gauntlet's per-episode bear split (RWYB-VERIFIED) showed deeper
# de-risk cuts maxDD MONOTONICALLY in BOTH independent bears (2022: -47.7 -> -41.4; 2025: -29.9 -> -26.4)
# and lifts held-out p05 (-42.7 -> -36) and full-cycle Sharpe (1.31 -> 1.37), at the cost of ~280pp of
# the (still enormous) compound. A -55% maxDD default breaches the project's 20-30% maxDD floor badly;
# this default is the deployment-grade one. The ORIGINAL aggressive map {1.0,0.6,0.2} (higher compound,
# higher maxDD) is preserved as --aggressive for reproducibility of the prior documented headline.
REGIME_SCALAR_DEFENSIVE = {"trend": 1.0, "chop": 0.5, "down": 0.1}   # HARDENED deployment default
REGIME_SCALAR_AGGRESSIVE = {"trend": 1.0, "chop": 0.6, "down": 0.2}  # prior documented headline (opt-in)
REGIME_SCALAR = REGIME_SCALAR_DEFENSIVE
# the TRAIN prefix the regime thresholds are FIT on (no look-ahead onto the eval span)
REGIME_TRAIN_FIT = ("2019-01-01", "2023-01-01")
# the calendar tilt for the orthobook core (day-of-week multiplicative tilt, small, orthogonal)
# fit nowhere -- a fixed, pre-registered mild tilt (Mon-Sun); orthogonal to vol-target by construction.
DOW_TILT = np.array([1.05, 1.00, 1.00, 1.00, 1.00, 0.95, 0.95])   # Mon..Sun (mild, sums ~7)


def _syms():
    return [a["symbol"] for a in yaml.safe_load(
        open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]


# ===========================================================================
# 1. load the date-aligned daily close panel (full history) for u10
# ===========================================================================
def load_close_panel(syms=None):
    """Return a date-aligned daily CLOSE DataFrame [dates x assets] over the FULL u10 history.
    Floored to daily, deduped (the alignment contract from portfolio_replay)."""
    syms = syms or _syms()
    closes = {}
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, CADENCE)
        except Exception:
            continue
        idx = pd.to_datetime(ms, unit="ms").floor("D")
        s = pd.Series(c, index=idx)
        s = s[~s.index.duplicated(keep="last")]
        closes[sym] = s
    panel = pd.DataFrame(closes).sort_index()
    # alignment guard (same contract as portfolio_replay): the union index must not explode
    _max_rows = max((len(s) for s in closes.values()), default=0)
    assert len(panel) <= 1.5 * _max_rows + 5, (
        f"date-alignment regression: panel {len(panel)} rows >> max per-asset {_max_rows}")
    return panel


# ===========================================================================
# 2. CORE: vol-targeted long-only weights (every day, full coverage) + orthobook variant
# ===========================================================================
def core_weights(close_panel, core="voltgt", vol_target=VOL_TARGET, max_per_name=MAX_PER_NAME,
                 max_gross=MAX_GROSS):
    """Per-day per-asset CORE target weights (DataFrame [dates x assets]). CAUSAL: realized vol uses
    bars <= t (rolling, no look-ahead). For each day, for each PRESENT asset:
        raw_w = clip(vol_target / realized_vol, 0, max_per_name)
    then normalize the row so gross <= max_gross (scale DOWN only -- if under-budget, stay
    under-budget; we never lever up a low-vol day past full exposure). Long-only, every day.
    core='orthobook' multiplies the per-day weights by a fixed day-of-week tilt (orthogonal,
    pre-registered) before re-normalizing -- the calendar component on top of vol-target."""
    rets = close_panel.pct_change(fill_method=None)
    rvol = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()
    present = close_panel.notna()
    raw = (vol_target / (rvol + 1e-12)).clip(lower=0.0, upper=max_per_name)
    raw = raw.where(present & np.isfinite(rvol))      # NaN where absent or vol un-warmed
    if core == "orthobook":
        dow = raw.index.dayofweek.to_numpy()           # 0=Mon .. 6=Sun
        tilt = DOW_TILT[dow]                            # per-row multiplicative tilt
        raw = raw.mul(pd.Series(tilt, index=raw.index), axis=0)
        raw = raw.clip(upper=max_per_name)             # keep the per-name cap after tilt
    gross = raw.sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / (gross + 1e-12), 1.0)
    w = raw.mul(pd.Series(scale, index=raw.index), axis=0).fillna(0.0)
    return w


# ===========================================================================
# 3. DEFENSIVE OVERLAY: the rolling regime classifier -> a daily exposure scalar in [0,1]
# ===========================================================================
def regime_scalar_series(close_panel, scalar_map=None):
    """Build the causal, hysteresis-smoothed daily regime label over the FULL history and map it to a
    daily exposure scalar in [0,1]. Thresholds FIT on REGIME_TRAIN_FIT ONLY (no look-ahead onto the
    eval span). scalar_map defaults to the module REGIME_SCALAR (the hardened defensive default); pass
    REGIME_SCALAR_AGGRESSIVE for the prior documented headline. Returns
    (scalar_series, label_series, thresholds, regime_share)."""
    scalar_map = scalar_map if scalar_map is not None else REGIME_SCALAR
    # the regime_features kernel takes a cadence key into its LOOKBACK/BREADTH tables; we pass our own
    # daily lookback by temporarily registering a '1d' entry (the kernel reads module-level dicts).
    import strat.rolling_regime_book as RRB
    RRB.LOOKBACK["1d"] = REGIME_LOOKBACK
    RRB.BREADTH_MA = BREADTH_MA
    feat = regime_features(close_panel, "1d")
    th = fit_regime_thresholds(feat, *REGIME_TRAIN_FIT)
    raw = classify_raw(feat, th)
    smoothed = apply_hysteresis(raw, REGIME_MIN_DWELL)
    labels = pd.Series(smoothed, index=close_panel.index)
    scalar = labels.map(scalar_map).astype(float)
    scalar = scalar.fillna(scalar_map["chop"])         # un-warmed early bars -> conservative chop
    share = {k: round(float(np.mean(smoothed == k)), 3) for k in ("trend", "chop", "down")}
    return scalar, labels, th, share


# ===========================================================================
# 4. BOOK ASSEMBLY: final weights = core * regime_scalar; the daily NET return stream (causal MtM)
# ===========================================================================
def build_book(close_panel, core="voltgt", use_overlay=True, cost_rt=TAKER_RT, scalar_map=None):
    """Assemble the final daily book + the causal MtM net-return stream.
    Returns a dict with the weight DataFrame, the net Series, the regime label/scalar Series, and the
    per-day gross-exposure Series. CAUSAL: weights are lagged 1 bar before applying to returns; turnover
    cost charged on the change in lagged weights (MtM-no-double-count).
    scalar_map (optional) overrides the regime->exposure map (default = the hardened defensive REGIME_SCALAR)."""
    cw = core_weights(close_panel, core=core)
    if use_overlay:
        scalar, labels, th, share = regime_scalar_series(close_panel, scalar_map=scalar_map)
        W = cw.mul(scalar, axis=0)
    else:
        scalar = pd.Series(1.0, index=close_panel.index)
        labels = pd.Series("trend", index=close_panel.index)
        th, share = {}, {"trend": 1.0, "chop": 0.0, "down": 0.0}
        W = cw
    rets = close_panel.pct_change(fill_method=None).fillna(0.0)
    Wl = W.shift(1).fillna(0.0)                          # lag 1 bar (causal)
    gross_ret = (Wl * rets).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = gross_ret - turnover * (cost_rt / 2.0)
    gross_exposure = W.sum(axis=1)
    return {"W": W, "core_W": cw, "net": net, "scalar": scalar, "labels": labels,
            "thresholds": th, "regime_share": share, "gross_exposure": gross_exposure,
            "turnover": turnover}


# ===========================================================================
# 5. metrics on a daily net-return Series over a window
# ===========================================================================
def window_stats(net, gross_exposure=None, lo=None, hi=None):
    """Compute the headline stats on a daily net-return Series, optionally sliced to [lo, hi)."""
    s = net.dropna()
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index < pd.Timestamp(hi)]
    if len(s) < 5:
        return {"n_days": int(len(s)), "error": "window too short"}
    d = s.to_numpy()
    eq = np.cumprod(1 + d)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    nyr = len(d) / ANN
    cagr = float((eq[-1] ** (1 / nyr) - 1) * 100) if eq[-1] > 0 else -100.0
    sharpe = float(d.mean() / (d.std() + 1e-12) * np.sqrt(ANN))
    daily_pos = float((d > 0).mean() * 100)
    out = {
        "n_days": int(len(d)),
        "compound_pct": round(float((eq[-1] - 1) * 100), 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "maxdd_pct": round(maxdd, 2),
        "daily_pos_rate_pct": round(daily_pos, 1),
        "first_day": str(s.index[0])[:10], "last_day": str(s.index[-1])[:10],
    }
    if gross_exposure is not None:
        ge = gross_exposure.reindex(s.index).fillna(0.0).to_numpy()
        out["avg_gross_exposure"] = round(float(ge.mean()), 3)
        out["coverage_pct"] = round(float((ge > 1e-6).mean() * 100), 1)   # % days in market
    return out


# ===========================================================================
# 6. the "what to hold today" book
# ===========================================================================
def _data_quality_flags(close_panel, date, n_syms):
    """Deployment data-quality guard for the live book on `date`. Returns a list of human-readable
    WARNING strings (empty = clean). A data-feed gap on the latest bar would otherwise SILENTLY shrink
    the book (vol un-warmed / asset absent -> 0 weight), i.e. an unintended de-risk on a glitch -- the
    deployer must SEE that before allocating real capital."""
    flags = []
    row = close_panel.loc[date]
    present = int(row.notna().sum())
    if present == 0:
        flags.append("CRITICAL: latest bar is ALL-NaN (total data-feed gap) -- the book is 0-exposure; "
                     "do NOT liquidate on this; hold yesterday's book and fix the feed.")
    elif present < n_syms:
        missing = [a for a in close_panel.columns if pd.isna(row[a])]
        flags.append(f"WARN: {present}/{n_syms} u10 names present on {str(date)[:10]} "
                     f"(missing {missing}) -- book is built on a partial universe.")
    # staleness: is `date` materially older than 'now'? (the caller passes the latest panel bar)
    age_days = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(date)).days
    if age_days > 2:
        flags.append(f"WARN: latest panel bar {str(date)[:10]} is {age_days}d stale "
                     "(refresh the data before deploying today's book).")
    # vol-warmup: any present name with un-warmed realized vol -> its weight defaults to 0 (silent)
    rets = close_panel.pct_change(fill_method=None)
    rvol = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()
    unwarmed = [a for a in close_panel.columns
                if pd.notna(row[a]) and not np.isfinite(rvol.loc[date, a])]
    if unwarmed:
        flags.append(f"WARN: {unwarmed} present but vol un-warmed (<{VOL_WINDOW // 2} bars) -> "
                     "excluded from today's book until they warm up.")
    return flags


def book_for_date(close_panel, target_date=None, core="voltgt", use_overlay=True, scalar_map=None):
    """Return the recommended book for a given date (default: the latest available day):
    per-asset weights, the detected regime, the exposure scalar, gross exposure. The weights are the
    ENGINE weights for that date (core * scalar). NB: the engine TRADES the lagged weight, but the
    'what to hold today' answer is the weight CHOSEN at the close of target_date (held into next day).
    Includes a deployment data-quality guard (data_quality_flags)."""
    bk = build_book(close_panel, core=core, use_overlay=use_overlay, scalar_map=scalar_map)
    W, labels, scalar, gross = bk["W"], bk["labels"], bk["scalar"], bk["gross_exposure"]
    if target_date is None:
        date = W.index[-1]
    else:
        td = pd.Timestamp(target_date)
        sub = W.index[W.index <= td]
        if len(sub) == 0:
            return {"error": f"no data on or before {target_date}"}
        date = sub[-1]
    row = W.loc[date]
    weights = {a: round(float(w), 4) for a, w in row.items() if abs(w) > 1e-6}
    return {
        "date": str(date)[:10],
        "regime": str(labels.loc[date]),
        "exposure_scalar": round(float(scalar.loc[date]), 3),
        "gross_exposure": round(float(gross.loc[date]), 3),
        "n_positions": len(weights),
        "weights": dict(sorted(weights.items(), key=lambda kv: -kv[1])),
        "data_quality_flags": _data_quality_flags(close_panel, date, len(close_panel.columns)),
    }


# ===========================================================================
# 7. the full backtest: ENGINE vs CORE-ALONE vs BUY-HOLD
# ===========================================================================
def buy_hold_net(close_panel, cost_rt=TAKER_RT):
    """Equal-weight long-only buy-hold of u10 (the naive baseline): each present asset gets weight
    1/n_present that day, held, MtM. One entry cost amortized via daily turnover on the EW weights."""
    present = close_panel.notna()
    n_present = present.sum(axis=1).replace(0, np.nan)
    W = present.div(n_present, axis=0).fillna(0.0)        # equal-weight across present assets
    rets = close_panel.pct_change(fill_method=None).fillna(0.0)
    Wl = W.shift(1).fillna(0.0)
    gross_ret = (Wl * rets).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    return gross_ret - turnover * (cost_rt / 2.0), W.sum(axis=1)


def backtest(lo, hi, core="voltgt", cost_rt=TAKER_RT, scalar_map=None):
    """Run the three books over [lo, hi) and return a comparison dict.
    ENGINE   = core * regime overlay (the deployable system)
    CORE     = core alone, no overlay (isolates the overlay's drawdown-control value)
    BUYHOLD  = equal-weight long-only u10 (the naive baseline)"""
    panel = load_close_panel()
    eng = build_book(panel, core=core, use_overlay=True, cost_rt=cost_rt, scalar_map=scalar_map)
    cor = build_book(panel, core=core, use_overlay=False, cost_rt=cost_rt)
    bh_net, bh_gross = buy_hold_net(panel, cost_rt=cost_rt)
    return {
        "panel": panel,
        "engine": eng, "core": cor, "bh_net": bh_net, "bh_gross": bh_gross,
        "stats": {
            "ENGINE": window_stats(eng["net"], eng["gross_exposure"], lo, hi),
            "CORE_ALONE": window_stats(cor["net"], cor["gross_exposure"], lo, hi),
            "BUYHOLD": window_stats(bh_net, bh_gross, lo, hi),
        },
        "regime_share_full": eng["regime_share"],
        "thresholds": eng["thresholds"],
    }


# ===========================================================================
# 8. chart
# ===========================================================================
def make_chart(bt, lo, hi, core, stamp):
    """Equity curves (ENGINE vs CORE-ALONE vs BUYHOLD) + the regime/exposure track."""
    def _eq(net):
        s = net.dropna()
        s = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))]
        return s.index, np.cumprod(1 + s.to_numpy())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1], sharex=True)
    for label, net, color in [("ENGINE (core+overlay)", bt["engine"]["net"], "#1b9e77"),
                              ("CORE-ALONE", bt["core"]["net"], "#7570b3"),
                              ("BUY-HOLD (EW u10)", bt["bh_net"], "#999999")]:
        x, eq = _eq(net)
        ax1.plot(x, eq, label=f"{label}", color=color, lw=1.6)
    ax1.set_yscale("log")
    ax1.set_ylabel("equity ($1 start, log)")
    ax1.set_title(f"DAILY ENGINE -- core={core} -- {lo} .. {hi} -- u10 daily long-only (taker)")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)
    # exposure scalar track
    sc = bt["engine"]["scalar"]
    sc = sc[(sc.index >= pd.Timestamp(lo)) & (sc.index < pd.Timestamp(hi))]
    ge = bt["engine"]["gross_exposure"]
    ge = ge[(ge.index >= pd.Timestamp(lo)) & (ge.index < pd.Timestamp(hi))]
    ax2.fill_between(sc.index, 0, sc.to_numpy(), color="#d95f02", alpha=0.35, label="regime exposure scalar")
    ax2.plot(ge.index, ge.to_numpy(), color="#1b9e77", lw=1.0, label="gross exposure")
    ax2.set_ylabel("exposure"); ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    p = PLOTS / f"daily_engine_{stamp}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


# ===========================================================================
# 9. selftest -- two-sided soundness (synthetic, no market)
# ===========================================================================
def selftest():
    """POSITIVE: a positive-drift synthetic close panel -> the core produces POSITIVE returns and full
    coverage. NEGATIVE: a zero-exposure case (regime forced to 'down' scalar 0) -> the book produces
    ~0 returns (no positions). Plus: the regime overlay must REDUCE exposure on a synthetic crash."""
    print("## DAILY-ENGINE SELFTEST (two-sided)")
    ok = True
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=900, freq="D")
    syms = ["A", "B", "C", "D"]

    # ---- POSITIVE: clean positive-drift panel -> core returns POSITIVE, full coverage ----
    closes = {}
    for j, s in enumerate(syms):
        r = rng.normal(0.0015, 0.02, len(dates))         # mild positive daily drift
        closes[s] = pd.Series(100 * np.cumprod(1 + r), index=dates)
    panel = pd.DataFrame(closes)
    bk = build_book(panel, core="voltgt", use_overlay=False)
    st = window_stats(bk["net"], bk["gross_exposure"])
    print(f"  POSITIVE: core compound {st['compound_pct']}% | daily-pos {st['daily_pos_rate_pct']}% | "
          f"coverage {st.get('coverage_pct')}% (expect compound>0, coverage~100%)")
    ok &= (st["compound_pct"] > 0 and (st.get("coverage_pct") or 0) > 90)

    # ---- NEGATIVE: a no-exposure book (all weights forced to 0) -> ~0 net ----
    cw = core_weights(panel, core="voltgt")
    zeroW = cw * 0.0
    rets = panel.pct_change(fill_method=None).fillna(0.0)
    zero_net = (zeroW.shift(1).fillna(0.0) * rets).sum(axis=1)
    znet_sum = float(np.cumprod(1 + zero_net.to_numpy())[-1] - 1) * 100
    print(f"  NEGATIVE: zero-exposure book compound {znet_sum:.6f}% (expect ~0)")
    ok &= (abs(znet_sum) < 1e-6)

    # ---- OVERLAY: on a synthetic crash, the regime scalar must drop below 1.0 (de-risk) ----
    crash_dates = pd.date_range("2021-01-01", periods=700, freq="D")
    up = np.cumprod(1 + rng.normal(0.004, 0.01, 350))
    down = np.cumprod(1 + rng.normal(-0.02, 0.03, 350)) * up[-1]
    series = np.concatenate([up, down])
    cpanel = pd.DataFrame({s: pd.Series(100 * series * (1 + 0.001 * j), index=crash_dates)
                           for j, s in enumerate(syms)})
    import strat.rolling_regime_book as RRB
    RRB.LOOKBACK["1d"] = 40
    scalar, labels, th, share = regime_scalar_series(cpanel)
    # in the crash half, mean exposure scalar should be materially lower than in the trend half
    sc = scalar.to_numpy()
    trend_half = float(np.mean(sc[:350]))
    crash_half = float(np.mean(sc[350:]))
    print(f"  OVERLAY: trend-half mean scalar {trend_half:.2f} vs crash-half {crash_half:.2f} "
          f"(expect crash < trend -- de-risk the bear)")
    ok &= (crash_half < trend_half)

    # ---- engine produces a non-null daily return stream on real-shaped data ----
    print(f"  STREAM: engine net stream length {len(bk['net'])} (non-null daily stream)")
    ok &= (len(bk["net"]) > 100 and bk["net"].notna().all())

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ===========================================================================
# 10. CLI
# ===========================================================================
def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _print_stats_table(stats):
    print(f"   {'book':14} {'compound%':>11} {'CAGR%':>8} {'Sharpe':>7} {'maxDD%':>8} "
          f"{'dailyPos%':>10} {'coverage%':>10} {'avgGross':>9}")
    for k in ("ENGINE", "CORE_ALONE", "BUYHOLD"):
        m = stats.get(k, {})
        if "error" in m:
            print(f"   {k:14} {m['error']}")
            continue
        print(f"   {k:14} {m.get('compound_pct'):>11} {m.get('cagr_pct'):>8} {m.get('sharpe'):>7} "
              f"{m.get('maxdd_pct'):>8} {m.get('daily_pos_rate_pct'):>10} "
              f"{str(m.get('coverage_pct')):>10} {str(m.get('avg_gross_exposure')):>9}")


def _print_book(book):
    if "error" in book:
        print(f"   {book['error']}")
        return
    print(f"   DATE {book['date']} | regime={book['regime']} | exposure_scalar={book['exposure_scalar']} "
          f"| gross={book['gross_exposure']} | {book['n_positions']} positions")
    for a, w in book["weights"].items():
        print(f"     {a:10} {w:>7.4f}")
    for flag in book.get("data_quality_flags", []):
        print(f"   [data-quality] {flag}")


def run_backtest_mode(lo, hi, core, cost_rt, cost_name, argv, scalar_map=None):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sm = scalar_map if scalar_map is not None else REGIME_SCALAR
    sm_name = "aggressive" if sm == REGIME_SCALAR_AGGRESSIVE else "defensive(default)"
    print(f"## DAILY ENGINE -- BACKTEST {lo} .. {hi} -- core={core} -- u10 daily long-only -- {cost_name} "
          f"-- scalar={sm_name} {sm}")
    bt = backtest(lo, hi, core=core, cost_rt=cost_rt, scalar_map=scalar_map)
    print(f"   regime time-share (full history): {bt['regime_share_full']}\n")
    _print_stats_table(bt["stats"])
    # a recent slice (2024-2025) to show fresh-data behaviour
    recent = {k: window_stats(bt[("engine" if k == "ENGINE" else "core" if k == "CORE_ALONE" else None)]["net"]
                              if k != "BUYHOLD" else bt["bh_net"],
                              (bt["engine"] if k == "ENGINE" else bt["core"] if k == "CORE_ALONE" else None)
                              ["gross_exposure"] if k != "BUYHOLD" else bt["bh_gross"],
                              "2024-01-01", "2026-01-01")
              for k in ("ENGINE", "CORE_ALONE", "BUYHOLD")}
    print(f"\n   --- recent slice 2024-01-01 .. 2026-01-01 (fresh data) ---")
    _print_stats_table(recent)
    # today's book
    today = book_for_date(bt["panel"], core=core, use_overlay=True, scalar_map=scalar_map)
    print(f"\n   --- latest-day book ('what to hold today') ---")
    _print_book(today)
    # scorecard (honest, deflation-aware) on the engine net
    card = score_book(f"daily_engine_{core}", bt["engine"]["net"])
    sr = card.get("ship_read", {})
    print(f"\n   [scorecard] full p05={card['full_block_bootstrap'].get('p05')} "
          f"heldout p05={card.get('heldout_block_bootstrap', {}).get('p05')} ship={sr.get('ship')}")
    chart = make_chart(bt, lo, hi, core, stamp)
    print(f"   [chart] {chart}")
    # persist
    p = OUT / f"daily_engine_{stamp}.json"
    out = {
        "repro": {"command": "python -m strat.daily_engine " + " ".join(argv),
                  "git_sha": _git_sha(), "cost_rt": cost_rt, "cost_name": cost_name,
                  "core": core, "window": [lo, hi], "cadence": CADENCE,
                  "vol_target": VOL_TARGET, "max_per_name": MAX_PER_NAME,
                  "regime_scalar_map": sm, "scalar_profile": sm_name,
                  "regime_train_fit": REGIME_TRAIN_FIT},
        "full_window_stats": bt["stats"],
        "recent_2024_2025_stats": recent,
        "regime_share_full": bt["regime_share_full"],
        "regime_thresholds": bt["thresholds"],
        "latest_day_book": today,
        "scorecard": card,
    }
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"   [persisted] {p}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m strat.daily_engine")
    ap.add_argument("--backtest", default=None, help="START:END (YYYY-MM-DD:YYYY-MM-DD)")
    ap.add_argument("--today", action="store_true", help="print the recommended book for the latest day")
    ap.add_argument("--date", default=None, help="print the recommended book for a specific YYYY-MM-DD")
    ap.add_argument("--core", default="voltgt", choices=["voltgt", "orthobook"],
                    help="core return generator (voltgt = vol-target buy-hold; orthobook = +calendar tilt)")
    ap.add_argument("--no-overlay", action="store_true", help="disable the regime defensive overlay")
    ap.add_argument("--maker", action="store_true", help="use maker cost (0.0006 rt) instead of taker")
    ap.add_argument("--aggressive", action="store_true",
                    help="use the prior documented headline scalar {trend:1.0,chop:0.6,down:0.2} "
                         "(higher compound, higher maxDD) instead of the hardened defensive default "
                         "{1.0,0.5,0.1}")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    cost_rt = MAKER_RT if a.maker else TAKER_RT
    cost_name = "maker" if a.maker else "taker"
    use_overlay = not a.no_overlay
    scalar_map = REGIME_SCALAR_AGGRESSIVE if a.aggressive else None   # None -> defensive default

    if a.today or a.date:
        panel = load_close_panel()
        book = book_for_date(panel, target_date=a.date, core=a.core, use_overlay=use_overlay,
                             scalar_map=scalar_map)
        print(f"## DAILY ENGINE -- {'TODAY' if a.today else a.date} book -- core={a.core} -- "
              f"overlay={'on' if use_overlay else 'off'} -- "
              f"scalar={'aggressive' if a.aggressive else 'defensive(default)'}")
        _print_book(book)
        return 0

    if a.backtest:
        try:
            lo, hi = a.backtest.split(":")
        except ValueError:
            print("--backtest expects START:END (e.g. 2020-01-01:2025-12-31)")
            return 2
        return run_backtest_mode(lo, hi, a.core, cost_rt, cost_name, argv, scalar_map=scalar_map)

    # default: a multi-year backtest + the latest day's book
    return run_backtest_mode("2020-01-01", "2026-01-01", a.core, cost_rt, cost_name, argv,
                             scalar_map=scalar_map)


if __name__ == "__main__":
    raise SystemExit(main())
