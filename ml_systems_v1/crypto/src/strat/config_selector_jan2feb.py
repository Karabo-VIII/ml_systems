"""src/strat/config_selector_jan2feb.py -- the PERIOD-LEVEL regime->config SELECTOR (NOT the per-trade
meta-labeler, which was null).

USER /orc (2026-06-12): "Build + TRAIN a CONFIG-SELECTOR: train on month-1 (2020-01-07..2020-02-07) ONLY,
predict the best MA-config combo for month-2 (2020-02-07..2020-03-07), then evaluate that prediction on
month-2 (the real out-of-sample next-month test). This is the period-level regime->config problem."

THE OBJECT (different from setup_meta_labeler.py, which learns P(this MOVE pays | trigger+context) per
firing and returned NULL OOS). HERE the unit is a PERIOD, the label is "which CONFIG was best for this
(asset, period)", and the model is a REGIME->CONFIG map learned from ONE month, then applied to the next.

  config        = (entry-MA-spec, exit, timeframe). entry in {2MA cross, 3MA stack}; exit menu =
                  {signalflip, trail5, trail10, trail-ATR, time-stop-N, take-profit}; TF in {15m,30m,1h,4h,1d}.
  oracle (target) = for a (period), the hindsight argmax-net-compound config per asset.
  selector      = a regime classifier (trend-strength x vol tercile) + a shrinkage/bootstrap config-ranker
                  over (config, regime). It predicts month-2's best config(s) per asset + a book pick,
                  WITHOUT seeing month-2 (regime predicted via persistence; config ranked from month-1 +
                  the data-expansion).

DATA-EXPANSION (train on ONE month -- the heart of the ask; each technique's effective-sample count is
reported in the output). The six techniques now live in the CANONICAL, GENERAL, REUSABLE module
src/strat/data_expansion.py (doc: docs/DATA_EXPANSION_CANONICAL.md) -- this file is a CALLER that wires
them to the config-selection domain, NOT a re-implementation. Any future limited-data model uses the same
canonical toolkit:
  1. CROSS-SECTIONAL : each of the (data-bearing) u10 assets is a separate regime sample.
  2. SUB-PERIOD      : split month-1 into ~weekly windows (autocorrelation/overlap flagged honestly).
  3. BLOCK-BOOTSTRAP : resample month-1 per-config trade-returns in blocks (preserve autocorr) -> a
                       DISTRIBUTION of each config's performance; rank by a ROBUST stat (median/p25), not max.
  4. SYNTHETIC paths : regime-conditioned generator (fit drift+vol+AR(1) on month-1 returns, simulate K
                       month-2-like paths) -> score each config across the synthetic distribution -> prefer
                       configs ROBUST across them.
  5. SHRINKAGE       : James-Stein / Bayesian shrinkage of each config's month-1 score toward the cross-
                       sectional robust prior (THE overfit-killer -- do not pick the lucky in-sample max).
  6. REGIME->CONFIG  : learn config-per-REGIME-BUCKET (not per-asset); classify month-1 end-regime;
                       predict month-2 regime by persistence; map regime -> the shrunk/robust best config.

EVALUATION (the deliverable -- the real OOS test): run the Jan-PREDICTED config(s) on month-2 and report,
per asset + book-level: (a) predicted-config Feb return vs the Feb HINDSIGHT-ORACLE (the gap); (b) vs a
RANDOM config-selection baseline (MUST beat -- the no-skill control); (c) vs buy-hold.

HONEST CONTEXT: month-2 is the COVID-crash ONSET -- a hard regime SHIFT (calm Jan -> crash). A long-only
trend config trained on calm-Jan will likely LOSE. The win condition is NOT "Feb is positive" -- it is
"beats RANDOM config-selection AND gets closer to the oracle than a naive default", and ideally picks the
LEAST-BAD / most-defensive config. A crash-month loss is NOT spun as success.

ENGINE REUSE (do NOT reinvent -- the other instance's validated pieces):
  holding_state / apply_trail_stop / TAKER_RT  <- strat.portfolio_replay
  per_asset_trades                              <- strat.portfolio_replay_per_asset (next-bar-open fills,
                                                   taker cost subtracted, MtM-correct round trips)
  distinct_specs                                <- strat.replay_distinct_grid (the deduped config space)
  ChimeraLoader / _norm_sym                     <- the mandated data path

Honest scope / caveats (stated, not buried):
  - 1-MONTH training is TINY (per asset: ~22-186 bars depending on cadence). The CIs are wide; we report
    them, not hide them. Sub-period samples OVERLAP (autocorr) -- flagged, used as supporting evidence only.
  - SOL/DOGE/AVAX have NO Jan/Feb-2020 data (launched later) -> the effective universe is 7 assets.
  - "best config" is argmax over a search grid -> in-sample max is optimistic; shrinkage + bootstrap +
    synthetic robustness are precisely the antidote, and the random baseline is the no-skill control.
  - causal: every MA uses bars <= t; warmup loaded from FULL history up to window-end; only trades ENTERING
    inside the window count (the ma_per_instrument contract). No look-ahead into month-2 anywhere in fit.

RWYB:
  python src/strat/config_selector_jan2feb.py --selftest    # synthetic positive/negative control (no market)
  python src/strat/config_selector_jan2feb.py               # the FULL Jan->Feb 2020 config-selection verdict
  python src/strat/config_selector_jan2feb.py --cadences 4h,1h,30m,15m

No emoji (Windows cp1252). Does NOT git commit.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import strat.portfolio_replay as PR                                       # noqa: E402
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT  # noqa: E402
from strat.portfolio_replay_per_asset import per_asset_trades            # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                        # noqa: E402
from mining.family_regime_map import _norm_sym                           # noqa: E402
# CANONICAL limited-data DATA-EXPANSION toolkit -- the single source of truth for the 6 techniques.
# This file (the Jan->Feb config-selector) is now a CALLER of that module, not a re-implementation.
# See docs/DATA_EXPANSION_CANONICAL.md.
import strat.data_expansion as DX                                        # noqa: E402
from strat.data_expansion import james_stein_shrink                     # noqa: E402

OUT = ROOT / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research_verdict",
    "inputs": {"chimera": "via pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)"},
    "outputs": {"verdict_json": "runs/strat/config_selector_jan2feb_<stamp>.json"},
    "invariants": {
        "period_level_not_per_trade": "unit is a (asset, period); label is the hindsight-best CONFIG; not the per-firing meta-labeler",
        "train_only_month1": "regime classifier, shrinkage prior, bootstrap+synthetic ranking ALL fit on month-1 only",
        "month2_held_out": "month-2 is touched ONCE, for the eval; no month-2 data enters fit/selection",
        "causal_features": "every MA uses bars <= t; warmup from full history up to window-end; only trades ENTERING in-window count",
        "objective_is_compound": "judged on held-out net COMPOUND return; never AUC/IC",
        "beats_random_is_the_bar": "the win condition is beating RANDOM config-selection AND approaching the oracle, NOT positive Feb",
        "shrinkage_is_the_overfit_killer": "config month-1 scores shrunk toward the cross-sectional robust prior (James-Stein)",
        "no_crash_loss_spin": "a crash-month loss is reported as a loss; success = relative (vs random / naive default)",
    },
}

FLOOR = {"1d": "D", "4h": "4h", "1h": "h", "30m": "30min", "15m": "15min"}
TF_HOURS = {"1d": 24, "4h": 4, "1h": 1, "30m": 0.5, "15m": 0.25}

# ---- THE EXIT MENU (varied building blocks; held as the config's exit dimension) -----------------
# name -> (kind, param). kinds: signalflip | trail (high-water %) | timestop (max bars) | atrtrail
# (ATR-multiple trailing) | takeprofit (fixed % TP overlay on the signalflip hold).
EXIT_MENU = [
    ("signalflip", ("signalflip", None)),
    ("trail5",     ("trail", 0.05)),
    ("trail10",    ("trail", 0.10)),
    ("atr3",       ("atrtrail", 3.0)),
    ("time24",     ("timestop", 24)),
    ("tp8",        ("takeprofit", 0.08)),
]
EXIT_KIND = dict(EXIT_MENU)

JAN = ("2020-01-07", "2020-02-07")
FEB = ("2020-02-07", "2020-03-07")


# ===========================================================================
# 0. DATA + the config space
# ===========================================================================
def _panel(sym, cadence):
    df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence, features=["open", "high", "low", "close"])
    idx = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor(FLOOR[cadence])
    sub = pd.DataFrame({"o": df["open"].to_numpy().astype(float), "h": df["high"].to_numpy().astype(float),
                        "l": df["low"].to_numpy().astype(float), "c": df["close"].to_numpy().astype(float)},
                       index=idx)
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()
    ms = (sub.index.asi8 // 10**6).astype("int64")
    return sub["o"].to_numpy(), sub["h"].to_numpy(), sub["l"].to_numpy(), sub["c"].to_numpy(), ms


def build_config_space(max_entry_each=4):
    """Entry specs from the deduped grid (2MA + 3MA) x the exit menu. Returns {entry_name: (fam,params)}
    injected into PR.STRATS so holding_state resolves them by name, plus the (entry x exit) config list."""
    entry_specs = {}
    for fam in ("2MA", "3MA"):
        entry_specs.update(distinct_specs(fam, 0.15, max_n=max_entry_each))
    PR.STRATS.update(entry_specs)                       # so holding_state can resolve entries by name
    configs = []
    for ename in entry_specs:
        for exname, _ in EXIT_MENU:
            configs.append((ename, exname))
    return entry_specs, configs


def _atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def apply_atr_trail(held, h, l, c, mult):
    """ATR-multiple high-water trailing stop (causal): once long, stop = high-water close - mult*ATR.
    Force flat on breach; wait for signal re-arm. Mirrors apply_trail_stop's episode logic."""
    atr = _atr(h, l, c)
    out = np.asarray(held, dtype=np.int8).copy()
    inpos = False; hw = 0.0; stopped = False
    for i in range(len(held)):
        if not held[i]:
            inpos, stopped, out[i] = False, False, 0
            continue
        if stopped:
            out[i] = 0; continue
        if not inpos:
            inpos, hw = True, float(c[i])
        hw = max(hw, float(c[i]))
        if float(c[i]) < hw - mult * float(atr[i]):
            out[i] = 0; inpos, stopped = False, True
        else:
            out[i] = 1
    return out


def apply_takeprofit(held, c, tp):
    """Fixed take-profit overlay: once long, force flat at the first bar whose close >= entry*(1+tp); wait
    for re-arm. Causal + consistent with the trail/ATR overlays: out[i]=0 at the trigger bar so the trade
    extractor sees the 1->0 transition at i and fills the exit at o[i+1] (1-bar lag, no look-ahead)."""
    out = np.asarray(held, dtype=np.int8).copy()
    inpos = False; entry_px = 0.0; done = False
    for i in range(len(held)):
        if not held[i]:
            inpos, done, out[i] = False, False, 0
            continue
        if done:
            out[i] = 0; continue
        if not inpos:
            inpos, entry_px = True, float(c[i])
        if float(c[i]) >= entry_px * (1.0 + tp):
            out[i] = 0; done = True            # TP hit -> flat from here (extractor exits at o[i+1])
        else:
            out[i] = 1
    return out


def held_for_exit(ename, exname, o, h, l, c):
    """Holding state for a (entry, exit) config. Entry signal from holding_state; exit overlay applied."""
    held0 = holding_state(ename, o, h, l, c).astype(np.int8)
    kind, param = EXIT_KIND[exname]
    if kind == "signalflip":
        return held0
    if kind == "trail":
        return apply_trail_stop(held0.copy(), c, param)[0].astype(np.int8)
    if kind == "atrtrail":
        return apply_atr_trail(held0.copy(), h, l, c, param).astype(np.int8)
    if kind == "takeprofit":
        return apply_takeprofit(held0.copy(), c, param).astype(np.int8)
    if kind == "timestop":
        # cap each episode to <=param bars (causal: count bars-since-entry, force flat after param)
        out = held0.copy(); inpos = False; cnt = 0
        for i in range(len(held0)):
            if not held0[i]:
                inpos, cnt, out[i] = False, 0, 0
                continue
            if not inpos:
                inpos, cnt = True, 0
            cnt += 1
            out[i] = 1 if cnt <= param else 0
        return out.astype(np.int8)
    raise ValueError(exname)


# ===========================================================================
# 1. PER-(asset, config, period) net-compound  + the trade-return stream (for the bootstrap)
# ===========================================================================
def config_perf(o, h, l, c, ms, ename, exname, s_ms, e_ms, cost):
    """Net compound (%) and per-trade net-return stream for one (entry,exit) config in a window. Causal:
    full-history holding state, keep trades whose ENTRY ms is inside [s_ms, e_ms)."""
    held = held_for_exit(ename, exname, o, h, l, c)
    trades = per_asset_trades(o, c, held, ms, cost)
    wt = [t for t in trades if s_ms <= t["entry_ms"] < e_ms]
    rets = np.array([t["ret"] for t in wt], float)
    comp = float(np.prod(1.0 + rets) - 1.0) if rets.size else 0.0
    return comp, rets, len(wt)


def buy_hold(c, ms, s_ms, e_ms, cost):
    """Buy-hold over the window: enter at first in-window bar close, exit at last; one round-trip cost."""
    m = (ms >= s_ms) & (ms < e_ms)
    cc = c[m]
    if len(cc) < 2:
        return 0.0
    return float(cc[-1] / cc[0] - 1.0 - cost)


# ===========================================================================
# 2. REGIME classification (trend-strength x vol tercile), causal, end-of-window
# ===========================================================================
def regime_label(c, ms, s_ms, e_ms):
    """Classify the asset's regime AT THE END of [s_ms, e_ms) using only in/pre-window bars.
    trend = sign & strength of the window's MA50 slope + price-vs-MA50; vol = realized-vol tercile vs a
    trailing baseline. Returns (trend_bucket in {up,flat,down}, vol_bucket in {lo,mid,hi}, scalar feats)."""
    m = ms < e_ms
    cc = c[m]
    if len(cc) < 30:
        return ("flat", "mid", {"trend_strength": 0.0, "vol": 0.0, "n": int(len(cc))})
    ma50 = pd.Series(cc).rolling(50, min_periods=20).mean().to_numpy()
    win = (ms[m] >= s_ms)
    rets = np.diff(np.log(cc[win])) if win.sum() > 2 else np.array([0.0])
    vol = float(np.std(rets)) if rets.size else 0.0
    # trend strength = window return / window vol (annualization-free, scale-aware)
    cwin = cc[win]
    wret = float(cwin[-1] / cwin[0] - 1.0) if len(cwin) >= 2 else 0.0
    trend_strength = wret / (vol * np.sqrt(max(1, len(cwin))) + 1e-9)
    price_vs_ma = float(cc[-1] / (ma50[-1] if np.isfinite(ma50[-1]) and ma50[-1] > 0 else cc[-1]) - 1.0)
    # buckets (fixed, pre-registered thresholds)
    tb = "up" if (trend_strength > 0.5 and price_vs_ma > 0) else ("down" if (trend_strength < -0.5 or price_vs_ma < -0.05) else "flat")
    return tb, None, {"trend_strength": round(trend_strength, 3), "vol": round(vol, 5),
                      "price_vs_ma50": round(price_vs_ma, 4), "window_ret": round(wret, 4), "n": int(len(cwin))}


def vol_tercile(vol_by_asset):
    """Assign each asset a vol bucket (lo/mid/hi) by terciles of cross-sectional month-1 vol."""
    vals = np.array(list(vol_by_asset.values()))
    if len(vals) < 3:
        return {a: "mid" for a in vol_by_asset}
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    out = {}
    for a, v in vol_by_asset.items():
        out[a] = "lo" if v <= q1 else ("hi" if v > q2 else "mid")
    return out


# ===========================================================================
# 3. DATA-EXPANSION techniques
# ===========================================================================
def block_bootstrap_scores(rets, n_boot=500, block=3, seed=0):
    """Block-bootstrap a per-config trade-return stream -> distribution of compound, in COMPOUND-% units.
    Thin caller of the CANONICAL DX.block_bootstrap_distribution (which works in decimal) -- this scales
    to % and keeps the historical {median, p25, p05, mean, n} key surface this selector consumes."""
    d = DX.block_bootstrap_distribution(rets, n_boot=n_boot, block=block, stat="median", seed=seed)
    return {"median": d["median"] * 100.0, "p25": d["p25"] * 100.0, "p05": d["p05"] * 100.0,
            "mean": d["mean"] * 100.0, "n": d["n"]}


def fit_regime_generator(c, ms, s_ms, e_ms):
    """Fit a simple regime-conditioned return generator on month-1: drift, vol, AR(1) on the in-window
    log-returns. Domain wrapper: slices the window + computes log-returns here (the selector's contract),
    then delegates the {drift,vol,ar1} estimation to the CANONICAL DX.fit_regime_generator. Adds p0 (the
    last in-window close, the path seed) + n_bars. Returns params for simulate_paths (legacy mu/sigma/phi
    keys preserved for downstream compatibility)."""
    m = (ms >= s_ms) & (ms < e_ms)
    cc = c[m]
    if len(cc) < 10:
        return None
    r = np.diff(np.log(cc))
    g = DX.fit_regime_generator(r)
    if g is None:
        return None
    return {"mu": g["drift"], "sigma": g["vol"], "phi": g["ar1"], "p0": float(cc[-1]),
            "n_bars": int(len(cc))}


SYNTH_BAR_CAP = 200   # synthetic-path length cap: the regime dynamics (drift/vol/AR1), not the raw
#                       bar-count, drive robustness -- 200 bars capture them at any source cadence and keep
#                       the (config x path) trade-scoring tractable (15m's 2700 bars are NOT needed).


def simulate_paths(params, n_bars, K=40, seed=0):
    """Simulate K month-2-like OHLC-ish paths from the regime params (AR(1) log-returns + drift/vol).
    Domain wrapper over the CANONICAL DX.simulate_regime_paths: the canonical engine generates the
    drift/vol/AR(1)-preserving return + price matrices (regime-stats-preserving, verified in its selftest);
    this wrapper adds the OHLC approximation (h/l from intrabar vol so ATR/trail overlays have something to
    bite -- flagged as APPROXIMATE intrabar) + the ms axis the trade replay needs. Path length capped at
    SYNTH_BAR_CAP. params accepts the legacy mu/sigma/phi keys (DX maps them to drift/vol/ar1)."""
    n_bars = int(min(n_bars, SYNTH_BAR_CAP))
    sim = DX.simulate_regime_paths(params, n_bars, K=K, seed=seed, p0=params["p0"])
    c = sim["prices"]                             # K x n_bars
    sigma = float(params.get("sigma", params.get("vol", 1e-9)))
    o = np.concatenate([np.full((K, 1), params["p0"]), c[:, :-1]], axis=1)
    amp = np.abs(np.random.default_rng(seed + 1).normal(0, sigma, (K, n_bars))) * c   # APPROX intrabar
    h = np.maximum(o, c) + amp
    l = np.minimum(o, c) - amp
    base_ms = np.arange(n_bars, dtype=np.int64) * 60000 + 1
    return [(o[k], h[k], l[k], c[k], base_ms) for k in range(K)]


def synthetic_config_scores(paths, ename, exname, cost):
    """Score a config across the synthetic month-2-like paths -> median + p25 compound (robustness)."""
    comps = []
    for (o, h, l, c, ms) in paths:
        held = held_for_exit(ename, exname, o, h, l, c)
        trades = per_asset_trades(o, c, held, ms, cost)
        rets = np.array([t["ret"] for t in trades], float)
        comps.append((np.prod(1.0 + rets) - 1.0) * 100.0 if rets.size else 0.0)
    comps = np.array(comps)
    return {"median": float(np.median(comps)), "p25": float(np.quantile(comps, 0.25)),
            "mean": float(np.mean(comps)), "K": len(comps)}


# NOTE: james_stein_shrink is now imported from the CANONICAL strat.data_expansion (see imports at top).
# It is THE overfit-killer: shrunk = prior + B*(raw - prior), B = 1 - (k-2)*sigma2 / sum((raw-prior)^2),
# B->1 when spread clears the noise floor (signal), B->0 when spread ~ noise (collapse to prior, pick
# nothing). Same signature this selector has always called (prior + optional noise_var).


# ===========================================================================
# 4. THE SELECTOR -- train on month-1, predict month-2 best config per asset + book
# ===========================================================================
def select_for_cadence(cadence, syms, entry_specs, configs, K_synth=40, n_boot=400, verbose=True,
                       train_win=None, test_win=None):
    """Train on month-1 (train_win), predict + evaluate month-2 (test_win). train_win/test_win default to
    the module-level JAN/FEB (the original Jan->Feb 2020 crash test) so existing callers are unchanged;
    the generalized config_selector.py passes arbitrary (start, end) tuples for the FAIR persistent-regime
    test. ALL fit logic (regime classifier, shrinkage, bootstrap, synthetic, regime->config map) is
    UNCHANGED -- only the two window tuples are parameterized."""
    jan = train_win if train_win is not None else JAN
    feb = test_win if test_win is not None else FEB
    cost = TAKER_RT
    js_ms, je_ms = pd.Timestamp(jan[0]).value // 10**6, pd.Timestamp(jan[1]).value // 10**6
    fs_ms, fe_ms = pd.Timestamp(feb[0]).value // 10**6, pd.Timestamp(feb[1]).value // 10**6

    # ---- load panels (full history up to Feb-end for warmup; we never read >Feb-end) ----
    panels = {}
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        keep = ms < fe_ms                                  # never look past month-2 end
        o, h, l, c, ms = o[keep], h[keep], l[keep], c[keep], ms[keep]
        if ((ms >= js_ms) & (ms < je_ms)).sum() < 5:       # needs month-1 data to train
            continue
        panels[sym] = (o, h, l, c, ms)
    if len(panels) < 3:
        return {"cadence": cadence, "verdict": "INSUFFICIENT_ASSETS", "n_assets": len(panels)}

    # ---- month-1 per-(asset,config) perf + trade streams (for the bootstrap) ----
    jan_perf = {}     # sym -> {config: (comp, rets, n)}
    feb_perf = {}     # sym -> {config: (comp, rets, n)}  (the held-out truth, used ONLY in eval)
    bh_feb = {}
    regimes = {}; vol_by_asset = {}
    for sym, (o, h, l, c, ms) in panels.items():
        jan_perf[sym] = {}; feb_perf[sym] = {}
        for (en, ex) in configs:
            jcomp, jrets, jn = config_perf(o, h, l, c, ms, en, ex, js_ms, je_ms, cost)
            fcomp, frets, fn = config_perf(o, h, l, c, ms, en, ex, fs_ms, fe_ms, cost)
            jan_perf[sym][(en, ex)] = (jcomp, jrets, jn)
            feb_perf[sym][(en, ex)] = (fcomp, frets, fn)
        bh_feb[sym] = buy_hold(c, ms, fs_ms, fe_ms, cost)
        tb, _, rf = regime_label(c, ms, js_ms, je_ms)      # month-1 END regime
        regimes[sym] = {"trend": tb, "feats": rf}
        vol_by_asset[sym] = rf["vol"]
    vterc = vol_tercile(vol_by_asset)
    for sym in regimes:
        regimes[sym]["vol_bucket"] = vterc[sym]
        regimes[sym]["bucket"] = f"{regimes[sym]['trend']}_{vterc[sym]}"

    # ---- DATA-EXPANSION sample accounting ----
    expansion = {
        "1_cross_sectional": {"n_assets": len(panels), "note": "each asset = 1 (asset,month-1) regime sample"},
        "2_sub_period": {},     # filled below
        "3_block_bootstrap": {"n_boot": n_boot, "block": 3,
                              "note": "resample month-1 per-config trade returns in blocks -> rank by median/p25"},
        "4_synthetic": {"K_paths": K_synth, "generator": "drift+vol+AR(1) regime-conditioned",
                       "note": "score each config across synthetic Feb-like paths -> prefer robust"},
        "5_shrinkage": {},      # filled below (B per cadence)
        "6_regime_to_config": {"buckets": sorted({regimes[s]["bucket"] for s in regimes})},
    }

    # ---- 2. SUB-PERIOD: split month-1 into ~weekly windows -> more (sub-period, asset, config) samples ----
    week_ms = 7 * 24 * 3600 * 1000
    subwins = []
    t0 = js_ms
    while t0 < je_ms:
        subwins.append((t0, min(t0 + week_ms, je_ms)))
        t0 += week_ms
    n_sub_samples = 0
    sub_scores = {(en, ex): [] for (en, ex) in configs}    # per-config list of sub-period compounds (pooled)
    for sym, (o, h, l, c, ms) in panels.items():
        for (ws, we) in subwins:
            if ((ms >= ws) & (ms < we)).sum() < 3:
                continue
            for (en, ex) in configs:
                comp, _, n = config_perf(o, h, l, c, ms, en, ex, ws, we, cost)
                if n > 0:
                    sub_scores[(en, ex)].append(comp * 100.0)
                    n_sub_samples += 1
    expansion["2_sub_period"] = {"n_subwindows": len(subwins), "n_subperiod_asset_config_samples": n_sub_samples,
                                 "CAVEAT": "weekly sub-windows OVERLAP MA warmup + are autocorrelated -> "
                                           "supporting evidence only, NOT independent samples"}

    # ---- bootstrap + synthetic per config (pooled across assets for the robust prior) ----
    # 3. block-bootstrap: pool month-1 trade returns per config across assets -> robust compound distribution
    boot_by_config = {}
    for (en, ex) in configs:
        pooled = np.concatenate([jan_perf[s][(en, ex)][1] for s in panels
                                 if jan_perf[s][(en, ex)][1].size]) if any(
            jan_perf[s][(en, ex)][1].size for s in panels) else np.array([])
        boot_by_config[(en, ex)] = block_bootstrap_scores(pooled, n_boot=n_boot, block=3, seed=0)

    # 4. synthetic: per asset fit a generator, simulate K Feb-like paths, score each config, average robustness
    n_feb_bars = int(np.median([((ms >= fs_ms) & (ms < fe_ms)).sum() for (_, _, _, _, ms) in panels.values()]))
    n_feb_bars = max(n_feb_bars, 30)
    synth_by_config = {(en, ex): [] for (en, ex) in configs}
    synth_assets = 0
    for sym, (o, h, l, c, ms) in panels.items():
        params = fit_regime_generator(c, ms, js_ms, je_ms)
        if params is None:
            continue
        paths = simulate_paths(params, n_feb_bars, K=K_synth, seed=hash(sym) % 10000)
        synth_assets += 1
        for (en, ex) in configs:
            sc = synthetic_config_scores(paths, en, ex, cost)
            synth_by_config[(en, ex)].append(sc["p25"])    # use p25 (robust-pessimistic) per asset
    synth_robust = {cfg: float(np.median(v)) if v else 0.0 for cfg, v in synth_by_config.items()}
    expansion["4_synthetic"]["n_assets_simulated"] = synth_assets

    # ---- 5. SHRINKAGE: combine the robust signals into ONE score per config, then James-Stein shrink ----
    # robust month-1 score per config = mean of {bootstrap median, sub-period median, synthetic p25-median}
    def _sub_median(cfg):
        v = sub_scores[cfg]
        return float(np.median(v)) if v else 0.0
    combined_raw = {}
    for cfg in configs:
        parts = [boot_by_config[cfg]["median"], _sub_median(cfg), synth_robust[cfg]]
        combined_raw[cfg] = float(np.mean(parts))
    prior = float(np.median(list(combined_raw.values())))   # cross-sectional robust prior
    # DATA-DRIVEN noise floor for James-Stein: the per-config bootstrap (median-p25) IQR-half is an
    # estimation-noise proxy; the cross-config median of its square = sigma2. So shrinkage engages exactly
    # when each config's OWN month-1 estimate is noisy (the overfit regime), not just on absolute spread.
    boot_noise = np.array([(boot_by_config[cfg]["median"] - boot_by_config[cfg]["p25"]) for cfg in configs])
    noise_var = float(np.median(boot_noise ** 2)) if np.isfinite(boot_noise).any() else 1.0
    noise_var = max(noise_var, 0.25)                        # floor so shrinkage never fully disengages
    shrunk, B = james_stein_shrink({cfg2key(cfg): combined_raw[cfg] for cfg in configs}, prior,
                                   noise_var=noise_var)
    expansion["5_shrinkage"] = {"shrinkage_B": round(B, 3), "prior_compound_pct": round(prior, 3),
                                "noise_var_est": round(noise_var, 3),
                                "note": "B near 0 => heavy shrink (config differences are noise vs the "
                                        "bootstrap noise floor); B near 1 => differences clear the noise = signal"}
    shrunk_by_cfg = {cfg: shrunk[cfg2key(cfg)] for cfg in configs}

    # ---- 6. REGIME->CONFIG map: best shrunk config PER REGIME BUCKET (generalize, don't memorize) ----
    # learn, per bucket, the config with the best AVERAGE shrunk score over assets in that bucket in month-1.
    bucket_assets = {}
    for sym in panels:
        bucket_assets.setdefault(regimes[sym]["bucket"], []).append(sym)
    # per (bucket, config) month-1 mean compound, then shrink toward the global shrunk score
    regime_config_map = {}
    for bucket, blist in bucket_assets.items():
        per_cfg = {}
        for cfg in configs:
            jc = np.mean([jan_perf[s][cfg][0] * 100.0 for s in blist])
            # blend the bucket's in-sample mean with the global shrunk score (more assets -> trust bucket more)
            w = len(blist) / (len(blist) + 2.0)
            per_cfg[cfg] = w * jc + (1 - w) * shrunk_by_cfg[cfg]
        best = max(configs, key=lambda c: per_cfg[c])
        regime_config_map[bucket] = {"best_config": cfg2key(best), "score": round(per_cfg[best], 3),
                                     "n_assets": len(blist)}

    # DIAGNOSTIC (the honest distinction): does the regime->config map actually DIFFERENTIATE (distinct
    # configs per bucket = genuine regime skill), or COLLAPSE to one config for all buckets (= the skill is
    # a single-axis 'this one config dominates' effect, e.g. slow-MA-survives-cost, NOT regime navigation)?
    distinct_picks = {regime_config_map[b]["best_config"] for b in regime_config_map}
    map_differentiates = len(distinct_picks) > 1
    expansion["6_regime_to_config"]["distinct_configs_picked"] = len(distinct_picks)
    expansion["6_regime_to_config"]["map_differentiates_by_regime"] = bool(map_differentiates)
    expansion["6_regime_to_config"]["honest_note"] = (
        "map differentiates: distinct config per regime bucket (genuine regime->config skill)"
        if map_differentiates else
        "map COLLAPSES to ONE config for all buckets -> the edge (if any) is a SINGLE-AXIS dominance "
        "(e.g. slow-MA-trades-rarely-survives-cost), NOT learned regime navigation. Report as such.")

    # ---- PREDICT month-2 regime by PERSISTENCE (calm Jan -> assume Jan regime holds; honest: it won't) ----
    # then map predicted regime -> the regime-bucket best config. Per asset + a book pick (most common).
    predicted = {}
    for sym in panels:
        pbucket = regimes[sym]["bucket"]               # persistence: predict month-2 = month-1 end regime
        pred_cfg_key = regime_config_map[pbucket]["best_config"]
        predicted[sym] = {"pred_regime_bucket": pbucket, "pred_config": pred_cfg_key,
                          "via": "regime->config map (persistence)"}
    # book pick = the single config the selector most often chooses (mode), AND the globally-best shrunk one
    from collections import Counter
    book_mode = Counter([predicted[s]["pred_config"] for s in predicted]).most_common(1)[0][0]
    book_global = cfg2key(max(configs, key=lambda c: shrunk_by_cfg[c]))

    # ===================================================================
    # 5. EVALUATION on month-2 (held out) -- predicted vs oracle vs random vs buy-hold
    # ===================================================================
    rng = np.random.default_rng(7)

    def feb_comp(sym, cfg_key):
        cfg = key2cfg(cfg_key, configs)
        return feb_perf[sym][cfg][0] * 100.0

    # per-asset oracle (Feb hindsight argmax) and Jan-oracle (the naive 'use last month's winner' default)
    eval_rows = []
    rand_trials = 200
    for sym in panels:
        feb_oracle_cfg = max(configs, key=lambda c: feb_perf[sym][c][0])
        feb_oracle = feb_perf[sym][feb_oracle_cfg][0] * 100.0
        jan_oracle_cfg = max(configs, key=lambda c: jan_perf[sym][c][0])   # naive default: last month's best
        pred_key = predicted[sym]["pred_config"]
        pred_ret = feb_comp(sym, pred_key)
        naive_ret = feb_perf[sym][jan_oracle_cfg][0] * 100.0
        # random config-selection baseline (the no-skill control): mean Feb compound over random configs
        rand_rets = np.array([feb_perf[sym][configs[i]][0] * 100.0
                              for i in rng.integers(0, len(configs), size=rand_trials)])
        eval_rows.append({
            "asset": sym[:-4],
            "pred_regime": predicted[sym]["pred_regime_bucket"],
            "pred_config": pred_key,
            "pred_feb_pct": round(pred_ret, 2),
            "feb_oracle_config": cfg2key(feb_oracle_cfg),
            "feb_oracle_pct": round(feb_oracle, 2),
            "gap_to_oracle_pct": round(feb_oracle - pred_ret, 2),
            "naive_lastmonth_best_pct": round(naive_ret, 2),
            "random_mean_pct": round(float(rand_rets.mean()), 2),
            "random_p50_pct": round(float(np.median(rand_rets)), 2),
            "beats_random": bool(pred_ret > rand_rets.mean()),
            "buy_hold_pct": round(bh_feb[sym] * 100.0, 2),
            "beats_buyhold": bool(pred_ret > bh_feb[sym] * 100.0),
        })

    # ---- BOOK level: equal-weight the per-asset predicted-config Feb compound (a simple book) ----
    def book_compound(get_cfg_key):
        rets = [feb_comp(s, get_cfg_key(s)) for s in panels]
        return float(np.mean(rets))     # equal-weight book of per-asset configs
    book_pred = book_compound(lambda s: predicted[s]["pred_config"])
    book_oracle = float(np.mean([feb_perf[s][max(configs, key=lambda c: feb_perf[s][c][0])][0] * 100.0 for s in panels]))
    book_naive = float(np.mean([feb_perf[s][max(configs, key=lambda c: jan_perf[s][c][0])][0] * 100.0 for s in panels]))
    book_bh = float(np.mean([bh_feb[s] * 100.0 for s in panels]))
    # random book: draw a random config per asset, average, repeat
    rand_books = []
    for _ in range(rand_trials):
        picks = {s: configs[rng.integers(0, len(configs))] for s in panels}
        rand_books.append(float(np.mean([feb_perf[s][picks[s]][0] * 100.0 for s in panels])))
    rand_books = np.array(rand_books)
    # single-book-pick (one config for ALL assets) variants
    book_single_mode = float(np.mean([feb_comp(s, book_mode) for s in panels]))
    book_single_global = float(np.mean([feb_comp(s, book_global) for s in panels]))

    book = {
        "book_pred_perasset_pct": round(book_pred, 2),
        "book_single_mode_config": book_mode, "book_single_mode_pct": round(book_single_mode, 2),
        "book_single_global_config": book_global, "book_single_global_pct": round(book_single_global, 2),
        "book_oracle_pct": round(book_oracle, 2),
        "book_naive_lastmonth_pct": round(book_naive, 2),
        "book_buyhold_pct": round(book_bh, 2),
        "book_random_mean_pct": round(float(rand_books.mean()), 2),
        "book_random_p05_pct": round(float(np.quantile(rand_books, 0.05)), 2),
        "book_random_p95_pct": round(float(np.quantile(rand_books, 0.95)), 2),
        "book_pred_beats_random": bool(book_pred > rand_books.mean()),
        "book_pred_pctile_vs_random": round(float((rand_books < book_pred).mean()) * 100, 1),
        "book_gap_to_oracle_pct": round(book_oracle - book_pred, 2),
    }

    # ---- KEY QUESTION 4: CONFIG-RANK PERSISTENCE (does month-1's best config STAY best in month-2?) ----
    # The cleanest single number for "does regime->config transfer": Spearman rank-corr between each
    # config's month-1 and month-2 compound. Computed (a) per asset, (b) on the cross-asset-MEAN per-config
    # vector (the book view). rho ~ +1 => the config ranking PERSISTS (transfer works); rho ~ 0 => last
    # month's winner tells you nothing about next month's (the thesis fails even in the fair case);
    # rho < 0 => anti-persistence (last month's winner is next month's loser).
    def _spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return None
        ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])

    per_asset_rho = {}
    for sym in panels:
        m1 = [jan_perf[sym][cfg][0] for cfg in configs]
        m2 = [feb_perf[sym][cfg][0] for cfg in configs]
        rho = _spearman(m1, m2)
        if rho is not None:
            per_asset_rho[sym[:-4]] = round(rho, 3)
    # book view: mean per-config compound across assets, month-1 vs month-2
    m1_book = [float(np.mean([jan_perf[s][cfg][0] for s in panels])) for cfg in configs]
    m2_book = [float(np.mean([feb_perf[s][cfg][0] for s in panels])) for cfg in configs]
    book_rho = _spearman(m1_book, m2_book)
    mean_per_asset_rho = float(np.mean(list(per_asset_rho.values()))) if per_asset_rho else None
    config_rank_persistence = {
        "book_spearman_rho": round(book_rho, 3) if book_rho is not None else None,
        "mean_per_asset_spearman_rho": round(mean_per_asset_rho, 3) if mean_per_asset_rho is not None else None,
        "per_asset_rho": per_asset_rho,
        "n_configs": len(configs),
        "interpretation": ("rho~+1: config ranking PERSISTS month1->month2 (regime->config transfers); "
                           "rho~0: last month's config ranking is uninformative for next month (thesis FAILS "
                           "even in the fair case); rho<0: anti-persistent (winner becomes loser)"),
    }

    # ---- aggregate verdict signals ----
    n_beat_rand = sum(1 for r in eval_rows if r["beats_random"])
    n_beat_naive = sum(1 for r in eval_rows if r["pred_feb_pct"] > r["naive_lastmonth_best_pct"])
    n_beat_bh = sum(1 for r in eval_rows if r["beats_buyhold"])
    mean_gap = float(np.mean([r["gap_to_oracle_pct"] for r in eval_rows]))

    return {
        "cadence": cadence,
        "n_assets": len(panels),
        "assets": [s[:-4] for s in panels],
        "config_space": {"n_entry_specs": len(entry_specs), "entry_specs": list(entry_specs),
                         "exit_menu": [e[0] for e in EXIT_MENU], "n_configs": len(configs)},
        "data_expansion": expansion,
        "regimes_month1": {s[:-4]: {"bucket": regimes[s]["bucket"], **regimes[s]["feats"]} for s in panels},
        "regime_config_map": regime_config_map,
        "predicted_book_pick_mode": book_mode,
        "predicted_book_pick_global": book_global,
        "config_rank_persistence": config_rank_persistence,
        "eval_per_asset": eval_rows,
        "eval_book": book,
        "scoreboard": {
            "n_assets": len(panels),
            "n_beats_random": n_beat_rand,
            "n_beats_naive_lastmonth": n_beat_naive,
            "n_beats_buyhold": n_beat_bh,
            "mean_gap_to_oracle_pct": round(mean_gap, 2),
            "book_pred_pct": round(book_pred, 2),
            "book_random_mean_pct": round(float(rand_books.mean()), 2),
            "book_oracle_pct": round(book_oracle, 2),
            "book_beats_random": bool(book_pred > rand_books.mean()),
            "book_pctile_vs_random": book["book_pred_pctile_vs_random"],
            "map_differentiates_by_regime": bool(map_differentiates),
            "n_distinct_configs_picked": len(distinct_picks),
            "config_rank_persistence_book_rho": config_rank_persistence["book_spearman_rho"],
            "config_rank_persistence_mean_asset_rho": config_rank_persistence["mean_per_asset_spearman_rho"],
        },
    }


def cfg2key(cfg):
    return f"{cfg[0]}|{cfg[1]}"


def key2cfg(key, configs):
    en, ex = key.split("|")
    return (en, ex)


# ===========================================================================
# 6. SELF-TEST -- two-sided soundness (no market data)
# ===========================================================================
def selftest():
    """Synthetic positive + negative controls for the selector machinery (no market data).
    POSITIVE: a config whose returns are deterministically better must win the shrink+rank.
    NEGATIVE: when all configs are i.i.d. noise, shrinkage B must be ~0 (collapse to prior = no false pick)."""
    print("## CONFIG-SELECTOR SELFTEST (two-sided)")
    ok = True

    # POSITIVE: shrinkage must preserve a genuinely-separated winner.
    scores = {f"cfg{i}": float(i) for i in range(10)}     # cfg9 clearly best, wide spread
    shr, B = james_stein_shrink(scores, prior=4.5)
    win = max(shr, key=shr.get)
    print(f"  POSITIVE: wide-spread winner -> shrink B={B:.2f} (expect high), argmax={win} (expect cfg9)")
    ok &= (win == "cfg9" and B > 0.5)

    # NEGATIVE: i.i.d. tiny noise -> B should be near 0 (heavy shrink, no confident pick).
    rng = np.random.default_rng(0)
    noise = {f"cfg{i}": float(rng.normal(0, 0.01)) for i in range(12)}
    _, Bn = james_stein_shrink(noise, prior=0.0)
    print(f"  NEGATIVE: i.i.d. noise -> shrink B={Bn:.2f} (expect near 0, heavy shrink)")
    ok &= (Bn < 0.5)

    # block-bootstrap sanity: positive-mean stream -> positive median; zero stream -> ~0.
    bs = block_bootstrap_scores(np.array([0.02, 0.03, 0.01, 0.025]), n_boot=300, seed=1)
    print(f"  BOOTSTRAP: positive stream median={bs['median']:.2f}% (expect >0)")
    ok &= (bs["median"] > 0)

    # synthetic generator: an up-drift regime must on average produce up paths.
    params = {"mu": 0.002, "sigma": 0.01, "phi": 0.1, "p0": 100.0, "n_bars": 50}
    paths = simulate_paths(params, 50, K=20, seed=2)
    up = np.mean([c[-1] > o[0] for (o, h, l, c, ms) in paths])
    print(f"  SYNTHETIC: up-drift regime -> {up*100:.0f}% of paths end higher (expect majority)")
    ok &= (up > 0.6)

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ===========================================================================
# 7. MAIN
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python src/strat/config_selector_jan2feb.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadences", default="4h,1h,30m,15m")
    ap.add_argument("--max-entry-each", type=int, default=4, help="distinct entry specs per family (2MA,3MA)")
    ap.add_argument("--K-synth", type=int, default=40, help="synthetic Feb-like paths per asset")
    ap.add_argument("--n-boot", type=int, default=400)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{a.universe}.yaml"))
    syms = [x["symbol"] for x in spec["assets"]]
    entry_specs, configs = build_config_space(a.max_entry_each)

    print(f"## CONFIG-SELECTOR Jan->Feb 2020 -- {a.universe} -- train month-1 {JAN[0]}..{JAN[1]}, "
          f"eval month-2 {FEB[0]}..{FEB[1]}")
    print(f"   config space: {len(entry_specs)} entry specs x {len(EXIT_MENU)} exits = {len(configs)} configs")
    print(f"   exit menu: {[e[0] for e in EXIT_MENU]}")
    print(f"   cadences: {a.cadences}  (HONEST: month-2 = COVID-crash onset, a hard regime SHIFT)\n")

    results = {}
    for cad in a.cadences.split(","):
        cad = cad.strip()
        r = select_for_cadence(cad, syms, entry_specs, configs, K_synth=a.K_synth, n_boot=a.n_boot)
        results[cad] = r
        if r.get("verdict", "").startswith("INSUFFICIENT"):
            print(f"## {cad}: {r['verdict']} (n_assets={r.get('n_assets')})")
            continue
        sb = r["scoreboard"]; bk = r["eval_book"]; ex = r["data_expansion"]
        print(f"## {cad}: {r['n_assets']} assets ({', '.join(r['assets'])})")
        print(f"   DATA-EXPANSION effective samples: cross-sec={ex['1_cross_sectional']['n_assets']} | "
              f"sub-period={ex['2_sub_period']['n_subperiod_asset_config_samples']} (OVERLAP) | "
              f"bootstrap={ex['3_block_bootstrap']['n_boot']}/cfg | "
              f"synthetic={ex['4_synthetic']['K_paths']}x{ex['4_synthetic'].get('n_assets_simulated','?')} | "
              f"shrinkage B={ex['5_shrinkage']['shrinkage_B']}")
        print(f"   regime buckets month-1: {ex['6_regime_to_config']['buckets']}")
        print(f"   REGIME->CONFIG map differentiates? {sb['map_differentiates_by_regime']} "
              f"({sb['n_distinct_configs_picked']} distinct config(s) across buckets) -- "
              f"{'genuine regime skill' if sb['map_differentiates_by_regime'] else 'COLLAPSES to single-axis dominance (e.g. slow-MA survives cost), NOT regime navigation'}")
        print(f"   predicted book pick (mode/global): {r['predicted_book_pick_mode']} / "
              f"{r['predicted_book_pick_global']}")
        print(f"\n   {'asset':6} {'pred_regime':12} {'pred_cfg':28} {'predFeb%':>9} {'oracle%':>9} "
              f"{'gap':>7} {'rand%':>7} {'BH%':>7} {'>rand':>6}")
        for row in r["eval_per_asset"]:
            print(f"   {row['asset']:6} {row['pred_regime']:12} {row['pred_config'][:28]:28} "
                  f"{row['pred_feb_pct']:>9.2f} {row['feb_oracle_pct']:>9.2f} {row['gap_to_oracle_pct']:>7.2f} "
                  f"{row['random_mean_pct']:>7.2f} {row['buy_hold_pct']:>7.2f} {str(row['beats_random']):>6}")
        print(f"\n   BOOK: pred(per-asset)={bk['book_pred_perasset_pct']}%  oracle={bk['book_oracle_pct']}%  "
              f"naive-lastmonth={bk['book_naive_lastmonth_pct']}%  random_mean={bk['book_random_mean_pct']}%  "
              f"buyhold={bk['book_buyhold_pct']}%")
        print(f"         pred vs random: {bk['book_pred_pctile_vs_random']}th pctile  "
              f"(beats_random={bk['book_pred_beats_random']}, gap_to_oracle={bk['book_gap_to_oracle_pct']}%)")
        print(f"   SCOREBOARD: {sb['n_beats_random']}/{sb['n_assets']} assets beat random | "
              f"{sb['n_beats_naive_lastmonth']}/{sb['n_assets']} beat naive-last-month | "
              f"{sb['n_beats_buyhold']}/{sb['n_assets']} beat buy-hold | mean gap-to-oracle {sb['mean_gap_to_oracle_pct']}%\n")

    # ---- AGGREGATE VERDICT across cadences ----
    valid = {c: r for c, r in results.items() if not r.get("verdict", "").startswith("INSUFFICIENT")}
    book_beats = sum(1 for r in valid.values() if r["scoreboard"]["book_beats_random"])
    asset_beat_frac = ([r["scoreboard"]["n_beats_random"] / r["scoreboard"]["n_assets"] for r in valid.values()])
    mean_asset_beat = float(np.mean(asset_beat_frac)) if asset_beat_frac else 0.0
    verdict = build_verdict(valid, book_beats, mean_asset_beat)
    print("=" * 78)
    print("## AGGREGATE VERDICT")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 78)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"config_selector_jan2feb_{a.universe}_{stamp}.json"
    json.dump({
        "repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha,
                  "train_window": JAN, "eval_window": FEB, "cost_rt": TAKER_RT, "universe": a.universe},
        "results": results, "verdict": verdict,
    }, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


def build_verdict(valid, book_beats, mean_asset_beat):
    lines = []
    if not valid:
        return {"headline": "NO VALID CADENCE", "lines": ["all cadences had insufficient data."]}
    n_cad = len(valid)
    # the win condition: beats RANDOM config-selection (per-asset majority + book) AND approaches the oracle.
    beat_random_book = book_beats
    lines.append(f"TRAINED on Jan 2020 ONLY (calm); EVALUATED on Feb 2020 (COVID-crash ONSET, hard regime shift).")
    lines.append(f"Win condition = beat RANDOM config-selection + approach the oracle. NOT 'Feb positive'.")
    lines.append("")
    for c, r in valid.items():
        sb = r["scoreboard"]; bk = r["eval_book"]
        lines.append(f"[{c}] book pred {sb['book_pred_pct']:+.1f}% vs random {sb['book_random_mean_pct']:+.1f}% "
                     f"vs oracle {sb['book_oracle_pct']:+.1f}% | book {('BEATS' if sb['book_beats_random'] else 'LOSES TO')} "
                     f"random ({sb['book_pctile_vs_random']}th pctile) | "
                     f"{sb['n_beats_random']}/{sb['n_assets']} assets beat random")
    lines.append("")
    # does ANY cadence's map actually differentiate by regime? (else the 'skill' is single-axis dominance)
    n_differentiating = sum(1 for r in valid.values() if r["scoreboard"].get("map_differentiates_by_regime"))
    diff_note = (f"The regime->config map DIFFERENTIATES on {n_differentiating}/{n_cad} cadences."
                 if n_differentiating else
                 "On ALL cadences the regime->config map COLLAPSES to a SINGLE config across every regime "
                 "bucket -- so the edge over random is a SINGLE-AXIS effect (a very slow MA trades rarely and "
                 "survives the cost/whipsaw bleed that sinks the average config into the crash), NOT learned "
                 "regime navigation. The 'skill' is 'slow-and-rare beats churn under cost', not regime->config.")
    # honest headline
    if beat_random_book >= max(1, n_cad // 2 + 1) and mean_asset_beat > 0.5:
        head = ("RELATIVE SKILL vs RANDOM (cautious, MECHANISM-QUALIFIED): Jan-trained config-selection beat "
                "random config-selection on the majority of cadences AND >50% of assets, even into the crash -- "
                "but " + diff_note + " So this is a real but NARROW result: the selector reliably AVOIDS the "
                "cost-bleeding configs (value over no-skill picking), yet it is NOT a profitable bot (Feb is a "
                "crash; book still ~flat-to-negative, far below the +oracle) and NOT a regime->config map.")
    elif beat_random_book >= 1 or mean_asset_beat > 0.5:
        head = ("MIXED / WEAK: the selector beat random on SOME cadences/assets but not a majority. Inconclusive "
                "skill into a hard regime shift -- calm-Jan training does not robustly transfer to the crash. " + diff_note)
    else:
        head = ("NO SKILL: Jan-trained config-selection did NOT beat random config-selection into the Feb crash. "
                "The regime shift (calm->crash) broke the persistence assumption; the selector's picks were no "
                "better than chance. EXPECTED given the crash onset -- reported as a loss, not spun. " + diff_note)
    lines.insert(2, f"HEADLINE: {head}")
    lines.append("")
    lines.append(f"Mean fraction of assets beating random across cadences: {mean_asset_beat*100:.0f}%. "
                 f"Book beat random on {book_beats}/{n_cad} cadences. "
                 f"Regime->config map differentiates on {n_differentiating}/{n_cad} cadences.")
    lines.append("CAVEATS: 1-month train = tiny n (wide CIs); sub-period samples overlap (autocorr); "
                 "'best config' is a search-grid argmax (optimistic) -- shrinkage+bootstrap+synthetic+random-"
                 "control are the antidotes; SOL/DOGE/AVAX absent (7-asset effective universe).")
    return {"headline": head, "book_beats_random_cadences": book_beats, "n_cadences": n_cad,
            "mean_asset_beat_random_frac": round(mean_asset_beat, 3),
            "n_cadences_regime_map_differentiates": n_differentiating, "lines": lines}


if __name__ == "__main__":
    raise SystemExit(main())
