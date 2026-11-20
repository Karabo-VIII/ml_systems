"""src/strat/setup_meta_labeler.py -- the CONDITIONAL setup-edge META-LABELER over a CONFIG GRID
(FULL-MECHANISM VERDICT build, 2026-06-11).

WHAT THIS IS (docs/STRATEGY_AGENT_DESIGN_2026_06_11.md S4/S7): the generalization of
`src/mining/mover_metalabel.py` from a single FIXED trigger to the USER'S FULL MECHANISM -- an agent
that NAVIGATES the MA-config search space (config-navigation), conditions ENTRY on causal context, and
exits the multi-candle MOVE by the user's RISK POLICY. It learns

    P( the multi-candle MOVE pays net cost | a config-X MA-cross fired AND causal context C at the cross )

POOLED ACROSS A GRID OF MA-CROSS CONFIGS, with THE CONFIG PARAMS THEMSELVES AS FEATURES. The model thus
learns WHICH config x context combos pay -- "navigating the config search space non-linearly", exactly
the user's ask. It trades ONLY the top-tau predicted subset (a binary ENTRY GATE, never a size
multiplier). This is the ONE ML use the project did NOT kill (D16: "ML-as-alpha dead; meta-labeler
survives"). It is NOT a forecaster (IC banned, D13), NOT predict-then-threshold (Kronos trap), NOT RL.

THE DELIVERABLE IS A VERDICT, NOT A BOT. For each cadence the run answers: does the meta-labeler's
selected subset beat (a) the UNCONDITIONAL pooled trigger AND (b) the regime+membership-matched random-
entry firewall, on UNSEEN, with OOS AUC>0.55, PBO<0.10, >=8/10 seeds OOS-positive?

HONEST FRAMING (THE CENTRAL CAVEAT -- stated before any result): each COMPONENT of this mechanism is
ALREADY null SEPARATELY in the dead-list:
  - config-navigation: MA-oracle / verify_dna_finding -- NO learnable per-move config at 30m..1d, the
    "DNA" is REGIME-driven not asset-feature (D-oracle decomposition).
  - conditional-entry : the EMA12x26 slice (this tool's prior build) + mover_metalabel -- OOS AUC 0.51-0.52.
  - exit-timing       : exit_capture_proxy / D61 -- smart exit-timing NULL on daily breakouts.
This run tests whether the JOINT mechanism (config-nav + conditional-entry + the user's 5% trailing risk
exit + the FINER 15m cadence) finds what the separate component-tests missed. An honest NULL is the
EXPECTED, HIGH-VALUE answer: it would mean the JOINT mechanism is genuinely refuted at 15m..4h with
INTERNAL features -- not just its parts. A non-null at ANY cadence = the FIRST measured edge -> champion-
gate it. **The config-grid is a BIG multiple-comparisons surface, so PBO is the CENTRAL risk: if PBO is
high the apparent edge is CONFIG-SEARCH OVERFIT, and the verdict must say so.**

PRE-REGISTRATION (fixed BEFORE any result was read):
  CONFIG-NAVIGATION TRIGGER GRID (the key addition):
      MA-cross over a GRID: fast in {5,8,10,12,20}, slow in {20,26,50,100,200} (fast<slow), type in
      {SMA, EMA}. Each (fast,slow,type) is a PAST-ONLY cross trigger (fast MA crosses ABOVE slow MA,
      confirmed at close, filled opens[t+1] by the harness -- Pattern T banned). ALL firings across ALL
      configs are POOLED into ONE dataset; the config params enter AS FEATURES.
  CADENCES (the standing HARD RULE -- never default one): 15m, 30m, 1h, 4h.
      1d is run only on demand (--cadence 1d); sub-15m needs the 1m dataset (a SEPARATE effort -- the
      1m liq/mover data exists but the MA-cross-on-1m pooling is out of this run's scope; STATED).
  EXIT -- THE USER'S RISK POLICY (held FIXED across all configs/cadences to isolate the entry+config edge):
      "risk 5%, track the current move and price" = a 5% TRAILING stop: initial stop at entry-5%
      (sl_pct=0.05), then trail to stay 5% below the running HIGH-since-entry (trail_pct=0.05; the
      ExitPolicy takes the TIGHTER/higher of the two for a long, so once price rises 5% the trail
      dominates the fixed stop -- "tracks the move"), exit on stop-hit OR a max-hold time-stop. Honors
      next-bar-open fill + pessimistic intrabar + prior-bar-hwm trail (setup_harness leak guards).
  LABEL: y = (net_pnl > 0) from setup_harness, taker 0.0024 round-trip (the SETUP OUTCOME across the
      MOVE). ALSO reported at net>cost-band (+0.0024) for D09 label-stability.
  FEATURES (ALL causal <= cross close; per-(asset,feature) z-scored from TRAIN STATS ONLY):
      CONFIG PARAMS (the navigation features): fast_len, slow_len, fast_slow_ratio, is_ema (type
                     one-hot), cfg_past_hitrate (the config's OWN past-only win-rate up to the firing).
      MA geometry  : fast/slow MA slopes, spread, price-to-fast, price-to-slow, stack, bars-since-cross.
      Chimera f41  : norm_funding, fund_rate_mean, norm_oi_change, liq_long_z30, liq_short_z30,
                     wh_whale_net_usd, norm_whale, norm_flow_imbalance, norm_vpin, norm_kyle_lambda,
                     regime_label  (read via pipeline.chimera_loader.ChimeraLoader -- the mandated path).
      Regime       : SMA-200 side (past-only) + past-only realized-vol tercile.
      Cross-asset  : BTC-relative 24h return + BTC-relative position vs its own SMA-50 at the cross.
      (WM-as-a-feature DEFERRED -- V1.1 re-training. v1 is built WITHOUT it; the hook `_wm_belief_features`
       is a clear no-op stub to add forecast_bundle moments [E[r],Std[r],P(r<0)] later as BELIEF features.)
  MODEL: LogisticRegression (LINEAR) -> HistGradientBoosting (NON-LINEAR; native NaN; interpretable-first
      escalation, escalate only on VAL lift). Binary ENTRY GATE at tau = 67th TRAIN percentile. Model,
      z-stats, imputation medians AND tau fit on TRAIN ONLY. Asymmetric loss pre-registered (false-LONG
      costs FALSE_LONG_PENALTY x a missed-winner -> class_weight on y=1, fixed, never tuned on UNSEEN).
  SPLIT: chronological 50/20/20/10 via the canonical WindowSpec date boundaries + a 400-bar purge buffer
      before each boundary. UNSEEN touched ONCE (for the verdict only).

THE GAUNTLET (run in order on the SELECTED-subset trade stream; each is a KILL gate; all already built):
  (1) OOS AUC > 0.55 floor + monotone TRAIN decile lift (DIAGNOSTIC only, never the objective).
  (2) random-entry FIREWALL (firewall.py), regime_matched=True AND membership_matched=True -- selected
      compound must beat the cost-matched null on ALL held-out windows.
  (3) shuffled-market control (shuffled_market_control.py, perm/block primary) -- edge must COLLAPSE.
  (4) PBO via CSCV (pbo_cscv.py, ship PBO<0.10) -- THE CENTRAL GATE: the config grid is a big multiple-
      comparisons surface; PBO is what catches config-overfit.
  (5) >=8/10 seeds OOS-positive + bootstrap p05>0.
  (6) battery Lens A/B/C (battery.py).
  (7) NO-SKILL controls (a RANDOM and a CONFIG-ONLY meta-labeler must FAIL) + POSITIVE control (a
      synthetic edge must SHIP) -- two-sided soundness.
  (8) per-regime report (bull/bear by SMA-200 side; NEVER aggregate).

OBJECTIVE = held-out COMPOUND of the selected subset, net of taker 0.0024, LONG-only spot lev=1.
NEVER AUC/IC (AUC>0.55 is a within-model diagnostic GATE only). UNSEEN touched once. No emoji (cp1252).

RWYB:
  python src/strat/setup_meta_labeler.py --selftest            # two-sided gate-logic check (no market data)
  python src/strat/setup_meta_labeler.py                       # the FULL-MECHANISM VERDICT (15m 30m 1h 4h on u10)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from strat.setup_harness import SetupHarness, ExitPolicy          # noqa: E402
from strat.firewall import random_entry_null                       # noqa: E402
from strat.battery import evaluate as battery_evaluate, block_bootstrap_p05_p95  # noqa: E402
from strat.pbo_cscv import pbo_cscv                                # noqa: E402
from agents._shared.shuffled_market_control import shuffled_market_control  # noqa: E402
from wealth_bot.harness import WindowSpec                          # noqa: E402

OUT = ROOT / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research_verdict",
    "inputs": {"chimera": "via pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)"},
    "outputs": {"verdict_json": "runs/strat/setup_metalabel_FULLMECH_verdict_<stamp>.json"},
    "invariants": {
        "ml_as_metalabeler_only": "classifier GATES a fixed causal MA-cross-grid trigger; never generates signals",
        "config_navigation": "firings POOLED across an MA-config grid; config params enter AS FEATURES (not tuned)",
        "train_only_fit": "model, z-stats, imputation medians, AND tau ALL fit from TRAIN rows only",
        "causal_features": "every feature computed from data <= the cross close (no look-ahead); cfg_past_hitrate is expanding past-only",
        "unseen_untouched": "no UNSEEN rows enter feature z-stats, model fit, imputation, or tau",
        "objective_is_compound": "judged on held-out COMPOUND; AUC>0.55 is a within-model diagnostic ONLY",
        "binary_entry_gate": "top-tau subset is ENTERED; this is NOT a position-size multiplier",
        "asymmetric_loss_preregistered": "false-LONG penalized > missed-winner via fixed class_weight",
        "purge_gap": "firings within 400 bars before a window boundary are dropped (normalization purge)",
        "pbo_is_central_gate": "the config grid is a multiple-comparisons surface; high PBO => config-search overfit",
        "no_predicted_return_as_reward": "the (deferred) WM enters as a BELIEF feature, never as label/reward",
    },
}

# ---- PRE-REGISTERED CONSTANTS (fixed before any result) ---------------------
# CONFIG-NAVIGATION GRID: fast in {5,8,10,12,20}, slow in {20,26,50,100,200} (fast<slow), type in {SMA,EMA}
MA_FAST_GRID = [5, 8, 10, 12, 20]
MA_SLOW_GRID = [20, 26, 50, 100, 200]
MA_TYPES = ["SMA", "EMA"]
MA_CONFIG_GRID = [(f, s, t) for t in MA_TYPES for f in MA_FAST_GRID for s in MA_SLOW_GRID if f < s]

# THE USER'S RISK EXIT: 5% trailing stop (initial stop entry-5%, trail 5% below running high) + time-stop.
RISK_SL_PCT = 0.05       # initial hard stop at entry - 5% ("risk 5%")
RISK_TRAIL_PCT = 0.05    # trail 5% below the running high-since-entry ("track the current move")

SMA_REGIME = 200
VOL_WINDOW = 30          # past-only realized-vol window for the regime tercile
TAKER_RT = 0.0024
TAU_Q = 0.67             # operating point: trade the top ~1/3 by predicted P
FALSE_LONG_PENALTY = 1.5  # asymmetric loss: a false-LONG costs 1.5x a missed-winner (class_weight on y=1)
PURGE_BARS = 400         # canonical normalization purge buffer before each window boundary
AUC_FLOOR = 0.55         # discrimination DIAGNOSTIC floor
SEED_GRID = list(range(10))   # 10 seeds; gauntlet floor = >=8 OOS-positive
SEEDS_OOS_POS_FLOOR = 8
PBO_SHIP = 0.10
SHUFFLE_GENUINE_FRAC = 0.20

# per-cadence pre-registered max-hold (a multi-day MOVE at each cadence)
MAX_HOLD = {"15m": 96, "30m": 64, "1h": 48, "4h": 24, "1d": 12}

# chimera f41 causal context columns (point-in-time bar values; native-NaN tolerant)
CHIMERA_FEATS = [
    "norm_funding", "fund_rate_mean", "norm_oi_change", "liq_long_z30", "liq_short_z30",
    "wh_whale_net_usd", "norm_whale", "norm_flow_imbalance", "norm_vpin", "norm_kyle_lambda",
    "regime_label",
]
# config-navigation features (the key addition -- the model learns which config x context pays)
CONFIG_FEATS = ["cfg_fast_len", "cfg_slow_len", "cfg_fast_slow_ratio", "cfg_is_ema", "cfg_past_hitrate"]
MA_FEATS = ["ema_fast_slope", "ema_slow_slope", "ma_spread", "price_to_fast", "price_to_slow",
            "ma_stack", "bars_since_cross"]
REGIME_FEATS = ["above_sma200", "vol_tercile"]
XASSET_FEATS = ["btc_r24h", "btc_vs_sma50"]
FEATURES = CONFIG_FEATS + MA_FEATS + CHIMERA_FEATS + REGIME_FEATS + XASSET_FEATS

# canonical date split (same boundaries setup_harness / firewall label trades by)
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def _u10_symbols() -> list[str]:
    import yaml
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / "u10.yaml"))
    return [a["symbol"] for a in spec["assets"]]


def _ma(x: np.ndarray, span: int, kind: str) -> np.ndarray:
    """Past-only MA (value at t uses bars <= t). SMA = rolling mean; EMA = ewm."""
    s = pd.Series(x)
    if kind == "EMA":
        return s.ewm(span=span, adjust=False).mean().to_numpy()
    return s.rolling(span, min_periods=span).mean().to_numpy()


# ===========================================================================
# 1. BUILD a per-asset OHLC frame with ALL chimera/regime/cross-asset causal context (config-INDEPENDENT)
# ===========================================================================
def build_base_frame(loader, sym: str, cadence: str, btc_ctx: Optional[dict]):
    """Load chimera once per asset; build the config-INDEPENDENT causal context (chimera f41, regime,
    cross-asset, price arrays). The per-config MA geometry + trigger are layered on top per config."""
    cols = (["timestamp", "open", "high", "low", "close"] + CHIMERA_FEATS)
    try:
        g = loader.load(sym, cadence=cadence, features=cols)
    except FileNotFoundError:
        return None
    d = g.to_dict(as_series=False)
    ts = np.asarray(d["timestamp"], dtype=np.int64)
    o = np.asarray(d["open"], float); h = np.asarray(d["high"], float)
    lo = np.asarray(d["low"], float); c = np.asarray(d["close"], float)
    n = len(c)
    if n < SMA_REGIME + 80:
        return None
    date = pd.to_datetime(ts, unit="ms")

    sma200 = pd.Series(c).rolling(SMA_REGIME, min_periods=SMA_REGIME).mean().to_numpy()
    ret1 = np.concatenate([[0.0], c[1:] / c[:-1] - 1.0])
    vol30 = pd.Series(ret1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy()
    vt = pd.Series(vol30).expanding(min_periods=VOL_WINDOW * 2).rank(pct=True).to_numpy()
    vol_tercile = np.where(np.isnan(vt), np.nan, np.floor(np.clip(vt, 0, 0.999) * 3))
    above_sma200 = np.where(np.isnan(sma200), np.nan, (c > sma200).astype(float))

    df = pd.DataFrame({"date": date, "open": o, "high": h, "low": lo, "close": c})
    for f in CHIMERA_FEATS:
        df[f] = np.asarray(d[f], float) if f in g.columns else np.nan
    df["above_sma200"] = above_sma200
    df["vol_tercile"] = vol_tercile
    df["btc_r24h"] = np.nan
    df["btc_vs_sma50"] = np.nan
    if btc_ctx is not None:
        idx = np.searchsorted(btc_ctx["ts"], ts, side="right") - 1   # last BTC bar <= this bar (causal)
        ok = idx >= 0
        df.loc[ok, "btc_r24h"] = btc_ctx["r24h"][idx[ok]]
        df.loc[ok, "btc_vs_sma50"] = btc_ctx["vs_sma50"][idx[ok]]
    return df, ts, o, h, lo, c, date


def btc_context(loader, cadence: str) -> dict:
    g = loader.load("BTCUSDT", cadence=cadence, features=["timestamp", "close"])
    d = g.to_dict(as_series=False)
    ts = np.asarray(d["timestamp"], dtype=np.int64)
    c = np.asarray(d["close"], float)
    bars_24h = {"1h": 24, "4h": 6, "15m": 96, "30m": 48, "1d": 1}.get(cadence, 6)
    r24h = np.full(len(c), np.nan)
    r24h[bars_24h:] = c[bars_24h:] / c[:-bars_24h] - 1.0
    sma50 = pd.Series(c).rolling(50, min_periods=50).mean().to_numpy()
    vs_sma50 = c / np.where(sma50 > 0, sma50, np.nan) - 1.0
    return {"ts": ts, "r24h": r24h, "vs_sma50": vs_sma50}


# ===========================================================================
# 2. ENUMERATE firings ACROSS THE CONFIG GRID -> per-firing causal features (incl config params) + label
# ===========================================================================
@dataclass
class Firing:
    sym: str
    cfg: tuple                # (fast, slow, type)
    entry_idx: int
    entry_ts: pd.Timestamp
    window: str
    net_pnl: float
    duration_bars: int
    feats: dict


def _config_trigger_and_geometry(c: np.ndarray, fast: int, slow: int, kind: str):
    """Past-only MA-cross trigger + the MA geometry features for ONE config. Returns (trigger bool array,
    geometry dict of arrays). fast MA crosses ABOVE slow MA this bar (uses MA[t] and MA[t-1] only)."""
    n = len(c)
    fma = _ma(c, fast, kind)
    sma = _ma(c, slow, kind)
    fast_above = fma > sma
    prev_above = np.concatenate([[False], fast_above[:-1]])
    trigger = fast_above & (~prev_above) & np.isfinite(fma) & np.isfinite(sma)

    bars_since = np.zeros(n)
    cnt = 0
    for i in range(n):
        cnt = 0 if not fast_above[i] else cnt + 1
        bars_since[i] = cnt
    f_slope = np.concatenate([[0.0], (fma[1:] - fma[:-1])]) / np.where(c > 0, c, np.nan)
    s_slope = np.concatenate([[0.0], (sma[1:] - sma[:-1])]) / np.where(c > 0, c, np.nan)
    geom = {
        "ema_fast_slope": f_slope, "ema_slow_slope": s_slope,
        "ma_spread": (fma - sma) / np.where(c > 0, c, np.nan),
        "price_to_fast": c / np.where(fma > 0, fma, np.nan) - 1.0,
        "price_to_slow": c / np.where(sma > 0, sma, np.nan) - 1.0,
        "ma_stack": ((c > fma) & (fma > sma)).astype(float),
        "bars_since_cross": bars_since,
    }
    return trigger, geom


def enumerate_config_firings(base, sym: str, policy: ExitPolicy, verbose=False) -> list[Firing]:
    """For EACH config in the grid: build the past-only cross trigger + MA geometry, run setup_harness for
    the SETUP OUTCOME (net_pnl across the user's risk-exit MOVE), attach the causal features INCLUDING the
    config params. cfg_past_hitrate is the config's OWN expanding past-only win-rate (causal: uses only
    the prior firings of THAT config on THAT asset). All firings across all configs are POOLED."""
    df_base, ts, o, h, lo, c, date = base
    firings: list[Firing] = []
    n_cfg_fired = 0
    for (fast, slow, kind) in MA_CONFIG_GRID:
        trigger, geom = _config_trigger_and_geometry(c, fast, slow, kind)
        if trigger.sum() == 0:
            continue
        df = df_base.copy()
        df["trigger"] = trigger
        for gk, gv in geom.items():
            df[gk] = gv
        h_run = SetupHarness(df, "trigger", policy, WIN, cost_rt=TAKER_RT)
        res = h_run.run()
        # config's OWN expanding past-only win-rate, in firing order (Laplace-smoothed; <= prior firings)
        cfg_firings = sorted(res.trades, key=lambda t: t["entry_idx"])
        wins_so_far = 0
        seen = 0
        for t in cfg_firings:
            past_hr = (wins_so_far + 1.0) / (seen + 2.0)   # Laplace prior 0.5; strictly prior firings
            i = t["entry_idx"]
            row = df.iloc[i]
            feats = {}
            # config-navigation features (the key addition)
            feats["cfg_fast_len"] = float(fast)
            feats["cfg_slow_len"] = float(slow)
            feats["cfg_fast_slow_ratio"] = float(fast) / float(slow)
            feats["cfg_is_ema"] = 1.0 if kind == "EMA" else 0.0
            feats["cfg_past_hitrate"] = float(past_hr)
            # geometry + chimera + regime + cross-asset (causal at the cross bar)
            for f in (MA_FEATS + CHIMERA_FEATS + REGIME_FEATS + XASSET_FEATS):
                v = row[f] if f in df.columns else np.nan
                feats[f] = float(v) if pd.notna(v) else np.nan
            feats = _wm_belief_features(feats, df, i)
            firings.append(Firing(sym=sym, cfg=(fast, slow, kind), entry_idx=i,
                                  entry_ts=pd.Timestamp(t["entry_ts"]), window=t["window"],
                                  net_pnl=float(t["net_pnl"]), duration_bars=int(t["duration_bars"]),
                                  feats=feats))
            # advance the causal counter AFTER recording (so past_hr uses only strictly-prior firings)
            wins_so_far += 1 if t["net_pnl"] > 0 else 0
            seen += 1
        n_cfg_fired += 1
    if verbose:
        print(f"    {sym:9s} configs-fired={n_cfg_fired}/{len(MA_CONFIG_GRID)} firings={len(firings)}", flush=True)
    return firings


def _wm_belief_features(feats: dict, df: pd.DataFrame, i: int) -> dict:
    """DEFERRED HOOK (V1.1 is re-training). When the frozen V1.1 forecaster lands, decode its twohot
    DISTRIBUTION moments [E[r], Std[r], P(r<0)] at bar i via src/wm/forecast_bundle.py and add them as
    BELIEF features here (detached, eval). It must enter as a belief feature ONLY -- using the decoded
    return as a label or predicted-reward trips CDAP no_predicted_return_as_realized_reward. v1 is built
    WITHOUT it; this is a clean no-op so the column set + wiring are ready."""
    return feats


# ===========================================================================
# 3. THE META-LABELER: TRAIN-only z-stats/imputation/model/tau; binary entry gate
# ===========================================================================
def _zscore_train_only(firings_by_sym: dict, train_mask_by_sym: dict):
    """Per-(asset,feature) z-stats from TRAIN firings only; returns (mu,sd) dict per sym."""
    stats = {}
    for sym, firs in firings_by_sym.items():
        tr = [f for f, m in zip(firs, train_mask_by_sym[sym]) if m]
        mu = {}; sd = {}
        for fname in FEATURES:
            col = np.array([f.feats[fname] for f in tr], float) if tr else np.array([np.nan])
            m = np.nanmean(col) if np.isfinite(col).any() else 0.0
            s = np.nanstd(col) if np.isfinite(col).any() else 1.0
            mu[fname] = float(m); sd[fname] = float(s if (np.isfinite(s) and s > 1e-12) else 1.0)
        stats[sym] = (mu, sd)
    return stats


def _matrix(firings: list[Firing], stats: dict) -> tuple[np.ndarray, np.ndarray]:
    X = np.full((len(firings), len(FEATURES)), np.nan)
    y = np.zeros(len(firings), dtype=int)
    for r, f in enumerate(firings):
        mu, sd = stats[f.sym]
        for j, fname in enumerate(FEATURES):
            v = f.feats[fname]
            X[r, j] = (v - mu[fname]) / sd[fname] if np.isfinite(v) else np.nan
        y[r] = 1 if f.net_pnl > 0 else 0
    return X, y


def fit_and_select(firings_by_sym: dict, seed: int, label_band: float = 0.0,
                   verbose: bool = False) -> dict:
    """Fit logistic + HGB on TRAIN only, pick tau on TRAIN, apply to all windows. Returns the selected
    trade streams per window + AUCs + the TRAIN decile lift. label_band: y = (net_pnl > label_band)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    train_mask_by_sym = {sym: [f.window == "TRAIN" for f in firs]
                         for sym, firs in firings_by_sym.items()}

    def purge_keep(firs):
        keep = [True] * len(firs)
        wlabels = [f.window for f in firs]
        boundary_idx = []
        for r in range(1, len(firs)):
            if wlabels[r] != wlabels[r - 1]:
                boundary_idx.append(firs[r].entry_idx)
        for r, f in enumerate(firs):
            for b in boundary_idx:
                if 0 <= (f.entry_idx - b) < PURGE_BARS:
                    keep[r] = False
                    break
        return keep

    purge_by_sym = {sym: purge_keep(firs) for sym, firs in firings_by_sym.items()}
    tr_keep_mask = {sym: [tm and pk for tm, pk in zip(train_mask_by_sym[sym], purge_by_sym[sym])]
                    for sym in firings_by_sym}
    stats = _zscore_train_only(firings_by_sym, tr_keep_mask)

    all_fir = [f for sym in firings_by_sym for f in firings_by_sym[sym]]
    keep_flat = [pk for sym in firings_by_sym for pk in purge_by_sym[sym]]
    X, _y_raw = _matrix(all_fir, stats)
    y = np.array([1 if f.net_pnl > label_band else 0 for f in all_fir], dtype=int)
    win_of = np.array([f.window for f in all_fir])
    keep = np.array(keep_flat, bool)

    tr = (win_of == "TRAIN") & keep
    if tr.sum() < 40 or len(set(y[tr])) < 2:
        return {"insufficient": True, "n_train": int(tr.sum())}

    class_weight = {0: 1.0, 1: FALSE_LONG_PENALTY}

    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Ximp = np.where(np.isnan(X), med, X)
    logit = LogisticRegression(max_iter=2000, class_weight=class_weight)
    logit.fit(Ximp[tr], y[tr])
    pl_all = logit.predict_proba(Ximp)[:, 1]

    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=seed, early_stopping=False,
                                         class_weight=class_weight)
    hgb.fit(X[tr], y[tr])
    p_all = hgb.predict_proba(X)[:, 1]

    def _auc(p, mask):
        if mask.sum() < 5 or len(set(y[mask])) < 2:
            return None
        return float(roc_auc_score(y[mask], p[mask]))

    oos = (win_of == "OOS")
    val = (win_of == "VAL")
    aucs = {"hgb_train": _auc(p_all, tr), "hgb_val": _auc(p_all, val), "hgb_oos": _auc(p_all, oos),
            "logit_train": _auc(pl_all, tr), "logit_val": _auc(pl_all, val), "logit_oos": _auc(pl_all, oos)}

    # interpretable-first escalation: SELECT with LOGISTIC by default; escalate to HGB only if HGB shows
    # lift over logit on VAL (the development window -- never OOS/UNSEEN). Also avoids HGB-memorization
    # degeneracy (train AUC->1.0 makes the 67th-pctile TRAIN tau ~0.98 -> nothing selected held-out).
    hgb_val, logit_val = aucs["hgb_val"], aucs["logit_val"]
    use_hgb = bool(hgb_val is not None and logit_val is not None and hgb_val > logit_val + 0.02)
    p_sel = p_all if use_hgb else pl_all
    operating_model = "hgb" if use_hgb else "logit"

    tau = float(np.quantile(p_sel[tr], TAU_Q))

    order = np.argsort(p_sel[tr])
    tr_idx = np.flatnonzero(tr)[order]
    deciles = []
    for dd in range(10):
        seg = tr_idx[int(dd / 10 * len(tr_idx)): int((dd + 1) / 10 * len(tr_idx))]
        if len(seg) == 0:
            continue
        nets = np.array([all_fir[i].net_pnl for i in seg])
        deciles.append({"decile": dd + 1, "mean_net": float(nets.mean()), "win": float((nets > 0).mean())})
    mono = sum(1 for k in range(1, len(deciles)) if deciles[k]["mean_net"] >= deciles[k - 1]["mean_net"])
    decile_monotone_frac = mono / max(1, len(deciles) - 1)

    selected = p_sel >= tau
    # per-firing predicted-P map (by object id) so the per-bar dedup keeps the highest-P config per bar
    p_by_id = {id(all_fir[i]): float(p_sel[i]) for i in range(len(all_fir))}
    sel_streams = {}        # FULL (config,bar) selected pool -- used for PBO / config-nav diagnostics
    sel_streams_dedup = {}  # per-bar deduped -- the REALIZED trade stream (matches the firewall)
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        m = (win_of == w) & selected
        stream = [all_fir[i] for i in np.flatnonzero(m)]
        sel_streams[w] = stream
        sel_streams_dedup[w] = _dedup_per_bar(stream, p_by_id)

    return {
        "insufficient": False, "aucs": aucs, "tau": tau, "operating_model": operating_model,
        "decile_lift": deciles, "decile_monotone_frac": decile_monotone_frac,
        "selected_streams": sel_streams, "selected_streams_dedup": sel_streams_dedup,
        "p_all": p_sel, "win_of": win_of,
        "all_fir": all_fir, "n_train_fit": int(tr.sum()),
    }


def _compound(nets) -> float:
    a = np.asarray(nets, float)
    return float((np.prod(1.0 + a) - 1.0) * 100) if a.size else 0.0


def _dedup_per_bar(firings: list, p_by_id: Optional[dict] = None) -> list:
    """Collapse firings that land on the SAME (sym, entry_idx) bar to ONE trade. Multiple configs can
    cross at the same bar, and under the FIXED exit policy they all produce the IDENTICAL physical trade
    (same fill, same exit, same net_pnl) -- so counting all of them in the realized compound/battery would
    double-count one trade up to ~24x (MEASURED: mean 2.3, max 24 configs/bar at 4h BTC). The TRAINING pool
    keeps every (config,bar) row (the config-navigation signal lives in the differing config features); only
    the REALIZED-TRADE reporting is deduped so it matches the firewall's per-bar 'selected' column. When a
    P-map is given, the surviving firing per bar is the highest-predicted-P one (the config the meta-labeler
    most wants); else the first encountered."""
    best = {}
    for f in firings:
        key = (f.sym, f.entry_idx)
        if key not in best:
            best[key] = f
        elif p_by_id is not None and p_by_id.get(id(f), -1) > p_by_id.get(id(best[key]), -1):
            best[key] = f
    return list(best.values())


# ===========================================================================
# 4. THE GAUNTLET on the SELECTED-subset stream
# ===========================================================================
def _selected_entry_col(df: pd.DataFrame, selected_entry_idxs: set) -> pd.DataFrame:
    df2 = df.copy()
    col = np.zeros(len(df2), bool)
    for i in selected_entry_idxs:
        if 0 <= i < len(col):
            col[i] = True
    df2["selected"] = col
    return df2


def run_firewall_on_selected(frames_by_sym: dict, sel_idxs_by_sym: dict, policy: ExitPolicy) -> dict:
    """Per-asset firewall (regime+membership matched) on the SELECTED subset, then pool the verdict. The
    selected entries are POOLED across configs (a bar may be a cross in several configs; the firewall
    treats it as a single 'selected' entry per bar -- the merged selected setup the meta-labeler trades)."""
    per_asset = {}
    for sym, df in frames_by_sym.items():
        sel = sel_idxs_by_sym.get(sym, set())
        if not sel:
            continue
        df2 = _selected_entry_col(df, sel)
        if df2["selected"].sum() == 0:
            continue
        h = SetupHarness(df2, "selected", policy, WIN, cost_rt=TAKER_RT)
        fw = random_entry_null(h, n_books=200, seed=7, regime_matched=True, membership_matched=True)
        per_asset[sym] = fw
    held = ["OOS", "UNSEEN"]
    beats_votes = {w: [] for w in held}
    pos_votes = {w: [] for w in held}
    for sym, fw in per_asset.items():
        for w in held:
            r = fw["per_window"].get(w, {})
            if r.get("n_trades", 0) > 0:
                beats_votes[w].append(bool(r.get("beats_null")))
                pos_votes[w].append((r.get("real") or 0) > 0)
    summary = {}
    for w in held:
        bv, pv = beats_votes[w], pos_votes[w]
        summary[w] = {"beats_frac": (float(np.mean(bv)) if bv else None),
                      "pos_frac": (float(np.mean(pv)) if pv else None),
                      "n_assets_fired": len(bv)}
    beats_held = all((summary[w]["beats_frac"] or 0) > 0.5 for w in held)
    pos_held = all((summary[w]["pos_frac"] or 0) > 0.5 for w in held)
    return {"per_asset": {s: fw["verdict"] for s, fw in per_asset.items()},
            "held_summary": summary, "beats_held": beats_held, "pos_held": pos_held,
            "verdict": ("REAL TIMING EDGE (selected beats regime+membership-matched null on held-out)"
                        if (beats_held and pos_held) else
                        "NO TIMING EDGE (selected does not beat the random-entry null on held-out)")}


def run_shuffled_control(selected_unseen_nets: list, selected_oos_nets: list) -> dict:
    """Shuffled-market control on the SELECTED policy's realized held-out per-trade NET stream."""
    stream = np.asarray(list(selected_oos_nets) + list(selected_unseen_nets), float)
    stream = stream[np.isfinite(stream)]
    if stream.size < 8:
        return {"verdict": "INSUFFICIENT", "n": int(stream.size)}

    def policy(returns: np.ndarray) -> np.ndarray:
        return np.asarray(returns, float)

    res = shuffled_market_control(stream, policy, n_surrogates=200, seed=7, block=2,
                                  genuine_frac_threshold=SHUFFLE_GENUINE_FRAC)
    return res.to_dict()


def run_pbo(firings_by_sym: dict, fit_res: dict) -> dict:
    """PBO via CSCV over a cross-section of META-LABELER CONFIGS (tau operating points). The config GRID
    is the multiple-comparisons surface this gate is built to police -- high PBO => config-search overfit."""
    all_fir = fit_res["all_fir"]
    p_all = fit_res["p_all"]
    win_of = fit_res["win_of"]
    mask = np.isin(win_of, ["VAL", "OOS", "UNSEEN"])
    fir = [all_fir[i] for i in np.flatnonzero(mask)]
    p = p_all[mask]
    if len(fir) < 64:
        return {"verdict": "INSUFFICIENT", "n": len(fir)}
    order = np.argsort([f.entry_ts.value for f in fir])
    fir = [fir[i] for i in order]; p = p[order]
    nets = np.array([f.net_pnl for f in fir])
    taus = np.quantile(p_all[win_of == "TRAIN"], np.linspace(0.30, 0.85, 12))
    cols = []
    for tq in taus:
        col = np.where(p >= tq, nets, 0.0)
        if np.std(col) > 1e-9:
            cols.append(col)
    if len(cols) < 2:
        return {"verdict": "INSUFFICIENT", "n_configs": len(cols)}
    R = np.column_stack(cols)
    T = R.shape[0]
    S = 16 if T >= 32 else (8 if T >= 16 else 4)
    try:
        res = pbo_cscv(R, S=S)
    except ValueError as e:
        return {"verdict": "INSUFFICIENT", "error": str(e), "T": int(T), "N": int(R.shape[1])}
    res["ship"] = bool(res["pbo"] < PBO_SHIP)
    return res


def run_seed_robustness(firings_by_sym: dict, label_band: float = 0.0) -> dict:
    """Re-fit across SEED_GRID; record per-seed UNSEEN+OOS selected compound + p05 bootstrap."""
    per_seed = []
    for s in SEED_GRID:
        fr = fit_and_select(firings_by_sym, seed=s, label_band=label_band)
        if fr.get("insufficient"):
            per_seed.append({"seed": s, "insufficient": True})
            continue
        oos = [f.net_pnl for f in fr["selected_streams_dedup"]["OOS"]]
        uns = [f.net_pnl for f in fr["selected_streams_dedup"]["UNSEEN"]]
        per_seed.append({"seed": s, "oos_compound": _compound(oos), "unseen_compound": _compound(uns),
                         "oos_pos": _compound(oos) > 0, "unseen_pos": _compound(uns) > 0,
                         "n_oos": len(oos), "n_uns": len(uns)})
    valid = [p for p in per_seed if not p.get("insufficient")]
    oos_pos = sum(1 for p in valid if p["oos_pos"])
    uns_pos = sum(1 for p in valid if p["unseen_pos"])
    fr0 = fit_and_select(firings_by_sym, seed=0, label_band=label_band)
    p05 = None
    if not fr0.get("insufficient"):
        held = [f.net_pnl for f in fr0["selected_streams_dedup"]["OOS"]] + \
               [f.net_pnl for f in fr0["selected_streams_dedup"]["UNSEEN"]]
        if len(held) >= 10:
            bb = block_bootstrap_p05_p95(held)
            p05 = bb["p05"]
    return {"per_seed": per_seed, "n_valid": len(valid),
            "oos_positive_seeds": oos_pos, "unseen_positive_seeds": uns_pos,
            "seeds_floor": SEEDS_OOS_POS_FLOOR, "bootstrap_p05": p05,
            "passes": bool(oos_pos >= SEEDS_OOS_POS_FLOOR and (p05 is not None and p05 > 0))}


def run_battery(fit_res: dict) -> dict:
    # realized-trade battery runs on the per-bar DEDUPED stream (no double-counting one physical trade)
    uns = [f.net_pnl for f in fit_res["selected_streams_dedup"]["UNSEEN"]]
    comps = {w: _compound([f.net_pnl for f in fit_res["selected_streams_dedup"][w]])
             for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
    if len(uns) < 2:
        return {"verdict": "INSUFFICIENT_UNSEEN", "n_unseen": len(uns), "comps": comps}
    eq = np.cumprod(1.0 + np.asarray(uns, float))
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    pairs = [(f.entry_ts, f.net_pnl) for f in fit_res["selected_streams_dedup"]["UNSEEN"]]
    # family_n reflects the config-grid x cadence selection surface (DSR/Holm deflation honesty)
    bat = battery_evaluate(uns, comps, dd, entry_pnl_pairs=pairs, family_n=len(MA_CONFIG_GRID))
    bat["comps"] = comps
    return bat


def per_regime_report(fit_res: dict) -> dict:
    out = {}
    for w in ("OOS", "UNSEEN"):
        bull = [f.net_pnl for f in fit_res["selected_streams_dedup"][w] if f.feats.get("above_sma200", 0) == 1.0]
        bear = [f.net_pnl for f in fit_res["selected_streams_dedup"][w]
                if f.feats.get("above_sma200", 1) == 0.0]
        out[w] = {"bull_n": len(bull), "bull_compound": _compound(bull),
                  "bear_n": len(bear), "bear_compound": _compound(bear)}
    return out


def unconditional_baseline(firings_by_sym: dict) -> dict:
    """The UNCONDITIONAL trigger (ALL configs fire, no meta-label gate) per window -- the thing the
    selected subset must BEAT ('does config-nav + conditioning add anything?'). Reported per-bar-DEDUPED
    (the apples-to-apples reference for the deduped selected stream: a bar where ANY config fired = one
    unconditional trade) AND pooled (every (config,bar), the raw 'trade every config cross' book -- which
    double-counts the same physical trade and is shown only to expose the cost/whipsaw bleed)."""
    out = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        firs_w = [f for sym in firings_by_sym for f in firings_by_sym[sym] if f.window == w]
        dedup = _dedup_per_bar(firs_w)
        out[w] = {"n": len(dedup), "compound": _compound([f.net_pnl for f in dedup]),
                  "n_pooled": len(firs_w), "compound_pooled": _compound([f.net_pnl for f in firs_w])}
    return out


# ===========================================================================
# 5. PER-CADENCE VERDICT
# ===========================================================================
def verdict_for_cadence(loader, cadence: str, syms: list[str], verbose: bool = True) -> dict:
    # THE USER'S RISK EXIT: 5% trailing stop (initial entry-5%) + max-hold time-stop. Held FIXED.
    policy = ExitPolicy(sl_pct=RISK_SL_PCT, trail_pct=RISK_TRAIL_PCT,
                        max_hold_bars=MAX_HOLD.get(cadence, 24))
    bctx = btc_context(loader, cadence)
    frames_by_sym = {}
    firings_by_sym = {}
    n_fire = 0
    cfg_fire_counts = {}
    for sym in syms:
        base = build_base_frame(loader, sym, cadence, None if sym == "BTCUSDT" else bctx)
        if base is None:
            continue
        firs = enumerate_config_firings(base, sym, policy, verbose=verbose)
        if not firs:
            continue
        # firewall needs an OHLC+selected frame; store the config-INDEPENDENT base df
        # (date/open/high/low/close + chimera/regime/cross-asset context). The 'selected' boolean column
        # is layered on per-asset by _selected_entry_col when the firewall runs.
        frames_by_sym[sym] = base[0]
        firings_by_sym[sym] = firs
        n_fire += len(firs)
        for f in firs:
            cfg_fire_counts[f.cfg] = cfg_fire_counts.get(f.cfg, 0) + 1

    n_configs_used = len({f.cfg for sym in firings_by_sym for f in firings_by_sym[sym]})
    if n_fire < 80:
        return {"cadence": cadence, "verdict": "INSUFFICIENT_FIRINGS", "n_firings": n_fire,
                "n_configs_in_grid": len(MA_CONFIG_GRID), "n_configs_fired": n_configs_used}

    uncond = unconditional_baseline(firings_by_sym)
    fit0 = fit_and_select(firings_by_sym, seed=0, label_band=0.0, verbose=verbose)
    if fit0.get("insufficient"):
        return {"cadence": cadence, "verdict": "INSUFFICIENT_TRAIN", "detail": fit0,
                "n_firings": n_fire, "n_configs_fired": n_configs_used}

    fit_band = fit_and_select(firings_by_sym, seed=0, label_band=TAKER_RT)

    sel_idxs_by_sym = {}
    for w in ("OOS", "UNSEEN", "TRAIN", "VAL"):
        for f in fit0["selected_streams"][w]:
            sel_idxs_by_sym.setdefault(f.sym, set()).add(f.entry_idx)

    # realized-trade compound on the per-bar DEDUPED stream (matches the firewall's per-bar 'selected')
    sel_comp = {w: _compound([f.net_pnl for f in fit0["selected_streams_dedup"][w]])
                for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
    sel_n = {w: len(fit0["selected_streams_dedup"][w]) for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
    sel_n_pooled = {w: len(fit0["selected_streams"][w]) for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}

    # GAUNTLET
    fw = run_firewall_on_selected(frames_by_sym, sel_idxs_by_sym, policy)
    shuf = run_shuffled_control([f.net_pnl for f in fit0["selected_streams_dedup"]["UNSEEN"]],
                                [f.net_pnl for f in fit0["selected_streams_dedup"]["OOS"]])
    pbo = run_pbo(firings_by_sym, fit0)
    seeds = run_seed_robustness(firings_by_sym, label_band=0.0)
    bat = run_battery(fit0)
    regime = per_regime_report(fit0)

    beats_uncond = {w: (sel_comp[w] > uncond[w]["compound"]) for w in ("OOS", "UNSEEN")}

    aucs = fit0["aucs"]
    op_model = fit0["operating_model"]
    op_oos_auc = aucs.get(f"{op_model}_oos")
    auc_ok = bool((op_oos_auc or 0) > AUC_FLOOR)
    fw_ok = bool(fw["beats_held"] and fw["pos_held"])
    shuf_ok = bool(shuf.get("verdict") == "GENUINE")
    pbo_ok = bool(pbo.get("ship", False))
    seeds_ok = bool(seeds["passes"])
    bat_ok = bool(bat.get("verdict", "FAIL") != "FAIL")
    beats_uncond_held = bool(beats_uncond["OOS"] and beats_uncond["UNSEEN"])

    ship = bool(auc_ok and fw_ok and shuf_ok and pbo_ok and seeds_ok and bat_ok and beats_uncond_held)
    verdict_str = ("CONDITIONAL EDGE (config-nav meta-labeler beats unconditional pooled trigger + firewall "
                   "on UNSEEN, AUC>0.55, PBO<0.10, >=8/10 seeds, shuffle-GENUINE, battery-recognized) -- "
                   "CHAMPION-GATE IT"
                   if ship else
                   "NO CONDITIONAL EDGE with internal features (the EXPECTED, high-value null -- the JOINT "
                   "mechanism (config-nav + conditional-entry + 5%-trail risk-exit) is refuted at this "
                   "cadence with internal features, re-confirming the separate component nulls jointly)")

    # config-overfit diagnosis (the central risk): if PBO is high, the apparent edge is config-search overfit
    pbo_val = pbo.get("pbo")
    config_overfit_flag = bool(pbo_val is not None and pbo_val >= PBO_SHIP)

    return {
        "cadence": cadence,
        "n_firings": n_fire,
        "n_assets": len(firings_by_sym),
        "config_navigation": {
            "n_configs_in_grid": len(MA_CONFIG_GRID),
            "n_configs_fired": n_configs_used,
            "config_params_as_features": CONFIG_FEATS,
            "firings_per_config_top10": dict(sorted(
                {f"{f}x{s}_{t}": cfg_fire_counts[(f, s, t)] for (f, s, t) in cfg_fire_counts}.items(),
                key=lambda kv: -kv[1])[:10]),
            "note": "ALL firings across ALL configs POOLED into one dataset; config params enter as features",
        },
        "policy": {"exit": "USER RISK 5%-trailing-stop", "sl_pct": RISK_SL_PCT,
                   "trail_pct": RISK_TRAIL_PCT, "max_hold_bars": MAX_HOLD.get(cadence, 24)},
        "unconditional_trigger": uncond,
        "selected_compound": sel_comp,
        "selected_n_trades_dedup": sel_n,
        "selected_n_firings_pooled": sel_n_pooled,
        "selected_beats_unconditional_held": beats_uncond,
        "aucs": aucs,
        "operating_model": op_model,
        "tau": fit0["tau"],
        "decile_lift": fit0["decile_lift"],
        "decile_monotone_frac": fit0["decile_monotone_frac"],
        "label_stability_D09": {
            "net_gt_0_unseen": sel_comp["UNSEEN"],
            "net_gt_costband_unseen": (None if fit_band.get("insufficient") else
                                       _compound([f.net_pnl for f in fit_band["selected_streams_dedup"]["UNSEEN"]])),
        },
        "config_overfit_flag": config_overfit_flag,
        "gauntlet": {
            "1_auc_floor": {"operating_model": op_model, "operating_oos_auc": op_oos_auc,
                            "hgb_oos": aucs.get("hgb_oos"), "logit_oos": aucs.get("logit_oos"),
                            "floor": AUC_FLOOR, "pass": auc_ok,
                            "decile_monotone_frac": fit0["decile_monotone_frac"]},
            "2_firewall": {"verdict": fw["verdict"], "held_summary": fw["held_summary"],
                           "per_asset": fw["per_asset"], "pass": fw_ok},
            "3_shuffled_control": {"verdict": shuf.get("verdict"),
                                   "overfit_fraction": shuf.get("overfit_fraction"), "pass": shuf_ok},
            "4_pbo": {"pbo": pbo.get("pbo"), "ship_thr": PBO_SHIP, "verdict": pbo.get("verdict"),
                      "config_overfit": config_overfit_flag, "detail": pbo, "pass": pbo_ok},
            "5_seed_robustness": {"oos_positive_seeds": seeds["oos_positive_seeds"],
                                  "floor": SEEDS_OOS_POS_FLOOR, "bootstrap_p05": seeds["bootstrap_p05"],
                                  "per_seed": seeds["per_seed"], "pass": seeds_ok},
            "6_battery": {"verdict": bat.get("verdict"), "jk2": bat.get("jk2"), "jk3": bat.get("jk3"),
                          "p05": bat.get("p05"), "n_eff": bat.get("n_eff"),
                          "maxdd": bat.get("unseen_maxdd_pct"), "pass": bat_ok},
            "8_per_regime": regime,
        },
        "SHIP": ship,
        "verdict": verdict_str,
    }


# ===========================================================================
# 6. NO-SKILL + POSITIVE controls (gate (7), two-sided soundness)
# ===========================================================================
def run_two_sided_controls(verbose: bool = True) -> dict:
    """Two-sided soundness on the META-LABELER's gauntlet, using synthetic firing streams:
      - NO-SKILL (random) meta-labeler: random predicted-P -> the selected subset is a random
        sub-sample of firings -> it must NOT beat the unconditional trigger (FAIL).
      - CONFIG-ONLY meta-labeler (the navigation no-skill control): select purely by a config-id signal
        that is INDEPENDENT of the per-firing winner -> the config dimension alone must NOT add timing
        lift (FAILS). This is the direct no-skill control for the config-navigation claim: if selecting
        by config alone 'wins', the gate is broken.
      - REGIME-ONLY meta-labeler: select by a single regime flag (a beta proxy) -> must FAIL.
      - POSITIVE: a synthetic firing stream where a known CAUSAL feature deterministically separates
        winners from losers -> the meta-labeler MUST DISCRIMINATE it (AUC high, selected >> unconditional)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)

    # ---- POSITIVE: a known conditional edge. feature f0 > 0 => winner.
    n = 2000
    f0 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 5))
    X = np.column_stack([f0, noise])
    base = np.where(f0 > 0, 0.04, -0.02) + rng.normal(0, 0.01, n)
    y = (base > 0).astype(int)
    tr = slice(0, 1000); oos = slice(1400, 2000)
    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=0, early_stopping=False)
    hgb.fit(X[tr], y[tr])
    p = hgb.predict_proba(X)[:, 1]
    tau = np.quantile(p[tr], TAU_Q)
    pos_auc = float(roc_auc_score(y[oos], p[oos]))
    sel_oos = base[oos][p[oos] >= tau]
    uncond_oos = base[oos]
    pos_sel_comp = _compound(sel_oos)
    pos_uncond_comp = _compound(uncond_oos)
    positive_ships = bool(pos_auc > AUC_FLOOR and pos_sel_comp > pos_uncond_comp)

    # ---- NO-SKILL: random predicted-P on the SAME positive stream.
    p_rand = rng.random(n)
    tau_r = np.quantile(p_rand[tr], TAU_Q)
    rand_sel = base[oos][p_rand[oos] >= tau_r]
    rand_auc = float(roc_auc_score(y[oos], p_rand[oos]))
    no_skill_fails = bool(rand_auc < AUC_FLOOR and _compound(rand_sel) <= pos_uncond_comp + 1.0)

    # ---- CONFIG-ONLY (the navigation no-skill control): a config-id assigned at random, INDEPENDENT of
    # the winner -> selecting by config alone gives ~unconditional performance (no timing lift).
    config_id = rng.integers(0, len(MA_CONFIG_GRID), n)
    fav_config = int(rng.integers(0, len(MA_CONFIG_GRID)))   # an arbitrarily 'chosen' config
    cfg_sel = base[oos][config_id[oos] == fav_config]
    # config-only fails iff selecting by config alone does NOT beat the unconditional by the positive margin
    config_only_fails = bool(len(cfg_sel) == 0 or
                             abs(_compound(cfg_sel) - pos_uncond_comp) < abs(pos_sel_comp - pos_uncond_comp))

    # ---- REGIME-ONLY: select by a single regime flag uncorrelated with the per-firing winner.
    regime_flag = (rng.random(n) > 0.5)
    reg_sel = base[oos][regime_flag[oos]]
    regime_only_fails = bool(abs(_compound(reg_sel) - pos_uncond_comp) < abs(pos_sel_comp - pos_uncond_comp))

    ok = positive_ships and no_skill_fails and config_only_fails and regime_only_fails
    if verbose:
        print("=" * 78)
        print("[setup_meta_labeler] TWO-SIDED CONTROLS (gate 7; synthetic, no market data)")
        print("=" * 78)
        print(f"  POSITIVE (known causal edge): OOS AUC={pos_auc:.3f}  selected={pos_sel_comp:+.2f}%  "
              f"unconditional={pos_uncond_comp:+.2f}%  -> SHIPS={positive_ships}")
        print(f"  NO-SKILL (random P):          OOS AUC={rand_auc:.3f}  selected={_compound(rand_sel):+.2f}% "
              f" -> FAILS(correctly)={no_skill_fails}")
        print(f"  CONFIG-ONLY (nav no-skill):   selected={_compound(cfg_sel):+.2f}% vs uncond "
              f"{pos_uncond_comp:+.2f}%  -> FAILS(correctly)={config_only_fails}")
        print(f"  REGIME-ONLY (beta proxy):     selected={_compound(reg_sel):+.2f}% vs uncond "
              f"{pos_uncond_comp:+.2f}%  -> FAILS(correctly)={regime_only_fails}")
        print(f"\n  [setup_meta_labeler] two-sided soundness: {'PASS' if ok else 'CHECK'} -- "
              f"{'gate ships a real conditional edge; rejects random / config-only / regime-only selection.' if ok else 'see flags.'}")
    return {"positive_ships": positive_ships, "no_skill_fails": no_skill_fails,
            "config_only_fails": config_only_fails, "regime_only_fails": regime_only_fails,
            "positive_oos_auc": pos_auc, "all_pass": ok}


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="FULL-MECHANISM conditional setup-edge META-LABELER -- VERDICT")
    ap.add_argument("--cadence", nargs="+", default=["15m", "30m", "1h", "4h"],
                    help="cadences to evaluate (do NOT default one); pre-registered = 15m 30m 1h 4h")
    ap.add_argument("--assets", nargs="+", default=None, help="override u10 with explicit assets")
    ap.add_argument("--selftest", action="store_true", help="run the two-sided control logic only (no data)")
    args = ap.parse_args()

    if args.selftest:
        res = run_two_sided_controls(verbose=True)
        return 0 if res["all_pass"] else 1

    from pipeline.chimera_loader import ChimeraLoader
    loader = ChimeraLoader()
    syms = [_norm_sym(a) for a in args.assets] if args.assets else _u10_symbols()

    t0 = time.time()
    print("=" * 78)
    print(f"FULL-MECHANISM META-LABELER VERDICT -- MA-config GRID ({len(MA_CONFIG_GRID)} configs) x "
          f"5%-trail risk exit, u10")
    print("=" * 78)
    controls = run_two_sided_controls(verbose=True)
    print()

    per_cadence = {}
    for cad in args.cadence:
        print(f"\n--- CADENCE {cad} ---")
        try:
            per_cadence[cad] = verdict_for_cadence(loader, cad, syms, verbose=True)
        except Exception as e:
            import traceback
            per_cadence[cad] = {"cadence": cad, "verdict": "ERROR", "error": str(e),
                                "trace": traceback.format_exc()}
            print(f"  ERROR on {cad}: {e}")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "setup_meta_labeler_FULLMECH", "git_sha": sha, "stamp": stamp,
        "objective": "held-out COMPOUND of the selected subset (taker 0.0024, LONG-only spot lev=1); "
                     "AUC is a within-model diagnostic ONLY",
        "mechanism": "config-navigation (MA-config grid pooled, config params as features) + conditional-"
                     "entry meta-labeler + the USER'S 5%-trailing-stop risk exit, tested as ONE joint thing",
        "pre_registration": {
            "config_grid": {"fast": MA_FAST_GRID, "slow": MA_SLOW_GRID, "types": MA_TYPES,
                            "n_configs": len(MA_CONFIG_GRID),
                            "rule": "fast<slow; each (fast,slow,type) a past-only cross trigger; ALL POOLED"},
            "config_features": CONFIG_FEATS,
            "cadences": args.cadence,
            "cadence_note": "15m/30m/1h/4h per the standing HARD RULE; 1d on demand; sub-15m needs the 1m "
                            "dataset (SEPARATE effort -- out of scope here)",
            "universe": "u10 (survivorship FLAGGED)",
            "exit": f"USER RISK 5%-trailing-stop (initial stop entry-{RISK_SL_PCT:.0%}, trail "
                    f"{RISK_TRAIL_PCT:.0%} below running high) + time-stop per cadence",
            "label": "y = net_pnl > 0 (taker 0.0024 RT); also reported at net>cost-band (D09)",
            "features": FEATURES, "wm_feature": "DEFERRED (V1.1 retraining; clean hook left)",
            "model": "LogisticRegression -> HistGradientBoosting; binary entry gate at tau=67th TRAIN pctile",
            "asymmetric_loss": f"false-LONG penalty {FALSE_LONG_PENALTY}x (class_weight on y=1)",
            "split": "chronological 50/20/20/10 (canonical WindowSpec dates) + 400-bar purge; UNSEEN once",
        },
        "two_sided_controls": controls,
        "per_cadence": per_cadence,
        "RWYB": {
            "gauntlet_gates_ran": ["auc_floor", "firewall(regime+membership)", "shuffled_control(perm/block)",
                                   "pbo_cscv(CENTRAL-config-overfit gate)", "seed_robustness(10)",
                                   "battery(A/B/C)", "two_sided_controls", "per_regime"],
            "two_sided": "no-skill (random + CONFIG-ONLY + regime-only) must FAIL; positive (synthetic) must SHIP",
            "config_nav_proof": "see per_cadence[*].config_navigation: n_configs_fired + firings POOLED",
            "objective": "held-out compound, never AUC/IC",
        },
        "honest_framing": [
            "each COMPONENT is already null SEPARATELY: config-nav (MA-oracle/verify_dna_finding -- no "
            "learnable per-move config at 30m-1d, regime-driven), conditional-entry (EMA12x26 slice + "
            "mover_metalabel OOS AUC 0.51-0.52), exit-timing (exit_capture_proxy/D61).",
            "this run tests the JOINT mechanism + config-navigation + the user's risk-exit + the 15m cadence.",
            "an honest NULL is the EXPECTED, high-value answer -- it means the JOINT mechanism is genuinely "
            "refuted at 15m-4h with INTERNAL features, not just its parts.",
            "a non-null at ANY cadence = the first measured edge -> champion-gate it.",
        ],
        "caveats": [
            "u10 CURRENT membership -> survivorship on absolute-level features (FLAGGED, not corrected)",
            "WM-as-feature DEFERRED (V1.1 retraining) -- v1 verdict is INTERNAL-features-only by design",
            "THE CONFIG GRID IS A BIG MULTIPLE-COMPARISONS SURFACE: PBO is the CENTRAL gate. If PBO>=0.10 "
            "(config_overfit_flag=True) the apparent edge is CONFIG-SEARCH OVERFIT, NOT a real edge.",
            "shuffled control runs on the realized held-out net stream (necessary-not-sufficient; composes "
            "with firewall+PBO+seeds)",
            "sub-15m (1m) MA-cross pooling is a SEPARATE effort -- NOT run here (stated, not silently skipped)",
        ],
    }
    out_path = OUT / f"setup_metalabel_FULLMECH_verdict_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    print("\n" + "=" * 78)
    print("VERDICT SUMMARY (FULL MECHANISM)")
    print("=" * 78)
    for cad, v in per_cadence.items():
        if v.get("verdict", "").startswith(("INSUFFICIENT", "ERROR")):
            print(f"  [{cad}] {v['verdict']}  ({v.get('n_firings','?')} firings, "
                  f"{v.get('n_configs_fired','?')} configs)")
            continue
        g = v["gauntlet"]
        cn = v["config_navigation"]
        print(f"  [{cad}] SHIP={v['SHIP']}  {v['verdict'][:70]}")
        print(f"        config-nav: {cn['n_configs_fired']}/{cn['n_configs_in_grid']} configs fired, "
              f"{v['n_firings']} pooled firings across {v['n_assets']} assets")
        print(f"        sel UNSEEN={v['selected_compound']['UNSEEN']:+.2f}% vs uncond "
              f"{v['unconditional_trigger']['UNSEEN']['compound']:+.2f}%  | "
              f"op={g['1_auc_floor']['operating_model']} OOS AUC={g['1_auc_floor']['operating_oos_auc']}")
        print(f"        gates: auc={g['1_auc_floor']['pass']} firewall={g['2_firewall']['pass']} "
              f"shuffle={g['3_shuffled_control']['pass']} pbo={g['4_pbo']['pass']}"
              f"(PBO={g['4_pbo']['pbo']}, config_overfit={g['4_pbo']['config_overfit']}) "
              f"seeds={g['5_seed_robustness']['pass']}({g['5_seed_robustness']['oos_positive_seeds']}/10) "
              f"battery={g['6_battery']['pass']}")
    print(f"\n  controls two-sided pass: {controls['all_pass']}")
    print(f"  ({time.time()-t0:.0f}s)  VERDICT JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_two_sided_controls(verbose=True)["all_pass"] and 0 or 1)
    sys.exit(main())
