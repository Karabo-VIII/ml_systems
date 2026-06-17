"""val_pepe_phase2_regime_gated.py — W2 VAL Phase 2 regime-gated + chimera filter

MAXX-INST-2026-05-26-NIGHT W2 (Opus domain-worker, 2nd-round)
Mandate: validate TRAIN-discovered Phase 2 regime-gated setups on VAL.
NO re-tuning. NO UNSEEN access. Past-only at every layer.

Locked Tier A setups (Bonferroni-survived in W1 permutation null tests):
  A1: BULL_RISK_ON x pepe_trending_up @ 1h  -> SMA(20, 21), E5 48h hold
  A2: BULL_RISK_ON x pepe_trending_up @ 30m -> SMA(10, 24), E5 48h hold
  A3: BULL_RISK_ON x pepe_trending_up @ 15m -> SMA(36, 48), E5 48h hold

Chimera filter for Tier A: bd_imbalance_l1 entry value > 60-bar past-only
rolling median (note: 60 bars are on the cadence timeline, but bd_imbalance_l1
is daily-grain — so this is effectively a coarse filter at sub-hourly cadences).

Locked Tier B setups (W2 grain-corrected chimera findings; cross-cadence robust):
  B1: BEAR_OR_RISK_OFF x pepe_chop @ 4h
      filter: premium_z90 daily_delta_7d < 0
      setup: SMA(20, 21) (W1 unified rule cell-winner at 1h applied at 4h scaled)
      actually use the regime-cell winner found at 4h: SMA(15,18) per master doc.
      Use indicator from §2.2 master doc.
  B2: CHOP_RISK_ON x pepe_trending_down @ 1h
      filter: te_btc_imb daily_zscore_30d > 1.0
      setup: SMA(5, 21) per master doc §2.1
  B3: BULL_RISK_ON x pepe_trending_down @ 4h
      filter: wh_whale_trade_count daily_delta_7d > 0
      setup: SMA(7, 9) per master doc §2.2

Methodology (cf. scripts/mine_pepe_train_regime_conditioned.py):
  1. Identify VAL mover days (PEPE daily-return >= 2%) on 1d chimera (VAL window = days 573-795).
  2. For each VAL mover day, classify regime (PEPE self-cell + BTC trend + risk_state) past-only.
  3. Per Tier A setup at cadence c:
     - locate 1h/30m/15m bar of the mover-day close
     - look back 72/144/288 bars (3 days) for first SMA fast crosses-above slow
     - if regime gate passes AND chimera filter passes (bd_imbalance_l1 > 60-bar past-only median)
     - enter at next-bar open
     - exit at +48h fixed hold (E5)
     - record capture
  4. Phase 1 baseline measurement: same methodology but no regime gate / chimera filter, using
     general best Phase 1 setup per cadence.
  5. Compute lift (Phase 2 - Phase 1) per setup.

Output:
  runs/audit/MAXX_2026_05_26/data/pepe_val_phase2_summary.json
  runs/audit/MAXX_2026_05_26/data/pepe_val_phase2_<setup_id>.json (one per setup)

canonical_seeds = {bag_seed=201, feat_seed=1201, rng_seed=8101}
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

ROOT = Path("c:/Users/karab/Documents/coding/ml_systems")
sys.path.insert(0, str(ROOT / "src"))

from wealth_bot.regime_router.regime_classifier import (
    RegimeClassifierConfig, classify_regime, recalibrate,
)

# ---------------------------------------------------------------------------
# Repro metadata
# ---------------------------------------------------------------------------
CANONICAL_SEEDS = {"bag_seed": 201, "feat_seed": 1201, "rng_seed": 8101}
GIT_SHA_AT_RUN = "c9be66b"  # latest commit per git log

OUT_DIR = ROOT / "runs" / "audit" / "MAXX_2026_05_26" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 50/PURGE/20/10 split (per mandate)
TRAIN_END_BAR_1D = 556          # day 555 inclusive
VAL_START_BAR_1D = 573          # day 573 inclusive (17-bar purge after TRAIN)
VAL_END_BAR_1D = 795            # day 795 inclusive (222-day VAL window)

# ---------------------------------------------------------------------------
# Cadence configs
# ---------------------------------------------------------------------------
CADENCE_CONFIGS = {
    "15m": {
        "chim_file": ROOT / "data/processed/chimera/15m/pepeusdt_v51_chimera_15m_20260522.parquet",
        "bars_per_day": 96,
        "lookback_bars": 288,           # 3 days @ 15m
        "exit_48h_bars": 192,           # 48h fixed hold
    },
    "30m": {
        "chim_file": ROOT / "data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet",
        "bars_per_day": 48,
        "lookback_bars": 144,
        "exit_48h_bars": 96,
    },
    "1h": {
        "chim_file": ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet",
        "bars_per_day": 24,
        "lookback_bars": 72,
        "exit_48h_bars": 48,
    },
    "4h": {
        "chim_file": ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet",
        "bars_per_day": 6,
        "lookback_bars": 18,
        "exit_48h_bars": 12,
    },
}

CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"
BTC_1D = ROOT / "data/processed/chimera/1d/btcusdt_v51_chimera_1d_20260522.parquet"

# ---------------------------------------------------------------------------
# Locked Phase 2 setup specs
# ---------------------------------------------------------------------------
TIER_A_SETUPS = {
    "A1_1h_BULL_RISK_ON_x_pepe_trending_up": {
        "cadence": "1h",
        "indicator": "SMA",
        "fast": 20,
        "slow": 21,
        "regime_macro": "BULL_RISK_ON",
        "regime_pepe": "pepe_trending_up",
        "chimera_filter": "bd_imbalance_l1_gt_med60",
        "exit_policy": "E5_48h",
        "train_median_capture": 0.1131,   # from master doc §2.1
        "bonferroni_survived": True,
    },
    "A2_30m_BULL_RISK_ON_x_pepe_trending_up": {
        "cadence": "30m",
        "indicator": "SMA",
        "fast": 10,
        "slow": 24,
        "regime_macro": "BULL_RISK_ON",
        "regime_pepe": "pepe_trending_up",
        "chimera_filter": "bd_imbalance_l1_gt_med60",
        "exit_policy": "E5_48h",
        "train_median_capture": 0.1307,  # from W1 unified
        "bonferroni_survived": True,
    },
    "A3_15m_BULL_RISK_ON_x_pepe_trending_up": {
        "cadence": "15m",
        "indicator": "SMA",
        "fast": 36,
        "slow": 48,
        "regime_macro": "BULL_RISK_ON",
        "regime_pepe": "pepe_trending_up",
        "chimera_filter": "bd_imbalance_l1_gt_med60",
        "exit_policy": "E5_48h",
        "train_median_capture": 0.1096,  # from W1 unified
        "bonferroni_survived": True,
    },
}

# Tier B — grain-corrected chimera signals (lower confidence)
# B1: BEAR_OR_RISK_OFF x pepe_chop @ 4h with premium_z90 falling
#     setup from master doc §2.2: SMA(15,18) (PROVISIONAL n=4 in TRAIN — small sample caveat)
TIER_B_SETUPS = {
    "B1_4h_BEAR_OR_RISK_OFF_x_pepe_chop": {
        "cadence": "4h",
        "indicator": "SMA",
        "fast": 15,
        "slow": 18,
        "regime_macro": "BEAR_OR_RISK_OFF",
        "regime_pepe": "pepe_chop",
        "chimera_filter": "premium_z90_d7delta_lt_0",
        "exit_policy": "E5_48h",
        "train_median_capture": 0.1487,  # PROVISIONAL n=4
        "bonferroni_survived": False,
        "provisional": True,
    },
    "B2_1h_CHOP_RISK_ON_x_pepe_trending_down": {
        "cadence": "1h",
        "indicator": "SMA",
        "fast": 5,
        "slow": 21,
        "regime_macro": "CHOP_RISK_ON",
        "regime_pepe": "pepe_trending_down",
        "chimera_filter": "te_btc_imb_z30_gt_1",
        "exit_policy": "E6_MFE_trail50",   # per master doc §2.1
        "train_median_capture": 0.0324,
        "bonferroni_survived": False,
    },
    "B3_4h_BULL_RISK_ON_x_pepe_trending_down": {
        "cadence": "4h",
        "indicator": "SMA",
        "fast": 7,
        "slow": 9,
        "regime_macro": "BULL_RISK_ON",
        "regime_pepe": "pepe_trending_down",
        "chimera_filter": "wh_whale_trade_count_d7delta_gt_0",
        "exit_policy": "E6_MFE_trail50",   # per master doc §2.2
        "train_median_capture": 0.0470,
        "bonferroni_survived": False,
    },
}

# Phase 1 baselines per master doc §1 — applied to VAL with no regime gate / no chimera filter
PHASE1_BASELINES = {
    "15m": {"indicator": "SMA", "fast": 12, "slow": 48, "train_median": 0.0585},
    "30m": {"indicator": "SMA", "fast": 18, "slow": 42, "train_median": 0.0453},
    "1h":  {"indicator": "SMA", "fast": 9,  "slow": 21, "train_median": 0.0469},
    "4h":  {"indicator": "SMA", "fast": 7,  "slow": 9,  "train_median": 0.0470},
}

# ---------------------------------------------------------------------------
# MA primitives (past-only, copy of mining script)
# ---------------------------------------------------------------------------
def compute_ma(prices: np.ndarray, kind: str, window: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    if kind == "SMA":
        if n >= window:
            csum = np.cumsum(prices)
            out[window - 1:] = (csum[window - 1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA":
        alpha = 2.0 / (window + 1.0)
        if n >= window:
            seed = prices[:window].mean()
            out[window - 1] = seed
            for i in range(window, n):
                out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    elif kind == "WMA":
        if n >= window:
            w = np.arange(1, window + 1, dtype=float)
            w_sum = w.sum()
            for i in range(window - 1, n):
                out[i] = (prices[i - window + 1:i + 1] * w).sum() / w_sum
    return out


def first_cross_up_in_window(fast_ma: np.ndarray, slow_ma: np.ndarray,
                              start: int, end: int) -> int:
    """Find first index in [max(start,1), end) where fast crosses above slow."""
    s = max(start, 1)
    if s >= end:
        return -1
    for i in range(s, end):
        if (not np.isnan(fast_ma[i - 1]) and not np.isnan(slow_ma[i - 1])
                and not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i])
                and fast_ma[i - 1] <= slow_ma[i - 1]
                and fast_ma[i] > slow_ma[i]):
            return i
    return -1


# ---------------------------------------------------------------------------
# BTC + risk_state past-only classifiers (copied from build_pepe_mover_day_signatures)
# ---------------------------------------------------------------------------
def btc_trend_at(closes: np.ndarray, t: int, sma_period: int = 30,
                  slope_lookback: int = 7) -> str:
    if t < sma_period + slope_lookback:
        return "warmup"
    sma_now = float(np.mean(closes[t - sma_period:t]))
    sma_lookback = float(np.mean(closes[t - sma_period - slope_lookback:t - slope_lookback]))
    sma_rising = sma_now > sma_lookback
    sma_falling = sma_now < sma_lookback
    close_t_minus_1 = float(closes[t - 1])
    if close_t_minus_1 > sma_now and sma_rising:
        return "bull"
    elif close_t_minus_1 < sma_now and sma_falling:
        return "bear"
    return "chop"


def btc_vol_state(closes: np.ndarray, t: int, window: int = 30,
                   p33_thresh: float = 0.020, p67_thresh: float = 0.035) -> str:
    if t < window + 1:
        return "warmup"
    chunk = closes[t - window - 1:t]
    if np.any(chunk <= 0):
        return "warmup"
    rets = np.diff(np.log(chunk))
    sigma = float(rets.std())
    if sigma < p33_thresh:
        return "low_vol"
    elif sigma >= p67_thresh:
        return "high_vol"
    return "med_vol"


def crypto_risk_state(btc_trend_s: str, btc_vol_s: str) -> str:
    if btc_trend_s in ("bull", "chop") and btc_vol_s in ("low_vol", "med_vol"):
        return "risk_on"
    return "risk_off"


def derive_macro_regime(btc_trend_s: str, risk_state_s: str) -> str:
    if btc_trend_s == "bear" or risk_state_s == "risk_off":
        return "BEAR_OR_RISK_OFF"
    if btc_trend_s == "bull" and risk_state_s == "risk_on":
        return "BULL_RISK_ON"
    if btc_trend_s == "chop" and risk_state_s == "risk_on":
        return "CHOP_RISK_ON"
    return "OTHER"


def derive_pepe_micro(pepe_self_cell: str) -> str:
    if pepe_self_cell == "WARMUP":
        return "WARMUP"
    if pepe_self_cell.startswith("trending_up"):
        return "pepe_trending_up"
    if pepe_self_cell.startswith("chop"):
        return "pepe_chop"
    if pepe_self_cell.startswith("trending_down"):
        return "pepe_trending_down"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Chimera filter helpers (PAST-ONLY)
# ---------------------------------------------------------------------------
def bd_imbalance_l1_gt_med60(df_cad: pl.DataFrame, entry_idx: int) -> tuple[bool, float]:
    """Returns (passes, entry_val). Filter: bd_imbalance_l1 at bar entry_idx-1 (lag-1)
    above past-60-bar rolling median."""
    if entry_idx < 60:
        return False, float("nan")
    vals = df_cad["bd_imbalance_l1"].to_numpy()[max(0, entry_idx - 60):entry_idx]
    if len(vals) < 30:
        return False, float("nan")
    med = float(np.nanmedian(vals))
    entry_val = float(vals[-1])
    return (entry_val > med), entry_val


def premium_z90_d7delta_lt_0(df_1d: pl.DataFrame, day_idx_1d: int) -> tuple[bool, float]:
    """premium_z90 7-day delta < 0 at the prior 1d bar. (entry day's premium_z90 -
    premium_z90 7 days back) < 0."""
    if day_idx_1d < 7:
        return False, float("nan")
    col = df_1d["premium_z90"].to_numpy()
    cur = float(col[day_idx_1d - 1])      # past-only: yesterday's value
    past = float(col[day_idx_1d - 8])
    if np.isnan(cur) or np.isnan(past):
        return False, float("nan")
    delta = cur - past
    return (delta < 0), delta


def te_btc_imb_z30_gt_1(df_1d: pl.DataFrame, day_idx_1d: int) -> tuple[bool, float]:
    """te_btc_imb z-score over past 30 days > 1.0 at lag-1."""
    if day_idx_1d < 31:
        return False, float("nan")
    col = df_1d["te_btc_imb"].to_numpy()
    history = col[day_idx_1d - 31:day_idx_1d - 1]
    if len(history) < 25:
        return False, float("nan")
    cur = float(col[day_idx_1d - 1])
    mu = float(np.nanmean(history))
    sd = float(np.nanstd(history))
    if sd == 0 or np.isnan(sd):
        return False, float("nan")
    z = (cur - mu) / sd
    return (z > 1.0), z


def wh_whale_trade_count_d7delta_gt_0(df_1d: pl.DataFrame, day_idx_1d: int) -> tuple[bool, float]:
    if day_idx_1d < 7:
        return False, float("nan")
    col = df_1d["wh_whale_trade_count"].to_numpy()
    cur = float(col[day_idx_1d - 1])
    past = float(col[day_idx_1d - 8])
    if np.isnan(cur) or np.isnan(past):
        return False, float("nan")
    delta = cur - past
    return (delta > 0), delta


CHIMERA_FILTER_FUNCS = {
    "bd_imbalance_l1_gt_med60": ("cadence", bd_imbalance_l1_gt_med60),
    "premium_z90_d7delta_lt_0": ("daily", premium_z90_d7delta_lt_0),
    "te_btc_imb_z30_gt_1": ("daily", te_btc_imb_z30_gt_1),
    "wh_whale_trade_count_d7delta_gt_0": ("daily", wh_whale_trade_count_d7delta_gt_0),
}


# ---------------------------------------------------------------------------
# Exit policies
# ---------------------------------------------------------------------------
def exit_E5_48h(entry_idx: int, opens: np.ndarray, closes: np.ndarray,
                 highs: np.ndarray, lows: np.ndarray, exit_48h_bars: int) -> tuple[int, float]:
    """48h fixed hold. Returns (exit_idx, captured_pct).

    captured_pct = exit_close / entry_open - 1.
    """
    n = len(closes)
    exit_idx = min(entry_idx + exit_48h_bars, n - 1)
    if entry_idx >= n - 1 or opens[entry_idx] <= 0:
        return -1, float("nan")
    return exit_idx, float(closes[exit_idx] / opens[entry_idx] - 1.0)


def exit_E6_MFE_trail50(entry_idx: int, opens: np.ndarray, closes: np.ndarray,
                         highs: np.ndarray, lows: np.ndarray, exit_48h_bars: int) -> tuple[int, float]:
    """MFE-trail50: exit when close falls 50% below MFE-from-entry. Cap at 48h."""
    n = len(closes)
    if entry_idx >= n - 1 or opens[entry_idx] <= 0:
        return -1, float("nan")
    entry_p = opens[entry_idx]
    mfe = 0.0
    for k in range(entry_idx, min(entry_idx + exit_48h_bars + 1, n)):
        cur_ret = highs[k] / entry_p - 1.0 if k > entry_idx else 0.0
        if cur_ret > mfe:
            mfe = cur_ret
        # check trail
        if mfe > 0:
            trail_thresh = entry_p * (1.0 + mfe * 0.5)
            if closes[k] < trail_thresh and k > entry_idx:
                return k, float(closes[k] / entry_p - 1.0)
    # cap
    exit_idx = min(entry_idx + exit_48h_bars, n - 1)
    return exit_idx, float(closes[exit_idx] / entry_p - 1.0)


EXIT_FUNCS = {
    "E5_48h": exit_E5_48h,
    "E6_MFE_trail50": exit_E6_MFE_trail50,
}


# ---------------------------------------------------------------------------
# VAL mover-day identification
# ---------------------------------------------------------------------------
def identify_val_mover_days(df_1d: pl.DataFrame) -> tuple[list[dict], dict]:
    """Identify VAL mover days at >=2% daily-return. Returns (list, summary)."""
    n = len(df_1d)
    df_1d = df_1d.sort("timestamp")
    df_1d = df_1d.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )
    closes = df_1d["close"].to_numpy()
    highs = df_1d["high"].to_numpy()

    daily_ret = np.zeros(n)
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0
    hi_move = np.zeros(n)
    hi_move[1:] = highs[1:] / closes[:-1] - 1.0

    movers = []
    for i in range(VAL_START_BAR_1D, min(VAL_END_BAR_1D + 1, n)):
        if daily_ret[i] >= 0.02:
            dt = df_1d["dt"][i]
            movers.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt)[:10],
                "ts_ms": int(df_1d["timestamp"][i]),
                "day_idx_1d": int(i),
                "daily_return": float(daily_ret[i]),
                "hi_move": float(hi_move[i]),
            })

    summary = {
        "val_window_bars": (VAL_START_BAR_1D, VAL_END_BAR_1D),
        "val_start_date": str(df_1d["dt"][VAL_START_BAR_1D].date()) if hasattr(df_1d["dt"][VAL_START_BAR_1D], "date") else None,
        "val_end_date": str(df_1d["dt"][min(VAL_END_BAR_1D, n-1)].date()) if hasattr(df_1d["dt"][min(VAL_END_BAR_1D, n-1)], "date") else None,
        "n_val_days": min(VAL_END_BAR_1D, n - 1) - VAL_START_BAR_1D + 1,
        "n_movers_2pct": len(movers),
        "base_mover_rate": len(movers) / max(1, min(VAL_END_BAR_1D, n - 1) - VAL_START_BAR_1D + 1),
    }
    return movers, summary


# ---------------------------------------------------------------------------
# Tag each VAL mover with regime
# ---------------------------------------------------------------------------
def tag_movers_with_regime(movers: list[dict], btc_1d: pl.DataFrame,
                            pepe_4h: pl.DataFrame, pepe_cfg: RegimeClassifierConfig,
                            btc_p33: float, btc_p67: float) -> list[dict]:
    """Tag each mover with past-only PEPE 4h regime cell + BTC macro + risk_state."""
    btc_ts = btc_1d["timestamp"].to_numpy()
    btc_close = btc_1d["close"].to_numpy()
    pepe_ts_4h = pepe_4h["timestamp"].to_numpy()
    pepe_close_4h = pepe_4h["close"].to_numpy()

    tagged = []
    for m in movers:
        ts_ms = m["ts_ms"]
        # PEPE 4h regime at the bar STRICTLY before entry ts (no peek)
        pepe_idx_4h = int(np.searchsorted(pepe_ts_4h, ts_ms, side="left") - 1)
        if pepe_idx_4h < 0:
            tagged.append({**m, "skip_reason": "no_pepe_4h_bar"})
            continue
        pepe_self_cell = classify_regime(pepe_close_4h, pepe_idx_4h + 1, pepe_cfg)
        pepe_micro = derive_pepe_micro(pepe_self_cell)

        # BTC trend / vol at strictly-before-entry day
        btc_idx = int(np.searchsorted(btc_ts, ts_ms, side="left") - 1)
        if btc_idx < 0:
            tagged.append({**m, "skip_reason": "no_btc_1d_bar"})
            continue
        btc_t = btc_idx + 1
        btc_tr = btc_trend_at(btc_close, btc_t, sma_period=30, slope_lookback=7)
        btc_v = btc_vol_state(btc_close, btc_t, window=30, p33_thresh=btc_p33, p67_thresh=btc_p67)
        risk_s = crypto_risk_state(btc_tr, btc_v)
        macro = derive_macro_regime(btc_tr, risk_s)

        tagged.append({
            **m,
            "pepe_self_cell": pepe_self_cell,
            "pepe_micro_regime": pepe_micro,
            "btc_trend": btc_tr,
            "btc_vol_state": btc_v,
            "crypto_risk_state": risk_s,
            "macro_regime": macro,
            "regime_cell": f"{macro}|{pepe_micro}",
        })
    return tagged


# ---------------------------------------------------------------------------
# Run a setup against VAL mover days
# ---------------------------------------------------------------------------
def run_setup_on_val_movers(
    setup_id: str, spec: dict, val_movers_tagged: list[dict],
    df_cad: pl.DataFrame, df_1d: pl.DataFrame,
    apply_regime_gate: bool, apply_chimera_filter: bool,
) -> dict:
    """Run a single setup against VAL mover days. Returns per-trade dict + summary."""
    cad = spec["cadence"]
    cad_cfg = CADENCE_CONFIGS[cad]
    kind = spec["indicator"]
    fast = spec["fast"]
    slow = spec["slow"]
    lookback = cad_cfg["lookback_bars"]
    exit_48h = cad_cfg["exit_48h_bars"]

    closes = df_cad["close"].to_numpy()
    opens = df_cad["open"].to_numpy()
    highs = df_cad["high"].to_numpy()
    lows = df_cad["low"].to_numpy()
    ts_arr = df_cad["timestamp"].to_numpy()

    fast_ma = compute_ma(closes, kind, fast)
    slow_ma = compute_ma(closes, kind, slow)

    chim_filter_kind, chim_filter_func = (None, None)
    if apply_chimera_filter and spec.get("chimera_filter"):
        chim_filter_kind, chim_filter_func = CHIMERA_FILTER_FUNCS[spec["chimera_filter"]]

    exit_func = EXIT_FUNCS[spec["exit_policy"]]

    per_trade = []
    for m in val_movers_tagged:
        if "skip_reason" in m:
            continue
        # Regime gate
        regime_gate_pass = True
        if apply_regime_gate:
            regime_gate_pass = (m.get("macro_regime") == spec["regime_macro"]
                                and m.get("pepe_micro_regime") == spec["regime_pepe"])
        if not regime_gate_pass:
            continue

        # locate cadence bar of daily close
        day_close_idx = int(np.searchsorted(ts_arr, m["ts_ms"], side="right") - 1)
        if day_close_idx < 0:
            continue

        window_start = max(0, day_close_idx - (lookback - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1

        # available move from window-start prices to day-close
        daily_close_price = float(closes[day_close_idx])
        avail_returns = []
        for i in range(window_start, day_close_idx):
            if i + 1 <= day_close_idx:
                ent_p = float(opens[i + 1])
                if ent_p > 0:
                    avail_returns.append(daily_close_price / ent_p - 1.0)
        available_move = max(avail_returns) if avail_returns else 0.0

        cross_idx = first_cross_up_in_window(fast_ma, slow_ma, signal_search_start, signal_search_end)
        if cross_idx == -1:
            per_trade.append({
                "date": m["date"], "ts_ms": m["ts_ms"],
                "regime_cell": m.get("regime_cell"),
                "available_move": available_move,
                "fired": False,
                "reason": "no_cross",
            })
            continue
        entry_idx = cross_idx + 1
        if entry_idx > day_close_idx or entry_idx >= len(opens):
            per_trade.append({
                "date": m["date"], "ts_ms": m["ts_ms"],
                "regime_cell": m.get("regime_cell"),
                "fired": False,
                "reason": "entry_past_close",
            })
            continue

        # Chimera filter check
        chim_pass = True
        chim_val = float("nan")
        if apply_chimera_filter and chim_filter_func is not None:
            if chim_filter_kind == "cadence":
                chim_pass, chim_val = chim_filter_func(df_cad, entry_idx)
            elif chim_filter_kind == "daily":
                day_idx_1d = m.get("day_idx_1d", -1)
                if day_idx_1d < 0:
                    chim_pass, chim_val = False, float("nan")
                else:
                    chim_pass, chim_val = chim_filter_func(df_1d, day_idx_1d)

        if not chim_pass:
            per_trade.append({
                "date": m["date"], "ts_ms": m["ts_ms"],
                "regime_cell": m.get("regime_cell"),
                "fired": False,
                "reason": "chimera_filter_blocked",
                "chimera_val": chim_val,
            })
            continue

        # Exit
        exit_idx, captured = exit_func(entry_idx, opens, closes, highs, lows, exit_48h)
        # The mandate uses E5 48h hold capture (close at +48h). For honest cross-comparison
        # with TRAIN methodology, capture = close-at-exit / entry-open - 1.
        if exit_idx < 0 or np.isnan(captured):
            per_trade.append({
                "date": m["date"], "ts_ms": m["ts_ms"],
                "regime_cell": m.get("regime_cell"),
                "fired": False,
                "reason": "exit_failed",
            })
            continue

        per_trade.append({
            "date": m["date"], "ts_ms": m["ts_ms"],
            "regime_cell": m.get("regime_cell"),
            "available_move": available_move,
            "fired": True,
            "entry_idx": int(entry_idx),
            "exit_idx": int(exit_idx),
            "entry_open": float(opens[entry_idx]),
            "exit_close": float(closes[exit_idx]),
            "captured_pct": float(captured),
            "capture_rate": float(captured / available_move) if available_move > 1e-9 else 0.0,
            "chimera_val": chim_val if not np.isnan(chim_val) else None,
        })

    # Summary
    fired_trades = [t for t in per_trade if t.get("fired")]
    captures = np.array([t["captured_pct"] for t in fired_trades]) if fired_trades else np.array([])
    summary = {
        "setup_id": setup_id,
        "n_val_movers_eligible": len([t for t in per_trade if apply_regime_gate is False or 1])
            if not apply_regime_gate else len(per_trade),
        "n_total_movers": len(val_movers_tagged),
        "n_regime_matched": len(per_trade),
        "n_fired": int(len(fired_trades)),
        "fire_rate": float(len(fired_trades) / max(1, len(per_trade))) if len(per_trade) > 0 else 0.0,
        "median_capture": float(np.median(captures)) if len(captures) else None,
        "mean_capture": float(np.mean(captures)) if len(captures) else None,
        "p10": float(np.quantile(captures, 0.10)) if len(captures) else None,
        "p90": float(np.quantile(captures, 0.90)) if len(captures) else None,
        "win_rate": float((captures > 0).mean()) if len(captures) else None,
    }
    return {
        "setup_id": setup_id,
        "spec": spec,
        "apply_regime_gate": apply_regime_gate,
        "apply_chimera_filter": apply_chimera_filter,
        "per_trade": per_trade,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ts_start = time.time()
    print(f"[*] W2 VAL Phase 2 regime-gated validation starting at {datetime.now(timezone.utc).isoformat()}")
    print(f"[*] VAL window: 1d bars [{VAL_START_BAR_1D}, {VAL_END_BAR_1D}] inclusive")

    # Load 1d chimera, identify VAL movers
    print("[*] Loading PEPE 1d chimera...")
    df_pepe_1d = pl.read_parquet(CHIM_1D).sort("timestamp")
    print(f"    n={df_pepe_1d.height} rows")

    val_movers, val_summary = identify_val_mover_days(df_pepe_1d)
    print(f"[*] VAL window: {val_summary['val_start_date']} -> {val_summary['val_end_date']}")
    print(f"    n_val_days={val_summary['n_val_days']}, n_movers={val_summary['n_movers_2pct']}, base_rate={val_summary['base_mover_rate']:.3f}")

    # Load BTC 1d
    print("[*] Loading BTC 1d chimera...")
    df_btc_1d = pl.read_parquet(BTC_1D).sort("timestamp")
    btc_ts = df_btc_1d["timestamp"].to_numpy()
    btc_close = df_btc_1d["close"].to_numpy()

    # Calibrate BTC vol thresholds on TRAIN-only window
    # TRAIN end = 1d bar 555 (day 556 exclusive). Need to map to BTC 1d index.
    # Use PEPE TRAIN end as alignment.
    train_end_ts = int(df_pepe_1d["timestamp"][TRAIN_END_BAR_1D - 1])
    btc_train_end_idx = int(np.searchsorted(btc_ts, train_end_ts, side="right") - 1)
    btc_train_closes = btc_close[:btc_train_end_idx + 1]
    btc_stds = []
    for t in range(31, len(btc_train_closes)):
        chunk = btc_train_closes[t - 30 - 1:t]
        if np.any(chunk <= 0):
            continue
        rets = np.diff(np.log(chunk))
        s = float(rets.std())
        if s > 0:
            btc_stds.append(s)
    btc_p33 = float(np.percentile(btc_stds, 33))
    btc_p67 = float(np.percentile(btc_stds, 67))
    print(f"[*] BTC 1d vol p33={btc_p33:.5f} p67={btc_p67:.5f} (TRAIN-only, n={len(btc_stds)})")

    # Calibrate PEPE 4h regime classifier on TRAIN-only PEPE 4h data
    print("[*] Loading PEPE 4h chimera...")
    df_pepe_4h_full = pl.read_parquet(CADENCE_CONFIGS["4h"]["chim_file"]).sort("timestamp")
    pepe_4h_train = df_pepe_4h_full.filter(pl.col("timestamp") <= train_end_ts)
    pepe_cfg = recalibrate(pepe_4h_train["close"].to_numpy())
    print(f"[*] PEPE 4h regime calibrated: p33={pepe_cfg.vol_p33:.5f} p67={pepe_cfg.vol_p67:.5f} "
          f"(TRAIN-only on {pepe_4h_train.height} bars)")

    # Tag VAL movers with regime
    print("[*] Tagging VAL movers with past-only regime cells...")
    val_movers_tagged = tag_movers_with_regime(val_movers, df_btc_1d, df_pepe_4h_full,
                                                  pepe_cfg, btc_p33, btc_p67)
    # Regime distribution
    from collections import Counter
    cell_dist = Counter(t.get("regime_cell", "SKIP") for t in val_movers_tagged)
    print(f"[*] VAL regime cell distribution:")
    for c, n in sorted(cell_dist.items(), key=lambda kv: -kv[1]):
        print(f"      {c:55s}  n={n}")

    # Compare to TRAIN distribution (per master doc §2.0)
    train_cell_dist = {
        "BULL_RISK_ON|pepe_trending_up": 66,
        "BULL_RISK_ON|pepe_chop": 17,
        "BULL_RISK_ON|pepe_trending_down": 11,
        "CHOP_RISK_ON|pepe_trending_up": 5,
        "CHOP_RISK_ON|pepe_chop": 10,
        "CHOP_RISK_ON|pepe_trending_down": 10,
        "BEAR_OR_RISK_OFF|pepe_trending_up": 15,
        "BEAR_OR_RISK_OFF|pepe_chop": 8,
        "BEAR_OR_RISK_OFF|pepe_trending_down": 35,
    }

    # Load chimera per cadence and pre-process
    chimera_dfs = {}
    for cad, cfg in CADENCE_CONFIGS.items():
        print(f"[*] Loading PEPE {cad} chimera...")
        df = pl.read_parquet(cfg["chim_file"]).sort("timestamp")
        chimera_dfs[cad] = df

    # Run Tier A setups (Phase 2 regime-gated + chimera filter)
    tier_a_results = {}
    tier_a_no_chimera_results = {}    # regime-gated, NO chimera filter
    for sid, spec in TIER_A_SETUPS.items():
        print(f"[A] Running {sid}...")
        df_cad = chimera_dfs[spec["cadence"]]
        result = run_setup_on_val_movers(
            sid, spec, val_movers_tagged, df_cad, df_pepe_1d,
            apply_regime_gate=True, apply_chimera_filter=True,
        )
        tier_a_results[sid] = result
        s = result["summary"]
        print(f"    [regime+chim] n_matched={s['n_regime_matched']} n_fired={s['n_fired']} "
              f"med_cap={s['median_capture']} win_rate={s['win_rate']}")
        # Also run WITHOUT chimera filter to isolate regime gate effect
        result_no_chim = run_setup_on_val_movers(
            sid + "_no_chim", spec, val_movers_tagged, df_cad, df_pepe_1d,
            apply_regime_gate=True, apply_chimera_filter=False,
        )
        tier_a_no_chimera_results[sid] = result_no_chim
        s2 = result_no_chim["summary"]
        print(f"    [regime-only] n_matched={s2['n_regime_matched']} n_fired={s2['n_fired']} "
              f"med_cap={s2['median_capture']} win_rate={s2['win_rate']}")
        # Persist per-setup JSON
        with open(OUT_DIR / f"pepe_val_phase2_{sid}.json", "w") as f:
            json.dump({
                "_meta": {
                    "task": f"VAL Phase 2 {sid}",
                    "instance": "MAXX-INST-2026-05-26-NIGHT",
                    "worker": "W2 (Opus)",
                    "ts_generated": datetime.now(timezone.utc).isoformat(),
                    "git_sha": GIT_SHA_AT_RUN,
                    "canonical_seeds": CANONICAL_SEEDS,
                    "val_only": True,
                    "val_window_bars_1d": (VAL_START_BAR_1D, VAL_END_BAR_1D),
                },
                "spec": spec,
                "summary": result["summary"],
                "per_trade": result["per_trade"],
            }, f, indent=2, default=str)

    # Tier B
    tier_b_results = {}
    for sid, spec in TIER_B_SETUPS.items():
        print(f"[B] Running {sid}...")
        df_cad = chimera_dfs[spec["cadence"]]
        result = run_setup_on_val_movers(
            sid, spec, val_movers_tagged, df_cad, df_pepe_1d,
            apply_regime_gate=True, apply_chimera_filter=True,
        )
        tier_b_results[sid] = result
        s = result["summary"]
        print(f"    n_matched={s['n_regime_matched']} n_fired={s['n_fired']} "
              f"med_cap={s['median_capture']} win_rate={s['win_rate']}")
        with open(OUT_DIR / f"pepe_val_phase2_{sid}.json", "w") as f:
            json.dump({
                "_meta": {
                    "task": f"VAL Phase 2 {sid}",
                    "instance": "MAXX-INST-2026-05-26-NIGHT",
                    "worker": "W2 (Opus)",
                    "ts_generated": datetime.now(timezone.utc).isoformat(),
                    "git_sha": GIT_SHA_AT_RUN,
                    "canonical_seeds": CANONICAL_SEEDS,
                    "val_only": True,
                    "val_window_bars_1d": (VAL_START_BAR_1D, VAL_END_BAR_1D),
                },
                "spec": spec,
                "summary": result["summary"],
                "per_trade": result["per_trade"],
            }, f, indent=2, default=str)

    # Phase 1 baselines on VAL (no regime gate, no chimera filter)
    phase1_results = {}
    for cad, baseline in PHASE1_BASELINES.items():
        print(f"[P1] Running phase1 baseline at {cad}...")
        df_cad = chimera_dfs[cad]
        # Use the regime-gate=False, chimera_filter=False mode by feeding a spec
        spec = {
            "cadence": cad,
            "indicator": baseline["indicator"],
            "fast": baseline["fast"],
            "slow": baseline["slow"],
            "regime_macro": None,
            "regime_pepe": None,
            "chimera_filter": None,
            "exit_policy": "E5_48h",
        }
        result = run_setup_on_val_movers(
            f"P1_{cad}_{baseline['indicator']}_{baseline['fast']}_{baseline['slow']}",
            spec, val_movers_tagged, df_cad, df_pepe_1d,
            apply_regime_gate=False, apply_chimera_filter=False,
        )
        phase1_results[cad] = result
        s = result["summary"]
        print(f"    n_fired={s['n_fired']} med_cap={s['median_capture']} win_rate={s['win_rate']}")
        # Persist per-trade for cell-restricted comparison
        with open(OUT_DIR / f"pepe_val_phase1_baseline_{cad}.json", "w") as f:
            json.dump({
                "_meta": {
                    "task": f"VAL Phase 1 baseline {cad}",
                    "instance": "MAXX-INST-2026-05-26-NIGHT",
                    "worker": "W2 (Opus)",
                    "ts_generated": datetime.now(timezone.utc).isoformat(),
                    "git_sha": GIT_SHA_AT_RUN,
                    "canonical_seeds": CANONICAL_SEEDS,
                    "val_only": True,
                },
                "spec": spec,
                "summary": result["summary"],
                "per_trade": result["per_trade"],
            }, f, indent=2, default=str)

    # Cell-restricted Phase 1 baselines (Phase 1 setup restricted to each regime cell of interest)
    print()
    print("[*] Computing CELL-RESTRICTED Phase 1 baselines for fair Phase2 vs Phase1 lift comparison...")
    phase1_cell_restricted = {}
    target_cells = [
        ("BULL_RISK_ON", "pepe_trending_up"),
        ("BEAR_OR_RISK_OFF", "pepe_chop"),
        ("CHOP_RISK_ON", "pepe_trending_down"),
        ("BULL_RISK_ON", "pepe_trending_down"),
    ]
    for cad in ["15m", "30m", "1h", "4h"]:
        p1 = phase1_results[cad]
        for macro, micro in target_cells:
            key = f"{cad}|{macro}|{micro}"
            # filter per_trade by regime_cell
            cell_trades = [t for t in p1["per_trade"]
                            if t.get("regime_cell") == f"{macro}|{micro}"]
            fired = [t for t in cell_trades if t.get("fired")]
            captures = np.array([t["captured_pct"] for t in fired]) if fired else np.array([])
            phase1_cell_restricted[key] = {
                "n_cell_movers": len(cell_trades),
                "n_fired": int(len(fired)),
                "fire_rate": float(len(fired) / max(1, len(cell_trades))) if cell_trades else 0.0,
                "median_capture": float(np.median(captures)) if len(captures) else None,
                "mean_capture": float(np.mean(captures)) if len(captures) else None,
                "win_rate": float((captures > 0).mean()) if len(captures) else None,
            }

    # Summary JSON
    summary_obj = {
        "_meta": {
            "task": "W2 VAL Phase 2 regime-gated + chimera-filter validation",
            "instance": "MAXX-INST-2026-05-26-NIGHT",
            "worker": "W2 (Opus, 2nd round)",
            "ts_generated": datetime.now(timezone.utc).isoformat(),
            "git_sha": GIT_SHA_AT_RUN,
            "canonical_seeds": CANONICAL_SEEDS,
            "val_only": True,
            "val_window_bars_1d": (VAL_START_BAR_1D, VAL_END_BAR_1D),
            "train_only_calibration": True,
        },
        "val_summary": val_summary,
        "val_regime_cell_distribution": dict(cell_dist),
        "train_regime_cell_distribution_reference": train_cell_dist,
        "tier_a_results": {sid: r["summary"] for sid, r in tier_a_results.items()},
        "tier_a_regime_only_no_chimera": {sid: r["summary"] for sid, r in tier_a_no_chimera_results.items()},
        "tier_b_results": {sid: r["summary"] for sid, r in tier_b_results.items()},
        "phase1_baselines": {cad: r["summary"] for cad, r in phase1_results.items()},
        "phase1_cell_restricted": phase1_cell_restricted,
        "tier_a_specs": TIER_A_SETUPS,
        "tier_b_specs": TIER_B_SETUPS,
        "phase1_specs": PHASE1_BASELINES,
        "btc_vol_thresholds": {"p33": btc_p33, "p67": btc_p67},
        "pepe_4h_regime_thresholds": {"p33": pepe_cfg.vol_p33, "p67": pepe_cfg.vol_p67},
        "elapsed_seconds": time.time() - ts_start,
    }
    with open(OUT_DIR / "pepe_val_phase2_summary.json", "w") as f:
        json.dump(summary_obj, f, indent=2, default=str)

    print(f"\n[*] Done. Total elapsed: {time.time() - ts_start:.1f}s")
    print(f"[*] Outputs at: {OUT_DIR / 'pepe_val_phase2_*.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
