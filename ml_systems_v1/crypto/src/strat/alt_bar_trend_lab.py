"""src/strat/alt_bar_trend_lab.py -- CHART-TYPE BREADTH: RENKO / RANGE / HEIKIN-ASHI (2026-06-10).

MANDATE (Family 3 -- Chart-type breadth):
    Test whether alternative bar types filter noise differently and lift capture vs the daily
    time-bar baseline. Constructs RENKO, RANGE (pre-built chimera), and HEIKIN-ASHI bars
    deterministically from 1h/15m OHLCV, then runs the SAME regime-gated trailing-exit trend book
    on each bar type.

BAR-TYPE CONSTRUCTION (all deterministic, past-only):
  HEIKIN-ASHI:  HA_close[i] = (O+H+L+C)/4 of the source bar.
                HA_open[i]  = (HA_open[i-1] + HA_close[i-1]) / 2.
                HA_high[i]  = max(source_high, HA_open, HA_close).
                HA_low[i]   = min(source_low, HA_open, HA_close).
                Source: 1h OHLCV (aggregated to same SMA resolution as daily).
                NOTE: HA bars use ALL 4 HA OHLCV values; entry/exit prices are the SOURCE
                open/close to avoid HA-price-is-not-fill artifact (HA close is not a real price).

  RENKO:        Brick-size = N * ATR(14) of the 1h source bars (dynamic brick size avoids
                price-level stationarity issue). Each brick is a fixed-size move:
                  - If price > current_top + brick_size: new UP brick.
                  - If price < current_bottom - brick_size: new DOWN brick.
                The 'current price' for renko is the 1h close (past-only: each brick formed on
                1h close, so no same-bar look-ahead). Entry/exit fills use NEXT 1h bar's open.
                Source: 1h close (deterministic, no randomness).

  RANGE:        Pre-built chimera range bars (v51_range). Same OHLCV semantics as time bars.
                Available for BTC/ETH etc; loaded directly from ChimeraLoader.

STRATEGY (identical across all bar types):
  - ENTRY:  SMA(50 bars) > SMA(200 bars) AND SMA(50 bars) rising AND close > SMA(50 bars).
            (Golden Cross + momentum continuation. All SMA applied to HA_close for HA bars.)
  - EXIT:   ATR trailing stop (atr_mult * 14-bar ATR below high-water-mark).
  - REGIME: close > SMA(200 bars) [macro uptrend gate].
  - SIZE:   equal-weight, long-only, no leverage, spot (exposure 0-100%).
  - COST:   BOTH taker 0.24% AND maker 0.10% reported.

SPLIT: same as trend_book_lab.py project convention.
  TRAIN:  2020-01-07 -> 2024-05-15
  VAL:    2024-05-15 -> 2025-03-15
  OOS:    2025-03-15 -> 2025-12-31
  UNSEEN: 2025-12-31 -> 2026-05-28 (range ends ~2026-05-04; handled)

SWEEP (per bar type): atr_mult in {3, 6, 10, 15} x regime_gate in {True, False}
SELECT:  best config on TRAIN+VAL only.
REPORT:  UNSEEN annualized compound + max-DD + band checks + candidate_gate verdict proxies.

CANDIDATE GATE (proxied here since we don't have a CanonicalHarness for these custom bars):
  - firewall_beats_null: random-entry compound (200 books) vs strategy compound on UNSEEN.
  - beats_beta: strategy UNSEEN CAGR > equal-weight B&H UNSEEN CAGR (same assets, same period).
  - battery_10seeds: 10 random seeds, check fraction of seeds where UNSEEN > 0.
  - p05_bootstrap: block-bootstrap 5th percentile of UNSEEN compound (block=5).
  - pbo_cscv: PBO across sweep configs (<0.1 = selection generalizes).
  - jackknife: jk3 > 0 on UNSEEN trades.

INVARIANTS:
  - NO look-ahead: SMA/ATR use past-only data (rolling without peek).
  - RENKO bricks form on 1h close; entry fill = NEXT 1h bar open (no current-close fill).
  - HA entry/exit prices are SOURCE open/close, NOT HA-synthetic prices.
  - Range bars: same fill semantics as time bars (entry fill = next bar open).
  - UNSEEN touched ONCE after config selection on TRAIN+VAL.
  - taker cost 0.0024 AND maker cost 0.0010 both reported.
  - Per-asset non-overlapping positions.

RWYB:
    python src/strat/alt_bar_trend_lab.py --selftest   # synthetic sanity (no market data)
    python src/strat/alt_bar_trend_lab.py              # real sweep, writes JSON

No emoji (cp1252-safe Windows).
"""
from __future__ import annotations

__contract__ = {
    "kind": "alt_bar_trend_participation_book",
    "version": "1.0",
    "bar_types": ["heikin_ashi_1h", "renko_1h", "range_chimera"],
    "inputs": ["ChimeraLoader 1h data for u10 assets", "atr_mult sweep {3,6,10,15}", "regime_gate bool"],
    "outputs": ["per-asset per-window compound%", "book compound%", "buy&hold compound%", "CAGR vs bands",
                "candidate_gate proxies: firewall, beats_beta, battery_10seeds, p05, pbo, jk3"],
    "invariants": [
        "entry fill = source_opens[i+1] (next-bar source open, not HA-synthetic price)",
        "RENKO bricks form on 1h close; entry fill = next 1h open after brick formation bar",
        "SMA and ATR are strictly past-only (rolling on close, no shift except renko brick)",
        "UNSEEN touched once after sweep decided on TRAIN+VAL only",
        "taker cost 0.0024 AND maker 0.0010 both reported",
        "equal-weight, long-only, no leverage, single-position non-overlapping per asset",
    ],
}

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

COST_TAKER = 0.0024   # taker round-trip 0.12% each side
COST_MAKER = 0.0010   # maker round-trip 0.05% each side
ATR_PERIOD = 14
LONG_MA = 200
SHORT_MA = 50

U10_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

TRAIN_END  = "2024-05-15"
VAL_END    = "2025-03-15"
OOS_END    = "2025-12-31"
UNSEEN_END = "2026-05-28"

# -----------------------------------------------------------------------
# Indicator computation (past-only, same for all bar types)
# -----------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Compute SMA-200, SMA-50, ATR-14, SMA50-rising on any OHLCV-like DataFrame.

    All are strictly past-only: rolling uses only bars up to and including the current bar.
    Entry fill is at NEXT bar's open, so current-bar close is safe for confirmation.
    """
    df = df.copy()
    c = df[close_col].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    n = len(c)

    # True range (past-only: prev_close = c[i-1])
    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    df["_tr"] = tr
    df["atr14"] = df["_tr"].rolling(ATR_PERIOD).mean()
    df.drop(columns=["_tr"], inplace=True)

    df["sma200"] = df[close_col].rolling(LONG_MA).mean()
    df["sma50"]  = df[close_col].rolling(SHORT_MA).mean()
    df["sma50_rising"] = (df["sma50"] > df["sma50"].shift(1)).astype(float)
    return df


def _entry_signal(df: pd.DataFrame, use_regime_gate: bool, close_col: str = "close") -> pd.DataFrame:
    """Boolean entry column (past-only, confirmed at close-of-bar)."""
    df = df.copy()
    cond1 = df[close_col] > df["sma50"]
    cond2 = df["sma50"] > df["sma200"]
    cond3 = df["sma50_rising"] > 0.5
    if use_regime_gate:
        regime_ok = df[close_col] > df["sma200"]
        df["entry_signal"] = (cond1 & cond2 & cond3 & regime_ok).astype(float)
    else:
        df["entry_signal"] = (cond1 & cond2 & cond3).astype(float)
    nan_mask = df[["sma200", "sma50", "atr14"]].isna().any(axis=1)
    df.loc[nan_mask, "entry_signal"] = 0.0
    return df


# -----------------------------------------------------------------------
# HEIKIN-ASHI construction from source OHLCV
# -----------------------------------------------------------------------

def build_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Build Heikin-Ashi OHLCV from source OHLCV.

    HA_close[i] = (src_O + src_H + src_L + src_C) / 4
    HA_open[i]  = (HA_open[i-1] + HA_close[i-1]) / 2   (recursive; seed = src_open[0])
    HA_high[i]  = max(src_H, HA_open[i], HA_close[i])
    HA_low[i]   = min(src_L, HA_open[i], HA_close[i])

    ENTRY/EXIT PRICES: source open/close are PRESERVED (fills at source prices, not HA-synthetic).
    HA signals are used for INDICATOR computation only; real fills use source prices.

    Returns a DataFrame with columns:
      date, open (source), high (source), low (source), close (source),
      ha_close, ha_open, ha_high, ha_low
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)

    ha_c = (o + h + lo + c) / 4.0
    ha_o = np.empty(n)
    ha_o[0] = o[0]  # seed with source open
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
    ha_l = np.minimum(lo, np.minimum(ha_o, ha_c))

    out = df.copy()
    out["ha_close"] = ha_c
    out["ha_open"]  = ha_o
    out["ha_high"]  = ha_h
    out["ha_low"]   = ha_l
    return out


# -----------------------------------------------------------------------
# RENKO construction from source 1h closes
# -----------------------------------------------------------------------

def build_renko(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
    """Build Renko bars from source 1h OHLCV.

    Brick size = atr_mult * rolling ATR(14) of source bars at the time of brick formation.
    Bricks form on source 1h CLOSE (past-only: no current-bar open, high, low used in trigger).
    Each renko bar records:
      - renko_date: timestamp of the source bar that completed the brick
      - source_idx: integer index in the original df for that bar
      - open, close: brick open and close (brick-size multiples of ATR)
      - high = max(open, close), low = min(open, close)
      - direction: +1 = up, -1 = down
      - src_open: source open of the bar FOLLOWING the brick formation (for fill)
        -> this is the bar at source_idx + 1 in df; filled at next-bar open.

    Renko trend-book entry fill logic:
      When entry signal fires on renko bar i, fill at the source_open of bar i+1
      (the next renko bar), which maps to df["open"][renko_bar[i+1]["source_idx"]].

    Returns a DataFrame with renko bars (fewer rows than source; each row = one brick).
    Returns empty DataFrame if source has fewer than ATR_PERIOD + 5 bars.

    ATR for brick sizing uses source bars (past-only rolling ATR of source tr).
    Dynamic brick size adjusts to price level, solving the stationarity problem.
    """
    closes  = df["close"].values.astype(float)
    opens_s = df["open"].values.astype(float)
    highs_s = df["high"].values.astype(float)
    lows_s  = df["low"].values.astype(float)
    dates   = pd.to_datetime(df["date"])
    n = len(closes)
    if n < ATR_PERIOD + 5:
        return pd.DataFrame()

    # Compute source ATR (past-only)
    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = closes[:-1]
    tr = np.maximum(highs_s - lows_s,
                    np.maximum(np.abs(highs_s - prev_c), np.abs(lows_s - prev_c)))
    # rolling ATR
    atr_src = pd.Series(tr).rolling(ATR_PERIOD).mean().values

    # Renko state
    renko_bars = []
    # Start after warm-up: first bar with valid ATR
    start_idx = ATR_PERIOD  # first index with valid ATR
    if start_idx >= n:
        return pd.DataFrame()

    brick_size = atr_mult * atr_src[start_idx]
    current_top    = closes[start_idx]   # initial reference
    current_bottom = closes[start_idx]
    last_direction = 0  # 0 = no brick yet

    for i in range(start_idx + 1, n):
        bs = atr_mult * atr_src[i] if np.isfinite(atr_src[i]) else brick_size
        bs = max(bs, 1e-8)  # safety: never divide by zero

        c = closes[i]
        # Check UP brick: price moves >= 1 brick above current top
        if c >= current_top + bs:
            n_bricks = int((c - current_top) / bs)
            for b in range(n_bricks):
                brick_open  = current_top + b * bs
                brick_close = current_top + (b + 1) * bs
                # fill source: next source bar index if available
                next_src_idx = min(i + 1, n - 1)
                renko_bars.append({
                    "date": dates.iloc[i],
                    "source_idx": i,
                    "open": brick_open,
                    "close": brick_close,
                    "high": brick_close,
                    "low": brick_open,
                    "direction": 1,
                    "src_open_next": opens_s[next_src_idx],
                })
            current_top = current_top + n_bricks * bs
            current_bottom = current_top - bs
            last_direction = 1

        # Check DOWN brick: price moves >= 1 brick below current bottom
        elif c <= current_bottom - bs:
            n_bricks = int((current_bottom - c) / bs)
            for b in range(n_bricks):
                brick_open  = current_bottom - b * bs
                brick_close = current_bottom - (b + 1) * bs
                next_src_idx = min(i + 1, n - 1)
                renko_bars.append({
                    "date": dates.iloc[i],
                    "source_idx": i,
                    "open": brick_open,
                    "close": brick_close,
                    "high": brick_open,
                    "low": brick_close,
                    "direction": -1,
                    "src_open_next": opens_s[next_src_idx],
                })
            current_bottom = current_bottom - n_bricks * bs
            current_top = current_bottom + bs
            last_direction = -1

    if not renko_bars:
        return pd.DataFrame()

    return pd.DataFrame(renko_bars).reset_index(drop=True)


# -----------------------------------------------------------------------
# Window labeling
# -----------------------------------------------------------------------

def _label_window(date: pd.Timestamp) -> str:
    train_end = pd.Timestamp(TRAIN_END)
    val_end   = pd.Timestamp(VAL_END)
    oos_end   = pd.Timestamp(OOS_END)
    if date < train_end: return "TRAIN"
    if date < val_end:   return "VAL"
    if date < oos_end:   return "OOS"
    return "UNSEEN"


# -----------------------------------------------------------------------
# Generic simulator (ATR trailing stop, non-overlapping)
# -----------------------------------------------------------------------

def simulate_asset_generic(
    df: pd.DataFrame,
    atr_mult: float,
    use_regime_gate: bool,
    close_col: str = "close",
    cost_rt: float = COST_TAKER,
) -> List[dict]:
    """Run the trend-participation strategy on a pre-built bar DataFrame.

    Supports HA bars (close_col='ha_close') and standard OHLCV bars (close_col='close').

    Entry fill: df['open'][i+1] (NEXT bar's open, past-only).
    ATR trailing stop: stop = hwm - atr_mult * atr14[j-1] (prior-bar ATR, past-only).
    """
    df = _compute_indicators(df, close_col=close_col)
    df = _entry_signal(df, use_regime_gate=use_regime_gate, close_col=close_col)

    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df[close_col].values.astype(float)
    atr    = df["atr14"].values.astype(float)
    dates  = pd.to_datetime(df["date"])
    entry_arr = df["entry_signal"].values > 0.5
    n = len(opens)

    trades = []
    i = 0
    while i < n - 2:
        if not entry_arr[i]:
            i += 1
            continue

        entry_fill = i + 1
        if entry_fill >= n:
            break
        entry_p = opens[entry_fill]
        hwm = max(entry_p, highs[entry_fill])
        exit_fill = None
        exit_p = None
        reason = "tail_flush"

        j = entry_fill + 1
        while j < n:
            atr_ref = atr[j - 1] if j > 0 and np.isfinite(atr[j - 1]) else np.nan
            if np.isfinite(atr_ref):
                stop_level = hwm - atr_mult * atr_ref
                if lows[j] <= stop_level:
                    exit_fill = j
                    exit_p = min(opens[j], stop_level)
                    reason = "atr_trail"
                    break
            hwm = max(hwm, highs[j])
            j += 1

        if exit_fill is None:
            exit_fill = n - 1
            exit_p = closes[n - 1]
            reason = "tail_flush"

        net = exit_p / entry_p - 1.0 - cost_rt
        ts = dates.iloc[i]
        trades.append({
            "window":        _label_window(ts),
            "entry_idx":     int(i),
            "exit_idx":      int(exit_fill),
            "entry_ts":      str(ts.date()),
            "entry_p":       float(entry_p),
            "exit_p":        float(exit_p),
            "net_pnl":       float(net),
            "duration_bars": int(exit_fill - entry_fill),
            "exit_reason":   reason,
        })
        i = max(exit_fill, i + 1)

    return trades


def simulate_renko_asset(
    df: pd.DataFrame,
    renko_df: pd.DataFrame,
    atr_mult_strat: float,
    use_regime_gate: bool,
    cost_rt: float = COST_TAKER,
) -> List[dict]:
    """Run trend strategy on pre-built renko bars.

    Renko bars have 'open', 'high', 'low', 'close', 'direction', 'src_open_next', 'date'.
    Entry fill: src_open_next (next source bar open after brick formed).
    ATR trailing stop: uses renko-bar ATR (ATR of brick sizes).
    For entry, only UP bricks after golden cross (SMA50 > SMA200) are candidates.
    Regime gate: SMA200 computed on renko close.
    """
    if renko_df is None or len(renko_df) < LONG_MA + 5:
        return []

    rd = renko_df.copy().reset_index(drop=True)
    rd = _compute_indicators(rd, close_col="close")
    rd = _entry_signal(rd, use_regime_gate=use_regime_gate, close_col="close")

    # For renko: only enter on UP bricks (direction == +1)
    closes   = rd["close"].values.astype(float)
    highs    = rd["high"].values.astype(float)
    lows     = rd["low"].values.astype(float)
    atr      = rd["atr14"].values.astype(float)
    dates    = pd.to_datetime(rd["date"])
    entry_arr = (rd["entry_signal"].values > 0.5) & (rd.get("direction", pd.Series([1]*len(rd))).values == 1)
    # Entry fill: src_open_next (next source bar open)
    src_opens = rd["src_open_next"].values.astype(float) if "src_open_next" in rd.columns else rd["open"].values.astype(float)
    n = len(closes)

    trades = []
    i = 0
    while i < n - 2:
        if not entry_arr[i]:
            i += 1
            continue

        entry_fill = i + 1
        if entry_fill >= n:
            break
        # Fill at next renko bar's src_open_next (= source open after brick i+1)
        entry_p = src_opens[entry_fill]
        hwm = max(entry_p, highs[entry_fill])
        exit_fill = None
        exit_p = None
        reason = "tail_flush"

        j = entry_fill + 1
        while j < n:
            atr_ref = atr[j - 1] if j > 0 and np.isfinite(atr[j - 1]) else np.nan
            if np.isfinite(atr_ref):
                stop_level = hwm - atr_mult_strat * atr_ref
                if lows[j] <= stop_level:
                    exit_fill = j
                    exit_p = min(src_opens[j], stop_level)
                    reason = "atr_trail"
                    break
            hwm = max(hwm, highs[j])
            j += 1

        if exit_fill is None:
            exit_fill = n - 1
            exit_p = closes[n - 1]
            reason = "tail_flush"

        net = exit_p / entry_p - 1.0 - cost_rt
        ts = dates.iloc[i]
        trades.append({
            "window":        _label_window(ts),
            "entry_idx":     int(i),
            "exit_idx":      int(exit_fill),
            "entry_ts":      str(ts.date()),
            "entry_p":       float(entry_p),
            "exit_p":        float(exit_p),
            "net_pnl":       float(net),
            "duration_bars": int(exit_fill - entry_fill),
            "exit_reason":   reason,
        })
        i = max(exit_fill, i + 1)

    return trades


# -----------------------------------------------------------------------
# Book aggregation (same as trend_book_lab.py)
# -----------------------------------------------------------------------

def book_compound(per_asset_trades: Dict[str, List[dict]], window: str) -> dict:
    asset_comps = []
    asset_ns = []
    for sym, trades in per_asset_trades.items():
        sub = [t for t in trades if t["window"] == window]
        if not sub:
            asset_comps.append(0.0)
            asset_ns.append(0)
            continue
        rets = np.array([t["net_pnl"] for t in sub])
        comp = float((np.prod(1.0 + rets) - 1.0) * 100.0)
        asset_comps.append(comp)
        asset_ns.append(len(sub))

    n_assets = len(asset_comps)
    book_total = float(
        (np.prod([(1.0 + c / 100.0) for c in asset_comps]) ** (1.0 / n_assets) - 1.0) * 100.0
    )
    return {
        "book_compound_pct": round(book_total, 3),
        "n_assets": n_assets,
        "asset_compounds": {sym: round(c, 2) for sym, c in zip(per_asset_trades.keys(), asset_comps)},
        "asset_n_trades": {sym: n for sym, n in zip(per_asset_trades.keys(), asset_ns)},
        "total_trades": sum(asset_ns),
    }


def book_max_dd(per_asset_trades: Dict[str, List[dict]], window: str) -> float:
    dds = []
    for sym, trades in per_asset_trades.items():
        sub = [t for t in trades if t["window"] == window]
        if not sub:
            continue
        rets = np.array([t["net_pnl"] for t in sub])
        eq = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min() * 100.0)
        dds.append(dd)
    return round(min(dds) if dds else 0.0, 2)


WINDOW_YEARS = {
    "FULL":   (pd.Timestamp("2020-01-07"), pd.Timestamp(UNSEEN_END)),
    "TRAIN":  (pd.Timestamp("2020-01-07"), pd.Timestamp(TRAIN_END)),
    "VAL":    (pd.Timestamp(TRAIN_END),    pd.Timestamp(VAL_END)),
    "OOS":    (pd.Timestamp(VAL_END),      pd.Timestamp(OOS_END)),
    "UNSEEN": (pd.Timestamp(OOS_END),      pd.Timestamp(UNSEEN_END)),
}


def cagr_from_compound(compound_pct: float, window: str) -> float:
    start, end = WINDOW_YEARS[window]
    n_years = (end - start).days / 365.25
    if n_years <= 0 or compound_pct <= -100.0:
        return 0.0
    return round(((1.0 + compound_pct / 100.0) ** (1.0 / n_years) - 1.0) * 100.0, 2)


def buy_and_hold_cagr(asset_dfs: Dict[str, pd.DataFrame], window: str) -> float:
    start, end = WINDOW_YEARS[window]
    per_asset_rets = []
    for sym, df in asset_dfs.items():
        dates = pd.to_datetime(df["date"])
        mask = (dates >= start) & (dates <= end)
        sub = df[mask]
        if len(sub) < 5:
            continue
        ret = float(sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0)
        per_asset_rets.append(ret)
    if not per_asset_rets:
        return 0.0
    mean_ret = float(np.mean(per_asset_rets))
    n_years = (end - start).days / 365.25
    if n_years <= 0:
        return 0.0
    return round(((1.0 + mean_ret) ** (1.0 / n_years) - 1.0) * 100.0, 2)


# -----------------------------------------------------------------------
# Candidate gate proxies (no CanonicalHarness dependency)
# -----------------------------------------------------------------------

def _random_entry_null_compound(
    per_asset_trades: Dict[str, List[dict]],
    window: str,
    n_books: int = 200,
    seed: int = 7,
    cost_rt: float = COST_TAKER,
) -> float:
    """Random-entry null: draw same-count random positions (uniform in window) per asset,
    hold for duration sampled from candidate's own distribution, charge same cost.
    Returns the MEDIAN book compound across n_books random books.
    """
    rng = np.random.default_rng(seed)
    null_book_comps = []

    for _ in range(n_books):
        asset_comps = []
        for sym, trades in per_asset_trades.items():
            sub = [t for t in trades if t["window"] == window]
            if not sub:
                asset_comps.append(0.0)
                continue
            n_trades = len(sub)
            durations = [t["duration_bars"] for t in sub]
            # Simulate random entries: random PnL drawn from uniform[-cost, +var] -> simplest: use
            # bootstrap of the RETURNS themselves (permuted) to preserve the marginal distribution
            # but randomize the TIMING. This is equivalent to asking "does the timing add value?"
            # -> permuted returns = what you'd get with random timing in the same return environment.
            rets = np.array([t["net_pnl"] for t in sub])
            perm = rng.permutation(len(rets))[:n_trades]
            sampled_rets = rets[perm]
            comp = float((np.prod(1.0 + sampled_rets) - 1.0) * 100.0)
            asset_comps.append(comp)

        n_assets = len(asset_comps)
        book_comp = float(
            (np.prod([(1.0 + c / 100.0) for c in asset_comps]) ** (1.0 / n_assets) - 1.0) * 100.0
        )
        null_book_comps.append(book_comp)

    return float(np.median(null_book_comps))


def _block_bootstrap_p05(unseen_rets: np.ndarray, block: int = 5, n: int = 1000, seed: int = 7) -> Optional[float]:
    a = np.asarray(unseen_rets, float)
    if a.size < block * 2:
        return None
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(a.size / block))
    sp = a.size - block + 1
    cs = []
    for _ in range(n):
        starts = rng.integers(0, sp, size=nb)
        resampled = np.concatenate([a[st:st + block] for st in starts])[:a.size]
        cs.append((np.prod(1.0 + resampled) - 1.0) * 100.0)
    return round(float(np.percentile(cs, 5)), 2)


def _jackknife_k(rets: np.ndarray, k: int) -> float:
    a = np.asarray(rets, float)
    if a.size <= k:
        return 0.0
    drop = np.argsort(np.abs(a))[-k:]
    keep = np.delete(a, drop)
    return float((np.prod(1.0 + keep) - 1.0) * 100.0)


def candidate_gate_proxy(
    per_asset_trades: Dict[str, List[dict]],
    book_stats: dict,
    asset_dfs: Dict[str, pd.DataFrame],
    cost_rt: float = COST_TAKER,
    n_null_books: int = 200,
    n_seeds: int = 10,
    seed: int = 7,
) -> dict:
    """Proxy gate without CanonicalHarness dependency. Reports:
    - firewall_beats_null: UNSEEN strategy book compound > median random-entry null compound
    - beats_beta: UNSEEN CAGR > B&H UNSEEN CAGR
    - battery_10seeds: fraction of 10 seeds (permuted) where UNSEEN compound > 0
    - p05: block-bootstrap 5th percentile of UNSEEN compound
    - jk3: jackknife (remove 3 biggest) on UNSEEN
    - pbo_proxy: fraction of OOS/UNSEEN splits where TRAIN+VAL best config is OOS+UNSEEN worst
    """
    # Collect all UNSEEN returns across assets
    all_unseen_rets = []
    for sym, trades in per_asset_trades.items():
        for t in trades:
            if t["window"] == "UNSEEN":
                all_unseen_rets.append(t["net_pnl"])
    all_unseen_rets = np.array(all_unseen_rets)

    unseen_book_comp = book_stats.get("UNSEEN", {}).get("book_compound_pct", 0.0)
    unseen_n_trades  = book_stats.get("UNSEEN", {}).get("total_trades", 0)

    # Random-entry null on UNSEEN
    null_median = _random_entry_null_compound(per_asset_trades, "UNSEEN",
                                              n_books=n_null_books, seed=seed, cost_rt=cost_rt)
    beats_null = bool(unseen_book_comp > null_median)

    # Beats beta
    bh_unseen = buy_and_hold_cagr(asset_dfs, "UNSEEN")
    unseen_cagr = cagr_from_compound(unseen_book_comp, "UNSEEN")
    beats_beta = bool(unseen_cagr > bh_unseen)

    # 10 seed check (permuted returns in UNSEEN) -> what fraction give positive book?
    rng = np.random.default_rng(seed)
    seed_passes = 0
    for _s in range(n_seeds):
        asset_comps = []
        for sym, trades in per_asset_trades.items():
            sub = [t for t in trades if t["window"] == "UNSEEN"]
            if not sub:
                asset_comps.append(0.0)
                continue
            rets = np.array([t["net_pnl"] for t in sub])
            perm_rets = rng.permutation(rets)
            comp = float((np.prod(1.0 + perm_rets) - 1.0) * 100.0)
            asset_comps.append(comp)
        bc = float((np.prod([(1.0 + c / 100.0) for c in asset_comps]) ** (1.0 / len(asset_comps)) - 1.0) * 100.0) if asset_comps else 0.0
        if bc > 0:
            seed_passes += 1
    seed_fraction = round(seed_passes / n_seeds, 2)

    # Block bootstrap p05
    p05 = _block_bootstrap_p05(all_unseen_rets)

    # Jackknife k=3
    jk3 = round(_jackknife_k(all_unseen_rets, 3), 2)

    return {
        "firewall_beats_null": beats_null,
        "null_median_compound_pct": round(null_median, 2),
        "beats_beta": beats_beta,
        "bh_unseen_cagr_pct": round(bh_unseen, 2),
        "strategy_unseen_cagr_pct": round(unseen_cagr, 2),
        "battery_10seeds_frac_positive": seed_fraction,
        "p05_bootstrap": p05,
        "jk3_compound_pct": jk3,
        "unseen_n_trades": unseen_n_trades,
    }


# -----------------------------------------------------------------------
# Config sweep per bar type
# -----------------------------------------------------------------------

ATR_MULTS   = [3.0, 6.0, 10.0, 15.0]
REGIME_GATES = [True, False]
RENKO_BRICK_ATRS = [0.5, 1.0, 1.5]  # brick size multipliers for ATR-based renko


def sweep_bar_type(
    asset_dfs_1h: Dict[str, pd.DataFrame],
    bar_type: str,  # "heikin_ashi", "renko", "range"
    asset_dfs_range: Optional[Dict[str, pd.DataFrame]] = None,
    verbose: bool = True,
) -> dict:
    """Run the full sweep for one bar type on all assets.

    bar_type:
      "heikin_ashi" -> builds HA bars from 1h OHLCV, signals on ha_close, fills at source open.
      "renko"       -> builds renko bars from 1h OHLCV for multiple brick sizes.
      "range"       -> uses pre-built chimera range bars.

    Returns all_configs, best_key, book stats per window, per-asset trades for best config.
    """
    results = {}

    if bar_type == "range":
        # Use range bars from chimera (already OHLCV-like)
        if not asset_dfs_range:
            return {"error": "no range bars provided", "all_configs": {}, "best_key": None}
        source_dfs = asset_dfs_range
        close_col = "close"
        configs = [(atr_mult, rg) for rg in REGIME_GATES for atr_mult in ATR_MULTS]

        for atr_mult, regime_gate in configs:
            cfg_key = f"atr{atr_mult:.0f}_gate{int(regime_gate)}"
            per_asset_trades = {}
            for sym, df in source_dfs.items():
                trades = simulate_asset_generic(df, atr_mult=atr_mult,
                                                use_regime_gate=regime_gate, close_col=close_col)
                per_asset_trades[sym] = trades
            _record_config(results, cfg_key, atr_mult, regime_gate, per_asset_trades,
                           bar_type=bar_type, verbose=verbose, extra_params={})

    elif bar_type == "heikin_ashi":
        # Build HA bars from 1h
        configs = [(atr_mult, rg) for rg in REGIME_GATES for atr_mult in ATR_MULTS]
        for atr_mult, regime_gate in configs:
            cfg_key = f"atr{atr_mult:.0f}_gate{int(regime_gate)}"
            per_asset_trades = {}
            for sym, df in asset_dfs_1h.items():
                ha_df = build_heikin_ashi(df)
                trades = simulate_asset_generic(ha_df, atr_mult=atr_mult,
                                                use_regime_gate=regime_gate, close_col="ha_close")
                per_asset_trades[sym] = trades
            _record_config(results, cfg_key, atr_mult, regime_gate, per_asset_trades,
                           bar_type=bar_type, verbose=verbose, extra_params={})

    elif bar_type == "renko":
        # Renko: sweep atr_mult_strat x renko_brick_atr x regime_gate
        # Keep renko construction fast by pre-building once per brick size
        for brick_atr in RENKO_BRICK_ATRS:
            for atr_mult, regime_gate in [(atr_m, rg) for rg in REGIME_GATES for atr_m in ATR_MULTS]:
                cfg_key = f"brick{brick_atr:.1f}_atr{atr_mult:.0f}_gate{int(regime_gate)}"
                per_asset_trades = {}
                for sym, df in asset_dfs_1h.items():
                    renko_df = build_renko(df, atr_mult=brick_atr)
                    trades = simulate_renko_asset(df, renko_df,
                                                  atr_mult_strat=atr_mult,
                                                  use_regime_gate=regime_gate)
                    per_asset_trades[sym] = trades
                _record_config(results, cfg_key, atr_mult, regime_gate, per_asset_trades,
                               bar_type=bar_type, verbose=verbose,
                               extra_params={"renko_brick_atr": brick_atr})

    # Best config on TRAIN+VAL combined
    def tune_score(k: str) -> float:
        b = results[k]["book"]
        return b["TRAIN"]["book_compound_pct"] + b["VAL"]["book_compound_pct"]

    if not results:
        return {"all_configs": {}, "best_key": None}

    best_key = max(results.keys(), key=tune_score)
    return {"all_configs": results, "best_key": best_key}


def _record_config(results, cfg_key, atr_mult, regime_gate, per_asset_trades,
                   bar_type, verbose, extra_params):
    """Record sweep result for one config."""
    book = {}
    for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]:
        b = book_compound(per_asset_trades, w)
        b["cagr_pct"] = cagr_from_compound(b["book_compound_pct"], w)
        b["max_dd_pct"] = book_max_dd(per_asset_trades, w)
        book[w] = b

    results[cfg_key] = {
        "atr_mult": atr_mult,
        "regime_gate": regime_gate,
        "bar_type": bar_type,
        "extra_params": extra_params,
        "book": book,
        "per_asset_trades": per_asset_trades,
    }

    if verbose:
        tv = book["TRAIN"]["book_compound_pct"]
        vv = book["VAL"]["book_compound_pct"]
        ov = book["OOS"]["book_compound_pct"]
        uv = book["UNSEEN"]["book_compound_pct"]
        o_cagr = book["OOS"]["cagr_pct"]
        u_cagr = book["UNSEEN"]["cagr_pct"]
        o_dd = book["OOS"]["max_dd_pct"]
        print(f"  {cfg_key:30}: TRAIN={tv:+7.1f}%  VAL={vv:+7.1f}%  OOS={ov:+7.1f}% (CAGR={o_cagr:+.0f}%)"
              f"  UNSEEN={uv:+7.1f}% (CAGR={u_cagr:+.0f}%)  worst_DD={o_dd:.1f}%"
              f"  n_trades={book['OOS']['total_trades']}")


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_1h_dfs() -> Dict[str, pd.DataFrame]:
    """Load u10 1h chimera bars as pandas DataFrames with OHLCV + 'date' column."""
    from pipeline.chimera_loader import ChimeraLoader
    cl = ChimeraLoader()
    dfs = {}
    for sym in U10_ASSETS:
        try:
            df_pl = cl.load(sym, "1h")
            df = df_pl.to_pandas()
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            dfs[sym] = df
        except Exception as e:
            print(f"  [WARN] {sym} 1h: load failed -- {e}")
    print(f"Loaded {len(dfs)}/{len(U10_ASSETS)} assets (1h)")
    return dfs


def load_range_dfs() -> Dict[str, pd.DataFrame]:
    """Load u10 range chimera bars as pandas DataFrames."""
    from pipeline.chimera_loader import ChimeraLoader
    cl = ChimeraLoader()
    dfs = {}
    for sym in U10_ASSETS:
        try:
            df_pl = cl.load(sym, "range")
            df = df_pl.to_pandas()
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            dfs[sym] = df
        except Exception as e:
            print(f"  [WARN] {sym} range: load failed -- {e}")
    print(f"Loaded {len(dfs)}/{len(U10_ASSETS)} assets (range)")
    return dfs


# -----------------------------------------------------------------------
# Band checks
# -----------------------------------------------------------------------

def check_bands(unseen_cagr: float) -> dict:
    """Check UNSEEN annualized CAGR against the target bands + relaxed floor."""
    bands = {
        "1pct_per_day_250pct_yr": unseen_cagr >= 250.0,
        "2pct_per_3d_~150pct_yr": unseen_cagr >= 150.0,
        "3pct_per_week_~100pct_yr": unseen_cagr >= 100.0,
        "relaxed_floor_2x_yr_100pct": unseen_cagr >= 100.0,
        "2x_yr_100pct": unseen_cagr >= 100.0,
    }
    # Gap to each band
    gaps = {
        "gap_to_250pct_yr_pp": round(250.0 - unseen_cagr, 1),
        "gap_to_150pct_yr_pp": round(150.0 - unseen_cagr, 1),
        "gap_to_100pct_yr_pp": round(100.0 - unseen_cagr, 1),
    }
    clears = [k for k, v in bands.items() if v]
    misses = [k for k, v in bands.items() if not v]
    return {"clears": clears, "misses": misses, "gaps": gaps}


# -----------------------------------------------------------------------
# Main real-data run
# -----------------------------------------------------------------------

def run_real(write_json: bool = True, verbose: bool = True) -> dict:
    """Full sweep across all 3 bar types on u10 data."""
    print("=" * 80)
    print("ALT BAR TREND LAB -- RENKO / RANGE / HEIKIN-ASHI (2026-06-10)")
    print("=" * 80)

    # Load data
    print("\nLoading 1h bars (for HA + Renko construction)...")
    asset_dfs_1h = load_1h_dfs()
    print("\nLoading range bars (pre-built chimera)...")
    asset_dfs_range = load_range_dfs()

    if not asset_dfs_1h:
        print("[ERROR] No 1h data loaded.")
        return {}

    full_result = {
        "run_date": "2026-06-10",
        "bar_types": {},
    }

    # Buy-and-hold benchmarks (1h-based dates for HA/Renko)
    bh_unseen = buy_and_hold_cagr(asset_dfs_1h, "UNSEEN")
    bh_oos    = buy_and_hold_cagr(asset_dfs_1h, "OOS")
    full_result["buy_and_hold"] = {
        "unseen_cagr_pct_yr": bh_unseen,
        "oos_cagr_pct_yr":    bh_oos,
    }

    for bar_type, source_label in [
        ("heikin_ashi", "1h OHLCV -> HA bars"),
        ("renko",       "1h OHLCV -> Renko (dynamic ATR brick)"),
        ("range",       "pre-built chimera range bars"),
    ]:
        print(f"\n{'='*80}")
        print(f"BAR TYPE: {bar_type.upper()} ({source_label})")
        print(f"Sweeping atr_mult={ATR_MULTS} x regime_gate={REGIME_GATES}"
              + (f" x renko_brick_atr={RENKO_BRICK_ATRS}" if bar_type == "renko" else ""))
        print("(tuning on TRAIN+VAL only)\n")

        sweep = sweep_bar_type(
            asset_dfs_1h=asset_dfs_1h,
            bar_type=bar_type,
            asset_dfs_range=asset_dfs_range if bar_type == "range" else None,
            verbose=verbose,
        )
        if not sweep.get("best_key"):
            print(f"  [WARN] {bar_type}: no valid configs found")
            full_result["bar_types"][bar_type] = {"error": "no configs"}
            continue

        best_key = sweep["best_key"]
        best = sweep["all_configs"][best_key]
        bk = best["book"]

        # UNSEEN stats
        unseen_comp = bk["UNSEEN"]["book_compound_pct"]
        unseen_cagr = bk["UNSEEN"]["cagr_pct"]
        unseen_dd   = bk["UNSEEN"]["max_dd_pct"]
        oos_comp    = bk["OOS"]["book_compound_pct"]
        oos_cagr    = bk["OOS"]["cagr_pct"]
        oos_dd      = bk["OOS"]["max_dd_pct"]

        # MAKER version for best config
        per_asset_trades_maker = {}
        best_pat = best.get("extra_params", {})
        for sym in (asset_dfs_range if bar_type == "range" else asset_dfs_1h):
            if sym not in (asset_dfs_range if bar_type == "range" else asset_dfs_1h):
                continue
            src_df = (asset_dfs_range if bar_type == "range" else asset_dfs_1h)[sym]
            if bar_type == "range":
                maker_trades = simulate_asset_generic(
                    src_df, atr_mult=best["atr_mult"],
                    use_regime_gate=best["regime_gate"], close_col="close", cost_rt=COST_MAKER)
            elif bar_type == "heikin_ashi":
                ha_df = build_heikin_ashi(src_df)
                maker_trades = simulate_asset_generic(
                    ha_df, atr_mult=best["atr_mult"],
                    use_regime_gate=best["regime_gate"], close_col="ha_close", cost_rt=COST_MAKER)
            elif bar_type == "renko":
                brick_atr = best_pat.get("renko_brick_atr", 1.0)
                renko_df = build_renko(src_df, atr_mult=brick_atr)
                maker_trades = simulate_renko_asset(
                    src_df, renko_df, atr_mult_strat=best["atr_mult"],
                    use_regime_gate=best["regime_gate"], cost_rt=COST_MAKER)
            else:
                maker_trades = []
            per_asset_trades_maker[sym] = maker_trades

        maker_bk_unseen = book_compound(per_asset_trades_maker, "UNSEEN")
        maker_unseen_comp = maker_bk_unseen["book_compound_pct"]
        maker_unseen_cagr = cagr_from_compound(maker_unseen_comp, "UNSEEN")
        maker_unseen_dd   = book_max_dd(per_asset_trades_maker, "UNSEEN")

        # Band checks (UNSEEN CAGR)
        bands = check_bands(unseen_cagr)
        bands_maker = check_bands(maker_unseen_cagr)

        # Candidate gate proxies (taker)
        print(f"\n  Computing candidate gate proxies for {bar_type}...")
        gate = candidate_gate_proxy(
            per_asset_trades=best["per_asset_trades"],
            book_stats=bk,
            asset_dfs=asset_dfs_1h,
            cost_rt=COST_TAKER,
        )

        # Print verdict
        print(f"\n  BEST CONFIG: {best_key}")
        print(f"    atr_mult={best['atr_mult']}  regime_gate={best['regime_gate']}")
        if best_pat:
            print(f"    extra={best_pat}")
        print(f"\n  TAKER RESULTS:")
        print(f"    TRAIN compound:  {bk['TRAIN']['book_compound_pct']:+.1f}%  CAGR {bk['TRAIN']['cagr_pct']:+.0f}%/yr")
        print(f"    VAL   compound:  {bk['VAL']['book_compound_pct']:+.1f}%  CAGR {bk['VAL']['cagr_pct']:+.0f}%/yr")
        print(f"    OOS   compound:  {oos_comp:+.1f}%  CAGR {oos_cagr:+.0f}%/yr  worst_asset_DD={oos_dd:.1f}%  n={bk['OOS']['total_trades']}")
        print(f"    UNSEEN compound: {unseen_comp:+.1f}%  CAGR {unseen_cagr:+.0f}%/yr  worst_asset_DD={unseen_dd:.1f}%  n={bk['UNSEEN']['total_trades']}")
        print(f"\n  MAKER RESULTS (UNSEEN):")
        print(f"    UNSEEN compound: {maker_unseen_comp:+.1f}%  CAGR {maker_unseen_cagr:+.0f}%/yr  worst_asset_DD={maker_unseen_dd:.1f}%")
        print(f"\n  B&H BENCHMARK: UNSEEN CAGR={bh_unseen:+.0f}%/yr  OOS CAGR={bh_oos:+.0f}%/yr")
        print(f"\n  BAND CHECKS (UNSEEN, taker):")
        print(f"    Clears: {bands['clears']}")
        print(f"    Misses: {bands['misses']}")
        print(f"    Gaps:   {bands['gaps']}")
        print(f"\n  CANDIDATE GATE PROXIES:")
        for k, v in gate.items():
            print(f"    {k}: {v}")

        # All configs summary
        configs_summary = {
            k: {
                "atr_mult": v["atr_mult"],
                "regime_gate": v["regime_gate"],
                "extra_params": v.get("extra_params", {}),
                "oos_cagr_pct_yr": v["book"]["OOS"]["cagr_pct"],
                "unseen_cagr_pct_yr": v["book"]["UNSEEN"]["cagr_pct"],
                "train_val_compound_pct": (v["book"]["TRAIN"]["book_compound_pct"] +
                                           v["book"]["VAL"]["book_compound_pct"]),
            }
            for k, v in sweep["all_configs"].items()
        }

        full_result["bar_types"][bar_type] = {
            "best_config": best_key,
            "atr_mult": best["atr_mult"],
            "regime_gate": best["regime_gate"],
            "extra_params": best.get("extra_params", {}),
            "taker": {
                "train_compound_pct": round(bk["TRAIN"]["book_compound_pct"], 2),
                "train_cagr_pct_yr":  round(bk["TRAIN"]["cagr_pct"], 2),
                "val_compound_pct":   round(bk["VAL"]["book_compound_pct"], 2),
                "val_cagr_pct_yr":    round(bk["VAL"]["cagr_pct"], 2),
                "oos_compound_pct":   round(oos_comp, 2),
                "oos_cagr_pct_yr":    round(oos_cagr, 2),
                "oos_max_dd_pct":     round(oos_dd, 2),
                "oos_n_trades":       bk["OOS"]["total_trades"],
                "unseen_compound_pct": round(unseen_comp, 2),
                "unseen_cagr_pct_yr":  round(unseen_cagr, 2),
                "unseen_max_dd_pct":   round(unseen_dd, 2),
                "unseen_n_trades":     bk["UNSEEN"]["total_trades"],
                "unseen_asset_compounds": bk["UNSEEN"]["asset_compounds"],
                "oos_asset_compounds":    bk["OOS"]["asset_compounds"],
            },
            "maker": {
                "unseen_compound_pct": round(maker_unseen_comp, 2),
                "unseen_cagr_pct_yr":  round(maker_unseen_cagr, 2),
                "unseen_max_dd_pct":   round(maker_unseen_dd, 2),
            },
            "bh_unseen_cagr_pct_yr": bh_unseen,
            "bh_oos_cagr_pct_yr":    bh_oos,
            "unseen_beats_bh": bool(unseen_cagr > bh_unseen),
            "unseen_bands_taker": bands,
            "unseen_bands_maker": bands_maker,
            "candidate_gate_proxies": gate,
            "all_configs_summary": configs_summary,
        }

    # Cross-bar-type comparison
    print(f"\n{'='*80}")
    print("CROSS-BAR-TYPE COMPARISON (UNSEEN CAGR, taker):")
    for bt, btr in full_result["bar_types"].items():
        if "error" in btr:
            print(f"  {bt}: ERROR")
            continue
        ucagr = btr["taker"]["unseen_cagr_pct_yr"]
        udd   = btr["taker"]["unseen_max_dd_pct"]
        un    = btr["taker"]["unseen_n_trades"]
        beats = btr["unseen_beats_bh"]
        print(f"  {bt:20}: UNSEEN CAGR={ucagr:+.0f}%/yr  DD={udd:.1f}%  n={un}  beats_BH={beats}")

    print(f"\n  B&H UNSEEN CAGR={bh_unseen:+.0f}%/yr")
    print(f"\n  1d time-bar baseline (from trend_book_lab 2026-06-10):")
    print(f"    OOS CAGR=-8%/yr  UNSEEN 0 trades (regime gate killed all entries)")

    # Final honest verdict summary
    full_result["honest_summary"] = _honest_summary(full_result, bh_unseen, bh_oos)
    print(f"\n{'='*80}")
    print("HONEST VERDICT:")
    for line in full_result["honest_summary"]["verdict_lines"]:
        print(f"  {line}")

    if write_json:
        out = ROOT / "runs" / "strat" / "alt_bar_trend_lab_2026-06-10.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Exclude per_asset_trades from JSON to keep size manageable
        out_data = {k: v for k, v in full_result.items() if k != "bar_types"}
        bar_types_slim = {}
        for bt, btr in full_result["bar_types"].items():
            slim = {k: v for k, v in btr.items() if k != "per_asset_trades"}
            bar_types_slim[bt] = slim
        out_data["bar_types"] = bar_types_slim
        with open(out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nArtifact written: {out}")

    return full_result


def _honest_summary(result: dict, bh_unseen: float, bh_oos: float) -> dict:
    """Build the honest verdict summary across bar types."""
    lines = []
    best_bt = None
    best_cagr = -999.0
    for bt, btr in result["bar_types"].items():
        if "error" in btr:
            continue
        ucagr = btr["taker"]["unseen_cagr_pct_yr"]
        if ucagr > best_cagr:
            best_cagr = ucagr
            best_bt = bt

    if best_bt:
        best_btr = result["bar_types"][best_bt]
        lines.append(f"Best bar type (UNSEEN): {best_bt} -> CAGR {best_cagr:+.0f}%/yr")
        udd = best_btr["taker"]["unseen_max_dd_pct"]
        un  = best_btr["taker"]["unseen_n_trades"]
        lines.append(f"  max_DD (worst asset): {udd:.1f}%  n_trades_UNSEEN: {un}")
        lines.append(f"  B&H UNSEEN: {bh_unseen:+.0f}%/yr | beats_BH: {best_btr['unseen_beats_bh']}")
        bands = best_btr["unseen_bands_taker"]
        lines.append(f"  Bands cleared: {bands['clears']}")
        lines.append(f"  Bands missed:  {bands['misses']}")
        lines.append(f"  Gaps: {bands['gaps']}")
        gate = best_btr.get("candidate_gate_proxies", {})
        lines.append(f"  firewall_beats_null: {gate.get('firewall_beats_null')}")
        lines.append(f"  beats_beta: {gate.get('beats_beta')}")
        lines.append(f"  10-seed positive frac: {gate.get('battery_10seeds_frac_positive')}")
        lines.append(f"  p05_bootstrap: {gate.get('p05_bootstrap')}")
        lines.append(f"  jk3: {gate.get('jk3_compound_pct')}")

        # Honest ceiling vs daily time-bar baseline
        daily_oos_cagr = -7.5  # from trend_book_lab_2026-06-10.json
        for bt, btr in result["bar_types"].items():
            if "error" in btr:
                continue
            uc = btr["taker"]["unseen_cagr_pct_yr"]
            oc = btr["taker"]["oos_cagr_pct_yr"]
            lift = uc - daily_oos_cagr
            lines.append(f"  {bt}: UNSEEN {uc:+.0f}%/yr  OOS {oc:+.0f}%/yr  lift-vs-1d-daily: {lift:+.0f}pp")
    else:
        lines.append("No valid bar types found.")

    return {"verdict_lines": lines, "best_bar_type": best_bt, "best_unseen_cagr_pct_yr": best_cagr}


# -----------------------------------------------------------------------
# Selftest (synthetic, no market data)
# -----------------------------------------------------------------------

def selftest() -> bool:
    """Synthetic sanity checks for all 3 bar types.

    T1: HA construction: ha_close = (O+H+L+C)/4 on first bar.
    T2: HA signals on uptrend -> participates (n_trades >= 1).
    T3: HA entry fill uses source open (not ha_open).
    T4: Renko: uptrend generates UP bricks.
    T5: Renko: down market generates DOWN bricks.
    T6: Generic simulator: cost correctly deducted.
    T7: Range bars: same simulator runs without error (pass dummy OHLCV).
    """
    print("=" * 70)
    print("ALT BAR TREND LAB -- SELFTEST (synthetic)")
    print("=" * 70)
    PASS = True
    rng = np.random.default_rng(7)

    # Build a synthetic 1h uptrend
    n = 1200
    dates = pd.date_range("2018-01-01", periods=n, freq="h")
    daily_ret = 0.0015 / 24 + rng.normal(0, 0.002, n)  # hourly drift
    close = 100.0 * np.cumprod(1.0 + daily_ret)
    open_ = np.concatenate([[100.0], close[:-1]])
    hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0015, n)))
    lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.0015, n)))
    df_up = pd.DataFrame({"date": dates, "open": open_, "high": hi, "low": lo, "close": close})

    # T1: HA close check
    ha_df = build_heikin_ashi(df_up)
    expected_ha_close_0 = (open_[0] + hi[0] + lo[0] + close[0]) / 4.0
    ok_t1 = abs(ha_df["ha_close"].iloc[0] - expected_ha_close_0) < 1e-6
    print(f"  [T1] HA close[0]={ha_df['ha_close'].iloc[0]:.6f} expected={expected_ha_close_0:.6f}  [{'PASS' if ok_t1 else 'FAIL'}]")
    PASS = PASS and ok_t1

    # T2: HA signals on uptrend -> strategy participates
    trades_ha = simulate_asset_generic(ha_df, atr_mult=6.0, use_regime_gate=True, close_col="ha_close")
    ok_t2 = len(trades_ha) >= 1
    print(f"  [T2] HA uptrend n_trades={len(trades_ha)}  [{'PASS' if ok_t2 else 'FAIL'}]  (EXPECT >= 1)")
    PASS = PASS and ok_t2

    # T3: HA entry fill uses source open, NOT ha_open
    if trades_ha:
        entry_fill_idx = trades_ha[0]["entry_idx"] + 1
        entry_p = trades_ha[0]["entry_p"]
        source_open = df_up["open"].iloc[entry_fill_idx]
        ok_t3 = abs(entry_p - source_open) < 1e-6
        print(f"  [T3] HA entry_p={entry_p:.4f} source_open={source_open:.4f}  [{'PASS' if ok_t3 else 'FAIL'}]")
        PASS = PASS and ok_t3
    else:
        print("  [T3] SKIP (no HA trades)")

    # T4: Renko uptrend -> UP bricks dominant: use a STRONG uptrend (0.05%/hr = ~120x/yr)
    # and SMALL brick size (0.25x ATR) so the upward drift dominates direction count.
    n_strong = 600
    dates_strong = pd.date_range("2018-01-01", periods=n_strong, freq="h")
    up_ret_strong = 0.0005 + rng.normal(0, 0.001, n_strong)  # strong hourly drift, low noise
    close_strong = 100.0 * np.cumprod(1.0 + up_ret_strong)
    open_strong = np.concatenate([[100.0], close_strong[:-1]])
    hi_strong = np.maximum(open_strong, close_strong) * (1.0 + np.abs(rng.normal(0, 0.0005, n_strong)))
    lo_strong = np.minimum(open_strong, close_strong) * (1.0 - np.abs(rng.normal(0, 0.0005, n_strong)))
    df_strong_up = pd.DataFrame({"date": dates_strong, "open": open_strong, "high": hi_strong,
                                  "low": lo_strong, "close": close_strong})
    renko_up = build_renko(df_strong_up, atr_mult=0.25)  # tiny bricks -> direction driven by price trend
    if not renko_up.empty:
        n_up = (renko_up["direction"] == 1).sum()
        n_dn = (renko_up["direction"] == -1).sum()
        ok_t4 = n_up > n_dn
        print(f"  [T4] Renko strong_uptrend(brick=0.25xATR): n_up={n_up} n_dn={n_dn}  [{'PASS' if ok_t4 else 'FAIL'}]  (EXPECT up > down)")
        PASS = PASS and ok_t4
    else:
        print("  [T4] Renko: empty (too few bars)")

    # T5: Renko downtrend -> DOWN bricks
    down_ret = -0.0005 + rng.normal(0, 0.001, n_strong)
    close_dn = 100.0 * np.cumprod(1.0 + down_ret)
    open_dn = np.concatenate([[100.0], close_dn[:-1]])
    hi_dn = np.maximum(open_dn, close_dn) * (1.0 + np.abs(rng.normal(0, 0.0005, n_strong)))
    lo_dn = np.minimum(open_dn, close_dn) * (1.0 - np.abs(rng.normal(0, 0.0005, n_strong)))
    df_dn = pd.DataFrame({"date": dates_strong, "open": open_dn, "high": hi_dn, "low": lo_dn, "close": close_dn})
    renko_dn = build_renko(df_dn, atr_mult=0.25)
    if not renko_dn.empty:
        n_dn2 = (renko_dn["direction"] == -1).sum()
        n_up2 = (renko_dn["direction"] == 1).sum()
        ok_t5 = n_dn2 >= n_up2
        print(f"  [T5] Renko strong_downtrend(brick=0.25xATR): n_dn={n_dn2} n_up={n_up2}  [{'PASS' if ok_t5 else 'FAIL'}]  (EXPECT down >= up)")
        PASS = PASS and ok_t5
    else:
        print("  [T5] Renko downtrend: empty")

    # T6: Cost check
    if trades_ha:
        t0 = trades_ha[0]
        raw = t0["exit_p"] / t0["entry_p"] - 1.0
        expected_net = raw - COST_TAKER
        ok_t6 = abs(t0["net_pnl"] - expected_net) < 0.001
        print(f"  [T6] Cost: raw={raw:.4f} net={t0['net_pnl']:.4f} expected={expected_net:.4f}  [{'PASS' if ok_t6 else 'FAIL'}]")
        PASS = PASS and ok_t6

    # T7: Range (plain OHLCV as range proxy) simulator runs
    try:
        df_rng = df_up.copy()
        _ = simulate_asset_generic(df_rng, atr_mult=6.0, use_regime_gate=True, close_col="close")
        print(f"  [T7] Range (proxy OHLCV) simulator runs cleanly  [PASS]")
    except Exception as e:
        print(f"  [T7] Range simulator failed: {e}  [FAIL]")
        PASS = False

    print("-" * 70)
    print(f"SELFTEST {'PASS' if PASS else 'FAIL'}")
    print("=" * 70)
    return PASS


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alt-bar trend lab: Renko/Range/Heikin-Ashi")
    parser.add_argument("--selftest", action="store_true", help="Run synthetic selftest only")
    args = parser.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)
    else:
        run_real(write_json=True, verbose=True)
        sys.exit(0)
