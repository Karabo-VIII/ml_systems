"""Canonical EXIT mechanism library for trading engines.

The project had no single exit module — exits were ad-hoc `exit_fn` params scattered
across engines. This consolidates the tested menu + the scoped-but-untested ones, with
the dead-list verdicts baked in as docstrings so we don't re-walk dead paths.

DEAD-LIST CONTEXT (read before using any exit):
  - D23: MFE trailing-stop exit → −18pp TRAIN+VAL, below baseline UNSEEN. Trailing stops
         FAILED on the MA entry. (NOTE: trailing worked on a DIFFERENT entry — breadth bounce.)
  - D61: "smart exit > fixed hold" → Δ−0.037, won 3/12. The "beats-null" was a hold-length
         artifact. BUT explicitly SCOPED: regime-conditional / take-profit / sub-bar exits
         remain UNTESTED.
  - prod/config.py RED TEAM: "ATR3 trailing stop destructive on 7/10 assets", "SL5 tolerable
         but always worse than baseline" → stop-losses DISABLED in prod.

CONCLUSION baked in: fixed time-hold + signal-exit is the project's robust default in crypto.
Stops/trailing are OFF by default. The library EXPOSES them for explicit testing but flags risk.

Every exit takes the standard signature and returns a boolean array `exit_now[t]` aligned to
the close array — True means "close the position at bar t". The engine combines entry +
these exits. Exits are CAUSAL (only use data ≤ t).

Exit types:
  TIME_CAP        — close after max_hold_bars (the current default backstop). ROBUST.
  SIGNAL_BREAK    — close when the entry signal/confirmation is no longer true. ROBUST.
  TAKE_PROFIT     — close when unrealized gain ≥ tp_pct. SCOPED-UNTESTED (D61).
  VOL_TARGET_EXIT — close when realized vol spikes beyond vol_mult×entry_vol. SCOPED.
  REGIME_EXIT     — close when the regime gate flips to bear mid-hold. ROBUST (gate logic).
  TRAILING_STOP   — close when price falls trail_pct from peak-since-entry. RISKY (D23).
  STOP_LOSS       — close when price falls sl_pct from entry. RISKY (RED TEAM).
  MIN_HOLD / COOLDOWN are ENTRY-side modifiers (see apply_min_hold_cooldown), not exits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# ENTRY-SIDE MODIFIERS (whipsaw control for high-frequency TFs)
# =============================================================================

def apply_min_hold_cooldown(raw_signal: np.ndarray, min_hold_bars: int = 0,
                            cooldown_bars: int = 0) -> np.ndarray:
    """Convert a raw per-bar long/flat signal into a held state with:
      - MIN_HOLD: once entered, stay in for >= min_hold_bars even if signal flickers off
        (kills the in-out-in churn that destroys profit factor at 15m-2h)
      - COOLDOWN: after an exit, block re-entry for cooldown_bars (prevents re-entering
        the same chop immediately)

    raw_signal: int/bool array, 1 = signal says long, 0 = flat
    Returns: int8 held array.

    This is the whipsaw fix. Causal (only looks back). Does NOT add look-ahead.
    """
    sig = np.asarray(raw_signal, dtype=np.int8)
    n = len(sig)
    held = np.zeros(n, dtype=np.int8)
    state = 0
    bars_in = 0
    bars_since_exit = 10**9
    for i in range(n):
        if state == 1:
            bars_in += 1
            # can only exit if past min-hold AND signal dropped
            if bars_in >= min_hold_bars and sig[i] == 0:
                state = 0
                bars_since_exit = 0
                held[i] = 0
            else:
                held[i] = 1
        else:
            bars_since_exit += 1
            # can only enter if past cooldown AND signal on
            if sig[i] == 1 and bars_since_exit >= cooldown_bars:
                state = 1
                bars_in = 1
                held[i] = 1
            else:
                held[i] = 0
    return held


# =============================================================================
# EXIT MECHANISMS (return exit_now[t] boolean, given an active position)
# =============================================================================

def exit_time_cap(n: int, entry_idx: int, max_hold_bars: int) -> int:
    """TIME_CAP (ROBUST default). Returns the forced-exit bar index."""
    return min(entry_idx + max_hold_bars, n - 1)


def exit_take_profit(close: np.ndarray, entry_idx: int, entry_price: float,
                     tp_pct: float) -> int | None:
    """TAKE_PROFIT (SCOPED-UNTESTED, D61). Exit at first bar where gain >= tp_pct.
    Returns exit bar index or None if never hit within the array."""
    for i in range(entry_idx + 1, len(close)):
        if np.isfinite(close[i]) and entry_price > 0:
            if (close[i] / entry_price - 1.0) >= tp_pct:
                return i
    return None


def exit_trailing_stop(close: np.ndarray, entry_idx: int, trail_pct: float) -> int | None:
    """TRAILING_STOP (RISKY, D23 — failed on MA entry). Exit when price falls trail_pct
    from the running peak since entry. Returns exit bar or None."""
    peak = close[entry_idx]
    for i in range(entry_idx + 1, len(close)):
        if not np.isfinite(close[i]):
            continue
        peak = max(peak, close[i])
        if peak > 0 and (close[i] / peak - 1.0) <= -trail_pct:
            return i
    return None


def exit_stop_loss(close: np.ndarray, entry_idx: int, entry_price: float,
                   sl_pct: float) -> int | None:
    """STOP_LOSS (RISKY, RED TEAM — destructive on 7/10 assets). Exit when price falls
    sl_pct from entry. Returns exit bar or None."""
    for i in range(entry_idx + 1, len(close)):
        if np.isfinite(close[i]) and entry_price > 0:
            if (close[i] / entry_price - 1.0) <= -sl_pct:
                return i
    return None


def realized_vol(close: np.ndarray, lookback: int) -> np.ndarray:
    """Causal rolling realized vol (std of bar returns)."""
    rets = np.zeros(len(close))
    rets[1:] = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0)
    return pd.Series(rets).rolling(lookback, min_periods=max(2, lookback // 2)).std().to_numpy()


# =============================================================================
# UNIFIED EXIT RESOLVER (combines confirmation-break + time-cap + optional extras)
# =============================================================================

def resolve_held(close: np.ndarray, confirmed: np.ndarray, max_hold_bars: int,
                 *, min_hold_bars: int = 0, cooldown_bars: int = 0,
                 take_profit_pct: float | None = None,
                 trailing_stop_pct: float | None = None,
                 stop_loss_pct: float | None = None,
                 regime_mask: np.ndarray | None = None) -> np.ndarray:
    """THE canonical held-state resolver. Given a per-bar `confirmed` entry signal,
    produce the held array using:
      - entry: confirmed AND past cooldown AND (regime bull if regime_mask given)
      - exit: EARLIEST of {confirmation breaks (after min_hold), time-cap, take-profit,
              trailing-stop, stop-loss, regime flip}

    Default behavior (no optional args) == the project's robust default:
    confirmation-break + time-cap. ROBUST per D61/D23/RED-TEAM.

    All causal. Returns int8 held array aligned to close.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    conf = np.asarray(confirmed, dtype=bool)
    held = np.zeros(n, dtype=np.int8)

    has_price_exit = (take_profit_pct is not None or trailing_stop_pct is not None
                      or stop_loss_pct is not None)

    i = 0
    bars_since_exit = 10**9
    while i < n:
        # ENTRY conditions
        can_enter = conf[i] and bars_since_exit >= cooldown_bars
        if regime_mask is not None and i < len(regime_mask):
            can_enter = can_enter and bool(regime_mask[i])
        if not can_enter:
            bars_since_exit += 1
            i += 1
            continue

        # OPEN a position at i
        entry_idx = i
        entry_price = c[i]
        cap_idx = min(entry_idx + max_hold_bars, n - 1)

        # confirmation-break / regime-flip exit (after min_hold) — find earliest
        exit_idx = cap_idx
        for t in range(entry_idx + 1, cap_idx + 1):
            if (t - entry_idx) < min_hold_bars:
                continue
            if not conf[t] or (regime_mask is not None and t < len(regime_mask) and not regime_mask[t]):
                exit_idx = t
                break

        # price-based exits (only if requested) — may pull the exit earlier
        if has_price_exit:
            if take_profit_pct is not None:
                tp = exit_take_profit(c[:exit_idx + 1], entry_idx, entry_price, take_profit_pct)
                if tp is not None and tp < exit_idx:
                    exit_idx = tp
            if trailing_stop_pct is not None:
                ts = exit_trailing_stop(c[:exit_idx + 1], entry_idx, trailing_stop_pct)
                if ts is not None and ts < exit_idx:
                    exit_idx = ts
            if stop_loss_pct is not None:
                sl = exit_stop_loss(c[:exit_idx + 1], entry_idx, entry_price, stop_loss_pct)
                if sl is not None and sl < exit_idx:
                    exit_idx = sl

        held[entry_idx:exit_idx] = 1   # held while in position (exit bar is flat)
        bars_since_exit = 0
        i = exit_idx if exit_idx > entry_idx else entry_idx + 1
    return held


# Cost presets (round-trip)
COST_MAKER = 0.0006      # maker limit-order RT (optimistic)
COST_TAKER = 0.0024      # taker market-order RT
COST_PROD = 0.0028       # taker + 2x slippage (prod/config.py: 0.001/side fee + 0.0002 slip)
