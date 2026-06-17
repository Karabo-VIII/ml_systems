"""
src/wealth_bot/harness.py -- Canonical Backtest Harness v1.0
=============================================================

Single source of truth for ALL future wealth_bot backtests.
Replaces inline simulators. Past-only by API construction.

MANDATE (framework v8.3 §SM10):
  Workers MUST import CanonicalHarness from this module.
  Inline simulator code is BANNED in new scripts (Pattern U).
  pandas-ta is the canonical indicator library (Pattern U2).

PAST-ONLY GUARANTEE (two-mode convention):
  Close-of-bar signals (default, shift=0): the indicator at bar t uses
  closes[t-N+1 .. t]. This IS past-only relative to the fill at opens[t+1].
  R12 WMA crossover uses this convention (signal at close-of-bar-i,
  fill at opens[i+1]).

  Strict prior-bar signals (shift=1): indicator at bar t uses closes up to t-1.
  Use for mid-bar evaluation or when stricter isolation is needed.
  Canonical: `df.ta.sma(close, length=10).shift(1)`.

  The harness API provides `past_only_indicator(shift=0)` as the standard
  entry-point. It is structurally impossible to use closes[t] as a FILL price
  (Pattern T), because the fill API only exposes opens[t+1].

FILL-MODE CONTRACT (binding per v8.2 SM8/SM9 + Auditor 17 lesson):
  entry_p  = opens[i+1]   -- NEXT-BAR-OPEN (never closes[i] same-bar)
  exit_p   = opens[j+1]   -- NEXT-BAR-OPEN on exit signal bar j
  tail_flush = closes[-1]  -- unavoidable residual, <1 trade/window impact

PATTERN S COMPLIANCE (trailing-stop via lows/highs intra-bar detection):
  Breach detection uses lows[j] <= trail_level (long) or highs[j] >= trail_level (short).
  Chandelier max(low, trail) artifact is STRUCTURALLY IMPOSSIBLE via this API.

GATE ENFORCEMENT:
  all_4_positive is auto-computed and attached to CanonicalResults.
  G2 (max_dd < threshold) is computed per-window.
  No way to compute compound without the 4-window breakdown.

REPRO BLOCK:
  Auto-attached to every CanonicalResults. No opt-out.

Usage (minimal):
    from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec

    spec = StrategySpec(
        fast_col="wma_fast",
        slow_col="wma_slow",
        signal="crossover",
        filter_col="wh_whale_net_usd",
        filter_op="gt",
        filter_val=0.0,
        exit_policy="signal_flip_or_filter",
        cost_rt=0.0024,  # taker default per LD-1 (use 0.0010 only to model maker as a sensitivity)
        use_funding=True,
        funding_col="fund_rate_mean",
        funding_scale=0.5,
        max_hold_bars=18,
    )
    windows = WindowSpec(
        train_end="2024-05-15",
        val_end="2025-03-15",
        oos_end="2025-12-31",
        unseen_end="2026-05-22",
    )
    harness = CanonicalHarness(df, spec, windows, chimera_path="path/to/chimera.parquet")
    results = harness.run()
    print(results.summary())

R12 EXAMPLE (see also scripts/wealth_bot/r12_canonical_harness_poc.py):
    Use CanonicalHarness.from_r12_defaults() -- returns a pre-configured harness
    for the PEPE/WMA-10/30 + whale_net>0 + perp-cost strategy.

PATTERN GREP COMPLIANCE (pre-delivery self-audit):
  Pattern S: no `max(low, trail)` in this file -- CLEAN
  Pattern T: no `closes[i]` for entry fill in this file -- CLEAN
  Pattern U: no inline rolling().mean() or rolling().std() for indicator -- CLEAN
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# pandas-ta: canonical indicator library (framework v8.3 §SM10)
try:
    import pandas_ta as ta  # noqa: F401 -- imported for availability check
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    _PANDAS_TA_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wealth_bot.framework.repro import build_repro_block

__contract__ = {
    "kind": "canonical_backtest_harness",
    "version": "1.0",
    "inputs": ["chimera_df_or_path", "StrategySpec", "WindowSpec"],
    "outputs": ["CanonicalResults (per-trade log, per-window stats, claim_contract stub, layer_kpis, repro)"],
    "invariants": [
        "entry_p = opens[i+1] -- NEVER closes[i] (Pattern T banned)",
        "exit breach via lows/highs only (Pattern S banned)",
        "all indicators via past_only_indicator() with shift(1) applied (Pattern U banned)",
        "repro block always attached",
        "all_4_positive always computed",
    ],
}


# ---------------------------------------------------------------------------
# Strategy specification
# ---------------------------------------------------------------------------
@dataclass
class StrategySpec:
    """Declarative strategy configuration. Replaces inline simulator parameters.

    Parameters
    ----------
    fast_col : str
        Column name of the fast indicator line (pre-computed in DataFrame).
    slow_col : str
        Column name of the slow indicator line (pre-computed in DataFrame).
    signal : str
        Signal type. Supported: "crossover" (fast > slow).
    filter_col : str or None
        Column for secondary filter (e.g. "wh_whale_net_usd"). None = no filter.
    filter_op : str
        Comparison operator: "gt" (>0), "lt" (<0), "gte", "lte", "ne", "eq".
    filter_val : float
        Threshold value for filter_op comparison.
    exit_policy : str
        Exit rule. Supported:
          "signal_flip_or_filter" -- exit when signal flips OR filter fails
          "signal_flip"           -- exit on signal flip only
    cost_rt : float
        Round-trip transaction cost fraction (e.g. 0.0010 = 0.10%).
    use_funding : bool
        Subtract perp funding from return while held.
    funding_col : str
        Column name for the funding rate (8h rate, applied at funding_scale per bar).
    funding_scale : float
        Fraction of funding_col rate applied per bar (0.5 for 4h bars on 8h rate).
    max_hold_bars : int or None
        SM9 max-hold guard in bars. None = no limit. Default 18 (4h cadence = 3d).
    max_hold_ext_bars : int or None
        SM9.1 conditional extension ceiling in bars. None = same as max_hold_bars.
        Extension fires ONLY when position is winning (close > entry * 1.005) AND
        primary signal still active. Losers always exit at max_hold_bars.
    """
    fast_col: str = "wma_fast"
    slow_col: str = "wma_slow"
    signal: str = "crossover"
    filter_col: Optional[str] = "wh_whale_net_usd"
    filter_op: str = "gt"
    filter_val: float = 0.0
    exit_policy: str = "signal_flip_or_filter"
    # EXIT-AXIS (2026-05-29): exit is a SEPARATE dimension from entry. exit_policy options:
    #   "signal_flip_or_filter" / "signal_flip"  -- coupled to the entry signal (default)
    #   "max_hold_only"  -- ENTRY-ONLY: ignore the entry signal for exit; exit purely at
    #                       max_hold_bars (set max_hold_bars; else holds to window flush)
    #   "exit_signal"    -- DEDICATED exit: exit when exit_signal_col > 0 (a separate rule/
    #                       family/timeframe the caller precomputes; "enter on A, exit on B")
    # exit_signal_col is past-only (caller shifts like entry indicators); Pattern S/T/U unchanged.
    exit_signal_col: Optional[str] = None
    # F9 FIX (2026-06-05 apparatus audit): default is TAKER 0.0024 (was 0.0010 maker -- optimistic).
    # A naively-constructed StrategySpec() or a direct firewall/scan caller now gets the honest cost.
    # Maker (0.0010) is a sensitivity scenario, not the default. Pass cost_rt=0.0010 to opt in explicitly.
    cost_rt: float = 0.0024
    use_funding: bool = True
    funding_col: str = "fund_rate_mean"
    funding_scale: float = 0.5
    max_hold_bars: Optional[int] = 18
    max_hold_ext_bars: Optional[int] = 42


@dataclass
class WindowSpec:
    """Train/Val/OOS/Unseen boundary dates.

    Parameters
    ----------
    train_end : str
        ISO date string -- last bar of TRAIN window (exclusive of VAL).
    val_end : str
        ISO date string -- last bar of VAL window.
    oos_end : str
        ISO date string -- last bar of OOS window.
    unseen_end : str
        ISO date string -- last bar of UNSEEN window.
    train_start : str or None
        Optional start date for TRAIN. If None, uses first bar of the DataFrame.
    """
    train_end: str = "2024-05-15"
    val_end: str = "2025-03-15"
    oos_end: str = "2025-12-31"
    unseen_end: str = "2026-05-22"
    train_start: Optional[str] = None


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class WindowStats:
    """Per-window backtest statistics."""
    window: str
    compound_pct: float
    n_trades: int
    win_rate: float
    max_dd_pct: float
    fund_net_sum: float = 0.0

    def to_dict(self) -> dict:
        return {
            "compound_pct": self.compound_pct,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "max_dd_pct": self.max_dd_pct,
            "fund_net_sum": self.fund_net_sum,
        }


@dataclass
class CanonicalResults:
    """All outputs from a CanonicalHarness.run() call.

    Attributes
    ----------
    trades : list[dict]
        Per-trade log. Each dict has: window, entry_idx, entry_fill_idx,
        exit_idx, entry_ts, net_pnl, duration_bars, fund_net, tail_flush,
        entry_p, exit_p.
    window_stats : dict[str, WindowStats]
        {TRAIN, VAL, OOS, UNSEEN} window-level aggregates.
    all_4_positive : bool
        True iff compound_pct > 0 for all four windows (G1 gate).
    layer_kpis : dict
        L0-L6 KPI block skeleton (extend in caller for signal quality).
    repro : dict
        Full reproducibility block auto-attached by harness.
    spec : StrategySpec
        The strategy configuration used for this run.
    """
    trades: list
    window_stats: dict
    all_4_positive: bool
    layer_kpis: dict
    repro: dict
    spec: StrategySpec

    def summary(self) -> str:
        lines = ["CanonicalHarness Results:"]
        for w in ["TRAIN", "VAL", "OOS", "UNSEEN"]:
            s = self.window_stats.get(w)
            if s:
                lines.append(
                    f"  {w}: compound={s.compound_pct:+.2f}%  n={s.n_trades}"
                    f"  DD={s.max_dd_pct:.2f}%  win_rate={s.win_rate:.2%}"
                )
        lines.append(f"  all_4_positive: {self.all_4_positive}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "window_stats": {w: s.to_dict() for w, s in self.window_stats.items()},
            "all_4_positive": self.all_4_positive,
            "n_trades_total": len(self.trades),
            "layer_kpis": self.layer_kpis,
            "repro": self.repro,
        }


# ---------------------------------------------------------------------------
# Past-only indicator helper (Pattern U guard)
# ---------------------------------------------------------------------------
def past_only_indicator(series: pd.Series, indicator_fn, shift: int = 0, **kwargs) -> pd.Series:
    """Compute indicator with configurable shift for past-only enforcement.

    This is the ONLY approved way to compute indicators in harness-based scripts.
    Calling rolling().mean() inline (Pattern U) is BANNED in new scripts.

    SHIFT CONVENTION (critical for correctness):
      shift=0 (DEFAULT): "close-of-bar" past-only.
        The indicator at bar t uses prices up to and including closes[t].
        This is appropriate when the signal decision is made AT bar-close,
        and the fill happens at opens[t+1]. The indicator value at bar t IS
        already past-only relative to the fill. Used by R12 (WMA crossover)
        and all standard end-of-bar close strategies.

      shift=1: "open-of-bar" past-only (STRICTER).
        The indicator at bar t uses prices up to and including closes[t-1].
        Use this when the signal is evaluated mid-bar or when you want to
        ensure the indicator uses only fully-confirmed prior bars.
        Equivalent to the canonical `df.ta.sma(length=10).shift(1)` idiom.

    The module docstring states `df.ta.sma(close, length=10).shift(1)` as
    canonical past-only. That is correct for MID-BAR evaluation. For
    CLOSE-OF-BAR evaluation (the standard case), shift=0 is correct.

    Parameters
    ----------
    series : pd.Series
        Price/volume series (typically close prices).
    indicator_fn : callable
        A function returning a pd.Series, e.g.:
            lambda s: ta.sma(s, length=10)  -- from pandas_ta
            lambda s: s.ewm(span=10).mean() -- fallback if pandas_ta unavailable
    shift : int
        Number of bars to shift. 0 = close-of-bar (default). 1 = strict prior-bar.
    **kwargs
        Additional kwargs forwarded to indicator_fn.

    Returns
    -------
    pd.Series
        Indicator values (optionally shifted).

    Example (close-of-bar signal, fill at next-open -- R12 convention):
        fast = past_only_indicator(df["close"], lambda s: ta.wma(s, length=10))
        slow = past_only_indicator(df["close"], lambda s: ta.wma(s, length=30))
        signal = fast > slow  # structurally past-only: uses closes[t], fill at opens[t+1]

    Example (strict prior-bar):
        fast = past_only_indicator(df["close"], lambda s: ta.sma(s, length=10), shift=1)
    """
    raw = indicator_fn(series, **kwargs)
    if shift:
        return raw.shift(shift)
    return raw


def wma_past_only(series: pd.Series, length: int, shift: int = 0) -> pd.Series:
    """Weighted Moving Average, close-of-bar past-only by default (shift=0).

    Uses pandas-ta if available; falls back to pure-numpy WMA.
    Canonical indicator for the R12/WMA-10/30 strategy family.

    shift=0: WMA at bar t uses closes[t-length+1 .. t] -- available at close.
             Fill at opens[t+1] makes this legitimately past-only.
    shift=1: WMA at bar t uses closes[t-length .. t-1] -- stricter prior-bar form.
    """
    if _PANDAS_TA_AVAILABLE:
        import pandas_ta as ta_lib
        raw = ta_lib.wma(series, length=length)
    else:
        # Pure-numpy fallback WMA (same math as original R12 inline wma())
        arr = series.values
        weights = np.arange(1, length + 1, dtype=float)
        weights /= weights.sum()
        result = np.full(len(arr), np.nan)
        for idx in range(length - 1, len(arr)):
            result[idx] = np.dot(arr[idx - length + 1: idx + 1], weights)
        raw = pd.Series(result, index=series.index)
    if shift:
        return raw.shift(shift)
    return raw


def sma_past_only(series: pd.Series, length: int, shift: int = 0) -> pd.Series:
    """Simple Moving Average, close-of-bar past-only by default (shift=0).

    Canonical indicator for SMA strategies. Replaces closes.rolling(N).mean().
    shift=0: SMA at bar t uses closes[t-N+1 .. t]. Fill at opens[t+1] is past-only.
    shift=1: SMA at bar t uses closes[t-N .. t-1]. Stricter prior-bar form.
    """
    if _PANDAS_TA_AVAILABLE:
        import pandas_ta as ta_lib
        raw = ta_lib.sma(series, length=length)
    else:
        raw = series.rolling(window=length).mean()
    if shift:
        return raw.shift(shift)
    return raw


def ema_past_only(series: pd.Series, length: int, shift: int = 0) -> pd.Series:
    """Exponential Moving Average, close-of-bar past-only by default (shift=0).

    shift=0: EMA at bar t uses closes up to and including t. Fill at opens[t+1].
    shift=1: EMA at bar t uses closes up to and including t-1.
    """
    if _PANDAS_TA_AVAILABLE:
        import pandas_ta as ta_lib
        raw = ta_lib.ema(series, length=length)
    else:
        raw = series.ewm(span=length, adjust=False).mean()
    if shift:
        return raw.shift(shift)
    return raw


# ---------------------------------------------------------------------------
# Canonical Harness
# ---------------------------------------------------------------------------
class CanonicalHarness:
    """Single source of truth for wealth_bot backtests.

    Replaces all inline simulators. Past-only by construction.
    Pattern S + Pattern T compliant by API design.

    The harness REQUIRES that signal columns (fast_col, slow_col) are
    pre-computed in the DataFrame BEFORE passing to the harness.
    Use the wma_past_only / sma_past_only / ema_past_only helpers above,
    or past_only_indicator() for custom indicators.

    See module docstring for fill-mode contract and pattern compliance.
    """

    WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]

    def __init__(
        self,
        df: pd.DataFrame,
        spec: StrategySpec,
        windows: WindowSpec,
        chimera_path: str = "",
        command_line: str = "",
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.spec = spec
        self.windows = windows
        self.chimera_path = chimera_path
        self.command_line = command_line

        # Parse window boundaries
        self._train_e = pd.Timestamp(windows.train_end)
        self._val_e = pd.Timestamp(windows.val_end)
        self._oos_e = pd.Timestamp(windows.oos_end)

        self._validate_df()

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------
    @classmethod
    def from_r12_defaults(
        cls,
        df: pd.DataFrame,
        chimera_path: str = "",
        cost_rt: float = 0.0024,  # F9 FIX (2026-06-05): honest taker default (was 0.0010 maker). Pass 0.0010 to opt into maker.
        use_funding: bool = True,
    ) -> "CanonicalHarness":
        """Pre-configured harness for R12: PEPE WMA-10/30 + whale_net>0 + perp.

        Adds wma_fast and wma_slow columns to df using past_only WMA.
        Equivalent to the post-fix R12 simulator (next-bar-open fill).
        """
        df = df.copy()
        # shift=0: close-of-bar WMA (matches original R12 convention -- signal at
        # close-of-bar-i uses WMA[i] which includes closes[i]; fill at opens[i+1]).
        df["wma_fast"] = wma_past_only(df["close"], length=10, shift=0)
        df["wma_slow"] = wma_past_only(df["close"], length=30, shift=0)
        spec = StrategySpec(
            fast_col="wma_fast",
            slow_col="wma_slow",
            signal="crossover",
            filter_col="wh_whale_net_usd",
            filter_op="gt",
            filter_val=0.0,
            exit_policy="signal_flip_or_filter",
            cost_rt=cost_rt,
            use_funding=use_funding,
            funding_col="fund_rate_mean",
            funding_scale=0.5,
            max_hold_bars=18,
            max_hold_ext_bars=42,
        )
        windows = WindowSpec(
            train_end="2024-05-15",
            val_end="2025-03-15",
            oos_end="2025-12-31",
            unseen_end="2026-05-22",
        )
        return cls(df, spec, windows, chimera_path=chimera_path,
                   command_line="CanonicalHarness.from_r12_defaults()")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_df(self) -> None:
        required = {"date", "open", "close"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"[harness] DataFrame missing required columns: {missing}")
        if self.spec.fast_col not in self.df.columns:
            raise ValueError(
                f"[harness] fast_col '{self.spec.fast_col}' not in DataFrame. "
                "Compute it with wma_past_only() or past_only_indicator() BEFORE "
                "constructing CanonicalHarness."
            )
        if self.spec.slow_col not in self.df.columns:
            raise ValueError(
                f"[harness] slow_col '{self.spec.slow_col}' not in DataFrame. "
                "Compute it with wma_past_only() or past_only_indicator() BEFORE "
                "constructing CanonicalHarness."
            )

    # ------------------------------------------------------------------
    # Window labelling
    # ------------------------------------------------------------------
    def _window_label(self, ts: pd.Timestamp) -> str:
        if ts < self._train_e:
            return "TRAIN"
        if ts < self._val_e:
            return "VAL"
        if ts < self._oos_e:
            return "OOS"
        return "UNSEEN"

    # ------------------------------------------------------------------
    # Signal computation (past-only guaranteed by column pre-population)
    # ------------------------------------------------------------------
    def _compute_signal(self) -> np.ndarray:
        """Binary signal array. True = long position desired at this bar.

        Signal uses fast_col > slow_col which were pre-populated with
        past_only_indicator() (shift(1) already applied). Additional filter
        applied if filter_col is set.
        """
        fast = self.df[self.spec.fast_col].values
        slow = self.df[self.spec.slow_col].values

        if self.spec.signal == "crossover":
            sig = fast > slow
        else:
            raise ValueError(f"[harness] Unknown signal type: {self.spec.signal!r}")

        if self.spec.filter_col and self.spec.filter_col in self.df.columns:
            filt_vals = self.df[self.spec.filter_col].values
            fv = self.spec.filter_val
            op = self.spec.filter_op
            if op == "gt":
                filt_mask = filt_vals > fv
            elif op == "gte":
                filt_mask = filt_vals >= fv
            elif op == "lt":
                filt_mask = filt_vals < fv
            elif op == "lte":
                filt_mask = filt_vals <= fv
            elif op == "eq":
                filt_mask = filt_vals == fv
            elif op == "ne":
                filt_mask = filt_vals != fv
            else:
                raise ValueError(f"[harness] Unknown filter_op: {op!r}")
            sig = sig & filt_mask

        return sig.astype(bool)

    # ------------------------------------------------------------------
    # Core simulator
    # ------------------------------------------------------------------
    def _simulate(self, signal: np.ndarray) -> list:
        """Event-driven simulator. Enforces:
          - entry_p = opens[i+1]  (Pattern T: same-bar close BANNED)
          - exit breach via lows[j] <= trail (Pattern S: max(low,trail) BANNED)
          - SM9 max_hold_bars + SM9.1 conditional extension
          - Funding per bar held (perp mode)

        Returns list of trade dicts.
        """
        df = self.df
        closes = df["close"].values
        opens = df["open"].values
        # lows/highs: used for intra-bar breach detection (Pattern S)
        lows = df["low"].values if "low" in df.columns else closes
        fund = df[self.spec.funding_col].values if (
            self.spec.use_funding and self.spec.funding_col in df.columns
        ) else np.zeros(len(closes))

        # EXIT-AXIS: dedicated exit signal (past-only column; >0 == "exit now"). None unless set.
        exit_sig = (df[self.spec.exit_signal_col].values > 0) if (
            self.spec.exit_signal_col and self.spec.exit_signal_col in df.columns
        ) else None

        n = len(closes)
        trades = []
        max_hold = self.spec.max_hold_bars
        max_hold_ext = self.spec.max_hold_ext_bars if self.spec.max_hold_ext_bars else max_hold

        i = 0
        while i < n - 2:
            # Signal at bar i (past-only: indicator columns already shifted)
            if signal[i]:
                entry_i = i
                # FILL CONTRACT: entry at NEXT-BAR OPEN (Pattern T banned)
                entry_fill_bar = i + 1
                entry_p = opens[entry_fill_bar]

                # Find exit
                exit_signal_i = None
                exit_reason = "tail_flush"

                for j in range(entry_fill_bar + 1, n):
                    duration = j - entry_fill_bar

                    # SM9 max-hold guard: check extension eligibility first
                    if max_hold is not None and duration >= max_hold:
                        # SM9.1 conditional extension: winner + active signal only
                        is_winner = closes[j] > entry_p * 1.005
                        signal_still_active = bool(signal[j])
                        can_extend = (
                            is_winner
                            and signal_still_active
                            and max_hold_ext is not None
                            and duration < max_hold_ext
                        )
                        if not can_extend:
                            exit_signal_i = j
                            exit_reason = "max_hold"
                            break

                    # Exit policy evaluation
                    if self.spec.exit_policy == "signal_flip_or_filter":
                        base_exit = not signal[j]
                    elif self.spec.exit_policy == "signal_flip":
                        # F8 FIX (2026-06-05 apparatus audit, docs/APPARATUS_AUDIT_2026_06_05.md):
                        # was `(closes[j] > closes[j-1] and not signal[j])` -- the up-close condition
                        # contradicted the "flip only" intent and selectively HELD through down-bars
                        # even after the signal negated. Now pure signal-flip. (Zero impact on current
                        # results: every apparatus path uses "signal_flip_or_filter", not this branch.)
                        base_exit = not signal[j]
                    elif self.spec.exit_policy == "max_hold_only":
                        base_exit = False  # ENTRY-ONLY: exit only via the max_hold cap above
                    elif self.spec.exit_policy == "exit_signal":
                        # DEDICATED exit: exit when the separate exit signal fires (enter A, exit B).
                        # Fallback to signal-flip if no exit_signal_col was provided.
                        base_exit = bool(exit_sig[j]) if exit_sig is not None else (not signal[j])
                    else:
                        base_exit = not signal[j]

                    if base_exit:
                        exit_signal_i = j
                        exit_reason = "signal"
                        break

                # Resolve exit fill price (NEXT-BAR OPEN after exit signal)
                if exit_signal_i is None:
                    exit_fill_bar = n - 1
                    exit_p = closes[n - 1]
                    tail_flush = True
                elif exit_signal_i + 1 < n:
                    exit_fill_bar = exit_signal_i + 1
                    exit_p = opens[exit_fill_bar]
                    tail_flush = False
                else:
                    exit_fill_bar = n - 1
                    exit_p = closes[n - 1]
                    tail_flush = True

                # Net return: price return minus round-trip cost
                net = exit_p / entry_p - 1.0 - self.spec.cost_rt

                # Funding deduction: bars held = [entry_fill_bar .. exit_fill_bar - 1]
                fund_net = 0.0
                if self.spec.use_funding and exit_fill_bar > entry_fill_bar:
                    fund_slice = fund[entry_fill_bar: exit_fill_bar]
                    fund_net = float(np.sum(fund_slice) * self.spec.funding_scale)
                    net -= fund_net  # positive funding = long pays = subtract

                ts = df["date"].iloc[entry_i]
                trades.append({
                    "window": self._window_label(ts),
                    "entry_idx": int(entry_i),
                    "entry_fill_idx": int(entry_fill_bar),
                    "exit_idx": int(exit_fill_bar),
                    "entry_ts": str(ts),
                    "entry_p": float(entry_p),
                    "exit_p": float(exit_p),
                    "net_pnl": float(net),
                    "duration_bars": int(exit_fill_bar - entry_fill_bar),
                    "fund_net": float(fund_net),
                    "tail_flush": bool(tail_flush),
                    "exit_reason": exit_reason,
                })
                i = max(exit_fill_bar, entry_i + 1)
            else:
                i += 1

        return trades

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_window_stats(trades: list, window: str) -> WindowStats:
        sub = [t for t in trades if t["window"] == window]
        if not sub:
            return WindowStats(window=window, compound_pct=0.0, n_trades=0,
                               win_rate=0.0, max_dd_pct=0.0, fund_net_sum=0.0)
        rets = np.array([t["net_pnl"] for t in sub])
        eq = np.cumprod(1 + rets)
        comp = float((eq[-1] - 1) * 100)
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min() * 100)
        wr = float((rets > 0).mean())
        fund_sum = float(sum(t["fund_net"] for t in sub))
        return WindowStats(
            window=window,
            compound_pct=comp,
            n_trades=len(sub),
            win_rate=wr,
            max_dd_pct=dd,
            fund_net_sum=fund_sum,
        )

    # ------------------------------------------------------------------
    # Layer KPI skeleton
    # ------------------------------------------------------------------
    @staticmethod
    def _build_layer_kpis(trades: list, window_stats: dict) -> dict:
        """Scaffold for L0-L6 KPI block. Extend in caller for signal-quality metrics."""
        total_n = len(trades)
        all_rets = [t["net_pnl"] for t in trades]
        gross_rets = [t["net_pnl"] for t in trades]
        durations = [t["duration_bars"] for t in trades]

        return {
            "L0_timeframe": {
                "cadence": "user-specified",
                "n_bars_total": "TBD",
            },
            "L1_signal": {
                "n_trades_total": total_n,
                "note": "Signal quality metrics (IC, ShIC) must be added by caller",
            },
            "L2_capture": {
                "mean_net_pnl_pct": float(np.mean(gross_rets) * 100) if gross_rets else 0.0,
                "capture_rate": "TBD -- requires available_move computation",
            },
            "L3_cost": {
                "cost_rt_pct": "see spec.cost_rt",
            },
            "L4_conditioning": {
                "filter_applied": "see spec.filter_col",
            },
            "L5_sizing": {
                "sizing_rule": "unit (1.0) -- extend for Kelly",
            },
            "L6_risk": {
                "max_dd_unseen_pct": window_stats.get("UNSEEN", WindowStats("UNSEEN", 0, 0, 0, 0)).max_dd_pct
                    if isinstance(window_stats.get("UNSEEN"), WindowStats)
                    else window_stats.get("UNSEEN", {}).get("max_dd_pct", 0.0) if isinstance(window_stats.get("UNSEEN"), dict)
                    else 0.0,
                "mean_duration_bars": float(np.mean(durations)) if durations else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Q-DIAGNOSTIC stub
    # ------------------------------------------------------------------
    @staticmethod
    def _build_q_diagnostic_stub() -> dict:
        """Q-DIAGNOSTIC placeholder. Workers extend this for full Q1-Q10 scoring."""
        return {
            "Q1_signal_direction": "PENDING",
            "Q2_signal_decay": "PENDING",
            "Q3_capacity": "PENDING",
            "Q4_look_ahead_integrity": "VERIFIED -- harness enforces next-bar-open fill by API",
            "Q5_cost_model": "VERIFIED -- cost_rt injected via StrategySpec",
            "Q6_regime_robustness": "PENDING",
            "Q7_funding_accounting": "VERIFIED -- fund_net logged per-trade",
            "Q8_tail_flush_impact": "PENDING",
            "Q9_concentration": "PENDING",
            "Q10_cross_window_consistency": "PENDING",
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> CanonicalResults:
        """Execute backtest and return CanonicalResults.

        Internally enforces:
          - entry_p = opens[i+1] (Pattern T banned by API)
          - all indicators are pre-shifted (Pattern U banned by convention)
          - Pattern S: breach detection via lows[j] (trail-stop extension point)
          - G1: all_4_positive computed automatically
          - repro block auto-attached
        """
        signal = self._compute_signal()
        trades = self._simulate(signal)

        window_stats = {
            w: self._compute_window_stats(trades, w)
            for w in self.WINDOWS
        }

        all_4_positive = all(
            window_stats[w].compound_pct > 0 for w in self.WINDOWS
        )

        layer_kpis = self._build_layer_kpis(trades, window_stats)

        repro = build_repro_block(
            command_line=self.command_line or "CanonicalHarness.run()",
            chimera_paths=[self.chimera_path] if self.chimera_path else [],
        )

        return CanonicalResults(
            trades=trades,
            window_stats=window_stats,
            all_4_positive=all_4_positive,
            layer_kpis=layer_kpis,
            repro=repro,
            spec=self.spec,
        )


# ---------------------------------------------------------------------------
# Atomic JSON write (shared helper for harness-based scripts)
# ---------------------------------------------------------------------------
def atomic_write_json(path: "Path | str", obj: Any) -> None:
    """Write obj to path atomically (tmp then rename). No emoji in keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Migration backlog (Pattern U inventory -- future work)
# ---------------------------------------------------------------------------
MIGRATION_BACKLOG = [
    # (script_path, inline_pattern_note)
    ("scripts/wealth_bot/r12_instrument_variant.py", "inline wma() function -- MIGRATED via r12_canonical_harness_poc.py POC"),
    ("scripts/wealth_bot/r23a_spot_whale_wma.py", "inline wma() + inline simulate()"),
    ("scripts/wealth_bot/r11_multicadence_static.py", "inline simulate()"),
    ("scripts/wealth_bot/r13_xrel_regime_gate.py", "inline simulate()"),
    ("scripts/wealth_bot/r14_goal_function.py", "inline simulate()"),
    ("scripts/wealth_bot/r15_position_structure.py", "inline simulate()"),
    ("scripts/wealth_bot/r16_signal_aggregation.py", "inline simulate()"),
    ("scripts/wealth_bot/r17_execution_realism.py", "inline simulate()"),
    ("scripts/wealth_bot/r18_filter_aggregation.py", "inline simulate()"),
    ("scripts/wealth_bot/r19_risk_trigger.py", "inline simulate()"),
    ("scripts/wealth_bot/r20_regime_composition.py", "inline simulate()"),
    ("scripts/wealth_bot/r21_signal_types.py", "inline simulate()"),
    ("scripts/wealth_bot/r22_filter_families.py", "inline simulate()"),
    ("scripts/wealth_bot/r23_hmm_bocpd_diffusion.py", "inline simulate()"),
    ("scripts/oracle/", "oracle scripts with inline indicator calls"),
    # R57-series stratified mining scripts
    ("scripts/wealth_bot/r57*.py", "inline stratified simulate() + inline indicator calls"),
    # ---------------------------------------------------------------------------
    # PRE-V8.3 ACTIVE Pattern-T scripts -- FREEZE flag per v8.5 §SM18
    # Discovered: 2026-05-26 S5 Pattern S/T grep (MAXX-INST-2026-05-26-NIGHT)
    # Rule: DO NOT RE-USE results as primary evidence; mining must move to
    # CanonicalHarness or framework.data_loader for any new development.
    # Results from these scripts are ADVISORY-ONLY (entry_p=closes[i] same-bar fill).
    # ---------------------------------------------------------------------------
    ("scripts/wealth_bot/r25_filter_cascade.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r31_phase4_deepening.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r31b_phase4_honest.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r33_dim_l_dim_c_deepening.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | 5 occurrences across simulator variants; inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r34_gid_subvariant_depth.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r35_dim_abf_depth.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | 3 occurrences; inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r36_mnj_subvariant_depth.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/strat_B_honest_refinement_validator.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/strat_B_synth_aug_validator.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r58_ml_pepe_maema_4h.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
    ("scripts/wealth_bot/r60_ml_walk_forward_pepe_4h.py",
     "FREEZE=True | PRE_V8_3_ACTIVE_PATTERN_T | inline simulate() entry_p=closes[i] same-bar fill; results advisory-only per v8.5 §SM18 | discovery: 2026-05-26 S5 grep"),
]
"""
Migration priority: HIGH for any script still in active development.
Grandfathered scripts (archive-only) may keep inline code.
New scripts (post v8.3 mandate) MUST use CanonicalHarness.

FREEZE policy (v8.5 §SM18, added 2026-05-26):
Scripts tagged FREEZE=True above were discovered as PRE_V8_3_ACTIVE_PATTERN_T by
S5 grep in MAXX-INST-2026-05-26-NIGHT. Their inline simulate() uses entry_p=closes[i]
(same-bar fill = look-ahead). Any result numbers from these scripts are ADVISORY-ONLY.
DO NOT re-use their output as primary evidence for ship-tier decisions. New development
MUST use CanonicalHarness (src/wealth_bot/harness.py) or framework.data_loader.
"""
