"""PLUG-IN ORACLE ENGINE -- indicator/cadence/driver-general hindsight upper bound.

================================================================================
HINDSIGHT UPPER BOUND -- descriptive, NOT a tradeable signal.
================================================================================

This is the PROPER, extensible oracle engine. It generalizes the verified
``ma_oracle_engine.MAOracleEngine`` (single indicator family, single cadence,
single driver) along THREE axes WITHOUT rewriting or breaking it:

  1. INDICATOR-GENERAL -- the signal generator is a plug-in (``Indicator``
     protocol). ``MAIndicator`` REUSES the verified causal primitives
     (``_sma`` / ``_ema`` / ``_crosses``) imported from ``ma_oracle_engine`` and
     reproduces the existing MA golden/death behavior EXACTLY. New families
     (rsi / macd / bollinger) are REGISTERED placeholders -- the registry is the
     extension point; their ``signal()`` raises NotImplementedError until built.

  2. CADENCE-GENERAL -- runs on the native ``1d`` bar OR any other cadence
     (4h / 1h / dollar / dib / range / runs_* / ...). There are TWO resolutions:

       * ``resolution="native"`` (DEFAULT, 2026-06-08) -- GENUINE multi-timeframe:
         a 4h move is analyzed on 4h bars, a 1h move on 1h bars. The series is the
         cadence's OWN bars (NO daily aggregation), so a 4h asset sees ~6x more
         bars than 1d and can enter on an intraday golden cross that a daily
         series cannot see. For ``cadence="1d"`` the native series IS the daily
         series (one bar per calendar date), so 1d is UNCHANGED.
       * ``resolution="daily"`` (back-compat) -- the prior behavior: every cadence
         is aggregated to a DAILY close (last bar of each calendar date), so all
         cadences collapse to the same daily series as 1d. Kept for reproducibility
         of the committed 1d/daily numbers; NOT genuine multi-timeframe.

     NATIVE-mode windowing (the load-bearing semantics):
       - decision bar    = last native bar with timestamp <= end of the query date
                           D (calendar end-of-day cutoff; date(bar) <= D).
       - lookback WINDOW = native bars whose calendar date is in
                           [D - lookback_days, D] (a CALENDAR-DAY window measured in
                           DAYS, but containing the NATIVE bars inside it -- so 4h
                           sees ~6x the bars 1d sees over the same span).
       - ranking         = close[decision]/close[window_start]-1 over the NATIVE
                           window (NOT the 1d-only rank_top_performers). For
                           ``cadence="1d"`` native ranking DELEGATES to the verified
                           ``rank_top_performers`` so the committed 1d ranking is
                           reproduced bit-for-bit (the test gate's invariant).
       - validity        = the rolling-validity driver works on the NATIVE series;
                           ``validity_windows`` are in DAYS and bound completed
                           round-trips by timestamp to [D - vw_days, D].
     The PRIOR (daily) bug: ``cadence`` was COSMETIC -- 4h/1h/dollar all aggregated
     to the SAME daily series as 1d, so all cadences gave IDENTICAL oracle results.
     Native resolution fixes that.

  3. DRIVER-GENERAL -- the "best (config, entry)" selection is a plug-in policy.
     ``rolling_validity`` (default) REUSES the decomposer's rolling-validity
     logic over the GIVEN ``validity_windows`` (the first window that yields a
     qualifying config wins), falling back to ``bounded_oneshot`` (the verified
     v1 max-captured pick, bounded into the lookback) when no config has enough
     completed round-trips.

WHAT "HINDSIGHT" MEANS HERE (unchanged from v1):
  Each individual config's entry SIGNAL is computed CAUSALLY (past-only:
  closes[:d_idx+1] only; a cross is the t-1 -> t sign change with no future
  leak). The hindsight is ONLY in selecting the best config / entry after the
  fact -- the allowed oracle move. The result is an UPPER BOUND on what an
  indicator-family long entry could have captured to D; it is NOT a predictive
  signal. ``hindsight=True`` is stamped on every row.

ADDITIVE GUARANTEE: this module imports and reuses
``ma_oracle_engine`` (primitives + rank_top_performers + grid) and the
``decomposer``'s daily-aggregation + rolling-validity logic. It does NOT modify
or duplicate either file's behavior.

--------------------------------------------------------------------------------
CLI:
    python src/oracle/engine.py --date 2026-05-20 [--universe u50]
        [--indicator ma] [--cadence 1d] [--lookback 30] [--top-n 25]
        [--validity-windows 180,365] [--driver rolling_validity]
        [--out runs/oracle/engine_<date>.csv]
"""
from __future__ import annotations

import argparse
import math
import sys
from abc import ABC, abstractmethod
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

__contract__ = {
    "kind": "plugin_oracle_engine",
    "inputs": [
        "ma_oracle_engine.MAOracleEngine (causal primitives + ranking + grid)",
        "oracle.decomposer (daily-aggregation idea + rolling-validity driver)",
        "chimera via pipeline.chimera_loader.ChimeraLoader (any cadence)",
    ],
    "outputs": {
        "callable": "OracleEngine.oracle(date, *, universe, indicator, cadence, "
                    "lookback_days, top_n, top_pct, validity_windows, driver) "
                    "-> pl.DataFrame",
        "csv": "runs/oracle/engine_<date>.csv",
    },
    "invariants": [
        "per-config indicator + cross signal is CAUSAL (past-only, closes[:d_idx+1])",
        "best-config / best-entry selection is hindsight (the allowed oracle move)",
        "capture_rate in [0,1]; entry_date <= date; entry/decision are past-only",
        "indicator-general: signal generator is a registry plug-in",
        "cadence-general: resolution='native' analyzes the cadence's OWN bars "
        "(genuine multi-timeframe; 4h move on 4h bars); resolution='daily' "
        "aggregates every cadence to a daily close (back-compat, collapses to 1d)",
        "native 1d == daily 1d (ranking DELEGATES to rank_top_performers at 1d) so "
        "the committed 1d behavior is reproduced bit-for-bit; only 4h/1h/event "
        "cadences change under native",
        "native lookback window = native bars with calendar date in "
        "[D - lookback_days, D]; rank = close[decision]/close[window_start]-1; "
        "validity_windows are in DAYS, bound completed round-trips by timestamp",
        "driver-general: rolling_validity over given windows, bounded_oneshot fallback",
        "output is a HINDSIGHT UPPER BOUND -- not a tradeable signal (hindsight=True)",
        "no emoji in prints (cp1252)",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(SRC / "pipeline"), str(SRC / "oracle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
# REUSE the verified causal primitives + ranking from v1 (do NOT duplicate).
from oracle.ma_oracle_engine import (  # noqa: E402
    MAOracleEngine, _to_date, _last_idx_le, _sma, _ema, _crosses,
    _build_ma_grid, _print_table_ascii,
)

HINDSIGHT_LABEL = "HINDSIGHT UPPER BOUND -- descriptive, not a tradeable signal."


# ============================================================================
# 1. INDICATOR PLUG-IN PROTOCOL
# ============================================================================
class Indicator(ABC):
    """A plug-in entry/exit signal generator.

    An indicator turns a CAUSAL daily close series + a config into golden/death
    cross indices (entries / exits). Every implementation MUST be past-only:
    output[t] depends only on closes[:t+1]. The engine slices the series to
    closes[:d_idx+1] before calling, so no row past D is ever visible; an
    indicator must additionally not peek forward WITHIN the array it is given.
    """

    name: str = "base"

    @abstractmethod
    def config_grid(self) -> list[dict]:
        """Return the list of config dicts this indicator sweeps."""
        raise NotImplementedError

    @abstractmethod
    def signal(self, dates: list, closes: np.ndarray, cfg: dict) -> dict:
        """Return at least {'golden_idx': [int...], 'death_idx': [int...]}.

        Indices are positions into ``closes`` (== ``dates``). CAUSAL: a cross at
        index t uses only closes[:t+1].
        """
        raise NotImplementedError


class MAIndicator(Indicator):
    """MA-family golden/death cross indicator.

    REUSES the verified ``_sma`` / ``_ema`` / ``_crosses`` from
    ``ma_oracle_engine`` -- this reproduces the existing MA behavior exactly.
    config_grid = the canonical 16 configs (SMA+EMA x fast{5,10,20} x
    slow{20,50,100}, fast<slow). cfg = {'family','fast','slow'}.
    """

    name = "ma"

    def __init__(self, fast=(5, 10, 20), slow=(20, 50, 100)):
        # _build_ma_grid yields (family, fast, slow) tuples == v1's grid order.
        self._grid = [
            {"family": fam, "fast": f, "slow": s}
            for (fam, f, s) in _build_ma_grid(fast, slow)
        ]

    def config_grid(self) -> list[dict]:
        return list(self._grid)

    def signal(self, dates, closes, cfg: dict) -> dict:
        fam, f, s = cfg["family"], cfg["fast"], cfg["slow"]
        if fam == "SMA":
            ma_f, ma_s = _sma(closes, f), _sma(closes, s)
        else:
            ma_f, ma_s = _ema(closes, f), _ema(closes, s)
        golden, death = _crosses(ma_f - ma_s)
        return {"golden_idx": golden, "death_idx": death}


class _NotImplementedIndicator(Indicator):
    """Registered-but-not-implemented placeholder.

    Present in the registry (so the API is indicator-general and the registry is
    the single extension point) and exposes a config_grid STUB, but ``signal``
    raises NotImplementedError until the family is built.
    """

    def __init__(self, name: str, grid_stub: list[dict]):
        self.name = name
        self._grid_stub = grid_stub

    def config_grid(self) -> list[dict]:
        return list(self._grid_stub)

    def signal(self, dates, closes, cfg: dict) -> dict:
        raise NotImplementedError(
            f"indicator '{self.name}' is registered but not implemented; "
            f"only 'ma' is implemented. Build a real {self.name.upper()}Indicator "
            f"and register it in INDICATOR_REGISTRY."
        )


# ============================================================================
# 3. INDICATOR REGISTRY -- the single extension point
# ============================================================================
INDICATOR_REGISTRY: dict[str, Indicator] = {
    "ma": MAIndicator(),
    # registered-but-not-implemented placeholders (config_grid stubs only).
    # These are REPLACED at the bottom of this module by the real implementations
    # from oracle.indicators_ta once that module is imported (deferred to avoid
    # a circular import: indicators_ta imports Indicator from here; we import
    # indicators_ta after the registry dict is defined).
    "rsi": _NotImplementedIndicator(
        "rsi", [{"period": p, "lo": 30, "hi": 70} for p in (7, 14, 21)]),
    "macd": _NotImplementedIndicator(
        "macd", [{"fast": 12, "slow": 26, "signal": 9}]),
    "bollinger": _NotImplementedIndicator(
        "bollinger", [{"period": p, "k": k} for p in (20,) for k in (2.0, 2.5)]),
}

# ---------------------------------------------------------------------------
# Wire real pandas_ta-backed implementations into the registry.
# Import is DEFERRED to the bottom of this module to avoid a circular import:
#   oracle.indicators_ta imports Indicator (the ABC) from oracle.engine.
#   oracle.engine imports oracle.indicators_ta (here, bottom-of-file).
# By the time Python reaches this line, Indicator + _NotImplementedIndicator +
# INDICATOR_REGISTRY are all fully defined, so indicators_ta can import
# Indicator safely.  The try/except is a hard-fail guard: if indicators_ta
# imports broken (e.g. pandas_ta unavailable), we surface the error
# immediately rather than silently leaving NotImplementedError stubs.
# ---------------------------------------------------------------------------
try:
    from oracle.indicators_ta import (  # noqa: E402
        RSIIndicator, MACDIndicator, BollingerIndicator,
    )
    INDICATOR_REGISTRY["rsi"]      = RSIIndicator()
    INDICATOR_REGISTRY["macd"]     = MACDIndicator()
    INDICATOR_REGISTRY["bollinger"] = BollingerIndicator()
except ImportError as _e:
    raise ImportError(
        f"oracle.indicators_ta could not be imported; rsi/macd/bollinger "
        f"will remain as NotImplementedError stubs. "
        f"Install pandas_ta and ensure numpy.NaN shim is present. "
        f"Original error: {_e}"
    ) from _e


# ============================================================================
# 4. THE ENGINE
# ============================================================================
class OracleEngine:
    """Plug-in oracle engine: indicator x cadence x driver general.

    HINDSIGHT UPPER BOUND -- descriptive, not a tradeable signal. The per-config
    signal is CAUSAL (past-only); only the best-config/best-entry selection is
    hindsight (the allowed oracle move).

    Reuses the verified v1 engine (``MAOracleEngine``) for ranking + the causal
    grid, and mirrors the decomposer's daily-aggregation + rolling-validity
    logic for cadence- and driver-generality.
    """

    hindsight = True  # marker: every output of this engine is a hindsight bound.

    def __init__(self, loader: ChimeraLoader | None = None,
                 default_resolution: str = "native"):
        self.loader = loader or ChimeraLoader()
        # v1 engine: canonical ranking (rank_top_performers) on the 1d bar.
        self.ma_engine = MAOracleEngine(self.loader)
        # per (sym, cadence) -> ascending (date, close) DAILY-aggregated frame cache.
        self._series_cache: dict[tuple[str, str], pl.DataFrame | None] = {}
        # per (sym, cadence) -> ascending (ts, date, close) NATIVE-resolution frame
        # cache (the cadence's OWN bars; NO daily aggregation).
        self._native_cache: dict[tuple[str, str], pl.DataFrame | None] = {}
        # PERF (2026-06-09): per (sym, cadence, resolution, indicator, cfg-tuple)
        # -> (golden_full, death_full) crosses computed ONCE over the WHOLE series.
        # The MA series + golden/death cross indices for a (sym, cadence, config)
        # are a function of the FULL price series and DO NOT depend on the
        # decision-day D: _sma/_ema are past-only (output[t] uses only closes[:t+1])
        # and a cross at index t depends only on spread[t-1],spread[t]. So the
        # crosses over closes[:d_idx+1] are EXACTLY the full-series crosses filtered
        # to idx <= d_idx (proven bit-exact in the probe). This removes the
        # O(D^2) per-decision-day signal recompute (panel.py sweeps a date range
        # and the prior code re-ran _sma/_ema/_crosses on closes[:d_idx+1] for
        # EVERY (date, config)). Per-D logic still gates causally to idx <= d_idx
        # via _signal_le below -- identical numbers, just not recomputed.
        self._signal_cache: dict[tuple, tuple[list, list]] = {}
        # When False, _signal_le recomputes on the truncated slice (the literal
        # pre-optimization path) -- used ONLY by the bit-exact proof to compare
        # cached vs uncached. Production default True (memoized full-series).
        self._use_signal_cache: bool = True
        # The resolution every series consumer (incl. _daily_series, which the
        # adaptive chooser + compare grader call) sees by DEFAULT. 'native'
        # (genuine multi-timeframe) makes those downstream callers consistent with
        # oracle()'s new native default so a 4h oracle is graded against a 4h model
        # (apples-to-apples). At 1d native == daily, so this is a no-op there.
        # oracle() temporarily overrides this from its `resolution=` argument.
        if default_resolution not in ("native", "daily"):
            raise ValueError(
                f"unknown default_resolution '{default_resolution}'")
        self.default_resolution = default_resolution

    # ---- data access (cadence-general series; resolution-aware) ----------
    def _daily_series(self, sym: str, cadence: str) -> pl.DataFrame | None:
        """Ascending (date, close[, ts]) frame for sym at a cadence.

        RESOLUTION-AWARE (2026-06-08): if ``self.default_resolution == 'native'``
        (the default) this returns the cadence's NATIVE bars (via _native_series)
        -- a (ts, date, close) frame with one row per native bar -- so downstream
        callers that read ``series["date"]`` / ``series["close"]`` (the adaptive
        chooser, the compare grader) operate at the SAME native resolution as the
        oracle's native default (apples-to-apples). At cadence='1d' the native
        series IS the daily series, so 1d is unchanged.

        If ``self.default_resolution == 'daily'`` (back-compat) it returns the
        DAILY-aggregated (date, close) frame: 1d -> native daily bar; other
        cadences -> last bar's close on each calendar date (the prior behavior,
        same idea as decomposer._daily_series). None if unavailable.
        """
        if self.default_resolution == "native":
            return self._native_series(sym, cadence)
        key = (sym, cadence)
        if key in self._series_cache:
            return self._series_cache[key]
        try:
            df = self.loader.load(sym, cadence=cadence, features=["close", "date"])
        except Exception:
            self._series_cache[key] = None
            return None
        if "date" not in df.columns or "close" not in df.columns:
            self._series_cache[key] = None
            return None
        df = df.select(["date", "close"]).drop_nulls()
        if cadence == "1d":
            out = df.unique(subset=["date"], keep="last").sort("date")
        else:
            out = (df.sort("date")
                     .group_by("date")
                     .agg(pl.col("close").last().alias("close"))
                     .sort("date"))
        self._series_cache[key] = out
        return out

    # ---- data access (NATIVE-resolution series -- genuine multi-timeframe) -
    def _native_series(self, sym: str, cadence: str) -> pl.DataFrame | None:
        """Ascending (ts, date, close) frame at the cadence's OWN bars.

        NO daily aggregation -- a 4h cadence yields the 4h bars, a 1h cadence the
        1h bars, so a 4h move is analyzed on 4h bars (genuine multi-timeframe).
        ``ts`` is the chimera 13-digit-ms timestamp (the within-day ordering key);
        ``date`` is the bar's calendar date (used to bound the calendar-day lookback
        window). For ``cadence="1d"`` this equals the daily series (chimera already
        has one bar per calendar date), so 1d native is identical to 1d daily.
        None if unavailable.
        """
        key = (sym, cadence)
        if key in self._native_cache:
            return self._native_cache[key]
        try:
            df = self.loader.load(
                sym, cadence=cadence, features=["close", "date", "timestamp"])
        except Exception:
            self._native_cache[key] = None
            return None
        cols = df.columns
        if "close" not in cols or "timestamp" not in cols:
            self._native_cache[key] = None
            return None
        sel = ["timestamp", "close"]
        if "date" in cols:
            sel = ["timestamp", "date", "close"]
        out = (df.select(sel)
                 .drop_nulls()
                 .rename({"timestamp": "ts"})
                 .unique(subset=["ts"], keep="last")
                 .sort("ts"))
        if "date" not in out.columns:
            # derive a calendar date from the ms timestamp (UTC) for windowing.
            out = out.with_columns(
                (pl.col("ts") * 1000).cast(pl.Datetime("us")).dt.date().alias("date"))
        self._native_cache[key] = out
        return out

    # ---- memoized per-config full-series crosses (the O(D^2) fix) ---------
    @staticmethod
    def _cfg_key(cfg: dict):
        """Stable hashable key for an indicator config dict (order-independent)."""
        return tuple(sorted(cfg.items()))

    def _full_crosses(self, sym, cadence, resolution, indicator: Indicator,
                      cfg: dict, full_closes: np.ndarray, full_dates):
        """(golden_full, death_full) over the WHOLE series, computed ONCE and
        memoized on the engine instance.

        Keyed on (sym, cadence, resolution, indicator.name, cfg-tuple). The
        crosses depend ONLY on the full price series (NOT on any decision-day D),
        because _sma/_ema are past-only and a cross at t uses only spread[t-1],
        spread[t]. This is the per-config computation the old code redid on
        closes[:d_idx+1] for every (date, config) -- the O(D^2) re-mining.

        ``resolution`` MUST be the resolution the ``full_closes`` series was
        loaded under (native vs daily) -- it is an EXPLICIT key component (NOT
        read from self.default_resolution, which a caller like capture_of_config
        restores before this runs), so native and daily series never collide and
        a stale instance flag can never mislabel a cached entry.
        """
        key = (sym, cadence, resolution, indicator.name, self._cfg_key(cfg))
        hit = self._signal_cache.get(key)
        if hit is not None:
            return hit
        sig = indicator.signal(full_dates, full_closes, cfg)
        golden = list(sig.get("golden_idx", []))
        death = list(sig.get("death_idx", []))
        self._signal_cache[key] = (golden, death)
        return golden, death

    def _signal_le(self, sym, cadence, resolution, indicator: Indicator,
                   cfg: dict, full_closes: np.ndarray, full_dates, d_idx: int):
        """Causal (golden, death) for THIS config as of decision-day index ``d_idx``.

        Returns the full-series crosses filtered to idx <= d_idx -- EXACTLY equal
        to ``indicator.signal(dates, closes[:d_idx+1], cfg)`` (bit-exact: a cross
        at index i <= d_idx depends only on closes[i-1],closes[i], independent of
        where the array is truncated). This is the drop-in for the per-D
        recompute. ``death_idx`` here are full-series indices but, like ``golden``,
        are all <= d_idx, so they index identically into closes[:d_idx+1].
        ``resolution`` keys the cache (see _full_crosses).

        When ``self._use_signal_cache`` is False (the proof reference path), this
        recomputes the signal on the TRUNCATED slice closes[:d_idx+1] -- the
        LITERAL pre-optimization behavior -- so cached vs uncached can be asserted
        frame-equal over a grid. Same numbers; one path memoizes, one does not."""
        if not self._use_signal_cache:
            past = full_closes[:d_idx + 1]
            sig = indicator.signal(full_dates, past, cfg)
            return (list(sig.get("golden_idx", [])),
                    list(sig.get("death_idx", [])))
        golden_full, death_full = self._full_crosses(
            sym, cadence, resolution, indicator, cfg, full_closes, full_dates)
        golden = [g for g in golden_full if g <= d_idx]
        death = [x for x in death_full if x <= d_idx]
        return golden, death

    # ---- per-config completed round-trip validity (driver primitive) -----
    @staticmethod
    def _config_rolling_validity(dates, past, golden, death, d, validity_window):
        """Mean capture_rate of COMPLETED round-trips (golden->death) whose ENTRY
        is within [D - validity_window, D]. Causal per round-trip (each rate uses
        only closes inside that round-trip's own [entry, exit]).

        Mirrors decomposer._config_rolling_validity but takes pre-computed
        golden/death so it is indicator-general. Returns (mean_rate|None, n).
        """
        if not golden:
            return None, 0
        death_sorted = sorted(death)
        rates = []
        for g in golden:
            entry_date = dates[g]
            if (d - entry_date).days > validity_window:
                continue
            ex = next((dx for dx in death_sorted if dx > g), None)
            if ex is None:
                continue  # not yet completed -> not a closed validity sample
            c_entry = float(past[g])
            if c_entry <= 0 or np.isnan(c_entry):
                continue
            c_exit = float(past[ex])
            captured = (c_exit - c_entry) / c_entry
            seg = past[g:ex + 1]
            c_min = float(np.min(seg))
            perfect = (c_exit - c_min) / c_min if c_min > 0 else 0.0
            if perfect > 0:
                rates.append(max(0.0, min(1.0, captured / perfect)))
            else:
                rates.append(0.0)
        if not rates:
            return None, 0
        return float(np.mean(rates)), len(rates)

    # ---- a single config's in-position-at-D candidate (causal) -----------
    def _config_candidate(self, dates, past, d, d_idx, cD, lookback_days,
                          indicator: Indicator, cfg: dict,
                          sym=None, cadence=None, resolution=None,
                          full_closes=None):
        """Run the indicator for one config; return an in-position-at-D candidate
        dict or None. Bounded: the selected golden cross must be within
        [D - lookback_days, D] (no unbounded-lookback artifact). CAUSAL: `past`
        is already closes[:d_idx+1].

        Crosses come from the memoized full-series cache sliced to idx <= d_idx
        (bit-exact equal to indicator.signal(dates, past, cfg)); the legacy
        recompute path is used only when sym/cadence/full_closes are not provided
        (the _uncached proof reference)."""
        if sym is not None and full_closes is not None:
            golden, death = self._signal_le(
                sym, cadence, resolution, indicator, cfg, full_closes, dates,
                d_idx)
        else:
            sig = indicator.signal(dates, past, cfg)
            golden = sig.get("golden_idx", [])
            death = sig.get("death_idx", [])
        if not golden:
            return None
        last_g = golden[-1]
        # in position at D iff no death cross strictly after the last golden.
        if any(dx > last_g for dx in death):
            return None
        entry_date = dates[last_g]
        days_back = (d - entry_date).days
        if days_back > lookback_days:        # BOUNDED LOOKBACK (no artifact)
            return None
        c_entry = float(past[last_g])
        if c_entry <= 0 or np.isnan(c_entry):
            return None
        captured = (cD - c_entry) / c_entry
        return {
            "cfg": cfg, "entry_idx": last_g, "entry_date": entry_date,
            "days_back": int(days_back), "captured": float(captured),
            "golden": golden, "death": death,
        }

    # ---- driver: pick best (config, entry) for one asset/cadence ---------
    def _best_for_asset(self, sym, cadence, d, lookback_days, indicator,
                        validity_windows, driver, min_valid_trades=3):
        """Return (best_dict | None) for one asset at one cadence under `driver`.

        rolling_validity: for each given validity window (in order), score the
        in-position bounded configs by mean completed-round-trip capture_rate;
        the FIRST window with >= min_valid_trades qualifying config wins, and we
        pick best (validity desc, captured desc). Falls back to bounded_oneshot
        (max captured) when no window qualifies. bounded_oneshot: max captured.
        """
        series = self._daily_series(sym, cadence)
        if series is None or len(series) == 0:
            return None
        dates = series["date"].to_list()
        closes = series["close"].to_numpy().astype(float)
        d_idx = _last_idx_le(dates, d)
        if d_idx is None:
            return None
        past = closes[:d_idx + 1]            # CAUSAL slice
        cD = float(past[d_idx])

        # perfect-entry oracle over [D - lookback_days, D] (capture_rate denom).
        win_lo = max(0, d_idx - lookback_days)
        c_min = float(np.min(closes[win_lo:d_idx + 1]))
        perfect_return = (cD - c_min) / c_min if c_min > 0 else 0.0

        candidates = []
        for cfg in indicator.config_grid():
            cand = self._config_candidate(dates, past, d, d_idx, cD,
                                          lookback_days, indicator, cfg,
                                          sym=sym, cadence=cadence,
                                          resolution=self.default_resolution,
                                          full_closes=closes)
            if cand is not None:
                candidates.append(cand)
        if not candidates:
            return {"in_position": False, "perfect_return": perfect_return,
                    "resolved_day": dates[d_idx]}

        validity_window_used = None
        validity_score = None
        used_fallback = False
        chosen = None

        if driver == "rolling_validity":
            for vw in validity_windows:
                scored = []
                for cand in candidates:
                    rv, n_tr = self._config_rolling_validity(
                        dates, past, cand["golden"], cand["death"], d, vw)
                    if rv is not None and n_tr >= min_valid_trades:
                        scored.append((rv, cand["captured"], cand))
                if scored:
                    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
                    validity_score, _, chosen = scored[0]
                    validity_window_used = vw
                    break
            if chosen is None:
                # fallback: bounded one-shot (max captured)
                used_fallback = True
                chosen = max(candidates, key=lambda c: c["captured"])
        else:  # bounded_oneshot
            chosen = max(candidates, key=lambda c: c["captured"])

        cap_rate = 0.0
        if perfect_return > 0:
            cap_rate = max(0.0, min(1.0, chosen["captured"] / perfect_return))
        return {
            "in_position": True,
            "resolved_day": dates[d_idx],
            "cfg": chosen["cfg"],
            "entry_date": chosen["entry_date"],
            "days_back": chosen["days_back"],
            "captured_return": float(chosen["captured"]),
            "perfect_return": float(perfect_return),
            "capture_rate": float(cap_rate),
            "validity_window_used": validity_window_used,
            "validity_score": validity_score,
            "used_fallback": used_fallback,
        }

    # ---- NATIVE ranking (genuine multi-timeframe) -----------------------
    def _native_rank(self, d, universe, cadence, lookback_days, top_n):
        """Rank assets by trailing return over the NATIVE calendar-day window.

        perf(sym) = close[decision]/close[window_start]-1, where:
          - decision   = last native bar with date <= D (end-of-day cutoff),
          - window_start = first native bar with date >= (D - lookback_days).
        So a 4h asset's rank is computed over its 4h bars inside the same calendar
        span the 1d asset uses (more bars, same span). Returns list[(sym, perf)]
        sorted desc, length <= top_n.
        """
        from datetime import timedelta
        cutoff = d - timedelta(days=lookback_days)
        rows = []
        for sym in self.loader.universes.list(universe):
            series = self._native_series(sym, cadence)
            if series is None or len(series) == 0:
                continue
            dates = series["date"].to_list()
            closes = series["close"].to_numpy().astype(float)
            d_idx = _last_idx_le(dates, d)
            if d_idx is None:
                continue
            # window_start = first bar with date >= cutoff (>= D - lookback_days).
            j = d_idx
            while j > 0 and dates[j - 1] >= cutoff:
                j -= 1
            c0 = float(closes[j])
            cD = float(closes[d_idx])
            if c0 <= 0 or np.isnan(c0) or np.isnan(cD):
                continue
            rows.append((sym, cD / c0 - 1.0))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:top_n]

    # ---- driver: pick best (config, entry) for one asset/cadence (NATIVE) -
    def _best_for_asset_native(self, sym, cadence, d, lookback_days, indicator,
                               validity_windows, driver, min_valid_trades=3):
        """NATIVE-resolution analog of ``_best_for_asset``.

        Runs the indicator on the cadence's OWN bars (NO daily aggregation). The
        lookback window is the NATIVE bars whose calendar date is in
        [D - lookback_days, D]; the in-position golden cross must fall inside that
        window. captured / perfect / capture_rate use the SAME formula as the
        daily path but on the native bars. Reports ``entry_ts`` / ``bars_back``
        (native bars between entry and decision) alongside ``entry_date`` (the
        calendar date of entry_ts, for readability), plus ``n_bars_in_window``.
        """
        from datetime import timedelta
        series = self._native_series(sym, cadence)
        if series is None or len(series) == 0:
            return None
        ts = series["ts"].to_list()
        dates = series["date"].to_list()
        closes = series["close"].to_numpy().astype(float)
        d_idx = _last_idx_le(dates, d)          # last native bar with date <= D
        if d_idx is None:
            return None
        past = closes[:d_idx + 1]               # CAUSAL slice (native bars)
        cD = float(past[d_idx])

        # native lookback window = bars with date in [D - lookback_days, D].
        cutoff = d - timedelta(days=lookback_days)
        win_lo = d_idx
        while win_lo > 0 and dates[win_lo - 1] >= cutoff:
            win_lo -= 1
        n_bars_in_window = d_idx - win_lo + 1
        c_min = float(np.min(closes[win_lo:d_idx + 1]))
        perfect_return = (cD - c_min) / c_min if c_min > 0 else 0.0

        # per-config in-position candidate, bounded to the native lookback window.
        # Crosses come from the memoized full-series cache sliced to idx <= d_idx
        # (bit-exact equal to indicator.signal(dates, past, cfg)) -- removes the
        # O(D^2) per-decision-day recompute.
        candidates = []
        for cfg in indicator.config_grid():
            golden, death = self._signal_le(
                sym, cadence, self.default_resolution, indicator, cfg,
                closes, dates, d_idx)
            if not golden:
                continue
            last_g = golden[-1]
            if any(dx > last_g for dx in death):
                continue                        # closed before/at D -> flat at D
            if last_g < win_lo:                 # BOUNDED to the native window
                continue
            c_entry = float(past[last_g])
            if c_entry <= 0 or np.isnan(c_entry):
                continue
            captured = (cD - c_entry) / c_entry
            candidates.append({
                "cfg": cfg, "entry_idx": last_g, "entry_ts": ts[last_g],
                "entry_date": dates[last_g],
                "bars_back": int(d_idx - last_g),
                "days_back": int((d - dates[last_g]).days),
                "captured": float(captured), "golden": golden, "death": death,
            })
        if not candidates:
            return {"in_position": False, "perfect_return": perfect_return,
                    "resolved_day": dates[d_idx],
                    "n_bars_in_window": int(n_bars_in_window)}

        validity_window_used = None
        validity_score = None
        used_fallback = False
        chosen = None
        if driver == "rolling_validity":
            for vw in validity_windows:
                scored = []
                for cand in candidates:
                    rv, n_tr = self._config_rolling_validity(
                        dates, past, cand["golden"], cand["death"], d, vw)
                    if rv is not None and n_tr >= min_valid_trades:
                        scored.append((rv, cand["captured"], cand))
                if scored:
                    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
                    validity_score, _, chosen = scored[0]
                    validity_window_used = vw
                    break
            if chosen is None:
                used_fallback = True
                chosen = max(candidates, key=lambda c: c["captured"])
        else:  # bounded_oneshot
            chosen = max(candidates, key=lambda c: c["captured"])

        cap_rate = 0.0
        if perfect_return > 0:
            cap_rate = max(0.0, min(1.0, chosen["captured"] / perfect_return))
        return {
            "in_position": True,
            "resolved_day": dates[d_idx],
            "cfg": chosen["cfg"],
            "entry_ts": chosen["entry_ts"],
            "entry_date": chosen["entry_date"],
            "bars_back": chosen["bars_back"],
            "days_back": chosen["days_back"],
            "captured_return": float(chosen["captured"]),
            "perfect_return": float(perfect_return),
            "capture_rate": float(cap_rate),
            "validity_window_used": validity_window_used,
            "validity_score": validity_score,
            "used_fallback": used_fallback,
            "n_bars_in_window": int(n_bars_in_window),
        }

    # ---- per-config capture grader (SAME computation the oracle ranks with) -
    def capture_of_config(self, sym, date, config, *, cadence="1d",
                          lookback_days=30, resolution="native",
                          indicator="ma"):
        """HINDSIGHT EVAL of a GIVEN config for one asset as of D, computed by the
        EXACT SAME native (or daily) window / perfect-return / entry logic the
        oracle uses to find its best config.

        This is the per-config kernel of ``_best_for_asset_native`` /
        ``_best_for_asset`` factored out so a SINGLE (given) config can be graded
        on the IDENTICAL scale as the oracle. The engine's ``oracle()`` best result
        equals ``max`` over the indicator grid of this function -- so when the
        compare grader uses THIS method, ``model_realized_capture <= oracle_capture``
        holds BY CONSTRUCTION (the ceiling is the per-asset max over the grid, and
        the model picks one config from the same grid graded the same way).

        resolution:
          * 'native' (default) -- decision bar = last native bar with date <= D;
            window = native bars with date in [D - lookback_days, D]; c_min over
            that window; entry = the config's last golden cross, which must fall
            inside the window with no later death cross by D. Identical to
            ``_best_for_asset_native``'s per-config branch.
          * 'daily' -- decision bar / window / entry use the index-based daily-path
            logic of ``_best_for_asset`` (``_config_candidate``): win_lo =
            max(0, d_idx - lookback_days), entry bounded by days_back <= lookback_days.

        config: 'FAM(fast,slow)' string (the shared _fmt_cfg form) or a
                {'family','fast','slow'} dict. indicator must be 'ma' (the
                implemented family).

        Returns a dict with the same keys the compare grader needs:
            sym, config, entry_date, entry_ts, bars_back, days_back,
            captured_return, perfect_return, capture_rate, in_position, note.
        The capture_rate is on the SAME scale as oracle_capture (<= it by
        construction when config is in the same grid).
        """
        from datetime import timedelta
        base = {
            "sym": sym,
            "config": config if isinstance(config, str) else None,
            "entry_date": None, "entry_ts": None, "bars_back": None,
            "days_back": None, "captured_return": 0.0, "perfect_return": 0.0,
            "capture_rate": 0.0, "in_position": False, "note": "",
        }
        if indicator != "ma":
            base["note"] = (
                f"capture_of_config only implements 'ma'; got '{indicator}'")
            return base
        if resolution not in ("native", "daily"):
            raise ValueError(
                f"unknown resolution '{resolution}'; use 'native' or 'daily'")
        cfg = config if isinstance(config, dict) else _parse_ma_config(config)
        if cfg is None:
            base["note"] = "no/unparseable config (model did not pick a config)"
            return base
        base["config"] = _fmt_cfg("ma", cfg)
        ind = INDICATOR_REGISTRY["ma"]

        native = (resolution == "native")
        d = _to_date(date)
        # Pin the series resolution for this grade so _native_series / _daily_series
        # return the same bars the oracle used. Restored in finally (no leak).
        _prev_default = self.default_resolution
        self.default_resolution = resolution
        try:
            if native:
                series = self._native_series(sym, cadence)
            else:
                series = self._daily_series(sym, cadence)
        finally:
            self.default_resolution = _prev_default
        if series is None or len(series) == 0:
            base["note"] = "no data"
            return base
        dates = series["date"].to_list()
        closes = series["close"].to_numpy().astype(float)
        ts = series["ts"].to_list() if "ts" in series.columns else None
        d_idx = _last_idx_le(dates, d)
        if d_idx is None:
            base["note"] = "date before first bar"
            return base
        past = closes[:d_idx + 1]            # CAUSAL slice
        cD = float(past[d_idx])

        # window + perfect_return -- IDENTICAL to the oracle's per-resolution logic.
        if native:
            # native: calendar-day window [D - lookback_days, D] over native bars
            # (mirrors _best_for_asset_native).
            cutoff = d - timedelta(days=lookback_days)
            win_lo = d_idx
            while win_lo > 0 and dates[win_lo - 1] >= cutoff:
                win_lo -= 1
        else:
            # daily: index-based last-N-bars window (mirrors _best_for_asset).
            win_lo = max(0, d_idx - lookback_days)
        c_min = float(np.min(closes[win_lo:d_idx + 1]))
        perfect_return = (cD - c_min) / c_min if c_min > 0 else 0.0
        base["perfect_return"] = float(perfect_return)

        # CAUSAL signal for THIS config (reuse the verified indicator primitives).
        # Memoized full-series crosses sliced to idx <= d_idx -- bit-exact equal
        # to ind.signal(dates, past, cfg) (a cross at i <= d_idx is independent of
        # where the series is truncated). ``resolution`` (the local, pinned for
        # the series load) keys the cache -- NOT self.default_resolution, which the
        # finally above already restored.
        golden, death = self._signal_le(
            sym, cadence, resolution, ind, cfg, closes, dates, d_idx)
        if not golden:
            base["note"] = "config has no golden cross by D (flat)"
            return base
        last_g = golden[-1]
        # in position at D iff no death cross strictly after the last golden.
        if any(dx > last_g for dx in death):
            base["note"] = "config closed (death cross) before/at D -> flat at D"
            return base
        # entry must fall inside the SAME bound the oracle used.
        if native:
            if last_g < win_lo:               # outside the native window
                base["note"] = ("config golden cross is outside the native "
                                "lookback window -> not an oracle candidate")
                return base
        else:
            if (d - dates[last_g]).days > lookback_days:   # daily-path bound
                base["note"] = ("config golden cross is older than lookback_days "
                                "-> not an oracle candidate")
                return base
        c_entry = float(past[last_g])
        if c_entry <= 0 or np.isnan(c_entry):
            base["note"] = "bad entry close"
            return base
        captured = (cD - c_entry) / c_entry
        cap_rate = 0.0
        if perfect_return > 0:
            cap_rate = max(0.0, min(1.0, captured / perfect_return))
        base.update({
            "entry_date": dates[last_g],
            "entry_ts": (ts[last_g] if ts is not None else None),
            "bars_back": int(d_idx - last_g) if native else None,
            "days_back": int((d - dates[last_g]).days),
            "captured_return": float(captured),
            "perfect_return": float(perfect_return),
            "capture_rate": float(cap_rate),
            "in_position": True,
            "note": "" if perfect_return > 0 else "no up-move in window; cap_rate=0",
        })
        return base

    # ---- the public oracle ----------------------------------------------
    def oracle(self, date, *, universe: str = "u50", indicator: str = "ma",
               cadence: str = "1d", lookback_days: int = 30, top_n: int = 25,
               top_pct: float | None = None,
               validity_windows: tuple[int, ...] = (180, 365),
               driver: str = "rolling_validity",
               resolution: str = "native") -> pl.DataFrame:
        """Top-N (or top-pct) performers as of `date`, with the best causal
        indicator (config, entry) under the chosen driver per asset.

        HINDSIGHT UPPER BOUND -- not a tradeable signal (hindsight=True on every
        row). The per-config signal is CAUSAL/past-only; only the best-config /
        best-entry pick is hindsight.

        indicator: key into INDICATOR_REGISTRY ('ma' implemented; rsi/macd/
                   bollinger registered placeholders).
        cadence: '1d' or any other cadence ('4h'/'1h'/'dollar'/'dib'/...).
        resolution: 'native' (DEFAULT) analyzes the cadence's OWN bars -- genuine
                   multi-timeframe (a 4h move on 4h bars; 4h sees ~6x more bars
                   than 1d). For cadence='1d' native == the daily series, so 1d is
                   UNCHANGED. 'daily' (back-compat) aggregates EVERY cadence to a
                   daily close (collapses all cadences to the 1d series -- the
                   prior behavior, kept for reproducibility).
        top_pct: if given, top_n = ceil(top_pct * universe_size).
        validity_windows: windows (in DAYS) the rolling_validity driver tries in order.
        driver: 'rolling_validity' (default) | 'bounded_oneshot'.
        """
        d = _to_date(date)
        if indicator not in INDICATOR_REGISTRY:
            raise KeyError(
                f"unknown indicator '{indicator}'; registered: "
                f"{sorted(INDICATOR_REGISTRY)}")
        ind = INDICATOR_REGISTRY[indicator]
        if driver not in ("rolling_validity", "bounded_oneshot"):
            raise ValueError(f"unknown driver '{driver}'")
        if resolution not in ("native", "daily"):
            raise ValueError(
                f"unknown resolution '{resolution}'; use 'native' or 'daily'")
        if isinstance(validity_windows, int):
            validity_windows = (validity_windows,)
        validity_windows = tuple(validity_windows)

        # resolve top_n from top_pct against the universe size if requested.
        if top_pct is not None:
            uni_size = len(self.loader.universes.list(universe))
            top_n = max(1, math.ceil(top_pct * uni_size))

        native = (resolution == "native")
        # Pin the series resolution for the duration of THIS call so the daily
        # path's _daily_series returns daily-aggregated bars when resolution=
        # 'daily' (back-compat) and native bars when 'native'. Restored in finally
        # so we never leak per-call state onto the engine instance.
        _prev_default = self.default_resolution
        self.default_resolution = resolution
        try:
            # RANKING:
            #  - daily resolution: REUSE v1's verified 1d ranking (identical top set).
            #  - native resolution at 1d: DELEGATE to rank_top_performers too, so the
            #    committed 1d ranking is reproduced bit-for-bit (the gate's invariant).
            #  - native resolution at 4h/1h/...: rank on the cadence's NATIVE window.
            if native and cadence != "1d":
                ranked = self._native_rank(
                    d, universe, cadence, lookback_days, top_n)
            else:
                ranked = self.ma_engine.rank_top_performers(
                    d, universe, lookback_days, top_n)
            return self._assemble_oracle_rows(
                ranked, native, sym_cadence=cadence, d=d,
                lookback_days=lookback_days, ind=ind,
                validity_windows=validity_windows, driver=driver,
                indicator=indicator, resolution=resolution)
        finally:
            self.default_resolution = _prev_default

    def _assemble_oracle_rows(self, ranked, native, *, sym_cadence, d,
                              lookback_days, ind, validity_windows, driver,
                              indicator, resolution):
        """Build the per-asset oracle output rows (native or daily path)."""
        cadence = sym_cadence
        rows = []
        for rank, (sym, perf) in enumerate(ranked, start=1):
            if native:
                best = self._best_for_asset_native(
                    sym, cadence, d, lookback_days, ind, validity_windows, driver)
            else:
                best = self._best_for_asset(
                    sym, cadence, d, lookback_days, ind, validity_windows, driver)
            base = {
                "sym": sym,
                "perf_rank": rank,
                "trailing_perf": round(float(perf), 6),
                "indicator": indicator,
                "cadence": cadence,
                "resolution": resolution,
                "best_config": None,
                "entry_date": None,
                "days_back": None,
                "bars_back": None,
                "n_bars_in_window": (best.get("n_bars_in_window") if best else None),
                "captured_return": 0.0,
                "perfect_return": (round(best["perfect_return"], 6)
                                   if best else 0.0),
                "capture_rate": 0.0,
                "in_position": False,
                "validity_window_used": None,
                "validity_score": None,
                "hindsight": True,
            }
            if best and best.get("in_position"):
                cfg = best["cfg"]
                base.update({
                    "best_config": _fmt_cfg(indicator, cfg),
                    "entry_date": str(best["entry_date"]),
                    "days_back": best["days_back"],
                    # bars_back is native-only (None in daily resolution).
                    "bars_back": best.get("bars_back"),
                    "captured_return": round(best["captured_return"], 6),
                    "perfect_return": round(best["perfect_return"], 6),
                    "capture_rate": round(best["capture_rate"], 6),
                    "in_position": True,
                    "validity_window_used": best["validity_window_used"],
                    "validity_score": (round(best["validity_score"], 6)
                                       if best["validity_score"] is not None
                                       else None),
                })
            rows.append(base)

        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows, infer_schema_length=None).sort("perf_rank")


# ============================================================================
# helpers
# ============================================================================
def _fmt_cfg(indicator: str, cfg: dict) -> str:
    """Render a config dict compactly. For 'ma' this matches v1's best_ti string
    'FAM(fast,slow)' so outputs are directly comparable."""
    if indicator == "ma":
        return f"{cfg['family']}({cfg['fast']},{cfg['slow']})"
    return ",".join(f"{k}={v}" for k, v in cfg.items())


def _parse_ma_config(config_str) -> dict | None:
    """Parse the canonical 'FAM(fast,slow)' string (the shared _fmt_cfg form for
    indicator='ma') back into {'family','fast','slow'}. Returns the dict unchanged
    if a dict is passed; None on anything that does not match (None / non-MA)."""
    if isinstance(config_str, dict):
        return config_str
    if not config_str or "(" not in config_str or ")" not in config_str:
        return None
    try:
        fam, rest = config_str.split("(", 1)
        inner = rest.rstrip(")")
        f_str, s_str = inner.split(",")
        fam = fam.strip().upper()
        if fam not in ("SMA", "EMA"):
            return None
        return {"family": fam, "fast": int(f_str), "slow": int(s_str)}
    except Exception:
        return None


def _reconcile_one(engine: OracleEngine, sym: str, date, cadence,
                   lookback_days, indicator, validity_windows, driver):
    """Hand reconciliation for one asset: print close series around the chosen
    entry, show the golden cross is real + past-only, and that
    captured_return = close[D]/close[entry]-1 reconciles."""
    d = _to_date(date)
    best = engine._best_for_asset(
        sym, cadence, d, lookback_days, INDICATOR_REGISTRY[indicator],
        tuple(validity_windows), driver)
    series = engine._daily_series(sym, cadence)
    dates = series["date"].to_list()
    closes = series["close"].to_numpy().astype(float)
    d_idx = _last_idx_le(dates, d)
    cD = closes[d_idx]
    print("\n--- HAND RECONCILIATION (one asset) ---")
    print(f"  asset            : {sym}")
    print(f"  cadence          : {cadence}  indicator={indicator}  driver={driver}")
    print(f"  query date D     : {date}  (resolved day: {dates[d_idx]})")
    print(f"  close[D]         : {cD:.8f}")
    if not best or not best.get("in_position"):
        print("  (no in-position config at D -- nothing to reconcile)")
        return
    e_idx = _last_idx_le(dates, best["entry_date"])
    c_entry = closes[e_idx]
    manual = (cD - c_entry) / c_entry
    print(f"  best_config      : {_fmt_cfg(indicator, best['cfg'])}")
    print(f"  entry_date       : {best['entry_date']}  (days_back={best['days_back']})")
    print(f"  close[entry]     : {c_entry:.8f}")
    # show the close series and the spread around the entry to prove the cross.
    cfg = best["cfg"]
    past = closes[:d_idx + 1]
    if indicator == "ma" and cfg["family"] == "SMA":
        ma_f, ma_s = _sma(past, cfg["fast"]), _sma(past, cfg["slow"])
    elif indicator == "ma":
        ma_f, ma_s = _ema(past, cfg["fast"]), _ema(past, cfg["slow"])
    else:
        ma_f = ma_s = None
    print("  -- close + MA spread around entry (proves golden cross, past-only) --")
    lo = max(0, e_idx - 3)
    hi = min(len(past) - 1, e_idx + 3)
    print(f"    {'idx':>5} {'date':>12} {'close':>14} "
          f"{'fast':>12} {'slow':>12} {'spread':>12} {'cross':>8}")
    for t in range(lo, hi + 1):
        sp = (ma_f[t] - ma_s[t]) if (ma_f is not None and not np.isnan(ma_f[t])
                                     and not np.isnan(ma_s[t])) else float("nan")
        sp_prev = (ma_f[t - 1] - ma_s[t - 1]) if (
            t > 0 and ma_f is not None and not np.isnan(ma_f[t - 1])
            and not np.isnan(ma_s[t - 1])) else float("nan")
        is_golden = (not np.isnan(sp) and not np.isnan(sp_prev)
                     and sp_prev <= 0 and sp > 0)
        fa = f"{ma_f[t]:.6f}" if (ma_f is not None and not np.isnan(ma_f[t])) else "nan"
        sl = f"{ma_s[t]:.6f}" if (ma_s is not None and not np.isnan(ma_s[t])) else "nan"
        spx = f"{sp:.6f}" if not np.isnan(sp) else "nan"
        mark = "GOLDEN" if is_golden else ("<-D" if t == d_idx else "")
        flag = "ENTRY" if t == e_idx else mark
        print(f"    {t:>5} {str(dates[t]):>12} {past[t]:>14.6f} "
              f"{fa:>12} {sl:>12} {spx:>12} {flag:>8}")
    print(f"  manual return    : (close[D]-close[entry])/close[entry] = "
          f"({cD:.8f}-{c_entry:.8f})/{c_entry:.8f} = {manual:.6f}")
    print(f"  engine captured  : {best['captured_return']:.6f}")
    print(f"  match (<1e-9)    : {abs(manual - best['captured_return']) < 1e-9}")
    print(f"  perfect_return   : {best['perfect_return']:.6f}  "
          f"capture_rate={best['capture_rate']:.6f}")


def main():
    ap = argparse.ArgumentParser(description=HINDSIGHT_LABEL)
    ap.add_argument("--date", required=True, help="query day D, YYYY-MM-DD")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--indicator", default="ma",
                    help=f"one of {sorted(INDICATOR_REGISTRY)}")
    ap.add_argument("--cadence", default="1d",
                    help="1d or any other cadence (4h/1h/dollar/dib/range/runs_*/...)")
    ap.add_argument("--resolution", default="native",
                    choices=["native", "daily"],
                    help="native (DEFAULT): analyze the cadence's own bars "
                         "(genuine multi-timeframe); daily: aggregate to a daily "
                         "close (back-compat, collapses all cadences to 1d)")
    ap.add_argument("--lookback", type=int, default=30,
                    help="trailing-return ranking window AND entry-bound + "
                         "perfect-entry window")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--top-pct", type=float, default=None,
                    help="if set, top_n = ceil(top_pct * universe_size)")
    ap.add_argument("--validity-windows", default="180,365",
                    help="comma list of validity windows (days), tried in order")
    ap.add_argument("--driver", default="rolling_validity",
                    choices=["rolling_validity", "bounded_oneshot"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--reconcile", action="store_true",
                    help="also print a hand reconciliation for the rank-1 asset")
    args = ap.parse_args()

    vws = tuple(int(x.strip()) for x in args.validity_windows.split(",") if x.strip())

    engine = OracleEngine()
    table = engine.oracle(
        args.date, universe=args.universe, indicator=args.indicator,
        cadence=args.cadence, lookback_days=args.lookback, top_n=args.top_n,
        top_pct=args.top_pct, validity_windows=vws, driver=args.driver,
        resolution=args.resolution)

    print("=" * 80)
    print(HINDSIGHT_LABEL)
    print(f"ORACLE-ENGINE date={args.date} universe={args.universe} "
          f"indicator={args.indicator} cadence={args.cadence} "
          f"resolution={args.resolution}")
    print(f"lookback={args.lookback}d top_n={args.top_n} "
          f"validity_windows={list(vws)} driver={args.driver}")
    print("=" * 80)
    if table.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return
    _print_table_ascii(table, max_rows=args.top_n)

    out = args.out or str(PROJECT_ROOT / "runs" / "oracle" /
                          f"engine_{args.date}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out)
    print(f"\nwrote: {out}")

    if args.reconcile:
        top_sym = table["sym"][0]
        _reconcile_one(engine, top_sym, args.date, args.cadence, args.lookback,
                       args.indicator, vws, args.driver)


if __name__ == "__main__":
    main()
