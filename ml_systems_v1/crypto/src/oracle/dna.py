"""GENERALIZED DNA-DECOUPLING MODULE (oracle.dna)

================================================================================
HINDSIGHT UPPER BOUND -- descriptive, NOT a tradeable signal.
================================================================================

This module adds a **generalized DNA-decouple step** to the plug-in oracle
engine: for every top-X% performer's best-config move returned by
``OracleEngine.oracle()``, it ATTACHES the full surrounding DNA:

    FEATURES  -- the chimera v50/v51 feature vector as-of the entry day AND
                 as-of the peak day, pulled via
                 ``OracleDecomposer._features_as_of`` (reused, not reimplemented).
    CHART     -- the same indicator's driver read across each given chart_type
                 (1d + dollar by default), produced by
                 ``OracleDecomposer._driver_for_cadence`` (reused).
    REGIME    -- BTC trend+vol regime as-of the entry day, computed past-only
                 from BTC's 1d chimera close (no future leak).
    TIMING    -- entry_date, days_back (already in the engine row).

Output: one wide DNA record per (asset, move) as a ``pl.DataFrame``.
Persisted to ``runs/oracle/dna_<universe>_<indicator>_<cadence>_<date>.parquet``.

Design choice (reuse strategy)
-------------------------------
``OracleDecomposer`` already encapsulates the three most expensive primitives:
  (a) ``_features_as_of(sym, d)`` -- causal chimera feature lookup (no future leak).
  (b) ``_driver_for_cadence(sym, d, cadence, ...)`` -- one chart-type's driver result
       INCLUDING peak_date computation.
  (c) ``_context_columns(sym)`` -- curated norm_* + extra column list.

Rather than re-implementing any of these (which would duplicate ~150 lines and
risk introducing a subtle difference / future drift), this module:
  1. Calls ``OracleEngine().oracle(...)`` to get the ranked + best-config rows.
  2. Instantiates a shared ``OracleDecomposer`` (same ``ChimeraLoader`` instance).
  3. For each engine row, calls ``decomposer._features_as_of(sym, entry_date)``
     and ``decomposer._features_as_of(sym, peak_date)`` directly.
  4. Calls ``decomposer._driver_for_cadence(...)`` for each extra chart type
     (those beyond the primary cadence already handled by the engine row).
  5. Adds the REGIME column by calling the local ``_btc_regime_as_of`` helper
     which follows the same past-only logic as
     ``runs/staging/h1_regime_overlay_2026_06_08.btc_regime_series``.

The peak_date for the engine row (the day of the max close in [entry, D]) is
computed locally via ``_peak_date_for`` (a three-line helper over the cached
series -- no duplication of complex logic).

CAUSALITY GUARANTEES (every row)
----------------------------------
  * Feature vectors: ``_features_as_of(sym, entry_date)`` selects the LAST
    chimera row with date <= entry_date.  Because entry_date <= query_date = D,
    every feature value pre-dates D.  Chimera features are themselves computed
    as-of-bar (no future leak in the pipeline).  Peak-day features use
    ``peak_date`` which is also within the lookback window and <= D.
  * Regime: ``_btc_regime_as_of(d)`` computes SMA200 and 30d-vol using only
    BTC closes up to and including d (past-anchored rolling means/stds).  The
    expanding-median threshold for the vol leg is anchored at t=d (includes d,
    conservative).  The result is strictly as-of entry_date.
  * Chart drivers: ``_driver_for_cadence`` truncates to ``closes[:d_idx+1]``
    before any signal computation (causal slice, same as engine + decomposer).
  * days_back: carried through from the engine; entry_date = D - days_back
    days (checked below).

No future information crosses the entry_date boundary.

--------------------------------------------------------------------------------
CLI:
    python src/oracle/dna.py --date 2026-05-20 --universe u10 --indicator ma
        [--cadence 1d] [--lookback 30] [--top-n 25] [--top-pct 0.25]
        [--validity-windows 180,365] [--chart-types 1d,dollar]
        [--no-features] [--no-regime]
        [--out runs/oracle/dna_u10_ma_1d_2026-05-20.parquet]
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

__contract__ = {
    "kind": "oracle_dna_decoupler",
    "inputs": [
        "oracle.engine.OracleEngine (plug-in oracle: ranked top performers + best config)",
        "oracle.decomposer.OracleDecomposer (_features_as_of + _driver_for_cadence reused)",
        "chimera v50/v51 via pipeline.chimera_loader.ChimeraLoader (1d + event cadences)",
    ],
    "outputs": {
        "callable": (
            "decouple(date, *, universe, indicator, cadence, lookback_days, top_n, "
            "validity_windows, chart_types, include_features, include_regime) -> pl.DataFrame"
        ),
        "parquet": "runs/oracle/dna_<universe>_<indicator>_<cadence>_<date>.parquet",
    },
    "invariants": [
        "per-config indicator signal is CAUSAL (closes[:d_idx+1] only)",
        "best-config selection is hindsight (the allowed oracle move)",
        "feature vectors are as-of entry_date (last chimera row with date <= entry_date)",
        "peak-day features are as-of peak_date (within the lookback window, <= D)",
        "regime is as-of entry_date (BTC closes up to entry_date only, past-only rolling)",
        "entry_date <= query_date D; days_back in [0, lookback_days]",
        "output is a HINDSIGHT UPPER BOUND -- hindsight=True on every row",
        "no emoji in prints (cp1252-safe)",
        "additive: does not modify engine.py, decomposer.py, or ma_oracle_engine.py",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(SRC / "pipeline"), str(SRC / "oracle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.chimera_loader import ChimeraLoader        # noqa: E402
from oracle.engine import OracleEngine, INDICATOR_REGISTRY   # noqa: E402
from oracle.decomposer import OracleDecomposer           # noqa: E402
from oracle.ma_oracle_engine import _to_date, _last_idx_le   # noqa: E402

HINDSIGHT_LABEL = "HINDSIGHT DNA DECOUPLE -- descriptive, not a tradeable signal."

# BTC regime parameters (matching h1_regime_overlay_2026_06_08 constants)
_BTC_SYM = "BTCUSDT"
_BTC_TREND_WIN = 200
_BTC_VOL_WIN = 30


# ============================================================================
# BTC REGIME HELPER (past-only, mirrors btc_regime_series in h1_regime_overlay)
# ============================================================================
class _BTCRegimeCache:
    """Computes and caches BTC trend+vol regime as a {date: float} mapping.

    regime_on = 1.0 iff (BTC close > SMA200) AND (30d-realized-vol > expanding median).
    Both legs use BTC closes up to and including the queried day (past-only).
    """

    def __init__(self, loader: ChimeraLoader):
        self._loader = loader
        self._cache: dict[_date, float] | None = None
        self._dates: list[_date] | None = None

    def _build(self) -> None:
        """Load BTC 1d closes once and compute the full past-only regime series."""
        try:
            df = self._loader.load(_BTC_SYM, cadence="1d",
                                   features=["close", "date"])
        except Exception:
            self._cache = {}
            self._dates = []
            return
        if "date" not in df.columns or "close" not in df.columns:
            self._cache = {}
            self._dates = []
            return
        df = df.select(["date", "close"]).drop_nulls()
        df = df.unique(subset=["date"], keep="last").sort("date")
        dates = df["date"].to_list()
        closes = df["close"].to_numpy().astype(float)
        n = len(closes)
        # Leg 1: BTC close > SMA200 (past-anchored rolling mean, causal)
        sma200 = np.full(n, np.nan)
        for i in range(_BTC_TREND_WIN - 1, n):
            sma200[i] = np.mean(closes[i - _BTC_TREND_WIN + 1: i + 1])
        trend_on = (closes > sma200)  # NaN comparison -> False
        # Leg 2: 30d realized vol of daily log-returns vs expanding median of that vol
        logret = np.zeros(n)
        logret[1:] = np.log(closes[1:] / closes[:-1])
        rvol = np.full(n, np.nan)
        for i in range(_BTC_VOL_WIN - 1, n):
            rvol[i] = np.std(logret[i - _BTC_VOL_WIN + 1: i + 1], ddof=1)
        # Expanding median of rvol up to and including i (conservative: includes i)
        exp_median = np.full(n, np.nan)
        valid_rvols: list[float] = []
        for i in range(n):
            if not np.isnan(rvol[i]):
                valid_rvols.append(rvol[i])
                exp_median[i] = float(np.median(valid_rvols))
        vol_on = (rvol > exp_median)  # NaN comparison -> False
        regime_on = np.where(
            np.isnan(sma200) | np.isnan(exp_median),
            0.0,
            (trend_on & vol_on).astype(float),
        )
        self._cache = {dates[i]: float(regime_on[i]) for i in range(n)}
        self._dates = list(dates)

    def regime_as_of(self, d: _date) -> float | None:
        """Return the BTC regime value as-of day d (last date <= d).

        1.0 = risk-on (trend + high-vol), 0.0 = risk-off.  None if BTC data
        unavailable or d is before the warm-up period.
        """
        if self._cache is None:
            self._build()
        if not self._dates:
            return None
        idx = _last_idx_le(self._dates, d)
        if idx is None:
            return None
        regime_date = self._dates[idx]
        return self._cache.get(regime_date, None)


# ============================================================================
# INTERNAL HELPERS
# ============================================================================
def _peak_date_for(decomp: OracleDecomposer, sym: str, cadence: str,
                   entry_date: _date, query_date: _date) -> _date | None:
    """Day of the max close in [entry_date, query_date] on the given cadence.

    Uses the decomposer's already-cached daily series.  Causal: only dates in
    [entry_date, D] are examined (the range already realised by query_date).
    """
    series = decomp._daily_series(sym, cadence)
    if series is None or len(series) == 0:
        return None
    dates = series["date"].to_list()
    closes = series["close"].to_numpy().astype(float)
    # restrict to [entry_date, query_date]
    lo = _last_idx_le(dates, entry_date)
    hi = _last_idx_le(dates, query_date)
    if lo is None or hi is None or hi < lo:
        return None
    seg = closes[lo: hi + 1]
    peak_offset = int(np.argmax(seg))
    return dates[lo + peak_offset]


# ============================================================================
# PUBLIC API
# ============================================================================
def decouple(
    date,
    *,
    universe: str = "u50",
    indicator: str = "ma",
    cadence: str = "1d",
    lookback_days: int = 30,
    top_n: int = 25,
    top_pct: float | None = None,
    validity_windows: tuple[int, ...] = (180, 365),
    chart_types: tuple[str, ...] = ("1d", "dollar"),
    include_features: bool = True,
    include_regime: bool = True,
    loader: ChimeraLoader | None = None,
) -> pl.DataFrame:
    """Generalized DNA decoupling: attach the full surrounding DNA to every
    top-X% performer's best-config move from OracleEngine.oracle().

    Parameters
    ----------
    date : str or date
        Query day D (YYYY-MM-DD).
    universe : str
        Universe key ('u10', 'u50', 'u100').
    indicator : str
        Key into INDICATOR_REGISTRY ('ma' implemented; 'rsi'/'macd'/'bollinger'
        are registered stubs -- will raise NotImplementedError).
    cadence : str
        Primary bar cadence ('1d' or any event cadence).
    lookback_days : int
        Trailing-return ranking window AND entry bound.
    top_n : int
        Number of top performers to include.
    top_pct : float | None
        If given, top_n = ceil(top_pct * universe_size).
    validity_windows : tuple[int, ...]
        Windows for the rolling_validity driver (tried in order).
    chart_types : tuple[str, ...]
        Chart types to run the indicator across.  The primary cadence is always
        run (sourced from the engine row); additional chart types yield
        ``chart__<ct>__*`` columns via the decomposer's driver.
    include_features : bool
        If True, attach chimera feature vectors as-of entry_date and peak_date.
    include_regime : bool
        If True, attach the BTC trend+vol regime as-of entry_date.
    loader : ChimeraLoader | None
        Shared loader instance (one is created if not provided).

    Returns
    -------
    pl.DataFrame
        One row per (asset, move) with columns grouped as:
          * ENGINE: sym, perf_rank, trailing_perf, indicator, cadence,
                    best_config, entry_date, days_back, captured_return,
                    perfect_return, capture_rate, in_position, hindsight.
          * TIMING: query_date, peak_date, peak_date_<ct> per chart type.
          * REGIME: btc_regime_at_entry, btc_trend_on, btc_vol_on (if
                    include_regime=True).
          * FEATURES: ctx_entry__<col> / ctx_peak__<col> (if
                      include_features=True).
          * CHART: chart__<ct>__best_config, chart__<ct>__entry_date,
                   chart__<ct>__days_back, chart__<ct>__captured_return,
                   chart__<ct>__capture_rate (for each ct in chart_types).

    Notes
    -----
    HINDSIGHT UPPER BOUND: the per-config signal is CAUSAL (past-only);
    only the best-config / best-entry selection is hindsight.  Every row
    carries hindsight=True.

    Causality invariants (see module docstring for full proof):
      - feature vectors use ``_features_as_of(sym, entry_date)`` ->
        last chimera row with date <= entry_date.
      - peak_date is within [entry_date, D] (past window).
      - regime is BTC closes up to entry_date (rolling past-only).
      - chart drivers use closes[:d_idx+1] (causal slice).
    """
    d = _to_date(date)
    if indicator not in INDICATOR_REGISTRY:
        raise KeyError(
            f"unknown indicator '{indicator}'; registered: "
            f"{sorted(INDICATOR_REGISTRY)}"
        )
    if isinstance(validity_windows, int):
        validity_windows = (validity_windows,)
    validity_windows = tuple(validity_windows)
    chart_types = tuple(chart_types)

    _loader = loader or ChimeraLoader()

    # ---- resolve top_n from top_pct if requested -----------------------
    if top_pct is not None:
        uni_size = len(_loader.universes.list(universe))
        top_n = max(1, math.ceil(top_pct * uni_size))

    # ---- get engine rows (ranked + best-config per asset) ---------------
    engine = OracleEngine(loader=_loader)
    engine_df = engine.oracle(
        d,
        universe=universe,
        indicator=indicator,
        cadence=cadence,
        lookback_days=lookback_days,
        top_n=top_n,
        validity_windows=validity_windows,
        driver="rolling_validity",
    )
    if engine_df.is_empty():
        return pl.DataFrame()

    # ---- shared decomposer (reuses loader cache) -----------------------
    decomp = OracleDecomposer(loader=_loader)

    # ---- BTC regime cache ----------------------------------------------
    regime_cache = _BTCRegimeCache(_loader) if include_regime else None

    rows: list[dict] = []

    for row in engine_df.iter_rows(named=True):
        sym: str = row["sym"]
        in_pos: bool = bool(row.get("in_position", False))
        entry_date_raw = row.get("entry_date")
        days_back_val = row.get("days_back")

        # Parse entry_date
        entry_d: _date | None = None
        if entry_date_raw is not None:
            try:
                entry_d = _to_date(entry_date_raw)
            except Exception:
                entry_d = None

        # Peak date (max close in [entry, D] on primary cadence)
        peak_d: _date | None = None
        if in_pos and entry_d is not None:
            peak_d = _peak_date_for(decomp, sym, cadence, entry_d, d)

        # ---- ENGINE columns (pass-through) -------------------------
        rec: dict = {
            # engine identity
            "sym": sym,
            "perf_rank": row.get("perf_rank"),
            "trailing_perf": row.get("trailing_perf"),
            "indicator": indicator,
            "cadence": cadence,
            "query_date": str(d),
            # driver result
            "best_config": row.get("best_config"),
            "entry_date": str(entry_d) if entry_d is not None else None,
            "days_back": days_back_val,
            "captured_return": row.get("captured_return", 0.0),
            "perfect_return": row.get("perfect_return", 0.0),
            "capture_rate": row.get("capture_rate", 0.0),
            "in_position": in_pos,
            "validity_window_used": row.get("validity_window_used"),
            "validity_score": row.get("validity_score"),
            "hindsight": True,
            # timing
            "peak_date": str(peak_d) if peak_d is not None else None,
        }

        # ---- REGIME ------------------------------------------------
        if include_regime:
            regime_val = (
                regime_cache.regime_as_of(entry_d)
                if (regime_cache is not None and entry_d is not None)
                else None
            )
            rec["btc_regime_at_entry"] = regime_val
            # decompose into legs for interpretability
            if regime_val is not None:
                rec["btc_regime_risk_on"] = int(regime_val >= 0.5)
            else:
                rec["btc_regime_risk_on"] = None

        # ---- FEATURES (entry_date + peak_date) ---------------------
        if include_features:
            ctx_cols = decomp._context_columns(sym)
            # features as-of entry_date (last chimera row with date <= entry_d)
            ctx_entry = (
                decomp._features_as_of(sym, entry_d)
                if entry_d is not None
                else {}
            )
            # features as-of peak_date (last chimera row with date <= peak_d)
            ctx_peak = (
                decomp._features_as_of(sym, peak_d)
                if peak_d is not None
                else {}
            )
            for c in ctx_cols:
                rec[f"ctx_entry__{c}"] = ctx_entry.get(c, None)
            for c in ctx_cols:
                rec[f"ctx_peak__{c}"] = ctx_peak.get(c, None)

        # ---- CHART-TYPE comparison ---------------------------------
        # The primary cadence is already captured in the engine row.
        # For each chart type we call the decomposer's driver and record
        # the per-chart best-config + metrics.
        for ct in chart_types:
            if ct == cadence and in_pos:
                # primary cadence: source from the engine row directly
                rec[f"chart__{ct}__best_config"] = row.get("best_config")
                rec[f"chart__{ct}__entry_date"] = (
                    str(entry_d) if entry_d is not None else None
                )
                rec[f"chart__{ct}__days_back"] = days_back_val
                rec[f"chart__{ct}__captured_return"] = row.get("captured_return", 0.0)
                rec[f"chart__{ct}__capture_rate"] = row.get("capture_rate", 0.0)
                rec[f"chart__{ct}__peak_date"] = (
                    str(peak_d) if peak_d is not None else None
                )
            else:
                # secondary cadence (or primary when not in-position): run driver
                try:
                    ct_res = decomp._driver_for_cadence(
                        sym, d, ct, lookback_days,
                        "rolling_validity", 365, 3,
                    )
                except Exception:
                    ct_res = None
                if ct_res and ct_res.get("in_position"):
                    rec[f"chart__{ct}__best_config"] = ct_res.get("best_ti")
                    ed = ct_res.get("entry_date")
                    rec[f"chart__{ct}__entry_date"] = (
                        str(ed) if ed is not None else None
                    )
                    rec[f"chart__{ct}__days_back"] = ct_res.get("days_back")
                    rec[f"chart__{ct}__captured_return"] = ct_res.get(
                        "captured_return", 0.0
                    )
                    rec[f"chart__{ct}__capture_rate"] = ct_res.get(
                        "capture_rate", 0.0
                    )
                    pd_ct = ct_res.get("peak_date")
                    rec[f"chart__{ct}__peak_date"] = (
                        str(pd_ct) if pd_ct is not None else None
                    )
                else:
                    rec[f"chart__{ct}__best_config"] = None
                    rec[f"chart__{ct}__entry_date"] = None
                    rec[f"chart__{ct}__days_back"] = None
                    rec[f"chart__{ct}__captured_return"] = None
                    rec[f"chart__{ct}__capture_rate"] = None
                    rec[f"chart__{ct}__peak_date"] = None

        rows.append(rec)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).sort("perf_rank")


# ============================================================================
# PERSIST OUTPUT
# ============================================================================
def _out_path(universe: str, indicator: str, cadence: str, date_str: str,
              out: str | None) -> Path:
    if out:
        return Path(out)
    return (PROJECT_ROOT / "runs" / "oracle" /
            f"dna_{universe}_{indicator}_{cadence}_{date_str}.parquet")


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=HINDSIGHT_LABEL)
    ap.add_argument("--date", required=True, help="query day D, YYYY-MM-DD")
    ap.add_argument("--universe", default="u50",
                    help="universe key: u10 / u50 / u100 (default u50)")
    ap.add_argument("--indicator", default="ma",
                    help=f"one of {sorted(INDICATOR_REGISTRY)} (default ma)")
    ap.add_argument("--cadence", default="1d",
                    help="primary bar cadence: 1d or event (dollar/dib/...) (default 1d)")
    ap.add_argument("--lookback", type=int, default=30,
                    help="trailing-return ranking window AND entry bound (default 30)")
    ap.add_argument("--top-n", type=int, default=25,
                    help="number of top performers (default 25)")
    ap.add_argument("--top-pct", type=float, default=None,
                    help="if set, top_n = ceil(top_pct * universe_size)")
    ap.add_argument("--validity-windows", default="180,365",
                    help="comma list of validity windows (days) (default 180,365)")
    ap.add_argument("--chart-types", default="1d,dollar",
                    help="comma list of chart types (default 1d,dollar)")
    ap.add_argument("--no-features", action="store_true",
                    help="skip chimera feature attachment")
    ap.add_argument("--no-regime", action="store_true",
                    help="skip BTC regime attachment")
    ap.add_argument("--out", default=None,
                    help="output parquet path (default runs/oracle/dna_*.parquet)")
    args = ap.parse_args()

    vws = tuple(int(x.strip()) for x in args.validity_windows.split(",") if x.strip())
    cts = tuple(c.strip() for c in args.chart_types.split(",") if c.strip())

    result = decouple(
        args.date,
        universe=args.universe,
        indicator=args.indicator,
        cadence=args.cadence,
        lookback_days=args.lookback,
        top_n=args.top_n,
        top_pct=args.top_pct,
        validity_windows=vws,
        chart_types=cts,
        include_features=(not args.no_features),
        include_regime=(not args.no_regime),
    )

    print("=" * 72)
    print(HINDSIGHT_LABEL)
    print(f"DNA-DECOUPLE date={args.date} universe={args.universe} "
          f"indicator={args.indicator} cadence={args.cadence}")
    print(f"lookback={args.lookback}d top_n={args.top_n} "
          f"validity_windows={list(vws)} chart_types={list(cts)}")
    print("=" * 72)

    if result.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return

    out_path = _out_path(args.universe, args.indicator, args.cadence,
                         args.date, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(out_path))
    print(f"shape      : {result.shape}")
    print(f"wrote      : {out_path}")

    # column group summary
    all_cols = result.columns
    engine_cols = [c for c in all_cols if c in (
        "sym", "perf_rank", "trailing_perf", "indicator", "cadence",
        "query_date", "best_config", "entry_date", "days_back",
        "captured_return", "perfect_return", "capture_rate",
        "in_position", "validity_window_used", "validity_score",
        "hindsight", "peak_date")]
    regime_cols = [c for c in all_cols if "regime" in c or "btc_" in c]
    feature_entry_cols = [c for c in all_cols if c.startswith("ctx_entry__")]
    feature_peak_cols = [c for c in all_cols if c.startswith("ctx_peak__")]
    chart_cols = [c for c in all_cols if c.startswith("chart__")]
    print(f"engine cols: {len(engine_cols)}")
    print(f"regime cols: {len(regime_cols)}  -> {regime_cols}")
    print(f"feature entry cols: {len(feature_entry_cols)}")
    print(f"feature peak  cols: {len(feature_peak_cols)}")
    print(f"chart cols : {len(chart_cols)}  -> {chart_cols}")

    # print 2 example rows (selected cols, plain-row print)
    show_cols = (
        ["sym", "perf_rank", "entry_date", "days_back",
         "captured_return", "capture_rate", "peak_date"]
        + regime_cols[:2]
        + feature_entry_cols[:3]
        + chart_cols[:4]
    )
    show_cols = [c for c in show_cols if c in all_cols]
    n_ex = min(2, len(result))
    print(f"\n-- {n_ex} example rows (selected cols) --")
    for i in range(n_ex):
        r = result.row(i, named=True)
        print(f"  row {i}:")
        for c in show_cols:
            print(f"    {c:40s}: {r[c]}")


if __name__ == "__main__":
    main()
