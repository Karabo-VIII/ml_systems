"""MA ADAPTIVE ENGINE -- a REALIZABLE, forward-testable (instrument, TI-config) CHOOSER.

================================================================================
REALIZABLE / PAST-ONLY -- the honest forward analog of the hindsight oracle.
================================================================================

This is the counterpart to ``oracle/engine.py``'s ``OracleEngine`` and the
``oracle/decomposer.py``'s driver.  The ORACLE is HINDSIGHT: it knows the
realized move to D and selects the config that maximized that realized capture
(the allowed oracle move).  This ADAPTIVE engine is REALIZABLE: at a decision
date D it uses ONLY past-only information (closes[:D] and chimera features
as-of <= D).  It does NOT know the future move.

THE KEY DISTINCTION (do not violate):
  - ORACLE picks the config by REALIZED capture (close[D]/close[entry]-1) -- a
    number unknown until D arrives -- and reports a hindsight upper bound.
  - ADAPTIVE picks the config by its PAST-ONLY ROLLING-VALIDITY: how well that
    config captured COMPLETED golden->death round-trips BEFORE D (the same
    ``_config_rolling_validity`` the decomposer uses).  This score is the model's
    forward CONFIDENCE in a config; it is NOT the realized capture.  The chooser
    then acts ONLY if that config is IN-POSITION at D (a live golden cross with
    no later death cross by D -- a forward entry signal it would actually trade).

A leak here would invalidate the whole comparison, so every selection input is
explicitly past-only:
  * the rolling-validity score sums COMPLETED round-trips with exit index <= D;
  * ``in_position_at_D`` is derived from crosses over closes[:d_idx+1] only;
  * the momentum pre-rank is a trailing return ending at D (close[D]/close[D-k]-1).
  * the BTC regime gate (``regime_cond``) and the market-state read
    (``state_cond``) are both past-only by construction (see those mechanisms).

OUTPUT COLUMNS mirror the oracle's so a side-by-side compare is apples-to-apples
MINUS any hindsight field:
    sym, chosen_config, mechanism, validity_window, chosen_score,
    in_position_at_D, momentum_rank, cadence, date
``chosen_score`` is the rolling-validity (past-only), NOT realized capture.
There is deliberately NO ``captured_return`` / ``capture_rate`` / ``perfect_return``
column -- those are realized (hindsight) quantities the chooser cannot know at D.

REUSE (cited):
  * ``oracle.engine.INDICATOR_REGISTRY`` / ``MAIndicator`` -- the indicator
    plug-in + the canonical 16-config MA grid + the causal ``signal()``.
  * ``oracle.engine.OracleEngine._config_rolling_validity`` -- the PAST-ONLY
    rolling-validity scorer (ported here as the realizable selection driver).
  * ``oracle.ma_oracle_engine._sma/_ema/_crosses/_to_date/_last_idx_le/
    _print_table_ascii`` -- verified causal primitives + helpers.
  * ``runs/staging/h1_regime_overlay_2026_06_08.py:btc_regime_series`` -- the
    past-only BTC regime (reused verbatim for the ``regime_cond`` mechanism).
  * ``firm.market_state.compute_state`` -- the cross-sectional state read for
    the ``state_cond`` mechanism.
  * ``pipeline.chimera_loader.ChimeraLoader`` -- data access (any cadence).

--------------------------------------------------------------------------------
CLI:
    python src/oracle/adaptive.py --date 2026-05-20 [--universe u50]
        [--cadence 1d] [--validity-window 365] [--mechanism rolling_validity]
        [--indicator ma] [--lookback 30] [--top-n 25]
        [--all]   (run choose_all: every mechanism side by side)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

__contract__ = {
    "kind": "ma_adaptive_chooser",
    "inputs": [
        "oracle.engine.INDICATOR_REGISTRY / MAIndicator (causal signal + 16-cfg grid)",
        "oracle.engine.OracleEngine._config_rolling_validity (past-only validity scorer)",
        "oracle.ma_oracle_engine primitives (_sma/_ema/_crosses/_last_idx_le/_to_date)",
        "runs/staging/h1_regime_overlay_2026_06_08.py:btc_regime_series (past-only BTC regime)",
        "firm.market_state.compute_state (cross-sectional state read)",
        "chimera via pipeline.chimera_loader.ChimeraLoader (any cadence)",
    ],
    "outputs": {
        "callable": "AdaptiveChooser.choose(date, *, universe, cadence, "
                    "validity_window, mechanism, indicator, lookback_days, top_n) "
                    "-> pl.DataFrame",
        "callable_all": "AdaptiveChooser.choose_all(date, ...) -> dict[mechanism]->pl.DataFrame",
    },
    "invariants": [
        "REALIZABLE / PAST-ONLY: every selection input uses only closes[:d_idx+1] "
        "and chimera features as-of <= D; NO future leak",
        "chosen_config is selected by PAST-ONLY rolling-validity, NOT realized capture",
        "chosen_score IS the rolling-validity (mean capture_rate of COMPLETED "
        "round-trips with exit <= D), NOT the realized capture to D",
        "in_position_at_D is a FORWARD entry signal: a live golden cross with no "
        "later death cross by D (derived from closes[:d_idx+1] only)",
        "NO hindsight field (no captured_return / capture_rate / perfect_return)",
        "momentum_rank is a trailing return ending at D (past-only pre-rank)",
        "regime_cond / state_cond gates are past-only by construction",
        "no emoji in prints (cp1252)",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
# NOTE: do NOT add src/firm to sys.path -- it contains a module `pipeline.py` that
# would shadow the real `pipeline` package.  firm is imported via the `firm.` prefix
# (src is on the path), so no firm dir needs to be on sys.path.
for _p in (str(SRC), str(SRC / "pipeline"), str(SRC / "oracle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
# REUSE the verified causal primitives + helpers from v1.
from oracle.ma_oracle_engine import (  # noqa: E402
    _to_date, _last_idx_le, _sma, _ema, _crosses, _print_table_ascii,
)
# REUSE the indicator plug-in registry + the past-only rolling-validity scorer.
from oracle.engine import INDICATOR_REGISTRY, OracleEngine  # noqa: E402

REALIZABLE_LABEL = ("REALIZABLE / PAST-ONLY chooser -- forward-testable, "
                    "NOT hindsight (no future move known at D).")


# ---------------------------------------------------------------------------
# REUSE: btc_regime_series from the staged H1 overlay (loaded by path -- it lives
# under runs/staging which is not an importable package).  This is the SAME
# past-only BTC regime (trend / trend+vol) verified in the H1 run.
# ---------------------------------------------------------------------------
def _load_btc_regime_fn():
    """Import btc_regime_series from the staged H1 overlay module by file path.

    Returns the function (callable on a btc_df) or None if the staged file is
    absent.  We import by path because runs/staging is not a package; the
    function itself only depends on pandas/numpy (no heavy harness imports run
    at function-call time)."""
    src = PROJECT_ROOT / "runs" / "staging" / "h1_regime_overlay_2026_06_08.py"
    if not src.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_h1_regime_overlay", str(src))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "btc_regime_series", None)
    except Exception:
        return None


# ============================================================================
# THE ADAPTIVE CHOOSER
# ============================================================================
class AdaptiveChooser:
    """REALIZABLE forward chooser of (instrument, TI-config).

    At a decision date D, for the candidate instruments (pre-ranked by a
    past-only trailing-return momentum), pick -- per asset, PAST-ONLY -- the best
    config by the selected mechanism and whether it is in-position at D.  The
    result is directly comparable to ``OracleEngine.oracle`` MINUS the hindsight
    capture fields: the oracle's "right answer" vs the model's realizable pick.
    """

    realizable = True  # marker: every output of this engine is a forward, past-only pick.

    def __init__(self, loader: ChimeraLoader | None = None,
                 min_valid_trades: int = 3):
        self.loader = loader or ChimeraLoader()
        # Reuse OracleEngine for: ranking (rank_top_performers), the daily-series
        # cadence aggregation (_daily_series), and the PAST-ONLY
        # _config_rolling_validity scorer.  We do NOT call its hindsight oracle().
        self.engine = OracleEngine(self.loader)
        self.min_valid_trades = int(min_valid_trades)
        self._btc_regime_fn = _load_btc_regime_fn()
        # caches
        self._regime_cache: dict[str, dict] = {}    # mode -> {date: regime_on}
        self._btc_df_cache = None

    # ---- past-only per-asset config selection (the realizable core) ------
    def _choose_config_for_asset(self, sym, cadence, d, validity_window,
                                 lookback_days, indicator):
        """PAST-ONLY: pick the best config for one asset at one cadence by
        rolling-validity, and report whether it is IN-POSITION at D.

        Returns a dict or None (no data / no usable config).  All inputs are
        closes[:d_idx+1] only -- no future leak.  The selection NEVER uses the
        realized capture to D; it uses only the rolling-validity score (completed
        round-trips with exit <= D) and the in-position-at-D forward signal.
        """
        series = self.engine._daily_series(sym, cadence)
        if series is None or len(series) == 0:
            return None
        dates = series["date"].to_list()
        closes = series["close"].to_numpy().astype(float)
        d_idx = _last_idx_le(dates, d)
        if d_idx is None:
            return None
        past = closes[:d_idx + 1]            # CAUSAL slice -- nothing past D visible

        ind = INDICATOR_REGISTRY[indicator]
        scored = []   # (rolling_validity, in_position_at_D, n_trades, cfg, last_golden_idx, has_open_golden)
        for cfg in ind.config_grid():
            sig = ind.signal(dates, past, cfg)
            golden = sig.get("golden_idx", [])
            death = sig.get("death_idx", [])
            # PAST-ONLY rolling-validity score for this config (completed round-trips
            # with entry in [D - validity_window, D] and exit <= D).  REUSE the
            # OracleEngine scorer -- identical to the decomposer's logic.
            rv, n_tr = OracleEngine._config_rolling_validity(
                dates, past, golden, death, d, validity_window)
            # forward in-position-at-D signal: a live golden cross with no later death.
            in_pos = False
            last_g = None
            if golden:
                last_g = golden[-1]
                in_pos = not any(dx > last_g for dx in death)
            scored.append({
                "cfg": cfg, "rolling_validity": rv, "n_trades": n_tr,
                "in_position": bool(in_pos), "last_golden_idx": last_g,
                "golden": golden, "death": death,
            })

        # Rank configs by PAST-ONLY rolling-validity (higher = historically captures
        # this asset's moves better).  We require >= min_valid_trades completed
        # round-trips for the score to be trusted; configs below that are de-prioritized.
        def _key(c):
            rv = c["rolling_validity"]
            trusted = (rv is not None and c["n_trades"] >= self.min_valid_trades)
            return (1 if trusted else 0, rv if rv is not None else -1.0)

        scored.sort(key=_key, reverse=True)
        best = scored[0]
        # entry day for the chosen config's live golden (if in position)
        entry_date = None
        days_back = None
        if best["in_position"] and best["last_golden_idx"] is not None:
            entry_date = dates[best["last_golden_idx"]]
            days_back = (d - entry_date).days
            # bounded: only treat as an actionable forward entry if within lookback.
            if days_back > lookback_days:
                # config is technically open but its entry predates the action window;
                # we still surface it but mark it not actionable-at-D.
                best = dict(best)
                best["in_position"] = False
        return {
            "sym": sym,
            "cfg": best["cfg"],
            "rolling_validity": best["rolling_validity"],
            "n_trades": best["n_trades"],
            "in_position_at_D": bool(best["in_position"]),
            "entry_date": entry_date,
            "days_back": days_back,
            "resolved_day": dates[d_idx],
            "d_idx": d_idx,
        }

    # ---- regime gate (REUSE btc_regime_series) --------------------------
    def _btc_df(self):
        if self._btc_df_cache is not None:
            return self._btc_df_cache
        try:
            import pandas as pd
            g = self.loader.load("BTCUSDT", cadence="1d", features=["close", "date"])
            d = g.to_dict(as_series=False)
            df = pd.DataFrame({
                "date": pd.to_datetime(np.asarray(d["date"])),
                "close": np.asarray(d["close"], float),
            }).sort_values("date").reset_index(drop=True)
            self._btc_df_cache = df
        except Exception:
            self._btc_df_cache = None
        return self._btc_df_cache

    def _btc_regime_on(self, d: _date, mode: str = "trend_only") -> bool | None:
        """PAST-ONLY BTC risk-on/off at day D via the REUSED btc_regime_series.

        Returns True (risk-on), False (risk-off), or None if unavailable. The
        regime function uses only BTC closes <= each day (no future leak); we read
        the regime value on the last regime row with date <= D."""
        if self._btc_regime_fn is None:
            return None
        if mode not in self._regime_cache:
            btc = self._btc_df()
            if btc is None:
                self._regime_cache[mode] = {}
            else:
                import pandas as pd
                reg = self._btc_regime_fn(btc, mode=mode)
                m = {}
                for dt, on in zip(reg["date"].tolist(), reg["regime_on"].tolist()):
                    m[pd.Timestamp(dt).date()] = float(on)
                self._regime_cache[mode] = m
        m = self._regime_cache[mode]
        if not m:
            return None
        # last regime date <= D
        keys = sorted(k for k in m.keys() if k <= d)
        if not keys:
            return None
        return m[keys[-1]] > 0.5

    # ---- market-state read (REUSE firm.market_state.compute_state) ------
    def _market_state(self, d: _date, ranked, lookback_days: int):
        """PAST-ONLY cross-sectional market state at D from the ranked universe's
        trailing returns (already computed past-only by rank_top_performers).
        Returns a MarketState (favourability in [0,1]) or None."""
        try:
            from firm.market_state import compute_state
        except Exception:
            return None
        if not ranked:
            return None
        # trailing returns per asset = the rank perf (past-only, ends at D).
        returns = {sym: float(perf) for sym, perf in ranked}
        try:
            return compute_state(returns)
        except Exception:
            return None

    # ---- the public chooser ---------------------------------------------
    def choose(self, date, *, universe: str = "u50", cadence: str = "1d",
               validity_window: int = 365, mechanism: str = "rolling_validity",
               indicator: str = "ma", lookback_days: int = 30,
               top_n: int = 25) -> pl.DataFrame:
        """REALIZABLE forward picks as of `date` under the chosen mechanism.

        For the candidate instruments (pre-ranked by past-only trailing-return
        momentum), choose -- PAST-ONLY -- the best config by the mechanism's
        scoring and whether it is in-position at D (a forward entry signal).

        Returns one row per chosen instrument with columns:
            sym, chosen_config, mechanism, validity_window, chosen_score,
            in_position_at_D, momentum_rank, cadence, date
        NO hindsight field (no realized capture).  ``chosen_score`` IS the
        past-only rolling-validity, not the realized capture to D.

        mechanism: 'rolling_validity' (core) | 'regime_cond' | 'state_cond'.
        """
        d = _to_date(date)
        if indicator not in INDICATOR_REGISTRY:
            raise KeyError(
                f"unknown indicator '{indicator}'; registered: "
                f"{sorted(INDICATOR_REGISTRY)}")
        if mechanism not in MECHANISMS:
            raise ValueError(
                f"unknown mechanism '{mechanism}'; registered: {sorted(MECHANISMS)}")
        return MECHANISMS[mechanism](
            self, d, universe, cadence, validity_window, indicator,
            lookback_days, top_n)

    def choose_all(self, date, *, universe: str = "u50", cadence: str = "1d",
                   validity_window: int = 365, indicator: str = "ma",
                   lookback_days: int = 30, top_n: int = 25) -> dict:
        """Run EVERY mechanism for one date and return each one's picks side by
        side: {mechanism_name: pl.DataFrame}."""
        out = {}
        for name in MECHANISMS:
            out[name] = self.choose(
                date, universe=universe, cadence=cadence,
                validity_window=validity_window, mechanism=name,
                indicator=indicator, lookback_days=lookback_days, top_n=top_n)
        return out

    # ---- mechanism implementations --------------------------------------
    def _rank_and_pick(self, d, universe, cadence, validity_window, indicator,
                       lookback_days, top_n):
        """Shared core: past-only momentum pre-rank, then per-asset config pick.
        Returns (ranked, list[pick-dict]).  Used by all mechanisms."""
        # REUSE the verified past-only trailing-return ranking from v1.
        ranked = self.engine.ma_engine.rank_top_performers(
            d, universe, lookback_days, top_n)
        picks = []
        for rank, (sym, perf) in enumerate(ranked, start=1):
            sel = self._choose_config_for_asset(
                sym, cadence, d, validity_window, lookback_days, indicator)
            picks.append((rank, sym, perf, sel))
        return ranked, picks

    def _row(self, rank, sym, sel, mechanism, validity_window, indicator,
             cadence, d, *, forced_in_position=None, score_override=None,
             note=None):
        """Build one output row (oracle-comparable columns, minus hindsight)."""
        cfg_str = None
        score = None
        in_pos = False
        if sel is not None:
            cfg_str = _fmt_cfg(indicator, sel["cfg"])
            score = sel["rolling_validity"]
            in_pos = sel["in_position_at_D"]
        if score_override is not None:
            score = score_override
        if forced_in_position is not None:
            in_pos = forced_in_position
        row = {
            "sym": sym,
            "chosen_config": cfg_str,
            "mechanism": mechanism,
            "validity_window": validity_window,
            "chosen_score": (round(float(score), 6) if score is not None else None),
            "in_position_at_D": bool(in_pos),
            "momentum_rank": rank,
            "cadence": cadence,
            "date": str(d),
        }
        if note is not None:
            row["note"] = note
        return row

    def _mech_rolling_validity(self, d, universe, cadence, validity_window,
                               indicator, lookback_days, top_n):
        """CORE mechanism: pick the highest past-only rolling-validity config that
        is in-position at D; rank instruments by (in_position AND chosen_score).
        The honest forward analog of the oracle's driver."""
        ranked, picks = self._rank_and_pick(
            d, universe, cadence, validity_window, indicator, lookback_days, top_n)
        rows = [self._row(r, sym, sel, "rolling_validity", validity_window,
                          indicator, cadence, d)
                for (r, sym, perf, sel) in picks]
        return _sort_picks(_to_df(rows))

    def _mech_regime_cond(self, d, universe, cadence, validity_window,
                          indicator, lookback_days, top_n):
        """REGIME-gated: same as rolling_validity, but GATED by the past-only BTC
        regime.  When BTC is risk-OFF at D the chooser ABSTAINS (no position /
        cash) -- every pick is forced in_position_at_D=False with a 'regime_off'
        note.  Risk-on at D -> identical to rolling_validity."""
        regime_on = self._btc_regime_on(d, mode="trend_only")
        ranked, picks = self._rank_and_pick(
            d, universe, cadence, validity_window, indicator, lookback_days, top_n)
        rows = []
        for (r, sym, perf, sel) in picks:
            if regime_on is False:
                # ABSTAIN: BTC risk-off -> cash for this date (de-risk the cohort).
                rows.append(self._row(r, sym, sel, "regime_cond", validity_window,
                                      indicator, cadence, d,
                                      forced_in_position=False, note="regime_off_abstain"))
            elif regime_on is None:
                rows.append(self._row(r, sym, sel, "regime_cond", validity_window,
                                      indicator, cadence, d, note="regime_unavailable"))
            else:
                rows.append(self._row(r, sym, sel, "regime_cond", validity_window,
                                      indicator, cadence, d, note="regime_on"))
        return _sort_picks(_to_df(rows))

    def _mech_state_cond(self, d, universe, cadence, validity_window,
                         indicator, lookback_days, top_n):
        """STATE-conditioned: condition on the past-only cross-sectional market
        state (firm.market_state.compute_state on the ranked universe's trailing
        returns).  When favourability < 0.5 (risk-off tape) we ABSTAIN; otherwise
        the chooser KEEPS only the top ``ceil(top_n * favourability)`` momentum
        names in-position (scale exposure by tape favourability).  The state read
        is past-only (trailing returns end at D)."""
        import math
        ranked, picks = self._rank_and_pick(
            d, universe, cadence, validity_window, indicator, lookback_days, top_n)
        state = self._market_state(d, ranked, lookback_days)
        fav = state.favourability if state is not None else None
        rows = []
        if fav is None:
            for (r, sym, perf, sel) in picks:
                rows.append(self._row(r, sym, sel, "state_cond", validity_window,
                                      indicator, cadence, d, note="state_unavailable"))
            return _sort_picks(_to_df(rows))
        if fav < 0.5:
            # risk-off tape -> abstain entirely (favourability below neutral).
            for (r, sym, perf, sel) in picks:
                rows.append(self._row(r, sym, sel, "state_cond", validity_window,
                                      indicator, cadence, d, forced_in_position=False,
                                      note=f"state_riskoff_abstain(fav={fav})"))
            return _sort_picks(_to_df(rows))
        # favourable tape -> keep top-K in-position, where K scales with favourability.
        n_keep = max(1, math.ceil(len(picks) * fav))
        for i, (r, sym, perf, sel) in enumerate(picks):
            keep = (i < n_keep)
            forced = None if keep else False
            note = f"state_keep(fav={fav},K={n_keep})" if keep else f"state_trim(fav={fav})"
            rows.append(self._row(r, sym, sel, "state_cond", validity_window,
                                  indicator, cadence, d,
                                  forced_in_position=forced, note=note))
        return _sort_picks(_to_df(rows))


# ============================================================================
# mechanism registry -- "try them all"
# ============================================================================
MECHANISMS = {
    "rolling_validity": AdaptiveChooser._mech_rolling_validity,
    "regime_cond": AdaptiveChooser._mech_regime_cond,
    "state_cond": AdaptiveChooser._mech_state_cond,
}


# ============================================================================
# helpers
# ============================================================================
def _fmt_cfg(indicator: str, cfg: dict) -> str:
    """Render a config dict compactly.  For 'ma' this matches the oracle's
    'FAM(fast,slow)' string so picks are directly comparable to the oracle."""
    if indicator == "ma":
        return f"{cfg['family']}({cfg['fast']},{cfg['slow']})"
    return ",".join(f"{k}={v}" for k, v in cfg.items())


def _to_df(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def _sort_picks(df: pl.DataFrame) -> pl.DataFrame:
    """Rank instruments by (in_position_at_D desc, chosen_score desc, momentum_rank
    asc) -- the in-position, highest-validity, top-momentum picks float to the top.
    Picks the model would actually act on come first."""
    if df.is_empty():
        return df
    return df.sort(
        by=["in_position_at_D", "chosen_score", "momentum_rank"],
        descending=[True, True, False],
        nulls_last=True,
    )


def main():
    ap = argparse.ArgumentParser(description=REALIZABLE_LABEL)
    ap.add_argument("--date", required=True, help="decision day D, YYYY-MM-DD")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--cadence", default="1d",
                    help="1d or an event cadence (dollar/dib/range/runs_*/...)")
    ap.add_argument("--validity-window", type=int, default=365,
                    help="trailing window (days) for the past-only validity score")
    ap.add_argument("--mechanism", default="rolling_validity",
                    choices=sorted(MECHANISMS))
    ap.add_argument("--indicator", default="ma",
                    help=f"one of {sorted(INDICATOR_REGISTRY)}")
    ap.add_argument("--lookback", type=int, default=30,
                    help="momentum pre-rank window AND actionable-entry bound")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--all", action="store_true",
                    help="run choose_all: every mechanism side by side")
    args = ap.parse_args()

    chooser = AdaptiveChooser()
    print("=" * 80)
    print(REALIZABLE_LABEL)
    print(f"ADAPTIVE-CHOOSER date={args.date} universe={args.universe} "
          f"cadence={args.cadence} indicator={args.indicator}")
    print(f"validity_window={args.validity_window}d lookback={args.lookback}d "
          f"top_n={args.top_n}")
    print("=" * 80)

    if args.all:
        all_picks = chooser.choose_all(
            args.date, universe=args.universe, cadence=args.cadence,
            validity_window=args.validity_window, indicator=args.indicator,
            lookback_days=args.lookback, top_n=args.top_n)
        for name, tbl in all_picks.items():
            print(f"\n--- mechanism: {name} ---")
            if tbl.is_empty():
                print("(no rows)")
                continue
            n_in = int(tbl["in_position_at_D"].sum())
            print(f"in_position_at_D picks: {n_in} / {len(tbl)}")
            _print_table_ascii(tbl, max_rows=args.top_n)
        return

    table = chooser.choose(
        args.date, universe=args.universe, cadence=args.cadence,
        validity_window=args.validity_window, mechanism=args.mechanism,
        indicator=args.indicator, lookback_days=args.lookback, top_n=args.top_n)
    if table.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return
    n_in = int(table["in_position_at_D"].sum())
    print(f"in_position_at_D picks: {n_in} / {len(table)}")
    _print_table_ascii(table, max_rows=args.top_n)


if __name__ == "__main__":
    main()
