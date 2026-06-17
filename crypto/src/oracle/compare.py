"""ORACLE-vs-MODEL side-by-side comparison harness -- the headline deliverable.

================================================================================
"Call either side by side and compare the two: what instrument and TI config the
 MODEL chose (adaptive, PAST-ONLY) vs what the ORACLE says is the right answer
 (HINDSIGHT)."
================================================================================

This module joins the two existing, verified engines for a given decision date D
and lays them side by side, one row per instrument:

  * the ORACLE'S "right answer" (HINDSIGHT) -- ``oracle.engine.OracleEngine.oracle``:
    the best causal (config, entry) under the chosen driver, and the realized
    capture_rate it achieved to D (the upper bound; the model cannot beat it).

  * the MODEL'S choice (REALIZABLE / PAST-ONLY) -- ``oracle.adaptive.AdaptiveChooser.choose``:
    the config the model picks at D using only closes[:D] (rolling-validity /
    regime / state mechanism), and whether it would actually be IN-POSITION at D.

  * the SCORING of the model's pick -- ``capture_of_config`` (below): a hindsight
    EVAL of the model's PAST-ONLY pick. We compute what that specific (model-chosen)
    config actually captured to D by DELEGATING to
    ``OracleEngine.capture_of_config`` -- the EXACT SAME native (or daily) window /
    perfect-return / entry kernel the oracle uses to find its best config. This is
    the allowed scoring move: the PICK is past-only, only the GRADING of an
    already-made pick is hindsight.

KEY DISTINCTION (do not violate):
  - model_config / model_inpos / model_score are PAST-ONLY (the model knows them at D).
  - oracle_config / oracle_capture and model_realized_capture are HINDSIGHT EVALs
    (graded after the move is known). The gap = oracle_capture - model_realized_capture
    is how much capture the model LEFT ON THE TABLE vs the hindsight ceiling.
  - If the model would NOT have entered (model_inpos == False) it captures NOTHING
    -> model_realized_capture = 0.0 (you cannot bank a move you did not enter).

CEILING INVARIANT (and the 2026-06-08 bug it had): the oracle is the per-asset MAX
over the MA grid -- a model that picks a config FROM THE SAME GRID, graded by the
SAME kernel, can NEVER exceed it. ``oracle_capture`` in compare() is the max over
the grid of ``OracleEngine.capture_of_config`` (the true hindsight ceiling), and
``model_realized_capture`` is that same kernel at the model's chosen config -- so
``model_realized_capture <= oracle_capture`` holds BY CONSTRUCTION at EVERY cadence
/ resolution. The PRIOR bug: capture_of_config had its OWN bespoke index-based
window + entry logic that DIVERGED from the oracle's native calendar-day window at
non-1d cadences, so the model's graded capture could EXCEED the oracle's (e.g.
2026-04-17 @ 4h: 9/10 assets violated). Fixed by delegating both sides to the one
engine kernel. The summary still flags any residual violation as a likely bug.

ADDITIVE GUARANTEE: this module imports and reuses OracleEngine, AdaptiveChooser,
and the MAOracleEngine primitives. It does NOT modify any of them. ``capture_of_config``
is a thin DELEGATE here (per the task spec: do not modify ma_oracle_engine.py).

--------------------------------------------------------------------------------
CLI:
    python src/oracle/compare.py --date 2026-05-20 [--universe u10] [--cadence 1d]
        [--validity-window 365] [--mechanism rolling_validity] [--lookback 30]
        [--top-n 25]
    python src/oracle/compare.py --start 2026-05-01 --end 2026-05-20 --step-days 5
        --grid     (run compare_grid over a small breadth of cadence x window x mechanism)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

__contract__ = {
    "kind": "oracle_vs_model_compare",
    "inputs": [
        "oracle.engine.OracleEngine.oracle (HINDSIGHT 'right answer' per asset)",
        "oracle.adaptive.AdaptiveChooser.choose (PAST-ONLY model pick per asset)",
        "oracle.ma_oracle_engine primitives (_sma/_ema/_crosses + MAOracleEngine "
        "daily series) -- reused by capture_of_config to GRADE the model's pick",
        "chimera via pipeline.chimera_loader.ChimeraLoader",
    ],
    "outputs": {
        "callable": "OracleVsModel.compare(date, *, universe, cadence, validity_window, "
                    "mechanism, indicator, lookback_days, top_n) -> pl.DataFrame",
        "callable_grid": "OracleVsModel.compare_grid(dates, *, cadences, "
                         "validity_windows, mechanisms, ...) -> pl.DataFrame (leaderboard)",
        "printer": "OracleVsModel.side_by_side(date, ...) -> str (ASCII table)",
        "robustness": "OracleVsModel.robustness(date_range, ...) -> dict (optional null tests)",
    },
    "invariants": [
        "the MODEL pick (model_config/model_inpos/model_score) is PAST-ONLY -- "
        "the model knows it at D with no future leak",
        "the SCORING (oracle_capture, model_realized_capture) is HINDSIGHT EVAL ONLY "
        "-- grading an already-made past-only pick (the allowed scoring move)",
        "capture_rate in [0,1]; capture_of_config reuses the SAME perfect-return "
        "denominator + capture definition as the oracle/MAOracleEngine",
        "model_realized_capture = capture_of_config(...) when model_inpos else 0.0 "
        "(no entry -> no captured move)",
        "capture_GAP = oracle_capture - model_realized_capture (capture left on table)",
        "CEILING: oracle picks the max-capture config -> oracle_capture >= "
        "model_realized_capture expected; a violation is flagged as a likely bug",
        "no emoji in prints (cp1252)",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(SRC / "pipeline"), str(SRC / "oracle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
# REUSE the verified causal primitives + helpers (do NOT duplicate / modify).
from oracle.ma_oracle_engine import (  # noqa: E402
    _to_date, _last_idx_le, _sma, _ema, _crosses, _print_table_ascii,
)
from oracle.engine import OracleEngine, _fmt_cfg  # noqa: E402
from oracle.adaptive import AdaptiveChooser  # noqa: E402

LABEL = ("ORACLE (hindsight 'right answer') vs MODEL (past-only choice) -- "
         "side by side. capture is hindsight EVAL of a past-only pick.")


# ============================================================================
# capture_of_config -- GRADE a SPECIFIC (model-chosen) config's realized capture
# ============================================================================
def _parse_ma_config(config_str: str) -> dict | None:
    """Parse the canonical 'FAM(fast,slow)' string (the shared _fmt_cfg form for
    indicator='ma') back into {'family','fast','slow'}. Returns None on anything
    that does not match (e.g. None, or a non-MA indicator string)."""
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


def capture_of_config(engine: OracleEngine, sym: str, date, config,
                      lookback_days: int = 30, cadence: str = "1d",
                      indicator: str = "ma", resolution: str = "native") -> dict:
    """HINDSIGHT EVAL of a GIVEN (model-chosen) config for one asset as of D.

    THIN DELEGATE (2026-06-08 ceiling-bug fix): this now calls
    ``OracleEngine.capture_of_config`` -- the EXACT SAME native (or daily) window /
    perfect-return / entry kernel the oracle uses to find its best config. Before
    the fix this function had its OWN bespoke window (index-based ``win_lo =
    d_idx - lookback_days``) and entry logic, which DIVERGED from the oracle's
    native calendar-day window at non-1d cadences -- so a model that picked one of
    the oracle's OWN grid configs could be graded with a HIGHER capture than the
    oracle (a ceiling VIOLATION, e.g. 2026-04-17 @ 4h). Delegating to the engine's
    factored kernel puts ``model_realized_capture`` on the IDENTICAL scale as
    ``oracle_capture`` so ``model_realized_capture <= oracle_capture`` holds by
    construction (the oracle's capture is that same kernel at the oracle-chosen
    config; the compare ceiling is the max over the grid -- see compare()).

    The signal that locates the entry is CAUSAL (a golden cross over closes[:d_idx+1],
    same as the oracle). The hindsight is ONLY in grading an already-fixed config.

    config: either a 'FAM(fast,slow)' string (the shared _fmt_cfg form) or a
            {'family','fast','slow'} dict. indicator must be 'ma' (the implemented
            family); other families return in_position=False with a note.
    resolution: 'native' (default) | 'daily' -- MUST match the resolution the
            compare()/oracle() run used so the grade is apples-to-apples.

    Returns a dict mirroring the engine kernel's keys (a superset of the prior
    return: adds entry_ts, bars_back):
        sym, config, entry_date, entry_ts, bars_back, days_back, captured_return,
        perfect_return, capture_rate, in_position, note.
    """
    return engine.capture_of_config(
        sym, date, config, cadence=cadence, lookback_days=lookback_days,
        resolution=resolution, indicator=indicator)


# ============================================================================
# THE COMPARISON HARNESS
# ============================================================================
class OracleVsModel:
    """Side-by-side ORACLE (hindsight) vs MODEL (past-only) (instrument, TI-config)
    comparison. One shared loader/engine backs both sides so the daily series, the
    ranking, and the capture definition are identical (apples-to-apples)."""

    def __init__(self, loader: ChimeraLoader | None = None):
        self.loader = loader or ChimeraLoader()
        self.oracle_engine = OracleEngine(self.loader)
        # AdaptiveChooser reuses the same OracleEngine internally; share the loader.
        self.chooser = AdaptiveChooser(self.loader)

    # ---- per-date side-by-side ------------------------------------------
    def compare(self, date, *, universe: str = "u10", cadence: str = "1d",
                validity_window: int = 365, mechanism: str = "rolling_validity",
                indicator: str = "ma", lookback_days: int = 30,
                top_n: int = 25) -> pl.DataFrame:
        """Run oracle() + adaptive.choose() for ``date``, JOIN on sym, and per row
        compute the side-by-side fields + the hindsight grade of the model's pick.

        Returns one row per instrument (the join of the two top-N sets) with:
            sym, perf_rank,
            oracle_config (the rolling-validity driver's pick -- the oracle's
                'right answer'), oracle_inpos,
            oracle_capture (the per-asset hindsight CEILING = MAX over the MA grid
                of engine.capture_of_config; the upper bound the model cannot beat),
            model_config, model_inpos, model_score (past-only validity),
            model_realized_capture (hindsight EVAL: engine.capture_of_config at the
                model's chosen config when model in-position, else 0.0 -- the SAME
                kernel as the ceiling, so model_realized_capture <= oracle_capture
                holds BY CONSTRUCTION),
            config_MATCH (oracle_config == model_config),
            capture_GAP  (oracle_capture - model_realized_capture, always >= 0).

        A one-line summary (config-match rate, mean capture_GAP, model vs oracle
        mean capture) is attached as ``df.attrs['summary']`` (a dict) AND can be
        rendered via summary_line(df).
        """
        d = _to_date(date)
        ora = self.oracle_engine.oracle(
            d, universe=universe, indicator=indicator, cadence=cadence,
            lookback_days=lookback_days, top_n=top_n,
            validity_windows=(validity_window,) if isinstance(validity_window, int)
            else tuple(validity_window),
            driver="rolling_validity")
        mod = self.chooser.choose(
            d, universe=universe, cadence=cadence, validity_window=validity_window,
            mechanism=mechanism, indicator=indicator, lookback_days=lookback_days,
            top_n=top_n)

        if ora.is_empty() and mod.is_empty():
            return pl.DataFrame()

        # Normalize the two sides to a common per-sym schema, then OUTER-join on sym
        # (the two top-N sets are ranked the same way, so they coincide; outer-join
        # is defensive against any edge mismatch).
        ora_s = (ora.select([
            pl.col("sym"),
            pl.col("perf_rank").alias("perf_rank"),
            pl.col("best_config").alias("oracle_config"),
            pl.col("in_position").alias("oracle_inpos"),
            pl.col("capture_rate").alias("oracle_capture"),
        ]) if not ora.is_empty() else
            pl.DataFrame(schema={"sym": pl.Utf8, "perf_rank": pl.Int64,
                                 "oracle_config": pl.Utf8, "oracle_inpos": pl.Boolean,
                                 "oracle_capture": pl.Float64}))
        mod_s = (mod.select([
            pl.col("sym"),
            pl.col("chosen_config").alias("model_config"),
            pl.col("in_position_at_D").alias("model_inpos"),
            pl.col("chosen_score").alias("model_score"),
            pl.col("momentum_rank").alias("model_rank"),
        ]) if not mod.is_empty() else
            pl.DataFrame(schema={"sym": pl.Utf8, "model_config": pl.Utf8,
                                 "model_inpos": pl.Boolean, "model_score": pl.Float64,
                                 "model_rank": pl.Int64}))

        j = ora_s.join(mod_s, on="sym", how="full", coalesce=True)
        if j.is_empty():
            return pl.DataFrame()

        # The MA grid -- the per-asset hindsight ceiling is the MAX capture over
        # exactly these configs (the model picks ONE of them, so it cannot exceed it).
        from oracle.engine import INDICATOR_REGISTRY
        grid = INDICATOR_REGISTRY[indicator].config_grid() if indicator == "ma" else []

        # per-row hindsight EVAL of the model's pick + the derived comparison fields.
        rows = []
        for rec in j.iter_rows(named=True):
            sym = rec["sym"]
            model_cfg = rec.get("model_config")
            model_inpos = bool(rec.get("model_inpos")) if rec.get("model_inpos") is not None else False
            oracle_cfg = rec.get("oracle_config")

            # ORACLE CEILING (the upper bound the model cannot beat) = MAX over the
            # SAME grid of engine.capture_of_config (the IDENTICAL kernel that grades
            # the model). This makes model_realized_capture <= oracle_capture hold BY
            # CONSTRUCTION at every cadence/resolution, independent of which config
            # the rolling-validity driver happened to surface as oracle_config. The
            # prior bug let the model exceed the oracle because the two sides were
            # graded by DIFFERENT windows; both now use this one kernel.
            oracle_cap = 0.0
            if grid:
                for cfg in grid:
                    g = self.oracle_engine.capture_of_config(
                        sym, d, cfg, cadence=cadence, lookback_days=lookback_days,
                        resolution="native", indicator=indicator)
                    if g["capture_rate"] > oracle_cap:
                        oracle_cap = float(g["capture_rate"])
            elif rec.get("oracle_capture") is not None:
                oracle_cap = float(rec["oracle_capture"])

            # GRADE the model's pick (hindsight EVAL). No entry -> captures nothing.
            # resolution='native' matches the oracle()/chooser default so the grade
            # uses the SAME native (calendar-day) window the oracle ranked with ->
            # model_realized_capture and oracle_capture are on the IDENTICAL scale.
            if model_inpos and model_cfg is not None:
                graded = capture_of_config(
                    self.oracle_engine, sym, d, model_cfg, lookback_days=lookback_days,
                    cadence=cadence, indicator=indicator, resolution="native")
                model_realized = float(graded["capture_rate"])
            else:
                model_realized = 0.0

            config_match = bool(oracle_cfg is not None and model_cfg is not None
                                and oracle_cfg == model_cfg)
            capture_gap = oracle_cap - model_realized
            rows.append({
                "sym": sym,
                "perf_rank": rec.get("perf_rank"),
                "oracle_config": oracle_cfg,
                "oracle_inpos": bool(rec.get("oracle_inpos")) if rec.get("oracle_inpos") is not None else False,
                "oracle_capture": round(oracle_cap, 6),
                "model_config": model_cfg,
                "model_inpos": model_inpos,
                "model_score": (round(float(rec["model_score"]), 6)
                                if rec.get("model_score") is not None else None),
                "model_realized_capture": round(model_realized, 6),
                "config_MATCH": config_match,
                "capture_GAP": round(capture_gap, 6),
            })

        df = pl.DataFrame(rows, infer_schema_length=None)
        # sort by perf_rank (the shared momentum pre-rank); nulls last.
        if "perf_rank" in df.columns:
            df = df.sort("perf_rank", nulls_last=True)
        # attach the one-line summary.
        df = df.with_columns()  # no-op to ensure a fresh frame
        summ = _compute_summary(df)
        try:
            df.attrs = {"summary": summ}  # type: ignore[attr-defined]
        except Exception:
            pass
        self._last_summary = summ
        return df

    # ---- exhaustive sweep -> leaderboard --------------------------------
    def compare_grid(self, dates, *, universe: str = "u10",
                     cadences=("1d",), validity_windows=(180, 365),
                     mechanisms=("rolling_validity", "regime_cond", "state_cond"),
                     indicator: str = "ma", lookback_days: int = 30,
                     top_n: int = 25) -> pl.DataFrame:
        """EXHAUSTIVE sweep: for each (date x cadence x validity_window x mechanism)
        run compare() and aggregate into a LEADERBOARD row (pooled over the dates):
            {cadence, validity_window, mechanism, n_dates, config_match_rate,
             mean_capture_gap, model_mean_capture, oracle_mean_capture,
             model_inpos_rate}.

        Returns the leaderboard sorted by model_mean_capture desc (best
        mechanism/cadence/window on top) -- the 'exhaustive across dimensions'
        deliverable.
        """
        dates = [_to_date(x) for x in dates]
        rows = []
        for cad in cadences:
            for vw in validity_windows:
                for mech in mechanisms:
                    # pool the per-asset rows across all dates for this combo.
                    match_n = match_d = 0
                    gaps = []
                    model_caps = []
                    oracle_caps = []
                    inpos_n = inpos_d = 0
                    n_dates_used = 0
                    for d in dates:
                        try:
                            cdf = self.compare(
                                d, universe=universe, cadence=cad,
                                validity_window=vw, mechanism=mech,
                                indicator=indicator, lookback_days=lookback_days,
                                top_n=top_n)
                        except Exception:
                            continue
                        if cdf.is_empty():
                            continue
                        n_dates_used += 1
                        # config-match rate: over rows where BOTH sides picked a config.
                        both = cdf.filter(
                            pl.col("oracle_config").is_not_null()
                            & pl.col("model_config").is_not_null())
                        match_d += both.height
                        match_n += int(both["config_MATCH"].sum()) if both.height else 0
                        gaps.extend(cdf["capture_GAP"].to_list())
                        model_caps.extend(cdf["model_realized_capture"].to_list())
                        oracle_caps.extend(cdf["oracle_capture"].to_list())
                        inpos_d += cdf.height
                        inpos_n += int(cdf["model_inpos"].sum())
                    rows.append({
                        "cadence": cad,
                        "validity_window": vw,
                        "mechanism": mech,
                        "n_dates": n_dates_used,
                        "config_match_rate": round(match_n / match_d, 4) if match_d else 0.0,
                        "mean_capture_gap": round(float(np.mean(gaps)), 6) if gaps else 0.0,
                        "model_mean_capture": round(float(np.mean(model_caps)), 6) if model_caps else 0.0,
                        "oracle_mean_capture": round(float(np.mean(oracle_caps)), 6) if oracle_caps else 0.0,
                        "model_inpos_rate": round(inpos_n / inpos_d, 4) if inpos_d else 0.0,
                    })
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows, infer_schema_length=None).sort(
            "model_mean_capture", descending=True)

    # ---- pretty side-by-side printer (one date) -------------------------
    def side_by_side(self, date, *, universe: str = "u10", cadence: str = "1d",
                     validity_window: int = 365, mechanism: str = "rolling_validity",
                     indicator: str = "ma", lookback_days: int = 30,
                     top_n: int = 25, to_stdout: bool = True) -> str:
        """Pretty plain-ASCII (no box chars) side-by-side of compare() for ONE date.
        Returns the rendered string (and prints it if to_stdout)."""
        d = _to_date(date)
        df = self.compare(
            d, universe=universe, cadence=cadence, validity_window=validity_window,
            mechanism=mechanism, indicator=indicator, lookback_days=lookback_days,
            top_n=top_n)
        lines = []
        lines.append("=" * 100)
        lines.append(LABEL)
        lines.append(f"date={d} universe={universe} cadence={cadence} indicator={indicator} "
                     f"mechanism={mechanism} validity_window={validity_window} "
                     f"lookback={lookback_days}d top_n={top_n}")
        lines.append("=" * 100)
        if df.is_empty():
            lines.append("(no rows -- no assets cover this date with >= lookback history)")
            out = "\n".join(lines)
            if to_stdout:
                print(out)
            return out
        # ascii table (reuse the verified ASCII printer by capturing it would print;
        # build the rows here for full control over the side-by-side column order).
        cols = ["sym", "perf_rank", "oracle_config", "oracle_capture",
                "model_config", "model_inpos", "model_score",
                "model_realized_capture", "config_MATCH", "capture_GAP"]
        show = df.select([c for c in cols if c in df.columns])
        lines.append(_ascii_table_str(show, max_rows=top_n))
        lines.append("-" * 100)
        lines.append(summary_line(df))
        out = "\n".join(lines)
        if to_stdout:
            print(out)
        return out

    # ---- optional robustness: does the model's SELECTION beat random + passive? ----
    def robustness(self, dates, *, universe: str = "u10", cadence: str = "1d",
                   validity_window: int = 365, mechanism: str = "rolling_validity",
                   indicator: str = "ma", lookback_days: int = 30, top_n: int = 25,
                   n_null: int = 300, seed: int = 7) -> dict:
        """OPTIONAL: pool the model's realized per-pick captures across ``dates`` and
        ask whether the model's SELECTION beats (a) random in-position selection and
        (b) a passive 'enter every in-position name' benchmark.

        - The model's pool = capture_rate of every name the model put IN-POSITION.
        - The random-selection null = for each date, pick the SAME count of names at
          random FROM THE in-position candidates and pool their captures; repeat
          n_null times -> a p50/p95 distribution of mean capture. (A bespoke
          selection-null; firewall.random_entry_null needs a full single-asset
          CanonicalHarness, which does not map to this cross-sectional config-pick,
          so we use the matched-count selection null here -- same spirit, runs cleanly.)
        - The passive benchmark = mean capture if you entered EVERY in-position name
          (no selection). evaluate_setup_chaser grades the pooled model returns.

        Returns a dict with the pooled stats + null percentiles + the chaser gate.
        Returns {'status': 'no_data'} if nothing is in-position across the dates.
        """
        from oracle.engine import OracleEngine as _OE  # local ref (already imported)
        try:
            from strat.battery import evaluate_setup_chaser
        except Exception:
            evaluate_setup_chaser = None

        dates = [_to_date(x) for x in dates]
        model_caps = []          # capture of each name the MODEL selected (in-position)
        passive_caps_per_date = []   # per-date list of ALL in-position-candidate captures
        model_counts = []        # how many the model selected per date
        for d in dates:
            cdf = self.compare(
                d, universe=universe, cadence=cadence, validity_window=validity_window,
                mechanism=mechanism, indicator=indicator, lookback_days=lookback_days,
                top_n=top_n)
            if cdf.is_empty():
                continue
            # the model's selected (in-position) names and their realized captures.
            sel = cdf.filter(pl.col("model_inpos"))
            sel_caps = sel["model_realized_capture"].to_list()
            model_caps.extend(sel_caps)
            model_counts.append(len(sel_caps))
            # the universe of in-position CANDIDATES for the random/passive baseline:
            # every name that COULD be entered (oracle in-position OR model in-position),
            # graded by capture_of_config under the same lookback. We approximate the
            # candidate pool by the union of in-position names' realized captures: the
            # model's own captures plus the oracle's capture for oracle-in-position names.
            cand = cdf.filter(pl.col("oracle_inpos") | pl.col("model_inpos"))
            cand_caps = []
            for rec in cand.iter_rows(named=True):
                # prefer the realized capture of whatever config is in-position for that name
                if rec["model_inpos"] and rec["model_config"] is not None:
                    cand_caps.append(float(rec["model_realized_capture"]))
                elif rec["oracle_inpos"]:
                    cand_caps.append(float(rec["oracle_capture"]))
            if cand_caps:
                passive_caps_per_date.append((len(sel_caps), cand_caps))

        if not model_caps:
            return {"status": "no_data",
                    "note": "no model in-position picks across the given dates"}

        model_mean = float(np.mean(model_caps))
        passive_pool = [c for (_, lst) in passive_caps_per_date for c in lst]
        passive_mean = float(np.mean(passive_pool)) if passive_pool else 0.0

        # matched-count random-selection null: per date, draw the model's count from
        # the candidate pool; pool across dates; repeat -> p50/p95 of mean capture.
        rng = np.random.default_rng(seed)
        null_means = []
        for _ in range(n_null):
            draw = []
            for (k, lst) in passive_caps_per_date:
                if k <= 0 or not lst:
                    continue
                idx = rng.integers(0, len(lst), size=k)
                draw.extend(np.asarray(lst)[idx].tolist())
            if draw:
                null_means.append(float(np.mean(draw)))
        null_p50 = float(np.percentile(null_means, 50)) if null_means else None
        null_p95 = float(np.percentile(null_means, 95)) if null_means else None
        beats_random = (null_p95 is not None and model_mean > null_p95)

        chaser = None
        if evaluate_setup_chaser is not None:
            # grade the pooled model captures as a setup-chaser book (UNSEEN proxy =
            # the pooled held-out captures; selective vs the passive-enter-everything mean).
            chaser = evaluate_setup_chaser(
                model_caps, {"UNSEEN": model_mean * 100.0}, -5.0,
                flat_benchmark_mean=passive_mean)

        return {
            "status": "ok",
            "n_dates": len(model_counts),
            "n_model_picks": len(model_caps),
            "model_mean_capture": round(model_mean, 6),
            "passive_mean_capture": round(passive_mean, 6),
            "selection_edge_vs_passive": round(model_mean - passive_mean, 6),
            "random_null_p50": round(null_p50, 6) if null_p50 is not None else None,
            "random_null_p95": round(null_p95, 6) if null_p95 is not None else None,
            "beats_random_selection": bool(beats_random),
            "chaser_gate": chaser,
            "note": ("model SELECTION value = model_mean - passive_mean (does picking "
                     "beat entering everything?) AND model_mean vs random_null_p95 "
                     "(does picking beat random same-count selection?)"),
        }


# ============================================================================
# summary + ascii helpers
# ============================================================================
def _compute_summary(df: pl.DataFrame) -> dict:
    """Compute the one-line summary stats over a compare() frame."""
    if df.is_empty():
        return {"n_rows": 0}
    both = df.filter(pl.col("oracle_config").is_not_null()
                     & pl.col("model_config").is_not_null())
    match_rate = (float(both["config_MATCH"].sum()) / both.height) if both.height else 0.0
    n_model_inpos = int(df["model_inpos"].sum()) if "model_inpos" in df.columns else 0
    n_oracle_inpos = int(df["oracle_inpos"].sum()) if "oracle_inpos" in df.columns else 0
    mean_gap = float(df["capture_GAP"].mean())
    model_mean = float(df["model_realized_capture"].mean())
    oracle_mean = float(df["oracle_capture"].mean())
    # ceiling sanity: any row where the model beat the hindsight oracle (a bug flag).
    violations = df.filter(pl.col("model_realized_capture") > pl.col("oracle_capture") + 1e-9)
    return {
        "n_rows": df.height,
        "config_match_rate": round(match_rate, 4),
        "n_both_picked": both.height,
        "n_model_inpos": n_model_inpos,
        "n_oracle_inpos": n_oracle_inpos,
        "mean_capture_GAP": round(mean_gap, 6),
        "model_mean_capture": round(model_mean, 6),
        "oracle_mean_capture": round(oracle_mean, 6),
        "ceiling_violations": violations.height,
        "ceiling_violation_syms": violations["sym"].to_list() if violations.height else [],
    }


def summary_line(df: pl.DataFrame) -> str:
    """Render the one-line summary for a compare() frame."""
    s = _compute_summary(df)
    if s.get("n_rows", 0) == 0:
        return "SUMMARY: (no rows)"
    line = (f"SUMMARY: rows={s['n_rows']}  config-match={s['config_match_rate']:.1%} "
            f"(of {s['n_both_picked']} both-picked)  "
            f"model_inpos={s['n_model_inpos']}/{s['n_rows']}  "
            f"oracle_inpos={s['n_oracle_inpos']}/{s['n_rows']}  "
            f"mean_capture_GAP={s['mean_capture_GAP']:+.4f}  "
            f"model_mean_capture={s['model_mean_capture']:.4f}  "
            f"oracle_mean_capture={s['oracle_mean_capture']:.4f}")
    if s["ceiling_violations"]:
        line += (f"\n  *** CEILING VIOLATION: model beat oracle on "
                 f"{s['ceiling_violations']} row(s) {s['ceiling_violation_syms']} "
                 f"-- LIKELY BUG (oracle is the max-capture config) ***")
    else:
        line += "\n  ceiling-sanity OK: model_realized_capture <= oracle_capture on every row."
    return line


def _ascii_table_str(table: pl.DataFrame, max_rows: int = 25) -> str:
    """Plain-ASCII table render (no Unicode box glyphs -> cp1252 safe). Returns the
    string (mirrors ma_oracle_engine._print_table_ascii but returns instead of prints)."""
    cols = table.columns
    rows = table.head(max_rows).rows()
    cells = [[("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
              for v in row] for row in rows]
    widths = [len(c) for c in cols]
    for row in cells:
        for i, v in enumerate(row):
            if len(v) > widths[i]:
                widths[i] = len(v)
    sep = "-+-".join("-" * w for w in widths)
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    out = [header, sep]
    for row in cells:
        out.append(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)))
    return "\n".join(out)


def _daterange(start, end, step_days: int):
    s, e = _to_date(start), _to_date(end)
    out = []
    cur = s
    while cur <= e:
        out.append(cur)
        cur = cur + timedelta(days=step_days)
    return out


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--date", default=None, help="single decision day D, YYYY-MM-DD")
    ap.add_argument("--start", default=None, help="grid start date (with --end/--step-days)")
    ap.add_argument("--end", default=None, help="grid end date")
    ap.add_argument("--step-days", type=int, default=5, help="grid date step (days)")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--validity-window", type=int, default=365)
    ap.add_argument("--mechanism", default="rolling_validity")
    ap.add_argument("--indicator", default="ma")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--grid", action="store_true",
                    help="run compare_grid over a small breadth (cadence x window x mechanism)")
    ap.add_argument("--robustness", action="store_true",
                    help="also run the optional selection-robustness check over the date range")
    args = ap.parse_args()

    cmp = OracleVsModel()

    if args.grid:
        if args.start and args.end:
            dates = _daterange(args.start, args.end, args.step_days)
        elif args.date:
            dates = [args.date]
        else:
            raise SystemExit("--grid needs either --date or --start/--end/--step-days")
        lb = cmp.compare_grid(
            dates, universe=args.universe, cadences=(args.cadence,),
            validity_windows=(180, 365),
            mechanisms=("rolling_validity", "regime_cond", "state_cond"),
            indicator=args.indicator, lookback_days=args.lookback, top_n=args.top_n)
        print("=" * 100)
        print("COMPARE-GRID LEADERBOARD (sorted by model_mean_capture desc) -- "
              f"n_dates={len(dates)} universe={args.universe} cadence={args.cadence}")
        print("=" * 100)
        if lb.is_empty():
            print("(empty leaderboard)")
        else:
            print(_ascii_table_str(lb, max_rows=100))
        return

    if not args.date:
        raise SystemExit("provide --date (single day) or --grid with --start/--end/--step-days")

    cmp.side_by_side(
        args.date, universe=args.universe, cadence=args.cadence,
        validity_window=args.validity_window, mechanism=args.mechanism,
        indicator=args.indicator, lookback_days=args.lookback, top_n=args.top_n)

    if args.robustness:
        if args.start and args.end:
            dates = _daterange(args.start, args.end, args.step_days)
        else:
            dates = [args.date]
        print("\n" + "=" * 100)
        print("OPTIONAL ROBUSTNESS (model SELECTION vs random + passive)")
        print("=" * 100)
        rob = cmp.robustness(
            dates, universe=args.universe, cadence=args.cadence,
            validity_window=args.validity_window, mechanism=args.mechanism,
            indicator=args.indicator, lookback_days=args.lookback, top_n=args.top_n)
        for k, v in rob.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
