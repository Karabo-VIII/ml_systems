"""CROSS-DIMENSIONAL ORACLE DECOMPOSER (v2, built ON ma_oracle_engine v1).

================================================================================
HINDSIGHT DNA EXTRACTOR -- descriptive, NOT a tradeable signal.
================================================================================

v1 (ma_oracle_engine.MAOracleEngine) answers, for a day D and the top-25 daily
performers, "what is the best MA-family entry that captured the max realized
return to D?".  This v2 decomposer takes that DRIVER and builds, per
(asset, best-TI move), ONE rich DNA record:

    DRIVER   -- the best MA-family config + entry, with a BOUNDED lookback so a
                200-day-old golden cross can no longer dominate via capture-rate
                clipping (the v1 artifact).  The MA grid is just the *driver*; the
                point of this engine is the CONTEXT around the move.
    CONTEXT  -- the REAL chimera v50/v51 feature vector for that asset, pulled
                AS-OF the entry day AND as-of the move's peak day (the 33 norm_*
                normalized features + regime/hurst/cross-asset descriptors).  This
                is the surrounding-context DNA we will later mine for "what
                co-occurs with a good capture".
    CHART    -- the same driver run across >= 2 chart types (1d native + dollar
                bars aggregated to a daily close), so we see how best-TI / capture
                differs by bar construction.

Output: one tidy decomposed record per (asset, move) ->
    runs/oracle/decomposer_<date>.parquet  (+ a readable .csv preview).
Queryable: decompose(date, universe, ...) -> pl.DataFrame.

--------------------------------------------------------------------------------
HONESTY / INVARIANTS (overseer WILL RWYB):
  * STILL HINDSIGHT.  The hindsight is in selecting the best config after the
    fact (the allowed oracle move) AND in centering the context on the realized
    move's peak.  The value is the DNA (what context co-occurs with good
    captures), not a tradeable signal.
  * The driver signal stays CAUSALLY past-only: every MA at day t uses closes up
    to and including t only; crosses are detected from t-1 -> t sign changes.  We
    inherit v1's _sma/_ema/_crosses and re-verify the no-future-leak truncation.
  * BOUNDED lookback: the selected golden cross must be at most `max_days_back`
    days before D (default = lookback_days = the ranking window).  This removes
    the v1 unbounded-entry artifact (days_back up to ~206).
  * Chimera features attached at the entry / peak day are the REAL loader values,
    AS-OF that day (no future leak -- the feature value recorded AT that day's
    bar; chimera features are themselves causal/as-of by construction).
  * No emoji in prints (cp1252-safe).

--------------------------------------------------------------------------------
DRIVER MODE (what we did, and why):
  Two selection modes are implemented; `decompose(...)` defaults to the ELEVATED
  one and records which was used in the `driver_mode` column.

    "bounded_oneshot" -- v1's max-captured pick, but the golden cross is
        constrained to entry within [D - max_days_back, D].  One-shot: the single
        best realized capture in-window.  Fast.

    "rolling_validity" (ELEVATED, default) -- for each (asset, chart_type, MA
        config) we score the config by how well it has historically captured this
        asset's moves over a trailing validity window: the MEAN capture_rate of
        the config's COMPLETED entries (golden->death round-trips) whose entry
        falls in [D - validity_window, D], measured against the per-trade
        perfect-entry oracle for that round-trip's own window.  We then pick the
        config with the best (rolling_validity, then in-position captured) and
        report the in-position entry for that winner.  This selects a config that
        *robustly* captures this asset's moves, not the single luckiest one-shot.
        Falls back to bounded_oneshot for an asset if it has < `min_valid_trades`
        completed entries in the validity window (flagged per row).

CLI:
    python src/oracle/decomposer.py --date 2026-05-20 [--universe u50]
        [--lookback 30] [--max-days-back 30] [--driver-mode rolling_validity]
        [--chart-types 1d,dollar] [--out runs/oracle/decomposer_<date>.parquet]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

__contract__ = {
    "kind": "oracle_decomposer",
    "inputs": [
        "ma_oracle_engine.MAOracleEngine (v1 driver: rank/best_ma_capture/crosses)",
        "chimera v50/v51 features via pipeline.chimera_loader.ChimeraLoader",
    ],
    "outputs": {
        "callable": "decompose(date, universe, lookback_days, ...) -> pl.DataFrame",
        "parquet": "runs/oracle/decomposer_<date>.parquet",
        "csv": "runs/oracle/decomposer_<date>.csv",
    },
    "invariants": [
        "DRIVER MA + cross signal is CAUSAL (past-only up to each day)",
        "best-config selection is hindsight (the allowed oracle move)",
        "BOUNDED lookback: selected entry within [D - max_days_back, D]",
        "chimera context features are REAL loader values as-of entry/peak day",
        "capture_rate in [0,1]; entry_date <= date; 0 <= days_back <= max_days_back",
        "output labeled HINDSIGHT -- not a tradeable signal",
        "no emoji in prints (cp1252)",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(SRC / "pipeline"), str(SRC / "oracle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from oracle.ma_oracle_engine import (  # noqa: E402
    MAOracleEngine, _to_date, _last_idx_le, _sma, _ema, _crosses,
)

HINDSIGHT_LABEL = "HINDSIGHT DNA -- descriptive, not a tradeable signal."

# Curated CONTEXT feature set pulled from chimera as-of the entry / peak day.
# The 33 norm_* features are the canonical normalized surface (std ~ 1); we add a
# handful of descriptive non-norm columns (regime, hurst, raw return, cross-asset)
# that carry direct interpretive value. Any column absent for an asset/cadence is
# simply skipped (recorded as null) -- we use what the real schema provides.
CONTEXT_NORM_PREFIX = "norm_"
CONTEXT_EXTRA = [
    "returns_clean", "regime_label", "hurst_regime",
    "xd_btc_return", "xd_btc_volatility", "xd_ma_distance", "xd_momentum_rank",
    "rv_rv_5m", "rv_jump_frac",
    "liq_total_usd", "fund_rate_mean", "s3_global_lsr",
    "premium_apr", "bs_basis_pct",
]


class OracleDecomposer:
    """Cross-dimensional DNA decomposer built on the v1 MA oracle engine."""

    def __init__(self, loader: ChimeraLoader | None = None,
                 fast=None, slow=None):
        self.loader = loader or ChimeraLoader()
        # The v1 engine carries the canonical grid + causal MA/cross primitives.
        if fast is not None and slow is not None:
            self.engine = MAOracleEngine(self.loader, fast=fast, slow=slow)
        else:
            self.engine = MAOracleEngine(self.loader)
        self.grid = self.engine.grid
        # per (sym, cadence) -> (date,close) frame cache
        self._series_cache: dict[tuple[str, str], pl.DataFrame | None] = {}
        # per (sym) -> full 1d feature frame cache (for context lookups)
        self._feat_cache: dict[str, pl.DataFrame | None] = {}

    # ---- data access ----------------------------------------------------
    def _daily_series(self, sym: str, cadence: str) -> pl.DataFrame | None:
        """Return an ascending (date, close) daily frame for sym at a cadence.

        For 1d this is the native daily bar.  For event-based cadences (dollar /
        dib / range / ...) we aggregate to a DAILY close = the LAST bar's close on
        each calendar date, so every chart type yields a comparable daily close
        series the same causal MA driver can run over.  None if unavailable.
        """
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
            # event bars: many per day -> last close of the day = daily close.
            out = (df.sort("date")
                     .group_by("date")
                     .agg(pl.col("close").last().alias("close"))
                     .sort("date"))
        self._series_cache[key] = out
        return out

    def _feature_frame(self, sym: str) -> pl.DataFrame | None:
        """Full 1d chimera frame (all columns) for context lookups, sorted by date."""
        if sym in self._feat_cache:
            return self._feat_cache[sym]
        try:
            df = self.loader.load(sym, cadence="1d")
        except Exception:
            self._feat_cache[sym] = None
            return None
        if "date" not in df.columns:
            self._feat_cache[sym] = None
            return None
        df = df.unique(subset=["date"], keep="last").sort("date")
        self._feat_cache[sym] = df
        return df

    def _context_columns(self, sym: str) -> list[str]:
        """The context columns actually present in this asset's 1d schema."""
        feat = self._feature_frame(sym)
        if feat is None:
            return []
        cols = feat.columns
        norm = [c for c in cols if c.startswith(CONTEXT_NORM_PREFIX)]
        extra = [c for c in CONTEXT_EXTRA if c in cols]
        # stable, de-duplicated order
        seen, out = set(), []
        for c in norm + extra:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _features_as_of(self, sym: str, d: _date) -> dict:
        """The REAL chimera feature row AS-OF day d (last bar with date <= d).

        Returns {col: value} for the curated context columns. No future leak: we
        select the row whose date is the last <= d, and read that row's own
        (causal/as-of) feature values.
        """
        feat = self._feature_frame(sym)
        if feat is None:
            return {}
        dates = feat["date"].to_list()
        idx = _last_idx_le(dates, d)
        if idx is None:
            return {}
        cols = self._context_columns(sym)
        row = feat.row(idx, named=True)
        out = {}
        for c in cols:
            v = row.get(c, None)
            if v is not None and isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = None
            out[c] = v
        return out

    # ---- causal per-config crosses on an arbitrary daily series ---------
    def _config_crosses(self, closes: np.ndarray, fam: str, f: int, s: int):
        """(golden_idx, death_idx) for one config over a causal daily close array."""
        if fam == "SMA":
            ma_f, ma_s = _sma(closes, f), _sma(closes, s)
        else:
            ma_f, ma_s = _ema(closes, f), _ema(closes, s)
        return _crosses(ma_f - ma_s)

    # ---- driver: bounded one-shot best capture --------------------------
    def _driver_bounded_oneshot(self, dates, closes, d_idx, max_days_back):
        """Best realized capture among configs IN POSITION at D whose golden cross
        is within [D - max_days_back, D].  Causal: only closes[:d_idx+1] are seen.

        Returns (best_dict | None).  best_dict has family/fast/slow/entry_idx/
        captured/entry_date/days_back.
        """
        d = dates[d_idx]
        past = closes[:d_idx + 1]
        cD = float(past[d_idx])
        best = None
        for (fam, f, s) in self.grid:
            golden, death = self._config_crosses(past, fam, f, s)
            if not golden:
                continue
            last_g = golden[-1]
            # still open at D?
            if any(dx > last_g for dx in death):
                continue
            entry_date = dates[last_g]
            days_back = (d - entry_date).days
            if days_back > max_days_back:        # BOUNDED LOOKBACK
                continue
            c_entry = float(past[last_g])
            if c_entry <= 0 or np.isnan(c_entry):
                continue
            captured = (cD - c_entry) / c_entry
            if best is None or captured > best["captured"]:
                best = dict(family=fam, fast=f, slow=s, entry_idx=last_g,
                            entry_date=entry_date, days_back=int(days_back),
                            captured=float(captured))
        return best

    # ---- driver: rolling-validity score per config ----------------------
    def _config_rolling_validity(self, dates, closes, d_idx, fam, f, s,
                                 validity_window):
        """Mean capture_rate of this config's COMPLETED round-trips (golden->death)
        whose ENTRY is within [D - validity_window, D]. Causal per round-trip:
        each capture_rate is computed only from closes inside that round-trip's
        own [entry, exit] window (no future relative to the round-trip).

        Returns (mean_capture_rate | None, n_trades).
        """
        d = dates[d_idx]
        past = closes[:d_idx + 1]
        golden, death = self._config_crosses(past, fam, f, s)
        if not golden:
            return None, 0
        death_sorted = sorted(death)
        rates = []
        for g in golden:
            entry_date = dates[g]
            if (d - entry_date).days > validity_window:
                continue
            # the first death cross strictly after this golden = the exit.
            ex = next((dx for dx in death_sorted if dx > g), None)
            if ex is None:
                continue  # not yet completed -> not a closed validity sample
            c_entry = float(past[g])
            c_exit = float(past[ex])
            if c_entry <= 0 or np.isnan(c_entry):
                continue
            captured = (c_exit - c_entry) / c_entry
            # per-trade perfect-entry oracle within [g, ex]
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

    def _driver_rolling_validity(self, dates, closes, d_idx, max_days_back,
                                 validity_window, min_valid_trades):
        """Pick the config with the best rolling validity that is ALSO in position
        at D with a bounded entry; report that config's in-position entry.

        Tie-break: (rolling_validity desc, captured desc).  Returns
        (best_dict | None, used_fallback: bool).  best_dict adds
        rolling_validity + n_valid_trades.
        """
        d = dates[d_idx]
        past = closes[:d_idx + 1]
        cD = float(past[d_idx])
        candidates = []
        for (fam, f, s) in self.grid:
            golden, death = self._config_crosses(past, fam, f, s)
            if not golden:
                continue
            last_g = golden[-1]
            if any(dx > last_g for dx in death):
                continue  # not in position at D
            entry_date = dates[last_g]
            days_back = (d - entry_date).days
            if days_back > max_days_back:        # BOUNDED LOOKBACK
                continue
            c_entry = float(past[last_g])
            if c_entry <= 0 or np.isnan(c_entry):
                continue
            captured = (cD - c_entry) / c_entry
            rv, n_tr = self._config_rolling_validity(
                dates, closes, d_idx, fam, f, s, validity_window)
            candidates.append(dict(
                family=fam, fast=f, slow=s, entry_idx=last_g,
                entry_date=entry_date, days_back=int(days_back),
                captured=float(captured),
                rolling_validity=rv, n_valid_trades=n_tr))
        if not candidates:
            return None, False
        scored = [c for c in candidates
                  if c["rolling_validity"] is not None
                  and c["n_valid_trades"] >= min_valid_trades]
        if scored:
            scored.sort(key=lambda c: (c["rolling_validity"], c["captured"]),
                        reverse=True)
            return scored[0], False
        # fallback: no config has enough closed validity samples -> bounded one-shot
        candidates.sort(key=lambda c: c["captured"], reverse=True)
        best = candidates[0]
        best["rolling_validity"] = None
        return best, True

    # ---- one chart type's driver result for an asset --------------------
    def _driver_for_cadence(self, sym, d, cadence, max_days_back, mode,
                            validity_window, min_valid_trades):
        """Run the driver on one chart type. Returns a dict (or None if no data /
        no in-position bounded config)."""
        series = self._daily_series(sym, cadence)
        if series is None or len(series) == 0:
            return None
        dates = series["date"].to_list()
        closes = series["close"].to_numpy().astype(float)
        d_idx = _last_idx_le(dates, d)
        if d_idx is None:
            return None

        # perfect-entry oracle over [D - max_days_back, D] for capture_rate
        win_lo = max(0, d_idx - max_days_back)
        c_min = float(np.min(closes[win_lo:d_idx + 1]))
        cD = float(closes[d_idx])
        perfect_return = (cD - c_min) / c_min if c_min > 0 else 0.0

        used_fallback = False
        if mode == "rolling_validity":
            best, used_fallback = self._driver_rolling_validity(
                dates, closes, d_idx, max_days_back, validity_window,
                min_valid_trades)
        else:
            best = self._driver_bounded_oneshot(dates, closes, d_idx, max_days_back)

        if best is None:
            return dict(cadence=cadence, resolved_day=dates[d_idx],
                        best_ti=None, entry_date=None, days_back=None,
                        captured_return=0.0, perfect_return=round(perfect_return, 6),
                        capture_rate=0.0, rolling_validity=None,
                        n_valid_trades=0, used_fallback=False, in_position=False)

        cap_rate = 0.0
        if perfect_return > 0:
            cap_rate = max(0.0, min(1.0, best["captured"] / perfect_return))
        # peak day = the day of the max close in [entry, D] (where the move topped)
        e_idx = best["entry_idx"]
        seg_idx = int(e_idx + int(np.argmax(closes[e_idx:d_idx + 1])))
        peak_date = dates[seg_idx]
        return dict(
            cadence=cadence, resolved_day=dates[d_idx],
            best_ti=f"{best['family']}({best['fast']},{best['slow']})",
            family=best["family"], fast=best["fast"], slow=best["slow"],
            entry_date=best["entry_date"], days_back=best["days_back"],
            captured_return=round(best["captured"], 6),
            perfect_return=round(perfect_return, 6),
            capture_rate=round(cap_rate, 6),
            rolling_validity=(round(best["rolling_validity"], 6)
                              if best.get("rolling_validity") is not None else None),
            n_valid_trades=best.get("n_valid_trades", 0),
            used_fallback=used_fallback, in_position=True,
            peak_date=peak_date)

    # ---- the full decomposed record per asset ---------------------------
    def decompose(self, date, universe: str = "u50", lookback_days: int = 30,
                  top_n: int = 25, max_days_back: int | None = None,
                  driver_mode: str = "rolling_validity",
                  chart_types: tuple[str, ...] = ("1d", "dollar"),
                  validity_window: int = 365,
                  min_valid_trades: int = 3) -> pl.DataFrame:
        """Top-N performers as of `date`, one DNA record per (asset, best-TI move).

        driver_mode: 'rolling_validity' (elevated, default) | 'bounded_oneshot'.
        chart_types: the cadences to run the driver across; the FIRST is canonical
                     for the headline driver/context (default '1d').
        Returns a tidy DataFrame; also the source for the parquet/csv writers.
        """
        d = _to_date(date)
        if max_days_back is None:
            max_days_back = lookback_days
        primary = chart_types[0]

        ranked = self.engine.rank_top_performers(d, universe, lookback_days, top_n)
        rows = []
        for rank, (sym, perf) in enumerate(ranked, start=1):
            per_chart = {}
            for ct in chart_types:
                res = self._driver_for_cadence(
                    sym, d, ct, max_days_back, driver_mode,
                    validity_window, min_valid_trades)
                if res is not None:
                    per_chart[ct] = res

            head = per_chart.get(primary)
            rec: dict = {
                "sym": sym, "perf_rank": rank,
                "trailing_perf": round(float(perf), 6),
                "query_date": str(d),
                "driver_mode": driver_mode,
                "primary_chart": primary,
                "max_days_back": max_days_back,
                "hindsight": True,
            }

            # ---- DRIVER (primary chart) ----
            if head is None or not head.get("in_position"):
                rec.update({
                    "best_ti": None, "entry_date": None, "days_back": None,
                    "peak_date": None, "captured_return": 0.0,
                    "perfect_return": (round(head["perfect_return"], 6)
                                       if head else 0.0),
                    "capture_rate": 0.0, "rolling_validity": None,
                    "n_valid_trades": 0, "validity_fallback": False,
                    "note": "no bounded in-position config on primary chart",
                })
            else:
                rec.update({
                    "best_ti": head["best_ti"],
                    "entry_date": str(head["entry_date"]),
                    "days_back": head["days_back"],
                    "peak_date": str(head["peak_date"]),
                    "captured_return": head["captured_return"],
                    "perfect_return": head["perfect_return"],
                    "capture_rate": head["capture_rate"],
                    "rolling_validity": head["rolling_validity"],
                    "n_valid_trades": head["n_valid_trades"],
                    "validity_fallback": head["used_fallback"],
                    "note": "",
                })

            # ---- CONTEXT: chimera features as-of entry day + peak day ----
            ctx_cols = self._context_columns(sym)
            rec["context_cols"] = ",".join(ctx_cols) if ctx_cols else ""
            entry_d = head["entry_date"] if (head and head.get("in_position")) else None
            peak_d = head["peak_date"] if (head and head.get("in_position")) else None
            ctx_entry = self._features_as_of(sym, entry_d) if entry_d else {}
            ctx_peak = self._features_as_of(sym, peak_d) if peak_d else {}
            for c in ctx_cols:
                rec[f"ctx_entry__{c}"] = ctx_entry.get(c, None)
            for c in ctx_cols:
                rec[f"ctx_peak__{c}"] = ctx_peak.get(c, None)

            # ---- CHART-TYPE comparison ----
            for ct in chart_types:
                r = per_chart.get(ct)
                rec[f"chart__{ct}__best_ti"] = r["best_ti"] if r else None
                rec[f"chart__{ct}__entry_date"] = (
                    str(r["entry_date"]) if (r and r.get("entry_date")) else None)
                rec[f"chart__{ct}__days_back"] = r["days_back"] if r else None
                rec[f"chart__{ct}__captured_return"] = (
                    r["captured_return"] if r else None)
                rec[f"chart__{ct}__capture_rate"] = r["capture_rate"] if r else None

            rows.append(rec)

        if not rows:
            return pl.DataFrame()
        # union schema (records can have different context-col sets per asset)
        return pl.DataFrame(rows, infer_schema_length=None).sort("perf_rank")


# ---- writers / preview --------------------------------------------------
def _write_outputs(table: pl.DataFrame, date_str: str, out_parquet: str | None):
    out_pq = out_parquet or str(
        PROJECT_ROOT / "runs" / "oracle" / f"decomposer_{date_str}.parquet")
    Path(out_pq).parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_pq)
    out_csv = out_pq.rsplit(".", 1)[0] + ".csv"
    # CSV preview: drop the wide ctx_* columns for readability; keep a digest.
    keep = [c for c in table.columns if not c.startswith("ctx_")]
    table.select(keep).write_csv(out_csv)
    return out_pq, out_csv


def _print_decomposed_record(table: pl.DataFrame, chart_types) -> None:
    """Print ONE top performer's full decomposed record in ASCII."""
    if table.is_empty():
        print("(no rows)")
        return
    # first in-position row, else row 0
    in_pos = table.filter(pl.col("best_ti").is_not_null())
    row = (in_pos.row(0, named=True) if not in_pos.is_empty()
           else table.row(0, named=True))
    print("=" * 72)
    print(f"FULL DECOMPOSED RECORD -- {row['sym']} (rank {row['perf_rank']})")
    print("=" * 72)
    print("-- DRIVER (bounded, %s) --" % row["driver_mode"])
    for k in ("best_ti", "entry_date", "days_back", "peak_date",
              "captured_return", "perfect_return", "capture_rate",
              "rolling_validity", "n_valid_trades", "validity_fallback",
              "max_days_back", "note"):
        print(f"  {k:18s}: {row.get(k)}")
    print("-- CONTEXT (chimera features AS-OF entry day) --")
    ent = {c[len('ctx_entry__'):]: row[c] for c in table.columns
           if c.startswith("ctx_entry__") and row[c] is not None}
    for k in sorted(ent):
        print(f"  {k:24s}: {ent[k]}")
    print("  (... features as-of PEAK day also attached: ctx_peak__*)")
    print("-- CHART-TYPE comparison --")
    hdr = f"  {'chart':10s} {'best_ti':14s} {'entry':12s} {'days_back':10s} {'captured':10s} {'cap_rate':9s}"
    print(hdr)
    for ct in chart_types:
        print(f"  {ct:10s} "
              f"{str(row.get(f'chart__{ct}__best_ti')):14s} "
              f"{str(row.get(f'chart__{ct}__entry_date')):12s} "
              f"{str(row.get(f'chart__{ct}__days_back')):10s} "
              f"{str(row.get(f'chart__{ct}__captured_return')):10s} "
              f"{str(row.get(f'chart__{ct}__capture_rate')):9s}")


def main():
    ap = argparse.ArgumentParser(description=HINDSIGHT_LABEL)
    ap.add_argument("--date", required=True, help="query day D, YYYY-MM-DD")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--lookback", type=int, default=30,
                    help="trailing-return ranking window")
    ap.add_argument("--max-days-back", type=int, default=None,
                    help="bound on the driver entry's days_back (default=lookback)")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--driver-mode", default="rolling_validity",
                    choices=["rolling_validity", "bounded_oneshot"])
    ap.add_argument("--chart-types", default="1d,dollar",
                    help="comma list; first is canonical (default 1d,dollar)")
    ap.add_argument("--validity-window", type=int, default=365)
    ap.add_argument("--min-valid-trades", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    chart_types = tuple(c.strip() for c in args.chart_types.split(",") if c.strip())
    dec = OracleDecomposer()
    table = dec.decompose(
        args.date, args.universe, args.lookback, args.top_n,
        max_days_back=args.max_days_back, driver_mode=args.driver_mode,
        chart_types=chart_types, validity_window=args.validity_window,
        min_valid_trades=args.min_valid_trades)

    print("=" * 72)
    print(HINDSIGHT_LABEL)
    mdb = args.max_days_back if args.max_days_back is not None else args.lookback
    print(f"DECOMPOSER date={args.date} universe={args.universe} "
          f"lookback={args.lookback}d max_days_back={mdb}d")
    print(f"driver_mode={args.driver_mode} chart_types={list(chart_types)} "
          f"top_n={args.top_n}")
    print("=" * 72)
    if table.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return

    if "days_back" in table.columns:
        dbs = [v for v in table["days_back"].to_list() if v is not None]
        if dbs:
            print(f"days_back across {len(dbs)} in-position assets: "
                  f"min={min(dbs)} max={max(dbs)} (bound={mdb})")
    out_pq, out_csv = _write_outputs(table, args.date, args.out)
    print(f"wrote: {out_pq}")
    print(f"wrote: {out_csv}")
    print()
    _print_decomposed_record(table, chart_types)


if __name__ == "__main__":
    main()
