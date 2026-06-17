"""Oracle panel / sweep builder -- persisted indexed store.

================================================================================
ORACLE PANEL -- incremental per-(universe, indicator, cadence) sweep store.
================================================================================

Builds a date-indexed panel by calling OracleEngine().oracle(date, ...) for a
list of dates and persisting all rows in a single parquet store keyed by
``query_date``. Designed for sweeping ranges (e.g., every 7th day over a year)
and for incremental top-up: dates already present in the store are skipped
(skip_existing=True), so re-running after new data arrives is cheap.

Store layout::

    runs/oracle/panel/panel_<universe>_<indicator>_<cadence>.parquet

Each row carries all columns emitted by OracleEngine.oracle() plus a
``query_date`` column (the date that was passed to oracle()).

Atomic writes (via pipeline.parquet_io.atomic_write_parquet) ensure the on-disk
store is never partially written.

Usage::

    from oracle.panel import build_panel, load_panel, date_range

    dates = date_range("2026-01-01", "2026-06-01", step_days=7)
    df = build_panel(dates, universe="u10", cadence="1d")
    # later top-up:
    df = build_panel(["2026-06-08"], universe="u10", cadence="1d")

CLI::

    python src/oracle/panel.py --start 2026-01-01 --end 2026-06-01 --step-days 7
        --universe u10 --indicator ma --cadence 1d

HINDSIGHT label: every row inherits hindsight=True from OracleEngine.oracle().
This store is a HINDSIGHT UPPER BOUND -- not tradeable signals.

No emoji in print statements (cp1252 safe).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date, timedelta as _timedelta
from pathlib import Path
from typing import Sequence, Tuple, Union

import polars as pl

__contract__ = {
    "kind": "oracle_panel_store",
    "inputs": [
        "oracle.engine.OracleEngine (per-date oracle)",
        "list[str] dates OR (start, end, step_days) range",
    ],
    "outputs": {
        "callable": "build_panel(dates, *, universe, indicator, cadence, ...) -> pl.DataFrame",
        "parquet": "runs/oracle/panel/panel_<universe>_<indicator>_<cadence>.parquet",
    },
    "invariants": [
        "one store per (universe, indicator, cadence) tuple",
        "incremental / skip-existing: dates already in store are not re-queried",
        "atomic write: tmp + os.replace; never a partial store on disk",
        "query_date column (str YYYY-MM-DD) indexes every row",
        "dates with no oracle rows are skipped + logged, no crash",
        "output is HINDSIGHT UPPER BOUND (hindsight=True on every row)",
        "no emoji in prints (cp1252 safe)",
    ],
}

# ---------------------------------------------------------------------------
# Bootstrap sys.path so this module can be run as __main__ from repo root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SRC), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from oracle.engine import OracleEngine  # noqa: E402

# Try to import the canonical atomic writer; fall back to a local impl.
try:
    from pipeline.parquet_io import atomic_write_parquet as _atomic_write
except ImportError:
    import os as _os

    def _atomic_write(df: pl.DataFrame, out_path: Path, **_kw) -> Path:  # type: ignore[misc]
        """Minimal fallback: write to .tmp then os.replace."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            df.write_parquet(tmp)
            _os.replace(str(tmp), str(out_path))
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise
        return out_path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def date_range(start: str, end: str, step_days: int = 1) -> list[str]:
    """Emit dates from ``start`` to ``end`` (inclusive) with ``step_days`` step.

    Args:
        start: 'YYYY-MM-DD' start date (inclusive).
        end:   'YYYY-MM-DD' end date (inclusive).
        step_days: calendar-day step between successive dates (default 1).

    Returns:
        List of 'YYYY-MM-DD' strings.
    """
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1; got {step_days}")
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    if d0 > d1:
        return []
    out: list[str] = []
    cur = d0
    step = _timedelta(days=step_days)
    while cur <= d1:
        out.append(cur.isoformat())
        cur += step
    return out


def _store_path(out_dir: Union[str, Path],
                universe: str, indicator: str, cadence: str) -> Path:
    """Canonical path for the (universe, indicator, cadence) store."""
    safe_cad = cadence.replace("/", "-")
    name = f"panel_{universe}_{indicator}_{safe_cad}.parquet"
    return Path(out_dir) / name


def _existing_query_dates(store: Path) -> set[str]:
    """Return the set of query_date strings already in the store.

    Returns empty set if the store does not exist or cannot be read.
    """
    if not store.exists():
        return set()
    try:
        schema = pl.read_parquet_schema(store)
        if "query_date" not in schema:
            print(f"[panel] WARNING: store exists but has no 'query_date' column; "
                  f"treating as empty ({store.name})", flush=True)
            return set()
        df = pl.read_parquet(store, columns=["query_date"])
        return set(df["query_date"].cast(pl.Utf8).to_list())
    except Exception as exc:
        print(f"[panel] WARNING: could not read existing store "
              f"({store.name}): {type(exc).__name__}: {exc}; treating as empty",
              flush=True)
        return set()


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_panel(
    dates: Union[Sequence[str], Tuple[str, str, int]],
    *,
    universe: str = "u50",
    indicator: str = "ma",
    cadence: str = "1d",
    lookback_days: int = 30,
    top_n: int = 25,
    validity_windows: Tuple[int, ...] = (180, 365),
    driver: str = "rolling_validity",
    out_dir: Union[str, Path] = "runs/oracle/panel",
    skip_existing: bool = True,
) -> pl.DataFrame:
    """Build (or top-up) the oracle panel store for (universe, indicator, cadence).

    For each date in ``dates``, call ``OracleEngine().oracle(date, ...)`` and
    append all rows (with a ``query_date`` column) into the persisted store.

    The store is a single parquet file at::

        <out_dir>/panel_<universe>_<indicator>_<cadence>.parquet

    Incremental / skip-existing
    ---------------------------
    When ``skip_existing=True`` (default), any ``query_date`` already present in
    the on-disk store is NOT re-queried. Only the missing dates are run, then the
    new rows are concatenated with the existing store and atomically written back.

    Date specification
    ------------------
    ``dates`` may be:

    - A list / sequence of 'YYYY-MM-DD' strings.
    - A 3-tuple ``(start, end, step_days)`` which is expanded via
      :func:`date_range`.

    Args:
        dates: dates to query (list of ISO strings, or (start, end, step) tuple).
        universe: universe key ('u10', 'u50', 'u100').
        indicator: indicator key ('ma'; 'rsi'/'macd'/'bollinger' are registered
                   placeholders -- calling them raises NotImplementedError).
        cadence: '1d' or an event cadence (dollar / dib / ...).
        lookback_days: passed to OracleEngine.oracle().
        top_n: top-N performers per date.
        validity_windows: rolling-validity windows tried in order.
        driver: 'rolling_validity' (default) | 'bounded_oneshot'.
        out_dir: directory for the store. Resolved relative to repo root when
                 the path is relative (cwd at call time).
        skip_existing: if True, skip dates already in the store (incremental).

    Returns:
        The full panel DataFrame (all dates, including previously-stored ones).
        Empty DataFrame if no dates produce any rows.
    """
    # -- Expand (start, end, step) tuple form ---------------------------------
    if isinstance(dates, tuple) and len(dates) == 3:
        dates = date_range(str(dates[0]), str(dates[1]), int(dates[2]))

    dates_list: list[str] = [str(d) for d in dates]
    if not dates_list:
        print("[panel] No dates provided; nothing to do.", flush=True)
        existing = load_panel(universe, indicator, cadence, out_dir=out_dir)
        return existing if existing is not None else pl.DataFrame()

    # -- Resolve out_dir (relative -> absolute from repo root) ----------------
    out_dir_path = Path(out_dir)
    if not out_dir_path.is_absolute():
        out_dir_path = _PROJECT_ROOT / out_dir_path
    out_dir_path.mkdir(parents=True, exist_ok=True)
    store = _store_path(out_dir_path, universe, indicator, cadence)

    # -- Determine which dates to actually run --------------------------------
    already_done: set[str] = set()
    if skip_existing:
        already_done = _existing_query_dates(store)
        if already_done:
            to_skip = [d for d in dates_list if d in already_done]
            print(f"[panel] skip_existing: {len(to_skip)}/{len(dates_list)} dates "
                  f"already in store; running {len(dates_list) - len(to_skip)} new.",
                  flush=True)

    to_run = [d for d in dates_list if d not in already_done]

    # -- Run missing dates ----------------------------------------------------
    engine = OracleEngine()
    new_frames: list[pl.DataFrame] = []
    vw = tuple(validity_windows)

    for d_str in to_run:
        try:
            frame = engine.oracle(
                d_str,
                universe=universe,
                indicator=indicator,
                cadence=cadence,
                lookback_days=lookback_days,
                top_n=top_n,
                validity_windows=vw,
                driver=driver,
            )
        except Exception as exc:
            print(f"[panel] SKIP {d_str}: oracle raised "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue

        if frame is None or (hasattr(frame, "is_empty") and frame.is_empty()):
            print(f"[panel] SKIP {d_str}: oracle returned no rows "
                  f"(insufficient data)", flush=True)
            continue

        # Stamp query_date as the first column for easy inspection.
        frame = frame.with_columns(
            pl.lit(d_str).alias("query_date")
        ).select(["query_date"] + [c for c in frame.columns if c != "query_date"])
        new_frames.append(frame)

    n_new = sum(len(f) for f in new_frames)
    print(f"[panel] Ran {len(to_run)} dates, got {n_new} new rows.", flush=True)

    # -- Merge with existing store + atomic-write -----------------------------
    if new_frames:
        new_df = pl.concat(new_frames, how="vertical_relaxed")

        if store.exists() and already_done:
            # Read existing, drop any rows whose query_date is in the new batch
            # (idempotency: if somehow the same date appears, last write wins).
            try:
                existing_df = pl.read_parquet(store)
                new_dates_set = set(new_df["query_date"].to_list())
                keep = existing_df.filter(
                    ~pl.col("query_date").is_in(list(new_dates_set))
                )
                full_df = pl.concat([keep, new_df],
                                     how="vertical_relaxed").sort("query_date")
            except Exception as exc:
                print(f"[panel] WARNING: could not merge with existing store "
                      f"({type(exc).__name__}: {exc}); writing new rows only.",
                      flush=True)
                full_df = new_df.sort("query_date")
        else:
            full_df = new_df.sort("query_date")

        _atomic_write(full_df, store)
        print(f"[panel] Store written: {store} ({len(full_df)} rows, "
              f"{full_df['query_date'].n_unique()} query dates)", flush=True)
    else:
        # Nothing new -- just return what was already stored.
        if store.exists():
            try:
                full_df = pl.read_parquet(store)
            except Exception as exc:
                print(f"[panel] WARNING: store unreadable "
                      f"({type(exc).__name__}: {exc}); returning empty.",
                      flush=True)
                full_df = pl.DataFrame()
        else:
            print("[panel] No new rows and no existing store; returning empty.",
                  flush=True)
            full_df = pl.DataFrame()

    return full_df


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_panel(
    universe: str,
    indicator: str,
    cadence: str,
    out_dir: Union[str, Path] = "runs/oracle/panel",
) -> "pl.DataFrame | None":
    """Load the persisted oracle panel for (universe, indicator, cadence).

    Returns the full DataFrame, or None if the store does not exist.
    """
    out_dir_path = Path(out_dir)
    if not out_dir_path.is_absolute():
        out_dir_path = _PROJECT_ROOT / out_dir_path
    store = _store_path(out_dir_path, universe, indicator, cadence)
    if not store.exists():
        print(f"[panel] store not found: {store}", flush=True)
        return None
    try:
        df = pl.read_parquet(store)
        print(f"[panel] loaded {store.name}: {df.shape} "
              f"({df['query_date'].n_unique() if 'query_date' in df.columns else '?'} "
              f"query dates)", flush=True)
        return df
    except Exception as exc:
        print(f"[panel] ERROR reading store ({store.name}): "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Oracle panel builder: sweep a date range, store results in a "
            "persisted indexed parquet. HINDSIGHT UPPER BOUND -- not a tradeable "
            "signal."
        )
    )
    date_grp = ap.add_mutually_exclusive_group()
    date_grp.add_argument(
        "--dates", default=None,
        help="Comma-separated list of YYYY-MM-DD dates to query.",
    )
    ap.add_argument("--start", default=None,
                    help="Range start (YYYY-MM-DD). Used with --end [--step-days].")
    ap.add_argument("--end", default=None,
                    help="Range end (YYYY-MM-DD, inclusive).")
    ap.add_argument("--step-days", type=int, default=1,
                    help="Calendar-day step for --start/--end range (default 1).")
    ap.add_argument("--universe", default="u50",
                    help="Universe key: u10, u50, u100 (default u50).")
    ap.add_argument("--indicator", default="ma",
                    help="Indicator key: ma (rsi/macd/bollinger are stubs).")
    ap.add_argument("--cadence", default="1d",
                    help="Bar cadence: 1d or event cadence (default 1d).")
    ap.add_argument("--lookback", type=int, default=30,
                    help="Lookback days for ranking + entry bound (default 30).")
    ap.add_argument("--top-n", type=int, default=25,
                    help="Top-N performers per date (default 25).")
    ap.add_argument("--validity-windows", default="180,365",
                    help="Comma list of validity windows in days (default 180,365).")
    ap.add_argument("--driver", default="rolling_validity",
                    choices=["rolling_validity", "bounded_oneshot"],
                    help="Driver: rolling_validity (default) or bounded_oneshot.")
    ap.add_argument("--out-dir", default="runs/oracle/panel",
                    help="Output directory for the store "
                         "(default runs/oracle/panel).")
    ap.add_argument("--no-skip", action="store_true",
                    help="Disable skip_existing -- re-run all dates.")
    ap.add_argument("--load-only", action="store_true",
                    help="Just load and print the existing store; no build.")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    vws = tuple(int(x.strip()) for x in args.validity_windows.split(",")
                if x.strip())

    if args.load_only:
        df = load_panel(args.universe, args.indicator, args.cadence,
                        out_dir=args.out_dir)
        if df is None:
            print("(no store found)")
        else:
            print(df)
        return

    # -- Resolve dates -------------------------------------------------------
    if args.dates:
        dates_list = [d.strip() for d in args.dates.split(",") if d.strip()]
    elif args.start and args.end:
        dates_list = date_range(args.start, args.end, args.step_days)
    else:
        print("ERROR: provide either --dates d1,d2,.. OR --start + --end",
              flush=True)
        sys.exit(1)

    print(f"[panel] Building panel for {len(dates_list)} dates | "
          f"universe={args.universe} indicator={args.indicator} "
          f"cadence={args.cadence} lookback={args.lookback}d "
          f"top_n={args.top_n} validity_windows={list(vws)} "
          f"driver={args.driver}", flush=True)

    df = build_panel(
        dates_list,
        universe=args.universe,
        indicator=args.indicator,
        cadence=args.cadence,
        lookback_days=args.lookback,
        top_n=args.top_n,
        validity_windows=vws,
        driver=args.driver,
        out_dir=args.out_dir,
        skip_existing=not args.no_skip,
    )

    if df.is_empty():
        print("[panel] Result: empty panel.")
    else:
        print(f"\n[panel] Result: shape={df.shape}")
        print(df.head(10))


if __name__ == "__main__":
    main()
