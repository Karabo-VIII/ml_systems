"""Oracle system unified entry point.

ONE-COMMAND access to the full oracle system without memorising 4 module CLIs.

    python src/oracle/run.py [subcommand] [options]

SUBCOMMANDS
-----------
  compare  (default) -- headline: oracle (hindsight right answer) vs model
                        (past-only choice) side by side for a single date.
  oracle              -- hindsight upper bound table for a single date.
  model               -- adaptive (past-only) model picks for a single date.
  dna                 -- DNA decoupling: attach full context to each oracle row.
  sweep               -- compare_grid leaderboard over a date range.
  doctor              -- preflight: import check, data load check, date range.

If no subcommand is given, 'compare' is used.

DEFAULT UNIVERSE: u10 (overridable via --universe on every subcommand).
DEFAULT DATE: the max valid date from data_date_range (auto-resolved per run).

ALL output is plain ASCII rows -- no Unicode box chars, no polars print().
No emoji anywhere (cp1252 safety).

USAGE EXAMPLES
--------------
  python src/oracle/run.py doctor --universe u10
  python src/oracle/run.py compare --date 2026-05-20 --universe u10
  python src/oracle/run.py oracle  --date 2026-05-20 --universe u10
  python src/oracle/run.py model   --date 2026-05-20 --universe u10
  python src/oracle/run.py dna     --date 2026-05-20 --universe u10
  python src/oracle/run.py sweep   --start 2026-05-01 --end 2026-05-20
                                   --step-days 5 --universe u10
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path so this file is runnable from the repo root without
# extra PYTHONPATH fiddling (the engines already do this themselves, but we
# do it here too so imports below work before any engine is imported).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent   # repo root
_SRC = _PROJECT_ROOT / "src"
for _p in (str(_SRC), str(_SRC / "pipeline"), str(_SRC / "oracle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__contract__ = {
    "kind": "oracle_unified_entrypoint",
    "inputs": [
        "oracle.engine.OracleEngine.oracle",
        "oracle.adaptive.AdaptiveChooser.choose / choose_all",
        "oracle.compare.OracleVsModel.compare / compare_grid / side_by_side",
        "oracle.dna.decouple",
        "oracle.ma_oracle_engine.MAOracleEngine.data_date_range",
        "pipeline.chimera_loader.ChimeraLoader",
    ],
    "outputs": {
        "oracle subcommand": "ASCII table to stdout",
        "model subcommand": "ASCII table to stdout",
        "compare subcommand": "ASCII side-by-side table + summary to stdout",
        "dna subcommand": "shape printed + parquet written to runs/oracle/",
        "sweep subcommand": "ASCII leaderboard to stdout + CSV to runs/oracle/",
        "doctor subcommand": "preflight report to stdout, exit 0 OK / exit 2 FAIL",
    },
    "invariants": [
        "no emoji in prints (cp1252-safe)",
        "no print(df) / no Unicode box chars anywhere",
        "default universe is u10",
        "default date auto-resolved to max valid date from data_date_range",
        "exits with code 2 on hard failure (doctor), 1 on soft warning",
    ],
}

# ============================================================================
# SHARED HELPERS
# ============================================================================

def _to_date(d) -> _date:
    """Parse any date-like value to a date object."""
    if isinstance(d, _date):
        return d
    return _date.fromisoformat(str(d))


def _print_rows(df, max_rows: int = 50) -> None:
    """Plain-ASCII table print (no polars print, no Unicode box chars).

    Safe on Windows cp1252 stdout.  Mirrors _print_table_ascii from
    ma_oracle_engine but RETURNS nothing and takes max_rows as a cap.
    """
    if df is None or df.is_empty():
        print("(empty table)")
        return
    cols = df.columns
    rows = df.head(max_rows).rows()
    cells = [
        [("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
         for v in row]
        for row in rows
    ]
    widths = [len(c) for c in cols]
    for row in cells:
        for i, v in enumerate(row):
            if len(v) > widths[i]:
                widths[i] = len(v)
    sep = "-+-".join("-" * w for w in widths)
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(header)
    print(sep)
    for row in cells:
        print(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)))
    if len(df) > max_rows:
        print(f"... ({len(df) - max_rows} more rows hidden, showing first {max_rows})")


def _resolve_default_date(universe: str) -> str | None:
    """Return the max valid date string from data_date_range for the universe.

    Returns None if the date range cannot be resolved (data missing).
    This is the auto-date when the user omits --date.
    """
    try:
        from oracle.ma_oracle_engine import MAOracleEngine
        from pipeline.chimera_loader import ChimeraLoader
        eng = MAOracleEngine(ChimeraLoader())
        _, max_d = eng.data_date_range(universe)
        if max_d is None:
            return None
        return str(max_d)
    except Exception:
        return None


def _ensure_date(date_str: str | None, universe: str) -> str:
    """Return a usable date string, auto-resolving if date_str is None."""
    if date_str is not None:
        return date_str
    resolved = _resolve_default_date(universe)
    if resolved is None:
        print("ERROR: --date not provided and auto-resolution from data_date_range failed.")
        print("  Ensure chimera data is present for the universe, or pass --date explicitly.")
        sys.exit(2)
    print(f"(auto-resolved --date to max valid date: {resolved})")
    return resolved


def _daterange(start: str, end: str, step_days: int) -> list[_date]:
    """Generate a list of dates from start to end (inclusive) stepping by step_days."""
    s, e = _to_date(start), _to_date(end)
    out = []
    cur = s
    while cur <= e:
        out.append(cur)
        cur = cur + timedelta(days=step_days)
    return out


# ============================================================================
# SUBCOMMAND: oracle
# ============================================================================

def cmd_oracle(args) -> None:
    """Hindsight upper-bound oracle table for a single date."""
    from oracle.engine import OracleEngine
    from oracle.ma_oracle_engine import _print_table_ascii

    date = _ensure_date(args.date, args.universe)
    vws = tuple(int(x.strip()) for x in args.validity_windows.split(",") if x.strip())

    print("=" * 80)
    print("ORACLE (hindsight upper bound) -- descriptive, NOT a tradeable signal.")
    print(f"date={date} universe={args.universe} indicator={args.indicator} "
          f"cadence={args.cadence}")
    print(f"lookback={args.lookback}d top_n={args.top_n} "
          f"validity_windows={list(vws)} driver={args.driver}")
    print("=" * 80)

    engine = OracleEngine()
    table = engine.oracle(
        date,
        universe=args.universe,
        indicator=args.indicator,
        cadence=args.cadence,
        lookback_days=args.lookback,
        top_n=args.top_n,
        validity_windows=vws,
        driver=args.driver,
    )

    if table.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return

    _print_rows(table, max_rows=args.top_n)


# ============================================================================
# SUBCOMMAND: model
# ============================================================================

def cmd_model(args) -> None:
    """Adaptive (past-only) model picks for a single date."""
    from oracle.adaptive import AdaptiveChooser, MECHANISMS

    date = _ensure_date(args.date, args.universe)

    print("=" * 80)
    print("ADAPTIVE MODEL (past-only / realizable) -- forward-testable pick.")
    print(f"date={date} universe={args.universe} cadence={args.cadence} "
          f"indicator={args.indicator}")
    print(f"mechanism={args.mechanism} validity_window={args.validity_window}d "
          f"lookback={args.lookback}d top_n={args.top_n}")
    print("=" * 80)

    chooser = AdaptiveChooser()
    table = chooser.choose(
        date,
        universe=args.universe,
        cadence=args.cadence,
        validity_window=args.validity_window,
        mechanism=args.mechanism,
        indicator=args.indicator,
        lookback_days=args.lookback,
        top_n=args.top_n,
    )

    if table.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return

    n_in = int(table["in_position_at_D"].sum()) if "in_position_at_D" in table.columns else "?"
    print(f"in_position_at_D picks: {n_in} / {len(table)}")
    _print_rows(table, max_rows=args.top_n)


# ============================================================================
# SUBCOMMAND: compare (DEFAULT)
# ============================================================================

def cmd_compare(args) -> None:
    """Oracle vs model side-by-side for a single date -- the headline subcommand."""
    from oracle.compare import OracleVsModel

    date = _ensure_date(args.date, args.universe)

    cmp = OracleVsModel()
    cmp.side_by_side(
        date,
        universe=args.universe,
        cadence=args.cadence,
        validity_window=args.validity_window,
        mechanism=args.mechanism,
        indicator=args.indicator,
        lookback_days=args.lookback,
        top_n=args.top_n,
    )


# ============================================================================
# SUBCOMMAND: dna
# ============================================================================

def cmd_dna(args) -> None:
    """DNA decoupling: run decouple() and save parquet + print shape."""
    from oracle.dna import decouple, _out_path

    date = _ensure_date(args.date, args.universe)
    vws = tuple(int(x.strip()) for x in args.validity_windows.split(",") if x.strip())
    cts = tuple(c.strip() for c in args.chart_types.split(",") if c.strip())

    print("=" * 72)
    print("DNA DECOUPLE -- hindsight, descriptive, NOT a tradeable signal.")
    print(f"date={date} universe={args.universe} indicator={args.indicator} "
          f"cadence={args.cadence}")
    print(f"lookback={args.lookback}d top_n={args.top_n} "
          f"validity_windows={list(vws)} chart_types={list(cts)}")
    print("=" * 72)

    result = decouple(
        date,
        universe=args.universe,
        indicator=args.indicator,
        cadence=args.cadence,
        lookback_days=args.lookback,
        top_n=args.top_n,
        validity_windows=vws,
        chart_types=cts,
        include_features=(not args.no_features),
        include_regime=(not args.no_regime),
    )

    if result.is_empty():
        print("(no rows -- no assets cover this date with >= lookback history)")
        return

    out_path = _out_path(args.universe, args.indicator, args.cadence, date, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(out_path))

    print(f"shape : {result.shape}")
    print(f"wrote : {out_path}")
    # Print a few key columns to give a sense of the data without polars print().
    key_cols = [c for c in ["sym", "perf_rank", "entry_date", "days_back",
                             "captured_return", "capture_rate"]
                if c in result.columns]
    print("\n-- key columns (first rows) --")
    _print_rows(result.select(key_cols), max_rows=min(10, len(result)))


# ============================================================================
# SUBCOMMAND: sweep
# ============================================================================

def cmd_sweep(args) -> None:
    """compare_grid leaderboard over a date range + save CSV."""
    from oracle.compare import OracleVsModel, _ascii_table_str

    if not args.start or not args.end:
        print("ERROR: sweep requires --start and --end")
        sys.exit(1)

    dates = _daterange(args.start, args.end, args.step_days)
    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]
    mechanisms = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    print("=" * 100)
    print("SWEEP -- compare_grid leaderboard over date range.")
    print(f"range={args.start} to {args.end} step={args.step_days}d "
          f"({len(dates)} dates)  universe={args.universe}")
    print(f"cadences={cadences} windows={windows} mechanisms={mechanisms}")
    print("=" * 100)

    cmp = OracleVsModel()
    lb = cmp.compare_grid(
        dates,
        universe=args.universe,
        cadences=tuple(cadences),
        validity_windows=tuple(windows),
        mechanisms=tuple(mechanisms),
        indicator=args.indicator,
        lookback_days=args.lookback,
        top_n=args.top_n,
    )

    if lb.is_empty():
        print("(empty leaderboard -- no dates had usable data)")
        return

    print(_ascii_table_str(lb, max_rows=100))

    out_path = (_PROJECT_ROOT / "runs" / "oracle" /
                f"leaderboard_{args.universe}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lb.write_csv(str(out_path))
    print(f"\nwrote: {out_path}")


# ============================================================================
# SUBCOMMAND: doctor
# ============================================================================

def cmd_doctor(args) -> None:
    """Preflight check: import each engine, load data, print date range.

    Exits 0 if all pass, 2 on hard fail.
    """
    errors = []
    warnings = []

    print("=" * 72)
    print("ORACLE DOCTOR -- preflight checks")
    print(f"universe={args.universe}")
    print("=" * 72)

    # --- 1. import checks ---
    engines = [
        ("oracle.engine.OracleEngine", "oracle.engine", "OracleEngine"),
        ("oracle.adaptive.AdaptiveChooser", "oracle.adaptive", "AdaptiveChooser"),
        ("oracle.compare.OracleVsModel", "oracle.compare", "OracleVsModel"),
        ("oracle.dna.decouple", "oracle.dna", "decouple"),
        ("oracle.ma_oracle_engine.MAOracleEngine", "oracle.ma_oracle_engine", "MAOracleEngine"),
    ]
    for label, modname, objname in engines:
        try:
            import importlib
            mod = importlib.import_module(modname)
            getattr(mod, objname)
            print(f"  [OK] import {label}")
        except Exception as exc:
            msg = f"import {label}: {exc}"
            errors.append(msg)
            print(f"  [FAIL] {msg}")

    # --- 2. ChimeraLoader import ---
    try:
        from pipeline.chimera_loader import ChimeraLoader
        print("  [OK] import pipeline.chimera_loader.ChimeraLoader")
    except Exception as exc:
        msg = f"import ChimeraLoader: {exc}"
        errors.append(msg)
        print(f"  [FAIL] {msg}")
        # If ChimeraLoader fails, nothing else will work.
        print("\nHARD FAIL -- cannot continue without ChimeraLoader.")
        sys.exit(2)

    # --- 3. BTCUSDT 1d data load ---
    try:
        loader = ChimeraLoader()
        df = loader.load("BTCUSDT", cadence="1d", features=["close", "date"])
        n_rows = len(df) if df is not None else 0
        if n_rows == 0:
            msg = "BTCUSDT 1d loaded but has 0 rows"
            warnings.append(msg)
            print(f"  [WARN] {msg}")
        else:
            print(f"  [OK] ChimeraLoader.load('BTCUSDT','1d') -> {n_rows} rows")
    except Exception as exc:
        msg = f"ChimeraLoader.load('BTCUSDT','1d') failed: {exc}"
        errors.append(msg)
        print(f"  [FAIL] {msg}")

    # --- 4. data_date_range for the universe ---
    try:
        from oracle.ma_oracle_engine import MAOracleEngine
        eng = MAOracleEngine(ChimeraLoader())
        min_d, max_d = eng.data_date_range(args.universe)
        if min_d is None or max_d is None:
            msg = f"data_date_range({args.universe}) returned None -- no loadable assets?"
            warnings.append(msg)
            print(f"  [WARN] {msg}")
        else:
            print(f"  [OK] data_date_range({args.universe}) = {min_d} to {max_d}")
            # --- 5. date-in-range check (if a date was supplied) ---
            if args.date:
                req = _to_date(args.date)
                if req < min_d:
                    msg = (f"requested date {req} is BEFORE data range start {min_d} "
                           f"-- queries on this date will return empty tables")
                    warnings.append(msg)
                    print(f"  [WARN] {msg}")
                elif req > max_d:
                    msg = (f"requested date {req} is AFTER data range end {max_d} "
                           f"-- queries on this date will return empty tables")
                    warnings.append(msg)
                    print(f"  [WARN] {msg}")
                else:
                    print(f"  [OK] requested date {req} is within data range")
    except Exception as exc:
        msg = f"data_date_range({args.universe}) failed: {exc}"
        errors.append(msg)
        print(f"  [FAIL] {msg}")

    # --- summary ---
    print("-" * 72)
    if errors:
        print(f"RESULT: FAIL -- {len(errors)} error(s), {len(warnings)} warning(s)")
        for e in errors:
            print(f"  ERROR  : {e}")
        for w in warnings:
            print(f"  WARNING: {w}")
        sys.exit(2)
    elif warnings:
        print(f"RESULT: PASS with {len(warnings)} warning(s)")
        for w in warnings:
            print(f"  WARNING: {w}")
        sys.exit(0)
    else:
        print("RESULT: PASS -- all checks OK")
        sys.exit(0)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared across most subcommands."""
    p.add_argument("--date", default=None,
                   help="query day D, YYYY-MM-DD "
                        "(default: max valid date from data_date_range)")
    p.add_argument("--universe", default="u10",
                   help="universe key: u10 / u50 / u100 (default u10)")
    p.add_argument("--indicator", default="ma",
                   help="indicator family: ma (default; rsi/macd/bollinger are stubs)")
    p.add_argument("--cadence", default="1d",
                   help="bar cadence: 1d or event cadence (default 1d)")
    p.add_argument("--lookback", type=int, default=30,
                   help="trailing return window + entry bound in days (default 30)")
    p.add_argument("--top-n", type=int, default=25,
                   help="number of top performers (default 25)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python src/oracle/run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="subcommand")

    # ---- oracle ----
    p_oracle = sub.add_parser("oracle", help="hindsight upper-bound table")
    _add_common_args(p_oracle)
    p_oracle.add_argument("--validity-windows", default="180,365",
                          help="comma list of validity windows (days) (default 180,365)")
    p_oracle.add_argument("--driver", default="rolling_validity",
                          choices=["rolling_validity", "bounded_oneshot"])

    # ---- model ----
    p_model = sub.add_parser("model", help="adaptive (past-only) model picks")
    _add_common_args(p_model)
    p_model.add_argument("--validity-window", type=int, default=365,
                         help="rolling validity window in days (default 365)")
    p_model.add_argument("--mechanism", default="rolling_validity",
                         choices=["rolling_validity", "regime_cond", "state_cond"],
                         help="selection mechanism (default rolling_validity)")

    # ---- compare (default) ----
    p_cmp = sub.add_parser("compare",
                           help="oracle vs model side-by-side (default subcommand)")
    _add_common_args(p_cmp)
    p_cmp.add_argument("--validity-window", type=int, default=365,
                       help="rolling validity window in days (default 365)")
    p_cmp.add_argument("--mechanism", default="rolling_validity",
                       choices=["rolling_validity", "regime_cond", "state_cond"])

    # ---- dna ----
    p_dna = sub.add_parser("dna", help="DNA decoupling (attach full context, save parquet)")
    _add_common_args(p_dna)
    p_dna.add_argument("--validity-windows", default="180,365",
                       help="comma list of validity windows (days) (default 180,365)")
    p_dna.add_argument("--chart-types", default="1d,dollar",
                       help="comma list of chart types (default 1d,dollar)")
    p_dna.add_argument("--no-features", action="store_true",
                       help="skip chimera feature attachment")
    p_dna.add_argument("--no-regime", action="store_true",
                       help="skip BTC regime attachment")
    p_dna.add_argument("--out", default=None,
                       help="output parquet path (default runs/oracle/dna_*.parquet)")

    # ---- sweep ----
    p_sweep = sub.add_parser("sweep", help="compare_grid leaderboard over date range")
    p_sweep.add_argument("--start", required=True, help="sweep start date YYYY-MM-DD")
    p_sweep.add_argument("--end", required=True, help="sweep end date YYYY-MM-DD")
    p_sweep.add_argument("--step-days", type=int, default=5,
                         help="step between dates in the sweep (default 5)")
    p_sweep.add_argument("--universe", default="u10")
    p_sweep.add_argument("--cadences", default="1d",
                         help="comma list of cadences (default 1d)")
    p_sweep.add_argument("--windows", default="180,365",
                         help="comma list of validity windows (default 180,365)")
    p_sweep.add_argument("--mechanisms", default="rolling_validity,regime_cond,state_cond",
                         help="comma list of mechanisms (default all 3)")
    p_sweep.add_argument("--indicator", default="ma")
    p_sweep.add_argument("--lookback", type=int, default=30)
    p_sweep.add_argument("--top-n", type=int, default=25)

    # ---- doctor ----
    p_doc = sub.add_parser("doctor", help="preflight: import + data + date-range checks")
    p_doc.add_argument("--universe", default="u10")
    p_doc.add_argument("--date", default=None,
                       help="optional date to check against data range")

    return ap


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = build_parser()
    # Check if the first positional looks like a subcommand; if not, inject 'compare'
    # so it is the default subcommand.
    known_subs = {"oracle", "model", "compare", "dna", "sweep", "doctor"}
    argv = sys.argv[1:]
    # A word is a subcommand if it matches exactly; anything else means the user
    # omitted the subcommand and passed flags directly (e.g. --date ...).
    if not argv or argv[0].startswith("-") or argv[0] not in known_subs:
        argv = ["compare"] + argv

    args = ap.parse_args(argv)

    dispatch = {
        "oracle": cmd_oracle,
        "model": cmd_model,
        "compare": cmd_compare,
        "dna": cmd_dna,
        "sweep": cmd_sweep,
        "doctor": cmd_doctor,
    }

    fn = dispatch.get(args.subcommand)
    if fn is None:
        ap.print_help()
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
