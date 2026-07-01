"""Pipeline I/O contracts: atomic writes + skip-fresh / delta predicates.

Single source of truth for durable parquet I/O across every forex producer:
  - atomic write: tmp + col-verify + os.replace (never a half-written parquet)
  - skip-fresh: output mtime vs input mtimes (with a force override)
  - delta state: decide fresh / append / rebuild for a date-keyed producer

Ported from the crypto pipeline framework (proven across the crypto ingest +
bar + feature producers). Behaviour is identical; only the docstrings are
forex-clean. Producers MUST use these helpers instead of inline tmp+rename or
inline mtime comparisons so the crash-safety + freshness contracts hold
everywhere.

Freshness decisions are LOUD (the caller prints why a stage was skipped).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_io",
    "inputs": {"args": ["df, out_path, required_cols (atomic_write); "
                        "out_path, input_paths, force (is_fresh)"]},
    "outputs": {"side_effects": "atomic parquet at out_path; bool predicate"},
    "invariants": {
        "atomic_write_no_partial": True,
        "tmp_cleaned_on_failure": True,
        "force_overrides_freshness": True,
    },
    "rationale": "Crash-safe parquet write + freshness/delta logic shared by all producers.",
}


def atomic_write_parquet(
    df,                                       # polars OR pandas DataFrame
    out_path: Path | str,
    required_cols: Optional[set[str]] = None,
    *,
    compression: str = "zstd",
    compression_level: int = 3,
) -> Path:
    """Atomically write a parquet, validating cols before rename.

    Accepts a polars.DataFrame OR pandas.DataFrame.

    Sequence:
      1. mkdir -p parent
      2. write to <out>.tmp
      3. read schema, verify required_cols subset
      4. os.replace(<out>.tmp -> <out>)  (single atomic syscall)

    Failure at any step deletes the tmp file and re-raises. The on-disk
    <out> is either the previous version OR the new validated version,
    never a half-written file.

    Returns the final out_path. Raises ValueError if required_cols is not a
    subset of the written schema.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    is_polars = isinstance(df, pl.DataFrame)
    if not is_polars:
        try:
            import pandas as _pd
            if isinstance(df, _pd.DataFrame):
                df = pl.from_pandas(df)
            else:
                raise TypeError(
                    f"atomic_write_parquet expects polars or pandas DataFrame; "
                    f"got {type(df).__name__}")
        except ImportError:
            raise TypeError(
                f"atomic_write_parquet got non-polars DataFrame and pandas "
                f"is not installed; type={type(df).__name__}")

    try:
        df.write_parquet(tmp_path, compression=compression,
                         compression_level=compression_level)
        if required_cols:
            written = set(pl.read_parquet_schema(tmp_path).keys())
            missing = set(required_cols) - written
            if missing:
                raise ValueError(
                    f"atomic_write_parquet: {out_path.name} missing required cols: "
                    f"{sorted(missing)} (wrote {len(written)} cols)")
        # os.replace is atomic on BOTH POSIX and Windows (NTFS): it overwrites
        # the target with no unlink-then-rename gap where a concurrent reader
        # could hit FileNotFoundError.
        os.replace(str(tmp_path), str(out_path))
        return out_path
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def is_fresh(
    out_path: Path | str,
    input_paths: Optional[Iterable[Path | str]] = None,
    *,
    force: bool = False,
) -> bool:
    """Skip-fresh predicate: True iff out_path is up to date relative to inputs.

    Decision tree:
      - force=True            -> False (caller will rebuild; LOUD)
      - out_path missing      -> False
      - no input_paths given  -> True (existence is sufficient)
      - else                  -> out_path.mtime >= max(input mtimes)

    OSError on stat (e.g. race with deletion) returns False conservatively.
    """
    out_path = Path(out_path)
    if force:
        return False
    if not out_path.exists():
        return False
    if not input_paths:
        return True
    try:
        out_mtime = out_path.stat().st_mtime
        newest_input = max(Path(p).stat().st_mtime for p in input_paths)
        return out_mtime >= newest_input
    except OSError:
        return False


def safe_unlink(path: Path | str) -> bool:
    """Delete path if it exists; return True if a file was removed."""
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
            return True
        except OSError as e:
            print(f"[io] safe_unlink({p}) failed: {type(e).__name__}: {e}",
                  flush=True)
            raise
    return False


def _normalize_date(d) -> "date | None":  # noqa: F821
    """Coerce assorted date-ish values to datetime.date.

    Accepts datetime.date, datetime.datetime, polars Date/Datetime, pandas
    Timestamp, ISO 'YYYY-MM-DD' strings, and ms/sec epoch ints. Returns None
    on parse failure.
    """
    from datetime import date as _date, datetime as _dt
    if d is None:
        return None
    if isinstance(d, _date) and not isinstance(d, _dt):
        return d
    if isinstance(d, _dt):
        return d.date()
    if isinstance(d, str):
        try:
            return _date.fromisoformat(d[:10])
        except (ValueError, TypeError):
            return None
    if isinstance(d, (int, float)):
        try:
            ts = float(d)
            if ts > 1e12:            # ms epoch
                ts /= 1000.0
            return _dt.utcfromtimestamp(ts).date()
        except (OSError, OverflowError, ValueError):
            return None
    if hasattr(d, "date"):
        try:
            r = d.date()
            return r if isinstance(r, _date) else None
        except Exception:
            return None
    return None


def validate_existing(out_path: Path | str,
                      *,
                      required_cols: Optional[set] = None,
                      max_null_rate: Optional[dict] = None,
                      min_rows: int = 1) -> tuple[bool, str]:
    """Sanity-check an existing parquet before trusting it for delta-append.

    Catches the corruption pattern where the schema is correct but VALUES are
    silently null/corrupted, so a pure-delta append would preserve corruption
    forever.

    Checks (all optional): required_cols present, per-column max null rate,
    min_rows. Returns (ok, reason); ok=False means rebuild from scratch.
    """
    p = Path(out_path)
    if not p.exists():
        return (False, f"file does not exist: {p}")
    try:
        schema_cols = set(pl.read_parquet_schema(p).keys())
    except Exception as e:
        return (False, f"schema read failed: {type(e).__name__}: {e}")
    if required_cols:
        missing = set(required_cols) - schema_cols
        if missing:
            return (False, f"missing required cols: {sorted(missing)}")
    if min_rows >= 1 or max_null_rate:
        try:
            cols_to_load = None
            if max_null_rate:
                cols_to_load = [c for c in max_null_rate if c in schema_cols]
            if not cols_to_load:
                any_col = next(iter(schema_cols), None)
                if any_col is None:
                    return (False, "parquet has zero columns")
                cols_to_load = [any_col]
            df = pl.read_parquet(p, columns=cols_to_load)
        except Exception as e:
            return (False, f"data read failed: {type(e).__name__}: {e}")
        if len(df) < min_rows:
            return (False, f"row count {len(df)} < min_rows {min_rows}")
        if max_null_rate and len(df) > 0:
            for col, max_rate in max_null_rate.items():
                if col not in df.columns:
                    continue
                try:
                    n_null = int(df[col].null_count())
                except Exception:
                    continue
                rate = n_null / len(df)
                if rate > max_rate:
                    return (False, f"col '{col}' null rate {rate:.1%} > "
                                   f"max {max_rate:.1%} ({n_null}/{len(df)} rows)")
    return (True, "ok")


def read_existing_dates(out_path: Path | str, date_col: str = "date") -> set:
    """Return the set of distinct datetime.date values from an existing parquet."""
    p = Path(out_path)
    if not p.exists():
        return set()
    try:
        df = pl.read_parquet(p, columns=[date_col])
        out: set = set()
        for v in df[date_col].to_list():
            nd = _normalize_date(v)
            if nd is not None:
                out.add(nd)
        return out
    except Exception as e:
        print(f"[io] read_existing_dates({p.name}) failed: "
              f"{type(e).__name__}: {e}; treating as empty", flush=True)
        return set()


def delta_state(out_path: Path | str,
                input_paths: Iterable[Path | str],
                *,
                force: bool = False,
                date_from_filename: Optional["callable"] = None,
                date_col: str = "date",
                required_cols: Optional[set] = None,
                max_null_rate: Optional[dict] = None) -> dict:
    """Decide between fresh / append / rebuild for a date-keyed producer.

    Instead of binary fresh-vs-rebuild (mtime), do a chain-link append when
    the existing output contains a strict subset of the input dates
    (set-difference on the per-file date). Fills mid-stream gaps AND trailing
    dates equally.

    Returns {"mode": "fresh"|"append"|"rebuild", "existing_dates": set,
             "new_inputs": list[Path], "reason": str}.
    """
    p_out = Path(out_path)
    inputs_list = [Path(p) for p in input_paths]

    if force:
        return {"mode": "rebuild", "existing_dates": set(),
                "new_inputs": inputs_list, "reason": "force"}
    if not p_out.exists():
        return {"mode": "rebuild", "existing_dates": set(),
                "new_inputs": inputs_list, "reason": "no existing output"}
    if required_cols or max_null_rate:
        ok, why = validate_existing(p_out, required_cols=required_cols,
                                    max_null_rate=max_null_rate)
        if not ok:
            return {"mode": "rebuild", "existing_dates": set(),
                    "new_inputs": inputs_list, "reason": f"existing corrupt: {why}"}
    if not inputs_list:
        return {"mode": "fresh", "existing_dates": set(),
                "new_inputs": [], "reason": "no inputs"}
    if date_from_filename is None:
        if is_fresh(p_out, inputs_list):
            return {"mode": "fresh", "existing_dates": set(),
                    "new_inputs": [], "reason": "mtime fresh"}
        return {"mode": "rebuild", "existing_dates": set(),
                "new_inputs": inputs_list, "reason": "mtime stale"}

    existing_dates = read_existing_dates(p_out, date_col=date_col)
    if not existing_dates:
        return {"mode": "rebuild", "existing_dates": existing_dates,
                "new_inputs": inputs_list,
                "reason": f"existing parquet has no '{date_col}' values"}

    inputs_with_dates = []
    for fp in inputs_list:
        try:
            raw_d = date_from_filename(fp)
        except Exception:
            continue
        d = _normalize_date(raw_d)
        if d is None:
            continue
        inputs_with_dates.append((fp, d))
    if not inputs_with_dates:
        return {"mode": "fresh", "existing_dates": existing_dates,
                "new_inputs": [], "reason": "no input dates parseable"}

    missing_dates = sorted({d for _, d in inputs_with_dates
                           if d not in existing_dates})
    if not missing_dates:
        return {"mode": "fresh", "existing_dates": existing_dates,
                "new_inputs": [],
                "reason": f"all {len(inputs_with_dates)} input dates already "
                          f"in output ({len(existing_dates)} dates)"}

    missing_set = set(missing_dates)
    new_inputs = sorted((fp for fp, d in inputs_with_dates if d in missing_set),
                        key=lambda p: str(p))
    return {"mode": "append", "existing_dates": existing_dates,
            "new_inputs": new_inputs,
            "reason": f"appending {len(new_inputs)} files "
                      f"({len(missing_dates)} missing dates)"}


def append_parquet(existing_path: Path | str,
                   new_df: pl.DataFrame,
                   *,
                   date_col: str = "date",
                   sort_col: str = "date",
                   required_cols: Optional[set] = None) -> Path:
    """Idempotent date-keyed append: drop existing rows whose date is in new_df,
    concat, sort, atomic-write. Re-running with overlapping new_df is a no-op
    (last write wins per date)."""
    p = Path(existing_path)
    if not p.exists():
        atomic_write_parquet(new_df.sort(sort_col), p, required_cols=required_cols)
        return p
    existing = pl.read_parquet(p)
    existing_dtype = existing.schema.get(date_col)
    new_key = new_df[date_col]
    if existing_dtype is not None and new_key.dtype != existing_dtype:
        try:
            new_key = new_key.cast(existing_dtype)
        except Exception:
            pass
    new_dates = set(new_key.to_list())
    keep = existing.filter(~pl.col(date_col).is_in(list(new_dates)))
    union = pl.concat([keep, new_df], how="vertical_relaxed").sort(sort_col)
    atomic_write_parquet(union, p, required_cols=required_cols)
    return p
