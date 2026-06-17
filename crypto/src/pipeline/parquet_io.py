"""Pipeline I/O contracts: atomic writes + skip-fresh predicates.

Replaces ~7 copy-pasted atomic-tmp-rename blocks across producers
(originally G-AUDIT-020). Single source of truth for:
  - atomic write: tmp + col-verify + rename (no half-written parquet)
  - skip-fresh: output mtime vs input mtimes (and force override)

Per @browser B7: writes are atomic. Per @browser B1: freshness decisions
are LOUD (caller prints why a stage was skipped).

CDAP contract: scripts MUST use these helpers instead of inline tmp+rename
or inline mtime comparisons. The audit hook (config/_invariants.yaml)
will grow rules to enforce this once migration completes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_io",
    "inputs": {"args": ["df", "out_path", "required_cols (atomic_write); "
                         "out_path, input_paths, force (is_fresh)"]},
    "outputs": {"side_effects": "atomic parquet at out_path; bool predicate"},
    "invariants": {
        "atomic_write_no_partial": True,
        "tmp_cleaned_on_failure": True,
        "force_overrides_freshness": True,
    },
    "rationale": "Eliminate copy-paste of G-AUDIT-020 atomic-write across 7+ producers.",
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

    Accepts polars.DataFrame OR pandas.DataFrame (Phase 6 polymorphism for
    Phase B7 retrofit of 10 producers; previously polars-only).

    Sequence:
      1. mkdir -p parent
      2. write to <out>.tmp
      3. read schema, verify required_cols subset
      4. unlink existing <out> if present
      5. rename <out>.tmp -> <out>

    Failure at any step deletes the tmp file and re-raises. The on-disk
    <out> is either the previous version OR the new validated version,
    never a half-written file.

    Returns:
        Path: the final out_path.

    Raises:
        ValueError: if required_cols is not a subset of the written schema.
        OSError / pl.exceptions.*: from the underlying write/rename.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # Polymorphism: if pandas DataFrame, convert to polars internally so the
    # rest of the function stays one code path.
    is_polars = isinstance(df, pl.DataFrame)
    if not is_polars:
        try:
            import pandas as _pd
            if isinstance(df, _pd.DataFrame):
                df_pl = pl.from_pandas(df)
            else:
                raise TypeError(
                    f"atomic_write_parquet expects polars or pandas DataFrame; "
                    f"got {type(df).__name__}")
        except ImportError:
            raise TypeError(
                f"atomic_write_parquet got non-polars DataFrame and pandas "
                f"is not installed; type={type(df).__name__}")
        df = df_pl

    try:
        df.write_parquet(tmp_path, compression=compression,
                          compression_level=compression_level)
        if required_cols:
            written = set(pl.read_parquet_schema(tmp_path).keys())
            missing = set(required_cols) - written
            if missing:
                raise ValueError(
                    f"atomic_write_parquet: {out_path.name} missing required cols: "
                    f"{sorted(missing)} (wrote {len(written)} cols)"
                )
        # os.replace is a single atomic syscall on BOTH POSIX and Windows
        # (NTFS): it overwrites the target with no unlink-then-rename gap where
        # a concurrent reader could hit FileNotFoundError (G-PIPE Windows race).
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

    OSError on stat (e.g., race with deletion) returns False conservatively.

    Use:
        if is_fresh(out, input_files, force=args.force):
            print(f"[{name}] SKIP fresh: {out.name}")
            return
        # else rebuild
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
    """Delete path if exists; return True if a file was removed.

    Use for force-rebuild: caller logs the deletion (B1 LOUD).
    """
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


def _normalize_date(d) -> "date | None":
    """Coerce assorted date-ish values to datetime.date.

    Accepts: datetime.date, datetime.datetime, polars Date/Datetime objects,
    pandas Timestamp, ISO 'YYYY-MM-DD' strings, and ms-epoch ints.
    Returns None on parse failure.
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
            # Heuristic: treat as ms-epoch if > 1e12, else seconds.
            ts = float(d)
            if ts > 1e12:
                ts /= 1000.0
            return _dt.utcfromtimestamp(ts).date()
        except (OSError, OverflowError, ValueError):
            return None
    # Last resort: many pandas/polars types support .date()
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

    Catches the corruption pattern that broke the prior build: features
    schema is correct but VALUES are silently null/corrupted, so a
    pure-delta append preserves the corruption forever.

    Checks (all optional; skipped if arg not provided):
      - required_cols: set of column names that MUST be present.
      - max_null_rate: {col_name: max_rate} where rate in [0, 1]. If any
        listed column's null rate exceeds max_rate, file is corrupt.
        Use this for KEY feature columns whose nullness would propagate.
      - min_rows: file with fewer rows than this is corrupt (default 1).

    Returns:
        (ok: bool, reason: str)
        ok=True  -> parquet is sane; safe to delta-append
        ok=False -> parquet is corrupt; reason explains. Caller should
                    fall through to full rebuild.

    Use:
        ok, why = validate_existing(out, required_cols={"date","asset"},
                                      max_null_rate={"xd_funding_spread": 0.10})
        if not ok:
            print(f"[stage] CORRUPT existing: {why}; force rebuild")
            # full rebuild instead of delta
    """
    p = Path(out_path)
    if not p.exists():
        return (False, f"file does not exist: {p}")

    # Schema check via lazy schema read (no data load).
    try:
        schema_cols = set(pl.read_parquet_schema(p).keys())
    except Exception as e:
        return (False, f"schema read failed: {type(e).__name__}: {e}")

    if required_cols:
        missing = set(required_cols) - schema_cols
        if missing:
            return (False, f"missing required cols: {sorted(missing)}")

    # Row-count + null-rate checks need actual data load. Use >= 1 (not > 1) so
    # a zero-row parquet is detected at the default min_rows=1 (a schema-only,
    # data-empty file otherwise silently passed validation).
    if min_rows >= 1 or max_null_rate:
        try:
            cols_to_load = ["__dummy__"]  # placeholder
            if max_null_rate:
                cols_to_load = [c for c in max_null_rate if c in schema_cols]
            if not cols_to_load or cols_to_load == ["__dummy__"]:
                # Need any column to count rows.
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
                    return (False,
                             f"col '{col}' null rate {rate:.1%} > "
                             f"max {max_rate:.1%} ({n_null}/{len(df)} rows)")

    return (True, "ok")


def read_existing_dates(out_path: Path | str,
                         date_col: str = "date") -> set:
    """Return the set of distinct datetime.date values from an existing parquet.

    Normalizes assorted date types (string / Date / Datetime / int) to
    datetime.date so callers can do set-comparison with date objects from
    `date_from_filename`. Empty set on missing/unreadable file.
    """
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
              f"{type(e).__name__}: {e}; treating as empty",
              flush=True)
        return set()


def delta_state(out_path: Path | str,
                 input_paths: Iterable[Path | str],
                 *,
                 force: bool = False,
                 date_from_filename: Optional["callable"] = None,
                 date_col: str = "date",
                 burn_from_first_gap: bool = False,
                 window_days: int = 0,
                 required_cols: Optional[set] = None,
                 max_null_rate: Optional[dict] = None) -> dict:
    """Decide between fresh / append / rebuild for a date-keyed producer.

    The "third state": instead of binary fresh-vs-rebuild (mtime), do a
    chain-link append when the existing output contains a strict subset
    of the input dates.

    Three semantic modes (toggled by flags):

    1. **Pure set-difference** (default; per-day independent producers):
       new_inputs = inputs whose date isn't in existing output.
       Fills mid-stream gaps AND trailing dates equally. Day X's output
       doesn't depend on day Y, so this is correct for bars / whale / liq /
       hawkes.

    2. **Burn-from-first-gap** (`burn_from_first_gap=True`; for stages
       where contiguity matters or you want defensive behavior):
       If a mid-stream gap exists in the existing output, find the first
       missing date and include EVERY input date >= that date in
       new_inputs (clearing existing rows for that range). Rebuilds the
       chain from the first crack onwards. Use when Binance has just
       backfilled a previously-missing date and you want the full segment
       reconstructed deterministically.

    3. **Windowed delta** (`window_days=W`; for rolling-feature producers
       like RV / TE / rolling z-scores):
       new_inputs = inputs with date >= (max(existing_dates) - W). This
       guarantees that any rolling-window feature whose window includes
       newly-arrived dates gets recomputed on the overlapping tail.
       Combine with burn_from_first_gap=True if Binance backfilled
       mid-stream and you also need rolling-feature continuity.

    Returns:
        {
          "mode": "fresh" | "append" | "rebuild",
          "existing_dates": set,
          "new_inputs": list[Path],
          "reason": str,                        # explains gap / window / etc.
        }

    Args:
        out_path: existing output parquet (may not exist).
        input_paths: candidate input files.
        force: True -> always 'rebuild' regardless.
        date_from_filename: callable Path -> date (or None to fall back
            to mtime-based is_fresh; in that case mode is 'fresh' or
            'rebuild', never 'append').
        date_col: column in existing output that holds the date.
        burn_from_first_gap: if True, on detecting a mid-stream gap, mark
            new_inputs as ALL inputs from the first-gap-date onwards (not
            just the gap itself).
        window_days: if >0, pad new_inputs backwards by this many days
            from existing's max date so rolling-window features get the
            overlap they need.
    """
    p_out = Path(out_path)
    inputs_list = [Path(p) for p in input_paths]

    if force:
        return {"mode": "rebuild", "existing_dates": set(),
                "new_inputs": inputs_list, "reason": "force"}

    if not p_out.exists():
        return {"mode": "rebuild", "existing_dates": set(),
                "new_inputs": inputs_list, "reason": "no existing output"}

    # CORRUPTION GUARD: if the caller declared schema/null expectations,
    # validate the existing parquet BEFORE trusting it for delta-append.
    # Catches the prior failure mode: schema looks right but feature
    # columns are silently null. Pure delta would preserve the corruption.
    if required_cols or max_null_rate:
        ok, why = validate_existing(p_out,
                                      required_cols=required_cols,
                                      max_null_rate=max_null_rate)
        if not ok:
            return {"mode": "rebuild", "existing_dates": set(),
                    "new_inputs": inputs_list,
                    "reason": f"existing corrupt: {why}"}

    if not inputs_list:
        return {"mode": "fresh", "existing_dates": set(),
                "new_inputs": [], "reason": "no inputs"}

    # Without date awareness, fall back to mtime-fresh-or-rebuild.
    if date_from_filename is None:
        if is_fresh(p_out, inputs_list):
            return {"mode": "fresh", "existing_dates": set(),
                    "new_inputs": [], "reason": "mtime fresh"}
        return {"mode": "rebuild", "existing_dates": set(),
                "new_inputs": inputs_list, "reason": "mtime stale"}

    # Date-aware delta path.
    existing_dates = read_existing_dates(p_out, date_col=date_col)
    if not existing_dates:
        return {"mode": "rebuild", "existing_dates": existing_dates,
                "new_inputs": inputs_list,
                "reason": f"existing parquet has no '{date_col}' values"}

    # Pair each input file with its decoded date (drop unparseable).
    # Normalize whatever date_from_filename returns (string / date / etc.)
    # to datetime.date for set-comparison with read_existing_dates.
    inputs_with_dates: list[tuple[Path, "date"]] = []
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
                "new_inputs": [],
                "reason": "no input dates parseable"}

    # Pure set-difference: dates in inputs but not in existing output.
    # Naturally catches both mid-stream gaps AND trailing-edge new dates.
    missing_dates = sorted({d for _, d in inputs_with_dates
                              if d not in existing_dates})

    if not missing_dates:
        return {"mode": "fresh", "existing_dates": existing_dates,
                "new_inputs": [],
                "reason": f"all {len(inputs_with_dates)} input dates "
                            f"already in output ({len(existing_dates)} dates)"}

    reason_parts: list[str] = []
    existing_max = max(existing_dates) if existing_dates else None
    is_mid_gap = (existing_max is not None
                   and missing_dates[0] <= existing_max)

    # Decide whether to use range-mode (everything >= cutoff) or
    # set-mode (only the specific missing dates).
    use_range_mode = False
    cutoff = missing_dates[0]

    if burn_from_first_gap and is_mid_gap:
        use_range_mode = True
        cutoff = missing_dates[0]
        reason_parts.append(
            f"mid-stream gap @ {cutoff} <= existing_max={existing_max}; "
            f"burn_from_first_gap=True (range from gap forward)")
    elif window_days > 0 and existing_max is not None:
        from datetime import timedelta as _td
        use_range_mode = True
        cutoff = existing_max - _td(days=window_days)
        # Always pull at least the first missing date into the rebuild.
        if cutoff > missing_dates[0]:
            cutoff = missing_dates[0]
        reason_parts.append(
            f"window={window_days}d overlap; cutoff={cutoff} "
            f"(rolling-feature recompute on tail)")
    else:
        # Pure set-diff: only the specific missing dates.
        if is_mid_gap:
            reason_parts.append(
                f"mid-stream gap @ {missing_dates[0]} <= "
                f"existing_max={existing_max}; pure set-diff fill-only")
        else:
            reason_parts.append(
                f"trailing-edge: first new={missing_dates[0]}, "
                f"existing_max={existing_max}")

    if use_range_mode:
        new_inputs = sorted(
            (fp for fp, d in inputs_with_dates if d >= cutoff),
            key=lambda p: str(p))
    else:
        missing_set = set(missing_dates)
        new_inputs = sorted(
            (fp for fp, d in inputs_with_dates if d in missing_set),
            key=lambda p: str(p))

    return {"mode": "append", "existing_dates": existing_dates,
            "new_inputs": new_inputs,
            "reason": (f"appending {len(new_inputs)} files "
                        f"({len(missing_dates)} missing dates) | "
                        + " | ".join(reason_parts))}


def panel_delta_state(out_path: Path | str,
                       per_asset_dates: dict,
                       *,
                       force: bool = False,
                       required_cols: Optional[set] = None,
                       max_null_rate: Optional[dict] = None,
                       window_days: int = 0) -> dict:
    """Delta state for a multi-asset PANEL parquet (rows = (asset, date)).

    Whale / liquidations / s3_metrics produce one panel across all assets.
    Per-asset deltas need (asset, date) co-keying: for each asset, find
    the dates already in the panel for THAT asset; new_dates per asset =
    candidate dates not in existing-for-asset.

    Args:
        out_path: panel parquet (rows have 'asset' + 'date' cols).
        per_asset_dates: dict {asset_str: list[date]} -- candidate dates
            we *could* build per asset (from input file globs).
        force / required_cols / max_null_rate: same as delta_state.
        window_days: if >0, extend new_dates per asset backwards by this
            many days from the asset's existing max date. Required for
            rolling-feature panels (rv_jump W=20-30 EMA, te_panel W=90)
            where new days need overlap context to compute correctly.
            The rebuilt rows REPLACE existing rows for those dates via
            append_panel_parquet's anti-join semantics.

    Returns:
        {
          "mode": "fresh" | "append" | "rebuild",
          "per_asset_new_dates": {asset: [date, ...]},
          "reason": str,
        }
    """
    p_out = Path(out_path)
    if force:
        return {"mode": "rebuild", "per_asset_new_dates": dict(per_asset_dates),
                "reason": "force"}
    if not p_out.exists():
        return {"mode": "rebuild", "per_asset_new_dates": dict(per_asset_dates),
                "reason": "no existing panel"}
    if required_cols or max_null_rate:
        ok, why = validate_existing(p_out,
                                      required_cols=required_cols,
                                      max_null_rate=max_null_rate)
        if not ok:
            return {"mode": "rebuild",
                    "per_asset_new_dates": dict(per_asset_dates),
                    "reason": f"existing corrupt: {why}"}

    # Read (asset, date) pairs from existing panel.
    try:
        df = pl.read_parquet(p_out, columns=["asset", "date"])
    except Exception as e:
        return {"mode": "rebuild", "per_asset_new_dates": dict(per_asset_dates),
                "reason": f"existing read failed: {type(e).__name__}: {e}"}

    # Group existing dates per asset for fast lookup.
    per_asset_existing: dict[str, set] = {}
    for asset, date_v in zip(df["asset"].to_list(), df["date"].to_list()):
        nd = _normalize_date(date_v)
        if nd is None:
            continue
        per_asset_existing.setdefault(str(asset), set()).add(nd)

    per_asset_new: dict[str, list] = {}
    total_new = 0
    from datetime import timedelta as _td
    for asset, candidate_dates in per_asset_dates.items():
        existing = per_asset_existing.get(asset, set())
        normalized_candidates = [(_normalize_date(d), d) for d in candidate_dates]
        normalized_candidates = [(nd, raw) for nd, raw in normalized_candidates
                                  if nd is not None]
        if window_days > 0 and existing:
            # Windowed: include the last `window_days` of existing context
            # so rolling-feature recomputes have proper warmup. The rebuilt
            # rows for those dates REPLACE the existing rows on append.
            # G-AUDIT-026: previously had a half-broken precedence-bug
            # filter immediately overwritten by the correct one. Removed
            # the dead branch so a future "cleanup" cannot re-activate it.
            existing_max = max(existing)
            cutoff = existing_max - _td(days=window_days)
            new_for_asset = sorted({nd for nd, _raw in normalized_candidates
                                      if nd >= cutoff})
        else:
            new_for_asset = sorted({nd for nd, _raw in normalized_candidates
                                      if nd not in existing})
        if new_for_asset:
            per_asset_new[asset] = new_for_asset
            total_new += len(new_for_asset)

    if total_new == 0:
        return {"mode": "fresh", "per_asset_new_dates": {},
                "reason": f"all (asset, date) pairs already in panel "
                           f"({len(df)} rows)"}

    return {"mode": "append", "per_asset_new_dates": per_asset_new,
            "reason": f"appending {total_new} new (asset, date) rows "
                        f"across {len(per_asset_new)} assets"}


def append_panel_parquet(existing_path: Path | str,
                          new_rows_df: pl.DataFrame,
                          *,
                          asset_col: str = "asset",
                          date_col: str = "date",
                          sort_cols: Optional[list] = None,
                          required_cols: Optional[set] = None) -> Path:
    """Append (asset, date)-keyed rows to a multi-asset panel.

    Drops existing rows whose (asset, date) appears in new_rows_df, concats,
    sorts by (asset, date) by default, atomic-writes.

    Defensive: reorders new_rows_df columns to match existing's schema
    BEFORE concat so polars' "schema names differ" error is avoided.
    Producers that emit rows in a different column order than the existing
    panel (a real bug observed 2026-05-01 in whale_activity) get tolerated
    here -- the producer should still be fixed, but this layer prevents
    a hard failure that breaks the pipeline cascade.
    """
    if sort_cols is None:
        sort_cols = [asset_col, date_col]
    p = Path(existing_path)
    if not p.exists():
        atomic_write_parquet(new_rows_df.sort(sort_cols), p,
                              required_cols=required_cols)
        return p
    existing = pl.read_parquet(p)

    # Reorder new_rows_df columns to match existing schema. This avoids
    # the polars "schema names differ" error during pl.concat.
    existing_cols = existing.columns
    new_cols = set(new_rows_df.columns)
    # Project to existing's column set, in existing's order.
    common = [c for c in existing_cols if c in new_cols]
    extra_in_new = [c for c in new_rows_df.columns if c not in existing_cols]
    if extra_in_new:
        # New rows have columns the existing doesn't -- this is a schema
        # extension. Keep them at the end (existing rows will get nulls).
        ordered = common + extra_in_new
    else:
        ordered = common
    if ordered != list(new_rows_df.columns):
        new_rows_df = new_rows_df.select(ordered)

    new_keys = new_rows_df.select([asset_col, date_col]).unique()
    keep = existing.join(new_keys, on=[asset_col, date_col], how="anti")
    union = pl.concat([keep, new_rows_df],
                       how="vertical_relaxed").sort(sort_cols)
    atomic_write_parquet(union, p, required_cols=required_cols)
    return p


def append_parquet(existing_path: Path | str,
                    new_df: pl.DataFrame,
                    *,
                    date_col: str = "date",
                    sort_col: str = "date",
                    required_cols: Optional[set] = None) -> Path:
    """Read existing parquet, drop rows whose date_col is in new_df, concat,
    sort by sort_col, atomic-write.

    The "drop existing rows for replaced dates" step ensures idempotency:
    re-running with overlapping new_df produces the same output (last
    write wins for any date appearing in both).

    Args:
        existing_path: parquet to read + overwrite atomically.
        new_df: rows to append (must contain date_col).
        date_col: column used to identify "rows for this date".
        sort_col: column to sort the union by before write.
        required_cols: passed through to atomic_write_parquet.

    Returns: the output Path.
    """
    p = Path(existing_path)
    if not p.exists():
        # Caller should have routed to atomic_write_parquet directly; defensive.
        atomic_write_parquet(new_df.sort(sort_col), p,
                              required_cols=required_cols)
        return p
    existing = pl.read_parquet(p)
    # Normalize the dedup key dtype: if existing date_col is e.g. epoch-ms int
    # but new_df has Date objects (or vice versa), is_in would match nothing and
    # silently produce duplicate rows for the same date. Cast new_df's key to the
    # existing dtype for the membership test.
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
