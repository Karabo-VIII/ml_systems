"""bidirectional.py -- forward/reverse execution helpers.

Phase 7 of the pipeline overhaul. Implements the "meet-in-the-middle" pattern
the user prefers: run TWO terminals against the same producer with
opposite `--reverse` flags. Terminal A processes oldest->newest (or A->Z);
Terminal B processes newest->oldest (or Z->A). They share a skip-if-exists
gate so neither redoes the other's work; they meet in the middle, ~2x faster.

REFERENCE IMPLEMENTATION
========================
`src/pipeline/fetch_all.py` already implements this for raw_aggtrades +
raw_funding. This module generalizes the pattern for ANY per-asset or
per-(asset, date) producer.

WHEN TO USE BIDIRECTIONAL
=========================
Two-terminal meet-in-middle gives ~2x speedup when:
  (a) work is partitionable along an axis (per-asset OR per-date)
  (b) per-task output is independent (no shared write target)
  (c) `should_skip()` correctly detects "another worker did this one"

DO NOT use bidirectional for:
  - Single-blob outputs (e.g. one large parquet aggregating all assets) -
    both workers would race-write the same file
  - Aggregation stages (basis_features_long, lob_proxy_daily, etc.) -
    output is one file built from many inputs; can't be partitioned
  - Stages with sequential state (e.g. cumulative running stats) -
    unless the state is partitioned by axis too

CONTRACT
========
Producers using this module MUST:
  1. Accept a `--reverse` flag (action=store_true)
  2. Pass that flag to `iter_assets()` / `iter_dates()` for traversal order
  3. Use `should_skip(output_path)` before each task to detect "already done"
  4. Use `atomic_write_parquet` (from parquet_io) so half-written files
     don't trick the other worker into "skipping" an incomplete file

USAGE
=====
    from pipeline.bidirectional import iter_assets, should_skip
    from pipeline.parquet_io import atomic_write_parquet

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--reverse", action="store_true",
                          help="Process A->Z (default) or Z->A (--reverse). "
                               "Run with `-r` in a second terminal for "
                               "meet-in-middle 2x speedup.")
        # ... other args
        args = ap.parse_args()

        assets = load_universe(args.universe)
        for asset in iter_assets(assets, reverse=args.reverse):
            out_path = OUT_DIR / f"{asset}_panel.parquet"
            if should_skip(out_path):
                continue                 # other worker did this one
            df = compute(asset)
            atomic_write_parquet(df, out_path, required_cols={...})

USER PATTERN
============
Two terminals, same command, opposite -r flags:

    Terminal A:  python src/pipeline/bars/dib_bars_fast.py --universe u100
    Terminal B:  python src/pipeline/bars/dib_bars_fast.py --universe u100 -r

Both finish in ~half the time of one terminal alone.
"""
from __future__ import annotations

__contract__ = {
    "kind": "framework_primitive",
    "owner": "pipeline/orchestration",
    "outputs": "iter_assets / iter_dates / should_skip helpers",
    "invariants": [
        "iter functions yield in forward (default) or reverse order",
        "should_skip uses file existence + size > 100 byte (matches fetch_all)",
        "atomic-write contract makes half-files visible only as final files",
        "no shared state -- helpers are pure",
    ],
}

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, List, Union

DateLike = Union[date, datetime, str]
AssetLike = str

MIN_OUTPUT_BYTES = 100   # match fetch_all's threshold


def iter_assets(assets: Iterable[AssetLike],
                  reverse: bool = False) -> Iterator[AssetLike]:
    """Yield asset symbols in forward (default) or reverse order.

    MEET-IN-THE-MIDDLE INVARIANT: for two terminals to converge without overlap,
    BOTH must be given the SAME input ordering (one with reverse=False, one with
    reverse=True). This function does not sort -- it preserves the caller's order
    so universe-priority ordering (e.g. BTC first) is respected. If you need
    alphabetical symmetry between independently-launched terminals, pass a
    pre-sorted list to both.

    Args:
        assets: iterable of asset symbols (any sortable type)
        reverse: if True, reverse the iteration order

    Yields:
        each asset symbol once, in the requested order
    """
    out = list(assets)
    if reverse:
        out = list(reversed(out))
    yield from out


def iter_dates(dates: Iterable[DateLike],
                 reverse: bool = False) -> Iterator[DateLike]:
    """Yield dates in forward (default) or reverse order.

    Args:
        dates: iterable of date-like values (date, datetime, or ISO string)
        reverse: if True, reverse the iteration order
    """
    out = list(dates)
    if reverse:
        out = list(reversed(out))
    yield from out


def date_range(start: DateLike, end: DateLike,
                  reverse: bool = False) -> List[date]:
    """Build an inclusive date range as a list, optionally reversed.

    Returns Python date objects regardless of input type.
    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    elif isinstance(start, datetime):
        start = start.date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()
    elif isinstance(end, datetime):
        end = end.date()
    out: List[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur + timedelta(days=1)
    if reverse:
        out.reverse()
    return out


def should_skip(out_path: Path | str, min_bytes: int = MIN_OUTPUT_BYTES) -> bool:
    """Return True if `out_path` already exists and is at least `min_bytes`.

    Matches the fetch_all convention: file present + size > 100 bytes is
    considered "another worker already did this task." The min_bytes
    threshold rules out 0-byte ghost files left by interrupted processes.

    USAGE INVARIANT: the producer MUST use atomic_write_parquet so half-
    written files never get seen by `should_skip`. Without atomic writes,
    a crashed producer's partial file could be skipped as "done" leaving
    silent data corruption.
    """
    p = Path(out_path)
    if not p.exists():
        return False
    try:
        return p.stat().st_size >= min_bytes
    except OSError:
        return False


def both_terminals_recipe(producer_cmd: str) -> str:
    """Return a human-readable recipe for running this producer bidirectionally.

    Args:
        producer_cmd: the base command (e.g. "python src/pipeline/bars/dib_bars_fast.py --universe u100")

    Returns:
        formatted text showing both terminal invocations.
    """
    return (
        f"=== Bidirectional run recipe ===\n"
        f"Terminal A (forward, oldest->newest / A->Z):\n"
        f"  {producer_cmd}\n\n"
        f"Terminal B (reverse, newest->oldest / Z->A):\n"
        f"  {producer_cmd} -r\n\n"
        f"Both meet in the middle; total wall-time ~half of one terminal alone.\n"
    )


__all__ = [
    "iter_assets", "iter_dates", "date_range",
    "should_skip", "both_terminals_recipe",
    "MIN_OUTPUT_BYTES",
]
