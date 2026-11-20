"""Purge-aware data splitter -- leak-proof date-based train/val/oos/unseen.

Replaces ad-hoc date filtering scattered across training scripts. Reads the
canonical split boundaries from config/data_config.yaml (frozen dates per
project invariant) and applies them with explicit purge gaps.

Why this matters:
  Per CLAUDE.md "Anti-Fragile" invariants:
    Train: 50% (oldest)
    Val:   20%
    OOS:   20%
    Unseen: 10% (newest)
  with 400-bar PURGE gaps between segments to prevent normalization leakage.

Without purge gaps, a rolling-window normalizer at the END of train can leak
into the BEGINNING of val (the same bar appears in two segments' rolling stats).

This helper provides a single function that returns 4 DataFrames with explicit
date-based boundaries + purge applied.

Public API:
    from pipeline.purge_split import split_chimera, get_split_dates
    train_df, val_df, oos_df, unseen_df = split_chimera(chimera_df, asset="BTC")
    dates = get_split_dates()  # {'train_end': '2023-07-01', ...}

Run as standalone:
    python src/pipeline/purge_split.py --asset BTC --cadence 1d
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from chimera_loader import ChimeraLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CONFIG = PROJECT_ROOT / "config" / "data_config.yaml"


@dataclass
class SplitBoundaries:
    train_end: str        # "2023-07-01"
    val_end: str          # "2024-05-15"
    oos_end: str          # "2025-03-15"
    unseen_start: str     # "2025-03-15"
    purge_bars: int       # 400 default

    @property
    def train_end_date(self) -> str:
        return self.train_end

    def as_dict(self) -> dict:
        return {
            "train_end": self.train_end,
            "val_end": self.val_end,
            "oos_end": self.oos_end,
            "unseen_start": self.unseen_start,
            "purge_bars": self.purge_bars,
        }


def get_split_dates(config_path: Path = DATA_CONFIG) -> SplitBoundaries:
    """Load frozen split dates from config/data_config.yaml."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s = cfg["splits"]
    return SplitBoundaries(
        train_end=s["train_end"],
        val_end=s["val_end"],
        oos_end=s["oos_end"],
        unseen_start=s["unseen_start"],
        purge_bars=int(s.get("purge_bars", 400)),
    )


def split_chimera(
    df: pl.DataFrame,
    boundaries: SplitBoundaries | None = None,
    timestamp_col: str = "timestamp",
    apply_purge: bool = True,
    purge_bars_override: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a chimera DataFrame into (train, val, oos, unseen) by date.

    Args:
        df: chimera DataFrame with a 'timestamp' (epoch ms) or 'date' column.
        boundaries: optional override; default loads from config.
        timestamp_col: which column to use for date filtering.
        apply_purge: if True, applies a purge+embargo gap at every segment
                     boundary -- drops the TAIL of train/val/oos (forward-return
                     label leakage) AND the HEAD of val/oos/unseen (trailing
                     normalization-window leakage). Capped per side at 25% of the
                     segment with a LOUD warning when the cadence forces the gap
                     below the requested `purge_bars`.

    Returns:
        (train, val, oos, unseen) tuples.
    """
    if boundaries is None:
        boundaries = get_split_dates()

    # Convert ts to date if needed
    if timestamp_col == "timestamp" and "timestamp" in df.columns:
        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("__split_date")
        )
    elif "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Date).alias("__split_date"))
    else:
        raise ValueError("DataFrame must have 'timestamp' or 'date' column")

    from datetime import date as _date
    train_end_d = _date.fromisoformat(boundaries.train_end)
    val_end_d = _date.fromisoformat(boundaries.val_end)
    oos_end_d = _date.fromisoformat(boundaries.oos_end)
    unseen_start_d = _date.fromisoformat(boundaries.unseen_start)
    train = df.filter(pl.col("__split_date") < train_end_d)
    val = df.filter(
        (pl.col("__split_date") >= train_end_d) & (pl.col("__split_date") < val_end_d)
    )
    oos = df.filter(
        (pl.col("__split_date") >= val_end_d) & (pl.col("__split_date") < oos_end_d)
    )
    unseen = df.filter(pl.col("__split_date") >= unseen_start_d)

    if apply_purge:
        n = purge_bars_override if purge_bars_override is not None else boundaries.purge_bars
        if n > 0:
            # Two leakage modes, both addressed:
            #  (1) forward-return LABEL leakage: an upstream segment's last ~h bars
            #      have labels that reach INTO the next segment -> drop the TAIL of
            #      train/val/oos.
            #  (2) trailing-window NORMALIZATION leakage: a downstream segment's
            #      first ~window bars have rolling features reaching BACK into the
            #      previous segment -> drop the HEAD of val/oos/unseen (embargo).
            # The previous code dropped only TAILS (so mode (2) leaked) AND capped
            # at 5% (so at 1d cadence the effective purge was ~10 bars, not 400).
            # Cap per side at 25% of the segment so coarse-cadence segments are not
            # erased, and WARN LOUDLY whenever the cap relaxes the requested gap.
            def _cap(seg_len: int, side: str, seg: str) -> int:
                capped = min(n, max(0, int(seg_len * 0.25)))
                if capped < n:
                    print(f"[purge_split] WARN {seg} {side}: requested purge {n} "
                          f"capped to {capped} (segment only {seg_len} bars; the "
                          f"{n}-bar gap cannot be fully applied at this cadence)",
                          flush=True)
                return capped

            tr_tail = _cap(len(train), "tail", "train")
            v_head, v_tail = _cap(len(val), "head", "val"), _cap(len(val), "tail", "val")
            o_head, o_tail = _cap(len(oos), "head", "oos"), _cap(len(oos), "tail", "oos")
            u_head = _cap(len(unseen), "head", "unseen")

            train = train.head(max(0, len(train) - tr_tail))
            val = val.slice(v_head).head(max(0, len(val) - v_head - v_tail))
            oos = oos.slice(o_head).head(max(0, len(oos) - o_head - o_tail))
            unseen = unseen.slice(u_head)

    # Drop the helper column
    train = train.drop("__split_date")
    val = val.drop("__split_date")
    oos = oos.drop("__split_date")
    unseen = unseen.drop("__split_date")

    return train, val, oos, unseen


def split_summary(df: pl.DataFrame, name: str = "chimera",
                  boundaries: SplitBoundaries | None = None) -> dict:
    """Return summary stats per split. Useful for sanity-checking before training."""
    train, val, oos, unseen = split_chimera(df, boundaries=boundaries)
    return {
        "name": name,
        "total_rows": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "oos_rows": len(oos),
        "unseen_rows": len(unseen),
        "train_pct": len(train) / max(len(df), 1),
        "val_pct": len(val) / max(len(df), 1),
        "oos_pct": len(oos) / max(len(df), 1),
        "unseen_pct": len(unseen) / max(len(df), 1),
    }


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--cadence", default="1d")
    args = ap.parse_args()

    boundaries = get_split_dates()
    print(f"[split] Frozen dates: {json.dumps(boundaries.as_dict(), indent=2)}")
    print()

    loader = ChimeraLoader()
    df = loader.load(args.asset, cadence=args.cadence)
    print(f"[split] Loaded {args.asset} {args.cadence}: {len(df)} rows")

    summary = split_summary(df, name=f"{args.asset}_{args.cadence}", boundaries=boundaries)
    print()
    print(f"[split] Split sizes:")
    for k, v in summary.items():
        if k == "name":
            continue
        if k.endswith("_pct"):
            print(f"  {k:<14}  {v:.1%}")
        else:
            print(f"  {k:<14}  {v:,}")


if __name__ == "__main__":
    main()
