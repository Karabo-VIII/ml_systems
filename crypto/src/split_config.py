"""Canonical split-boundary single source of truth.

Per R28/R29 audit (2026-05-15): 14+ files across src/analysis/ and
src/strategy/ml/ hardcoded the same split boundaries (TRAIN_END = 2024-05-15
= 1715731200000 ms; VAL_END = 2025-03-15 = 1741996800000 ms). If the
canonical boundary changes, the old approach required 14+ manual edits;
miss one = lookahead bias on that consumer.

This module is the SINGLE SOURCE OF TRUTH. All consumers must import from here.

Per CLAUDE.md invariant: 'Data split: 50/20/20/10 (train/val/oos/unseen).
Unseen segment NEVER touched during development -- reserved for backtesting
only. Purge gap: 400 bars between segments to prevent normalization leakage.'

Usage:
    from split_config import TRAIN_END_MS, VAL_END_MS, OOS_END_MS, UNSEEN_START_MS
    from split_config import TRAIN_END_DATE, VAL_END_DATE, OOS_END_DATE
    from split_config import PURGE_BARS, PURGE_DAYS

For asset-relative or rolling splits (newer convention per R26/R28 used by
Mover Oracle, predict_movers_from_behavior, build_direction_oracle, etc.):
    from split_config import ROLLING_TRAIN_END_DATE, ROLLING_VAL_START_DATE
    from split_config import ROLLING_VAL_END_DATE, ROLLING_TEST_START_DATE
    from split_config import ROLLING_TEST_END_DATE, ROLLING_PURGE_DAYS
"""
from __future__ import annotations

import pandas as pd

__contract__ = {
    "kind": "config_constants",
    "owner": "pipeline/split-discipline",
    "purpose": "single source of truth for temporal split boundaries",
    "invariants": [
        "All TRAIN/VAL/OOS/UNSEEN consumers import from THIS module",
        "Boundaries match CLAUDE.md 50/20/20/10 invariant",
        "Purge gap 400 bars (or 7-30 days depending on cadence) between segments",
        "UNSEEN segment NEVER touched during development; reserved for backtesting",
    ],
}

# ─── LEGACY split (used by src/strategy/ml/ + src/analysis/, pre-2025-Q4) ────
# These are the values that 14+ files were hardcoding identically.
# Anchored to: TRAIN ends 2024-05-15, VAL ends 2025-03-15.
# Originally codified in src/strategy/ml/training_data_extractor.py.
TRAIN_END_DATE = "2024-05-15"   # inclusive
VAL_END_DATE   = "2025-03-15"   # inclusive
OOS_END_DATE   = "2025-12-31"   # inclusive (rest = UNSEEN)
UNSEEN_START_DATE = "2026-01-01" # first day of UNSEEN

TRAIN_END_MS = int(pd.Timestamp(TRAIN_END_DATE).value // 1_000_000)  # 1715731200000
VAL_END_MS   = int(pd.Timestamp(VAL_END_DATE).value // 1_000_000)    # 1741996800000
OOS_END_MS   = int(pd.Timestamp(OOS_END_DATE).value // 1_000_000)
UNSEEN_START_MS = int(pd.Timestamp(UNSEEN_START_DATE).value // 1_000_000)

PURGE_BARS = 400          # bars (cadence-relative -- meaning depends on user)
PURGE_DAYS = 7            # canonical 7-day purge for ML training (Mover Oracle convention)

# ─── ROLLING / CONTEMPORARY split (Mover Oracle / Layer-4 / 2025-Q4 onward) ──
# Used by:
#   scripts/oracle/build_mover_oracle.py
#   scripts/oracle/build_direction_oracle.py
#   scripts/oracle/predict_movers_from_behavior.py
# Convention: rolling window with explicit 7-day purge gap.
# UNSEEN = Jan-Apr 2026 (the 4 months used by v3 paper-trade-replay deploy gate).
ROLLING_TRAIN_END_DATE   = "2025-09-30"
ROLLING_VAL_START_DATE   = "2025-10-07"  # 7d purge after TRAIN
ROLLING_VAL_END_DATE     = "2025-12-31"
ROLLING_TEST_START_DATE  = "2026-01-07"  # 7d purge after VAL
ROLLING_TEST_END_DATE    = "2026-04-30"
ROLLING_PURGE_DAYS       = 7

# ─── v3 paper-trade-replay deploy gate window ────────────────────────────────
# Matches the 4-month UNSEEN window used by paper_trade_replay_v3 + log file
# naming (paper_trade_replay_v3_<BLEND>_<UNIV>_<START>_<END>.json).
V3_REPLAY_START_DATE = "2026-01-01"
V3_REPLAY_END_DATE   = "2026-04-30"
V3_REPLAY_MONTHS = [
    ("Jan 2026", "20260101_20260131", 31),
    ("Feb 2026", "20260201_20260228", 28),
    ("Mar 2026", "20260301_20260331", 31),
    ("Apr 2026", "20260401_20260430", 30),
]


def split_indices(timestamps_ms, mode: str = "legacy"):
    """Return (train_mask, val_mask, oos_mask, unseen_mask) given ms timestamps.

    mode='legacy': uses TRAIN_END_MS / VAL_END_MS / OOS_END_MS / UNSEEN_START_MS
    mode='rolling': uses ROLLING_*_DATE constants converted to ms
    """
    import numpy as np
    ts = np.asarray(timestamps_ms)
    if mode == "rolling":
        tr_end = int(pd.Timestamp(ROLLING_TRAIN_END_DATE).value // 1_000_000)
        vl_st  = int(pd.Timestamp(ROLLING_VAL_START_DATE).value // 1_000_000)
        vl_end = int(pd.Timestamp(ROLLING_VAL_END_DATE).value // 1_000_000)
        ts_st  = int(pd.Timestamp(ROLLING_TEST_START_DATE).value // 1_000_000)
        ts_end = int(pd.Timestamp(ROLLING_TEST_END_DATE).value // 1_000_000)
        return (
            ts <= tr_end,
            (ts >= vl_st) & (ts <= vl_end),
            (ts >= ts_st) & (ts <= ts_end),
            ts > ts_end,
        )
    return (
        ts <= TRAIN_END_MS,
        (ts > TRAIN_END_MS) & (ts <= VAL_END_MS),
        (ts > VAL_END_MS) & (ts <= OOS_END_MS),
        ts > OOS_END_MS,
    )


if __name__ == "__main__":
    print("Split Config -- canonical boundaries (R29)")
    print(f"  LEGACY:  TRAIN_END {TRAIN_END_DATE} ({TRAIN_END_MS}) "
          f"VAL_END {VAL_END_DATE} ({VAL_END_MS})")
    print(f"           OOS_END {OOS_END_DATE} ({OOS_END_MS})")
    print(f"           UNSEEN_START {UNSEEN_START_DATE} ({UNSEEN_START_MS})")
    print(f"  ROLLING: TRAIN_END {ROLLING_TRAIN_END_DATE}  "
          f"VAL {ROLLING_VAL_START_DATE} -> {ROLLING_VAL_END_DATE}  "
          f"TEST {ROLLING_TEST_START_DATE} -> {ROLLING_TEST_END_DATE}")
    print(f"  V3 REPLAY: {V3_REPLAY_START_DATE} -> {V3_REPLAY_END_DATE}")
    print(f"  PURGE: {PURGE_BARS} bars / {PURGE_DAYS}d legacy / {ROLLING_PURGE_DAYS}d rolling")
