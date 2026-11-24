"""Contract-level constants. Thresholds and windows live in ``config/pipeline.yaml``."""

from __future__ import annotations

CONTRACT_VERSION = "2.4.1"
PRODUCER = "pipeline"

# Core chart/ML fabrics. Time bars are real OHLCV aggregates, not "last dollar bar".
CORE_BAR_TYPES: tuple[str, ...] = (
    "time_1m",
    "time_5m",
    "time_30m",
    "time_1h",
    "time_2h",
    "time_4h",
    "time_6h",
    "time_8h",
    "time_1d",
    "dollar",
)
OPTIONAL_BAR_TYPES: tuple[str, ...] = ("tick", "volume")
ALL_BAR_TYPES: tuple[str, ...] = CORE_BAR_TYPES + OPTIONAL_BAR_TYPES

# Research fabrics deliberately omitted from v2 (DIB, runs, range, adaptive-vol).
OMITTED_BAR_TYPES: tuple[str, ...] = ("dib", "runs_tick", "runs_volume", "range", "adaptive_vol")

TIME_WIDTH_MS: dict[str, int] = {
    "time_1m": 60_000,
    "time_5m": 5 * 60_000,
    "time_30m": 30 * 60_000,
    "time_1h": 60 * 60_000,
    "time_2h": 2 * 60 * 60_000,
    "time_4h": 4 * 60 * 60_000,
    "time_6h": 6 * 60 * 60_000,
    "time_8h": 8 * 60 * 60_000,
    "time_1d": 24 * 60 * 60_000,
}

DAY_MS = 86_400_000
HOUR_MS = 3_600_000

# 2h, 6h, and 8h (and the rest) divide 24 hours, so every bucket lines up
# with 00:00:00 UTC. close_of uses the Unix epoch, which is midnight UTC.
assert all(DAY_MS % w == 0 for w in TIME_WIDTH_MS.values())

# Peer as-of tolerance for dollar bars (old MAX_JOIN_STALENESS was 10 minutes).
DOLLAR_PEER_TOLERANCE_MS = 10 * 60_000

# Contiguous hole in BTC that fails the gap gate.
BTC_MAX_GAP_MS = 15 * 60_000

HORIZONS: tuple[int, ...] = (1, 4, 16, 64)
CLIP_H1 = 0.15
CLIP_OTHER = 0.50

HISTORY_START = "2020-01-01"

# Columns that are allowed to be constant (identifiers, flags, indicators).
CONSTANT_ALLOW = {
    "asset",
    "bar_type",
    "split",
    "regime_label",
    "is_u10",
    "is_u50",
    "is_u100",
    "is_warmup",
    "asset_dna",
    "listing_date",
    "bar_id",
}

MARKET_COLUMNS = (
    "asset",
    "bar_type",
    "bar_id",
    "ts_open_ms",
    "ts_ms",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "volume_usd",
    "buy_volume",
    "sell_volume",
    "buy_usd",
    "sell_usd",
    "n_trades",
    "signed_usd",
)

# Families that are not ported. Documented so a later reader does not reintroduce them.
DROPPED_FAMILIES = {
    "long_short_ratio": "v1 wrote 1.0 when Vision had no column. Invented constants are forbidden.",
    "token_unlocks": "v1 pointed at ingest/coingecko_unlocks.py, which does not exist.",
    "bookTicker": "declared in the v1 manifest and never implemented.",
    "xex": "deferred. An empty venue frame used to raise KeyError and abort the job.",
    "te": "deferred. The v1 producer emitted 0 rows without a real peer set.",
    "dib_runs_range_adaptive": "research-only; omitted from v2.",
    "target_voladj_as_feature": "optional label only; the denominator is future volatility.",
    "norm_bar_duration_on_time_bars": "the clock is fixed, so the column would be dead.",
}
