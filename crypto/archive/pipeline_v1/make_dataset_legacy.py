# CDAP contract — no `from __future__` in this file so contract goes at top.
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "chimera_legacy",
    "inputs": {
        "args": ["--workers", "--asset", "--assets", "--universe {u10|u50|u100}",
                 "--skip-phase2", "--phase2-only", "--force", "--single-asset (internal)"],
        "upstream": "data/raw/<SYM>USDT/{aggTrades,funding,metrics}/*.parquet",
        "config_keys": ["data.start_date", "data.assets"],
    },
    "outputs": {
        "files": "data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera*_*.parquet",
        "expected_columns": 41,         # 34 base + 7 cross-asset
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "coverage_report_at_end": True,
        "phase2_after_all_phase1": True,    # cross-asset enrichment requires complete phase1
    },
    "rationale": "Gold legacy chimera; V1-V14 inference depends on this output.",
}

"""
V4 Dataset Builder (LEGACY V50) -- Dollar bars + feature engineering + cross-asset enrichment.

Single-file pipeline that produces fully enriched chimera parquets (41 features):
  Phase 1: Per-asset processing (raw trades -> dollar bars -> 34 base features + targets + regime_label)
  Phase 2: Cross-asset enrichment (all assets -> 7 cross-asset features -> 41 total)

Output: data/processed/chimera_legacy/<sym>usdt_v50_chimera_<YYYYMMDD>.parquet
        (41 features, 10 targets, regime_label; date suffix = data-end date in UTC)
Note: Models select features by name from settings.FEATURE_LIST.

Parallelism (--workers N):
  Phase 1 IS parallelized via ProcessPool: spawns N child processes, one
  per asset, each with --workers 1 + polars_threads = cpu // N. Linear
  scaling up to ~4 workers; beyond that polars-thread division floors and
  gains shrink.
  Phase 2 IS parallelized via ThreadPool: min(N, |U|, 8) threads share
  the all_data dict in-memory and write to per-asset distinct paths.
  Phase 2 alone is 15-20 min on u50 (cross-asset join + xd_* compute on
  3000+ bars across 48 assets); ThreadPool brings it to 4-6 min at N=4.
  ProcessPool would be memory-prohibitive (all_data is 1-5 GB at u50);
  ThreadPool works because polars releases the GIL on most ops.
  Net: --workers N speeds up BOTH phases by ~Nx (Phase 2 capped at 8).

Universe scope:
  --universe / --asset filters PHASE 1 ingestion only. Phase 2
  cross-asset enrichment iterates over EVERY chimera_legacy file on disk
  (`_layout.list_v50_assets()`), regardless of the --universe argument.
  Rationale: cross-sectional ranks (xd_momentum_rank, xd_cross_return_mean)
  must be computed against the full investable universe present on disk;
  silently restricting to a subset would give different xd_* values
  depending on which universe was last built, making the feature path-
  dependent. The cost is harmless: Phase 2 just re-enriches each on-disk
  file (idempotent: strips existing xd_* before recompute, line 690-693).
  If you want the legacy chimera for u10 only, delete the non-u10 files
  from data/processed/chimera_legacy/dollar/ before running.

Usage:
    python src/pipeline/make_dataset_legacy.py    # legacy v50 chimera (V1-V14 inference)
    # For v51 SOTA chimera (current pipeline), use: python src/pipeline/make_dataset.py
"""
import os
import polars as pl
import yaml
import sys
import gc
import time
from pathlib import Path
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import sota_shared_logic_v50 as physics
import layout as _layout
from progress import phase_log as _phase_log

def _chimera_legacy_pl(phase, msg, *, counters=None):
    _phase_log("chimera_legacy", phase, msg, counters=counters)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LEGACY_DIR = _layout.chimera_legacy_dir()  # data/processed/chimera_legacy/dollar/

# V4 Required Features (34 base + 8 targets + regime label)
REQUIRED_FEATURES = [
    # Legacy (0-12)
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    # Extended (13-17)
    "norm_ma_distance",
    "norm_whale", "norm_efficiency",
    "norm_return_4", "norm_return_16",
    # Tier 1 (18-20)
    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
    # Hawkes (21-24)
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # Tier 2 IC-boosting (25-29)
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # SOTA Tier 3 (30-33) -- institutional-grade additions
    "norm_yz_volatility", "norm_cs_spread",
    "norm_perm_entropy", "norm_kyle_lambda",
    # Targets
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
    "target_voladj_1", "target_voladj_4", "target_voladj_16", "target_voladj_64",
    "regime_label",
]

# Cross-asset feature names (7 extra dimensions)
XD_FEATURES = [
    "xd_btc_return",         # BTC's norm_return_1 (leader signal, pass-through)
    "xd_btc_volatility",     # BTC's norm_hl_spread (risk regime, pass-through)
    "xd_funding_spread",     # Asset funding minus BTC funding (or BTC vs mean)
    "xd_cross_return_mean",  # Mean of non-BTC/non-self assets' norm_return_1
    "xd_cross_vol_mean",     # Mean of non-BTC/non-self assets' norm_vol_cluster
    "xd_ma_distance",        # Cross-sectional: asset's SMA-200 trend vs market avg
    "xd_momentum_rank",      # Cross-sectional: asset's return rank vs all peers (0-1)
]

# Cross-asset constants
NORM_WINDOW = 200
BTC_SYMBOL = "BTCUSDT"
MAX_JOIN_STALENESS_MS = 10 * 60 * 1000         # 10 min (~2 bars tolerance) for cross-asset joins
MAX_FUNDING_STALENESS_MS = 24 * 60 * 60 * 1000  # 24h for funding (reported 3x/day)
MAX_OI_STALENESS_MS = 48 * 60 * 60 * 1000       # 48h for OI metrics (reported daily)


# =============================================================================
# CONFIG
# =============================================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# PHASE 1: PER-ASSET PROCESSING
# =============================================================================

def standardise_time_column(df, col_name="timestamp"):
    if col_name not in df.columns:
        return df
    if df.schema[col_name] == pl.String:
        df = df.with_columns(
            pl.col(col_name).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .dt.timestamp("ms").cast(pl.Int64)
        )
    else:
        df = df.with_columns(pl.col(col_name).cast(pl.Int64))

    # Normalize timestamp magnitude: microseconds (16-digit) -> milliseconds (13-digit)
    df = df.with_columns(
        pl.when(pl.col(col_name) > 1_000_000_000_000_000)
        .then(pl.col(col_name) // 1000)
        .otherwise(pl.col(col_name))
        .alias(col_name)
    )

    # Validate timestamp range (13-digit ms: 2020-01-01 to 2035-01-01)
    MIN_TS = 1_577_836_800_000  # 2020-01-01 UTC
    MAX_TS = 2_051_222_400_000  # 2035-01-01 UTC
    ts_min = df[col_name].min()
    ts_max = df[col_name].max()
    if ts_min is not None and ts_max is not None:
        if ts_min < MIN_TS or ts_max > MAX_TS:
            _chimera_legacy_pl("WARN", f"Timestamps out of valid range: min={ts_min}, max={ts_max}")

    return df.sort(col_name)


def generate_dollar_bars(df, dollar_threshold, start_cum_val, bar_id_offset):
    """
    Generate dollar bars with proper residual and offset tracking.

    Args:
        df: Trade data with price, qty, timestamp, is_buyer_maker columns
        dollar_threshold: Dollar value per bar
        start_cum_val: Residual dollar value from previous chunk (< threshold)
        bar_id_offset: Starting bar_id for this chunk (cumulative from prior chunks)

    Returns:
        (bars_df, new_residual, new_bar_id_offset)
    """
    df = df.with_columns(
        (pl.col("price") * pl.col("qty")).fill_null(0.0).alias("dollar_value")
    )

    # local_bar_id = floor((running_dollar_total + residual) / threshold)
    # NOTE: floor() gives deterministic >= boundary: when cum_sum exactly equals
    # threshold, floor(1.0) = 1 -> new bar. No ambiguity at boundaries.
    df = df.with_columns(
        ((pl.col("dollar_value").cum_sum() + start_cum_val) / dollar_threshold)
        .floor().cast(pl.Int64).alias("local_bar_id")
    )

    # Apply global offset to make bar_ids unique across all chunks
    df = df.with_columns(
        (pl.col("local_bar_id") + pl.lit(bar_id_offset)).alias("bar_id")
    )

    bars = df.group_by("bar_id", maintain_order=True).agg([
        pl.col("timestamp").last().alias("timestamp"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.col("dollar_value").sum().alias("volume_usd"),
        (pl.col("dollar_value").filter(pl.col("is_buyer_maker") == True).sum())
        .fill_null(0).alias("sell_vol"),
        (pl.col("dollar_value").filter(pl.col("is_buyer_maker") == False).sum())
        .fill_null(0).alias("buy_vol"),
        pl.len().alias("tick_count")
    ])

    # Compute residual and new offset
    chunk_total = df["dollar_value"].sum()
    if chunk_total is None or chunk_total == 0:
        return bars, start_cum_val, bar_id_offset

    # Residual = leftover dollars that didn't complete a full bar
    residual = float((chunk_total + start_cum_val) % dollar_threshold)

    # New offset based on local bar_ids produced
    max_local = df["local_bar_id"].max()
    if max_local is not None:
        if residual > 1e-9:
            # Last bar incomplete -- next chunk continues it (reuse that bar_id)
            new_offset = bar_id_offset + int(max_local)
        else:
            # All bars complete -- next chunk starts fresh
            new_offset = bar_id_offset + int(max_local) + 1
    else:
        new_offset = bar_id_offset

    return bars, residual, new_offset


def process_symbol(symbol, dollar_thresh):
    """Phase 1: Raw trades -> dollar bars -> 34 base features + targets."""
    clean_sym = symbol.replace("/", "").upper()
    asset_raw = RAW_DIR / clean_sym
    agg_path = asset_raw / "aggTrades"
    trade_files = []
    if agg_path.exists():
        trade_files = sorted(list(agg_path.glob("*.parquet")))
    if not trade_files:
        _chimera_legacy_pl("WARN", f"No data for {clean_sym}")
        return

    _chimera_legacy_pl("BUILD", f"Processing {clean_sym} [V4 SOTA]...")
    accumulated_bars = []
    residual_value = 0.0
    bar_id_offset = 0
    failed_chunks = 0

    # Dollar Bars
    for i in tqdm(range(0, len(trade_files), 5), desc="   Sampling"):
        chunk = trade_files[i : i + 5]
        try:
            df_chunk = pl.read_parquet(chunk)
            df_chunk = standardise_time_column(df_chunk, "timestamp")
            bars, residual_value, bar_id_offset = generate_dollar_bars(
                df_chunk, dollar_thresh, residual_value, bar_id_offset
            )
            accumulated_bars.append(bars)
            del df_chunk
            gc.collect()
        except Exception as e:
            failed_chunks += 1
            _chimera_legacy_pl("WARN", f"Chunk {i//5} failed ({chunk[0].name}...): {e}")

    if not accumulated_bars:
        return

    total_chunks = max((len(trade_files) + 4) // 5, 1)
    if failed_chunks > 0:
        fail_pct = failed_chunks / total_chunks * 100
        _chimera_legacy_pl("WARN", f"{failed_chunks}/{total_chunks} chunks failed ({fail_pct:.1f}%)")
        if fail_pct > 5.0:
            _chimera_legacy_pl("FAIL", f"CRITICAL: Aborting {clean_sym}: {fail_pct:.1f}% chunk failure rate exceeds 5% threshold")
            return

    # Merge cross-chunk boundary bars (same bar_id at chunk edges)
    df_full = pl.concat(accumulated_bars).sort("bar_id")
    df_full = df_full.group_by("bar_id", maintain_order=True).agg([
        pl.col("timestamp").last(), pl.col("open").first(), pl.col("high").max(),
        pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
        pl.col("volume_usd").sum(), pl.col("sell_vol").sum(), pl.col("buy_vol").sum(),
        pl.col("tick_count").sum()
    ]).sort("timestamp")

    # -- Join Funding Rate --
    fund_path = asset_raw / "funding"
    if fund_path.exists():
        fund_files = list(fund_path.glob("*.parquet"))
        if fund_files:
            try:
                df_fund = pl.read_parquet(fund_files)
                rename = {}
                if "fundingRate" in df_fund.columns:
                    rename["fundingRate"] = "funding_rate"
                if "calcTime" in df_fund.columns:
                    rename["calcTime"] = "timestamp"
                if rename:
                    df_fund = df_fund.rename(rename)
                df_fund = (
                    standardise_time_column(df_fund, "timestamp")
                    .select(["timestamp", "funding_rate"])
                    .unique(subset=["timestamp"], keep="last")
                    .sort("timestamp")
                )
                df_full = df_full.join_asof(
                    df_fund, on="timestamp", strategy="backward",
                    tolerance=MAX_FUNDING_STALENESS_MS,
                )
            except Exception:
                pass

    # -- Join Open Interest --
    metrics_path = asset_raw / "metrics"
    if metrics_path.exists():
        metrics_files = list(metrics_path.glob("*.parquet"))
        if metrics_files:
            try:
                df_oi = pl.read_parquet(metrics_files)
                if "open_interest_val" in df_oi.columns and "timestamp" in df_oi.columns:
                    df_oi = standardise_time_column(df_oi, "timestamp")
                    df_oi = (
                        df_oi.select(["timestamp", "open_interest_val"])
                        .unique(subset=["timestamp"], keep="last")
                        .sort("timestamp")
                    )
                    df_full = df_full.join_asof(
                        df_oi, on="timestamp", strategy="backward",
                        tolerance=MAX_OI_STALENESS_MS,
                    )
                    _chimera_legacy_pl("OK", f"OI data joined ({len(df_oi)} records)")
                else:
                    _chimera_legacy_pl("WARN", f"OI data missing expected columns")
            except Exception as e:
                _chimera_legacy_pl("WARN", f"OI join failed: {e}")

    # -- Physics Engine (34 base features) --
    _chimera_legacy_pl("BUILD", f"PHYSICS: Computing V4 SOTA Physics (34 base features)...")
    try:
        df_final = physics.calculate_v50_features(df_full)

        # Validate all required features exist
        missing = [c for c in REQUIRED_FEATURES if c not in df_final.columns]
        if missing:
            _chimera_legacy_pl("FAIL", f"CRITICAL: Missing features: {missing}")
            return

        # Compute data-end-date for filename suffix
        if "timestamp" in df_final.columns:
            from datetime import datetime as _dt, timezone as _tz
            ts_max = df_final["timestamp"].max()
            data_end_date = _dt.fromtimestamp(ts_max / 1000.0, tz=_tz.utc).date()
        else:
            from datetime import datetime as _dt, timezone as _tz
            data_end_date = _dt.now(_tz.utc).date()

        LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        out_file = _layout.chimera_v50_path(clean_sym, data_end_date)

        # Atomic write: temp file -> verify -> rename (crash-safe).
        # Validate by COLUMN NAME, not column COUNT — a 56-col chimera missing
        # the 7 xd_* enrichment features would pass a count-only check.
        tmp_file = LEGACY_DIR / f"{clean_sym.lower()}usdt_v50_chimera.tmp.parquet"
        df_final.write_parquet(tmp_file)
        _verify_cols = set(pl.read_parquet_schema(tmp_file).keys())
        missing_required = [c for c in REQUIRED_FEATURES if c not in _verify_cols]
        if missing_required:
            tmp_file.unlink()
            print(f"    [CRITICAL] Phase 1 verification failed: missing {len(missing_required)} cols: "
                  f"{missing_required[:5]}...")
            return
        if out_file.exists():
            out_file.unlink()
        tmp_file.rename(out_file)
        # NOTE: NO gc_older_dated here. Phase 1 produces a 34-feature chimera
        # WITHOUT xd_* — if we GC older snapshots here, Phase 2 failure leaves
        # the canonical dir with a partial chimera AND no fallback. GC is
        # deferred to Phase 2 (after xd_* enrichment validates).

        # Report
        n_bars = len(df_final)
        has_oi = "norm_oi_change" in df_final.columns and df_final["norm_oi_change"].std() > 0.01
        _chimera_legacy_pl("OK", f"Saved: {out_file.name} ({n_bars:,} bars, OI: {'active' if has_oi else 'zero-filled'})")

    except Exception as e:
        _chimera_legacy_pl("FAIL", f"Physics Failed: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# PHASE 2: CROSS-ASSET ENRICHMENT (34 -> 41 features)
# =============================================================================
#
# Cross-asset features capture inter-market dynamics that per-asset features miss:
#   - BTC leader signal (BTC return propagates to alts with lag)
#   - BTC volatility regime (systemic risk indicator)
#   - Funding spread (relative positioning vs BTC / cross-asset mean)
#   - Cross-asset return mean (market breadth, excluding BTC)
#   - Cross-asset volatility mean (systemic risk, excluding BTC)
#
# Design decisions (per oracle red team 2026-02-21):
#   - NO double z-scoring: BTC return/vol are already normalized, passed through directly
#   - Single z-score only for COMPUTED features (spread, cross-means)
#   - BTC funding spread = BTC vs cross-asset mean (NOT self-self = 0)
#   - Cross-means EXCLUDE BTC to decorrelate from xd_btc_return/vol
#   - Clip pass-through features to [-5, 5] for safety


def _rolling_zscore(series: pl.Series, window: int = NORM_WINDOW) -> pl.Series:
    """Rolling z-score with clip and forward-fill (matches sota_shared_logic_v50)."""
    mu = series.rolling_mean(window)
    sigma = series.rolling_std(window)
    z = ((series - mu) / (sigma + 1e-5)).clip(-5.0, 5.0)
    return z.fill_null(strategy="forward").fill_null(0.0)


def _build_slim_cache(all_data: dict) -> dict:
    """One-time per-asset extraction of slim sorted views.

    Hoisting these out of the per-target hot path eliminates O(N^2)
    redundant select+sort calls inside _compute_cross_asset_features.
    Each entry: {'bar': (ts, ret, vol), 'fund': (ts, fund), 'ma': (ts, ma),
                 'ret_named': (ts, _ret_<sym>) for the rank panel}.
    """
    ts_col = "timestamp"
    cache = {}
    for sym, df in all_data.items():
        entry = {}
        # Slim view for cross-asset return / vol mean (reused per target call).
        entry["bar"] = df.select([
            pl.col(ts_col),
            pl.col("norm_return_1").alias("_o_ret"),
            pl.col("norm_vol_cluster").alias("_o_vol"),
        ]).sort(ts_col)
        # Slim view for funding (used for BTC funding pass-through).
        entry["fund"] = df.select([
            pl.col(ts_col),
            pl.col("norm_funding").alias("_o_fund"),
        ]).sort(ts_col)
        # Slim view for cross-sectional MA distance.
        if "norm_ma_distance" in df.columns:
            entry["ma"] = df.select([
                pl.col(ts_col),
                pl.col("norm_ma_distance").alias("_o_ma"),
            ]).sort(ts_col)
        else:
            entry["ma"] = None
        # Per-asset return slim for the rank panel (one join per target).
        if "norm_return_1" in df.columns:
            entry["ret_named"] = df.select([
                pl.col(ts_col),
                pl.col("norm_return_1").alias(f"_ret_{sym}"),
            ]).sort(ts_col)
        else:
            entry["ret_named"] = None
        # BTC return + vol slim for the BTC pass-through path.
        if sym == BTC_SYMBOL:
            entry["btc_bar"] = df.select([
                pl.col(ts_col),
                pl.col("norm_return_1").alias("_btc_ret"),
                pl.col("norm_hl_spread").alias("_btc_vol"),
            ]).sort(ts_col)
            entry["btc_fund"] = df.select([
                pl.col(ts_col),
                pl.col("norm_funding").alias("_btc_fund"),
            ]).sort(ts_col)
        cache[sym] = entry
    return cache


def _compute_cross_asset_features(
    target_symbol: str,
    target_df: pl.DataFrame,
    all_data: dict,
    slim_cache: dict | None = None,
) -> pl.DataFrame:
    """
    Compute 5 cross-asset features for target_symbol using data from all assets.

    Uses join_asof (backward strategy, 30-min tolerance) to align other assets'
    features onto the target asset's timestamps.

    Args:
        slim_cache: Optional precomputed per-asset slim views (built once via
            _build_slim_cache). If None, falls back to recomputing per-call
            (legacy path; ~24x slower at u50). Always pass slim_cache from
            enrich_all_chimera for the fast path.
    """
    btc_symbol = BTC_SYMBOL
    ts_col = "timestamp"

    # Start with the target's full dataframe (sort ONCE; every join_asof below
    # depends on sorted ts so we hoist the sort out of the loop).
    result = target_df.clone().sort(ts_col)

    # ---- Features 1-2: BTC return & volatility (direct pass-through) --------
    if target_symbol == btc_symbol:
        result = result.with_columns([
            pl.col("norm_return_1").clip(-5.0, 5.0).alias("xd_btc_return"),
            pl.col("norm_hl_spread").clip(-5.0, 5.0).alias("xd_btc_volatility"),
        ])
    else:
        btc_entry = slim_cache.get(btc_symbol) if slim_cache else None
        btc_df = all_data.get(btc_symbol)
        if btc_entry is not None and "btc_bar" in btc_entry:
            btc_bar_slim = btc_entry["btc_bar"]
            btc_fund_slim = btc_entry["btc_fund"]
        elif btc_df is not None:
            # Legacy fallback (slim_cache=None): rebuild slim views inline.
            btc_bar_slim = btc_df.select([
                pl.col(ts_col),
                pl.col("norm_return_1").alias("_btc_ret"),
                pl.col("norm_hl_spread").alias("_btc_vol"),
            ]).sort(ts_col)
            btc_fund_slim = btc_df.select([
                pl.col(ts_col),
                pl.col("norm_funding").alias("_btc_fund"),
            ]).sort(ts_col)
        else:
            btc_bar_slim = btc_fund_slim = None

        if btc_bar_slim is not None:
            # Bar-frequency join (30min tolerance) + funding join (24h tolerance).
            # result is sorted at function top; no re-sort required.
            result = result.join_asof(
                btc_bar_slim, on=ts_col, strategy="backward",
                tolerance=MAX_JOIN_STALENESS_MS,
            ).join_asof(
                btc_fund_slim, on=ts_col, strategy="backward",
                tolerance=MAX_FUNDING_STALENESS_MS,
            )
            result = result.with_columns([
                pl.col("_btc_ret").fill_null(0.0).clip(-5.0, 5.0).alias("xd_btc_return"),
                pl.col("_btc_vol").fill_null(0.0).clip(-5.0, 5.0).alias("xd_btc_volatility"),
            ])
            result = result.drop(["_btc_ret", "_btc_vol"])
        else:
            result = result.with_columns([
                pl.lit(0.0).alias("xd_btc_return"),
                pl.lit(0.0).alias("xd_btc_volatility"),
            ])

    # ---- Feature 3: Funding spread ------------------------------------------
    if target_symbol == btc_symbol:
        # P3 FIX: BTC's funding spread (self vs altcoin mean) is near-constant
        # because altcoins move together. Self-referential signal wastes capacity.
        # Set to 0 for BTC -- altcoins still get meaningful (asset vs BTC) spread.
        result = result.with_columns([
            pl.lit(0.0).alias("_raw_funding_spread"),
        ])
    else:
        if "_btc_fund" in result.columns:
            result = result.with_columns([
                (pl.col("norm_funding") - pl.col("_btc_fund").fill_null(0.0)).alias("_raw_funding_spread"),
            ])
            result = result.drop(["_btc_fund"])
        else:
            result = result.with_columns([
                pl.lit(0.0).alias("_raw_funding_spread"),
            ])

    # ---- Features 4-5: Cross-asset return/vol means -------------------------
    non_btc_others = [s for s in all_data
                      if s != target_symbol and s != btc_symbol]

    ret_accum = pl.lit(0.0).alias("_ret_sum")
    vol_accum = pl.lit(0.0).alias("_vol_sum")
    # P2 FIX: Track per-row valid join count (not static n_others)
    # Stale joins produce null -> fill_null(0.0) biases mean low
    ret_valid = pl.lit(0).cast(pl.Int32).alias("_n_ret_valid")
    vol_valid = pl.lit(0).cast(pl.Int32).alias("_n_vol_valid")
    result = result.with_columns([ret_accum, vol_accum, ret_valid, vol_valid])

    n_others = 0
    for other_sym in non_btc_others:
        # Use precomputed slim view if available; else fall back to inline build.
        if slim_cache and other_sym in slim_cache:
            other_slim = slim_cache[other_sym]["bar"]
        else:
            other_df = all_data.get(other_sym)
            if other_df is None:
                continue
            other_slim = other_df.select([
                pl.col(ts_col),
                pl.col("norm_return_1").alias("_o_ret"),
                pl.col("norm_vol_cluster").alias("_o_vol"),
            ]).sort(ts_col)

        # result is already sorted (top of function); no re-sort.
        result = result.join_asof(
            other_slim, on=ts_col, strategy="backward",
            tolerance=MAX_JOIN_STALENESS_MS,
        )

        result = result.with_columns([
            (pl.col("_ret_sum") + pl.col("_o_ret").fill_null(0.0)).alias("_ret_sum"),
            (pl.col("_vol_sum") + pl.col("_o_vol").fill_null(0.0)).alias("_vol_sum"),
            # Increment valid count only when join succeeded (non-null)
            (pl.col("_n_ret_valid") + pl.col("_o_ret").is_not_null().cast(pl.Int32)).alias("_n_ret_valid"),
            (pl.col("_n_vol_valid") + pl.col("_o_vol").is_not_null().cast(pl.Int32)).alias("_n_vol_valid"),
        ])
        result = result.drop(["_o_ret", "_o_vol"])
        n_others += 1

    if n_others > 0:
        # Divide by per-row valid count (clamp to 1 to avoid div-by-zero)
        result = result.with_columns([
            (pl.col("_ret_sum") / pl.col("_n_ret_valid").clip(1)).alias("_raw_cross_return_mean"),
            (pl.col("_vol_sum") / pl.col("_n_vol_valid").clip(1)).alias("_raw_cross_vol_mean"),
        ])
    else:
        result = result.with_columns([
            pl.lit(0.0).alias("_raw_cross_return_mean"),
            pl.lit(0.0).alias("_raw_cross_vol_mean"),
        ])
    result = result.drop(["_ret_sum", "_vol_sum", "_n_ret_valid", "_n_vol_valid"])

    # ---- Feature 6: Cross-sectional MA distance ----------------------------
    # "Is this asset trending more or less than the market?"
    # Uses norm_ma_distance (per-asset SMA-200 distance, already normalized).
    # Cross-asset spread = asset's norm_ma_distance - mean(all other assets').
    # Then rolling z-scored for stationarity.
    if "norm_ma_distance" in target_df.columns:
        # BUG-6 FIX: use per-row valid count (not static n_ma) to handle stale joins
        result = result.with_columns([
            pl.lit(0.0).alias("_ma_sum"),
            pl.lit(0).cast(pl.Int32).alias("_n_ma_valid"),
        ])

        for other_sym in sorted(all_data.keys()):
            if other_sym == target_symbol:
                continue
            # Cache hit: precomputed slim view; cache miss: inline build (legacy).
            if slim_cache and other_sym in slim_cache:
                other_ma = slim_cache[other_sym]["ma"]
                if other_ma is None:
                    continue
            else:
                other_df = all_data.get(other_sym)
                if other_df is None or "norm_ma_distance" not in other_df.columns:
                    continue
                other_ma = other_df.select([
                    pl.col(ts_col),
                    pl.col("norm_ma_distance").alias("_o_ma"),
                ]).sort(ts_col)

            # result already sorted (top of function); no re-sort.
            result = result.join_asof(
                other_ma, on=ts_col, strategy="backward",
                tolerance=MAX_JOIN_STALENESS_MS,
            )
            result = result.with_columns([
                (pl.col("_ma_sum") + pl.col("_o_ma").fill_null(0.0)).alias("_ma_sum"),
                (pl.col("_n_ma_valid") + pl.col("_o_ma").is_not_null().cast(pl.Int32)).alias("_n_ma_valid"),
            ])
            result = result.drop(["_o_ma"])

        result = result.with_columns([
            pl.when(pl.col("_n_ma_valid") > 0)
            .then(pl.col("norm_ma_distance") - pl.col("_ma_sum") / pl.col("_n_ma_valid").clip(1))
            .otherwise(0.0)
            .alias("_raw_xd_ma_distance"),
        ])
        result = result.drop(["_ma_sum", "_n_ma_valid"])
    else:
        result = result.with_columns([
            pl.lit(0.0).alias("_raw_xd_ma_distance"),
        ])

    # ---- Feature 7: Cross-sectional momentum rank ----------------------------
    # "Where does this asset rank in the return leaderboard right now?"
    # Pure relative strength signal — orthogonal to absolute return levels.
    # Rank = (count of assets with lower return) / (N-1), normalized to [0, 1].
    # rank=1.0 = strongest performer, rank=0.0 = weakest.
    # Then z-scored for model consumption.
    all_returns: dict = {}
    for other_sym in sorted(all_data.keys()):
        # Cache hit: reuse precomputed per-asset return slim (named _ret_<sym>).
        if slim_cache and other_sym in slim_cache:
            cached_ret = slim_cache[other_sym]["ret_named"]
            if cached_ret is None:
                continue
            all_returns[other_sym] = cached_ret
            continue
        # Legacy fallback when slim_cache unavailable.
        other_df = all_data.get(other_sym)
        if other_df is None or "norm_return_1" not in other_df.columns:
            continue
        all_returns[other_sym] = other_df.select([
            pl.col(ts_col),
            pl.col("norm_return_1").alias(f"_ret_{other_sym}"),
        ]).sort(ts_col)

    if len(all_returns) > 1:
        # Join all assets' returns onto target timestamps
        rank_df = result.select([pl.col(ts_col)]).sort(ts_col)
        for sym, ret_df in all_returns.items():
            rank_df = rank_df.join_asof(
                ret_df, on=ts_col, strategy="backward",
                tolerance=MAX_JOIN_STALENESS_MS,
            )

        # Compute rank: count how many assets have lower return than target
        target_ret_col = f"_ret_{target_symbol}"
        if target_ret_col in rank_df.columns:
            ret_cols = [c for c in rank_df.columns if c.startswith("_ret_")]
            n_assets = len(ret_cols)
            # Count assets with lower return (row-wise)
            rank_expr = pl.lit(0).cast(pl.Float64)
            for rc in ret_cols:
                if rc == target_ret_col:
                    continue
                rank_expr = rank_expr + (
                    pl.col(rc).fill_null(0.0) < pl.col(target_ret_col).fill_null(0.0)
                ).cast(pl.Float64)
            rank_df = rank_df.with_columns(
                (rank_expr / max(n_assets - 1, 1)).alias("_raw_momentum_rank")
            )
            # Join back to result
            result = result.sort(ts_col).join_asof(
                rank_df.select([pl.col(ts_col), pl.col("_raw_momentum_rank")]).sort(ts_col),
                on=ts_col, strategy="backward",
                tolerance=MAX_JOIN_STALENESS_MS,
            )
        else:
            result = result.with_columns(pl.lit(0.5).alias("_raw_momentum_rank"))
    else:
        result = result.with_columns(pl.lit(0.5).alias("_raw_momentum_rank"))

    # ---- Normalize COMPUTED features only (single z-score) ------------------
    result = result.with_columns([
        _rolling_zscore(result["_raw_funding_spread"]).alias("xd_funding_spread"),
        _rolling_zscore(result["_raw_cross_return_mean"]).alias("xd_cross_return_mean"),
        _rolling_zscore(result["_raw_cross_vol_mean"]).alias("xd_cross_vol_mean"),
        _rolling_zscore(result["_raw_xd_ma_distance"]).alias("xd_ma_distance"),
        _rolling_zscore(result["_raw_momentum_rank"]).alias("xd_momentum_rank"),
    ])

    # Drop raw intermediates
    raw_cols = [c for c in result.columns if c.startswith("_raw_")]
    result = result.drop(raw_cols)

    return result


def enrich_all_chimera(workers: int = 1):
    """
    Phase 2: Enrich all chimera parquet files with cross-asset features in-place.

    Reads all 34-feature chimera files, computes 7 cross-asset features per asset,
    and overwrites each chimera with the 41-feature version.

    Idempotent: strips existing XD columns before recomputing.

    Args:
        workers: Phase 2 thread count (default 1 = serial). ThreadPool used since
            per-asset work is independent and polars releases the GIL on heavy
            ops. Capped to min(workers, |U|, 8) inside.

    Returns True on success, False on failure.
    """
    print(f"\n{'='*70}")
    print("  PHASE 2: CROSS-ASSET FEATURE ENRICHMENT (30 -> 41 features)")
    print(f"{'='*70}")
    print(f"\n  Source: {LEGACY_DIR}")

    # Load all latest-dated chimera files from chimera_legacy/
    asset_paths = []
    for sym_u in _layout.list_v50_assets():
        p = _layout.chimera_v50_latest(sym_u)
        if p is not None and p.exists():
            asset_paths.append((sym_u, p))
    if not asset_paths:
        print("  [ERROR] No v50 chimera files found in chimera_legacy/. Phase 1 may have failed.")
        return False

    all_data = {}
    for symbol, f in asset_paths:
        df = pl.read_parquet(f)
        all_data[symbol] = df
        print(f"  Loaded {symbol}: {len(df):,} bars, {len(df.columns)} cols")

    print(f"\n  Loaded {len(all_data)} assets: {', '.join(sorted(all_data.keys()))}")
    print(f"  Computing 7 cross-asset features per asset...")
    print(f"  Design: NO double z-score, BTC funding vs mean, cross-means exclude BTC\n")

    # Strip any existing XD columns before recomputing (idempotent rerun)
    for symbol in all_data:
        df = all_data[symbol]
        existing_xd = [c for c in df.columns if c.startswith("xd_")]
        if existing_xd:
            all_data[symbol] = df.drop(existing_xd)
            _chimera_legacy_pl("OK", f"INFO: {symbol}: stripped {len(existing_xd)} existing XD columns for recompute")

    # Record pre-enrichment row counts for validation
    pre_enrich_rows = {sym: len(df) for sym, df in all_data.items()}

    n_total = len(all_data)

    # PERF FIX (2026-05-01): hoist per-asset slim view extraction OUT of the
    # per-target hot path. Old code did N x N redundant select+sort calls
    # (~2,300 at u50). Building the cache once here drops that to N
    # extractions total. The N join_asofs per target still happen (necessary
    # for timestamp alignment), but each one operates on an already-extracted
    # already-sorted slim DataFrame -- ~3-5x speedup at u50.
    print(f"  Building slim-view cache for {n_total} assets...", flush=True)
    _slim_t0 = time.time()
    slim_cache = _build_slim_cache(all_data)
    print(f"  Slim cache built in {time.time()-_slim_t0:.1f}s", flush=True)

    def _enrich_one(symbol: str) -> tuple[str, bool, str]:
        """Per-asset Phase 2 worker: compute xd_*, verify, atomic write.

        Returns (symbol, ok, message). Thread-safe: reads all_data + slim_cache
        (read-only after Phase 1 load), writes to per-asset distinct paths.
        """
        try:
            target_df = all_data[symbol]
            xd_df = _compute_cross_asset_features(symbol, target_df, all_data,
                                                   slim_cache=slim_cache)
            if len(xd_df) != pre_enrich_rows[symbol]:
                return (symbol, False,
                        f"row count changed {pre_enrich_rows[symbol]:,} -> {len(xd_df):,}")
            missing = [f for f in XD_FEATURES if f not in xd_df.columns]
            if missing:
                return (symbol, False, f"missing features: {missing}")
            # std/mean sanity (warn-only, not fatal)
            warnings = []
            for feat in XD_FEATURES:
                std = xd_df[feat].std()
                mean = xd_df[feat].mean()
                if std is not None and (std < 0.1 or std > 4.0):
                    warnings.append(f"{feat} std={std:.3f} mean={mean:.3f}")
            if "timestamp" in xd_df.columns:
                from datetime import datetime as _dt, timezone as _tz
                ts_max = xd_df["timestamp"].max()
                data_end_date = _dt.fromtimestamp(ts_max / 1000.0, tz=_tz.utc).date()
            else:
                from datetime import datetime as _dt, timezone as _tz
                data_end_date = _dt.now(_tz.utc).date()
            out_path = _layout.chimera_v50_path(symbol, data_end_date)
            tmp_path = LEGACY_DIR / f"{symbol.lower()}_v50_chimera.tmp.parquet"
            LEGACY_DIR.mkdir(parents=True, exist_ok=True)
            xd_df.write_parquet(tmp_path)
            _verify_cols = set(pl.read_parquet_schema(tmp_path).keys())
            full_required = set(REQUIRED_FEATURES) | set(XD_FEATURES)
            still_missing = full_required - _verify_cols
            if still_missing:
                tmp_path.unlink(missing_ok=True)
                return (symbol, False,
                        f"missing {len(still_missing)} required cols: {sorted(still_missing)[:5]}")
            if out_path.exists():
                out_path.unlink()
            tmp_path.rename(out_path)
            # GC older dated v50 chimera files NOW (after enrichment validated).
            # gc_older_dated is per-prefix so concurrent calls on different
            # symbols don't race.
            _layout.gc_older_dated(LEGACY_DIR, f"{symbol.lower()}_v50_chimera")
            msg = f"-> {out_path.name}: {len(xd_df):,} bars, {len(xd_df.columns)} cols"
            if warnings:
                msg += f"  [warnings: {len(warnings)}]"
            return (symbol, True, msg)
        except Exception as e:
            return (symbol, False, f"{type(e).__name__}: {e}")

    # Phase 2 parallelism: ThreadPool because (a) each per-asset call only
    # reads all_data (no mutation), (b) per-asset writes go to distinct paths,
    # (c) polars releases the GIL on heavy ops -> linear speedup, (d) shared
    # all_data dict (~1-5 GB at u50) makes ProcessPool memory-prohibitive.
    #
    # HARD CAP at 2 (added 2026-05-24 after 3 rc=0xC0000005 ACCESS_VIOLATION
    # crashes on 2026-05-16/21/23). Phase 2 thread peak memory is dominated
    # by Feature 7's rank_df wide join (line 695-747), which materializes
    # N-1 per-asset columns simultaneously: ~1-3 GB per thread for large
    # assets. With 4 concurrent threads + ~4-8 GB shared all_data, peak
    # commit charge exceeds the 32 GB host -> Windows kills with AV.
    # At workers=2: 2 × ~2 GB working + 6 GB all_data = ~10 GB peak, fits.
    # TODO follow-up: replace Feature 7's wide-join with a streaming
    # accumulator (one asset at a time, drop column after extraction).
    # That structural fix would safely allow workers>=4 even at u200+.
    p2_workers_requested = workers or 1
    p2_workers = max(1, min(p2_workers_requested, n_total, 2))
    print(f"  Phase 2 parallelism: {p2_workers} thread(s) "
          f"(requested {p2_workers_requested}, capped at min(workers, |U|, 8))",
          flush=True)

    success_count = 0
    if p2_workers <= 1:
        for i, symbol in enumerate(sorted(all_data.keys()), start=1):
            pct = 100.0 * (i - 1) / n_total
            print(f"  [Phase2 {i}/{n_total}] enriching {symbol} ({pct:.0f}% complete)",
                  flush=True)
            sym, ok, msg = _enrich_one(symbol)
            if ok:
                success_count += 1
                print(f"    {msg}", flush=True)
            else:
                _chimera_legacy_pl("FAIL", f"{sym}: {msg}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        completed = 0
        with ThreadPoolExecutor(max_workers=p2_workers) as ex:
            futures = {ex.submit(_enrich_one, sym): sym for sym in sorted(all_data.keys())}
            for fut in as_completed(futures):
                sym, ok, msg = fut.result()
                completed += 1
                pct = 100.0 * completed / n_total
                if ok:
                    success_count += 1
                    print(f"  [Phase2 {completed}/{n_total} {pct:.0f}%] {sym}  {msg}",
                          flush=True)
                else:
                    print(f"  [Phase2 {completed}/{n_total} {pct:.0f}%] [FAIL] {sym}: {msg}",
                          flush=True)

    print(f"\n  [DONE] Enriched {success_count}/{len(all_data)} assets")
    print(f"  Output: {LEGACY_DIR}/<sym>usdt_v50_chimera_<DATE>.parquet (41 features + regime_label)")

    # 2026-05-17 FAIL-LOUD coverage guard: today's session exposed a silent
    # early-termination at 15/87 with workers=4 + universe=u100 (race + worker
    # exception). The Phase 2 returned True success despite 17% coverage,
    # which downstream chimera_v51 then propagated as backward_compat FAILS
    # across 22 assets. From now on, ANY shortfall vs the expected cohort
    # (all_data keys) is a HARD FAIL with non-zero exit code.
    expected_keys = set(all_data.keys())
    produced_keys = set()
    for f in LEGACY_DIR.glob("*_v50_chimera_*.parquet"):
        sym = f.name.split("_v50_")[0].upper()
        if sym in expected_keys:
            # Verify the file actually has xd_* cols (Phase 2 enrichment ran)
            try:
                cols = pl.read_parquet_schema(f).keys()
                if "xd_btc_return" in cols:
                    produced_keys.add(sym)
            except Exception:
                pass
    missing = sorted(expected_keys - produced_keys)
    if missing:
        print(f"\n  [HARD FAIL] Phase 2 under-coverage: {len(missing)}/{len(expected_keys)} "
              f"missing xd_* enrichment", flush=True)
        print(f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}",
              flush=True)
        print(f"  Re-run: python src/pipeline/make_dataset_legacy.py --phase2-only "
              f"--universe u100 --workers 1   # workers=1 for race-free", flush=True)
        sys.exit(2)  # was _sys.exit (undefined in this scope -> NameError on the
                     # very coverage-fail path this guard exists to handle)
    print(f"  [COVERAGE-OK] {len(produced_keys)}/{len(expected_keys)} assets verified "
          f"with xd_* enrichment", flush=True)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    # P10 FIX: Raise on missing config instead of silently running with 0 assets
    try:
        conf = load_config()
    except FileNotFoundError:
        raise ValueError(
            f"Config file not found: {CONFIG_PATH}\n"
            "Create config/data_config.yaml with asset definitions."
        )
    except Exception as e:
        raise ValueError(f"Failed to load config {CONFIG_PATH}: {e}")

    DEFAULT_THRESH = 100_000  # $100K default for assets not in config

    # Discover assets: config + any raw data directories not in config
    symbols = []
    config_assets = {}
    if 'assets' in conf:
        for s, p in conf['assets'].items():
            if p.get("is_active", True):
                symbols.append(s)
                config_assets[s.replace("/", "").upper()] = p

    # Auto-discover: scan data/raw/ for asset directories with aggTrades
    # Auto-calibrate bar size from universe screener volume data or raw trade volume
    universe_volumes = {}
    universe_path = PROJECT_ROOT / "data" / "prod_state" / "universe.json"
    if universe_path.exists():
        try:
            import json as _json
            with open(universe_path) as _f:
                _universe = _json.load(_f)
            for a in _universe.get("assets", []):
                # asset is lowercase like "btcusdt"
                universe_volumes[a["asset"].upper()] = a.get("volume_24h", 0)
        except Exception:
            pass

    raw_dir = PROJECT_ROOT / conf['data']['raw_dir']
    if raw_dir.exists():
        for d in sorted(raw_dir.iterdir()):
            if d.is_dir() and (d / "aggTrades").exists():
                name = d.name.upper()
                if name.endswith("USDT") and name not in config_assets:
                    pair = name[:-4] + "/USDT"
                    if pair not in symbols:
                        # Calibrate bar size: daily_volume / 288 (target ~5min bars)
                        daily_vol = universe_volumes.get(name, 0)
                        if daily_vol > 0:
                            thresh = max(int(daily_vol / 288), 10_000)
                            print(f"  [AUTO] {pair}: vol=${daily_vol:,.0f}/day -> bar=${thresh:,}")
                        else:
                            # Estimate from raw data: sample first day of trades
                            thresh = DEFAULT_THRESH
                            try:
                                agg_dir = d / "aggTrades"
                                first_file = sorted(agg_dir.glob("*.parquet"))[-1]
                                sample = pl.read_parquet(first_file)
                                if len(sample) > 0 and "price" in sample.columns and "qty" in sample.columns:
                                    day_vol = float((sample["price"] * sample["qty"]).sum())
                                    if day_vol > 0:
                                        thresh = max(int(day_vol / 288), 10_000)
                                        print(f"  [AUTO] {pair}: sampled vol=${day_vol:,.0f}/day -> bar=${thresh:,}")
                            except Exception:
                                print(f"  [AUTO] {pair}: no volume data, using ${DEFAULT_THRESH:,}")
                        symbols.append(pair)
                        config_assets[name] = {"dollar_bar_size": thresh}

    # CLI: --workers for Phase 1 parallelism
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--workers", type=int, default=1,
                    help="Phase 1 + Phase 2 worker count (default 1). "
                         "Phase 1: spawns N child processes (one per asset, "
                         "each with --workers 1, polars_threads=cpu//N). "
                         "4-8 GB RAM per Phase 1 worker; 2-4 fits 32+ GB. "
                         "Phase 2: ThreadPool of min(N, |U|, 8) threads. "
                         "Reuses the in-memory all_data dict (~1-5 GB at u50) "
                         "so no per-worker memory blowup. Polars releases GIL "
                         "on heavy ops -> linear speedup up to ~4-8 threads.")
    _p.add_argument("--skip-phase2", action="store_true",
                    help="Skip cross-asset enrichment (useful for partial / per-asset rebuilds)")
    _p.add_argument("--phase2-only", action="store_true",
                    help="Skip Phase 1 (assume per-asset chimera already on disk) and run "
                         "ONLY the cross-asset xd_* enrichment. Use when a prior Phase 1 "
                         "completed but Phase 2 was interrupted, leaving 41-feature outputs "
                         "without xd_*. Bypasses --workers / --asset / --universe; reads "
                         "every available chimera_legacy/dollar/<sym>_v50_chimera_*.parquet.")
    _p.add_argument("--single-asset", default=None,
                    help="Internal: invoked by parallel workers to build one asset only.")
    _p.add_argument("--single-thresh", type=int, default=None,
                    help="Internal: dollar-bar threshold for --single-asset.")
    _p.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Restrict to a universe (filters config-discovered symbols).")
    _p.add_argument("--asset", default=None,
                    help="Single asset (e.g. BTC). Deprecated alias for --assets [SYM].")
    # 2026-05-21 contract retrofit: --assets plural added.
    _p.add_argument("--assets", nargs="+", default=None,
                    help="Asset list (BTC or BTCUSDT format). Overrides --universe / --asset.")
    _p.add_argument("--force", action="store_true",
                    help="Force fresh rebuild: delete existing dated chimera_legacy "
                         "snapshots in OUT_DIR for the resolved universe before rebuild. "
                         "Bypasses any per-asset cache-hit logic.")
    # Phase 7 bidirectional pattern
    _p.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration (Z->A) for meet-in-middle "
                         "2x speedup. Run two terminals: one without -r, one with. "
                         "Per-asset chimera files are independent; safe with --workers.")
    _args, _ = _p.parse_known_args()

    # Phase 7 bidirectional: reverse symbol list if requested
    if _args.reverse and symbols:
        symbols.reverse()
        print(f"[chimera_legacy] REVERSE mode: iterating {len(symbols)} "
              f"symbols Z->A (meet-in-middle pattern)", flush=True)

    # Phase 8: centralized listing_dates marker. chimera_legacy processes
    # per-asset chimera files that already self-filter to post-listing
    # dates (input bars don't exist pre-listing), but the import + marker
    # keeps the consumer crawler green and documents the contract.
    try:
        import sys as _ld_sys
        from pathlib import Path as _ld_Path
        _ld_sys.path.insert(0, str(_ld_Path(__file__).resolve().parents[1]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing  # noqa: F401
    except ImportError:
        pass

    # @browser B1: --force LOUD + delete prior dated snapshot
    if _args.force:
        print(f"[FORCE] regenerating chimera_legacy from scratch", flush=True)
        # Pre-delete dated snapshots for the resolved universe before Phase 1
        # (snapshots are written per-asset via process_symbol; --force ensures
        # we don't reuse stale outputs).
        # Defer to after universe filter so we delete only resolved-asset snapshots.

    # 2026-05-21 contract retrofit: --assets > --asset > --universe
    if _args.assets:
        target_set = set(a.upper().replace("USDT", "") + "USDT" for a in _args.assets)
        symbols = [s for s in symbols if s.replace("/", "").upper() in target_set]
        print(f"  [--assets] filtered to {len(symbols)} symbols")
    elif _args.asset:
        sym_u = _args.asset.upper().replace("USDT", "")
        target = f"{sym_u}/USDT"
        symbols = [s for s in symbols if s.replace("/", "").upper() == f"{sym_u}USDT"]
        print(f"  [--asset {_args.asset}] filtered to {len(symbols)} symbols")
    elif _args.universe:
        try:
            sys.path.insert(0, str(current_dir))
            from universe_loader import UniverseLoader as _UL
            univ = set(s.upper() for s in _UL.load().list(_args.universe))
            symbols = [s for s in symbols
                       if s.replace("/", "").upper() in univ]
            print(f"  [--universe {_args.universe}] filtered to {len(symbols)} symbols")
        except Exception as e:
            _chimera_legacy_pl("WARN", f"universe={_args.universe} load failed ({e}); using all discovered")

    # Single-asset mode (used by parallel workers)
    if _args.single_asset:
        process_symbol(_args.single_asset, _args.single_thresh or 100_000)
        gc.collect()
        return

    print(f"[START] V4 Data Factory (41 features: 34 base + 7 cross-asset + regime_label)")

    # 2026-05-21 CRITICAL FIX: --force + --phase2-only previously triggered
    # the delete-block below BEFORE the phase2_only short-circuit, which wiped
    # all v50 chimera files when the user just wanted to re-enrich xd_*.
    # Recovery in that scenario requires full Phase 1 rebuild from raw aggTrades.
    # Now: when --phase2-only is set, the delete-block is SKIPPED entirely
    # (--force in phase2-only context means "re-enrich even if xd_* present",
    # not "delete all v50 files"). See enrich_all_chimera() which already
    # strips existing xd_* cols before recomputing (idempotent).
    if _args.phase2_only:
        if _args.force:
            print(f"[--phase2-only] --force noted; will RE-ENRICH xd_* cols (idempotent strip+recompute). "
                  f"NOT deleting v50 chimera files.", flush=True)
        print(f"\n[--phase2-only] skipping Phase 1; running cross-asset xd_* "
              f"enrichment on existing chimera_legacy/dollar/ snapshots")
        ok = enrich_all_chimera(workers=_args.workers)
        sys.exit(0 if ok else 2)

    # @browser B1: --force snapshot deletion BEFORE Phase 1 spawns
    if _args.force:
        n_deleted = 0
        for sym in symbols:
            sym_l = sym.replace("/", "").lower()
            for old in LEGACY_DIR.glob(f"{sym_l}_v50_chimera*.parquet"):
                try:
                    old.unlink()
                    n_deleted += 1
                except Exception:
                    pass
        print(f"[FORCE] deleted {n_deleted} prior chimera_legacy snapshots before rebuild",
              flush=True)

    # 2026-05-21 contract retrofit: pre-flight skip-existing for Phase 1.
    # Skip assets whose v50 chimera output already has today's date stamp.
    # @browser B1: skip is LOUD; --force overrides.
    if not _args.force and symbols:
        from datetime import datetime as _dt2, timezone as _tz2
        _today = _dt2.now(_tz2.utc).date()
        _skipped = []
        _keep = []
        for sym in symbols:
            sym_l = sym.replace("/", "").lower()
            existing = list(LEGACY_DIR.glob(f"{sym_l}_v50_chimera*.parquet"))
            existing_dates = []
            for f in existing:
                stem_parts = f.stem.split("_")
                for token in stem_parts[::-1]:
                    if len(token) == 8 and token.isdigit():
                        try:
                            existing_dates.append(_dt2.strptime(token, "%Y%m%d").date())
                            break
                        except ValueError:
                            pass
            if existing_dates and max(existing_dates) >= _today:
                _skipped.append(sym)
            else:
                _keep.append(sym)
        if _skipped:
            print(f"[chimera_legacy] skip-existing: {len(_skipped)} assets fresh "
                  f"(stamp >= {_today}); --force to rebuild", flush=True)
        symbols = _keep
        if not symbols:
            print("[chimera_legacy] all assets already fresh; nothing to do in Phase 1.", flush=True)

    # Phase 1: Per-asset processing (dollar bars -> 34 features + targets)
    print(f"\n{'='*70}")
    print(f"  PHASE 1: PER-ASSET PROCESSING ({len(symbols)} assets, workers={_args.workers})")
    print(f"{'='*70}")

    phase1_jobs = []
    for sym in symbols:
        clean = sym.replace("/", "").upper()
        thresh = config_assets.get(clean, {}).get("dollar_bar_size", DEFAULT_THRESH)
        phase1_jobs.append((sym, thresh))

    if _args.workers <= 1:
        n_total = len(phase1_jobs)
        for i, (sym, thresh) in enumerate(phase1_jobs, start=1):
            print(f"\n  [Phase1 {i}/{n_total}] processing {sym}", flush=True)
            process_symbol(sym, thresh)
            gc.collect()
    else:
        # Subprocess pool — each asset is a fully isolated child running this
        # script with a single --asset flag. OS reclaims memory between assets.
        # NOTE: process_symbol doesn't currently accept --asset; we drive
        # via env var SINGLE_ASSET_OVERRIDE which Phase 1 honors below.
        import subprocess as _sp
        from pathlib import Path as _P

        log_dir = PROJECT_ROOT / "logs" / "make_dataset_legacy"
        log_dir.mkdir(parents=True, exist_ok=True)

        cpu = os.cpu_count() or 8
        polars_threads = max(2, cpu // _args.workers)
        print(f"  Worker thread budget: cpu={cpu} workers={_args.workers} "
              f"-> polars_threads={polars_threads} per worker")

        running = []
        completed = 0

        def _spawn(sym: str, thresh: int):
            # G-AUDIT-028: thread --force into Phase 1 child cmd explicitly.
            # Parent already deletes prior snapshots before spawn (line ~1067),
            # so children write into empty target -- this is redundant but
            # makes the contract explicit (not relying on implicit emptiness).
            cmd = [sys.executable, str(_P(__file__).resolve()),
                   "--workers", "1", "--skip-phase2",
                   "--single-asset", sym, "--single-thresh", str(thresh)]
            if getattr(_args, "force", False):
                cmd.append("--force")
            log = open(log_dir / f"{sym.replace('/','_')}.log", "w", encoding="utf-8")
            env = os.environ.copy()
            env["POLARS_MAX_THREADS"] = str(polars_threads)
            env["RAYON_NUM_THREADS"] = str(polars_threads)
            return (sym, _sp.Popen(cmd, stdout=log, stderr=_sp.STDOUT, env=env,
                                   cwd=str(PROJECT_ROOT)), log)

        import time as _t
        spawned = 0
        for sym, thresh in phase1_jobs:
            while len(running) >= _args.workers:
                for idx, (s, p, lf) in enumerate(running):
                    rc = p.poll()
                    if rc is not None:
                        lf.close()
                        completed += 1
                        running.pop(idx)
                        marker = "OK" if rc == 0 else f"FAIL(exit={rc})"
                        pct = 100.0 * completed / len(phase1_jobs)
                        print(f"  [Phase1 {completed}/{len(phase1_jobs)}] {s} {marker} "
                              f"({pct:.0f}% complete, {len(running)} in flight)", flush=True)
                        break
                else:
                    _t.sleep(0.3)
                    continue
            running.append(_spawn(sym, thresh))
            spawned += 1
            print(f"  [Phase1 spawn {spawned}/{len(phase1_jobs)}] {sym} "
                  f"(running={len(running)}, completed={completed})", flush=True)
        # drain
        while running:
            for idx, (s, p, lf) in enumerate(running):
                rc = p.poll()
                if rc is not None:
                    lf.close()
                    completed += 1
                    running.pop(idx)
                    marker = "OK" if rc == 0 else f"FAIL(exit={rc})"
                    pct = 100.0 * completed / len(phase1_jobs)
                    print(f"  [Phase1 {completed}/{len(phase1_jobs)}] {s} {marker} "
                          f"({pct:.0f}% complete, {len(running)} in flight)", flush=True)
                    break
            else:
                import time as _t2
                _t2.sleep(0.3)

    # Phase 2: Cross-asset enrichment (34 -> 41 features)
    # Must happen AFTER all per-asset processing (requires all chimera files)
    if _args.skip_phase2:
        print("\n[SKIP] Phase 2 (--skip-phase2): cross-asset enrichment not run.")
        return
    success = enrich_all_chimera(workers=_args.workers)
    if not success:
        # G-AUDIT-023: Phase 2 partial failure was previously WARN-only and
        # the script exited rc=0, so refresh.py marked chimera_legacy SUCCESS
        # while per-asset chimera files shipped with 14 features (no xd_*).
        # Hard-fail so downstream consumers (V1.0/V1.1) see an explicit halt.
        print("[FAIL] Cross-asset enrichment incomplete. Some chimera files "
              "have 14 features only (xd_* missing).")
        print("[FAIL] Models select by FEATURE_LIST name -> training will "
              "silently fail with NaN xd_* cols. Halting.")
        sys.exit(2)

    print(f"\n[DONE] V4 Data Factory complete.")

    # Coverage report (uniform across pipeline stages)
    try:
        sys.path.insert(0, str(current_dir))
        from coverage_report import print_coverage_report
        # Determine which asset built outputs landed
        produced = set()
        for f in LEGACY_DIR.glob("*usdt_v50_chimera*.parquet"):
            stem = f.stem.lower()
            if "usdt_" in stem:
                sym_l = stem.split("usdt_", 1)[0]
                produced.add(sym_l.upper() + "USDT")
        expected = set(s.replace("/", "").upper() for s in symbols)
        ok_set = produced & expected
        err_set = expected - produced
        print_coverage_report(
            stage_name="chimera_legacy",
            universe=_args.universe,
            expected_assets=expected,
            ok_assets=ok_set,
            err_assets=err_set,
            extra_lines=[f"Phase 2 (xsec enrichment): {'OK' if success else 'INCOMPLETE'}",
                         f"Workers: {_args.workers}"],
        )
    except Exception as e:
        print(f"[coverage] WARN: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
