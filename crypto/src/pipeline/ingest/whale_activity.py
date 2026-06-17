"""Whale activity — derived from aggTrades large-trade events.

True WhaleAlert-style on-chain transfer feed requires paid API. Instead, we
derive proxy signals from existing aggTrades:

    whale_trade_count       count of trades with value > $1M
    whale_trade_count_500k  count of trades with value > $500K
    whale_buy_usd           sum of value of whale trades hitting the ask (taker buy)
    whale_sell_usd          sum of value of whale trades hitting the bid (taker sell)
    whale_net_usd           whale_buy - whale_sell

These capture "large market-aggressive trades" which correlate strongly with
whale-wallet activity for CEX-held inventory. Not a substitute for on-chain
netflow but a reasonable proxy.

Output:
    data/frontier/whale/whale_activity_daily.parquet
"""
from __future__ import annotations

import concurrent.futures
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Framework primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parquet_io import (atomic_write_parquet, panel_delta_state,
                         append_panel_parquet)
from cli import add_standard_args, resolve_assets
# Pattern P (fix_logs/INDEX.md): aggTrades raw format drifted in 2025-Q3 +
# 2026-Q1 (us timestamps, unsorted rows). prepare_aggtrades normalizes ts
# scale + sorts. Idempotent. Mandate per fix log: call after every aggTrades
# pl.read_parquet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
from _aggtrades_utils import prepare_aggtrades


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("whale", phase, message, **kw)


def _date_from_aggtrades_path(fp):
    """BTCUSDT-aggTrades-2024-01-01.parquet -> date(2024, 1, 1)."""
    parts = Path(fp).stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return pd.Timestamp("-".join(parts[-3:])).date()
    except Exception:
        return None

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "whale_activity_daily.parquet"

WHALE_THRESHOLDS = {
    "BTCUSDT": (500_000, 1_000_000),
    "ETHUSDT": (250_000, 500_000),
    "SOLUSDT": (100_000, 250_000),
    "BNBUSDT": (100_000, 250_000),
    "XRPUSDT": (50_000, 100_000),
    "DOGEUSDT": (50_000, 100_000),
    "ADAUSDT": (50_000, 100_000),
    "AVAXUSDT": (50_000, 100_000),
    "LINKUSDT": (50_000, 100_000),
    "LTCUSDT": (50_000, 100_000),
}


def process_day(fp: Path, t_small: float, t_large: float) -> dict | None:
    try:
        # Pattern P: normalize ts scale + sort (idempotent), then filter.
        raw = prepare_aggtrades(pl.read_parquet(fp))
        df = raw.with_columns(
            (pl.col("price") * pl.col("qty")).alias("value")
        ).filter(pl.col("value") >= t_small)
    except Exception as e:
        # Loud log so corrupt aggTrades parquets surface (otherwise day silently
        # disappears from the whale panel — caller treats None as 'no whales').
        _pl("FAIL", f"whale_read_err: {fp.name}: {type(e).__name__}: {str(e)[:80]}")
        return None
    parts = fp.stem.split("-")
    date = pd.to_datetime("-".join(parts[-3:]))

    if df.is_empty():
        return {
            "date": date,
            "whale_trade_count": 0, "whale_trade_count_500k": 0,
            "whale_buy_usd": 0.0, "whale_sell_usd": 0.0, "whale_net_usd": 0.0,
        }

    pdf = df.to_pandas()
    # is_buyer_maker=False means TAKER BUY (aggressive buy)
    buy_mask = ~pdf["is_buyer_maker"]
    sell_mask = pdf["is_buyer_maker"]
    large_mask = pdf["value"] >= t_large

    # Column order MUST match the existing-panel schema (date, whale_trade_count,
    # whale_trade_count_500k, whale_buy_usd, whale_sell_usd, whale_net_usd) so
    # delta-append can pl.concat without schema-name-order mismatch errors.
    return {
        "date": date,
        "whale_trade_count": int(large_mask.sum()),                          # above 1m
        "whale_trade_count_500k": int((~large_mask).sum() + int(large_mask.sum())),  # all above 500k
        "whale_buy_usd": float(pdf.loc[buy_mask, "value"].sum()),
        "whale_sell_usd": float(pdf.loc[sell_mask, "value"].sum()),
        "whale_net_usd": float(pdf.loc[buy_mask, "value"].sum() - pdf.loc[sell_mask, "value"].sum()),
    }


def process_asset(symbol: str, day_workers: int = 8,
                   fps_subset: list | None = None) -> pd.DataFrame:
    """Process either ALL aggTrades files for symbol (fps_subset=None) or
    a specific subset (delta-mode: only the new dates).
    """
    t_small, t_large = WHALE_THRESHOLDS.get(symbol, (50_000, 100_000))
    if fps_subset is None:
        fps = sorted(glob.glob(str(RAW / symbol / "aggTrades" / f"{symbol}-aggTrades-*.parquet")))
    else:
        fps = list(fps_subset)
    asset = symbol.replace("USDT", "")
    print(f"[{asset}] threshold_small=${t_small}, threshold_large=${t_large}, "
          f"{len(fps)} files, day_workers={day_workers}", flush=True)

    rows = []
    if not fps:
        return pd.DataFrame(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=day_workers) as ex:
        futs = {ex.submit(process_day, Path(fp), t_small, t_large): fp for fp in fps}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                r["asset"] = asset
                rows.append(r)
            if i % 300 == 0:
                _pl("BUILD", f"{asset}: {i}/{len(fps)}")

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=8, date_window=False)
    args = ap.parse_args()

    symbols = resolve_assets(args, default=list(WHALE_THRESHOLDS.keys()),
                              stage_name="whale")

    n_default = sum(1 for s in symbols if s not in WHALE_THRESHOLDS)
    if n_default:
        missing = [s for s in symbols if s not in WHALE_THRESHOLDS]
        print(f"[whale] FALLBACK: {n_default} assets use default (50K,100K) thresholds: "
              f"{missing[:5]}{'+' + str(n_default-5) + ' more' if n_default > 5 else ''}",
              flush=True)

    # Build per-asset (date -> filepath) maps so the delta planner can match
    # candidate dates to the input files we'd actually process.
    per_asset_files: dict[str, dict] = {}
    per_asset_dates: dict[str, list] = {}
    # Phase 8: centralized listing_dates defensive filter
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing
    except ImportError:
        _is_pre_listing = lambda *a, **k: False
    for symbol in symbols:
        fps = sorted(glob.glob(str(RAW / symbol / "aggTrades" / f"{symbol}-aggTrades-*.parquet")))
        date_to_fp: dict = {}
        for fp in fps:
            d = _date_from_aggtrades_path(fp)
            if d is not None and not _is_pre_listing(symbol, d):
                date_to_fp[d] = fp
        if date_to_fp:
            per_asset_files[symbol] = date_to_fp
            per_asset_dates[symbol.replace("USDT", "")] = sorted(date_to_fp.keys())

    # Panel-mode delta (corruption guards: required cols + max null rate
    # on key feature whale_net_usd; if existing panel has too many nulls,
    # full rebuild instead of append-on-corruption).
    delta = panel_delta_state(
        OUT_PATH, per_asset_dates,
        force=args.force,
        required_cols={"date", "asset", "whale_net_usd"},
        max_null_rate={"whale_net_usd": 0.05})
    _pl("BUILD", f"{delta['mode']}: {delta['reason']}")

    if args.dry_run:
        n = sum(len(v) for v in delta["per_asset_new_dates"].values())
        print(f"[whale] dry-run: {delta['mode']}, "
              f"{n} (asset, date) rows would be processed across "
              f"{len(delta['per_asset_new_dates'])} assets")
        return

    if delta["mode"] == "fresh":
        return

    # Process only the new (asset, date) pairs.
    frames = []
    for symbol in symbols:
        asset_root = symbol.replace("USDT", "")
        new_dates = delta["per_asset_new_dates"].get(asset_root, [])
        if not new_dates:
            continue
        if delta["mode"] == "rebuild":
            # Full rebuild: process every file we have for this asset.
            fps_to_run = sorted(per_asset_files.get(symbol, {}).values())
        else:
            # Append: only the files for new dates.
            date_to_fp = per_asset_files.get(symbol, {})
            fps_to_run = [date_to_fp[d] for d in new_dates if d in date_to_fp]
        df = process_asset(symbol, day_workers=args.workers,
                            fps_subset=fps_to_run)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        print("[whale] ERROR: 0 assets produced data; aborting", flush=True)
        sys.exit(2)

    new_rows = pl.from_pandas(pd.concat(frames, ignore_index=True))

    if delta["mode"] == "append":
        append_panel_parquet(OUT_PATH, new_rows,
                              required_cols={"date", "asset"})
    else:  # rebuild
        atomic_write_parquet(new_rows.sort(["asset", "date"]), OUT_PATH,
                              required_cols={"date", "asset"})

    print(f"[whale] saved: {OUT_PATH} (mode={delta['mode']}, "
          f"{len(new_rows)} new rows)", flush=True)


if __name__ == "__main__":
    main()
