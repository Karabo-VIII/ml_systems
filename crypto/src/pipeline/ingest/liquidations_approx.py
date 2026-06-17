"""Liquidations approximation from aggTrades (existing raw data, no API).

Background: Binance stopped publishing historical liquidation snapshots in 2022
for retail protection. The public real-time WS (!forceOrder@arr) only captures
forward-looking. For backtest, we approximate from aggTrades via the signature:

    Long liquidation:  aggressive MARKET SELL (is_buyer_maker=True) + large size
                       → forced position close on a long
    Short liquidation: aggressive MARKET BUY (is_buyer_maker=False) + large size
                       → forced position close on a short

"Large" is asset-specific (BTC: >$50k, altcoins: >$10k). We use per-asset
percentile thresholds (p99.5 of trade value per day) to normalize.

Daily aggregate per asset:
    liq_long_usd    = sum of qualifying sell-maker trades
    liq_short_usd   = sum of qualifying buy-maker trades
    liq_long_count  = count of qualifying sell-maker trades
    liq_short_count = count of qualifying buy-maker trades
    liq_delta_usd   = short - long (net squeeze pressure)

Hypothesis: days with large liq_long spikes represent capitulation (forced
long-close) and are FOLLOWED by mean-reversion rallies 24-72h later.
Reverse: liq_short spikes represent short squeezes, often near tops.

Output: data/frontier/liquidations/liq_daily_approx.parquet
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
# Pattern P (fix_logs/INDEX.md): aggTrades raw format drifted twice in 2025-Q3
# and 2026-Q1 (us timestamps, unsorted rows). prepare_aggtrades normalizes
# ts scale + sorts. Idempotent — safe to call even when output is order-
# independent (defense-in-depth against future ts-aware logic).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
from _aggtrades_utils import prepare_aggtrades


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("liq", phase, message, **kw)


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
OUT_PATH = OUT_DIR / "liq_daily_approx.parquet"

# Value thresholds per asset (USD) for "potential liquidation" — tuned so ~1%
# of trades qualify. Per-asset thresholds reflect typical trade sizes.
THRESHOLDS_USD = {
    "BTC": 100_000, "ETH": 50_000, "SOL": 25_000, "BNB": 30_000, "XRP": 15_000,
    "DOGE": 10_000, "ADA": 10_000, "AVAX": 10_000, "LINK": 10_000, "LTC": 10_000,
}


def process_day_file(fp: Path, asset: str, threshold_usd: float) -> dict | None:
    try:
        # Pattern P: prepare_aggtrades normalizes timestamp scale (us->ms)
        # and sorts; idempotent. Per fix_logs/INDEX.md mandate "MUST call
        # prepare_aggtrades immediately after any pl.read_parquet(aggTrades_path)".
        raw = prepare_aggtrades(pl.read_parquet(fp))
        df = raw.with_columns(
            (pl.col("price") * pl.col("qty")).alias("value_usd")
        ).filter(pl.col("value_usd") >= threshold_usd)
    except Exception as e:
        # B1 no-silent-failure: surface corrupt aggTrades parquet so the day
        # doesn't silently disappear from the liquidation panel.
        _pl("FAIL", f"liq_read_err: {asset} {fp.name}: {type(e).__name__}: {str(e)[:80]}")
        return None
    if len(df) == 0:
        # still return a zero row so the day is recorded
        date = pd.to_datetime(fp.stem.split("-")[-3] + "-" + fp.stem.split("-")[-2] + "-" + fp.stem.split("-")[-1])
        return {"date": date, "asset": asset, "liq_long_usd": 0.0, "liq_short_usd": 0.0,
                "liq_long_count": 0, "liq_short_count": 0}
    # is_buyer_maker=True  => aggressive SELL hit a resting BID => LONG LIQUIDATION
    # is_buyer_maker=False => aggressive BUY hit a resting ASK  => SHORT LIQUIDATION
    long_liq = df.filter(pl.col("is_buyer_maker"))
    short_liq = df.filter(~pl.col("is_buyer_maker"))

    # Extract date from filename: ASSET-aggTrades-YYYY-MM-DD.parquet
    parts = fp.stem.split("-")
    date_str = "-".join(parts[-3:])
    date = pd.to_datetime(date_str)

    return {
        "date": date,
        "asset": asset,
        "liq_long_usd": float(long_liq["value_usd"].sum()),
        "liq_short_usd": float(short_liq["value_usd"].sum()),
        "liq_long_count": int(len(long_liq)),
        "liq_short_count": int(len(short_liq)),
    }


def process_asset(symbol: str, threshold_usd: float, day_workers: int = 8,
                   fps_subset: list | None = None) -> pd.DataFrame:
    """Build daily liq aggregates for `symbol` over fps_subset (delta) or
    every aggTrades file on disk (full rebuild when fps_subset=None)."""
    if fps_subset is None:
        fps = sorted(glob.glob(str(RAW / symbol / "aggTrades" / f"{symbol}-aggTrades-*.parquet")))
    else:
        fps = list(fps_subset)
    if not fps:
        return pd.DataFrame()
    asset = symbol.replace("USDT", "")
    print(f"[{asset}] {len(fps)} daily files, threshold=${threshold_usd}, "
          f"day_workers={day_workers}", flush=True)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=day_workers) as ex:
        futs = {ex.submit(process_day_file, Path(fp), asset, threshold_usd): fp for fp in fps}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
            done += 1
            if done % 200 == 0:
                _pl("OK", f"{asset}: {done}/{len(fps)}")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=8, date_window=False)
    args = ap.parse_args()

    DEFAULT_THRESH = 100_000
    syms = resolve_assets(args, default=[s + "USDT" for s in THRESHOLDS_USD],
                           stage_name="liq")
    items = [(s.replace("USDT", ""),
              THRESHOLDS_USD.get(s.replace("USDT", ""), DEFAULT_THRESH))
             for s in syms]

    n_default = sum(1 for root, _ in items if root not in THRESHOLDS_USD)
    if n_default:
        missing = [root for root, _ in items if root not in THRESHOLDS_USD]
        print(f"[liq] FALLBACK: {n_default} assets use default ${DEFAULT_THRESH:,} threshold: "
              f"{missing[:5]}{'+' + str(n_default-5) + ' more' if n_default > 5 else ''}",
              flush=True)

    # Build per-asset (date -> filepath) maps for the delta planner.
    per_asset_files: dict[str, dict] = {}
    per_asset_dates: dict[str, list] = {}
    # Phase 8: centralized listing_dates pre-listing filter (defensive;
    # aggTrades inputs already self-filter, but the contract keeps the
    # listing_date_consumer crawler green and protects against future
    # producers that might emit pre-listing aggTrades files).
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing
    except ImportError:
        _is_pre_listing = lambda *a, **k: False
    for root, _thresh in items:
        sym_full = f"{root}USDT"
        fps = sorted(glob.glob(str(RAW / sym_full / "aggTrades" / f"{sym_full}-aggTrades-*.parquet")))
        date_to_fp: dict = {}
        for fp in fps:
            d = _date_from_aggtrades_path(fp)
            if d is not None and not _is_pre_listing(sym_full, d):
                date_to_fp[d] = fp
        if date_to_fp:
            per_asset_files[sym_full] = date_to_fp
            per_asset_dates[root] = sorted(date_to_fp.keys())

    delta = panel_delta_state(
        OUT_PATH, per_asset_dates,
        force=args.force,
        required_cols={"date", "asset", "liq_long_usd", "liq_short_usd"},
        max_null_rate={"liq_long_usd": 0.05, "liq_short_usd": 0.05})
    _pl("BUILD", f"{delta['mode']}: {delta['reason']}")

    if args.dry_run:
        n = sum(len(v) for v in delta["per_asset_new_dates"].values())
        print(f"[liq] dry-run: {delta['mode']}, {n} (asset, date) rows would "
              f"be processed across {len(delta['per_asset_new_dates'])} assets")
        return

    if delta["mode"] == "fresh":
        return

    all_frames = []
    for root, thresh in items:
        sym_full = f"{root}USDT"
        new_dates = delta["per_asset_new_dates"].get(root, [])
        if not new_dates:
            continue
        if delta["mode"] == "rebuild":
            fps_to_run = sorted(per_asset_files.get(sym_full, {}).values())
        else:
            date_to_fp = per_asset_files.get(sym_full, {})
            fps_to_run = [date_to_fp[d] for d in new_dates if d in date_to_fp]
        df = process_asset(sym_full, thresh, day_workers=args.workers,
                            fps_subset=fps_to_run)
        if len(df) > 0:
            all_frames.append(df)

    if not all_frames:
        print("[liq] ERROR: 0 assets produced data; aborting (no panel written)",
              flush=True)
        sys.exit(2)

    new_panel = (pd.concat(all_frames, ignore_index=True)
                 .sort_values(["asset", "date"]).reset_index(drop=True))
    new_panel["liq_delta_usd"] = new_panel["liq_short_usd"] - new_panel["liq_long_usd"]
    new_panel["liq_total_usd"] = new_panel["liq_long_usd"] + new_panel["liq_short_usd"]

    new_panel_pl = pl.from_pandas(new_panel)

    if delta["mode"] == "append":
        append_panel_parquet(OUT_PATH, new_panel_pl,
                              required_cols={"date", "asset", "liq_long_usd",
                                              "liq_short_usd"})
    else:
        atomic_write_parquet(new_panel_pl, OUT_PATH,
                              required_cols={"date", "asset", "liq_long_usd",
                                              "liq_short_usd"})

    print(f"[liq] saved: {OUT_PATH} (mode={delta['mode']}, "
          f"{len(new_panel)} new rows)", flush=True)


if __name__ == "__main__":
    main()
