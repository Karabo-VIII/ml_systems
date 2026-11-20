"""Sub-bar (1m) liquidation-proxy builder from local tick aggTrades.

WHY THIS EXISTS (Fork B instrumentation, 2026-06-10):
The dead-list refuted liquidation-cascade entries HARD at daily/4h (D47/D48/D52),
and D67 names LEADING/sub-bar data as the ONLY un-refuted fork avenue. But no
sub-4h liquidation test has ever been run because the existing liq_* features are
DAILY aggregates broadcast across intraday bars (artifact A6: .shift(1) at 4h still
leaks 5/6 bars). This tool closes that gap: it re-derives the liquidation proxy at
1-MINUTE resolution directly from the local tick aggTrades (132GB on disk), making
an honest event-clock cascade study possible for the first time.

Methodology is IDENTICAL to the daily approximation (src/pipeline/ingest/
liquidations_approx.py) so numbers reconcile:
    Long liquidation:  aggressive MARKET SELL (is_buyer_maker=True) + value >= per-asset USD threshold
    Short liquidation: aggressive MARKET BUY  (is_buyer_maker=False) + value >= per-asset USD threshold

Per-asset output (data/processed/liq_subbar/<SYM>_1m.parquet), one row per 1m bar:
    minute_ts (13-digit ms, start of minute, UTC)
    open, high, low, close            -- from aggTrades prices
    vol_usd, n_trades                 -- total flow
    buy_aggr_usd, sell_aggr_usd       -- aggressive flow by side (all sizes)
    liq_long_usd,  liq_long_cnt       -- large aggressive sells  (long-liq proxy)
    liq_short_usd, liq_short_cnt      -- large aggressive buys   (short-liq proxy)

Strictly causal: every row is an aggregate of trades INSIDE that minute only.
Downstream consumers must lag by >=1 bar for any conditioning decision.

Run (full u10 build, background-friendly):
  python -m mining.liq_subbar --universe u10 --start-date 2021-01-01 --workers 6
  python -m mining.liq_subbar --assets BTC --start-date 2021-01-01 --end-date 2021-01-31   # smoke
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import glob
import json
import sys
import time
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# liquidations_approx self-bootstraps its own sibling imports (parquet_io, cli, bars)
sys.path.insert(0, str(ROOT / "src" / "pipeline" / "ingest"))

from pipeline.parquet_io import atomic_write_parquet                    # noqa: E402
from pipeline.bars._aggtrades_utils import normalize_ts_to_ms           # noqa: E402
from liquidations_approx import THRESHOLDS_USD                          # noqa: E402

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "liq_subbar"
RUNS_OUT = ROOT / "runs" / "mining"
RUNS_OUT.mkdir(parents=True, exist_ok=True)

# Fallback threshold for assets not in the canonical dict. MUST equal the daily
# tool's DEFAULT_THRESH (liquidations_approx.py) or 1m day-sums will not reconcile.
DEFAULT_THRESHOLD_USD = 100_000

__contract__ = {
    "kind": "producer",
    "inputs": {"aggTrades": "data/raw/<SYM>/aggTrades/<SYM>-aggTrades-YYYY-MM-DD.parquet"},
    "outputs": {"liq_subbar_1m": "data/processed/liq_subbar/<SYM>_1m.parquet"},
    "invariants": {
        "ts_ms_13digit": "minute_ts in [1.5e12, 2.0e12]",
        "causal": "each row aggregates trades strictly inside its minute",
        "method_parity": "threshold + side semantics identical to liquidations_approx.py",
        "atomic_write": "atomic_write_parquet (G-AUDIT-020)",
    },
}

REQUIRED_COLS = {
    "minute_ts", "open", "high", "low", "close", "vol_usd", "n_trades",
    "buy_aggr_usd", "sell_aggr_usd",
    "liq_long_usd", "liq_long_cnt", "liq_short_usd", "liq_short_cnt",
}


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def _asset_key(sym: str) -> str:
    """BTCUSDT -> BTC (threshold dict keys are base-asset)."""
    return sym[:-4] if sym.endswith("USDT") else sym


def _day_files(sym: str, start: dt.date, end: dt.date) -> list[Path]:
    files = sorted(glob.glob(str(RAW / sym / "aggTrades" / f"{sym}-aggTrades-*.parquet")))
    out = []
    for f in files:
        parts = Path(f).stem.split("-")
        try:
            d = dt.date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
        except (ValueError, IndexError):
            continue
        if start <= d <= end:
            out.append(Path(f))
    return out


def process_day(fp: Path, threshold_usd: float) -> pl.DataFrame | None:
    """One aggTrades day file -> 1440-or-fewer 1m bars. Returns None on read failure."""
    try:
        # Pattern P semantics (normalize ts scale, sort asc) but with a STABLE sort:
        # aggTrades has no trade-id column and ~50% of rows share a timestamp, so an
        # unstable sort makes minute open/close non-deterministic (V50->V51 tick_seq
        # bug class). maintain_order=True keeps file order (= exchange id order) on ties.
        df = normalize_ts_to_ms(pl.read_parquet(fp)).sort("timestamp", maintain_order=True)
    except Exception as e:  # B1 no-silent-failure: caller logs the day
        print(f"  READ-FAIL {fp.name}: {type(e).__name__}: {str(e)[:80]}")
        return None
    if df.is_empty():
        return None
    df = df.with_columns([
        (pl.col("price") * pl.col("qty")).alias("value_usd"),
        ((pl.col("timestamp") // 60_000) * 60_000).alias("minute_ts"),
    ])
    is_liq = pl.col("value_usd") >= threshold_usd
    is_sell = pl.col("is_buyer_maker")  # aggressive SELL hit a resting bid
    bars = (
        df.group_by("minute_ts", maintain_order=True)
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("value_usd").sum().alias("vol_usd"),
            pl.len().alias("n_trades"),
            pl.col("value_usd").filter(~is_sell).sum().alias("buy_aggr_usd"),
            pl.col("value_usd").filter(is_sell).sum().alias("sell_aggr_usd"),
            pl.col("value_usd").filter(is_liq & is_sell).sum().alias("liq_long_usd"),
            (is_liq & is_sell).sum().alias("liq_long_cnt"),
            pl.col("value_usd").filter(is_liq & ~is_sell).sum().alias("liq_short_usd"),
            (is_liq & ~is_sell).sum().alias("liq_short_cnt"),
        ])
    )
    # filter()-over-empty-group sums yield null -> 0.0 for flow columns
    flow_cols = ["buy_aggr_usd", "sell_aggr_usd", "liq_long_usd", "liq_short_usd"]
    bars = bars.with_columns([pl.col(c).fill_null(0.0) for c in flow_cols])
    return bars


def build_asset(sym: str, start: dt.date, end: dt.date, workers: int, force: bool) -> dict:
    sym = _norm_sym(sym)
    out_path = OUT_DIR / f"{sym}_1m.parquet"
    if out_path.exists() and not force:
        print(f"[{sym}] SKIP (exists; --force to rebuild): {out_path}")
        return {"sym": sym, "status": "skipped"}
    threshold = THRESHOLDS_USD.get(_asset_key(sym), DEFAULT_THRESHOLD_USD)
    files = _day_files(sym, start, end)
    if not files:
        print(f"[{sym}] NO aggTrades files in [{start}, {end}] -- skipping")
        return {"sym": sym, "status": "no_data"}
    t0 = time.time()
    print(f"[{sym}] {len(files)} day files, threshold ${threshold:,.0f}, workers={workers}")
    frames: list[pl.DataFrame] = []
    fails = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_day, fp, threshold): fp for fp in files}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            done += 1
            if res is None:
                fails += 1
            else:
                frames.append(res)
            if done % 200 == 0:
                print(f"[{sym}] {done}/{len(files)} days ({time.time()-t0:.0f}s)")
    if not frames:
        print(f"[{sym}] ALL {len(files)} days failed -- nothing written")
        return {"sym": sym, "status": "all_failed", "fails": fails}
    full = pl.concat(frames).sort("minute_ts")
    # invariant: 13-digit ms
    ts_min, ts_max = full["minute_ts"].min(), full["minute_ts"].max()
    assert 1.5e12 < ts_min < 2.0e12 and 1.5e12 < ts_max < 2.0e12, \
        f"minute_ts out of 13-digit ms range: [{ts_min}, {ts_max}]"
    atomic_write_parquet(full, out_path, required_cols=REQUIRED_COLS)
    elapsed = time.time() - t0
    meta = {
        "sym": sym, "status": "built", "rows": len(full), "days_ok": len(frames),
        "days_failed": fails, "threshold_usd": threshold,
        "ts_range": [int(ts_min), int(ts_max)], "elapsed_s": round(elapsed, 1),
        "out": str(out_path.relative_to(ROOT)),
        "liq_long_usd_total": float(full["liq_long_usd"].sum()),
        "liq_short_usd_total": float(full["liq_short_usd"].sum()),
    }
    print(f"[{sym}] DONE rows={meta['rows']:,} days={meta['days_ok']} fails={fails} "
          f"({elapsed:.0f}s) -> {out_path.name}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="1m liquidation-proxy builder from aggTrades")
    ap.add_argument("--assets", nargs="+", default=None, help="e.g. BTC ETH SOL")
    ap.add_argument("--universe", default=None, help="u10/u50/u100 (config/universes/)")
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--end-date", default="2026-05-28")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
    else:
        ap.error("provide --assets or --universe")

    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for sym in syms:  # assets serial; per-day threads inside (IO-bound)
        results.append(build_asset(sym, start, end, args.workers, args.force))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = RUNS_OUT / f"liq_subbar_build_{stamp}.json"
    report.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"BUILD REPORT -> {report}")
    built = sum(1 for r in results if r.get("status") == "built")
    failed = sum(1 for r in results if r.get("status") in ("all_failed", "no_data"))
    print(f"SUMMARY: built={built} skipped={sum(1 for r in results if r.get('status')=='skipped')} failed={failed}")
    return 2 if (failed and not built) else 0


if __name__ == "__main__":
    sys.exit(main())
