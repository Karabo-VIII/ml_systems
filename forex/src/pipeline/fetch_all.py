"""Unified fetch entrypoint — routes to the selected data source.

Default source = Dukascopy (free tick bulk archive; the deep primary). Secondary
sources route to their ingest modules. This is the single command a user runs to
populate the bronze layer.

    # primary: Dukascopy ticks for the 7 majors, last few days
    python src/pipeline/fetch_all.py --universe u7 --start 2026-06-25 --end 2026-07-01

    # a single pair
    python src/pipeline/fetch_all.py --pairs EURUSD --start 2026-06-30 --end 2026-07-01

    # secondary sources
    python src/pipeline/fetch_all.py --source histdata --pairs EURUSD --start 2024-01 --end 2024-03
    python src/pipeline/fetch_all.py --source oanda --pairs EURUSD --granularity 1h --start 2026-06-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli
from pipeline.dispatch import run_per_task
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_all",
    "inputs": {"args": ["--source", "--pairs", "--universe", "--start", "--end", "--workers"]},
    "outputs": {"files": "data/raw/<PAIR>/... (source-dependent)"},
    "invariants": {"source_routing_explicit": True, "no_silent_fallback": True},
    "rationale": "Single fetch entrypoint; bronze layer for everything downstream.",
}


def _run_dukascopy(args) -> None:
    from pipeline.ingest import dukascopy
    pairs = cli.resolve_pairs(args, stage_name="fetch")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    U = UniverseLoader.load()
    throttle = getattr(args, "throttle_ms", 120)
    tasks = [(p, start, end, U.price_divisor(p), args.force, throttle) for p in pairs]
    if args.dry_run:
        for t in tasks:
            print(f"  would fetch {t[0]} {t[1]}..{t[2]} divisor={t[3]}")
        return

    def _worker(pair, s, e, div, force, thr):
        return dukascopy.fetch_pair_range(pair, s, e, div, force=force, throttle_ms=thr)

    run_per_task(tasks, _worker, workers=args.workers,
                 mode="thread" if args.workers > 1 else "serial",
                 stage_name="fetch",
                 progress_summary_keys=["written", "skipped", "closed", "ticks"])


def _run_histdata(args) -> None:
    from pipeline.ingest import histdata
    pairs = cli.resolve_pairs(args, stage_name="fetch")
    tmp = histdata.layout.DATA / "_histdata_tmp"
    # histdata takes YYYY-MM months; tolerate a full YYYY-MM-DD by slicing.
    for p in pairs:
        for y, m in histdata._months(args.start[:7], args.end[:7]):
            csv = histdata.download_month(p, y, m, tmp)
            if csv is None:
                continue
            print(f"[fetch/histdata] {histdata.ingest_csv(Path(csv), p, y, m)}", flush=True)


def _run_oanda(args) -> None:
    from pipeline.ingest import oanda
    pairs = cli.resolve_pairs(args, stage_name="fetch")
    gran = oanda.CADENCE_TO_GRAN[args.granularity]
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    for p in pairs:
        try:
            print(f"[fetch/oanda] {oanda.ingest(p, gran, start, end)}", flush=True)
        except Exception as e:
            print(f"[fetch/oanda] {p} FAIL: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forex unified fetcher")
    cli.add_standard_args(ap, default_workers=4, default_start="2020-01-01", with_source=True)
    ap.add_argument("--throttle-ms", type=int, default=120,
                    help="Dukascopy: polite pause between hourly requests.")
    ap.add_argument("--granularity", default="1h",
                    help="OANDA only: bar cadence (1m/5m/15m/30m/1h/4h/1d).")
    args = ap.parse_args()

    print(f"[fetch] source={args.source}", flush=True)
    if args.source == "dukascopy":
        _run_dukascopy(args)
    elif args.source == "histdata":
        _run_histdata(args)
    elif args.source == "oanda":
        _run_oanda(args)
    else:
        raise SystemExit(f"unknown source {args.source!r}")
    print("[fetch] DONE.", flush=True)


if __name__ == "__main__":
    main()
