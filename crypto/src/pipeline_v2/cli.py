"""One CLI. ``run_pipeline``, ``refresh``, and ``refresh_full`` are not carried over.

    python -m pipeline_v2.cli run --start 2020-01-01 --end 2026-11-18 --universe u10
    python -m pipeline_v2.cli status --data-root data/pipeline_v2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from pipeline_v2.constants import CORE_BAR_TYPES
from pipeline_v2.contract import load_splits, load_windows
from pipeline_v2.orchestrator import RunConfig, run, status
from pipeline_v2.paths import default_data_root
from pipeline_v2.symbols import parse_symbol_list


def _date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline_v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="fetch, build bars, features, dataset, validate")
    run_p.add_argument("--start", required=True, type=_date)
    run_p.add_argument("--end", required=True, type=_date)
    run_p.add_argument("--assets", nargs="*", default=None, help="BTC, BTCUSDT, or BTC/USDT")
    run_p.add_argument("--universe", default=None, help="u10, u50, or u100")
    run_p.add_argument("--stages", default="fetch,bars,features,dataset,validate")
    run_p.add_argument("--bar-types", default=",".join(CORE_BAR_TYPES))
    run_p.add_argument("--data-root", type=Path, default=None)
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--include-oi", action="store_true", help="compute norm_oi_change (flagged)")
    run_p.add_argument("--voladj", action="store_true", help="optional future-vol labels")
    run_p.add_argument("--clipped", action="store_true", help="optional clipped return labels")
    run_p.add_argument("--timeout", type=int, default=120)

    st = sub.add_parser("status", help="list gold files under the data root")
    st.add_argument("--data-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.data_root or default_data_root()
    if args.cmd == "status":
        return status(root)
    assets = parse_symbol_list(args.assets) if args.assets else []
    if not assets and not args.universe:
        print("pass --assets or --universe", file=sys.stderr)
        return 2
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bar_types = [s.strip() for s in args.bar_types.split(",") if s.strip()]
    cfg = RunConfig(
        start=args.start,
        end=args.end,
        assets=assets,
        universe=args.universe,
        stages=stages,
        bar_types=bar_types,
        data_root=root,
        force=args.force,
        dry_run=args.dry_run,
        include_oi=args.include_oi,
        voladj=args.voladj,
        clipped=args.clipped,
        windows=load_windows(),
        splits=load_splits(),
        fetch_timeout=args.timeout,
    )
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
