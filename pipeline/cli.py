"""One CLI. ``run_pipeline``, ``refresh``, and ``refresh_full`` are not carried over.

    python -m pipeline.cli run --start 2020-01-01 --end 2026-11-20 --universe u10
    python -m pipeline.cli run --market fx --start 2024-01-02 --end 2024-01-02 --universe majors
    python -m pipeline.cli status --data-root data
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Runnable as `python pipeline/cli.py` from the repository root.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.constants import CORE_BAR_TYPES
from pipeline.paths import default_data_root
from pipeline.symbols import parse_symbol_list


def _date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="run accepts --start --end --assets --universe --force --market",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="fetch, build bars, features, dataset, validate")
    run_p.add_argument("--start", required=True, type=_date)
    run_p.add_argument("--end", required=True, type=_date)
    run_p.add_argument("--assets", nargs="*", default=None, help="BTC, BTCUSDT, or BTC/USDT")
    run_p.add_argument("--universe", default=None, help="crypto: u10, u50, u100. fx: majors")
    run_p.add_argument("--market", default="crypto", choices=("crypto", "fx"))
    run_p.add_argument("--stages", default="fetch,bars,features,dataset,validate")
    run_p.add_argument("--bar-types", default=None, help="default: crypto core bars, or fx time bars")
    run_p.add_argument("--data-root", type=Path, default=None)
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--include-oi", action="store_true", help="compute norm_oi_change (flagged)")
    run_p.add_argument("--voladj", action="store_true", help="optional future-vol labels")
    run_p.add_argument("--clipped", action="store_true", help="optional clipped return labels")
    run_p.add_argument("--timeout", type=int, default=120)

    st = sub.add_parser(
        "status",
        help="report, without fetching, how complete each asset/stage is for a range",
    )
    st.add_argument("--start", type=_date, default=None)
    st.add_argument("--end", type=_date, default=None)
    st.add_argument("--assets", nargs="*", default=None)
    st.add_argument("--universe", default=None)
    st.add_argument("--market", default="crypto", choices=("crypto", "fx"))
    st.add_argument("--stages", default="fetch,bars,features,dataset")
    st.add_argument("--bar-types", default=None, help="default: canonical core bars")
    st.add_argument("--data-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Imported after --help so the flag check does not need the data stack.
    from pipeline.config import load_splits, load_windows
    from pipeline.constants import TIME_WIDTH_MS
    from pipeline.orchestrator import RunConfig, run, status
    from pipeline.symbols import canonical_fx_symbol
    root = args.data_root or default_data_root()
    if args.cmd == "status":
        if args.start is None or args.end is None:
            return status(root)
        if args.market == "fx":
            assets = []
            for item in args.assets or []:
                for part in item.split(","):
                    if part.strip():
                        assets.append(canonical_fx_symbol(part))
        else:
            assets = parse_symbol_list(args.assets) if args.assets else []
        if not assets and not args.universe:
            print("status with --start/--end needs --assets or --universe", file=sys.stderr)
            return 2
        if args.bar_types:
            bar_types = [s.strip() for s in args.bar_types.split(",") if s.strip()]
        elif args.market == "fx":
            bar_types = list(TIME_WIDTH_MS)
        else:
            bar_types = list(CORE_BAR_TYPES)
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        cfg = RunConfig(
            start=args.start, end=args.end, assets=assets, universe=args.universe,
            stages=stages, bar_types=bar_types, data_root=root, market=args.market,
        )
        return status(cfg=cfg)
    if args.market == "fx":
        assets = []
        for item in args.assets or []:
            for part in item.split(","):
                if part.strip():
                    assets.append(canonical_fx_symbol(part))
    else:
        assets = parse_symbol_list(args.assets) if args.assets else []
    if not assets and not args.universe:
        print("pass --assets or --universe", file=sys.stderr)
        return 2
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.bar_types:
        bar_types = [s.strip() for s in args.bar_types.split(",") if s.strip()]
    elif args.market == "fx":
        bar_types = list(TIME_WIDTH_MS)
    else:
        bar_types = list(CORE_BAR_TYPES)
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
        market=args.market,
    )
    try:
        return run(cfg)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
