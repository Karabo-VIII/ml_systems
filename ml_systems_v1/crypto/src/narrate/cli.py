"""src/narrate/cli.py -- command-line interface for the narrate engine.

Runnable as:
    PYTHONPATH=src python -m narrate --asset BTC --cadence 4h --start 2025-10-01 --end 2025-11-01
    python -m narrate --asset BTC --charts
    python -m narrate --asset ETH --json

Flags:
    --asset       (required) asset symbol, e.g. BTC, BTCUSDT, ETH
    --cadence     chart type / cadence (default: 4h)
    --start       ISO date string for window start (optional)
    --end         ISO date string for window end (optional)
    --charts      run narrate_across_charts instead of single-cadence narrate
    --foundation  enable MOMENT foundation-model layer (with_foundation=True)
    --artifacts   enable artifact layer (with_artifacts=True)
    --json        emit to_dict() JSON instead of prose text

Default output: .to_text() prose.
No emoji (Windows cp1252 safe).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure src/ is on the path when this module is run directly (not via -m with PYTHONPATH already set)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m narrate",
        description="Descriptive market intelligence engine for an asset/period.",
    )
    p.add_argument("--asset", required=True,
                   help="Asset symbol, e.g. BTC or BTCUSDT.")
    p.add_argument("--cadence", default="4h",
                   help="Chart type / cadence (default: 4h). Ignored when --charts is set.")
    p.add_argument("--start", default=None,
                   help="ISO date string for window start, e.g. 2025-10-01. Optional.")
    p.add_argument("--end", default=None,
                   help="ISO date string for window end, e.g. 2025-11-01. Optional.")
    p.add_argument("--charts", action="store_true",
                   help="Run narrate_across_charts (multi-chart comparison) instead of single cadence.")
    p.add_argument("--foundation", action="store_true",
                   help="Enable the MOMENT foundation-model layer (requires momentfm installed).")
    p.add_argument("--artifacts", action="store_true",
                   help="Enable the trained-artifact layer.")
    p.add_argument("--json", action="store_true", dest="emit_json",
                   help="Emit structured JSON (to_dict()) instead of prose text.")
    return p


def main(argv=None) -> int:
    """Entry point. Returns process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.charts:
        # multi-chart comparison mode
        try:
            from narrate.charts import narrate_across_charts
            result = narrate_across_charts(
                asset=args.asset,
                start=args.start,
                end=args.end,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR ({type(exc).__name__}): {exc}")
            return 1

        if args.emit_json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(result.to_text())
        return 0

    # single-cadence mode
    try:
        from narrate import narrate
        nr = narrate(
            asset=args.asset,
            cadence=args.cadence,
            start=args.start,
            end=args.end,
            with_foundation=args.foundation,
            with_artifacts=args.artifacts,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR ({type(exc).__name__}): {exc}")
        return 1

    if args.emit_json:
        print(json.dumps(nr.to_dict(), indent=2, default=str))
    else:
        print(nr.to_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
