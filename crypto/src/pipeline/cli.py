"""Standard CLI surface for pipeline producers.

Replaces ~15 ad-hoc argparse setups. Adds the canonical flags used across
every producer:
    --workers N           parallel worker count
    --force               force rebuild ignoring fresh checks
    --universe u10/u50/u100   resolves to asset list
    --assets BTCUSDT ...  explicit override
    --dry-run             print task list, exit without dispatch
    --start / --end       date window (skip via date_window=False)

Per @browser:
  - B5: universe propagation explicit and LOUD (resolve_assets prints)
  - B1: --force is LOUD; the script must say "[force] rebuilding X"
  - B3: fallbacks announce themselves
"""
from __future__ import annotations

import argparse
import datetime
from typing import Optional

# Resolve UniverseLoader once, at import, via the dual-path fallback (works
# whether src/ or src/pipeline/ is on sys.path). Avoids mutating sys.path on
# every resolve_assets() call (which previously grew sys.path and risked
# shadowing a local universe_loader.py).
try:
    from universe_loader import UniverseLoader as _UniverseLoader  # type: ignore
except ImportError:  # pragma: no cover - path-dependent
    try:
        from pipeline.universe_loader import UniverseLoader as _UniverseLoader  # type: ignore
    except ImportError:
        _UniverseLoader = None  # type: ignore

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_cli",
    "inputs": {"args": ["argparse.ArgumentParser"]},
    "outputs": {"side_effects": "argparse args; loud universe banner"},
    "invariants": {
        "loud_universe_announce": True,
        "no_silent_fallback": True,
    },
    "rationale": "Canonical CLI flags across all producers; eliminates per-script drift.",
}


# Canonical u10 default kept here so producers don't reinvent it.
# NOTE: this is a FALLBACK mirror of config/universes/u10.yaml (the source of
# truth). Used only when neither --assets nor --universe is given. If u10.yaml
# membership changes, update this list too (or prefer --universe u10).
DEFAULT_U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def add_standard_args(
    ap: argparse.ArgumentParser,
    *,
    default_workers: int = 1,
    date_window: bool = True,
    default_start: str = "2023-01-01",
    default_end: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Add canonical pipeline flags to an argparse parser.

    Adds: --workers --force --universe --assets --dry-run [--start --end]

    Returns the same parser (for chaining). Idempotent if already added
    (raises argparse.ArgumentError on conflict — caller should call once).

    Note (2026-05-24 fix): default_start was "2024-01-01" pre-fix, which
    silently dropped all 2023 bars on every fresh rebuild that called this
    helper (4 bar producers + multi_venue features). Default_end was a
    hardcoded "2026-05-01" that grew stale by the day. Now: start anchored
    on Binance Vision earliest-availability (2023-01-01); end computed at
    parse-time as TODAY UTC (exclusive upper bound -> includes through
    yesterday, which is the latest day with a stable Binance Vision file).
    """
    ap.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"Parallel worker count (default {default_workers}). "
             f"Stage chooses process vs thread vs serial; refresh.py's "
             f"--workers overrides this at orchestrator level.")
    ap.add_argument(
        "--force", action="store_true",
        help="Force rebuild even when output is fresh.")
    ap.add_argument(
        "--universe", default=None, choices=["u10", "u50", "u100"],
        help="Resolve assets via UniverseLoader. Default: u10 hardcoded list.")
    ap.add_argument(
        "--assets", nargs="+", default=None,
        help="Explicit asset list (BTCUSDT format, with or without USDT). "
             "Overrides --universe.")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved task list and exit without dispatch.")
    if date_window:
        # Compute default_end at parse time so it always tracks today.
        # Binance Vision daily files lag by ~1 day; exclusive `today` upper
        # bound includes everything through yesterday.
        if default_end is None:
            default_end = datetime.date.today().isoformat()
        ap.add_argument(
            "--start", default=default_start,
            help=f"Start date inclusive (YYYY-MM-DD). Default {default_start} "
                 f"(Binance Vision earliest-availability anchor).")
        ap.add_argument(
            "--end", default=default_end,
            help=f"End date exclusive (YYYY-MM-DD). Default {default_end} "
                 f"(today UTC, computed at parse time).")
    return ap


def resolve_assets(
    args: argparse.Namespace,
    *,
    default: Optional[list[str]] = None,
    suffix: str = "USDT",
    stage_name: str = "universe",
) -> list[str]:
    """Resolve --assets / --universe / fallback to a concrete BTCUSDT-format list.

    Priority:
        --assets        > overrides everything
        --universe X    > UniverseLoader.list(X)
        default         > given default (or DEFAULT_U10)

    Always announces the resolution per @browser B5. Fallbacks announce
    themselves per @browser B3.

    Args:
        args: argparse Namespace with .assets / .universe attributes.
        default: list to use when neither --assets nor --universe given.
        suffix: append "USDT" to bare symbols (set "" to disable).
        stage_name: prefix for the log line (e.g., "[hawkes] universe ...").
    """
    if default is None:
        default = list(DEFAULT_U10)

    def _add_suffix(syms: list[str]) -> list[str]:
        if not suffix:
            return [s.upper() for s in syms]
        return [s.upper() if s.upper().endswith(suffix) else s.upper() + suffix
                 for s in syms]

    if getattr(args, "assets", None):
        out = _add_suffix(list(args.assets))
        print(f"[{stage_name}] universe: --assets ({len(out)} explicit)",
              flush=True)
        return out

    universe = getattr(args, "universe", None)
    if universe:
        if _UniverseLoader is None:
            # No silent downgrade: the caller explicitly asked for a universe;
            # falling back to a smaller default would silently under-cover.
            raise RuntimeError(
                f"[{stage_name}] --universe {universe} requested but UniverseLoader "
                f"is not importable; refusing to silently fall back to a "
                f"{len(default)}-asset default.")
        try:
            raw = _UniverseLoader.load().list(universe)
        except Exception as e:
            raise RuntimeError(
                f"[{stage_name}] --universe {universe} load failed "
                f"({type(e).__name__}: {e}); refusing silent fallback to a "
                f"{len(default)}-asset default.") from e
        out = _add_suffix(list(raw))
        print(f"[{stage_name}] universe: {universe} ({len(out)} assets)", flush=True)
        return out

    print(f"[{stage_name}] universe: default ({len(default)} assets) "
          f"-- pass --universe u50 to extend", flush=True)
    return list(default)
