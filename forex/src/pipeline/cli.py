"""Standard CLI surface for forex pipeline producers.

Canonical flags shared across every producer so behaviour stays consistent:
    --workers N              parallel worker count
    --force                  force rebuild ignoring fresh checks
    --universe u7/u28/u52    resolves to a pair list via UniverseLoader
    --pairs EURUSD ...       explicit override (EUR/USD | EURUSD | eur_usd ok)
    --source dukascopy|...   data source selector (fetchers only)
    --dry-run                print resolved task list and exit
    --start / --end          date window (YYYY-MM-DD)

Universe propagation is explicit and LOUD; fallbacks announce themselves (no
silent under-coverage).
"""
from __future__ import annotations

import argparse
import datetime
from typing import Optional

# Resolve UniverseLoader once, via dual-path fallback (works whether src/ or
# src/pipeline/ is on sys.path).
try:
    from universe_loader import UniverseLoader as _UniverseLoader  # type: ignore
except ImportError:  # pragma: no cover - path-dependent
    try:
        from pipeline.universe_loader import UniverseLoader as _UniverseLoader  # type: ignore
    except ImportError:
        _UniverseLoader = None  # type: ignore

try:
    from layout import normalize_pair as _normalize_pair  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from pipeline.layout import normalize_pair as _normalize_pair  # type: ignore
    except ImportError:
        _normalize_pair = None  # type: ignore

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_cli",
    "inputs": {"args": ["argparse.ArgumentParser"]},
    "outputs": {"side_effects": "argparse args; loud universe banner"},
    "invariants": {"loud_universe_announce": True, "no_silent_fallback": True},
    "rationale": "Canonical CLI flags across all producers; eliminates per-script drift.",
}

UNIVERSE_CHOICES = ["u7", "u28", "u52"]
SOURCE_CHOICES = ["dukascopy", "histdata", "oanda"]

# Fallback mirror of config/universes/u7.yaml (source of truth). Used only when
# neither --pairs nor --universe is given.
DEFAULT_U7 = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]


def add_standard_args(
    ap: argparse.ArgumentParser,
    *,
    default_workers: int = 4,
    date_window: bool = True,
    default_start: str = "2010-01-01",
    default_end: Optional[str] = None,
    with_source: bool = False,
) -> argparse.ArgumentParser:
    """Add canonical pipeline flags. Returns the same parser (for chaining)."""
    ap.add_argument("--workers", type=int, default=default_workers,
                    help=f"Parallel worker count (default {default_workers}).")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild even when output is fresh.")
    ap.add_argument("--universe", default=None, choices=UNIVERSE_CHOICES,
                    help="Resolve pairs via UniverseLoader. Default: u7 majors.")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Explicit pair list (EUR/USD | EURUSD | eur_usd). "
                         "Overrides --universe.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print resolved task list and exit without dispatch.")
    if with_source:
        ap.add_argument("--source", default="dukascopy", choices=SOURCE_CHOICES,
                        help="Data source (default dukascopy = free tick bulk archive).")
    if date_window:
        if default_end is None:
            default_end = datetime.date.today().isoformat()
        ap.add_argument("--start", default=default_start,
                        help=f"Start date inclusive (YYYY-MM-DD). Default {default_start}.")
        ap.add_argument("--end", default=default_end,
                        help=f"End date exclusive (YYYY-MM-DD). Default {default_end} (today).")
    return ap


def resolve_pairs(
    args: argparse.Namespace,
    *,
    default: Optional[list[str]] = None,
    stage_name: str = "universe",
) -> list[str]:
    """Resolve --pairs / --universe / fallback to a concrete flat-UPPER pair list.

    Priority: --pairs > --universe > default (u7). Always announces the
    resolution; refuses a silent fallback when a universe was explicitly named.
    """
    if default is None:
        default = list(DEFAULT_U7)

    def _flatten(syms: list[str]) -> list[str]:
        out = []
        for s in syms:
            if _normalize_pair is not None:
                out.append(_normalize_pair(s).flat)
            else:
                out.append(s.upper().replace("/", "").replace("_", ""))
        return out

    if getattr(args, "pairs", None):
        out = _flatten(list(args.pairs))
        print(f"[{stage_name}] pairs: --pairs ({len(out)} explicit)", flush=True)
        return out

    universe = getattr(args, "universe", None)
    if universe:
        if _UniverseLoader is None:
            raise RuntimeError(
                f"[{stage_name}] --universe {universe} requested but UniverseLoader "
                f"is not importable; refusing to silently fall back to a "
                f"{len(default)}-pair default.")
        try:
            raw = _UniverseLoader.load().list(universe)
        except Exception as e:
            raise RuntimeError(
                f"[{stage_name}] --universe {universe} load failed "
                f"({type(e).__name__}: {e}); refusing silent fallback.") from e
        out = _flatten(list(raw))
        print(f"[{stage_name}] universe: {universe} ({len(out)} pairs)", flush=True)
        return out

    print(f"[{stage_name}] universe: default u7 ({len(default)} pairs) "
          f"-- pass --universe u28 to extend", flush=True)
    return list(default)
