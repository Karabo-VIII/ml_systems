"""
Pipeline Coverage Report — shared utility
=========================================

Drop-in helper for any pipeline stage to emit a uniform end-of-run
"expected vs actual coverage" line so the user can see at a glance
whether the stage produced output for every asset in the requested
universe.

Usage (in any stage script's main()):

    from pipeline.coverage_report import resolve_universe_assets, print_coverage_report

    resolved_assets = resolve_universe_assets(universe="u50",
                                               asset_arg=None,
                                               fallback_assets=DEFAULT_ASSETS)

    # ... run the stage, accumulate per-asset results ...

    print_coverage_report(
        stage_name="hawkes_branching",
        universe=args.universe,
        expected_assets=resolved_assets,
        ok_assets=ok_set,
        err_assets=err_set,
        skipped_assets=skipped_set,
    )

Output looks like:

    ╔═══════════════════════════════════════════════════════════════╗
    ║ COVERAGE — hawkes_branching (universe=u50)                    ║
    ║   Expected: 50 assets                                          ║
    ║   Produced: 47 OK · 1 ERR · 0 SKIP · 2 MISSING                ║
    ║   Missing : ALGOUSDT, BLURUSDT                                 ║
    ║   Errored : SUIUSDT                                            ║
    ║   Status  : INCOMPLETE (94%)                                   ║
    ╚═══════════════════════════════════════════════════════════════╝

Status values:
    OK        — every expected asset produced output, no errors
    PARTIAL   — most produced (>=80%) but some missing/errored
    INCOMPLETE — significant gaps (<80% completion)
    EMPTY     — no output produced
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src" / "pipeline") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src" / "pipeline"))


def resolve_universe_assets(universe: Optional[str] = None,
                             asset_arg: Optional[str] = None,
                             fallback_assets: Optional[Sequence[str]] = None,
                             strip_usdt: bool = True) -> list:
    """Resolve the list of expected assets for a stage.

    Order of precedence:
        asset_arg > universe yaml > fallback_assets > 10-asset majors

    Args:
        universe:        u10 / u50 / u100 (yaml-driven via universe_loader)
        asset_arg:       single asset (e.g. "BTC" or "BTCUSDT")
        fallback_assets: list to use if universe loading fails
        strip_usdt:      if True, return ["BTC", "ETH"]; else ["BTCUSDT", "ETHUSDT"]

    Returns:
        list of asset symbols.
    """
    DEFAULT = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]

    def _norm(a: str) -> str:
        a = a.upper()
        if strip_usdt and a.endswith("USDT"):
            return a[:-4]
        if (not strip_usdt) and (not a.endswith("USDT")):
            return a + "USDT"
        return a

    if asset_arg:
        return [_norm(asset_arg)]

    if universe:
        try:
            from universe_loader import UniverseLoader
            syms = UniverseLoader.load().list(universe)
            return [_norm(s) for s in syms]
        except Exception as e:
            print(f"[coverage] WARN universe={universe} failed to load: {e}; falling back", file=sys.stderr)

    if fallback_assets:
        return [_norm(a) for a in fallback_assets]
    return [_norm(a) for a in DEFAULT]


def _box(lines: list, title: Optional[str] = None) -> str:
    """ASCII-only fixed-width box (Windows cp1252 safe — no Unicode glyphs)."""
    width = max(60, max((len(line) for line in lines), default=60) + 4)
    top = "+" + "-" * (width - 2) + "+"
    if title:
        title_line = f"| {title}".ljust(width - 1) + "|"
    out = [top]
    if title:
        out.append(title_line)
        out.append("|" + "-" * (width - 2) + "|")
    for line in lines:
        out.append("| " + line.ljust(width - 4) + " |")
    out.append(top)
    return "\n".join(out)


def print_coverage_report(stage_name: str,
                           universe: Optional[str],
                           expected_assets: Iterable[str],
                           ok_assets: Iterable[str],
                           err_assets: Optional[Iterable[str]] = None,
                           skipped_assets: Optional[Iterable[str]] = None,
                           extra_lines: Optional[list] = None,
                           file=sys.stdout) -> dict:
    """Print a uniform coverage report and return a summary dict.

    Args:
        stage_name:     human-readable stage key (e.g. "hawkes_branching")
        universe:       universe label or None
        expected_assets: assets the stage SHOULD have produced output for
        ok_assets:       assets that completed successfully
        err_assets:      assets that errored (excluded from ok)
        skipped_assets:  assets that were skipped (e.g. resume / no source data)
        extra_lines:    optional list of additional info lines

    Returns:
        dict with keys: status, n_expected, n_ok, n_err, n_skip, n_missing,
                        missing, errored, skipped, completion_pct
    """
    expected = set(s.upper() for s in expected_assets)
    ok = set(s.upper() for s in ok_assets)
    err = set(s.upper() for s in (err_assets or ()))
    skip = set(s.upper() for s in (skipped_assets or ()))
    accounted = ok | err | skip
    missing = expected - accounted

    n_exp = len(expected)
    n_ok = len(ok)
    n_err = len(err)
    n_skip = len(skip)
    n_miss = len(missing)
    pct = (n_ok / n_exp * 100.0) if n_exp > 0 else 0.0

    if n_exp == 0:
        status = "EMPTY"
    elif n_ok == n_exp:
        status = "OK"
    elif pct >= 80.0:
        status = f"PARTIAL ({pct:.0f}%)"
    else:
        status = f"INCOMPLETE ({pct:.0f}%)"

    title = f"COVERAGE -- {stage_name}" + (f" (universe={universe})" if universe else "")
    lines = [
        f"Expected: {n_exp} assets" + (f"  [{universe}]" if universe else ""),
        f"Produced: {n_ok} OK | {n_err} ERR | {n_skip} SKIP | {n_miss} MISSING",
    ]
    if missing:
        miss_list = ", ".join(sorted(missing)[:10])
        if n_miss > 10:
            miss_list += f", +{n_miss - 10} more"
        lines.append(f"Missing : {miss_list}")
    if err:
        err_list = ", ".join(sorted(err)[:10])
        if n_err > 10:
            err_list += f", +{n_err - 10} more"
        lines.append(f"Errored : {err_list}")
    if skip:
        skip_list = ", ".join(sorted(skip)[:8])
        if n_skip > 8:
            skip_list += f", +{n_skip - 8} more"
        lines.append(f"Skipped : {skip_list}")
    lines.append(f"Status  : {status}")
    if extra_lines:
        lines.extend(extra_lines)

    print("", file=file)
    print(_box(lines, title=title), file=file)
    print("", file=file)

    return {
        "stage": stage_name,
        "universe": universe,
        "status": status,
        "n_expected": n_exp,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_skip": n_skip,
        "n_missing": n_miss,
        "missing": sorted(missing),
        "errored": sorted(err),
        "skipped": sorted(skip),
        "completion_pct": pct,
    }


# Smoke test
if __name__ == "__main__":
    expected = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
    ok = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA"]
    err = ["AVAX"]
    skip = []

    summary = print_coverage_report(
        stage_name="smoke_test",
        universe="u10",
        expected_assets=expected,
        ok_assets=ok,
        err_assets=err,
        skipped_assets=skip,
    )

    assert summary["n_expected"] == 10
    assert summary["n_ok"] == 7
    assert summary["n_err"] == 1
    assert summary["n_missing"] == 2
    assert "LINK" in summary["missing"]
    assert "LTC" in summary["missing"]
    assert summary["status"].startswith("INCOMPLETE"), summary["status"]  # 7/10 = 70% < 80%
    print("PASS: coverage_report smoke")

    # Resolve universe smoke
    a = resolve_universe_assets(asset_arg="BTC")
    assert a == ["BTC"]
    a = resolve_universe_assets(universe="u10")
    print(f"u10 resolved: {a[:5]}... ({len(a)} total)")
