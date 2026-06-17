"""Multi-asset panel orchestrator — builds the T2 panel layer.

Wraps the existing per-feature panel builders under src/pipeline/{ingest,features}/ (relocated 2026-04-29 from src/frontier/)
and runs them sequentially. Output: data/processed/panels/daily/<panel>_<DATE>.parquet.

These panels are inputs to frontier_consolidator (T3) which joins them
into per-asset frontier silver. Without these panels the v51 chimera
build is partial (validate stage hard-fails on schema_count).

Built panels:
    s3 / basis / liquidations / whale / top_trader / etf

Usage:
    python src/pipeline/build_panels.py                # all panels
    python src/pipeline/build_panels.py --panels s3 basis  # subset
    python src/pipeline/build_panels.py --skip-existing      # don't rebuild fresh
"""
from __future__ import annotations

# CDAP contract — declared after __future__ per PEP-236.
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "build_panels",
    "inputs": {
        "args": ["--panels", "--skip-existing"],
        "upstream": ["data/raw_external/*", "data/raw/*/aggTrades/*.parquet"],
    },
    "outputs": {
        "files": "data/processed/panels/daily/*.parquet",
        "panel_kinds": ["s3", "basis", "liquidations", "whale", "etf",
                        "rv_jumps", "te", "top_trader"],
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "coverage_report_at_end": True,
        "universe_agnostic": True,
    },
}

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

# Panel name -> (script, expected_output_glob_relative_to_processed_panels)
# The glob is checked for "skip if exists" mode and for output verification.
PANEL_BUILDERS = {
    # Glob MUST match the CANONICAL panel output (the file downstream readers
    # consume), not auxiliary per-asset outputs. Provenance: 2026-04-29
    # top_trader failed with FileNotFoundError on s3_metrics_panel.parquet
    # because skip-existing glob `s3_*.parquet` matched per-asset
    # `s3_metrics_daily_<SYM>.parquet` files and silently skipped the s3
    # consolidator -- the canonical panel never (re)wrote.
    "s3":           ("src/pipeline/ingest/binance_s3_metrics.py",
                     "daily/s3_metrics_panel.parquet"),
    "whale":        ("src/pipeline/ingest/whale_activity.py",
                     "daily/whale_activity_daily.parquet"),
    "liquidations": ("src/pipeline/ingest/liquidations_approx.py",
                     "daily/liq_daily_approx.parquet"),
    # liq_features depends on liquidations panel above; ordering matters.
    "liq_features": ("src/pipeline/features/liq_features.py",
                     "daily/liq_features_long.parquet"),
    # spot_klines: fetches Binance Vision daily 1d klines for basis math.
    # Universe-aware. Slow on cold cache (~5-10 min for u50 over 2 years).
    "spot_klines":  ("src/pipeline/ingest/binance_spot_klines.py",
                     "daily/spot_klines_daily.parquet"),
    # basis depends on spot_klines + chimera_legacy perp daily close.
    # Re-enabled 2026-04-29 (G-FRONTIER-004 closed) -- spot_klines now built
    # by the prior stage, chimera_legacy is a Phase-1 prerequisite.
    "basis":        ("src/pipeline/features/basis_signals.py",
                     "daily/basis_features_long.parquet"),
    "top_trader":   ("src/pipeline/features/top_trader_signals.py",
                     "daily/s3_features_long.parquet"),
    "etf":          ("src/pipeline/ingest/etf_flows.py",
                     "daily/btc_etf_flows.parquet"),
    # rv_jumps + te write `<name>_<YYYYMMDD>.parquet` (dated, layout v3).
    # Glob captures the dated canonical panel; only the panel itself starts
    # with this exact prefix, so the glob is unambiguous.
    "rv_jumps":     ("src/pipeline/features/realized_volatility.py",
                     "daily/rv_jump_panel_*.parquet"),
    "te":           ("src/pipeline/features/transfer_entropy_panel.py",
                     "daily/te_panel_*.parquet"),
    # token_unlocks: CONCEDE -- the underlying coingecko fetcher is a
    # heuristic proxy (synthetic days_to_unlock from hash of cg_id). Author
    # explicitly says "DO NOT trade live on this proxy". Real implementation
    # requires paid token-unlock API. Kept here so Option C STUB detector
    # surfaces the gap; NOT wired into chimera registry.
    "token_unlocks":("src/pipeline/ingest/coingecko_unlocks.py",
                     "daily/token_unlocks_*.parquet"),
    # multi_venue_listings: real fetchers (Binance/Bybit/OKX exchangeInfo).
    # Produces an EVENT catalogue; the per-day derive step is multi_venue_features.
    "multi_venue_listings": ("src/pipeline/ingest/multi_venue_listings.py",
                              "daily/multi_venue_listings.parquet"),
    # multi_venue_features depends on multi_venue_listings -- ordered after.
    # Generates per-(date, asset) days_since_listed features for chimera join.
    "multi_venue_features": ("src/pipeline/features/multi_venue_features.py",
                              "daily/multi_venue_features.parquet"),
    # lob_proxy: per-asset/day BAR-LEVEL files (one per (sym, day)). Strategies
    # that need bar resolution read these directly. NOT joined to chimera.
    "lob_proxy":    ("src/pipeline/features/lob_proxy_panel.py",
                     "daily/lob_proxy_*USDT_*.parquet"),
    # lob_proxy_daily depends on lob_proxy -- aggregates the per-(sym, day)
    # bar-level files into a single long-format panel for chimera join.
    "lob_proxy_daily": ("src/pipeline/features/lob_proxy_daily.py",
                         "daily/lob_proxy_daily.parquet"),
    # book_depth_profile_daily (2026-05-03): consumes binance.vision bookDepth
    # historical archive (2024-01 onward, 16+ months for top-10). Distinct
    # stream from lob_proxy (aggTrades-derived); ADDITIVE chimera join.
    "book_depth_profile_daily": ("src/pipeline/features/book_depth_profile_daily.py",
                                  "daily/book_depth_profile_daily.parquet"),
}

PANELS_DIR = PROJECT_ROOT / "data" / "processed" / "panels"


def run_one(panel: str, script: str, universe: str | None = None,
             expected_glob: str | None = None) -> tuple[str, float]:
    """Invoke a panel builder. Streams sub-builder stdout live.

    Threads --universe through to sub-builders that accept it. Per @browser B5,
    only sub-builders with --universe in their CLI get the flag; others
    (etf_flows, top_trader, basis) are universe-agnostic by design.

    Returns (status, elapsed) where status is one of:
      'OK'        rc=0 AND a fresh canonical output file was touched
      'STUB'      rc=0 but NO fresh output (smoke-only / dry run)
      'FAIL'      rc != 0
      'TIMEOUT'   > 1h wall-clock

    The STUB classification (Option C, 2026-04-29) catches the smoke-
    only-stub-passes-as-OK class of false positives; previously these
    showed [OK] in the dashboard while writing nothing to disk.
    """
    cmd = [PYTHON, script]
    UNIVERSE_AWARE = {
        "src/pipeline/ingest/binance_s3_metrics.py",
        "src/pipeline/ingest/whale_activity.py",
        "src/pipeline/ingest/liquidations_approx.py",
        "src/pipeline/ingest/binance_spot_klines.py",
        "src/pipeline/features/realized_volatility.py",
        "src/pipeline/features/transfer_entropy_panel.py",
        "src/pipeline/features/lob_proxy_panel.py",
    }
    if universe and script in UNIVERSE_AWARE:
        cmd.extend(["--universe", universe])
    print(f"  [START] {panel}: {' '.join(cmd[:3])}{' ...' if len(cmd)>3 else ''}", flush=True)
    t0 = time.time()
    # Snapshot pre-run mtime for the canonical output (if any). After the
    # subprocess returns, compare to detect a fresh write -- guards against
    # smoke-only stubs that exit rc=0 without producing data.
    pre_mtime = -1.0
    if expected_glob:
        try:
            matches = list(PANELS_DIR.glob(expected_glob))
            if matches:
                pre_mtime = max(m.stat().st_mtime for m in matches)
        except Exception:
            pre_mtime = -1.0
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=3600)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"  [FAIL] {panel}: exit {proc.returncode}", flush=True)
            return ("FAIL", elapsed)
        # rc=0 -- check whether the canonical output was actually touched.
        if expected_glob:
            try:
                matches = list(PANELS_DIR.glob(expected_glob))
                post_mtime = (max(m.stat().st_mtime for m in matches)
                               if matches else -1.0)
            except Exception:
                post_mtime = -1.0
            # No matches OR post_mtime <= pre_mtime (file not refreshed) -> STUB.
            if post_mtime < 0 or post_mtime <= pre_mtime:
                print(f"  [STUB] {panel}: exit 0 but {expected_glob} "
                      f"not refreshed (smoke-only or dry-run); "
                      f"this panel produced NO new data", flush=True)
                return ("STUB", elapsed)
        print(f"  [OK] {panel}: {elapsed:.1f}s", flush=True)
        return ("OK", elapsed)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {panel}: > 1h", flush=True)
        return ("TIMEOUT", time.time() - t0)


def has_recent_output(glob_rel: str, max_age_days: float = 7) -> bool:
    """True if any matching file is younger than max_age_days."""
    matches = list(PANELS_DIR.glob(glob_rel))
    if not matches:
        return False
    age_days = (time.time() - max(f.stat().st_mtime for f in matches)) / 86400
    return age_days < max_age_days


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", nargs="+", default=list(PANEL_BUILDERS.keys()),
                    choices=list(PANEL_BUILDERS.keys()),
                    help="Panels to build (default: all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip panels with output <7 days old")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Threaded through to universe-aware sub-builders "
                         "(s3, whale, liq, rv_jumps, te). Universe-agnostic "
                         "panels (etf, top_trader) ignore this flag.")
    ap.add_argument("--force", action="store_true",
                    help="Force fresh rebuild: overrides --skip-existing AND deletes "
                         "prior dated panel snapshots in panels/daily/ before rebuild.")
    ap.add_argument("--via-refresh", action="store_true",
                    help="Delegate to the new DAG runner (refresh.py). Smarter "
                         "incremental rebuild via content-hash skip. Recommended "
                         "for new work; legacy path kept for compat.")
    args = ap.parse_args()

    # Opt-in delegation to the DAG runner.
    if args.via_refresh:
        print("[build_panels] --via-refresh: delegating to refresh.py "
              "(target=frontier_silver walks all panel deps)", flush=True)
        cmd = [PYTHON, str(PROJECT_ROOT / "src" / "pipeline" / "refresh.py"),
               "--target", "frontier_silver"]
        if args.universe:
            cmd.extend(["--scope", args.universe])
        if args.force:
            cmd.append("--force")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return proc.returncode

    # @browser B1: --force LOUD; overrides --skip-existing
    if args.force:
        if args.skip_existing:
            print("[FORCE] overriding --skip-existing", flush=True)
            args.skip_existing = False
        n_deleted = 0
        for panel in args.panels:
            _, glob_rel = PANEL_BUILDERS[panel]
            for old in PANELS_DIR.glob(glob_rel):
                try:
                    old.unlink()
                    n_deleted += 1
                except Exception:
                    pass
        print(f"[FORCE] deleted {n_deleted} prior panel snapshots before rebuild", flush=True)

    # @browser B1: universe LOUD
    print(f"\n{'='*70}")
    print(f"BUILD PANELS  panels={args.panels}  skip_existing={args.skip_existing}  "
          f"universe={args.universe or 'sub-builder default (mostly u10)'}")
    print(f"{'='*70}\n")

    n_ok = n_fail = n_skipped = n_stub = 0
    stub_panels: list[str] = []
    total_t = 0.0
    n_total = len(args.panels)
    for i, panel in enumerate(args.panels, start=1):
        script, glob_rel = PANEL_BUILDERS[panel]
        pct = 100.0 * (i - 1) / n_total
        print(f"\n  [Panel {i}/{n_total}] {panel} ({pct:.0f}% suite complete)", flush=True)
        if not (PROJECT_ROOT / script).exists():
            print(f"  [SKIP-MISSING] {panel}: builder script not found ({script})", flush=True)
            n_skipped += 1
            continue
        if args.skip_existing and has_recent_output(glob_rel):
            print(f"  [SKIP-FRESH] {panel}: recent output exists (<7d)", flush=True)
            n_skipped += 1
            continue
        status, t = run_one(panel, script, universe=args.universe,
                             expected_glob=glob_rel)
        total_t += t
        if status == "OK":
            n_ok += 1
        elif status == "STUB":
            n_stub += 1
            stub_panels.append(panel)
        else:
            n_fail += 1

    print(f"\n{'='*70}")
    print(f"DONE: {n_ok} ok / {n_stub} stub / {n_fail} fail / {n_skipped} skipped / {total_t:.1f}s total")
    if stub_panels:
        print(f"STUBS (rc=0 but no panel data written): {', '.join(stub_panels)}")
    print(f"{'='*70}")

    # Panel-level coverage report (not per-asset — panels are multi-asset blobs).
    # Treats each panel as an "asset" for the report's bookkeeping.
    try:
        if str(Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        from coverage_report import print_coverage_report
        ok_panels = set()
        for panel in args.panels:
            _, glob_rel = PANEL_BUILDERS[panel]
            if any(PANELS_DIR.glob(glob_rel)):
                ok_panels.add(panel.upper())
        # NOTE: coverage_report's `expected_assets` param is repurposed here
        # to count PANEL KINDS (not assets). build_panels coverage =
        # "did each requested panel kind produce output?" — so 7 expected
        # means 7 panel kinds (s3, whale, liq, top_trader, etf, rv_jumps, te),
        # NOT 7 assets. The extra_lines header makes this explicit.
        print_coverage_report(
            stage_name="build_panels",
            universe=None,                 # build_panels is universe-agnostic
            expected_assets=[p.upper() for p in args.panels],
            ok_assets=ok_panels,
            err_assets=set(),
            extra_lines=[
                f"NOTE: coverage rows = PANEL KINDS, not assets.",
                f"  Each panel is a multi-asset blob. To check per-asset",
                f"  coverage, inspect the per-panel parquet (e.g. te_panel)",
                f"  or run hawkes/frontier coverage at u50.",
                f"Build summary: {n_ok} new / {n_stub} stub / {n_fail} fail / {n_skipped} skipped",
            ],
        )
    except Exception as e:
        print(f"[coverage] WARN: {type(e).__name__}: {e}", flush=True)
    # Stubs are not failures (rc=0) but degrade coverage; orchestrator can
    # still proceed. Failures (rc!=0) trigger non-zero exit.
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
