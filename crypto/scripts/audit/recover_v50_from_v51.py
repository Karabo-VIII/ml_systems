"""recover_v50_from_v51.py — Recover deleted chimera_legacy/dollar v50 files from v51.

CONTEXT (2026-05-21): an earlier run of
  python src/pipeline/make_dataset_legacy.py --phase2-only --workers 4 --force
hit a bug in make_dataset_legacy.py where --force deletes existing v50 chimera
snapshots BEFORE the --phase2-only short-circuit. Result: all 87 v50 files in
data/processed/chimera_legacy/dollar/ wiped, then Phase 2 had nothing to enrich.

The bug is now fixed (lines 1126-1141 of make_dataset_legacy.py: phase2-only
check moved BEFORE the force-delete block). But the data is gone.

RECOVERY APPROACH:
  v51 chimera files at data/processed/chimera/dollar/ are intact and contain
  58 of the 61 v50-required columns (everything except 3 of the 7 xd_*
  cross-asset features). So we can:
    1. (this script) Reconstruct v50 from v51 by selecting v50 columns +
       DROPPING all xd_* (they'll be recomputed fresh in step 2).
    2. (next step) Run: python src/pipeline/make_dataset_legacy.py --phase2-only
       to recompute all 7 xd_* features against the freshly-restored v50 base.

OUTPUT:
  data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_<YYYYMMDD>.parquet
  (one per asset, ~700-900 MB each; 87 files; ~60-75 GB total)

Usage:
  python scripts/audit/recover_v50_from_v51.py --dry-run    # preview, no writes
  python scripts/audit/recover_v50_from_v51.py              # execute
  python scripts/audit/recover_v50_from_v51.py --assets BTC ETH   # subset

Verification: after recovery, run
  python src/pipeline/make_dataset_legacy.py --phase2-only --workers 1
  python src/audit/check_invariants.py
"""
from __future__ import annotations

__contract__ = {
    "kind": "recovery_script",
    "stage": "recover_v50_from_v51",
    "inputs": {
        "args": ["--assets", "--universe", "--dry-run", "--force"],
        "upstream": "data/processed/chimera/dollar/*_v51_chimera_*.parquet",
    },
    "outputs": {
        "files": "data/processed/chimera_legacy/dollar/*_v50_chimera_*.parquet",
    },
    "invariants": {
        "atomic_write": True,
        "column_allowlist_enforced": True,
        "drops_xd_for_phase2_recompute": True,
        "no_xrel_or_te_or_rv_leakage": True,
    },
}

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
V51_DIR = ROOT / "data" / "processed" / "chimera" / "dollar"
V50_DIR = ROOT / "data" / "processed" / "chimera_legacy" / "dollar"

# Canonical v50 column manifest (mirrors REQUIRED_FEATURES + XD_FEATURES +
# essentials from src/pipeline/make_dataset_legacy.py).
# xd_* DELIBERATELY EXCLUDED — Phase 2 will recompute all 7 from scratch
# (idempotent strip+recompute per line 805-807 of make_dataset_legacy.py).
ESSENTIAL_COLS = [
    "timestamp", "bar_id",
    "open", "high", "low", "close", "volume",
    "volume_usd", "buy_vol", "sell_vol", "tick_count",
]
BASE_FEATURES_34 = [
    # Legacy (0-12)
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    # Extended (13-17)
    "norm_ma_distance", "norm_whale", "norm_efficiency",
    "norm_return_4", "norm_return_16",
    # Tier 1 (18-20)
    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
    # Hawkes (21-24)
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # Tier 2 IC-boosting (25-29)
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # SOTA Tier 3 (30-33)
    "norm_yz_volatility", "norm_cs_spread", "norm_perm_entropy", "norm_kyle_lambda",
]
TARGETS_8 = [
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
    "target_voladj_1", "target_voladj_4", "target_voladj_16", "target_voladj_64",
]
REGIME = ["regime_label"]

V50_MANIFEST = ESSENTIAL_COLS + BASE_FEATURES_34 + TARGETS_8 + REGIME  # 54 cols
# After Phase 2 enrichment: 54 + 7 xd_* = 61 final v50 cols.

# Naming pattern: btcusdt_v51_chimera_20260519.parquet → btcusdt_v50_chimera_20260519.parquet
V51_NAME_RE = re.compile(r"^([a-z0-9]+usdt)_v51_chimera_(\d{8})\.parquet$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Restrict recovery to these assets (BTC or BTCUSDT). Default: all 87.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Restrict via UniverseLoader. Overridden by --assets.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview the per-asset plan; no writes.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing v50 files. Default: skip if v50 already on disk.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallelism (default 1 = serial). Each task reads ~1GB and writes ~800MB; "
                         "IO-bound. Cap 4 to avoid disk thrash.")
    return ap.parse_args()


def resolve_asset_filter(args: argparse.Namespace) -> set | None:
    """Return a set of lowercase 'btcusdt'-style names, or None for all."""
    if args.assets:
        out = set()
        for a in args.assets:
            a_u = a.upper().replace("USDT", "")
            out.add(f"{a_u.lower()}usdt")
        print(f"[recover] --assets filter: {len(out)} assets")
        return out
    if args.universe:
        try:
            sys.path.insert(0, str(ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader  # type: ignore
            syms = UniverseLoader.load().list(args.universe)
            out = set(s.lower() if s.lower().endswith("usdt") else s.lower() + "usdt"
                      for s in syms)
            print(f"[recover] --universe {args.universe}: {len(out)} assets")
            return out
        except Exception as e:
            print(f"[recover] FALLBACK: --universe {args.universe} load failed ({e}); using all")
            return None
    return None


def recover_one(v51_path: Path, force: bool, dry_run: bool) -> dict:
    """Recover one v50 file from v51. Returns result dict."""
    m = V51_NAME_RE.match(v51_path.name)
    if not m:
        return {"path": v51_path.name, "status": "skip", "reason": "name pattern"}
    sym_l, stamp = m.group(1), m.group(2)
    v50_path = V50_DIR / f"{sym_l}_v50_chimera_{stamp}.parquet"

    if v50_path.exists() and not force:
        return {"path": v51_path.name, "status": "skip_exists",
                "reason": f"v50 already at {v50_path.name}"}

    if dry_run:
        return {"path": v51_path.name, "status": "would_recover",
                "v50_path": str(v50_path.name)}

    t0 = time.time()
    # Read v51 schema to determine which manifest cols are actually present.
    schema_cols = set(pl.read_parquet_schema(v51_path).keys())
    cols_to_read = [c for c in V50_MANIFEST if c in schema_cols]
    missing_from_v51 = [c for c in V50_MANIFEST if c not in schema_cols]

    df = pl.read_parquet(v51_path, columns=cols_to_read)
    n_rows = len(df)

    # Atomic write
    V50_DIR.mkdir(parents=True, exist_ok=True)
    tmp = v50_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=3)

    # Schema verification before rename
    written_cols = set(pl.read_parquet_schema(tmp).keys())
    expected = set(cols_to_read)
    if written_cols != expected:
        tmp.unlink(missing_ok=True)
        return {"path": v51_path.name, "status": "fail",
                "reason": f"written schema mismatch: extra={written_cols-expected}, "
                          f"missing={expected-written_cols}"}

    if v50_path.exists():
        v50_path.unlink()
    tmp.rename(v50_path)

    return {
        "path": v51_path.name, "status": "ok",
        "v50_path": v50_path.name,
        "rows": n_rows, "cols_written": len(cols_to_read),
        "cols_missing_in_v51": missing_from_v51,
        "elapsed_s": time.time() - t0,
        "size_mb": v50_path.stat().st_size / 1024 / 1024,
    }


def main():
    args = parse_args()

    if not V51_DIR.exists():
        print(f"[FATAL] v51 source dir missing: {V51_DIR}")
        return 2

    v51_files = sorted(V51_DIR.glob("*_v51_chimera_*.parquet"))
    if not v51_files:
        print(f"[FATAL] no v51 files found at {V51_DIR}")
        return 2

    asset_filter = resolve_asset_filter(args)
    if asset_filter:
        v51_files = [f for f in v51_files
                      if V51_NAME_RE.match(f.name) and
                      V51_NAME_RE.match(f.name).group(1) in asset_filter]
    print(f"[recover] {len(v51_files)} v51 files to process "
          f"(force={args.force}, dry_run={args.dry_run}, workers={args.workers})")
    print(f"[recover] v50 manifest: {len(V50_MANIFEST)} cols "
          f"(11 essential + 34 base + 8 targets + 1 regime; xd_* deliberately dropped)")
    print(f"[recover] xd_* features will be recomputed by --phase2-only after recovery.")
    print()

    t_start = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for i, fp in enumerate(v51_files, 1):
            r = recover_one(fp, force=args.force, dry_run=args.dry_run)
            results.append(r)
            sym = V51_NAME_RE.match(fp.name).group(1)
            stat = r["status"]
            if stat == "ok":
                print(f"  [{i}/{len(v51_files)}] {sym}: OK {r['cols_written']} cols, "
                      f"{r['rows']:,} rows, {r['size_mb']:.0f}MB, {r['elapsed_s']:.1f}s")
            elif stat == "would_recover":
                print(f"  [{i}/{len(v51_files)}] {sym}: WOULD RECOVER -> {r['v50_path']}")
            elif stat == "skip_exists":
                print(f"  [{i}/{len(v51_files)}] {sym}: SKIP ({r['reason']})")
            else:
                print(f"  [{i}/{len(v51_files)}] {sym}: {stat.upper()} {r.get('reason','')}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n_workers = min(args.workers, len(v51_files), 4)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(recover_one, fp, args.force, args.dry_run): fp
                       for fp in v51_files}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                fp = futures[fut]
                sym = V51_NAME_RE.match(fp.name).group(1)
                stat = r["status"]
                print(f"  [{i}/{len(v51_files)}] {sym}: {stat.upper()} "
                      f"{r.get('cols_written','-')} cols  {r.get('elapsed_s', 0):.1f}s",
                      flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] in ("ok", "would_recover"))
    n_skip = sum(1 for r in results if r["status"] in ("skip", "skip_exists"))
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_xd_missing = sum(1 for r in results
                       if r.get("cols_missing_in_v51") and
                       any(c.startswith("xd_") for c in r["cols_missing_in_v51"]))

    print()
    print("=" * 70)
    print(f"RECOVERY SUMMARY  elapsed={elapsed:.0f}s")
    print(f"  OK / would_recover: {n_ok}")
    print(f"  skipped:            {n_skip}")
    print(f"  failed:             {n_fail}")
    if not args.dry_run and n_ok > 0:
        total_size_mb = sum(r.get("size_mb", 0) for r in results)
        total_rows = sum(r.get("rows", 0) for r in results)
        print(f"  total written:      {total_size_mb/1024:.1f}GB, {total_rows:,} rows")
    if results and any(r.get("cols_missing_in_v51") for r in results):
        sample = next(r for r in results if r.get("cols_missing_in_v51"))
        print(f"\n  Note: {len(sample['cols_missing_in_v51'])} cols missing in v51 "
              f"(expected; xd_* will be recomputed in Phase 2):")
        for c in sample["cols_missing_in_v51"][:6]:
            print(f"    - {c}")

    print()
    if not args.dry_run and n_ok > 0:
        print("NEXT STEP — recompute xd_* cross-asset features via Phase 2:")
        print("  python src/pipeline/make_dataset_legacy.py --phase2-only --workers 1")
        print()
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
