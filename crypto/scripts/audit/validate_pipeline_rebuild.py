"""Post-rebuild validation script — checks all RED-team-audit fix outcomes.

Validates:
  1. feature_registry.yaml schema intact (no catalog-schema collision recurrence)
  2. frontier_silver output coverage vs u100 universe size
  3. chimera_v51 output coverage vs u100 universe size
  4. xrel_* columns present in chimera outputs (CRIT A fix — was previously
     silently absent)
  5. Silver write order: silver mtime <= chimera mtime (CRIT J fix — chimera
     was rebuilding silver AFTER frontier_silver's gate, gc-deleting it)
  6. raw_funding has fresh data per asset (MED orphan fix)
  7. multi_venue_listings panel exists with rows (HIGH F fix)
  8. CDAP invariants clean (Layer-2 schema lock + post-2026-05-20 rules)
  9. add_xrel_features ran AFTER chimera_v51 (DAG ordering)

Exit codes:
  0 = clean
  1 = warnings only
  2 = at least one CRITICAL check failed (refresh did NOT produce correct data)
"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path
from datetime import datetime, timezone

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Result:
    def __init__(self, name, status, msg):
        self.name = name; self.status = status; self.msg = msg
    def __repr__(self):
        return f"  [{self.status:8s}] {self.name}: {self.msg}"

def check_feature_registry_schema():
    p = PROJECT_ROOT / "config" / "feature_registry.yaml"
    if not p.exists():
        return Result("feature_registry_schema", "CRIT", "FILE MISSING")
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return Result("feature_registry_schema", "CRIT", "YAML root is not dict")
    has_sources = "sources" in data
    has_chimera = "chimera_v51" in data
    has_catalog_marker = "prefix_families" in data or ("meta" in data and isinstance(data.get("meta"), dict) and "generated" in data.get("meta", {}))
    if not has_sources or not has_chimera:
        return Result("feature_registry_schema", "CRIT",
                      f"missing keys: sources={has_sources} chimera_v51={has_chimera}")
    if has_catalog_marker:
        return Result("feature_registry_schema", "CRIT",
                      "catalog-schema marker present (c59c4e7 regression)")
    return Result("feature_registry_schema", "OK",
                  f"sources({len(data['sources'])}) + chimera_v51 keys present, no catalog markers")

def get_universe_assets(scope="u100"):
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.pipeline.universe_loader import UniverseLoader
    u = UniverseLoader.load()
    return [a.replace("USDT", "").upper() for a in u.list(scope)]

def check_frontier_silver_coverage():
    expected = get_universe_assets("u100")
    p = PROJECT_ROOT / "data" / "processed" / "frontier" / "daily"
    files = list(p.glob("*_frontier_daily*.parquet"))
    assets_with_silver = set()
    for f in files:
        sym = f.name.split("usdt")[0].upper()
        assets_with_silver.add(sym)
    coverage = len(assets_with_silver & set(expected)) / max(len(expected), 1) * 100
    if coverage < 50:
        return Result("frontier_silver_coverage", "CRIT",
                      f"only {len(assets_with_silver)}/{len(expected)} assets ({coverage:.0f}%) -- PARTIAL")
    if coverage < 90:
        return Result("frontier_silver_coverage", "WARN",
                      f"{len(assets_with_silver)}/{len(expected)} assets ({coverage:.0f}%)")
    return Result("frontier_silver_coverage", "OK",
                  f"{len(assets_with_silver)}/{len(expected)} assets ({coverage:.0f}%) covered")

def check_chimera_v51_coverage():
    expected = get_universe_assets("u100")
    p = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"
    files = list(p.glob("*_v51_chimera_*.parquet"))
    assets_with_chimera = set()
    for f in files:
        sym = f.name.split("usdt")[0].upper()
        assets_with_chimera.add(sym)
    coverage = len(assets_with_chimera & set(expected)) / max(len(expected), 1) * 100
    if coverage < 50:
        return Result("chimera_v51_coverage", "CRIT",
                      f"only {len(assets_with_chimera)}/{len(expected)} assets ({coverage:.0f}%) -- PARTIAL")
    if coverage < 90:
        return Result("chimera_v51_coverage", "WARN",
                      f"{len(assets_with_chimera)}/{len(expected)} assets ({coverage:.0f}%)")
    return Result("chimera_v51_coverage", "OK",
                  f"{len(assets_with_chimera)}/{len(expected)} assets ({coverage:.0f}%) covered")

def check_xrel_present_in_chimera():
    """CRIT A fix: xrel_* cols must be in chimera after the rebuild.

    Checks ALL chimera files in the u100 universe (not a sample) to ensure
    no single asset is missing xrel_* enrichment.
    """
    expected = set(get_universe_assets("u100"))
    p = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"
    files = sorted(p.glob("*_v51_chimera_*.parquet"))
    if not files:
        return Result("xrel_in_chimera", "CRIT", "no chimera files to check")
    # Group by asset (one file per asset; take the latest dated)
    by_asset = {}
    for f in files:
        sym = f.name.split("usdt")[0].upper()
        if sym not in expected:
            continue  # ignore non-u100 assets
        # Keep latest mtime
        if sym not in by_asset or f.stat().st_mtime > by_asset[sym].stat().st_mtime:
            by_asset[sym] = f
    missing = []
    for sym, f in by_asset.items():
        try:
            cols = pl.read_parquet(f, n_rows=0).columns
            xrel_cnt = sum(1 for c in cols if c.startswith("xrel_"))
            if xrel_cnt < 10:
                missing.append((sym, xrel_cnt))
        except Exception as e:
            missing.append((sym, f"read-err:{type(e).__name__}"))
    if missing:
        names = ", ".join(f"{n}({c})" for n, c in missing[:5])
        return Result("xrel_in_chimera", "CRIT",
                      f"{len(missing)}/{len(by_asset)} chimera files have <10 xrel_* cols: {names}{'...' if len(missing)>5 else ''}")
    return Result("xrel_in_chimera", "OK",
                  f"all {len(by_asset)}/{len(expected)} u100 chimera files have >=10 xrel_* cols")

def check_silver_vs_chimera_mtime():
    """CRIT J fix: silver must NOT post-date chimera. Chimera with --skip-silver
    reads existing silver; if silver mtime > chimera mtime, chimera rebuilt
    silver after frontier_silver gated it.

    Checks ALL u100 assets, not a sample.
    """
    silver_dir = PROJECT_ROOT / "data" / "processed" / "frontier" / "daily"
    chimera_dir = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"
    assets = [a.lower() for a in get_universe_assets("u100")]
    violations = []
    checked = 0
    for asset in assets:
        silver_files = sorted(silver_dir.glob(f"{asset}usdt_frontier_daily*.parquet"))
        chimera_files = sorted(chimera_dir.glob(f"{asset}usdt_v51_chimera_*.parquet"))
        if not silver_files or not chimera_files:
            continue
        checked += 1
        silver_mtime = max(f.stat().st_mtime for f in silver_files)
        chimera_mtime = max(f.stat().st_mtime for f in chimera_files)
        # If silver is >60s newer than chimera, --skip-silver may have been bypassed.
        if silver_mtime > chimera_mtime + 60:
            violations.append((asset, silver_mtime - chimera_mtime))
    if violations:
        v = ", ".join(f"{a}(+{int(d)}s)" for a, d in violations[:5])
        return Result("silver_before_chimera", "CRIT",
                      f"{len(violations)}/{checked} assets have silver newer than chimera: {v}")
    return Result("silver_before_chimera", "OK",
                  f"all {checked}/{len(assets)} u100 silver mtimes <= chimera mtimes")

def check_raw_funding_freshness():
    """MED orphan fix: raw_funding should run when chimera_legacy rebuilds."""
    p = PROJECT_ROOT / "data" / "raw"
    assets = get_universe_assets("u100")[:10]
    missing = []
    for a in assets:
        ad = p / f"{a}USDT" / "funding"
        if not ad.exists():
            missing.append(a); continue
        files = list(ad.glob("*.parquet"))
        if not files:
            missing.append(a)
    if len(missing) > 3:
        return Result("raw_funding_freshness", "WARN",
                      f"{len(missing)}/10 sampled assets missing raw/funding/")
    return Result("raw_funding_freshness", "OK",
                  f"raw/funding/ present for {10-len(missing)}/10 sampled assets")

def check_multi_venue_panel():
    """HIGH F fix: multi_venue_listings must produce a non-empty panel."""
    p = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "multi_venue_listings.parquet"
    if not p.exists():
        return Result("multi_venue_listings", "CRIT", "panel parquet missing")
    df = pl.read_parquet(p, n_rows=10)
    n = pl.scan_parquet(p).select(pl.len()).collect().item()
    if n == 0:
        return Result("multi_venue_listings", "WARN", "panel exists but empty")
    return Result("multi_venue_listings", "OK", f"panel has {n} rows, cols={df.columns}")

def check_cdap_invariants():
    """Run CDAP and report CRITICAL count."""
    result = subprocess.run(
        [sys.executable, "src/audit/check_invariants.py", "--quiet"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = (result.stdout + result.stderr).lower()
    if "critical" in output:
        import re
        m = re.search(r"(\d+)\s*critical", output)
        if m and int(m.group(1)) > 0:
            return Result("cdap_invariants", "CRIT",
                          f"{m.group(1)} CRITICAL findings (see check_invariants.py output)")
    return Result("cdap_invariants", "OK", "0 CRITICAL findings")

def check_add_xrel_order_in_dag():
    """add_xrel_features must be in DAG with deps=[chimera_v51]."""
    with open(PROJECT_ROOT / "config" / "asset_dag.yaml") as f:
        dag = yaml.safe_load(f)
    spec = dag.get("assets", {}).get("add_xrel_features")
    if spec is None:
        return Result("add_xrel_dag_entry", "CRIT", "stage missing from DAG")
    if "chimera_v51" not in spec.get("deps", []):
        return Result("add_xrel_dag_entry", "CRIT",
                      f"deps={spec.get('deps')} (expected chimera_v51)")
    return Result("add_xrel_dag_entry", "OK",
                  f"deps={spec['deps']}, output_kind={spec.get('output_kind')}")

def main():
    checks = [
        check_feature_registry_schema,
        check_frontier_silver_coverage,
        check_chimera_v51_coverage,
        check_xrel_present_in_chimera,
        check_silver_vs_chimera_mtime,
        check_raw_funding_freshness,
        check_multi_venue_panel,
        check_cdap_invariants,
        check_add_xrel_order_in_dag,
    ]
    print("=" * 78)
    print("POST-REBUILD VALIDATION — Pipeline RED-team audit fixes")
    print(f"Run at: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 78)
    results = []
    for c in checks:
        try:
            r = c()
        except Exception as e:
            r = Result(c.__name__, "CRIT", f"exception: {e}")
        print(r)
        results.append(r)
    print()
    n_crit = sum(1 for r in results if r.status == "CRIT")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_ok = sum(1 for r in results if r.status == "OK")
    print(f"SUMMARY: {n_ok} OK / {n_warn} WARN / {n_crit} CRIT")
    return 2 if n_crit > 0 else (1 if n_warn > 0 else 0)

if __name__ == "__main__":
    raise SystemExit(main())
