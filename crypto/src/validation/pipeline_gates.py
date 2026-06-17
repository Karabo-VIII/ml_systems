"""Pipeline-layer validation gates.

Single source for "is this data product ready for downstream consumption?"
Designed to be called from THREE places:
  1) The producer script itself, post-write (fail-fast).
  2) The DAG runner (refresh.py) after a stage rebuild.
  3) The consumer (model training, strategy backtest) before reading.

Each gate returns a (ok, message, severity) tuple. Severity:
  'critical'  hard fail; consumer must not proceed
  'warn'      degraded but usable; emit warning
  'info'      informational; not blocking

The gates are intentionally cheap to call (microseconds-to-seconds) so
they can run inline at every layer boundary without slowing the pipeline.

CLI:
  python src/validation/pipeline_gates.py --asset BTC --gate v51_chimera
  python src/validation/pipeline_gates.py --gate frontier_silver --asset BTC
  python src/validation/pipeline_gates.py --gate-suite default --asset BTC
"""
from __future__ import annotations

# CDAP contract.
__contract__ = {
    "kind": "validation_module",
    "stage": "pipeline_gates",
    "outputs": {"format": "(ok: bool, message: str, severity: str)"},
}

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    name: str
    ok: bool
    message: str
    severity: str = "info"

    def render(self) -> str:
        icon = "OK  " if self.ok else ("WARN" if self.severity == "warn" else "FAIL")
        return f"  [{icon}] {self.name}: {self.message}"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_schema(p: Path):
    import polars as pl
    return set(pl.read_parquet_schema(p).keys())


def _latest_chimera_v51(asset: str) -> Optional[Path]:
    import layout as _layout
    sym = asset.upper() if asset.upper().endswith("USDT") else asset.upper() + "USDT"
    return _layout.chimera_v51_latest(sym, "dollar")


def _latest_frontier_silver(asset: str) -> Optional[Path]:
    import layout as _layout
    sym = asset.upper() if asset.upper().endswith("USDT") else asset.upper() + "USDT"
    return _layout.frontier_daily_latest(sym)


def _latest_chimera_legacy(asset: str) -> Optional[Path]:
    import layout as _layout
    sym = asset.upper() if asset.upper().endswith("USDT") else asset.upper() + "USDT"
    return _layout.chimera_v50_latest(sym)


# ─── Individual gates ────────────────────────────────────────────────────────

def gate_chimera_legacy_xd(asset: str) -> GateResult:
    """Legacy v50 chimera must carry the xd_* cross-asset features."""
    p = _latest_chimera_legacy(asset)
    if p is None or not p.exists():
        return GateResult("chimera_legacy_xd", False,
                           f"no chimera_legacy for {asset}", "critical")
    cols = _read_schema(p)
    required = {"xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
                "xd_btc_volatility", "xd_cross_return_mean"}
    missing = required - cols
    if missing:
        return GateResult("chimera_legacy_xd", False,
                           f"missing xd_* cols: {sorted(missing)}",
                           "critical")
    return GateResult("chimera_legacy_xd", True,
                       f"all 5 required xd_* present in {p.name}")


def gate_v51_schema(asset: str) -> GateResult:
    """v51 chimera must carry the registry-declared feature set."""
    p = _latest_chimera_v51(asset)
    if p is None or not p.exists():
        return GateResult("v51_schema", False,
                           f"no v51 chimera for {asset}", "critical")
    cols = _read_schema(p)
    required = {
        "timestamp", "bar_id", "close",
        "target_return_1", "target_return_4", "target_return_16", "target_return_64",
        "norm_flow_imbalance", "norm_hawkes_imbalance",
        "xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
        # New: te_*, rv_*, mv_*, lob_*
        "te_in_btc", "te_imb",
    }
    missing = required - cols
    if missing:
        return GateResult("v51_schema", False,
                           f"missing required cols: {sorted(missing)}",
                           "critical")
    return GateResult("v51_schema", True,
                       f"all required cols present ({len(cols)} total)")


def gate_v51_nan_budget(asset: str, max_high_nan_features: int = 50,
                         high_nan_threshold: float = 0.30) -> GateResult:
    """Count features with > threshold NaN. Blocks training if too many.

    Default: warn if >50 features have >30% NaN. Critical if >80.
    """
    import polars as pl
    p = _latest_chimera_v51(asset)
    if p is None or not p.exists():
        return GateResult("v51_nan_budget", False,
                           f"no v51 chimera for {asset}", "critical")
    df = pl.read_parquet(p)
    n = df.height
    high_nan = []
    for c in df.columns:
        if df[c].dtype.is_numeric():
            null_pct = df[c].null_count() / max(n, 1)
            if null_pct > high_nan_threshold:
                high_nan.append((c, null_pct))
    n_high = len(high_nan)
    msg = f"{n_high} features > {high_nan_threshold:.0%} NaN"
    if n_high > 80:
        return GateResult("v51_nan_budget", False, msg, "critical")
    if n_high > max_high_nan_features:
        return GateResult("v51_nan_budget", False, msg + " (limit "
                           f"{max_high_nan_features})", "warn")
    return GateResult("v51_nan_budget", True, msg)


def gate_v51_required_features_not_all_nan(asset: str,
                                             features: Optional[list] = None) -> GateResult:
    """Critical-features must NOT be 100% NaN. Catches the te_panel-mismatch
    class of bugs (where a feature column is present but every row null).
    """
    import polars as pl
    p = _latest_chimera_v51(asset)
    if p is None or not p.exists():
        return GateResult("v51_critical_not_null", False,
                           f"no v51 chimera for {asset}", "critical")
    if features is None:
        features = ["xd_btc_return", "xd_momentum_rank",
                     "te_in_btc", "te_imb",
                     "norm_flow_imbalance", "norm_hawkes_imbalance"]
    df = pl.read_parquet(p, columns=[c for c in features
                                        if c in _read_schema(p)])
    fully_null = []
    for c in df.columns:
        if df[c].null_count() == df.height:
            fully_null.append(c)
    if fully_null:
        return GateResult("v51_critical_not_null", False,
                           f"features 100% NaN: {fully_null}",
                           "critical")
    return GateResult("v51_critical_not_null", True,
                       f"{len(df.columns)} critical features carry data")


def gate_frontier_silver_minimum(asset: str, min_features: int = 60) -> GateResult:
    """Frontier silver must have at least N feature columns."""
    p = _latest_frontier_silver(asset)
    if p is None or not p.exists():
        return GateResult("frontier_silver_min", False,
                           f"no silver for {asset}", "critical")
    cols = _read_schema(p)
    n_features = len([c for c in cols if c not in {"date", "asset"}])
    if n_features < min_features:
        return GateResult("frontier_silver_min", False,
                           f"only {n_features} features, expected >= {min_features}",
                           "warn")
    return GateResult("frontier_silver_min", True,
                       f"{n_features} features (>= {min_features})")


# ─── Suites ──────────────────────────────────────────────────────────────────

def suite_default(asset: str) -> list[GateResult]:
    """Standard pre-train gate suite for one asset."""
    return [
        gate_chimera_legacy_xd(asset),
        gate_frontier_silver_minimum(asset),
        gate_v51_schema(asset),
        gate_v51_required_features_not_all_nan(asset),
        gate_v51_nan_budget(asset),
    ]


def suite_pipeline_only(asset: str) -> list[GateResult]:
    """Faster suite: just the schema-existence checks. No NaN budget scan."""
    return [
        gate_chimera_legacy_xd(asset),
        gate_frontier_silver_minimum(asset),
        gate_v51_schema(asset),
        gate_v51_required_features_not_all_nan(asset),
    ]


SUITES = {"default": suite_default, "pipeline_only": suite_pipeline_only}


# ─── Public entry: callable from refresh.py + consumers ──────────────────────

def run_suite(suite_name: str, asset: str) -> tuple[bool, list[GateResult]]:
    """Run a named suite. Returns (overall_ok, results).

    overall_ok = no result has severity='critical' AND ok=False.
    """
    if suite_name not in SUITES:
        raise ValueError(f"unknown suite {suite_name!r}; known: {list(SUITES)}")
    results = SUITES[suite_name](asset)
    has_critical_fail = any(
        (not r.ok) and r.severity == "critical" for r in results
    )
    return (not has_critical_fail), results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", required=True, help="Asset symbol (e.g. BTC)")
    ap.add_argument("--gate", default=None,
                    help="Single gate name: chimera_legacy_xd, v51_schema, "
                         "v51_nan_budget, v51_critical_not_null, frontier_silver_min")
    ap.add_argument("--gate-suite", default=None, choices=list(SUITES),
                    help=f"Suite: {list(SUITES)}")
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()

    if args.gate:
        fn_map = {
            "chimera_legacy_xd": gate_chimera_legacy_xd,
            "v51_schema": gate_v51_schema,
            "v51_nan_budget": gate_v51_nan_budget,
            "v51_critical_not_null": gate_v51_required_features_not_all_nan,
            "frontier_silver_min": gate_frontier_silver_minimum,
        }
        if args.gate not in fn_map:
            print(f"unknown gate {args.gate!r}; known: {list(fn_map)}", file=sys.stderr)
            return 2
        results = [fn_map[args.gate](args.asset)]
        ok = all(r.ok or r.severity != "critical" for r in results)
    else:
        suite_name = args.gate_suite or "default"
        ok, results = run_suite(suite_name, args.asset)

    if args.json:
        import json
        print(json.dumps([{"name": r.name, "ok": r.ok, "msg": r.message,
                            "sev": r.severity} for r in results], indent=2))
    else:
        print(f"\nGATE RESULTS  asset={args.asset}  "
              f"({sum(1 for r in results if r.ok)}/{len(results)} ok)")
        for r in results:
            print(r.render())
        print()
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
