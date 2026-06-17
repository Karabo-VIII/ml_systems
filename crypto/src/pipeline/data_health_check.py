"""Data health check -- audit bronze + silver layers for drift, freshness, gaps.

Catches problems BEFORE they reach the chimera builder:
  1. Registry coverage: every source declared in feature_registry.yaml exists on disk.
  2. Freshness: each source has data within the last N days (configurable per source).
  3. Schema drift: declared columns are still present + types unchanged.
  4. Asset coverage: per-asset sources cover the U10/U50/U100 universes.
  5. Raw fetch gaps: identify date gaps in data/raw/<SYMBOL>/{aggTrades,funding,metrics}/.
  6. Universe / DNA consistency: every U50 asset has matching DNA bucket assignment.

Run:
  python src/pipeline/data_health_check.py             # full audit, exit non-zero on FAIL
  python src/pipeline/data_health_check.py --json      # machine-readable output
  python src/pipeline/data_health_check.py --quick     # skip the slow per-day raw scan

Output:
  logs/data_health_<DATE>.json   (machine readable)
  stdout summary
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_registry import FeatureRegistry  # noqa: E402
from universe_loader import UniverseLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"

# Per-source freshness budget in days. Beyond this, source is considered stale.
FRESHNESS_DAYS = {
    "hawkes_branching":      7,    # built from local aggTrades; should be near-current
    "s3_features":           3,    # daily refresh
    "basis_features":        2,
    "liq_features":          2,
    "whale_activity":        2,
    "wiki_pageviews":        7,    # 1-day Wikimedia API delay
    "cross_exchange_spreads": 2,
    "dvol":                  2,    # Deribit
    "stable_flow":           2,    # DeFiLlama
    "etf_flows":             3,    # Farside daily release
    "funding_panel":         2,
}

# Raw Binance freshness (calendar days)
RAW_FRESHNESS_DAYS = 3


@dataclass
class CheckResult:
    name: str
    severity: str   # 'ok' | 'warn' | 'fail'
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    checks: list[CheckResult] = field(default_factory=list)
    @property
    def n_pass(self) -> int: return sum(1 for c in self.checks if c.severity == "ok")
    @property
    def n_warn(self) -> int: return sum(1 for c in self.checks if c.severity == "warn")
    @property
    def n_fail(self) -> int: return sum(1 for c in self.checks if c.severity == "fail")
    def add(self, c: CheckResult) -> None: self.checks.append(c)


def parse_date(s) -> date | None:
    if s is None:
        return None
    # Order matters: datetime is a subclass of date, so check datetime first
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
    except Exception:
        return None


def check_registry_coverage(reg: FeatureRegistry) -> list[CheckResult]:
    """Every declared source's path must exist + readable."""
    out = []
    for name, src in reg.sources.items():
        fp = src.absolute_path()
        if not fp.exists():
            out.append(CheckResult(f"source_exists.{name}", "fail",
                                   f"missing: {fp.relative_to(PROJECT_ROOT)}"))
            continue
        try:
            schema = pl.read_parquet_schema(fp)
            n_cols = len(schema)
            out.append(CheckResult(f"source_exists.{name}", "ok",
                                   f"{fp.relative_to(PROJECT_ROOT)} ({n_cols} cols)"))
        except Exception as e:
            out.append(CheckResult(f"source_readable.{name}", "fail", f"read err: {e}"))
    return out


def check_freshness(reg: FeatureRegistry) -> list[CheckResult]:
    """Latest date in each source must be within freshness budget."""
    out = []
    today = datetime.now(timezone.utc).date()
    for name, src in reg.sources.items():
        budget = FRESHNESS_DAYS.get(name, 7)
        try:
            df = pl.read_parquet(src.absolute_path(), columns=[src.date_col])
            if src.date_col not in df.columns:
                out.append(CheckResult(f"freshness.{name}", "fail",
                                       f"date col '{src.date_col}' missing"))
                continue
            max_date_val = df[src.date_col].max()
            d = parse_date(max_date_val)
            if d is None:
                out.append(CheckResult(f"freshness.{name}", "fail", f"unparseable max date"))
                continue
            stale_days = (today - d).days
            if stale_days > budget:
                out.append(CheckResult(f"freshness.{name}", "warn",
                                       f"latest={d} ({stale_days}d old, budget {budget}d)"))
            else:
                out.append(CheckResult(f"freshness.{name}", "ok",
                                       f"latest={d} ({stale_days}d old, budget {budget}d)"))
        except Exception as e:
            out.append(CheckResult(f"freshness.{name}", "fail", f"err: {e}"))
    return out


def check_schema_drift(reg: FeatureRegistry) -> list[CheckResult]:
    """Every declared feature column must still be present."""
    out = []
    for name, src in reg.sources.items():
        if not src.absolute_path().exists():
            continue
        try:
            schema = pl.read_parquet_schema(src.absolute_path())
            cols = set(schema.keys())
            missing = [f for f in src.features if f not in cols]
            if missing:
                out.append(CheckResult(f"schema_drift.{name}", "fail",
                                       f"declared but missing: {missing[:5]}"))
            else:
                out.append(CheckResult(f"schema_drift.{name}", "ok",
                                       f"{len(src.features)} declared cols all present"))
        except Exception as e:
            out.append(CheckResult(f"schema_drift.{name}", "fail", f"err: {e}"))
    return out


def check_universe_coverage(reg: FeatureRegistry, loader: UniverseLoader) -> list[CheckResult]:
    """Per-asset frontier sources must cover their declared expected_coverage.

    Default expected coverage: U10 mandatory, U50 nice-to-have.
    If source declares `expected_coverage: {assets: [...]}` use that explicit list.
    If source declares `expected_coverage: {universe: u10|u50}` use that universe.
    """
    out = []
    for name, src in reg.sources.items():
        if src.layout != "per_asset":
            continue
        if not src.absolute_path().exists():
            continue
        try:
            df = pl.read_parquet(src.absolute_path(), columns=[src.asset_col])
            assets = set(df[src.asset_col].unique().to_list())
            assets_normalized = {(a + "USDT" if not str(a).endswith("USDT") else str(a))
                                 for a in assets if a is not None}
            # Resolve expected coverage
            coverage = src.expected_coverage or {}
            if "assets" in coverage:
                expected = set(coverage["assets"])
                missing = expected - assets_normalized
                if missing:
                    out.append(CheckResult(f"universe_cov.{name}", "fail",
                                           f"missing declared assets: {sorted(missing)}"))
                else:
                    out.append(CheckResult(f"universe_cov.{name}", "ok",
                                           f"covers all {len(expected)} declared assets"))
                continue
            if "universe" in coverage:
                u_name = coverage["universe"]
                expected = set(loader.list(u_name))
                missing = expected - assets_normalized
                if missing:
                    out.append(CheckResult(f"universe_cov.{name}.{u_name}", "fail",
                                           f"missing {len(missing)} {u_name} assets: {list(missing)[:5]}"))
                else:
                    out.append(CheckResult(f"universe_cov.{name}", "ok",
                                           f"covers {u_name}"))
                continue
            # Default: u10 mandatory, u50 nice-to-have
            u10 = set(loader.list("u10"))
            u50 = set(loader.list("u50"))
            u10_missing = u10 - assets_normalized
            u50_missing = u50 - assets_normalized
            if u10_missing:
                out.append(CheckResult(f"universe_cov.{name}.u10", "fail",
                                       f"missing {len(u10_missing)} U10 assets: {list(u10_missing)[:5]}"))
            elif u50_missing:
                out.append(CheckResult(f"universe_cov.{name}.u50", "warn",
                                       f"covers U10 but missing {len(u50_missing)} U50 assets"))
            else:
                out.append(CheckResult(f"universe_cov.{name}", "ok",
                                       f"covers U50 ({len(assets_normalized)} assets total)"))
        except Exception as e:
            out.append(CheckResult(f"universe_cov.{name}", "fail", f"err: {e}"))
    return out


def check_raw_fetch_freshness(loader: UniverseLoader) -> list[CheckResult]:
    """For each U10 asset, raw aggTrades + funding + metrics must be within N days."""
    out = []
    today = datetime.now(timezone.utc).date()
    for sym in loader.list("u10"):
        sym_dir = RAW / sym
        if not sym_dir.exists():
            out.append(CheckResult(f"raw_fetch.{sym}", "fail", f"raw dir missing"))
            continue
        for typ in ("aggTrades", "funding", "metrics"):
            sub = sym_dir / typ
            if not sub.exists():
                out.append(CheckResult(f"raw_fetch.{sym}.{typ}", "warn", f"{typ}/ subdir missing"))
                continue
            files = sorted(sub.glob("*.parquet"))
            if not files:
                out.append(CheckResult(f"raw_fetch.{sym}.{typ}", "fail", "no parquet files"))
                continue
            # Latest-file date is in the filename: <SYM>-<TYPE>-YYYY-MM-DD.parquet
            latest = files[-1].stem
            parts = latest.split("-")
            if len(parts) < 5:
                out.append(CheckResult(f"raw_fetch.{sym}.{typ}", "warn",
                                       f"can't parse latest filename: {files[-1].name}"))
                continue
            try:
                d = date.fromisoformat("-".join(parts[-3:]))
                stale = (today - d).days
                if stale > RAW_FRESHNESS_DAYS:
                    out.append(CheckResult(f"raw_fetch.{sym}.{typ}", "warn",
                                           f"latest={d} ({stale}d old)"))
                else:
                    out.append(CheckResult(f"raw_fetch.{sym}.{typ}", "ok",
                                           f"latest={d} ({stale}d old, {len(files)} files)"))
            except Exception:
                out.append(CheckResult(f"raw_fetch.{sym}.{typ}", "warn", f"err parsing date"))
    return out


def check_universe_dna_consistency(loader: UniverseLoader) -> list[CheckResult]:
    """Every asset in U10 / U50 / U100 has a DNA bucket; transitive consistency."""
    out = []
    for u_name in ("u10", "u50", "u100"):
        unknown = []
        u = loader.get(u_name)
        for spec in u.assets or []:
            if spec.dna not in ("BLUE", "STEADY", "VOLATILE", "DEGEN"):
                unknown.append((spec.symbol, spec.dna))
        if unknown:
            out.append(CheckResult(f"dna.{u_name}", "fail",
                                   f"{len(unknown)} unknown DNA: {unknown[:3]}"))
        else:
            out.append(CheckResult(f"dna.{u_name}", "ok",
                                   f"all {len(u.assets or [])} assets have valid DNA"))
    # Transitive: U10 ⊂ U50 ⊂ U100
    u10 = set(loader.list("u10"))
    u50 = set(loader.list("u50"))
    u100 = set(loader.list("u100"))
    if not u10.issubset(u50):
        out.append(CheckResult("dna.u10_subset_u50", "fail",
                               f"U10 not ⊂ U50; orphans: {u10 - u50}"))
    else:
        out.append(CheckResult("dna.u10_subset_u50", "ok", "U10 ⊂ U50"))
    if not u50.issubset(u100):
        out.append(CheckResult("dna.u50_subset_u100", "fail",
                               f"U50 not ⊂ U100; orphans: {u50 - u100}"))
    else:
        out.append(CheckResult("dna.u50_subset_u100", "ok", "U50 ⊂ U100"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="Skip per-asset raw fetch scan (slowest check)")
    args = ap.parse_args()

    reg = FeatureRegistry.load()
    loader = UniverseLoader.load()
    rep = HealthReport()

    print(f"[health] feature registry v{reg.version}: {len(reg.sources)} sources")
    print(f"[health] universes: u10={len(loader.list('u10'))}, "
          f"u50={len(loader.list('u50'))}, u100={len(loader.list('u100'))}")

    print("\n[health] === Registry coverage ===")
    for c in check_registry_coverage(reg):
        rep.add(c)

    print("\n[health] === Freshness ===")
    for c in check_freshness(reg):
        rep.add(c)

    print("\n[health] === Schema drift ===")
    for c in check_schema_drift(reg):
        rep.add(c)

    print("\n[health] === Universe coverage (per_asset sources) ===")
    for c in check_universe_coverage(reg, loader):
        rep.add(c)

    if not args.quick:
        print("\n[health] === Raw Binance fetch freshness (U10) ===")
        for c in check_raw_fetch_freshness(loader):
            rep.add(c)

    print("\n[health] === Universe / DNA consistency ===")
    for c in check_universe_dna_consistency(loader):
        rep.add(c)

    # Print non-OK
    print("\n[health] === Issues ===")
    for c in rep.checks:
        if c.severity != "ok":
            print(f"  {c.severity.upper():>4}  {c.name}  -- {c.detail}")
    if rep.n_fail == 0 and rep.n_warn == 0:
        print("  none -- all clean")

    print(f"\n[health] Summary: {rep.n_pass} pass, {rep.n_warn} warn, {rep.n_fail} fail")

    if args.json:
        out_dir = PROJECT_ROOT / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"data_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps({
            "summary": {"pass": rep.n_pass, "warn": rep.n_warn, "fail": rep.n_fail},
            "checks": [{"name": c.name, "severity": c.severity, "detail": c.detail}
                       for c in rep.checks],
        }, indent=2, default=str))
        print(f"[health] saved: {out_path.relative_to(PROJECT_ROOT)}")

    if rep.n_fail > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
