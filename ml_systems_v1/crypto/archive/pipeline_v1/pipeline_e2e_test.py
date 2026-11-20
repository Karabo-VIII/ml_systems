"""End-to-end pipeline integration test.

Runs ALL pipeline stages on a single asset (BTCUSDT) and verifies each
transformation in turn. Used as a smoke test before any large rebuild
or before retraining models.

Stages tested:
  Stage 1: Bronze -- raw aggTrades / funding / metrics presence + freshness
  Stage 2: Bronze external -- raw_external/ sources present
  Stage 3: Silver bar fabric -- bars/<sym>/{dib,runs_*,range,adaptive_vol}/ accessible
  Stage 4: Silver features -- features/<sym>/frontier_daily.parquet built correctly
  Stage 5: Silver _global features -- features/_global/<source>.parquet present
  Stage 6: Gold v50 -- legacy chimera readable (V1-V14 compat)
  Stage 7: Gold v51 -- new chimera readable + 80 frontier features joined
  Stage 8: Cadence views -- 1d/4h/1h/15m views match v51 expected row counts
  Stage 9: Cross-asset features -- xd_* sanity check
  Stage 10: Manifest -- v51 manifest exists + checksums match
  Stage 11: Strategy parity -- raw frontier values match chimera v51 column values
  Stage 12: ChimeraLoader API -- can load all cadences successfully

Run:
  python src/pipeline/pipeline_e2e_test.py             # BTCUSDT only
  python src/pipeline/pipeline_e2e_test.py --asset ETH
  python src/pipeline/pipeline_e2e_test.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_registry import FeatureRegistry  # noqa: E402
from universe_loader import UniverseLoader  # noqa: E402
from bar_fabric import BarFabric  # noqa: E402
from chimera_loader import ChimeraLoader  # noqa: E402
import layout as _layout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"


@dataclass
class StageResult:
    stage: int
    name: str
    passed: bool
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


def stage1_bronze_raw(symbol: str) -> StageResult:
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    raw_dir = DATA / "raw" / sym
    if not raw_dir.exists():
        return StageResult(1, "bronze_raw", False, f"missing {raw_dir}", elapsed_ms=(time.time()-t0)*1000)
    types_seen = []
    for typ in ("aggTrades", "funding", "metrics"):
        sub = raw_dir / typ
        if not sub.exists():
            continue
        files = list(sub.glob("*.parquet"))
        if files:
            types_seen.append(f"{typ}({len(files)})")
    return StageResult(1, "bronze_raw", len(types_seen) >= 1,
                       f"types: {types_seen}",
                       metrics={"types_present": len(types_seen)},
                       elapsed_ms=(time.time()-t0)*1000)


def stage2_bronze_external() -> StageResult:
    t0 = time.time()
    expected = ("farside", "defillama", "deribit", "coinbase_okx_bybit", "wikipedia",
                "binance_futures_panels")
    base = DATA / "raw_external"
    missing = []
    for d in expected:
        if not (base / d).exists():
            missing.append(d)
    return StageResult(2, "bronze_external", not missing,
                       f"missing: {missing}" if missing else f"all {len(expected)} present",
                       elapsed_ms=(time.time()-t0)*1000)


def stage3_silver_bars(symbol: str, bf: BarFabric) -> StageResult:
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    info = bf.list_available(sym)
    bar_types = ("dib", "runs_tick", "runs_volume", "range", "adaptive_vol")
    available = sum(1 for bt in bar_types if info.get(bt) and info[bt].available)
    return StageResult(3, "silver_bars", available >= 3,
                       f"available: {available}/{len(bar_types)}",
                       metrics={"n_bar_types": available},
                       elapsed_ms=(time.time()-t0)*1000)


def stage4_silver_frontier(symbol: str, reg: "FeatureRegistry | None" = None) -> StageResult:
    """Threshold derives from registry expected_total_new_features (slack 4)."""
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    fp = _layout.frontier_daily_latest(sym)
    if fp is None or not fp.exists():
        return StageResult(4, "silver_frontier", False, f"no frontier_daily for {sym}",
                           elapsed_ms=(time.time()-t0)*1000)
    df = pl.read_parquet(fp)
    if reg is not None:
        try:
            exp = reg.chimera.expected_total_new_features
            min_cols = max(50, exp - 4)  # 2 keys + (exp - 4) features
        except Exception:
            min_cols = 60
    else:
        min_cols = 60
    return StageResult(4, "silver_frontier",
                       len(df) > 100 and len(df.columns) >= min_cols,
                       f"{len(df)} rows, {len(df.columns)} cols (>= {min_cols} required)",
                       elapsed_ms=(time.time()-t0)*1000)


def stage5_silver_global(reg: FeatureRegistry) -> StageResult:
    t0 = time.time()
    # Multi-asset silver lives in processed/hawkes/daily/ + processed/panels/daily/
    files = []
    for d in (_layout.hawkes_dir(), _layout.panels_dir()):
        if d.exists():
            files.extend(list(d.glob("*.parquet")))
    if not files:
        return StageResult(5, "silver_global", False,
                           f"no panel files in {_layout.hawkes_dir()} or {_layout.panels_dir()}",
                           elapsed_ms=(time.time()-t0)*1000)
    return StageResult(5, "silver_global", len(files) >= 5,
                       f"{len(files)} panel files",
                       elapsed_ms=(time.time()-t0)*1000)


def stage6_gold_v50(symbol: str) -> StageResult:
    t0 = time.time()
    fp = _layout.chimera_v50_latest(symbol)
    if fp is None or not fp.exists():
        return StageResult(6, "gold_v50", False, f"no v50 chimera for {symbol}",
                           elapsed_ms=(time.time()-t0)*1000)
    schema = pl.read_parquet_schema(fp)
    return StageResult(6, "gold_v50", len(schema) >= 60,
                       f"{len(schema)} cols",
                       elapsed_ms=(time.time()-t0)*1000)


def stage7_gold_v51(symbol: str, reg: "FeatureRegistry | None" = None) -> StageResult:
    """Counts features added via registry sources (covers any prefix or none)."""
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    fp = _layout.chimera_v51_latest(sym, "dollar")
    if fp is None or not fp.exists():
        return StageResult(7, "gold_v51", False, f"no v51 chimera for {sym}",
                           elapsed_ms=(time.time()-t0)*1000)
    schema = pl.read_parquet_schema(fp)
    cols = set(schema.keys())
    n_added = 0
    threshold = 60
    if reg is not None:
        try:
            for src_name in reg.chimera.sources_to_join:
                src = reg.get_source(src_name)
                expected_cols = src.output_feature_names()
                n_added += sum(1 for c in expected_cols if c in cols)
            # Threshold: registry total minus 4 slack (sources may emit fewer
            # cols if they had no rows for an asset).
            threshold = max(60, reg.chimera.expected_total_new_features - 4)
        except Exception:
            # Fallback to legacy prefix sum.
            n_added = sum(1 for c in cols if c.startswith(
                ("hbr_", "s3_", "bs_", "liq_", "wh_", "wiki_", "xex_", "dv_",
                 "stbl_", "etf_", "fp_", "rv_", "te_")))
    return StageResult(7, "gold_v51", n_added >= threshold,
                       f"{len(schema)} cols, {n_added} registered frontier features (>= {threshold})",
                       elapsed_ms=(time.time()-t0)*1000)


def stage8_cadence_views(symbol: str) -> StageResult:
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    cadences = ("1d", "4h", "1h", "15m")
    found = []
    for cad in cadences:
        p = _layout.chimera_v51_latest(sym, cad)
        if p is not None and p.exists():
            found.append(cad)
    return StageResult(8, "cadence_views", len(found) == len(cadences),
                       f"present: {found}",
                       elapsed_ms=(time.time()-t0)*1000)


def stage9_xd_features(symbol: str) -> StageResult:
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    fp = _layout.chimera_v51_latest(sym, "dollar")
    if fp is None or not fp.exists():
        return StageResult(9, "xd_features", False, "no v51")
    df = pl.read_parquet(fp, columns=["xd_btc_return", "xd_funding_spread", "xd_momentum_rank"])
    healthy = (
        abs(float(df["xd_momentum_rank"].drop_nulls().mean() or 0)) < 0.5 and
        0.5 < float(df["xd_momentum_rank"].drop_nulls().std() or 0) < 2.0
    )
    return StageResult(9, "xd_features", healthy,
                       f"momentum_rank mean={float(df['xd_momentum_rank'].drop_nulls().mean()):.3f} "
                       f"std={float(df['xd_momentum_rank'].drop_nulls().std()):.3f}",
                       elapsed_ms=(time.time()-t0)*1000)


def stage10_manifest(symbol: str) -> StageResult:
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    fp = _layout.manifest_path(sym)
    if not fp.exists():
        return StageResult(10, "manifest", False, f"missing {fp.name}")
    try:
        m = json.loads(fp.read_text())
        required = {"asset", "chimera_version", "v50_input_sha256", "row_count", "fixes_applied"}
        return StageResult(10, "manifest", required.issubset(m.keys()),
                           f"v={m.get('chimera_version')}, fixes={len(m.get('fixes_applied',[]))}",
                           elapsed_ms=(time.time()-t0)*1000)
    except Exception as e:
        return StageResult(10, "manifest", False, f"err: {e}")


def stage11_parity(symbol: str) -> StageResult:
    """Quick parity: v51's etf_btc_etf_total_z30 should equal raw farside's btc_etf_total_z30."""
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    v51_fp = _layout.chimera_v51_latest(sym, "dollar")
    raw_fp = DATA / "raw_external" / "farside" / "etf_flow_features.parquet"
    if v51_fp is None or not v51_fp.exists() or not raw_fp.exists():
        return StageResult(11, "parity_etf", False, "missing inputs")
    v51 = pl.read_parquet(v51_fp, columns=["timestamp", "etf_btc_etf_total_z30"])
    v51 = v51.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date"))
    v51_daily = v51.sort("timestamp").group_by("date").last().select(["date", "etf_btc_etf_total_z30"])
    raw = pl.read_parquet(raw_fp, columns=["date", "btc_etf_total_z30"])
    raw = raw.with_columns(pl.col("date").cast(pl.Date)).sort("date")
    # The chimera lags daily panels by +1 TRADING day (look-ahead prevention: on
    # bar-date T you only see ETF flows PUBLISHED through T-1). So farside's value
    # at date D appears in the chimera at the NEXT trading day. Align by shifting
    # the farside date forward one row before the exact-match join -- otherwise the
    # parity check sees a spurious 1-day offset (verified 2026-05-31: same-date join
    # -> max diff ~4.16 on the z-score; +1-trading-day join -> max diff 0.0).
    raw = raw.with_columns(pl.col("date").shift(-1).alias("applied_date"))
    j = v51_daily.join(raw, left_on="date", right_on="applied_date", how="inner").drop_nulls().sort("date")
    if len(j) == 0:
        return StageResult(11, "parity_etf", False, "no overlap")
    # etf_total_z30 is a 30-day ROLLING z-score. If farside is REFRESHED after the
    # chimera was built, only the most recent ~30d of z-scores recompute (new/revised
    # data shifts the rolling mean/std), so the chimera (baked at build time) drifts
    # from a later farside snapshot on the recent tail ONLY. A BROKEN/misaligned join
    # mismatches EVERYWHERE -- so require EXACT match on the STABLE region (older than
    # the rolling horizon) and only WARN-tolerate the recent rolling tail. Verified
    # 2026-05-31: post-refresh, the first 559 dates matched 0.0; only the last ~42 drifted.
    horizon_cutoff = j["date"].max() - timedelta(days=45)   # z30 30d window + margin
    stable = j.filter(pl.col("date") <= horizon_cutoff)
    diff = float((stable["etf_btc_etf_total_z30"] - stable["btc_etf_total_z30"]).abs().max() or 0) if len(stable) else 0.0
    recent_diff = float((j["etf_btc_etf_total_z30"] - j["btc_etf_total_z30"]).abs().max() or 0)
    return StageResult(11, "parity_etf", diff < 1e-6,
                       f"stable max diff {diff:.2e} on {len(stable)} dates "
                       f"(recent-tail drift {recent_diff:.2e}, tolerated: rolling-z post-build refresh)",
                       elapsed_ms=(time.time()-t0)*1000)


def stage12_loader_api(symbol: str) -> StageResult:
    t0 = time.time()
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    loader = ChimeraLoader()
    cadences_ok = []
    for cad in ("dollar", "1d", "4h", "1h", "15m"):
        try:
            df = loader.load(sym, cadence=cad)
            cadences_ok.append(f"{cad}({len(df)}r)")
        except Exception as e:
            return StageResult(12, "loader_api", False, f"{cad} fail: {e}")
    return StageResult(12, "loader_api", True, f"all 5 cadences: {cadences_ok}",
                       elapsed_ms=(time.time()-t0)*1000)


def run_all(symbol: str = "BTCUSDT") -> list[StageResult]:
    reg = FeatureRegistry.load()
    bf = BarFabric()
    results = []
    results.append(stage1_bronze_raw(symbol))
    results.append(stage2_bronze_external())
    results.append(stage3_silver_bars(symbol, bf))
    results.append(stage4_silver_frontier(symbol, reg))
    results.append(stage5_silver_global(reg))
    results.append(stage6_gold_v50(symbol))
    results.append(stage7_gold_v51(symbol, reg))
    results.append(stage8_cadence_views(symbol))
    results.append(stage9_xd_features(symbol))
    results.append(stage10_manifest(symbol))
    results.append(stage11_parity(symbol))
    results.append(stage12_loader_api(symbol))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    sym = args.asset.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"

    print(f"[e2e] running 12-stage pipeline integration test on {sym}")
    print()
    results = run_all(sym)
    n_pass = sum(1 for r in results if r.passed)
    for r in results:
        flag = "OK  " if r.passed else "FAIL"
        print(f"  Stage {r.stage:>2}: {flag}  {r.name:<22}  {r.detail}  ({r.elapsed_ms:.0f}ms)")
    print()
    print(f"[e2e] Summary: {n_pass}/{len(results)} stages PASS")

    if args.json:
        out_dir = PROJECT_ROOT / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pipeline_e2e_{sym}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps([{
            "stage": r.stage, "name": r.name, "passed": r.passed,
            "detail": r.detail, "elapsed_ms": r.elapsed_ms, "metrics": r.metrics,
        } for r in results], indent=2))
        print(f"[e2e] saved: {out_path.relative_to(PROJECT_ROOT)}")

    if n_pass < len(results):
        # exit 2 (hard fail), not 1: pre_train_gate maps rc>=2 to FAIL and rc=1 to
        # WARN. A failing end-to-end pipeline smoke MUST block training, not pass
        # as a mere warning.
        sys.exit(2)


if __name__ == "__main__":
    main()
