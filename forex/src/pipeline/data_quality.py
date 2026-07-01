"""FX data-quality gate + pipeline metrics (the 'validating data + metrics' layer).

Computes per-(pair, cadence) metrics on the silver bars and applies FX-specific
quality gates. The checks encode what a crypto pipeline does NOT need:

  - WEEKEND-GAP AWARENESS: the Fri-close -> Sun-open silence is a STRUCTURAL
    feature of the 24/5 calendar, not missing data. Gaps are classified
    weekend-expected vs intra-session-anomalous.
  - SPREAD SANITY: ask < bid (negative spread) must never happen; spread is
    reported in pips (per-pair pip size, JPY-aware).
  - OHLC INTEGRITY: high >= max(open,close), low <= min(open,close), high >= low.
  - PRICE FINITENESS + POSITIVITY, null rates, zero-tick bars.

Exit code = worst status across all (pair, cadence): 0 clean, 1 warn, 2 fail.
A machine-readable metrics JSON is written under runs/pipeline_metrics/.

    python src/pipeline/data_quality.py --universe u7 --cadences 1m 1h 1d
    python src/pipeline/data_quality.py --pairs EURUSD --report   # metrics only, exit 0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_validator",
    "stage": "data_quality",
    "inputs": {"upstream": "data/processed/bars/<cadence>/<pair>_<cadence>_*.parquet",
               "args": ["--pairs", "--universe", "--cadences", "--report"]},
    "outputs": {"metrics_json": "runs/pipeline_metrics/<ts>.json",
                "exit_code": "0 clean / 1 warn / 2 fail (worst across pairs)"},
    "invariants": {"weekend_gap_not_flagged_as_missing": True, "negative_spread_is_fail": True,
                   "ohlc_integrity_checked": True, "pips_are_pair_aware": True},
    "rationale": "The FX validation + pipeline-metrics gate before any modelling.",
}

CADENCE_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
              "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# Broad price-plausibility bands by pair class (catches a wrong price_divisor).
def _price_band(pair: str, dna: str) -> tuple[float, float]:
    p = pair.upper()
    if p.startswith("XAU"):      # gold
        return (400.0, 6000.0)
    if p.startswith("XAG"):      # silver
        return (1.0, 200.0)
    if p.startswith("XPT") or p.startswith("XPD"):
        return (100.0, 5000.0)
    if p.endswith("JPY"):
        return (20.0, 400.0)
    if dna == "EXOTIC":
        return (0.001, 100.0)    # exotics span a wide range (USDHKD~7.8, USDTRY~40)
    return (0.3, 3.0)            # majors / non-JPY crosses


def _is_weekend_gap(t_start_ms: int, t_end_ms: int) -> bool:
    """A gap that spans the weekend close (Fri ~21:00 UTC -> Sun ~21:00 UTC) is
    expected. Heuristic: gap > 20h and it starts on Friday or Saturday (UTC)."""
    gap_h = (t_end_ms - t_start_ms) / 3_600_000
    start = datetime.fromtimestamp(t_start_ms / 1000, tz=timezone.utc)
    return gap_h > 20.0 and start.weekday() in (4, 5)   # Fri / Sat


def bar_metrics(pair: str, cadence: str, U: UniverseLoader) -> dict:
    """Load the latest bar file for (pair, cadence) and compute metrics + status."""
    path = layout.bars_latest(pair, cadence)
    m: dict = {"pair": pair, "cadence": cadence, "status": "pass", "issues": []}
    if path is None or not path.exists():
        m["status"] = "fail"
        m["issues"].append("no_bar_file")
        m["n_bars"] = 0
        return m
    df = pl.read_parquet(path)
    n = len(df)
    m["file"] = path.name
    m["n_bars"] = n
    if n == 0:
        m["status"] = "fail"
        m["issues"].append("empty_bar_file")
        return m

    ts = df["timestamp"].to_list()
    m["first"] = datetime.fromtimestamp(ts[0] / 1000, tz=timezone.utc).isoformat()
    m["last"] = datetime.fromtimestamp(ts[-1] / 1000, tz=timezone.utc).isoformat()
    m["span_days"] = round((ts[-1] - ts[0]) / 86_400_000, 2)

    pip = U.pip(pair)
    dna = U.dna_for(pair)

    # --- price finiteness / positivity / plausibility ---
    close = df["close"]
    n_nonfinite = int((~close.is_finite()).sum()) + int(close.is_null().sum())
    n_nonpos = int((close <= 0).sum())
    lo, hi = _price_band(pair, dna)
    med = float(close.median())
    m["price_median"] = round(med, 6)
    m["price_min"] = round(float(close.min()), 6)
    m["price_max"] = round(float(close.max()), 6)
    price_out_of_band = not (lo <= med <= hi)

    # --- OHLC integrity ---
    integ_viol = int(df.select(
        ((pl.col("high") < pl.col("low")) |
         (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close")) |
         (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close")))
        .sum().alias("v"))["v"][0])
    m["ohlc_integrity_violations"] = integ_viol

    # --- spread sanity (bid/ask cols present on tick-built bars) ---
    if "bid_close" in df.columns and "ask_close" in df.columns:
        neg_spread = int((df["ask_close"] < df["bid_close"]).sum())
        m["negative_spread_bars"] = neg_spread
        if "spread_mean" in df.columns:
            m["spread_mean_pips"] = round(float(df["spread_mean"].mean()) / pip, 3)
            m["spread_p99_pips"] = round(float(df["spread_max"].quantile(0.99)) / pip, 3)
    else:
        neg_spread = 0

    # --- zero-tick + null rates ---
    if "tick_count" in df.columns:
        m["zero_tick_bars"] = int((df["tick_count"] == 0).sum())
    m["null_close_bars"] = int(close.is_null().sum())

    # --- gaps (session-aware) ---
    step = CADENCE_MS[cadence]
    weekend_gaps = anomalous_gaps = 0
    max_anom_gap_h = 0.0
    for i in range(1, n):
        gap = ts[i] - ts[i - 1]
        if gap <= int(1.5 * step):
            continue
        if _is_weekend_gap(ts[i - 1], ts[i]):
            weekend_gaps += 1
        else:
            anomalous_gaps += 1
            max_anom_gap_h = max(max_anom_gap_h, gap / 3_600_000)
    m["weekend_gaps"] = weekend_gaps
    m["anomalous_gaps"] = anomalous_gaps
    m["max_anomalous_gap_h"] = round(max_anom_gap_h, 2)

    # --- status roll-up ---
    if n_nonfinite or n_nonpos or integ_viol or neg_spread or m.get("null_close_bars", 0):
        m["status"] = "fail"
        if n_nonfinite:
            m["issues"].append(f"nonfinite_close={n_nonfinite}")
        if n_nonpos:
            m["issues"].append(f"nonpositive_close={n_nonpos}")
        if integ_viol:
            m["issues"].append(f"ohlc_integrity={integ_viol}")
        if neg_spread:
            m["issues"].append(f"negative_spread={neg_spread}")
        if m.get("null_close_bars", 0):
            m["issues"].append(f"null_close={m['null_close_bars']}")
    else:
        warn = []
        if price_out_of_band:
            warn.append(f"price_median_{med}_outside_{lo}-{hi}(check_divisor)")
        if anomalous_gaps > max(5, int(0.02 * n)):
            warn.append(f"anomalous_gaps={anomalous_gaps}")
        if m.get("zero_tick_bars", 0) > int(0.05 * n):
            warn.append(f"zero_tick_bars={m['zero_tick_bars']}")
        if warn:
            m["status"] = "warn"
            m["issues"].extend(warn)
    return m


_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description="FX data-quality gate + pipeline metrics")
    cli.add_standard_args(ap, default_workers=1, date_window=False)
    ap.add_argument("--cadences", nargs="+", default=["1m", "1h", "1d"],
                    help="Cadences to check (default 1m 1h 1d).")
    ap.add_argument("--report", action="store_true",
                    help="Metrics-only mode: print + write JSON, always exit 0.")
    args = ap.parse_args()

    pairs = cli.resolve_pairs(args, stage_name="data_quality")
    U = UniverseLoader.load()
    all_metrics = []
    worst = 0
    print(f"\n{'pair':<8} {'cad':<4} {'status':<5} {'bars':>7} "
          f"{'spread_pips':>11} {'wknd':>5} {'anom':>5}  issues", flush=True)
    print("-" * 78, flush=True)
    for p in pairs:
        for cad in args.cadences:
            m = bar_metrics(p, cad, U)
            all_metrics.append(m)
            worst = max(worst, _STATUS_RANK[m["status"]])
            print(f"{p:<8} {cad:<4} {m['status']:<5} {m.get('n_bars', 0):>7} "
                  f"{m.get('spread_mean_pips', float('nan')):>11} "
                  f"{m.get('weekend_gaps', 0):>5} {m.get('anomalous_gaps', 0):>5}  "
                  f"{','.join(m['issues'][:3])}", flush=True)

    # Write metrics JSON.
    out_dir = layout.PROJECT_ROOT / "runs" / "pipeline_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"metrics_{stamp}.json"
    out_path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "worst_status": {0: "pass", 1: "warn", 2: "fail"}[worst],
        "metrics": all_metrics,
    }, indent=2), encoding="utf-8")
    print(f"\n[data_quality] metrics -> {out_path.relative_to(layout.PROJECT_ROOT)}", flush=True)
    print(f"[data_quality] worst status: {{0:'pass',1:'warn',2:'fail'}}[{worst}]", flush=True)

    if args.report:
        sys.exit(0)
    sys.exit(worst)


if __name__ == "__main__":
    main()
