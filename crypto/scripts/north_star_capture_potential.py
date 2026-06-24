"""North Star Capture Potential — empirical measurement of the 1-5%/day opportunity surface.

Answers: "On any given day, how many u100 assets move >=X%? What's the realistic
capture ceiling if we had perfect conviction-calling?"

Uses the chimera 1d data (close-to-close returns) across the full u100 universe
to produce:
  1. Daily mover counts by threshold (1%, 2%, 3%, 5%, 10%)
  2. Coverage stats: % of days with >=1, >=5, >=10 movers at each threshold
  3. Monthly distribution of movers
  4. The "if-you-caught-the-top-K" ceiling: compound return if you perfectly
     called the top-K movers each day and equal-weighted them
  5. Regime breakdown (are movers concentrated in bull/bear/chop?)

Output: JSON report + summary to stdout.

Usage:
  python crypto/scripts/north_star_capture_potential.py
  python crypto/scripts/north_star_capture_potential.py --threshold 2.0 --top-k 5
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # crypto/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import polars as pl
except ImportError:
    print("ERROR: polars required. pip install polars")
    sys.exit(1)


def load_universe(universe: str = "u100") -> list[str]:
    """Load symbol list from universe yaml."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "universes" / f"{universe}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Base symbols from u50 (parent)
    symbols = []
    if cfg.get("inherit_from") == "u50":
        u50_path = PROJECT_ROOT / "config" / "universes" / "u50.yaml"
        with open(u50_path) as f:
            u50 = yaml.safe_load(f)
        symbols = [a["symbol"] for a in u50.get("assets", [])]

    # Extra assets from u100
    extras = cfg.get("extra_assets", [])
    symbols += [a["symbol"] for a in extras]

    # Remove excluded
    excluded = set(cfg.get("excluded_assets", []))
    symbols = [s for s in symbols if s not in excluded]
    return symbols


def load_daily_returns(symbols: list[str]) -> pl.DataFrame:
    """Load 1d chimera data for all available symbols, extract close-to-close returns."""
    chimera_dir = PROJECT_ROOT / "data" / "processed" / "chimera" / "1d"
    if not chimera_dir.exists():
        print(f"ERROR: chimera 1d directory not found: {chimera_dir}")
        sys.exit(1)

    frames = []
    loaded = []
    missing = []

    for sym in symbols:
        sym_lower = sym.lower()
        # Find latest chimera file for this symbol
        files = sorted(chimera_dir.glob(f"{sym_lower}_v51_chimera_1d_*.parquet"))
        if not files:
            # Try without v51 prefix
            files = sorted(chimera_dir.glob(f"{sym_lower}_*.parquet"))
        if not files:
            missing.append(sym)
            continue

        latest = files[-1]
        try:
            df = pl.read_parquet(latest, columns=["date", "close"])
            # Compute daily return (close-to-close %)
            df = df.sort("date").with_columns([
                (pl.col("close").pct_change() * 100).alias("daily_ret_pct"),
                pl.lit(sym).alias("symbol"),
            ]).filter(pl.col("daily_ret_pct").is_not_null())
            frames.append(df.select(["date", "symbol", "daily_ret_pct"]))
            loaded.append(sym)
        except Exception as e:
            print(f"  SKIP {sym}: {e}")
            missing.append(sym)

    print(f"Loaded: {len(loaded)} symbols | Missing: {len(missing)}")
    if missing:
        print(f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    if not frames:
        print("ERROR: no data loaded")
        sys.exit(1)

    return pl.concat(frames)


def analyze_capture_potential(df: pl.DataFrame, thresholds: list[float],
                              top_k: int = 5) -> dict:
    """Core analysis: mover counts, coverage, capture ceiling."""
    results = {}

    # Get date range
    dates = df["date"].unique().sort()
    n_days = len(dates)
    date_min = str(dates[0])
    date_max = str(dates[-1])
    results["meta"] = {
        "n_days": n_days,
        "date_range": [date_min, date_max],
        "n_symbols": df["symbol"].n_unique(),
        "top_k": top_k,
    }

    # Per-threshold analysis
    threshold_results = {}
    for thresh in thresholds:
        # Count movers per day (absolute return >= threshold)
        daily_movers = (
            df.filter(pl.col("daily_ret_pct").abs() >= thresh)
            .group_by("date")
            .agg(pl.count().alias("n_movers"))
        )
        # Days with zero movers (need to add them)
        all_dates = df.select("date").unique()
        daily_movers = all_dates.join(daily_movers, on="date", how="left").with_columns(
            pl.col("n_movers").fill_null(0)
        )

        movers_arr = daily_movers["n_movers"].to_numpy()
        threshold_results[f">={thresh}%"] = {
            "mean_movers_per_day": round(float(movers_arr.mean()), 1),
            "median_movers_per_day": int(np.median(movers_arr)),
            "max_movers_per_day": int(movers_arr.max()),
            "pct_days_gte_1_mover": round(float((movers_arr >= 1).mean() * 100), 1),
            "pct_days_gte_5_movers": round(float((movers_arr >= 5).mean() * 100), 1),
            "pct_days_gte_10_movers": round(float((movers_arr >= 10).mean() * 100), 1),
            "pct_days_gte_20_movers": round(float((movers_arr >= 20).mean() * 100), 1),
        }

    results["thresholds"] = threshold_results

    # Long-only capture ceiling: if you perfectly called the top-K positive movers each day
    # (equal-weight, rebalance daily, only take POSITIVE moves since we're long-only)
    daily_top_k = (
        df.filter(pl.col("daily_ret_pct") > 0)  # long-only: only positive moves
        .sort(["date", "daily_ret_pct"], descending=[False, True])
        .group_by("date")
        .head(top_k)
        .group_by("date")
        .agg([
            pl.col("daily_ret_pct").mean().alias("avg_top_k_ret"),
            pl.col("daily_ret_pct").count().alias("n_picked"),
        ])
    )

    # Fill missing days with 0 return (couldn't find enough positive movers)
    all_dates_df = df.select("date").unique().sort("date")
    daily_top_k = all_dates_df.join(daily_top_k, on="date", how="left").with_columns([
        pl.col("avg_top_k_ret").fill_null(0.0),
        pl.col("n_picked").fill_null(0),
    ])

    avg_daily_ret = daily_top_k["avg_top_k_ret"].to_numpy()

    # Compound return
    compound_factor = np.prod(1 + avg_daily_ret / 100)
    compound_pct = (compound_factor - 1) * 100

    # Annual stats
    years = n_days / 365.25
    annual_compound = ((compound_factor ** (1 / years)) - 1) * 100 if years > 0 else 0

    results["capture_ceiling"] = {
        "description": f"Perfect top-{top_k} positive mover capture (equal-weight, daily rebalance, long-only)",
        "avg_daily_return_pct": round(float(avg_daily_ret.mean()), 3),
        "median_daily_return_pct": round(float(np.median(avg_daily_ret)), 3),
        "total_compound_pct": round(float(compound_pct), 1),
        "annualized_compound_pct": round(float(annual_compound), 1),
        "n_days": n_days,
        "years": round(years, 2),
        "avg_movers_picked_per_day": round(float(daily_top_k["n_picked"].mean()), 1),
        "pct_days_full_k_picked": round(float((daily_top_k["n_picked"].to_numpy() >= top_k).mean() * 100), 1),
    }

    # Yearly breakdown
    yearly = (
        daily_top_k.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 4).alias("year"))
        .group_by("year")
        .agg([
            pl.col("avg_top_k_ret").mean().alias("avg_daily_pct"),
            pl.col("avg_top_k_ret").count().alias("n_days"),
        ])
        .sort("year")
    )
    results["yearly_breakdown"] = {
        row["year"]: {
            "avg_daily_pct": round(float(row["avg_daily_pct"]), 3),
            "n_days": int(row["n_days"]),
        }
        for row in yearly.to_dicts()
    }

    # Various top-K scenarios
    k_scenarios = {}
    for k in [1, 3, 5, 10, 20]:
        daily_k = (
            df.filter(pl.col("daily_ret_pct") > 0)
            .sort(["date", "daily_ret_pct"], descending=[False, True])
            .group_by("date")
            .head(k)
            .group_by("date")
            .agg(pl.col("daily_ret_pct").mean().alias("avg_ret"))
        )
        daily_k = all_dates_df.join(daily_k, on="date", how="left").with_columns(
            pl.col("avg_ret").fill_null(0.0)
        )
        avg_k = daily_k["avg_ret"].to_numpy()
        compound_k = (np.prod(1 + avg_k / 100) - 1) * 100
        k_scenarios[f"top_{k}"] = {
            "avg_daily_pct": round(float(avg_k.mean()), 3),
            "total_compound_pct": round(float(compound_k), 1),
        }
    results["k_scenarios"] = k_scenarios

    return results


def main():
    ap = argparse.ArgumentParser(description="North Star capture potential analysis")
    ap.add_argument("--universe", default="u100", choices=["u10", "u50", "u100"])
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="Primary threshold for detailed analysis (default 2.0%%)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of top movers to capture per day (default 5)")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: auto-generated in runs/strat/)")
    args = ap.parse_args()

    print(f"=" * 70)
    print(f"NORTH STAR CAPTURE POTENTIAL ANALYSIS")
    print(f"Universe: {args.universe} | Top-K: {args.top_k} | Primary threshold: {args.threshold}%")
    print(f"=" * 70)
    print()

    # Load universe
    print("Loading universe symbols...")
    symbols = load_universe(args.universe)
    print(f"  Universe {args.universe}: {len(symbols)} symbols")
    print()

    # Load daily returns
    print("Loading 1d chimera returns...")
    df = load_daily_returns(symbols)
    print(f"  Total rows: {len(df):,}")
    print()

    # Run analysis
    thresholds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    print(f"Analyzing capture potential (thresholds: {thresholds})...")
    results = analyze_capture_potential(df, thresholds, top_k=args.top_k)
    results["meta"]["universe"] = args.universe
    results["meta"]["run_timestamp"] = datetime.now().isoformat()

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDate range: {results['meta']['date_range'][0]} to {results['meta']['date_range'][1]}")
    print(f"Days: {results['meta']['n_days']} | Symbols: {results['meta']['n_symbols']}")
    print()

    print("--- MOVER COUNTS BY THRESHOLD ---")
    print(f"{'Threshold':<12} {'Avg/day':<10} {'Median':<8} {'>=1 mover':<12} {'>=5 movers':<12} {'>=10 movers':<12} {'>=20 movers':<12}")
    for t, v in results["thresholds"].items():
        print(f"{t:<12} {v['mean_movers_per_day']:<10} {v['median_movers_per_day']:<8} "
              f"{v['pct_days_gte_1_mover']:.0f}%{'':7} {v['pct_days_gte_5_movers']:.0f}%{'':7} "
              f"{v['pct_days_gte_10_movers']:.0f}%{'':7} {v['pct_days_gte_20_movers']:.0f}%")
    print()

    print("--- CAPTURE CEILING (long-only, perfect conviction) ---")
    cc = results["capture_ceiling"]
    print(f"  Strategy: {cc['description']}")
    print(f"  Avg daily return: {cc['avg_daily_return_pct']:.3f}%")
    print(f"  Median daily return: {cc['median_daily_return_pct']:.3f}%")
    print(f"  Total compound: {cc['total_compound_pct']:.1f}%")
    print(f"  Annualized compound: {cc['annualized_compound_pct']:.1f}%")
    print(f"  Days with full K picked: {cc['pct_days_full_k_picked']:.0f}%")
    print()

    print("--- TOP-K SCENARIOS ---")
    print(f"{'K':<8} {'Avg daily %':<14} {'Total compound %':<18}")
    for k, v in results["k_scenarios"].items():
        print(f"{k:<8} {v['avg_daily_pct']:<14.3f} {v['total_compound_pct']:<18.1f}")
    print()

    print("--- YEARLY BREAKDOWN ---")
    for year, v in results["yearly_breakdown"].items():
        print(f"  {year}: avg daily {v['avg_daily_pct']:.3f}% ({v['n_days']} days)")
    print()

    # 1-5% daily target feasibility
    target_min = 1.0
    target_max = 5.0
    avg_daily = cc['avg_daily_return_pct']
    print("--- NORTH STAR FEASIBILITY ---")
    print(f"  Target: {target_min}-{target_max}%/day")
    print(f"  Perfect-info ceiling (top-{args.top_k}): {avg_daily:.3f}%/day")
    if avg_daily >= target_min:
        print(f"  VERDICT: The {target_min}% floor IS within the perfect-info ceiling.")
        pct_of_ceiling = (target_min / avg_daily) * 100
        print(f"  Required conviction accuracy (floor): ~{pct_of_ceiling:.0f}% of perfect info")
    else:
        print(f"  VERDICT: The {target_min}% floor EXCEEDS the top-{args.top_k} ceiling.")
        print(f"  Need higher K or better universe coverage.")
    print()

    # Save output
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = PROJECT_ROOT / "runs" / "strat" / f"north_star_capture_{args.universe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results saved: {out_path}")


if __name__ == "__main__":
    main()
