"""
MA x Kyle/VPIN/Hawkes Conjunction Filter Test
==============================================
Tests whether order-flow filters (Kyle lambda, VPIN, Hawkes imbalance)
improve MA-cross signal quality when used as conjunction gates.

Discipline: TRAIN+VAL only (2020-01 to 2025-09). UNSEEN never touched.
Unconditional measurement (no event-day filter bias).
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("C:/Users/karab/Documents/coding/v4_crypto_stystem")
CHIMERA_1D = BASE / "data/processed/chimera/1d"
EVENT_SNAP = BASE / "runs/oracle_layer3/ma_ema_permutation/event_ma_snapshot.parquet"
PAIR_SUMMARY = BASE / "runs/oracle_layer3/ma_ema_permutation/pair_summary.parquet"
OUT_DIR = BASE / "runs/dossiers/MA_x_KYLE_conjunction_2026_05_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TRAIN+VAL cutoff: 2025-09-30 (OOS starts Oct 2025; UNSEEN after that)
TRAIN_VAL_END = "2025-09-30"

ORDER_FLOW_COLS = ["norm_kyle_lambda", "norm_vpin", "norm_hawkes_imbalance"]
FILTER_THRESHOLD = 2.0  # top ~5-10% by absolute z-score

# Cost model: taker round-trip ~0.1% per side = 0.2% round-trip
COST_RT = 0.002  # 0.2%


def load_chimera_orderflow(asset: str) -> pd.DataFrame | None:
    """Load 1d chimera, return DataFrame with date + order-flow columns."""
    symbol = asset.lower() + "usdt"
    # Find the latest chimera file for this asset
    files = list(CHIMERA_1D.glob(f"{symbol}_v51_chimera_1d_*.parquet"))
    if not files:
        return None
    latest = sorted(files)[-1]
    df = pd.read_parquet(latest)
    # Ensure date column is string-compatible
    if "date" not in df.columns:
        return None
    df["date_str"] = df["date"].astype(str)
    keep = ["date_str"] + [c for c in ORDER_FLOW_COLS if c in df.columns]
    return df[keep].rename(columns={"date_str": "date"})


def compute_ma_signal(row, ma_type: str, fast: int, slow: int) -> int:
    """Derive MA-cross signal from snapshot row. +1=long, -1=short."""
    fast_col = f"{ma_type}_{fast}"
    slow_col = f"{ma_type}_{slow}"
    if fast_col not in row.index or slow_col not in row.index:
        return 0
    fv, sv = row[fast_col], row[slow_col]
    if pd.isna(fv) or pd.isna(sv):
        return 0
    return 1 if fv > sv else -1


def sharpe(pnl_series: pd.Series) -> float:
    if len(pnl_series) < 5 or pnl_series.std() == 0:
        return np.nan
    return pnl_series.mean() / pnl_series.std() * np.sqrt(252)


def hit_rate(pnl_series: pd.Series) -> float:
    return (pnl_series > 0).mean()


def evaluate_cell(events_subset: pd.DataFrame, ma_type: str, fast: int, slow: int) -> dict:
    """Compute signal quality for a (ma_type, fast, slow) cell on events_subset."""
    results = []
    for _, row in events_subset.iterrows():
        sig = compute_ma_signal(row, ma_type, fast, slow)
        if sig == 0:
            continue
        # PnL = signal direction * magnitude_signed - cost
        raw_pnl = sig * row["magnitude_signed"] / 100.0
        net_pnl = raw_pnl - COST_RT
        results.append(net_pnl)
    if not results:
        return {"n_fires": 0, "sharpe": np.nan, "hit_rate": np.nan, "mean_pnl_pct": np.nan}
    s = pd.Series(results)
    return {
        "n_fires": len(s),
        "sharpe": sharpe(s),
        "hit_rate": hit_rate(s),
        "mean_pnl_pct": s.mean() * 100,
    }


def main():
    print("=" * 60)
    print("MA x Order-Flow Conjunction Filter Test")
    print("=" * 60)

    # Step 1: Load event snapshot, restrict to TRAIN+VAL
    events = pd.read_parquet(EVENT_SNAP)
    events["date"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    events = events[events["date"] <= TRAIN_VAL_END].copy()
    print(f"Events (TRAIN+VAL): {len(events):,} rows, {events['date'].min()} to {events['date'].max()}")

    # Step 2: Load top MA pairs from pair_summary (universe-wide)
    pairs = pd.read_parquet(PAIR_SUMMARY)
    top_pairs = pairs.nlargest(10, "sharpe_proxy")[["ma_type", "fast", "slow", "sharpe_proxy", "n_signaled"]].reset_index(drop=True)
    print(f"\nTop 10 MA pairs (universe-wide baseline):")
    print(top_pairs.to_string(index=False))

    # Step 3: Load order-flow data for all assets in event snapshot
    print("\nLoading order-flow data...")
    asset_of_map = {}
    for asset in events["asset"].unique():
        df = load_chimera_orderflow(asset)
        if df is not None:
            asset_of_map[asset] = df.set_index("date")
    print(f"Loaded order-flow for {len(asset_of_map)} / {events['asset'].nunique()} assets")

    # Step 4: Join order-flow to events
    of_rows = []
    for _, row in events.iterrows():
        asset = row["asset"]
        date = row["date"]
        of_data = {"kyle": np.nan, "vpin": np.nan, "hawkes": np.nan}
        if asset in asset_of_map:
            asset_df = asset_of_map[asset]
            if date in asset_df.index:
                arow = asset_df.loc[date]
                if "norm_kyle_lambda" in arow.index:
                    of_data["kyle"] = arow["norm_kyle_lambda"]
                if "norm_vpin" in arow.index:
                    of_data["vpin"] = arow["norm_vpin"]
                if "norm_hawkes_imbalance" in arow.index:
                    of_data["hawkes"] = arow["norm_hawkes_imbalance"]
        of_rows.append(of_data)

    of_df = pd.DataFrame(of_rows, index=events.index)
    events = pd.concat([events, of_df], axis=1)

    # Coverage stats
    for col in ["kyle", "vpin", "hawkes"]:
        pct = events[col].notna().mean() * 100
        extreme_pct = (events[col].abs() >= FILTER_THRESHOLD).sum() / len(events) * 100
        print(f"  {col}: {pct:.1f}% coverage, {extreme_pct:.1f}% >= |{FILTER_THRESHOLD}|")

    print(f"\nRunning conjunction tests across top {len(top_pairs)} MA pairs...")

    comparison_rows = []

    for _, pair_row in top_pairs.iterrows():
        ma_type = pair_row["ma_type"]
        fast = int(pair_row["fast"])
        slow = int(pair_row["slow"])
        label = f"{ma_type}({fast},{slow})"

        # Baseline: all events
        baseline = evaluate_cell(events, ma_type, fast, slow)

        # Kyle conjunction: fire only when |kyle| >= threshold
        kyle_mask = events["kyle"].abs() >= FILTER_THRESHOLD
        kyle_events = events[kyle_mask]
        kyle = evaluate_cell(kyle_events, ma_type, fast, slow)

        # VPIN conjunction
        vpin_mask = events["vpin"].abs() >= FILTER_THRESHOLD
        vpin_events = events[vpin_mask]
        vpin = evaluate_cell(vpin_events, ma_type, fast, slow)

        # Hawkes conjunction
        hawkes_mask = events["hawkes"].abs() >= FILTER_THRESHOLD
        hawkes_events = events[hawkes_mask]
        hawkes = evaluate_cell(hawkes_events, ma_type, fast, slow)

        # Triple conjunction: all three extreme
        triple_mask = kyle_mask & vpin_mask & hawkes_mask
        triple_events = events[triple_mask]
        triple = evaluate_cell(triple_events, ma_type, fast, slow)

        row = {
            "pair": label,
            # Baseline
            "base_sh": round(baseline["sharpe"], 3),
            "base_n": baseline["n_fires"],
            "base_hr": round(baseline["hit_rate"], 3),
            "base_pnl": round(baseline["mean_pnl_pct"], 3),
            # Kyle
            "kyle_sh": round(kyle["sharpe"], 3) if not np.isnan(kyle["sharpe"]) else "N/A",
            "kyle_n": kyle["n_fires"],
            "kyle_hr": round(kyle["hit_rate"], 3) if not np.isnan(kyle["hit_rate"]) else "N/A",
            "kyle_pnl": round(kyle["mean_pnl_pct"], 3) if not np.isnan(kyle["mean_pnl_pct"]) else "N/A",
            # VPIN
            "vpin_sh": round(vpin["sharpe"], 3) if not np.isnan(vpin["sharpe"]) else "N/A",
            "vpin_n": vpin["n_fires"],
            "vpin_hr": round(vpin["hit_rate"], 3) if not np.isnan(vpin["hit_rate"]) else "N/A",
            "vpin_pnl": round(vpin["mean_pnl_pct"], 3) if not np.isnan(vpin["mean_pnl_pct"]) else "N/A",
            # Hawkes
            "hawkes_sh": round(hawkes["sharpe"], 3) if not np.isnan(hawkes["sharpe"]) else "N/A",
            "hawkes_n": hawkes["n_fires"],
            "hawkes_hr": round(hawkes["hit_rate"], 3) if not np.isnan(hawkes["hit_rate"]) else "N/A",
            "hawkes_pnl": round(hawkes["mean_pnl_pct"], 3) if not np.isnan(hawkes["mean_pnl_pct"]) else "N/A",
            # Triple
            "triple_sh": round(triple["sharpe"], 3) if not np.isnan(triple["sharpe"]) else "N/A",
            "triple_n": triple["n_fires"],
        }

        # Verdict: conjunction passes if sh >= 2x baseline AND n >= 20
        def verdict(conj_sh, conj_n, base_sh):
            if conj_n < 10:
                return "SPARSE"
            if isinstance(conj_sh, str):
                return "N/A"
            if conj_n < 20:
                return "LOW-N"
            if conj_sh >= 2 * base_sh and conj_sh > 0.5:
                return "PASS"
            if conj_sh > base_sh:
                return "MARGINAL"
            return "FAIL"

        row["kyle_verdict"] = verdict(kyle["sharpe"], kyle["n_fires"], baseline["sharpe"])
        row["vpin_verdict"] = verdict(vpin["sharpe"], vpin["n_fires"], baseline["sharpe"])
        row["hawkes_verdict"] = verdict(hawkes["sharpe"], hawkes["n_fires"], baseline["sharpe"])
        row["triple_verdict"] = verdict(triple["sharpe"], triple["n_fires"], baseline["sharpe"])

        comparison_rows.append(row)
        print(f"  {label}: base_sh={baseline['sharpe']:.3f} | kyle_sh={kyle['sharpe']:.3f}(n={kyle['n_fires']}) [{row['kyle_verdict']}] | vpin_sh={vpin['sharpe']:.3f}(n={vpin['n_fires']}) [{row['vpin_verdict']}] | hawkes_sh={hawkes['sharpe']:.3f}(n={hawkes['n_fires']}) [{row['hawkes_verdict']}]")

    result_df = pd.DataFrame(comparison_rows)

    # Step 5: Extended per-asset test for best conjunction
    # For each filter, find which assets have PASS verdict
    print("\n--- EXTENDED: Per-asset test on best filter ---")
    best_filter_results = []
    for filter_name in ["kyle", "vpin", "hawkes"]:
        mask_col = filter_name
        mask = events[mask_col].abs() >= FILTER_THRESHOLD
        filtered_events = events[mask]
        # Test universal best pair: SMA(13,34) from memory doc
        for ma_type, fast, slow in [("SMA", 13, 34), ("SMA", 28, 29), ("SMA", 2, 70)]:
            # Per-asset breakdown
            for asset in sorted(events["asset"].unique()):
                asset_mask = filtered_events["asset"] == asset
                asset_events = filtered_events[asset_mask]
                if len(asset_events) < 5:
                    continue
                res = evaluate_cell(asset_events, ma_type, fast, slow)
                if res["n_fires"] >= 15 and not np.isnan(res["sharpe"]) and res["sharpe"] >= 1.5:
                    best_filter_results.append({
                        "filter": filter_name,
                        "pair": f"{ma_type}({fast},{slow})",
                        "asset": asset,
                        "n_fires": res["n_fires"],
                        "sharpe": round(res["sharpe"], 3),
                        "hit_rate": round(res["hit_rate"], 3),
                        "mean_pnl_pct": round(res["mean_pnl_pct"], 3),
                    })

    per_asset_df = pd.DataFrame(best_filter_results)
    if len(per_asset_df) > 0:
        per_asset_df = per_asset_df.sort_values("sharpe", ascending=False)
        print(f"Found {len(per_asset_df)} per-asset cells with Sh >= 1.5 (n >= 15):")
        print(per_asset_df.to_string(index=False))
    else:
        print("No per-asset cells cleared Sh >= 1.5 with n >= 15")

    # Save outputs
    result_df.to_csv(OUT_DIR / "conjunction_comparison.csv", index=False)
    if len(per_asset_df) > 0:
        per_asset_df.to_csv(OUT_DIR / "per_asset_pass_cells.csv", index=False)

    # Summary verdict
    print("\n" + "=" * 60)
    print("SUMMARY VERDICTS")
    print("=" * 60)
    for filter_name in ["kyle", "vpin", "hawkes", "triple"]:
        col = f"{filter_name}_verdict"
        if col in result_df.columns:
            counts = result_df[col].value_counts().to_dict()
            print(f"  {filter_name}: {counts}")

    # DSR correction: n_trials = 10 pairs x 4 filters = 40
    n_trials = 40
    # Bailey 2014: DSR = Sh * sqrt((1 - gamma + gamma * Sh^2) / n_trials)
    # Simplified: if best conjunction Sh exists, apply correction
    pass_cells = [r for r in comparison_rows if r.get("kyle_verdict") == "PASS"
                  or r.get("vpin_verdict") == "PASS" or r.get("hawkes_verdict") == "PASS"]
    if pass_cells:
        print(f"\n{len(pass_cells)} PASS cells found. DSR correction (n_trials={n_trials}):")
        for pc in pass_cells:
            for fn in ["kyle", "vpin", "hawkes"]:
                sh = pc.get(f"{fn}_sh")
                if isinstance(sh, float) and pc.get(f"{fn}_verdict") == "PASS":
                    gamma = 0.717
                    dsr = sh * np.sqrt((1 - gamma + gamma * sh ** 2) / n_trials)
                    print(f"  {pc['pair']} x {fn}: raw_sh={sh:.3f}, DSR={dsr:.3f} {'SURVIVES(>0.9)' if dsr > 0.9 else 'DEFLATED'}")
    else:
        print("\nNo PASS cells found -- DSR correction not applicable.")

    print(f"\nOutputs written to: {OUT_DIR}")
    return result_df, per_asset_df


if __name__ == "__main__":
    main()
