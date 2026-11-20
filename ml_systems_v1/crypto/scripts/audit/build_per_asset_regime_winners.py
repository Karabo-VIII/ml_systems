"""build_per_asset_regime_winners.py -- per-asset × per-own-regime winner cells.

User insight (2026-05-20):
  "We have a new and interesting dimension: own_bull winner, own_bear winner,
   own_chop winner, etc. So each asset now has regime-specific setups that
   statistically skew things favorably, catching wins even when the weather
   within the asset isn't green."

ARCHITECTURE:
  For each asset A, for each own-regime R in {bull, chop, bear, crash}:
    Identify the cell that performs best on A's events when A is in regime R.
    Filter: n_signaled_in_regime >= 3, mean > cost, hit >= 0.4.
    Pick by sharpe-proxy descending.

  Output: data/processed/per_asset_regime_winners.parquet
    Columns: asset, own_regime, ma_type, fast, slow, n_in_regime, mean_pct,
             hit, sum_pct, sharpe, exit_recommendation

  Deploy rule: at each (asset, date):
    1. Look up A's own_regime today.
    2. If A has a winner cell for that regime, that cell is the candidate.
       Otherwise asset is cash for today (no qualifying cell).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
OUT = ROOT / "data" / "processed" / "per_asset_regime_winners.parquet"
OUT_MD = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "PER_REGIME_WINNERS.md"

MIN_N_IN_REGIME = 3
MIN_MEAN_PCT = 0.0  # already cost-deducted in profile stats
MIN_HIT = 0.40
MIN_SHARPE = -0.05  # tolerate slightly-negative if hit + asym is favorable


def parse_regime(j: str) -> dict:
    try:
        d = json.loads(j)
        return d if d else None
    except Exception:
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    profile = pd.read_parquet(PROFILE)
    print(f"Loaded profile: {len(profile)} cells across {profile['asset'].nunique()} assets")

    winners = []
    for asset, asset_cells in profile.groupby("asset"):
        bucket = asset_cells["bucket"].iloc[0]
        for regime in ("bull", "chop", "bear", "crash"):
            # For each candidate cell on this asset, parse its per-own-regime stats
            candidates = []
            for _, c in asset_cells.iterrows():
                stats = parse_regime(c[f"own_{regime}"])
                if stats is None:
                    continue
                n = stats.get("n", 0)
                mean_pct = stats.get("mean", 0)
                hit = stats.get("hit", 0)
                sharpe = stats.get("sharpe", 0)
                if (n >= MIN_N_IN_REGIME and mean_pct > MIN_MEAN_PCT
                    and hit >= MIN_HIT and sharpe > MIN_SHARPE):
                    candidates.append({
                        "asset": asset, "bucket": bucket, "own_regime": regime,
                        "cell_id": c["cell_id"], "ma_type": c["ma_type"],
                        "fast": int(c["fast"]), "slow": int(c["slow"]),
                        "n_in_regime": int(n),
                        "mean_pct_in_regime": float(mean_pct),
                        "hit_in_regime": float(hit),
                        "sum_pct_in_regime": float(stats.get("sum", 0)),
                        "sharpe_in_regime": float(sharpe),
                        "overall_sharpe": float(c["sharpe"]),
                    })
            if candidates:
                # Pick the cell with highest sharpe_in_regime; tiebreak by mean_pct
                candidates_sorted = sorted(candidates,
                                            key=lambda x: (-x["sharpe_in_regime"], -x["mean_pct_in_regime"]))
                best = candidates_sorted[0]
                winners.append(best)

    if not winners:
        print("[FATAL] no per-regime winners found")
        return 2

    winners_df = pd.DataFrame(winners)
    print(f"\nPer-asset × per-regime winners: {len(winners_df)} cells")
    print(f"\nDistribution:")
    print(winners_df.groupby(["bucket", "own_regime"]).size().unstack(fill_value=0).to_string())

    # Coverage: how many assets have winners in each regime?
    print(f"\nAssets with winner per regime:")
    cov = winners_df.groupby("own_regime")["asset"].nunique()
    n_total_assets = profile["asset"].nunique()
    for r in ("bull", "chop", "bear", "crash"):
        print(f"  own_{r:<6}: {cov.get(r, 0):2d}/{n_total_assets} assets ({cov.get(r, 0)/n_total_assets*100:.0f}%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    winners_df.to_parquet(OUT, index=False)
    print(f"\n[OK] wrote {OUT}")

    # Markdown
    lines = [
        "# Per-Asset × Per-Own-Regime MA/EMA Winners (2026-05-20)\n",
        "**Method**: for each asset × own-regime cell, take the per-asset MA/EMA cell with",
        "the highest Sharpe-proxy ON THAT REGIME'S EVENTS. Quality gates:",
        f"  - n_in_regime ≥ {MIN_N_IN_REGIME}",
        f"  - mean_pct ≥ {MIN_MEAN_PCT}% (cost-deducted)",
        f"  - hit ≥ {MIN_HIT*100:.0f}%",
        f"  - sharpe ≥ {MIN_SHARPE}",
        "",
        f"**Output**: {len(winners_df)} winners across {winners_df['asset'].nunique()} assets",
        "",
        "## A. Coverage by own-regime",
        "",
        "| regime | assets with winner | % of profile universe |",
        "|---|---:|---:|",
    ]
    for r in ("bull", "chop", "bear", "crash"):
        n = cov.get(r, 0)
        lines.append(f"| {r} | {n} | {n/n_total_assets*100:.1f}% |")

    lines += ["", "## B. Bucket × regime distribution", ""]
    bd = winners_df.groupby(["bucket", "own_regime"]).size().unstack(fill_value=0)
    lines.append("| bucket | " + " | ".join(bd.columns.tolist()) + " | total |")
    lines.append("|---|" + "|".join(["---:"] * (len(bd.columns) + 1)) + "|")
    for b, row in bd.iterrows():
        lines.append(f"| {b} | " + " | ".join(str(v) for v in row) + f" | {row.sum()} |")

    lines += ["", "## C. All winner cells (sorted by asset, regime)", ""]
    lines.append("| asset | bucket | regime | type | (fast, slow) | n | mean % | hit % | sum % | regime Sharpe |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for _, w in winners_df.sort_values(["asset", "own_regime"]).iterrows():
        lines.append(f"| {w['asset']} | {w['bucket']} | {w['own_regime']} | {w['ma_type']} | "
                     f"({w['fast']}, {w['slow']}) | {w['n_in_regime']} | "
                     f"{w['mean_pct_in_regime']:+.3f} | {w['hit_in_regime']*100:.1f} | "
                     f"{w['sum_pct_in_regime']:+.2f} | {w['sharpe_in_regime']:+.4f} |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
