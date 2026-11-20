"""build_bucket_dna_fallback.py -- bucket-default MA/EMA cell for assets without
   a per-asset profile (e.g., new listings absent from TRAIN).

User mandate (2026-05-20): "There are assets that did not exist in the train set,
but might exist in VAL and OOS. How do we develop profiles for those?"

APPROACH (Tier-1 bucket-DNA fallback):
  For each DNA bucket (BLUE / STEADY / VOLATILE / DEGEN):
    1. Take all cousin-set cells from per_asset_ma_ema_profile for assets in that bucket.
    2. Find the (ma_type, fast, slow) configuration that appears most often, weighted
       by per-asset Sharpe-proxy. That's the bucket's "default cell."
    3. Output one (ma_type, fast, slow, regime_tag, exit_policy) per bucket.

  At deploy time, when an asset A has no profile:
    a. Look up A's bucket from universes yaml.
    b. Use bucket_default[A.bucket] as A's fallback cell.
    c. Tag as regime-tag = "BULL_AND_CHOP" (conservative default).

EXTENSIONS (later):
  - Tier-2: sector-DNA fallback (more specific, e.g., DEGEN-meme vs DEGEN-AI)
  - Tier-3: JIT mini-profile (once asset has ≥60 days of data, build a simplified
            single-fold per-asset cell)
"""
from __future__ import annotations

import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
U50_YAML = ROOT / "config" / "universes" / "u50.yaml"
U100_YAML = ROOT / "config" / "universes" / "u100.yaml"
OUT = ROOT / "data" / "processed" / "bucket_dna_fallback.parquet"
OUT_MD = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "BUCKET_FALLBACK.md"


def load_universe():
    """Load u100 universe with asset → bucket mapping."""
    with open(U50_YAML) as f:
        u50 = yaml.safe_load(f)
    with open(U100_YAML) as f:
        u100 = yaml.safe_load(f)
    asset_to_bucket = {}
    for a in u50["assets"]:
        sym = a["symbol"].replace("USDT", "")
        asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    for a in u100.get("extra_assets", []):
        if a.get("status") != "ready": continue
        sym = a["symbol"].replace("USDT", "")
        asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    return asset_to_bucket


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("BUCKET-DNA FALLBACK CELL DERIVATION")
    print("="*78)

    profile = pd.read_parquet(PROFILE)
    cousins = profile[profile["is_cousin_set_member"]].copy()
    print(f"Profile cousin-set members: {len(cousins)} cells across {cousins['asset'].nunique()} assets")

    # For each bucket, derive the MOST COMMON (ma_type, fast, slow) weighted by Sharpe.
    bucket_defaults = []
    for bucket, sub in cousins.groupby("bucket"):
        # Score each (ma_type, fast, slow) tuple by sum-of-Sharpe across this bucket
        tup_scores = (sub.groupby(["ma_type", "fast", "slow"])
                       .agg(n_assets=("asset", "nunique"),
                            sum_sharpe=("sharpe", "sum"),
                            mean_sharpe=("sharpe", "mean"),
                            mean_mean_pct=("overall_mean_pct", "mean"),
                            mean_hit=("overall_hit", "mean"))
                       .reset_index()
                       .sort_values(["n_assets", "sum_sharpe"], ascending=False))
        if tup_scores.empty:
            continue
        # Top configuration = appears in most assets AND has highest collective Sharpe
        top = tup_scores.iloc[0]
        n_assets_bucket = sub["asset"].nunique()
        print(f"\n{bucket} ({n_assets_bucket} assets in profile):")
        print(f"  top tuples (by n_assets × sum_sharpe):")
        for _, t in tup_scores.head(5).iterrows():
            print(f"    {t['ma_type']}({t['fast']:>3}, {t['slow']:>3}) — n={int(t['n_assets'])} "
                  f"sum_Sh={t['sum_sharpe']:+.3f} mean_Sh={t['mean_sharpe']:+.3f} mean_pct={t['mean_mean_pct']:+.3f} hit={t['mean_hit']*100:.1f}%")
        bucket_defaults.append({
            "bucket": bucket,
            "ma_type": top["ma_type"],
            "fast": int(top["fast"]),
            "slow": int(top["slow"]),
            "n_assets_in_bucket_with_this_config": int(top["n_assets"]),
            "mean_sharpe_proxy": float(top["mean_sharpe"]),
            "mean_pct_per_event": float(top["mean_mean_pct"]),
            "mean_hit_rate": float(top["mean_hit"]),
            # Regime-tag: be conservative for fallback assets (no per-asset data to know better)
            "regime_tag": "BULL_AND_CHOP",
            "derivation": "modal_cousin_cell_per_bucket_weighted_by_sharpe",
        })

    defaults_df = pd.DataFrame(bucket_defaults)
    print(f"\n=== BUCKET DEFAULTS ===")
    print(defaults_df[["bucket", "ma_type", "fast", "slow", "n_assets_in_bucket_with_this_config",
                       "mean_sharpe_proxy", "mean_hit_rate"]].to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    defaults_df.to_parquet(OUT, index=False)
    print(f"\n[OK] wrote {OUT}")

    # Universe coverage report
    asset_to_bucket = load_universe()
    profile_assets = set(profile["asset"].unique())
    universe_assets = set(asset_to_bucket.keys())
    unprofiled = universe_assets - profile_assets
    print(f"\n=== UNIVERSE COVERAGE ===")
    print(f"  u100 universe: {len(universe_assets)} assets")
    print(f"  Profile coverage: {len(profile_assets)} assets ({len(profile_assets)/len(universe_assets)*100:.1f}%)")
    print(f"  Unprofiled (will use fallback): {len(unprofiled)} assets")
    if unprofiled:
        unprofiled_buckets = {}
        for a in unprofiled:
            b = asset_to_bucket[a]
            unprofiled_buckets.setdefault(b, []).append(a)
        print(f"\n  Unprofiled by bucket:")
        for b, assets in unprofiled_buckets.items():
            print(f"    {b:<10}: {len(assets):2d} assets — {sorted(assets)[:10]}{'...' if len(assets)>10 else ''}")

    # Markdown
    lines = [
        "# Bucket-DNA Fallback Cells for Unprofiled Assets (2026-05-20)\n",
        "**Purpose**: provide a deployable MA/EMA cell for assets without per-asset profile",
        "(typically new listings absent from TRAIN window).\n",
        "**Method**: for each DNA bucket, modal (ma_type, fast, slow) across cousin-set cells",
        "of profiled assets in that bucket, weighted by Sharpe-proxy.\n",
        "## Default cells",
        "",
        "| bucket | type | (fast, slow) | n_assets_w_this | mean Sharpe | mean PnL/event | mean hit % |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in defaults_df.iterrows():
        lines.append(f"| {r['bucket']} | {r['ma_type']} | ({r['fast']}, {r['slow']}) | "
                     f"{int(r['n_assets_in_bucket_with_this_config'])} | "
                     f"{r['mean_sharpe_proxy']:+.4f} | {r['mean_pct_per_event']:+.3f}% | "
                     f"{r['mean_hit_rate']*100:.1f}% |")
    lines += [
        "",
        "## Universe coverage",
        "",
        f"- u100 universe: {len(universe_assets)} assets",
        f"- Profile coverage: {len(profile_assets)} assets ({len(profile_assets)/len(universe_assets)*100:.1f}%)",
        f"- Unprofiled (fallback users): {len(unprofiled)} assets",
        "",
        "**Tier-1 fallback policy**: unprofiled assets deploy with the bucket-default cell,",
        "regime_tag=BULL_AND_CHOP (conservative — own_bull and own_chop days only).",
        "",
        "**Future Tier-2/3** (not built this iteration):",
        "- Sector-DNA fallback (DEGEN-meme vs DEGEN-AI: different default cells)",
        "- JIT mini-profile (once asset has ≥60 days data, build simplified profile)",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
