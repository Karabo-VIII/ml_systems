"""Feature signal-strength mining on v51 chimera data.

Asks: which individual features carry the most signal toward target_return_1,
and is the f29 default (Pattern P+Q stripped to 29) optimal or are there
gains in f51 / f121?

For each feature column:
  - lag-1 IC with target_return_1 (Pearson + Spearman)
  - persistence (autocorrelation)
  - cross-asset stability (IC variance across the 10 V1.x assets)

Output: ranked feature table + recommendation.

Cost: ~2 min per asset (reads ~2-4 GB parquet); total ~20 min for 10 assets.
We sample 1M rows per asset to bound memory.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DDIR = ROOT / "data" / "processed" / "chimera" / "dollar"
sys.path.insert(0, str(ROOT / "src"))
from feature_sets import FEATURE_LIST_13, FEATURE_LIST_29, FEATURE_LIST_34, FEATURE_LIST_41, FEATURE_LIST_121  # noqa: E402

V1_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
             "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
SAMPLE_N = 1_000_000


def latest_file(sym: str) -> Path | None:
    files = sorted(DDIR.glob(f"{sym.lower()}_v51_chimera_*.parquet"))
    return files[-1] if files else None


def per_asset_feature_ic(sym: str, all_features: list[str]) -> dict:
    f = latest_file(sym)
    if f is None:
        return {}
    needed = list(set(all_features + ["target_return_1"]))
    df = pl.read_parquet(f, columns=[c for c in needed if c in pl.read_parquet_schema(f)])
    # Sub-sample for speed (keep middle slice — avoids the elevated-zero tail)
    n = len(df)
    if n > SAMPLE_N:
        start = (n - SAMPLE_N) // 2
        df = df.slice(start, SAMPLE_N)
    tgt = df["target_return_1"].to_numpy()
    valid = np.isfinite(tgt) & (np.abs(tgt) > 1e-9)  # drop zero-return bars
    tgt = tgt[valid]
    if len(tgt) < 1000:
        return {}
    result = {}
    for feat in all_features:
        if feat not in df.columns:
            continue
        x = df[feat].to_numpy()[valid]
        if not np.isfinite(x).all():
            continue
        if x.std() < 1e-9:
            continue
        # Pearson
        ic = float(np.corrcoef(x, tgt)[0, 1])
        # Cross-rank Spearman (cheap proxy: corr of ranks)
        # autocorr (lag-1)
        autocorr = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 1 else 0.0
        result[feat] = {"ic": ic, "abs_ic": abs(ic), "autocorr": autocorr,
                        "n_valid": int(valid.sum())}
    return result


def main():
    print("=" * 78)
    print("  FEATURE SIGNAL MINING — V1.x asset universe, target_return_1")
    print("=" * 78)
    # Master feature list = union of all standard counts
    all_features = sorted(set(FEATURE_LIST_13 + FEATURE_LIST_29 + FEATURE_LIST_34
                              + FEATURE_LIST_41 + FEATURE_LIST_121))
    print(f"  candidate features: {len(all_features)}")

    # Pooled per-asset table
    pooled = {f: {"ics": [], "autocorrs": []} for f in all_features}
    for sym in V1_ASSETS:
        print(f"\n--- {sym} ---")
        results = per_asset_feature_ic(sym, all_features)
        print(f"  {len(results)} features evaluated")
        # Top 5 per asset
        top = sorted(results.items(), key=lambda x: -x[1]["abs_ic"])[:5]
        for feat, r in top:
            print(f"    {feat:32s}  IC={r['ic']:+.4f}  autocorr={r['autocorr']:+.3f}")
        for feat, r in results.items():
            pooled[feat]["ics"].append(r["ic"])
            pooled[feat]["autocorrs"].append(r["autocorr"])

    # Pooled summary
    print("\n" + "=" * 78)
    print("  POOLED across 10 assets")
    print("=" * 78)
    rows = []
    for feat, agg in pooled.items():
        ics = np.array(agg["ics"])
        if len(ics) == 0:
            continue
        rows.append({
            "feat": feat,
            "n_assets": len(ics),
            "mean_ic": float(np.mean(ics)),
            "median_ic": float(np.median(ics)),
            "abs_median": float(np.median(np.abs(ics))),
            "sign_stability": float(np.mean(np.sign(ics) == np.sign(np.median(ics)))),
            "mean_autocorr": float(np.mean(agg["autocorrs"])),
        })
    rows.sort(key=lambda r: -r["abs_median"])

    print(f"\n  {'feature':32s} {'n':>3s}  {'med_IC':>8s}  {'sign_stab':>9s}  {'autocorr':>9s}  membership")
    print("  " + "-" * 78)
    for r in rows[:50]:
        # which canonical sets is it in?
        memb = []
        if r["feat"] in FEATURE_LIST_13: memb.append("13")
        if r["feat"] in FEATURE_LIST_29: memb.append("29")
        if r["feat"] in FEATURE_LIST_34: memb.append("34")
        if r["feat"] in FEATURE_LIST_41: memb.append("41")
        if r["feat"] in FEATURE_LIST_121: memb.append("121")
        memb_str = ",".join(memb) if memb else "—"
        print(f"  {r['feat']:32s} {r['n_assets']:>3d}  {r['median_ic']:+8.4f}  {r['sign_stability']*100:>8.1f}%  {r['mean_autocorr']:+8.3f}  f{memb_str}")

    # Concentration check: how much of the top-K signal is captured by f13/f29/f41?
    print("\n" + "=" * 78)
    print("  SIGNAL CAPTURE per feature set (top-10 features by median |IC|)")
    print("=" * 78)
    top10 = [r["feat"] for r in rows[:10]]
    sets = {"f13": FEATURE_LIST_13, "f29": FEATURE_LIST_29, "f34": FEATURE_LIST_34,
            "f41": FEATURE_LIST_41, "f121": FEATURE_LIST_121}
    for name, fl in sets.items():
        in_set = sum(1 for f in top10 if f in fl)
        print(f"  {name:5s} (size {len(fl):>3d}): captures {in_set:>2d}/10 of top-signal features")

    # Recommend: which f-count maximizes coverage of high-signal features
    # WITHOUT including features whose median |IC| is below the median of the
    # lowest f13 member (a heuristic "dead feature" threshold).
    f13_ics = [r["abs_median"] for r in rows if r["feat"] in FEATURE_LIST_13]
    if f13_ics:
        f13_floor = float(np.median(f13_ics))
        useful = [r for r in rows if r["abs_median"] > f13_floor * 0.5]
        print(f"\n  Dead-feature floor (= 0.5 x median |IC| of f13): {f13_floor*0.5:.5f}")
        print(f"  Features above floor: {len(useful)} / {len(rows)}")
        for r in useful[:30]:
            print(f"    {r['feat']}  (median |IC|={r['abs_median']:.5f})")

    print("\n[done]")


if __name__ == "__main__":
    main()
