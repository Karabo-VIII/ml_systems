"""
Dataset Inspector -- Comprehensive quality validation for processed chimera parquets.

Combines base feature inspection (OHLCV, bar structure, 34 features, 10 targets)
with cross-asset feature inspection (7 XD features, leakage, multicollinearity)
into a single pass per file.

Outputs:
  - Console: Per-asset quality report (shape, nulls, features, targets, distributions)
  - Plots:   plots/dataset_*_YYYY-MM-DD.png (date-stamped, with --plots flag)

Usage:
    python src/pipeline/inspect_dataset.py                 # Full report
    python src/pipeline/inspect_dataset.py --plots         # + diagnostic plots
    python src/pipeline/inspect_dataset.py --strict        # CI gate mode (exit code)
    python src/pipeline/inspect_dataset.py --strict --plots
"""
import polars as pl
import numpy as np
import sys
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Date subfolder: same day overwrites, different days get new folder
_DATE_SUBDIR = date.today().isoformat()


def _save_plot(fig, name: str):
    """Save plot into date subfolder and close figure."""
    import matplotlib.pyplot as plt
    stem = name.removesuffix(".png")
    out_dir = PLOTS_DIR / _DATE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved: plots/{_DATE_SUBDIR}/{stem}.png")

# 34 base features (V50 pipeline: 13 legacy + 5 extended + 3 tier1 + 4 hawkes + 5 ic-boost + 4 SOTA)
BASE_FEATURES = [
    # Legacy (0-12)
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    # Extended (13-17)
    "norm_ma_distance", "norm_whale", "norm_efficiency",
    "norm_return_4", "norm_return_16",
    # Tier 1 (18-20)
    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
    # Hawkes (21-24)
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # IC-boost Tier 2 (25-29)
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # SOTA Tier 3 (30-33)
    "norm_yz_volatility", "norm_cs_spread",
    "norm_perm_entropy", "norm_kyle_lambda",
]

# 7 cross-asset features
XD_FEATURES = [
    "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
    "xd_cross_return_mean", "xd_cross_vol_mean", "xd_ma_distance",
    "xd_momentum_rank",
]

ALL_FEATURES = BASE_FEATURES + XD_FEATURES

TARGETS = [
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
    "target_return_50", "target_vol_20",
]

VOLADJ_TARGETS = [
    "target_voladj_1", "target_voladj_4", "target_voladj_16", "target_voladj_64",
]

OHLCV_COLS = ["open", "high", "low", "close", "volume", "volume_usd",
              "buy_vol", "sell_vol", "tick_count"]

# XD feature categories
PASS_THROUGH_XD = ["xd_btc_return", "xd_btc_volatility"]
COMPUTED_XD = ["xd_funding_spread", "xd_cross_return_mean", "xd_cross_vol_mean",
               "xd_ma_distance", "xd_momentum_rank"]


def safe_corr(a, b):
    """Pearson correlation that handles zero-variance arrays without RuntimeWarning."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 30:
        return 0.0
    a_std, b_std = np.std(a), np.std(b)
    if a_std < 1e-10 or b_std < 1e-10:
        return 1.0 if a_std < 1e-10 and b_std < 1e-10 else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ts_to_date(ts_ms):
    try:
        return datetime(1970, 1, 1) + timedelta(milliseconds=int(ts_ms))
    except Exception:
        return None


def inspect_dataset(file_path):
    """Inspect a single processed chimera parquet file. Returns comprehensive result dict."""
    # Strip date suffix (e.g. _20260422) and _v50_chimera tag to get symbol
    stem = file_path.stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit() and len(tail) == 8:
            stem = stem.rsplit("_", 1)[0]  # drop date suffix
    sym = stem.replace("_v50_chimera", "").upper()
    result = {
        "symbol": sym, "file": file_path.name,
        "rows": 0, "cols": 0, "size_mb": 0.0,
        # Bar structure
        "date_start": None, "date_end": None, "days_span": 0,
        "bars_per_day": 0.0,
        "bar_id_unique": True, "bar_id_gaps": 0,
        "ohlc_valid": True, "ohlc_errors": 0,
        "ts_format_ok": True,
        # Base features
        "missing_base_features": [], "missing_xd_features": [],
        "missing_targets": [],
        "feature_nulls": {}, "target_nulls": {}, "xd_nulls": {},
        "feature_stats": {},  # mean, std, min, max, zero_pct
        "target_stats": {},
        "xd_stats": {},       # mean, std, min, max, zero_pct, abs_gt_5_pct
        "zero_tail_50": 0,
        "funding_zero_pct": 0.0,
        "oi_active": False, "hurst_active": False,
        # Cross-asset
        "has_xd": False,
        "xd_leakage": {},     # feat -> worst_corr
        "xd_base_corr": {},   # feat -> {max_corr, max_base_feat}
        "btc_self_corr_return": None,
        "btc_self_corr_vol": None,
        "btc_funding_fixed": False,
        # Issues and warnings
        "issues": [], "warnings": [],
        "df": None,
    }

    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"  [FAIL] Could not read {file_path.name}: {e}")
        return result

    result["rows"] = len(df)
    result["cols"] = len(df.columns)
    result["size_mb"] = file_path.stat().st_size / (1024 * 1024)

    if len(df) == 0:
        return result

    result["df"] = df

    # ==== BAR STRUCTURE ====

    # Timestamp range
    if "timestamp" in df.columns:
        ts_min = int(df["timestamp"].min())
        ts_max = int(df["timestamp"].max())
        result["date_start"] = ts_to_date(ts_min)
        result["date_end"] = ts_to_date(ts_max)
        if result["date_start"] and result["date_end"]:
            result["days_span"] = max(1, (result["date_end"] - result["date_start"]).days)
            result["bars_per_day"] = result["rows"] / result["days_span"]
        result["ts_format_ok"] = 1_000_000_000_000 < ts_min < 9_000_000_000_000

    # Bar ID uniqueness
    if "bar_id" in df.columns:
        n_unique = df["bar_id"].n_unique()
        result["bar_id_unique"] = (n_unique == len(df))
        if not result["bar_id_unique"]:
            result["bar_id_gaps"] = len(df) - n_unique
        bar_ids = df["bar_id"].to_numpy()
        if len(bar_ids) > 1:
            diffs = np.diff(bar_ids)
            result["bar_id_gaps"] = int(np.sum(diffs != 1))

    # OHLC integrity
    bad_ohlc = df.filter(pl.col("high") < pl.col("low")).height
    result["ohlc_errors"] = bad_ohlc
    result["ohlc_valid"] = (bad_ohlc == 0)

    # ==== BASE FEATURES (34) ====

    result["missing_base_features"] = [f for f in BASE_FEATURES if f not in df.columns]
    result["missing_targets"] = [t for t in TARGETS if t not in df.columns]

    for f in BASE_FEATURES:
        if f in df.columns:
            col = df[f]
            total = len(col)
            zero_count = col.filter(col == 0.0).len()
            result["feature_nulls"][f] = col.null_count()
            result["feature_stats"][f] = {
                "mean": float(col.mean()) if col.mean() is not None else 0.0,
                "std": float(col.std()) if col.std() is not None else 0.0,
                "min": float(col.min()) if col.min() is not None else 0.0,
                "max": float(col.max()) if col.max() is not None else 0.0,
                "zero_pct": (zero_count / total * 100) if total > 0 else 0.0,
                "null_pct": (col.null_count() / total * 100) if total > 0 else 0.0,
            }

    # Target statistics
    for t in TARGETS:
        if t in df.columns:
            col = df[t]
            total = len(col)
            zero_count = col.filter(col == 0.0).len()
            result["target_nulls"][t] = col.null_count()
            result["target_stats"][t] = {
                "mean": float(col.mean()) if col.mean() is not None else 0.0,
                "std": float(col.std()) if col.std() is not None else 0.0,
                "min": float(col.min()) if col.min() is not None else 0.0,
                "max": float(col.max()) if col.max() is not None else 0.0,
                "zero_pct": (zero_count / total * 100) if total > 0 else 0.0,
            }

    # Vol-adjusted target statistics
    result["voladj_stats"] = {}
    result["missing_voladj_targets"] = [t for t in VOLADJ_TARGETS if t not in df.columns]
    result["has_voladj"] = len(result["missing_voladj_targets"]) == 0
    for t in VOLADJ_TARGETS:
        if t in df.columns:
            col = df[t]
            total = len(col)
            zero_count = col.filter(col == 0.0).len()
            result["voladj_stats"][t] = {
                "mean": float(col.mean()) if col.mean() is not None else 0.0,
                "std": float(col.std()) if col.std() is not None else 0.0,
                "min": float(col.min()) if col.min() is not None else 0.0,
                "max": float(col.max()) if col.max() is not None else 0.0,
                "zero_pct": (zero_count / total * 100) if total > 0 else 0.0,
                "null_pct": (col.null_count() / total * 100) if total > 0 else 0.0,
            }

    # Zero tail check for target_return_50
    if "target_return_50" in df.columns:
        tail = df["target_return_50"].tail(100).to_numpy()
        result["zero_tail_50"] = int(np.sum(np.abs(tail) < 1e-9))

    # Funding zero percentage
    if "norm_funding" in df.columns:
        fz = df["norm_funding"].filter(df["norm_funding"] == 0.0).len()
        result["funding_zero_pct"] = (fz / len(df)) * 100

    # OI activity check
    if "norm_oi_change" in df.columns:
        oi_std = df["norm_oi_change"].std()
        result["oi_active"] = (oi_std is not None and oi_std > 0.01)

    # Hurst activity check
    if "hurst_regime" in df.columns:
        h_mean = df["hurst_regime"].mean()
        h_std = df["hurst_regime"].std()
        result["hurst_active"] = not (0.49 < (h_mean or 0.5) < 0.51 and (h_std or 0) < 0.01)

    # ==== REGIME LABEL ====
    if "regime_label" in df.columns:
        regime_counts = df["regime_label"].value_counts().sort("regime_label")
        total = len(df)
        regime_dist = {}
        for row in regime_counts.iter_rows():
            label, count = row[0], row[1]
            name = {0: "bear", 1: "neutral", 2: "bull"}.get(label, f"unknown({label})")
            regime_dist[name] = f"{count:,} ({count/total*100:.1f}%)"
        result["regime_dist"] = regime_dist
    else:
        result["regime_dist"] = {}

    # ==== XD FEATURES (7) ====

    result["missing_xd_features"] = [f for f in XD_FEATURES if f not in df.columns]
    result["has_xd"] = len(result["missing_xd_features"]) == 0

    for feat in XD_FEATURES:
        if feat not in df.columns:
            continue
        s = df[feat]
        total = len(s)
        result["xd_nulls"][feat] = s.null_count()
        result["xd_stats"][feat] = {
            "mean": float(s.mean()) if s.mean() is not None else None,
            "std": float(s.std()) if s.std() is not None else None,
            "min": float(s.min()) if s.min() is not None else None,
            "max": float(s.max()) if s.max() is not None else None,
            "zero_pct": float((s == 0.0).sum() / total * 100),
            "abs_gt_5_pct": float((s.abs() > 5.0).sum() / total * 100),
        }

        std = result["xd_stats"][feat]["std"]

        # Check pass-through features
        if feat in PASS_THROUGH_XD:
            if std is not None and (std < 0.3 or std > 3.0):
                result["warnings"].append(f"{feat}: std={std:.3f} (pass-through, expected ~1.0)")
        elif feat in COMPUTED_XD:
            if std is not None and (std < 0.3 or std > 3.0):
                result["warnings"].append(f"{feat}: std={std:.3f} (z-scored, expected ~1.0)")

        # Dead feature detection
        if std is not None and std < 0.01:
            result["issues"].append(f"{feat}: DEAD FEATURE (std={std:.6f})")

        if s.null_count() > 0:
            result["issues"].append(f"{feat}: {s.null_count():,} null values")

    # BTC self-reference test
    is_btc = "BTC" in sym
    if is_btc and "xd_btc_return" in df.columns and "norm_return_1" in df.columns:
        corr = safe_corr(
            df["xd_btc_return"].to_numpy()[:10000],
            df["norm_return_1"].to_numpy()[:10000],
        )
        result["btc_self_corr_return"] = corr

    if is_btc and "xd_btc_volatility" in df.columns and "norm_hl_spread" in df.columns:
        corr = safe_corr(
            df["xd_btc_volatility"].to_numpy()[:10000],
            df["norm_hl_spread"].to_numpy()[:10000],
        )
        result["btc_self_corr_vol"] = corr

    # BTC funding spread fix validation
    if is_btc and "xd_funding_spread" in df.columns:
        fund_std = float(df["xd_funding_spread"].std()) if df["xd_funding_spread"].std() is not None else 0.0
        if fund_std < 0.1:
            result["issues"].append(
                f"BTC xd_funding_spread still near-constant: std={fund_std:.6f}"
            )
        else:
            result["btc_funding_fixed"] = True

    # Leakage test (stratified: start, middle, end)
    if "target_return_1" in df.columns and result["has_xd"]:
        target_full = df["target_return_1"].to_numpy()
        n = len(target_full)
        sample_size = min(50000, n // 3)

        samples = [
            ("start", 0, sample_size),
            ("middle", n // 2 - sample_size // 2, n // 2 + sample_size // 2),
            ("end", n - sample_size, n),
        ]

        for feat in XD_FEATURES:
            if feat not in df.columns:
                continue
            feat_full = df[feat].to_numpy()
            worst_corr = 0.0
            worst_region = ""

            for region_name, start, end in samples:
                fv = feat_full[start:end]
                tv = target_full[start:end]
                corr = safe_corr(fv, tv)
                if abs(corr) > abs(worst_corr):
                    worst_corr = corr
                    worst_region = region_name

            result["xd_leakage"][feat] = worst_corr
            if abs(worst_corr) > 0.10:
                result["issues"].append(
                    f"POTENTIAL LEAKAGE: {feat} -> target_return_1 corr={worst_corr:.4f} "
                    f"(worst region: {worst_region})"
                )
            elif abs(worst_corr) > 0.05:
                result["warnings"].append(
                    f"Moderate XD->target corr: {feat} -> target_return_1 corr={worst_corr:.4f} "
                    f"(worst region: {worst_region})"
                )

    # Multicollinearity: XD vs base features
    if result["has_xd"]:
        for xd_feat in XD_FEATURES:
            if xd_feat not in df.columns:
                continue
            max_corr = 0.0
            max_base = ""
            n_corr = min(50000, len(df))
            xd_vals = df[xd_feat].to_numpy()[:n_corr]

            for base_feat in BASE_FEATURES:
                if base_feat not in df.columns:
                    continue
                base_vals = df[base_feat].to_numpy()[:n_corr]
                corr = abs(safe_corr(xd_vals, base_vals))
                if corr > max_corr:
                    max_corr = corr
                    max_base = base_feat

            result["xd_base_corr"][xd_feat] = {"max_corr": max_corr, "max_base_feat": max_base}
            if max_corr > 0.90:
                result["warnings"].append(
                    f"High multicollinearity: {xd_feat} vs {max_base} corr={max_corr:.4f}"
                )

    # ========================================================================
    # ENHANCED VALIDATION (P7/P9 upgrades)
    # ========================================================================

    n_sample = min(200_000, len(df))  # P7: 200K sample (was 50K)
    result["enhanced"] = {}

    # (A) All-feature leakage: base features vs ALL target horizons
    all_horizons = ["target_return_1", "target_return_4", "target_return_16", "target_return_64"]
    all_features_for_leak = [f for f in BASE_FEATURES + XD_FEATURES if f in df.columns]
    horizon_targets = {h: df[h].to_numpy()[:n_sample] for h in all_horizons if h in df.columns}

    leak_matrix = {}  # {feat: {horizon: corr}}
    for feat in all_features_for_leak:
        feat_vals = df[feat].to_numpy()[:n_sample]
        leak_matrix[feat] = {}
        for h_name, h_vals in horizon_targets.items():
            corr = safe_corr(feat_vals, h_vals)
            leak_matrix[feat][h_name] = corr
            if abs(corr) > 0.15:
                result["issues"].append(
                    f"LEAKAGE: {feat} -> {h_name} corr={corr:.4f}"
                )
    result["enhanced"]["leak_matrix"] = leak_matrix

    # (B) Autocorrelation (lag-1) per feature -- high autocorrelation = memorization risk
    acf_lag1 = {}
    for feat in all_features_for_leak[:30]:  # base features only (speed)
        vals = df[feat].to_numpy()
        if len(vals) > 100:
            c = safe_corr(vals[1:n_sample], vals[:n_sample-1])
            acf_lag1[feat] = c
            if abs(c) > 0.95:
                result["warnings"].append(
                    f"Very high autocorrelation: {feat} lag-1 ACF={c:.4f} (memorization risk)"
                )
    result["enhanced"]["acf_lag1"] = acf_lag1

    # (C) Base-to-base feature correlation matrix (top pairs)
    base_present = [f for f in BASE_FEATURES if f in df.columns]
    high_corr_pairs = []
    for i, f1 in enumerate(base_present):
        v1 = df[f1].to_numpy()[:n_sample].astype(np.float64)
        v1_std = np.std(v1[np.isfinite(v1)])
        if v1_std < 1e-10:
            continue  # Skip dead features (all zeros/constant)
        for f2 in base_present[i+1:]:
            v2 = df[f2].to_numpy()[:n_sample].astype(np.float64)
            v2_std = np.std(v2[np.isfinite(v2)])
            if v2_std < 1e-10:
                continue  # Skip dead features
            c = abs(safe_corr(v1, v2))
            if c > 0.70:
                high_corr_pairs.append((f1, f2, c))
    high_corr_pairs.sort(key=lambda x: -x[2])
    result["enhanced"]["high_corr_pairs"] = high_corr_pairs[:10]
    for f1, f2, c in high_corr_pairs:
        if c > 0.90:
            result["warnings"].append(f"Redundant features: {f1} vs {f2} corr={c:.4f}")

    # (D) Regime balance check (P9)
    if "regime_label" in df.columns:
        regime_counts = df["regime_label"].value_counts().sort("regime_label")
        regime_dict = {int(row[0]): int(row[1]) for row in regime_counts.iter_rows()}
        total = sum(regime_dict.values())
        result["enhanced"]["regime_balance"] = {
            k: round(v / total * 100, 1) for k, v in regime_dict.items()
        }
        # Flag if any regime > 80% or < 5%
        for regime_id, pct in result["enhanced"]["regime_balance"].items():
            if pct > 80:
                result["warnings"].append(
                    f"Regime imbalance: regime {regime_id} is {pct}% of data"
                )
            elif pct < 5:
                result["warnings"].append(
                    f"Rare regime: regime {regime_id} is only {pct}% of data"
                )

    # (E) Feature importance: correlation with target_return_1 (sorted)
    if "target_return_1" in df.columns:
        t1 = df["target_return_1"].to_numpy()[:n_sample]
        importance = {}
        for feat in all_features_for_leak:
            fv = df[feat].to_numpy()[:n_sample]
            importance[feat] = safe_corr(fv, t1)
        # Sort by absolute correlation
        importance = dict(sorted(importance.items(), key=lambda x: -abs(x[1])))
        result["enhanced"]["feature_importance_h1"] = importance

    return result


def print_summary(results):
    """Print comprehensive console summary."""
    print("=" * 110)
    print("  DATASET INSPECTION REPORT (V4 SOTA -- Base + Cross-Asset)")
    print("=" * 110)

    # Overview table
    print(f"\n  {'Asset':<12} {'Bars':>12} {'Cols':>6} {'Size(MB)':>10} "
          f"{'Days':>6} {'Bars/Day':>10} {'Date Start':>12} {'Date End':>12} "
          f"{'BarID OK':>9} {'OHLC OK':>8} {'XD':>4}")
    print(f"  {'-'*114}")

    total_bars = 0
    total_size = 0.0
    for r in results:
        total_bars += r["rows"]
        total_size += r["size_mb"]
        d_start = r["date_start"].strftime("%Y-%m-%d") if r["date_start"] else "N/A"
        d_end = r["date_end"].strftime("%Y-%m-%d") if r["date_end"] else "N/A"
        bid_ok = "YES" if r["bar_id_unique"] else f"NO ({r['bar_id_gaps']} gaps)"
        ohlc_ok = "YES" if r["ohlc_valid"] else f"NO ({r['ohlc_errors']})"
        n_xd = len(XD_FEATURES)
        xd_ok = f"{n_xd}/{n_xd}" if r["has_xd"] else f"{n_xd-len(r['missing_xd_features'])}/{n_xd}"

        print(f"  {r['symbol']:<12} {r['rows']:>12,} {r['cols']:>6} {r['size_mb']:>10.1f} "
              f"{r['days_span']:>6} {r['bars_per_day']:>10.1f} {d_start:>12} {d_end:>12} "
              f"{bid_ok:>9} {ohlc_ok:>8} {xd_ok:>4}")

    print(f"  {'-'*114}")
    print(f"  {'TOTAL':<12} {total_bars:>12,} {'':>6} {total_size:>10.1f}")

    # Base feature health
    print(f"\n  BASE FEATURE HEALTH")
    print(f"  {'Feature':<22} ", end="")
    for r in results:
        print(f"{r['symbol']:>14}", end="")
    print()
    print(f"  {'-'*22} " + "-" * 14 * len(results))

    for f in BASE_FEATURES:
        print(f"  {f:<22} ", end="")
        for r in results:
            stats = r["feature_stats"].get(f)
            if stats is None:
                cell = "MISSING"
            elif stats["null_pct"] > 1:
                cell = "NULL {:.0f}%".format(stats["null_pct"])
            elif stats["std"] < 0.001:
                cell = "DEAD"
            else:
                cell = "OK (s={:.2f})".format(stats["std"])
            print(f"{cell:>14}", end="")
        print()

    # XD feature health
    if any(r["has_xd"] for r in results):
        print(f"\n  CROSS-ASSET FEATURE HEALTH (XD)")
        print(f"  {'Feature':<24} ", end="")
        for r in results:
            print(f"{r['symbol']:>14}", end="")
        print()
        print(f"  {'-'*24} " + "-" * 14 * len(results))

        for feat in XD_FEATURES:
            # Std row
            print(f"  {feat:<24} ", end="")
            for r in results:
                stats = r["xd_stats"].get(feat)
                if stats is None:
                    cell = "MISSING"
                elif stats["std"] is not None and stats["std"] < 0.01:
                    cell = "DEAD"
                elif stats["std"] is not None:
                    cell = f"s={stats['std']:.3f}"
                else:
                    cell = "N/A"
                print(f"{cell:>14}", end="")
            print()

    # Target health
    print(f"\n  TARGET HEALTH")
    print(f"  {'Target':<22} ", end="")
    for r in results:
        print(f"{r['symbol']:>14}", end="")
    print()
    print(f"  {'-'*22} " + "-" * 14 * len(results))

    for t in TARGETS:
        print(f"  {t:<22} ", end="")
        for r in results:
            stats = r["target_stats"].get(t)
            if stats is None:
                cell = "MISSING"
            elif stats["zero_pct"] > 50:
                cell = "WARN {:.0f}%z".format(stats["zero_pct"])
            else:
                cell = "OK (s={:.3f})".format(stats["std"])
            print(f"{cell:>14}", end="")
        print()

    # Vol-adjusted target health
    if any(r.get("has_voladj") for r in results):
        print(f"\n  VOL-ADJUSTED TARGET HEALTH")
        print(f"  {'Target':<22} ", end="")
        for r in results:
            print(f"{r['symbol']:>14}", end="")
        print()
        print(f"  {'-'*22} " + "-" * 14 * len(results))

        for t in VOLADJ_TARGETS:
            print(f"  {t:<22} ", end="")
            for r in results:
                stats = r.get("voladj_stats", {}).get(t)
                if stats is None:
                    cell = "MISSING"
                elif stats.get("null_pct", 0) > 1:
                    cell = "NULL {:.0f}%".format(stats["null_pct"])
                else:
                    cell = "OK (s={:.3f})".format(stats["std"])
                print(f"{cell:>14}", end="")
            print()

    # Diagnostic checks
    print(f"\n  DIAGNOSTIC CHECKS")
    print(f"  {'Check':<30} ", end="")
    for r in results:
        print(f"{r['symbol']:>14}", end="")
    print()
    print(f"  {'-'*30} " + "-" * 14 * len(results))

    checks = [
        ("Timestamp Format (13-digit)", lambda r: "OK" if r["ts_format_ok"] else "BAD"),
        ("Bar ID Unique", lambda r: "OK" if r["bar_id_unique"] else f"FAIL({r['bar_id_gaps']})"),
        ("OHLC Geometry", lambda r: "OK" if r["ohlc_valid"] else f"FAIL({r['ohlc_errors']})"),
        ("Hurst Active", lambda r: "YES" if r["hurst_active"] else "DEAD"),
        ("OI Change Active", lambda r: "YES" if r["oi_active"] else "ZERO-FILL"),
        ("Funding Zero %", lambda r: f"{r['funding_zero_pct']:.1f}%"),
        ("Tail Corruption (last 100)", lambda r: f"{r['zero_tail_50']}/100 zeros"),
        ("XD Features Present", lambda r: f"{len(XD_FEATURES)}/{len(XD_FEATURES)}" if r["has_xd"] else f"{len(XD_FEATURES)-len(r['missing_xd_features'])}/{len(XD_FEATURES)}"),
        ("Regime Bear %", lambda r: r.get("regime_dist", {}).get("bear", "N/A")),
        ("Regime Neutral %", lambda r: r.get("regime_dist", {}).get("neutral", "N/A")),
        ("Regime Bull %", lambda r: r.get("regime_dist", {}).get("bull", "N/A")),
    ]

    for label, fn in checks:
        print(f"  {label:<30} ", end="")
        for r in results:
            print(f"{fn(r):>14}", end="")
        print()

    # Leakage test (XD -> target)
    if any(r["xd_leakage"] for r in results):
        print(f"\n  LEAKAGE TEST (XD feature -> target_return_1 correlation)")
        print(f"  {'Feature':<24} ", end="")
        for r in results:
            print(f"{r['symbol']:>14}", end="")
        print()
        print(f"  {'-'*24} " + "-" * 14 * len(results))

        for feat in XD_FEATURES:
            print(f"  {feat:<24} ", end="")
            for r in results:
                leak = r["xd_leakage"].get(feat)
                if leak is not None:
                    flag = "!!!" if abs(leak) > 0.10 else "?" if abs(leak) > 0.05 else ""
                    cell = f"{leak:+.4f}{flag}"
                else:
                    cell = "N/A"
                print(f"{cell:>14}", end="")
            print()

    # Multicollinearity
    if any(r["xd_base_corr"] for r in results):
        print(f"\n  MULTICOLLINEARITY (max |corr| of XD vs any base feature)")
        print(f"  {'Feature':<24} ", end="")
        for r in results:
            print(f"{r['symbol']:>14}", end="")
        print()
        print(f"  {'-'*24} " + "-" * 14 * len(results))

        for feat in XD_FEATURES:
            print(f"  {feat:<24} ", end="")
            for r in results:
                xbc = r["xd_base_corr"].get(feat, {})
                mc = xbc.get("max_corr")
                if mc is not None:
                    flag = "!!!" if mc > 0.90 else "!" if mc > 0.70 else ""
                    cell = f"{mc:.4f}{flag}"
                else:
                    cell = "N/A"
                print(f"{cell:>14}", end="")
            print()

    # Enhanced validation results
    if any(r.get("enhanced") for r in results):
        # Regime balance
        has_regime = any(r.get("enhanced", {}).get("regime_balance") for r in results)
        if has_regime:
            print(f"\n  REGIME BALANCE (P9)")
            print(f"  {'Regime':<12} ", end="")
            for r in results:
                print(f"{r['symbol']:>10}", end="")
            print()
            for regime_id, label in [(0, "Bear"), (1, "Neutral"), (2, "Bull")]:
                print(f"  {label:<12} ", end="")
                for r in results:
                    rb = r.get("enhanced", {}).get("regime_balance", {})
                    pct = rb.get(regime_id, 0)
                    flag = "!" if pct > 80 or pct < 5 else ""
                    print(f"{pct:>8.1f}%{flag}", end="")
                print()

        # Feature importance (top 10, first asset only for brevity)
        first_enh = next((r.get("enhanced", {}) for r in results if r.get("enhanced")), {})
        importance = first_enh.get("feature_importance_h1", {})
        if importance:
            print(f"\n  FEATURE IMPORTANCE (corr with target_return_1, {results[0]['symbol']})")
            for i, (feat, corr) in enumerate(importance.items()):
                if i >= 15:
                    break
                bar = "+" * int(abs(corr) * 200) if abs(corr) > 0.005 else "."
                print(f"    {feat:<30} {corr:+.4f}  {bar}")

        # High-correlation feature pairs
        for r in results[:1]:  # first asset only
            pairs = r.get("enhanced", {}).get("high_corr_pairs", [])
            if pairs:
                print(f"\n  HIGH-CORRELATION FEATURE PAIRS ({r['symbol']}, |corr| > 0.70)")
                for f1, f2, c in pairs[:8]:
                    flag = " REDUNDANT!" if c > 0.90 else ""
                    print(f"    {f1:<28} vs {f2:<28} corr={c:.4f}{flag}")

        # Autocorrelation summary
        for r in results[:1]:
            acf = r.get("enhanced", {}).get("acf_lag1", {})
            if acf:
                high_acf = [(f, c) for f, c in acf.items() if abs(c) > 0.80]
                if high_acf:
                    print(f"\n  HIGH AUTOCORRELATION (lag-1 ACF > 0.80, {r['symbol']})")
                    for feat, c in sorted(high_acf, key=lambda x: -abs(x[1])):
                        print(f"    {feat:<30} ACF={c:.4f}")

    # Issues and warnings
    total_issues = sum(len(r["issues"]) for r in results)
    total_warnings = sum(len(r["warnings"]) for r in results)

    if total_issues > 0:
        print(f"\n  [ISSUES] ({total_issues} total)")
        for r in results:
            for issue in r["issues"]:
                print(f"    [{r['symbol']}] {issue}")

    if total_warnings > 0:
        print(f"\n  [WARNINGS] ({total_warnings} total)")
        for r in results:
            for warn in r["warnings"]:
                print(f"    [{r['symbol']}] {warn}")

    print(f"\n  {'='*60}")
    if total_issues == 0:
        print(f"  [PASS] All {len(results)} assets passed dataset inspection")
    else:
        print(f"  [FAIL] {total_issues} issues found across {len(results)} assets")
    print(f"  Warnings: {total_warnings}")
    print(f"  {'='*60}")


def generate_plots(results):
    """Generate comprehensive diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    valid = [r for r in results if r["df"] is not None and r["rows"] > 0]
    if not valid:
        print("  [WARN] No valid datasets for plotting")
        return

    symbols = [r["symbol"] for r in valid]
    n_assets = len(symbols)
    colors = plt.cm.Set2(np.linspace(0, 1, n_assets))

    # ---- Plot 1: Bar count comparison ----
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = [r["rows"] for r in valid]
    bar_plot = ax.bar(symbols, [b / 1e6 for b in bars], color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
    for i, (b, v) in enumerate(zip(bar_plot, bars)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Bars (Millions)")
    ax.set_title("Dollar Bars: Bar Count per Asset")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, "dataset_bar_count.png")

    # ---- Plot 2: Bars per day distribution ----
    fig, axes = plt.subplots(2, (n_assets + 1) // 2, figsize=(16, 8), squeeze=False)
    axes_flat = axes.flatten()
    for i, r in enumerate(valid):
        ax = axes_flat[i]
        df = r["df"]
        if "timestamp" not in df.columns:
            continue
        ts = df["timestamp"].to_numpy()
        days = (ts - ts[0]) // (86400 * 1000)
        _, counts = np.unique(days, return_counts=True)
        ax.hist(counts, bins=50, color=colors[i], alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(np.median(counts), color="red", linestyle="--", linewidth=1.5,
                   label=f"Median: {np.median(counts):.0f}")
        ax.axvline(288, color="green", linestyle=":", linewidth=1.5, label="Target: 288")
        ax.set_title(f"{r['symbol']}", fontsize=10)
        ax.set_xlabel("Bars/Day")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(n_assets, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Dollar Bars: Bars-per-Day Distribution (Target: 288 = ~5min)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "dataset_bars_per_day.png")

    # ---- Plot 3: Base feature distribution heatmap (mean/std) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(5, len(BASE_FEATURES) * 0.45)))
    means = np.zeros((len(BASE_FEATURES), n_assets))
    stds = np.zeros((len(BASE_FEATURES), n_assets))
    for j, r in enumerate(valid):
        for i, f in enumerate(BASE_FEATURES):
            stats = r["feature_stats"].get(f, {})
            means[i, j] = stats.get("mean", 0)
            stds[i, j] = stats.get("std", 0)

    im1 = ax1.imshow(means, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax1.set_xticks(range(n_assets))
    ax1.set_xticklabels(symbols, rotation=45, ha="right")
    ax1.set_yticks(range(len(BASE_FEATURES)))
    ax1.set_yticklabels(BASE_FEATURES, fontsize=8)
    ax1.set_title("Feature Means (should be ~0)")
    fig.colorbar(im1, ax=ax1, shrink=0.8)
    for i in range(len(BASE_FEATURES)):
        for j in range(n_assets):
            ax1.text(j, i, f"{means[i,j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(means[i,j]) > 0.5 else "black")

    im2 = ax2.imshow(stds, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
    ax2.set_xticks(range(n_assets))
    ax2.set_xticklabels(symbols, rotation=45, ha="right")
    ax2.set_yticks(range(len(BASE_FEATURES)))
    ax2.set_yticklabels(BASE_FEATURES, fontsize=8)
    ax2.set_title("Feature Std Devs (should be ~1)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    for i in range(len(BASE_FEATURES)):
        for j in range(n_assets):
            ax2.text(j, i, f"{stds[i,j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if stds[i,j] > 2 else "black")

    fig.suptitle("Base Feature Distribution Health (Rolling Z-Score Normalized)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "dataset_base_feature_heatmap.png")

    # ---- Plot 4: Target distribution histograms ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for idx, t in enumerate(TARGETS):
        ax = axes_flat[idx]
        for j, r in enumerate(valid):
            if t in r["df"].columns:
                data = r["df"][t].drop_nulls().to_numpy()
                if len(data) > 50000:
                    data = np.random.choice(data, 50000, replace=False)
                ax.hist(data, bins=100, alpha=0.4, label=r["symbol"], density=True)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("Target Return Distributions (All Assets)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "dataset_target_distributions.png")

    # ---- Plot 5: Price series (close) for all assets ----
    fig, axes = plt.subplots(n_assets, 1, figsize=(16, 3 * n_assets), sharex=False)
    if n_assets == 1:
        axes = [axes]
    for i, r in enumerate(valid):
        ax = axes[i]
        df = r["df"]
        if "timestamp" not in df.columns or "close" not in df.columns:
            continue
        step = max(1, len(df) // 5000)
        ts = df["timestamp"].gather_every(step).to_numpy()
        close = df["close"].gather_every(step).to_numpy()
        dates = [datetime(1970, 1, 1) + timedelta(milliseconds=int(t)) for t in ts]
        ax.plot(dates, close, linewidth=0.5, color=colors[i])
        ax.set_ylabel("Close ($)")
        ax.set_title(f"{r['symbol']} ({r['rows']:,} bars)", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Dollar Bar Price Series (All Assets)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "dataset_price_series.png")

    # ---- Plot 6: Feature time-series sample (first asset) ----
    sample_r = valid[0]
    df_sample = sample_r["df"]
    present_features = [f for f in BASE_FEATURES if f in df_sample.columns]
    n_feat = len(present_features)
    fig, axes = plt.subplots(n_feat, 1, figsize=(16, 2.5 * n_feat), sharex=True)
    if n_feat == 1:
        axes = [axes]
    step = max(1, len(df_sample) // 3000)
    ts_arr = df_sample["timestamp"].gather_every(step).to_numpy()
    dates = [datetime(1970, 1, 1) + timedelta(milliseconds=int(t)) for t in ts_arr]
    for j, feat in enumerate(present_features):
        ax = axes[j]
        vals = df_sample[feat].gather_every(step).to_numpy()
        ax.plot(dates, vals, linewidth=0.4, color="#1976D2")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(feat, fontsize=7, rotation=0, ha="right", labelpad=80)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Base Feature Time Series: {sample_r['symbol']} (sampled)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "dataset_base_feature_timeseries.png")

    # ---- Plot 7: Correlation matrix of base features (first asset) ----
    if len(present_features) > 1:
        feat_data = df_sample.select(present_features).to_numpy()
        if feat_data.shape[0] > 100000:
            idx = np.random.choice(feat_data.shape[0], 100000, replace=False)
            feat_data = feat_data[idx]
        corr = np.corrcoef(feat_data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(present_features, rotation=90, fontsize=7)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(present_features, fontsize=7)
        ax.set_title(f"Base Feature Correlation Matrix: {sample_r['symbol']}")
        for ii in range(n_feat):
            for jj in range(n_feat):
                color = "white" if abs(corr[ii, jj]) > 0.6 else "black"
                ax.text(jj, ii, f"{corr[ii,jj]:.2f}", ha="center", va="center", fontsize=5, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        _save_plot(fig, "dataset_base_feature_correlation.png")

    # ---- Plot 8: Volume USD over time ----
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, r in enumerate(valid):
        df = r["df"]
        if "timestamp" not in df.columns or "volume_usd" not in df.columns:
            continue
        step = max(1, len(df) // 3000)
        ts = df["timestamp"].gather_every(step).to_numpy()
        vol = df["volume_usd"].gather_every(step).to_numpy()
        dates = [datetime(1970, 1, 1) + timedelta(milliseconds=int(t)) for t in ts]
        window = min(50, len(vol) // 10) if len(vol) > 100 else 1
        if window > 1:
            vol_smooth = np.convolve(vol, np.ones(window)/window, mode="valid")
            dates_smooth = dates[:len(vol_smooth)]
        else:
            vol_smooth = vol
            dates_smooth = dates
        ax.plot(dates_smooth, vol_smooth / 1e6, linewidth=0.8, label=r["symbol"], alpha=0.8)
    ax.set_ylabel("Volume USD per Bar (M$, smoothed)")
    ax.set_title("Dollar Bar Volume Over Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, "dataset_volume_timeseries.png")

    # ---- Plot 9: Null/Zero summary heatmap ----
    all_cols = BASE_FEATURES + TARGETS
    zero_matrix = np.zeros((len(all_cols), n_assets))
    for j, r in enumerate(valid):
        for i, col in enumerate(all_cols):
            if col in BASE_FEATURES:
                stats = r["feature_stats"].get(col, {})
            else:
                stats = r["target_stats"].get(col, {})
            zero_matrix[i, j] = stats.get("zero_pct", 100.0)

    fig, ax = plt.subplots(figsize=(max(8, n_assets * 1.5), max(8, len(all_cols) * 0.4)))
    im = ax.imshow(zero_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=50)
    ax.set_xticks(range(n_assets))
    ax.set_xticklabels(symbols, rotation=45, ha="right")
    ax.set_yticks(range(len(all_cols)))
    ax.set_yticklabels(all_cols, fontsize=7)
    ax.set_title("Zero-Value Percentage (Base Features + Targets)")
    for i in range(len(all_cols)):
        for j in range(n_assets):
            val = zero_matrix[i, j]
            color = "white" if val > 25 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, label="Zero %", shrink=0.7)
    fig.tight_layout()
    _save_plot(fig, "dataset_zero_heatmap.png")

    # ---- Plot 10: XD feature std across assets ----
    if any(r["has_xd"] for r in valid):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_assets)
        width = 0.15
        for k, feat in enumerate(XD_FEATURES):
            xd_stds = [r["xd_stats"].get(feat, {}).get("std", 0) or 0 for r in valid]
            ax.bar(x + k * width, xd_stds, width, alpha=0.7, label=feat)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="target std=1.0")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(symbols, rotation=45, ha="right")
        ax.set_ylabel("Standard Deviation")
        ax.set_title("XD Feature Standard Deviations by Asset")
        ax.legend(fontsize=7)
        fig.tight_layout()
        _save_plot(fig, "dataset_xd_feature_stds.png")

    # ---- Plot 11: XD leakage correlation heatmap ----
    if any(r["xd_leakage"] for r in valid):
        leak_matrix = []
        for r in valid:
            row = [r["xd_leakage"].get(f, 0) for f in XD_FEATURES]
            leak_matrix.append(row)
        leak_matrix = np.array(leak_matrix)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(leak_matrix.T, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(n_assets))
        ax.set_xticklabels(symbols, rotation=45)
        ax.set_yticks(range(len(XD_FEATURES)))
        ax.set_yticklabels(XD_FEATURES)
        for i in range(len(XD_FEATURES)):
            for j in range(n_assets):
                ax.text(j, i, f"{leak_matrix[j, i]:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, label="Correlation with target_return_1")
        ax.set_title("Leakage Test: XD Feature -> Future Return Correlation")
        fig.tight_layout()
        _save_plot(fig, "dataset_xd_leakage_heatmap.png")

    # ---- Plot 12: XD feature time-series sample (first asset) ----
    if valid[0]["has_xd"]:
        df_xd = valid[0]["df"]
        present_xd = [f for f in XD_FEATURES if f in df_xd.columns]
        if present_xd:
            fig, axes = plt.subplots(len(present_xd), 1, figsize=(14, 2.5 * len(present_xd)), sharex=True)
            if len(present_xd) == 1:
                axes = [axes]
            n_plot = min(5000, len(df_xd))
            for i, feat in enumerate(present_xd):
                vals = df_xd[feat].to_numpy()[:n_plot]
                axes[i].plot(vals, linewidth=0.5, alpha=0.8)
                axes[i].set_ylabel(feat.replace("xd_", ""), fontsize=8)
                axes[i].axhline(y=0, color="gray", linestyle="--", alpha=0.3)
                axes[i].set_ylim(-5, 5)
            axes[0].set_title(f"XD Feature Time-Series ({valid[0]['symbol']}, first {n_plot:,} bars)")
            axes[-1].set_xlabel("Bar Index")
            fig.tight_layout()
            _save_plot(fig, "dataset_xd_feature_timeseries.png")

    print(f"\n  [OK] All plots saved to {PLOTS_DIR / _DATE_SUBDIR}/")


def run_strict_gates(results):
    """
    Run all CLAUDE.md invariant checks as hard gates. Returns (all_pass, failures).

    Gates (13 total):
      Base (7):
        1. Timestamp is 13-digit milliseconds
        2. bar_id is unique
        3. OHLC geometry valid (high >= low)
        4. All 34 base features present
        5. All 6 targets present
        6. Base feature std > 0.01 (no dead features)
        7. Target tail: <10 zeros in last 100 rows of target_return_50
      Cross-Asset (4):
        8. All 7 XD features present
        9. No null values in XD features
       10. No dead XD features (std < 0.01)
       11. No leakage (|corr with target_return_1| < 0.10)
      Enhanced (2):
       12. No base feature -> any horizon leakage (|corr| < 0.15)
       13. No extreme regime imbalance (no regime > 90%)
    """
    failures = []
    for r in results:
        sym = r["symbol"]

        # Gate 1: Timestamp format
        if not r["ts_format_ok"]:
            failures.append(f"{sym}: Timestamp not 13-digit milliseconds")

        # Gate 2: Bar ID unique
        if not r["bar_id_unique"]:
            failures.append(f"{sym}: bar_id not unique ({r['bar_id_gaps']} gaps/dupes)")

        # Gate 3: OHLC geometry
        if not r["ohlc_valid"]:
            failures.append(f"{sym}: OHLC geometry invalid ({r['ohlc_errors']} bars)")

        # Gate 4: All 34 base features present
        if r["missing_base_features"]:
            failures.append(f"{sym}: Missing base features: {r['missing_base_features']}")

        # Gate 5: All 6 targets present
        if r["missing_targets"]:
            failures.append(f"{sym}: Missing targets: {r['missing_targets']}")

        # Gate 6: No dead base features
        for feat_name, stats in r["feature_stats"].items():
            if stats["std"] < 0.01:
                failures.append(f"{sym}: Base feature {feat_name} is dead (std={stats['std']:.6f})")

        # Gate 7: Target tail integrity
        # Threshold 10: fill_null corruption creates 50+ zeros bunched at tail end.
        # Legitimate zeros (price unchanged over 50 bars) are scattered and fewer,
        # especially for low-price altcoins during consolidation (e.g., LINK ~9% zero rate).
        if r["zero_tail_50"] >= 10:
            failures.append(f"{sym}: Tail corruption ({r['zero_tail_50']} zeros in last 100 of target_return_50)")

        # Gate 8: All 7 XD features present
        if not r["has_xd"]:
            failures.append(f"{sym}: Missing XD features: {r['missing_xd_features']}")

        # Gate 9: No nulls in XD features
        xd_null_sum = sum(r["xd_nulls"].values())
        if xd_null_sum > 0:
            failures.append(f"{sym}: {xd_null_sum} null values in XD features")

        # Gate 10: No dead XD features
        for feat_name, stats in r["xd_stats"].items():
            if stats["std"] is not None and stats["std"] < 0.01:
                failures.append(f"{sym}: XD feature {feat_name} is dead (std={stats['std']:.6f})")

        # Gate 11: No leakage
        for feat_name, corr in r["xd_leakage"].items():
            if abs(corr) > 0.10:
                failures.append(f"{sym}: Leakage {feat_name} -> target_return_1 corr={corr:.4f}")

        # Gate 12: Enhanced leakage (base features vs all horizons)
        enh = r.get("enhanced", {})
        leak_matrix = enh.get("leak_matrix", {})
        for feat_name, horizon_corrs in leak_matrix.items():
            for h_name, corr in horizon_corrs.items():
                if abs(corr) > 0.15:
                    failures.append(f"{sym}: Leakage {feat_name} -> {h_name} corr={corr:.4f}")

        # Gate 13: Regime balance (no regime > 90%)
        regime_bal = enh.get("regime_balance", {})
        for regime_id, pct in regime_bal.items():
            if pct > 90:
                failures.append(f"{sym}: Extreme regime imbalance: regime {regime_id} = {pct}%")

    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="Dataset Inspector (Base + Cross-Asset)")
    parser.add_argument("--plots", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--strict", action="store_true",
                        help="Return exit code 1 if critical checks fail (CI gate mode)")
    args = parser.parse_args()

    # post-2026-04-26 layout: legacy v50 chimeras live under processed/chimera_legacy/
    legacy_dir = PROCESSED_DIR / "chimera_legacy"
    files = sorted(legacy_dir.glob("*_v50_chimera*.parquet")) if legacy_dir.exists() else []
    if not files:
        print(f"[ERROR] No V50 datasets found in {legacy_dir}")
        if args.strict:
            sys.exit(1)
        return

    # Auto-save log to logs/pipeline/
    import io, contextlib
    LOG_SAVE_DIR = PROJECT_ROOT / "logs" / "pipeline"
    LOG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_SAVE_DIR / f"inspect_dataset_{date.today().isoformat()}.log"
    log_buffer = io.StringIO()

    class TeeWriter:
        """Write to both stdout and a buffer."""
        def __init__(self, *writers):
            self.writers = writers
        def write(self, s):
            for w in self.writers:
                w.write(s)
        def flush(self):
            for w in self.writers:
                w.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, log_buffer)

    print(f"[START] Inspecting {len(files)} processed datasets...")

    results = []
    for f in files:
        print(f"  Loading {f.name}...", end="\r")
        results.append(inspect_dataset(f))

    print_summary(results)

    if args.plots:
        print(f"\n  Generating diagnostic plots...")
        generate_plots(results)

    # Strict gate checks
    if args.strict:
        print(f"\n{'='*70}")
        print(f"  STRICT GATE CHECKS (13 gates: 7 base [34 features] + 4 cross-asset [7 XD] + 2 enhanced)")
        print(f"{'='*70}")
        all_pass, failures = run_strict_gates(results)
        if all_pass:
            print(f"  [PASS] All {len(results)} assets passed all 13 gates")
        else:
            for fail in failures:
                print(f"  [GATE FAIL] {fail}")
            print(f"\n  [FAIL] {len(failures)} gate failures across {len(results)} assets")

        # Save log before exit
        sys.stdout = original_stdout
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(log_buffer.getvalue())
        print(f"  Log saved to {log_path}")

        # Cleanup df references
        for r in results:
            r["df"] = None

        sys.exit(0 if all_pass else 1)

    # Cleanup df references
    for r in results:
        r["df"] = None

    print(f"\n[DONE] Inspection complete.")
    if not args.plots:
        print(f"  Tip: re-run with --plots for diagnostic plots in {PLOTS_DIR / _DATE_SUBDIR}")

    # Save log
    sys.stdout = original_stdout
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(log_buffer.getvalue())
    print(f"  Log saved to {log_path}")


if __name__ == "__main__":
    main()
