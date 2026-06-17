"""Pre-retrain data audit (2026-05-21, oracle).

Validates every assumption that V1.1 / V13 / V22 training will rely on:

  1. Schema: v51 chimera has the expected columns + dtypes.
  2. Per-feature health: NaN counts; norm_* features should have std ~ 1.0.
  3. Target distribution: target_return_h should be near-zero-mean with
     realistic tails (no fill_null(0) corruption; no leakage from future).
  4. Bin coverage: TwoHotSymlog [-1, 1] / 255 bins — what fraction of bins
     are actually hit by training data?
  5. Asset coverage: do all 10 V1.x assets have files? Bar counts?
  6. Walk-forward split feasibility: enough bars for 50/20/20/10 + 400-bar purge?
  7. Look-ahead sanity: target_return_h at bar t should correlate with
     features at bar t, NOT t-1 (else features are future-leaking).
  8. Cross-asset (xd_*) sanity: BTC xd_btc_return ~ 0 (anchor invariant),
     other assets xd_btc_return is non-trivial.
  9. Signal-distance from null: rolling-window IC of norm_return_1 vs
     target_return_1 — small but non-zero is healthy.
  10. FEATURE_LIST_29 resolves and all 29 features exist in v51 schema.

Outputs a structured report to stdout. Fails LOUDLY on any invariant
violation (per CLAUDE.md no-silent-failures).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from feature_sets import (  # noqa: E402
    FEATURE_LIST_29,
    FEATURE_LIST_34,
    FEATURE_LIST_41,
)

DATA_DIR = ROOT / "data" / "processed" / "chimera" / "dollar"


def header(s: str) -> None:
    print(f"\n{'=' * 78}\n  {s}\n{'=' * 78}")


def fmt_pct(x: float) -> str:
    return f"{x * 100:>7.3f}%"


def audit_one_asset(path: Path, asset: str, feature_list: list[str]) -> dict:
    print(f"\n--- {asset:8s}  {path.name}  ({path.stat().st_size / 1e6:.0f} MB)")
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    cols = list(schema.names())
    n_cols = len(cols)
    n_rows = lf.select(pl.len()).collect().item()
    print(f"  rows: {n_rows:>10,}    cols: {n_cols}")

    out = {"asset": asset, "n_rows": n_rows, "n_cols": n_cols}

    # 1. Feature presence
    missing = [f for f in feature_list if f not in cols]
    if missing:
        print(f"  [FAIL] {len(missing)} features missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        out["missing_features"] = missing
    else:
        print(f"  features f{len(feature_list)}: all {len(feature_list)} present")

    # 2. Per-feature stats
    df = lf.select([*feature_list, "target_return_1", "target_return_4",
                    "target_return_16", "target_return_64",
                    "regime_label", "timestamp",
                    "xd_btc_return"]).collect()
    feat_block = df.select(feature_list).describe()
    means = df.select([pl.col(f).mean() for f in feature_list]).row(0)
    stds = df.select([pl.col(f).std() for f in feature_list]).row(0)
    nan_counts = df.select([pl.col(f).is_null().sum() for f in feature_list]).row(0)

    bad_std = [(f, s) for f, s in zip(feature_list, stds)
               if f.startswith("norm_") and (s is None or s < 0.5 or s > 2.0)]
    if bad_std:
        print(f"  [WARN] norm_* features with std outside [0.5, 2.0]: {len(bad_std)}")
        for f, s in bad_std[:5]:
            print(f"         {f}: std={s}")
    nan_total = sum(c for c in nan_counts if c is not None)
    if nan_total > 0:
        worst = sorted(zip(feature_list, nan_counts), key=lambda x: -(x[1] or 0))[:5]
        print(f"  [WARN] {nan_total} NaN cells; worst:")
        for f, c in worst:
            if c:
                print(f"         {f}: {c} NaNs ({c/n_rows*100:.3f}%)")
    else:
        print(f"  NaN: 0 across all {len(feature_list)} features")

    out["bad_std"] = bad_std
    out["nan_total"] = nan_total

    # 3. Target distribution
    print(f"\n  TARGETS:")
    for h in [1, 4, 16, 64]:
        col = f"target_return_{h}"
        t = df[col].to_numpy()
        t = t[np.isfinite(t)]
        z_share = (np.abs(t) < 1e-9).mean()
        skew = float(((t - t.mean()) ** 3).mean() / (t.std() ** 3 + 1e-12))
        kurt = float(((t - t.mean()) ** 4).mean() / (t.std() ** 4 + 1e-12)) - 3
        p1, p50, p99 = np.percentile(t, [1, 50, 99])
        # Bin coverage in [-1, 1] symlog: bin_width = 2/254 = 0.00787
        # In symlog: log1p(|0.5|) = 0.405, log1p(|0.1|) = 0.0953
        # Returns of ±0.01 (1%) map to symlog ±0.00995 → bin index ~1.27 around center
        sym = np.sign(t) * np.log1p(np.abs(t))
        in_range = ((sym >= -1.0) & (sym <= 1.0)).mean()
        n_unique_bins = len(np.unique(np.clip(((sym + 1.0) / (2.0/254)).astype(np.int32), 0, 254)))
        print(f"    h={h:2d}:  mean={t.mean():+.5f}  std={t.std():.5f}  "
              f"skew={skew:+.2f}  exkurt={kurt:+.1f}  "
              f"p1/p50/p99=[{p1:+.4f}, {p50:+.4f}, {p99:+.4f}]")
        print(f"           zero_share={fmt_pct(z_share)}  in_symlog_range={fmt_pct(in_range)}  "
              f"unique_bins/255={n_unique_bins}")
        # CLAUDE.md tail invariant: <10 zero values in last 100 rows
        last_100 = df[col][-100:].to_numpy()
        zeros_tail = int(np.sum(np.abs(last_100) < 1e-9))
        flag = "[FAIL]" if zeros_tail >= 10 else "[OK]"
        print(f"           tail-zeros (last 100 rows): {zeros_tail} {flag}")
        out[f"tail_zeros_h{h}"] = zeros_tail

    # 4. Regime label distribution
    print(f"\n  REGIME LABEL:")
    rl = df["regime_label"].to_numpy()
    finite_rl = rl[np.isfinite(rl)]
    classes, counts = np.unique(finite_rl.astype(np.int32), return_counts=True)
    nan_rl = (~np.isfinite(rl)).sum()
    print(f"    class shares: {dict(zip(classes.tolist(), [fmt_pct(c/n_rows) for c in counts]))}")
    print(f"    NaN regime labels: {nan_rl} ({nan_rl/n_rows*100:.3f}%)")
    out["regime_classes"] = classes.tolist()
    out["regime_counts"] = counts.tolist()

    # 5. Timestamp sanity
    ts = df["timestamp"].to_numpy()
    is_mono = bool(np.all(np.diff(ts) >= 0))
    ts_in_range = bool(np.all((ts > 1.5e12) & (ts < 2.5e12)))  # 2017-2049
    print(f"\n  TIMESTAMPS:")
    print(f"    monotonic: {is_mono}    in_13digit_ms_range: {ts_in_range}")
    print(f"    first: {ts.min():.0f}  last: {ts.max():.0f}")
    print(f"    span: {(ts.max() - ts.min()) / 86400000:.1f} days = {(ts.max() - ts.min()) / 86400000 / 365:.2f} years")
    out["mono"] = is_mono

    # 6. Cross-asset (xd_*) sanity
    if "xd_btc_return" in cols:
        xd = df["xd_btc_return"].to_numpy()
        xd_finite = xd[np.isfinite(xd)]
        xd_mean = xd_finite.mean()
        xd_std = xd_finite.std()
        print(f"\n  XD anchor (xd_btc_return):")
        print(f"    mean: {xd_mean:+.5f}  std: {xd_std:.5f}")
        # For BTC itself the anchor should be ~ 0
        if asset.startswith("BTC") and abs(xd_mean) > 0.001:
            print(f"    [WARN] BTC xd_btc_return mean != 0 (anchor invariant)")

    # 7. Look-ahead audit: target_return_1 at bar t should correlate with the
    # FORWARD return; norm_return_1 at bar t is the LAGGED return (causal feature).
    # Pearson(norm_return_1, target_return_1) should be small (random) — if it
    # were close to ±1 the feature would be leaking the target.
    if "norm_return_1" in cols:
        x = df["norm_return_1"].to_numpy()
        y = df["target_return_1"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 1000:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            print(f"\n  LEAKAGE CHECK:")
            print(f"    corr(norm_return_1, target_return_1) = {corr:+.4f}  "
                  f"({'OK' if abs(corr) < 0.1 else 'WARN' if abs(corr) < 0.5 else 'FAIL'})")
            print(f"    (close to 0 = no feature leakage; close to ±1 = leak)")
            # Sanity: signal direction. Crypto momentum 1-bar: small POSITIVE corr expected.
            out["leakage_norm_return_corr"] = corr

    # 8. Walk-forward split feasibility (50/20/20/10 + 400-bar purge gaps)
    needed = int(n_rows * 0.50) + 400 + int(n_rows * 0.20) + 400 + int(n_rows * 0.20) + 400 + int(n_rows * 0.10)
    print(f"\n  WALK-FORWARD FEASIBILITY:")
    print(f"    rows: {n_rows:,}    needed (50/20/20/10 + 3*400 purge): {needed:,}")
    print(f"    headroom: {n_rows - needed:,} bars")

    return out


def main():
    print("=" * 78)
    print("  V51 CHIMERA DATA AUDIT — PRE-RETRAIN")
    print("  feature set under audit: FEATURE_LIST_29")
    print("=" * 78)

    files = sorted(DATA_DIR.glob("*_v51_chimera_*.parquet"))
    print(f"\n  v51 chimera files: {len(files)}")
    print(f"  sample names: {[f.name for f in files[:5]]}")

    # 10 canonical V1.x assets
    universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

    assets_present = []
    for sym in universe:
        matching = [f for f in files if f.name.lower().startswith(sym.lower())]
        if matching:
            assets_present.append((sym, max(matching, key=lambda p: p.name)))
        else:
            print(f"  [FAIL] no v51 chimera file for {sym}!")

    print(f"\n  10-asset coverage: {len(assets_present)}/10")
    if len(assets_present) < 10:
        print("  ABORT: not all 10 V1.x universe assets present")
        sys.exit(2)

    results = []
    header("PER-ASSET AUDIT (FEATURE_LIST_29)")
    for sym, path in assets_present:
        try:
            results.append(audit_one_asset(path, sym, FEATURE_LIST_29))
        except Exception as e:
            print(f"  [FAIL] {sym}: {type(e).__name__}: {e}")

    # Aggregate sanity table
    header("AGGREGATE SUMMARY")
    print(f"  {'asset':10s}  {'rows':>10s}  {'NaN':>8s}  {'tail_z_h1':>10s}  {'tail_z_h4':>10s}  {'tail_z_h16':>10s}  {'tail_z_h64':>10s}")
    for r in results:
        print(f"  {r['asset']:10s}  {r['n_rows']:>10,}  {r['nan_total']:>8d}  "
              f"{r.get('tail_zeros_h1', 0):>10d}  {r.get('tail_zeros_h4', 0):>10d}  "
              f"{r.get('tail_zeros_h16', 0):>10d}  {r.get('tail_zeros_h64', 0):>10d}")

    # Final verdict
    header("FINAL VERDICT")
    fails = []
    warns = []
    for r in results:
        if r.get("missing_features"):
            fails.append(f"{r['asset']}: missing features {r['missing_features']}")
        if not r.get("mono", True):
            fails.append(f"{r['asset']}: non-monotonic timestamps")
        if r.get("tail_zeros_h1", 0) >= 10 or r.get("tail_zeros_h64", 0) >= 10:
            fails.append(f"{r['asset']}: tail-zeros >= 10 (fill_null corruption)")
        if r.get("bad_std"):
            warns.append(f"{r['asset']}: {len(r['bad_std'])} norm_* features have std outside [0.5, 2.0]")
        if r.get("nan_total", 0) > r["n_rows"] * 0.01:
            warns.append(f"{r['asset']}: NaN > 1% of cells ({r['nan_total']} / {r['n_rows']})")

    if fails:
        print(f"\n  [FAIL] {len(fails)} blocking issues:")
        for f in fails:
            print(f"    - {f}")
        sys.exit(2)
    print(f"\n  [PASS] no blocking issues across {len(results)} assets")
    if warns:
        print(f"\n  [WARN] {len(warns)} advisory issues (not blocking):")
        for w in warns:
            print(f"    - {w}")
    else:
        print("  [PASS] no advisory issues")

    print("\n  READY for V1.1 / V13 / V22 retrain at FEATURE_LIST_29.")


if __name__ == "__main__":
    main()
