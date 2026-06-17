"""
Linear Baseline -- Ridge Regression on 13-121 Features (Shared V0)

Establishes the IC floor achievable with a simple linear model (no DL).
This is the benchmark ALL world models (V1-V19) must beat.

If linear IC ~ DL IC, the DL architecture buys minimal improvement.
If DL IC >> linear IC, temporal/nonlinear modeling is justified.

Uses walk-forward split with purge gap for fair comparison.

Usage:
    python linear_baseline.py                       # 41 features (default)
    python linear_baseline.py --features 13         # legacy V1.0 base
    python linear_baseline.py --features 29         # f29 (Pattern P; no dead features)
    python linear_baseline.py --features 121        # full v51 frontier (requires v51 chimera)
    python linear_baseline.py --workers 8           # parallel ridge fits
    python linear_baseline.py --full                # simple 90/10 (no purge gap)

Post-2026-04-27 upgrade:
  - Centralized feature config (src/feature_sets.py) supports all 18 counts
  - ProcessPoolExecutor for per-(asset, horizon) ridge sweeps
  - Shuffled IC iterates ALL REWARD_HORIZONS (was: hard-coded h=1, h=16)
  - v51 chimera read path for f46+ counts (TrainingLoader)
"""
import warnings
import os
import numpy as np
import polars as pl
import sys
import argparse
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ridge handles multicollinearity via regularization -- suppress scipy's
# ill-conditioned matrix warnings (expected with 37 correlated features)
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import (
    DATA_DIR, FEATURE_LIST, FEATURE_LIST_13, ASSET_TO_IDX, ASSET_LIST,
    REWARD_HORIZONS, INPUT_DIM, PURGE_GAP_BARS, RIDGE_ALPHAS,
    N_SHUFFLE_SEEDS, LOG_DIR,
    FEATURE_LIST_18, FEATURE_LIST_21, FEATURE_LIST_25,
    FEATURE_LIST_30, FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
    FEATURE_GROUPS,
    # TRAIN_RATIO/VAL_RATIO replaced by date-based splits via _load_split_boundaries
    get_feature_config, list_supported_features,
)
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets

# Workers (top-level functions for ProcessPoolExecutor — must be importable)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _workers import (
    ridge_sweep_worker,
    ablation_worker,
    shuffled_ic_worker,
)


def _date_to_ms(d_str: str) -> int:
    """YYYY-MM-DD -> epoch milliseconds (UTC)."""
    from datetime import datetime, timezone
    return int(datetime.strptime(d_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def _load_split_boundaries():
    """Read frozen split dates from config/data_config.yaml. Returns dict with epoch ms."""
    import sys as _sys
    from pathlib import Path as _P
    pipeline_dir = _P(__file__).resolve().parents[4] / "src" / "pipeline"
    if str(pipeline_dir) not in _sys.path:
        _sys.path.insert(0, str(pipeline_dir))
    from purge_split import get_split_dates  # noqa: E402
    b = get_split_dates()
    return {
        "train_end_ms": _date_to_ms(b.train_end),
        "val_end_ms":   _date_to_ms(b.val_end),
        "oos_end_ms":   _date_to_ms(b.oos_end),
        "purge_bars":   int(b.purge_bars),
    }


def get_dated_split_indices(timestamps, boundaries):
    """Calendar-aligned 4-way split index lookup for one asset.

    Returns dict with keys: train_end, val_start, val_end, oos_start, oos_end,
    unseen_start, n. All bars between train_end..val_start (etc.) are PURGE.
    """
    import numpy as _np
    n = len(timestamps)
    train_end   = int(_np.searchsorted(timestamps, boundaries["train_end_ms"]))
    val_end     = int(_np.searchsorted(timestamps, boundaries["val_end_ms"]))
    oos_end     = int(_np.searchsorted(timestamps, boundaries["oos_end_ms"]))
    purge       = boundaries["purge_bars"]
    val_start   = min(train_end + purge, n)
    oos_start   = min(val_end + purge, n)
    unseen_start = min(oos_end + purge, n)
    return {
        "train_end": train_end, "val_start": val_start, "val_end": val_end,
        "oos_start": oos_start, "oos_end": oos_end,
        "unseen_start": unseen_start, "n": n,
    }


def load_data(feature_list=None, use_v51: bool = False):
    """Load all asset data, return list of (features, targets, asset_name, timestamps).

    Layout v3 dedup: picks the LATEST dated chimera per asset (e.g. for BTC the
    file with the newest YYYYMMDD suffix wins).

    use_v51=True: read from v51 chimera (data/processed/chimera/dollar/...)
                  via the post-2026-04-26 layout, supports the full 80
                  frontier features. Required for f46+.
    use_v51=False (default): read from v50 chimera_legacy (V0 historical path).
    """
    if feature_list is None:
        feature_list = FEATURE_LIST

    if use_v51:
        from pathlib import Path as _P
        v51_dir = _P(__file__).resolve().parents[4] / "data" / "processed" / "chimera" / "dollar"
        files = sorted(v51_dir.glob("*_v51_chimera_*.parquet"))
    else:
        files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))

    # Layout v3 dedup: pick NEWEST VALID dated snapshot per asset.
    # Falls back to an older snapshot if the latest is missing required
    # feature columns (defends against partial chimera builds).
    by_asset: dict = {}
    for f in files:
        sym_l = f.stem.split("_")[0]
        asset_name = sym_l.upper()
        if asset_name not in ASSET_TO_IDX:
            continue
        by_asset.setdefault(asset_name, []).append(f)

    required_cols = set(feature_list)
    selected = []
    for asset_name, candidates in by_asset.items():
        candidates.sort(key=lambda p: p.name, reverse=True)  # newest first
        chosen = None
        for cand in candidates:
            try:
                cols = set(pl.read_parquet_schema(cand).keys())
            except Exception:
                continue
            missing = required_cols - cols
            if not missing:
                chosen = cand
                break
            if cand is candidates[0]:
                print(f"  [WARN] {asset_name}: latest snapshot {cand.name} missing "
                      f"{len(missing)} cols ({sorted(missing)[:3]}...); falling back to older")
        if chosen is None:
            print(f"  [ERROR] {asset_name}: NO valid snapshot; skipping")
            continue
        selected.append(chosen)
    files = sorted(selected)

    all_data = []
    for f in files:
        sym_l = f.stem.split("_")[0]
        asset_name = sym_l.upper()
        df = pl.read_parquet(f)
        df = selective_drop_nulls(df, feature_list, REWARD_HORIZONS, asset_name)
        feats, targets = extract_features_targets(df, feature_list, REWARD_HORIZONS, asset_name)
        ts = df["timestamp"].to_numpy().astype(np.int64) if "timestamp" in df.columns else None
        all_data.append((feats, targets, asset_name, ts))
        print(f"  {asset_name}: {len(feats):,} bars")

    return all_data


def _resolve_polars_threads(workers: int) -> int:
    """How many polars/numpy threads each worker should get."""
    cpu = os.cpu_count() or 8
    return max(2, cpu // max(1, workers))


def compute_ic(preds, reals):
    """Compute Pearson IC with p-value."""
    mask = np.isfinite(preds) & np.isfinite(reals)
    p, r = preds[mask], reals[mask]
    if len(p) < 30 or np.std(p) < 1e-10 or np.std(r) < 1e-10:
        return 0.0, 1.0, 0
    ic = float(np.corrcoef(p, r)[0, 1])
    n = len(p)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 2)))
    return ic, p_value, n


def compute_rank_ic(preds, reals):
    """Compute Spearman rank IC."""
    mask = np.isfinite(preds) & np.isfinite(reals)
    p, r = preds[mask], reals[mask]
    if len(p) < 30:
        return 0.0
    return float(scipy_stats.spearmanr(p, r).statistic)


def run_linear_baseline(use_walk_forward=True, n_features=41, workers: int = 1):
    """
    Train ridge regression per-asset, per-horizon (parallelizable).

    n_features: any count from feature_sets.SUPPORTED_COUNTS
                {13, 18, 21, 25, 29, 30, 34, 37, 41,
                 46, 60, 73, 78, 81, 84, 97, 110, 121}
    workers:    1 = sequential (default; safe under all RAM regimes)
                2-8 = parallel via ProcessPoolExecutor
    """
    # Resolve via central registry (post-2026-04-27 centralization)
    feature_list, input_dim, _base_dim = get_feature_config(n_features)
    use_v51 = n_features > 41   # f46+ requires v51 chimera

    print("=" * 70)
    print(f"  LINEAR BASELINE (Ridge Regression on {input_dim} Features)")
    print(f"  Source: {'v51 chimera (frontier)' if use_v51 else 'v50 chimera (legacy)'}")
    print(f"  Workers: {workers}")
    print("  Purpose: IC floor for ALL world models (V1-V19)")
    print("=" * 70)

    all_data = load_data(feature_list, use_v51=use_v51)
    if not all_data:
        print("  [ERROR] No data found.")
        return

    # Calendar-aligned 4-way split: dates frozen in config/data_config.yaml
    # (same boundaries V1.x training uses — apples-to-apples IC floor)
    boundaries = _load_split_boundaries()
    print(f"\n  Split boundaries (calendar-aligned, all assets):")
    print(f"    train_end={boundaries['train_end_ms']}, val_end={boundaries['val_end_ms']}, "
          f"oos_end={boundaries['oos_end_ms']}, purge_bars={boundaries['purge_bars']}")

    results = {}

    # Build the (asset, horizon) task list, plus per-task X/y slices
    tasks = []
    splits = {}
    for feats, targets, asset_name, ts in all_data:
        if ts is None:
            raise ValueError(f"{asset_name}: chimera missing 'timestamp' column "
                             f"(required for calendar-aligned splits)")
        idx = get_dated_split_indices(ts, boundaries)
        train_end, val_start, val_end = idx["train_end"], idx["val_start"], idx["val_end"]
        if not use_walk_forward:
            val_start = train_end  # disable purge for --full mode (legacy 90/10 logic)
        if val_start >= val_end:
            val_start = train_end
        results[asset_name] = {}
        splits[asset_name] = (train_end, val_start, val_end, idx["n"])
        for h in REWARD_HORIZONS:
            tasks.append((
                asset_name,
                feats[:train_end],
                feats[val_start:val_end],
                targets[h][:train_end],
                targets[h][val_start:val_end],
                tuple(RIDGE_ALPHAS),
                int(h),
                _resolve_polars_threads(workers),
            ))

    # Print per-asset split sizes once
    for asset_name, (te, vs, ve, n) in splits.items():
        print(f"  {asset_name}: train={te:,} val={ve - vs:,} (purge={vs - te})")

    print(f"\n  Running {len(tasks)} ridge sweeps "
          f"({len(all_data)} assets x {len(REWARD_HORIZONS)} horizons x {len(RIDGE_ALPHAS)} alphas)")

    # Dispatch
    out_iter = []
    if workers <= 1:
        for t in tasks:
            out_iter.append(ridge_sweep_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(ridge_sweep_worker, t) for t in tasks]
            for fut in as_completed(futs):
                out_iter.append(fut.result())

    for r in out_iter:
        a, h = r["asset"], r["horizon"]
        results[a][h] = r
        sig = ("***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01
               else "*" if r["p_value"] < 0.05 else "")
        print(f"    {a:<10} t+{h:<3} IC:{r['ic']:+.4f}{sig:<4} "
              f"RankIC:{r['rank_ic']:+.4f} Dir:{r['dir_acc']*100:.1f}% "
              f"alpha={r['best_alpha']} n={r['n']}")

    # --- Feature Importance (using best alpha on pooled data) ---
    print(f"\n{'='*70}")
    print("  FEATURE IMPORTANCE (Pooled Ridge Coefficients)")
    print(f"{'='*70}")

    for h in REWARD_HORIZONS:
        X_all, y_all = [], []
        for feats, targets, _, ts in all_data:
            train_end = get_dated_split_indices(ts, boundaries)["train_end"]
            X_all.append(feats[:train_end])
            y_all.append(targets[h][:train_end])

        X_pooled = np.concatenate(X_all)
        y_pooled = np.concatenate(y_all)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_pooled)

        model = Ridge(alpha=1.0)
        model.fit(X_s, y_pooled)
        coefs = np.abs(model.coef_)
        importance = coefs / coefs.sum() * 100

        print(f"\n  t+{h} (top features by |coefficient|):")
        ranked = sorted(
            zip(feature_list, model.coef_, importance),
            key=lambda x: abs(x[1]), reverse=True,
        )
        for name, coef, imp in ranked:
            bar = "#" * int(imp)
            print(f"    {name:<24} coef:{coef:+.6f}  imp:{imp:4.1f}% [{bar}]")

    # --- Feature Ablation (drop-one IC change) ---
    print(f"\n{'='*70}")
    print("  FEATURE ABLATION (Drop-One IC Change, Pooled, t+1)")
    print(f"{'='*70}")

    X_all, y_all = [], []
    X_val_all, y_val_all = [], []
    for feats, targets, _, ts in all_data:
        idx = get_dated_split_indices(ts, boundaries)
        train_end, val_start, val_end = idx["train_end"], idx["val_start"], idx["val_end"]
        if not use_walk_forward:
            val_start = train_end
        if val_start >= val_end:
            val_start = train_end
        X_all.append(feats[:train_end])
        y_all.append(targets[1][:train_end])
        X_val_all.append(feats[val_start:val_end])
        y_val_all.append(targets[1][val_start:val_end])

    X_train_pooled = np.concatenate(X_all)
    y_train_pooled = np.concatenate(y_all)
    X_val_pooled = np.concatenate(X_val_all)
    y_val_pooled = np.concatenate(y_val_all)

    scaler = StandardScaler()
    X_ts = scaler.fit_transform(X_train_pooled)
    X_vs = scaler.transform(X_val_pooled)

    model = Ridge(alpha=1.0)
    model.fit(X_ts, y_train_pooled)
    baseline_preds = model.predict(X_vs)
    baseline_ic, _, _ = compute_ic(baseline_preds, y_val_pooled)
    print(f"\n  Baseline (all {input_dim} features): IC = {baseline_ic:+.4f}")

    ablation_results = []
    for drop_idx, feat_name in enumerate(feature_list):
        keep = [i for i in range(input_dim) if i != drop_idx]
        X_ts_abl = scaler.fit_transform(X_train_pooled[:, keep])
        X_vs_abl = scaler.transform(X_val_pooled[:, keep])

        model_abl = Ridge(alpha=1.0)
        model_abl.fit(X_ts_abl, y_train_pooled)
        preds_abl = model_abl.predict(X_vs_abl)
        ic_abl, _, _ = compute_ic(preds_abl, y_val_pooled)
        delta = ic_abl - baseline_ic
        ablation_results.append((feat_name, ic_abl, delta))

    ablation_results.sort(key=lambda x: x[2])
    print(f"\n  {'Feature':<24} {'IC w/o':>8} {'Delta':>8}  Impact")
    print(f"  {'-'*56}")
    for name, ic_wo, delta in ablation_results:
        impact = "CRITICAL" if delta < -0.003 else "USEFUL" if delta < -0.001 else "MARGINAL" if delta < 0 else "NOISE"
        print(f"  {name:<24} {ic_wo:+.4f}  {delta:+.4f}   {impact}")

    # --- Shuffled IC (linear model) ---
    # Iterate ALL REWARD_HORIZONS now (was hardcoded to [1, 16] only).
    print(f"\n{'='*70}")
    print(f"  LINEAR SHUFFLED IC (anti-memorization check, h={REWARD_HORIZONS})")
    print(f"{'='*70}")

    polars_threads = _resolve_polars_threads(workers)

    # Build (asset, horizon, seed) task list
    shuf_tasks = []
    for feats, targets, asset_name, ts in all_data:
        split = get_dated_split_indices(ts, boundaries)["train_end"]
        for h in REWARD_HORIZONS:
            for seed in range(N_SHUFFLE_SEEDS):
                shuf_tasks.append((
                    asset_name, int(h), int(seed),
                    feats[:split], targets[h][:split],
                    1.0, polars_threads,  # alpha=1.0 for shuffled-IC test
                ))

    shuf_results = []
    if workers <= 1:
        for t in shuf_tasks:
            shuf_results.append(shuffled_ic_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(shuffled_ic_worker, t) for t in shuf_tasks]
            for fut in as_completed(futs):
                shuf_results.append(fut.result())

    # Aggregate by horizon
    by_horizon: dict = {h: [] for h in REWARD_HORIZONS}
    for r in shuf_results:
        by_horizon[r["horizon"]].append(r["ic_shuffled"])
    for h in REWARD_HORIZONS:
        ics = by_horizon[h]
        mean_shic = float(np.mean(ics)) if ics else 0.0
        std_shic = float(np.std(ics)) if ics else 0.0
        print(f"  t+{h:<3} Shuffled IC: {mean_shic:+.4f} +/- {std_shic:.4f} "
              f"(n={len(ics)} = {N_SHUFFLE_SEEDS} seeds x {len(all_data)} assets)")

    # --- Feature Group Masking Comparison (multi-head equivalent) ---
    print(f"\n{'='*70}")
    print("  FEATURE GROUP MASKING (multi-head IC comparison)")
    print(f"  Train with all features, evaluate with groups masked to zero")
    print(f"{'='*70}")

    n_full = len(FEATURE_LIST_37)
    max_train_masking = 500_000  # subsample to avoid OOM on pooled 10-asset data
    max_val_masking = 200_000
    all_feat_data = load_data(FEATURE_LIST_37)
    if all_feat_data:
        X_all_tr, X_all_val, y_all_tr, y_all_val = [], [], {}, {}
        for h in REWARD_HORIZONS:
            y_all_tr[h] = []
            y_all_val[h] = []

        for feats, targets, _, ts in all_feat_data:
            idx = get_dated_split_indices(ts, boundaries)
            train_end, val_start, val_end = idx["train_end"], idx["val_start"], idx["val_end"]
            if not use_walk_forward:
                val_start = train_end
            if val_start >= val_end:
                val_start = train_end
            X_all_tr.append(feats[:train_end])
            X_all_val.append(feats[val_start:val_end])
            for h in REWARD_HORIZONS:
                y_all_tr[h].append(targets[h][:train_end])
                y_all_val[h].append(targets[h][val_start:val_end])

        X_train_full = np.concatenate(X_all_tr)
        X_val_full = np.concatenate(X_all_val)
        y_train_all = {h: np.concatenate(y_all_tr[h]) for h in REWARD_HORIZONS}
        y_val_all = {h: np.concatenate(y_all_val[h]) for h in REWARD_HORIZONS}

        # Subsample to avoid OOM (pooled ~18M rows x 37 features)
        rng_mask = np.random.default_rng(42)
        if len(X_train_full) > max_train_masking:
            tr_idx = rng_mask.choice(len(X_train_full), max_train_masking, replace=False)
            X_train_full = X_train_full[tr_idx]
            y_train_all = {h: y_train_all[h][tr_idx] for h in REWARD_HORIZONS}
        if len(X_val_full) > max_val_masking:
            vl_idx = rng_mask.choice(len(X_val_full), max_val_masking, replace=False)
            X_val_full = X_val_full[vl_idx]
            y_val_all = {h: y_val_all[h][vl_idx] for h in REWARD_HORIZONS}
        print(f"\n  Pooled data: train={len(X_train_full):,}, val={len(X_val_full):,} (subsampled)")

        full_feat_names = list(FEATURE_LIST_37)

        # Define feature group masks (which indices to KEEP for each configuration)
        mask_configs = [
            ("f37 (full)",         list(range(37))),
            ("f30 (base only)",    list(range(30))),
            ("f25 (base+hawkes)",  list(range(25))),
            ("f21 (base+tier1)",   list(range(21))),
            ("f18 (extended)",     list(range(18))),
            ("f13 (legacy)",       list(range(13))),
            ("f13+hawkes",         list(range(13)) + list(range(21, 25))),
            ("f13+ic_boost",       list(range(13)) + list(range(25, 30))),
            ("f13+xd",            list(range(13)) + list(range(30, 37))),
            ("hawkes only (4)",    list(range(21, 25))),
            ("ic_boost only (5)",  list(range(25, 30))),
            ("xd only (7)",        list(range(30, 37))),
        ]

        print(f"\n  {'Configuration':<24}", end="")
        for h in REWARD_HORIZONS:
            print(f"  {'t+' + str(h):>8}", end="")
        print(f"  {'Avg IC':>8}")
        print(f"  {'-'*24}" + "-" * 10 * (len(REWARD_HORIZONS) + 1))

        for config_name, keep_idx in mask_configs:
            ics = []
            for h in REWARD_HORIZONS:
                X_tr_masked = X_train_full.copy()
                X_vl_masked = X_val_full.copy()
                mask_out = [i for i in range(n_full) if i not in keep_idx]
                if mask_out:
                    X_tr_masked[:, mask_out] = 0.0
                    X_vl_masked[:, mask_out] = 0.0

                scaler_m = StandardScaler()
                X_tr_s = scaler_m.fit_transform(X_tr_masked)
                X_vl_s = scaler_m.transform(X_vl_masked)

                model_m = Ridge(alpha=1.0)
                model_m.fit(X_tr_s, y_train_all[h])
                preds_m = model_m.predict(X_vl_s)
                ic_m, _, _ = compute_ic(preds_m, y_val_all[h])
                ics.append(ic_m)

            avg = np.mean(ics)
            print(f"  {config_name:<24}", end="")
            for ic_val in ics:
                print(f"  {ic_val:>+8.4f}", end="")
            print(f"  {avg:>+8.4f}")

        # Delta analysis (contribution of each group)
        print(f"\n  GROUP CONTRIBUTION (IC delta when adding group to base13):")
        base13_ics = {}
        base13_idx = list(range(13))
        for h in REWARD_HORIZONS:
            X_tr_b = X_train_full.copy()
            X_vl_b = X_val_full.copy()
            mask_out = [i for i in range(n_full) if i not in base13_idx]
            X_tr_b[:, mask_out] = 0.0
            X_vl_b[:, mask_out] = 0.0
            scaler_b = StandardScaler()
            X_tr_s = scaler_b.fit_transform(X_tr_b)
            X_vl_s = scaler_b.transform(X_vl_b)
            model_b = Ridge(alpha=1.0)
            model_b.fit(X_tr_s, y_train_all[h])
            preds_b = model_b.predict(X_vl_s)
            ic_b, _, _ = compute_ic(preds_b, y_val_all[h])
            base13_ics[h] = ic_b

        group_deltas = {}
        for group_name, group_feats in FEATURE_GROUPS.items():
            if group_name == "base13":
                continue
            group_idx = [full_feat_names.index(f) for f in group_feats if f in full_feat_names]
            combined_idx = base13_idx + group_idx
            deltas_h = {}
            for h in REWARD_HORIZONS:
                X_tr_g = X_train_full.copy()
                X_vl_g = X_val_full.copy()
                mask_out = [i for i in range(n_full) if i not in combined_idx]
                X_tr_g[:, mask_out] = 0.0
                X_vl_g[:, mask_out] = 0.0
                scaler_g = StandardScaler()
                X_tr_s = scaler_g.fit_transform(X_tr_g)
                X_vl_s = scaler_g.transform(X_vl_g)
                model_g = Ridge(alpha=1.0)
                model_g.fit(X_tr_s, y_train_all[h])
                preds_g = model_g.predict(X_vl_s)
                ic_g, _, _ = compute_ic(preds_g, y_val_all[h])
                deltas_h[h] = ic_g - base13_ics[h]
            group_deltas[group_name] = deltas_h

        print(f"\n  {'Group':<20}", end="")
        for h in REWARD_HORIZONS:
            print(f"  {'dt+' + str(h):>8}", end="")
        print(f"  {'Avg dIC':>8}")
        print(f"  {'-'*20}" + "-" * 10 * (len(REWARD_HORIZONS) + 1))

        for group_name, deltas_h in group_deltas.items():
            avg_delta = np.mean(list(deltas_h.values()))
            verdict = "HELPFUL" if avg_delta > 0.001 else "NEUTRAL" if avg_delta > -0.001 else "HARMFUL"
            print(f"  +{group_name:<19}", end="")
            for h in REWARD_HORIZONS:
                print(f"  {deltas_h[h]:>+8.4f}", end="")
            print(f"  {avg_delta:>+8.4f}  {verdict}")

    # --- Aggregate Summary ---
    print(f"\n{'='*70}")
    print("  AGGREGATE SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Per-Horizon Average IC (across assets):")
    print(f"  {'Horizon':<10} {'Linear IC':>10} {'DL Gate':>10} {'Margin to Gate':>15}")
    print(f"  {'-'*50}")

    # DL gate threshold from CLAUDE.md: IC > 0.015
    dl_gate = 0.015

    for h in REWARD_HORIZONS:
        ics = [results[a][h]["ic"] for a in results]
        avg_linear = np.mean(ics)
        margin = avg_linear - dl_gate
        print(f"  t+{h:<7} {avg_linear:>+10.4f} {dl_gate:>+10.4f} {margin:>+15.4f}")

    print(f"\n  Interpretation:")
    avg_all = np.mean([results[a][h]["ic"] for a in results for h in REWARD_HORIZONS])
    print(f"  - Linear baseline average IC: {avg_all:+.4f}")
    print(f"  - DL validation gate (CLAUDE.md): IC > 0.015")
    if avg_all > 0.020:
        print(f"  - [STRONG] Linear model captures strong signal from features alone")
        print(f"  - DL models must significantly exceed {avg_all:.4f} to justify complexity")
    elif avg_all > 0.010:
        print(f"  - [MODERATE] Linear model captures moderate signal")
        print(f"  - DL models add value through nonlinear + temporal modeling")
    else:
        print(f"  - [WEAK] Linear model captures weak/no signal")
        print(f"  - Either features are weak or nonlinear modeling is essential")

    # Save results to log
    log_file = LOG_DIR / "linear_baseline_results.txt"
    with open(log_file, "w") as f:
        f.write(f"Linear Baseline Results\n")
        f.write(f"{'='*50}\n\n")
        for asset_name in sorted(results.keys()):
            f.write(f"{asset_name}:\n")
            for h in REWARD_HORIZONS:
                r = results[asset_name][h]
                f.write(f"  t+{h}: IC={r['ic']:+.4f} RankIC={r['rank_ic']:+.4f} "
                        f"Dir={r['dir_acc']*100:.1f}% alpha={r['best_alpha']}\n")
            f.write("\n")
        f.write(f"\nAverage IC: {avg_all:+.4f}\n")
    print(f"\n  Results saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Baseline IC Analysis")
    parser.add_argument("--full", action="store_true",
                        help="Use simple 90/10 split without purge gap")
    parser.add_argument(
        "--features", type=int,
        choices=list_supported_features(),
        default=41,
        help=("Feature count to use. Supported counts come from "
              "src/feature_sets.py (post-2026-04-27 centralization). "
              "13/18/21/25/29/30/34/37/41 = v50 schema (legacy chimera). "
              "46/60/73/78/81/84/97/110/121 = v51 frontier cuts (require v51 chimera). "
              "Default: 41 (full v50)."),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=("Per-task parallel workers (default 1 = sequential). "
              "Each worker fits one (asset, horizon) ridge sweep. "
              "Set 4-8 on multi-core machines; sklearn Ridge is small "
              "so process-spawn overhead is meaningful — workers=2 is a "
              "safe sweet spot."),
    )
    args = parser.parse_args()

    if args.full and not args.full:
        pass  # silence linter

    import multiprocessing
    multiprocessing.freeze_support()
    run_linear_baseline(
        use_walk_forward=not args.full,
        n_features=args.features,
        workers=args.workers,
    )
