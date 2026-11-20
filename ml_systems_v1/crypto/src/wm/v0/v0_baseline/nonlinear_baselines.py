"""
Non-Linear Baselines -- Polynomial, GBT, MLP on 13-121 Features

Three non-linear models that define the IC ceiling WITHOUT temporal modeling.
If V1-V19 can't beat these, the sequential architecture adds no value.

Model shapes:
  Polynomial:  y = sum(w_ij * x_i * x_j) + sum(w_i * x_i) + b  (parabolic)
  GBT:         Ensemble of axis-aligned decision trees           (piecewise constant)
  MLP:         2-layer feedforward neural net                    (smooth non-linear)

Post-2026-04-27 upgrade: parallel per-(asset, model, horizon) fits via
ProcessPoolExecutor + central feature_sets registry.

Usage:
    python nonlinear_baselines.py                       # 41 features (default)
    python nonlinear_baselines.py --features 13         # legacy base
    python nonlinear_baselines.py --features 121        # full v51 frontier
    python nonlinear_baselines.py --model poly          # Only polynomial
    python nonlinear_baselines.py --workers 4           # parallel fits
    python nonlinear_baselines.py --full                # simple 90/10 (no purge gap)
"""
import numpy as np
import sys
import argparse
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # sklearn convergence
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import (
    FEATURE_LIST, FEATURE_LIST_13, FEATURE_LIST_18, FEATURE_LIST_21,
    FEATURE_LIST_25, FEATURE_LIST_30, FEATURE_LIST_34, FEATURE_LIST_37,
    FEATURE_LIST_41, FEATURE_LIST_121,
    ASSET_TO_IDX, REWARD_HORIZONS, PURGE_GAP_BARS, N_SHUFFLE_SEEDS, LOG_DIR,
    # TRAIN_RATIO/VAL_RATIO replaced by date-based splits via _load_split_boundaries
    get_feature_config, list_supported_features,
)
from linear_baseline import (
    load_data, compute_ic, compute_rank_ic, _resolve_polars_threads,
    _load_split_boundaries, get_dated_split_indices,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from _workers import nonlinear_fit_worker
import os


# ─── Model Builders ─────────────────────────────────────────────────────────

def build_polynomial(X_train, y_train, X_val, degree=2):
    """Polynomial Ridge: parabolic interactions between features."""
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    scaler = StandardScaler()

    # Subsample if large (poly expansion is O(features^2) columns)
    max_train = 200_000
    if len(X_train) > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    X_tp = poly.fit_transform(X_sub)
    X_vp = poly.transform(X_val)
    X_ts = scaler.fit_transform(X_tp)
    X_vs = scaler.transform(X_vp)

    # alpha=100: high-dim polynomial needs strong regularization
    model = Ridge(alpha=100.0)
    model.fit(X_ts, y_sub)
    return model.predict(X_vs)


def build_gbt(X_train, y_train, X_val):
    """Gradient Boosted Trees: histogram-based (10-50x faster than classic GBT)."""
    scaler = StandardScaler()

    # Subsample training data (HistGBT is faster but still O(n * trees))
    max_train = 200_000
    if len(X_train) > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    X_ts = scaler.fit_transform(X_sub)
    X_vs = scaler.transform(X_val)

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=50,
        max_bins=128,
        random_state=42,
    )
    model.fit(X_ts, y_sub)
    return model.predict(X_vs)


def build_mlp(X_train, y_train, X_val):
    """MLP: 2-layer feedforward network (smooth non-linear)."""
    scaler = StandardScaler()

    # Subsample if large (MLP training is O(n * hidden * epochs))
    max_train = 200_000
    if len(X_train) > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    X_ts = scaler.fit_transform(X_sub)
    X_vs = scaler.transform(X_val)

    # Cap hidden layers: baseline doesn't need huge capacity
    n_feat = X_ts.shape[1]
    h1 = min(n_feat * 4, 64)
    h2 = min(n_feat * 2, 32)
    model = MLPRegressor(
        hidden_layer_sizes=(h1, h2),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=512,
        random_state=42,
    )
    model.fit(X_ts, y_sub)
    return model.predict(X_vs)


MODEL_BUILDERS = {
    "poly": ("Polynomial Ridge (degree=2)", build_polynomial),
    "gbt":  ("Gradient Boosted Trees",      build_gbt),
    "mlp":  ("MLP (2-layer feedforward)",    build_mlp),
}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_nonlinear_baselines(use_walk_forward=True, n_features=41, models=None,
                              workers: int = 1):
    """
    Train non-linear baselines per-(asset, model, horizon) — parallelizable.

    Args:
        n_features: any from feature_sets.SUPPORTED_COUNTS
        models: list of model keys to run, or None for all
        workers: ProcessPoolExecutor max_workers (default 1 = sequential).
    """
    # Resolve via central registry (post-2026-04-27 centralization)
    feature_list, input_dim, _base_dim = get_feature_config(n_features)
    use_v51 = n_features > 41

    if models is None:
        models = list(MODEL_BUILDERS.keys())

    print("=" * 70)
    print(f"  NON-LINEAR BASELINES ({input_dim} Features)")
    print(f"  Models: {', '.join(models)}")
    print(f"  Source: {'v51 chimera (frontier)' if use_v51 else 'v50 chimera (legacy)'}")
    print(f"  Workers: {workers}")
    print(f"  Purpose: IC ceiling WITHOUT temporal modeling")
    print("=" * 70)

    all_data = load_data(feature_list, use_v51=use_v51)
    if not all_data:
        print("  [ERROR] No data found.")
        return

    # Calendar-aligned 4-way split: dates frozen in config/data_config.yaml
    # (same boundaries V1.x training uses — apples-to-apples IC ceiling)
    boundaries = _load_split_boundaries()
    print(f"\n  Split: train_end={boundaries['train_end_ms']}, "
          f"val_end={boundaries['val_end_ms']}, "
          f"oos_end={boundaries['oos_end_ms']} (purge {boundaries['purge_bars']} bars)")

    # {model_key: {asset: {horizon: {ic, rank_ic, dir_acc}}}}
    all_results = {m: {} for m in models}
    polars_threads = _resolve_polars_threads(workers)

    # Build (asset, model, horizon) task list
    tasks = []
    for feats, targets, asset_name, ts in all_data:
        if ts is None:
            raise ValueError(f"{asset_name}: missing timestamp column")
        idx = get_dated_split_indices(ts, boundaries)
        train_end, val_start, val_end = idx["train_end"], idx["val_start"], idx["val_end"]
        if not use_walk_forward:
            val_start = train_end
        if val_start >= val_end:
            val_start = train_end

        X_train = feats[:train_end]
        X_val = feats[val_start:val_end]

        # Subsample large assets for poly/MLP (workers retain 200K row cap internally)
        for model_key in models:
            all_results[model_key][asset_name] = {}
            for h in REWARD_HORIZONS:
                y_train = targets[h][:train_end]
                y_val = targets[h][val_start:val_end]
                tasks.append((asset_name, model_key, int(h), X_train, y_train,
                              X_val, y_val, polars_threads))

    print(f"\n  Running {len(tasks)} fits "
          f"({len(all_data)} assets x {len(models)} models x {len(REWARD_HORIZONS)} horizons)")

    out = []
    if workers <= 1:
        for t in tasks:
            out.append(nonlinear_fit_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(nonlinear_fit_worker, t) for t in tasks]
            for fut in as_completed(futs):
                out.append(fut.result())

    for r in out:
        a, m, h = r["asset"], r["model_key"], r["horizon"]
        all_results[m][a][h] = {"ic": r["ic"], "rank_ic": r["rank_ic"],
                                 "dir_acc": r["dir_acc"]}
        if r.get("ok", True):
            print(f"    [{m:<4}] {a:<10} t+{h:<3} IC:{r['ic']:+.4f} "
                  f"RankIC:{r['rank_ic']:+.4f} Dir:{r['dir_acc']*100:.1f}% "
                  f"({r['elapsed_s']:.1f}s)")
        else:
            print(f"    [{m:<4}] {a:<10} t+{h:<3} FAILED: {r.get('err', '')}")

    # ─── Shuffled IC ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SHUFFLED IC (anti-memorization check, h=1 only)")
    print(f"{'='*70}")

    # ShIC only needs h=1 (the generalizing horizon) and 3 seeds is sufficient
    shic_seeds = min(N_SHUFFLE_SEEDS, 3)
    # Use 3 highest-volume assets for speed (BTC, ETH, SOL)
    shic_assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    shic_data = [(f, t, a, ts) for f, t, a, ts in all_data if a in shic_assets]
    if not shic_data:
        shic_data = all_data[:3]  # fallback

    for model_key in models:
        _, builder = MODEL_BUILDERS[model_key]
        shuffled_ics = []
        for seed in range(shic_seeds):
            rng = np.random.default_rng(42 + seed * 1000)
            s_preds_all, s_reals_all = [], []

            for feats, targets, _, ts in shic_data:
                split = get_dated_split_indices(ts, boundaries)["train_end"]
                indices = np.arange(split)
                rng.shuffle(indices)

                X_shuf = feats[indices]
                y_shuf = targets[1][indices]

                mid = int(len(X_shuf) * 0.80)
                try:
                    preds_sh = builder(X_shuf[:mid], y_shuf[:mid], X_shuf[mid:])
                    s_preds_all.extend(preds_sh)
                    s_reals_all.extend(y_shuf[mid:])
                except Exception:
                    pass

            if s_preds_all:
                ic_sh, _, _ = compute_ic(np.array(s_preds_all), np.array(s_reals_all))
                shuffled_ics.append(ic_sh)

        if shuffled_ics:
            mean_shic = np.mean(shuffled_ics)
            print(f"  [{model_key:<4}] t+1  Shuffled IC: {mean_shic:+.4f} "
                  f"(avg of {len(shuffled_ics)} seeds, {len(shic_data)} assets)")

    # ─── Comparison Table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE (Average IC Across Assets)")
    print(f"{'='*70}")

    dl_gate = 0.015  # from CLAUDE.md

    header = f"  {'Model':<28}"
    for h in REWARD_HORIZONS:
        header += f"  {'t+' + str(h):>6}"
    header += f"  {'Avg':>6}"
    print(header)
    print(f"  {'-'*68}")

    for model_key in models:
        model_name = MODEL_BUILDERS[model_key][0]
        row = f"  {model_name:<28}"
        h_avgs = []
        for h in REWARD_HORIZONS:
            ics = [all_results[model_key][a][h]["ic"]
                   for a in all_results[model_key]
                   if h in all_results[model_key][a]]
            avg = np.mean(ics) if ics else 0.0
            h_avgs.append(avg)
            row += f"  {avg:+.4f}"
        overall = np.mean(h_avgs)
        row += f"  {overall:+.4f}"
        print(row)

    print(f"  {'DL Gate (CLAUDE.md)':<28}", end="")
    for _ in REWARD_HORIZONS:
        print(f"  {dl_gate:+.4f}", end="")
    print(f"  {dl_gate:+.4f}")

    # ─── Direction Accuracy Table ──────────────────────────────────────────
    print(f"\n  {'Model':<28}", end="")
    for h in REWARD_HORIZONS:
        print(f"  {'t+' + str(h):>6}", end="")
    print(f"  {'Avg':>6}")
    print(f"  {'-'*68}")

    for model_key in models:
        model_name = MODEL_BUILDERS[model_key][0]
        row = f"  {model_name:<28}"
        h_avgs = []
        for h in REWARD_HORIZONS:
            accs = [all_results[model_key][a][h]["dir_acc"]
                    for a in all_results[model_key]
                    if h in all_results[model_key][a]]
            avg = np.mean(accs) if accs else 0.5
            h_avgs.append(avg)
            row += f"  {avg*100:5.1f}%"
        overall = np.mean(h_avgs)
        row += f"  {overall*100:5.1f}%"
        print(row)

    # ─── Interpretation ───────────────────────────────────────────────────
    print(f"\n  Interpretation:")
    for model_key in models:
        model_name = MODEL_BUILDERS[model_key][0]
        all_ics = [all_results[model_key][a][h]["ic"]
                   for a in all_results[model_key]
                   for h in REWARD_HORIZONS
                   if h in all_results[model_key][a]]
        avg = np.mean(all_ics) if all_ics else 0.0
        if avg > 0.030:
            verdict = "STRONG non-linear signal. DL temporal models must beat this."
        elif avg > 0.015:
            verdict = "Moderate non-linear signal. DL adds temporal context."
        elif avg > 0.005:
            verdict = "Weak signal. Non-linear helps marginally."
        else:
            verdict = "Negligible. Non-linear fits noise, not signal."
        print(f"  - {model_name}: avg IC = {avg:+.4f} -- {verdict}")

    # ─── Save results (per-model file to avoid overwrite) ────────────────
    model_tag = "_".join(models)
    log_file = LOG_DIR / f"nonlinear_{model_tag}_f{input_dim}_results.txt"
    with open(log_file, "w") as f:
        f.write("Non-Linear Baseline Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Features: {input_dim}\n\n")
        for model_key in models:
            model_name = MODEL_BUILDERS[model_key][0]
            f.write(f"\n{model_name}\n{'-'*40}\n")
            for asset_name in sorted(all_results[model_key].keys()):
                f.write(f"  {asset_name}:\n")
                for h in REWARD_HORIZONS:
                    if h in all_results[model_key][asset_name]:
                        r = all_results[model_key][asset_name][h]
                        f.write(f"    t+{h}: IC={r['ic']:+.4f} RankIC={r['rank_ic']:+.4f} "
                                f"Dir={r['dir_acc']*100:.1f}%\n")
    print(f"\n  Results saved to {log_file}")

    # Also append to combined log (accumulates across runs)
    combined_log = LOG_DIR / "nonlinear_baselines_results.txt"
    with open(combined_log, "a") as f:
        f.write(f"\n\n{'='*60}\n")
        f.write(f"Features: {input_dim} | Models: {', '.join(models)}\n")
        f.write(f"{'='*60}\n")
        for model_key in models:
            model_name = MODEL_BUILDERS[model_key][0]
            f.write(f"\n{model_name}\n{'-'*40}\n")
            for asset_name in sorted(all_results[model_key].keys()):
                f.write(f"  {asset_name}:\n")
                for h in REWARD_HORIZONS:
                    if h in all_results[model_key][asset_name]:
                        r = all_results[model_key][asset_name][h]
                        f.write(f"    t+{h}: IC={r['ic']:+.4f} RankIC={r['rank_ic']:+.4f} "
                                f"Dir={r['dir_acc']*100:.1f}%\n")
    print(f"  Also appended to {combined_log}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Non-Linear Baseline IC Analysis")
    parser.add_argument("--full", action="store_true",
                        help="Use simple 90/10 split without purge gap")
    parser.add_argument("--features", type=int,
                        choices=list_supported_features(),
                        default=41,
                        help="Feature count from src/feature_sets.py registry "
                             "(13/18/21/25/29/30/34/37/41 = v50; 46-121 = v51 frontier).")
    parser.add_argument("--model", type=str, choices=["poly", "gbt", "mlp"], default=None,
                        help="Run only one model type (default: all 3)")
    parser.add_argument("--workers", type=int, default=1,
                        help="ProcessPoolExecutor workers (default 1). Tree models "
                             "use internal threads — each worker gets cpu//workers threads "
                             "automatically. Set 2-4 on multi-core; subsample caps per "
                             "builder still apply (200K rows).")
    args = parser.parse_args()

    models = [args.model] if args.model else None
    run_nonlinear_baselines(
        use_walk_forward=not args.full,
        n_features=args.features,
        models=models,
        workers=args.workers,
    )
