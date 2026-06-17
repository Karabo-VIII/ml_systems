"""V0 baseline workers — top-level functions for ProcessPoolExecutor.

Worker functions MUST be top-level (importable, picklable) for
ProcessPoolExecutor to dispatch them across worker processes. This module
hosts those workers.

Each worker:
  - Re-imports its needed modules (Windows spawn semantics → fresh interpreter)
  - Caps polars/numpy thread pools to avoid oversubscription with N workers
  - Returns a small picklable dict (no live model objects)
"""
from __future__ import annotations

import os
from typing import Any


def _set_thread_caps(threads: int) -> None:
    """Cap intra-process thread pools (numpy MKL, polars rayon, OpenMP).

    Set BEFORE importing numpy so the BLAS layer respects the cap.
    Idempotent (safe to call multiple times).
    """
    threads = max(1, int(threads))
    s = str(threads)
    for k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, s)


# ─── Ridge sweep worker ──────────────────────────────────────────────────────

def ridge_sweep_worker(args_tuple) -> dict:
    """Fit Ridge for ONE (asset, horizon, alpha-sweep) tuple. Returns best.

    args_tuple = (asset_name, X_train, X_val, y_train, y_val, alphas, threads)
    Returns: {asset, horizon, ic, rank_ic, dir_acc, p_value, best_alpha, n}

    Reads of X_train etc. come via shared-memory pickling (these are np arrays
    sized ~MBs; pickle overhead dominates short fits unless tasks are big).
    """
    asset_name, X_train, X_val, y_train, y_val, alphas, horizon, threads = args_tuple
    _set_thread_caps(threads)

    # Lazy imports inside worker (Windows spawn isolates these)
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy import stats as _stats

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_ic = -np.inf
    best_alpha = 1.0
    best_preds = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_val_s)
        mask = np.isfinite(preds) & np.isfinite(y_val)
        if mask.sum() < 30 or np.std(preds[mask]) < 1e-10 or np.std(y_val[mask]) < 1e-10:
            ic = 0.0
        else:
            ic = float(np.corrcoef(preds[mask], y_val[mask])[0, 1])
        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha
            best_preds = preds

    if best_preds is None:
        return {"asset": asset_name, "horizon": horizon, "ic": 0.0,
                "rank_ic": 0.0, "dir_acc": 0.5, "p_value": 1.0,
                "best_alpha": 1.0, "n": 0}

    mask = np.isfinite(best_preds) & np.isfinite(y_val)
    p, r = best_preds[mask], y_val[mask]
    n = int(mask.sum())
    if n < 30 or np.std(p) < 1e-10 or np.std(r) < 1e-10:
        ic, p_value = 0.0, 1.0
    else:
        ic = float(np.corrcoef(p, r)[0, 1])
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
        p_value = float(2 * (1 - _stats.t.cdf(abs(t_stat), n - 2)))
    rank_ic = float(_stats.spearmanr(p, r).statistic) if n >= 30 else 0.0
    dir_acc = float(np.mean(np.sign(p) == np.sign(r))) if n > 0 else 0.5

    return {
        "asset": asset_name,
        "horizon": horizon,
        "ic": ic,
        "rank_ic": rank_ic,
        "dir_acc": dir_acc,
        "p_value": p_value,
        "best_alpha": float(best_alpha),
        "n": n,
    }


# ─── Drop-one ablation worker ────────────────────────────────────────────────

def ablation_worker(args_tuple) -> dict:
    """Fit Ridge with ONE feature dropped. Returns IC + delta.

    args_tuple = (drop_idx, feat_name, X_train, X_val, y_train, y_val,
                  alpha, baseline_ic, threads)
    """
    (drop_idx, feat_name, X_train, X_val, y_train, y_val,
     alpha, baseline_ic, threads) = args_tuple
    _set_thread_caps(threads)

    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    keep = [i for i in range(X_train.shape[1]) if i != drop_idx]
    scaler = StandardScaler()
    X_ts = scaler.fit_transform(X_train[:, keep])
    X_vs = scaler.transform(X_val[:, keep])

    model = Ridge(alpha=alpha)
    model.fit(X_ts, y_train)
    preds = model.predict(X_vs)

    mask = np.isfinite(preds) & np.isfinite(y_val)
    if mask.sum() < 30 or np.std(preds[mask]) < 1e-10 or np.std(y_val[mask]) < 1e-10:
        ic_abl = 0.0
    else:
        ic_abl = float(np.corrcoef(preds[mask], y_val[mask])[0, 1])
    return {
        "drop_idx": int(drop_idx),
        "feat_name": feat_name,
        "ic_abl": ic_abl,
        "delta": ic_abl - baseline_ic,
    }


# ─── Shuffled IC worker (per asset, per seed, per horizon) ───────────────────

def shuffled_ic_worker(args_tuple) -> dict:
    """Compute shuffled IC for ONE (asset, horizon, seed) tuple.

    args_tuple = (asset_name, horizon, seed, feats_train, targets_h_train,
                  alpha, threads)
    Returns: {asset, horizon, seed, ic_shuffled, n}
    """
    (asset_name, horizon, seed, feats, targets_h, alpha, threads) = args_tuple
    _set_thread_caps(threads)

    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42 + seed * 1000)
    indices = np.arange(len(feats))
    rng.shuffle(indices)

    X_shuf = feats[indices]
    y_shuf = targets_h[indices]
    mid = int(len(X_shuf) * 0.80)

    scaler = StandardScaler()
    X_ts = scaler.fit_transform(X_shuf[:mid])
    X_vs = scaler.transform(X_shuf[mid:])

    model = Ridge(alpha=alpha)
    model.fit(X_ts, y_shuf[:mid])
    preds = model.predict(X_vs)
    y_val = y_shuf[mid:]

    mask = np.isfinite(preds) & np.isfinite(y_val)
    if mask.sum() < 30 or np.std(preds[mask]) < 1e-10 or np.std(y_val[mask]) < 1e-10:
        ic = 0.0
    else:
        ic = float(np.corrcoef(preds[mask], y_val[mask])[0, 1])

    return {
        "asset": asset_name,
        "horizon": horizon,
        "seed": int(seed),
        "ic_shuffled": ic,
        "n": int(mask.sum()),
    }


# ─── Nonlinear (CatBoost/LightGBM/XGB) per-(asset, model, horizon) worker ────

def nonlinear_fit_worker(args_tuple) -> dict:
    """Fit ONE nonlinear model for one (asset, model_key, horizon) tuple.

    args_tuple = (asset_name, model_key, horizon, X_train, y_train, X_val,
                  y_val, threads)
    Tree-ensemble libraries (CatBoost/LightGBM/XGB) have their own internal
    threadpool — we pass `threads` as their n_jobs/thread_count so total
    cores used = workers × threads stays = cpu_count.

    Returns: {asset, model_key, horizon, ic, rank_ic, dir_acc, elapsed_s, ok}
    """
    asset_name, model_key, horizon, X_train, y_train, X_val, y_val, threads = args_tuple
    _set_thread_caps(threads)

    import time as _t
    import numpy as np
    from scipy import stats as _stats

    # Local model builder (must run inside worker so libs use the capped threads)
    def _build_predict(model_key, X_tr, y_tr, X_v):
        if model_key == "poly":
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler
            from sklearn.linear_model import Ridge
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            X_tr_p = poly.fit_transform(X_tr)
            X_v_p = poly.transform(X_v)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_p)
            X_v_s = scaler.transform(X_v_p)
            m = Ridge(alpha=10.0)
            m.fit(X_tr_s, y_tr)
            return m.predict(X_v_s)
        if model_key == "gbt":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            m.fit(X_tr, y_tr)
            return m.predict(X_v)
        if model_key == "mlp":
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_v_s = scaler.transform(X_v)
            m = MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", learning_rate_init=1e-3,
                max_iter=200, batch_size=256,
                early_stopping=True, validation_fraction=0.1,
                random_state=42,
            )
            m.fit(X_tr_s, y_tr)
            return m.predict(X_v_s)
        raise ValueError(f"unknown model_key={model_key}")

    t0 = _t.time()
    try:
        preds = _build_predict(model_key, X_train, y_train, X_val)
    except Exception as e:
        return {
            "asset": asset_name, "model_key": model_key, "horizon": horizon,
            "ic": 0.0, "rank_ic": 0.0, "dir_acc": 0.5,
            "elapsed_s": _t.time() - t0, "ok": False,
            "err": f"{type(e).__name__}: {e}",
        }
    elapsed = _t.time() - t0

    mask = np.isfinite(preds) & np.isfinite(y_val)
    if mask.sum() < 30 or np.std(preds[mask]) < 1e-10 or np.std(y_val[mask]) < 1e-10:
        ic, rank_ic, dir_acc = 0.0, 0.0, 0.5
    else:
        ic = float(np.corrcoef(preds[mask], y_val[mask])[0, 1])
        rank_ic = float(_stats.spearmanr(preds[mask], y_val[mask]).statistic)
        dir_acc = float(np.mean(np.sign(preds[mask]) == np.sign(y_val[mask])))

    return {
        "asset": asset_name, "model_key": model_key, "horizon": horizon,
        "ic": ic, "rank_ic": rank_ic, "dir_acc": dir_acc,
        "elapsed_s": elapsed, "ok": True,
    }
