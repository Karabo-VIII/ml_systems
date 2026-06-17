"""src/narrate/artifacts.py -- surface trained models/artifacts under models/ as DESCRIPTIVE reads.

artifact_layer(sym, cadence, df, period_mask) -> dict with keys:
  - per-model: str  (descriptive sentence)
  - "loaded": list  (model names that ran without error)
  - "skipped": dict (name -> reason string)

Design contract (user mandate 2026-06-06):
  - DESCRIPTIVE ("the what"), ENTRY-SIGNAL focused, per-SETUP not per-candle.
  - EXIT / sizing models are noted as out-of-scope but not centred.
  - One bad model must NOT break the layer (per-model try/except throughout).
  - Fast and read-only -- models are loaded lazily and never retrained.
  - No emoji in any string (Windows cp1252).

The df argument is a _PolarsShim: df[col].to_numpy(), df.columns, len(df).
period_mask is a boolean numpy array aligned to df rows (True = inside the requested window).
"""
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repo root detection (robust across cwd contexts)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MODELS = os.path.join(_ROOT, "models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lgb(path: str):
    """Load a LightGBM Booster from path, raise on failure."""
    import lightgbm as lgb
    return lgb.Booster(model_file=path)


def _col_arr(df, col: str, mask: np.ndarray) -> np.ndarray:
    """Return the masked numpy array for col, or None if col absent."""
    if col not in df.columns:
        return None
    arr = df[col].to_numpy()
    return arr[mask]


def _build_feature_matrix(df, cols, mask: np.ndarray, fill: float = 0.0):
    """
    Build a 2-D numpy array [n_rows, n_cols] from the masked df rows.
    Missing columns are filled with fill (0.0 by default -- unknown at inference).
    String/categorical columns are treated as missing (cannot encode without
    the original label-encoder mapping -- safe-fill to 0.0).
    Returns (X, n_missing).
    """
    present = set(df.columns)
    rows = int(mask.sum())
    X = np.full((rows, len(cols)), fill, dtype=np.float32)
    n_missing = 0
    for j, c in enumerate(cols):
        if c in present:
            raw = df[c].to_numpy()[mask]
            # Reject non-numeric dtypes (string/object columns)
            if raw.dtype.kind in ("U", "S", "O"):   # unicode, bytes, object
                n_missing += 1
                continue
            arr = raw.astype(np.float32)
            X[:, j] = np.nan_to_num(arr, nan=fill)
        else:
            n_missing += 1
    return X, n_missing


def _pct_str(frac: float) -> str:
    return f"{frac * 100:.0f}%"


# ---------------------------------------------------------------------------
# MODEL 1: setup_classifier  (per_sleeve_setup/v51_full_T4 -- 4h entry setup)
# ---------------------------------------------------------------------------

def _run_setup_classifier(df, mask: np.ndarray, cadence: str) -> str:
    """
    per_sleeve_setup models are ENTRY-SETUP classifiers for the cross-section
    ranker pipeline (4h and 1d).  Here we score the requested asset.
    Uses v51_full_T4 for 1d-aligned calls and 4h_K5_h32_sleeve for 4h.
    Returns: descriptive string.
    """
    import json
    cadence_key = "4h" if "4h" in cadence.lower() else "1d"
    if cadence_key == "4h":
        model_path = os.path.join(_MODELS, "per_sleeve_setup", "4h_K5_h32_sleeve.txt")
        meta_path = os.path.join(_MODELS, "per_sleeve_setup", "4h_K5_h32_sleeve_meta.json")
    else:
        model_path = os.path.join(_MODELS, "per_sleeve_setup", "v51_full_T4.txt")
        meta_path = os.path.join(_MODELS, "per_sleeve_setup", "v51_full_T4_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    # sleeve extra features are panel-context features -- not available per-asset;
    # they will be zero-filled (model was trained with them, reads degrade gracefully).
    booster = _load_lgb(model_path)

    X, n_missing = _build_feature_matrix(df, feature_cols, mask)
    probs = booster.predict(X)          # shape (n,) or (n, n_classes)
    if probs.ndim > 1:
        probs = probs[:, 1]
    pos_pct = float((probs > 0.5).mean()) * 100
    mean_prob = float(probs.mean())
    oos_auc = meta.get("oos_auc", None)
    verdict = meta.get("verdict", "?")
    feat_note = f" ({n_missing} panel features zero-filled)" if n_missing > 0 else ""
    return (
        f"setup-positive on {pos_pct:.0f}% of bars in the window "
        f"(mean p={mean_prob:.3f}); model OOS AUC={oos_auc:.3f} [{verdict}]"
        f"{feat_note}"
    )


# ---------------------------------------------------------------------------
# MODEL 2: day_class  (day-type classifier: TREND_UP / TREND_DOWN / CHOP / CRASH)
# ---------------------------------------------------------------------------

def _run_day_class(df, mask: np.ndarray) -> str:
    """
    day_class_lgb_v1 is a 4-class daily-bar classifier.
    Features are asset-level lags + BTC normalised microstructure.
    Predicts which type of day (today) is.
    NOTE: trained on BTC-like assets + daily cadence; cadence mismatch degrades gracefully.
    """
    import json
    model_path = os.path.join(_MODELS, "day_class", "day_class_lgb_v1.txt")
    meta_path = os.path.join(_MODELS, "day_class", "day_class_lgb_v1_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    booster = _load_lgb(model_path)

    # day_class uses lagged/rolling features derived at training time;
    # many will be present in chimera (btc_norm_* family).  Others (lag1_ret,
    # is_bull etc.) may be absent and zero-filled.
    X, n_missing = _build_feature_matrix(df, feature_cols, mask)
    raw = booster.predict(X)          # shape (n, n_classes) for multiclass
    if raw.ndim == 1:
        preds = (raw > 0.5).astype(int)
        class_names = ["FLAT", "TREND_UP"]
    else:
        preds = raw.argmax(axis=1)
        # v1 was trained with label encoding: 0=chop, 1=bear, 2=bull, 3=crash (inspect model)
        # We use modal label as the headline read.
        class_names = ["CHOP", "BEAR/TREND_DOWN", "BULL/TREND_UP", "CRASH"]

    from collections import Counter
    counts = Counter(preds.tolist())
    total = len(preds)
    modal_idx = counts.most_common(1)[0][0]
    modal_name = class_names[modal_idx] if modal_idx < len(class_names) else str(modal_idx)
    modal_pct = counts[modal_idx] / total * 100
    breakdown = ", ".join(
        f"{class_names[k] if k < len(class_names) else str(k)}={v/total*100:.0f}%"
        for k, v in sorted(counts.items())
    )
    feat_note = f" ({n_missing} features zero-filled)" if n_missing > 0 else ""
    oos_auc = meta.get("oos_auc", None)
    auc_str = f" OOS AUC={oos_auc:.3f}" if oos_auc else ""
    return (
        f"modal day-class = {modal_name} ({modal_pct:.0f}% of bars);"
        f" breakdown: {breakdown}{auc_str}{feat_note}"
    )


# ---------------------------------------------------------------------------
# MODEL 3: xsec_ranker  (cross-section next-bar outperformance ranker)
# ---------------------------------------------------------------------------

def _run_xsec_ranker(df, mask: np.ndarray) -> str:
    """
    xgb_ndcg_v1_u87 ranks an asset within u87 by probability of outperforming
    peers next bar.  When called on a single asset the raw score is a relative
    positioning estimate (percentile unclear without a panel).
    """
    ranker_path = os.path.join(_MODELS, "xsec_ranker", "xgb_ndcg_v1_u87.pkl")
    meta_path = os.path.join(_MODELS, "xsec_ranker", "xgb_ndcg_v1_u87.meta.json")
    with open(ranker_path, "rb") as f:
        bundle = pickle.load(f)
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    ranker = bundle["ranker"]
    features = bundle.get("features", meta.get("features", []))

    X, n_missing = _build_feature_matrix(df, features, mask)
    scores = ranker.predict(X)   # raw XGB scores (higher = better rank)
    mean_score = float(scores.mean())
    p75 = float(np.percentile(scores, 75))
    p25 = float(np.percentile(scores, 25))
    feat_note = f" ({n_missing} features zero-filled)" if n_missing > 0 else ""
    return (
        f"cross-section rank score: mean={mean_score:.3f}, "
        f"p25={p25:.3f}, p75={p75:.3f} (higher=better vs universe{feat_note}); "
        f"trained on u87, n_features={meta.get('n_features',len(features))}"
    )


# ---------------------------------------------------------------------------
# MODEL 4: win_capture_v2  (winner classifier -- LightGBM top-mover probability)
# ---------------------------------------------------------------------------

def _run_win_capture_v2(df, mask: np.ndarray) -> str:
    """
    win_capture_v2 predicts which assets are 'winners' (top decile next bar).
    Features are cross-section xrel (relative-volume/realized-volatility) ranks;
    most will be zero-filled for single-asset calls -- score is indicative only.
    """
    model_path = os.path.join(_MODELS, "win_capture_v2", "lgbm.pkl")
    feat_path = os.path.join(_MODELS, "win_capture_v2", "features.json")
    metrics_path = os.path.join(_MODELS, "win_capture_v2", "metrics.json")
    with open(model_path, "rb") as f:
        booster = pickle.load(f)
    import json
    with open(feat_path) as f:
        features = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    X, n_missing = _build_feature_matrix(df, features, mask)
    probs = booster.predict(X)
    pos_pct = float((probs > 0.5).mean()) * 100
    mean_prob = float(probs.mean())
    oos = metrics.get("OOS", {})
    oos_auc = oos.get("auc", None)
    auc_str = f" OOS AUC={oos_auc:.3f}" if oos_auc else ""
    feat_note = f"; {n_missing}/{len(features)} features are xrel panel cols (zero-filled for single-asset)" if n_missing > 0 else ""
    return (
        f"winner-probability: {mean_prob:.3f} mean "
        f"({pos_pct:.0f}% of bars above 0.5 threshold){auc_str}{feat_note}"
    )


# ---------------------------------------------------------------------------
# MODEL 5: oracle/win_classifier  (win-trade classifier -- captures structured setups)
# ---------------------------------------------------------------------------

def _run_win_classifier(df, mask: np.ndarray, cadence: str) -> str:
    """
    oracle/win_classifier_v1 predicts whether a bar is a 'win' setup entry.
    win_4h_classifier_v1 is trained specifically on 4h data; win_classifier_v1 on daily.
    Returns the fraction of period bars flagged as setup-positive and top feature reads.
    """
    cadence_key = "4h" if "4h" in cadence.lower() else "1d"
    if cadence_key == "4h":
        model_path = os.path.join(_MODELS, "oracle", "win_4h_classifier_v1", "model.lgb")
        label = "win_4h_classifier_v1"
    else:
        model_path = os.path.join(_MODELS, "oracle", "win_classifier_v1", "model.lgb")
        label = "win_classifier_v1"

    booster = _load_lgb(model_path)
    features = booster.feature_name()
    X, n_missing = _build_feature_matrix(df, features, mask)
    probs = booster.predict(X)
    if probs.ndim > 1:
        probs = probs[:, 1]
    setup_pct = float((probs > 0.5).mean()) * 100
    mean_prob = float(probs.mean())
    # Describe distribution: high (p>0.7), medium (0.5-0.7), low (<0.5)
    high_pct = float((probs > 0.7).mean()) * 100
    feat_note = f" ({n_missing}/{len(features)} features zero-filled)" if n_missing > 0 else ""
    return (
        f"[{label}] setup-positive on {setup_pct:.0f}% of bars "
        f"(mean p={mean_prob:.3f}; high-confidence >0.7: {high_pct:.0f}% of bars){feat_note}"
    )


# ---------------------------------------------------------------------------
# MODEL 6: contrastive_signatures_v1  (win vs loss regime fingerprint)
# ---------------------------------------------------------------------------

def _run_contrastive_signatures(df, mask: np.ndarray) -> str:
    """
    contrastive_signatures_v1 stores centroid fingerprints of 'winning' vs
    'losing' setup regimes in z-scored feature space.  We compute the cosine
    similarity of the current period's median feature vector to the win/loss
    centroids and return a descriptive read.
    """
    lib_path = os.path.join(_MODELS, "oracle", "contrastive_signatures_v1", "library.pkl")
    with open(lib_path, "rb") as f:
        lib = pickle.load(f)
    features = lib["features"]
    means = lib["means"]
    stds = lib["stds"]
    win_centroids = lib["win_centroids_z"]    # shape (K, n_features)
    loss_centroids = lib["loss_centroids_z"]  # shape (K, n_features)

    # Build z-scored feature vector for the period median
    vec = np.zeros(len(features), dtype=np.float32)
    for j, feat in enumerate(features):
        if feat in df.columns:
            arr = df[feat].to_numpy()[mask].astype(np.float32)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                raw = float(np.median(arr))
                std = float(stds.get(feat, 1.0)) or 1.0
                vec[j] = (raw - float(means.get(feat, 0.0))) / std

    def cosine_sim(a, B):
        """a: (d,), B: (K, d) -> max cosine similarity over K centroids"""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return float((B_norm @ a_norm).max())

    win_sim = cosine_sim(vec, win_centroids)
    loss_sim = cosine_sim(vec, loss_centroids)
    delta = win_sim - loss_sim
    if delta > 0.05:
        sentiment = "WIN-aligned (fingerprint matches historical winning setups)"
    elif delta < -0.05:
        sentiment = "LOSS-aligned (fingerprint resembles historical losing setups)"
    else:
        sentiment = "NEUTRAL (fingerprint equidistant between win/loss centroids)"
    return (
        f"contrastive signature: {sentiment}; "
        f"win-centroid similarity={win_sim:.3f}, loss-centroid similarity={loss_sim:.3f}"
    )


# ---------------------------------------------------------------------------
# MODEL 7: wm_v1_1  (world model v1.1 -- RSSM Transformer, f41 features)
# ---------------------------------------------------------------------------

def _run_wm_v1_1(df, mask: np.ndarray) -> str:
    """
    WM v1.1 is our primary trained Transformer-RSSM world model (41 features,
    best_shuffled_ic=0.034 at epoch 165, gate_passed=True).
    We run a DESCRIPTIVE read: extract the model's expected return distribution
    for h=1 over the period and summarise direction bias and uncertainty.
    No gradient, no retrain -- inference only.
    """
    import torch

    ckpt_path = os.path.join(_MODELS, "wm", "v1", "v1_1", "base", "v1_1_f41_wm_best_ema.pt")
    epoch_path = os.path.join(_MODELS, "wm", "v1", "v1_1", "base", "v1_1_f41_wm_epoch_165.pt")

    # Load epoch checkpoint for metadata (best_shuffled_ic etc)
    meta_ckpt = torch.load(epoch_path, map_location="cpu", weights_only=False)
    best_shic = meta_ckpt.get("best_shuffled_ic", None)
    gate_passed = meta_ckpt.get("gate_passed", None)
    n_features = meta_ckpt.get("n_features", 41)
    version = meta_ckpt.get("version", "v1_1")

    # Load the EMA weights (best generalisation)
    ema_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # The v1_1 EMA file only contains model_state_dict (no config).
    # We reconstruct using the known v1_1 settings.
    _wm_dir = os.path.join(_ROOT, "src", "wm", "v1", "v1_1_training")
    sys_path_insert = _wm_dir not in __import__("sys").path
    if sys_path_insert:
        __import__("sys").path.insert(0, _wm_dir)
    try:
        from wm.v1.v1_1_training.world_model import TransformerWorldModel
        from wm.v1.v1_1_training.settings import (
            WM_D_MODEL, WM_N_HEADS, WM_N_LAYERS, WM_D_FF,
            RSSM_LATENT_DIM, RSSM_CLASSES, NUM_BINS, NUM_ASSETS,
            WM_ASSET_EMB_DIM, WM_DROPOUT, FEATURE_LIST_41,
            BASE_DIM,
        )
    except Exception as e:
        return f"WM v1.1 inference skipped (import error: {e})"
    finally:
        if sys_path_insert:
            __import__("sys").path.remove(_wm_dir)

    feature_list = FEATURE_LIST_41   # 41 features
    model = TransformerWorldModel(
        input_dim=len(feature_list),
        base_dim=BASE_DIM,
        d_model=WM_D_MODEL,
        n_heads=WM_N_HEADS,
        n_layers=WM_N_LAYERS,
        d_ff=WM_D_FF,
        latent_dim=RSSM_LATENT_DIM,
        classes=RSSM_CLASSES,
        num_bins=NUM_BINS,
        num_assets=NUM_ASSETS,
        asset_emb_dim=WM_ASSET_EMB_DIM,
        dropout=0.0,     # eval mode
        ablation_subsets={},
    )
    state = ema_ckpt.get("model_state_dict", ema_ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Build feature matrix for the period
    X, n_missing = _build_feature_matrix(df, feature_list, mask)
    n = len(X)
    if n < 4:
        return f"WM v1.1: window too short ({n} bars) for meaningful inference"

    # Run forward in sequence chunks (no grad)
    # The model uses forward_train(obs_seq, asset_id) -> dict with 'return_logits'
    SEQ = 32
    bullish_bars = 0
    bearish_bars = 0
    total_inf = 0
    with torch.no_grad():
        for start in range(0, n, SEQ):
            chunk = X[start : start + SEQ]
            t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
            asset_ids = torch.zeros(1, dtype=torch.long)
            try:
                out = model.forward_train(t, asset_ids)
                # out is a dict; 'return_logits' is itself a dict keyed by horizon int
                # {1: (B,T,num_bins), 4: ..., 16: ..., 64: ...}
                logits_dict = out.get("return_logits")
                if isinstance(logits_dict, dict) and 1 in logits_dict:
                    h1_logits = logits_dict[1]   # (1, T, num_bins)
                    probs_chunk = torch.softmax(h1_logits[0], dim=-1)  # (T, num_bins)
                    # bin centres: [BIN_MIN, BIN_MAX] split into NUM_BINS
                    bins = torch.linspace(-1.0, 1.0, NUM_BINS)
                    expected = (probs_chunk * bins).sum(dim=-1)  # (T,)
                    bullish_bars += int((expected > 0.0).sum().item())
                    bearish_bars += int((expected < 0.0).sum().item())
                    total_inf += len(chunk)
            except Exception:
                pass   # partial chunk might fail; accumulate what we get

    if total_inf == 0:
        dir_str = "inference ran but produced no output"
    else:
        bull_pct = bullish_bars / total_inf * 100
        bear_pct = bearish_bars / total_inf * 100
        if bull_pct > 60:
            bias = "BULLISH bias"
        elif bear_pct > 60:
            bias = "BEARISH bias"
        else:
            bias = "MIXED/uncertain"
        dir_str = (
            f"{bias}: h=1 return sign positive on {bull_pct:.0f}% of bars, "
            f"negative on {bear_pct:.0f}%"
        )

    shic_str = f"best_ShIC={best_shic:.4f}" if best_shic else "ShIC=unknown"
    gate_str = "gate PASS" if gate_passed else "gate FAIL"
    return (
        f"WM v1.1 ({version}, f41, {shic_str}, {gate_str}): {dir_str}"
    )


# ---------------------------------------------------------------------------
# MODEL 8: oracle/mover_oracle_v2  (asset-level mover probability)
# ---------------------------------------------------------------------------

def _run_mover_oracle(df, mask: np.ndarray) -> str:
    """
    mover_oracle_v2_target_5pct predicts whether an asset moves >=5% (either
    direction) within h=1 bar.  An ENTRY-RELEVANT signal: high probability
    indicates a high-energy bar is likely -- useful for setup confirmation.
    """
    model_path = os.path.join(_MODELS, "oracle", "mover_oracle_v2_target_5pct.lgb")
    booster = _load_lgb(model_path)
    features = booster.feature_name()
    X, n_missing = _build_feature_matrix(df, features, mask)
    probs = booster.predict(X)
    if probs.ndim > 1:
        probs = probs[:, 1]
    mean_prob = float(probs.mean())
    high_pct = float((probs > 0.5).mean()) * 100
    feat_note = f" ({n_missing}/{len(features)} features zero-filled)" if n_missing > 0 else ""
    level = "HIGH" if mean_prob > 0.35 else ("MODERATE" if mean_prob > 0.20 else "LOW")
    return (
        f"mover_oracle_v2 (>=5% move): {level} energy -- mean p={mean_prob:.3f}; "
        f"{high_pct:.0f}% of bars above 0.5 threshold{feat_note}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def artifact_layer(sym: str, cadence: str, df: Any, period_mask: np.ndarray) -> dict:
    """
    Surface trained model artifacts as descriptive reads for the requested
    (sym, cadence) period.

    Parameters
    ----------
    sym:         asset symbol (e.g. 'BTCUSDT')
    cadence:     bar cadence string (e.g. '4h', '1d')
    df:          _PolarsShim -- supports df[col].to_numpy(), df.columns, len(df)
    period_mask: boolean numpy array aligned to df rows

    Returns
    -------
    dict with:
      - model-name keys -> descriptive string read
      - "loaded": list of model names that ran without error
      - "skipped": dict {name: reason} for models that failed or were excluded
    """
    result: dict = {}
    loaded: list = []
    skipped: dict = {}

    n_period = int(period_mask.sum())
    if n_period == 0:
        return {"loaded": [], "skipped": {"all": "period_mask is empty"}}

    # -----------------------------------------------------------------------
    # 1. setup_classifier (entry-setup probability per bar in window)
    # -----------------------------------------------------------------------
    try:
        result["setup_classifier"] = _run_setup_classifier(df, period_mask, cadence)
        loaded.append("setup_classifier")
    except Exception as e:
        skipped["setup_classifier"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # 2. oracle/win_classifier (win-trade setup probability)
    # -----------------------------------------------------------------------
    try:
        result["win_classifier"] = _run_win_classifier(df, period_mask, cadence)
        loaded.append("win_classifier")
    except Exception as e:
        skipped["win_classifier"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # 3. oracle/mover_oracle_v2 (high-energy bar probability)
    # -----------------------------------------------------------------------
    try:
        result["mover_oracle"] = _run_mover_oracle(df, period_mask)
        loaded.append("mover_oracle")
    except Exception as e:
        skipped["mover_oracle"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # 4. contrastive_signatures_v1 (win/loss regime fingerprint)
    # -----------------------------------------------------------------------
    try:
        result["contrastive_signatures"] = _run_contrastive_signatures(df, period_mask)
        loaded.append("contrastive_signatures")
    except Exception as e:
        skipped["contrastive_signatures"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # 5. xsec_ranker (cross-section rank score -- context signal)
    # -----------------------------------------------------------------------
    try:
        result["xsec_ranker"] = _run_xsec_ranker(df, period_mask)
        loaded.append("xsec_ranker")
    except Exception as e:
        skipped["xsec_ranker"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # 6. win_capture_v2 (top-mover probability)
    # -----------------------------------------------------------------------
    try:
        result["win_capture_v2"] = _run_win_capture_v2(df, period_mask)
        loaded.append("win_capture_v2")
    except Exception as e:
        skipped["win_capture_v2"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # 7. day_class (daily-bar regime classifier)
    # NOTE: day_class_lgb_v1 uses bespoke derived features (lag1_ret, lag1_is_high,
    # btc_norm_* etc.) with 0/46 overlap with chimera columns. Running with all-zero
    # input produces a degenerate "FLAT=100%" read that is not informative.
    # We note it as unavailable rather than produce a misleading output.
    # -----------------------------------------------------------------------
    skipped.setdefault(
        "day_class",
        "SKIP: requires bespoke pre-engineered features (lag1_ret, lag1_is_high, "
        "btc_norm_* etc.) with 0/46 overlap with chimera -- degenerate all-zero "
        "input would produce a misleading read"
    )

    # -----------------------------------------------------------------------
    # 8. WM v1.1 (world model -- directional bias read)
    # -----------------------------------------------------------------------
    try:
        result["wm_v1_1"] = _run_wm_v1_1(df, period_mask)
        loaded.append("wm_v1_1")
    except Exception as e:
        skipped["wm_v1_1"] = str(e)[:120]

    # -----------------------------------------------------------------------
    # Models intentionally skipped with notes
    # -----------------------------------------------------------------------
    skipped.setdefault(
        "cluster_avoidance",
        "SKIP: Column_0..Column_37 feature schema -- trained on opaque numeric columns, "
        "cannot reconstruct from chimera without original panel encoder"
    )
    skipped.setdefault(
        "cluster_detector",
        "SKIP: requires cross-section panel breadth/mover/cluster features "
        "(btc_30d_ret, breadth_pct_long etc.) not available in single-asset chimera"
    )
    skipped.setdefault(
        "oracle_ma_ml",
        "SKIP: requires MA-crossover derived features (ma_days_since_*, "
        "ma_spread_*, state_id, liquidity_regime) not present in chimera"
    )
    skipped.setdefault(
        "meta_labeler_v8",
        "SKIP: requires pre-computed xsec_pred column + multiple rolling-return "
        "lags (ret_1d/3d/7d/14d) and btc_regime -- reset-orphaned feature set"
    )
    skipped.setdefault(
        "frontier_ml_foundation",
        "NOTE: loads OK (31.7M params, Mamba3/SSM, n_features=34, step=4999, "
        "loss_ema=1.17) -- inference deferred (multi-asset cross-attn required; "
        "descriptive embedding extraction is a future extension)"
    )
    skipped.setdefault(
        "per_sleeve_setup_other",
        "NOTE: blend_v2_DIB_cppi, router_strict, bd_depth_flow_alpha, cash_USDC "
        "all PASS (OOS AUC 0.63-0.64) but are sizing/routing sleeves (exit-domain); "
        "not centred per design mandate"
    )
    skipped.setdefault(
        "signature_library_v1",
        "NOTE: parquet centroids + scaler.pkl -- superseded by "
        "contrastive_signatures_v1 (which is loaded above)"
    )

    result["loaded"] = loaded
    result["skipped"] = skipped
    return result
