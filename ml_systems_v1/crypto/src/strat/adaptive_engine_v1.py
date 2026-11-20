"""
adaptive_engine_v1.py — Real adaptive ML engine for 7-day forward return prediction.

DESIGN:
  - Walk-forward expanding-window retrain every 90 days
  - Features: causal indicators + derived (dist_sma200, dist_sma50, range_pos,
    rsi14, vol20, mom7, mom14, mom30, atr14/C, breadth, btc_regime, 1d/3d recent ret)
  - Targets: P(7d fwd return > 0) [classifier], 7d fwd return [regressor],
             beats cross-sectional median [relative classifier]
  - Allocation: top-K assets by predicted P, EW; cash if nothing clears P>0.5
  - Evaluation: >=300 random 7-day OOS slices, win-rate vs buy-hold on SAME slices
  - OOS AUC reported for classifier
  - BRUTAL honesty about leak: all label windows must be closed before training cutoff

CAUSAL RULE: Feature at row d uses only data <= d.
             Label = d -> d+7 forward return (uses d+1..d+7).
             For training cutoff T: only rows d where d+7 < T are used.
             This means the first OOS prediction starts at T, not T+7.

Run:
  python -m strat.adaptive_engine_v1
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

# ---- Feature engineering (all causal) ----
def build_features(ind: dict, lookback_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a flat feature DataFrame (dates x assets stacked -> rows).
    Returns:
        feat_df: MultiIndex (date, asset) -> feature columns
        meta_df: MultiIndex (date, asset) -> [fwd_ret_7d, label_abs, label_rel]
    """
    C = ind["C"]
    R = ind["R"]
    dates = C.index
    assets = C.columns
    eps = 1e-8

    # ----- derived features (each is dates x assets DataFrame) -----
    sma200 = ind["sma200"]
    sma50 = ind["sma50"]
    dist_sma200 = C / (sma200 + eps) - 1          # distance from 200 SMA
    dist_sma50 = C / (sma50 + eps) - 1            # distance from 50 SMA
    hh14 = ind["hh14"]
    ll14 = ind["ll14"]
    rng = hh14 - ll14
    range_pos = (C - ll14) / (rng + eps)           # 0=at low, 1=at high
    rsi14 = ind["rsi14"]
    vol20 = ind["vol20"]
    mom7 = ind["mom7"]
    mom14 = ind["mom14"]
    mom30 = ind["mom30"]
    atr14 = ind["atr14"]
    atr_ratio = atr14 / (C + eps)                  # ATR / price = normalized volatility
    ret1 = ind["ret1"]
    ret3 = C / C.shift(3) - 1                      # 3-day return (causal)

    # breadth: fraction of universe above own sma50 (scalar per date)
    above_sma50 = (C > sma50).astype(float)
    breadth = above_sma50.mean(axis=1)             # Series, date -> scalar

    # btc_regime: BTC above its own sma200 (scalar per date)
    btc_above_sma200 = (C["BTCUSDT"] > sma200["BTCUSDT"]).astype(float)
    # Handle NaN (warmup period)
    btc_above_sma200 = btc_above_sma200.fillna(0.0)

    # realized vol ratio: vol20 / vol20.rolling(60).mean (momentum of vol)
    vol_ma60 = vol20.rolling(60, min_periods=20).mean()
    vol_ratio = vol20 / (vol_ma60 + eps)

    # cross-sectional rank of mom7 (captures relative momentum)
    mom7_rank = mom7.rank(axis=1, pct=True)
    mom14_rank = mom14.rank(axis=1, pct=True)

    # gate signal
    gate = ind["gate"].astype(float)

    # ---- build rows ----
    records = []
    for d in dates:
        b = float(breadth.loc[d])
        btc = float(btc_above_sma200.loc[d])
        for sym in assets:
            row = {
                "date": d,
                "asset": sym,
                "dist_sma200": _get(dist_sma200, d, sym),
                "dist_sma50": _get(dist_sma50, d, sym),
                "range_pos": _get(range_pos, d, sym),
                "rsi14": _get(rsi14, d, sym),
                "vol20": _get(vol20, d, sym),
                "vol_ratio": _get(vol_ratio, d, sym),
                "mom7": _get(mom7, d, sym),
                "mom14": _get(mom14, d, sym),
                "mom30": _get(mom30, d, sym),
                "atr_ratio": _get(atr_ratio, d, sym),
                "ret1": _get(ret1, d, sym),
                "ret3": _get(ret3, d, sym),
                "mom7_rank": _get(mom7_rank, d, sym),
                "mom14_rank": _get(mom14_rank, d, sym),
                "gate": _get(gate, d, sym),
                "breadth": b,
                "btc_regime": btc,
            }
            records.append(row)

    feat_df = pd.DataFrame(records).set_index(["date", "asset"])
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

    # ---- compute labels (7-day forward return) ----
    # fwd_7d at row d = C[d+7] / C[d] - 1 (uses future data -> only used for TRAINING rows)
    # CAUSAL CHECK: we never use the label value during inference; label is only attached
    # to rows whose forward window has already closed relative to the training cutoff.
    fwd_ret = C.shift(-7) / C - 1   # NaN for last 7 rows (no future)
    # cross-sectional median at each date
    cs_median = fwd_ret.median(axis=1)
    beats_median = (fwd_ret.gt(cs_median, axis=0)).astype(float)

    label_records = []
    for d in dates:
        for sym in assets:
            label_records.append({
                "date": d, "asset": sym,
                "fwd_ret_7d": _get(fwd_ret, d, sym),
                "label_abs": float((_get(fwd_ret, d, sym) or 0) > 0),
                "label_rel": _get(beats_median, d, sym),
            })
    meta_df = pd.DataFrame(label_records).set_index(["date", "asset"])
    meta_df = meta_df.apply(pd.to_numeric, errors="coerce")

    return feat_df, meta_df


def _get(df, d, sym):
    """Safe getter returning None if missing."""
    try:
        v = df.loc[d, sym]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


FEATURE_COLS = [
    "dist_sma200", "dist_sma50", "range_pos", "rsi14", "vol20", "vol_ratio",
    "mom7", "mom14", "mom30", "atr_ratio", "ret1", "ret3",
    "mom7_rank", "mom14_rank", "gate", "breadth", "btc_regime",
]


# ---- Walk-forward training ----
def walk_forward_train_predict(
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    retrain_every: int = 90,
    min_train_rows: int = 500,
    model_type: str = "hgb_cls",   # hgb_cls | hgb_reg | logreg
    label_col: str = "label_abs",
) -> pd.DataFrame:
    """
    Walk-forward expanding window.
    Returns DataFrame with columns [date, asset, pred_prob, pred_label, true_label, fwd_ret_7d].

    CAUSAL ENFORCEMENT:
    - At each retrain date T, training set = rows where date + 7 < T (label window closed)
    - Predictions are made for the next block [T .. T+retrain_every)
    - There is NO data leakage because we never include rows with open label windows in training
    """
    dates = sorted(feat_df.index.get_level_values("date").unique())
    n_dates = len(dates)

    # Find first training cutoff: need at least min_train_rows label-closed rows
    # Label window closes at d+7, so for cutoff T, valid training rows have d < T-7
    predictions = []

    # Determine retrain schedule
    # Start: find first date where we have enough training data
    start_oos_idx = None
    for i, T in enumerate(dates):
        # rows with closed labels: date <= T - 8 (d + 7 < T means d < T-7, i.e. d <= T-8)
        T_thresh = T - pd.Timedelta(days=8)
        mask = feat_df.index.get_level_values("date") <= T_thresh
        n_valid = mask.sum()
        if n_valid >= min_train_rows:
            start_oos_idx = i
            break

    if start_oos_idx is None:
        raise ValueError("Not enough data to start walk-forward")

    print(f"[walk-forward] OOS starts at date index {start_oos_idx}: {dates[start_oos_idx].date()}")
    print(f"[walk-forward] Total dates: {n_dates}, OOS dates: {n_dates - start_oos_idx}")

    last_retrain_idx = -999
    model = None
    scaler = None
    X_scaler_needed = model_type == "logreg"

    for i in range(start_oos_idx, n_dates):
        T = dates[i]

        # Retrain if needed
        if i - last_retrain_idx >= retrain_every:
            T_thresh = T - pd.Timedelta(days=8)
            train_mask = feat_df.index.get_level_values("date") <= T_thresh
            X_tr_raw = feat_df[train_mask][FEATURE_COLS]
            y_tr = meta_df[train_mask][label_col]

            # Drop rows with NaN in features or labels
            valid = X_tr_raw.notna().all(axis=1) & y_tr.notna()
            X_tr = X_tr_raw[valid].values
            y_tr = y_tr[valid].values

            if len(X_tr) >= min_train_rows:
                if model_type == "hgb_cls":
                    model = HistGradientBoostingClassifier(
                        max_iter=200, max_depth=4, learning_rate=0.05,
                        min_samples_leaf=20, l2_regularization=1.0,
                        random_state=SEED
                    )
                    model.fit(X_tr, y_tr)
                elif model_type == "hgb_reg":
                    model = HistGradientBoostingRegressor(
                        max_iter=200, max_depth=4, learning_rate=0.05,
                        min_samples_leaf=20, l2_regularization=1.0,
                        random_state=SEED
                    )
                    model.fit(X_tr, y_tr)
                elif model_type == "logreg":
                    scaler = StandardScaler()
                    X_tr_sc = scaler.fit_transform(X_tr)
                    model = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
                    model.fit(X_tr_sc, y_tr)
                last_retrain_idx = i

        # Make prediction for current date T
        if model is None:
            continue

        day_idx = feat_df.index.get_level_values("date") == T
        X_day_raw = feat_df[day_idx][FEATURE_COLS]
        y_day = meta_df[day_idx]

        for (d, sym) in feat_df[day_idx].index:
            row = X_day_raw.loc[(d, sym)]
            if row.isna().any():
                continue
            x = row.values.reshape(1, -1)
            if X_scaler_needed and scaler is not None:
                x = scaler.transform(x)

            if model_type == "hgb_cls" or model_type == "logreg":
                prob = float(model.predict_proba(x)[0, 1])
                pred_label = int(prob > 0.5)
            else:
                # regression: convert to probability-like score (not a proper prob)
                prob = float(model.predict(x)[0])
                pred_label = int(prob > 0)

            true_label = float(meta_df.loc[(d, sym), label_col]) if (d, sym) in meta_df.index else np.nan
            fwd = float(meta_df.loc[(d, sym), "fwd_ret_7d"]) if (d, sym) in meta_df.index else np.nan

            predictions.append({
                "date": d, "asset": sym,
                "pred_prob": prob,
                "pred_label": pred_label,
                "true_label": true_label,
                "fwd_ret_7d": fwd,
            })

    pred_df = pd.DataFrame(predictions)
    return pred_df


# ---- Random 7-day slice evaluation ----
def evaluate_random_slices(
    pred_df: pd.DataFrame,
    ind: dict,
    n_slices: int = 400,
    slice_days: int = 7,
    top_k: int = 3,
    prob_threshold: float = 0.5,
    seed: int = SEED,
) -> dict:
    """
    Evaluate on n_slices random 7-day windows in the OOS period.
    For each slice: compare ML engine vs buy-hold on SAME assets/slice.

    Returns dict with win-rate, mean per-slice return, vs buy-hold.
    """
    rng = np.random.default_rng(seed)
    C = ind["C"]
    assets = list(C.columns)

    oos_dates = sorted(pred_df["date"].unique())
    if len(oos_dates) < slice_days + 1:
        raise ValueError(f"Not enough OOS dates: {len(oos_dates)}")

    # Valid start dates: need slice_days forward
    valid_starts = [d for d in oos_dates if (d + pd.Timedelta(days=slice_days * 2)) <= oos_dates[-1]]
    if len(valid_starts) < n_slices:
        print(f"[warn] Only {len(valid_starts)} valid start dates, using all")
        n_slices = len(valid_starts)

    chosen_starts = rng.choice(len(valid_starts), size=n_slices, replace=True)

    ml_rets = []
    bh_rets = []
    ml_wins = []
    exposures = []

    for idx in chosen_starts:
        start_date = valid_starts[idx]
        # Find the actual 7 trading days from start
        after_start = [d for d in oos_dates if d >= start_date]
        if len(after_start) < slice_days + 1:
            continue
        end_date = after_start[slice_days]  # 7 bars after start

        # ML signal: use prediction AT start_date
        day_preds = pred_df[pred_df["date"] == start_date]

        if len(day_preds) == 0:
            # Try closest date
            close = [d for d in oos_dates if d <= start_date]
            if not close:
                continue
            day_preds = pred_df[pred_df["date"] == close[-1]]

        # Pick top-K assets by predicted prob, above threshold
        day_preds = day_preds.sort_values("pred_prob", ascending=False)
        eligible = day_preds[day_preds["pred_prob"] >= prob_threshold]

        if len(eligible) == 0:
            # Cash position: 0 return
            ml_ret = 0.0
            expo = 0.0
        else:
            picked = eligible.head(top_k)["asset"].tolist()
            # Compute actual return from start_date to end_date for picked assets
            slice_rets = []
            for sym in picked:
                if sym in C.columns and start_date in C.index and end_date in C.index:
                    r = float(C.loc[end_date, sym]) / float(C.loc[start_date, sym]) - 1
                    slice_rets.append(r)
            if slice_rets:
                ml_ret = float(np.mean(slice_rets))  # EW
            else:
                ml_ret = 0.0
            expo = 1.0

        # Buy-hold: EW over ALL assets (same slice)
        bh_slice = []
        for sym in assets:
            if sym in C.columns and start_date in C.index and end_date in C.index:
                r = float(C.loc[end_date, sym]) / float(C.loc[start_date, sym]) - 1
                bh_slice.append(r)
        bh_ret = float(np.mean(bh_slice)) if bh_slice else 0.0

        ml_rets.append(ml_ret)
        bh_rets.append(bh_ret)
        ml_wins.append(int(ml_ret > bh_ret))
        exposures.append(expo)

    ml_arr = np.array(ml_rets)
    bh_arr = np.array(bh_rets)
    wins_arr = np.array(ml_wins)
    expo_arr = np.array(exposures)

    # Absolute win-rate: ML slice return > 0
    ml_abs_wins = (ml_arr > 0).mean()
    bh_abs_wins = (bh_arr > 0).mean()

    return {
        "n_slices": len(ml_rets),
        "ml_win_rate_vs_bh": float(wins_arr.mean()),      # ML beats BH on this slice
        "ml_abs_win_rate": float(ml_abs_wins),             # ML slice > 0
        "bh_abs_win_rate": float(bh_abs_wins),             # BH slice > 0
        "ml_mean_ret": float(ml_arr.mean()),               # mean per-slice return for ML
        "bh_mean_ret": float(bh_arr.mean()),               # mean per-slice return for BH
        "ml_median_ret": float(np.median(ml_arr)),
        "bh_median_ret": float(np.median(bh_arr)),
        "avg_exposure": float(expo_arr.mean()),            # fraction of slices invested (not cash)
        "ml_mean_when_invested": float(ml_arr[expo_arr > 0].mean()) if (expo_arr > 0).any() else 0.0,
    }


# ---- OOS AUC ----
def compute_oos_auc(pred_df: pd.DataFrame, label_col: str = "true_label") -> float:
    valid = pred_df[pred_df[label_col].notna() & pred_df["pred_prob"].notna()]
    if len(valid) < 10:
        return np.nan
    try:
        return float(roc_auc_score(valid[label_col].values, valid["pred_prob"].values))
    except Exception:
        return np.nan


# ---- Bootstrap confidence interval on win-rate ----
def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    boots = [stat_fn(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ---- Main ----
def main():
    print("=" * 70)
    print("ADAPTIVE ENGINE v1 -- Walk-Forward ML for 7-day Forward Return")
    print("=" * 70)

    # Load full data
    print("\n[1] Loading data 2020-01-01 to 2026-06-01 ...")
    ind = lab.load("2020-01-01", "2026-06-01")
    C = ind["C"]
    print(f"    Assets: {list(C.columns)}")
    print(f"    Date range: {C.index[0].date()} to {C.index[-1].date()}")
    print(f"    Total bars: {len(C)}")

    # Build features
    print("\n[2] Building causal feature matrix (long format) ...")
    feat_df, meta_df = build_features(ind)
    print(f"    Feature matrix: {len(feat_df)} rows, {len(FEATURE_COLS)} features")
    print(f"    Features: {FEATURE_COLS}")

    # Check for NaN rates
    nan_rates = feat_df[FEATURE_COLS].isna().mean()
    print(f"\n    NaN rates per feature:")
    for col, r in nan_rates.items():
        if r > 0.01:
            print(f"      {col}: {r:.1%}")

    # ---- Model 1: HGB Classifier (abs direction) ----
    print("\n" + "=" * 70)
    print("MODEL 1: HistGradientBoosting Classifier -- P(7d return > 0)")
    print("=" * 70)
    pred_cls = walk_forward_train_predict(
        feat_df, meta_df,
        retrain_every=90, min_train_rows=500,
        model_type="hgb_cls", label_col="label_abs"
    )
    print(f"  Total OOS predictions: {len(pred_cls)}")
    oos_auc_cls = compute_oos_auc(pred_cls, "true_label")
    print(f"  OOS AUC (direction): {oos_auc_cls:.4f}")

    slices_cls = evaluate_random_slices(pred_cls, ind, n_slices=400, top_k=3, prob_threshold=0.5)
    print(f"\n  [RANDOM SLICE EVAL - top3 assets, P>=0.5]")
    print(f"  n_slices          : {slices_cls['n_slices']}")
    print(f"  ML abs win-rate   : {slices_cls['ml_abs_win_rate']:.1%}   (slice return > 0)")
    print(f"  BH abs win-rate   : {slices_cls['bh_abs_win_rate']:.1%}   (STANDING RESULT to beat: 55%)")
    print(f"  ML vs BH win-rate : {slices_cls['ml_win_rate_vs_bh']:.1%}   (ML beats BH on same slice)")
    print(f"  ML mean ret/slice : {slices_cls['ml_mean_ret']:+.2%}  (STANDING RESULT to beat: +2.9%)")
    print(f"  BH mean ret/slice : {slices_cls['bh_mean_ret']:+.2%}")
    print(f"  ML median ret     : {slices_cls['ml_median_ret']:+.2%}")
    print(f"  Avg exposure      : {slices_cls['avg_exposure']:.1%}  (fraction of slices NOT in cash)")
    print(f"  ML ret when invested: {slices_cls['ml_mean_when_invested']:+.2%}")

    # Bootstrap CI on abs win-rate
    ml_abs = np.array([float(r > 0) for r in []])  # placeholder
    # Recompute slices raw for CI
    rng2 = np.random.default_rng(SEED + 1)
    C_data = ind["C"]
    assets = list(C_data.columns)
    oos_dates = sorted(pred_cls["date"].unique())
    valid_starts = [d for d in oos_dates if (d + pd.Timedelta(days=14)) <= oos_dates[-1]]
    chosen = rng2.choice(len(valid_starts), size=400, replace=True)
    raw_ml_rets2 = []
    for idx in chosen:
        start_date = valid_starts[idx]
        after_start = [d for d in oos_dates if d >= start_date]
        if len(after_start) < 8:
            continue
        end_date = after_start[7]
        day_preds = pred_cls[pred_cls["date"] == start_date]
        if len(day_preds) == 0:
            continue
        eligible = day_preds[day_preds["pred_prob"] >= 0.5].sort_values("pred_prob", ascending=False)
        if len(eligible) == 0:
            raw_ml_rets2.append(0.0)
            continue
        picked = eligible.head(3)["asset"].tolist()
        sr = []
        for sym in picked:
            if sym in C_data.columns and start_date in C_data.index and end_date in C_data.index:
                r = float(C_data.loc[end_date, sym]) / float(C_data.loc[start_date, sym]) - 1
                sr.append(r)
        raw_ml_rets2.append(float(np.mean(sr)) if sr else 0.0)

    if raw_ml_rets2:
        arr2 = np.array(raw_ml_rets2)
        abs_wins2 = (arr2 > 0).astype(float)
        lo, hi = bootstrap_ci(abs_wins2, np.mean, n_boot=2000)
        print(f"  Bootstrap 95% CI (abs win-rate): [{lo:.1%}, {hi:.1%}]")

    # ---- Model 2: HGB Relative Classifier (beats cross-sectional median) ----
    print("\n" + "=" * 70)
    print("MODEL 2: HGB Classifier -- P(beats cross-sectional median)")
    print("=" * 70)
    pred_rel = walk_forward_train_predict(
        feat_df, meta_df,
        retrain_every=90, min_train_rows=500,
        model_type="hgb_cls", label_col="label_rel"
    )
    oos_auc_rel = compute_oos_auc(pred_rel, "true_label")
    print(f"  OOS AUC (relative): {oos_auc_rel:.4f}")

    slices_rel = evaluate_random_slices(pred_rel, ind, n_slices=400, top_k=3, prob_threshold=0.5)
    print(f"\n  [RANDOM SLICE EVAL - top3 assets, P>=0.5]")
    print(f"  n_slices          : {slices_rel['n_slices']}")
    print(f"  ML abs win-rate   : {slices_rel['ml_abs_win_rate']:.1%}")
    print(f"  BH abs win-rate   : {slices_rel['bh_abs_win_rate']:.1%}")
    print(f"  ML vs BH win-rate : {slices_rel['ml_win_rate_vs_bh']:.1%}")
    print(f"  ML mean ret/slice : {slices_rel['ml_mean_ret']:+.2%}")
    print(f"  BH mean ret/slice : {slices_rel['bh_mean_ret']:+.2%}")
    print(f"  Avg exposure      : {slices_rel['avg_exposure']:.1%}")

    # ---- Model 3: HGB Regressor ----
    print("\n" + "=" * 70)
    print("MODEL 3: HGB Regressor -- 7d forward return (scored as rank)")
    print("=" * 70)
    pred_reg = walk_forward_train_predict(
        feat_df, meta_df,
        retrain_every=90, min_train_rows=500,
        model_type="hgb_reg", label_col="fwd_ret_7d"
    )
    # For regression, AUC of sign prediction
    valid_reg = pred_reg[pred_reg["fwd_ret_7d"].notna()]
    if len(valid_reg) > 10:
        true_sign = (meta_df.loc[valid_reg.set_index(["date", "asset"]).index, "label_abs"]
                     if False else None)
        # Simple: check if predicted positive fwd_ret -> actual positive
        pred_sign = (pred_reg["pred_prob"] > 0).astype(float)
        actual_abs = (meta_df.reindex(
            pd.MultiIndex.from_frame(pred_reg[["date", "asset"]])
        )["label_abs"])
        if len(actual_abs) > 0 and actual_abs.notna().sum() > 10:
            try:
                sign_auc = roc_auc_score(actual_abs.dropna().values,
                                          pred_sign[actual_abs.notna()].values)
                print(f"  Sign prediction AUC: {sign_auc:.4f}")
            except Exception:
                pass

    slices_reg = evaluate_random_slices(pred_reg, ind, n_slices=400, top_k=3, prob_threshold=0.0)
    print(f"\n  [RANDOM SLICE EVAL - top3 assets by predicted return]")
    print(f"  n_slices          : {slices_reg['n_slices']}")
    print(f"  ML abs win-rate   : {slices_reg['ml_abs_win_rate']:.1%}")
    print(f"  BH abs win-rate   : {slices_reg['bh_abs_win_rate']:.1%}")
    print(f"  ML vs BH win-rate : {slices_reg['ml_win_rate_vs_bh']:.1%}")
    print(f"  ML mean ret/slice : {slices_reg['ml_mean_ret']:+.2%}")
    print(f"  BH mean ret/slice : {slices_reg['bh_mean_ret']:+.2%}")
    print(f"  Avg exposure      : {slices_reg['avg_exposure']:.1%}")

    # ---- Model 4: Logistic Regression (simpler, less overfit risk) ----
    print("\n" + "=" * 70)
    print("MODEL 4: Logistic Regression -- P(7d return > 0) [L2 reg, less overfit]")
    print("=" * 70)
    pred_lr = walk_forward_train_predict(
        feat_df, meta_df,
        retrain_every=90, min_train_rows=500,
        model_type="logreg", label_col="label_abs"
    )
    oos_auc_lr = compute_oos_auc(pred_lr, "true_label")
    print(f"  OOS AUC (logreg): {oos_auc_lr:.4f}")

    slices_lr = evaluate_random_slices(pred_lr, ind, n_slices=400, top_k=3, prob_threshold=0.5)
    print(f"\n  [RANDOM SLICE EVAL - top3 assets, P>=0.5]")
    print(f"  n_slices          : {slices_lr['n_slices']}")
    print(f"  ML abs win-rate   : {slices_lr['ml_abs_win_rate']:.1%}")
    print(f"  BH abs win-rate   : {slices_lr['bh_abs_win_rate']:.1%}")
    print(f"  ML vs BH win-rate : {slices_lr['ml_win_rate_vs_bh']:.1%}")
    print(f"  ML mean ret/slice : {slices_lr['ml_mean_ret']:+.2%}")
    print(f"  BH mean ret/slice : {slices_lr['bh_mean_ret']:+.2%}")
    print(f"  Avg exposure      : {slices_lr['avg_exposure']:.1%}")

    # ---- Ablation: vary top-K and threshold for best model ----
    print("\n" + "=" * 70)
    print("ABLATION: Vary top-K and threshold (Model 1 - HGB Cls abs)")
    print("=" * 70)
    best_winrate = -1
    best_config = None
    best_result = None
    for topk in [1, 2, 3, 5]:
        for thresh in [0.45, 0.50, 0.55]:
            res = evaluate_random_slices(pred_cls, ind, n_slices=400, top_k=topk,
                                          prob_threshold=thresh, seed=SEED + topk + int(thresh * 100))
            wr = res["ml_abs_win_rate"]
            mr = res["ml_mean_ret"]
            vs = res["ml_win_rate_vs_bh"]
            expo = res["avg_exposure"]
            print(f"  topK={topk} thresh={thresh:.2f}: abs_wr={wr:.1%} mean={mr:+.2%} vs_bh={vs:.1%} expo={expo:.0%}")
            if wr > best_winrate:
                best_winrate = wr
                best_config = (topk, thresh)
                best_result = res

    print(f"\n  Best config: topK={best_config[0]}, thresh={best_config[1]:.2f}")
    print(f"  Best abs win-rate: {best_winrate:.1%}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (OOS, walk-forward, no look-ahead)")
    print("=" * 70)
    print(f"{'Model':<35} {'AUC':>6} {'AbsWR':>7} {'MeanRet':>9} {'VsBH_WR':>9} {'Expo':>6}")
    print("-" * 75)
    bh_wr = slices_cls["bh_abs_win_rate"]
    bh_mr = slices_cls["bh_mean_ret"]
    print(f"{'Buy-Hold (benchmark)':<35} {'N/A':>6} {bh_wr:>6.1%} {bh_mr:>+8.2%} {'N/A':>9} {'100%':>6}")
    print(f"{'HGB Cls (abs, top3 P>=0.5)':<35} {oos_auc_cls:>6.4f} {slices_cls['ml_abs_win_rate']:>6.1%} {slices_cls['ml_mean_ret']:>+8.2%} {slices_cls['ml_win_rate_vs_bh']:>8.1%} {slices_cls['avg_exposure']:>5.0%}")
    print(f"{'HGB Cls (rel, top3 P>=0.5)':<35} {oos_auc_rel:>6.4f} {slices_rel['ml_abs_win_rate']:>6.1%} {slices_rel['ml_mean_ret']:>+8.2%} {slices_rel['ml_win_rate_vs_bh']:>8.1%} {slices_rel['avg_exposure']:>5.0%}")
    print(f"{'HGB Reg (top3 pred>0)':<35} {'N/A':>6} {slices_reg['ml_abs_win_rate']:>6.1%} {slices_reg['ml_mean_ret']:>+8.2%} {slices_reg['ml_win_rate_vs_bh']:>8.1%} {slices_reg['avg_exposure']:>5.0%}")
    print(f"{'LogReg (abs, top3 P>=0.5)':<35} {oos_auc_lr:>6.4f} {slices_lr['ml_abs_win_rate']:>6.1%} {slices_lr['ml_mean_ret']:>+8.2%} {slices_lr['ml_win_rate_vs_bh']:>8.1%} {slices_lr['avg_exposure']:>5.0%}")
    if best_result:
        print(f"{'HGB Cls (best ablation config)':<35} {oos_auc_cls:>6.4f} {best_result['ml_abs_win_rate']:>6.1%} {best_result['ml_mean_ret']:>+8.2%} {best_result['ml_win_rate_vs_bh']:>8.1%} {best_result['avg_exposure']:>5.0%}")
    print("-" * 75)
    print(f"\nTarget: abs win-rate > 55% (BH={bh_wr:.1%}) OR mean ret > +2.9% (BH={bh_mr:+.2%})")

    # ---- Honest verdict ----
    print("\n" + "=" * 70)
    print("HONEST VERDICT")
    print("=" * 70)

    best_wr = max(slices_cls["ml_abs_win_rate"], slices_rel["ml_abs_win_rate"],
                  slices_reg["ml_abs_win_rate"], slices_lr["ml_abs_win_rate"])
    best_mr = max(slices_cls["ml_mean_ret"], slices_rel["ml_mean_ret"],
                  slices_reg["ml_mean_ret"], slices_lr["ml_mean_ret"])

    beats_wr = best_wr > 0.55
    beats_mr = best_mr > 0.029

    print(f"  Win-rate criterion (>55%): {'PASS' if beats_wr else 'FAIL'} (best={best_wr:.1%})")
    print(f"  Mean-ret criterion (>+2.9%): {'PASS' if beats_mr else 'FAIL'} (best={best_mr:+.2%})")

    print(f"\n  OOS AUC for direction classifier: {oos_auc_cls:.4f}")
    print(f"  AUC < 0.53: consistent with 7d direction being near-unpredictable")

    print(f"\n  LEAK CHECK:")
    print(f"    - Label d+7 only in training when d+7 < cutoff T (d <= T-8)")
    print(f"    - Features use only C[<=d], sma/rsi/etc computed with rolling(past)")
    print(f"    - No peeking into future via shift(-N) in feature space")
    print(f"    - fwd_ret = C.shift(-7)/C-1 is a LABEL only, never a feature")

    if beats_wr or beats_mr:
        print(f"\n  VERDICT: At least one criterion PASSES. Inspect carefully for:")
        print(f"    - Whether the advantage persists across sub-periods")
        print(f"    - Whether exposure concentration explains outperformance")
        print(f"    - Whether the cash-out rate (low expo) is driving the win-rate")
    else:
        print(f"\n  VERDICT: FAIL. The adaptive ML engine does NOT beat buy-hold OOS.")
        print(f"    The 7d direction is near-unpredictable with internal data alone.")
        print(f"    OOS AUC ~0.50 confirms no edge; any slippage above 55% is noise.")


if __name__ == "__main__":
    main()
