"""
gb_meta_engine.py -- Gradient-Boosting Meta-Labeler Engine (Tournament Cycle 1)

ENGINE: sklearn HistGradientBoostingClassifier, expanding walk-forward.
Target: P(asset 7d-fwd > 0).
Allocation: top-K assets with P >= 0.5 (EW); CASH if nothing clears P>0.5.
Evaluation: >=300 random 7-day OOS slices.
Reports: OOS AUC, positive-rate, mean vs buy-hold, cash-rate-in-down-weeks.

CAUSAL RULE enforcement:
  Feature at row d: uses only data <= d.
  Label d -> d+7: uses future close at d+7 (LABEL ONLY, never a feature).
  Training at cutoff T: only rows where d+7 < T (label window CLOSED before T).
  No global scaler/threshold fit on full data.

Run: python -m strat.gb_meta_engine
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except ImportError:
    HAS_HGB = False
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)


# ============================================================
# FEATURE ENGINEERING (fully vectorized, CAUSAL)
# ============================================================

def build_feature_matrix(ind: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build features in panel (date x asset) format, then stack to long format.
    All features are causal at row d (use only data <= d).
    Labels are 7d forward returns (future data, attached separately for training only).

    Returns:
        feat_long  : DataFrame (MultiIndex date,asset) x feature_cols
        label_abs  : Series (MultiIndex date,asset) = 1 if fwd_7d > 0 else 0
        fwd_ret_7d : Series (MultiIndex date,asset) = raw 7d forward return
    """
    C    = ind["C"]
    eps  = 1e-8

    # --- Panel features (all dates x assets, causal) ---
    sma200     = ind["sma200"]
    sma50      = ind["sma50"]
    dist_sma200 = C / (sma200 + eps) - 1
    dist_sma50  = C / (sma50  + eps) - 1
    hh14       = ind["hh14"]
    ll14       = ind["ll14"]
    rng14      = hh14 - ll14
    range_pos  = (C - ll14) / (rng14 + eps)      # 0=at 14d low, 1=at 14d high
    rsi14      = ind["rsi14"]
    vol20      = ind["vol20"]
    mom7       = ind["mom7"]
    mom14      = ind["mom14"]
    mom30      = ind["mom30"]
    atr14      = ind["atr14"]
    atr_ratio  = atr14 / (C + eps)
    ret1       = ind["ret1"]
    ret3       = C / C.shift(3) - 1
    ret14      = C / C.shift(14) - 1

    # vol momentum: vol20 vs 60d avg
    vol_ma60  = vol20.rolling(60, min_periods=20).mean()
    vol_ratio = vol20 / (vol_ma60 + eps)

    # cross-sectional ranks (relative momentum)
    mom7_rank  = mom7.rank(axis=1, pct=True)
    mom14_rank = mom14.rank(axis=1, pct=True)

    gate = ind["gate"].astype(float)

    # breadth: fraction of assets above sma50 (scalar per date, broadcast)
    above_sma50 = (C > sma50).astype(float)
    breadth = above_sma50.mean(axis=1)

    # btc_regime: BTC above its sma200 (scalar per date)
    btc_regime = (C["BTCUSDT"] > sma200["BTCUSDT"]).astype(float).fillna(0.0)

    # --- stack to long format ---
    # Each panel df gets stacked -> MultiIndex (date, asset)
    panels = {
        "dist_sma200": dist_sma200,
        "dist_sma50":  dist_sma50,
        "range_pos":   range_pos,
        "rsi14":       rsi14,
        "vol20":       vol20,
        "vol_ratio":   vol_ratio,
        "mom7":        mom7,
        "mom14":       mom14,
        "mom30":       mom30,
        "atr_ratio":   atr_ratio,
        "ret1":        ret1,
        "ret3":        ret3,
        "ret14":       ret14,
        "mom7_rank":   mom7_rank,
        "mom14_rank":  mom14_rank,
        "gate":        gate,
    }

    stacked = {}
    for name, df in panels.items():
        s = df.stack(dropna=False)
        s.index.names = ["date", "asset"]
        stacked[name] = s

    feat_long = pd.DataFrame(stacked)

    # broadcast scalars (breadth, btc_regime) to all assets
    breadth_broad = breadth.reindex(C.index).fillna(method="ffill")
    btc_broad     = btc_regime.reindex(C.index).fillna(0.0)
    # replicate per asset via index level 0
    feat_long["breadth"]    = breadth_broad.reindex(feat_long.index.get_level_values("date")).values
    feat_long["btc_regime"] = btc_broad.reindex(feat_long.index.get_level_values("date")).values

    # --- labels (FUTURE data, never used as feature) ---
    fwd_ret = C.shift(-7) / C - 1   # NaN for last 7 rows (no future close)
    fwd_long = fwd_ret.stack(dropna=False)
    fwd_long.index.names = ["date", "asset"]
    fwd_ret_long = fwd_long
    label_abs = (fwd_ret_long > 0).astype(float).where(fwd_ret_long.notna(), other=np.nan)

    return feat_long, label_abs, fwd_ret_long


FEATURE_COLS = [
    "dist_sma200", "dist_sma50", "range_pos", "rsi14", "vol20", "vol_ratio",
    "mom7", "mom14", "mom30", "atr_ratio", "ret1", "ret3", "ret14",
    "mom7_rank", "mom14_rank", "gate", "breadth", "btc_regime",
]


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def walk_forward_predict(
    feat_long: pd.DataFrame,
    label_abs: pd.Series,
    fwd_ret_long: pd.Series,
    retrain_every: int = 90,
    min_train_rows: int = 500,
    model_type: str = "hgb",   # "hgb" | "logreg"
) -> pd.DataFrame:
    """
    Expanding walk-forward: retrain every `retrain_every` trading days.
    CAUSAL: training at date T uses only rows with date <= T - 8
            (so label window d+7 < T is closed).
    Returns pred_df with columns: date, asset, pred_prob, true_label, fwd_ret_7d.
    """
    dates = sorted(feat_long.index.get_level_values("date").unique())
    n_dates = len(dates)

    # Find first OOS index (need min_train_rows label-closed rows)
    start_oos_idx = None
    for i, T in enumerate(dates):
        T_thresh = T - pd.Timedelta(days=8)
        mask = feat_long.index.get_level_values("date") <= T_thresh
        valid_mask = mask & feat_long[FEATURE_COLS].notna().all(axis=1) & label_abs.notna()
        if valid_mask.sum() >= min_train_rows:
            start_oos_idx = i
            break

    if start_oos_idx is None:
        raise ValueError("Not enough data for walk-forward")

    print(f"  [WF] OOS starts: {dates[start_oos_idx].date()}  (idx {start_oos_idx}/{n_dates})")

    last_retrain = -999
    model = None
    scaler = None
    predictions = []

    for i in range(start_oos_idx, n_dates):
        T = dates[i]

        # Retrain?
        if i - last_retrain >= retrain_every:
            T_thresh = T - pd.Timedelta(days=8)
            tr_mask = (feat_long.index.get_level_values("date") <= T_thresh)
            X_raw = feat_long[FEATURE_COLS][tr_mask]
            y_raw = label_abs[tr_mask]
            valid = X_raw.notna().all(axis=1) & y_raw.notna()
            X_tr = X_raw[valid].values
            y_tr = y_raw[valid].values

            if len(X_tr) >= min_train_rows:
                if model_type == "hgb" and HAS_HGB:
                    model = HistGradientBoostingClassifier(
                        max_iter=200, max_depth=4, learning_rate=0.05,
                        min_samples_leaf=20, l2_regularization=1.0,
                        random_state=SEED
                    )
                    model.fit(X_tr, y_tr)
                    scaler = None
                else:
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(X_tr)
                    model = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
                    model.fit(X_sc, y_tr)
                last_retrain = i
                print(f"  [WF] Retrained at {T.date()} on {len(X_tr)} rows ({model_type})")

        if model is None:
            continue

        # Predict for all assets at date T
        day_mask = feat_long.index.get_level_values("date") == T
        X_day_raw = feat_long[FEATURE_COLS][day_mask]
        valid_day = X_day_raw.notna().all(axis=1)
        if not valid_day.any():
            continue
        X_day = X_day_raw[valid_day].values
        day_idx = feat_long[day_mask][valid_day].index

        if scaler is not None:
            X_day = scaler.transform(X_day)
        probs = model.predict_proba(X_day)[:, 1]

        for j, (d, sym) in enumerate(day_idx):
            tl = label_abs.get((d, sym), np.nan)
            fr = fwd_ret_long.get((d, sym), np.nan)
            predictions.append({
                "date": d, "asset": sym,
                "pred_prob": float(probs[j]),
                "true_label": float(tl) if not pd.isna(tl) else np.nan,
                "fwd_ret_7d": float(fr) if not pd.isna(fr) else np.nan,
            })

    return pd.DataFrame(predictions)


# ============================================================
# SLICE EVALUATION (random 7-day OOS slices)
# ============================================================

def evaluate_slices(
    pred_df: pd.DataFrame,
    ind: dict,
    n_slices: int = 400,
    top_k: int = 3,
    prob_threshold: float = 0.50,
    seed: int = SEED,
) -> dict:
    """
    Draw n_slices random 7-day windows from OOS period.
    For each slice:
      - ML engine: hold top-K assets with P >= threshold (EW); CASH if none.
      - BH: EW over all assets.
    Key extras: down-week detection (BH < 0 weeks) -> cash rate for ML in those weeks.
    """
    rng = np.random.default_rng(seed)
    C = ind["C"]
    assets = list(C.columns)
    oos_dates = sorted(pred_df["date"].unique())
    all_dates  = list(C.index)

    # Precompute date index for fast lookup
    date_pos = {d: i for i, d in enumerate(all_dates)}

    # Valid start dates: need 7+ bars ahead in OOS
    valid_starts = [d for d in oos_dates
                    if (d + pd.Timedelta(days=slice_days_buffer(7))) <= oos_dates[-1]]
    n_slices = min(n_slices, len(valid_starts))
    chosen_idxs = rng.choice(len(valid_starts), size=n_slices, replace=True)

    ml_rets, bh_rets, exposures = [], [], []

    for idx in chosen_idxs:
        start_d = valid_starts[idx]
        # Find end_date = 7 trading bars after start (using OOS dates as proxy)
        after = [d for d in oos_dates if d >= start_d]
        if len(after) < 8:
            continue
        end_d = after[7]

        # BH: EW over all assets
        bh_slice = []
        for sym in assets:
            if start_d in C.index and end_d in C.index:
                v0, v1 = C.loc[start_d, sym], C.loc[end_d, sym]
                if pd.notna(v0) and pd.notna(v1) and v0 > 0:
                    bh_slice.append(v1 / v0 - 1)
        bh_ret = float(np.mean(bh_slice)) if bh_slice else 0.0

        # ML: pick top-K by prob >= threshold
        day_pred = pred_df[pred_df["date"] == start_d]
        if len(day_pred) == 0:
            # fallback: closest prior date
            prior = [d for d in oos_dates if d <= start_d]
            if prior:
                day_pred = pred_df[pred_df["date"] == prior[-1]]

        eligible = day_pred[day_pred["pred_prob"] >= prob_threshold].sort_values(
            "pred_prob", ascending=False)

        if len(eligible) == 0:
            ml_ret = 0.0
            expo = 0.0
        else:
            picked = eligible.head(top_k)["asset"].tolist()
            sr = []
            for sym in picked:
                if sym in C.columns and start_d in C.index and end_d in C.index:
                    v0, v1 = C.loc[start_d, sym], C.loc[end_d, sym]
                    if pd.notna(v0) and pd.notna(v1) and v0 > 0:
                        sr.append(v1 / v0 - 1)
            ml_ret = float(np.mean(sr)) if sr else 0.0
            expo = 1.0

        ml_rets.append(ml_ret)
        bh_rets.append(bh_ret)
        exposures.append(expo)

    ml = np.array(ml_rets)
    bh = np.array(bh_rets)
    expo = np.array(exposures)

    # Down weeks = slices where BH < 0
    down_mask  = bh < 0
    up_mask    = bh >= 0
    cash_mask  = expo == 0.0

    down_weeks_count  = int(down_mask.sum())
    cash_in_down      = float(cash_mask[down_mask].mean()) if down_mask.any() else np.nan
    ml_ret_in_down    = float(ml[down_mask].mean()) if down_mask.any() else np.nan
    ml_ret_in_up      = float(ml[up_mask].mean()) if up_mask.any() else np.nan
    bh_ret_in_down    = float(bh[down_mask].mean()) if down_mask.any() else np.nan

    return {
        "n_slices":        int(len(ml)),
        "ml_pos_rate":     float((ml > 0).mean()),          # ML slice > 0
        "bh_pos_rate":     float((bh > 0).mean()),          # BH slice > 0
        "ml_mean":         float(ml.mean()),
        "bh_mean":         float(bh.mean()),
        "ml_median":       float(np.median(ml)),
        "bh_median":       float(np.median(bh)),
        "ml_vs_bh_wr":     float((ml > bh).mean()),         # ML beats BH on same slice
        "avg_exposure":    float(expo.mean()),
        "ml_when_inv":     float(ml[expo > 0].mean()) if (expo > 0).any() else 0.0,
        "down_weeks":      down_weeks_count,
        "down_pct":        float(down_mask.mean()),
        "cash_in_down":    cash_in_down,                    # KEY: cash when market down
        "ml_ret_in_down":  ml_ret_in_down,                  # ML return in down weeks
        "ml_ret_in_up":    ml_ret_in_up,
        "bh_ret_in_down":  bh_ret_in_down,
    }


def slice_days_buffer(n):
    """Calendar days buffer to ensure n trading days exist."""
    return n * 2 + 5


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 72)
    print("GB META-LABELER ENGINE -- Walk-Forward, OOS Evaluation")
    print("Tournament Cycle 1 -- CAUSAL, No-Leak")
    print("=" * 72)

    # 1. Load data
    print("\n[1] Loading data 2020-01-01 -> 2026-05-31 ...")
    ind = lab.load("2020-01-01", "2026-06-01")
    C = ind["C"]
    print(f"    Assets: {list(C.columns)}")
    print(f"    Date range: {C.index[0].date()} -> {C.index[-1].date()}")
    print(f"    Bars: {len(C)}")

    # 2. Build feature matrix
    print("\n[2] Building vectorized causal feature matrix ...")
    feat_long, label_abs, fwd_ret_long = build_feature_matrix(ind)
    n_rows = len(feat_long)
    nan_rate = feat_long[FEATURE_COLS].isna().mean()
    print(f"    Rows: {n_rows}  Features: {len(FEATURE_COLS)}")
    high_nan = {c: f"{v:.1%}" for c, v in nan_rate.items() if v > 0.05}
    if high_nan:
        print(f"    High NaN features (>5%): {high_nan}")
    label_valid = label_abs.notna().sum()
    label_pos_rate = float((label_abs > 0).sum()) / label_valid if label_valid else np.nan
    print(f"    Label coverage: {label_valid}/{n_rows}  Base pos-rate: {label_pos_rate:.1%}")

    # 3. Walk-forward: HGB classifier
    print("\n[3] Walk-forward: HistGradientBoostingClassifier (P(7d>0))")
    model_tag = "HGB" if HAS_HGB else "LogReg (HGB unavailable)"
    pred_hgb = walk_forward_predict(
        feat_long, label_abs, fwd_ret_long,
        retrain_every=90, min_train_rows=500,
        model_type="hgb",
    )
    print(f"    OOS predictions: {len(pred_hgb)}")

    # OOS AUC
    valid_auc = pred_hgb[pred_hgb["true_label"].notna() & pred_hgb["pred_prob"].notna()]
    oos_auc = np.nan
    if len(valid_auc) >= 50:
        try:
            oos_auc = float(roc_auc_score(valid_auc["true_label"].values,
                                          valid_auc["pred_prob"].values))
        except Exception as e:
            print(f"    AUC error: {e}")
    print(f"    OOS AUC: {oos_auc:.4f}")

    # 4. Random slice evaluation (400 slices)
    print("\n[4] Random 7-day slice evaluation (n=400, top_k=3, P>=0.5) ...")
    res = evaluate_slices(pred_hgb, ind, n_slices=400, top_k=3, prob_threshold=0.50)

    # 5. Walk-forward: LogReg (comparison / ablation)
    print("\n[5] Walk-forward: LogisticRegression (comparison) ...")
    pred_lr = walk_forward_predict(
        feat_long, label_abs, fwd_ret_long,
        retrain_every=90, min_train_rows=500,
        model_type="logreg",
    )
    auc_lr = np.nan
    valid_lr = pred_lr[pred_lr["true_label"].notna() & pred_lr["pred_prob"].notna()]
    if len(valid_lr) >= 50:
        try:
            auc_lr = float(roc_auc_score(valid_lr["true_label"].values,
                                          valid_lr["pred_prob"].values))
        except Exception:
            pass
    res_lr = evaluate_slices(pred_lr, ind, n_slices=400, top_k=3, prob_threshold=0.50)

    # 6. Ablation: vary top-K and threshold for HGB
    print("\n[6] Ablation: top-K x threshold (HGB model) ...")
    ablations = []
    for topk in [1, 2, 3, 5]:
        for thr in [0.45, 0.50, 0.55]:
            r = evaluate_slices(pred_hgb, ind, n_slices=400, top_k=topk,
                                prob_threshold=thr, seed=SEED + topk + int(thr * 100))
            ablations.append((topk, thr, r))
            print(f"    topK={topk} thr={thr:.2f}: pos_rate={r['ml_pos_rate']:.1%} "
                  f"mean={r['ml_mean']:+.2%} vs_bh_wr={r['ml_vs_bh_wr']:.1%} "
                  f"expo={r['avg_exposure']:.0%} cash_in_down={r['cash_in_down']:.1%}")

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    print("\n" + "=" * 72)
    print("RESULTS TABLE (OOS, walk-forward, causal, no leak)")
    print("=" * 72)
    hdr = f"{'Model':<38} {'AUC':>6} {'PosRate':>8} {'Mean':>8} {'VsBH_WR':>8} {'Expo':>6} {'CashDown':>9}"
    print(hdr)
    print("-" * 84)
    bh_pr = res["bh_pos_rate"]
    bh_mn = res["bh_mean"]
    print(f"{'Buy-Hold (EW benchmark)':<38} {'--':>6} {bh_pr:>7.1%} {bh_mn:>+7.2%} {'--':>8} {'100%':>6} {'--':>9}")
    print(f"{'HGB Classifier (top3, P>=0.5)':<38} {oos_auc:>6.4f} {res['ml_pos_rate']:>7.1%} "
          f"{res['ml_mean']:>+7.2%} {res['ml_vs_bh_wr']:>7.1%} {res['avg_exposure']:>5.0%} "
          f"{res['cash_in_down']:>8.1%}")
    print(f"{'LogReg (top3, P>=0.5)':<38} {auc_lr:>6.4f} {res_lr['ml_pos_rate']:>7.1%} "
          f"{res_lr['ml_mean']:>+7.2%} {res_lr['ml_vs_bh_wr']:>7.1%} {res_lr['avg_exposure']:>5.0%} "
          f"{res_lr['cash_in_down']:>8.1%}")
    print("-" * 84)
    print(f"\nTargets: pos_rate > {bh_pr:.1%}  |  mean > {bh_mn:+.2%}")
    print(f"NEVER-NEGATIVE mechanism: CASH when P<0.5 for all assets.")

    print("\n" + "=" * 72)
    print("DOWN-WEEK BEHAVIOR (structural-physics check)")
    print("=" * 72)
    print(f"  Total OOS slices evaluated: {res['n_slices']}")
    print(f"  Down weeks (BH<0):          {res['down_weeks']} ({res['down_pct']:.1%} of slices)")
    print(f"  Cash rate in down weeks:    {res['cash_in_down']:.1%}  <- KEY: does engine avoid?")
    print(f"  ML return in down weeks:    {res['ml_ret_in_down']:+.2%}  (BH in down: {res['bh_ret_in_down']:+.2%})")
    print(f"  ML return in up weeks:      {res['ml_ret_in_up']:+.2%}")
    print(f"  ML return when invested:    {res['ml_when_inv']:+.2%}")

    # ============================================================
    # HONEST VERDICT
    # ============================================================
    print("\n" + "=" * 72)
    print("HONEST VERDICT")
    print("=" * 72)

    beats_pos_rate = res["ml_pos_rate"] > bh_pr
    beats_mean     = res["ml_mean"] > bh_mn

    print(f"  Pos-rate criterion (>{bh_pr:.1%}): {'PASS' if beats_pos_rate else 'FAIL'} "
          f"(HGB={res['ml_pos_rate']:.1%})")
    print(f"  Mean-ret criterion (>{bh_mn:+.2%}): {'PASS' if beats_mean else 'FAIL'} "
          f"(HGB={res['ml_mean']:+.2%})")
    print(f"  OOS AUC: {oos_auc:.4f}  (chance=0.500; useful edge requires >0.53)")
    print(f"\n  CAUSAL LEAK CHECK:")
    print(f"    - Features at row d: only C[<=d], rolling(past), shift(+N) causal")
    print(f"    - Labels (fwd_ret = C.shift(-7)/C-1): NEVER used as feature")
    print(f"    - Training cutoff T: rows with date <= T-8 only (label window closed)")
    print(f"    - No global scaler fitted on full data: StandardScaler fitted on train only")
    print(f"    - mom7/mom14 rank: rank() applied per-date on past data only")

    if beats_pos_rate or beats_mean:
        print(f"\n  STATUS: CRITERIA MET -- inspect sub-period stability + exposure bias.")
        print(f"  CAUTION: check whether cash-in-down >= 50% is driving pos-rate mechanically.")
        print(f"  (Cash=0% always beats any negative week -> inflates pos-rate if exposure low)")
    else:
        print(f"\n  STATUS: FAIL -- does not beat buy-hold OOS by either criterion.")
        print(f"  OOS AUC ~0.50 confirms 7d direction is near-unpredictable from internal data.")
        print(f"  The never-negative mechanism (cash when P<0.5) reduces mean return.")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")

    # Save results
    out = {
        "model": model_tag,
        "oos_auc": float(oos_auc) if not np.isnan(oos_auc) else None,
        "slices_hgb": {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                       for k, v in res.items()},
        "slices_lr":  {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                       for k, v in res_lr.items()},
        "bh_pos_rate": float(bh_pr),
        "bh_mean": float(bh_mn),
        "beats_pos_rate": bool(beats_pos_rate),
        "beats_mean": bool(beats_mean),
    }
    out_path = Path(__file__).resolve().parents[2] / "runs" / "strat" / "gb_meta_engine_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
