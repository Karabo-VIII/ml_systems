"""
engine_tournament_ensemble.py -- MODEL-DIVERSITY ensemble engine (tournament cycle 1).

ENGINE: Logistic + RandomForest + MLP (sklearn), trained on 3 LABEL variants:
  L1: 7d-up sign (binary classification)
  L2: beats-cross-sectional-median (binary classification)
  L3: magnitude (regression, 7d forward return)

Walk-forward with a ROLLING 1-year training window (fixed size, fast).
Per-asset scores averaged -> allocate to top-K gated assets.

CAUSAL RULE (verified): features at row d use only data <= d.
  - Rolling window [d-365 .. d-HOLD] for training.
  - Labels use forward return d -> d+HOLD, only available after row d+HOLD.
  - Scaler fit on training window only, never on full data.
  - No global thresholds/parameters fit on full history.

WIN CONDITION: beat EW buy-hold positive-rate (~55%) and mean (+2.9%) on >=300 random 7d slices.
STRUCTURAL PHYSICS: long-only spot => cash on down weeks is best possible case.

RWYB: python -m strat.engine_tournament_ensemble
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HOLD = 7         # forward return horizon in days
TOPK = 3         # top-K assets to hold
REBAL = 7        # rebalance positions every N days
REFIT = 30       # refit models every N days
TRAIN_WIN = 252  # rolling training window size (1 year of trading days)
TRAIN_MIN = 120  # min training samples before first prediction
N_SLICES = 300   # random 7-day slices for evaluation
SEED = 42
N_EST = 20       # RF n_estimators (speed/accuracy tradeoff)


# ---------------------------------------------------------------------------
# Pre-compute full feature panel (all causal by construction)
# ---------------------------------------------------------------------------

def build_feature_panel(ind: dict):
    """
    Build (D, A, F) feature array and feature names.
    All features use rolling/shift on past data -> causal per row.

    Returns:
        arr: np.ndarray (D, A, F)
        feat_names: list of str
        fwd_arr: np.ndarray (D, A), fwd_arr[i] = C[i+HOLD]/C[i] - 1
        gate_arr: np.ndarray (D, A) bool
    """
    C  = ind["C"]
    D, A = C.shape
    assets = list(C.columns)

    frames = {}
    frames["mom7"]      = ind["mom7"]
    frames["mom14"]     = ind["mom14"]
    frames["mom30"]     = ind["mom30"]
    frames["rsi14"]     = ind["rsi14"]
    frames["vol20"]     = ind["vol20"]
    frames["atr_pct"]   = (ind["atr14"] / (C + 1e-12)).fillna(0)
    frames["dist50"]    = (C / (ind["sma50"]  + 1e-12) - 1).fillna(0)
    frames["dist200"]   = (C / (ind["sma200"] + 1e-12) - 1).fillna(0)

    hh = ind["hh14"]
    ll = ind["ll14"]
    frames["range_pos"] = ((C - ll) / (hh - ll + 1e-12)).fillna(0.5)
    frames["ret1"]      = ind["ret1"]
    frames["gate_f"]    = ind["gate"].astype(float)

    # BTC regime: scalar per day broadcast to all assets
    btc_sym = next((c for c in assets if "BTC" in c.upper()), assets[0])
    btc_m7 = ind["mom7"][btc_sym].values  # (D,)
    btc_arr = np.tile(btc_m7[:, None], (1, A))  # (D, A)
    frames["btc_mom7"] = pd.DataFrame(btc_arr, index=C.index, columns=C.columns)

    # Breadth: fraction above sma50
    above50 = (C > ind["sma50"]).fillna(False).mean(axis=1).values  # (D,)
    frames["breadth"] = pd.DataFrame(
        np.tile(above50[:, None], (1, A)), index=C.index, columns=C.columns
    )

    # 3d return
    frames["ret3"] = (C / (C.shift(3) + 1e-12) - 1).fillna(0)

    feat_names = list(frames.keys())
    F = len(feat_names)

    arr = np.zeros((D, A, F), dtype=np.float32)
    for fi, fn in enumerate(feat_names):
        arr[:, :, fi] = frames[fn].values.astype(np.float32)

    # Forward return array
    Cv = C.values.astype(np.float64)
    fwd_arr = np.full((D, A), np.nan, dtype=np.float64)
    fwd_arr[:D - HOLD] = Cv[HOLD:] / (Cv[:D - HOLD] + 1e-12) - 1

    gate_arr = ind["gate"].values.astype(bool)

    return arr, feat_names, fwd_arr, gate_arr


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------

def make_labels(y_flat: np.ndarray, y_full_rows: np.ndarray) -> dict:
    """
    y_flat: (N,) flat array of forward returns (from the training window)
    y_full_rows: (T, A) forward returns per row (for median label)
    Returns dict of labels all aligned to y_flat.
    """
    L1 = (y_flat > 0).astype(int)

    # L2: beats cross-sectional median per row
    T, A = y_full_rows.shape
    L2_2d = np.zeros((T, A), dtype=int)
    for t in range(T):
        row = y_full_rows[t]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        med = np.nanmedian(row[valid])
        L2_2d[t, valid] = (row[valid] > med).astype(int)
    # Flatten -- we'll need to mask same as y_flat
    # L2_flat built by caller to match valid mask
    L2_flat = L2_2d.flatten()

    L3 = y_flat.copy()
    return {"L1": L1, "L2": L2_flat, "L3": L3}


# ---------------------------------------------------------------------------
# Walk-forward weight builder (vectorized training)
# ---------------------------------------------------------------------------

def build_weight_matrix(ind: dict, models_config: str = "ensemble") -> pd.DataFrame:
    """
    models_config: "ensemble" -> LR+RF+MLP x 3 labels
                   "lr_L1", "rf_L1", "mlp_L1", "rf_L2", "rf_L3" -> single model/label
    """
    C = ind["C"]
    D, A = C.shape
    dates = C.index
    assets = C.columns

    print(f"    Pre-computing feature panel ({D}d x {A}a)...", flush=True)
    arr, feat_names, fwd_arr, gate_arr = build_feature_panel(ind)
    F = arr.shape[2]

    W = np.zeros((D, A), dtype=np.float32)

    # fitted model state
    scaler = None
    clf_L1_models = []  # list of fitted classifiers for L1
    clf_L2_models = []  # list of fitted classifiers for L2
    reg_L3_models = []  # list of fitted regressors for L3
    models_ready = False

    last_refit = -999
    last_rebal = -999
    last_w = np.zeros(A, dtype=np.float32)

    print(f"    Walk-forward loop (REFIT={REFIT}d, REBAL={REBAL}d)...", flush=True)
    n_refits = 0

    for i in range(D):
        # ---- REFIT ----
        if i - last_refit >= REFIT:
            # Training window: [i - TRAIN_WIN - HOLD .. i - HOLD]
            # All rows whose HOLD-day forward window closed before row i
            train_end = i - HOLD       # last valid training row (fwd closes at i)
            train_start = max(0, train_end - TRAIN_WIN)

            if train_end - train_start >= TRAIN_MIN:
                # Feature block: (T, A, F)
                X_block = arr[train_start:train_end].astype(np.float64)   # (T, A, F)
                y_block = fwd_arr[train_start:train_end]                   # (T, A)
                T_blk = train_end - train_start

                # Flatten to panel
                X_flat = X_block.reshape(T_blk * A, F)
                y_flat  = y_block.reshape(T_blk * A)

                # Valid mask: no NaN in features or label
                valid = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y_flat))
                X_tr = X_flat[valid]
                y_tr  = y_flat[valid]

                if len(X_tr) >= TRAIN_MIN:
                    # Build labels: L2 needs per-row median
                    # Flatten valid mask back to (T_blk, A) shape for L2
                    valid_2d = valid.reshape(T_blk, A)
                    y_L2_2d = np.zeros((T_blk, A), dtype=int)
                    for t in range(T_blk):
                        row_y = y_block[t]
                        row_v = valid_2d[t]
                        if row_v.sum() == 0:
                            continue
                        med = np.nanmedian(row_y[row_v])
                        y_L2_2d[t, row_v] = (row_y[row_v] > med).astype(int)
                    y_L2_flat = y_L2_2d.reshape(T_blk * A)[valid]

                    y_L1 = (y_tr > 0).astype(int)
                    y_L2 = y_L2_flat
                    y_L3 = y_tr

                    # Fit scaler on training data only
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(X_tr)

                    # Define which models to fit
                    if models_config == "ensemble":
                        clf_L1_specs = [
                            ("lr", LogisticRegression(max_iter=300, C=0.1,
                                                       random_state=SEED, solver="lbfgs")),
                            ("rf", RandomForestClassifier(n_estimators=N_EST, max_depth=4,
                                                           random_state=SEED, n_jobs=-1)),
                            ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150,
                                                   random_state=SEED)),
                        ]
                        clf_L2_specs = [
                            ("lr", LogisticRegression(max_iter=300, C=0.1,
                                                       random_state=SEED, solver="lbfgs")),
                            ("rf", RandomForestClassifier(n_estimators=N_EST, max_depth=4,
                                                           random_state=SEED, n_jobs=-1)),
                            ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150,
                                                   random_state=SEED)),
                        ]
                        reg_L3_specs = [
                            ("rf", RandomForestRegressor(n_estimators=N_EST, max_depth=4,
                                                          random_state=SEED, n_jobs=-1)),
                            ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=150,
                                                  random_state=SEED)),
                            ("rf2", RandomForestRegressor(n_estimators=N_EST, max_depth=3,
                                                           random_state=SEED + 1, n_jobs=-1)),
                        ]
                    elif models_config == "lr_L1":
                        clf_L1_specs = [("lr", LogisticRegression(max_iter=300, C=0.1,
                                                                    random_state=SEED, solver="lbfgs"))]
                        clf_L2_specs = []
                        reg_L3_specs = []
                    elif models_config == "rf_L1":
                        clf_L1_specs = [("rf", RandomForestClassifier(n_estimators=N_EST, max_depth=4,
                                                                        random_state=SEED, n_jobs=-1))]
                        clf_L2_specs = []
                        reg_L3_specs = []
                    elif models_config == "mlp_L1":
                        clf_L1_specs = [("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150,
                                                                random_state=SEED))]
                        clf_L2_specs = []
                        reg_L3_specs = []
                    elif models_config == "rf_L2":
                        clf_L1_specs = []
                        clf_L2_specs = [("rf", RandomForestClassifier(n_estimators=N_EST, max_depth=4,
                                                                        random_state=SEED, n_jobs=-1))]
                        reg_L3_specs = []
                    elif models_config == "rf_L3":
                        clf_L1_specs = []
                        clf_L2_specs = []
                        reg_L3_specs = [("rf", RandomForestRegressor(n_estimators=N_EST, max_depth=4,
                                                                      random_state=SEED, n_jobs=-1))]
                    else:
                        clf_L1_specs = []
                        clf_L2_specs = []
                        reg_L3_specs = []

                    clf_L1_models = []
                    if len(np.unique(y_L1)) > 1:
                        for nm, m in clf_L1_specs:
                            try:
                                m.fit(X_sc, y_L1)
                                clf_L1_models.append(m)
                            except Exception:
                                pass

                    clf_L2_models = []
                    if len(np.unique(y_L2)) > 1:
                        for nm, m in clf_L2_specs:
                            try:
                                m.fit(X_sc, y_L2)
                                clf_L2_models.append(m)
                            except Exception:
                                pass

                    reg_L3_models = []
                    for nm, m in reg_L3_specs:
                        try:
                            m.fit(X_sc, y_L3)
                            reg_L3_models.append(m)
                        except Exception:
                            pass

                    models_ready = len(clf_L1_models) + len(clf_L2_models) + len(reg_L3_models) > 0
                    n_refits += 1
                    last_refit = i

        # ---- REBALANCE ----
        if i - last_rebal >= REBAL and models_ready and scaler is not None:
            X_now = arr[i].astype(np.float64)  # (A, F)
            has_nan = np.isnan(X_now).any(axis=1)  # (A,)
            X_clean = np.where(np.isnan(X_now), 0.0, X_now)
            X_sc_now = scaler.transform(X_clean)
            gate_now = gate_arr[i]

            scores_list = []

            for m in clf_L1_models:
                try:
                    prob = m.predict_proba(X_sc_now)
                    classes = list(m.classes_)
                    pos_idx = classes.index(1) if 1 in classes else -1
                    s = prob[:, pos_idx] if pos_idx >= 0 else np.full(A, 0.5)
                    scores_list.append(s)
                except Exception:
                    pass

            for m in clf_L2_models:
                try:
                    prob = m.predict_proba(X_sc_now)
                    classes = list(m.classes_)
                    pos_idx = classes.index(1) if 1 in classes else -1
                    s = prob[:, pos_idx] if pos_idx >= 0 else np.full(A, 0.5)
                    scores_list.append(s)
                except Exception:
                    pass

            for m in reg_L3_models:
                try:
                    pred = m.predict(X_sc_now)
                    ranks = pred.argsort().argsort().astype(float)
                    s = ranks / max(1.0, A - 1.0)
                    scores_list.append(s)
                except Exception:
                    pass

            if scores_list:
                ens = np.mean(scores_list, axis=0)
                ens[has_nan] = -1.0
                ens = np.where(gate_now, ens, -2.0)
                valid = ens > -1.5
                if valid.sum() > 0:
                    k = min(TOPK, int(valid.sum()))
                    top_idx = np.argsort(ens)[-k:]
                    w = np.zeros(A, dtype=np.float32)
                    w[top_idx] = 1.0 / k
                    last_w = w
                else:
                    last_w = np.zeros(A, dtype=np.float32)
            else:
                last_w = np.zeros(A, dtype=np.float32)

            last_rebal = i

        W[i] = last_w

    print(f"    Done. {n_refits} refits performed.", flush=True)
    return pd.DataFrame(W.astype(np.float64), index=dates, columns=assets)


# ---------------------------------------------------------------------------
# Random-slice evaluation
# ---------------------------------------------------------------------------

def random_slice_eval(W: pd.DataFrame, ind: dict, n_slices: int = N_SLICES,
                      hold: int = HOLD, label: str = "engine") -> dict:
    """
    Sample n_slices random 7-day windows.
    Engine: position = W.shift(1) (lagged 1 bar, no lookahead).
    BH: EW all assets.
    """
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    n = len(W)
    max_start = n - hold - 1

    rng_local = np.random.default_rng(SEED + 1)
    starts = rng_local.integers(1, max_start + 1, size=n_slices)  # start>=1 so shift(1) valid

    W_arr = W.values
    R_arr = R.values

    engine_rets = []
    bh_rets     = []

    for s in starts:
        e = s + hold
        # Engine: position at bar t = W[t-1] (shift-1 lag)
        pos = W_arr[s - 1:e - 1]        # (hold, A)
        r   = R_arr[s:e]                 # (hold, A)
        L   = min(len(pos), len(r))
        eng_compound = float(np.prod(1 + (pos[:L] * r[:L]).sum(axis=1)) - 1)

        bh_daily     = R_arr[s:e].mean(axis=1)
        bh_compound  = float(np.prod(1 + bh_daily) - 1)

        engine_rets.append(eng_compound)
        bh_rets.append(bh_compound)

    engine_rets = np.array(engine_rets)
    bh_rets     = np.array(bh_rets)

    down_mask = bh_rets < 0
    up_mask   = ~down_mask

    return {
        "label": label,
        "n_slices": n_slices,
        "positive_rate":     float(np.mean(engine_rets > 0)),
        "mean_return":       float(np.mean(engine_rets)),
        "bh_positive_rate":  float(np.mean(bh_rets > 0)),
        "bh_mean_return":    float(np.mean(bh_rets)),
        "beat_bh_rate":      float(np.mean(engine_rets > bh_rets)),
        "down_week_count":   int(down_mask.sum()),
        "down_week_eng_mean": float(np.mean(engine_rets[down_mask])) if down_mask.sum() > 0 else float("nan"),
        "up_week_count":     int(up_mask.sum()),
        "up_week_eng_mean":  float(np.mean(engine_rets[up_mask])) if up_mask.sum() > 0 else float("nan"),
        # Cash rate: fraction of slices where engine is mostly flat (avg exposure < 5%)
        "cash_pct": float(np.mean([W_arr[s:s + hold].sum(axis=1).mean() < 0.05
                                    for s in starts])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("ENGINE TOURNAMENT CYCLE 1: MODEL-DIVERSITY ENSEMBLE", flush=True)
    print("=" * 70, flush=True)

    print("\n[1] Loading data 2020-01..2026-05...", flush=True)
    ind = ml.load(start="2020-01-01", end="2026-06-01")
    C = ind["C"]
    print(f"    {len(C.columns)} assets, {len(C)} dates "
          f"({C.index[0].date()} -> {C.index[-1].date()})", flush=True)
    print(f"    Assets: {list(C.columns)}", flush=True)
    print(f"    Data load: {time.time()-t_start:.1f}s", flush=True)

    # EW buy-hold baseline
    print("\n[2] EW buy-hold baseline...", flush=True)
    W_bh = pd.DataFrame(1.0 / len(C.columns), index=C.index, columns=C.columns)
    bh_result = random_slice_eval(W_bh, ind, label="EW_buy_hold")
    print(f"    positive_rate={bh_result['positive_rate']:.3f}  "
          f"mean={bh_result['mean_return']*100:.2f}%", flush=True)

    # ---- Engine variants ----
    variants = [
        ("ensemble",  "ENSEMBLE(LR+RF+MLP x L1+L2+L3)"),
        ("lr_L1",     "LR_L1(sign)"),
        ("rf_L1",     "RF_L1(sign)"),
        ("mlp_L1",    "MLP_L1(sign)"),
        ("rf_L2",     "RF_L2(beat-median)"),
        ("rf_L3",     "RF_L3(magnitude)"),
    ]

    results = [bh_result]
    for cfg, name in variants:
        t0 = time.time()
        print(f"\n[3] Building: {name}...", flush=True)
        W = build_weight_matrix(ind, models_config=cfg)
        elapsed = time.time() - t0
        print(f"    Computing random-slice evaluation...", flush=True)
        res = random_slice_eval(W, ind, label=name)
        results.append(res)
        print(f"    positive_rate={res['positive_rate']:.3f}  "
              f"mean={res['mean_return']*100:.2f}%  "
              f"elapsed={elapsed:.0f}s", flush=True)

    # ---- Report ----
    total = time.time() - t_start
    print("\n", flush=True)
    print("=" * 78, flush=True)
    print(f"RESULTS: Random {HOLD}-day Slice Evaluation (n_slices={N_SLICES}, "
          f"total_time={total:.0f}s)", flush=True)
    print(f"  Settings: HOLD={HOLD}d, TOPK={TOPK}, REBAL={REBAL}d, "
          f"REFIT={REFIT}d, TRAIN_WIN={TRAIN_WIN}d, N_EST={N_EST}", flush=True)
    print("=" * 78, flush=True)

    cols = ["Engine", "PosRate", "Mean%", "BeatBH%", "DownWk%", "UpWk%", "Cash%"]
    widths = [34, 8, 8, 8, 9, 9, 7]
    print("".join(c.rjust(w) for c, w in zip(cols, widths)), flush=True)
    print("-" * sum(widths), flush=True)

    for r in results:
        dw = r.get("down_week_eng_mean", float("nan"))
        uw = r.get("up_week_eng_mean", float("nan"))
        dw_s = f"{dw*100:.2f}%" if not np.isnan(dw) else "N/A"
        uw_s = f"{uw*100:.2f}%" if not np.isnan(uw) else "N/A"
        cp_s = f"{r.get('cash_pct', 0)*100:.1f}%" if "cash_pct" in r else "N/A"
        row = [r["label"],
               f"{r['positive_rate']:.3f}",
               f"{r['mean_return']*100:.2f}%",
               f"{r['beat_bh_rate']:.3f}",
               dw_s, uw_s, cp_s]
        print("".join(v.rjust(w) for v, w in zip(row, widths)), flush=True)

    print(flush=True)
    print("WIN CONDITION: positive_rate > 0.55 AND mean return > +2.9%", flush=True)
    print(flush=True)

    # Ensemble-specific verdict
    ens = next(r for r in results if "ENSEMBLE" in r["label"])
    bh  = bh_result
    best_single = max((r for r in results if "ENSEMBLE" not in r["label"] and r["label"] != "EW_buy_hold"),
                      key=lambda r: r["positive_rate"])

    print("DOWN-WEEK ANALYSIS:", flush=True)
    dw_m = ens.get("down_week_eng_mean", float("nan"))
    print(f"  Down weeks (BH<0): {ens['down_week_count']}/{N_SLICES} "
          f"({ens['down_week_count']/N_SLICES*100:.1f}%)", flush=True)
    print(f"  Ensemble mean in down weeks: {dw_m*100:.2f}%  "
          f"(0%=perfect cash, negative=partial exposure)", flush=True)
    print(flush=True)

    print("VERDICT:", flush=True)
    pr_beat = ens["positive_rate"] > bh["positive_rate"]
    mn_beat = ens["mean_return"]   > bh["mean_return"]
    wins    = ens["positive_rate"] > 0.55 and ens["mean_return"] > 0.029
    ens_best_pr = ens["positive_rate"] > best_single["positive_rate"]
    ens_best_mn = ens["mean_return"]   > best_single["mean_return"]

    print(f"  Ensemble positive_rate {ens['positive_rate']:.3f} vs BH {bh['positive_rate']:.3f} "
          f"-> {'BEATS' if pr_beat else 'LOSES'}", flush=True)
    print(f"  Ensemble mean {ens['mean_return']*100:.2f}% vs BH {bh['mean_return']*100:.2f}% "
          f"-> {'BEATS' if mn_beat else 'LOSES'}", flush=True)
    print(f"  WIN CONDITION (pr>0.55 AND mean>2.9%): {'PASS' if wins else 'FAIL'}", flush=True)
    print(f"  Best single model: {best_single['label']} "
          f"pr={best_single['positive_rate']:.3f} "
          f"mean={best_single['mean_return']*100:.2f}%", flush=True)
    print(f"  Ensemble vs best single (positive_rate): {'BETTER' if ens_best_pr else 'WORSE/SAME'}", flush=True)
    print(f"  Ensemble vs best single (mean return):   {'BETTER' if ens_best_mn else 'WORSE/SAME'}", flush=True)

    return results


if __name__ == "__main__":
    main()
