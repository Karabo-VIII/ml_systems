"""src/strat/referee_harness.py -- INDEPENDENT adversarial referee for the engine tournament.

Goal: ONE canonical leak-free evaluation so the 6 engines are comparable on the SAME
slice sampler and the SAME buy-hold baseline. Primary job: independently RE-DERIVE the
ML engine OOS random-slice positive-rate with a STRICT walk-forward (no future feature,
no train/test label overlap, no full-data scaling).

CANONICAL CONVENTIONS (every engine judged identically):
  - Slice = 7 CONSECUTIVE TRADING DAYS sampled uniformly from the OOS region.
  - Engine 7d return = compound of the engine's daily book return over the slice,
    where the book = positions LAGGED 1 bar, taker cost on |dpos|. (signal at d acted at d+1)
  - BH baseline = the SAME book mechanic with W = EW over assets-with-valid-price
    (fillna(0)=cash for pre-listing), lagged 1 bar, same cost. Cadence-invariant.
  - OOS region: walk-forward predictions only. ML labels are STRICTLY closed before
    the prediction date (label window d->d+7 must end <= T-1 day of the train cutoff).
  - n_slices >= 500, K=3 independent seeds reported (answer-frequency).

No emoji (cp1252). Does NOT git commit.
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

COST = lab.COST  # taker round-trip


# ============================================================
# CANONICAL BOOK RETURN from a weight matrix W (dates x assets)
# ============================================================
def book_daily_returns(W: pd.DataFrame, ind: dict) -> pd.Series:
    """Engine daily return: positions LAGGED 1 bar, taker cost on |dpos|. Identical to lab.evaluate."""
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    return bret


def bh_ew_weights(ind: dict) -> pd.DataFrame:
    """EW buy-hold: equal weight across assets that have a valid price that day (pre-listing = cash).
    This is the cadence-invariant fixed-EW convention (fillna(0)=cash) from project memory.
    """
    C = ind["C"]
    present = C.notna().astype(float)
    n = present.sum(axis=1).replace(0, np.nan)
    W = present.div(n, axis=0).fillna(0.0)
    return W


# ============================================================
# CANONICAL RANDOM-SLICE EVALUATOR (7 consecutive trading days)
# ============================================================
def slice_stats(bret: pd.Series, bh: pd.Series, oos_start: str, oos_end: str,
                n_slices: int, slice_days: int, seed: int) -> dict:
    """Random 7-consecutive-trading-day slices from [oos_start, oos_end). Same slices for eng & bh."""
    rng = np.random.default_rng(seed)
    idx = bret.index
    m = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))
    oos_idx = idx[m]
    if len(oos_idx) < slice_days + 5:
        return {"error": "insufficient OOS data", "n_oos": len(oos_idx)}
    max_start = len(oos_idx) - slice_days
    eng, bhr = [], []
    for _ in range(n_slices):
        si = rng.integers(0, max_start)
        sl = oos_idx[si: si + slice_days]
        eng.append(float((1 + bret.loc[sl]).prod() - 1))
        bhr.append(float((1 + bh.loc[sl]).prod() - 1))
    eng = np.array(eng); bhr = np.array(bhr)
    down = bhr < 0
    return {
        "n_slices": n_slices,
        "pos_rate": round(100 * float((eng > 0).mean()), 1),
        "mean_pct": round(100 * float(eng.mean()), 2),
        "median_pct": round(100 * float(np.median(eng)), 2),
        "p05_pct": round(100 * float(np.percentile(eng, 5)), 2),
        "beat_bh_pct": round(100 * float((eng > bhr).mean()), 1),
        "down_wk_eng_mean": round(100 * float(eng[down].mean()), 2) if down.any() else None,
        "down_wk_eng_posrate": round(100 * float((eng[down] > 0).mean()), 1) if down.any() else None,
        "down_wk_cash_rate": round(100 * float((np.abs(eng[down]) < 1e-9).mean()), 1) if down.any() else None,
        "n_down": int(down.sum()),
    }


def bh_slice_stats(bh: pd.Series, oos_start: str, oos_end: str,
                   n_slices: int, slice_days: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = bh.index
    m = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))
    oos_idx = idx[m]
    max_start = len(oos_idx) - slice_days
    r = []
    for _ in range(n_slices):
        si = rng.integers(0, max_start)
        sl = oos_idx[si: si + slice_days]
        r.append(float((1 + bh.loc[sl]).prod() - 1))
    r = np.array(r)
    return {
        "n_slices": n_slices,
        "pos_rate": round(100 * float((r > 0).mean()), 1),
        "mean_pct": round(100 * float(r.mean()), 2),
        "median_pct": round(100 * float(np.median(r)), 2),
        "p05_pct": round(100 * float(np.percentile(r, 5)), 2),
    }


# ============================================================
# STRICT ML WALK-FORWARD (independent re-derivation)
# ============================================================
def build_features(ind: dict):
    """Causal features at row d use data <= d only. Label = d->d+7 fwd return (>0 -> 1)."""
    C = ind["C"]; eps = 1e-8
    panels = {
        "dist_sma200": C / (ind["sma200"] + eps) - 1,
        "dist_sma50":  C / (ind["sma50"] + eps) - 1,
        "range_pos":   (C - ind["ll14"]) / ((ind["hh14"] - ind["ll14"]) + eps),
        "rsi14":       ind["rsi14"],
        "vol20":       ind["vol20"],
        "mom7":        ind["mom7"],
        "mom14":       ind["mom14"],
        "mom30":       ind["mom30"],
        "ret1":        ind["ret1"],
        "ret3":        C / C.shift(3) - 1,
        "mom7_rank":   ind["mom7"].rank(axis=1, pct=True),
        "mom14_rank":  ind["mom14"].rank(axis=1, pct=True),
        "gate":        ind["gate"].astype(float),
    }
    # scalar breadth + btc regime broadcast per asset
    breadth = (C > ind["sma50"]).astype(float).mean(axis=1)
    btc_reg = (C["BTCUSDT"] > ind["sma200"]["BTCUSDT"]).astype(float).fillna(0.0)
    stacked = {}
    for name, df in panels.items():
        s = df.stack(dropna=False); s.index.names = ["date", "asset"]; stacked[name] = s
    feat = pd.DataFrame(stacked)
    dl = feat.index.get_level_values("date")
    feat["breadth"] = breadth.reindex(dl).values
    feat["btc_regime"] = btc_reg.reindex(dl).values
    cols = list(panels.keys()) + ["breadth", "btc_regime"]
    fwd = (C.shift(-7) / C - 1)  # LABEL ONLY
    fl = fwd.stack(dropna=False); fl.index.names = ["date", "asset"]
    label = (fl > 0).astype(float).where(fl.notna(), np.nan)
    return feat, cols, label, fl


def strict_walk_forward(ind: dict, oos_start: str, retrain_every: int = 90,
                        min_train: int = 500, model_type: str = "hgb"):
    """STRICT: training at predict-date T uses ONLY rows whose label window CLOSED before T.
    A row at date d has label window d -> d+7. It is train-eligible at cutoff T iff
    d + 7 trading days <= T - 1 (label fully observed strictly before T). We enforce this
    using the trading-day index position, NOT calendar days (the lab is daily but uses
    calendar-gap-free trading bars; +8 calendar in the original is a sloppy proxy).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feat, cols, label, fwd = build_features(ind)
    all_dates = list(ind["C"].index)
    pos_of = {d: i for i, d in enumerate(all_dates)}
    # train-eligible label-close position: row at date-pos p is eligible at predict-pos i iff p+7 <= i-1
    dl_pos = np.array([pos_of[d] for d in feat.index.get_level_values("date")])
    feat_X = feat[cols]
    valid_feat = feat_X.notna().all(axis=1).values
    valid_lbl = label.notna().values
    Xv = feat_X.values
    yv = label.values

    oos_pos = pos_of.get(pd.Timestamp(oos_start))
    if oos_pos is None:
        # nearest >= oos_start
        oos_pos = next(i for i, d in enumerate(all_dates) if d >= pd.Timestamp(oos_start))

    model = None; scaler = None; last_retrain = -10**9
    preds = []
    for i in range(oos_pos, len(all_dates)):
        T = all_dates[i]
        if i - last_retrain >= retrain_every:
            # STRICT label-closure: row pos p eligible iff p+7 <= i-1  ->  p <= i-8
            tr = (dl_pos <= i - 8) & valid_feat & valid_lbl
            if tr.sum() >= min_train:
                Xtr = Xv[tr]; ytr = yv[tr]
                if model_type == "hgb":
                    model = HistGradientBoostingClassifier(
                        max_iter=200, max_depth=4, learning_rate=0.05,
                        min_samples_leaf=20, l2_regularization=1.0, random_state=42)
                    model.fit(Xtr, ytr); scaler = None
                else:
                    scaler = StandardScaler(); Xs = scaler.fit_transform(Xtr)
                    model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                    model.fit(Xs, ytr)
                last_retrain = i
        if model is None:
            continue
        day = (dl_pos == i) & valid_feat
        if not day.any():
            continue
        Xday = Xv[day]
        if scaler is not None:
            Xday = scaler.transform(Xday)
        p = model.predict_proba(Xday)[:, 1]
        idx_day = feat.index[day]
        lbl_day = label.values[day]
        fwd_day = fwd.values[day]
        for j, (d, sym) in enumerate(idx_day):
            preds.append({"date": d, "asset": sym, "prob": float(p[j]),
                          "label": float(lbl_day[j]) if not pd.isna(lbl_day[j]) else np.nan,
                          "fwd": float(fwd_day[j]) if not pd.isna(fwd_day[j]) else np.nan})
    return pd.DataFrame(preds)


def ml_weight_matrix(pred_df: pd.DataFrame, ind: dict, top_k: int, thr: float) -> pd.DataFrame:
    """Build a daily W from ML predictions: on each prediction date, hold top-K assets with prob>=thr (EW),
    carry until the next prediction date. CASH if none clear thr. Causal: W at d acted at d+1 by book_daily_returns.
    """
    C = ind["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    by_date = {d: g for d, g in pred_df.groupby("date")}
    pred_dates = sorted(by_date.keys())
    if not pred_dates:
        return W
    pd_set = set(pred_dates)
    cur = {}
    for d in C.index:
        if d in pd_set:
            g = by_date[d]
            elig = g[g["prob"] >= thr].sort_values("prob", ascending=False).head(top_k)
            if len(elig) == 0:
                cur = {}
            else:
                w = 1.0 / len(elig)
                cur = {s: w for s in elig["asset"].tolist()}
        if d >= pred_dates[0]:
            for s, wv in cur.items():
                if s in W.columns:
                    W.loc[d, s] = wv
    return W


# ============================================================
# DIVERSIFICATION / SMOOTHING CEILING (no prediction)
# ============================================================
def no_prediction_ceiling(ind: dict, oos_start: str, oos_end: str, n_slices: int, seeds: list) -> dict:
    """Max positive-rate achievable with NO prediction: pure EW BH and gated-EW (cash down-trends).
    Gated = EW over assets above their own SMA200; cash when none. This is the structural ceiling
    of 'smoothing/regime gating without forecasting'.
    """
    C = ind["C"]; gate = ind["gate"]
    # gated EW
    g = gate.astype(float)
    gW = g.div(g.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)
    gated_b = book_daily_returns(gW, ind)
    out = {}
    for tag, b in [("EW_BH", bh_b), ("Gated_EW", gated_b)]:
        prs = [slice_stats(b, bh_b, oos_start, oos_end, n_slices, 7, s)["pos_rate"] for s in seeds]
        means = [slice_stats(b, bh_b, oos_start, oos_end, n_slices, 7, s)["mean_pct"] for s in seeds]
        out[tag] = {"pos_rate_mean": round(float(np.mean(prs)), 1), "pos_rate_seeds": prs,
                    "mean_pct_mean": round(float(np.mean(means)), 2)}
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    OOS_START = "2022-01-01"   # canonical OOS: the regime-router's boundary (hardest, includes the bear)
    OOS_END = "2026-06-01"
    N = 500
    SEEDS = [11, 23, 42]

    print("=" * 76)
    print("INDEPENDENT REFEREE HARNESS -- canonical leak-free re-derivation")
    print(f"OOS: {OOS_START} -> {OOS_END} | n_slices={N} | seeds={SEEDS} | 7 consecutive trading days")
    print("=" * 76)

    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)

    # --- BH baseline (canonical) ---
    print("\n[BH] canonical EW buy-hold baseline:")
    bh_seed = {s: bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS}
    bh_pr = [bh_seed[s]["pos_rate"] for s in SEEDS]
    bh_mn = [bh_seed[s]["mean_pct"] for s in SEEDS]
    print(f"  pos_rate seeds={bh_pr} mean={round(float(np.mean(bh_pr)),1)}%")
    print(f"  mean_pct seeds={bh_mn} mean={round(float(np.mean(bh_mn)),2)}%")

    # --- STRICT ML walk-forward re-derivation (HGB) ---
    print("\n[ML] strict walk-forward (HGB), STRICT label-closure (p<=i-8 trading-bars)...")
    pred = strict_walk_forward(ind, OOS_START, model_type="hgb")
    from sklearn.metrics import roc_auc_score
    va = pred[pred["label"].notna() & pred["prob"].notna()]
    auc = float(roc_auc_score(va["label"], va["prob"])) if len(va) > 50 else np.nan
    print(f"  OOS preds: {len(pred)}  AUC: {auc:.4f}")

    # ML weight matrix + slice stats, across top_k/thr configs
    ml_configs = [(3, 0.50), (2, 0.45), (1, 0.45), (5, 0.50)]
    ml_results = {}
    for (k, thr) in ml_configs:
        W = ml_weight_matrix(pred, ind, k, thr)
        b = book_daily_returns(W, ind)
        prs = [slice_stats(b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in prs]; mn = [x["mean_pct"] for x in prs]
        bw = [x["beat_bh_pct"] for x in prs]
        dn = [x["down_wk_cash_rate"] for x in prs]
        ml_results[(k, thr)] = {"pos_rate": round(float(np.mean(pr)), 1), "pos_rate_seeds": pr,
                                "mean_pct": round(float(np.mean(mn)), 2), "beat_bh": round(float(np.mean(bw)), 1),
                                "down_cash": round(float(np.mean(dn)), 1),
                                "avg_expo": round(float((W.sum(axis=1) > 0).loc[C.index >= OOS_START].mean()), 2)}
        print(f"  HGB top{k} thr{thr}: pos_rate={ml_results[(k,thr)]['pos_rate']}% (seeds {pr}) "
              f"mean={ml_results[(k,thr)]['mean_pct']}% beat_bh={ml_results[(k,thr)]['beat_bh']}% "
              f"down_cash={ml_results[(k,thr)]['down_cash']}% expo={ml_results[(k,thr)]['avg_expo']}")

    # --- No-prediction ceiling ---
    print("\n[CEILING] no-prediction smoothing/gating ceiling:")
    ceil = no_prediction_ceiling(ind, OOS_START, OOS_END, N, SEEDS)
    for k, v in ceil.items():
        print(f"  {k}: pos_rate={v['pos_rate_mean']}% (seeds {v['pos_rate_seeds']}) mean={v['mean_pct_mean']}%")

    # --- Adaptive regime-router (re-run through canonical harness) ---
    print("\n[ROUTER] adaptive regime-router through CANONICAL slices:")
    import strat.adaptive_meta_engine as ame
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    Wr = ame.build_weight_matrix(ind, vthr)
    rb = book_daily_returns(Wr, ind)
    rprs = [slice_stats(rb, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    rpr = [x["pos_rate"] for x in rprs]; rmn = [x["mean_pct"] for x in rprs]
    rdn = [x["down_wk_eng_mean"] for x in rprs]; rbw = [x["beat_bh_pct"] for x in rprs]
    router = {"pos_rate": round(float(np.mean(rpr)), 1), "pos_rate_seeds": rpr,
              "mean_pct": round(float(np.mean(rmn)), 2), "beat_bh": round(float(np.mean(rbw)), 1),
              "down_wk_mean": round(float(np.mean(rdn)), 2),
              "avg_expo": round(float((Wr.sum(axis=1)).loc[C.index >= OOS_START].mean()), 2)}
    print(f"  router: pos_rate={router['pos_rate']}% (seeds {rpr}) mean={router['mean_pct']}% "
          f"beat_bh={router['beat_bh']}% down_wk_mean={router['down_wk_mean']}% expo={router['avg_expo']}")

    out = {
        "oos": [OOS_START, OOS_END], "n_slices": N, "seeds": SEEDS,
        "bh": {"pos_rate": round(float(np.mean(bh_pr)), 1), "pos_rate_seeds": bh_pr,
               "mean_pct": round(float(np.mean(bh_mn)), 2)},
        "ml_auc": auc if not np.isnan(auc) else None,
        "ml": {f"top{k}_thr{thr}": v for (k, thr), v in ml_results.items()},
        "ceiling": ceil,
        "router": router,
        "runtime_s": round(time.time() - t0, 1),
    }
    outp = ROOT.parent / "runs" / "strat" / "referee_harness_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
