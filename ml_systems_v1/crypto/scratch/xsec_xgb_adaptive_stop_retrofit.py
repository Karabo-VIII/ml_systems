"""G6 retrofit: Adaptive triple-barrier on xsec K=1 stop variant.

Closes G6 from gap audit 2026-04-25.

Reference:
  Lopez de Prado, M. (2018). Advances in Financial Machine Learning, Ch. 3.
    Triple-barrier method with vol-scaled exits.
  Memory: frontier_final_state_2026_04_23.md C2 finding -- adaptive (2-3sigma * sqrt(H))
    stops gave 3-4x better mean trade vs fixed -10% stop on xsec K=5 picks.

What this does:
  1. Loads the same panel as scratch/xsec_xgb_walkforward.py (XGB rank:ndcg, K=1 FULL).
  2. Compares fixed -10% stop vs adaptive (k * sigma_7d * sqrt(3)) stop.
  3. Walk-forward across 3 windows (Oct24-Mar25, Mar25-Sep25, Sep25-Apr26).
  4. Output: sharpe / DD / total return per (window x stop_mode).

Run:
  python scratch/xsec_xgb_adaptive_stop_retrofit.py
"""
from __future__ import annotations

import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs" / "g6_adaptive_stop_retrofit"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAKER_RT = 0.08

# Load meta-labeler v8 once
META_PKL = ROOT / "models" / "meta_labeler" / "v8_catboost.pkl"
meta_obj = pickle.load(open(META_PKL, "rb")) if META_PKL.exists() else None
meta_model = meta_obj["model"] if isinstance(meta_obj, dict) and "model" in meta_obj else None
meta_features = meta_obj.get("feature_names", []) if isinstance(meta_obj, dict) else []


def build_panel() -> pd.DataFrame:
    """Same panel construction as xsec_xgb_walkforward.py."""
    all_fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
    btc = pl.read_parquet(DATA / "btcusdt_v50_chimera.parquet").to_pandas()
    btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms").dt.date
    btc_d = btc.groupby("date").agg({"close": "last"}).reset_index()
    btc_d["btc_30d"] = btc_d["close"].pct_change(30)
    btc_d["btc_ret_1d"] = btc_d["close"].pct_change()
    btc_d["btc_ret_7d"] = btc_d["close"].pct_change(7)
    btc_d["date"] = pd.to_datetime(btc_d["date"])

    rows = []
    for fp in all_fps:
        df = pl.read_parquet(fp).to_pandas()
        if len(df) < 1000:
            continue
        a = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        agg = {"close": "last", "open": "first", "high": "max", "low": "min", "volume": "sum"}
        for c in df.columns:
            if c.startswith("norm_") or c.startswith("xd_") or c == "hurst_regime":
                agg[c] = "last"
        d = df.groupby("date").agg(agg).reset_index()
        d["ret_1d"] = d["close"].pct_change()
        d["ret_3d"] = d["close"].pct_change(3)
        d["ret_7d"] = d["close"].pct_change(7)
        d["ret_14d"] = d["close"].pct_change(14)
        d["vol_7d"] = d["ret_1d"].rolling(7).std()
        d["vol_14d"] = d["ret_1d"].rolling(14).std()
        d["vol_30d"] = d["ret_1d"].rolling(30).std()
        d["hl"] = (d["high"] - d["low"]) / d["open"]
        d["fwd_1d"] = d["close"].shift(-1) / d["close"] - 1
        d["fwd_2d"] = d["close"].shift(-2) / d["close"] - 1
        d["fwd_3d"] = d["close"].shift(-3) / d["close"] - 1
        d["max_fwd_3d"] = np.maximum.reduce([d["fwd_1d"], d["fwd_2d"], d["fwd_3d"]])
        d["min_fwd_3d"] = np.minimum.reduce([d["fwd_1d"], d["fwd_2d"], d["fwd_3d"]])
        d["bc_ratio"] = (d["volume"] / d["volume"].rolling(30).mean()).fillna(1.0)
        d["bc_ratio_trend_3d"] = d["bc_ratio"] - d["bc_ratio"].shift(3)
        d["flow_persistence"] = d.get("norm_flow_imbalance", pd.Series(np.zeros(len(d)))).rolling(7).mean()
        d["vpin_spike"] = ((d.get("norm_vpin", pd.Series(np.zeros(len(d)))) -
                           d.get("norm_vpin", pd.Series(np.zeros(len(d)))).rolling(30).mean()) > 0).astype(float)
        d["asset"] = a
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True).dropna(subset=["fwd_3d"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(btc_d[["date", "btc_30d", "btc_ret_1d", "btc_ret_7d"]], on="date", how="left")
    panel["asset_vs_btc_7d"] = panel["ret_7d"] - panel["btc_ret_7d"]
    panel["btc_regime"] = np.sign(panel["btc_ret_7d"].fillna(0))
    return panel


def score_meta(te_df: pd.DataFrame) -> np.ndarray:
    if meta_model is None:
        return np.ones(len(te_df)) * 0.5
    X_cols = []
    for f in meta_features:
        if f in te_df.columns:
            X_cols.append(te_df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(te_df)))
    X = np.vstack(X_cols).T if X_cols else np.zeros((len(te_df), 1))
    try:
        p = meta_model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.flatten()
    except Exception:
        return np.ones(len(te_df)) * 0.5


def adaptive_stop_for(row: pd.Series, k: float, horizon_days: int = 3) -> float:
    """k * sigma_7d * sqrt(H), floored at 5% to avoid getting stopped on tiny moves."""
    sig = row.get("vol_7d", 0.02)
    if pd.isna(sig) or sig <= 0:
        sig = 0.02
    return float(max(0.05, k * sig * np.sqrt(horizon_days)))


def bt_full(te: pd.DataFrame, K_long: int, stop_mode: str, k_adaptive: float = 2.0,
            regime_gate: bool = True, meta_gate: bool = True, meta_thresh: float = 0.45,
            bear_thresh: float = -0.15) -> dict:
    """K=K_long FULL stack with selectable stop."""
    daily = []
    n_meta_reject = n_regime_block = n_stops_hit = 0
    for d in sorted(te["date"].unique()):
        grp = te[te["date"] == d]
        if len(grp) < K_long:
            daily.append(0); continue
        btc30 = float(grp["btc_30d"].iloc[0]) if len(grp) else 0.0
        if regime_gate and (pd.isna(btc30) or btc30 < bear_thresh):
            n_regime_block += 1
            daily.append(0); continue
        long_r = 0.0
        if K_long > 0:
            top = grp.sort_values("pred", ascending=False).head(K_long).copy()
            if meta_gate:
                p_wins = score_meta(top)
                top = top.assign(p_win=p_wins)
                kept = top[top["p_win"] >= meta_thresh]
                n_meta_reject += (len(top) - len(kept))
                top = kept
                if len(top) == 0:
                    daily.append(0); continue
            rs = []
            for _, r in top.iterrows():
                p = r["fwd_3d"]
                # Stop logic
                if stop_mode == "fixed_10":
                    stop = 0.10
                elif stop_mode == "adaptive_k2":
                    stop = adaptive_stop_for(r, k_adaptive)
                elif stop_mode == "adaptive_k3":
                    stop = adaptive_stop_for(r, 3.0)
                elif stop_mode == "tighter_min10_k2":
                    # Pareto attempt: take TIGHTER of fixed_10 and 2*sigma*sqrt(3)
                    stop = min(0.10, adaptive_stop_for(r, 2.0))
                elif stop_mode == "wider_max10_k3":
                    # Take WIDER of fixed_10 and 3*sigma*sqrt(3) -- give more room on vol days
                    stop = max(0.10, adaptive_stop_for(r, 3.0))
                else:
                    stop = 0.10
                if r["min_fwd_3d"] < -stop:
                    p = -stop
                    n_stops_hit += 1
                rs.append(p)
            long_r = float(np.mean(rs)) * 100.0 - MAKER_RT
        daily.append(long_r / 3.0)
    r = np.array(daily)
    if r.std() == 0:
        return {"total": 0, "sharpe": 0, "dd": 0, "n_stops": n_stops_hit,
                "meta_rej": n_meta_reject, "reg_blk": n_regime_block}
    eq = np.cumprod(1 + r / 100)
    cm = np.maximum.accumulate(eq)
    return {
        "total": (eq[-1] - 1) * 100,
        "sharpe": r.mean() * 252 / (r.std() * np.sqrt(252)),
        "dd": ((eq - cm) / cm).min() * 100,
        "n_stops": n_stops_hit,
        "meta_rej": n_meta_reject,
        "reg_blk": n_regime_block,
    }


def main() -> None:
    print("[g6] building panel...")
    panel = build_panel()
    print(f"[g6] panel: {panel.shape}")

    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    panel["rank_target"] = panel.groupby("date")["fwd_3d"].rank(pct=True).apply(lambda x: int(x * 31))

    feat_list = [c for c in panel.columns if c.startswith("norm_") or c.startswith("xd_")]
    feat_list += ["ret_1d", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "vol_30d", "hl"]
    feat_list = [c for c in feat_list if c in panel.columns]

    import xgboost as xgb

    def fit_predict(train_end: str, test_start: str, test_end: str) -> pd.DataFrame | None:
        tr = panel[panel["date"] < train_end].copy()
        te = panel[(panel["date"] >= test_start) & (panel["date"] < test_end)].copy()
        if len(tr) < 5000 or len(te) < 500:
            return None
        tr_g = tr.groupby("date").size().values
        ranker = xgb.XGBRanker(
            objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
            max_depth=6, n_estimators=500, random_state=42, eval_metric="ndcg@5",
        )
        ranker.fit(tr[feat_list].fillna(0).values, tr["rank_target"].values, group=tr_g, verbose=False)
        te["pred"] = ranker.predict(te[feat_list].fillna(0).values)
        return te

    windows = [
        ("2024-10-01", "2024-10-01", "2025-03-16", "WF1"),
        ("2025-03-16", "2025-03-16", "2025-09-01", "WF2"),
        ("2025-09-01", "2025-09-01", "2026-04-16", "WF3"),
        ("2024-10-01", "2024-10-01", "2026-04-16", "COMBINED"),
    ]
    rows = []
    stop_modes = ["fixed_10", "adaptive_k2", "adaptive_k3", "tighter_min10_k2", "wider_max10_k3"]
    for K_long in (1, 5):
        print(f"\n[g6] === K={K_long} FULL: 5 stop modes ===")
        print(f"{'Window':<10} {'StopMode':<18} {'Total%':>10} {'Sh':>7} {'DD%':>8} {'N_stops':>8}")
        for train_end, ts, te_end, label in windows:
            te = fit_predict(train_end, ts, te_end)
            if te is None:
                continue
            for stop_mode in stop_modes:
                r = bt_full(te, K_long=K_long, stop_mode=stop_mode)
                rows.append({"K": K_long, "window": label, "stop_mode": stop_mode, **r})
                print(f"{label:<10} {stop_mode:<18} {r['total']:>+8.1f}% {r['sharpe']:>+6.2f} {r['dd']:>+7.1f}% {r['n_stops']:>8}")

    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "results.csv", index=False)
    print(f"\n[g6] saved: {LOG_DIR/'results.csv'}")

    # Summary: per K, find Pareto improvement on COMBINED
    for K_long in (1, 5):
        combined = df[(df["window"] == "COMBINED") & (df["K"] == K_long)].copy()
        if len(combined) < 2:
            continue
        baseline = combined[combined["stop_mode"] == "fixed_10"].iloc[0]
        winners = combined[(combined["sharpe"] > baseline["sharpe"]) & (combined["dd"] > baseline["dd"])]
        print(f"\n[g6] === K={K_long} COMBINED summary ===")
        print(f"  Baseline (fixed_10):  Sh {baseline['sharpe']:+.2f}  DD {baseline['dd']:+.1f}%  Total {baseline['total']:+.1f}%")
        if len(winners) > 0:
            best = winners.sort_values("sharpe", ascending=False).iloc[0]
            print(f"  PARETO WIN ({best['stop_mode']}):  Sh {best['sharpe']:+.2f}  DD {best['dd']:+.1f}%  Total {best['total']:+.1f}%")
        else:
            best_sh = combined.sort_values("sharpe", ascending=False).iloc[0]
            best_dd = combined.sort_values("dd", ascending=False).iloc[0]
            print(f"  No strict Pareto win. Best Sharpe: {best_sh['stop_mode']} ({best_sh['sharpe']:+.2f}). "
                  f"Best DD: {best_dd['stop_mode']} ({best_dd['dd']:+.1f}%)")


if __name__ == "__main__":
    main()
