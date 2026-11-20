"""Combined research: setup classifier threshold sweep + fade-on-xgb.

Items from research plan:
    #1 Setup classifier threshold sweep (fade gate at 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    #3 Apply fade gate to xgb_K3_long_WEALTH40

Reuses scripts/xsec_setup_gate.py's simulate_xsec_gated function via import.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Load classifiers
SETUP_CLFS = {}
for name in ["bounce_ml", "fade_ml", "swing_ml"]:
    with open(MODELS / name / f"{name}_v1.pkl", "rb") as f:
        SETUP_CLFS[name.replace("_ml", "")] = pickle.load(f)

META_PKL = MODELS / "meta_labeler" / "v8_catboost.pkl"
meta_obj = pickle.load(open(META_PKL, "rb"))
meta_model = meta_obj["model"]
meta_features = meta_obj.get("feature_names", [])

MAKER_RT = 0.08
TRAIN_END = "2024-10-01"
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"
CAPITAL = 10000.0

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

# Build panel
all_fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
rows = []
for fp in all_fps:
    asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
    if asset not in UNIVERSE:
        continue
    try:
        df = pl.read_parquet(fp).to_pandas()
    except Exception:
        continue
    if len(df) < 1000:
        continue
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    agg = {"close": "last", "open": "first", "high": "max", "low": "min", "volume": "sum"}
    for c in df.columns:
        if c.startswith("norm_") or c.startswith("xd_") or c == "hurst_regime":
            agg[c] = "last"
    d = df.groupby("date").agg(agg).reset_index()
    d["ret_d"] = d["close"].pct_change()
    d["ret_3d"] = d["close"].pct_change(3)
    d["ret_7d"] = d["close"].pct_change(7)
    d["ret_14d"] = d["close"].pct_change(14)
    d["vol_7d"] = d["ret_d"].rolling(7).std()
    d["vol_30d"] = d["ret_d"].rolling(30).std()
    d["hl"] = (d["high"] - d["low"]) / d["open"]
    d["fwd_1d"] = d["close"].shift(-1) / d["close"] - 1
    d["fwd_2d"] = d["close"].shift(-2) / d["close"] - 1
    d["fwd_3d"] = d["close"].shift(-3) / d["close"] - 1
    d["max_fwd_3d"] = np.maximum.reduce([d["fwd_1d"], d["fwd_2d"], d["fwd_3d"]])
    d["min_fwd_3d"] = np.minimum.reduce([d["fwd_1d"], d["fwd_2d"], d["fwd_3d"]])
    d["asset"] = asset
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=["fwd_3d"])
panel["date"] = pd.to_datetime(panel["date"])

btc = pl.read_parquet(DATA / "btcusdt_v50_chimera.parquet", columns=["timestamp", "close"]).to_pandas()
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms").dt.date
btc_d = btc.groupby("date").agg({"close": "last"}).reset_index()
btc_d["btc_30d"] = btc_d["close"].pct_change(30)
btc_d["date"] = pd.to_datetime(btc_d["date"])
panel = panel.merge(btc_d[["date", "btc_30d"]], on="date", how="left")

feat_list = [c for c in panel.columns if c.startswith("norm_") or c.startswith("xd_")]
feat_list += ["ret_d", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "vol_30d", "hl", "hurst_regime"]
feat_list = [c for c in feat_list if c in panel.columns]
panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
panel["rank_target"] = panel.groupby("date")["fwd_3d"].rank(pct=True).apply(lambda x: int(x * 31))
tr = panel[panel["date"] < TRAIN_END]
te = panel[(panel["date"] >= TEST_START) & (panel["date"] < TEST_END)].copy()
tr_g = tr.groupby("date").size().values

print("[info] training ranker...")
t0 = time.time()
ranker = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
                       max_depth=6, n_estimators=500, random_state=42, eval_metric="ndcg@5")
ranker.fit(tr[feat_list].fillna(0).values, tr["rank_target"].values, group=tr_g, verbose=False)
te["pred"] = ranker.predict(te[feat_list].fillna(0).values)
print(f"[info] trained in {time.time() - t0:.1f}s")


def score_meta(df):
    X_cols = []
    for f in meta_features:
        if f in df.columns:
            X_cols.append(df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(df)))
    X = np.vstack(X_cols).T
    p = meta_model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p.flatten()


def score_setup(df, name):
    clf = SETUP_CLFS[name]["clf"]
    feats = SETUP_CLFS[name]["features"]
    X_cols = []
    for f in feats:
        if f in df.columns:
            X_cols.append(df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(df)))
    X = np.vstack(X_cols).T
    p = clf.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p.flatten()


te["p_win"] = score_meta(te)
te["p_bounce"] = score_setup(te, "bounce")
te["p_fade"] = score_setup(te, "fade")
te["p_swing"] = score_setup(te, "swing")


def run_sim(K_long, K_short, stop, regime_gate, meta_thresh, setup_gate, setup_thresh):
    dates = sorted(te["date"].unique())
    daily_rets = []
    for d in dates:
        grp = te[te["date"] == d]
        if len(grp) < K_long + K_short:
            daily_rets.append(0.0); continue
        if regime_gate:
            btc30 = float(grp["btc_30d"].iloc[0]) if len(grp) else 0
            if pd.isna(btc30) or btc30 < -0.15:
                daily_rets.append(0.0); continue
        long_r = short_r = 0.0
        if K_long > 0:
            top = grp.sort_values("pred", ascending=False).head(K_long).copy()
            if meta_thresh > 0:
                top = top[top["p_win"] >= meta_thresh]
            if setup_gate in ("bounce", "fade", "swing"):
                top = top[top[f"p_{setup_gate}"] >= setup_thresh]
            elif setup_gate == "any":
                mask = ((top["p_bounce"] >= setup_thresh) |
                        (top["p_fade"] >= setup_thresh) |
                        (top["p_swing"] >= setup_thresh))
                top = top[mask]
            rs = []
            for _, r in top.iterrows():
                p = r["fwd_3d"]
                if stop and r["min_fwd_3d"] < -stop:
                    p = -stop
                rs.append(p)
            long_r = (np.mean(rs) * 100 - MAKER_RT) if rs else 0.0
        if K_short > 0:
            bot = grp.sort_values("pred", ascending=True).head(K_short)
            rs = []
            for _, r in bot.iterrows():
                p = -r["fwd_3d"]
                if stop and r["max_fwd_3d"] > stop:
                    p = -stop
                rs.append(p)
            short_r = (np.mean(rs) * 100 - MAKER_RT) if rs else 0.0
        n_sides = (1 if K_long > 0 else 0) + (1 if K_short > 0 else 0)
        daily_rets.append((long_r + short_r) / max(n_sides, 1) / 3)
    r = np.array(daily_rets) / 100.0
    eq = CAPITAL * np.cumprod(1 + r)
    total = (eq[-1] / CAPITAL - 1) * 100
    days = len(dates)
    days_span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days_span) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    return {"days": days, "total_ret": total, "cagr": cagr, "sharpe": sharpe, "dd": dd}


# ==========================================================================
# Experiment A: Fade-gate threshold sweep on xsec K=10+10
# ==========================================================================
print("\n" + "=" * 80)
print("A. FADE-GATE THRESHOLD SWEEP -- xsec K=10+10 delta-neutral on U50_LIQUID")
print("=" * 80)
print(f"{'fade_thresh':<14} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7} {'ret%':>8}")
print("-" * 80)
sweep_A = []
for thresh in [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    gate = "fade" if thresh > 0 else "none"
    m = run_sim(K_long=10, K_short=10, stop=0.10, regime_gate=True,
                meta_thresh=0.45, setup_gate=gate, setup_thresh=thresh)
    sweep_A.append({"fade_thresh": thresh, **m})
    print(f"{thresh:<14.2f} {m['days']:>5} {m['cagr']:>+7.2f} {m['sharpe']:>+5.2f} "
          f"{m['dd']:>+6.2f} {m['total_ret']:>+7.2f}")

# ==========================================================================
# Experiment B: Fade-gate applied to xgb_K3_long_WEALTH40 (long-only K=3, no regime gate)
# ==========================================================================
print("\n" + "=" * 80)
print("B. FADE-GATE ON xgb_K3_long_WEALTH40 (K=3 long-only, no regime/meta)")
print("=" * 80)
print(f"{'fade_thresh':<14} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7} {'ret%':>8}")
print("-" * 80)
sweep_B = []
for thresh in [0.0, 0.35, 0.40, 0.45, 0.50, 0.55]:
    gate = "fade" if thresh > 0 else "none"
    m = run_sim(K_long=3, K_short=0, stop=0.10, regime_gate=False,
                meta_thresh=0, setup_gate=gate, setup_thresh=thresh)
    sweep_B.append({"fade_thresh": thresh, **m})
    print(f"{thresh:<14.2f} {m['days']:>5} {m['cagr']:>+7.2f} {m['sharpe']:>+5.2f} "
          f"{m['dd']:>+6.2f} {m['total_ret']:>+7.2f}")

# ==========================================================================
# Experiment C: Bounce + swing thresholds (for completeness)
# ==========================================================================
print("\n" + "=" * 80)
print("C. Quick check: bounce+swing at best fade threshold (0.45) on xsec K=10+10")
print("=" * 80)
sweep_C = []
for gate in ["none", "bounce", "swing"]:
    for thresh in [0.45]:
        m = run_sim(K_long=10, K_short=10, stop=0.10, regime_gate=True,
                    meta_thresh=0.45, setup_gate=gate if gate != "none" else "none",
                    setup_thresh=thresh)
        sweep_C.append({"gate": gate, "thresh": thresh, **m})
        print(f"{gate:<10} thresh={thresh:.2f}  CAGR {m['cagr']:>+7.2f}%  "
              f"Sh {m['sharpe']:>+5.2f}  DD {m['dd']:>+6.2f}%")

# Save
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "threshold_sweep_research.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_A_fade_sweep_xsec_K10_10": sweep_A,
        "experiment_B_fade_on_xgb_K3_long": sweep_B,
        "experiment_C_other_gates_xsec_K10_10": sweep_C,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
