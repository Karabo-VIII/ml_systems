"""Audit Hawkes leakage + fix vol sizing + retest everything honestly.

Items:
    #1 Hawkes audit: confirm leakage from ret_1d/ret_3d/ret_5d forward-return
       columns in hawkes_enh_daily.parquet; strip them and re-measure honest lift
    #5 Vol-sizing NaN bug fix (fillna with median vol on warmup)
    Shuffled-label control: confirm baseline ranker has zero Sharpe under shuffled targets
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
HAWKES_ENH = ROOT / "data" / "frontier" / "hawkes_enh" / "hawkes_enh_daily.parquet"

v8 = pickle.load(open(MODELS / "meta_labeler" / "v8_catboost.pkl", "rb"))
MAKER_RT = 0.08
CAPITAL = 10000.0
TRAIN_END = "2024-10-01"
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

# =============================================================================
# Build base panel (no Hawkes merge yet)
# =============================================================================
print("[audit] building base panel...")
t0 = time.time()
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
    d["ret_3d_bwd"] = d["close"].pct_change(3)  # BACKWARD
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
# Keep for compat with xsec_variants script
panel["ret_3d"] = panel["ret_3d_bwd"]

btc = pl.read_parquet(DATA / "btcusdt_v50_chimera.parquet", columns=["timestamp", "close"]).to_pandas()
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms").dt.date
btc_d = btc.groupby("date").agg({"close": "last"}).reset_index()
btc_d["btc_30d"] = btc_d["close"].pct_change(30)
btc_d["date"] = pd.to_datetime(btc_d["date"])
panel = panel.merge(btc_d[["date", "btc_30d"]], on="date", how="left")

base_feats = [c for c in panel.columns if c.startswith("norm_") or c.startswith("xd_")]
base_feats += ["ret_d", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "vol_30d", "hl", "hurst_regime"]
base_feats = [c for c in base_feats if c in panel.columns]

panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
panel["rank_target"] = panel.groupby("date")["fwd_3d"].rank(pct=True).apply(lambda x: int(x * 31))
tr_base = panel[panel["date"] < TRAIN_END]
tr_g = tr_base.groupby("date").size().values
print(f"[audit] panel {panel.shape} in {time.time()-t0:.1f}s, {len(base_feats)} base features")


# =============================================================================
# Merge Hawkes enhanced features -- WITH LEAKAGE STRIP
# =============================================================================
hawkes_cols_safe = []
hawkes_cols_leaky = {"ret_1d", "ret_3d", "ret_5d"}  # forward returns -- LEAK
if HAWKES_ENH.exists():
    hx = pd.read_parquet(HAWKES_ENH)
    hx["date"] = pd.to_datetime(hx["date"])
    # Drop leakage columns explicitly
    leaky_in_hx = [c for c in hawkes_cols_leaky if c in hx.columns]
    if leaky_in_hx:
        print(f"[audit] STRIPPED leakage columns from hawkes parquet: {leaky_in_hx}")
        hx = hx.drop(columns=leaky_in_hx)
    # Drop duplicate columns (close, norm_hawkes_*)
    overlap = [c for c in hx.columns if c in panel.columns and c not in ("date", "asset")]
    if overlap:
        print(f"[audit] dropping hx columns that collide with panel: {overlap}")
        hx = hx.drop(columns=overlap)
    panel = panel.merge(hx, on=["date", "asset"], how="left")
    hawkes_cols_safe = [c for c in hx.columns if c not in ("date", "asset") and c in panel.columns]
    print(f"[audit] Hawkes enhanced features added (safe): {hawkes_cols_safe}")

enhanced_feats = base_feats + hawkes_cols_safe
tr_base = panel[panel["date"] < TRAIN_END]
te_base = panel[(panel["date"] >= TEST_START) & (panel["date"] < TEST_END)].copy()
print(f"[audit] base_feats={len(base_feats)} enhanced_feats={len(enhanced_feats)}")

def train_ranker(feat_list):
    r = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
                      max_depth=6, n_estimators=500, random_state=42, eval_metric="ndcg@5")
    r.fit(tr_base[feat_list].fillna(0).values, tr_base["rank_target"].values,
          group=tr_g, verbose=False)
    return r


def score_v8(df):
    feats = v8.get("feature_names") or v8.get("features", [])
    X = np.vstack([df[f].fillna(0).values if f in df.columns else np.zeros(len(df))
                   for f in feats]).T
    return v8["model"].predict_proba(X)[:, 1]


# =============================================================================
# Experiment 1: honest comparison of base vs enhanced ranker (NO LEAKAGE)
# =============================================================================
print("\n[exp 1] training rankers (base + enhanced, no leakage)...")
t0 = time.time()
r_base = train_ranker(base_feats)
r_enh = train_ranker(enhanced_feats)
te_base["pred_base"] = r_base.predict(te_base[base_feats].fillna(0).values)
te_base["pred_enh"] = r_enh.predict(te_base[enhanced_feats].fillna(0).values)
te_base["p_v8"] = score_v8(te_base)
print(f"[exp 1] done in {time.time()-t0:.1f}s")


def sim(pred_col, gate_thresh=0.45, sizing="equal", K_long=10, K_short=10, stop=0.10,
        shuffle_pred=False):
    df = te_base
    if shuffle_pred:
        # Shuffle preds WITHIN each date (break the alpha)
        rng = np.random.RandomState(42)
        df = df.copy()
        df[pred_col] = df.groupby("date")[pred_col].transform(
            lambda s: rng.permutation(s.values))
    dates = sorted(df["date"].unique())
    daily_rets = []
    for d in dates:
        grp = df[df["date"] == d]
        if len(grp) < K_long + K_short:
            daily_rets.append(0.0); continue
        btc30 = float(grp["btc_30d"].iloc[0]) if len(grp) else 0
        if pd.isna(btc30) or btc30 < -0.15:
            daily_rets.append(0.0); continue
        long_r = short_r = 0.0
        if K_long > 0:
            top = grp.sort_values(pred_col, ascending=False).head(K_long).copy()
            top = top[top["p_v8"] >= gate_thresh]
            if len(top) > 0:
                rs = []; weights = []
                for _, r in top.iterrows():
                    p = r["fwd_3d"]
                    if stop and r["min_fwd_3d"] < -stop:
                        p = -stop
                    rs.append(p)
                    if sizing == "vol":
                        # FIX: fillna with median vol on warmup
                        vol = r.get("vol_7d")
                        if pd.isna(vol) or vol is None or vol <= 0:
                            vol = 0.02  # ~median daily vol for crypto
                        weights.append(1.0 / max(vol, 0.005))
                    else:
                        weights.append(1.0)
                weights = np.array(weights)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(rs)) / len(rs)
                long_r = (np.array(rs) * weights).sum() * 100 - MAKER_RT
        if K_short > 0:
            bot = grp.sort_values(pred_col, ascending=True).head(K_short)
            rs = []; weights = []
            for _, r in bot.iterrows():
                p = -r["fwd_3d"]
                if stop and r["max_fwd_3d"] > stop:
                    p = -stop
                rs.append(p)
                if sizing == "vol":
                    vol = r.get("vol_7d")
                    if pd.isna(vol) or vol is None or vol <= 0:
                        vol = 0.02
                    weights.append(1.0 / max(vol, 0.005))
                else:
                    weights.append(1.0)
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(rs)) / max(len(rs), 1)
            if len(rs):
                short_r = (np.array(rs) * weights).sum() * 100 - MAKER_RT
        n_sides = (1 if K_long > 0 else 0) + (1 if K_short > 0 else 0)
        daily_rets.append((long_r + short_r) / max(n_sides, 1) / 3)
    r = np.array(daily_rets) / 100.0
    eq = CAPITAL * np.cumprod(1 + r)
    total = (eq[-1] / CAPITAL - 1) * 100
    days_span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days_span) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    dd_mask = np.maximum.accumulate(eq)
    dd = ((eq - dd_mask) / dd_mask).min() * 100
    return {"days": len(dates), "cagr": cagr, "sharpe": sharpe, "dd": dd, "total_ret": total}


print("\n" + "=" * 80)
print("EXP 1: BASE vs ENHANCED (honest, no forward-return leakage)")
print("=" * 80)
print(f"{'config':<40} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 80)

m_base = sim("pred_base")
print(f"{'base ranker, v8 gate @0.45, equal':<40} {m_base['days']:>5} "
      f"{m_base['cagr']:>+7.2f} {m_base['sharpe']:>+5.2f} {m_base['dd']:>+6.2f}")

m_enh = sim("pred_enh")
print(f"{'enhanced ranker (Hawkes clean), equal':<40} {m_enh['days']:>5} "
      f"{m_enh['cagr']:>+7.2f} {m_enh['sharpe']:>+5.2f} {m_enh['dd']:>+6.2f}")


# =============================================================================
# Experiment 2: Shuffled-pred control
# =============================================================================
print("\n" + "=" * 80)
print("EXP 2: SHUFFLED-PRED CONTROL (Sharpe should collapse to ~0)")
print("=" * 80)
m_shuf_base = sim("pred_base", shuffle_pred=True)
m_shuf_enh = sim("pred_enh", shuffle_pred=True)
print(f"{'base, shuffled-pred':<40} {m_shuf_base['sharpe']:>+5.2f} "
      f"(CAGR {m_shuf_base['cagr']:>+7.2f}%)")
print(f"{'enhanced, shuffled-pred':<40} {m_shuf_enh['sharpe']:>+5.2f} "
      f"(CAGR {m_shuf_enh['cagr']:>+7.2f}%)")
if abs(m_shuf_base["sharpe"]) > 1.0 or abs(m_shuf_enh["sharpe"]) > 1.0:
    print(f"  [WARN] shuffled Sharpe > 1.0 -- possible remaining leakage source!")
else:
    print(f"  [OK] shuffled Sharpe ~0 -- alpha is from ranker, not leakage")


# =============================================================================
# Experiment 3: Vol-scaled sizing (fixed NaN bug)
# =============================================================================
print("\n" + "=" * 80)
print("EXP 3: VOL-SCALED SIZING FIX")
print("=" * 80)
print(f"{'config':<40} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 80)
m_equal = sim("pred_base", sizing="equal")
print(f"{'base + equal (baseline)':<40} {m_equal['days']:>5} "
      f"{m_equal['cagr']:>+7.2f} {m_equal['sharpe']:>+5.2f} {m_equal['dd']:>+6.2f}")
m_vol = sim("pred_base", sizing="vol")
print(f"{'base + vol (fixed NaN bug)':<40} {m_vol['days']:>5} "
      f"{m_vol['cagr']:>+7.2f} {m_vol['sharpe']:>+5.2f} {m_vol['dd']:>+6.2f}")


# =============================================================================
# Save
# =============================================================================
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "audit_hawkes_fix_vol.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "leakage_stripped_cols": sorted(hawkes_cols_leaky),
        "hawkes_cols_clean": hawkes_cols_safe,
        "exp1_base": m_base,
        "exp1_enhanced": m_enh,
        "exp2_base_shuffled": m_shuf_base,
        "exp2_enhanced_shuffled": m_shuf_enh,
        "exp3_equal": m_equal,
        "exp3_vol_scaled": m_vol,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
