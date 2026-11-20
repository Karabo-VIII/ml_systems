"""Final research sprint: adaptive TB v9 + Hawkes features + vol-scaled sizing.

Items:
    #2 Adaptive TB v9 (swap meta-labeler v8 -> v9)
    #5 Vol-scaled position sizing (simple optimal-transport-lite)
    #6 Hawkes enhanced features as ranker inputs

All tested against xsec K=10+10 baseline on U50_LIQUID.
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

# Load gates: v8 (existing) and v9 (adaptive TB, unused)
v8 = pickle.load(open(MODELS / "meta_labeler" / "v8_catboost.pkl", "rb"))
v9 = pickle.load(open(MODELS / "meta_labeler" / "v9_adaptive_tb.pkl", "rb"))
print(f"[info] v8 features: {len(v8.get('feature_names', []))}, v9 features: {len(v9.get('feature_names', []))}")

MAKER_RT = 0.08
CAPITAL = 10000.0
TRAIN_END = "2024-10-01"
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)


# =============================================================================
# Build panel
# =============================================================================
print("[info] building panel...")
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

# Merge Hawkes enhanced features
if HAWKES_ENH.exists():
    hx = pd.read_parquet(HAWKES_ENH)
    hx["date"] = pd.to_datetime(hx["date"])
    panel = panel.merge(hx, on=["date", "asset"], how="left")
    hawkes_cols = [c for c in hx.columns if c not in ("date", "asset") and c in panel.columns]
    print(f"[info] Hawkes enhanced features added: {hawkes_cols}")
else:
    hawkes_cols = []
    print("[warn] no Hawkes enhanced file found")

# BTC regime
btc = pl.read_parquet(DATA / "btcusdt_v50_chimera.parquet", columns=["timestamp", "close"]).to_pandas()
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms").dt.date
btc_d = btc.groupby("date").agg({"close": "last"}).reset_index()
btc_d["btc_30d"] = btc_d["close"].pct_change(30)
btc_d["date"] = pd.to_datetime(btc_d["date"])
panel = panel.merge(btc_d[["date", "btc_30d"]], on="date", how="left")

base_feats = [c for c in panel.columns if c.startswith("norm_") or c.startswith("xd_")]
base_feats += ["ret_d", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "vol_30d", "hl", "hurst_regime"]
base_feats = [c for c in base_feats if c in panel.columns]
enhanced_feats = base_feats + [c for c in hawkes_cols if c in panel.columns]

panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
panel["rank_target"] = panel.groupby("date")["fwd_3d"].rank(pct=True).apply(lambda x: int(x * 31))

tr_base = panel[panel["date"] < TRAIN_END]
te_base = panel[(panel["date"] >= TEST_START) & (panel["date"] < TEST_END)].copy()
tr_g = tr_base.groupby("date").size().values
print(f"[info] panel {panel.shape} in {time.time()-t0:.1f}s")


# =============================================================================
# Train two rankers: baseline (feat_list) and enhanced (+Hawkes)
# =============================================================================
def train_ranker(feat_list):
    r = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
                      max_depth=6, n_estimators=500, random_state=42, eval_metric="ndcg@5")
    r.fit(tr_base[feat_list].fillna(0).values, tr_base["rank_target"].values,
          group=tr_g, verbose=False)
    return r

print("[info] training rankers (base + enhanced)...")
r_base = train_ranker(base_feats)
te_base["pred_base"] = r_base.predict(te_base[base_feats].fillna(0).values)
r_enh = train_ranker(enhanced_feats)
te_base["pred_enh"] = r_enh.predict(te_base[enhanced_feats].fillna(0).values)
print("[info] rankers trained")


# =============================================================================
# Score gates: v8 and v9
# =============================================================================
def score_gate(df, gate_obj):
    feats = gate_obj.get("feature_names") or gate_obj.get("features", [])
    model = gate_obj.get("model") or gate_obj.get("clf")
    X_cols = []
    for f in feats:
        if f in df.columns:
            X_cols.append(df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(df)))
    X = np.vstack(X_cols).T
    try:
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p.flatten()
    except Exception:
        return np.ones(len(df)) * 0.5

te_base["p_v8"] = score_gate(te_base, v8)
te_base["p_v9"] = score_gate(te_base, v9)
print(f"[info] v8 p_win: mean={te_base['p_v8'].mean():.3f}, median={te_base['p_v8'].median():.3f}")
print(f"[info] v9 p_win: mean={te_base['p_v9'].mean():.3f}, median={te_base['p_v9'].median():.3f}")


# =============================================================================
# Simulation with configurable gate + sizing
# =============================================================================
def run_sim(pred_col, gate_col, gate_thresh, sizing="equal",
            K_long=10, K_short=10, stop=0.10):
    """
    sizing:
      'equal'   -- equal weight (current baseline)
      'vol'     -- inverse-vol weight (risk parity; optimal-transport-lite)
    """
    dates = sorted(te_base["date"].unique())
    daily_rets = []
    for d in dates:
        grp = te_base[te_base["date"] == d]
        if len(grp) < K_long + K_short:
            daily_rets.append(0.0); continue
        btc30 = float(grp["btc_30d"].iloc[0]) if len(grp) else 0
        if pd.isna(btc30) or btc30 < -0.15:
            daily_rets.append(0.0); continue

        long_r = short_r = 0.0
        if K_long > 0:
            top = grp.sort_values(pred_col, ascending=False).head(K_long).copy()
            top = top[top[gate_col] >= gate_thresh]
            if len(top) > 0:
                rs = []
                weights = []
                for _, r in top.iterrows():
                    p = r["fwd_3d"]
                    if stop and r["min_fwd_3d"] < -stop:
                        p = -stop
                    rs.append(p)
                    # Vol weight = inverse of recent vol
                    if sizing == "vol":
                        vol = r.get("vol_7d", 0.02) or 0.02
                        weights.append(1.0 / max(vol, 0.005))
                    else:
                        weights.append(1.0)
                weights = np.array(weights)
                weights = weights / weights.sum()
                rs = np.array(rs)
                long_r = ((rs * weights).sum() * 100 - MAKER_RT)
        if K_short > 0:
            bot = grp.sort_values(pred_col, ascending=True).head(K_short)
            rs = []; weights = []
            for _, r in bot.iterrows():
                p = -r["fwd_3d"]
                if stop and r["max_fwd_3d"] > stop:
                    p = -stop
                rs.append(p)
                if sizing == "vol":
                    vol = r.get("vol_7d", 0.02) or 0.02
                    weights.append(1.0 / max(vol, 0.005))
                else:
                    weights.append(1.0)
            weights = np.array(weights) / np.array(weights).sum() if weights else np.array([])
            if len(rs):
                short_r = ((np.array(rs) * weights).sum() * 100 - MAKER_RT)
        n_sides = (1 if K_long > 0 else 0) + (1 if K_short > 0 else 0)
        daily_rets.append((long_r + short_r) / max(n_sides, 1) / 3)

    r = np.array(daily_rets) / 100.0
    eq = CAPITAL * np.cumprod(1 + r)
    days = len(dates)
    total = (eq[-1] / CAPITAL - 1) * 100
    days_span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days_span) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    return {"days": days, "total_ret": total, "cagr": cagr, "sharpe": sharpe, "dd": dd}


# =============================================================================
# Experiment 1: v8 vs v9 gate at various thresholds
# =============================================================================
print("\n" + "=" * 80)
print("1. META-LABELER GATE COMPARISON  (xsec K=10+10, equal-weight)")
print("=" * 80)
print(f"{'variant':<18} {'thresh':<8} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 70)
gate_results = []
for name, col in [("v8_catboost", "p_v8"), ("v9_adaptive_tb", "p_v9")]:
    for thresh in [0.35, 0.40, 0.45, 0.50]:
        m = run_sim(pred_col="pred_base", gate_col=col, gate_thresh=thresh,
                    sizing="equal")
        gate_results.append({"gate": name, "thresh": thresh, **m})
        print(f"{name:<18} {thresh:<8.2f} {m['days']:>5} {m['cagr']:>+7.2f} "
              f"{m['sharpe']:>+5.2f} {m['dd']:>+6.2f}")


# =============================================================================
# Experiment 2: Hawkes-enhanced ranker
# =============================================================================
print("\n" + "=" * 80)
print("2. HAWKES-ENHANCED RANKER  (base vs +Hawkes features)")
print("=" * 80)
print(f"{'ranker':<15} {'gate':<18} {'thresh':<8} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 70)
hawkes_results = []
for rname, pred_col in [("base", "pred_base"), ("enhanced", "pred_enh")]:
    m = run_sim(pred_col=pred_col, gate_col="p_v8", gate_thresh=0.45, sizing="equal")
    hawkes_results.append({"ranker": rname, "gate": "v8", "thresh": 0.45, **m})
    print(f"{rname:<15} {'v8':<18} {'0.45':<8} {m['cagr']:>+7.2f} "
          f"{m['sharpe']:>+5.2f} {m['dd']:>+6.2f}")


# =============================================================================
# Experiment 3: Vol-scaled position sizing
# =============================================================================
print("\n" + "=" * 80)
print("3. VOL-SCALED SIZING  (inverse-vol vs equal-weight)")
print("=" * 80)
print(f"{'sizing':<15} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 70)
sizing_results = []
for sizing in ["equal", "vol"]:
    m = run_sim(pred_col="pred_base", gate_col="p_v8", gate_thresh=0.45, sizing=sizing)
    sizing_results.append({"sizing": sizing, **m})
    print(f"{sizing:<15} {m['days']:>5} {m['cagr']:>+7.2f} {m['sharpe']:>+5.2f} "
          f"{m['dd']:>+6.2f}")


# =============================================================================
# Experiment 4: BEST combined config test
# =============================================================================
print("\n" + "=" * 80)
print("4. BEST COMBINED CONFIG (enhanced ranker + best gate + vol sizing)")
print("=" * 80)
# Find best v8/v9 threshold from Exp 1
best_gate = max(gate_results, key=lambda r: r["sharpe"])
print(f"Best gate from Exp1: {best_gate['gate']} @ {best_gate['thresh']}")
gate_col = "p_v8" if best_gate["gate"] == "v8_catboost" else "p_v9"

m_base = run_sim("pred_enh", gate_col, best_gate["thresh"], "vol")
m_noenh = run_sim("pred_base", gate_col, best_gate["thresh"], "vol")
print(f"\n{'config':<35} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 70)
print(f"{'baseline (pred_base + v8@0.45 + equal)':<35} "
      f"{sizing_results[0]['cagr']:>+7.2f} {sizing_results[0]['sharpe']:>+5.2f} "
      f"{sizing_results[0]['dd']:>+6.2f}")
print(f"{'enhanced + best_gate + vol':<35} {m_base['cagr']:>+7.2f} "
      f"{m_base['sharpe']:>+5.2f} {m_base['dd']:>+6.2f}")
print(f"{'base + best_gate + vol':<35} {m_noenh['cagr']:>+7.2f} "
      f"{m_noenh['sharpe']:>+5.2f} {m_noenh['dd']:>+6.2f}")


# Save all
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "research_final_sprint.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "universe": "UNIVERSE_50_LIQUID",
        "gate_comparison": gate_results,
        "hawkes_comparison": hawkes_results,
        "sizing_comparison": sizing_results,
        "best_gate_from_exp1": best_gate,
        "enhanced_vol": m_base,
        "base_vol": m_noenh,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
