"""Wire setup classifiers (bounce/fade/swing) into xsec ranker gate.

The setup classifiers are trained CatBoost binaries stored in:
    models/bounce_ml/bounce_ml_v1.pkl
    models/fade_ml/fade_ml_v1.pkl
    models/swing_ml/swing_ml_v1.pkl

Each takes 45 features (all norm_*, xd_*, ret_d/3d/7d, hl, hurst_regime) and
outputs p(setup_valid).

Strategy: extend the xsec K=10+10 delta-neutral variant (best Sharpe from the
K-sweep) with an additional setup-classifier gate. Gate variants:
    none          -- baseline (just meta_gate v8)
    any           -- require ANY of {bounce, fade, swing} p > threshold
    all           -- require ALL p > threshold (strict)
    bounce_only   -- bounce gate only
    fade_only     -- fade gate only
    swing_only    -- swing gate only

Also runs a time-slice validation:
    2025-01-01 -> 2025-07-01  (6mo)
    2025-07-01 -> 2026-01-01  (6mo)
    2026-01-01 -> 2026-04-22  (4mo)

Each slice's Sharpe reported for each variant, to check robustness.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import glob, numpy as np, pandas as pd, polars as pl
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Setup classifiers
SETUP_CLFS = {}
for name in ["bounce_ml", "fade_ml", "swing_ml"]:
    with open(MODELS / name / f"{name}_v1.pkl", "rb") as f:
        SETUP_CLFS[name.replace("_ml", "")] = pickle.load(f)

# CatBoost v8 meta-labeler (existing)
META_PKL = MODELS / "meta_labeler" / "v8_catboost.pkl"
meta_obj = pickle.load(open(META_PKL, "rb")) if META_PKL.exists() else None
meta_model = meta_obj["model"] if isinstance(meta_obj, dict) and "model" in meta_obj else None
meta_features = meta_obj.get("feature_names", []) if isinstance(meta_obj, dict) else []

MAKER_RT = 0.08
TRAIN_END = "2024-10-01"
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"
CAPITAL = 10000.0

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(f"[info] loaded setup classifiers: {list(SETUP_CLFS.keys())}")
print(f"[info] loaded meta-labeler: {meta_model is not None}")

# Load U50 universe
sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)
print(f"[info] universe = UNIVERSE_50_LIQUID ({len(UNIVERSE)} assets)")

# Build panel (shamelessly adapted from xsec_variants_daily_equity.py)
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

# BTC regime
btc = pl.read_parquet(DATA / "btcusdt_v50_chimera.parquet", columns=["timestamp", "close"]).to_pandas()
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms").dt.date
btc_d = btc.groupby("date").agg({"close": "last"}).reset_index()
btc_d["btc_30d"] = btc_d["close"].pct_change(30)
btc_d["btc_ret_7d"] = btc_d["close"].pct_change(7)
btc_d["date"] = pd.to_datetime(btc_d["date"])
panel = panel.merge(btc_d[["date", "btc_30d", "btc_ret_7d"]], on="date", how="left")
panel["ret_7d"] = panel["ret_7d"]  # already computed
panel["btc_regime"] = np.sign(panel["btc_ret_7d"].fillna(0))

feat_list = [c for c in panel.columns if c.startswith("norm_") or c.startswith("xd_")]
feat_list += ["ret_d", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "vol_30d", "hl", "hurst_regime"]
feat_list = [c for c in feat_list if c in panel.columns]

panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
panel["rank_target"] = panel.groupby("date")["fwd_3d"].rank(pct=True).apply(lambda x: int(x * 31))

tr = panel[panel["date"] < TRAIN_END]
te = panel[(panel["date"] >= TEST_START) & (panel["date"] < TEST_END)].copy()
tr_g = tr.groupby("date").size().values

print(f"[info] panel: {panel.shape}, train {len(tr)} rows, test {len(te)} rows")
print("[info] training XGBRanker...")
t0 = time.time()
ranker = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
                       max_depth=6, n_estimators=500, random_state=42, eval_metric="ndcg@5")
ranker.fit(tr[feat_list].fillna(0).values, tr["rank_target"].values, group=tr_g, verbose=False)
te["pred"] = ranker.predict(te[feat_list].fillna(0).values)
print(f"[info] trained in {time.time() - t0:.1f}s")


# Meta-labeler p_win scorer
def score_meta(df: pd.DataFrame) -> np.ndarray:
    if meta_model is None:
        return np.ones(len(df)) * 0.5
    X_cols = []
    for f in meta_features:
        if f in df.columns:
            X_cols.append(df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(df)))
    X = np.vstack(X_cols).T if X_cols else np.zeros((len(df), 1))
    try:
        p = meta_model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.flatten()
    except Exception:
        return np.ones(len(df)) * 0.5


# Setup classifiers scorer -- produce p per (bounce, fade, swing)
def score_setups(df: pd.DataFrame) -> dict:
    """Return dict of name -> p_win array aligned to df."""
    out = {}
    for name, obj in SETUP_CLFS.items():
        clf = obj["clf"]
        feats = obj["features"]
        X_cols = []
        for f in feats:
            if f in df.columns:
                X_cols.append(df[f].fillna(0).values)
            else:
                X_cols.append(np.zeros(len(df)))
        X = np.vstack(X_cols).T
        try:
            p = clf.predict_proba(X)
            out[name] = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.flatten()
        except Exception as e:
            print(f"[warn] {name} scorer failed: {e}; using 0.5")
            out[name] = np.ones(len(df)) * 0.5
    return out


print("[info] scoring meta-labeler + setup classifiers...")
t0 = time.time()
te["p_win"] = score_meta(te)
setup_ps = score_setups(te)
for name, p in setup_ps.items():
    te[f"p_{name}"] = p
print(f"[info] done in {time.time() - t0:.1f}s")
print(f"[info] p_win stats: mean={te['p_win'].mean():.3f}, median={te['p_win'].median():.3f}")
for name in setup_ps:
    print(f"[info] p_{name} stats: mean={te[f'p_{name}'].mean():.3f}, median={te[f'p_{name}'].median():.3f}")


def simulate_xsec_gated(te_in: pd.DataFrame, K_long: int, K_short: int,
                        stop: float = 0.10,
                        meta_thresh: float = 0.45,
                        setup_gate: str = "none",
                        setup_thresh: float = 0.50,
                        date_from: str | None = None,
                        date_to: str | None = None,
                        variant_name: str = ""):
    """Run the xsec sim with stackable meta + setup gates."""
    sub = te_in
    if date_from:
        sub = sub[sub["date"] >= date_from]
    if date_to:
        sub = sub[sub["date"] < date_to]
    dates = sorted(sub["date"].unique())
    if len(dates) < 10:
        return None
    daily_rets = []
    for d in dates:
        grp = sub[sub["date"] == d]
        if len(grp) < K_long + K_short:
            daily_rets.append(0.0)
            continue
        btc30 = float(grp["btc_30d"].iloc[0]) if len(grp) else 0.0
        if pd.isna(btc30) or btc30 < -0.15:
            daily_rets.append(0.0)
            continue
        long_r = short_r = 0.0
        if K_long > 0:
            top = grp.sort_values("pred", ascending=False).head(K_long).copy()
            # Apply meta-gate
            if meta_thresh > 0:
                top = top[top["p_win"] >= meta_thresh]
            # Apply setup-gate
            if setup_gate == "any":
                mask = ((top["p_bounce"] >= setup_thresh) |
                        (top["p_fade"] >= setup_thresh) |
                        (top["p_swing"] >= setup_thresh))
                top = top[mask]
            elif setup_gate == "all":
                mask = ((top["p_bounce"] >= setup_thresh) &
                        (top["p_fade"] >= setup_thresh) &
                        (top["p_swing"] >= setup_thresh))
                top = top[mask]
            elif setup_gate in ("bounce", "fade", "swing"):
                top = top[top[f"p_{setup_gate}"] >= setup_thresh]
            # Commit trades
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
    n = len(dates)
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    cum = np.maximum.accumulate(eq)
    dd = ((eq - cum) / cum).min() * 100
    return {"variant": variant_name, "n_days": n, "total_ret_pct": total,
            "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd}


# =============================================================================
# Experiment 1: Setup-gate variants on K=10+10 full window
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Setup-gate variants at K=10+10 (full 2025-01 to 2026-04)")
print("=" * 80)
print(f"{'variant':<28} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7} {'ret%':>8}")
print("-" * 80)
gate_results = []
for setup_gate in ["none", "any", "all", "bounce", "fade", "swing"]:
    for thresh in [0.50] if setup_gate != "none" else [0.0]:
        r = simulate_xsec_gated(te, K_long=10, K_short=10,
                                setup_gate=setup_gate, setup_thresh=thresh,
                                variant_name=f"K10_10_setup_{setup_gate}")
        if r is not None:
            gate_results.append(r)
            print(f"{r['variant']:<28} {r['n_days']:>5} {r['cagr_pct']:>+7.2f} "
                  f"{r['sharpe']:>+5.2f} {r['max_dd_pct']:>+6.2f} {r['total_ret_pct']:>+7.2f}")

# =============================================================================
# Experiment 2: K=10+10 time-slice robustness
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: K=10+10 time-slice validation (3 non-overlap windows)")
print("=" * 80)
slices = [
    ("2025-01-01 -> 2025-07-01", "2025-01-01", "2025-07-01"),
    ("2025-07-01 -> 2026-01-01", "2025-07-01", "2026-01-01"),
    ("2026-01-01 -> 2026-04-22", "2026-01-01", "2026-04-22"),
]
print(f"{'slice':<30} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 80)
slice_results = []
for label, d0, d1 in slices:
    r = simulate_xsec_gated(te, K_long=10, K_short=10, setup_gate="none",
                            date_from=d0, date_to=d1, variant_name=label)
    if r is not None:
        slice_results.append(r)
        print(f"{label:<30} {r['n_days']:>5} {r['cagr_pct']:>+7.2f} "
              f"{r['sharpe']:>+5.2f} {r['max_dd_pct']:>+6.2f}")

# Save
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "xsec_setup_gate_results.json"
with open(out_file, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "universe": "UNIVERSE_50_LIQUID",
        "K_long": 10, "K_short": 10,
        "gate_results": gate_results,
        "slice_results": slice_results,
    }, f, indent=2, default=str)
print(f"\n[saved] {out_file}")
