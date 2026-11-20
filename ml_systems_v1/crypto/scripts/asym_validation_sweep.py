"""Three validation experiments bundled together:

    (A) Family E -- oversold bounce prototype
        Entry: ret_3d < -0.15 AND ret_1d > +0.02
        Stop: recent 3d low; Target: 50% of 3d drawdown recovery
    (B) No-ranker baseline -- random-pick within meta-gate + regime filter
        Replaces XGBRanker picks with random subset. Confirms whether ranker
        adds alpha (memory/shuffle-control finding).
    (C) Shuffled-fade-gate control -- shuffle p_fade within each date
        on the xgb_K3_long_WEALTH40 sleeve. If shuffled gives similar CAGR
        uplift, the fade@0.50 Pareto win was noise.
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

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

MAKER_RT = 0.08
CAPITAL = 10000.0
TRAIN_END = "2024-10-01"
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"

v8 = pickle.load(open(MODELS / "meta_labeler" / "v8_catboost.pkl", "rb"))
fade = pickle.load(open(MODELS / "fade_ml" / "fade_ml_v1.pkl", "rb"))


# =============================================================================
# Panel build
# =============================================================================
print("[panel] building...")
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
    d["fwd_5d"] = d["close"].shift(-5) / d["close"] - 1
    d["min_fwd_5d"] = d[["fwd_1d", "fwd_2d", "fwd_3d", d.columns[-1] if "fwd_5d" in d.columns else "fwd_3d"]].min(axis=1)
    d["min_fwd_3d"] = np.minimum.reduce([d["fwd_1d"], d["fwd_2d"], d["fwd_3d"]])
    d["max_fwd_3d"] = np.maximum.reduce([d["fwd_1d"], d["fwd_2d"], d["fwd_3d"]])
    d["low_3d"] = d["low"].rolling(3, min_periods=1).min()
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

print(f"[panel] {panel.shape} in {time.time()-t0:.1f}s")


# =============================================================================
# (A) FAMILY E: oversold bounce
# =============================================================================
print("\n" + "=" * 80)
print("(A) FAMILY E: OVERSOLD BOUNCE")
print("=" * 80)

def sim_oversold_bounce(threshold_drop=-0.15, reversal_min=0.02, target_recov=0.50,
                        time_stop_days=5, max_concurrent=10, pct_per_trade=0.10):
    """Entry: ret_3d <= threshold_drop AND ret_d >= reversal_min.
    Stop: at 3d low (on entry day).
    Target: entry + target_recov * (entry - 3d_low).
    Time stop: time_stop_days.
    """
    test = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)].copy()
    test = test.sort_values(["asset", "date"]).reset_index(drop=True)

    all_dates = sorted(test["date"].unique())
    lookups = {d: grp for d, grp in test.groupby("date")}

    cash = CAPITAL
    positions = {}
    trades = []
    daily_eq = []

    for d in all_dates:
        lkup = lookups.get(d)
        if lkup is None: continue
        # MTM + check exits
        close_map = dict(zip(lkup["asset"], lkup["close"]))
        low_map = dict(zip(lkup["asset"], lkup["low"]))
        high_map = dict(zip(lkup["asset"], lkup["high"]))
        closed = []
        for asset, pos in list(positions.items()):
            if asset not in close_map:
                continue
            low = low_map[asset]; high = high_map[asset]; close = close_map[asset]
            exit_price = None; reason = None
            if low <= pos["stop"]:
                exit_price = pos["stop"]; reason = "stop"
            elif high >= pos["target"]:
                exit_price = pos["target"]; reason = "target"
            elif (d - pos["entry_date"]).days >= time_stop_days:
                exit_price = close; reason = "time"
            if exit_price is not None:
                size = pos["size"]
                pnl = size * (exit_price / pos["entry_price"] - 1)
                pnl -= size * (MAKER_RT / 200.0)
                cash += size + pnl
                net = (exit_price / pos["entry_price"] - 1) * 100 - MAKER_RT
                trades.append({"asset": asset, "entry_date": pos["entry_date"], "exit_date": d,
                                "net_ret_pct": net, "exit_reason": reason})
                closed.append(asset)
        for a in closed: del positions[a]

        # New entries: oversold with reversal
        if len(positions) < max_concurrent:
            cands = lkup[(lkup["ret_3d"] <= threshold_drop) & (lkup["ret_d"] >= reversal_min)]
            for _, row in cands.iterrows():
                asset = row["asset"]
                if asset in positions: continue
                entry = row["close"]
                low3d = row["low_3d"]
                drop = entry - low3d
                if drop <= 0: continue
                stop = low3d * 0.995  # slightly below 3d low
                target = entry + target_recov * drop
                size = cash * pct_per_trade
                if size > cash: break
                cash -= size + size * (MAKER_RT / 200.0)
                positions[asset] = {"entry_date": d, "entry_price": entry,
                                     "stop": stop, "target": target, "size": size}
                if len(positions) >= max_concurrent: break

        pos_val = sum(p["size"] * (close_map.get(a, p["entry_price"]) / p["entry_price"])
                      for a, p in positions.items() if a in close_map)
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_pos": len(positions)})

    eq_df = pd.DataFrame(daily_eq)
    tr_df = pd.DataFrame(trades)
    if len(eq_df) < 2 or len(tr_df) == 0:
        return {"status": "insufficient", "n_trades": len(tr_df)}
    eq = eq_df["equity"].values
    dr = np.diff(eq) / eq[:-1]
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    cm = np.maximum.accumulate(eq)
    dd = ((eq - cm) / cm).min() * 100
    r = tr_df["net_ret_pct"].values
    wins = r[r > 0]; losses = r[r <= 0]
    hit = len(wins) / len(r)
    asym = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else float("inf")
    kelly_g = (hit * np.log1p(wins.mean()/100 if len(wins) else 0)
               + (1 - hit) * np.log1p(losses.mean()/100 if len(losses) else 0))
    return {
        "n_days": len(eq_df), "n_trades": len(r),
        "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd, "total_ret_pct": total,
        "hit_rate": hit, "mean_win_pct": wins.mean() if len(wins) else 0,
        "mean_loss_pct": losses.mean() if len(losses) else 0,
        "asymmetry_ratio": asym, "kelly_log_g_per_trade": kelly_g,
    }


print(f"{'config':<30} {'days':>4} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} "
      f"{'n_tr':>4} {'hit%':>4} {'asym':>5} {'kelly':>6}")
print("-" * 80)
oversold_results = []
for (thr, rev, tgt, ts, label) in [
    (-0.15, 0.02, 0.50, 5, "oversold_-15_+2_tgt50_ts5"),
    (-0.10, 0.02, 0.40, 5, "oversold_-10_+2_tgt40_ts5"),
    (-0.15, 0.01, 0.50, 3, "oversold_-15_+1_tgt50_ts3"),
    (-0.20, 0.03, 0.60, 7, "oversold_-20_+3_tgt60_ts7"),
]:
    s = sim_oversold_bounce(thr, rev, tgt, ts)
    s["label"] = label
    oversold_results.append(s)
    if s.get("status") == "insufficient":
        print(f"{label:<30} (insufficient, n_trades={s['n_trades']})")
        continue
    print(f"{label:<30} {s['n_days']:>4} {s['cagr_pct']:>+6.2f} {s['sharpe']:>+4.2f} "
          f"{s['max_dd_pct']:>+5.2f} {s['n_trades']:>4} {s['hit_rate']*100:>3.0f} "
          f"{s['asymmetry_ratio']:>4.2f} {s['kelly_log_g_per_trade']:>+5.4f}")


# =============================================================================
# (B) NO-RANKER BASELINE: random pick within meta-gate
# =============================================================================
print("\n" + "=" * 80)
print("(B) NO-RANKER BASELINE (random pick within meta-gate + regime)")
print("=" * 80)

# Score meta v8
def score_v8(df):
    feats = v8.get("feature_names") or v8.get("features", [])
    X = np.vstack([df[f].fillna(0).values if f in df.columns else np.zeros(len(df))
                   for f in feats]).T
    return v8["model"].predict_proba(X)[:, 1]

te = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)].copy()
te["p_v8"] = score_v8(te)
print(f"[info] test panel {te.shape}, p_v8 mean={te['p_v8'].mean():.3f}")


def sim_random_pick(K_long=10, K_short=10, meta_thresh=0.45, stop=0.10, seed=42):
    """Random K_long + K_short pick from meta-approved, delta-neutral basket."""
    rng = np.random.RandomState(seed)
    dates = sorted(te["date"].unique())
    daily_rets = []
    for d in dates:
        grp = te[te["date"] == d]
        if len(grp) < K_long + K_short:
            daily_rets.append(0.0); continue
        btc30 = float(grp["btc_30d"].iloc[0]) if len(grp) else 0
        if pd.isna(btc30) or btc30 < -0.15:
            daily_rets.append(0.0); continue
        approved = grp[grp["p_v8"] >= meta_thresh]
        if len(approved) < (K_long + K_short):
            daily_rets.append(0.0); continue
        # Random permutation
        idx = rng.permutation(len(approved))
        long_idx = idx[:K_long]
        short_idx = idx[K_long:K_long + K_short]
        top = approved.iloc[long_idx]
        bot = approved.iloc[short_idx]
        long_rs = [(-stop if r["min_fwd_3d"] < -stop else r["fwd_3d"]) for _, r in top.iterrows()]
        short_rs = [(-stop if r["max_fwd_3d"] > stop else -r["fwd_3d"]) for _, r in bot.iterrows()]
        long_r = (np.mean(long_rs) * 100 - MAKER_RT) if long_rs else 0
        short_r = (np.mean(short_rs) * 100 - MAKER_RT) if short_rs else 0
        daily_rets.append((long_r + short_r) / 2 / 3)

    r = np.array(daily_rets) / 100.0
    eq = CAPITAL * np.cumprod(1 + r)
    total = (eq[-1] / CAPITAL - 1) * 100
    days_span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days_span) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    return {"n_days": len(dates), "cagr_pct": cagr, "sharpe": sharpe,
            "max_dd_pct": dd, "total_ret_pct": total}

print(f"{'trial':<20} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 60)
noranker_results = []
for seed in [42, 1, 7, 100, 2026]:
    s = sim_random_pick(seed=seed)
    s["seed"] = seed
    noranker_results.append(s)
    print(f"random_seed_{seed:<14} {s['cagr_pct']:>+7.2f} {s['sharpe']:>+5.2f} {s['max_dd_pct']:>+6.2f}")

mean_sh = np.mean([r["sharpe"] for r in noranker_results])
mean_cagr = np.mean([r["cagr_pct"] for r in noranker_results])
print(f"\n  MEAN across 5 seeds: Sharpe {mean_sh:+.2f}, CAGR {mean_cagr:+.2f}%")
print(f"  Baseline xsec-K10+10 ranker: Sharpe ~4.04, CAGR ~105%")
if abs(mean_sh - 4.04) < 1.0:
    print("  [CONFIRMED] random-pick Sharpe ~= ranker Sharpe. XGBRanker adds no alpha.")
elif mean_sh > 4.04:
    print("  [STRONG] random-pick BEATS ranker -- ranker is net-negative.")
else:
    print("  [REFUTED] random-pick < ranker -- XGBRanker adds real alpha.")


# =============================================================================
# (C) SHUFFLED-FADE-GATE CONTROL
# =============================================================================
print("\n" + "=" * 80)
print("(C) SHUFFLED FADE-GATE CONTROL on xgb_K3_long_WEALTH40")
print("=" * 80)

# Need ranker for xgb_K3_long variant
base_feats = [c for c in panel.columns if c.startswith("norm_") or c.startswith("xd_")]
base_feats += ["ret_d", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "vol_30d", "hl", "hurst_regime"]
base_feats = [c for c in base_feats if c in panel.columns]
panel["rank_target"] = panel.groupby("date")["fwd_3d"].rank(pct=True).apply(lambda x: int(x * 31))
tr_base = panel[panel["date"] < TRAIN_END]
tr_g = tr_base.groupby("date").size().values
print(f"[ranker] training XGBRanker...")
t0 = time.time()
ranker = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
                       max_depth=6, n_estimators=500, random_state=42, eval_metric="ndcg@5")
ranker.fit(tr_base[base_feats].fillna(0).values, tr_base["rank_target"].values,
           group=tr_g, verbose=False)
te["pred"] = ranker.predict(te[base_feats].fillna(0).values)

# Score fade p_win
fclf = fade["clf"]; ffeats = fade["features"]
X_fade = np.vstack([te[f].fillna(0).values if f in te.columns else np.zeros(len(te))
                     for f in ffeats]).T
te["p_fade"] = fclf.predict_proba(X_fade)[:, 1]
print(f"[info] trained+scored in {time.time()-t0:.1f}s; p_fade mean={te['p_fade'].mean():.3f}")


def sim_xgb_K3_long(gate="none", gate_thresh=0.50, shuffle_p_fade=False, seed=42):
    """Long K=3, no regime gate, no meta. Optional fade gate (maybe shuffled)."""
    df = te.copy()
    if shuffle_p_fade:
        rng = np.random.RandomState(seed)
        df["p_fade"] = df.groupby("date")["p_fade"].transform(
            lambda s: rng.permutation(s.values))
    dates = sorted(df["date"].unique())
    daily_rets = []
    for d in dates:
        grp = df[df["date"] == d]
        if len(grp) < 3:
            daily_rets.append(0.0); continue
        top = grp.sort_values("pred", ascending=False).head(3).copy()
        if gate == "fade":
            top = top[top["p_fade"] >= gate_thresh]
        rs = [(-0.10 if r["min_fwd_3d"] < -0.10 else r["fwd_3d"]) for _, r in top.iterrows()]
        if rs:
            daily_rets.append((np.mean(rs) * 100 - MAKER_RT) / 3 / 100)
        else:
            daily_rets.append(0.0)
    r = np.array(daily_rets)
    eq = CAPITAL * np.cumprod(1 + r)
    total = (eq[-1] / CAPITAL - 1) * 100
    days_span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days_span) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    return {"cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd, "total_ret_pct": total}


print(f"{'variant':<40} {'CAGR%':>8} {'Sh':>6} {'DD%':>7}")
print("-" * 70)
c_results = []
m_base = sim_xgb_K3_long(gate="none")
print(f"{'xgb_K3_long baseline (no gate)':<40} {m_base['cagr_pct']:>+7.2f} "
      f"{m_base['sharpe']:>+5.2f} {m_base['max_dd_pct']:>+6.2f}")
c_results.append({"variant": "baseline_no_gate", **m_base})

m_fade = sim_xgb_K3_long(gate="fade", gate_thresh=0.50)
print(f"{'xgb_K3_long + fade@0.50 (real p_fade)':<40} {m_fade['cagr_pct']:>+7.2f} "
      f"{m_fade['sharpe']:>+5.2f} {m_fade['max_dd_pct']:>+6.2f}")
c_results.append({"variant": "fade_real", **m_fade})

# 5 shuffled seeds
for seed in [42, 1, 7, 100, 2026]:
    m_shuf = sim_xgb_K3_long(gate="fade", gate_thresh=0.50,
                              shuffle_p_fade=True, seed=seed)
    m_shuf["seed"] = seed
    c_results.append({"variant": f"fade_shuffled_seed_{seed}", **m_shuf})
    print(f"{'xgb_K3_long + fade@0.50 shuffled (s=' + str(seed) + ')':<40} "
          f"{m_shuf['cagr_pct']:>+7.2f} {m_shuf['sharpe']:>+5.2f} {m_shuf['max_dd_pct']:>+6.2f}")

shuf_mean_cagr = np.mean([r["cagr_pct"] for r in c_results if "shuffled" in r["variant"]])
shuf_mean_sh = np.mean([r["sharpe"] for r in c_results if "shuffled" in r["variant"]])
print(f"\n  Real fade gate: CAGR {m_fade['cagr_pct']:+.2f}%, Sharpe {m_fade['sharpe']:+.2f}")
print(f"  Shuffled mean:  CAGR {shuf_mean_cagr:+.2f}%, Sharpe {shuf_mean_sh:+.2f}")
if m_fade["cagr_pct"] > shuf_mean_cagr + 20:
    print(f"  [CONFIRMED] Real fade gate adds >+20pp CAGR over shuffled -- signal is real")
else:
    print(f"  [REFUTED] Real fade gate within noise of shuffled -- fade@0.50 win was noise")

# Save
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "asym_validation_sweep.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "A_oversold_bounce": oversold_results,
        "B_no_ranker_baseline": noranker_results,
        "C_shuffled_fade_control": c_results,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
