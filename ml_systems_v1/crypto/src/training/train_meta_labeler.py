"""
Meta-Labeler Training — López de Prado "corrective AI"
=========================================================

For each engine signal that fired in TRAIN+VAL, we know (via triple-barrier)
whether the resulting trade was a WIN or LOSS. Train a LightGBM binary
classifier:

    input  = [engine_name_onehot, asset_onehot, asset_regime_onehot,
              engine_signal_magnitude, recent_vol, recent_return_5d,
              recent_return_30d, bucket_onehot, market_breadth]
    target = win_label (1 if trade return > 0, else 0)

At inference: for each (engine, asset, t), P(win | context) becomes a
CONFIDENCE MULTIPLIER on the signal. Position size = signal × P(win).

Effect: engine fires, but position is only full-size if the meta-labeler
is confident. When the model is unsure, position shrinks — preserving
capital during regimes where the engine's signal is unreliable.

Reference: López de Prado (2018) AFML Ch. 3 "Triple-Barrier Method"
           Hudson & Thames (2022) "Does Meta Labeling Add to Signal Efficacy?"

Run:
    python src/training/train_meta_labeler.py --universe 24 --hold 10 --k-up 1.5 --k-down 1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "strategy"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))

from universe import UNIVERSE_10, UNIVERSE_24, DNA_BUCKET
from cost_model import SPOT_COST
from engine_autopsy_tracker import classify_asset_regime
from triple_barrier_exit import TripleBarrierExit, TripleBarrierConfig
from multi_engine_v3_backtest import load_daily_asset_data
from v4_walkforward_backtest import compute_splits, build_v2_engines

MODEL_DIR = PROJECT_ROOT / "models" / "meta_labeler"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def extract_context(asset_data: Dict, asset: str, t: int) -> Dict[str, float]:
    """Build per-bar context feature vector for meta-labeler input."""
    d = asset_data[asset]
    r = d.get("returns")
    c = d.get("close")
    if r is None or c is None or t >= len(r):
        return None
    if t < 30:
        return None

    recent_30 = r[t-30:t]
    recent_5 = r[t-5:t]
    vol_30 = float(np.std(recent_30) * np.sqrt(365))
    ret_5d = float(np.sum(recent_5)) if len(recent_5) else 0.0
    ret_30d = float(np.sum(recent_30)) if len(recent_30) else 0.0
    # Additional features if present
    out = {
        "vol_30": vol_30,
        "ret_5d": ret_5d,
        "ret_30d": ret_30d,
        "abs_ret_5d": abs(ret_5d),
        "signed_vol_30": vol_30 * (1 if ret_30d > 0 else -1),
    }
    # Include some structural features if available
    for feat in ["norm_funding", "norm_oi_change", "norm_whale", "norm_efficiency"]:
        if feat in d and t < len(d[feat]):
            v = d[feat][t]
            out[feat] = float(v) if np.isfinite(v) else 0.0
    return out


def build_labels(asset_data, engines, names, warmup, train_end, val_end,
                  tb_cfg: TripleBarrierConfig, cost: float):
    """Walk through TRAIN+VAL, record (engine, asset, t, signal, outcome, context)."""
    records = []
    print(f"  Scanning bars {warmup}-{val_end} ({val_end-warmup} days) for signals...")
    t0 = time.time()
    for eng in engines:
        eng_name = eng.cfg.name
        for asset in names:
            d = asset_data[asset]
            c = d.get("close")
            r = d.get("returns")
            high = d.get("high", c)
            low = d.get("low", c)
            if c is None:
                continue
            tb = TripleBarrierExit(tb_cfg)
            # Feed signals sequentially; when engine fires, open TB pos
            for t in range(warmup, val_end):
                # Check existing barrier
                if tb.is_open(asset):
                    status = tb.check(asset, t, high[t] if t < len(high) else c[t],
                                      low[t] if t < len(low) else c[t], c[t])
                    if status.startswith("exit"):
                        ret_trade, pos = tb.close(asset, t, c[t], status)
                        if pos is not None:
                            win = 1 if ret_trade > 2 * cost / 2 else 0
                            ctx = extract_context(asset_data, asset, pos.entry_t)
                            if ctx is not None:
                                records.append({
                                    "engine": eng_name, "asset": asset,
                                    "bucket": DNA_BUCKET.get(asset, "UNKNOWN"),
                                    "entry_t": pos.entry_t, "exit_t": t,
                                    "regime": classify_asset_regime(c, r, pos.entry_t),
                                    "hold_bars": t - pos.entry_t,
                                    "ret_trade": ret_trade, "win": win,
                                    "reason": status.replace("exit_", ""),
                                    **{f"ctx_{k}": v for k, v in ctx.items()},
                                })
                # New entry: engine signal > 0.1
                try:
                    sig = eng.compute_signals({asset: d}, t).get(asset, 0.0)
                except Exception:
                    continue
                if not tb.is_open(asset) and sig > 0.1:
                    daily_sigma = max(tb_cfg.min_sigma,
                                       float(np.std(r[max(0, t-30):t])) if t > 30 else 0.02)
                    tb.open(asset, t, c[t], daily_sigma)
                    # store signal magnitude for feature
                    records_last = {"_entry_sig": float(sig)} if sig else None
    print(f"  Scanning done in {time.time()-t0:.1f}s")
    print(f"  Recorded {len(records)} complete trades")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=int, default=24, choices=[10, 24])
    parser.add_argument("--k-up", type=float, default=1.5)
    parser.add_argument("--k-down", type=float, default=1.0)
    parser.add_argument("--max-hold", type=int, default=20)
    args = parser.parse_args()

    universe = UNIVERSE_24 if args.universe == 24 else UNIVERSE_10
    print(f"Meta-Labeler Training — U{args.universe}, TB k_up={args.k_up} "
          f"k_down={args.k_down} max_hold={args.max_hold}")
    print("=" * 70)
    t0 = time.time()
    names, ret_matrix, asset_data = load_daily_asset_data(universe)
    n_days = ret_matrix.shape[0]
    print(f"Loaded {len(names)} assets, {n_days} days ({time.time()-t0:.1f}s)")

    warmup = 120
    splits = compute_splits(n_days, warmup=warmup)
    tb_cfg = TripleBarrierConfig(k_up=args.k_up, k_down=args.k_down,
                                   max_hold=args.max_hold)
    cost = SPOT_COST.round_trip()
    engines = build_v2_engines(names, asset_data, n_days, args.universe)

    # Build trades dataset on TRAIN+VAL (no OOS leak)
    records = build_labels(asset_data, engines, names, warmup,
                             splits.train_end, splits.val_end, tb_cfg, cost)
    if not records:
        print("No trades recorded — aborting.")
        return

    df = pl.DataFrame(records)
    out_trades = PROJECT_ROOT / "logs" / f"meta_labeler_trades_u{args.universe}.csv"
    df.write_csv(out_trades)
    print(f"\nTrades written: {out_trades}")

    # Summary (avoid polars pretty-print which uses unicode box chars)
    print(f"\nTrade outcome distribution:")
    by_reason = df.group_by("reason").agg(pl.len().alias("n"), pl.col("win").mean().alias("win_rate"),
                                            pl.col("ret_trade").mean().alias("avg_ret"))
    for r in by_reason.iter_rows(named=True):
        print(f"  reason={r['reason']:<12} n={r['n']:>6}  win_rate={r['win_rate']:.3f}  avg_ret={r['avg_ret']:+.4f}")

    print(f"\nWin rate by engine:")
    by_eng = df.group_by("engine").agg(pl.len().alias("n"), pl.col("win").mean().alias("win_rate"),
                                          pl.col("ret_trade").mean().alias("avg_ret")).sort("win_rate", descending=True)
    for r in by_eng.iter_rows(named=True):
        print(f"  engine={r['engine']:<25} n={r['n']:>6}  win_rate={r['win_rate']:.3f}  avg_ret={r['avg_ret']:+.4f}")

    # Train LightGBM classifier
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    # Features: one-hot engine + asset + regime + bucket + numeric context
    df_pd = df.to_pandas()
    # Label encoding
    df_pd["engine_id"] = df_pd["engine"].astype("category").cat.codes
    df_pd["asset_id"] = df_pd["asset"].astype("category").cat.codes
    df_pd["regime_id"] = df_pd["regime"].astype("category").cat.codes
    df_pd["bucket_id"] = df_pd["bucket"].astype("category").cat.codes
    feature_cols = ["engine_id", "asset_id", "regime_id", "bucket_id", "hold_bars"]
    feature_cols += [c for c in df_pd.columns if c.startswith("ctx_")]

    X = df_pd[feature_cols].fillna(0).values
    y = df_pd["win"].values

    # Time-based split: train on first 70%, val on last 30%
    n = len(df_pd)
    split_idx = int(0.70 * n)
    X_tr, X_v = X[:split_idx], X[split_idx:]
    y_tr, y_v = y[:split_idx], y[split_idx:]

    print(f"\nTraining: X {X_tr.shape} y balance={y_tr.mean():.3f}")
    print(f"Val:      X {X_v.shape} y balance={y_v.mean():.3f}")

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols,
                              categorical_feature=["engine_id", "asset_id", "regime_id", "bucket_id"])
    val_data = lgb.Dataset(X_v, label=y_v, reference=train_data,
                            feature_name=feature_cols,
                            categorical_feature=["engine_id", "asset_id", "regime_id", "bucket_id"])
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "verbose": -1,
    }
    t1 = time.time()
    model = lgb.train(params, train_data, num_boost_round=300,
                      valid_sets=[train_data, val_data], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(25), lgb.log_evaluation(50)])
    print(f"Training done in {time.time()-t1:.1f}s, best iter {model.best_iteration}")

    # Val AUC
    from sklearn.metrics import roc_auc_score, brier_score_loss
    p_val = model.predict(X_v)
    auc = roc_auc_score(y_v, p_val)
    brier = brier_score_loss(y_v, p_val)
    print(f"\nVal AUC: {auc:.3f}  (>0.5 = better than random)")
    print(f"Val Brier: {brier:.4f} (<0.25 = useful calibration)")

    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])
    print("\nTop 10 features by gain:")
    for f, g in imp[:10]:
        print(f"  {f:<25} {g:>10.0f}")

    # Save
    out = MODEL_DIR / f"meta_labeler_u{args.universe}.txt"
    model.save_model(str(out), num_iteration=model.best_iteration)
    (MODEL_DIR / f"features_u{args.universe}.txt").write_text("\n".join(feature_cols))
    # Save categorical mappings so inference can encode
    mappings = {
        "engines": sorted(df_pd["engine"].unique().tolist()),
        "assets": sorted(df_pd["asset"].unique().tolist()),
        "regimes": sorted(df_pd["regime"].unique().tolist()),
        "buckets": sorted(df_pd["bucket"].unique().tolist()),
    }
    import json
    (MODEL_DIR / f"mappings_u{args.universe}.json").write_text(json.dumps(mappings, indent=2))
    print(f"\nModel: {out}")
    print(f"Features: {MODEL_DIR / f'features_u{args.universe}.txt'}")
    print(f"Mappings: {MODEL_DIR / f'mappings_u{args.universe}.json'}")


if __name__ == "__main__":
    main()
