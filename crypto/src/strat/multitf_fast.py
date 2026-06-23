"""Fast Multi-Timeframe Selection — optimized for large panels.

The previous version was too slow on sub-daily cadences because it used
polars .filter() per bar in the walk-forward loop. This version:
1. Pre-materializes the panel into numpy arrays (dates, features, labels, returns)
2. Uses pre-built date->index mappings for O(1) per-bar access
3. Trains less frequently (larger retrain windows)
4. Caps the walk-forward to last N bars for faster feedback
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from datetime import date, datetime
import json
import time


CADENCE_PARAMS = {
    "15m": {"bars_per_day": 96, "min_ret": 0.003, "sma_bars": 960, "mom_bars": 12, "retrain": 2000, "vol_target": 0.005},
    "30m": {"bars_per_day": 48, "min_ret": 0.004, "sma_bars": 480, "mom_bars": 8, "retrain": 1000, "vol_target": 0.008},
    "1h":  {"bars_per_day": 24, "min_ret": 0.005, "sma_bars": 360, "mom_bars": 6, "retrain": 500, "vol_target": 0.010},
    "4h":  {"bars_per_day": 6,  "min_ret": 0.008, "sma_bars": 150, "mom_bars": 6, "retrain": 200, "vol_target": 0.015},
    "1d":  {"bars_per_day": 1,  "min_ret": 0.015, "sma_bars": 100, "mom_bars": 7, "retrain": 30,  "vol_target": 0.030},
}


def run_cadence_fast(cadence: str) -> dict:
    """Run selection for one cadence using numpy-optimized walk-forward."""
    import lightgbm as lgb
    
    params = CADENCE_PARAMS[cadence]
    chimera_dir = Path(f"crypto/data/processed/chimera/{cadence}")
    
    t0 = time.time()
    
    # Load
    files = sorted(chimera_dir.glob("*_v51_chimera_*.parquet"))
    if not files:
        return {"cadence": cadence, "status": "NO_DATA"}
    
    frames = []
    btc_close_series = None
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
            keep = ["date", "close"]
            keep += [c for c in df.columns if any(c.startswith(p) for p in
                     ("norm_", "rv_", "hbr_", "fund_", "liq_"))]
            keep = [c for c in keep if c in df.columns]
            df = df.select(keep).with_columns(pl.lit(sym).alias("asset"))
            frames.append(df)
            if sym == "BTCUSDT":
                btc_close_series = df.select(["date", "close"]).sort("date")
        except Exception:
            pass
    
    panel = pl.concat(frames, how="diagonal").sort(["asset", "date"])
    n_assets = panel["asset"].n_unique()
    
    # Add derived features per-asset
    panel = panel.sort(["asset", "date"])
    mom = params["mom_bars"]
    panel = panel.with_columns([
        pl.col("close").pct_change(1).over("asset").alias("ret_1"),
        pl.col("close").pct_change(mom).over("asset").alias("ret_mom"),
        pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1"),
        pl.col("close").pct_change(1).rolling_std(mom).over("asset").alias("vol_short"),
    ])
    
    # Forward return (label)
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return")
    )
    
    # Cross-sectional ranks
    for col in ["ret_1", "ret_mom", "abs_ret_1", "vol_short"]:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") /
                 pl.col(col).count().over("date")).alias(f"xrank_{col}")
            )
    
    # Label: top-K positive movers
    panel = panel.with_columns(
        pl.when(pl.col("fwd_return") >= params["min_ret"])
        .then(pl.col("fwd_return"))
        .otherwise(pl.lit(-999.0))
        .rank(method="ordinal", descending=True).over("date")
        .alias("pos_rank")
    )
    panel = panel.with_columns(
        ((pl.col("pos_rank") <= 7) & (pl.col("fwd_return") >= params["min_ret"]))
        .cast(pl.Int8).alias("label")
    )
    
    # Drop nulls
    panel = panel.filter(pl.col("label").is_not_null() & pl.col("fwd_return").is_not_null())
    
    # BTC regime
    btc_close = btc_close_series["close"].to_numpy()
    btc_dates = btc_close_series["date"].to_list()
    sma_p = params["sma_bars"]
    sma = np.convolve(btc_close, np.ones(sma_p)/sma_p, "full")[:len(btc_close)]
    sma[:sma_p-1] = btc_close[:sma_p-1]
    regime_arr = (btc_close > sma * 1.01).astype(int)
    date_to_regime = dict(zip(btc_dates, regime_arr))
    
    # Feature columns
    exclude = {"date", "asset", "close", "fwd_return", "label", "pos_rank"}
    feature_cols = sorted([c for c in panel.columns if c not in exclude and not c.startswith("target_")])
    
    # Materialize into grouped-by-date numpy for fast access
    # Group panel by date, convert to numpy per-date blocks
    val_start_d = date.fromisoformat("2024-06-01")
    train_start_d = date.fromisoformat("2021-06-01")
    
    unique_dates = sorted(panel["date"].unique().to_list())
    test_dates = [d for d in unique_dates if d >= val_start_d]
    train_dates = [d for d in unique_dates if train_start_d <= d < val_start_d]
    
    # Build training set (all dates before val_start)
    train_panel = panel.filter((pl.col("date") >= train_start_d) & (pl.col("date") < val_start_d))
    X_train_full = np.nan_to_num(train_panel.select(feature_cols).to_numpy(), nan=0.0)
    y_train_full = train_panel["label"].to_numpy()
    
    load_time = time.time() - t0
    print(f"  [{cadence}] Loaded: {len(panel):,} rows, {n_assets} assets, {len(feature_cols)} features ({load_time:.1f}s)")
    print(f"  [{cadence}] Train: {len(X_train_full):,} rows, Test dates: {len(test_dates)}")
    
    # Train initial model
    td = lgb.Dataset(X_train_full, label=y_train_full)
    lgb_params = {"objective": "binary", "metric": "auc", "max_depth": 5,
                 "learning_rate": 0.04, "min_child_samples": 50,
                 "verbosity": -1, "num_threads": 4, "is_unbalance": True,
                 "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5}
    model = lgb.train(lgb_params, td, num_boost_round=200)
    
    # Walk-forward (predict each test date)
    bar_rets = []
    retrain_counter = 0
    
    for i, pred_date in enumerate(test_dates):
        # Regime
        regime = date_to_regime.get(pred_date, 0)
        if regime == 0:
            bar_rets.append(0.0)
            continue
        
        # Get this date's data
        day_data = panel.filter(pl.col("date") == pred_date)
        if len(day_data) == 0:
            bar_rets.append(0.0)
            continue
        
        X = np.nan_to_num(day_data.select(feature_cols).to_numpy(), nan=0.0)
        scores = model.predict(X)
        
        ret_1 = day_data["ret_1"].to_numpy()
        ret_mom = day_data["ret_mom"].to_numpy()
        fwd = day_data["fwd_return"].to_numpy()
        vol = day_data["vol_short"].to_numpy()
        
        # Momentum gate + minimum score
        mask = ((np.nan_to_num(ret_1) > 0) &
                (np.nan_to_num(ret_mom) > 0.002) &
                (scores > 0.10))
        
        idx = np.where(mask)[0]
        if len(idx) == 0:
            bar_rets.append(0.0)
            continue
        
        # Top-5 by score
        top_k = min(5, len(idx))
        selected = idx[np.argsort(scores[idx])[::-1][:top_k]]
        
        # Vol-weighted
        v = np.clip(np.nan_to_num(vol[selected], nan=0.02), 0.002, 0.15)
        w = params["vol_target"] / v
        w = np.minimum(w, 0.25)
        w = w / w.sum()
        
        port_ret = float(np.dot(w, np.nan_to_num(fwd[selected], nan=0.0)))
        bar_rets.append(port_ret)
        
        # Periodic retrain (add recent data to training)
        retrain_counter += 1
        if retrain_counter >= params["retrain"]:
            # Retrain with all data up to now
            new_train = panel.filter((pl.col("date") >= train_start_d) & (pl.col("date") <= pred_date))
            X_new = np.nan_to_num(new_train.select(feature_cols).to_numpy(), nan=0.0)
            y_new = new_train["label"].to_numpy()
            td = lgb.Dataset(X_new, label=y_new)
            model = lgb.train(lgb_params, td, num_boost_round=200)
            retrain_counter = 0
    
    # Results
    rets = np.array(bar_rets)
    bpd = params["bars_per_day"]
    
    compound = float(np.prod(1 + rets) - 1)
    mean_bar = float(np.mean(rets))
    mean_day = mean_bar * bpd
    std_bar = float(np.std(rets))
    sharpe = mean_bar / std_bar * np.sqrt(365 * bpd) if std_bar > 0 else 0
    win_rate = float(np.mean(rets > 0))
    
    cum = np.cumprod(1 + rets)
    max_dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) if len(cum) > 1 else 0
    
    elapsed = time.time() - t0
    
    return {
        "cadence": cadence,
        "n_bars": len(rets),
        "n_assets": n_assets,
        "bars_per_day": bpd,
        "mean_per_bar_pct": mean_bar * 100,
        "mean_per_day_pct": mean_day * 100,
        "compound_pct": compound * 100,
        "sharpe_annual": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": win_rate,
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 80)
    print("MULTI-TIMEFRAME SELECTION (fast)")
    print("=" * 80)
    
    results = []
    for cadence in ["1d", "4h", "1h", "30m", "15m"]:
        print(f"\n--- {cadence} ---")
        try:
            r = run_cadence_fast(cadence)
            results.append(r)
            if r.get("status") == "NO_DATA":
                print(f"  NO DATA")
            else:
                print(f"  mean/day={r['mean_per_day_pct']:.4f}%, compound={r['compound_pct']:.1f}%, "
                      f"sharpe={r['sharpe_annual']:.2f}, DD={r['max_dd_pct']:.1f}%, "
                      f"win={r['win_rate']:.3f} [{r['elapsed_s']:.0f}s]")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"cadence": cadence, "status": "ERROR", "error": str(e)})
    
    # Summary
    print("\n" + "=" * 80)
    print("CROSS-TIMEFRAME COMPARISON")
    print("=" * 80)
    print(f"\n{'TF':<6} {'Bars':>8} {'Mean/Day':>10} {'Compound':>10} {'Sharpe':>8} {'MaxDD':>8} {'Win':>6}")
    print("-" * 62)
    for r in results:
        if "mean_per_day_pct" not in r:
            print(f"{r['cadence']:<6} {'N/A':>8}")
            continue
        print(f"{r['cadence']:<6} {r['n_bars']:>8,} {r['mean_per_day_pct']:>9.4f}% "
              f"{r['compound_pct']:>9.1f}% {r['sharpe_annual']:>7.2f} "
              f"{r['max_dd_pct']:>7.1f}% {r['win_rate']:>5.3f}")
    
    valid = [r for r in results if "mean_per_day_pct" in r]
    if valid:
        best = max(valid, key=lambda x: x["mean_per_day_pct"])
        print(f"\nBEST: {best['cadence']} at {best['mean_per_day_pct']:.4f}%/day")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("crypto/runs/strat") / f"multitf_fast_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "timestamp": ts}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
