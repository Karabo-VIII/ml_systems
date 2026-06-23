"""Multi-Timeframe Mover Selection — run the same engine at all cadences.

Tests the selection engine at 15m, 30m, 1h, 4h, 1d to see if there's a
sweet spot where selection skill + opportunity frequency + cost reality
converge on the best daily ROI.

Key questions:
- Does selection skill improve at faster cadences? (more data to learn from)
- Does the opportunity per-bar shrink faster than frequency grows?
- Is there a cost cliff that kills sub-1h? (D60 found 1h cost wall at -0.43%)
- Does the dead-list's "no selection alpha at 1d" hold at 4h/1h/15m?

Architecture: same LightGBM regime-gated momentum selection as V3/4h,
parametrized by cadence. Each cadence gets its own run.
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime
import json


@dataclass
class MultiTFConfig:
    chimera_base: Path = Path("crypto/data/processed/chimera")
    
    # Per-cadence parameters (bars_per_day, min_return, sma_period, momentum_lookback)
    cadence_params: dict = None
    
    # Common
    top_k: int = 7
    portfolio_top_n: int = 5
    max_weight: float = 0.25
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.04
    val_start: str = "2024-06-01"
    train_start: str = "2021-06-01"
    
    def __post_init__(self):
        if self.cadence_params is None:
            self.cadence_params = {
                "15m": {"bars_per_day": 96, "min_ret": 0.003, "sma_bars": 960, "mom_bars": 12, "retrain": 960, "vol_target": 0.005, "min_mom_short": 0.0005, "min_mom_long": 0.002},
                "30m": {"bars_per_day": 48, "min_ret": 0.004, "sma_bars": 480, "mom_bars": 8, "retrain": 480, "vol_target": 0.008, "min_mom_short": 0.001, "min_mom_long": 0.003},
                "1h":  {"bars_per_day": 24, "min_ret": 0.005, "sma_bars": 360, "mom_bars": 6, "retrain": 360, "vol_target": 0.010, "min_mom_short": 0.001, "min_mom_long": 0.004},
                "4h":  {"bars_per_day": 6,  "min_ret": 0.008, "sma_bars": 150, "mom_bars": 6, "retrain": 150, "vol_target": 0.015, "min_mom_short": 0.001, "min_mom_long": 0.005},
                "1d":  {"bars_per_day": 1,  "min_ret": 0.015, "sma_bars": 100, "mom_bars": 7, "retrain": 30,  "vol_target": 0.030, "min_mom_short": 0.002, "min_mom_long": 0.008},
            }


def run_cadence(cadence: str, config: MultiTFConfig) -> dict:
    """Run selection engine for a single cadence."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")
    
    params = config.cadence_params[cadence]
    chimera_dir = config.chimera_base / cadence
    
    # Load
    files = sorted(chimera_dir.glob("*_v51_chimera_*.parquet"))
    if not files:
        return {"cadence": cadence, "status": "NO_DATA"}
    
    frames = []
    btc_data = None
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
            keep = ["date", "open", "high", "low", "close", "volume"]
            keep += [c for c in df.columns if any(c.startswith(p) for p in
                     ("norm_", "rv_", "hbr_", "bd_", "fund_", "liq_", "bs_"))]
            keep = [c for c in keep if c in df.columns]
            df = df.select(keep).with_columns(pl.lit(sym).alias("asset"))
            frames.append(df)
            if sym == "BTCUSDT":
                btc_data = df.select(["date", "close"]).sort("date")
        except Exception:
            pass
    
    panel = pl.concat(frames, how="diagonal").sort(["asset", "date"])
    n_assets = panel["asset"].n_unique()
    
    # Features
    mom_bars = params["mom_bars"]
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns([
        pl.col("close").pct_change(1).over("asset").alias("ret_1"),
        pl.col("close").pct_change(mom_bars).over("asset").alias("ret_mom"),
        pl.col("close").pct_change(mom_bars * 2).over("asset").alias("ret_mom2"),
        pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1"),
        pl.col("close").pct_change(mom_bars).over("asset").abs().alias("abs_ret_mom"),
        pl.col("close").pct_change(1).rolling_std(mom_bars).over("asset").alias("vol_short"),
        pl.col("close").pct_change(1).rolling_std(mom_bars * 4).over("asset").alias("vol_long"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("range_pct"),
        (pl.col("volume") / pl.col("volume").rolling_mean(mom_bars * 4).over("asset")).alias("vol_ratio"),
    ])
    
    # Cross-sectional
    for col in ["ret_1", "ret_mom", "abs_ret_1", "abs_ret_mom", "vol_short", "vol_long", "range_pct", "vol_ratio"]:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") /
                 pl.col(col).count().over("date")).alias(f"xrank_{col}")
            )
    
    # Label
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return")
    )
    panel = panel.with_columns(
        pl.when(pl.col("fwd_return") >= params["min_ret"])
        .then(pl.col("fwd_return"))
        .otherwise(pl.lit(-999.0))
        .rank(method="ordinal", descending=True).over("date")
        .alias("pos_rank")
    )
    panel = panel.with_columns(
        ((pl.col("pos_rank") <= config.top_k) &
         (pl.col("fwd_return") >= params["min_ret"]))
        .cast(pl.Int8).alias("is_top_mover")
    )
    
    # Regime
    btc_close = btc_data["close"].to_numpy()
    sma_p = params["sma_bars"]
    sma = np.convolve(btc_close, np.ones(sma_p)/sma_p, "full")[:len(btc_close)]
    sma[:sma_p-1] = np.nan
    regime = np.zeros(len(btc_close), dtype=int)
    state = 0
    for i in range(sma_p, len(btc_close)):
        if np.isnan(sma[i]): continue
        if state == 0 and btc_close[i] > sma[i] * 1.015:
            state = 1
        elif state == 1 and btc_close[i] < sma[i] * 0.985:
            state = 0
        regime[i] = state
    
    btc_regime_df = btc_data.sort("date").with_columns(pl.Series("regime", regime))
    panel = panel.join(btc_regime_df.select(["date", "regime"]), on="date", how="left")
    
    # Feature columns
    exclude = {"date", "asset", "open", "high", "low", "close", "volume",
               "fwd_return", "is_top_mover", "pos_rank", "regime"}
    feature_cols = sorted([c for c in panel.columns if c not in exclude and not c.startswith("target_")])
    
    # Walk-forward
    panel = panel.filter(pl.col("is_top_mover").is_not_null() & pl.col("fwd_return").is_not_null())
    val_start_d = date.fromisoformat(config.val_start)
    train_start_d = date.fromisoformat(config.train_start)
    all_dates = sorted(panel.filter(pl.col("date") >= val_start_d)["date"].unique().to_list())
    
    model = None
    bar_rets = []
    n_bull = 0
    n_bear = 0
    retrain_every = params["retrain"]
    
    for i, pred_date in enumerate(all_dates):
        # Regime
        reg_row = btc_regime_df.filter(pl.col("date") == pred_date)
        if len(reg_row) == 0 or int(reg_row["regime"][0]) == 0:
            bar_rets.append(0.0)
            n_bear += 1
            continue
        
        # Retrain
        if model is None or (i % retrain_every == 0):
            train_df = panel.filter((pl.col("date") >= train_start_d) & (pl.col("date") < pred_date))
            if len(train_df) < 3000:
                bar_rets.append(0.0)
                continue
            X = np.nan_to_num(train_df.select(feature_cols).to_numpy(), nan=0.0)
            y = train_df["is_top_mover"].to_numpy()
            td = lgb.Dataset(X, label=y)
            lgb_params = {"objective": "binary", "metric": "auc",
                         "max_depth": config.max_depth, "learning_rate": config.learning_rate,
                         "min_child_samples": 50, "verbosity": -1, "num_threads": 4,
                         "is_unbalance": True, "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5}
            model = lgb.train(lgb_params, td, num_boost_round=config.n_estimators)
        
        # Predict
        pred_df = panel.filter(pl.col("date") == pred_date)
        if len(pred_df) == 0:
            bar_rets.append(0.0)
            continue
        
        X_pred = np.nan_to_num(pred_df.select(feature_cols).to_numpy(), nan=0.0)
        scores = model.predict(X_pred)
        
        # Gates
        ret_1 = np.nan_to_num(pred_df["ret_1"].to_numpy())
        ret_mom = np.nan_to_num(pred_df["ret_mom"].to_numpy())
        
        mask = ((ret_1 >= params["min_mom_short"]) &
                (ret_mom >= params["min_mom_long"]) &
                (scores > 0.10))
        
        candidates_idx = np.where(mask)[0]
        if len(candidates_idx) == 0:
            bar_rets.append(0.0)
            n_bull += 1
            continue
        
        # Top-N
        cand_scores = scores[candidates_idx]
        top_k = min(config.portfolio_top_n, len(candidates_idx))
        selected = candidates_idx[np.argsort(cand_scores)[::-1][:top_k]]
        
        # Vol-targeted
        vol_vals = pred_df["vol_short"].to_numpy()[selected]
        vol_vals = np.clip(np.nan_to_num(vol_vals, nan=0.02), 0.001, 0.20)
        weights = params["vol_target"] / vol_vals
        weights = np.minimum(weights, config.max_weight)
        weights = weights / weights.sum()
        
        # Return (NO trailing stop)
        fwd = np.nan_to_num(pred_df["fwd_return"].to_numpy()[selected], nan=0.0)
        bar_rets.append(float(np.dot(weights, fwd)))
        n_bull += 1
    
    # Results
    rets = np.array(bar_rets)
    bars_per_day = params["bars_per_day"]
    
    compound = float(np.prod(1 + rets) - 1)
    mean_bar = float(np.mean(rets))
    mean_day = mean_bar * bars_per_day
    std_bar = float(np.std(rets))
    sharpe = mean_bar / std_bar * np.sqrt(365 * bars_per_day) if std_bar > 0 else 0
    win_rate = float(np.mean(rets > 0))
    
    cum = np.cumprod(1 + rets)
    max_dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) if len(cum) > 0 else 0
    
    return {
        "cadence": cadence,
        "n_bars": len(rets),
        "n_assets": n_assets,
        "n_features": len(feature_cols),
        "n_bull_bars": n_bull,
        "n_bear_bars": n_bear,
        "bars_per_day": bars_per_day,
        "mean_per_bar_pct": mean_bar * 100,
        "mean_per_day_pct": mean_day * 100,
        "compound_pct": compound * 100,
        "sharpe_annual": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": win_rate,
    }


def run_all_timeframes(config: MultiTFConfig = None):
    """Run across all timeframes and compare."""
    if config is None:
        config = MultiTFConfig()
    
    print("=" * 80)
    print("MULTI-TIMEFRAME MOVER SELECTION")
    print("=" * 80)
    
    results = []
    cadences = ["1d", "4h", "1h", "30m", "15m"]
    
    for cadence in cadences:
        print(f"\n{'='*40}")
        print(f"  RUNNING: {cadence}")
        print(f"{'='*40}")
        r = run_cadence(cadence, config)
        results.append(r)
        
        if r.get("status") == "NO_DATA":
            print(f"  SKIP: no data")
            continue
        
        print(f"  bars={r['n_bars']:,}, assets={r['n_assets']}, features={r['n_features']}")
        print(f"  mean/bar={r['mean_per_bar_pct']:.4f}%, mean/day={r['mean_per_day_pct']:.4f}%")
        print(f"  compound={r['compound_pct']:.1f}%, sharpe={r['sharpe_annual']:.2f}, DD={r['max_dd_pct']:.1f}%")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY — CROSS-TIMEFRAME COMPARISON")
    print("=" * 80)
    print(f"\n{'Cadence':<8} {'Bars':>8} {'Mean/Bar':>10} {'Mean/Day':>10} {'Compound':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-" * 80)
    
    for r in results:
        if r.get("status") == "NO_DATA":
            print(f"{r['cadence']:<8} {'NO DATA':>8}")
            continue
        print(f"{r['cadence']:<8} {r['n_bars']:>8,} {r['mean_per_bar_pct']:>9.4f}% {r['mean_per_day_pct']:>9.4f}% {r['compound_pct']:>9.1f}% {r['sharpe_annual']:>7.2f} {r['max_dd_pct']:>7.1f}% {r['win_rate']:>7.3f}")
    
    # Find best cadence by daily return
    valid = [r for r in results if r.get("status") != "NO_DATA"]
    if valid:
        best = max(valid, key=lambda x: x["mean_per_day_pct"])
        print(f"\n  BEST CADENCE BY DAILY RETURN: {best['cadence']} ({best['mean_per_day_pct']:.4f}%/day)")
        
        best_sharpe = max(valid, key=lambda x: x["sharpe_annual"])
        print(f"  BEST CADENCE BY SHARPE: {best_sharpe['cadence']} (Sharpe {best_sharpe['sharpe_annual']:.2f})")
        
        if best["mean_per_day_pct"] >= 1.0:
            print(f"\n  ** NORTH STAR 1%/day ACHIEVED at {best['cadence']} **")
        elif best["mean_per_day_pct"] >= 0.5:
            print(f"\n  ** Promising: {best['mean_per_day_pct']:.3f}%/day at {best['cadence']} **")
    
    # Save
    output_dir = Path("crypto/runs/strat")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"version": "multitf_v1", "results": results, "timestamp": ts}
    with open(output_dir / f"multitf_selection_{ts}.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to crypto/runs/strat/multitf_selection_{ts}.json")
    
    return results


if __name__ == "__main__":
    run_all_timeframes()
