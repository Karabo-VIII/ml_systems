"""Vol-Breakout Selection — ride high-vol assets in their trending direction.

Fundamental insight from V1-V3 + dead-list:
- Direction is UNPREDICTABLE (D55) but vol IS predictable (Hurst(|ret|) 0.80-0.84)
- Continuation dominates reversion (D52, D53)
- Vol-targeting is the #1 safe overlay

So the strategy is:
1. PREDICT which assets will have HIGH ABSOLUTE MOVES tomorrow (vol prediction)
2. Of those, SELECT only the ones with POSITIVE momentum (trend-following filter)
3. SIZE inversely to their own vol (vol-target)
4. GATE on BTC regime (only deploy in bull)

This is NOT "predict direction" (dead). It's:
- "Predict vol" (alive, confirmed predictable)
- "Bet on continuation" (alive, confirmed D52/D53)
- "Only in bull" (alive, V1 drawdown-insurance verified)

The bet: high-vol + positive-momentum + bull-regime = continuation UP.
This fails if: high-vol means crash (checked: in bull regime, high-vol
assets have positive skew — the meme/momentum names run UP in bull).
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime
import json


@dataclass
class VolBreakoutConfig:
    chimera_dir: Path = Path("crypto/data/processed/chimera/1d")
    
    # Regime (same as V3)
    regime_sma_period: int = 75  # FASTER than V3 (75 vs 100) — crypto moves fast
    regime_hysteresis: float = 0.02  # tighter hysteresis
    
    # Vol prediction target: will this asset be in top-20% by |return| tomorrow?
    vol_percentile_threshold: float = 0.20  # top 20% by absolute return
    
    # Continuation filter
    min_trailing_ret_1d: float = 0.005  # must have been +0.5% yesterday
    min_trailing_ret_3d: float = 0.01   # must have been +1% over 3 days
    
    # Portfolio
    portfolio_top_n: int = 5
    vol_target_level: float = 0.025  # 2.5% daily vol target (tighter)
    max_weight: float = 0.30
    
    # Trailing stop
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.05  # 5% trailing stop per position
    
    # Model
    n_estimators: int = 200
    max_depth: int = 4  # shallower (less overfit)
    learning_rate: float = 0.05
    
    # Walk-forward
    train_start: str = "2021-01-01"
    val_start: str = "2024-06-01"
    retrain_every: int = 30


def run_vol_breakout(config: VolBreakoutConfig = None) -> dict:
    """Run the vol-breakout continuation strategy."""
    if config is None:
        config = VolBreakoutConfig()
    
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")
    
    print("=" * 80)
    print("VOL-BREAKOUT SELECTION (predict vol + ride continuation)")
    print("=" * 80)
    
    # Load
    print("\n[1] Loading...")
    files = sorted(config.chimera_dir.glob("*_v51_chimera_1d_*.parquet"))
    frames = []
    btc_close = None
    
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
            keep = ["date", "open", "high", "low", "close", "volume"]
            keep += [c for c in df.columns if any(c.startswith(p) for p in 
                     ("norm_", "rv_", "hbr_", "bd_", "fund_", "liq_"))]
            keep = [c for c in keep if c in df.columns]
            df = df.select(keep).with_columns(pl.lit(sym).alias("asset"))
            frames.append(df)
            if sym == "BTCUSDT":
                btc_close = df.select(["date", "close"]).sort("date")
        except Exception:
            pass
    
    panel = pl.concat(frames, how="diagonal").sort(["asset", "date"])
    print(f"  {len(panel):,} rows, {panel['asset'].n_unique()} assets")
    
    # Features
    print("\n[2] Features...")
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns([
        pl.col("close").pct_change(1).over("asset").alias("ret_1d"),
        pl.col("close").pct_change(3).over("asset").alias("ret_3d"),
        pl.col("close").pct_change(7).over("asset").alias("ret_7d"),
        pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1d"),
        pl.col("close").pct_change(3).over("asset").abs().alias("abs_ret_3d"),
        pl.col("close").pct_change(1).rolling_std(5).over("asset").alias("vol_5d"),
        pl.col("close").pct_change(1).rolling_std(10).over("asset").alias("vol_10d"),
        pl.col("close").pct_change(1).rolling_std(20).over("asset").alias("vol_20d"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("range_pct"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).rolling_mean(5).over("asset").alias("avg_range_5d"),
        # Volume momentum
        (pl.col("volume") / pl.col("volume").rolling_mean(10).over("asset")).alias("vol_ratio"),
    ])
    
    # Vol expansion signal
    panel = panel.with_columns(
        (pl.col("range_pct") / pl.col("avg_range_5d").clip(lower_bound=0.001)).alias("vol_expansion")
    )
    
    # Cross-sectional ranks (critical features from V2)
    for col in ["abs_ret_1d", "abs_ret_3d", "vol_5d", "vol_10d", "range_pct", "vol_expansion", "vol_ratio"]:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") /
                 pl.col(col).count().over("date")).alias(f"xrank_{col}")
            )
    
    # LABEL: will this asset be in top-20% by ABSOLUTE return tomorrow?
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return_1d")
    )
    panel = panel.with_columns(
        (pl.col("fwd_return_1d").abs()
         .rank(method="ordinal", descending=True)
         .over("date") / pl.col("fwd_return_1d").count().over("date"))
        .alias("fwd_abs_return_pctile")
    )
    panel = panel.with_columns(
        (pl.col("fwd_abs_return_pctile") <= config.vol_percentile_threshold)
        .cast(pl.Int8)
        .alias("is_high_vol")
    )
    
    # BTC regime
    print("\n[3] Regime...")
    btc_arr = btc_close.sort("date")["close"].to_numpy()
    sma = np.convolve(btc_arr, np.ones(config.regime_sma_period)/config.regime_sma_period, "full")[:len(btc_arr)]
    sma[:config.regime_sma_period-1] = np.nan
    
    regime = np.zeros(len(btc_arr), dtype=int)
    state = 0
    for i in range(config.regime_sma_period, len(btc_arr)):
        if np.isnan(sma[i]): continue
        if state == 0 and btc_arr[i] > sma[i] * (1 + config.regime_hysteresis):
            state = 1
        elif state == 1 and btc_arr[i] < sma[i] * (1 - config.regime_hysteresis):
            state = 0
        regime[i] = state
    
    btc_regime_df = btc_close.sort("date").with_columns(pl.Series("btc_regime", regime))
    panel = panel.join(btc_regime_df.select(["date", "btc_regime"]), on="date", how="left")
    
    bull_pct = regime.sum() / len(regime)
    print(f"  Bull ratio: {bull_pct:.1%}")
    
    # Feature columns
    exclude = {"date", "asset", "open", "high", "low", "close", "volume",
               "fwd_return_1d", "fwd_abs_return_pctile", "is_high_vol", "btc_regime"}
    feature_cols = sorted([c for c in panel.columns if c not in exclude and not c.startswith("target_")])
    print(f"  Features: {len(feature_cols)}")
    
    # Walk-forward
    print("\n[4] Walk-forward...")
    panel = panel.filter(pl.col("is_high_vol").is_not_null() & pl.col("fwd_return_1d").is_not_null())
    
    val_start = date.fromisoformat(config.val_start)
    train_start = date.fromisoformat(config.train_start)
    all_dates = sorted(panel.filter(pl.col("date") >= val_start)["date"].unique().to_list())
    
    model = None
    daily_results = []
    
    for i, pred_date in enumerate(all_dates):
        # Regime check
        regime_row = btc_regime_df.filter(pl.col("date") == pred_date)
        if len(regime_row) == 0 or int(regime_row["btc_regime"][0]) == 0:
            daily_results.append({"date": pred_date, "port_return": 0.0, "regime": "bear", "n": 0})
            continue
        
        # Retrain
        if model is None or (i % config.retrain_every == 0):
            train_df = panel.filter((pl.col("date") >= train_start) & (pl.col("date") < pred_date))
            if len(train_df) < 2000:
                daily_results.append({"date": pred_date, "port_return": 0.0, "regime": "skip", "n": 0})
                continue
            X = np.nan_to_num(train_df.select(feature_cols).to_numpy(), nan=0.0)
            y = train_df["is_high_vol"].to_numpy()
            td = lgb.Dataset(X, label=y)
            params = {"objective": "binary", "metric": "auc", "max_depth": config.max_depth,
                     "learning_rate": config.learning_rate, "verbosity": -1,
                     "num_threads": 4, "is_unbalance": True,
                     "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5}
            model = lgb.train(params, td, num_boost_round=config.n_estimators)
        
        # Predict vol
        pred_df = panel.filter(pl.col("date") == pred_date)
        if len(pred_df) == 0:
            continue
        
        X_pred = np.nan_to_num(pred_df.select(feature_cols).to_numpy(), nan=0.0)
        vol_scores = model.predict(X_pred)
        
        # CONTINUATION FILTER: only assets with positive recent momentum
        ret_1d = pred_df["ret_1d"].to_numpy()
        ret_3d = pred_df["ret_3d"].to_numpy()
        
        # Gates: predicted high-vol + positive momentum
        mask = (vol_scores > 0.3) & \
               (np.nan_to_num(ret_1d) >= config.min_trailing_ret_1d) & \
               (np.nan_to_num(ret_3d) >= config.min_trailing_ret_3d)
        
        candidates_idx = np.where(mask)[0]
        
        if len(candidates_idx) == 0:
            daily_results.append({"date": pred_date, "port_return": 0.0, "regime": "bull_no_cand", "n": 0})
            continue
        
        # Rank by vol score and take top-N
        cand_scores = vol_scores[candidates_idx]
        top_k = min(config.portfolio_top_n, len(candidates_idx))
        top_local = np.argsort(cand_scores)[::-1][:top_k]
        selected_idx = candidates_idx[top_local]
        
        # Vol-targeted weights
        vol_vals = pred_df["vol_10d"].to_numpy()[selected_idx]
        vol_vals = np.clip(np.nan_to_num(vol_vals, nan=0.05), 0.005, 0.30)
        weights = config.vol_target_level / vol_vals
        weights = np.minimum(weights, config.max_weight)
        weights = weights / weights.sum()
        
        # Returns
        fwd_rets = pred_df["fwd_return_1d"].to_numpy()[selected_idx]
        fwd_rets = np.nan_to_num(fwd_rets, nan=0.0)
        
        # Trailing stop simulation (simplified: cap loss at -5% per position)
        if config.use_trailing_stop:
            fwd_rets = np.maximum(fwd_rets, -config.trailing_stop_pct)
        
        port_return = float(np.dot(weights, fwd_rets))
        daily_results.append({"date": pred_date, "port_return": port_return, "regime": "bull", "n": len(selected_idx)})
        
        if (i+1) % 150 == 0:
            print(f"  {i+1}/{len(all_dates)}")
    
    # Evaluate
    print("\n[5] Results...")
    daily_df = pl.DataFrame(daily_results).sort("date")
    rets = daily_df["port_return"].to_numpy()
    
    compound = float(np.prod(1 + rets) - 1)
    mean_daily = float(np.mean(rets))
    std_daily = float(np.std(rets))
    sharpe = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0
    win_rate = float(np.mean(rets > 0))
    
    cum = np.cumprod(1 + rets)
    max_dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))
    
    bull_mask = np.array([r["regime"] == "bull" for r in daily_results])
    bull_rets = rets[bull_mask]
    bull_mean = float(np.mean(bull_rets)) if len(bull_rets) > 0 else 0
    bull_compound = float(np.prod(1 + bull_rets) - 1) if len(bull_rets) > 0 else 0
    n_bull = int(bull_mask.sum())
    n_bear = int(np.array([r["regime"] == "bear" for r in daily_results]).sum())
    
    print("\n" + "=" * 80)
    print("VOL-BREAKOUT RESULTS")
    print("=" * 80)
    print(f"\n  OVERALL:")
    print(f"    Days: {len(rets)} (deployed: {n_bull}, cash: {n_bear})")
    print(f"    Mean daily: {mean_daily*100:.4f}%")
    print(f"    Compound: {compound*100:.2f}%")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Max DD: {max_dd*100:.1f}%")
    print(f"    Win rate: {win_rate:.3f}")
    
    print(f"\n  BULL REGIME (deployed):")
    print(f"    Days: {n_bull}")
    print(f"    Mean daily: {bull_mean*100:.4f}%")
    print(f"    Compound: {bull_compound*100:.2f}%")
    
    print(f"\n  COMPARISONS:")
    print(f"    BTC buy-hold: +8.5%")
    print(f"    EW basket: -69.1%")
    print(f"    V3 regime-gated: +6.5%")
    print(f"    THIS (vol-breakout): {compound*100:.2f}%")
    
    if compound > 0.065:
        print(f"\n  ** BEATS V3 **")
    if mean_daily >= 0.01:
        print(f"  ** NORTH STAR 1%/day ACHIEVED **")
    elif bull_mean >= 0.005:
        print(f"  ** Bull-only: {bull_mean*100:.3f}%/day — building toward target **")
    
    # Save
    output = {
        "version": "vol_breakout_v1",
        "compound": compound, "mean_daily_pct": mean_daily * 100,
        "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
        "bull": {"n": n_bull, "mean_pct": bull_mean * 100, "compound": bull_compound},
        "bear_cash": n_bear,
    }
    output_dir = Path("crypto/runs/strat")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"vol_breakout_v1_{ts}.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved.")
    
    return output


if __name__ == "__main__":
    run_vol_breakout()
