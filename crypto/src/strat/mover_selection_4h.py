"""Mover Selection at 4H cadence — the sub-day engine.

WHY 4H:
- 6 decision points per day (vs 1 at daily) → 6x more opportunities
- Target per-bar: 0.5-1% (much more achievable than 3-5% at daily)
- Full 243-feature chimera available (same feature set as 1d)
- 14K rows per asset (6x more training data than 1d)
- Executable by a retail trader (no sub-second required)
- Dead-list D60 killed 1h (cost wall -0.43% OOS) but 4h is UNEXPLORED at this resolution

ARCHITECTURE:
1. Load 4h chimera panel (104 assets x 14K rows = ~1.4M row panel)
2. Label: top-K movers per 4h bar (>= 1% positive move)
3. Features: chimera norm_ + derived momentum/vol + cross-sectional
4. Model: LightGBM walk-forward (same as V1-V3 but at 4h)
5. Regime gate: BTC regime at 4h (more responsive than daily SMA)
6. Portfolio: top-5 per bar, vol-weighted, no trailing stop

KEY DIFFERENCE FROM DAILY:
- The opportunity per bar is smaller (0.5-2%) but MORE FREQUENT
- 6 bars/day means even 0.3%/bar compounds to 1.8%/day
- The selection problem is easier: "which assets are breaking out NOW"
  (momentum at 4h is more actionable than at 1d)
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime
import json


@dataclass
class Selection4HConfig:
    chimera_dir: Path = Path("crypto/data/processed/chimera/4h")
    
    # Label
    top_k: int = 7  # top-7 movers per bar
    min_return_threshold: float = 0.008  # 0.8% minimum (lower than daily)
    
    # Regime (faster at 4h)
    regime_sma_bars: int = 150  # 150 bars = 25 days (fast for 4h)
    regime_hysteresis: float = 0.015  # 1.5%
    
    # Momentum gate
    min_momentum_1bar: float = 0.001  # positive last bar
    min_momentum_6bar: float = 0.005  # +0.5% over last day (6 bars)
    
    # Portfolio
    portfolio_top_n: int = 5
    vol_target: float = 0.015  # 1.5% per-bar vol target (lower at 4h)
    max_weight: float = 0.25
    
    # Model
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.04
    min_child_samples: int = 50  # more data available
    
    # Walk-forward
    train_start: str = "2021-06-01"
    val_start: str = "2024-06-01"
    retrain_every: int = 150  # retrain every 150 bars (~25 days)


def run_4h_selection(config: Selection4HConfig = None) -> dict:
    """Run the 4h mover selection engine."""
    if config is None:
        config = Selection4HConfig()
    
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")
    
    print("=" * 80)
    print("4H MOVER SELECTION ENGINE")
    print("=" * 80)
    
    # Load
    print("\n[1] Loading 4h chimera panel...")
    files = sorted(config.chimera_dir.glob("*_v51_chimera_4h_*.parquet"))
    if not files:
        # Try alternate naming
        files = sorted(config.chimera_dir.glob("*_v51_chimera_*.parquet"))
    
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
        except Exception as e:
            pass
    
    panel = pl.concat(frames, how="diagonal").sort(["asset", "date"])
    n_assets = panel["asset"].n_unique()
    print(f"  {len(panel):,} rows, {n_assets} assets")
    
    # Features
    print("\n[2] Feature engineering...")
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns([
        pl.col("close").pct_change(1).over("asset").alias("ret_1bar"),
        pl.col("close").pct_change(6).over("asset").alias("ret_6bar"),   # 1 day
        pl.col("close").pct_change(12).over("asset").alias("ret_12bar"),  # 2 days
        pl.col("close").pct_change(42).over("asset").alias("ret_42bar"),  # 7 days
        pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1bar"),
        pl.col("close").pct_change(6).over("asset").abs().alias("abs_ret_6bar"),
        pl.col("close").pct_change(1).rolling_std(6).over("asset").alias("vol_6bar"),
        pl.col("close").pct_change(1).rolling_std(24).over("asset").alias("vol_24bar"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("range_pct"),
        (pl.col("volume") / pl.col("volume").rolling_mean(24).over("asset")).alias("vol_ratio"),
    ])
    
    # Cross-sectional ranks
    for col in ["ret_1bar", "ret_6bar", "abs_ret_1bar", "abs_ret_6bar",
                "vol_6bar", "vol_24bar", "range_pct", "vol_ratio"]:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") /
                 pl.col(col).count().over("date")).alias(f"xrank_{col}")
            )
    
    # Label: top-K positive movers this bar
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return")
    )
    panel = panel.with_columns(
        pl.when(pl.col("fwd_return") >= config.min_return_threshold)
        .then(pl.col("fwd_return"))
        .otherwise(pl.lit(-999.0))
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("pos_rank")
    )
    panel = panel.with_columns(
        ((pl.col("pos_rank") <= config.top_k) &
         (pl.col("fwd_return") >= config.min_return_threshold))
        .cast(pl.Int8)
        .alias("is_top_mover")
    )
    
    # Regime (BTC 4h)
    print("\n[3] BTC regime at 4h...")
    btc_close = btc_data["close"].to_numpy()
    sma = np.convolve(btc_close, np.ones(config.regime_sma_bars)/config.regime_sma_bars, "full")[:len(btc_close)]
    sma[:config.regime_sma_bars-1] = np.nan
    regime = np.zeros(len(btc_close), dtype=int)
    state = 0
    for i in range(config.regime_sma_bars, len(btc_close)):
        if np.isnan(sma[i]): continue
        if state == 0 and btc_close[i] > sma[i] * (1 + config.regime_hysteresis):
            state = 1
        elif state == 1 and btc_close[i] < sma[i] * (1 - config.regime_hysteresis):
            state = 0
        regime[i] = state
    
    btc_regime_df = btc_data.sort("date").with_columns(pl.Series("regime", regime))
    panel = panel.join(btc_regime_df.select(["date", "regime"]), on="date", how="left")
    bull_pct = float(regime[config.regime_sma_bars:].mean())
    print(f"  Bull: {bull_pct:.1%}")
    
    # Feature columns
    exclude = {"date", "asset", "open", "high", "low", "close", "volume",
               "fwd_return", "is_top_mover", "pos_rank", "regime"}
    feature_cols = sorted([c for c in panel.columns if c not in exclude and not c.startswith("target_")])
    print(f"  Features: {len(feature_cols)}")
    
    # Walk-forward
    print("\n[4] Walk-forward (4h bars)...")
    panel = panel.filter(pl.col("is_top_mover").is_not_null() & pl.col("fwd_return").is_not_null())
    
    val_start = date.fromisoformat(config.val_start)
    train_start = date.fromisoformat(config.train_start)
    all_dates = sorted(panel.filter(pl.col("date") >= val_start)["date"].unique().to_list())
    print(f"  Bars to predict: {len(all_dates)}")
    
    model = None
    daily_results = []
    bar_results = []
    
    for i, pred_date in enumerate(all_dates):
        # Regime check
        reg_row = btc_regime_df.filter(pl.col("date") == pred_date)
        if len(reg_row) == 0 or int(reg_row["regime"][0]) == 0:
            bar_results.append({"date": pred_date, "return": 0.0, "regime": "bear", "n": 0})
            continue
        
        # Retrain
        if model is None or (i % config.retrain_every == 0):
            train_df = panel.filter((pl.col("date") >= train_start) & (pl.col("date") < pred_date))
            if len(train_df) < 5000:
                bar_results.append({"date": pred_date, "return": 0.0, "regime": "skip", "n": 0})
                continue
            X = np.nan_to_num(train_df.select(feature_cols).to_numpy(), nan=0.0)
            y = train_df["is_top_mover"].to_numpy()
            td = lgb.Dataset(X, label=y)
            params = {"objective": "binary", "metric": "auc", "max_depth": config.max_depth,
                     "learning_rate": config.learning_rate, "min_child_samples": config.min_child_samples,
                     "verbosity": -1, "num_threads": 4, "is_unbalance": True,
                     "feature_fraction": 0.65, "bagging_fraction": 0.75, "bagging_freq": 5}
            model = lgb.train(params, td, num_boost_round=config.n_estimators)
        
        # Predict
        pred_df = panel.filter(pl.col("date") == pred_date)
        if len(pred_df) == 0:
            continue
        
        X_pred = np.nan_to_num(pred_df.select(feature_cols).to_numpy(), nan=0.0)
        scores = model.predict(X_pred)
        
        # Momentum gate
        ret_1bar = pred_df["ret_1bar"].to_numpy()
        ret_6bar = pred_df["ret_6bar"].to_numpy()
        
        mask = ((np.nan_to_num(ret_1bar) >= config.min_momentum_1bar) &
                (np.nan_to_num(ret_6bar) >= config.min_momentum_6bar) &
                (scores > 0.12))
        
        candidates_idx = np.where(mask)[0]
        if len(candidates_idx) == 0:
            bar_results.append({"date": pred_date, "return": 0.0, "regime": "bull_no_cand", "n": 0})
            continue
        
        # Top-N by score
        cand_scores = scores[candidates_idx]
        top_k = min(config.portfolio_top_n, len(candidates_idx))
        top_local = np.argsort(cand_scores)[::-1][:top_k]
        selected_idx = candidates_idx[top_local]
        
        # Vol-targeted weights
        vol_vals = pred_df["vol_6bar"].to_numpy()[selected_idx]
        vol_vals = np.clip(np.nan_to_num(vol_vals, nan=0.02), 0.003, 0.15)
        weights = config.vol_target / vol_vals
        weights = np.minimum(weights, config.max_weight)
        weights = weights / weights.sum()
        
        # Returns (NO trailing stop — honest)
        fwd_rets = pred_df["fwd_return"].to_numpy()[selected_idx]
        fwd_rets = np.nan_to_num(fwd_rets, nan=0.0)
        port_return = float(np.dot(weights, fwd_rets))
        
        bar_results.append({"date": pred_date, "return": port_return, "regime": "bull", "n": len(selected_idx)})
        
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(all_dates)} bars...")
    
    # Evaluate
    print("\n[5] Evaluation...")
    rets = np.array([r["return"] for r in bar_results])
    n_bars = len(rets)
    
    compound = float(np.prod(1 + rets) - 1)
    mean_per_bar = float(np.mean(rets))
    mean_per_day = mean_per_bar * 6  # 6 bars per day
    std_bar = float(np.std(rets))
    sharpe = mean_per_bar / std_bar * np.sqrt(365 * 6) if std_bar > 0 else 0
    win_rate = float(np.mean(rets > 0))
    
    cum = np.cumprod(1 + rets)
    max_dd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))
    
    bull_mask = np.array([r["regime"] == "bull" for r in bar_results])
    bull_rets = rets[bull_mask]
    bull_mean = float(np.mean(bull_rets)) if len(bull_rets) > 0 else 0
    n_bull = int(bull_mask.sum())
    n_bear = int(np.array([r["regime"] == "bear" for r in bar_results]).sum())
    
    # Per-day aggregation
    n_days_approx = n_bars / 6
    
    print("\n" + "=" * 80)
    print("4H MOVER SELECTION — RESULTS")
    print("=" * 80)
    print(f"\n  BARS: {n_bars} (bull: {n_bull}, bear/cash: {n_bear})")
    print(f"  APPROX DAYS: {n_days_approx:.0f}")
    print(f"\n  PER-BAR (4h):")
    print(f"    Mean: {mean_per_bar*100:.4f}%")
    print(f"    Compound: {compound*100:.2f}%")
    print(f"    Win rate: {win_rate:.3f}")
    print(f"    Sharpe (ann): {sharpe:.2f}")
    print(f"    Max DD: {max_dd*100:.1f}%")
    print(f"\n  PER-DAY (x6 bars):")
    print(f"    Mean daily: {mean_per_day*100:.4f}%")
    print(f"\n  BULL REGIME:")
    print(f"    Mean per bar: {bull_mean*100:.4f}%")
    print(f"    Mean per day (x6): {bull_mean*6*100:.4f}%")
    
    print(f"\n  VERDICT:")
    if mean_per_day >= 0.01:
        print(f"    ** NORTH STAR 1%/day: ACHIEVED ({mean_per_day*100:.3f}%/day) **")
    elif mean_per_day >= 0.005:
        print(f"    ** Promising: {mean_per_day*100:.3f}%/day (0.5%+) **")
    elif mean_per_day > 0:
        print(f"    ** Positive: {mean_per_day*100:.4f}%/day — need more signal **")
    else:
        print(f"    ** Negative at 4h — same conclusion as daily **")
    
    # Save
    output = {
        "version": "4h_selection_v1",
        "n_bars": n_bars, "n_bull_bars": n_bull,
        "per_bar": {"mean_pct": mean_per_bar * 100, "compound": compound, "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate},
        "per_day": {"mean_pct": mean_per_day * 100},
        "bull_only": {"mean_per_bar_pct": bull_mean * 100, "mean_per_day_pct": bull_mean * 6 * 100},
    }
    output_dir = Path("crypto/runs/strat")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"mover_selection_4h_v1_{ts}.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved.")
    
    return output


if __name__ == "__main__":
    run_4h_selection()
