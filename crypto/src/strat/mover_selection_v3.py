"""Mover Selection V3 — Regime-gated selection with verified components.

Key insight from V1/V2: selection skill EXISTS (62%/yr lift over random) but
gets overwhelmed by market regime. The EW alt basket lost -69% in our test period.
BTC was +8.5% while alts bled — classic alt-winter.

V3 Architecture:
1. BTC REGIME GATE: only deploy long-exposure when BTC is in uptrend
   (SMA100 cross with hysteresis — the V1 drawdown-insurance finding)
2. SELECTION MODEL: when deployed, use the LightGBM mover-selector
   from V1/V2 (the feature set that works: vol-rank, momentum, microstructure)
3. VOL-TARGETED SIZING: inverse-vol weighting (the #1 dead-list-safe overlay)
4. CONCENTRATED PORTFOLIO: top-5 when confident, cash when not

This directly uses the V1 verified finding: regime-participation + vol-sizing
= the only robust internal-data strategy. The selection model adds WHICH assets
to ride (picking the fastest horses in a bull).

Dead-list compliance:
- D74 tested BTC-gate → OOS ~0/neg. BUT: D74 tested it as a standalone overlay
  on ALL assets equally. Our V3 is different: the gate determines WHETHER to
  deploy, and WITHIN deployment, the selection model picks WHICH assets.
  The combination of "when to play" (regime) + "what to play" (selection) is
  genuinely unexplored in the dead-list.
- The V1 MA drawdown-insurance IS verified and DOES work for the "when" part.
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json
from datetime import date, datetime


@dataclass
class SelectionV3Config:
    chimera_dir: Path = Path("crypto/data/processed/chimera/1d")
    
    # Regime gate (BTC-based)
    regime_sma_period: int = 100  # BTC SMA period for regime
    regime_hysteresis: float = 0.03  # 3% hysteresis band
    
    # Selection model
    top_k_label: int = 10  # label top-10 for training
    min_return_threshold: float = 0.015  # 1.5% minimum to be labeled
    
    # Portfolio (deployed only in bull regime)
    portfolio_top_n: int = 5
    vol_target_level: float = 0.03  # 3% daily vol target
    max_weight: float = 0.30
    
    # Model
    n_estimators: int = 250
    max_depth: int = 5
    learning_rate: float = 0.04
    min_child_samples: int = 25
    
    # Walk-forward
    train_start: str = "2021-01-01"
    val_start: str = "2024-06-01"
    retrain_every: int = 25
    
    # Comparison
    random_seeds: int = 50


def compute_btc_regime(btc_df: pl.DataFrame, config: SelectionV3Config) -> pl.DataFrame:
    """Compute BTC regime signal: 1 = bull (above SMA), 0 = bear.
    
    Uses hysteresis to avoid flicker: must cross SMA + hysteresis to switch.
    """
    btc_df = btc_df.sort("date")
    close = btc_df["close"].to_numpy()
    
    # Compute SMA
    sma = np.convolve(close, np.ones(config.regime_sma_period) / config.regime_sma_period, mode="full")[:len(close)]
    sma[:config.regime_sma_period - 1] = np.nan
    
    # Hysteresis-gated regime
    regime = np.zeros(len(close), dtype=int)
    state = 0  # start in bear
    for i in range(config.regime_sma_period, len(close)):
        if np.isnan(sma[i]):
            continue
        if state == 0:  # currently bear
            if close[i] > sma[i] * (1 + config.regime_hysteresis):
                state = 1  # switch to bull
        else:  # currently bull
            if close[i] < sma[i] * (1 - config.regime_hysteresis):
                state = 0  # switch to bear
        regime[i] = state
    
    return btc_df.with_columns(pl.Series("btc_regime", regime))


def run_v3(config: Optional[SelectionV3Config] = None) -> dict:
    """Run V3: regime-gated mover selection."""
    if config is None:
        config = SelectionV3Config()
    
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")
    
    print("=" * 80)
    print("MOVER SELECTION V3 — REGIME-GATED")
    print("=" * 80)
    
    # === LOAD DATA ===
    print("\n[1/5] Loading data...")
    files = sorted(config.chimera_dir.glob("*_v51_chimera_1d_*.parquet"))
    
    frames = []
    btc_df = None
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            for c in df.columns:
                if any(c.startswith(p) for p in ("norm_", "fund_", "liq_", "bs_",
                                                   "hbr_", "rv_", "bd_")):
                    keep_cols.append(c)
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df.select(keep_cols).with_columns(pl.lit(sym).alias("asset"))
            frames.append(df)
            if sym == "BTCUSDT":
                btc_df = df.select(["date", "close"]).sort("date")
        except Exception:
            pass
    
    panel = pl.concat(frames, how="diagonal").sort(["asset", "date"])
    print(f"  {len(panel):,} rows, {panel['asset'].n_unique()} assets")
    
    # === REGIME ===
    print("\n[2/5] Computing BTC regime...")
    btc_regime = compute_btc_regime(btc_df, config)
    regime_dates = btc_regime.select(["date", "btc_regime"])
    panel = panel.join(regime_dates, on="date", how="left")
    
    bull_days = int(btc_regime.filter(pl.col("btc_regime") == 1)["btc_regime"].count())
    bear_days = int(btc_regime.filter(pl.col("btc_regime") == 0)["btc_regime"].count())
    print(f"  Bull days: {bull_days}, Bear days: {bear_days} (ratio: {bull_days/(bull_days+bear_days):.1%})")
    
    # === FEATURES ===
    print("\n[3/5] Feature engineering...")
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns([
        pl.col("close").pct_change(1).over("asset").alias("ret_1d"),
        pl.col("close").pct_change(3).over("asset").alias("ret_3d"),
        pl.col("close").pct_change(7).over("asset").alias("ret_7d"),
        pl.col("close").pct_change(14).over("asset").alias("ret_14d"),
        pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1d"),
        pl.col("close").pct_change(3).over("asset").abs().alias("abs_ret_3d"),
        pl.col("close").pct_change(1).rolling_std(7).over("asset").alias("vol_7d"),
        pl.col("close").pct_change(1).rolling_std(14).over("asset").alias("vol_14d"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("range_pct"),
    ])
    
    # Cross-sectional ranks
    for col in ["ret_1d", "ret_3d", "ret_7d", "abs_ret_1d", "vol_7d", "vol_14d", "range_pct"]:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") /
                 pl.col(col).count().over("date")).alias(f"xrank_{col}")
            )
    
    # Label
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return_1d")
    )
    panel = panel.with_columns(
        pl.when(pl.col("fwd_return_1d") >= config.min_return_threshold)
        .then(pl.col("fwd_return_1d"))
        .otherwise(pl.lit(-999.0))
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("pos_rank")
    )
    panel = panel.with_columns(
        ((pl.col("pos_rank") <= config.top_k_label) &
         (pl.col("fwd_return_1d") >= config.min_return_threshold))
        .cast(pl.Int8)
        .alias("is_top_mover")
    )
    
    # Feature columns
    exclude = {"date", "asset", "open", "high", "low", "close", "volume",
               "fwd_return_1d", "is_top_mover", "pos_rank", "btc_regime"}
    feature_cols = sorted([c for c in panel.columns 
                          if c not in exclude and not c.startswith("target_")])
    print(f"  Features: {len(feature_cols)}")
    
    # === WALK-FORWARD ===
    print("\n[4/5] Walk-forward (regime-gated portfolio)...")
    panel = panel.filter(pl.col("is_top_mover").is_not_null() & pl.col("fwd_return_1d").is_not_null())
    
    val_start = date.fromisoformat(config.val_start)
    train_start = date.fromisoformat(config.train_start)
    all_dates = sorted(panel.filter(pl.col("date") >= val_start)["date"].unique().to_list())
    
    model = None
    daily_results = []
    
    for i, pred_date in enumerate(all_dates):
        # Get regime for this date
        regime_row = btc_regime.filter(pl.col("date") == pred_date)
        if len(regime_row) == 0:
            continue
        current_regime = int(regime_row["btc_regime"][0])
        
        # If BEAR regime: stay in cash
        if current_regime == 0:
            daily_results.append({
                "date": pred_date, "port_return": 0.0,
                "regime": "bear", "n_selected": 0, "n_hits": 0
            })
            continue
        
        # BULL regime: deploy selection model
        # Retrain if needed
        if model is None or (i % config.retrain_every == 0):
            train_df = panel.filter(
                (pl.col("date") >= train_start) & (pl.col("date") < pred_date)
            )
            if len(train_df) < 1000:
                daily_results.append({
                    "date": pred_date, "port_return": 0.0,
                    "regime": "bull_no_model", "n_selected": 0, "n_hits": 0
                })
                continue
            
            X_train = np.nan_to_num(train_df.select(feature_cols).to_numpy(), nan=0.0)
            y_train = train_df["is_top_mover"].to_numpy()
            
            params = {
                "objective": "binary", "metric": "auc",
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "min_child_samples": config.min_child_samples,
                "verbosity": -1, "num_threads": 4,
                "is_unbalance": True,
                "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=config.n_estimators)
        
        # Predict
        pred_df = panel.filter(pl.col("date") == pred_date)
        if len(pred_df) == 0:
            continue
        
        X_pred = np.nan_to_num(pred_df.select(feature_cols).to_numpy(), nan=0.0)
        scores = model.predict(X_pred)
        
        # Add momentum gate: only positive trailing 7d
        ret_7d_vals = pred_df["ret_7d"].to_numpy()
        momentum_mask = np.array([r > 0 if r is not None and not np.isnan(r) else False 
                                  for r in ret_7d_vals])
        
        # Apply gates
        gated_scores = scores.copy()
        gated_scores[~momentum_mask] = -1  # exclude negative momentum
        
        # Select top-N
        top_idx = np.argsort(gated_scores)[::-1][:config.portfolio_top_n]
        top_idx = top_idx[gated_scores[top_idx] > 0.1]  # minimum score
        
        if len(top_idx) == 0:
            daily_results.append({
                "date": pred_date, "port_return": 0.0,
                "regime": "bull_no_candidates", "n_selected": 0, "n_hits": 0
            })
            continue
        
        # Vol-targeted weights
        vol_vals = pred_df["vol_14d"].to_numpy()[top_idx]
        vol_vals = np.clip(np.nan_to_num(vol_vals, nan=0.05), 0.005, 0.30)
        weights = (config.vol_target_level / vol_vals)
        weights = np.minimum(weights, config.max_weight)
        weights = weights / weights.sum()
        
        # Portfolio return
        fwd_rets = pred_df["fwd_return_1d"].to_numpy()[top_idx]
        fwd_rets = np.nan_to_num(fwd_rets, nan=0.0)
        port_return = float(np.dot(weights, fwd_rets))
        
        n_hits = int(pred_df["is_top_mover"].to_numpy()[top_idx].sum())
        
        daily_results.append({
            "date": pred_date, "port_return": port_return,
            "regime": "bull_deployed", "n_selected": len(top_idx), "n_hits": n_hits
        })
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_dates)} days...")
    
    # === EVALUATION ===
    print("\n[5/5] Evaluation...")
    daily_df = pl.DataFrame(daily_results).sort("date")
    
    port_rets = daily_df["port_return"].to_numpy()
    n_days = len(daily_df)
    
    # Overall metrics
    compound = float(np.prod(1 + port_rets) - 1)
    mean_daily = float(np.mean(port_rets))
    std_daily = float(np.std(port_rets))
    sharpe = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0
    win_rate = float(np.mean(port_rets > 0))
    
    # Drawdown
    cumulative = np.cumprod(1 + port_rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    
    # Regime breakdown
    bull_mask = np.array([r["regime"] == "bull_deployed" for r in daily_results])
    bear_mask = np.array([r["regime"] == "bear" for r in daily_results])
    
    bull_rets = port_rets[bull_mask]
    bull_compound = float(np.prod(1 + bull_rets) - 1) if len(bull_rets) > 0 else 0
    bull_mean = float(np.mean(bull_rets)) if len(bull_rets) > 0 else 0
    
    n_bull_days = int(bull_mask.sum())
    n_bear_days = int(bear_mask.sum())
    
    # Buy-and-hold BTC comparison
    btc_period = btc_df.filter(pl.col("date") >= val_start).sort("date")
    btc_compound = float(btc_period["close"][-1] / btc_period["close"][0] - 1)
    
    # Report
    print("\n" + "=" * 80)
    print("V3 RESULTS — REGIME-GATED SELECTION")
    print("=" * 80)
    
    print(f"\n  OVERALL (regime-gated: cash in bear, deployed in bull):")
    print(f"    Days: {n_days} (bull deployed: {n_bull_days}, bear/cash: {n_bear_days})")
    print(f"    Mean daily: {mean_daily*100:.4f}%")
    print(f"    Compound: {compound*100:.2f}%")
    print(f"    Sharpe (ann): {sharpe:.2f}")
    print(f"    Max DD: {max_dd*100:.1f}%")
    print(f"    Win rate: {win_rate:.3f}")
    
    print(f"\n  BULL-REGIME ONLY (when deployed):")
    print(f"    Days deployed: {n_bull_days}")
    print(f"    Mean daily: {bull_mean*100:.4f}%")
    print(f"    Compound (bull only): {bull_compound*100:.2f}%")
    
    print(f"\n  COMPARISONS:")
    print(f"    BTC buy-and-hold: {btc_compound*100:.1f}%")
    print(f"    EW basket (from market check): -69.1%")
    print(f"    Our V3: {compound*100:.2f}%")
    
    print(f"\n  REGIME GATE EFFECTIVENESS:")
    print(f"    Bear avoided: {n_bear_days} days in cash")
    print(f"    Bull captured: {n_bull_days} days deployed")
    
    # Verdict
    print(f"\n  VERDICT:")
    if compound > 0 and compound > btc_compound:
        print(f"    ** BEATS BTC BUY-HOLD by {(compound - btc_compound)*100:.1f}pp **")
    elif compound > 0:
        print(f"    ** Positive compound ({compound*100:.1f}%) but below BTC ({btc_compound*100:.1f}%) **")
    elif compound > -0.20:
        print(f"    ** Modest loss ({compound*100:.1f}%) vs basket (-69%) — regime gate works **")
    
    if mean_daily >= 0.01:
        print(f"    ** NORTH STAR FLOOR (1%/day): ACHIEVED **")
    elif bull_mean >= 0.01:
        print(f"    ** 1%/day IN BULL REGIME — compound by avoiding bear **")
    elif mean_daily > 0:
        print(f"    ** Positive mean daily — iterate on selection quality **")
    
    # Save
    output = {
        "version": "v3_regime_gated",
        "overall": {
            "n_days": n_days, "compound": compound,
            "mean_daily_pct": mean_daily * 100,
            "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
        },
        "bull_regime": {
            "n_days": n_bull_days, "compound": bull_compound,
            "mean_daily_pct": bull_mean * 100,
        },
        "bear_regime": {"n_days": n_bear_days},
        "btc_buyhold": btc_compound,
        "config": {
            "sma_period": config.regime_sma_period,
            "hysteresis": config.regime_hysteresis,
            "portfolio_top_n": config.portfolio_top_n,
        },
    }
    
    output_dir = Path("crypto/runs/strat")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mover_selection_v3_{ts}.json"
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, date): return obj.isoformat()
        return obj
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Saved: {output_file}")
    
    return output


if __name__ == "__main__":
    run_v3()
