"""Mover Selection V2 — iterating on V1 findings.

V1 Diagnosis:
- Has genuine selection skill (hit rate 3x random)
- But raw capture is insufficient (0.15%/day, compounds negative)
- Too many "big movers DOWN" slip through (direction-agnostic selection)
- High variance kills compounding (MaxDD -88%)

V2 Changes:
1. ASYMMETRIC TARGET: predict "top-K POSITIVE movers" more strictly
   (require >2% return, not just top-K rank)
2. MOMENTUM GATE: only consider assets with positive trailing momentum
   (crude but effective pre-filter that respects the "trend > reversion" finding)
3. VOL-TARGETING: scale portfolio weights inversely to predicted vol
4. MINIMUM CONFIDENCE: only select assets above a score threshold
5. BROADER TOP-K: select top-10, not top-5 (more diversification)
6. FEATURE ENGINEERING: add cross-sectional volatility rank features
   that specifically target the MAGNITUDE signal (D55 confirmed: vol is
   predictable, direction isn't)

The key insight from the dead-list: vol-targeting is the "zero dead-list risk"
overlay that should go on everything. And momentum-filtering aligns with
D52/D53: "continuation dominates" — ride the trend, don't fight it.
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
class SelectionV2Config:
    chimera_dir: Path = Path("crypto/data/processed/chimera/1d")
    
    # Label: top-K positive movers (stricter than V1)
    top_k: int = 10
    min_return_threshold: float = 0.02  # must be +2% to be labeled positive
    
    # Momentum gate
    momentum_gate: bool = True
    momentum_lookback: int = 7  # 7-day trailing return must be > 0
    
    # Vol-targeting
    vol_target: bool = True
    vol_lookback: int = 14  # days for vol estimation
    vol_target_level: float = 0.04  # target daily vol (4%)
    
    # Score threshold (minimum confidence to select)
    min_score_threshold: float = 0.15  # model must be >15% confident
    
    # Portfolio
    portfolio_top_n: int = 7  # select top-7 (more diversification than V1)
    max_position_weight: float = 0.25  # cap any single position
    
    # Model
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.03
    min_child_samples: int = 30
    
    # Walk-forward
    train_start: str = "2021-01-01"
    val_start: str = "2024-06-01"
    min_train_days: int = 400
    retrain_every: int = 20  # retrain every 20 days (more adaptive)
    
    # Evaluation
    random_seeds: int = 50


def load_and_prepare(config: SelectionV2Config) -> pl.DataFrame:
    """Load panel and add all features."""
    files = sorted(config.chimera_dir.glob("*_v51_chimera_1d_*.parquet"))
    
    frames = []
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
            # Keep relevant columns
            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            for c in df.columns:
                if any(c.startswith(p) for p in ("norm_", "fund_", "liq_", "bs_", 
                                                   "hbr_", "rv_", "bd_", "xrel_")):
                    keep_cols.append(c)
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df.select(keep_cols)
            df = df.with_columns(pl.lit(sym).alias("asset"))
            frames.append(df)
        except Exception:
            pass
    
    panel = pl.concat(frames, how="diagonal").sort(["asset", "date"])
    
    # === MOMENTUM FEATURES (the continuation signal) ===
    panel = panel.with_columns([
        pl.col("close").pct_change(1).over("asset").alias("ret_1d"),
        pl.col("close").pct_change(3).over("asset").alias("ret_3d"),
        pl.col("close").pct_change(7).over("asset").alias("ret_7d"),
        pl.col("close").pct_change(14).over("asset").alias("ret_14d"),
        pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1d"),
        pl.col("close").pct_change(3).over("asset").abs().alias("abs_ret_3d"),
        pl.col("close").pct_change(7).over("asset").abs().alias("abs_ret_7d"),
    ])
    
    # === VOLATILITY FEATURES (the predictable channel) ===
    panel = panel.with_columns([
        pl.col("close").pct_change(1).rolling_std(7).over("asset").alias("vol_7d"),
        pl.col("close").pct_change(1).rolling_std(14).over("asset").alias("vol_14d"),
        pl.col("close").pct_change(1).rolling_std(30).over("asset").alias("vol_30d"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("range_pct"),
        # Vol expansion signal (today's range vs trailing average)
        ((pl.col("high") - pl.col("low")) / pl.col("open")).rolling_mean(7).over("asset").alias("avg_range_7d"),
    ])
    
    # Vol expansion ratio
    panel = panel.with_columns(
        (pl.col("range_pct") / pl.col("avg_range_7d").clip(lower_bound=0.001)).alias("vol_expansion")
    )
    
    # === CROSS-SECTIONAL FEATURES (relative position) ===
    rank_cols = ["ret_1d", "ret_3d", "ret_7d", "abs_ret_1d", "abs_ret_3d",
                 "vol_7d", "vol_14d", "range_pct", "vol_expansion"]
    for col in rank_cols:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") /
                 pl.col(col).count().over("date")).alias(f"xrank_{col}")
            )
    
    # === LABEL: forward positive mover ===
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return_1d")
    )
    
    # Top-K POSITIVE movers per day (must exceed threshold)
    panel = panel.with_columns(
        pl.when(pl.col("fwd_return_1d") >= config.min_return_threshold)
        .then(pl.col("fwd_return_1d"))
        .otherwise(pl.lit(-999.0))
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("pos_return_rank")
    )
    
    panel = panel.with_columns(
        ((pl.col("pos_return_rank") <= config.top_k) &
         (pl.col("fwd_return_1d") >= config.min_return_threshold))
        .cast(pl.Int8)
        .alias("is_top_mover")
    )
    
    return panel


def get_feature_cols(panel: pl.DataFrame) -> list[str]:
    """Get feature columns (everything except targets, ids, and dates)."""
    exclude = {"date", "asset", "open", "high", "low", "close", "volume",
               "fwd_return_1d", "is_top_mover", "pos_return_rank", "return_rank"}
    return sorted([c for c in panel.columns if c not in exclude and not c.startswith("target_")])


def run_v2(config: Optional[SelectionV2Config] = None) -> dict:
    """Run the V2 selection engine."""
    if config is None:
        config = SelectionV2Config()
    
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")
    
    print("=" * 80)
    print("MOVER SELECTION V2")
    print("=" * 80)
    
    # Load and prepare
    print("\n[1/4] Loading and preparing data...")
    panel = load_and_prepare(config)
    print(f"  Panel: {len(panel):,} rows, {panel['asset'].n_unique()} assets")
    
    feature_cols = get_feature_cols(panel)
    print(f"  Features: {len(feature_cols)}")
    
    # Filter to valid rows
    panel = panel.filter(pl.col("is_top_mover").is_not_null() & pl.col("fwd_return_1d").is_not_null())
    
    # Walk-forward
    print("\n[2/4] Walk-forward prediction...")
    val_start = date.fromisoformat(config.val_start)
    train_start = date.fromisoformat(config.train_start)
    all_dates = sorted(panel.filter(pl.col("date") >= val_start)["date"].unique().to_list())
    print(f"  Predicting {len(all_dates)} days...")
    
    results_rows = []
    model = None
    
    for i, pred_date in enumerate(all_dates):
        # Retrain check
        if model is None or (i % config.retrain_every == 0):
            train_df = panel.filter(
                (pl.col("date") >= train_start) & (pl.col("date") < pred_date)
            )
            if len(train_df) < config.min_train_days * 5:
                continue
            
            X_train = np.nan_to_num(train_df.select(feature_cols).to_numpy(), nan=0.0)
            y_train = train_df["is_top_mover"].to_numpy()
            
            params = {
                "objective": "binary",
                "metric": "auc",
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "min_child_samples": config.min_child_samples,
                "verbosity": -1,
                "num_threads": 4,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=config.n_estimators)
        
        # Predict
        pred_df = panel.filter(pl.col("date") == pred_date)
        if len(pred_df) == 0:
            continue
        
        X_pred = np.nan_to_num(pred_df.select(feature_cols).to_numpy(), nan=0.0)
        scores = model.predict(X_pred)
        
        for j, row in enumerate(pred_df.iter_rows(named=True)):
            results_rows.append({
                "date": row["date"],
                "asset": row["asset"],
                "score": float(scores[j]),
                "is_top_mover": int(row["is_top_mover"]),
                "fwd_return_1d": float(row["fwd_return_1d"]),
                "ret_7d": float(row["ret_7d"]) if row.get("ret_7d") is not None else 0.0,
                "vol_14d": float(row["vol_14d"]) if row.get("vol_14d") is not None else 0.05,
            })
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_dates)} days done...")
    
    predictions = pl.DataFrame(results_rows)
    print(f"  Predictions: {len(predictions):,} rows")
    
    # === PORTFOLIO CONSTRUCTION WITH GATES ===
    print("\n[3/4] Portfolio construction (momentum gate + vol-targeting)...")
    
    daily_portfolios = []
    
    for pred_date in predictions["date"].unique().sort().to_list():
        day_data = predictions.filter(pl.col("date") == pred_date)
        
        # Gate 1: Momentum filter (only positive trailing 7d)
        if config.momentum_gate:
            day_data = day_data.filter(pl.col("ret_7d") > 0)
        
        # Gate 2: Minimum score threshold
        day_data = day_data.filter(pl.col("score") >= config.min_score_threshold)
        
        if len(day_data) == 0:
            # No candidates pass gates — stay in cash (0 return)
            daily_portfolios.append({"date": pred_date, "port_return": 0.0,
                                    "n_selected": 0, "n_hits": 0})
            continue
        
        # Select top-N by score
        day_data = day_data.sort("score", descending=True).head(config.portfolio_top_n)
        
        # Vol-targeting weights
        if config.vol_target:
            vols = day_data["vol_14d"].to_numpy()
            vols = np.clip(vols, 0.005, 0.30)  # clip extreme vols
            inv_vol = 1.0 / vols
            weights = inv_vol / inv_vol.sum()
            # Cap max position
            weights = np.minimum(weights, config.max_position_weight)
            weights = weights / weights.sum()  # re-normalize
        else:
            weights = np.ones(len(day_data)) / len(day_data)
        
        returns = day_data["fwd_return_1d"].to_numpy()
        port_return = float(np.dot(weights, returns))
        n_hits = int(day_data["is_top_mover"].sum())
        
        daily_portfolios.append({
            "date": pred_date,
            "port_return": port_return,
            "n_selected": len(day_data),
            "n_hits": n_hits,
        })
    
    daily_df = pl.DataFrame(daily_portfolios).sort("date")
    port_rets = daily_df["port_return"].to_numpy()
    
    # === EVALUATION ===
    print("\n[4/4] Evaluation...")
    
    n_days = len(daily_df)
    compound = float(np.prod(1 + port_rets) - 1)
    mean_daily = float(np.mean(port_rets))
    std_daily = float(np.std(port_rets))
    sharpe = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0
    win_rate = float(np.mean(port_rets > 0))
    
    # Max drawdown
    cumulative = np.cumprod(1 + port_rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    
    # Cash days (stayed out)
    n_cash = int((daily_df["n_selected"] == 0).sum())
    
    # Hit rate
    total_selected = daily_df["n_selected"].sum()
    total_hits = daily_df["n_hits"].sum()
    hit_rate = float(total_hits / total_selected) if total_selected > 0 else 0
    
    # Random baseline
    rng = np.random.default_rng(42)
    random_means = []
    for seed in range(config.random_seeds):
        r = np.random.default_rng(seed)
        rand_rets = []
        for _, row in enumerate(daily_df.iter_rows(named=True)):
            if row["n_selected"] == 0:
                rand_rets.append(0.0)
                continue
            day_preds = predictions.filter(pl.col("date") == row["date"])
            if config.momentum_gate:
                day_preds = day_preds.filter(pl.col("ret_7d") > 0)
            if len(day_preds) < config.portfolio_top_n:
                rand_rets.append(0.0)
                continue
            idx = r.choice(len(day_preds), size=min(config.portfolio_top_n, len(day_preds)), replace=False)
            rand_rets.append(float(day_preds["fwd_return_1d"].to_numpy()[idx].mean()))
        random_means.append(float(np.mean(rand_rets)))
    
    random_mean = float(np.mean(random_means))
    
    # Feature importance
    if model is not None:
        importance = model.feature_importance(importance_type="gain")
        top_features = sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:15]
    else:
        top_features = []
    
    # Report
    print("\n" + "=" * 80)
    print("V2 RESULTS")
    print("=" * 80)
    print(f"\n  MODEL PORTFOLIO (top-{config.portfolio_top_n}, momentum-gated, vol-targeted):")
    print(f"    Days traded: {n_days} (cash days: {n_cash})")
    print(f"    Mean daily: {mean_daily*100:.4f}%")
    print(f"    Compound: {compound*100:.2f}%")
    print(f"    Sharpe (ann): {sharpe:.2f}")
    print(f"    Hit rate: {hit_rate:.3f}")
    print(f"    Max DD: {max_dd*100:.1f}%")
    print(f"    Win rate: {win_rate:.3f}")
    
    print(f"\n  RANDOM BASELINE (momentum-gated, same N):")
    print(f"    Mean daily: {random_mean*100:.4f}%")
    
    print(f"\n  LIFT:")
    print(f"    vs random: {(mean_daily - random_mean)*100:.4f}%/day")
    print(f"    Annualized lift: {(mean_daily - random_mean)*365*100:.1f}%/yr")
    
    if top_features:
        print(f"\n  TOP FEATURES (by gain):")
        for feat, imp in top_features[:10]:
            print(f"    {feat:<35} {imp:.0f}")
    
    # Verdict
    print(f"\n  VERDICT:")
    if mean_daily >= 0.01:
        print(f"    ** NORTH STAR FLOOR (1%/day): ACHIEVED **")
    elif mean_daily >= 0.005:
        print(f"    ** Promising: >0.5%/day **")
    elif mean_daily > random_mean * 1.5:
        print(f"    ** Genuine skill: {(mean_daily/random_mean - 1)*100:.0f}% lift over random **")
    else:
        print(f"    ** Insufficient lift over random **")
    
    output = {
        "version": "v2",
        "config": {
            "top_k": config.top_k,
            "portfolio_top_n": config.portfolio_top_n,
            "momentum_gate": config.momentum_gate,
            "vol_target": config.vol_target,
            "min_score_threshold": config.min_score_threshold,
        },
        "model": {
            "n_days": n_days, "n_cash_days": n_cash,
            "compound_return": compound, "mean_daily_return": mean_daily,
            "std_daily_return": std_daily, "sharpe_annual": sharpe,
            "hit_rate": hit_rate, "max_drawdown": max_dd, "win_rate": win_rate,
            "mean_daily_pct": mean_daily * 100,
        },
        "random_baseline": {"mean_daily_pct": random_mean * 100},
        "lift_vs_random_pct_day": (mean_daily - random_mean) * 100,
        "top_features": top_features[:15] if top_features else [],
    }
    
    # Save
    output_dir = Path("crypto/runs/strat")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mover_selection_v2_{ts}.json"
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, date):
            return obj.isoformat()
        return obj
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Saved: {output_file}")
    
    return output


if __name__ == "__main__":
    run_v2()
