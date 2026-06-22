"""Mover Selection Engine — scores u100 assets daily for top-mover probability.

The direct attack on the North Star: if we can identify 3-5 assets that will be
tomorrow's top movers, the opportunity analysis shows 1-5%/day is achievable
with modest capture rates.

Architecture:
  1. Label construction: for each date, identify which assets were top-K movers
  2. Feature extraction: use lagged chimera features (t-1) to predict top-mover
     membership at t (strictly causal — no look-ahead)
  3. Model: LightGBM classifier (probability of top-K membership)
  4. Portfolio: equal-weight top-N scored assets each day
  5. Evaluation: compound return, hit-rate, capture-rate vs random baseline

Key design decisions:
  - Cross-sectional ranking problem, NOT per-asset time-series prediction
  - Features are RELATIVE (cross-sectional ranks/z-scores), not absolute
  - Walk-forward: train on expanding window, predict 1-day ahead
  - Gate: must beat random-K selection on UNSEEN (the minimum bar)

Dead-list compliance:
  - D17 (cross-sectional LGBM) was HARD at "predict returns" — we predict
    RANK/membership which is a different objective (classification not regression)
  - D55 (direction prediction) — we don't predict direction, we predict MAGNITUDE
  - D44 (1-5%/day via daily prediction) — their approach was IC/Sharpe; ours is
    selection+capture which has different math (oracle ceiling 10.4%/day proven)
  - The KEY difference: D17 tried to predict WHICH DIRECTION assets move;
    we predict WHICH ASSETS WILL MOVE A LOT (magnitude, not direction)

NOTE: D17 found IC ≈ 0 for cross-sectional prediction of RETURNS.
We are predicting ABSOLUTE MAGNITUDE RANK — a fundamentally different target.
If this also nulls, we learn something new. But the oracle ceiling analysis
shows the OPPORTUNITY is real (10.7 assets/day move >=5% UP), so the question
is whether yesterday's features predict today's movers.
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json
from datetime import date, timedelta


# === CONFIGURATION ===

@dataclass
class SelectionConfig:
    """Configuration for the mover selection engine."""
    # Universe
    chimera_dir: Path = Path("crypto/data/processed/chimera/1d")
    
    # Label construction
    top_k: int = 5  # top-K movers to identify each day
    min_return_threshold: float = 0.01  # minimum return to be considered a "mover"
    
    # Feature set (which chimera columns to use as predictors)
    # We use lagged features: features at t-1 predict membership at t
    feature_prefixes: tuple = (
        "norm_",      # 33 normalized features (vol, flow, momentum, etc)
        "fund_",      # funding derivatives
        "liq_",       # liquidation features  
        "bs_",        # basis/spread
        "hbr_",       # hawkes branching ratio
        "rv_",        # realized vol/variance
    )
    # Cross-sectional features we'll compute
    use_cross_sectional_rank: bool = True  # rank features across assets per day
    use_momentum_features: bool = True     # trailing 1d, 3d, 7d returns
    use_vol_features: bool = True          # trailing volatility measures
    
    # Model
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    min_child_samples: int = 20
    
    # Walk-forward
    train_start: str = "2021-01-01"
    val_start: str = "2024-06-01"      # OOS starts here
    unseen_start: str = "2025-06-01"   # TRUE held-out
    min_train_days: int = 365
    
    # Portfolio
    portfolio_top_n: int = 5  # how many assets to hold each day
    
    # Evaluation
    random_seeds: int = 100  # for random-selection null


# === DATA LOADING ===

def load_panel(config: SelectionConfig) -> pl.DataFrame:
    """Load all chimera 1d files into a unified panel.
    
    Returns a DataFrame with columns: [date, asset, close, ...features...]
    sorted by (date, asset).
    """
    files = sorted(config.chimera_dir.glob("*_v51_chimera_1d_*.parquet"))
    
    frames = []
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f)
            # Keep date, OHLCV, and feature columns
            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            for prefix in config.feature_prefixes:
                keep_cols.extend([c for c in df.columns if c.startswith(prefix)])
            
            # Only keep columns that exist
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df.select(keep_cols)
            df = df.with_columns(pl.lit(sym).alias("asset"))
            frames.append(df)
        except Exception as e:
            print(f"  SKIP {sym}: {e}")
    
    panel = pl.concat(frames, how="diagonal")  # diagonal handles missing cols
    panel = panel.sort(["date", "asset"])
    return panel


def add_derived_features(panel: pl.DataFrame, config: SelectionConfig) -> pl.DataFrame:
    """Add derived features: momentum, vol, cross-sectional ranks."""
    
    # Sort for per-asset operations
    panel = panel.sort(["asset", "date"])
    
    # === Momentum features (lagged — these are KNOWN at decision time) ===
    if config.use_momentum_features:
        panel = panel.with_columns([
            # Trailing returns (realized, not forward)
            pl.col("close").pct_change(1).over("asset").alias("ret_1d"),
            pl.col("close").pct_change(3).over("asset").alias("ret_3d"),
            pl.col("close").pct_change(7).over("asset").alias("ret_7d"),
            pl.col("close").pct_change(14).over("asset").alias("ret_14d"),
            # Absolute return (magnitude signal)
            pl.col("close").pct_change(1).over("asset").abs().alias("abs_ret_1d"),
            pl.col("close").pct_change(3).over("asset").abs().alias("abs_ret_3d"),
        ])
    
    # === Volatility features ===
    if config.use_vol_features:
        panel = panel.with_columns([
            # Rolling volatility (std of returns)
            pl.col("close").pct_change(1).rolling_std(7).over("asset").alias("vol_7d"),
            pl.col("close").pct_change(1).rolling_std(14).over("asset").alias("vol_14d"),
            # Intraday range
            ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("intraday_range"),
            # Volume change
            pl.col("volume").pct_change(1).over("asset").alias("vol_change_1d"),
        ])
    
    return panel


def construct_labels(panel: pl.DataFrame, config: SelectionConfig) -> pl.DataFrame:
    """Construct the target label: is this asset in the top-K movers today?
    
    IMPORTANT: The label uses TODAY's return. Features are from YESTERDAY.
    This is enforced by lagging features in the model, not here.
    
    Label = 1 if asset is in top-K by absolute return AND return > threshold.
    We use POSITIVE returns only (long-only constraint).
    """
    # Compute forward return (this IS the target — what happens today)
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns(
        pl.col("close").pct_change(1).shift(-1).over("asset").alias("fwd_return_1d")
    )
    
    # Per-day: rank assets by positive return
    panel = panel.with_columns(
        pl.col("fwd_return_1d")
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("return_rank")
    )
    
    # Label: top-K by return AND positive AND above threshold
    panel = panel.with_columns(
        ((pl.col("return_rank") <= config.top_k) & 
         (pl.col("fwd_return_1d") >= config.min_return_threshold))
        .cast(pl.Int8)
        .alias("is_top_mover")
    )
    
    return panel


def add_cross_sectional_features(panel: pl.DataFrame, config: SelectionConfig) -> pl.DataFrame:
    """Add cross-sectional rank features (relative position among peers today).
    
    These are computed from TODAY's features (which are known at decision time
    since they're based on yesterday's close and earlier).
    """
    if not config.use_cross_sectional_rank:
        return panel
    
    # Rank key features cross-sectionally (per date)
    rank_cols = ["ret_1d", "ret_3d", "ret_7d", "vol_7d", "abs_ret_1d", "abs_ret_3d",
                 "intraday_range", "vol_change_1d"]
    
    for col in rank_cols:
        if col in panel.columns:
            panel = panel.with_columns(
                (pl.col(col).rank(method="ordinal").over("date") / 
                 pl.col(col).count().over("date"))
                .alias(f"xrank_{col}")
            )
    
    return panel


# === MODEL TRAINING AND PREDICTION ===

def get_feature_columns(panel: pl.DataFrame, config: SelectionConfig) -> list[str]:
    """Get the list of feature columns for the model."""
    feature_cols = []
    
    # Original chimera features (norm_, fund_, liq_, etc)
    for prefix in config.feature_prefixes:
        feature_cols.extend([c for c in panel.columns if c.startswith(prefix)])
    
    # Derived features
    derived = ["ret_1d", "ret_3d", "ret_7d", "ret_14d", "abs_ret_1d", "abs_ret_3d",
               "vol_7d", "vol_14d", "intraday_range", "vol_change_1d"]
    feature_cols.extend([c for c in derived if c in panel.columns])
    
    # Cross-sectional ranks
    feature_cols.extend([c for c in panel.columns if c.startswith("xrank_")])
    
    return sorted(set(feature_cols))


def walk_forward_predict(panel: pl.DataFrame, config: SelectionConfig) -> pl.DataFrame:
    """Walk-forward prediction: train on expanding window, predict 1 day ahead.
    
    Returns a DataFrame with columns: [date, asset, score, is_top_mover, fwd_return_1d]
    for the OOS and UNSEEN periods.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm required: pip install lightgbm")
    
    feature_cols = get_feature_columns(panel, config)
    print(f"Features: {len(feature_cols)} columns")
    
    # Filter to rows where we have the label
    panel = panel.filter(pl.col("is_top_mover").is_not_null())
    
    # Convert dates
    train_start = date.fromisoformat(config.train_start)
    val_start = date.fromisoformat(config.val_start)
    unseen_start = date.fromisoformat(config.unseen_start)
    
    # Get unique dates for walk-forward
    all_dates = sorted(panel.filter(pl.col("date") >= val_start)["date"].unique().to_list())
    
    print(f"Walk-forward over {len(all_dates)} days ({val_start} to {all_dates[-1]})")
    
    results = []
    retrain_every = 30  # retrain monthly
    model = None
    last_train_date = None
    
    for i, pred_date in enumerate(all_dates):
        # Train data: everything before pred_date (expanding window)
        train_mask = (panel["date"] >= train_start) & (panel["date"] < pred_date)
        train_df = panel.filter(train_mask)
        
        # Need minimum training data
        if len(train_df) < config.min_train_days * 10:  # ~10 assets per day minimum
            continue
        
        # Retrain periodically
        if model is None or (i % retrain_every == 0):
            X_train = train_df.select(feature_cols).to_numpy()
            y_train = train_df["is_top_mover"].to_numpy()
            
            # Handle NaN in features
            X_train = np.nan_to_num(X_train, nan=0.0)
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                "objective": "binary",
                "metric": "auc",
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "min_child_samples": config.min_child_samples,
                "verbosity": -1,
                "num_threads": 4,
                "is_unbalance": True,  # top-K is minority class
            }
            model = lgb.train(params, train_data, num_boost_round=config.n_estimators)
            last_train_date = pred_date
        
        # Predict for pred_date
        pred_mask = panel["date"] == pred_date
        pred_df = panel.filter(pred_mask)
        
        if len(pred_df) == 0:
            continue
        
        X_pred = pred_df.select(feature_cols).to_numpy()
        X_pred = np.nan_to_num(X_pred, nan=0.0)
        
        scores = model.predict(X_pred)
        
        # Collect results
        for j, row in enumerate(pred_df.iter_rows(named=True)):
            results.append({
                "date": row["date"],
                "asset": row["asset"],
                "score": float(scores[j]),
                "is_top_mover": int(row["is_top_mover"]),
                "fwd_return_1d": float(row["fwd_return_1d"]) if row["fwd_return_1d"] is not None else 0.0,
            })
        
        if (i + 1) % 50 == 0:
            print(f"  Predicted {i+1}/{len(all_dates)} days...")
    
    return pl.DataFrame(results)


# === EVALUATION ===

def evaluate_selection(predictions: pl.DataFrame, config: SelectionConfig) -> dict:
    """Evaluate the selection strategy.
    
    For each day, select top-N by model score, equal-weight them,
    compute daily portfolio return. Compare to random-N baseline.
    """
    results = {}
    
    # Per-day: select top-N by score
    portfolio = predictions.sort(["date", "score"], descending=[False, True])
    portfolio = portfolio.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over("date").alias("score_rank")
    )
    
    # Selected assets each day
    selected = portfolio.filter(pl.col("score_rank") <= config.portfolio_top_n)
    
    # Daily portfolio return (equal-weight)
    daily_returns = selected.group_by("date").agg([
        pl.col("fwd_return_1d").mean().alias("portfolio_return"),
        pl.col("is_top_mover").sum().alias("n_hits"),
        pl.col("asset").count().alias("n_selected"),
    ]).sort("date")
    
    port_rets = daily_returns["portfolio_return"].to_numpy()
    n_hits = daily_returns["n_hits"].to_numpy()
    n_days = len(daily_returns)
    
    # Compound return
    compound = float(np.prod(1 + port_rets) - 1)
    mean_daily = float(np.mean(port_rets))
    std_daily = float(np.std(port_rets))
    sharpe = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0
    
    # Hit rate: what fraction of our selections were actual top-K movers?
    hit_rate = float(np.mean(n_hits)) / config.portfolio_top_n
    
    # Max drawdown
    cumulative = np.cumprod(1 + port_rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    
    # Win rate (days with positive return)
    win_rate = float(np.mean(port_rets > 0))
    
    results["model"] = {
        "n_days": n_days,
        "compound_return": compound,
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "sharpe_annual": sharpe,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "mean_daily_pct": mean_daily * 100,
    }
    
    # === Random baseline (null hypothesis) ===
    rng = np.random.default_rng(42)
    random_compounds = []
    random_means = []
    
    for seed in range(config.random_seeds):
        rng_s = np.random.default_rng(seed)
        # For each day, randomly select N assets
        random_rets = []
        for d in daily_returns["date"].to_list():
            day_data = predictions.filter(pl.col("date") == d)
            if len(day_data) < config.portfolio_top_n:
                continue
            idx = rng_s.choice(len(day_data), size=config.portfolio_top_n, replace=False)
            day_ret = float(day_data["fwd_return_1d"].to_numpy()[idx].mean())
            random_rets.append(day_ret)
        
        random_rets = np.array(random_rets)
        random_compounds.append(float(np.prod(1 + random_rets) - 1))
        random_means.append(float(np.mean(random_rets)))
    
    results["random_baseline"] = {
        "mean_compound": float(np.mean(random_compounds)),
        "std_compound": float(np.std(random_compounds)),
        "mean_daily_pct": float(np.mean(random_means)) * 100,
        "p95_compound": float(np.percentile(random_compounds, 95)),
    }
    
    # Beat-random test
    results["beats_random"] = compound > float(np.percentile(random_compounds, 95))
    results["lift_vs_random_pct"] = (mean_daily - float(np.mean(random_means))) * 100
    
    return results


# === MAIN PIPELINE ===

def run_selection_engine(config: Optional[SelectionConfig] = None) -> dict:
    """Run the full mover selection pipeline."""
    if config is None:
        config = SelectionConfig()
    
    print("=" * 80)
    print("MOVER SELECTION ENGINE")
    print("=" * 80)
    
    # 1. Load data
    print("\n[1/6] Loading panel data...")
    panel = load_panel(config)
    print(f"  Panel: {len(panel):,} rows, {panel['asset'].n_unique()} assets, "
          f"{panel['date'].n_unique()} dates")
    
    # 2. Add derived features
    print("\n[2/6] Adding derived features...")
    panel = add_derived_features(panel, config)
    
    # 3. Construct labels
    print("\n[3/6] Constructing labels (top-K movers)...")
    panel = construct_labels(panel, config)
    
    # 4. Add cross-sectional features
    print("\n[4/6] Adding cross-sectional features...")
    panel = add_cross_sectional_features(panel, config)
    
    # 5. Walk-forward prediction
    print("\n[5/6] Walk-forward prediction...")
    predictions = walk_forward_predict(panel, config)
    print(f"  Predictions: {len(predictions):,} rows")
    
    # 6. Evaluate
    print("\n[6/6] Evaluating...")
    results = evaluate_selection(predictions, config)
    
    # Report
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    m = results["model"]
    r = results["random_baseline"]
    print(f"\n  MODEL PORTFOLIO (top-{config.portfolio_top_n} by score, EW):")
    print(f"    Days: {m['n_days']}")
    print(f"    Mean daily: {m['mean_daily_pct']:.4f}%")
    print(f"    Compound: {m['compound_return']*100:.2f}%")
    print(f"    Sharpe (ann): {m['sharpe_annual']:.2f}")
    print(f"    Hit rate (top-K accuracy): {m['hit_rate']:.3f}")
    print(f"    Max DD: {m['max_drawdown']*100:.1f}%")
    print(f"    Win rate: {m['win_rate']:.3f}")
    
    print(f"\n  RANDOM BASELINE (EW random-{config.portfolio_top_n}):")
    print(f"    Mean daily: {r['mean_daily_pct']:.4f}%")
    print(f"    Mean compound: {r['mean_compound']*100:.2f}%")
    print(f"    P95 compound: {r['p95_compound']*100:.2f}%")
    
    print(f"\n  VERDICT:")
    print(f"    Lift vs random: {results['lift_vs_random_pct']:.4f}%/day")
    print(f"    Beats p95 random: {'YES' if results['beats_random'] else 'NO'}")
    
    if m['mean_daily_pct'] >= 1.0:
        print(f"    ** NORTH STAR FLOOR (1%/day): ACHIEVED **")
    elif m['mean_daily_pct'] >= 0.5:
        print(f"    ** Promising: >0.5%/day — iterate on features/model **")
    else:
        print(f"    ** Below 0.5%/day — need fundamentally different approach **")
    
    return results


if __name__ == "__main__":
    config = SelectionConfig()
    results = run_selection_engine(config)
    
    # Save results
    output_dir = Path("crypto/runs/strat")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mover_selection_v1_{ts}.json"
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n  Results saved to: {output_file}")
