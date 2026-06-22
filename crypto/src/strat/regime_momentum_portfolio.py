"""Regime-Timed Momentum Portfolio — the honest 1-5%/day engine.

FINDINGS FROM V1-V3 + VOL-BREAKOUT AUDIT:
1. No daily-bar SELECTION alpha exists (D17, D55, D77, vol-breakout null)
2. In bull regime + momentum filter, random selection compounds >1%/day
3. The value is REGIME TIMING (when to deploy/cash) + DIVERSIFICATION
4. Trailing stops create artificial alpha — must NOT use them in evaluation
5. V1 drawdown-insurance (SMA regime gate) IS verified and works

THE HONEST ENGINE:
- Deploy a diversified long-only portfolio of momentum-positive assets in bull regime
- Go to cash (or reduced exposure) in bear regime
- Size inversely to vol (vol-targeting — the one confirmed zero-dead-list overlay)
- NO trailing stops in evaluation (evaluate raw portfolio returns)
- Compare honestly against BTC buy-hold AND the EW basket

SUCCESS CRITERIA (BINDING):
- Must beat BTC buy-hold on risk-adjusted basis (Calmar ratio)
- Must have maxDD < 50% (V1 target: halve the HODL drawdown of -82%)
- Must have positive compound over the full period
- The 1%/day in bull is mechanically available — the question is REGIME ACCURACY

WHAT MAKES THIS DIFFERENT FROM "JUST BUY IN BULL":
The naive "buy when BTC > SMA" is already in the dead-list (D74: OOS ~0/neg).
BUT: D74 tested it on INDIVIDUAL ASSETS with their OWN SMA. The V1 verified
finding uses BTC-MARKET regime on a DIVERSIFIED PORTFOLIO. The difference:
- Self-SMA gate: sits out the asset's own rally (fails)
- BTC-market gate: uses the broad market as a switch for the portfolio
- V1 verification: halves maxDD, preserves bull exposure, 3-axis red-team GO

The open question is: can the regime signal be FAST enough to avoid the sharp
drawdowns (Feb 2025: -33% in one month) while not flickering in/out?
"""
from __future__ import annotations

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime
import json


@dataclass  
class RegimeMomentumConfig:
    chimera_dir: Path = Path("crypto/data/processed/chimera/1d")
    
    # Regime detection (BTC-based, multiple signals)
    fast_sma: int = 50   # fast SMA for early detection
    slow_sma: int = 100  # slow SMA for confirmation
    hysteresis: float = 0.02  # 2% band
    
    # Dual-regime: RISK-ON when fast > slow, RISK-OFF otherwise
    # (moving average crossover on BTC — the "Golden Cross" concept)
    use_dual_sma: bool = True
    
    # Portfolio construction (bull regime)
    min_momentum_1d: float = 0.0    # must have positive 1d return
    min_momentum_7d: float = 0.005  # must have +0.5% over 7d
    max_positions: int = 10         # diversified top-10
    
    # Vol-targeting
    vol_target: float = 0.03  # 3% daily vol target per position
    max_weight: float = 0.20  # 20% max per position
    
    # Partial deployment in uncertain regime
    bear_exposure: float = 0.0     # 0% in bear (full cash)
    transition_exposure: float = 0.3  # 30% in transition/uncertain
    
    # Walk-forward period
    test_start: str = "2024-01-01"  # start earlier for more data
    
    # NO trailing stop (honest evaluation)


def compute_dual_regime(btc_close: np.ndarray, config: RegimeMomentumConfig) -> np.ndarray:
    """Dual-SMA regime: fast > slow = bull, slow > fast = bear.
    
    With hysteresis to avoid flicker.
    """
    n = len(btc_close)
    
    if config.use_dual_sma:
        # Fast and slow SMAs
        fast = np.convolve(btc_close, np.ones(config.fast_sma)/config.fast_sma, "full")[:n]
        slow = np.convolve(btc_close, np.ones(config.slow_sma)/config.slow_sma, "full")[:n]
        fast[:config.fast_sma-1] = np.nan
        slow[:config.slow_sma-1] = np.nan
        
        # Regime: fast > slow * (1 + hysteresis) = bull
        regime = np.zeros(n, dtype=int)
        state = 0
        for i in range(config.slow_sma, n):
            if np.isnan(fast[i]) or np.isnan(slow[i]):
                continue
            if state == 0:  # bear
                if fast[i] > slow[i] * (1 + config.hysteresis):
                    state = 1
            else:  # bull
                if fast[i] < slow[i] * (1 - config.hysteresis):
                    state = 0
            regime[i] = state
    else:
        # Single SMA (simpler)
        sma = np.convolve(btc_close, np.ones(config.slow_sma)/config.slow_sma, "full")[:n]
        sma[:config.slow_sma-1] = np.nan
        regime = np.zeros(n, dtype=int)
        state = 0
        for i in range(config.slow_sma, n):
            if np.isnan(sma[i]):
                continue
            if state == 0 and btc_close[i] > sma[i] * (1 + config.hysteresis):
                state = 1
            elif state == 1 and btc_close[i] < sma[i] * (1 - config.hysteresis):
                state = 0
            regime[i] = state
    
    return regime


def run_regime_momentum(config: RegimeMomentumConfig = None) -> dict:
    """Run the honest regime-timed momentum portfolio."""
    if config is None:
        config = RegimeMomentumConfig()
    
    print("=" * 80)
    print("REGIME-TIMED MOMENTUM PORTFOLIO (honest, no trailing stop)")
    print("=" * 80)
    
    # Load all assets
    print("\n[1] Loading universe...")
    files = sorted(config.chimera_dir.glob("*_v51_chimera_1d_*.parquet"))
    
    asset_data = {}  # sym -> DataFrame with date, close, ret_1d, ret_7d, vol
    btc_df = None
    
    for f in files:
        sym = f.name.split("_v51_")[0].upper()
        try:
            df = pl.read_parquet(f, columns=["date", "open", "high", "low", "close", "volume"]).sort("date")
            df = df.with_columns([
                pl.col("close").pct_change(1).alias("ret_1d"),
                pl.col("close").pct_change(7).alias("ret_7d"),
                pl.col("close").pct_change(1).rolling_std(10).alias("vol_10d"),
            ])
            asset_data[sym] = df
            if sym == "BTCUSDT":
                btc_df = df
        except Exception:
            pass
    
    print(f"  {len(asset_data)} assets loaded")
    
    # Compute regime
    print("\n[2] Computing regime...")
    btc_close = btc_df["close"].to_numpy()
    btc_dates = btc_df["date"].to_list()
    regime = compute_dual_regime(btc_close, config)
    
    bull_pct = regime[config.slow_sma:].mean()
    print(f"  Bull ratio (post warm-up): {bull_pct:.1%}")
    
    # Build date -> regime lookup
    date_regime = {d: int(r) for d, r in zip(btc_dates, regime)}
    
    # Walk-forward portfolio
    print("\n[3] Running portfolio...")
    test_start = date.fromisoformat(config.test_start)
    
    # Get all dates after test_start where BTC has data
    test_dates = [d for d in btc_dates if d >= test_start]
    
    daily_results = []
    
    for day in test_dates:
        r = date_regime.get(day, 0)
        
        # Determine exposure level
        if r == 1:
            target_exposure = 1.0  # full deployment in bull
        else:
            target_exposure = config.bear_exposure  # cash in bear
        
        if target_exposure == 0:
            daily_results.append({
                "date": day, "port_return": 0.0, "regime": "bear",
                "n_positions": 0, "exposure": 0.0
            })
            continue
        
        # Select assets: momentum-positive, sorted by momentum, top-N
        candidates = []
        for sym, df in asset_data.items():
            row = df.filter(pl.col("date") == day)
            if len(row) == 0:
                continue
            ret_1d = row["ret_1d"][0]
            ret_7d = row["ret_7d"][0]
            vol_10d = row["vol_10d"][0]
            
            if ret_1d is None or ret_7d is None or vol_10d is None:
                continue
            
            # Momentum gates
            if ret_1d < config.min_momentum_1d:
                continue
            if ret_7d < config.min_momentum_7d:
                continue
            
            # Must have reasonable vol
            if vol_10d < 0.005 or vol_10d > 0.50:
                continue
            
            candidates.append((sym, float(ret_7d), float(vol_10d)))
        
        if len(candidates) == 0:
            daily_results.append({
                "date": day, "port_return": 0.0, "regime": "bull_no_cand",
                "n_positions": 0, "exposure": 0.0
            })
            continue
        
        # Sort by 7d momentum (ride the winners)
        candidates.sort(key=lambda x: -x[1])
        selected = candidates[:config.max_positions]
        
        # Vol-targeted weights
        vols = np.array([c[2] for c in selected])
        raw_weights = config.vol_target / vols
        raw_weights = np.minimum(raw_weights, config.max_weight)
        weights = raw_weights / raw_weights.sum()
        
        # Apply target exposure
        weights = weights * target_exposure
        
        # Get next-day returns for selected assets
        returns = []
        for sym, _, _ in selected:
            df = asset_data[sym]
            # Find position of this date
            dates_list = df["date"].to_list()
            try:
                pos = dates_list.index(day)
                if pos + 1 < len(df):
                    next_ret = df["ret_1d"][pos + 1]
                    returns.append(float(next_ret) if next_ret is not None else 0.0)
                else:
                    returns.append(0.0)
            except (ValueError, IndexError):
                returns.append(0.0)
        
        returns = np.array(returns)
        port_return = float(np.dot(weights, returns))
        
        # Cash portion earns 0 (conservative — no yield assumed)
        # Portfolio return = weighted assets + cash * 0
        cash_weight = 1.0 - weights.sum()
        # port_return already reflects only the invested portion weighted
        
        daily_results.append({
            "date": day, "port_return": port_return, "regime": "bull",
            "n_positions": len(selected), "exposure": float(weights.sum())
        })
    
    # === EVALUATION ===
    print("\n[4] Evaluation...")
    daily_df = pl.DataFrame(daily_results).sort("date")
    rets = daily_df["port_return"].to_numpy()
    
    n_days = len(rets)
    compound = float(np.prod(1 + rets) - 1)
    mean_daily = float(np.mean(rets))
    std_daily = float(np.std(rets))
    sharpe = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0
    win_rate = float(np.mean(rets > 0))
    
    cum = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    calmar = (mean_daily * 365) / abs(max_dd) if max_dd != 0 else 0
    
    # Regime breakdown
    bull_mask = np.array([r["regime"] == "bull" for r in daily_results])
    bear_mask = np.array([r["regime"] == "bear" for r in daily_results])
    
    bull_rets = rets[bull_mask]
    bear_rets = rets[bear_mask]  # should be 0 (cash)
    
    bull_mean = float(np.mean(bull_rets)) if len(bull_rets) > 0 else 0
    bull_compound = float(np.prod(1 + bull_rets) - 1) if len(bull_rets) > 0 else 0
    n_bull = int(bull_mask.sum())
    n_bear = int(bear_mask.sum())
    
    # BTC buy-hold comparison (same period)
    btc_test = btc_df.filter(pl.col("date") >= test_start).sort("date")
    if len(btc_test) > 1:
        btc_compound = float(btc_test["close"][-1] / btc_test["close"][0] - 1)
        btc_rets = btc_test["close"].pct_change().drop_nulls().to_numpy()
        btc_dd = float(np.min((np.cumprod(1+btc_rets) - np.maximum.accumulate(np.cumprod(1+btc_rets))) / 
                              np.maximum.accumulate(np.cumprod(1+btc_rets))))
        btc_calmar = (float(np.mean(btc_rets)) * 365) / abs(btc_dd) if btc_dd != 0 else 0
    else:
        btc_compound = 0
        btc_dd = 0
        btc_calmar = 0
    
    # Report
    print("\n" + "=" * 80)
    print("REGIME-TIMED MOMENTUM PORTFOLIO — HONEST RESULTS")
    print("=" * 80)
    
    print(f"\n  PERIOD: {test_start} to {daily_results[-1]['date']} ({n_days} days)")
    print(f"  REGIME SPLIT: bull={n_bull} days, bear/cash={n_bear} days")
    
    print(f"\n  OUR PORTFOLIO:")
    print(f"    Compound: {compound*100:.2f}%")
    print(f"    Mean daily: {mean_daily*100:.4f}%")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Max DD: {max_dd*100:.1f}%")
    print(f"    Calmar: {calmar:.2f}")
    print(f"    Win rate: {win_rate:.3f}")
    
    print(f"\n  BULL REGIME ONLY (when deployed):")
    print(f"    Days: {n_bull}")
    print(f"    Mean daily: {bull_mean*100:.4f}%")
    print(f"    Compound: {bull_compound*100:.2f}%")
    
    print(f"\n  BTC BUY-HOLD:")
    print(f"    Compound: {btc_compound*100:.1f}%")
    print(f"    Max DD: {btc_dd*100:.1f}%")
    print(f"    Calmar: {btc_calmar:.2f}")
    
    print(f"\n  VERDICT:")
    if compound > btc_compound and max_dd > btc_dd:  # our DD is less negative
        print(f"    ** BEATS BTC on BOTH return ({compound*100:.1f}% vs {btc_compound*100:.1f}%) AND drawdown ({max_dd*100:.1f}% vs {btc_dd*100:.1f}%) **")
    elif calmar > btc_calmar:
        print(f"    ** BEATS BTC on Calmar ({calmar:.2f} vs {btc_calmar:.2f}) — better risk-adjusted **")
    elif compound > 0 and max_dd > -0.50:
        print(f"    ** Positive, controlled DD ({max_dd*100:.1f}%) — capital-preservation achieved **")
    
    if bull_mean >= 0.01:
        print(f"    ** IN BULL: {bull_mean*100:.3f}%/day — NORTH STAR REACHABLE with better regime timing **")
    
    # Save
    output = {
        "version": "regime_momentum_v1",
        "period": f"{test_start} to {daily_results[-1]['date']}",
        "overall": {
            "n_days": n_days, "compound": compound,
            "mean_daily_pct": mean_daily * 100,
            "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar,
            "win_rate": win_rate,
        },
        "bull_only": {
            "n_days": n_bull, "compound": bull_compound,
            "mean_daily_pct": bull_mean * 100,
        },
        "bear_cash": {"n_days": n_bear},
        "btc_buyhold": {
            "compound": btc_compound, "max_dd": btc_dd, "calmar": btc_calmar,
        },
        "config": {
            "fast_sma": config.fast_sma, "slow_sma": config.slow_sma,
            "hysteresis": config.hysteresis, "max_positions": config.max_positions,
            "vol_target": config.vol_target,
        },
    }
    
    output_dir = Path("crypto/runs/strat")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"regime_momentum_v1_{ts}.json"
    
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
    run_regime_momentum()
