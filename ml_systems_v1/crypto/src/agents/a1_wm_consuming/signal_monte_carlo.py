"""
Signal Monte Carlo Analysis & Strategy Optimization
====================================================
Phase 1: Extract WM predictions vs actual returns (Monte Carlo episodes)
Phase 2: Signal quality analysis (IC per model/horizon/asset)
Phase 3: Offline strategy grid search (cost-aware, holding periods)
Phase 4: Environment validation of top strategies (IS + OOS)

Usage:
    python src/agents/a1_wm_consuming/signal_monte_carlo.py --ensemble --episodes 30
    python src/agents/a1_wm_consuming/signal_monte_carlo.py --per-model --episodes 20
    python src/agents/a1_wm_consuming/signal_monte_carlo.py --ensemble --episodes 30 --top 10
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from datetime import datetime
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DEVICE, NUM_ASSETS, REWARD_HORIZONS, INITIAL_CAPITAL,
    EPISODE_LENGTH, PER_ASSET_OBS_DIM, MAX_POSITION_FRAC,
    MAX_GROSS_EXPOSURE, TAKER_FEE_BPS, SLIPPAGE_BPS,
    FUNDING_RATE_HOURLY, BARS_PER_HOUR, BARS_PER_DAY,
    ASSET_LIST, PURGE_GAP_BARS, AGENT_LOG_DIR,
)
from environment import WorldModelTradingEnv
from train_agent import (
    load_world_model, load_ensemble_model, load_data,
    DEFAULT_ENSEMBLE_MODELS, PROJECT_ROOT,
)

HORIZONS = [1, 4, 16, 64]
FEE_RATE = (TAKER_FEE_BPS + SLIPPAGE_BPS) / 10_000  # 0.0006
FUNDING_PER_BAR = FUNDING_RATE_HOURLY / BARS_PER_HOUR


# ==========================================================================
# Phase 1: Signal Extraction
# ==========================================================================

def extract_signals(env, n_episodes, seed):
    """Run DoNothing episodes, collect predictions + actual returns.

    Returns list of episode dicts, each with:
        preds: [steps, assets, 4]     (h1, h4, h16, h64 predictions)
        actuals: [steps, assets, 4]   (corresponding actual forward returns)
        regime: [steps, assets, 3]    (bear/neutral/bull probs)
        uncertainty: [steps, assets]  (posterior entropy)
    """
    episodes = []
    zero_action = np.zeros(NUM_ASSETS, dtype=np.float32)

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        ep_preds, ep_actuals, ep_regime, ep_unc = [], [], [], []

        for step in range(EPISODE_LENGTH):
            step_p = np.zeros((NUM_ASSETS, 4), dtype=np.float32)
            step_a = np.zeros((NUM_ASSETS, 4), dtype=np.float32)
            step_r = np.zeros((NUM_ASSETS, 3), dtype=np.float32)
            step_u = np.zeros(NUM_ASSETS, dtype=np.float32)

            for i in range(NUM_ASSETS):
                off = i * PER_ASSET_OBS_DIM
                step_p[i] = obs[off:off + 4]
                step_r[i] = obs[off + 4:off + 7]
                step_u[i] = obs[off + 7]

                if i in env.asset_data and i in env.current_bar_indices:
                    bar = env.current_bar_indices[i]
                    d = env.asset_data[i]
                    if bar < d["n_bars"]:
                        step_a[i, 0] = d["target_return_1"][bar]
                        step_a[i, 1] = d["target_return_4"][bar]
                        step_a[i, 2] = d["target_return_16"][bar]
                        step_a[i, 3] = d["target_return_64"][bar]

            ep_preds.append(step_p)
            ep_actuals.append(step_a)
            ep_regime.append(step_r)
            ep_unc.append(step_u)

            obs, _, done, _ = env.step(zero_action)
            if done:
                break

        episodes.append({
            "preds": np.array(ep_preds),
            "actuals": np.array(ep_actuals),
            "regime": np.array(ep_regime),
            "uncertainty": np.array(ep_unc),
        })

    return episodes


# ==========================================================================
# Phase 2: Signal Quality Analysis
# ==========================================================================

def spearman_ic(pred, actual):
    """Spearman rank correlation (no scipy dependency)."""
    n = len(pred)
    if n < 10:
        return 0.0
    rp = np.argsort(np.argsort(pred)).astype(float)
    ra = np.argsort(np.argsort(actual)).astype(float)
    d = rp - ra
    return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def compute_signal_quality(episodes):
    """Compute IC, hit rate, regime stats, persistence from extracted signals."""
    # Pool all predictions and actuals
    all_p = {h: [] for h in range(4)}
    all_a = {h: [] for h in range(4)}
    per_asset_p = {h: {a: [] for a in range(NUM_ASSETS)} for h in range(4)}
    per_asset_a = {h: {a: [] for a in range(NUM_ASSETS)} for h in range(4)}

    for ep in episodes:
        for h in range(4):
            all_p[h].extend(ep["preds"][:, :, h].flatten())
            all_a[h].extend(ep["actuals"][:, :, h].flatten())
            for a in range(NUM_ASSETS):
                per_asset_p[h][a].extend(ep["preds"][:, a, h])
                per_asset_a[h][a].extend(ep["actuals"][:, a, h])

    # IC per horizon
    ic = {}
    hit_rate = {}
    for h_idx, h in enumerate(HORIZONS):
        p = np.array(all_p[h_idx])
        a = np.array(all_a[h_idx])
        ic[h] = spearman_ic(p, a)
        correct = ((p > 0) & (a > 0)) | ((p < 0) & (a < 0))
        hit_rate[h] = float(correct.mean())

    # IC per asset (average across horizons)
    asset_ic = {}
    for a in range(NUM_ASSETS):
        ics = []
        for h_idx in range(4):
            p = np.array(per_asset_p[h_idx][a])
            act = np.array(per_asset_a[h_idx][a])
            if len(p) > 50:
                ics.append(spearman_ic(p, act))
        asset_ic[ASSET_LIST[a]] = float(np.mean(ics)) if ics else 0.0

    # Conditional IC (top quartile predictions only)
    conditional_ic = {}
    for h_idx, h in enumerate(HORIZONS):
        p = np.array(all_p[h_idx])
        a = np.array(all_a[h_idx])
        thresh = np.percentile(np.abs(p), 75)
        mask = np.abs(p) >= thresh
        if mask.sum() > 50:
            conditional_ic[h] = spearman_ic(p[mask], a[mask])
        else:
            conditional_ic[h] = 0.0

    # Regime stats
    all_regime = np.concatenate([ep["regime"] for ep in episodes], axis=0)
    mean_regime = all_regime.mean(axis=(0, 1))
    max_regime = all_regime.max(axis=(0, 1))
    # What fraction of timesteps have any regime > various thresholds
    regime_above = {}
    for thresh in [0.40, 0.45, 0.50, 0.60]:
        frac = (all_regime.max(axis=-1) > thresh).mean()
        regime_above[thresh] = float(frac)

    # Uncertainty stats
    all_unc = np.concatenate([ep["uncertainty"] for ep in episodes], axis=0)

    # Signal persistence (autocorrelation of h=1 predictions)
    autocorrs = []
    for ep in episodes:
        for a in range(NUM_ASSETS):
            p = ep["preds"][:, a, 0]
            if len(p) > 10:
                autocorrs.append(float(np.corrcoef(p[:-1], p[1:])[0, 1]))

    # Prediction magnitude stats (for threshold calibration)
    pred_stats = {}
    for h_idx, h in enumerate(HORIZONS):
        p = np.abs(np.array(all_p[h_idx]))
        pred_stats[h] = {
            "mean": float(p.mean()),
            "p50": float(np.percentile(p, 50)),
            "p75": float(np.percentile(p, 75)),
            "p90": float(np.percentile(p, 90)),
            "p95": float(np.percentile(p, 95)),
        }

    return {
        "ic": ic,
        "conditional_ic": conditional_ic,
        "hit_rate": hit_rate,
        "asset_ic": asset_ic,
        "mean_regime": mean_regime.tolist(),
        "max_regime": max_regime.tolist(),
        "regime_above": regime_above,
        "mean_uncertainty": float(all_unc.mean()),
        "std_uncertainty": float(all_unc.std()),
        "signal_persistence": float(np.mean(autocorrs)) if autocorrs else 0.0,
        "pred_stats": pred_stats,
        "n_samples": len(all_p[0]),
    }


def print_signal_report(q, label):
    """Print formatted signal quality report."""
    print(f"\n  {'=' * 65}")
    print(f"  SIGNAL QUALITY: {label}")
    print(f"  {'=' * 65}")
    print(f"  Samples: {q['n_samples']:,}")

    print(f"\n  IC (Spearman) by Horizon:")
    print(f"  {'Horizon':<10s} {'IC':>8s} {'Cond.IC':>9s} {'HitRate':>8s} {'|Pred| p50':>12s} {'p90':>8s}")
    for h in HORIZONS:
        ps = q["pred_stats"][h]
        print(f"  h={h:<7d} {q['ic'][h]:>+8.4f} {q['conditional_ic'][h]:>+9.4f} "
              f"{q['hit_rate'][h]:>7.1%} {ps['p50']:>12.6f} {ps['p90']:>8.6f}")

    print(f"\n  IC by Asset (avg across horizons, sorted):")
    for name, ic_val in sorted(q["asset_ic"].items(), key=lambda x: -x[1]):
        bar = "+" * max(0, int(ic_val * 500))
        print(f"    {name:<12s} IC={ic_val:>+.4f} {bar}")

    mr = q["mean_regime"]
    print(f"\n  Regime Probs (bear/neutral/bull):")
    print(f"    Mean: [{mr[0]:.3f}, {mr[1]:.3f}, {mr[2]:.3f}]")
    print(f"    Max:  [{q['max_regime'][0]:.3f}, {q['max_regime'][1]:.3f}, {q['max_regime'][2]:.3f}]")
    for thresh, frac in sorted(q["regime_above"].items()):
        print(f"    Timesteps with max(regime) > {thresh:.2f}: {frac:.1%}")

    print(f"\n  Uncertainty: mean={q['mean_uncertainty']:.3f}, std={q['std_uncertainty']:.3f}")
    print(f"  Signal Persistence (h=1 autocorr): {q['signal_persistence']:.3f}")


# ==========================================================================
# Phase 3: Offline Strategy Simulation
# ==========================================================================

def simulate_episode_offline(preds, actuals_h1, hold_period, scale, threshold,
                             horizon_weights, mode="linear", fixed_size=0.10,
                             zero_funding=False):
    """Simulate a cost-aware holding strategy on one episode.

    Args:
        preds: [steps, assets, 4] predictions
        actuals_h1: [steps, assets] 1-bar actual returns
        hold_period: bars between rebalances
        scale: position = clip(signal * scale) for linear mode
        threshold: minimum |signal| to trade
        horizon_weights: [4] weights for combining horizon predictions
        mode: "linear" (proportional) or "binary" (fixed size)
        fixed_size: position size for binary mode
        zero_funding: if True, no funding costs (simulates spot trading)

    Returns dict with: final_value, net_pnl, costs, max_dd, sharpe, n_trades
    """
    n_steps = len(preds)
    hw = np.array(horizon_weights, dtype=np.float32)
    hw_sum = hw.sum()
    if hw_sum > 0:
        hw = hw / hw_sum

    positions = np.zeros(NUM_ASSETS, dtype=np.float32)
    portfolio = INITIAL_CAPITAL
    total_costs = 0.0
    steps_since_rebal = hold_period  # trigger rebalance on first step
    n_trades = 0
    values = [portfolio]

    for step in range(n_steps):
        steps_since_rebal += 1

        # Rebalance?
        if steps_since_rebal >= hold_period:
            signals = preds[step] @ hw  # [assets]
            if mode == "linear":
                target = np.clip(signals * scale, -MAX_POSITION_FRAC, MAX_POSITION_FRAC)
                target = np.where(np.abs(signals) >= threshold, target, 0.0)
            else:  # binary
                target = np.where(np.abs(signals) >= threshold,
                                  np.sign(signals) * fixed_size, 0.0)

            # Gross exposure cap
            gross = np.abs(target).sum()
            if gross > MAX_GROSS_EXPOSURE:
                target *= MAX_GROSS_EXPOSURE / gross

            # Transaction cost
            delta = np.abs(target - positions).sum()
            if delta > 1e-8:
                cost = delta * portfolio * FEE_RATE
                portfolio -= cost
                total_costs += cost
                n_trades += 1

            positions = target.copy()
            steps_since_rebal = 0

        # PnL from actual returns
        pnl = (positions * actuals_h1[step]).sum() * portfolio
        portfolio += pnl

        # Funding cost (skip for spot trading)
        if not zero_funding:
            funding = np.abs(positions).sum() * portfolio * FUNDING_PER_BAR
            portfolio -= funding
            total_costs += funding

        # Floor at zero (can't go negative)
        portfolio = max(portfolio, 0.0)
        values.append(portfolio)

    # Compute metrics
    values = np.array(values)
    returns = np.diff(values) / (values[:-1] + 1e-10)
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / (peak + 1e-10)
    max_dd = float(drawdowns.max())

    if len(returns) > 1 and returns.std() > 1e-12:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * BARS_PER_DAY))
    else:
        sharpe = 0.0

    return {
        "final_value": float(portfolio),
        "net_pnl": float(portfolio - INITIAL_CAPITAL),
        "costs": float(total_costs),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_trades": n_trades,
    }


def simulate_topn_offline(preds, actuals_h1, hold_period, n_top, pos_size,
                          horizon_idx=0, zero_funding=False):
    """Simulate TopN market-neutral strategy with holding period."""
    n_steps = len(preds)
    positions = np.zeros(NUM_ASSETS, dtype=np.float32)
    portfolio = INITIAL_CAPITAL
    total_costs = 0.0
    steps_since_rebal = hold_period
    n_trades = 0
    values = [portfolio]

    for step in range(n_steps):
        steps_since_rebal += 1

        if steps_since_rebal >= hold_period:
            h_preds = preds[step, :, horizon_idx]
            ranked = np.argsort(h_preds)
            target = np.zeros(NUM_ASSETS, dtype=np.float32)
            for idx in ranked[:n_top]:
                target[idx] = -pos_size
            for idx in ranked[-n_top:]:
                target[idx] = pos_size

            delta = np.abs(target - positions).sum()
            if delta > 1e-8:
                cost = delta * portfolio * FEE_RATE
                portfolio -= cost
                total_costs += cost
                n_trades += 1

            positions = target.copy()
            steps_since_rebal = 0

        pnl = (positions * actuals_h1[step]).sum() * portfolio
        portfolio += pnl

        if not zero_funding:
            funding = np.abs(positions).sum() * portfolio * FUNDING_PER_BAR
            portfolio -= funding
            total_costs += funding

        portfolio = max(portfolio, 0.0)
        values.append(portfolio)

    values = np.array(values)
    returns = np.diff(values) / (values[:-1] + 1e-10)
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / (peak + 1e-10)

    if len(returns) > 1 and returns.std() > 1e-12:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * BARS_PER_DAY))
    else:
        sharpe = 0.0

    return {
        "final_value": float(portfolio),
        "net_pnl": float(portfolio - INITIAL_CAPITAL),
        "costs": float(total_costs),
        "max_dd": float(drawdowns.max()),
        "sharpe": sharpe,
        "n_trades": n_trades,
    }


def grid_search(episodes, verbose=True, zero_funding=False):
    """Sweep strategy parameters on collected signal data.

    Args:
        episodes: list of episode signal dicts from extract_signals()
        verbose: print progress
        zero_funding: simulate spot trading (no funding costs)

    Returns sorted list of (config_dict, avg_metrics) tuples.
    """
    horizon_configs = {
        "h1": [1, 0, 0, 0],
        "h4": [0, 1, 0, 0],
        "h16": [0, 0, 1, 0],
        "h64": [0, 0, 0, 1],
        "short": [0.6, 0.4, 0, 0],
        "long": [0, 0, 0.4, 0.6],
        "equal": [0.25, 0.25, 0.25, 0.25],
    }

    # Precompute prediction percentiles for adaptive thresholds
    all_preds_flat = np.concatenate([ep["preds"][:, :, 0] for ep in episodes]).flatten()
    abs_preds = np.abs(all_preds_flat)
    p50 = float(np.percentile(abs_preds, 50))
    p75 = float(np.percentile(abs_preds, 75))
    p90 = float(np.percentile(abs_preds, 90))

    results = []

    # --- Linear mode configs ---
    # Extended scales: predictions are ~0.00002 median, need 1000-5000 for meaningful positions
    hold_periods = [4, 8, 16, 32, 64, 128, 256]
    scales = [100, 500, 1000, 2000, 5000]
    thresholds = [0.0, p50, p75, p90]

    n_linear = len(horizon_configs) * len(hold_periods) * len(scales) * len(thresholds)
    if verbose:
        print(f"\n  Grid search: {n_linear} linear + ", end="")

    for h_name, h_weights in horizon_configs.items():
        for hold, scale, thresh in product(hold_periods, scales, thresholds):
            ep_results = []
            for ep in episodes:
                r = simulate_episode_offline(
                    ep["preds"], ep["actuals"][:, :, 0],
                    hold, scale, thresh, h_weights, mode="linear",
                    zero_funding=zero_funding,
                )
                ep_results.append(r)

            avg = _avg_metrics(ep_results)
            results.append(({
                "type": "linear",
                "horizon": h_name,
                "hold_period": hold,
                "scale": scale,
                "threshold": round(thresh, 6),
                "threshold_pct": _thresh_label(thresh, p50, p75, p90),
            }, avg))

    # --- Binary mode configs ---
    fixed_sizes = [0.05, 0.10, 0.15]
    binary_thresholds = [p50, p75, p90]
    binary_holds = [8, 16, 32, 64]

    n_binary = len(horizon_configs) * len(binary_holds) * len(fixed_sizes) * len(binary_thresholds)
    if verbose:
        print(f"{n_binary} binary + ", end="")

    for h_name, h_weights in horizon_configs.items():
        for hold, fsize, thresh in product(binary_holds, fixed_sizes, binary_thresholds):
            ep_results = []
            for ep in episodes:
                r = simulate_episode_offline(
                    ep["preds"], ep["actuals"][:, :, 0],
                    hold, 0, thresh, h_weights, mode="binary", fixed_size=fsize,
                    zero_funding=zero_funding,
                )
                ep_results.append(r)

            avg = _avg_metrics(ep_results)
            results.append(({
                "type": "binary",
                "horizon": h_name,
                "hold_period": hold,
                "fixed_size": fsize,
                "threshold": round(thresh, 6),
                "threshold_pct": _thresh_label(thresh, p50, p75, p90),
            }, avg))

    # --- TopN configs ---
    topn_holds = [8, 16, 32, 64]
    topn_n = [2, 3]
    topn_sizes = [0.05, 0.08, 0.10]
    topn_horizons = [0, 1, 2]  # h1, h4, h16

    n_topn = len(topn_holds) * len(topn_n) * len(topn_sizes) * len(topn_horizons)
    if verbose:
        print(f"{n_topn} topN configs")

    for hold, nt, ps, h_idx in product(topn_holds, topn_n, topn_sizes, topn_horizons):
        ep_results = []
        for ep in episodes:
            r = simulate_topn_offline(
                ep["preds"], ep["actuals"][:, :, 0],
                hold, nt, ps, horizon_idx=h_idx,
                zero_funding=zero_funding,
            )
            ep_results.append(r)

        avg = _avg_metrics(ep_results)
        results.append(({
            "type": "topN",
            "horizon": HORIZONS[h_idx],
            "hold_period": hold,
            "n_top": nt,
            "pos_size": ps,
        }, avg))

    # Sort by net PnL (the user wants profit)
    results.sort(key=lambda x: -x[1]["net_pnl"])
    return results


def _avg_metrics(ep_results):
    """Average metrics across episodes."""
    keys = ["final_value", "net_pnl", "costs", "max_dd", "sharpe", "n_trades"]
    avg = {}
    for k in keys:
        vals = [r[k] for r in ep_results]
        avg[k] = float(np.mean(vals))
    avg["std_pnl"] = float(np.std([r["net_pnl"] for r in ep_results]))
    avg["win_rate"] = float(np.mean([1.0 if r["net_pnl"] > 0 else 0.0
                                      for r in ep_results]))
    return avg


def _thresh_label(thresh, p50, p75, p90):
    """Label threshold by its percentile."""
    if thresh <= 1e-10:
        return "none"
    if abs(thresh - p50) < 1e-10:
        return "p50"
    if abs(thresh - p75) < 1e-10:
        return "p75"
    if abs(thresh - p90) < 1e-10:
        return "p90"
    return f"{thresh:.6f}"


def print_grid_results(results, top_n, label):
    """Print top grid search results."""
    print(f"\n  {'=' * 90}")
    print(f"  TOP {top_n} STRATEGIES: {label}")
    print(f"  {'=' * 90}")
    print(f"  {'#':>3s} {'Type':<8s} {'Horizon':<8s} {'Hold':>5s} {'Scale/Size':>10s} "
          f"{'Thresh':>7s} {'NetPnL':>9s} {'Sharpe':>8s} {'MaxDD':>7s} {'WinRate':>8s} "
          f"{'Costs':>8s} {'Trades':>7s}")
    print(f"  {'-' * 90}")

    for i, (cfg, met) in enumerate(results[:top_n]):
        ctype = cfg["type"]
        horizon = str(cfg.get("horizon", ""))
        hold = cfg["hold_period"]

        if ctype == "linear":
            size_str = f"s={cfg['scale']}"
        elif ctype == "binary":
            size_str = f"fs={cfg['fixed_size']}"
        else:
            size_str = f"n={cfg.get('n_top', '')},ps={cfg.get('pos_size', '')}"

        thresh_str = cfg.get("threshold_pct", "")

        pnl_str = f"${met['net_pnl']:>+.2f}"
        print(f"  {i + 1:>3d} {ctype:<8s} {horizon:<8s} {hold:>5d} {size_str:>10s} "
              f"{thresh_str:>7s} {pnl_str:>9s} {met['sharpe']:>+8.2f} "
              f"{met['max_dd'] * 100:>6.1f}% {met['win_rate']:>7.0%} "
              f"${met['costs']:>7.1f} {met['n_trades']:>7.1f}")


# ==========================================================================
# Phase 4: Environment Validation Strategies
# ==========================================================================

class CostAwareHoldStrategy:
    """Strategy with minimum holding period and signal threshold.

    For use with the actual WorldModelTradingEnv (act() interface).
    """

    def __init__(self, config):
        self.config = config
        self.name = self._make_name(config)

        hw = config.get("horizon_weights", [1, 0, 0, 0])
        self.horizon_weights = np.array(hw, dtype=np.float32)
        s = self.horizon_weights.sum()
        if s > 0:
            self.horizon_weights /= s

        self.hold_period = config["hold_period"]
        self.threshold = config.get("threshold", 0.0)
        self.mode = config.get("type", "linear")
        self.scale = config.get("scale", 100)
        self.fixed_size = config.get("fixed_size", 0.10)

        self._target = np.zeros(NUM_ASSETS, dtype=np.float32)
        self._steps = self.hold_period  # trigger rebalance on first call

    def reset(self):
        self._target = np.zeros(NUM_ASSETS, dtype=np.float32)
        self._steps = self.hold_period

    def act(self, obs):
        self._steps += 1
        if self._steps >= self.hold_period:
            signals = np.zeros(NUM_ASSETS, dtype=np.float32)
            for i in range(NUM_ASSETS):
                off = i * PER_ASSET_OBS_DIM
                preds = obs[off:off + 4]
                signals[i] = preds @ self.horizon_weights

            if self.mode == "linear":
                target = np.clip(signals * self.scale,
                                 -MAX_POSITION_FRAC, MAX_POSITION_FRAC)
                target = np.where(np.abs(signals) >= self.threshold, target, 0.0)
            else:  # binary
                target = np.where(np.abs(signals) >= self.threshold,
                                  np.sign(signals) * self.fixed_size, 0.0)

            gross = np.abs(target).sum()
            if gross > MAX_GROSS_EXPOSURE:
                target *= MAX_GROSS_EXPOSURE / gross

            self._target = target
            self._steps = 0

        return self._target

    @staticmethod
    def _make_name(cfg):
        t = cfg.get("type", "linear")
        h = cfg.get("horizon", "?")
        hp = cfg["hold_period"]
        if t == "linear":
            return f"L_{h}_h{hp}_s{cfg.get('scale', 0)}"
        elif t == "topN":
            return f"TopN_{h}_h{hp}_n{cfg.get('n_top', 3)}"
        else:
            return f"B_{h}_h{hp}_fs{cfg.get('fixed_size', 0)}"


class TopNHoldStrategy:
    """TopN market-neutral strategy with holding period."""

    def __init__(self, config):
        self.config = config
        self.hold_period = config["hold_period"]
        self.n_top = config.get("n_top", 3)
        self.pos_size = config.get("pos_size", 0.10)
        self.h_idx = HORIZONS.index(config.get("horizon", 1))
        self.name = f"TopN_h{config.get('horizon', 1)}_hp{self.hold_period}_n{self.n_top}"

        self._target = np.zeros(NUM_ASSETS, dtype=np.float32)
        self._steps = self.hold_period

    def reset(self):
        self._target = np.zeros(NUM_ASSETS, dtype=np.float32)
        self._steps = self.hold_period

    def act(self, obs):
        self._steps += 1
        if self._steps >= self.hold_period:
            h_preds = np.array([obs[i * PER_ASSET_OBS_DIM + self.h_idx]
                                for i in range(NUM_ASSETS)])
            ranked = np.argsort(h_preds)

            target = np.zeros(NUM_ASSETS, dtype=np.float32)
            for idx in ranked[:self.n_top]:
                target[idx] = -self.pos_size
            for idx in ranked[-self.n_top:]:
                target[idx] = self.pos_size

            self._target = target
            self._steps = 0

        return self._target


def validate_in_env(strategy, env, n_episodes, seed):
    """Run a strategy in the real environment and collect metrics."""
    episode_values = []
    episode_costs = []
    episode_sharpes = []
    episode_max_dds = []
    episode_turnovers = []

    for ep in range(n_episodes):
        strategy.reset()
        obs = env.reset(seed=seed + ep)
        done = False
        total_cost = 0.0
        n_steps = 0
        step_returns = []
        portfolio_values = [env.portfolio_value]
        peak = env.portfolio_value
        max_dd = 0.0
        total_turnover = 0.0
        prev_positions = np.zeros(NUM_ASSETS, dtype=np.float32)

        while not done:
            action = strategy.act(obs)
            obs, reward, done, info = env.step(action)
            total_cost += info.get("txn_cost", 0) + info.get("funding_cost", 0)
            n_steps += 1

            pv = info.get("portfolio_value", env.portfolio_value)
            portfolio_values.append(pv)
            if len(portfolio_values) >= 2 and portfolio_values[-2] > 0:
                step_returns.append(
                    (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                )
            peak = max(peak, pv)
            if peak > 0:
                max_dd = max(max_dd, (peak - pv) / peak)

            positions = info.get("positions", prev_positions)
            total_turnover += np.abs(positions - prev_positions).sum()
            prev_positions = positions.copy()

        episode_values.append(info.get("portfolio_value", INITIAL_CAPITAL))
        episode_costs.append(total_cost)
        episode_max_dds.append(max_dd)
        episode_turnovers.append(total_turnover / max(n_steps, 1))

        if len(step_returns) > 1:
            sr = np.array(step_returns)
            ep_sharpe = (sr.mean() / (sr.std() + 1e-8)) * np.sqrt(252 * BARS_PER_DAY)
            episode_sharpes.append(ep_sharpe)

    return {
        "sharpe": float(np.mean(episode_sharpes)) if episode_sharpes else 0.0,
        "max_drawdown": float(np.max(episode_max_dds)) if episode_max_dds else 0.0,
        "win_rate": float(np.mean([1.0 if v > INITIAL_CAPITAL else 0.0
                                    for v in episode_values])),
        "mean_turnover": float(np.mean(episode_turnovers)),
        "mean_cost": float(np.mean(episode_costs)),
        "mean_final_value": float(np.mean(episode_values)),
        "net_pnl": float(np.mean(episode_values) - INITIAL_CAPITAL),
    }


# ==========================================================================
# Phase 5: Per-Model Analysis
# ==========================================================================

def per_model_analysis(n_episodes, seed):
    """Load each V1 model individually, compute signal quality."""
    models_to_test = [
        ("v1_0", 13),
        ("v1_1", 13),
        ("v1_2", 18),
        ("v1_3", 18),
        ("v1_4", 18),
    ]

    all_quality = {}
    data_cache = {}  # {n_features: segments}

    for variant, n_feat in models_to_test:
        print(f"\n  --- Loading {variant} (f{n_feat}) ---")
        try:
            model, revin, feature_list, load_data_fn = load_world_model(
                variant, n_feat, False
            )

            if n_feat not in data_cache:
                segments = load_data(load_data_fn, feature_list)
                data_cache[n_feat] = (segments, feature_list)
            else:
                segments, feature_list = data_cache[n_feat]

            env = WorldModelTradingEnv(
                world_model=model,
                data_segments=segments,
                feature_list=feature_list,
                mode="train",
                revin=revin,
                episode_length=EPISODE_LENGTH,
                seed=seed,
            )

            episodes = extract_signals(env, n_episodes, seed)
            quality = compute_signal_quality(episodes)
            all_quality[variant] = quality

            del model, env
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    [FAIL] {variant}: {e}")

    return all_quality


def print_model_comparison(all_quality):
    """Print per-model IC comparison table."""
    print(f"\n  {'=' * 70}")
    print(f"  PER-MODEL SIGNAL QUALITY COMPARISON")
    print(f"  {'=' * 70}")

    print(f"\n  {'Model':<15s}", end="")
    for h in HORIZONS:
        print(f" {'h=' + str(h):>8s}", end="")
    print(f" {'Persist':>8s} {'BestH':>6s}")
    print(f"  {'-' * 70}")

    for variant, q in sorted(all_quality.items()):
        print(f"  {variant:<15s}", end="")
        best_h = max(HORIZONS, key=lambda h: q["ic"][h])
        for h in HORIZONS:
            ic = q["ic"][h]
            marker = " *" if h == best_h else "  "
            print(f" {ic:>+6.4f}{marker}", end="")
        print(f" {q['signal_persistence']:>8.3f} h={best_h}")

    print(f"\n  Conditional IC (top 25% strongest predictions):")
    print(f"  {'Model':<15s}", end="")
    for h in HORIZONS:
        print(f" {'h=' + str(h):>8s}", end="")
    print()
    print(f"  {'-' * 55}")

    for variant, q in sorted(all_quality.items()):
        print(f"  {variant:<15s}", end="")
        for h in HORIZONS:
            cic = q["conditional_ic"][h]
            print(f" {cic:>+8.4f}", end="")
        print()


# ==========================================================================
# Phase 6: Main
# ==========================================================================

HORIZON_WEIGHT_MAP = {
    "h1": [1, 0, 0, 0],
    "h4": [0, 1, 0, 0],
    "h16": [0, 0, 1, 0],
    "h64": [0, 0, 0, 1],
    "short": [0.6, 0.4, 0, 0],
    "long": [0, 0, 0.4, 0.6],
    "equal": [0.25, 0.25, 0.25, 0.25],
}


def main():
    parser = argparse.ArgumentParser(
        description="Signal Monte Carlo Analysis & Strategy Optimization"
    )
    parser.add_argument("--world-model", type=str, default="v1_0")
    parser.add_argument("--features", type=int, choices=[13, 18, 19], default=13)
    parser.add_argument("--revin", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=30,
                        help="MC episodes for signal extraction (default: 30)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use V1.E ensemble")
    parser.add_argument("--ensemble-models", type=str, default=None)
    parser.add_argument("--per-model", action="store_true",
                        help="Run per-model signal comparison (slower)")
    parser.add_argument("--top", type=int, default=15,
                        help="How many top strategies to show/validate")
    parser.add_argument("--val-episodes", type=int, default=20,
                        help="Episodes for environment validation (default: 20)")
    parser.add_argument("--spot", action="store_true",
                        help="Zero funding costs (simulate spot trading)")
    args = parser.parse_args()

    use_ensemble = args.ensemble
    model_label = "V1.E ensemble" if use_ensemble else args.world_model

    print("=" * 70)
    print(f"  SIGNAL MONTE CARLO ANALYSIS")
    print("=" * 70)
    print(f"  Model: {model_label}")
    print(f"  MC Episodes: {args.episodes}")
    print(f"  Validation Episodes: {args.val_episodes}")
    print(f"  Spot Mode (no funding): {args.spot}")
    print(f"  Device: {DEVICE}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    # ---- Per-model analysis (optional) ----
    if args.per_model:
        print("\n  [PHASE 0] Per-model signal comparison...")
        all_quality = per_model_analysis(
            n_episodes=min(args.episodes, 15), seed=args.seed
        )
        print_model_comparison(all_quality)

    # ---- Load primary model ----
    print(f"\n  [PHASE 1] Loading model and extracting signals...")
    if use_ensemble:
        ensemble_keys = None
        if args.ensemble_models:
            ensemble_keys = [k.strip() for k in args.ensemble_models.split(",")]
        model, revin, feature_list, load_data_fn = load_ensemble_model(ensemble_keys)
    else:
        model, revin, feature_list, load_data_fn = load_world_model(
            args.world_model, args.features, args.revin
        )

    segments = load_data(load_data_fn, feature_list)

    # Create environments
    env_train = WorldModelTradingEnv(
        world_model=model, data_segments=segments, feature_list=feature_list,
        mode="train", revin=revin, episode_length=EPISODE_LENGTH, seed=args.seed,
    )
    env_val = WorldModelTradingEnv(
        world_model=model, data_segments=segments, feature_list=feature_list,
        mode="val", revin=revin, episode_length=EPISODE_LENGTH,
        seed=args.seed + 1000,
    )

    # ---- Extract signals ----
    print(f"  Extracting IS signals ({args.episodes} episodes)...")
    is_episodes = extract_signals(env_train, args.episodes, args.seed)

    print(f"  Extracting OOS signals ({args.episodes} episodes)...")
    oos_episodes = extract_signals(env_val, args.episodes, args.seed + 5000)

    # ---- Signal quality ----
    print(f"\n  [PHASE 2] Signal quality analysis...")
    is_quality = compute_signal_quality(is_episodes)
    oos_quality = compute_signal_quality(oos_episodes)
    print_signal_report(is_quality, f"IN-SAMPLE ({model_label})")
    print_signal_report(oos_quality, f"OUT-OF-SAMPLE ({model_label})")

    # ---- Break-even analysis ----
    print(f"\n  {'=' * 65}")
    print(f"  BREAK-EVEN ANALYSIS")
    print(f"  {'=' * 65}")
    print(f"  Cost per round trip: {FEE_RATE * 2 * 10000:.1f} bps")
    print(f"  Funding per bar: {FUNDING_PER_BAR * 10000:.4f} bps")
    print(f"\n  Required holding period to break even (per horizon, at scale=100):")
    for h_idx, h in enumerate(HORIZONS):
        ic = is_quality["ic"][h]
        pred_mean = is_quality["pred_stats"][h]["mean"]
        if ic > 0 and pred_mean > 0:
            # Expected edge per bar: scale * E[|pred|] * IC * portfolio
            # Cost per round trip: scale * E[|pred|] * 2 * FEE_RATE * portfolio
            # Break-even hold: 2 * FEE_RATE / IC
            be_hold = 2 * FEE_RATE / (ic * pred_mean * 100 + 1e-15)
            print(f"    h={h:<4d} IC={ic:>+.4f} mean|pred|={pred_mean:.6f} "
                  f"=> break-even hold ~{be_hold:.0f} bars")
        else:
            print(f"    h={h:<4d} IC={ic:>+.4f} -- no positive edge")

    # ---- Grid search ----
    print(f"\n  [PHASE 3] Strategy grid search (offline, IS data)...")
    is_results = grid_search(is_episodes, verbose=True, zero_funding=args.spot)

    n_profitable = sum(1 for _, m in is_results if m["net_pnl"] > 0)
    n_total = len(is_results)
    print(f"  Profitable configs: {n_profitable} / {n_total} ({n_profitable / n_total:.1%})")

    print_grid_results(is_results, args.top, "IN-SAMPLE (offline simulation)")

    # ---- Also run grid on OOS for comparison ----
    print(f"\n  Running grid search on OOS data...")
    oos_results = grid_search(oos_episodes, verbose=False, zero_funding=args.spot)
    print_grid_results(oos_results, args.top, "OUT-OF-SAMPLE (offline simulation)")

    # ---- Cross-reference: which IS winners also profit OOS? ----
    print(f"\n  {'=' * 90}")
    print(f"  IS/OOS CROSS-VALIDATION (top {args.top} IS strategies checked on OOS)")
    print(f"  {'=' * 90}")
    # Build OOS lookup by config
    oos_lookup = {}
    for cfg, met in oos_results:
        key = json.dumps(cfg, sort_keys=True)
        oos_lookup[key] = met

    surviving = 0
    print(f"  {'#':>3s} {'Type':<8s} {'Horizon':<8s} {'IS PnL':>9s} {'OOS PnL':>9s} "
          f"{'IS Sharpe':>10s} {'OOS Sharpe':>11s} {'Verdict':>8s}")
    print(f"  {'-' * 70}")

    for i, (cfg, is_met) in enumerate(is_results[:args.top]):
        key = json.dumps(cfg, sort_keys=True)
        oos_met = oos_lookup.get(key, {"net_pnl": 0, "sharpe": 0})
        verdict = "PASS" if oos_met["net_pnl"] > 0 else "FAIL"
        if oos_met["net_pnl"] > 0:
            surviving += 1
        print(f"  {i + 1:>3d} {cfg['type']:<8s} {str(cfg.get('horizon', '')):<8s} "
              f"${is_met['net_pnl']:>+8.2f} ${oos_met['net_pnl']:>+8.2f} "
              f"{is_met['sharpe']:>+10.2f} {oos_met['sharpe']:>+11.2f} "
              f"[{verdict}]")

    print(f"\n  Surviving OOS: {surviving} / {args.top}")

    # ---- Phase 4: Environment validation of top IS+OOS survivors ----
    # Collect configs that profit both IS and OOS
    env_candidates = []
    for cfg, is_met in is_results[:args.top * 2]:
        key = json.dumps(cfg, sort_keys=True)
        oos_met = oos_lookup.get(key, {"net_pnl": 0})
        if is_met["net_pnl"] > 0 and oos_met["net_pnl"] > 0:
            env_candidates.append(cfg)
        if len(env_candidates) >= args.top:
            break

    if env_candidates:
        print(f"\n  [PHASE 4] Environment validation ({len(env_candidates)} strategies, "
              f"{args.val_episodes} episodes each)...")
        print(f"  {'Strategy':<35s} {'IS PnL':>9s} {'IS Sharpe':>10s} "
              f"{'OOS PnL':>9s} {'OOS Sharpe':>11s} {'OOS MaxDD':>9s} {'Verdict':>8s}")
        print(f"  {'-' * 95}")

        env_results = []
        for cfg in env_candidates:
            if cfg["type"] == "topN":
                strat = TopNHoldStrategy(cfg)
            else:
                # Add horizon_weights for env strategy
                h_name = cfg.get("horizon", "h1")
                cfg_with_weights = dict(cfg)
                cfg_with_weights["horizon_weights"] = HORIZON_WEIGHT_MAP.get(
                    h_name, [1, 0, 0, 0])
                strat = CostAwareHoldStrategy(cfg_with_weights)

            is_env = validate_in_env(strat, env_train, args.val_episodes, args.seed + 100)
            oos_env = validate_in_env(strat, env_val, args.val_episodes, args.seed + 200)

            verdict = "PASS" if oos_env["net_pnl"] > 0 else "FAIL"
            print(f"  {strat.name:<35s} "
                  f"${is_env['net_pnl']:>+8.2f} {is_env['sharpe']:>+10.2f} "
                  f"${oos_env['net_pnl']:>+8.2f} {oos_env['sharpe']:>+11.2f} "
                  f"{oos_env['max_drawdown'] * 100:>8.1f}% [{verdict}]")

            env_results.append({
                "config": cfg,
                "strategy_name": strat.name,
                "is_env": is_env,
                "oos_env": oos_env,
            })
    else:
        print(f"\n  [PHASE 4] No strategies survived both IS and OOS -- skipping env validation.")
        env_results = []

    # ---- Save results ----
    tag = "v1e_ensemble" if use_ensemble else f"{args.world_model}_f{args.features}"
    results = {
        "model": model_label,
        "episodes": args.episodes,
        "seed": args.seed,
        "spot_mode": args.spot,
        "elapsed_sec": time.time() - t0,
        "signal_quality_is": is_quality,
        "signal_quality_oos": oos_quality,
        "top_strategies_is": [
            {"config": cfg, "metrics": met}
            for cfg, met in is_results[:args.top]
        ],
        "top_strategies_oos": [
            {"config": cfg, "metrics": met}
            for cfg, met in oos_results[:args.top]
        ],
        "env_validation": env_results,
        "n_profitable_is": n_profitable,
        "n_total_configs": n_total,
    }

    results_path = (AGENT_LOG_DIR /
                    f"signal_mc_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Results: {results_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
