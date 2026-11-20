# Sizing Theory (Kelly variants + portfolio construction + fat-tail handling)

> The current trader skill cites Kelly + HRP. That is one sizing method and one portfolio-construction method, both with known failure modes in crypto. This file documents the full taxonomy with project-empirical anchors for when each fails.

## The sizing problem

Given:
- A signal with expected edge μ (per-trade return)
- Cost c (round-trip transaction cost)
- Variance σ² (per-trade return variance)
- Holding period h (bars)

Question: what fraction f of capital to allocate?

## Method 1: Cost-adjusted Kelly

`f* = (μ - c/h) / σ²`

In `crypto/src/wealth_bot/bot/position_sizer.py` (`PositionSizer.size()`) per CLAUDE.md. (Post-2026-06-04-reset home; the archived `src/strategy/` path is dead.)

Assumptions:
- Returns are i.i.d.
- Returns are approximately Gaussian.
- μ and σ² are known (not estimated with error).
- Infinite horizon (we keep playing the same game).
- Bankruptcy is the only loss state we're avoiding.

When Kelly works:
- Liquid assets with stable distributions (BTC, ETH for the most part).
- Well-estimated μ (large sample size, recent data).
- Holding period long enough for cost amortization (1-2 day in this project).

When Kelly fails (and how it fails):
- **Fat tails**: σ² understates risk in heavy-tailed distributions. Kelly oversizes. Memecoins, post-listing assets, regulatory windows.
- **Uncertain μ**: if μ has a wide posterior, full-Kelly is too aggressive. Use Bayesian Kelly (shrunk to half).
- **Non-stationarity**: μ today ≠ μ tomorrow. Kelly assumes stationarity.
- **Drawdown intolerance**: full Kelly maximizes log-growth but has ~50% expected DD. Most operators (and this project) can't tolerate that.

**Project default**: half-Kelly. Empirically: ~25% growth hit for ~50% DD reduction (Pratt 1964, Thorp 2000).

**Project current**: 1/8-Kelly for H18 (commit 4ba027b 2026-05-27). Defensive given:
- H18 paper-trade is unproven in live.
- Whale-flow data source under-verified (gap G2).
- General regime uncertainty post-halving.

## Method 2: Fractional-Kelly ladder

Sizing as a function of certainty:

| Certainty about edge | Recommended Kelly fraction |
|---|---|
| Backtest only, < 30 paper trades | 1/16-Kelly |
| 30-100 paper trades positive | 1/8-Kelly |
| 100+ paper trades + 60+ live trades positive | 1/4-Kelly |
| Multi-year live history, low DD | 1/2-Kelly (cap) |
| Anything | NEVER full-Kelly (DD spec exceeds acceptable) |

This is the **fractional ladder** referenced in `RISK_PLAYBOOK.md` sizing-asymmetry table.

## Method 3: Volatility-targeting

`position_size = target_vol / realized_vol`

Targets a fixed portfolio volatility rather than a fixed fraction. Resizes inverse to recent realized vol.

When to use:
- Multi-asset portfolio where you want equal risk contribution.
- Regime where vol is itself the signal (high vol = pullback risk).

Empirical in this project: not yet used in live, mentioned in `position_sizer.py`. Should be considered for memecoin sleeves where Kelly's σ² is unreliable but realized-vol is observable.

Composition with Kelly: `f_final = min(f_Kelly_fractional, f_vol_target)`. The lower wins.

## Method 4: Max-drawdown-target sizing

`position_size = max_acceptable_DD / max_historical_DD * base_size`

Caps sizing by historical DD experience. If historical DD was 15% and you accept 10%, scale by 0.67.

When useful:
- After a real DD event — anchor sizing on realized DD, not Kelly's σ²-derived DD.
- Memecoin sleeves where DD is the binding constraint (not return).

## Method 5: Bayesian Kelly (sizing under uncertainty)

`f* = (μ_posterior - c/h) / (σ²_posterior + σ²_posterior_mean)`

The denominator inflates by the posterior variance on the mean estimate itself. With small n, the posterior on μ is wide, so f* shrinks.

Practical heuristic without full Bayes: shrink Kelly fraction by `1 - 1/sqrt(n)` where n = number of trades.
- n = 25: f' = 0.80 * f_Kelly
- n = 100: f' = 0.90 * f_Kelly
- n = 400: f' = 0.95 * f_Kelly

Use during INCUBATION -> PAPER and PAPER -> LIVE_SMALL stage transitions.

## Method 6: Anti-Kelly for fat-tailed assets

When tails are heavy (kurtosis > 5, or empirically: PEPE-class memecoins):
- Use **fixed-fraction** sizing (e.g., 0.5% notional per trade).
- Do NOT use Kelly. σ² lies in fat-tailed regimes.
- Empirical floor: 1/16-Kelly to 1/32-Kelly depending on regime.

Per gold-standard dossier rule: PEPE sleeves cap at 1/16-Kelly regardless of vol-target.

## Portfolio construction methods

### Method A: Equal-weight (1/N)

Simplest. Each sleeve gets 1/N of risk budget. Robust to estimation error.

Failure mode: ignores correlations. Two highly-correlated sleeves take 2x risk.

### Method B: HRP (Hierarchical Risk Parity, Lopez de Prado 2016) — project default

`position_sizer.hrp_weights(corr_matrix)` — single-linkage clustering on correlation, bisects clusters by inverse-variance.

Failure mode: assumes correlations are stable. In tail events (e.g., 2022 LUNA collapse, 2023 SVB), all crypto correlations → 1. HRP underestimates this.

Mitigation: stress-test HRP weights under "all corr = 1" scenario; cap any single sleeve's weight at 0.40 regardless of HRP recommendation.

### Method C: ERC (Equal Risk Contribution)

Each sleeve contributes equally to portfolio variance. Like HRP but with stricter equality constraint.

Failure mode: requires reliable covariance matrix. Same tail issue as HRP.

When to use: portfolios of > 10 sleeves where HRP clustering becomes unstable.

### Method D: Max-diversification (Choueifaty 2011)

Maximize the ratio `(weighted_avg_vol) / (portfolio_vol)`. Highest "diversification ratio."

Failure mode: tends to over-weight low-vol assets. Not useful in crypto where vol is signal.

### Method E: Min-variance

Optimize portfolio for minimum variance. Quadratic programming.

Failure mode: sensitive to covariance estimation error. Lopez de Prado 2014 showed min-variance underperforms HRP in out-of-sample tests.

### Method F: Mean-CVaR

Minimize Conditional Value-at-Risk (expected loss in worst 5% of outcomes). Better for fat-tailed assets.

Failure mode: needs lots of data to estimate tails. Compute-heavy.

When useful: portfolio of memecoin sleeves where mean-variance is misleading.

### Method G: Black-Litterman blend

Blend prior (e.g., HRP weights) with explicit views (e.g., "I believe sleeve X has higher Sharpe than implied"). Bayesian update.

Failure mode: requires explicit view + confidence on each. Easier said than done.

When useful: when trader has strong qualitative view that should override quant weights.

## Decision tree: which method to use

```
Is the sleeve in INCUBATION stage?
├── Yes → no sizing yet (backtest only)
└── No → continue

Is the sleeve in PAPER stage?
├── Yes → fractional Kelly (1/16 to 1/8) based on Bayesian shrinkage
└── No → continue

Is the sleeve on a fat-tailed asset (PEPE-class, memecoin, post-listing < 60d)?
├── Yes → ANTI-KELLY: fixed-fraction or 1/16-Kelly cap. Skip Kelly entirely.
└── No → continue

Is the sleeve in LIVE_SMALL stage (< 50 live trades)?
├── Yes → 1/8-Kelly (defensive)
└── No → continue

Is the sleeve in LIVE_SCALE stage with > 100 live trades and DD < 10%?
├── Yes → 1/4-Kelly (or 1/2-Kelly with formal authorization)
└── No → 1/8-Kelly (default)

Portfolio construction across sleeves:
├── < 3 sleeves → equal-weight (1/N)
├── 3-10 sleeves → HRP with single-sleeve cap at 0.40
├── 10+ sleeves → ERC with single-sleeve cap at 0.20
├── Memecoin-heavy portfolio → mean-CVaR with explicit tail estimation
└── Strong qualitative view → Black-Litterman blend on top of HRP prior
```

## Common sizing failure modes (what gets traders rekt)

1. **Sizing on Kelly with single-seed numbers** — full-Kelly on a backtest with luck-of-init. 2026-05-24 LSTM +44.6% / DQN +40.9% were init-luck per multi-seed audit. Always use multi-seed median for μ.
2. **Ignoring correlation in tail events** — HRP says each sleeve at 0.20, but in a crash all sleeves move together = 1.0 effective exposure. Stress-test.
3. **Scaling into decay** — adding notional while IC is falling. Always size on most-recent IC, not deploy-time IC.
4. **Symmetric sizing on DD** — see `RISK_PLAYBOOK.md`. Sizing change must be asymmetric (faster down than up).
5. **Sizing for return, not for risk** — picking Kelly fraction to "hit target ROI." This is backwards. Pick risk budget first, return follows.

## Hard sizing caps (this project's North Star)

- LONG-ONLY (no shorts).
- SPOT only (no PERP for now).
- LEV ≤ 1 (no leverage).
- Max 0.5x Kelly (no full-Kelly under any condition).
- Max 0.40 portfolio weight per sleeve (HRP-recommended weights capped).
- Max 0.05 portfolio risk-of-ruin (defined as P(portfolio DD > 30%)).

Any sleeve YAML or RiskController parameter that violates these = automatic reject.

## CDAP wiring

| Rule | Severity | What it checks |
|---|---|---|
| `trader_max_kelly_fraction_le_half` | critical | All sleeve YAML `max_kelly_fraction <= 0.5` |
| `trader_long_only_in_sleeve_yaml` | critical | All sleeve YAML `position_range_min >= 0` |
| `trader_spot_mode_in_live_sleeves` | critical | Live-stage sleeves use SPOT mode (not PERP) |
| `trader_hrp_single_sleeve_cap` | warn | HRP weights computed with cap 0.40 applied |

## Cross-references

- RISK_PLAYBOOK.md — fat-tail override + sizing asymmetry.
- LIFECYCLE.md — stage gates determine Kelly fraction.
- CRYPTO_MICROSTRUCTURE.md — funding, cascades, depegs all affect sizing posterior.
