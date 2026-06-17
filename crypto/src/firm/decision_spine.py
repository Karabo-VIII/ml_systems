"""src/firm/decision_spine.py -- the PROBABILISTIC / DECISION-THEORETIC SPINE of the firm harness.

The piece the project had only SCATTERED (Kelly in multi_task_heads, Monte-Carlo in src/agents/a1_wm_consuming, expectancy nowhere
coherent): the single layer that turns a PROBABILISTIC FORECAST of a setup's outcome into a RISK-BOUNDED, SIZED BET,
honestly accounting for COST, model UNCERTAINTY, and CALIBRATION. A world-class firm does not bet on a point estimate;
it bets the EXPECTANCY net of cost, sized by Kelly HAIRCUT for how uncertain the edge is, capped by risk -- and
DEFAULTS TO NO-TRADE when the edge does not clear cost + uncertainty.

Pure functions (no I/O, no look-ahead, deterministic). The forecast is a per-SETUP distribution over net return
(mean + std), consistent with the project's unit-of-trading (a setup across a move), NOT a per-candle point IC.

__contract__ is declared for CDAP's contract_loader (AST-discovered, no import side-effects).
No emoji (Windows cp1252).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["forecast (mean_net_return, std, optional p_win)", "round_trip_cost", "bankroll", "risk_limits"],
    "outputs": ["Decision(action BET|NO_TRADE, fraction, notional, ev, kelly_raw, kelly_haircut, reason)"],
    "invariants": [
        "NO-TRADE default: edge must clear cost AND the confidence floor, else fraction=0",
        "fraction never exceeds max_fraction (risk cap) nor full Kelly (no over-betting)",
        "uncertainty HAIRCUT: higher forecast std -> smaller fraction (penalize edge uncertainty)",
        "all returns are NET of round-trip cost; deterministic; no look-ahead",
    ],
}


# --------------------------------------------------------------------------- core decision-theory primitives
def expectancy(p_win: float, avg_win: float, avg_loss: float, cost: float = 0.0) -> float:
    """EV per unit staked, net of round-trip cost. avg_loss is a POSITIVE magnitude. The trader's bread-and-butter."""
    p_win = min(max(p_win, 0.0), 1.0)
    return p_win * avg_win - (1.0 - p_win) * abs(avg_loss) - abs(cost)


def kelly_fraction(p_win: float, win_loss_ratio: float) -> float:
    """Full-Kelly stake fraction for a binary bet: f* = p - (1-p)/b, b = avg_win/avg_loss. Clamped to [0,1]."""
    if win_loss_ratio <= 0:
        return 0.0
    f = p_win - (1.0 - p_win) / win_loss_ratio
    return min(max(f, 0.0), 1.0)


def kelly_haircut_for_uncertainty(edge: float, edge_std: float, risk_aversion: float = 1.0) -> float:
    """Shrink the *effective edge* by a penalty for how UNCERTAIN the edge estimate is (a lower-confidence edge).
    effective_edge = max(0, edge - risk_aversion * edge_std). Returns the shrink RATIO in [0,1] to scale Kelly by.
    A firm never bets full Kelly on a noisy edge -- this is the principled reason quarter-Kelly is folklore."""
    if edge <= 0:
        return 0.0
    eff = edge - risk_aversion * max(edge_std, 0.0)
    return min(max(eff / edge, 0.0), 1.0)


def prob_profit(mean_net: float, std: float) -> float:
    """P(net return > 0) under a Gaussian forecast -- the calibrated win-probability for the setup."""
    if std <= 1e-12:
        return 1.0 if mean_net > 0 else 0.0
    # P(X>0) = Phi(mean/std); Phi via erf
    return 0.5 * (1.0 + math.erf((mean_net / std) / math.sqrt(2.0)))


def brier_score(forecasts: list, outcomes: list) -> float:
    """Calibration: mean squared error of probabilistic forecasts vs binary outcomes. Lower = better calibrated."""
    pairs = [(min(max(p, 0.0), 1.0), 1.0 if o else 0.0) for p, o in zip(forecasts, outcomes)]
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else float("nan")


# --------------------------------------------------------------------------- the coherent forecast -> bet decision
@dataclass(frozen=True)
class Forecast:
    mean_net_return: float   # forecast mean NET return for the setup (already net of expected cost if known)
    std: float               # OUTCOME dispersion (1 sigma of the setup's return) -- the bet's intrinsic risk
    edge_se: float | None = None  # OPTIONAL: standard error of the EDGE ESTIMATE (e.g. backtest SE). When given,
                                  # applies the uncertainty HAIRCUT. This is a DIFFERENT uncertainty from `std`
                                  # (outcome dispersion); conflating them double-counts risk.
    horizon_bars: int = 1


@dataclass(frozen=True)
class RiskLimits:
    base_kelly_fraction: float = 0.25   # quarter-Kelly base (folklore, now principled via the haircut)
    max_fraction: float = 0.20          # hard cap on bankroll fraction per bet
    confidence_floor: float = 0.55      # require P(profit) >= this to bet at all
    min_edge_after_cost: float = 0.0    # require mean net edge strictly above cost + this
    regime_floor: float = 0.0           # require regime_posterior >= this to bet (0 = no regime gate)


@dataclass(frozen=True)
class Decision:
    action: str              # "BET" | "NO_TRADE"
    fraction: float          # bankroll fraction to stake (0 if NO_TRADE)
    notional: float          # bankroll * fraction
    p_profit: float
    ev: float                # expectancy per unit, net of cost
    kelly_raw: float
    kelly_haircut: float     # the uncertainty shrink ratio applied
    reason: str


def decide(forecast: Forecast, round_trip_cost: float, bankroll: float,
           limits: RiskLimits | None = None, risk_aversion: float = 1.0,
           regime_posterior: float = 1.0, loss_aversion: float = 1.0) -> Decision:
    """Turn a probabilistic setup forecast into a risk-bounded sized bet -- the full decision_node keystone:
    {forecast, cost, regime_posterior, asymmetric loss_aversion} -> sized position. NO-TRADE is the default unless the
    edge clears cost AND the confidence floor AND the regime is favourable enough; the stake is uncertainty-haircut
    Kelly, scaled by the regime posterior and shrunk by loss-aversion, capped by risk.
      regime_posterior in [0,1] : P(the regime favours this setup) -- an unfavourable regime shrinks (or vetoes) the bet.
      loss_aversion >= 1        : asymmetric loss -- weights the downside more, so a loss-averse desk bets smaller."""
    limits = limits or RiskLimits()
    regime_posterior = min(max(regime_posterior, 0.0), 1.0)
    loss_aversion = max(loss_aversion, 1e-9)
    edge_net = forecast.mean_net_return - abs(round_trip_cost)
    p = prob_profit(edge_net, forecast.std)

    # NO-TRADE gates (the firm discipline -- do not bet a non-edge / a wrong-regime / a low-confidence setup)
    if edge_net <= limits.min_edge_after_cost:
        return Decision("NO_TRADE", 0.0, 0.0, p, edge_net, 0.0, 0.0,
                        f"edge_net {edge_net:.4f} <= floor {limits.min_edge_after_cost:.4f} (cost not cleared)")
    if p < limits.confidence_floor:
        return Decision("NO_TRADE", 0.0, 0.0, p, edge_net, 0.0, 0.0,
                        f"P(profit) {p:.3f} < confidence_floor {limits.confidence_floor:.2f}")
    if regime_posterior < limits.regime_floor:
        return Decision("NO_TRADE", 0.0, 0.0, p, edge_net, 0.0, 0.0,
                        f"regime_posterior {regime_posterior:.2f} < regime_floor {limits.regime_floor:.2f} (wrong regime)")

    # continuous Kelly for a Gaussian forecast: f* = mu / sigma^2 (leverage-optimal; ALWAYS scaled by the quarter-
    # Kelly base + the risk cap below, because full continuous Kelly ignores fat tails + estimate error).
    var = max(forecast.std ** 2, 1e-9)
    k_raw = max(edge_net / var, 0.0)
    # uncertainty HAIRCUT: only when an EDGE-ESTIMATE standard error is supplied (else rely on base + cap).
    haircut = 1.0 if forecast.edge_se is None else kelly_haircut_for_uncertainty(edge_net, forecast.edge_se, risk_aversion)
    # regime posterior SCALES the stake (unfavourable regime -> smaller); loss-aversion SHRINKS it (asymmetric loss).
    frac = min(k_raw * limits.base_kelly_fraction * haircut * regime_posterior / loss_aversion, limits.max_fraction)

    if frac <= 1e-6:
        return Decision("NO_TRADE", 0.0, 0.0, p, edge_net, round(k_raw, 4), round(haircut, 4),
                        "stake driven to ~0 (uncertainty haircut / regime / loss-aversion)")
    return Decision("BET", round(frac, 5), round(bankroll * frac, 2), round(p, 4),
                    round(edge_net, 5), round(k_raw, 4), round(haircut, 4),
                    f"BET {frac*100:.2f}% (contKelly {k_raw:.2f} x base {limits.base_kelly_fraction} x haircut {haircut:.2f} "
                    f"x regime {regime_posterior:.2f} / lossAv {loss_aversion:.2f}; cap {limits.max_fraction})")


def _selftest():
    print("=== decision_spine selftest ===")
    bk = 10_000.0
    # 1) a real edge, modest uncertainty -> BET
    d = decide(Forecast(mean_net_return=0.04, std=0.05), round_trip_cost=0.0024, bankroll=bk)
    print(f"  edge 4% / std 5%  -> {d.action} frac={d.fraction} notional={d.notional} p={d.p_profit} :: {d.reason}")
    assert d.action == "BET" and 0 < d.fraction <= 0.20
    # 2) edge does not clear cost -> NO_TRADE
    d2 = decide(Forecast(mean_net_return=0.001, std=0.05), round_trip_cost=0.0024, bankroll=bk)
    print(f"  edge 0.1% (< cost) -> {d2.action} :: {d2.reason}")
    assert d2.action == "NO_TRADE"
    # 3) decent mean but HUGE uncertainty -> haircut/confidence kills it
    d3 = decide(Forecast(mean_net_return=0.03, std=0.40), round_trip_cost=0.0024, bankroll=bk)
    print(f"  edge 3% / std 40% -> {d3.action} :: {d3.reason}")
    assert d3.action == "NO_TRADE"
    # 4) regime posterior + asymmetric loss-aversion scale the stake (use a below-cap edge so scaling is visible)
    f = Forecast(mean_net_return=0.0424, std=0.30)  # edge_net 0.04, frac lands below the 0.20 cap
    base = decide(f, 0.0024, bk)
    weak = decide(f, 0.0024, bk, regime_posterior=0.4)
    averse = decide(f, 0.0024, bk, loss_aversion=3.0)
    print(f"  base frac={base.fraction} | weak-regime(0.4)={weak.fraction} | loss-averse(3x)={averse.fraction}")
    assert base.action == "BET" and weak.fraction < base.fraction and averse.fraction < base.fraction
    vetoed = decide(f, 0.0024, bk, limits=RiskLimits(regime_floor=0.5), regime_posterior=0.3)
    print(f"  regime veto (floor .5, post .3) -> {vetoed.action} :: {vetoed.reason}")
    assert vetoed.action == "NO_TRADE"
    # 5) calibration sanity
    b = brier_score([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    print(f"  brier(well-calibrated set)={b:.3f} (low good)")
    assert b < 0.1
    print("  ALL PASS -- NO-TRADE default holds; edge+uncertainty gate works; calibration computes.")


if __name__ == "__main__":
    _selftest()
