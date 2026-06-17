"""src/firm/pipeline.py -- the firm harness END-TO-END: candidates + market snapshot -> a risk-bounded BOOK.

Wires the engines into one callable, the way a desk actually runs:
  market_state.compute_state  -> the regime favourability (risk-on/off)
  decision_spine.decide        -> size each candidate (cost gate, confidence floor, Kelly, regime, loss-aversion, NO-TRADE)
  portfolio.allocate           -> combine the BET candidates into a risk-budgeted, correlation-aware book
  risk.check_book              -> the firm-level VaR/gross/drawdown gate (SCALE or HALT)
This is the showcase that the src/firm/ engines COMPOSE into a coherent firm pipeline. Pure / deterministic; the
forecasts are the only external input (produced upstream by the WM/discovery layers -- out of scope here). No emoji.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .decision_spine import Forecast, RiskLimits, decide
from .market_state import compute_state
from .portfolio import Position, PortfolioLimits, allocate
from .risk import RiskConfig, RiskState, check_book

__contract__ = {
    "kind": "firm_pipeline",
    "inputs": ["candidates [{asset, forecast(mean,std), cost}]", "market_returns {asset: ret}", "bankroll", "RiskState"],
    "outputs": ["Book(final_weights, market_state, decisions, allocation, risk_verdict, trace)"],
    "invariants": [
        "regime favourability from market_state feeds every decision's regime_posterior",
        "only BET decisions enter the portfolio; NO_TRADE candidates are dropped (not zero-weighted in)",
        "final weights = portfolio weights * risk scale_factor; a HALT yields an empty book",
    ],
}


@dataclass(frozen=True)
class Candidate:
    asset: str
    forecast: Forecast
    cost: float = 0.0024
    loss_aversion: float = 1.0


@dataclass(frozen=True)
class Book:
    final_weights: dict
    favourability: float
    decisions: dict
    risk_action: str
    risk_scale: float
    notes: list = field(default_factory=list)


def run_book(candidates: list, market_returns: dict, bankroll: float = 10_000.0,
             corr: list | None = None, risk_state: RiskState | None = None,
             dlimits: RiskLimits | None = None, plimits: PortfolioLimits | None = None,
             rcfg: RiskConfig | None = None) -> Book:
    ms = compute_state(market_returns)
    decisions, positions, vols, notes = {}, [], {}, []
    for c in candidates:
        d = decide(c.forecast, c.cost, bankroll, limits=dlimits,
                   regime_posterior=ms.favourability, loss_aversion=c.loss_aversion)
        decisions[c.asset] = d
        if d.action == "BET":
            positions.append(Position(c.asset, d.fraction, c.forecast.std))
            vols[c.asset] = c.forecast.std
    if not positions:
        return Book({}, ms.favourability, decisions, "NO_BOOK", 0.0, ["no BET candidates cleared the spine"])

    alloc = allocate(positions, corr, plimits)
    rv = check_book(alloc.weights, vols, corr, risk_state, rcfg)
    if rv.action == "HALT":
        return Book({}, ms.favourability, decisions, "HALT", 0.0, rv.breaches + ["risk HALT -> flat book"])
    final = {a: round(w * rv.scale_factor, 5) for a, w in alloc.weights.items()}
    notes = [f"regime fav={ms.favourability}", f"book vol-targeted; risk {rv.action} x{rv.scale_factor}"] + rv.breaches
    return Book(final, ms.favourability, decisions, rv.action, rv.scale_factor, notes)


def _selftest():
    print("=== firm pipeline (end-to-end) selftest ===")
    # a risk-on universe + 3 candidates (2 with edge, 1 without)
    market = {f"M{i}": 0.02 + 0.003 * (i % 4) for i in range(20)}  # broad-up
    cands = [
        Candidate("SOL", Forecast(0.05, 0.08)),     # clear edge -> BET
        Candidate("ETH", Forecast(0.03, 0.06)),     # edge -> BET
        Candidate("XRP", Forecast(0.001, 0.05)),    # below cost -> NO_TRADE
    ]
    book = run_book(cands, market, 10_000.0)
    print(f"  favourability={book.favourability} risk={book.risk_action} scale={book.risk_scale}")
    print(f"  decisions: " + ", ".join(f"{k}={v.action}" for k, v in book.decisions.items()))
    print(f"  final book weights: {book.final_weights}")
    print(f"  notes: {book.notes}")
    assert book.decisions["XRP"].action == "NO_TRADE", "below-cost candidate must be dropped"
    assert "SOL" in book.final_weights and "ETH" in book.final_weights and "XRP" not in book.final_weights
    assert book.risk_action in ("PASS", "SCALE")
    # risk-OFF regime: the same candidates should produce a SMALLER book (regime feeds sizing)
    stressed = {f"M{i}": (-0.05 if i % 2 else 0.04) for i in range(20)}
    book2 = run_book(cands, stressed, 10_000.0)
    sum1 = sum(abs(w) for w in book.final_weights.values())
    sum2 = sum(abs(w) for w in book2.final_weights.values())
    print(f"  risk-on gross={sum1:.4f} vs stressed gross={sum2:.4f} (stressed should be <=)")
    assert sum2 <= sum1 + 1e-9
    print("  ALL PASS -- market_state -> decide -> portfolio -> risk composes into one risk-bounded book.")


if __name__ == "__main__":
    _selftest()
