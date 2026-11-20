"""src/firm/regime_playbook.py -- the REGIME-PLAYBOOK ROUTER: the meta-mindset engine that picks the RIGHT playbook for
the CURRENT regime. The single deepest trading truth the mechanical engines don't encode on their own: a strategy that
prints in a trend bleeds in a range, and what saves you in a crisis is not what makes you money in a melt-up. So the
desk does not run ONE playbook -- it ROUTES between playbooks by regime, and tightens or loosens the mindset emphasis +
the size with it.

Consumes market_state.MarketState (breadth/dispersion/risk-on-off) -> a Playbook(name, archetype, entry_bias,
size_multiplier, mindset_emphasis). The size_multiplier feeds decision_spine (an extra regime scale) / portfolio
(book-level), and mindset_emphasis tells trader_mindset which principles to weight in THIS regime. Maps to the
archetype master map (docs/MARKET_STRATEGY_ARCHETYPES.md): PRIMARY swing+breakout in trend, mean-rev only in confirmed
ranges, defensive/liquidation-reversal in crisis, AVOID forcing trades in chop. Pure / deterministic. __contract__ for
CDAP. No emoji (cp1252).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["MarketState(breadth, dispersion, momentum, risk_on_off, favourability)"],
    "outputs": ["Playbook(regime, name, archetype, entry_bias, size_multiplier, mindset_emphasis, avoid)"],
    "invariants": [
        "size_multiplier shrinks as the tape gets stressed/risk-off (survival-first); trend gets the most size",
        "each regime names the archetype to RUN and the traps to AVOID; deterministic from MarketState",
    ],
}


@dataclass(frozen=True)
class Playbook:
    regime: str                  # TREND_UP | RANGE | HIGH_VOL | RISK_OFF
    name: str
    archetype: str               # the strategy MODE to run (from the archetype master map)
    entry_bias: str              # what kind of entry the regime rewards
    size_multiplier: float       # extra regime scale on top of the decision_spine sizing (survival-first)
    mindset_emphasis: list       # which trader_mindset principles matter MOST here
    avoid: str = ""              # the trap to avoid in this regime
    notes: list = field(default_factory=list)


def route(ms) -> Playbook:
    """MarketState -> the regime's playbook. Thresholds on the risk_on_off composite + dispersion stress."""
    roo = ms.risk_on_off
    # dispersion stress: high cross-sectional dispersion = stressed/idiosyncratic tape
    stressed = ms.dispersion > 0.04

    if roo >= 0.4 and not stressed:
        return Playbook("TREND_UP", "ride the trend", "swing + breakout (PRIMARY)",
                        "pullback-into-trend / breakout-and-hold", 1.0,
                        ["asymmetry", "regime_fit", "selectivity"],
                        avoid="fading strength / counter-trend mean-reversion",
                        notes=["broad advance, contained dispersion -> press winners, trail wide"])
    if roo <= -0.4:
        return Playbook("RISK_OFF", "defend + hunt forced flow", "event-driven (liquidation-reversal) / cash",
                        "capitulation reclaim / liquidation-cascade reversal -- small, late, confirmed", 0.3,
                        ["survival", "edge_source", "invalidation_defined"],
                        avoid="catching the knife / sizing up into a downtrend",
                        notes=["broad decline -> capital preservation first; only forced-flow reversals, tiny size"])
    if stressed:
        return Playbook("HIGH_VOL", "reduce + be selective", "intraday-momentum (fast) / stand aside",
                        "only the highest-conviction, fastest setups -- or no trade", 0.4,
                        ["survival", "selectivity", "asymmetry"],
                        avoid="normal-size swing entries into whipsaw",
                        notes=["high dispersion -> whipsaw risk; cut size, demand a fat pitch, NO-TRADE is fine"])
    return Playbook("RANGE", "fade the edges (confirmed range only)", "mean-reversion (CONDITIONAL)",
                    "fade range extremes with confirmation; tight invalidation", 0.6,
                    ["not_crowded", "invalidation_defined", "selectivity"],
                    avoid="breakout-chasing in chop / assuming a trend",
                    notes=["neutral breadth, low momentum -> range tactics, but only when the range is CONFIRMED"])


def _selftest():
    print("=== regime_playbook selftest ===")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from firm.market_state import compute_state
    cases = {
        "broad-up": {f"A{i}": 0.03 + 0.004 * (i % 3) for i in range(20)},
        "broad-down": {f"A{i}": -0.04 - 0.004 * (i % 3) for i in range(20)},
        "high-vol/mixed": {f"A{i}": (0.10 if i % 2 else -0.09) for i in range(20)},
        "flat/range": {f"A{i}": 0.001 * (1 if i % 2 else -1) for i in range(20)},
    }
    pbs = {}
    for label, rets in cases.items():
        ms = compute_state(rets)
        pb = route(ms)
        pbs[label] = pb
        print(f"  {label:16} roo={ms.risk_on_off:+.2f} disp={ms.dispersion:.3f} -> {pb.regime:9} | {pb.archetype} | size x{pb.size_multiplier} | avoid: {pb.avoid}")
    assert pbs["broad-up"].regime == "TREND_UP" and pbs["broad-up"].size_multiplier == 1.0
    assert pbs["broad-down"].regime == "RISK_OFF" and pbs["broad-down"].size_multiplier <= 0.3
    assert pbs["high-vol/mixed"].regime in ("HIGH_VOL", "RISK_OFF") and pbs["high-vol/mixed"].size_multiplier <= 0.4
    assert pbs["flat/range"].regime == "RANGE"
    # survival ordering: trend sizes biggest, risk-off smallest
    assert pbs["broad-up"].size_multiplier > pbs["flat/range"].size_multiplier > pbs["broad-down"].size_multiplier
    print("  ALL PASS -- regime routes to the right archetype + size; survival-first sizing ordering holds.")


if __name__ == "__main__":
    _selftest()
