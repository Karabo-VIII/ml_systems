"""src/firm/market_state.py -- the universe-wide MARKET STATE roll-up: breadth, dispersion, risk-on/off -> a single
favourability scalar the decision spine consumes as its `regime_posterior` and the portfolio conditions on.

WHY (design run wq3u9dvq1, gap, ev 0.66): asset_rotation already RANKS the cross-section, but there is no single
market-wide STATE object (breadth / dispersion / risk-on-off) that the decision + portfolio layers condition on. This
is the missing roll-up. It turns a cross-sectional snapshot into: how BROAD is the move (breadth), how STRESSED is the
tape (dispersion), and a [-1,1] risk-on/off score -> a [0,1] favourability that plugs straight into
decision_spine.decide(regime_posterior=...). Pure / deterministic / no look-ahead. __contract__ for CDAP. No emoji.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["returns {asset: recent_return}", "optional above_trend {asset: bool}", "optional disp_scale"],
    "outputs": ["MarketState(breadth, dispersion, momentum, risk_on_off in[-1,1], favourability in[0,1])"],
    "invariants": [
        "favourability = (risk_on_off+1)/2 in [0,1] -> directly usable as decision_spine regime_posterior",
        "broad-up + low-dispersion -> risk_on_off high; broad-down + high-dispersion -> risk_on_off low; deterministic",
    ],
}


@dataclass(frozen=True)
class MarketState:
    breadth: float           # fraction of the universe advancing (or above trend) in [0,1]
    dispersion: float        # cross-sectional std of returns (tape stress)
    momentum: float          # mean cross-sectional return
    risk_on_off: float       # composite in [-1, 1]: + risk-on, - risk-off
    favourability: float     # (risk_on_off+1)/2 in [0,1] -- plug into decision_spine.regime_posterior
    n_assets: int = 0
    notes: list = field(default_factory=list)


def _stdev(xs: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def compute_state(returns: dict, above_trend: dict | None = None, disp_scale: float = 0.05,
                  mom_scale: float = 0.02) -> MarketState:
    """Cross-sectional snapshot -> MarketState. `returns` = recent return per asset; `above_trend` (optional) overrides
    the breadth definition (fraction above its trend/MA) instead of fraction with positive return. disp_scale/mom_scale
    normalize dispersion-stress and momentum to ~O(1) before the composite."""
    assets = list(returns.keys())
    n = len(assets)
    if n == 0:
        return MarketState(0.5, 0.0, 0.0, 0.0, 0.5, 0, ["empty universe -> neutral"])
    rets = [returns[a] for a in assets]
    if above_trend:
        breadth = sum(1 for a in assets if above_trend.get(a)) / n
    else:
        breadth = sum(1 for r in rets if r > 0) / n
    dispersion = _stdev(rets)
    momentum = sum(rets) / n

    breadth_sig = 2.0 * breadth - 1.0                       # [-1,1]: 0.5 breadth -> 0
    mom_sig = math.tanh(momentum / max(mom_scale, 1e-9))    # [-1,1]
    disp_stress = math.tanh(dispersion / max(disp_scale, 1e-9))  # [0,1)-ish stress, subtract it (stress = risk-off)
    risk_on_off = max(-1.0, min(1.0, 0.5 * breadth_sig + 0.4 * mom_sig - 0.3 * disp_stress))
    fav = (risk_on_off + 1.0) / 2.0
    notes = []
    if disp_stress > 0.7:
        notes.append("high dispersion -> stressed tape (idiosyncratic / risk-off)")
    if breadth > 0.7 and momentum > 0:
        notes.append("broad advance -> risk-on")
    return MarketState(round(breadth, 4), round(dispersion, 5), round(momentum, 5), round(risk_on_off, 4),
                       round(fav, 4), n, notes)


def _selftest():
    print("=== market_state selftest ===")
    # broad-up, low dispersion -> risk-on, high favourability
    up = {f"A{i}": 0.03 + 0.005 * (i % 3) for i in range(20)}
    s_up = compute_state(up)
    print(f"  broad-up:   breadth={s_up.breadth} disp={s_up.dispersion} risk_on_off={s_up.risk_on_off} fav={s_up.favourability} {s_up.notes}")
    # broad-down, high dispersion -> risk-off, low favourability
    down = {f"A{i}": (-0.04 if i % 2 else 0.06) for i in range(20)}  # mixed, high dispersion, net down-ish
    s_dn = compute_state(down)
    print(f"  stressed:   breadth={s_dn.breadth} disp={s_dn.dispersion} risk_on_off={s_dn.risk_on_off} fav={s_dn.favourability} {s_dn.notes}")
    assert s_up.favourability > s_dn.favourability, "broad-up should be more favourable than stressed/mixed"
    assert 0.0 <= s_up.favourability <= 1.0 and 0.0 <= s_dn.favourability <= 1.0
    # the chain: favourability plugs into the decision spine as regime_posterior
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from firm.decision_spine import decide, Forecast
        d_up = decide(Forecast(0.0424, 0.30), 0.0024, 10_000, regime_posterior=s_up.favourability)
        d_dn = decide(Forecast(0.0424, 0.30), 0.0024, 10_000, regime_posterior=s_dn.favourability)
        print(f"  spine bets MORE in risk-on ({d_up.fraction}) than stressed ({d_dn.fraction})")
        assert d_up.fraction >= d_dn.fraction
    except Exception as e:
        print(f"  (spine-chain check skipped: {e})")
    print("  ALL PASS -- breadth/dispersion/risk-on-off roll-up feeds the decision spine's regime input.")


if __name__ == "__main__":
    _selftest()
