"""src/firm/crypto_market.py -- CRYPTO AS A MARKET: the structural facts of crypto + their TRADING IMPLICATION + which
harness engine each one informs. The narrate layer (src/narrate/crypto_context.py) describes the facts; this turns each
fact into a DECISION-RELEVANT implication and wires it to the engine that should act on it -- so "crypto is its own
market" stops being a slogan and becomes routing into the spine / mindset / regime / risk engines.

WHY (user, 2026-06-06): "research crypto as a market or any of a million other things." A desk that trades crypto like
equities dies; the edge AND the risk both come from crypto's structural peculiarities (24/7, perp funding as crowding,
liquidation reflexivity, BTC-beta dominance, venue fragmentation, narrative reflexivity). This engine is the queryable
map of those peculiarities and what each MEANS for a trade. Pure / deterministic. __contract__ for CDAP. No emoji.
"""
from __future__ import annotations

from dataclasses import dataclass

__contract__ = {
    "kind": "firm_engine",
    "inputs": ["approach/signal-family (optional filter)"],
    "outputs": ["Characteristic(key, fact, implication, informs) map + regime markers"],
    "invariants": ["each characteristic carries a TRADING IMPLICATION + the harness engine it informs; descriptive, no look-ahead"],
}


@dataclass(frozen=True)
class Characteristic:
    key: str
    fact: str
    implication: str       # what it MEANS for a trade (the decision-relevant part)
    informs: str           # which harness engine should act on it
    crypto_specific: bool = True


# Sourced + extended from src/narrate/crypto_context.py (the descriptive layer) into decision-relevant implications.
CHARACTERISTICS = [
    Characteristic("always_on", "24/7/365 trading -- no close, no circuit breakers",
                   "Risk accrues continuously; there is NO overnight-gap protection and weekend liquidity thins -- a "
                   "gap can happen at any hour. Size + stops must assume a move can occur while you sleep.", "risk"),
    Characteristic("perp_funding", "Perpetual funding + basis dominate positioning/price discovery",
                   "Persistently POSITIVE funding = crowded longs paying to stay = FADE-prone (priced in); negative = "
                   "crowded shorts. Read funding/basis BEFORE classic indicators -- it is the cleanest crowding gauge.",
                   "trader_mindset(not_crowded) + market_state"),
    Characteristic("liquidation_reflexivity", "Retail 10-125x leverage -> liquidation CASCADES (reflexive loop)",
                   "Forced liquidations feed back into price and trigger more -- a tradeable FORCED-FLOW reversal source "
                   "(the project's top open avenue) AND a tail risk to size for. Hunt the reversal, defend the cascade.",
                   "regime_playbook(RISK_OFF) + hypothesis_register"),
    Characteristic("btc_beta", "BTC-beta dominance -- most alts inherit BTC's direction",
                   "An alt 'signal' is usually BTC-beta in disguise. BETA-RESIDUALIZE before claiming an idiosyncratic "
                   "edge, or you will ship beta and call it alpha (the canonical false-positive here).",
                   "decision_spine(edge_source) + candidate_gate(benchmark)"),
    Characteristic("fragmentation", "Liquidity fragmented across venues; thin vs notional turnover",
                   "Depth shocks move price hard; fills are venue-dependent. p_fill + slippage are FIRST-CLASS -- "
                   "fixed-fill backtests are optimistic (expect 25-50% of fixed-backtest equity live).", "execution/cost"),
    Characteristic("narrative_attention", "Narrative + retail attention drive reflexive moves",
                   "Attention spikes precede/accompany moves -- a reflexivity INPUT, not noise. But consensus narrative "
                   "is already priced (second-order: who is left to buy?).", "market_state + trader_mindset(whats_priced_in)"),
]
_BY_KEY = {c.key: c for c in CHARACTERISTICS}

# crypto-specific REGIME markers (beyond price): the things to read that equities don't have.
REGIME_MARKERS = [
    "funding extreme (|funding| high) -> crowded positioning -> fade/contrarian bias",
    "open-interest spike + price stall -> leverage build-up -> cascade risk",
    "liquidation cluster printing -> forced-flow reversal window opening",
    "BTC-beta regime: are alts trading WITH btc (risk-on broad) or decoupling (idiosyncratic)?",
    "basis stress (perp-spot divergence) -> positioning stress / arb pressure",
]


def characteristics(informs: str | None = None) -> list:
    if informs:
        return [c for c in CHARACTERISTICS if informs.lower() in c.informs.lower()]
    return list(CHARACTERISTICS)


def implications_for(approach: str) -> list:
    """The crypto-nature implications most relevant to a given approach/signal family (best-effort keyword match)."""
    a = (approach or "").lower()
    hits = []
    if any(k in a for k in ("ma", "ema", "trend", "momentum", "breakout")):
        hits += [_BY_KEY["btc_beta"], _BY_KEY["always_on"]]          # trend signals: residualize beta, gap risk
    if any(k in a for k in ("liquidation", "cascade", "reversal", "event")):
        hits += [_BY_KEY["liquidation_reflexivity"], _BY_KEY["fragmentation"]]
    if any(k in a for k in ("funding", "carry", "basis", "positioning", "mean")):
        hits += [_BY_KEY["perp_funding"], _BY_KEY["narrative_attention"]]
    return hits or list(CHARACTERISTICS)  # unknown approach -> the full map (don't hide context)


def _selftest():
    print("=== crypto_market selftest ===")
    print(f"  {len(CHARACTERISTICS)} characteristics, {len(REGIME_MARKERS)} regime markers")
    for c in CHARACTERISTICS:
        print(f"  - {c.key:22} informs {c.informs}")
    ma = implications_for("adaptive MA trend")
    print(f"  implications_for('adaptive MA'): {[c.key for c in ma]}")
    assert any(c.key == "btc_beta" for c in ma), "MA approach MUST flag BTC-beta residualization"
    liq = implications_for("liquidation cascade reversal")
    assert any(c.key == "liquidation_reflexivity" for c in liq)
    assert characteristics(informs="risk"), "filter by informed-engine works"
    print("  ALL PASS -- each characteristic carries a trading implication + a harness hook; approach-routing works.")


if __name__ == "__main__":
    _selftest()
