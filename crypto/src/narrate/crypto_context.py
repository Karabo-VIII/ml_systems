"""src/narrate/crypto_context.py -- crypto AS ITS OWN MARKET.

Crypto is not equities, FX, or commodities, and reading its tape with those instincts is a category error. This
module encodes the structural characteristics that make crypto distinct and -- crucially -- how each one CHANGES the
way the narrator should read a signal. The narrator pulls the relevant caveats for whichever families are active so
every read is crypto-aware by construction.

User mandate (2026-06-06): "the engines you are building should understand crypto as a market (its characteristics)
because crypto is different from stocks, currencies, commodities, etc."
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CryptoTrait:
    key: str
    title: str
    vs_other_markets: str       # how it differs from equities/FX/commodities
    narration_implication: str  # how it changes the READ
    families: tuple             # feature families this trait governs


CRYPTO_TRAITS: list[CryptoTrait] = [
    CryptoTrait(
        "always_on", "24/7/365, no close, no circuit breakers",
        "Equities/FX have sessions, overnight gaps, halts, and exchange-enforced limit-downs. Crypto never closes; "
        "there is no halt to absorb a cascade and no opening auction to reset price discovery.",
        "Volatility can expand without a circuit-breaker brake -- moves run further and faster. There are no "
        "overnight gaps, so intrabar continuity holds; but weekends/holidays are THIN (wicks, slippage).",
        ("volatility", "liquidity", "orderflow")),
    CryptoTrait(
        "perp_funding", "Perpetual futures + funding dominate price discovery",
        "No equity/commodity analogue. The perpetual swap (not spot) is where most leverage and volume live; the "
        "funding rate is the periodic payment that tethers perp to spot.",
        "Funding is a FIRST-CLASS positioning gauge, not an exotic. Persistently positive funding = crowded longs "
        "paying to stay (fade-prone); negative = crowded shorts. Read funding/basis BEFORE classic indicators.",
        ("derivatives", "positioning")),
    CryptoTrait(
        "liquidation_reflexivity", "Leverage + liquidation cascades (reflexivity)",
        "Retail-accessible 10-125x leverage is unique. Forced liquidations feed back into price, which triggers more "
        "liquidations -- a reflexive loop equities largely lack.",
        "Liquidation spikes are CAUSAL flow, not just a coincident reading. Long-liq cascades overshoot to the "
        "downside (capitulation = contrarian-long context); short-liq squeezes overshoot up. Treat liq events as "
        "regime punctuation.",
        ("liquidation", "derivatives", "volatility")),
    CryptoTrait(
        "btc_beta", "BTC-beta dominance",
        "Equities have sector/market factors but no single name IS the market. In crypto, BTC (then ETH) is the beta; "
        "most alts inherit direction and risk-on/off from it.",
        "ALWAYS read an alt in BTC context: is this move idiosyncratic or just beta? A bullish alt read during a BTC "
        "dump is suspect. ~2/3 of alt variance is idiosyncratic on average, but regime/direction is BTC-led.",
        ("cross_asset", "momentum")),
    CryptoTrait(
        "fragmentation", "Fragmented venues, thin books vs notional turnover",
        "Equities centralize on a few lit venues with deep books; crypto liquidity is split across dozens of "
        "exchanges with books thin relative to the notional that trades.",
        "Kyle-lambda/impact and thin-book flags matter more -- modest flow moves price. Cross-exchange spreads widen "
        "in stress. Depth reads are fragile and venue-specific.",
        ("liquidity", "orderflow")),
    CryptoTrait(
        "retail_reflexive", "Retail-heavy, narrative- & attention-driven",
        "No earnings, dividends, or cash-flow anchor for most tokens; value is reflexive/network-driven. Retail "
        "participation and social attention are larger drivers than in institutional markets.",
        "Social/attention spikes are reflexive fuel, not noise. Momentum can self-sustain past 'fair value' because "
        "there is no fundamental anchor pulling it back. Crowding cuts both ways harder.",
        ("social", "momentum", "positioning")),
    CryptoTrait(
        "onchain_visibility", "On-chain + large-print whale flow is observable",
        "In equities, large-holder activity is largely hidden until filings. In crypto, large prints and on-chain "
        "movements are partially observable in near-real-time.",
        "Whale net-flow is an actionable read, not an after-the-fact disclosure. Sustained whale buying under a flat "
        "tape = quiet accumulation; the reverse = distribution.",
        ("whale",)),
    CryptoTrait(
        "power_law_dispersion", "Power-law dispersion, fat tails, survivorship churn",
        "A handful of names dominate moves on any day; listings/delistings churn the alt universe; return tails are "
        "fatter than equities.",
        "Cross-sectional RANK matters: being a relative leader/laggard is information. Treat extreme single-asset "
        "moves as regime, not anomaly. Beware survivorship when reasoning historically.",
        ("cross_asset", "volatility")),
]

CRYPTO_TRAITS_BY_FAMILY: dict[str, list[CryptoTrait]] = {}
for _t in CRYPTO_TRAITS:
    for _fam in _t.families:
        CRYPTO_TRAITS_BY_FAMILY.setdefault(_fam, []).append(_t)


def caveats_for_families(active_families) -> list[str]:
    """Crypto-specific narration caveats relevant to the families present in a read (deduped, order-stable)."""
    seen, out = set(), []
    for fam in active_families:
        for t in CRYPTO_TRAITS_BY_FAMILY.get(fam, []):
            if t.key not in seen:
                seen.add(t.key)
                out.append(f"[{t.title}] {t.narration_implication}")
    return out


def headline() -> str:
    return ("Crypto reads differently: 24/7 with no circuit breakers, perp-funding-driven positioning, "
            "liquidation-cascade reflexivity, BTC-beta dominance, fragmented thin books, and retail/attention "
            "reflexivity. Funding, liquidations, basis and positioning are first-class here, not exotic.")
