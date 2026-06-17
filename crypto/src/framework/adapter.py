"""The MarketAdapter CONTRACT -- the ONE thing a new market must supply to flow through the solutioning pipeline.

Everything else in the pipeline (the 8-axis decomposition lattice, the validation gates, the workspace store, the
firm meta-layer, the autonomy loop) is MARKET-AGNOSTIC. A market becomes pluggable by implementing this small
interface: a data loader (the feature panel), a cost model (the binding constraint), and a universe. The crypto
adapter is the existing apparatus (chimera_loader + maker/taker cost + u10/u50/u100); a stocks adapter would supply
equities OHLCV+fundamentals + a broker cost model + an index universe -- and reuse the IDENTICAL gates and storage.

This is a CONTRACT (Protocol), not an implementation -- no strategy is built here. See docs/SOLUTIONING_PIPELINE.md.
No emoji (cp1252).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Sequence, Any


@runtime_checkable
class MarketAdapter(Protocol):
    """Implement this for a new market (stocks, FX, commodities, ...). The pipeline depends ONLY on these methods."""

    #: short market key used in the workspace path, e.g. "crypto", "stocks".
    market: str

    def universe(self, tier: str = "default") -> Sequence[str]:
        """Return the list of instrument symbols in a named universe tier (crypto: u10/u50/u100; stocks: sp500/...)."""
        ...

    def load(self, symbol: str, cadence: str, features: Sequence[str] | None = None) -> Any:
        """Return a time-ordered feature panel (DataFrame-like) for one instrument at a cadence.
        MUST be point-in-time / no look-ahead. Required columns: a timestamp/date, close, and the market's features.
        Crypto: pipeline.chimera_loader.ChimeraLoader.load. Stocks: an equities OHLCV+fundamentals loader."""
        ...

    def cost_model(self) -> "CostModel":
        """Return the market's realistic execution-cost model (the binding constraint). Crypto: maker/taker +
        calibrated p_fill (config/maker_cost_calibration.yaml). Stocks: commission + spread + borrow."""
        ...

    def cadences(self) -> Sequence[str]:
        """Canonical cadences/bar-types available for this market (crypto: 1d/4h/1h/30m/15m/dollar/...; stocks: 1d/1h/...)."""
        ...

    def feature_families(self) -> dict[str, Sequence[str]]:
        """Map family -> feature columns (the decomposition's signal axis). Crypto: narrate.feature_map families;
        stocks: price/volume/fundamental/flow/sentiment families."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """Realistic round-trip execution cost -- the pipeline's stage-03/05 gates depend on this being honest.
    MARKET-AGNOSTIC: each market models its own cost structure (crypto: maker/taker + calibrated p_fill;
    equities: commission + spread + borrow + impact; FX: spread + swap). No crypto maker/taker is baked in here."""

    def round_trip(self, symbol: str, side: str = "long", notional: float = 0.0, venue: str | None = None) -> float:
        """Fractional round-trip cost (e.g. crypto taker 0.0024). The adapter folds in spread/commission/impact/
        borrow/funding/calibrated-fill as appropriate for ITS market -- the pipeline only needs the honest number."""
        ...


# Onboarding a new market (the repeatable steps):
#   1. implement a MarketAdapter (+ CostModel) for the market's data + costs.
#   2. `python -m framework.pipeline init <market> _market`
#   3. run stages 00->02 (decompose with the 8-axis lattice; mine; the SAME candidate_gate apparatus is reused).
#   4. per instrument: `init <market> <SYMBOL>`, run stages 03->06 through the gates.
# The dead-list, gates, storage, meta-layer, and autonomy loop all carry over unchanged.

CRYPTO_ADAPTER_NOTE = (
    "Crypto's adapter is realized (not yet wrapped in this Protocol): universe=config/universes/*.yaml; "
    "load=pipeline.chimera_loader.ChimeraLoader; cost=src/strat fill/cost model + config/maker_cost_calibration.yaml; "
    "cadences=1d/4h/1h/30m/15m/dollar/dib; feature_families=narrate.feature_map. Wrapping it in CryptoAdapter(MarketAdapter) "
    "is a small, mechanical follow-up if/when a second market is onboarded."
)
