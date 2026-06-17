"""src/narrate/ -- the DESCRIPTIVE market-intelligence foundation ("the what", before any strategy).

This package answers a different question than `src/strat/` (which VALIDATES a candidate edge) or the WM
(which PREDICTS returns). It answers: *given an asset, a period, and a chart type -- what is the market DOING?*
It narrates state, structure, regime, flow, positioning, and notable events by decomposing ALL of chimera into
human-readable family reads, optionally augmented by our own trained artifacts and a downloaded time-series
foundation model (MOMENT) -- each validated against what we already know.

DESIGN STANCE (user mandate 2026-06-06):
  - DESCRIPTIVE, not predictive. Price is the hard thing; this layer narrates the WHAT, it does not forecast.
  - ENTRY-SIGNAL framing only. We hunt conditions that precede setups. The EXIT is a separate decomposable
    domain (trailing / fixed / volatility) and is explicitly OUT OF SCOPE here.
  - PER-SETUP, not per-candle. A read describes a multi-candle STATE, never a single-bar prediction.
  - Chart-type aware. The same period on time bars vs dollar/dib/range bars tells different stories; the engine
    can narrate each and compare what each chart "sees".

Public entry point: narrate(asset, cadence, start, end) -> MarketNarration (structured + prose).
"""
from .narrator import narrate, MarketNarration  # noqa: F401

__all__ = ["narrate", "MarketNarration"]
