"""Market source interface. Crypto is implemented. FX is a slot, not an adapter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class MarketSource(Protocol):
    """What a venue must provide.

    An FX adapter can implement this later (session calendar, weekend closes).
    No FX adapter is shipped. Crypto-only series (funding, open interest) simply
    do not register for instruments that are not crypto.
    """

    market: str

    def list_instruments(self) -> list[str]:
        ...

    def fetch_ticks_or_bars(self, instrument: str, start: str, end: str) -> list[pl.DataFrame]:
        ...

    def calendar(self) -> str:
        """``24/7`` for crypto. FX would return a session calendar id."""
        ...

    def default_lags(self) -> dict[str, int]:
        ...
