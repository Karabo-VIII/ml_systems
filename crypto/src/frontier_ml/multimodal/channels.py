"""Multi-modal side channels with explicit lag (Hole 6 closure).

Reads from existing pipeline panels and per-asset chimera_legacy:
    - chimera norm_funding       (8h cadence, fwd-filled to bar cadence)
    - chimera norm_oi_change     (continuous)
    - panels/etf_flows           (daily; for BTC/ETH only)
    - panels/multi_venue_listings (event-driven)
    - data/raw_external/*        (DXY, S&P -- if present)

Public API:
    ChannelBank(asset_id, lag_bars=1).load_aligned(timestamps) -> dict[name, np.ndarray]

Each channel carries an explicit lag in bars (default 1). The
WalkForwardSplitter purge gap MUST be >= max(channel.lag_bars * bar_seconds_to_days)
or the longest lookback window used in feature construction (whichever is greater)
to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANELS_DIR = PROJECT_ROOT / "data" / "processed" / "panels" / "daily"
LEGACY_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"


@dataclass
class ChannelSpec:
    name: str
    source: str           # 'chimera' | 'panel'
    column: str
    lag_bars: int = 1     # bars of explicit lag (Hole 6 hygiene)
    fillna: float = 0.0   # safe fillna AFTER alignment


# Default channel set. Add/remove rows here to expand the modality.
# All values are PANEL- or PER-ASSET-CHIMERA-sourced; no live API calls.
DEFAULT_CHANNELS: List[ChannelSpec] = [
    ChannelSpec(name="funding",    source="chimera", column="norm_funding",    lag_bars=1),
    ChannelSpec(name="oi_delta",   source="chimera", column="norm_oi_change", lag_bars=1),
    ChannelSpec(name="hawkes_imb", source="chimera", column="norm_hawkes_imbalance", lag_bars=1),
    ChannelSpec(name="vpin",       source="chimera", column="norm_vpin",       lag_bars=1),
    # Daily panels (joined by date):
    ChannelSpec(name="etf_btc_total", source="panel", column="Total",         lag_bars=1, fillna=0.0),
]


@dataclass
class ChannelBank:
    asset: str
    lag_bars_default: int = 1
    specs: List[ChannelSpec] = field(default_factory=lambda: list(DEFAULT_CHANNELS))

    _chimera_df: Optional[pl.DataFrame] = field(default=None, init=False, repr=False)
    _panels: Dict[str, pl.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def _load_chimera(self) -> pl.DataFrame:
        if self._chimera_df is None:
            cands = sorted(LEGACY_DIR.glob(f"{self.asset.lower()}usdt_v50_chimera_*.parquet"))
            if not cands:
                raise FileNotFoundError(f"chimera_legacy not found for {self.asset}")
            chim_cols = ["timestamp"] + [s.column for s in self.specs if s.source == "chimera"]
            chim_cols = list(dict.fromkeys(chim_cols))  # dedupe preserving order
            self._chimera_df = pl.read_parquet(cands[-1], columns=chim_cols)
        return self._chimera_df

    def _load_panel(self, name: str) -> Optional[pl.DataFrame]:
        # Panel naming convention: data/processed/panels/daily/<panel>.parquet
        # E.g. etf_btc_total -> btc_etf_flows.parquet
        if name in self._panels:
            return self._panels[name]
        candidates = {
            "etf_btc_total": PANELS_DIR / "btc_etf_flows.parquet",
            "etf_eth_total": PANELS_DIR / "eth_etf_flows.parquet",
        }
        p = candidates.get(name)
        if p is None or not p.exists():
            return None
        df = pl.read_parquet(p)
        self._panels[name] = df
        return df

    def load_aligned(self, timestamps_ms: np.ndarray) -> Dict[str, np.ndarray]:
        """For an array of bar-close timestamps (ms), return per-channel arrays
        with the explicit lag applied (we read each value at timestamps_ms - lag).

        Returns:
            dict[channel_name] -> (N,) float32 array.
        """
        out: Dict[str, np.ndarray] = {}
        chim = self._load_chimera()
        ts_chim = chim["timestamp"].to_numpy()

        for spec in self.specs:
            if spec.source == "chimera":
                # join_asof (backward) to find each timestamp's bar value
                col = chim[spec.column].to_numpy().astype(np.float32)
                # apply lag: shift the OUTPUT by `lag_bars` bars
                lagged = np.empty_like(col)
                if spec.lag_bars > 0:
                    lagged[:spec.lag_bars] = spec.fillna
                    lagged[spec.lag_bars:] = col[:-spec.lag_bars]
                else:
                    lagged = col
                # asof lookup at each requested timestamp
                idx = np.searchsorted(ts_chim, timestamps_ms, side="right") - 1
                idx = np.clip(idx, 0, len(lagged) - 1)
                vals = lagged[idx]
                out[spec.name] = np.nan_to_num(vals, nan=spec.fillna)

            elif spec.source == "panel":
                df = self._load_panel(spec.name)
                if df is None:
                    out[spec.name] = np.full(len(timestamps_ms), spec.fillna, dtype=np.float32)
                    continue
                # Panel rows are daily (date column). Convert ms -> day, asof.
                if "date" not in df.columns:
                    out[spec.name] = np.full(len(timestamps_ms), spec.fillna, dtype=np.float32)
                    continue
                dts = df["date"].to_numpy()
                # Convert dates to int64 ns timestamps for searchsorted
                dts_ms = (dts.astype("datetime64[ms]").astype(np.int64))
                col = df[spec.column].to_numpy().astype(np.float32) if spec.column in df.columns else np.full(len(df), spec.fillna, dtype=np.float32)
                # Lag in BARS isn't the right unit for daily panels; treat lag_bars
                # as days here (smallest meaningful daily lag). Default 1 bar => 1 day.
                lag_days = max(1, spec.lag_bars)
                # shift down by `lag_days` rows
                lagged = np.empty_like(col)
                if lag_days < len(col):
                    lagged[:lag_days] = spec.fillna
                    lagged[lag_days:] = col[:-lag_days]
                else:
                    lagged[:] = spec.fillna
                idx = np.searchsorted(dts_ms, timestamps_ms, side="right") - 1
                idx = np.clip(idx, 0, len(lagged) - 1)
                vals = lagged[idx]
                out[spec.name] = np.nan_to_num(vals, nan=spec.fillna)

        return out

    @property
    def n_channels(self) -> int:
        return len(self.specs)

    @property
    def channel_names(self) -> List[str]:
        return [s.name for s in self.specs]


def smoke():
    """Verify ChannelBank works on BTC."""
    bank = ChannelBank(asset="BTC")
    chim = bank._load_chimera()
    ts = chim["timestamp"].to_numpy()[-100:]
    out = bank.load_aligned(ts)
    print(f"[mm-channels] BTC channels:")
    for name, arr in out.items():
        finite = np.isfinite(arr).sum()
        print(f"   {name:20s}  shape={arr.shape}  finite={finite}/{len(arr)}  "
              f"mean={arr.mean():+.4f}  std={arr.std():.4f}")


if __name__ == "__main__":
    smoke()
