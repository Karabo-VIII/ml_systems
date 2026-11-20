"""ChimeraLoader -- strategy-facing data API.

Single entrypoint for strategies to load chimera data:
    from pipeline.chimera_loader import ChimeraLoader
    loader = ChimeraLoader()
    df = loader.load("BTCUSDT", cadence="1d")                  # default: v51 1d view
    df = loader.load("BTCUSDT", cadence="dollar")              # raw dollar bars (v51)
    df = loader.load_universe("u50", cadence="1d")             # all U50 assets, daily
    df = loader.load_universe("u10", cadence="1d", features=["close", "etf_btc_etf_total_z30"])

What it abstracts:
  - v50 vs v51 selection (always prefers v51 if available)
  - Cadence resolution (1d / 4h / 1h / 15m / dollar)
  - Universe filtering (returns only U10 / U50 / U100 members)
  - Date range subsetting
  - Feature subsetting (single columns argument; performance win)
  - Universe membership flags (is_u10/u50/u100) + DNA bucket: returned via the
    LoadResult metadata when load(..., with_meta=True). NOTE: load_universe()
    returns a plain panel DataFrame (with an 'asset' column) and does NOT inject
    per-row membership/DNA columns.

Strategies should NOT do `pl.read_parquet("data/processed/...")` directly anymore.
This is the canonical import.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from bar_fabric import BarFabric  # noqa: E402
from universe_loader import UniverseLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CADENCE_MAP = {
    "dollar": "v51",     # raw dollar bars (v51 = v50 + frontier features)
    "1d": "v51_1d",
    "4h": "v51_4h",
    "1h": "v51_1h",
    "30m": "v51_30m",
    "15m": "v51_15m",
    # 2026-05-29 L2: feature-enriched ALT chart types (same 184-feature surface on
    # dib/runs/range/adaptive_vol bars) -> built by make_chimera_bars.py. Lets the
    # strat/WM layer mine any indicator on any chart type under feature conditioning:
    #   ChimeraLoader().load("PEPEUSDT", cadence="dib", features=[...])
    "dib": "v51_dib",
    "runs_tick": "v51_runs_tick",
    "runs_volume": "v51_runs_volume",
    "range": "v51_range",
    "adaptive_vol": "v51_adaptive_vol",
}


@dataclass
class LoadResult:
    """Wrapped chimera load result with metadata."""
    df: pl.DataFrame
    symbol: str
    cadence: str
    asset_dna: str
    is_u10: bool
    is_u50: bool
    is_u100: bool
    n_rows: int
    n_features: int


class ChimeraLoader:
    """Strategy-facing single API for chimera reads."""

    def __init__(self, project_root: Path = PROJECT_ROOT,
                 prefer_v51: bool = True):
        self.root = project_root
        self.bf = BarFabric(project_root)
        self.universes = UniverseLoader.load()
        self.prefer_v51 = prefer_v51

    def load(self, symbol: str,
             cadence: str = "dollar",
             features: list[str] | None = None,
             date_range: tuple[str, str] | None = None,
             universe: str | None = None,
             with_meta: bool = False) -> pl.DataFrame | LoadResult:
        """Load chimera for a single symbol at the requested cadence.

        Args:
            symbol: BTC or BTCUSDT
            cadence: 'dollar' | '1d' | '4h' | '1h' | '30m' | '15m'
            features: column subset for performance. Always includes timestamp.
            date_range: (start_iso, end_iso) tuple for date filter
            universe: 'u10' | 'u50' | 'u100' to assert membership; raises if not member
            with_meta: if True, return LoadResult with metadata; else just DataFrame.
        """
        sym_u = symbol.upper()
        if not sym_u.endswith("USDT"):
            sym_u += "USDT"

        if universe and not self.universes.is_in(sym_u, universe):
            raise ValueError(f"{sym_u} is not in {universe}")

        bar_type = CADENCE_MAP.get(cadence)
        if bar_type is None:
            raise ValueError(f"unknown cadence: {cadence}; known={list(CADENCE_MAP.keys())}")

        cols = features
        if cols is not None:
            # Always include timestamp + key index columns if present.
            # Preserve the caller's requested order (set() would scramble it and
            # break models that expect features in a fixed column order); append
            # index cols only if not already requested.
            keep = ["timestamp", "bar_id"]
            ordered = list(cols) + [k for k in keep if k not in cols]
            seen: set = set()
            cols = [c for c in ordered if not (c in seen or seen.add(c))]

        try:
            df = self.bf.load(sym_u, bar_type, columns=cols, date_range=date_range)
        except FileNotFoundError:
            if self.prefer_v51 and bar_type.startswith("v51"):
                # fallback to legacy v50 dollar
                if cadence == "dollar":
                    df = self.bf.load(sym_u, "dollar", columns=cols, date_range=date_range)
                    if features is not None:
                        missing = [c for c in features if c not in df.columns]
                        if missing:
                            print(f"[chimera_loader] WARN {sym_u}: v51 absent, fell back "
                                  f"to v50 dollar; {len(missing)} requested frontier "
                                  f"feature(s) unavailable: {missing[:8]}", flush=True)
                else:
                    raise
            else:
                raise

        if with_meta:
            return LoadResult(
                df=df,
                symbol=sym_u,
                cadence=cadence,
                asset_dna=self.universes.dna_for(sym_u),
                is_u10=self.universes.is_in(sym_u, "u10"),
                is_u50=self.universes.is_in(sym_u, "u50"),
                is_u100=self.universes.is_in(sym_u, "u100"),
                n_rows=len(df),
                n_features=len(df.columns),
            )
        return df

    def load_universe(self, universe: str = "u50",
                      cadence: str = "1d",
                      features: list[str] | None = None,
                      date_range: tuple[str, str] | None = None,
                      add_asset_col: bool = True,
                      skip_missing: bool = True) -> pl.DataFrame:
        """Load and concatenate chimera for all assets in a universe.

        Returns a single panel DataFrame with an 'asset' column.
        """
        symbols = self.universes.list(universe)
        frames = []
        skipped = []
        for sym in symbols:
            try:
                df = self.load(sym, cadence=cadence, features=features, date_range=date_range)
                if add_asset_col:
                    df = df.with_columns(pl.lit(sym).alias("asset"))
                frames.append(df)
            except FileNotFoundError:
                if skip_missing:
                    skipped.append(sym)
                    continue
                raise
        if skipped:
            print(f"[chimera_loader] WARN {universe}/{cadence}: {len(skipped)}/"
                  f"{len(symbols)} assets missing chimera, skipped: {skipped[:12]}",
                  flush=True)
        if not frames:
            raise FileNotFoundError(f"no chimera found for any asset in {universe}")
        # Schema-align before concat (in case cadences differ)
        common_cols = set(frames[0].columns)
        for f in frames[1:]:
            common_cols &= set(f.columns)
        common_cols = sorted(common_cols)
        return pl.concat([f.select(common_cols) for f in frames], how="vertical_relaxed")


def main():
    """Smoke test."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--universe", default=None)
    args = ap.parse_args()
    loader = ChimeraLoader()
    if args.universe:
        df = loader.load_universe(args.universe, cadence=args.cadence)
        print(f"Universe {args.universe}, cadence {args.cadence}: {df.shape}, "
              f"{df['asset'].n_unique()} assets")
    else:
        result = loader.load(args.asset, cadence=args.cadence, with_meta=True)
        print(f"Asset {result.symbol}, cadence {result.cadence}:")
        print(f"  rows: {result.n_rows:,}")
        print(f"  features: {result.n_features}")
        print(f"  dna: {result.asset_dna}")
        print(f"  in u10/u50/u100: {result.is_u10}/{result.is_u50}/{result.is_u100}")


if __name__ == "__main__":
    main()
