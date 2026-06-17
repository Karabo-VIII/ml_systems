"""Bar Fabric -- unified loader for all bar types per asset.

Single API to load any bar type for any asset:
    from pipeline.bar_fabric import BarFabric
    bf = BarFabric()
    df = bf.load("BTCUSDT", "dollar")          # v50 dollar bars (from chimera_legacy)
    df = bf.load("BTCUSDT", "dib")             # DIB bars
    df = bf.load("BTCUSDT", "runs_tick")       # tick runs
    df = bf.load("BTCUSDT", "v51_1d")          # v51 daily-cadence chimera
    info = bf.list_available("BTCUSDT")        # what bar types exist for this asset

Layouts (post-2026-04-26 cleanup, see src/pipeline/layout.py):
    dollar       -> data/processed/chimera_legacy/<sym>usdt_v50_chimera_<DATE>.parquet
    v51          -> data/processed/chimera/<sym>usdt_v51_chimera_<DATE>.parquet
    v51_<cad>    -> data/processed/chimera/<sym>usdt_v51_chimera_<cad>_<DATE>.parquet
    dib/runs_tick/runs_volume/range/adaptive_vol -> data/processed/bars/<sym>usdt_<type>_<DATE>.parquet

The `year` argument is no longer used (bars are now consolidated per asset).

All returns are polars DataFrames (lazy-loadable on request).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

__contract__ = {
    "kind": "loader",
    "inputs": ["data/processed/{chimera,chimera_legacy,bars}/..."],
    "outputs": {"callable": "BarFabric.load(symbol, bar_type, columns=, date_range=, lazy=)"},
    "invariants": [
        "date_range filter applied in BOTH eager and lazy paths",
        "columns order preserved (delegated to pl.read_parquet)",
    ],
}

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import layout as _layout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"


KNOWN_BAR_TYPES = (
    "dollar",       # legacy v50 dollar bars
    "v51",          # v51 chimera (dollar bars + 80 frontier features)
    "v51_1d", "v51_4h", "v51_1h", "v51_30m", "v51_15m",  # cadence views
    # feature-enriched alt-bar chimeras (2026-05-29 L2):
    "v51_dib", "v51_runs_tick", "v51_runs_volume", "v51_range", "v51_adaptive_vol",
    # raw (feature-less) alt bars:
    "dib",
    "runs_tick", "runs_volume",
    "range",
    "adaptive_vol",
)


@dataclass
class BarTypeInfo:
    bar_type: str
    available: bool
    rows: int = 0
    files: list[Path] = None
    schema: dict | None = None


class BarFabric:
    """Single point of access for all bar types."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.root = project_root
        self.processed = project_root / "data" / "processed"

    def _resolve_path(self, symbol: str, bar_type: str, year: int | None = None) -> Path:
        # year is accepted for backward-compat but ignored: bars are consolidated.
        if bar_type == "dollar":
            p = _layout.chimera_v50_latest(symbol)
        elif bar_type == "v51":
            p = _layout.chimera_v51_latest(symbol, "dollar")
        elif bar_type in ("v51_1d", "v51_4h", "v51_1h", "v51_30m", "v51_15m",
                          "v51_dib", "v51_runs_tick", "v51_runs_volume",
                          "v51_range", "v51_adaptive_vol"):
            cadence = bar_type.split("_", 1)[1]   # 'v51_runs_tick' -> 'runs_tick'
            p = _layout.chimera_v51_latest(symbol, cadence)
        elif bar_type in ("dib", "runs_tick", "runs_volume", "range", "adaptive_vol"):
            p = _layout.bars_latest(symbol, bar_type)
        else:
            raise ValueError(f"unknown bar_type: {bar_type}; known={KNOWN_BAR_TYPES}")
        if p is None:
            # Return a deterministic non-existent path for callers that test .exists()
            sym_l, _ = _layout.normalize_asset(symbol)
            return self.processed / "MISSING" / f"{sym_l}_{bar_type}.parquet"
        return p

    def load(self, symbol: str, bar_type: str = "v51", year: int | None = None,
             columns: list[str] | None = None,
             date_range: tuple[str, str] | None = None,
             lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """Load bars for (symbol, bar_type).

        Args:
            symbol: BTC / BTCUSDT / etc. (USDT suffix added if missing)
            bar_type: see KNOWN_BAR_TYPES.
            year: for year-partitioned types (dib, range); None = latest available.
            columns: subset to read (faster). None = all columns.
            date_range: (start, end) ISO strings to filter; works on any df with a
                'date' or 'timestamp' column.
            lazy: return LazyFrame instead of DataFrame.

        Returns:
            DataFrame or LazyFrame.
        """
        path = self._resolve_path(symbol, bar_type, year)
        if not path.exists():
            raise FileNotFoundError(f"no bars for {symbol} {bar_type} (year={year}): {path}")
        if lazy:
            df = pl.scan_parquet(path)
            if columns:
                df = df.select(columns)
            if date_range:
                df = self._filter_date_range_lazy(df, date_range)
            return df
        df = pl.read_parquet(path, columns=columns)
        if date_range:
            df = self._filter_date_range(df, date_range)
        return df

    def _filter_date_range(self, df: pl.DataFrame, date_range: tuple[str, str]) -> pl.DataFrame:
        from datetime import datetime as _dt, date as _date
        start, end = date_range
        if "timestamp" in df.columns and df["timestamp"].dtype.is_integer():
            # epoch ms
            ts_start = int(_dt.fromisoformat(start).timestamp() * 1000)
            ts_end = int(_dt.fromisoformat(end).timestamp() * 1000)
            return df.filter(pl.col("timestamp").is_between(ts_start, ts_end))
        if "date" in df.columns:
            sd = _date.fromisoformat(start)
            ed = _date.fromisoformat(end)
            return df.filter(pl.col("date").is_between(sd, ed))
        return df

    def _filter_date_range_lazy(self, df: pl.LazyFrame, date_range: tuple[str, str]) -> pl.LazyFrame:
        # Previously a no-op passthrough that silently returned ALL rows when
        # lazy=True + date_range was set. Now applies the same filter as the
        # eager path (on epoch-ms `timestamp`, else `date`).
        from datetime import datetime as _dt, date as _date
        start, end = date_range
        names = df.collect_schema().names()
        if "timestamp" in names:
            ts_start = int(_dt.fromisoformat(start).timestamp() * 1000)
            ts_end = int(_dt.fromisoformat(end).timestamp() * 1000)
            return df.filter(pl.col("timestamp").is_between(ts_start, ts_end))
        if "date" in names:
            sd = _date.fromisoformat(start)
            ed = _date.fromisoformat(end)
            return df.filter(pl.col("date").is_between(sd, ed))
        return df

    def list_available(self, symbol: str) -> dict[str, BarTypeInfo]:
        """Return bar-type availability map for a symbol."""
        out = {}
        for bt in KNOWN_BAR_TYPES:
            try:
                p = self._resolve_path(symbol, bt)
            except Exception:
                continue
            available = p.exists()
            info = BarTypeInfo(bar_type=bt, available=available, files=[])
            if available:
                try:
                    schema = pl.read_parquet_schema(p)
                    info.rows = pl.read_parquet(p, columns=[next(iter(schema))]).height
                    info.schema = {k: str(v) for k, v in schema.items()}
                    info.files = [p]
                except Exception:
                    pass
            out[bt] = info
        return out

    def list_universe_assets(self, bar_type: str = "v51") -> list[str]:
        """Return all assets that have data for this bar_type (post-2026-04-26 layout)."""
        if bar_type == "dollar":
            return _layout.list_v50_assets()
        if bar_type == "v51":
            return _layout.list_v51_assets()
        if bar_type.startswith("v51_"):
            cadence = bar_type.split("_", 1)[1]
            out = []
            for sym_u in _layout.list_v51_assets():
                if _layout.chimera_v51_latest(sym_u, cadence) is not None:
                    out.append(sym_u)
            return sorted(out)
        # bars (post-2026-04-26 v3: each bartype in its own subfolder)
        try:
            d = _layout.bars_dir(bar_type)
        except ValueError:
            return []
        if not d.exists():
            return []
        out = set()
        for f in d.glob(f"*_{bar_type}_*.parquet"):
            stem = f.stem
            if "_" in stem:
                without_date = stem.rsplit("_", 1)[0]
                tag = f"_{bar_type}"
                if without_date.endswith(tag):
                    sym_l = without_date[: -len(tag)]
                    out.add(sym_l.upper())
        return sorted(out)


def main():
    """CLI: print bar-type inventory for an asset."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    args = ap.parse_args()
    bf = BarFabric()
    sym = args.asset.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    print(f"BarFabric inventory for {sym}:")
    info = bf.list_available(sym)
    for bt, i in info.items():
        if i.available:
            print(f"  {bt:14s} {i.rows:>10,} rows  {i.files[0].relative_to(PROJECT_ROOT) if i.files else ''}")
        else:
            print(f"  {bt:14s} (not available)")


if __name__ == "__main__":
    main()
