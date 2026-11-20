"""Read v2 gold. This is the loader strat and world-model code should call.

``ChimeraLoader`` still reads v1 gold through the archive. New training and
research should use :func:`load_gold` (or ``data_api.load_v2``) instead of
``from pipeline.chimera_loader import ChimeraLoader``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pipeline_v2.paths import default_data_root
from pipeline_v2.symbols import canonical_symbol


def gold_dir(root: Path, bar_type: str) -> Path:
    return root / "gold" / bar_type


def latest_gold_path(root: Path, asset: str, bar_type: str) -> Path | None:
    asset = canonical_symbol(asset)
    directory = gold_dir(root, bar_type)
    if not directory.exists():
        return None
    files = sorted(p for p in directory.glob(f"{asset}_*.parquet") if p.is_file())
    return files[-1] if files else None


def load_gold(
    asset: str,
    bar_type: str = "time_1h",
    *,
    root: Path | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load the latest gold file for one asset and bar type.

    Raises FileNotFoundError when nothing has been built. Does not fall back
    to v1 chimera files.
    """
    root = root or default_data_root()
    path = latest_gold_path(root, asset, bar_type)
    if path is None:
        raise FileNotFoundError(
            f"no v2 gold for {canonical_symbol(asset)} {bar_type} under {root / 'gold'}. "
            f"Build it with: python -m pipeline_v2.cli run --assets {canonical_symbol(asset)} "
            f"--bar-types {bar_type}"
        )
    df = pl.read_parquet(path)
    if columns is not None:
        have = [c for c in columns if c in df.columns]
        df = df.select(have)
    return df


def load_gold_universe(
    assets: list[str],
    bar_type: str = "time_1h",
    *,
    root: Path | None = None,
    columns: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for asset in assets:
        try:
            out[canonical_symbol(asset)] = load_gold(asset, bar_type, root=root, columns=columns)
        except FileNotFoundError as exc:
            print(f"[gold] {exc}")
    return out
