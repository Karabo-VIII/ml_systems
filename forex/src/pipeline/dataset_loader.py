"""DatasetLoader — the canonical way to consume the gold dataset.

Mirrors crypto's ChimeraLoader invariant ("read via the loader, not raw
pl.read_parquet"): one API that resolves the latest dated file, drops purge
rows, filters by split, and separates feature columns from targets/metadata.

    from pipeline.dataset_loader import DatasetLoader as DL
    train = DL.load("EURUSD", "1h", split="train")      # ready-to-model frame
    X_cols = DL.feature_cols(train); y_cols = DL.target_cols(train)
    panel = DL.load_multi(["EURUSD", "GBPUSD"], "1h")    # stacked multi-pair
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout

__contract__ = {
    "kind": "framework_helper",
    "stage": "dataset_loader",
    "inputs": {"upstream": "data/processed/dataset/<cadence>/<pair>_dataset_*.parquet"},
    "outputs": {"api": "load / load_multi / feature_cols / target_cols"},
    "invariants": {"read_via_loader_not_raw": True, "purge_dropped_by_default": True,
                   "split_filter_supported": True},
    "rationale": "Single consumption API for the gold dataset (ChimeraLoader analog).",
}

_META_COLS = {"timestamp", "pair", "dna", "pip", "price_divisor", "split", "is_purge",
              "rate_base", "rate_quote"}


class DatasetLoader:
    @staticmethod
    def load(pair: str, cadence: str, *, split: str | None = None,
             drop_purge: bool = True) -> pl.DataFrame:
        """Load the latest gold dataset for (pair, cadence). Filters purge rows
        (default) and an optional split ('train'/'val'/'oos'/'unseen')."""
        path = layout.dataset_latest(pair, cadence)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"no dataset for {pair} {cadence}; run make_dataset.py first")
        df = pl.read_parquet(path)
        if drop_purge and "is_purge" in df.columns:
            df = df.filter(~pl.col("is_purge"))
        if split is not None:
            df = df.filter(pl.col("split") == split)
        return df

    @staticmethod
    def load_multi(pairs: list[str], cadence: str, *, split: str | None = None,
                   drop_purge: bool = True) -> pl.DataFrame:
        """Stack multiple pairs' datasets into one long panel (adds nothing;
        each row already carries its `pair`). Skips pairs with no dataset."""
        frames = []
        for p in pairs:
            try:
                frames.append(DatasetLoader.load(p, cadence, split=split,
                                                 drop_purge=drop_purge))
            except FileNotFoundError:
                continue
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")

    @staticmethod
    def target_cols(df: pl.DataFrame) -> list[str]:
        """The forward-return TARGET columns (the only non-causal columns)."""
        return [c for c in df.columns if c.startswith("fwd_")]

    @staticmethod
    def feature_cols(df: pl.DataFrame) -> list[str]:
        """Model input columns: everything that is neither a target nor metadata."""
        return [c for c in df.columns
                if c not in _META_COLS and not c.startswith("fwd_")]
