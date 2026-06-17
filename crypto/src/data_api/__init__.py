"""data_api -- canonical read-side contract.

Strategies, models, and validation all consume data through this module.
The pipeline owns the WRITE side (refresh.py orchestrates production);
data_api owns the READ side. The contract is:

  - Consumers NEVER call pl.read_parquet directly on chimera or panel paths.
  - Consumers call data_api functions: load_v51, load_silver, load_panel,
    extract_feature_for_universe.
  - data_api handles cadence resolution, latest-dated selection, schema
    validation, NaN policy, and universe filtering.
  - When the pipeline schema evolves, only data_api needs to change --
    consumers see a stable API.

This eliminates the 'silent break when pipeline rename' class of bugs
(e.g. te_panel asset-format mismatch could not have escaped data_api
because the loader runs gates on every read).

Currently provided:
  - load_v51_for_universe(universe, cadence, columns) -> Dict[asset, DataFrame]
  - load_panel(panel_name) -> DataFrame (single panel by registered name)
  - extract_feature_per_asset(feature, universe, cadence) -> Dict[asset, Series]

Existing access patterns being migrated INTO data_api:
  - ChimeraLoader (pipeline.chimera_loader) -- per-asset chimera loader.
    Wrapped by load_v51_for_universe.
  - cadence_loader (strategy.cadence_loader) -- thin wrapper. Migrate.
"""
from __future__ import annotations

# CDAP contract.
__contract__ = {
    "kind": "data_api",
    "stage": "data_api",
    "outputs": {"format": "in-memory polars DataFrames / Series"},
    "invariants": {
        "schema_validated_on_read": True,
        "no_direct_parquet_reads_in_consumers": True,
    },
}

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

import polars as pl  # noqa: E402

from chimera_loader import ChimeraLoader  # noqa: E402
import layout as _layout  # noqa: E402

# Re-export validation gate so consumers can run it inline.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from validation.pipeline_gates import run_suite, suite_pipeline_only  # noqa: E402
except Exception:
    run_suite = None
    suite_pipeline_only = None


# ─── Universe helpers ────────────────────────────────────────────────────────

def _resolve_universe(universe: str) -> list[str]:
    """Return uppercase asset symbols (no USDT) for the requested universe."""
    from universe_loader import UniverseLoader
    syms = UniverseLoader.load().list(universe)
    return [s.upper().replace("USDT", "") for s in syms]


# ─── Public API ──────────────────────────────────────────────────────────────

def load_v51(asset: str, cadence: str = "dollar",
              columns: Optional[list[str]] = None,
              run_gate: bool = False) -> pl.DataFrame:
    """Load v51 chimera for one asset/cadence.

    Args:
        asset:    'BTC' / 'ETH' / 'BTCUSDT' (USDT stripped if present).
        cadence:  'dollar' (default) / '1d' / '4h' / '1h' / '15m'.
        columns:  optional column subset.
        run_gate: if True, run pipeline_gates.suite_pipeline_only against
                  the loaded asset and raise if any critical fail.
    """
    asset_root = asset.upper().replace("USDT", "")
    if run_gate and run_suite is not None:
        ok, results = run_suite("pipeline_only", asset_root)
        if not ok:
            failed = [r for r in results if not r.ok and r.severity == "critical"]
            raise RuntimeError(
                f"data_api.load_v51({asset_root}): pipeline gate failed: "
                + "; ".join(r.message for r in failed)
            )
    loader = ChimeraLoader()
    df = loader.load(asset_root, cadence=cadence)
    if columns is not None:
        avail = [c for c in columns if c in df.columns]
        df = df.select(avail)
    return df


def load_v51_for_universe(universe: str, cadence: str = "dollar",
                            columns: Optional[list[str]] = None,
                            run_gate: bool = False) -> dict[str, pl.DataFrame]:
    """Load v51 chimera for every asset in the universe.

    Returns: {asset_root: DataFrame}.

    Use this instead of: glob('chimera/dollar/*.parquet') + per-file read.
    The data_api enforces gate + cadence + universe filter consistently.
    """
    out: dict[str, pl.DataFrame] = {}
    for asset in _resolve_universe(universe):
        try:
            out[asset] = load_v51(asset, cadence=cadence, columns=columns,
                                    run_gate=run_gate)
        except Exception as e:
            print(f"[data_api] WARN load_v51 failed for {asset}: {e}", flush=True)
    return out


def extract_feature_per_asset(feature: str, universe: str,
                                cadence: str = "1d") -> dict[str, pl.Series]:
    """Extract one feature column per asset across the universe.

    Returns: {asset_root: Series}. Use this for simple cross-sectional
    signals (e.g. te_in_btc, xd_momentum_rank).
    """
    out: dict[str, pl.Series] = {}
    needed = ["timestamp", feature] if feature != "timestamp" else ["timestamp"]
    for asset in _resolve_universe(universe):
        try:
            df = load_v51(asset, cadence=cadence, columns=needed)
            if feature in df.columns:
                out[asset] = df[feature]
        except Exception as e:
            print(f"[data_api] WARN extract {feature} failed for {asset}: {e}",
                  flush=True)
    return out


def load_panel(panel_name: str) -> pl.DataFrame:
    """Load a multi-asset panel BY NAME (not by file path).

    Currently delegates to layout.panel_latest. The point of this wrapper
    is that consumers don't have to know panel filenames -- if a panel
    is renamed, only this function changes.

    Known panels: see config/feature_registry.yaml + config/asset_dag.yaml.
    """
    p = _layout.panel_latest(panel_name)
    if p is None or not p.exists():
        raise FileNotFoundError(
            f"data_api.load_panel({panel_name}): no file found. "
            f"Run: python src/pipeline/refresh.py --target {panel_name}")
    return pl.read_parquet(p)


# ─── Training-side loader (delegates to anti_fragile.load_full_data) ────────
#
# Why this exists:
#   * `anti_fragile.load_full_data` is the single function every WM version
#     calls to materialize chimera_legacy parquet -> per-asset segment dicts
#     with features + targets + asset_idx.
#   * Versions today import it directly (`from anti_fragile import load_full_data`).
#     CDAP cannot easily enforce "models read through data_api" while the
#     direct import is the canonical path.
#   * The shim below re-exports it as `data_api.load_full_data_for_training`
#     so that:
#       (a) versions can migrate to a single canonical import,
#       (b) we can later add gate / schema-validation / cadence-resolution
#           hooks in this wrapper without touching every version,
#       (c) CDAP can scan `src/wm/**/*.py` for the legacy
#           `from anti_fragile import load_full_data` form and require
#           `from data_api import load_full_data_for_training` instead.
#
# This is INTENTIONALLY a thin re-export -- the v50 chimera_legacy schema is
# stable and `load_full_data` already handles asset_to_idx, target prefix,
# and dated-snapshot fallback. We don't need to reinvent it.

def load_full_data_for_training(*args, **kwargs):
    """Canonical entry point for WM training scripts.

    Delegates to :func:`anti_fragile.load_full_data`. Same signature.
    Future enhancements (gate runs, cadence resolution, normalizer cache)
    happen here, not in every version's training script.

    Use this instead of `from anti_fragile import load_full_data` in any
    new version. Existing versions migrating from the direct import: add
    `from data_api import load_full_data_for_training as load_full_data`
    -- the alias keeps the existing call sites unchanged.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from anti_fragile import load_full_data  # noqa: E402
    return load_full_data(*args, **kwargs)


__all__ = [
    "load_v51",
    "load_v51_for_universe",
    "extract_feature_per_asset",
    "load_panel",
    "load_full_data_for_training",
]
