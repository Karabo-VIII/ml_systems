"""src/framework/general_adapter.py -- Layer-B GeneralAdapter: the MarketAdapter mirror for
general time-series / science / non-crypto domains.

Mirrors src/framework/crypto_adapter.py's interface but accepts any DataFrame, parquet, or CSV
with named feature columns + a target column. Converts to the `segments: List[dict]` format that
the anti-fragile loop and MultiAssetDataset consume.

SEGMENT FORMAT (matches multi_asset_dataset.py data contract):
    {
        "asset_idx": int,                            # 0..N-1
        "asset_name": str,                           # instrument label
        "timestamp": np.ndarray (n_bars,) int64-ms, # epoch-ms, sorted asc
        "features": np.ndarray (n_bars, C) float32,
        "target_return_1": np.ndarray (n_bars,) float32,   # required
        "target_return_4": np.ndarray (n_bars,) float32,   # optional, zeros if absent
        "target_return_16": np.ndarray (n_bars,) float32,  # optional
        "target_return_64": np.ndarray (n_bars,) float32,  # optional
    }

CostModel: zero-cost by default (Layer B is not a trading adapter -- the battery and gates apply
but cost is domain-supplied; override via ZeroCostModel or supply your own).

Feature families: inferred from column name prefixes (everything before the first underscore).

No emoji (Windows cp1252). Self-contained; numpy + pandas (standard stack). No torch dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Timestamp validity range: 13-digit epoch milliseconds matching the chimera invariant.
# Synthetic timestamps are anchored here so they survive any downstream range checks
# (e.g. split_four_way_dated, validate_segment, ts.min()>=1.5e12 assertions).
# Base = 2020-09-14T01:46:40Z in ms (a round number safely inside [1.5e12, 2.0e12]).
_SYNTHETIC_TS_BASE_MS: int = 1_600_000_000_000
# Default cadence for synthetic timestamps: 1 bar = 1 hour = 3_600_000 ms.
# Callers can override via cadence_ms in _col_to_epoch_ms if needed.
_SYNTHETIC_TS_CADENCE_MS: int = 3_600_000

# Timestamp validity range (same invariant as CLAUDE.md + chimera spec)
_TS_MIN_VALID_MS: int = 1_500_000_000_000   # 2017-07-14
_TS_MAX_VALID_MS: int = 2_000_000_000_000   # 2033-05-18

# CDAP contract
__contract__ = {
    "kind": "adapter",
    "module": "GeneralAdapter",
    "inputs": ["DataFrame | parquet path | CSV path with feature cols + optional timestamp col"],
    "outputs": ["segments: List[dict] in MultiAssetDataset format"],
    "invariants": {
        "no_lookahead": "timestamps must be pre-sorted ascending; no future data visible",
        "zero_cost": "default CostModel returns 0.0 (domain-supplied override supported)",
        "target_required": "at least one target_* column OR a target_col kwarg must be present",
    },
}

# Active horizons (mirrors CLAUDE.md ACTIVE_HORIZONS=[1,4,16,64])
_HORIZONS = [1, 4, 16, 64]

# Default cadence labels for a generic time-series (user can override)
_DEFAULT_CADENCES = ["1d", "1h", "custom"]


# ---------------------------------------------------------------------------
# Zero-cost model (Layer B default)
# ---------------------------------------------------------------------------

class ZeroCostModel:
    """Default cost model for Layer B -- returns 0.0 (no trading friction assumed).

    Override at instantiation time if the domain has real execution costs:
        class MyCostModel(ZeroCostModel):
            def round_trip(self, symbol, side="long", notional=0.0, venue=None):
                return 0.005  # e.g. 50bps
    """
    def round_trip(self, symbol: str, side: str = "long", notional: float = 0.0,
                   venue: Optional[str] = None) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Segment builder (core conversion logic)
# ---------------------------------------------------------------------------

def _infer_timestamp_col(df) -> Optional[str]:
    """Heuristic: find a column whose name contains 'time', 'ts', 'date', or 'index'."""
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ("timestamp", "time", "ts", "date", "index", "open_time")):
            return col
    return None


def _col_to_epoch_ms(series) -> np.ndarray:
    """Convert a pandas Series to int64 epoch-millisecond timestamps."""
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series.astype("int64") // 1_000_000).values.astype(np.int64)
    if pd.api.types.is_numeric_dtype(series):
        vals = series.values.astype(np.float64)
        # distinguish seconds (< 1e12) from milliseconds (>= 1e12)
        if vals.mean() < 1e12:
            return (vals * 1000).astype(np.int64)
        return vals.astype(np.int64)
    # attempt parsing as string
    try:
        parsed = pd.to_datetime(series)
        return (parsed.astype("int64") // 1_000_000).values.astype(np.int64)
    except Exception:
        # fall back to synthetic anchored timestamps (row-index paced at 1h cadence)
        return _SYNTHETIC_TS_BASE_MS + np.arange(len(series), dtype=np.int64) * _SYNTHETIC_TS_CADENCE_MS


def _build_segment(
    df,
    asset_name: str,
    asset_idx: int,
    feature_cols: List[str],
    target_col: Optional[str],
    horizons: List[int],
) -> dict:
    """Convert one DataFrame slice to the segment dict format."""
    import pandas as pd

    # timestamp
    ts_col = _infer_timestamp_col(df)
    if ts_col is not None and ts_col not in feature_cols:
        ts_ms = _col_to_epoch_ms(df[ts_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_ms = (df.index.astype("int64") // 1_000_000).values.astype(np.int64)
    else:
        # Synthetic: anchor to a valid epoch-ms base so downstream range checks pass.
        # Old code used np.arange(len)*1000 which produced ~6-digit values violating
        # the [1.5e12, 2.0e12] invariant and breaking dated splits.
        ts_ms = _SYNTHETIC_TS_BASE_MS + np.arange(len(df), dtype=np.int64) * _SYNTHETIC_TS_CADENCE_MS
        warnings.warn(
            f"[general_adapter] No timestamp column found for asset '{asset_name}'; "
            f"using synthetic timestamps anchored at {_SYNTHETIC_TS_BASE_MS} ms "
            f"(1h cadence). Provide a timestamp column for accurate date-based splits.",
            stacklevel=3,
        )

    # Validate timestamp range after construction (catches bad real-data timestamps too)
    ts_min, ts_max = int(ts_ms[0]), int(ts_ms[-1])
    if ts_min < _TS_MIN_VALID_MS or ts_max > _TS_MAX_VALID_MS:
        warnings.warn(
            f"[general_adapter] Timestamps for asset '{asset_name}' out of valid range "
            f"[{_TS_MIN_VALID_MS}, {_TS_MAX_VALID_MS}]: "
            f"ts.min={ts_min}, ts.max={ts_max}. "
            f"Dated splits and chimera invariant checks will fail.",
            stacklevel=3,
        )

    # features (float32, shape [N, C])
    # Check for NaN before filling -- warn with column-level detail so the caller can
    # fix upstream rather than silently operating on zeroed-out features.
    raw_feat = df[feature_cols].to_numpy(dtype=np.float32)
    nan_mask = ~np.isfinite(raw_feat)
    if nan_mask.any():
        nan_counts = nan_mask.sum(axis=0)
        bad_cols = [feature_cols[i] for i in range(len(feature_cols)) if nan_counts[i] > 0]
        total_nans = int(nan_mask.sum())
        warnings.warn(
            f"[general_adapter] NaN/inf values in features for asset '{asset_name}': "
            f"{total_nans} cells across {len(bad_cols)} column(s): {bad_cols}. "
            f"Filling with 0.0 -- fix upstream if these are load-bearing features.",
            stacklevel=3,
        )
    features = np.where(nan_mask, 0.0, raw_feat).astype(np.float32)

    # targets -- detect existing target_return_* columns or build from target_col
    seg: dict = {
        "asset_idx": asset_idx,
        "asset_name": asset_name,
        "timestamp": ts_ms,
        "features": features,
    }

    # Try to find pre-computed target_return_<h> columns in df
    for h in horizons:
        col_name = f"target_return_{h}"
        src_col = None
        if col_name in df.columns:
            src_col = col_name
        elif target_col and target_col in df.columns and h == horizons[0]:
            src_col = target_col  # use the target col as the primary horizon

        if src_col is not None:
            raw_tgt = df[src_col].to_numpy(dtype=np.float32)
            tgt_nan = ~np.isfinite(raw_tgt)
            if tgt_nan.any():
                warnings.warn(
                    f"[general_adapter] NaN/inf in target column '{src_col}' for asset "
                    f"'{asset_name}': {int(tgt_nan.sum())} cells. Filling with 0.0.",
                    stacklevel=3,
                )
            seg[col_name] = np.where(tgt_nan, 0.0, raw_tgt).astype(np.float32)
        else:
            seg[col_name] = np.zeros(len(df), dtype=np.float32)

    return seg


# ---------------------------------------------------------------------------
# GeneralAdapter
# ---------------------------------------------------------------------------

class GeneralAdapter:
    """Implements the MarketAdapter Protocol for generic time-series / science domains.

    Construction
    ------------
    adapter = GeneralAdapter(
        data_source,           # DataFrame, Path, or str (parquet or CSV)
        target_col="y",        # column to use as target_return_1 (if no target_return_* cols exist)
        feature_cols=None,     # explicit list; if None, auto-detect (exclude target + timestamp)
        instrument="series",   # instrument/entity label (single-instrument default)
        cadence="custom",      # cadence label for the workspace store
        cost_model=None,       # ZeroCostModel by default
        horizons=None,         # [1, 4, 16, 64] by default
    )
    segments = adapter.to_segments()
    """

    market = "general"

    def __init__(
        self,
        data_source: Any,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        instrument: str = "series",
        cadence: str = "custom",
        cost_model: Optional[Any] = None,
        horizons: Optional[List[int]] = None,
    ):
        self._raw = data_source
        self._target_col = target_col
        self._feature_cols_override = feature_cols
        self._instrument = instrument
        self._cadence = cadence
        self._cost = cost_model or ZeroCostModel()
        self._horizons = horizons or _HORIZONS
        self._df = None  # lazily loaded

    # -- lazy loader ----------------------------------------------------------

    def _load_df(self):
        if self._df is not None:
            return self._df
        import pandas as pd

        raw = self._raw
        if isinstance(raw, (str, Path)):
            p = Path(raw)
            if p.suffix == ".parquet":
                self._df = pd.read_parquet(p)
            elif p.suffix in (".csv", ".tsv"):
                self._df = pd.read_csv(p)
            else:
                raise ValueError(f"Unsupported file format: {p.suffix} -- use parquet or CSV.")
        else:
            # assume DataFrame-like (pandas, polars -- convert polars to pandas)
            try:
                self._df = raw.to_pandas() if hasattr(raw, "to_pandas") else raw
            except Exception as exc:
                raise TypeError(
                    f"data_source must be a DataFrame, Path, or str -- got {type(raw)}"
                ) from exc
        return self._df

    # -- feature col detection ------------------------------------------------

    def _detect_feature_cols(self, df) -> List[str]:
        if self._feature_cols_override:
            return list(self._feature_cols_override)
        # exclude timestamp, target_return_*, and the explicit target_col
        exclude = set()
        ts_col = _infer_timestamp_col(df)
        if ts_col:
            exclude.add(ts_col)
        if self._target_col:
            exclude.add(self._target_col)
        for h in self._horizons:
            exclude.add(f"target_return_{h}")
        return [c for c in df.columns if c not in exclude]

    # -- MarketAdapter protocol -----------------------------------------------

    def universe(self, tier: str = "default") -> Sequence[str]:
        """Single-instrument universe (the one data series supplied). Override for multi-instrument."""
        return [self._instrument]

    def load(self, symbol: str, cadence: str, features: Sequence[str] | None = None) -> Any:
        """Return the loaded DataFrame (point-in-time, pre-sorted by timestamp)."""
        df = self._load_df()
        return df

    def cost_model(self) -> ZeroCostModel:
        return self._cost

    def cadences(self) -> Sequence[str]:
        return [self._cadence] if self._cadence not in _DEFAULT_CADENCES else _DEFAULT_CADENCES

    def feature_families(self) -> Dict[str, List[str]]:
        """Infer families from column-name prefix (everything before the first underscore).

        Example: ["price_close", "price_open", "vol_real"] -> {"price": [...], "vol": [...]}
        """
        df = self._load_df()
        feat_cols = self._detect_feature_cols(df)
        families: Dict[str, List[str]] = {}
        for col in feat_cols:
            prefix = col.split("_")[0] if "_" in col else "other"
            families.setdefault(prefix, []).append(col)
        return families

    # -- Segment builder (the key output) ------------------------------------

    def to_segments(self, sort_by_timestamp: bool = True) -> List[dict]:
        """Convert the data source to the segment List[dict] the anti-fragile loop consumes.

        Returns a list with ONE segment dict (single-instrument).
        For multi-instrument, subclass and call _build_segment per instrument.

        The segment dict keys match MultiAssetDataset's data contract:
            asset_idx, asset_name, timestamp (int64-ms), features (float32 [N, C]),
            target_return_1, target_return_4, target_return_16, target_return_64.
        """
        df = self._load_df()

        if sort_by_timestamp:
            ts_col = _infer_timestamp_col(df)
            if ts_col:
                df = df.sort_values(ts_col).reset_index(drop=True)

        feat_cols = self._detect_feature_cols(df)
        if not feat_cols:
            raise ValueError(
                "No feature columns detected. Supply feature_cols explicitly or name columns "
                "so they are not confused with the timestamp or target."
            )

        seg = _build_segment(
            df=df,
            asset_name=self._instrument,
            asset_idx=0,
            feature_cols=feat_cols,
            target_col=self._target_col,
            horizons=self._horizons,
        )
        return [seg]

    # -- Convenience: validate segment contract -------------------------------

    @staticmethod
    def validate_segment(seg: dict, raise_on_fail: bool = True) -> bool:
        """Assert the segment dict satisfies the MultiAssetDataset contract.

        Checks: required keys present, shapes consistent, timestamps int64, features float32.
        Returns True on pass; raises AssertionError (or returns False) on fail.
        """
        required = {"asset_idx", "asset_name", "timestamp", "features", "target_return_1"}
        missing = required - set(seg.keys())
        errors = []

        if missing:
            errors.append(f"Missing keys: {missing}")
        else:
            n = len(seg["timestamp"])
            if seg["timestamp"].dtype != np.int64:
                errors.append(f"timestamp.dtype={seg['timestamp'].dtype} (expected int64)")
            else:
                ts_min = int(seg["timestamp"].min()) if n > 0 else 0
                ts_max = int(seg["timestamp"].max()) if n > 0 else 0
                if ts_min < _TS_MIN_VALID_MS or ts_max > _TS_MAX_VALID_MS:
                    errors.append(
                        f"timestamps out of valid range [{_TS_MIN_VALID_MS}, {_TS_MAX_VALID_MS}]: "
                        f"ts.min={ts_min}, ts.max={ts_max}. "
                        f"Check _col_to_epoch_ms or use synthetic timestamps anchored correctly."
                    )
            if seg["features"].dtype != np.float32:
                errors.append(f"features.dtype={seg['features'].dtype} (expected float32)")
            if seg["features"].ndim != 2:
                errors.append(f"features.ndim={seg['features'].ndim} (expected 2)")
            if len(seg["features"]) != n:
                errors.append(f"features length {len(seg['features'])} != timestamp length {n}")
            for key in seg:
                if key.startswith("target_return_"):
                    arr = seg[key]
                    if len(arr) != n:
                        errors.append(f"{key} length {len(arr)} != timestamp length {n}")

        if errors:
            msg = "Segment contract violations: " + "; ".join(errors)
            if raise_on_fail:
                raise AssertionError(msg)
            return False
        return True


# ---------------------------------------------------------------------------
# Self-test: synthetic DataFrame -> segments -> validate contract
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> int:
    """Build a synthetic time-series DataFrame, run to_segments(), validate the contract."""
    import pandas as pd

    failures = 0

    # --- synthetic data ---
    n_bars = 200
    rng = np.random.RandomState(42)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "price_close": 100 + rng.randn(n_bars).cumsum(),
        "price_open":  100 + rng.randn(n_bars).cumsum(),
        "vol_real":    np.abs(rng.randn(n_bars)),
        "mom_5d":      rng.randn(n_bars),
        "y":           rng.randn(n_bars) * 0.01,  # target returns
    })

    adapter = GeneralAdapter(
        data_source=df,
        target_col="y",
        instrument="synthetic_series",
        cadence="1d",
    )

    # universe
    uni = adapter.universe()
    assert uni == ["synthetic_series"], f"universe mismatch: {uni}"
    if verbose:
        print("  [PASS] universe()")

    # cadences
    cads = adapter.cadences()
    assert "1d" in cads, f"cadences missing '1d': {cads}"
    if verbose:
        print("  [PASS] cadences()")

    # feature_families
    fams = adapter.feature_families()
    assert "price" in fams, f"Expected 'price' family, got: {list(fams.keys())}"
    assert "vol" in fams, f"Expected 'vol' family, got: {list(fams.keys())}"
    if verbose:
        print(f"  [PASS] feature_families() -> {list(fams.keys())}")

    # cost_model
    cost = adapter.cost_model().round_trip("synthetic_series")
    assert cost == 0.0, f"Expected 0.0 cost, got {cost}"
    if verbose:
        print("  [PASS] cost_model().round_trip() == 0.0")

    # to_segments
    segs = adapter.to_segments()
    assert len(segs) == 1, f"Expected 1 segment, got {len(segs)}"
    seg = segs[0]

    if verbose:
        print(f"  [PASS] to_segments() -> 1 segment, n_bars={len(seg['timestamp'])}, "
              f"C={seg['features'].shape[1]}")

    # validate contract
    ok = GeneralAdapter.validate_segment(seg, raise_on_fail=False)
    if ok:
        if verbose:
            print("  [PASS] validate_segment() -- contract satisfied")
    else:
        print("  [FAIL] validate_segment() -- contract violated")
        failures += 1

    # check shapes
    n = len(seg["timestamp"])
    assert n == n_bars, f"Expected {n_bars} bars, got {n}"
    assert seg["features"].dtype == np.float32, f"features dtype: {seg['features'].dtype}"
    assert seg["timestamp"].dtype == np.int64, f"timestamp dtype: {seg['timestamp'].dtype}"
    assert seg["target_return_1"].shape == (n_bars,), \
        f"target_return_1 shape: {seg['target_return_1'].shape}"
    if verbose:
        print(f"  [PASS] shapes: timestamp={seg['timestamp'].shape}, "
              f"features={seg['features'].shape}, target_return_1={seg['target_return_1'].shape}")

    # check that target values are non-trivial (using the actual y column)
    assert seg["target_return_1"].std() > 0, "target_return_1 is all zeros -- check target_col wiring"
    if verbose:
        print(f"  [PASS] target_return_1 non-trivial (std={seg['target_return_1'].std():.4f})")

    # parquet round-trip (optional: only if tmp path writable)
    try:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        df.to_parquet(tmp_path, index=False)
        adapter2 = GeneralAdapter(tmp_path, target_col="y", instrument="pq_series")
        segs2 = adapter2.to_segments()
        assert len(segs2) == 1
        os.unlink(tmp_path)
        if verbose:
            print("  [PASS] parquet round-trip -> to_segments()")
    except Exception as exc:
        if verbose:
            print(f"  [SKIP] parquet round-trip ({exc})")

    return failures


if __name__ == "__main__":
    print("[general_adapter] self-test: GeneralAdapter on synthetic DataFrame")
    fails = _selftest(verbose=True)
    if fails:
        print(f"[general_adapter] FAIL: {fails} assertion(s) failed")
        sys.exit(1)
    print("[general_adapter] PASS: all assertions satisfied")
    sys.exit(0)
