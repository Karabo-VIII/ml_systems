"""src/narrate/selftest_narrate.py -- DATA-FREE selftest for the narrate engine.

Builds a synthetic in-memory frame matching the _PolarsShim interface, then exercises:
  - narrate.state.compute_state()
  - narrate.feature_map.coverage_report()

Assertions:
  - families resolve (at least one FamilyRead is returned)
  - FamilyRead objects have finite scores
  - percentile math is in [0, 100]
  - no crash
  - events list returns (list, possibly empty)

No chimera load, no network, no GPU. Completes in < 2 seconds.
Prints PASS or FAIL and exits nonzero on failure.
No emoji (Windows cp1252 safe).
"""
from __future__ import annotations

import os
import sys
import traceback

# Ensure src/ is on the path so `narrate` is importable when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------

class _SyntheticCol:
    """Minimal column object matching the _PolarsShim _Col interface."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def to_numpy(self):
        return self._arr.copy()


class _SyntheticFrame:
    """Minimal frame object matching the _PolarsShim interface expected by state.compute_state.

    Must expose:
      - .columns  list of str
      - __len__() -> int
      - __getitem__(col: str) -> object with .to_numpy()
    """
    def __init__(self, data: dict):
        self._data = {k: np.asarray(v) for k, v in data.items()}
        self.columns = list(data.keys())

    def __len__(self):
        return len(next(iter(self._data.values()))) if self._data else 0

    def __getitem__(self, col):
        if col not in self._data:
            raise KeyError(col)
        return _SyntheticCol(self._data[col])


def _make_synthetic_frame(n: int = 80) -> _SyntheticFrame:
    """Build a synthetic chimera-like frame with norm_* columns + OHLCT."""
    rng = np.random.default_rng(seed=42)

    # ---- base price series ----
    close = 30_000.0 * np.cumprod(1.0 + 0.002 * rng.standard_normal(n))
    high = close * (1.0 + 0.005 * np.abs(rng.standard_normal(n)))
    low = close * (1.0 - 0.005 * np.abs(rng.standard_normal(n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    ts = np.arange(n, dtype=np.int64) * 14_400_000 + 1_700_000_000_000  # 4h steps from a 2023 epoch

    data: dict = {
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }

    # ---- norm_* columns representing key families ----
    # structure
    data["norm_ma_distance"] = rng.standard_normal(n)
    data["norm_deviation"] = rng.standard_normal(n)
    data["norm_efficiency"] = rng.uniform(0.0, 1.0, n)
    # momentum
    data["norm_return_1"] = rng.standard_normal(n)
    data["norm_return_4"] = rng.standard_normal(n)
    data["norm_return_16"] = rng.standard_normal(n)
    data["norm_momentum_accel"] = rng.standard_normal(n)
    # volatility
    data["norm_yz_volatility"] = np.abs(rng.standard_normal(n)) + 0.1
    data["norm_vol_ratio"] = np.abs(rng.standard_normal(n)) + 0.5
    # orderflow
    data["norm_flow_imbalance"] = rng.standard_normal(n)
    data["norm_vpin"] = np.abs(rng.standard_normal(n))
    # derivatives
    data["norm_funding"] = rng.standard_normal(n) * 0.3
    data["norm_oi_change"] = rng.standard_normal(n)
    # liquidation event flags (sparse binary)
    liq_flag = np.zeros(n, dtype=np.float64)
    liq_flag[rng.integers(0, n, size=3)] = 1.0
    data["liq_capitulation"] = liq_flag
    short_panic = np.zeros(n, dtype=np.float64)
    short_panic[rng.integers(0, n, size=2)] = 1.0
    data["liq_short_panic"] = short_panic
    # regime label (categorical -- must degrade gracefully)
    data["regime_label"] = np.array([1, 2, 1, 2] * (n // 4) + [1] * (n % 4), dtype=object)
    # cross-asset
    data["xd_btc_return"] = rng.standard_normal(n)
    data["xd_momentum_rank"] = rng.uniform(0.0, 1.0, n)

    return _SyntheticFrame(data)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def run_selftest() -> bool:
    """Run all assertions. Returns True = PASS, False = FAIL."""
    failures = []

    # ---------- 1. build synthetic frame ----------
    try:
        df = _make_synthetic_frame(n=80)
        n = len(df)
        assert n == 80, f"expected n=80, got {n}"
        assert "norm_return_1" in df.columns
        assert "timestamp" in df.columns
        assert "close" in df.columns
    except Exception:
        failures.append(("synthetic frame construction", traceback.format_exc()))
        return _report(failures)  # abort early; nothing else can run

    # period mask: last 20 bars; reference mask: all bars
    period_mask = np.zeros(n, dtype=bool)
    period_mask[60:] = True
    ref_mask = np.ones(n, dtype=bool)

    # ---------- 2. compute_state ----------
    try:
        from narrate.state import compute_state
        reads, events, meta = compute_state(df, period_mask, ref_mask)
    except Exception:
        failures.append(("compute_state import/call", traceback.format_exc()))
        return _report(failures)

    # check reads is a non-empty dict
    try:
        assert isinstance(reads, dict), f"reads must be dict, got {type(reads)}"
        assert len(reads) > 0, "no FamilyRead returned -- families did not resolve"
    except AssertionError as exc:
        failures.append(("reads non-empty", str(exc)))

    # check FamilyRead objects have finite scores and valid percentiles
    try:
        from narrate.state import FamilyRead
        for fam, r in reads.items():
            assert isinstance(r, FamilyRead), f"reads[{fam!r}] is not a FamilyRead"
            assert np.isfinite(r.score), f"reads[{fam!r}].score is not finite: {r.score}"
            assert 0.0 <= r.intensity_pctile <= 100.0, (
                f"reads[{fam!r}].intensity_pctile={r.intensity_pctile} outside [0, 100]"
            )
            assert r.direction in {
                "bullish", "bearish", "neutral", "elevated", "compressed", "normal", "n/a"
            }, f"reads[{fam!r}].direction={r.direction!r} is unknown"
            # salient list has tuples
            for item in r.salient:
                assert len(item) == 5, f"salient item length={len(item)}, expected 5"
                _c, _t, _v, pct, _pol = item
                assert 0.0 <= pct <= 100.0, f"salient pct={pct} outside [0,100]"
    except AssertionError as exc:
        failures.append(("FamilyRead structure", str(exc)))
    except Exception:
        failures.append(("FamilyRead structure", traceback.format_exc()))

    # check events is a list (may be empty; liq flags were set so expect at least 1)
    try:
        assert isinstance(events, list), f"events must be list, got {type(events)}"
    except AssertionError as exc:
        failures.append(("events list type", str(exc)))

    # check meta is a dict with n_bars key
    try:
        assert isinstance(meta, dict), f"meta must be dict"
        assert "n_bars" in meta, "meta missing 'n_bars'"
        assert meta["n_bars"] == int(period_mask.sum()), (
            f"meta n_bars={meta['n_bars']} != period bars={int(period_mask.sum())}"
        )
    except AssertionError as exc:
        failures.append(("meta dict", str(exc)))

    # ---------- 3. coverage_report ----------
    try:
        from narrate.feature_map import coverage_report
        cov = coverage_report(df.columns)
        assert isinstance(cov, dict), "coverage_report must return dict"
        assert "n_curated" in cov, "coverage_report missing n_curated"
        assert "families_present" in cov, "coverage_report missing families_present"
        assert cov["n_curated"] >= 0, "n_curated must be >= 0"
        assert cov.get("curated_pct", 0.0) >= 0.0, "curated_pct must be >= 0"
        fams = cov["families_present"]
        assert isinstance(fams, dict), "families_present must be dict"
        assert len(fams) > 0, "families_present must have at least one entry"
    except Exception:
        failures.append(("coverage_report", traceback.format_exc()))

    # ---------- 4. percentile edge cases (degenerate inputs) ----------
    try:
        from narrate.state import _pctile
        empty = np.array([], dtype=np.float64)
        assert not np.isfinite(_pctile(empty, 0.5)), "_pctile on empty array should return non-finite"
        sorted_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = _pctile(sorted_arr, 3.0)
        assert 0.0 <= v <= 100.0, f"_pctile returned {v} outside [0,100]"
        v_low = _pctile(sorted_arr, 0.0)
        assert v_low == 0.0, f"_pctile below min should be 0.0, got {v_low}"
        v_high = _pctile(sorted_arr, 100.0)
        assert v_high == 100.0, f"_pctile above max should be 100.0, got {v_high}"
    except Exception:
        failures.append(("_pctile edge cases", traceback.format_exc()))

    # ---------- 5. FAMILIES / FAMILY_ORDER are importable and consistent ----------
    try:
        from narrate.feature_map import FAMILIES, FAMILY_ORDER, FEATURES, classify, group_columns
        assert isinstance(FAMILIES, dict) and len(FAMILIES) > 0, "FAMILIES must be non-empty dict"
        assert isinstance(FAMILY_ORDER, list) and len(FAMILY_ORDER) > 0, "FAMILY_ORDER must be non-empty list"
        for k in FAMILY_ORDER:
            assert k in FAMILIES, f"FAMILY_ORDER key {k!r} not in FAMILIES"
        # classify should return a known family or 'meta'/'target'/'misc'
        assert classify("norm_return_1") in FAMILIES or classify("norm_return_1") in ("meta", "target", "misc")
        assert classify("timestamp") == "meta"
        # group_columns round-trips all cols to a dict of lists
        grouped = group_columns(df.columns)
        assert isinstance(grouped, dict)
        all_grouped_cols = [c for lst in grouped.values() for c in lst]
        for c in df.columns:
            assert c in all_grouped_cols, f"column {c!r} missing from group_columns output"
    except Exception:
        failures.append(("feature_map imports/consistency", traceback.format_exc()))

    return _report(failures)


def _report(failures: list) -> bool:
    if failures:
        print("FAIL")
        for name, detail in failures:
            print(f"  [FAIL] {name}")
            for line in detail.strip().splitlines():
                print(f"    {line}")
        return False
    print("PASS")
    return True


if __name__ == "__main__":
    ok = run_selftest()
    sys.exit(0 if ok else 1)
