"""Unit tests for MultiAssetDataset (src/wm/_shared/multi_asset_dataset.py).

Covers:
  T1. Basic shape contract: [B, A, T, F] obs + [B, A, T] mask + [B, A] asset_ids
  T2. Causality: window content matches the aligned past (no look-ahead)
  T3. Mask correctness: gap in asset timestamps produces mask=False for that slot
  T4. Different-length segments with a deliberate gap handled correctly
  T5. Memory efficiency: index arrays are numpy (no giant tuple list)
  T6. Inner-join mode (min_assets_present = n_assets) produces only fully-present samples
  T7. Partial-coverage mode (min_assets_present = 1) keeps samples with missing assets
  T8. DataLoader collate produces correct stacked shapes
  T9. coverage_stats() reports correct fractions
  T10. Raises on missing timestamps
  T11. btc_pace anchor strategy uses asset_idx=0 timestamps

Run:
    python -m pytest tests/test_multi_asset_dataset.py -v
or standalone:
    python tests/test_multi_asset_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "wm" / "_shared"))

from multi_asset_dataset import (
    MultiAssetDataset,
    AnchorSchedule,
    build_aligned_index,
    _asof_backward,
    build_multi_asset_collate_fn,
)


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def _make_segment(asset_idx: int, n_bars: int, n_features: int,
                  base_ts: int = 1_700_000_000_000,
                  bar_interval_ms: int = 60_000,
                  ts_gap_start: int = None, ts_gap_len: int = 0,
                  rng: np.random.Generator = None) -> dict:
    """Build a synthetic segment dict.

    ts_gap_start: bar index where a timestamp gap is introduced.
    ts_gap_len: size of gap in bars (timestamps jump by bar_interval * gap_len+1).
    """
    if rng is None:
        rng = np.random.default_rng(asset_idx * 1000)

    ts = base_ts + np.arange(n_bars) * bar_interval_ms
    if ts_gap_start is not None and ts_gap_len > 0:
        # Introduce a gap: bars at ts_gap_start onward are shifted forward
        ts[ts_gap_start:] += ts_gap_len * bar_interval_ms

    return {
        "asset_idx": asset_idx,
        "asset_name": f"ASSET{asset_idx}",
        "timestamp": ts.astype(np.int64),
        "features": rng.standard_normal((n_bars, n_features)).astype(np.float32),
        "target_return_1": rng.standard_normal(n_bars).astype(np.float32),
        "target_return_4": rng.standard_normal(n_bars).astype(np.float32),
    }


# ───────────────────────────────────────────────────────────────────
# T1: Basic shape contract
# ───────────────────────────────────────────────────────────────────

def test_basic_shape_contract():
    """[B, A, T, F] obs + [B, A, T] mask + [B, A] asset_ids."""
    rng = np.random.default_rng(0)
    A, T, F, N = 3, 32, 7, 500
    segments = [_make_segment(a, N, F, rng=rng) for a in range(A)]
    ds = MultiAssetDataset(segments, seq_len=T, reward_horizons=[1, 4], n_assets=A)

    assert len(ds) > 0, "Expected at least one sample"
    item = ds[0]

    assert item["obs"].shape == (A, T, F), f"obs shape mismatch: {item['obs'].shape}"
    assert item["mask"].shape == (A, T), f"mask shape: {item['mask'].shape}"
    assert item["asset_ids"].shape == (A,), f"asset_ids shape: {item['asset_ids'].shape}"
    assert item["asset_ids"].tolist() == list(range(A)), "asset_ids must be [0..A-1]"
    for h in [1, 4]:
        assert item["targets"][h].shape == (A, T), \
            f"targets[{h}] shape: {item['targets'][h].shape}"
    assert isinstance(item["anchor_ts_ms"], int), "anchor_ts_ms must be int"

    print("T1 PASS: shape contract  obs=%s mask=%s asset_ids=%s"
          % (tuple(item["obs"].shape), tuple(item["mask"].shape),
             tuple(item["asset_ids"].shape)))


# ───────────────────────────────────────────────────────────────────
# T2: Causality (no look-ahead)
# ───────────────────────────────────────────────────────────────────

def test_no_lookahead():
    """Window content must be the T bars ending at or before anchor_ts."""
    rng = np.random.default_rng(7)
    A, T, F = 2, 16, 4
    base_ts = 1_700_000_000_000
    interval_ms = 60_000
    N = 300
    segments = [_make_segment(a, N, F, base_ts=base_ts,
                               bar_interval_ms=interval_ms, rng=rng)
                for a in range(A)]

    ds = MultiAssetDataset(segments, seq_len=T, reward_horizons=[1], n_assets=A)

    for i in range(min(10, len(ds))):
        item = ds[i]
        anchor_ts = item["anchor_ts_ms"]

        for a in range(A):
            seg = segments[a]
            asset_ts = seg["timestamp"]
            # Compute expected end bar via asof backward
            end_idx_arr = _asof_backward(np.array([anchor_ts]), asset_ts)
            end_idx = int(end_idx_arr[0])
            if end_idx < T - 1:
                # Not enough history; this asset should be masked
                assert not item["mask"][a, 0].item(), \
                    f"T2 FAIL sample={i} asset={a}: expected mask=False but got True"
                continue
            start_idx = end_idx - T + 1
            expected_feats = seg["features"][start_idx:start_idx + T]  # [T, F]
            actual_feats = item["obs"][a].numpy()  # [T, F]
            np.testing.assert_array_almost_equal(
                actual_feats, expected_feats, decimal=5,
                err_msg=f"T2 FAIL: sample={i} asset={a} window mismatch"
            )
            # All features in the window must be at or before anchor_ts
            window_ts = asset_ts[start_idx:start_idx + T]
            assert window_ts[-1] <= anchor_ts, \
                f"T2 FAIL: window end ts {window_ts[-1]} > anchor {anchor_ts} (look-ahead)"

    print("T2 PASS: no look-ahead (checked %d samples x %d assets)" % (min(10, len(ds)), A))


# ───────────────────────────────────────────────────────────────────
# T3: Mask correctness on a deliberate timestamp gap
# ───────────────────────────────────────────────────────────────────

def test_mask_gap():
    """Asset starting late -> mask=False on early anchors; True once asset has T bars.

    The asof-backward join is backward-only: an anchor AFTER an asset's last bar
    still produces a valid window (it looks back into the asset's history). The mask
    only goes False when an asset doesn't yet have enough history to fill a T-bar
    window -- i.e., when the anchor falls before asset[T-1]-th bar.

    We engineer this by making asset 1 start 200 bars LATER than asset 0. Every
    anchor in the first 200 bars of the series hits a region where asset 1 has
    fewer than T bars of history -> mask[1] = False.
    """
    rng = np.random.default_rng(42)
    T, F = 32, 5
    base_ts = 1_700_000_000_000
    interval_ms = 60_000

    # Asset 0: full 400-bar coverage (anchor pace)
    seg0 = {
        "asset_idx": 0, "asset_name": "ASSET0",
        "timestamp": (base_ts + np.arange(400) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((400, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(400).astype(np.float32),
    }

    # Asset 1: starts 200 bars LATER than asset 0 (late-start = 200 * interval_ms)
    late_start_ms = 200 * interval_ms
    seg1 = {
        "asset_idx": 1, "asset_name": "ASSET1",
        "timestamp": (base_ts + late_start_ms + np.arange(200) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((200, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(200).astype(np.float32),
    }

    segments = [seg0, seg1]
    ds = MultiAssetDataset(segments, seq_len=T, reward_horizons=[1],
                            n_assets=2, min_assets_present=1)

    # asset 1 needs T-1 bars of history before an anchor can produce a full window.
    # asset 1's (T-1)-th bar has ts = base_ts + late_start_ms + (T-1)*interval_ms
    asset1_first_valid_anchor = base_ts + late_start_ms + (T - 1) * interval_ms

    n_asset1_masked = 0
    n_asset1_present = 0

    for i in range(len(ds)):
        item = ds[i]
        anchor = item["anchor_ts_ms"]
        m1 = item["mask"][1, 0].item()  # asset 1, first time step

        if anchor < asset1_first_valid_anchor:
            assert not m1, (
                f"T3 FAIL: anchor={anchor} < first_valid={asset1_first_valid_anchor} "
                f"but mask[1] is True (should be False -- not enough history)"
            )
            n_asset1_masked += 1
        else:
            if m1:
                n_asset1_present += 1

    assert n_asset1_masked > 0, "T3 FAIL: expected samples where asset 1 is too early (masked)"
    assert n_asset1_present > 0, "T3 FAIL: expected samples where asset 1 is present"

    print("T3 PASS: late-start mask  asset1_masked=%d  asset1_present=%d"
          % (n_asset1_masked, n_asset1_present))


# ───────────────────────────────────────────────────────────────────
# T4: Different-length segments with deliberate timestamp gap
# ───────────────────────────────────────────────────────────────────

def test_different_lengths_and_gap():
    """3 assets with different lengths + a gap in asset 2 timestamps."""
    rng = np.random.default_rng(99)
    T, F = 24, 6
    base_ts = 1_700_000_000_000
    interval_ms = 60_000

    seg0 = _make_segment(0, 600, F, base_ts=base_ts, bar_interval_ms=interval_ms, rng=rng)
    seg1 = _make_segment(1, 400, F, base_ts=base_ts, bar_interval_ms=interval_ms, rng=rng)
    # Asset 2: 300 bars but with a 50-bar timestamp gap starting at bar 100
    seg2 = _make_segment(2, 300, F, base_ts=base_ts, bar_interval_ms=interval_ms,
                          ts_gap_start=100, ts_gap_len=50, rng=rng)

    segments = [seg0, seg1, seg2]
    ds = MultiAssetDataset(segments, seq_len=T, reward_horizons=[1, 4],
                            n_assets=3, min_assets_present=1)

    assert len(ds) > 0, "T4 FAIL: no samples produced"

    # Verify shapes on a few samples
    for i in range(min(5, len(ds))):
        item = ds[i]
        assert item["obs"].shape == (3, T, F)
        assert item["mask"].shape == (3, T)
        for h in [1, 4]:
            assert item["targets"][h].shape == (3, T)

    # Verify coverage stats makes sense (asset 1 shorter -> lower coverage)
    stats = ds.coverage_stats()
    assert 0 < stats["per_asset_coverage"][1] <= stats["per_asset_coverage"][0], \
        "T4: asset 1 (shorter) should have <= coverage of asset 0 (longer)"
    assert stats["n_samples"] == len(ds)

    print("T4 PASS: different lengths + gap  n_samples=%d  coverage=%s"
          % (stats["n_samples"],
             {k: "%.2f" % v for k, v in stats["per_asset_coverage"].items()}))


# ───────────────────────────────────────────────────────────────────
# T5: Memory efficiency -- index is numpy, not a Python list of tuples
# ───────────────────────────────────────────────────────────────────

def test_memory_is_numpy_index():
    """per_asset_starts and sample_anchors must be numpy arrays."""
    rng = np.random.default_rng(1)
    segments = [_make_segment(a, 500, 4, rng=rng) for a in range(3)]
    ds = MultiAssetDataset(segments, seq_len=32, n_assets=3)

    assert isinstance(ds.sample_anchors, np.ndarray), \
        "T5 FAIL: sample_anchors should be np.ndarray"
    assert isinstance(ds.per_asset_starts, np.ndarray), \
        "T5 FAIL: per_asset_starts should be np.ndarray"
    assert ds.per_asset_starts.dtype == np.int32, \
        f"T5 FAIL: per_asset_starts dtype should be int32, got {ds.per_asset_starts.dtype}"
    assert ds.sample_anchors.dtype == np.int64, \
        f"T5 FAIL: sample_anchors dtype should be int64, got {ds.sample_anchors.dtype}"
    assert ds.per_asset_starts.shape == (len(ds), 3), \
        f"T5 FAIL: starts shape {ds.per_asset_starts.shape}"

    print("T5 PASS: index is numpy  starts=%s dtype=%s"
          % (ds.per_asset_starts.shape, ds.per_asset_starts.dtype))


# ───────────────────────────────────────────────────────────────────
# T6: Inner-join mode (min_assets_present = n_assets)
# ───────────────────────────────────────────────────────────────────

def test_inner_join_mode():
    """With min_assets_present=n_assets, every sample has all assets present.

    Asset 1 starts 200 bars later. Inner-join drops the early anchors where
    asset 1 has no history; partial mode keeps them. The two should differ.
    """
    rng = np.random.default_rng(5)
    A, T, F = 3, 20, 4
    base_ts = 1_700_000_000_000
    interval_ms = 60_000

    seg0 = {
        "asset_idx": 0,
        "timestamp": (base_ts + np.arange(400) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((400, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(400).astype(np.float32),
    }
    # Asset 1 starts 200 bars later (fewer early-history bars)
    seg1 = {
        "asset_idx": 1,
        "timestamp": (base_ts + 200 * interval_ms + np.arange(200) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((200, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(200).astype(np.float32),
    }
    seg2 = {
        "asset_idx": 2,
        "timestamp": (base_ts + np.arange(400) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((400, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(400).astype(np.float32),
    }
    segments = [seg0, seg1, seg2]

    ds_inner = MultiAssetDataset(segments, seq_len=T, n_assets=A,
                                  min_assets_present=A)
    ds_outer = MultiAssetDataset(segments, seq_len=T, n_assets=A,
                                  min_assets_present=1)

    # Inner join must have strictly fewer samples (early anchors dropped)
    assert len(ds_inner) < len(ds_outer), \
        "T6 FAIL: inner join should have < samples than partial coverage mode"

    # Every sample in inner join must have all masks True
    for i in range(min(20, len(ds_inner))):
        item = ds_inner[i]
        assert item["mask"].all(), \
            f"T6 FAIL: sample {i} in inner-join mode has False mask entries"

    print("T6 PASS: inner_join=%d < outer=%d samples; all masks True"
          % (len(ds_inner), len(ds_outer)))


# ───────────────────────────────────────────────────────────────────
# T7: Partial-coverage mode keeps samples with missing assets
# ───────────────────────────────────────────────────────────────────

def test_partial_coverage_mode():
    """min_assets_present=1 keeps anchors where only some assets have data.

    Asset 1 starts 300 bars LATER than asset 0, so early anchors produce
    mask=False for asset 1. This means per_asset_coverage[1] < 1.0 and
    inner_join_coverage < 1.0.
    """
    rng = np.random.default_rng(3)
    T, F = 20, 5
    base_ts = 1_700_000_000_000
    interval_ms = 60_000

    # Asset 0: full coverage, 400 bars (anchor pace)
    seg0 = {
        "asset_idx": 0, "asset_name": "ASSET0",
        "timestamp": (base_ts + np.arange(400) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((400, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(400).astype(np.float32),
    }
    # Asset 1: starts 300 bars later, only 100 bars
    late_start = 300 * interval_ms
    seg1 = {
        "asset_idx": 1, "asset_name": "ASSET1",
        "timestamp": (base_ts + late_start + np.arange(100) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((100, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(100).astype(np.float32),
    }
    segments = [seg0, seg1]

    ds = MultiAssetDataset(segments, seq_len=T, n_assets=2, min_assets_present=1)
    stats = ds.coverage_stats()

    # Asset 1 starts late: many early anchors won't have T bars -> coverage < 1.0
    assert stats["per_asset_coverage"][1] < 1.0, \
        "T7 FAIL: asset 1 should not be present for all samples (starts late)"
    # Asset 0 is the anchor source (btc_pace) so its coverage is always 1.0
    assert stats["per_asset_coverage"][0] > 0.99, \
        f"T7 FAIL: asset 0 coverage should be ~1.0, got {stats['per_asset_coverage'][0]}"
    # inner_join_coverage should be less than 1.0 when asset 1 is partial
    assert stats["inner_join_coverage"] < 1.0, \
        "T7 FAIL: inner_join_coverage should be < 1.0 when asset 1 is partial"

    print("T7 PASS: partial coverage  asset0=%.2f asset1=%.2f inner_join=%.2f"
          % (stats["per_asset_coverage"][0], stats["per_asset_coverage"][1],
             stats["inner_join_coverage"]))


# ───────────────────────────────────────────────────────────────────
# T8: DataLoader collate produces correct shapes
# ───────────────────────────────────────────────────────────────────

def test_dataloader_collate():
    """DataLoader with collate_fn produces [B, A, T, F] obs."""
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(11)
    A, T, F, N = 4, 16, 5, 300
    B = 6
    horizons = [1, 4]
    segments = [_make_segment(a, N, F, rng=rng) for a in range(A)]

    ds = MultiAssetDataset(segments, seq_len=T, reward_horizons=horizons, n_assets=A)
    collate = build_multi_asset_collate_fn(horizons)
    loader = DataLoader(ds, batch_size=B, collate_fn=collate,
                        shuffle=False, num_workers=0)

    batch = next(iter(loader))
    assert batch["obs"].shape == (B, A, T, F), \
        f"T8 FAIL: obs shape {batch['obs'].shape} != ({B}, {A}, {T}, {F})"
    assert batch["asset_ids"].shape == (B, A), \
        f"T8 FAIL: asset_ids shape {batch['asset_ids'].shape}"
    assert batch["mask"].shape == (B, A, T), \
        f"T8 FAIL: mask shape {batch['mask'].shape}"
    for h in horizons:
        assert batch["targets"][h].shape == (B, A, T), \
            f"T8 FAIL: targets[{h}] shape {batch['targets'][h].shape}"
    assert len(batch["anchor_ts_ms"]) == B

    print("T8 PASS: DataLoader collate  obs=%s mask=%s"
          % (tuple(batch["obs"].shape), tuple(batch["mask"].shape)))


# ───────────────────────────────────────────────────────────────────
# T9: coverage_stats correctness
# ───────────────────────────────────────────────────────────────────

def test_coverage_stats():
    """coverage_stats per_asset fractions match manual count.

    Asset 1 starts 250 bars later so its coverage fraction is < 1.0.
    """
    rng = np.random.default_rng(55)
    T, F = 16, 4
    base_ts = 1_700_000_000_000
    interval_ms = 60_000

    seg0 = {
        "asset_idx": 0,
        "timestamp": (base_ts + np.arange(500) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((500, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(500).astype(np.float32),
    }
    # Asset 1 starts 250 bars later (so early anchors miss it)
    seg1 = {
        "asset_idx": 1,
        "timestamp": (base_ts + 250 * interval_ms + np.arange(250) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((250, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(250).astype(np.float32),
    }
    segments = [seg0, seg1]

    ds = MultiAssetDataset(segments, seq_len=T, n_assets=2, min_assets_present=1)
    stats = ds.coverage_stats()

    # Manual: count starts >= 0 per asset
    manual_asset0 = float((ds.per_asset_starts[:, 0] >= 0).mean())
    manual_asset1 = float((ds.per_asset_starts[:, 1] >= 0).mean())

    assert abs(stats["per_asset_coverage"][0] - manual_asset0) < 1e-6, \
        f"T9 FAIL: asset0 coverage {stats['per_asset_coverage'][0]} != manual {manual_asset0}"
    assert abs(stats["per_asset_coverage"][1] - manual_asset1) < 1e-6, \
        f"T9 FAIL: asset1 coverage {stats['per_asset_coverage'][1]} != manual {manual_asset1}"
    assert stats["n_samples"] == len(ds), "T9 FAIL: n_samples mismatch"
    # Asset 1 starts late: its coverage must be < 1.0
    assert stats["per_asset_coverage"][1] < 1.0, \
        f"T9 FAIL: asset1 coverage should be < 1.0 (starts late), got {stats['per_asset_coverage'][1]}"

    print("T9 PASS: coverage_stats  asset0=%.3f asset1=%.3f inner_join=%.3f"
          % (stats["per_asset_coverage"][0], stats["per_asset_coverage"][1],
             stats["inner_join_coverage"]))


# ───────────────────────────────────────────────────────────────────
# T10: Raises on missing timestamps
# ───────────────────────────────────────────────────────────────────

def test_raises_on_missing_timestamps():
    """Segment without 'timestamp' key must raise ValueError."""
    rng = np.random.default_rng(77)
    seg0 = _make_segment(0, 200, 4, rng=rng)
    # Strip timestamp from asset 1
    seg1 = {
        "asset_idx": 1,
        "asset_name": "ASSET1",
        "features": rng.standard_normal((200, 4)).astype(np.float32),
        "target_return_1": rng.standard_normal(200).astype(np.float32),
    }
    try:
        MultiAssetDataset([seg0, seg1], seq_len=16, n_assets=2)
        assert False, "T10 FAIL: should have raised ValueError for missing timestamp"
    except ValueError as e:
        assert "timestamp" in str(e).lower(), \
            f"T10 FAIL: wrong error message: {e}"
    print("T10 PASS: raises ValueError on missing timestamp")


# ───────────────────────────────────────────────────────────────────
# T11: btc_pace anchor strategy uses asset_idx=0 timestamps
# ───────────────────────────────────────────────────────────────────

def test_btc_pace_anchor_strategy():
    """btc_pace anchors must exactly equal asset_idx=0 timestamps."""
    rng = np.random.default_rng(22)
    A, T, F = 3, 20, 4
    seg0 = _make_segment(0, 300, F, rng=rng)
    seg1 = _make_segment(1, 200, F, rng=rng)
    seg2 = _make_segment(2, 250, F, rng=rng)
    segments = [seg0, seg1, seg2]

    schedule = AnchorSchedule(strategy="btc_pace", btc_asset_idx=0)
    ds = MultiAssetDataset(segments, seq_len=T, n_assets=A,
                            anchor_schedule=schedule, min_assets_present=1)

    # Every anchor_ts_ms in the dataset must be in asset 0's timestamps
    ts0_set = set(seg0["timestamp"].tolist())
    for i in range(min(50, len(ds))):
        anchor = ds[i]["anchor_ts_ms"]
        assert anchor in ts0_set, \
            f"T11 FAIL: anchor_ts {anchor} not in asset 0 timestamps"

    print("T11 PASS: btc_pace anchors all in asset_idx=0 timestamps (%d checked)"
          % min(50, len(ds)))


# ───────────────────────────────────────────────────────────────────
# T12: Inner-join coverage diagnostic (real-world 10-asset simulation)
# ───────────────────────────────────────────────────────────────────

def test_inner_join_coverage_10_assets():
    """Simulate 10-asset scenario and report inner-join survival rate.

    This is an informational test: it does NOT assert a specific threshold
    (the actual fraction depends on bar-type alignment). It measures and
    prints how many timestamps survive the strict inner-join vs the partial-
    coverage default. The fraction is a key V12 trainability input.

    For fixed-cadence OHLCV data (e.g. 4h bars at aligned wall-clock times),
    expect ~80-95% survival. For dollar-bars with asset-specific trade counts,
    expect much lower (<20%) -- that's the honest finding about cross-asset
    trainability the task specification requested.
    """
    rng = np.random.default_rng(0)
    A, T, F = 10, 64, 13
    N_bars = 2000  # 2000 bars per asset, all at the same base cadence
    # Use identical timestamps for all assets (simulates wall-clock aligned bars)
    base_ts = 1_700_000_000_000
    interval_ms = 4 * 3_600_000  # 4h cadence

    segments = []
    for a in range(A):
        # Vary the length slightly: some assets are shorter
        n = N_bars - rng.integers(0, 200)
        seg = _make_segment(a, int(n), F,
                             base_ts=base_ts, bar_interval_ms=interval_ms, rng=rng)
        segments.append(seg)

    # Partial mode
    ds_partial = MultiAssetDataset(segments, seq_len=T, n_assets=A,
                                    min_assets_present=1)
    # Inner-join mode
    try:
        ds_inner = MultiAssetDataset(segments, seq_len=T, n_assets=A,
                                      min_assets_present=A)
        n_inner = len(ds_inner)
    except ValueError:
        n_inner = 0  # No samples survived inner-join

    n_partial = len(ds_partial)
    survival = n_inner / max(n_partial, 1)

    print("T12 INFO: 10-asset inner-join coverage diagnostic")
    print("  partial samples (min_present=1):       %d" % n_partial)
    print("  inner-join samples (min_present=%d): %d" % (A, n_inner))
    print("  inner-join survival rate:              %.1f%%" % (survival * 100))
    if n_inner > 0:
        stats = ds_inner.coverage_stats()
        print("  mean assets present (inner-join): %.2f / %d"
              % (stats["mean_assets_present"], A))
    print("  NOTE: at real dollar-bar cadences where different assets have")
    print("  different trade rates, survival may be much lower (<20%).")
    print("  Use min_assets_present < n_assets and rely on the mask.")

    # Soft assertion: at least some partial-mode samples exist
    assert n_partial > 0, "T12 FAIL: no samples even in partial mode"


# ───────────────────────────────────────────────────────────────────
# T13: absent_target_nan flag (P4)
# ───────────────────────────────────────────────────────────────────

def test_absent_target_nan():
    """P4: absent_target_nan=True fills absent-asset TARGET slots with NaN.

    Checks:
    - With absent_target_nan=True: an asset that is absent at an anchor has
      NaN target slots and mask=False for those slots.
    - The obs slots for the absent asset remain 0.0 (not NaN).
    - With absent_target_nan=False (default): absent target slots are 0.0
      (backward-compat).
    """
    rng = np.random.default_rng(314)
    T, F = 20, 5
    base_ts = 1_700_000_000_000
    interval_ms = 60_000

    # Asset 0: full coverage (anchor pace), 400 bars
    seg0 = {
        "asset_idx": 0, "asset_name": "ASSET0",
        "timestamp": (base_ts + np.arange(400) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((400, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(400).astype(np.float32),
    }
    # Asset 1: starts 300 bars later -> early anchors have no asset-1 data
    late_start = 300 * interval_ms
    seg1 = {
        "asset_idx": 1, "asset_name": "ASSET1",
        "timestamp": (base_ts + late_start + np.arange(100) * interval_ms).astype(np.int64),
        "features": rng.standard_normal((100, F)).astype(np.float32),
        "target_return_1": rng.standard_normal(100).astype(np.float32),
    }
    segments = [seg0, seg1]

    # Build with absent_target_nan=True
    ds_nan = MultiAssetDataset(
        segments, seq_len=T, reward_horizons=[1], n_assets=2,
        min_assets_present=1, absent_target_nan=True,
    )
    # Build with absent_target_nan=False (default)
    ds_zero = MultiAssetDataset(
        segments, seq_len=T, reward_horizons=[1], n_assets=2,
        min_assets_present=1, absent_target_nan=False,
    )

    found_absent_nan = False
    found_absent_zero = False

    for i in range(len(ds_nan)):
        item_nan = ds_nan[i]
        item_zero = ds_zero[i]
        # Find a sample where asset 1 is absent (mask[1] is all False)
        if not item_nan["mask"][1, 0].item():
            # Absent asset -- target_return_1 for asset 1 should be NaN
            t_nan = item_nan["targets"][1][1]   # [T] for asset 1
            t_zero = item_zero["targets"][1][1]  # [T] for asset 1

            assert torch.all(torch.isnan(t_nan)), (
                f"T13 FAIL: absent-asset targets with absent_target_nan=True should be NaN, "
                f"got: {t_nan}"
            )
            assert torch.all(t_zero == 0.0), (
                f"T13 FAIL: absent-asset targets with absent_target_nan=False should be 0.0, "
                f"got: {t_zero}"
            )
            # Obs must stay 0.0 regardless (NaN in obs would corrupt attention)
            obs_nan = item_nan["obs"][1]  # [T, F] for asset 1
            assert not torch.any(torch.isnan(obs_nan)), (
                f"T13 FAIL: obs should always be 0.0 for absent assets (never NaN), "
                f"got NaN at sample {i}"
            )
            assert torch.all(obs_nan == 0.0), (
                f"T13 FAIL: obs should be 0.0 for absent assets, got: {obs_nan}"
            )
            found_absent_nan = True
            found_absent_zero = True
            break

    assert found_absent_nan, (
        "T13 FAIL: no absent-asset sample found in ds_nan -- "
        "check that asset 1 really starts late enough"
    )
    assert found_absent_zero, (
        "T13 FAIL: no absent-asset sample found in ds_zero"
    )

    # Sanity: present-asset (asset 0) targets must NOT be NaN (only absent slots get NaN)
    item0 = ds_nan[len(ds_nan) - 1]  # late sample where both assets should be present
    # Just check asset 0 (anchor source, always present in last sample)
    assert not torch.any(torch.isnan(item0["targets"][1][0])), \
        "T13 FAIL: present-asset targets should not be NaN"

    print("T13 PASS: absent_target_nan  NaN-for-absent=True, 0.0-for-absent=False (default), obs=0.0 always")


# ───────────────────────────────────────────────────────────────────
# Main runner
# ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run standalone without pytest (for RWYB verification)
    import traceback

    tests = [
        test_basic_shape_contract,
        test_no_lookahead,
        test_mask_gap,
        test_different_lengths_and_gap,
        test_memory_is_numpy_index,
        test_inner_join_mode,
        test_partial_coverage_mode,
        test_dataloader_collate,
        test_coverage_stats,
        test_raises_on_missing_timestamps,
        test_btc_pace_anchor_strategy,
        test_inner_join_coverage_10_assets,
        test_absent_target_nan,
    ]

    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception:
            failed.append(t.__name__)
            traceback.print_exc()

    print("")
    print("=" * 60)
    print("RESULTS: %d / %d passed" % (passed, len(tests)))
    if failed:
        print("FAILED: %s" % ", ".join(failed))
        sys.exit(1)
    else:
        print("ALL PASS")
        sys.exit(0)


# pytest compatibility: import pytest only when running under pytest
try:
    import pytest
except ImportError:
    # Provide a minimal stub so T7 pytest.approx works when run standalone
    class _PytestStub:
        @staticmethod
        def approx(val, abs=None, rel=None):
            class _Approx:
                def __init__(self, v, a):
                    self.v, self.a = v, a
                def __eq__(self, other):
                    return abs(other - self.v) <= self.a
            return _Approx(val, abs or 1e-6)
    pytest = _PytestStub()
