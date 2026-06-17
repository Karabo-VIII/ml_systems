"""MultiAssetDataset -- synchronized N-asset batches for V12 cross-asset attention.

Status: 2026-06-10 COMPLETE. Full timestamp-aligned implementation with
mask-based partial-asset support. Replaces the 2026-05-16 scaffold.

Why this exists
---------------

V12's CrossAssetAttention in world_model.py expects synchronized N-asset batches:

    Input:  multi_obs [B, A, T, F]  -- A assets, same anchor timestamps
            multi_asset_ids [B, A]  -- asset indices

V1.x's AntifragileDataset returns single-asset (obs, asset_id, targets)
tuples. With N_assets=1 per batch, the cross-asset attention is a no-op.
This dataset lifts that constraint.

Data contract
-------------

Input: list of per-asset segment dicts (same format AntifragileDataset uses),
each with:
    {
        "asset_idx": int,                            # 0..N-1
        "asset_name": str,                           # "BTCUSDT" etc.
        "timestamp": np.ndarray (n_bars,) int64-ms,  # epoch ms, SORTED ASC
        "features": np.ndarray (n_bars, F) float32,
        "target_return_<h>": np.ndarray (n_bars,) float32,  # one per horizon h
        # optional:
        "regime_label": np.ndarray (n_bars,) int64,
        "weight": np.ndarray (n_bars,) float32,
    }

Output of __getitem__: synchronized N-asset slice at a chosen anchor TS
    {
        "obs":           torch.Tensor [N_assets, T, F],
        "asset_ids":     torch.Tensor [N_assets]   (long),
        "targets":       {h: torch.Tensor [N_assets, T] float32},
        "mask":          torch.Tensor [N_assets, T] bool   # True where data is valid
        "anchor_ts_ms":  int,                       # the alignment timestamp
    }

DataLoader stacks these into:
    obs:        [B, N_assets, T, F]
    asset_ids:  [B, N_assets]
    targets[h]: [B, N_assets, T]
    mask:       [B, N_assets, T]

Trainer flattens (B, N_assets) -> (B*N_assets) for the per-asset encoder,
then reshapes back for the cross-asset attention head.

Causality guarantee
-------------------
The asof backward search is backward-only: for anchor timestamp t, we use
the bar with the LARGEST timestamp <= t. The window is [end_bar - T + 1,
end_bar]. No future bars are ever included. The mask has no look-ahead either:
missing-asset slots are filled with zeros, mask=False; the trainer must NOT
compute loss on mask=False positions.
"""
from __future__ import annotations

import warnings as _warnings_module  # alias to avoid name collision with local `warnings` vars

__contract__ = {
    "kind": "wm_dataset",
    "owner": "wm/_shared",
    "inputs": ["list of per-asset segment dicts with timestamp + features + targets"],
    "outputs": ["[B, A, T, F] obs tensor", "[B, A, T] mask tensor",
                "[B, A] asset_ids tensor", "{h: [B, A, T]} targets dict"],
    "invariants": [
        "every __getitem__ returns exactly N_assets slices (some masked)",
        "timestamps within a returned batch are monotonic per asset (causal)",
        "asof join is backward-only -- never look-ahead from future bars",
        "missing-asset bars: obs=zeros, targets=zeros (or NaN when absent_target_nan=True), mask=False",
        "index arrays (per_asset_starts, sample_anchors) are numpy int32/int64 -- no Python lists",
        "min_assets_present <= n_assets enforced at construction",
    ],
}

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ───────────────────────────────────────────────────────────────────
# Asof-join helper (backward search, vectorized)
# ───────────────────────────────────────────────────────────────────

def _asof_backward(target_ts: np.ndarray, asset_ts: np.ndarray) -> np.ndarray:
    """For each value in `target_ts`, return the index in `asset_ts` of the
    LATEST bar at or before that timestamp. Returns -1 when no such bar exists.

    `asset_ts` MUST be sorted ascending (guaranteed by the chimera producer).

    Uses np.searchsorted(side='right') which gives the insertion point AFTER
    equal values; subtract 1 to get the largest-LE index.
    """
    idx = np.searchsorted(asset_ts, target_ts, side="right") - 1
    return idx  # -1 means "no bar at or before target_ts"


# ───────────────────────────────────────────────────────────────────
# Anchor-TS schedule
# ───────────────────────────────────────────────────────────────────

@dataclass
class AnchorSchedule:
    """Determines which timestamps are used as alignment anchors.

    Options:
      - "btc_pace": one anchor per BTC bar (BTC is most active; avoids
                    inflating anchor count vs actual bar density)
      - "fixed_grid_ms": one anchor every N ms regardless of bar density
      - "wallclock_hour": alias for fixed_grid_ms at 1h intervals

    For dollar-bar data, "btc_pace" is the natural default.
    For fixed-cadence (OHLCV 4h), "fixed_grid_ms" at the cadence interval
    gives a denser index at the cost of many repeated asof lookups.
    """
    strategy: str = "btc_pace"
    btc_asset_idx: int = 0
    fixed_grid_ms: int = 3_600_000   # 1 hour

    def build_anchors(self, segments: list[dict]) -> np.ndarray:
        if self.strategy == "btc_pace":
            for seg in segments:
                if seg.get("asset_idx") == self.btc_asset_idx:
                    ts = seg["timestamp"]
                    if not isinstance(ts, np.ndarray):
                        ts = np.asarray(ts, dtype=np.int64)
                    return ts.copy()
            raise ValueError(
                f"btc_pace selected but no segment with asset_idx="
                f"{self.btc_asset_idx} found in segments"
            )
        elif self.strategy in ("fixed_grid_ms", "wallclock_hour"):
            step = self.fixed_grid_ms
            t_min = min(seg["timestamp"][0] for seg in segments)
            t_max = max(seg["timestamp"][-1] for seg in segments)
            return np.arange(t_min, t_max + 1, step, dtype=np.int64)
        raise ValueError(f"unknown anchor strategy: {self.strategy!r}")


# ───────────────────────────────────────────────────────────────────
# Index build (timestamp-aligned, mask-aware)
# ───────────────────────────────────────────────────────────────────

def build_aligned_index(
    segments: list[dict],
    anchors: np.ndarray,
    seq_len: int,
    n_assets: int,
    min_assets_present: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor timestamp, compute the slice start index for every asset.

    An asset's start is -1 (absent/masked) when either:
      (a) the asset has no segment in `segments`
      (b) the asof-backward lookup returns an end_idx < seq_len - 1 (not enough
          historical bars to fill a full window ending at this anchor)

    A sample (anchor row) is KEPT when the number of assets with start >= 0
    is >= `min_assets_present`. Rows below that threshold are discarded.

    Returns:
        sample_anchors:    (n_valid,) int64 -- kept anchor timestamps
        per_asset_starts:  (n_valid, n_assets) int32 -- start bar index per
                           asset, -1 = absent (trainer must mask this slot)
    """
    # Build seg_by_idx with duplicate-detection: a duplicate asset_idx would silently
    # overwrite the first segment, causing one instrument to vanish from the index.
    seg_by_idx: dict[int, dict] = {}
    for s in segments:
        idx = s["asset_idx"]
        if idx in seg_by_idx:
            _warnings_module.warn(
                f"[build_aligned_index] Duplicate asset_idx={idx} in segments list. "
                f"The second segment (asset_name={s.get('asset_name', '?')!r}) will "
                f"OVERWRITE the first. Deduplicate before calling this function.",
                stacklevel=2,
            )
        seg_by_idx[idx] = s

    n_anchors = len(anchors)
    starts = np.full((n_anchors, n_assets), -1, dtype=np.int32)

    for asset_idx in range(n_assets):
        seg = seg_by_idx.get(asset_idx)
        if seg is None:
            continue
        asset_ts = seg["timestamp"]
        if not isinstance(asset_ts, np.ndarray):
            asset_ts = np.asarray(asset_ts, dtype=np.int64)

        # end_idx[i] = index in asset_ts of the last bar at or before anchor[i]
        end_indices = _asof_backward(anchors, asset_ts)
        # window: [end_idx - seq_len + 1, end_idx]  (inclusive on both ends)
        candidate_starts = end_indices - seq_len + 1
        # Valid when the window start is non-negative (enough historical bars)
        valid = candidate_starts >= 0
        starts[valid, asset_idx] = candidate_starts[valid]

    # Count valid assets per anchor and filter by min_assets_present threshold
    n_present = (starts >= 0).sum(axis=1)  # (n_anchors,)
    keep = n_present >= min_assets_present
    return anchors[keep], starts[keep]


# ───────────────────────────────────────────────────────────────────
# Dataset class
# ───────────────────────────────────────────────────────────────────

class MultiAssetDataset(Dataset):
    """Synchronized N-asset dataset for V12 cross-asset attention training.

    Each __getitem__ returns [N_assets, T, F] obs + targets aligned to a
    single anchor timestamp. Assets missing data at that anchor are filled
    with zeros and flagged mask=False.

    Memory efficiency: the index (sample_anchors + per_asset_starts) stores
    only int32/int64 index arrays. Raw feature arrays stay in the original
    segment dicts (never duplicated). At __getitem__ time only one seq_len
    window is copied per asset per sample.

    Parameters
    ----------
    segments : list[dict]
        Per-asset segment dicts (same format as AntifragileDataset). Each MUST
        contain a "timestamp" key (int64 ms, sorted ascending).
    seq_len : int
        Number of time steps T per sample window.
    reward_horizons : list[int]
        Return horizons to include in targets (e.g. [1, 4, 16, 64]).
    anchor_schedule : AnchorSchedule, optional
        Which timestamps to align on. Defaults to btc_pace (asset_idx=0).
    n_assets : int, optional
        Total number of asset slots. Auto-detected from max asset_idx + 1 if
        not provided. Explicitly setting is recommended for exact slot count.
    min_assets_present : int, optional
        Minimum number of assets that must have data at a given anchor for the
        sample to be included. Default=1 (keep any anchor with >= 1 asset).
        Set to n_assets for a strict inner-join (no missing assets allowed).
    absent_target_nan : bool, optional
        When True, absent-asset TARGET slots are filled with NaN instead of 0.0.
        This makes a mask-ignoring trainer produce an immediate loud NaN rather
        than silently training toward 0. Obs slots always stay 0.0 regardless of
        this flag. Default=False (current behaviour, backward-compatible).
    """

    def __init__(
        self,
        segments: list[dict],
        seq_len: int = 96,
        reward_horizons: Optional[list[int]] = None,
        anchor_schedule: Optional[AnchorSchedule] = None,
        n_assets: Optional[int] = None,
        min_assets_present: int = 1,
        absent_target_nan: bool = False,
    ):
        if not segments:
            raise ValueError("MultiAssetDataset: segments list is empty")

        self.segments = segments
        self.seq_len = int(seq_len)
        self.reward_horizons = list(reward_horizons or [1, 4, 16, 64])

        # Auto-detect n_assets
        if n_assets is None:
            n_assets = max(s["asset_idx"] for s in segments) + 1
        self.n_assets = int(n_assets)

        if self.n_assets < 2:
            raise ValueError(
                "MultiAssetDataset requires n_assets >= 2. "
                "Use AntifragileDataset for single-asset training."
            )

        min_assets_present = max(1, min(int(min_assets_present), self.n_assets))
        self._min_assets_present = min_assets_present
        self._absent_target_nan = bool(absent_target_nan)

        # Validate that all segments carry timestamps
        for s in segments:
            if "timestamp" not in s:
                raise ValueError(
                    f"Segment for asset_idx={s.get('asset_idx', '?')} is missing "
                    f"'timestamp'. MultiAssetDataset requires timestamps for alignment."
                )

        # Build aligned numpy index
        schedule = anchor_schedule or AnchorSchedule()
        raw_anchors = schedule.build_anchors(segments)
        self.sample_anchors, self.per_asset_starts = build_aligned_index(
            segments, raw_anchors,
            seq_len=self.seq_len,
            n_assets=self.n_assets,
            min_assets_present=min_assets_present,
        )

        if len(self.sample_anchors) == 0:
            raise ValueError(
                f"MultiAssetDataset: no valid samples after alignment "
                f"(n_assets={self.n_assets}, min_assets_present={min_assets_present}, "
                f"seq_len={seq_len}). "
                f"Check that segments have overlapping timestamp ranges."
            )

        # Fast lookup by asset_idx (duplicate guard -- mirrors the check in build_aligned_index)
        self._seg_by_idx: dict[int, dict] = {}
        for s in segments:
            idx = s["asset_idx"]
            if idx in self._seg_by_idx:
                _warnings_module.warn(
                    f"[MultiAssetDataset] Duplicate asset_idx={idx} in segments list. "
                    f"Segment asset_name={s.get('asset_name', '?')!r} OVERWRITES the "
                    f"previously registered segment for that index. "
                    f"Deduplicate segments before constructing MultiAssetDataset.",
                    stacklevel=2,
                )
            self._seg_by_idx[idx] = s

        # Infer feature width F once (consistent across all assets)
        self._n_features = self._infer_n_features()

    def _infer_n_features(self) -> int:
        """Return F from the first segment that has features."""
        for a in range(self.n_assets):
            seg = self._seg_by_idx.get(a)
            if seg is not None and "features" in seg:
                return seg["features"].shape[1]
        raise ValueError("No segment with 'features' found in MultiAssetDataset")

    def __len__(self) -> int:
        return int(len(self.sample_anchors))

    def __getitem__(self, idx: int) -> dict:
        """Return one synchronized multi-asset sample.

        Returns
        -------
        dict with keys:
            obs:           torch.Tensor [N_assets, T, F]  float32
            asset_ids:     torch.Tensor [N_assets]        int64
            targets:       dict {h: torch.Tensor [N_assets, T] float32}
            mask:          torch.Tensor [N_assets, T]     bool
                           True = real data; False = zero-padded (missing asset/bars)
            anchor_ts_ms:  int (epoch ms)
        """
        anchor_ts = int(self.sample_anchors[idx])
        starts = self.per_asset_starts[idx]   # (n_assets,) int32

        F = self._n_features
        T = self.seq_len
        A = self.n_assets

        obs = np.zeros((A, T, F), dtype=np.float32)
        # When absent_target_nan=True, initialise targets with NaN so a mask-ignoring
        # trainer gets a loud NaN for absent-asset slots rather than silently training
        # toward 0.  Obs slots always stay 0.0 (NaN in obs would propagate through
        # attention and corrupt present-asset representations too).
        _target_fill = np.nan if self._absent_target_nan else 0.0
        targets: dict[int, np.ndarray] = {
            h: np.full((A, T), _target_fill, dtype=np.float32) for h in self.reward_horizons
        }
        mask = np.zeros((A, T), dtype=np.bool_)
        asset_ids = np.arange(A, dtype=np.int64)

        for a in range(A):
            start = int(starts[a])
            seg = self._seg_by_idx.get(a)
            if seg is None or start < 0:
                # Absent asset: obs=zeros, targets=zeros, mask=False (already)
                continue
            end = start + T
            obs[a] = seg["features"][start:end]
            mask[a, :] = True
            for h in self.reward_horizons:
                key = f"target_return_{h}"
                if key in seg:
                    arr = seg[key]
                    # Guard against a window that runs off the end of a short segment
                    avail = min(T, len(arr) - start)
                    if avail > 0:
                        targets[h][a, :avail] = arr[start:start + avail]
                        if avail < T:
                            # Partial window: mask the tail as missing
                            mask[a, avail:] = False

        return {
            "obs": torch.from_numpy(obs),
            "asset_ids": torch.from_numpy(asset_ids),
            "targets": {h: torch.from_numpy(targets[h]) for h in self.reward_horizons},
            "mask": torch.from_numpy(mask),
            "anchor_ts_ms": anchor_ts,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_stats(self) -> dict:
        """Report asset presence statistics across all samples.

        Returns a dict:
            n_samples: total valid samples
            mean_assets_present: average number of assets with data per sample
            per_asset_coverage: {asset_idx: fraction of samples where asset is present}
            inner_join_coverage: fraction of samples where ALL assets are present
        """
        n_samples = len(self.sample_anchors)
        present = (self.per_asset_starts >= 0)  # (n_samples, n_assets) bool
        per_asset = present.mean(axis=0)  # (n_assets,)
        inner_join = present.all(axis=1).mean()
        return {
            "n_samples": n_samples,
            "mean_assets_present": float(present.sum(axis=1).mean()),
            "per_asset_coverage": {a: float(per_asset[a]) for a in range(self.n_assets)},
            "inner_join_coverage": float(inner_join),
        }


# ───────────────────────────────────────────────────────────────────
# get_multi_loss wiring spec (imported by V12 trainer once the guard
# is removed; also importable as a standalone helper for unit tests)
# ───────────────────────────────────────────────────────────────────

def build_multi_asset_collate_fn(reward_horizons: list[int]):
    """Return a DataLoader collate_fn for MultiAssetDataset batches.

    The default torch collate would try to stack the 'targets' dict values,
    which works but produces inconsistent key types. This collate is explicit.
    """
    def collate(batch: list[dict]) -> dict:
        obs = torch.stack([b["obs"] for b in batch])           # [B, A, T, F]
        asset_ids = torch.stack([b["asset_ids"] for b in batch])   # [B, A]
        mask = torch.stack([b["mask"] for b in batch])         # [B, A, T]
        targets = {
            h: torch.stack([b["targets"][h] for b in batch])   # [B, A, T]
            for h in reward_horizons
        }
        anchor_ts = [b["anchor_ts_ms"] for b in batch]
        return {
            "obs": obs,
            "asset_ids": asset_ids,
            "targets": targets,
            "mask": mask,
            "anchor_ts_ms": anchor_ts,
        }
    return collate


# ───────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Synthetic 3-asset smoke. Verifies alignment + shape contract."""
    rng = np.random.default_rng(42)
    base_ts = 1_700_000_000_000
    segments = []
    for asset_idx in range(3):
        n = 1000
        ts = base_ts + np.arange(n) * 60_000 + rng.integers(-5_000, 5_000, n)
        ts.sort()
        F = 5
        seg = {
            "asset_idx": asset_idx,
            "asset_name": f"ASSET{asset_idx}",
            "timestamp": ts.astype(np.int64),
            "features": rng.standard_normal((n, F)).astype(np.float32),
            "target_return_1": rng.standard_normal(n).astype(np.float32),
            "target_return_4": rng.standard_normal(n).astype(np.float32),
        }
        segments.append(seg)

    ds = MultiAssetDataset(segments, seq_len=64, reward_horizons=[1, 4], n_assets=3)
    assert len(ds) > 100, f"expected >100 samples, got {len(ds)}"
    sample = ds[0]
    assert sample["obs"].shape == (3, 64, 5), sample["obs"].shape
    assert sample["mask"].shape == (3, 64)
    assert sample["asset_ids"].tolist() == [0, 1, 2]
    assert sample["targets"][1].shape == (3, 64)
    assert isinstance(sample["anchor_ts_ms"], int)
    assert sample["mask"].all(), "synthetic data should have all bars present"
    print("[MultiAssetDataset] smoke PASS  "
          "(%d samples, obs=%s, first_anchor_ts=%d)"
          % (len(ds), tuple(sample["obs"].shape), sample["anchor_ts_ms"]))


if __name__ == "__main__":
    _smoke_test()
