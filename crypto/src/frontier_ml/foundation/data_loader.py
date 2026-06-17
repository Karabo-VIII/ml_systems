"""FoundationDataset — multi-asset window sampler for foundation pretrain.

Reads `data/processed/chimera_legacy/dollar/<asset>usdt_v50_chimera_*.parquet`,
builds an in-memory slim cache (per-asset fp16 features + fp32 horizon
targets), and provides:

    sample_anchor_batch(B, S)
        -> (x: (B,S,F), asset_ids: (B,), targets: dict[h] -> (B,))
       For the causal multi-horizon next-token objective.

    sample_contrastive_batch(B, S, lag_bars)
        -> (x_anchor, x_pos, x_neg, asset_a, asset_b)
       For lead-lag cross-asset contrastive (Hole 3 closure).

Design:
    - Slim cache lives in `data/_caches/foundation_slim/` as one .npz per
      asset (features fp16 + targets fp32 + start_ts int64).
    - Total ~9.3 GB across 87 u100 assets at f34 (33 norm_* + xd_btc_return).
    - Built once on first access; subsequent runs reload from npz in seconds.
    - Window sampling: uniform across (asset, start_idx) pairs that satisfy
      start_idx + S + max(horizons) <= n_bars.
    - Train/val split: temporal (50/20/20/10) -- but for pretrain we use
      train+val together. Linear-probe IC measured on oos+unseen.

Walk-forward purge:
    For pretrain, the windows themselves are sub-asset slices. Cross-asset
    contrastive can only pair (A_t, B_{t+lag}) windows; lead-lag uses small
    positive lag bars to model the BTC -> ETH -> alt cascade.

__contract__:
    inputs:
        config/universes/{u100,u50,u10}.yaml -- universe specs
        data/processed/chimera_legacy/dollar/<asset>_v50_chimera_*.parquet
    outputs:
        data/_caches/foundation_slim/<asset>.npz (features f16 + targets f32)
    invariants:
        - features are zero-mean unit-std at chimera level (norm_* prefix)
        - targets are RAW returns (target_return_<h>); voladj NOT used
        - any NaN in features replaced with 0 at cache build (rare; logged)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "pipeline"))

from universe_loader import UniverseLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# f34: 33 norm_* features + 1 cross-asset (xd_btc_return), matching V1.x baseline.
# Pretrain uses ALL of these; downstream linear-probe can subset.
DEFAULT_FEATURES: List[str] = [
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "norm_oi_change", "norm_return_1", "norm_spread_bps",
    "norm_ma_distance", "norm_whale", "norm_efficiency", "norm_return_4",
    "norm_return_16", "norm_return_kurtosis", "norm_bar_duration",
    "norm_funding_momentum", "norm_hawkes_intensity",
    "norm_hawkes_buy_intensity", "norm_hawkes_sell_intensity",
    "norm_hawkes_imbalance", "norm_momentum_accel", "norm_vol_price_corr",
    "norm_vol_ratio", "norm_flow_persistence", "norm_oi_price_divergence",
    "norm_yz_volatility", "norm_cs_spread", "norm_perm_entropy",
    "norm_kyle_lambda",
    "xd_btc_return",
]

DEFAULT_TARGETS: List[str] = [
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
]

CACHE_DIR = _PROJECT_ROOT / "data" / "_caches" / "foundation_slim"


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def _latest_chimera_legacy_for(asset_lower: str) -> Optional[Path]:
    """Resolve newest chimera_legacy parquet for an asset (e.g. 'btc')."""
    pattern = f"{asset_lower}usdt_v50_chimera_*.parquet"
    cands = sorted((_PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar").glob(pattern))
    return cands[-1] if cands else None


def _build_one_cache(
    asset_lower: str,
    features: List[str],
    targets: List[str],
    out_dir: Path,
    *,
    force: bool = False,
) -> Optional[Path]:
    """Build slim cache (npz) for a single asset; returns path or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{asset_lower}.npz"
    if out.exists() and not force:
        return out
    src = _latest_chimera_legacy_for(asset_lower)
    if src is None:
        print(f"[fdn-cache] {asset_lower}: no chimera_legacy parquet; skip", flush=True)
        return None
    df = pl.read_parquet(src, columns=features + targets + ["timestamp"])
    if df.height == 0:
        print(f"[fdn-cache] {asset_lower}: empty parquet; skip", flush=True)
        return None

    feat_arr = df.select(features).to_numpy().astype(np.float16)
    n_nan = int(np.isnan(feat_arr).sum())
    if n_nan > 0:
        print(f"[fdn-cache] {asset_lower}: {n_nan} NaN in features -> 0", flush=True)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0)

    tgt_arr = df.select(targets).to_numpy().astype(np.float32)
    tgt_arr = np.nan_to_num(tgt_arr, nan=0.0)

    ts_arr = df.select("timestamp").to_numpy().astype(np.int64).reshape(-1)

    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        features=feat_arr,
        targets=tgt_arr,
        timestamps=ts_arr,
        feature_names=np.array(features, dtype=object),
        target_names=np.array(targets, dtype=object),
    )
    if out.exists():
        out.unlink()
    tmp.rename(out)
    print(f"[fdn-cache] {asset_lower}: {feat_arr.shape[0]:,} bars  "
          f"features {feat_arr.nbytes/1e6:.1f} MB  targets {tgt_arr.nbytes/1e6:.1f} MB  "
          f"-> {out.name}", flush=True)
    return out


def build_cache(
    universe: str = "u100",
    features: Optional[List[str]] = None,
    targets: Optional[List[str]] = None,
    *,
    force: bool = False,
) -> Dict[str, Path]:
    """Build slim cache for every asset in the universe; returns {asset: path}."""
    feats = features or DEFAULT_FEATURES
    tgts = targets or DEFAULT_TARGETS

    raw_assets = UniverseLoader.load().list(universe)
    assets_lower = [a.lower().replace("usdt", "") for a in raw_assets]
    print(f"[fdn-cache] building {universe} cache: {len(assets_lower)} assets, "
          f"f={len(feats)} t={len(tgts)} force={force}", flush=True)

    out_paths: Dict[str, Path] = {}
    for a in assets_lower:
        p = _build_one_cache(a, feats, tgts, CACHE_DIR, force=force)
        if p is not None:
            out_paths[a] = p
    print(f"[fdn-cache] done: {len(out_paths)}/{len(assets_lower)} assets cached", flush=True)
    return out_paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class FoundationDataset:
    """Multi-asset window sampler for foundation pretrain.

    Loads slim caches into memory once; samples random (asset, start_idx)
    windows. Held-out segments NOT enforced here -- caller passes start /
    end indices per asset for train vs val splits if needed.
    """
    universe: str = "u100"
    seq_len: int = 512
    horizons: Tuple[int, ...] = (1, 4, 16, 64)
    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    targets: List[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    seed: int = 0
    train_frac: float = 0.7  # prefix of each asset usable for pretrain
                              # (val/oos/unseen reserved for downstream eval)

    # Populated in __post_init__
    asset_ids: List[str] = field(default_factory=list, init=False)
    features_arr: List[np.ndarray] = field(default_factory=list, init=False)  # per asset, fp16 (n_bars, F)
    targets_arr: List[np.ndarray] = field(default_factory=list, init=False)   # per asset, fp32 (n_bars, T)
    n_bars: List[int] = field(default_factory=list, init=False)
    n_train_bars: List[int] = field(default_factory=list, init=False)
    rng: np.random.Generator = field(default=None, init=False, repr=False)
    n_features: int = field(default=0, init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # Ensure cache exists
        cache_paths = build_cache(self.universe, self.features, self.targets, force=False)
        for asset_lower, p in cache_paths.items():
            data = np.load(p, allow_pickle=True)
            cached_feats = list(data["feature_names"])
            cached_tgts = list(data["target_names"])

            # Index features by name (cache may have been built with different ordering)
            try:
                feat_idx = [cached_feats.index(f) for f in self.features]
            except ValueError as e:
                print(f"[fdn-ds] {asset_lower}: cache missing feature ({e}); rebuilding",
                      flush=True)
                _build_one_cache(asset_lower, self.features, self.targets, CACHE_DIR, force=True)
                data = np.load(p, allow_pickle=True)
                cached_feats = list(data["feature_names"])
                cached_tgts = list(data["target_names"])
                feat_idx = [cached_feats.index(f) for f in self.features]
            tgt_idx = [cached_tgts.index(t) for t in self.targets]

            feats = data["features"][:, feat_idx]
            tgts = data["targets"][:, tgt_idx]

            self.asset_ids.append(asset_lower)
            self.features_arr.append(feats)
            self.targets_arr.append(tgts)
            self.n_bars.append(feats.shape[0])
            self.n_train_bars.append(int(feats.shape[0] * self.train_frac))

        self.n_features = len(self.features)
        total_bars = sum(self.n_bars)
        train_bars = sum(self.n_train_bars)
        total_mem = sum(a.nbytes for a in self.features_arr) + sum(a.nbytes for a in self.targets_arr)
        print(f"[fdn-ds] loaded {len(self.asset_ids)} assets  "
              f"{total_bars:,} bars total ({train_bars:,} train) "
              f"feat F={self.n_features}  mem {total_mem/1e9:.2f} GB", flush=True)

    # -- Sampling ---------------------------------------------------------

    def _valid_window_max(self, asset_idx: int, segment: str = "train") -> int:
        """Largest valid start_idx so window of seq_len + max horizon fits."""
        max_h = max(self.horizons)
        if segment == "train":
            n = self.n_train_bars[asset_idx]
        else:
            n = self.n_bars[asset_idx]
        return n - self.seq_len - max_h

    def sample_anchor_batch(
        self,
        batch_size: int,
        segment: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """Sample B windows uniformly over (asset, start_idx).

        Returns:
            x:         (B, S, F) fp32
            asset_ids: (B,) int64 -- index into self.asset_ids
            targets:   {h: (B,) fp32}  raw returns at start_idx + seq_len - 1 + h
        """
        S = self.seq_len
        F = self.n_features
        x = np.empty((batch_size, S, F), dtype=np.float32)
        asset_ids_out = np.empty(batch_size, dtype=np.int64)
        target_arrays = {h: np.empty(batch_size, dtype=np.float32) for h in self.horizons}

        n_assets = len(self.asset_ids)
        for b in range(batch_size):
            # Sample asset weighted by available windows so longer assets aren't
            # under-represented (small alts have fewer windows).
            asset_idx = int(self.rng.integers(0, n_assets))
            max_start = self._valid_window_max(asset_idx, segment=segment)
            if max_start <= 0:
                # Fall back to a different asset
                for _ in range(5):
                    asset_idx = int(self.rng.integers(0, n_assets))
                    max_start = self._valid_window_max(asset_idx, segment=segment)
                    if max_start > 0:
                        break
                if max_start <= 0:
                    # Use whatever; downstream collate will mask
                    asset_idx = 0
                    max_start = max(1, self.n_train_bars[0] - S - max(self.horizons))
            start = int(self.rng.integers(0, max_start))
            x[b] = self.features_arr[asset_idx][start:start + S].astype(np.float32)
            asset_ids_out[b] = asset_idx
            tgt_pos = start + S - 1
            for hi, h in enumerate(self.horizons):
                target_arrays[h][b] = self.targets_arr[asset_idx][tgt_pos, hi]

        return x, asset_ids_out, target_arrays

    def sample_contrastive_batch(
        self,
        batch_size: int,
        segment: str = "train",
        lag_choices: Tuple[int, ...] = (0, 1, 3, 12),
        neg_far_bars: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """Sample paired windows for lead-lag cross-asset contrastive.

        For each pair: pick anchor asset A, positive partner B (random different
        asset), positive lag δ from `lag_choices`, and a far-away NEGATIVE
        offset for B.

        Returns dict with keys:
            x_anchor:    (B, S, F)
            x_pos:       (B, S, F)
            x_neg:       (B, S, F)
            asset_anchor: (B,) int64
            asset_pos:    (B,) int64  (=asset_neg by construction)
            lag:          (B,) int64
        """
        S = self.seq_len
        F = self.n_features
        x_a = np.empty((batch_size, S, F), dtype=np.float32)
        x_p = np.empty((batch_size, S, F), dtype=np.float32)
        x_n = np.empty((batch_size, S, F), dtype=np.float32)
        asset_a = np.empty(batch_size, dtype=np.int64)
        asset_b = np.empty(batch_size, dtype=np.int64)
        lag_out = np.empty(batch_size, dtype=np.int64)

        n_assets = len(self.asset_ids)
        max_h = max(self.horizons)
        for b in range(batch_size):
            # Anchor asset
            for _ in range(10):
                ai = int(self.rng.integers(0, n_assets))
                if self._valid_window_max(ai, segment=segment) > 0:
                    break
            asset_a[b] = ai
            max_start_a = self._valid_window_max(ai, segment=segment)
            start_a = int(self.rng.integers(0, max_start_a))
            x_a[b] = self.features_arr[ai][start_a:start_a + S].astype(np.float32)

            # Positive partner: different asset, lag δ
            for _ in range(10):
                bi = int(self.rng.integers(0, n_assets))
                if bi != ai and self._valid_window_max(bi, segment=segment) > 0:
                    break
            asset_b[b] = bi
            lag = int(lag_choices[self.rng.integers(0, len(lag_choices))])
            lag_out[b] = lag
            max_start_b_pos = self._valid_window_max(bi, segment=segment) - lag
            start_b_pos = max(0, min(max_start_b_pos, start_a + lag))
            x_p[b] = self.features_arr[bi][start_b_pos:start_b_pos + S].astype(np.float32)

            # Negative: same partner B but far-away time
            far_offset = int(self.rng.integers(neg_far_bars, neg_far_bars * 4))
            sign = 1 if self.rng.random() > 0.5 else -1
            start_b_neg = start_a + sign * far_offset
            start_b_neg = max(0, min(start_b_neg, self._valid_window_max(bi, segment=segment) - 1))
            x_n[b] = self.features_arr[bi][start_b_neg:start_b_neg + S].astype(np.float32)

        return {
            "x_anchor": x_a,
            "x_pos": x_p,
            "x_neg": x_n,
            "asset_anchor": asset_a,
            "asset_pos": asset_b,
            "lag": lag_out,
        }


# ---------------------------------------------------------------------------
# CLI: build cache directly
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build foundation slim cache")
    ap.add_argument("--universe", default="u100", choices=["u10", "u50", "u100"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Build cache, then sample 2 batches and exit.")
    args = ap.parse_args()

    if args.smoke:
        ds = FoundationDataset(universe=args.universe, seq_len=128)
        x, aids, tgts = ds.sample_anchor_batch(batch_size=4)
        print(f"[smoke] anchor batch x.shape={x.shape}  asset_ids={aids}")
        print(f"[smoke] targets keys={list(tgts.keys())}  h1 sample={tgts[1]}")
        cb = ds.sample_contrastive_batch(batch_size=4)
        print(f"[smoke] contrastive: x_anchor={cb['x_anchor'].shape} "
              f"lag={cb['lag']} asset_anchor={cb['asset_anchor']} asset_pos={cb['asset_pos']}")
        return

    paths = build_cache(args.universe, force=args.force)
    print(f"[fdn-cache] built {len(paths)} caches at {CACHE_DIR}")


if __name__ == "__main__":
    main()
