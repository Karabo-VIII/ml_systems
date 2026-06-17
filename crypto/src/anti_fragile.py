# CDAP contract
__contract__ = {
    "kind": "splitter",
    "module": "WalkForwardSplitter",
    "outputs": {
        "splits": "(train, val, oos, unseen) by index ranges with purge gaps",
    },
    "invariants": {
        "purge_gap_bars_min": 400,
        "split_ratio_default": [0.50, 0.20, 0.20, 0.10],
        "no_overlap_between_segments": True,
    },
    "rationale": "G-AUDIT-002: purge gap = 0 caused leakage in xsec ranker.",
}

"""
Anti-Fragile Training Framework for World Models V1-V4

Core principles:
  1. Walk-forward with purge gap (eliminates temporal leakage)
  2. Shuffled IC as primary model selection metric (detects memorization)
  3. Rich augmentation pipeline (temporal jitter, mixup, reversal)
  4. Regime-balanced sampling (prevents neutral-bias)
  5. Overfitting monitor (alerts when contiguous IC >> shuffled IC)

Usage in any trainer:
    from anti_fragile import (
        WalkForwardSplitter, AntifragileAugmentor, ShuffledICTracker,
        OverfitMonitor, regime_balanced_sampler, ANTIFRAGILE_DEFAULTS,
        N_MODELS, IC_THRESHOLD_UNADJUSTED, IC_THRESHOLD_BONFERRONI,
    )
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats as scipy_stats


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AntifragileConfig:
    """Anti-fragile training hyperparameters."""

    # Walk-forward validation
    # CRITICAL FIX: purge_gap must be >= normalization window to prevent data leakage.
    # The rolling z-score in sota_shared_logic_v50.py uses WINDOW_ADAPTIVE=200 bars,
    # so any validation bar within 200 bars of training data has normalization statistics
    # contaminated by training data. With seq_len=96, we need:
    #   purge_gap = max(normalization_window, seq_len) + safety_margin
    #            = WINDOW_ADAPTIVE(200) + hurst_window(200) = 400 bars
    # Hurst R/S uses window=200, then z-scored with window=200 (cascading dependency)
    purge_gap_bars: int = 400           # >= hurst(200) + z_score(200) cascading window
    n_walk_forward_folds: int = 3       # Number of temporal folds
    min_train_ratio: float = 0.50       # Minimum training data in first fold

    # 4-way data split: 50% train / 20% val / 20% OOS / 10% unseen
    # OOS = post-training evaluation (model selection, ensemble weighting)
    # Unseen = never touched during development (backtesting only)
    train_ratio: float = 0.50
    val_ratio: float = 0.20
    oos_ratio: float = 0.20
    unseen_ratio: float = 0.10

    # Shuffled IC (computed every N epochs)
    shuffled_ic_every: int = 10         # Compute shuffled IC every N epochs (10 = allow generalization before checking)
    shuffled_ic_folds: int = 3          # K-fold for shuffled IC per seed
    # 10 seeds gives ~0.32x sigma standard error — sufficient for 80-90% confidence.
    # Original was 3 (too noisy), then 20 (too slow at ~15 min/check).
    shuffled_ic_seeds: int = 10         # Number of random seeds for shuffled IC
    shuffled_ic_min: float = 0.015      # Minimum shuffled IC to save best model
    # 2026-06-10 OOM fix: cap bars per asset in the ShIC compute. On the full
    # multi-million-bar chimera, feats[fold_indices] (~len x 41 x 4 bytes) + the
    # full-length forward pass + corrcoef's float64 doubling exhaust host RAM.
    # ShIC is a diagnostic; a 90k random subset is statistically ample (per-fold
    # corr over ~30k points). 0 = uncapped (old behaviour).
    shuffled_ic_max_bars: int = 90000

    # Augmentation
    aug_temporal_jitter: int = 4        # Max temporal shift (bars)
    aug_mixup_alpha: float = 0.2        # Mixup interpolation (0 = off)
    aug_time_reverse_prob: float = 0.0  # DISABLED: flips features but NOT targets (corrupts 15% of samples)
    aug_noise_std: float = 0.02         # Gaussian noise (existing)
    aug_feat_drop: float = 0.10         # Feature dropout (existing)
    aug_block_swap_prob: float = 0.0    # DISABLED: creates RSSM state discontinuities at swap boundaries

    # Regime-balanced sampling
    regime_balance: bool = True         # Oversample rare regimes
    regime_balance_power: float = 0.5   # Smoothing power (0=uniform, 1=full inverse)

    # Overfitting detection
    overfit_ic_gap_warn: float = 0.10   # Warn if contiguous IC - shuffled IC > this
    overfit_ic_gap_stop: float = 0.30   # Stop training if gap exceeds this

    # Label smoothing -- DISABLED (accelerates temporal memorization, see CLAUDE.md)
    label_smoothing: float = 0.0


# Sensible defaults
ANTIFRAGILE_DEFAULTS = AntifragileConfig()

# =============================================================================
# MULTI-COMPARISON CORRECTION (Bonferroni)
# =============================================================================
# With 9 model architectures (V1-V9), each tested against IC_THRESHOLD=0.015,
# the probability of at least one spurious pass increases ~9x under the null
# hypothesis (family-wise error rate). Bonferroni correction divides the
# per-model significance threshold by the number of comparisons:
#
#   Bonferroni-corrected threshold = IC_THRESHOLD / N_MODELS
#                                  = 0.015 / 9 = 0.00167
#
# This is conservative (controls FWER). If too strict in practice, consider
# Holm-Bonferroni (step-down) or Benjamini-Hochberg (FDR control) instead.
#
# NOTE: The per-model gate check (shuffled_ic_min=0.015) is the UNADJUSTED
# threshold used during individual training runs. The Bonferroni threshold
# is APPLIED at the cohort-selection stage by src/wm/wm_tournament.py (2026-05-29),
# which computes the family size DYNAMICALLY from run_all_training.MODELS. This
# constant is the static fallback / reference. Individual training still uses the
# unadjusted threshold as a minimum quality bar.
N_MODELS = 19                                          # was 9 (stale); active non-archived cohort
IC_THRESHOLD_UNADJUSTED = 0.015                        # Per-model IC gate
IC_THRESHOLD_BONFERRONI = IC_THRESHOLD_UNADJUSTED / N_MODELS  # ~0.00079 at N=19


# =============================================================================
# WALK-FORWARD DATA SPLITTER WITH PURGE GAP
# =============================================================================

class WalkForwardSplitter:
    """
    Walk-forward cross-validation with purge gap and 4-way data split.

    Primary mode (split_four_way): Fixed 50/20/20/10 split:
        Train [0, 60%] --- gap --- Val [60%+gap, 80%] --- gap --- OOS [80%+gap, 90%] --- gap --- Unseen [90%+gap, 100%]

    Legacy mode (split_segments): Walk-forward CV with expanding training window:
        Fold 1: Train [0, 50%] --- gap --- Val [50%+gap, 60%]
        Fold 2: Train [0, 65%] --- gap --- Val [65%+gap, 80%]
        Fold 3: Train [0, 80%] --- gap --- Val [80%+gap, 100%]

    Purge gaps (400 bars) break autocorrelation between splits.
    """

    def __init__(self, config: AntifragileConfig = None):
        self.config = config or ANTIFRAGILE_DEFAULTS

    def split_segments(
        self,
        all_segments: List[dict],
        fold: int = -1,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Split segments into train/val for a given fold.

        Args:
            all_segments: Full data segments (each has 'features', 'asset_idx', targets)
            fold: Which fold to use (-1 = last fold = primary validation)

        Returns:
            (train_segments, val_segments)
        """
        n_folds = self.config.n_walk_forward_folds
        if fold == -1:
            fold = n_folds - 1

        train_segments = []
        val_segments = []

        for seg in all_segments:
            n_bars = len(seg["features"])
            purge = self.config.purge_gap_bars

            # Compute split points for this fold
            # Each fold uses more training data
            min_train = int(n_bars * self.config.min_train_ratio)
            remaining = n_bars - min_train
            fold_size = remaining // n_folds

            train_end = min_train + fold * fold_size
            val_start = train_end + purge
            val_end = min(train_end + fold_size + purge, n_bars)

            if val_start >= n_bars or val_end - val_start < 96:
                # Not enough data for this fold, use what we can
                val_start = max(train_end + purge, n_bars - 96)
                val_end = n_bars

            # Build segments
            train_seg = {"features": seg["features"][:train_end], "asset_idx": seg["asset_idx"]}
            val_seg = {"features": seg["features"][val_start:val_end], "asset_idx": seg["asset_idx"]}

            for key in seg:
                if key.startswith("target_return_") or key == "regime_label":
                    train_seg[key] = seg[key][:train_end]
                    val_seg[key] = seg[key][val_start:val_end]

            if len(train_seg["features"]) >= 96 and len(val_seg["features"]) >= 96:
                train_segments.append(train_seg)
                val_segments.append(val_seg)

        return train_segments, val_segments

    def split_all_folds(
        self,
        all_segments: List[dict],
    ) -> List[Tuple[List[dict], List[dict]]]:
        """Return all walk-forward folds."""
        folds = []
        for fold in range(self.config.n_walk_forward_folds):
            train_segs, val_segs = self.split_segments(all_segments, fold=fold)
            if train_segs and val_segs:
                folds.append((train_segs, val_segs))
        return folds


    def split_four_way(
        self,
        all_segments: List[dict],
    ) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
        """
        Split segments into train/val/oos/unseen using fixed ratios.

        Data split: 60% train / 20% val / 10% OOS / 10% unseen
        Purge gaps inserted between each split to prevent leakage.

        Returns:
            (train_segments, val_segments, oos_segments, unseen_segments)
        """
        train_segments = []
        val_segments = []
        oos_segments = []
        unseen_segments = []
        purge = self.config.purge_gap_bars
        min_len = 96  # minimum bars for a usable segment (one sequence window)

        # Pre-flight: compute the minimum bars any single segment needs for all four
        # splits to be non-empty.  Formula (worst-case, each split gets exactly min_len):
        #   need >= train_ratio_bars + purge + val_ratio_bars + purge
        #          + oos_ratio_bars + purge + unseen_ratio_bars
        # With the 50/20/20/10 ratios and purge=400, roughly:
        #   train(min_len=96) + purge + val(min_len=96) + purge
        #   + oos(min_len=96) + purge + unseen(min_len=96)  = 3*purge + 4*min_len
        # We compute the exact minimum by inverting the split-point formulas.
        # n * val_ratio - purge >= min_len  -->  n >= (min_len + purge) / val_ratio
        # Similarly for oos and unseen.  Train is the largest slice so it easily satisfies.
        val_ratio = self.config.val_ratio
        oos_ratio = self.config.oos_ratio
        unseen_ratio = self.config.unseen_ratio
        # Minimum bars so val window (val_end - val_start) >= min_len
        min_for_val = int(np.ceil((min_len + purge) / val_ratio)) + 1
        # Minimum bars so oos window >= min_len
        min_for_oos = int(np.ceil((min_len + purge) / oos_ratio)) + 1
        # Minimum bars so unseen window >= min_len
        min_for_unseen = int(np.ceil((min_len + purge) / unseen_ratio)) + 1
        min_bars_required = max(min_for_val, min_for_oos, min_for_unseen)

        for seg in all_segments:
            n_bars = len(seg["features"])

            # Guard: raise early with a clear message so upstream code doesn't silently
            # produce an empty DataLoader (AntifragileDataset([], ...) is len 0 -- hard
            # to diagnose).  The crypto path uses 50k+ bar segments; this guard is only
            # triggered on small synthetic / test / Layer-B datasets.
            if n_bars < min_bars_required:
                raise ValueError(
                    f"split_four_way: segment (asset_idx={seg.get('asset_idx', '?')}) "
                    f"has {n_bars} bars -- too small to produce non-empty val/oos/unseen splits. "
                    f"Minimum required: {min_bars_required} bars "
                    f"(given purge_gap_bars={purge}, seq_len={min_len}, "
                    f"split_ratios train={self.config.train_ratio}/"
                    f"val={val_ratio}/oos={oos_ratio}/unseen={unseen_ratio}). "
                    f"Reduce purge_gap_bars or provide more data."
                )

            # Split points
            train_end = int(n_bars * self.config.train_ratio)
            val_end = int(n_bars * (self.config.train_ratio + val_ratio))
            oos_end = int(n_bars * (self.config.train_ratio + val_ratio + oos_ratio))

            val_start = train_end + purge
            oos_start = val_end + purge
            unseen_start = oos_end + purge

            def _slice_seg(seg, start, end):
                s = {"features": seg["features"][start:end], "asset_idx": seg["asset_idx"]}
                for key in seg:
                    if key.startswith("target_return_") or key == "regime_label":
                        s[key] = seg[key][start:end]
                return s

            # Build segments (skip if too short for one sequence)
            t_seg = _slice_seg(seg, 0, train_end)
            v_seg = _slice_seg(seg, val_start, val_end)
            o_seg = _slice_seg(seg, oos_start, oos_end)
            u_seg = _slice_seg(seg, unseen_start, n_bars)

            if len(t_seg["features"]) >= min_len:
                train_segments.append(t_seg)
            if len(v_seg["features"]) >= min_len:
                val_segments.append(v_seg)
            if len(o_seg["features"]) >= min_len:
                oos_segments.append(o_seg)
            if len(u_seg["features"]) >= min_len:
                unseen_segments.append(u_seg)

        return train_segments, val_segments, oos_segments, unseen_segments

    def split_four_way_dated(
        self,
        all_segments: List[dict],
        boundaries: Optional[dict] = None,
    ) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
        """Calendar-aligned 4-way split using frozen dates from config/data_config.yaml.

        Eliminates cross-asset calendar overlap leakage: every asset's train/val/oos/unseen
        boundary falls on the SAME calendar date, so xd_btc_return in AVAX's val window
        cannot be derived from BTC bars that BTC's training set used.

        Each segment must carry a 'timestamp' key (epoch ms). Use load_full_data() with
        chimera files containing the timestamp column.

        Args:
            all_segments: list of segment dicts with 'features', 'timestamp', etc.
            boundaries: dict with 'train_end', 'val_end', 'oos_end' as 'YYYY-MM-DD' strings,
                        plus 'purge_bars' (int). If None, loads from config/data_config.yaml.

        Returns:
            (train_segments, val_segments, oos_segments, unseen_segments)
        """
        from datetime import datetime, timezone

        if boundaries is None:
            # Lazy import to avoid circular: pipeline.purge_split → chimera_loader → ...
            import sys as _sys
            from pathlib import Path as _Path
            _pipeline_dir = _Path(__file__).resolve().parent / "pipeline"
            if str(_pipeline_dir) not in _sys.path:
                _sys.path.insert(0, str(_pipeline_dir))
            from purge_split import get_split_dates  # noqa: E402
            b = get_split_dates()
            boundaries = b.as_dict()

        # Convert YYYY-MM-DD strings → epoch ms
        def _date_to_ms(d_str: str) -> int:
            dt = datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)

        train_end_ms = _date_to_ms(boundaries["train_end"])
        val_end_ms = _date_to_ms(boundaries["val_end"])
        oos_end_ms = _date_to_ms(boundaries["oos_end"])
        purge = int(boundaries.get("purge_bars", self.config.purge_gap_bars))

        train_segments: list[dict] = []
        val_segments: list[dict] = []
        oos_segments: list[dict] = []
        unseen_segments: list[dict] = []
        min_len = 96

        def _slice_seg(seg: dict, start_idx: int, end_idx: int) -> dict:
            s = {"features": seg["features"][start_idx:end_idx],
                 "asset_idx": seg["asset_idx"]}
            for key in seg:
                if key.startswith("target_return_") or key == "regime_label" or key == "timestamp":
                    s[key] = seg[key][start_idx:end_idx]
            return s

        for seg in all_segments:
            if "timestamp" not in seg:
                raise ValueError(
                    f"split_four_way_dated requires 'timestamp' in each segment. "
                    f"Use load_full_data() against a chimera with a timestamp column."
                )
            ts = seg["timestamp"]
            n = len(ts)

            # np.searchsorted assumes monotonic; chimera timestamps are sorted by bar_id
            train_end = int(np.searchsorted(ts, train_end_ms))
            val_end = int(np.searchsorted(ts, val_end_ms))
            oos_end = int(np.searchsorted(ts, oos_end_ms))

            val_start = min(train_end + purge, n)
            oos_start = min(val_end + purge, n)
            unseen_start = min(oos_end + purge, n)

            t_seg = _slice_seg(seg, 0, train_end)
            v_seg = _slice_seg(seg, val_start, val_end)
            o_seg = _slice_seg(seg, oos_start, oos_end)
            u_seg = _slice_seg(seg, unseen_start, n)

            if len(t_seg["features"]) >= min_len:
                train_segments.append(t_seg)
            if len(v_seg["features"]) >= min_len:
                val_segments.append(v_seg)
            if len(o_seg["features"]) >= min_len:
                oos_segments.append(o_seg)
            if len(u_seg["features"]) >= min_len:
                unseen_segments.append(u_seg)

        return train_segments, val_segments, oos_segments, unseen_segments


def load_full_data(data_dir, feature_list, asset_to_idx, reward_horizons,
                   file_glob="*_v51_chimera*.parquet",
                   target_prefix=None):
    """
    Load full (unsplit) data from parquet files.

    Returns list of segment dicts, each with:
      - 'features': np.ndarray [N, C]
      - 'asset_idx': int
      - 'target_return_{h}': np.ndarray [N] for each horizon h

    Args:
        file_glob: glob pattern for parquet files. DEFAULT IS v51 (clean returns,
                   no silent fill_null(0), per V50_TO_V51_FIXES). To intentionally
                   load v50 legacy data, pass the legacy glob explicitly.
                   Before 2026-05-17 default was v50 which inflated WM ROI
                   claims; corrected per connector_integrity_crawler A1.
        target_prefix: Column name prefix for targets. None = 'target_return' (raw).
                       Use 'target_voladj' to opt in to vol-adjusted targets.
                       Segment dicts always use 'target_return_{h}' as internal keys
                       regardless of source column, for downstream compatibility.
    """
    import polars as pl
    from pathlib import Path

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(file_glob))
    if not files:
        print(f"  [ERROR] No data files found in {data_dir}")
        return None

    # Deduplicate: layout v3 keeps multiple dated snapshots per asset
    # (<sym>usdt_v50_chimera_<YYYYMMDD>.parquet). Pick the NEWEST VALID
    # snapshot per asset — falls back to older snapshot if newer is missing
    # required feature columns (defends against partial pipeline builds).
    by_asset: dict[str, list["Path"]] = {}
    for f in files:
        asset_name = f.stem.split("_")[0].upper()
        if asset_name not in asset_to_idx:
            continue
        by_asset.setdefault(asset_name, []).append(f)

    selected = []
    required_cols = set(feature_list)
    for asset_name, candidates in by_asset.items():
        candidates.sort(key=lambda p: p.name, reverse=True)  # newest first
        chosen = None
        for cand in candidates:
            try:
                cols = set(pl.read_parquet_schema(cand).keys())
            except Exception:
                continue
            missing = required_cols - cols
            if not missing:
                chosen = cand
                break
            if cand is candidates[0]:
                print(f"  [WARN] {asset_name}: latest snapshot {cand.name} missing "
                      f"{len(missing)} cols ({sorted(missing)[:3]}...); "
                      f"falling back to older snapshot")
        if chosen is None:
            print(f"  [ERROR] {asset_name}: NO snapshot has all required columns; skipping")
            continue
        selected.append(chosen)
    files = sorted(selected)

    all_segments = []
    total_bars = 0

    for f in files:
        asset_name = f.stem.split("_")[0].upper()
        if asset_name not in asset_to_idx:
            continue

        asset_idx = asset_to_idx[asset_name]

        try:
            df = pl.read_parquet(f)
            rows_before = len(df)

            # Resolve target prefix (default: raw returns, not voladj)
            # Voladj targets create a vol shortcut -- model predicts vol, not returns.
            if target_prefix is None:
                effective_prefix = "target_return"
            else:
                effective_prefix = target_prefix

            # Column-selective drop_nulls: only drop rows where MODEL inputs are null
            # (not auxiliary columns like buy_vol, sell_vol, volume_usd)
            drop_subset = list(feature_list) + [
                f"{effective_prefix}_{h}" for h in reward_horizons
            ]
            drop_subset = [c for c in drop_subset if c in df.columns]
            df = df.drop_nulls(subset=drop_subset)

            rows_after = len(df)
            pct_dropped = (1 - rows_after / max(rows_before, 1)) * 100
            if pct_dropped > 10:
                print(f"  [WARN] {asset_name}: Dropped {pct_dropped:.1f}% of rows "
                      f"during null removal ({rows_before:,} -> {rows_after:,})")

            # Validate schema and data quality BEFORE extracting numpy arrays
            missing_feat = [fn for fn in feature_list if fn not in df.columns]
            if missing_feat:
                raise ValueError(
                    f"{asset_name}: Missing feature columns: {missing_feat}. "
                    f"Available: {[c for c in df.columns if c.startswith('norm_') or c.startswith('hurst') or c.startswith('xd_')]}"
                )

            target_cols = [f"{effective_prefix}_{h}" for h in reward_horizons]
            missing_tgt = [t for t in target_cols if t not in df.columns]
            if missing_tgt:
                raise ValueError(
                    f"{asset_name}: Missing target columns: {missing_tgt}"
                )

            # Check for degenerate features (std < 0.01 = dead signal)
            for feat_name in feature_list:
                std_val = df[feat_name].std()
                if std_val is not None and std_val < 0.01:
                    print(f"  [WARN] {asset_name}: Feature {feat_name} "
                          f"has std={std_val:.6f} (degenerate)")

            # Check for degenerate targets (std < 1e-6 = no signal)
            for t in target_cols:
                std_val = df[t].std()
                if std_val is not None and std_val < 1e-6:
                    raise ValueError(
                        f"{asset_name}: Target {t} has std={std_val:.9f} (degenerate)"
                    )

            # Extract numpy arrays (no zero-padding -- validated above)
            feat_arrays = [
                df[feat_name].to_numpy().astype(np.float32)
                for feat_name in feature_list
            ]
            feats = np.column_stack(feat_arrays)

            seg = {"features": feats, "asset_idx": asset_idx}
            for h in reward_horizons:
                src_col = f"{effective_prefix}_{h}"
                # Internal key is always target_return_{h} for downstream compat
                seg[f"target_return_{h}"] = df[src_col].to_numpy().astype(np.float32)

            # Include precomputed SMA regime labels if available
            if "regime_label" in df.columns:
                seg["regime_label"] = df["regime_label"].to_numpy().astype(np.int64)

            # Carry timestamp (epoch ms) for date-based splitting (split_four_way_dated)
            # Chimera invariant: timestamp is 13-digit milliseconds in [1.5e12, 2.0e12]
            if "timestamp" in df.columns:
                seg["timestamp"] = df["timestamp"].to_numpy().astype(np.int64)

            all_segments.append(seg)
            total_bars += len(feats)
            print(f"  {asset_name}: {len(feats):,} bars ({pct_dropped:.1f}% null-dropped)")

        except Exception as e:
            print(f"  [ERROR] Loading {f.name}: {e}")

    if not all_segments:
        return None

    print(f"  Total: {total_bars:,} bars across {len(all_segments)} assets")
    return all_segments


# =============================================================================
# ANTI-FRAGILE AUGMENTATION PIPELINE
# =============================================================================

class AntifragileAugmentor:
    """
    Rich data augmentation pipeline for time-series.

    Augmentations (applied stochastically per sample):
      1. Gaussian noise injection (existing)
      2. Feature dropout (existing)
      3. Temporal jitter: shift sequence start by ±N bars
      4. Mixup: interpolate between two sequences
      5. Time reversal: flip time axis (features remain, targets invalidated)
      6. Block swap: swap two random blocks within the sequence
    """

    def __init__(self, config: AntifragileConfig = None):
        self.config = config or ANTIFRAGILE_DEFAULTS

    def augment_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Augment a single observation tensor [T, C] or batch [B, T, C].

        Applies noise, feature dropout, and block swap.
        Temporal jitter and mixup are handled at dataset level.
        """
        aug = obs.clone()

        # 1. Gaussian noise
        if self.config.aug_noise_std > 0:
            aug = aug + torch.randn_like(aug) * self.config.aug_noise_std

        # 2. Feature dropout (zero entire columns)
        if self.config.aug_feat_drop > 0:
            if aug.dim() == 3:
                B, T, C = aug.shape
                feat_mask = (torch.rand(B, 1, C, device=aug.device) > self.config.aug_feat_drop).float()
            else:
                T, C = aug.shape
                feat_mask = (torch.rand(1, C, device=aug.device) > self.config.aug_feat_drop).float()
            aug = aug * feat_mask

        # 3. Block swap (swap two random blocks)
        if self.config.aug_block_swap_prob > 0 and torch.rand(1).item() < self.config.aug_block_swap_prob:
            aug = self._block_swap(aug)

        # 4. Time reversal (flip time axis)
        if self.config.aug_time_reverse_prob > 0 and torch.rand(1).item() < self.config.aug_time_reverse_prob:
            if aug.dim() == 3:
                aug = aug.flip(1)  # [B, T, C] -> flip T
            else:
                aug = aug.flip(0)  # [T, C] -> flip T

        return aug

    def _block_swap(self, obs: torch.Tensor) -> torch.Tensor:
        """Swap two random blocks of 8-16 bars within the sequence."""
        if obs.dim() == 3:
            T = obs.shape[1]
        else:
            T = obs.shape[0]

        block_size = np.random.randint(8, min(17, T // 4 + 1))
        if T < block_size * 3:
            return obs

        # Pick two non-overlapping blocks
        a_start = np.random.randint(0, T - block_size * 2)
        b_start = np.random.randint(a_start + block_size, T - block_size)

        aug = obs.clone()
        if aug.dim() == 3:
            tmp = aug[:, a_start:a_start+block_size, :].clone()
            aug[:, a_start:a_start+block_size, :] = aug[:, b_start:b_start+block_size, :]
            aug[:, b_start:b_start+block_size, :] = tmp
        else:
            tmp = aug[a_start:a_start+block_size, :].clone()
            aug[a_start:a_start+block_size, :] = aug[b_start:b_start+block_size, :]
            aug[b_start:b_start+block_size, :] = tmp

        return aug

    def mixup_batch(
        self,
        obs: torch.Tensor,
        targets: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Mixup augmentation: interpolate between pairs of samples in batch.

        obs_mixed = lambda * obs_i + (1 - lambda) * obs_j
        target_mixed = lambda * target_i + (1 - lambda) * target_j

        Only applied with probability proportional to mixup_alpha.
        Alpha=0.2 means most samples are close to original (lambda ~= 1).
        """
        if self.config.aug_mixup_alpha <= 0:
            return obs, targets

        B = obs.shape[0]
        if B < 2:
            return obs, targets

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.config.aug_mixup_alpha, self.config.aug_mixup_alpha)
        lam = max(lam, 1 - lam)  # Ensure lambda >= 0.5 (closer to original)

        # Random permutation for mixing pairs
        perm = torch.randperm(B, device=obs.device)

        obs_mixed = lam * obs + (1 - lam) * obs[perm]

        targets_mixed = {}
        for h, t in targets.items():
            if isinstance(h, int):
                # Continuous targets: interpolate
                targets_mixed[h] = lam * t + (1 - lam) * t[perm]
            else:
                # Discrete labels (e.g. regime_label): keep dominant sample's labels
                targets_mixed[h] = t

        return obs_mixed, targets_mixed

    def temporal_jitter_index(self, start: int, max_len: int, seq_len: int) -> int:
        """
        Apply temporal jitter to a sequence start index.

        Shifts start by ±aug_temporal_jitter bars (clamped to valid range).
        """
        if self.config.aug_temporal_jitter <= 0:
            return start

        jitter = np.random.randint(
            -self.config.aug_temporal_jitter,
            self.config.aug_temporal_jitter + 1,
        )
        new_start = max(0, min(start + jitter, max_len - seq_len))
        return new_start


# =============================================================================
# SHUFFLED IC TRACKER (for in-training overfitting detection)
# =============================================================================

class ShuffledICTracker:
    """
    Computes shuffled IC during training to detect overfitting.

    Instead of using contiguous validation data, shuffles the full dataset
    and computes IC on random folds. If model is memorizing temporal order,
    shuffled IC will be near zero.

    SEMANTICS (documented 2026-05-29 per RED-team audit): this shuffles ROWS
    then re-windows them into seq_len sequences. So it tests "can the model
    predict from a SCRAMBLED sequence" -- i.e. SEQUENCE-SCRAMBLING robustness --
    NOT strictly "does the per-bar cross-sectional signal survive permutation"
    (the target-permutation test). The two coincide for a feed-forward predictor
    but diverge for a strong recurrent/sequence model, which a scrambled window
    legitimately cannot use -> such a model can score LOW ShIC for a benign
    reason (fed an incoherent sequence), not memorization. The gate DIRECTION is
    right (low ShIC + high IC => suspicious); read a near-zero ShIC on a recurrent
    model as "sequence-dependent", confirm memorization with the held-out OOS gap,
    not ShIC alone. A pure target-permutation variant is a documented future option.
    """

    def __init__(self, config: AntifragileConfig = None):
        self.config = config or ANTIFRAGILE_DEFAULTS
        self.history = {
            "epoch": [],
            "contiguous_ic": [],
            "shuffled_ic": [],
            "ic_gap": [],
        }

    @torch.no_grad()
    def compute_shuffled_ic(
        self,
        model: torch.nn.Module,
        all_segments: List[dict],
        predict_fn,
        horizon: int = 1,
        seed: int = 42,
    ) -> float:
        """
        Compute IC on shuffled data folds.

        Args:
            model: The world model (eval mode)
            all_segments: Full data segments (not split)
            predict_fn: Function(model, obs_batch, asset_ids) -> predictions
            horizon: Which return horizon to evaluate
            seed: Random seed for reproducibility

        Returns:
            Average shuffled IC across folds
        """
        model.eval()

        # Free VRAM before 300 forward passes (10 seeds x 10 assets x 3 folds).
        # Training + validation fill the CUDA cache; without this, OOM on 8GB GPUs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_seeds = self.config.shuffled_ic_seeds
        seeds = [seed + i * 1000 for i in range(n_seeds)]
        seed_ics = []

        from tqdm import tqdm
        print(f"  [ShIC] Computing shuffled IC ({n_seeds} seeds x {len(all_segments)} assets x {self.config.shuffled_ic_folds} folds)")

        pbar = tqdm(seeds, desc="  ShIC seeds", leave=False,
                    bar_format="  {l_bar}{bar:30}{r_bar}")

        for s in pbar:
            # 2026-05-10 OOM mitigation: ShIC compute accumulates GPU memory
            # across seeds; V3 OOM'd at seed 7/10 during gauntlet btg3bfzn5,
            # V4 same pattern earlier. Empty cache per-seed bounds peak usage.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rng = np.random.default_rng(s)
            all_ics = []

            for seg in all_segments:
                feats = seg["features"]
                targets = seg.get(f"target_return_{horizon}", None)
                asset_idx = seg["asset_idx"]

                if targets is None or len(feats) < 192:
                    continue

                # Create shuffled indices
                n = len(feats)
                indices = np.arange(n)
                rng.shuffle(indices)

                # 2026-06-10 OOM fix: cap bars per asset so the fancy-index copy
                # feats[fold_indices] + the forward pass + corrcoef stay bounded on
                # the full multi-million-bar chimera (the user's V1.1 run OOM'd here).
                cap = getattr(self.config, "shuffled_ic_max_bars", 90000)
                if cap and n > cap:
                    indices = indices[:cap]
                    n = cap

                fold_size = n // self.config.shuffled_ic_folds

                for fold in range(self.config.shuffled_ic_folds):
                    fold_start = fold * fold_size
                    fold_end = fold_start + fold_size if fold < self.config.shuffled_ic_folds - 1 else n

                    fold_indices = indices[fold_start:fold_end]
                    fold_feats = feats[fold_indices]
                    fold_targets = targets[fold_indices]

                    # Predict on shuffled data
                    preds = predict_fn(model, fold_feats, asset_idx, horizon)

                    if len(preds) < 30:
                        continue

                    # Align lengths
                    min_len = min(len(preds), len(fold_targets))
                    p = preds[:min_len]
                    r = fold_targets[:min_len]

                    mask = np.isfinite(p) & np.isfinite(r)
                    p, r = p[mask], r[mask]

                    if len(p) > 30 and np.std(p) > 1e-10 and np.std(r) > 1e-10:
                        # Memory-light Pearson in float32 (np.corrcoef promotes to
                        # float64 + concatenates p,r -- doubling RAM; the OOM site).
                        pf = p.astype(np.float32); rf = r.astype(np.float32)
                        pf = pf - pf.mean(); rf = rf - rf.mean()
                        denom = float(np.sqrt(float((pf * pf).sum()))) * float(np.sqrt(float((rf * rf).sum())))
                        ic = (float((pf * rf).sum()) / denom) if denom > 1e-12 else 0.0
                        if np.isfinite(ic):
                            all_ics.append(ic)
                    del fold_feats, preds

            if all_ics:
                seed_ics.append(float(np.mean(all_ics)))

            avg_so_far = np.mean(seed_ics) if seed_ics else 0.0
            pbar.set_postfix(IC=f"{avg_so_far:.4f}")

        result = float(np.mean(seed_ics)) if seed_ics else 0.0
        print(f"  [ShIC] Result: IC={result:.4f}", flush=True)
        return result

    def record(self, epoch: int, contiguous_ic: float, shuffled_ic: float):
        """Record IC values for monitoring."""
        self.history["epoch"].append(epoch)
        self.history["contiguous_ic"].append(contiguous_ic)
        self.history["shuffled_ic"].append(shuffled_ic)
        self.history["ic_gap"].append(contiguous_ic - shuffled_ic)


# =============================================================================
# OVERFITTING MONITOR
# =============================================================================

class OverfitMonitor:
    """
    Monitors training for signs of overfitting.

    Tracks:
      - Contiguous IC vs Shuffled IC gap
      - Training loss vs validation loss gap
      - Gradient norm trends
      - IC stability across folds
    """

    def __init__(self, config: AntifragileConfig = None):
        self.config = config or ANTIFRAGILE_DEFAULTS
        self.ic_tracker = ShuffledICTracker(config)

    def check_overfit(
        self,
        contiguous_ic: float,
        shuffled_ic: float,
        epoch: int,
    ) -> Tuple[bool, str]:
        """
        Check if model is overfitting.

        Returns:
            (should_stop, reason)
        """
        self.ic_tracker.record(epoch, contiguous_ic, shuffled_ic)

        ic_gap = contiguous_ic - shuffled_ic

        # Only flag memorization if ShIC was established as meaningful (above min threshold)
        # and then dropped. Near-zero ShIC (positive or negative) is noise, not a
        # memorization signal — especially with RevIN + RSSM where shuffled sequences
        # have different per-sequence normalization stats than contiguous sequences.
        shic_meaningful = shuffled_ic > self.config.shuffled_ic_min  # default 0.015
        if shic_meaningful and contiguous_ic > 0 and ic_gap > self.config.overfit_ic_gap_stop:
            return True, (
                f"OVERFIT DETECTED: IC gap = {ic_gap:.4f} > {self.config.overfit_ic_gap_stop} "
                f"(contiguous={contiguous_ic:.4f}, shuffled={shuffled_ic:.4f})"
            )

        if ic_gap > self.config.overfit_ic_gap_warn and shic_meaningful:
            print(
                f"  [WARN] IC gap = {ic_gap:.4f} > {self.config.overfit_ic_gap_warn} "
                f"(contiguous={contiguous_ic:.4f}, shuffled={shuffled_ic:.4f})"
            )

        return False, ""


# =============================================================================
# REGIME-BALANCED SAMPLING
# =============================================================================

def compute_regime_weights(
    segments: List[dict],
    horizon: int = 1,
    power: float = 0.5,
) -> np.ndarray:
    """
    Compute per-sample weights for regime-balanced sampling.

    Oversamples underrepresented regimes (bearish, bullish)
    and undersamples dominant regime (neutral).

    Args:
        segments: Data segments with target returns
        horizon: Which return horizon to use for regime classification
        power: Smoothing power (0=uniform, 1=full inverse-frequency)

    Returns:
        Array of per-sample weights aligned with dataset indices
    """
    all_weights = []

    for seg in segments:
        key = f"target_return_{horizon}"
        if key not in seg:
            all_weights.append(np.ones(len(seg["features"]), dtype=np.float32))
            continue

        targets = seg[key]
        ret_std = np.std(targets) + 1e-6

        # Classify regimes
        regimes = np.ones(len(targets), dtype=int)  # neutral
        regimes[targets > ret_std * 0.5] = 2  # bullish
        regimes[targets < -ret_std * 0.5] = 0  # bearish

        # Count regime frequencies
        counts = np.bincount(regimes, minlength=3).astype(float)
        counts = np.maximum(counts, 1.0)

        # Inverse frequency weighting with smoothing
        freq = counts / counts.sum()
        weights_per_regime = (1.0 / freq) ** power
        weights_per_regime = weights_per_regime / weights_per_regime.sum() * 3.0

        # Assign weights per sample
        sample_weights = np.array([weights_per_regime[r] for r in regimes], dtype=np.float32)
        all_weights.append(sample_weights)

    return all_weights


# =============================================================================
# ANTI-FRAGILE DATASET (drop-in replacement for MultiAssetSequenceDataset)
# =============================================================================

class AntifragileDataset(torch.utils.data.Dataset):
    """
    Anti-fragile dataset with rich augmentation.

    Drop-in replacement for existing MultiAssetSequenceDataset.
    Adds: temporal jitter, time reversal, block swap.
    Mixup is applied at batch level in the training loop.
    """

    def __init__(
        self,
        segments: List[dict],
        seq_len: int = 96,
        reward_horizons: list = None,
        augment: bool = False,
        config: AntifragileConfig = None,
        sample_weights: List[np.ndarray] = None,
        stride: int | None = None,
    ):
        """
        Args:
            stride: window stride. Default (None) uses seq_len // 4 for
                legacy behavior (V1.x/V3/V4/V6/V8 — supervises every bar
                so stride=24 keeps a healthy sample count without
                excessive overlap). Pass stride=1 for last-bar-supervision
                models (V22/V25 Timer-XL pattern) so consecutive bars
                each become a supervised target. 2026-05-21 oracle
                validation fix.
        """
        self.seq_len = seq_len
        self.segments = segments
        self.augment = augment
        self.config = config or ANTIFRAGILE_DEFAULTS
        self.reward_horizons = reward_horizons or [1, 4, 16, 64]
        self.augmentor = AntifragileAugmentor(self.config) if augment else None

        # Build window index -- MEMORY-EFFICIENT (2026-06-10 cohort-wide hardening). The old code
        # materialized a Python LIST of (seg_idx, start) tuples, one per window. On the full ~30M-bar
        # dataset with the V22/V25 Timer-XL small-stride pattern that is ~30M tuples ~= 3GB of host
        # RAM on top of the ~5GB feature tensors -> MemoryError (anti_fragile.py:982): the ENTIRE WM
        # cohort could not train on the full data. Storing the index as int32 numpy arrays cuts it
        # ~10x (~360MB) and is behaviour-preserving: identical windows (np.arange == range), identical
        # windowed-mean weights (vectorized via prefix sums). External callers use get_sampler()/
        # .weights only (audited) -- .index is internal, so this is a safe swap.
        if stride is None:
            stride = max(1, seq_len // 4)
        self.stride = stride

        seg_id_parts, start_parts, weight_parts = [], [], []
        for seg_idx, seg in enumerate(segments):
            feats = seg["features"]
            n_samples = len(feats) - seq_len  # Valid positions: 0 to len-seq_len (inclusive)
            if n_samples <= 0:
                continue
            starts = np.arange(0, n_samples, self.stride, dtype=np.int32)
            seg_id_parts.append(np.full(starts.shape, seg_idx, dtype=np.int32))
            start_parts.append(starts)

            seg_weights = sample_weights[seg_idx] if sample_weights else None
            if seg_weights is not None:
                # windowed mean over [start, min(start+seq_len, len)] per start, vectorized
                sw = np.asarray(seg_weights, dtype=np.float64)
                pref = np.concatenate(([0.0], np.cumsum(sw)))
                ends = np.minimum(starts.astype(np.int64) + seq_len, len(sw))
                wins = (pref[ends] - pref[starts.astype(np.int64)]) / np.maximum(ends - starts, 1)
                weight_parts.append(wins.astype(np.float32))
            else:
                weight_parts.append(np.ones(starts.shape, dtype=np.float32))

        if seg_id_parts:
            self._index_seg = np.concatenate(seg_id_parts)
            self._index_start = np.concatenate(start_parts)
            self.weights = np.concatenate(weight_parts)
        else:
            self._index_seg = np.zeros(0, dtype=np.int32)
            self._index_start = np.zeros(0, dtype=np.int32)
            self.weights = np.zeros(0, dtype=np.float32)

        # Normalize weights
        if self.weights.sum() > 0:
            self.weights = self.weights / self.weights.mean()

    def __len__(self):
        return len(self._index_seg)

    def __getitem__(self, idx):
        seg_idx = int(self._index_seg[idx]); start = int(self._index_start[idx])
        seg = self.segments[seg_idx]

        # Apply temporal jitter during training
        if self.augment and self.augmentor:
            start = self.augmentor.temporal_jitter_index(
                start, len(seg["features"]), self.seq_len
            )

        obs = torch.from_numpy(
            seg["features"][start:start + self.seq_len].copy()
        ).float()
        asset = torch.tensor(seg["asset_idx"], dtype=torch.long)

        targets = {}
        for h in self.reward_horizons:
            key = f"target_return_{h}"
            targets[h] = torch.from_numpy(
                seg[key][start:start + self.seq_len].copy()
            ).float()

        # Include precomputed regime labels if available
        if "regime_label" in seg:
            targets["regime_label"] = torch.from_numpy(
                seg["regime_label"][start:start + self.seq_len].copy()
            ).long()

        # Forward-regime labels (V1_FORWARD_REGIME flag; absent when flag is OFF).
        # Stored as float32 with NaN at the tail (no-future rows); collate_fn packs
        # them into targets["forward_regime_labels"] for get_loss's guarded aux block.
        if "fwd_bear" in seg:
            targets["fwd_bear"]  = torch.from_numpy(
                seg["fwd_bear"][start:start + self.seq_len].copy()).float()
            targets["fwd_trend"] = torch.from_numpy(
                seg["fwd_trend"][start:start + self.seq_len].copy()).float()
            targets["fwd_move"]  = torch.from_numpy(
                seg["fwd_move"][start:start + self.seq_len].copy()).float()

        # Apply augmentation
        if self.augment and self.augmentor:
            obs = self.augmentor.augment_obs(obs)

        return obs, targets, asset

    def get_sampler(self):
        """Get weighted sampler for regime-balanced training."""
        if self.config.regime_balance and np.std(self.weights) > 0.01:
            return torch.utils.data.WeightedRandomSampler(
                weights=self.weights.tolist(),
                num_samples=len(self),
                replacement=True,
            )
        return None


# =============================================================================
# PREDICTION HELPER (used by ShuffledICTracker)
# =============================================================================

def make_predict_fn(seq_len, device, model_type="rssm", batch_size=64, revin=None):
    """
    Create a predict function compatible with ShuffledICTracker.

    Batches multiple sequences for GPU efficiency (~64x faster than batch-1).

    NOTE: RevIN MUST be applied here when the model was trained with RevIN.
    The model expects RevIN-normalized inputs (per-sequence mean=0, std=1 + learned
    affine). Without RevIN, the model receives features at a different scale than
    training, producing noise predictions and ShIC ~= 0 regardless of actual
    cross-sectional signal.

    RevIN does NOT create a confound with shuffling:
    - RevIN normalizes per-sequence to the same learned scale whether contiguous
      or shuffled
    - Shuffled sequences have different raw statistics, but RevIN equalizes them
    - The model needs this normalization to produce meaningful predictions

    Args:
        seq_len: Sequence length
        device: torch device
        model_type: Unused (kept for backward compat). Both RSSM and JEPA models
                    expose the same forward_train() -> {"return_logits": ...} API.
        batch_size: Number of sequences to process in parallel
        revin: RevIN module matching the one used during training. Required for
               correct ShIC measurement when model was trained with RevIN.

    Returns:
        predict_fn(model, feats, asset_idx, horizon) -> np.ndarray
    """
    @torch.no_grad()
    def predict_fn(model, feats, asset_idx, horizon):
        """Run model on data, return predictions array."""
        model.eval()
        n_samples = len(feats)
        predictions = np.full(n_samples, np.nan, dtype=np.float32)

        # Non-overlapping stride to prevent IC inflation
        indices = list(range(0, n_samples - seq_len, seq_len))
        if not indices and n_samples >= seq_len:
            indices = [0]

        # Process in batches for GPU efficiency
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            obs_list = []
            for i in batch_indices:
                obs_list.append(feats[i:i+seq_len])

            obs = torch.from_numpy(np.stack(obs_list)).float().to(device)
            asset = torch.full((len(obs_list),), asset_idx, dtype=torch.long, device=device)

            # Apply RevIN normalization (must match training distribution)
            if revin is not None:
                obs = revin(obs, mode='norm')

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model.forward_train(obs, asset)
                logits = outputs["return_logits"][horizon]
                preds = model.bucketer.decode(logits).cpu().numpy()  # [B, T]

            for b, i in enumerate(batch_indices):
                for j in range(seq_len):
                    timestep = i + j
                    if timestep < n_samples:
                        if np.isnan(predictions[timestep]):
                            predictions[timestep] = preds[b, j]
                        else:
                            predictions[timestep] = (predictions[timestep] + preds[b, j]) / 2.0

        return predictions

    return predict_fn



# =============================================================================
# ANTI-FRAGILE EARLY STOPPING
# =============================================================================

class AntifragileEarlyStopping:
    """
    Early stopping based on SHUFFLED IC (not contiguous val loss).

    Three stopping criteria:
      1. Patience exhausted on shuffled IC
      2. Overfitting detected (IC gap too large)
      3. Standard val_loss patience (backup)
    """

    def __init__(
        self,
        patience_epochs: int = 40,
        val_every: int = 5,
        config: AntifragileConfig = None,
    ):
        self.patience_epochs = patience_epochs
        self.val_every = val_every
        self.config = config or ANTIFRAGILE_DEFAULTS

        self.best_shuffled_ic = -float("inf")
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.overfit_monitor = OverfitMonitor(config)

    def step(
        self,
        epoch: int,
        val_loss: float,
        contiguous_ic: float,
        shuffled_ic: float = None,
    ) -> Tuple[bool, str, bool]:
        """
        Check if training should stop.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            contiguous_ic: IC on contiguous validation data
            shuffled_ic: IC on shuffled data (computed every N epochs)

        Returns:
            (should_stop, reason, is_new_best)
        """
        is_new_best = False

        # Update val_loss tracking
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            is_new_best = True
            self.patience_counter = 0
        else:
            self.patience_counter += self.val_every

        # If shuffled IC is provided, use it as primary metric
        if shuffled_ic is not None:
            if shuffled_ic > self.best_shuffled_ic:
                self.best_shuffled_ic = shuffled_ic
                is_new_best = True
                self.patience_counter = 0

            # Check for overfitting
            should_stop, reason = self.overfit_monitor.check_overfit(
                contiguous_ic, shuffled_ic, epoch
            )
            if should_stop:
                return True, reason, False

        # Check patience
        if self.patience_counter >= self.patience_epochs:
            return True, f"Patience exhausted ({self.patience_epochs} epochs)", False

        return False, "", is_new_best


# =============================================================================
# UTILITY: Collate function (shared by all versions)
# =============================================================================

def collate_fn_generic(batch, reward_horizons):
    """Generic collate function for dict targets."""
    obs = torch.stack([b[0] for b in batch])
    asset = torch.stack([b[2] for b in batch])
    targets = {}
    for h in reward_horizons:
        targets[h] = torch.stack([b[1][h] for b in batch])
    return obs, targets, asset


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_antifragile_header(version: str, config: AntifragileConfig):
    """Print anti-fragile training configuration."""
    print(f"\n  Anti-Fragile Training Enabled:")
    print(f"    Walk-Forward Folds:    {config.n_walk_forward_folds}")
    print(f"    Purge Gap:             {config.purge_gap_bars} bars (>= normalization window)")
    print(f"    Shuffled IC every:     {config.shuffled_ic_every} epochs")
    print(f"    Shuffled IC seeds:     {config.shuffled_ic_seeds}")
    print(f"    Shuffled IC folds:     {config.shuffled_ic_folds} per seed")
    print(f"    Temporal Jitter:       +/-{config.aug_temporal_jitter} bars")
    print(f"    Mixup Alpha:           {config.aug_mixup_alpha}")
    print(f"    Time Reverse Prob:     {config.aug_time_reverse_prob}")
    print(f"    Block Swap Prob:       {config.aug_block_swap_prob}")
    print(f"    Regime Balanced:       {config.regime_balance}")
    print(f"    Overfit Gap Warn:      {config.overfit_ic_gap_warn}")
    print(f"    Overfit Gap Stop:      {config.overfit_ic_gap_stop}")
    # NOTE: label_smoothing disabled (0.0) -- accelerates memorization. See CLAUDE.md.
    print(f"    IC Threshold (raw):    {IC_THRESHOLD_UNADJUSTED}")
    print(f"    IC Threshold (Bonf.):  {IC_THRESHOLD_BONFERRONI:.5f} (for {N_MODELS}-model selection)")
