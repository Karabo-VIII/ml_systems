"""
V10 Meta-Ensemble Router Trainer

Training strategy:
  1. Load validation data (same format as V1-V9 trainers)
  2. Pre-compute predictions from all frozen V1-V9 models (one-time cost)
  3. Cache predictions per sample in a simple dataset
  4. Train the tiny router (~10K params) on cached predictions
  5. Compare ensemble IC to individual model ICs

The key insight: base model inference is the expensive step (~minutes per model).
Once predictions are cached, router training is extremely fast because the router
is only ~10K parameters operating on pre-computed tensors.

VRAM management: only one base model is loaded at a time during caching.
Router training uses negligible GPU memory.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import gc
import time
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = Path(__file__).resolve().parent      # src/wm/v10/v10_meta/
_GROUP_DIR = _THIS_DIR.parent                     # src/wm/v10/
_WM_DIR = _GROUP_DIR.parent                       # src/wm/   (post-2026-04-29 layout)
_SRC_DIR = _WM_DIR.parent                         # src/
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_GROUP_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_DIR))
if str(_WM_DIR) not in sys.path:
    sys.path.insert(0, str(_WM_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# V1 components (TwoHotSymlog) needed regardless of which models are enabled
_V1_GROUP = str(_WM_DIR / "v1")
if _V1_GROUP not in sys.path:
    sys.path.insert(0, _V1_GROUP)

from v10_meta.settings import (
    DEVICE, PROJECT_ROOT, DATA_DIR, MODEL_DIR, LOG_DIR,
    FEATURE_LIST, ASSET_LIST, ASSET_TO_IDX, NUM_ASSETS,
    REWARD_HORIZONS, INPUT_DIM, NUM_BINS, BIN_MIN, BIN_MAX,
    META_N_MODELS, META_CONTEXT_DIM, META_ROUTER_HIDDEN, META_TEMPERATURE,
    META_PER_HORIZON_ROUTING, META_MODEL_ENABLED,
    META_LR, META_WEIGHT_DECAY, META_EPOCHS, META_STEPS_PER_EPOCH,
    META_BATCH_SIZE, META_VAL_EVERY, META_GRAD_CLIP,
    WM_SEQ_LEN, META_CACHE_PREDICTIONS, GATE_IC_MIN,
    IS_WINDOWS, NUM_WORKERS,
    get_lr_for_epoch,
)
from v10_meta.meta_ensemble import (
    MetaRouter, MetaEnsemble, MetaContextComputer, count_parameters,
)
from anti_fragile import (
    WalkForwardSplitter, AntifragileConfig, AntifragileDataset,
    compute_regime_weights,
)
# Read-side contract (Phase D.10): load_full_data via data_api.
from data_api import load_full_data_for_training as load_full_data
from log_utils import setup_logging, teardown_logging


# ==============================================================================
# CACHED PREDICTIONS DATASET
# ==============================================================================

class CachedPredictionDataset(Dataset):
    """
    Dataset of pre-computed model predictions + targets.

    Each sample contains:
      - predictions from each enabled model (per horizon)
      - ground truth targets (per horizon)
      - context vector for the router
      - asset index

    This avoids running frozen models during router training iterations.
    """

    def __init__(
        self,
        cached_preds: dict,
        targets: dict,
        asset_indices: np.ndarray,
        context_vectors: np.ndarray,
        enabled_models: list,
    ):
        """
        Args:
            cached_preds: {version: {horizon: np.ndarray [N]}} model predictions
            targets: {horizon: np.ndarray [N]} ground truth
            asset_indices: np.ndarray [N] asset index per sample
            context_vectors: np.ndarray [N, META_CONTEXT_DIM]
            enabled_models: list of version ints
        """
        self.cached_preds = cached_preds
        self.targets = targets
        self.asset_indices = asset_indices
        self.context_vectors = context_vectors
        self.enabled_models = enabled_models
        self.n_samples = len(asset_indices)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Stack predictions from all models for each horizon
        # Shape per horizon: [n_models]
        preds = {}
        for h in REWARD_HORIZONS:
            model_preds = []
            for v in self.enabled_models:
                model_preds.append(self.cached_preds[v][h][idx])
            preds[h] = torch.tensor(model_preds, dtype=torch.float32)

        tgt = {}
        for h in REWARD_HORIZONS:
            tgt[h] = torch.tensor(self.targets[h][idx], dtype=torch.float32)

        ctx = torch.from_numpy(self.context_vectors[idx]).float()
        asset = torch.tensor(self.asset_indices[idx], dtype=torch.long)

        return preds, tgt, ctx, asset


def collate_cached(batch):
    """Collate function for CachedPredictionDataset."""
    preds = {h: torch.stack([b[0][h] for b in batch]) for h in REWARD_HORIZONS}
    targets = {h: torch.stack([b[1][h] for b in batch]) for h in REWARD_HORIZONS}
    context = torch.stack([b[2] for b in batch])
    asset = torch.stack([b[3] for b in batch])
    return preds, targets, context, asset


# ==============================================================================
# PREDICTION CACHING
# ==============================================================================

@torch.no_grad()
def cache_predictions_for_model(
    version: int,
    segments: list,
    model_paths: dict,
    seq_len: int = WM_SEQ_LEN,
    batch_size: int = 32,
) -> dict:
    """
    Run a single frozen model on all segments and cache decoded return predictions.

    Returns:
        {horizon: np.ndarray [total_timesteps]} of scalar predictions
    """
    from v10_meta.meta_ensemble import MetaEnsemble

    # Load model
    ensemble_helper = MetaEnsemble.__new__(MetaEnsemble)
    ensemble_helper.model_paths = model_paths
    ensemble_helper.enabled_models = sorted(model_paths.keys())

    # Add version's group and training dirs to sys.path
    v_group_dir = str(_SRC_DIR / f"v{version}")
    v_train_dir = str(Path(v_group_dir) / f"v{version}_training")
    if v_group_dir not in sys.path:
        sys.path.insert(0, v_group_dir)
    if v_train_dir not in sys.path:
        sys.path.insert(0, v_train_dir)

    from v10_meta.meta_ensemble import _MODEL_REGISTRY
    import importlib

    # Clear cached bare imports to prevent cross-contamination between versions
    for cached_key in ["components", "settings"]:
        sys.modules.pop(cached_key, None)

    module_path, class_name = _MODEL_REGISTRY[version]
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    model = ModelClass()
    path = model_paths[version]
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)

    # Support both formats:
    #   - new: {"model_state_dict": ..., "revin_state_dict": ...}
    #   - legacy (pre-fix): raw state_dict
    revin = None
    # strict=False per CLAUDE.md schema-compat invariant: meta-ensemble loads
    # ckpts from multiple architectures (V1.0/V1.1/V1.4/...) which may drift
    # in keys (e.g. cap-scaling rollback). Warn on non-trivial mismatches so
    # silent partial-loads don't escape detection.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        _m, _u = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if len(_m) > 5 or len(_u) > 0:
            print(f"  [meta load WARN] {path.name}: missing={len(_m)} unexpected={len(_u)}")
        if "revin_state_dict" in ckpt:
            from revin import RevIN
            revin = RevIN(num_features=INPUT_DIM).to(DEVICE)
            revin.load_state_dict(ckpt["revin_state_dict"], strict=False)
            revin.eval()
    else:
        # Legacy flat state_dict -- RevIN params not saved, use identity
        _m, _u = model.load_state_dict(ckpt, strict=False)
        if len(_m) > 5 or len(_u) > 0:
            print(f"  [meta load WARN legacy] {path.name}: missing={len(_m)} unexpected={len(_u)}")

    model.to(DEVICE)
    model.eval()

    from v1_0_training.components import TwoHotSymlog
    bucketer = TwoHotSymlog(NUM_BINS, BIN_MIN, BIN_MAX, DEVICE)

    # Collect predictions across all segments
    all_preds = {h: [] for h in REWARD_HORIZONS}
    total_samples = 0

    for seg in segments:
        feats = seg["features"]
        asset_idx = seg["asset_idx"]
        n = len(feats)

        if n < seq_len:
            continue

        # Process non-overlapping windows for clean predictions
        # Use stride = seq_len to avoid double-counting
        window_starts = list(range(0, n - seq_len + 1, seq_len))

        for batch_start in range(0, len(window_starts), batch_size):
            batch_indices = window_starts[batch_start:batch_start + batch_size]
            obs_list = []
            for i in batch_indices:
                obs_list.append(feats[i:i + seq_len])

            obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
            asset = torch.full(
                (len(obs_list),), asset_idx, dtype=torch.long, device=DEVICE
            )

            # Apply RevIN normalization (same as training-time preprocessing)
            if revin is not None:
                with torch.no_grad():
                    obs = revin(obs, mode='norm')

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                outputs = model.forward_train(obs, asset)

            for h in REWARD_HORIZONS:
                logits = outputs["return_logits"][h]
                decoded = model.bucketer.decode(logits)  # [B, T]
                # Take the last timestep prediction from each window
                # (most informed prediction, uses full context)
                last_preds = decoded[:, -1].cpu().numpy()
                all_preds[h].append(last_preds)

            total_samples += len(batch_indices)

    # Free model and RevIN from VRAM
    del model
    if revin is not None:
        del revin
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Concatenate
    result = {}
    for h in REWARD_HORIZONS:
        if all_preds[h]:
            result[h] = np.concatenate(all_preds[h])
        else:
            result[h] = np.array([], dtype=np.float32)

    return result


def cache_all_predictions(
    segments: list,
    model_paths: dict,
    enabled_models: list,
) -> tuple:
    """
    Cache predictions from all enabled models.

    Returns:
        (cached_preds, targets, asset_indices, n_samples)
        cached_preds: {version: {horizon: np.ndarray [N]}}
        targets: {horizon: np.ndarray [N]}
        asset_indices: np.ndarray [N]
    """
    print("\n  [CACHE] Pre-computing predictions from frozen models...")
    print(f"  Models to cache: V{', V'.join(str(v) for v in enabled_models)}")
    print("-" * 60)

    t0 = time.time()
    cached_preds = {}
    n_samples = None

    for v in enabled_models:
        vt = time.time()
        print(f"  [CACHE] Running V{v}...", end="", flush=True)
        preds = cache_predictions_for_model(
            v, segments, {v: model_paths[v]},
        )
        cached_preds[v] = preds

        # Verify consistent sample counts
        n = len(preds[REWARD_HORIZONS[0]])
        if n_samples is None:
            n_samples = n
        elif n != n_samples:
            # Truncate to minimum
            n_samples = min(n_samples, n)
            print(f" [WARN] Sample count mismatch (got {n}, expected {n_samples})", end="")

        elapsed = time.time() - vt
        print(f" {n:,} samples in {elapsed:.1f}s")

    # Truncate all to same length
    for v in enabled_models:
        for h in REWARD_HORIZONS:
            cached_preds[v][h] = cached_preds[v][h][:n_samples]

    # Extract aligned targets and asset indices
    targets = {h: [] for h in REWARD_HORIZONS}
    asset_indices = []

    for seg in segments:
        feats = seg["features"]
        asset_idx = seg["asset_idx"]
        n = len(feats)

        if n < WM_SEQ_LEN:
            continue

        window_starts = list(range(0, n - WM_SEQ_LEN + 1, WM_SEQ_LEN))
        for i in window_starts:
            # Target at the last timestep of each window (aligned with prediction)
            for h in REWARD_HORIZONS:
                key = f"target_return_{h}"
                last_idx = i + WM_SEQ_LEN - 1
                if last_idx < len(seg[key]):
                    targets[h].append(seg[key][last_idx])
                else:
                    targets[h].append(0.0)
            asset_indices.append(asset_idx)

    for h in REWARD_HORIZONS:
        targets[h] = np.array(targets[h][:n_samples], dtype=np.float32)
    asset_indices = np.array(asset_indices[:n_samples], dtype=np.int64)

    total_time = time.time() - t0
    print(f"\n  [OK] Cached {n_samples:,} samples from {len(enabled_models)} models "
          f"in {total_time:.1f}s")

    return cached_preds, targets, asset_indices, n_samples


# ==============================================================================
# CONTEXT VECTOR COMPUTATION
# ==============================================================================

def compute_context_vectors(
    cached_preds: dict,
    targets: dict,
    asset_indices: np.ndarray,
    enabled_models: list,
    n_samples: int,
    window_size: int = 500,
) -> np.ndarray:
    """
    Compute context vectors for each sample based on rolling IC.

    For each sample, we compute rolling IC from a trailing window of predictions
    vs actuals. This gives the router information about how well each model
    has been performing recently.

    Args:
        cached_preds: {version: {horizon: [N]}} predictions
        targets: {horizon: [N]} actuals
        asset_indices: [N]
        enabled_models: list of version ints
        n_samples: number of samples
        window_size: lookback for rolling IC

    Returns:
        np.ndarray [N, META_CONTEXT_DIM]
    """
    print("  [CTX] Computing context vectors (rolling IC, regime, volatility)...")

    context_vectors = np.zeros((n_samples, META_CONTEXT_DIM), dtype=np.float32)

    # Pre-compute rolling ICs per model per horizon
    for v_idx, v in enumerate(range(1, 10)):
        for h_idx, h in enumerate(REWARD_HORIZONS):
            if v not in cached_preds:
                continue

            preds_v = cached_preds[v][h]
            actuals = targets[h]

            # Compute rolling rank correlation (IC)
            for i in range(n_samples):
                start = max(0, i - window_size)
                if i - start < 30:
                    # Not enough history, use zero
                    continue

                p = preds_v[start:i]
                a = actuals[start:i]

                mask = np.isfinite(p) & np.isfinite(a)
                p_valid = p[mask]
                a_valid = a[mask]

                if len(p_valid) > 30 and np.std(p_valid) > 1e-10 and np.std(a_valid) > 1e-10:
                    ic = float(np.corrcoef(p_valid, a_valid)[0, 1])
                    if np.isfinite(ic):
                        context_vectors[i, v_idx * 4 + h_idx] = np.clip(ic * 10.0, -1.0, 1.0)

    # Regime from target returns
    ret_1 = targets[1]
    ret_std = np.std(ret_1) + 1e-8
    for i in range(n_samples):
        start = max(0, i - window_size)
        window = ret_1[start:i]  # exclude current bar (look-ahead fix)
        if len(window) < 10:
            continue

        # Simple regime: fraction of positive/negative returns
        n_pos = np.sum(window > ret_std * 0.5)
        n_neg = np.sum(window < -ret_std * 0.5)
        n_total = len(window)

        context_vectors[i, 36] = (n_neg / n_total) * 2.0 - 1.0  # bear
        context_vectors[i, 37] = ((n_total - n_pos - n_neg) / n_total) * 2.0 - 1.0  # neutral
        context_vectors[i, 38] = (n_pos / n_total) * 2.0 - 1.0  # bull

        # Volatility
        context_vectors[i, 39] = np.clip(np.std(window) * 10.0, -1.0, 1.0)

    # Mean IC per horizon
    for h_idx, h in enumerate(REWARD_HORIZONS):
        for i in range(n_samples):
            ics = []
            for v in enabled_models:
                v_idx_local = v - 1  # v is 1-indexed
                ic_val = context_vectors[i, v_idx_local * 4 + h_idx]
                if abs(ic_val) > 1e-10:
                    ics.append(ic_val)
            if ics:
                context_vectors[i, 40 + h_idx] = np.clip(np.mean(ics), -1.0, 1.0)

    # IC variance for horizons 1 and 4
    for pair_idx, h in enumerate(REWARD_HORIZONS[:2]):
        h_idx = pair_idx
        for i in range(n_samples):
            ics = []
            for v in enabled_models:
                v_idx_local = v - 1
                ic_val = context_vectors[i, v_idx_local * 4 + h_idx]
                if abs(ic_val) > 1e-10:
                    ics.append(ic_val)
            if len(ics) > 1:
                context_vectors[i, 44 + pair_idx] = np.clip(np.var(ics) * 100.0, 0.0, 1.0)

    print(f"  [OK] Context vectors computed: shape={context_vectors.shape}")
    return context_vectors


# ==============================================================================
# VALIDATION
# ==============================================================================

def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute rank correlation (Information Coefficient)."""
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    p = predictions[mask]
    a = actuals[mask]
    if len(p) < 30 or np.std(p) < 1e-10 or np.std(a) < 1e-10:
        return 0.0
    ic = float(np.corrcoef(p, a)[0, 1])
    return ic if np.isfinite(ic) else 0.0


@torch.no_grad()
def validate_ensemble(
    router: MetaRouter,
    val_dataset: CachedPredictionDataset,
    enabled_models: list,
    device: str = DEVICE,
) -> dict:
    """
    Validate the meta-ensemble on cached validation predictions.

    Computes:
      - Ensemble IC per horizon
      - Individual model IC per horizon (for comparison)
      - Ensemble loss (MSE between weighted prediction and target)
      - Average routing weights

    Returns:
        dict with metrics
    """
    router.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=META_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cached,
        drop_last=False,
    )

    # Collect ensemble and individual predictions
    ensemble_preds = {h: [] for h in REWARD_HORIZONS}
    individual_preds = {v: {h: [] for h in REWARD_HORIZONS} for v in enabled_models}
    all_targets = {h: [] for h in REWARD_HORIZONS}
    all_weights = {h: [] for h in REWARD_HORIZONS}
    losses = []

    for preds, targets, context, asset in val_loader:
        context = context.to(device)

        # Get routing weights
        weights = router(context)  # {horizon: [B, n_models]}

        for h in REWARD_HORIZONS:
            w = weights[h]  # [B, n_models]
            p = preds[h].to(device)  # [B, n_models]
            t = targets[h].to(device)  # [B]

            # Weighted ensemble prediction
            ens_pred = (p * w).sum(dim=-1)  # [B]
            ensemble_preds[h].append(ens_pred.cpu().numpy())
            all_targets[h].append(t.cpu().numpy())
            all_weights[h].append(w.cpu().numpy())

            # Individual model predictions
            for m_idx, v in enumerate(enabled_models):
                individual_preds[v][h].append(p[:, m_idx].cpu().numpy())

            # MSE loss
            loss = F.mse_loss(ens_pred, t)
            losses.append(loss.item())

    # Compute metrics
    result = {"loss": float(np.mean(losses))}

    # Ensemble IC per horizon
    for h in REWARD_HORIZONS:
        ens_p = np.concatenate(ensemble_preds[h])
        tgt = np.concatenate(all_targets[h])
        result[f"ensemble_ic_{h}"] = compute_ic(ens_p, tgt)

    # Mean ensemble IC
    result["ensemble_ic"] = float(np.mean([
        result[f"ensemble_ic_{h}"] for h in REWARD_HORIZONS
    ]))

    # Individual model IC per horizon
    for v in enabled_models:
        for h in REWARD_HORIZONS:
            ind_p = np.concatenate(individual_preds[v][h])
            tgt = np.concatenate(all_targets[h])
            result[f"v{v}_ic_{h}"] = compute_ic(ind_p, tgt)
        result[f"v{v}_ic"] = float(np.mean([
            result[f"v{v}_ic_{h}"] for h in REWARD_HORIZONS
        ]))

    # Average routing weights
    for h in REWARD_HORIZONS:
        avg_w = np.concatenate(all_weights[h]).mean(axis=0)  # [n_models]
        for m_idx, v in enumerate(enabled_models):
            result[f"weight_v{v}_h{h}"] = float(avg_w[m_idx])

    router.train()
    return result


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train_meta_ensemble():
    """
    Train the V10 Meta-Ensemble router.

    Steps:
      1. Load data
      2. Walk-forward split
      3. Cache predictions from all available frozen models
      4. Build context vectors
      5. Train the router on cached predictions
      6. Validate and compare ensemble vs individual model ICs
    """
    log_path = setup_logging(LOG_DIR, "v10_train")
    torch.set_float32_matmul_precision("medium")

    print("=" * 70)
    print("  V10 META-ENSEMBLE ROUTER TRAINER")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Platform:     {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Strategy:     Pre-compute frozen model predictions, train tiny router")
    print(f"  Router:       MLP hidden={META_ROUTER_HIDDEN}, per_horizon={META_PER_HORIZON_ROUTING}")
    print(f"  Context dim:  {META_CONTEXT_DIM}")
    print(f"  Temperature:  {META_TEMPERATURE}")
    print(f"  LR:           {META_LR}")
    print(f"  Epochs:       {META_EPOCHS}")
    print(f"  Batch size:   {META_BATCH_SIZE}")

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print(f"\n  Loading data from {DATA_DIR}")
    print("-" * 60)
    all_segments = load_full_data(
        DATA_DIR, FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS
    )
    if all_segments is None:
        print("[FAIL] No valid data. Exiting.")
        teardown_logging()
        return False

    # =========================================================================
    # STEP 2: Walk-Forward Split
    # =========================================================================
    af_config = AntifragileConfig()
    splitter = WalkForwardSplitter(af_config)
    train_segments, val_segments, oos_segments, unseen_segments = \
        splitter.split_four_way(all_segments)

    if not train_segments or not val_segments:
        print("[FAIL] Walk-forward split produced empty sets. Exiting.")
        teardown_logging()
        return False

    n_train = sum(len(s["features"]) for s in train_segments)
    n_val = sum(len(s["features"]) for s in val_segments)
    n_oos = sum(len(s["features"]) for s in oos_segments)
    n_unseen = sum(len(s["features"]) for s in unseen_segments)
    n_total = n_train + n_val + n_oos + n_unseen
    print(f"\n  Data Split:")
    print(f"    Train:  {n_train:>12,} bars ({n_train/n_total*100:.1f}%)")
    print(f"    Val:    {n_val:>12,} bars ({n_val/n_total*100:.1f}%)")
    print(f"    OOS:    {n_oos:>12,} bars ({n_oos/n_total*100:.1f}%)")
    print(f"    Unseen: {n_unseen:>12,} bars ({n_unseen/n_total*100:.1f}%) [held out]")

    # =========================================================================
    # STEP 3: Discover and Cache Predictions
    # =========================================================================
    # Find available model checkpoints
    model_paths = {}
    enabled_models = []
    for v in range(1, 10):
        if not META_MODEL_ENABLED.get(v, True):
            continue
        model_dir = PROJECT_ROOT / "models" / f"v{v}" / f"v{v}" / "base"
        for name in [f"v{v}_wm_best_ema.pt", f"v{v}_wm_best.pt", f"v{v}_wm_weights.pt"]:
            path = model_dir / name
            if path.exists():
                model_paths[v] = path
                enabled_models.append(v)
                break

    if len(enabled_models) < 2:
        print(f"\n[FAIL] Need at least 2 model checkpoints for ensemble. "
              f"Found {len(enabled_models)}: V{', V'.join(str(v) for v in enabled_models)}")
        print("  Train more base models (V1-V9) first.")
        teardown_logging()
        return False

    print(f"\n  Found {len(enabled_models)} model checkpoints: "
          f"V{', V'.join(str(v) for v in enabled_models)}")

    # Cache predictions on TRAINING data
    print("\n  --- Caching TRAIN predictions ---")
    train_preds, train_targets, train_assets, n_train_samples = cache_all_predictions(
        train_segments, model_paths, enabled_models,
    )

    # Cache predictions on VALIDATION data
    print("\n  --- Caching VAL predictions ---")
    val_preds, val_targets, val_assets, n_val_samples = cache_all_predictions(
        val_segments, model_paths, enabled_models,
    )

    if n_train_samples < 100 or n_val_samples < 50:
        print(f"[FAIL] Too few samples: train={n_train_samples}, val={n_val_samples}")
        teardown_logging()
        return False

    # =========================================================================
    # STEP 4: Compute Context Vectors
    # =========================================================================
    train_context = compute_context_vectors(
        train_preds, train_targets, train_assets, enabled_models, n_train_samples,
    )
    val_context = compute_context_vectors(
        val_preds, val_targets, val_assets, enabled_models, n_val_samples,
    )

    # =========================================================================
    # STEP 5: Build Datasets
    # =========================================================================
    train_ds = CachedPredictionDataset(
        train_preds, train_targets, train_assets, train_context, enabled_models,
    )
    val_ds = CachedPredictionDataset(
        val_preds, val_targets, val_assets, val_context, enabled_models,
    )

    print(f"\n  Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=META_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cached,
        drop_last=True,
    )

    # =========================================================================
    # STEP 6: Initialize Router
    # =========================================================================
    n_active = len(enabled_models)
    router = MetaRouter(
        context_dim=META_CONTEXT_DIM,
        n_models=n_active,
        hidden_dim=META_ROUTER_HIDDEN,
        per_horizon=META_PER_HORIZON_ROUTING,
        temperature=META_TEMPERATURE,
    ).to(DEVICE)

    print(f"\n  MetaRouter parameters: {count_parameters(router):,}")
    print(f"  Active models: {n_active}")

    optimizer = optim.AdamW(
        router.parameters(),
        lr=META_LR,
        weight_decay=META_WEIGHT_DECAY,
    )

    # =========================================================================
    # STEP 7: Initial Baseline (uniform weights / individual models)
    # =========================================================================
    print("\n  Computing baseline metrics (before training)...")
    baseline_metrics = validate_ensemble(router, val_ds, enabled_models)

    print(f"\n  Baseline Ensemble IC: {baseline_metrics['ensemble_ic']:.4f}")
    for h in REWARD_HORIZONS:
        print(f"    Horizon {h:2d}: ensemble={baseline_metrics[f'ensemble_ic_{h}']:.4f}", end="")
        for v in enabled_models:
            print(f" | V{v}={baseline_metrics[f'v{v}_ic_{h}']:.4f}", end="")
        print()

    # Best individual model IC
    best_individual_ic = max(baseline_metrics[f"v{v}_ic"] for v in enabled_models)
    best_individual_v = max(enabled_models, key=lambda v: baseline_metrics[f"v{v}_ic"])
    print(f"\n  Best individual model: V{best_individual_v} "
          f"(IC={best_individual_ic:.4f})")

    # =========================================================================
    # STEP 8: Training Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TRAINING ROUTER")
    print("=" * 70)

    best_val_ic = -float("inf")
    best_val_loss = float("inf")

    for epoch in range(META_EPOCHS):
        router.train()

        # Set LR for this epoch
        current_lr = get_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        epoch_losses = []
        epoch_ics = {h: [] for h in REWARD_HORIZONS}
        train_iter = iter(train_loader)

        steps_this_epoch = min(META_STEPS_PER_EPOCH, len(train_loader))
        pbar = tqdm(range(steps_this_epoch), desc=f"Epoch {epoch+1:3d}", leave=False)

        for step in pbar:
            try:
                preds, targets, context, asset = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                preds, targets, context, asset = next(train_iter)

            context = context.to(DEVICE)

            # Forward: get routing weights
            weights = router(context)  # {horizon: [B, n_models]}

            # Compute weighted ensemble loss
            total_loss = torch.tensor(0.0, device=DEVICE)
            for h in REWARD_HORIZONS:
                w = weights[h]  # [B, n_models]
                p = preds[h].to(DEVICE)  # [B, n_models]
                t = targets[h].to(DEVICE)  # [B]

                # Weighted prediction
                ens_pred = (p * w).sum(dim=-1)  # [B]

                # Huber loss (robust to outliers in crypto returns)
                h_loss = F.huber_loss(ens_pred, t, delta=0.1)
                total_loss = total_loss + h_loss

                # Track IC for logging
                with torch.no_grad():
                    ep = ens_pred.cpu().numpy()
                    et = t.cpu().numpy()
                    if np.std(ep) > 1e-10 and np.std(et) > 1e-10:
                        ic = float(np.corrcoef(ep, et)[0, 1])
                        if np.isfinite(ic):
                            epoch_ics[h].append(ic)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), META_GRAD_CLIP)
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if step % 50 == 0:
                avg_loss = np.mean(epoch_losses[-50:]) if epoch_losses else 0
                pbar.set_postfix(L=f"{avg_loss:.4f}", LR=f"{current_lr:.1e}")

        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        ic_str = " | ".join([
            f"IC{h}:{np.mean(epoch_ics[h]):.4f}" if epoch_ics[h] else f"IC{h}:---"
            for h in REWARD_HORIZONS
        ])
        print(f"  Ep {epoch+1:3d} | Loss: {avg_loss:.4f} | {ic_str} | LR: {current_lr:.1e}")

        # Validation
        if (epoch + 1) % META_VAL_EVERY == 0 or epoch == META_EPOCHS - 1:
            val_metrics = validate_ensemble(router, val_ds, enabled_models)
            val_ic = val_metrics["ensemble_ic"]
            val_loss = val_metrics["loss"]

            save_marker = ""
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_val_loss = val_loss
                # Save best router
                torch.save(router.state_dict(), MODEL_DIR / "v10_router_best.pt")
                save_marker = " *BEST*"

            # Print per-horizon comparison
            print(f"  -- VAL | Loss: {val_loss:.4f} | Ensemble IC: {val_ic:.4f}{save_marker}")
            for h in REWARD_HORIZONS:
                ens_ic = val_metrics[f"ensemble_ic_{h}"]
                ind_strs = []
                for v in enabled_models:
                    v_ic = val_metrics[f"v{v}_ic_{h}"]
                    ind_strs.append(f"V{v}={v_ic:.4f}")
                print(f"       H{h:2d}: ens={ens_ic:.4f} | {' '.join(ind_strs)}")

            # Print average routing weights
            weight_strs = []
            for v in enabled_models:
                avg_w = np.mean([val_metrics[f"weight_v{v}_h{h}"] for h in REWARD_HORIZONS])
                weight_strs.append(f"V{v}={avg_w:.2f}")
            print(f"       Avg weights: {' '.join(weight_strs)}")

            # Save latest
            state = {
                "epoch": epoch,
                "router_state_dict": router.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "enabled_models": enabled_models,
                "best_val_ic": best_val_ic,
                "best_val_loss": best_val_loss,
                "n_active_models": n_active,
                "version": "v10_meta_ensemble",
            }
            torch.save(state, MODEL_DIR / "v10_router_latest.pt")

    # =========================================================================
    # STEP 9: Final Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    # Load best router for final eval
    best_state = torch.load(MODEL_DIR / "v10_router_best.pt", map_location=DEVICE, weights_only=True)
    router.load_state_dict(best_state)
    final_metrics = validate_ensemble(router, val_ds, enabled_models)

    print(f"\n  Final Ensemble IC: {final_metrics['ensemble_ic']:.4f}")
    print(f"  Best Individual IC: {best_individual_ic:.4f} (V{best_individual_v})")

    improvement = final_metrics["ensemble_ic"] - best_individual_ic
    improvement_pct = (improvement / max(abs(best_individual_ic), 1e-6)) * 100
    print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # Per-horizon breakdown
    print(f"\n  Per-Horizon IC Comparison:")
    print(f"  {'Horizon':>8s} | {'Ensemble':>10s} | {'Best Indiv':>10s} | {'Delta':>8s}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8}")
    for h in REWARD_HORIZONS:
        ens_ic = final_metrics[f"ensemble_ic_{h}"]
        best_ind_h = max(final_metrics[f"v{v}_ic_{h}"] for v in enabled_models)
        delta = ens_ic - best_ind_h
        print(f"  {h:8d} | {ens_ic:10.4f} | {best_ind_h:10.4f} | {delta:+8.4f}")

    # Routing weight summary
    print(f"\n  Final Routing Weights (avg across horizons):")
    for v in enabled_models:
        per_h = []
        for h in REWARD_HORIZONS:
            per_h.append(final_metrics[f"weight_v{v}_h{h}"])
        avg_w = np.mean(per_h)
        h_str = " ".join([f"H{h}={w:.2f}" for h, w in zip(REWARD_HORIZONS, per_h)])
        print(f"    V{v}: avg={avg_w:.3f} | {h_str}")

    # Gate check
    gate_passed = final_metrics["ensemble_ic"] >= GATE_IC_MIN
    if gate_passed:
        print(f"\n  [OK] Ensemble IC {final_metrics['ensemble_ic']:.4f} >= {GATE_IC_MIN} -- GATE PASSED")
    else:
        print(f"\n  [FAIL] Ensemble IC {final_metrics['ensemble_ic']:.4f} < {GATE_IC_MIN} -- GATE FAILED")
        print("  Consider training more base models or adjusting router hyperparameters.")

    # Diversity analysis
    if len(enabled_models) >= 2:
        print(f"\n  Model Correlation Matrix (horizon 1 predictions):")
        corr_matrix = np.zeros((len(enabled_models), len(enabled_models)))
        for i, vi in enumerate(enabled_models):
            for j, vj in enumerate(enabled_models):
                pi = val_preds[vi][1]
                pj = val_preds[vj][1]
                mask = np.isfinite(pi) & np.isfinite(pj)
                if mask.sum() > 30:
                    corr = float(np.corrcoef(pi[mask], pj[mask])[0, 1])
                    corr_matrix[i, j] = corr if np.isfinite(corr) else 0.0

        # Print correlation matrix
        header = "       " + "  ".join([f"V{v:d}" for v in enabled_models])
        print(f"  {header}")
        for i, vi in enumerate(enabled_models):
            row_str = "  ".join([f"{corr_matrix[i,j]:5.2f}" for j in range(len(enabled_models))])
            print(f"    V{vi}: {row_str}")

        avg_corr = np.mean([
            corr_matrix[i, j]
            for i in range(len(enabled_models))
            for j in range(i+1, len(enabled_models))
        ]) if len(enabled_models) > 1 else 0.0
        print(f"  Average pairwise correlation: {avg_corr:.3f}")

        # Theoretical IC boost: IC_ens = IC_avg * sqrt(K / (1 + (K-1)*rho))
        avg_ind_ic = np.mean([final_metrics[f"v{v}_ic"] for v in enabled_models])
        K = len(enabled_models)
        rho = max(avg_corr, 0.01)
        theoretical_boost = avg_ind_ic * np.sqrt(K / (1 + (K - 1) * rho))
        print(f"  Theoretical IC (Markowitz): {theoretical_boost:.4f} "
              f"(K={K}, rho={rho:.2f}, IC_avg={avg_ind_ic:.4f})")
        print(f"  Actual ensemble IC: {final_metrics['ensemble_ic']:.4f}")

    print(f"\n  Router weights:  {MODEL_DIR / 'v10_router_best.pt'}")
    print(f"  Latest state:    {MODEL_DIR / 'v10_router_latest.pt'}")

    teardown_logging()
    return gate_passed


def main_cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="V10 meta-ensemble router trainer. Trains a tiny MLP "
                    "that routes between frozen V1-V14 model predictions."
    )
    parser.add_argument("--features", type=int, default=34,
                        help="Pseudo-feature count for run_all_training compat. "
                             "V10 reads frozen predictions from V1-V14 checkpoints, "
                             "so the actual feature schema is determined by those.")
    args = parser.parse_args()
    success = train_meta_ensemble()
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(2)


if __name__ == "__main__":
    main_cli()
