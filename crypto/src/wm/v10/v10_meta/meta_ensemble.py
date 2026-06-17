"""
V10 Meta-Ensemble -- Dynamic Routing Across Frozen V1-V9 Models

Architecture:
  1. Run each frozen V{N} model on input -> get return predictions
  2. Compute context vector from rolling IC, regime, volatility
  3. Router MLP produces per-model weights (optionally per-horizon)
  4. Final prediction = weighted average of model predictions

The router is ~10K params. All base models are frozen.
VRAM management: models loaded/unloaded as needed (8GB constraint).

IC_ensemble = IC_single * sqrt(K / (1 + (K-1)*rho))
With 9 diverse architectures, rho should be low -> large IC boost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib
from pathlib import Path
from collections import deque

import sys

# Ensure this module's dir and project src/ are on the path
_THIS_DIR = Path(__file__).resolve().parent      # src/wm/v10/v10_meta/
_GROUP_DIR = _THIS_DIR.parent                     # src/wm/v10/
_SRC_DIR = _GROUP_DIR.parent                      # src/
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_GROUP_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# V1 components (TwoHotSymlog) needed regardless of which models are enabled
_V1_GROUP = str(_SRC_DIR / "v1")
if _V1_GROUP not in sys.path:
    sys.path.insert(0, _V1_GROUP)

from v10_meta.settings import (
    DEVICE, PROJECT_ROOT, REWARD_HORIZONS, NUM_BINS, BIN_MIN, BIN_MAX,
    META_N_MODELS, META_CONTEXT_DIM, META_ROUTER_HIDDEN, META_TEMPERATURE,
    META_PER_HORIZON_ROUTING, META_MODEL_ENABLED, INPUT_DIM,
)


# =========================================================================
# MODEL CLASS REGISTRY
# Maps version int -> (package_name, class_name) for dynamic import
# =========================================================================
_MODEL_REGISTRY = {
    1: ("v1_0_training.world_model", "TransformerWorldModel"),
    2: ("v2_training.world_model", "JEPAWorldModel"),
    3: ("v3_training.world_model", "WaveNetGRUWorldModel"),
    4: ("v4_training.world_model", "MambaWorldModel"),
    5: ("v5_training.world_model", "HybridMambaAttentionWorldModel"),
    6: ("v6_training.world_model", "CausalJEPAWorldModel"),
    7: ("v7_training.world_model", "ViTWorldModel"),
    8: ("v8_training.world_model", "NeuralODEWorldModel"),
    9: ("v9_training.world_model", "MoEWorldModel"),
}


class MetaRouter(nn.Module):
    """
    Dynamic routing MLP that produces per-model weights.

    Input: context vector (rolling IC per model, regime probs, volatility)
    Output: softmax weights over N models (optionally per-horizon)

    ~10K params total.
    """

    def __init__(
        self,
        context_dim: int = META_CONTEXT_DIM,
        n_models: int = META_N_MODELS,
        hidden_dim: int = META_ROUTER_HIDDEN,
        per_horizon: bool = META_PER_HORIZON_ROUTING,
        temperature: float = META_TEMPERATURE,
    ):
        super().__init__()
        self.n_models = n_models
        self.per_horizon = per_horizon
        self.temperature = temperature
        self.n_horizons = len(REWARD_HORIZONS)

        output_dim = n_models * self.n_horizons if per_horizon else n_models

        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize to near-uniform weights."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor) -> dict:
        """
        Compute routing weights.

        Args:
            context: [B, context_dim]

        Returns:
            dict of {horizon: [B, n_models]} softmax weights
            If not per_horizon, all horizons share the same weights.
        """
        logits = self.net(context)  # [B, output_dim]

        if self.per_horizon:
            # Reshape to [B, n_horizons, n_models]
            logits = logits.view(-1, self.n_horizons, self.n_models)
            weights = F.softmax(logits / self.temperature, dim=-1)
            return {
                h: weights[:, i, :]
                for i, h in enumerate(REWARD_HORIZONS)
            }
        else:
            weights = F.softmax(logits / self.temperature, dim=-1)  # [B, n_models]
            return {h: weights for h in REWARD_HORIZONS}


class MetaEnsemble(nn.Module):
    """
    V10 Meta-Ensemble: Dynamic routing across frozen V1-V9 models.

    VRAM management: Models are loaded on-demand. On 8GB GPU,
    only one model is kept on GPU at a time during prediction caching.
    """

    def __init__(self, model_paths: dict = None, router: MetaRouter = None):
        """
        Args:
            model_paths: dict of {version_int: path_to_checkpoint}
                e.g., {1: "models/wm/v1/v1_0/base/v1_0_f13_wm_best_ema.pt", ...}
            router: MetaRouter instance (created if None)
        """
        super().__init__()

        self.model_paths = model_paths or {}
        self.enabled_models = []  # List of version ints that have checkpoints

        # Discover available models
        if not self.model_paths:
            self._discover_models()
        else:
            self.enabled_models = sorted(self.model_paths.keys())

        self.n_active_models = len(self.enabled_models)

        # Create router sized to actual number of active models
        self.router = router or MetaRouter(n_models=self.n_active_models)

        # We don't hold models in memory -- load on-demand during prediction
        # TwoHot decoder (shared, for decoding logits from base models)
        from v1_0_training.components import TwoHotSymlog
        self.bucketer = TwoHotSymlog(NUM_BINS, BIN_MIN, BIN_MAX, DEVICE)

    def _discover_models(self):
        """Find available model checkpoints."""
        for v in range(1, 10):
            if not META_MODEL_ENABLED.get(v, True):
                continue

            model_dir = PROJECT_ROOT / "models" / f"v{v}" / f"v{v}" / "base"
            # Try EMA weights first, then best (V2/V6 pattern), then regular weights
            for name in [f"v{v}_wm_best_ema.pt", f"v{v}_wm_best.pt", f"v{v}_wm_weights.pt"]:
                path = model_dir / name
                if path.exists():
                    self.model_paths[v] = path
                    self.enabled_models.append(v)
                    break

        print(f"  [OK] Found {len(self.enabled_models)} model checkpoints: "
              f"V{', V'.join(str(v) for v in self.enabled_models)}")

    def _load_model(self, version: int):
        """
        Load a single model onto GPU, plus its RevIN instance.

        Uses the model registry to dynamically import the correct class.
        Each version's module uses different import patterns internally,
        so we add the version's directory to sys.path before importing.

        Returns:
            (model, revin): both on DEVICE, both in eval() with requires_grad=False.
            revin may be None if checkpoint predates RevIN save (identity passthrough).
        """
        path = self.model_paths[version]

        if version not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown version: {version}")

        module_path, class_name = _MODEL_REGISTRY[version]

        # Add the version's group and training directories to sys.path so that
        # package imports (e.g., 'from v1_0_training.world_model import ...') and
        # internal bare imports (e.g., 'from components import ...') resolve.
        v_group_dir = str(_SRC_DIR / f"v{version}")
        # V1 base was renamed to v1_0_training; all others follow v{N}_training
        train_dir_name = "v1_0_training" if version == 1 else f"v{version}_training"
        v_train_dir = str(Path(v_group_dir) / train_dir_name)
        if v_group_dir not in sys.path:
            sys.path.insert(0, v_group_dir)
        if v_train_dir not in sys.path:
            sys.path.insert(0, v_train_dir)

        # Clear cached bare imports to prevent cross-contamination
        # (each version has its own components.py and settings.py)
        for cached_key in ["components", "settings"]:
            sys.modules.pop(cached_key, None)

        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)

        model = ModelClass()
        # weights_only=False: V11-V14 checkpoints store non-tensor metadata
        # (version strings, gate status) that weights_only=True refuses to
        # deserialize -> V10 could not load those base models. Our own trusted ckpts.
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        # Support both formats:
        #   - new: {"model_state_dict": ..., "revin_state_dict": ...}
        #   - legacy (pre-fix): raw state_dict (plain dict of param_name -> tensor)
        # strict=False per CLAUDE.md schema-compat invariant (meta-ensemble
        # loads ckpts across architectures; emit WARN on non-trivial mismatch).
        revin = None
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            _m, _u = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if len(_m) > 5 or len(_u) > 0:
                print(f"  [meta load WARN] {path.name}: missing={len(_m)} unexpected={len(_u)}")
            if "revin_state_dict" in ckpt:
                from revin import RevIN
                revin = RevIN(num_features=INPUT_DIM).to(DEVICE)
                revin.load_state_dict(ckpt["revin_state_dict"], strict=False)
                revin.eval()
                for p in revin.parameters():
                    p.requires_grad = False
        else:
            # Legacy flat state_dict -- RevIN params not saved, use identity
            _m, _u = model.load_state_dict(ckpt, strict=False)
            if len(_m) > 5 or len(_u) > 0:
                print(f"  [meta load WARN legacy] {path.name}: missing={len(_m)} unexpected={len(_u)}")

        model.to(DEVICE)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Clean up after loading to prevent contamination of next model
        for cached_key in ["components", "settings"]:
            sys.modules.pop(cached_key, None)

        return model, revin

    @torch.no_grad()
    def get_model_predictions(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
    ) -> dict:
        """
        Run all enabled models and collect decoded return predictions.

        Manages VRAM by loading/unloading models one at a time.

        Args:
            obs_seq: [B, T, INPUT_DIM]
            asset_id: [B]

        Returns:
            dict of {version: {horizon: [B, T] predictions}}
        """
        predictions = {}

        for v in self.enabled_models:
            model, revin = self._load_model(v)

            # Apply RevIN normalization (same as training-time preprocessing)
            obs_input = revin(obs_seq, mode='norm') if revin is not None else obs_seq

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                outputs = model.forward_train(obs_input, asset_id)

            # Decode return logits to scalar predictions
            preds = {}
            for h in REWARD_HORIZONS:
                logits = outputs["return_logits"][h]
                preds[h] = model.bucketer.decode(logits)  # [B, T]

            predictions[v] = preds

            # Free VRAM
            del model
            if revin is not None:
                del revin
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions

    def forward(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        context: torch.Tensor,
        precomputed_preds: dict = None,
    ) -> dict:
        """
        Meta-ensemble forward pass.

        Args:
            obs_seq: [B, T, INPUT_DIM]
            asset_id: [B]
            context: [B, META_CONTEXT_DIM] routing context
            precomputed_preds: optional dict of pre-computed model predictions
                (avoids re-running models during router training)

        Returns:
            dict with:
                'return_preds': {horizon: [B, T]} weighted predictions
                'weights': {horizon: [B, n_models]} routing weights
                'per_model_preds': {version: {horizon: [B, T]}} individual predictions
        """
        # Get model predictions
        if precomputed_preds is not None:
            all_preds = precomputed_preds
        else:
            all_preds = self.get_model_predictions(obs_seq, asset_id)

        # Compute routing weights
        weights = self.router(context)  # {horizon: [B, n_models]}

        # Weighted average of predictions
        ensemble_preds = {}
        for h in REWARD_HORIZONS:
            w = weights[h]  # [B, n_active_models]

            # Stack predictions from all models: [B, T, n_active_models]
            pred_stack = torch.stack(
                [all_preds[v][h] for v in self.enabled_models],
                dim=-1,
            )

            # Weighted average: [B, T]
            # Expand weights to [B, 1, n_active_models] for broadcasting
            w_expanded = w.unsqueeze(1)
            ensemble_preds[h] = (pred_stack * w_expanded).sum(dim=-1)

        return {
            "return_preds": ensemble_preds,
            "weights": weights,
            "per_model_preds": all_preds,
        }


class MetaContextComputer:
    """
    Computes the META_CONTEXT_DIM context vector for the router.

    Context layout (46 dimensions):
      [0:36]  Rolling IC per model per horizon (9 models * 4 horizons)
      [36:39] Regime probabilities (bear, neutral, bull)
      [39:40] Volatility level
      [40:44] Mean IC per horizon (across all models)
      [44:46] IC variance per horizon (measures model disagreement)
    """

    def __init__(self, lookback: int = 2000):
        self.lookback = lookback
        self.reset()

    def reset(self):
        self._ic_per_model = {}  # {(version, horizon): deque}
        self._regime_probs = []
        self._volatility = []

    def update_ic(self, version: int, horizon: int, ic_value: float):
        """Record an IC measurement for a given model and horizon."""
        key = (version, horizon)
        if key not in self._ic_per_model:
            self._ic_per_model[key] = deque(maxlen=self.lookback)
        self._ic_per_model[key].append(ic_value)

    def update_regime(self, probs: np.ndarray):
        """Record regime probabilities [bear, neutral, bull]."""
        self._regime_probs.append(probs)
        if len(self._regime_probs) > self.lookback:
            self._regime_probs = self._regime_probs[-self.lookback:]

    def update_volatility(self, vol: float):
        """Record a volatility measurement."""
        self._volatility.append(vol)
        if len(self._volatility) > self.lookback:
            self._volatility = self._volatility[-self.lookback:]

    def get_context(self) -> np.ndarray:
        """Build the META_CONTEXT_DIM context vector from accumulated stats."""
        ctx = np.zeros(META_CONTEXT_DIM, dtype=np.float32)

        # [0:36] Rolling IC per model per horizon
        for v_idx, v in enumerate(range(1, 10)):
            for h_idx, h in enumerate(REWARD_HORIZONS):
                key = (v, h)
                if key in self._ic_per_model and len(self._ic_per_model[key]) > 0:
                    ic = float(np.mean(list(self._ic_per_model[key])))
                    # Scale IC to [-1, 1] range (typical IC is 0.01-0.10)
                    ctx[v_idx * 4 + h_idx] = np.clip(ic * 10.0, -1.0, 1.0)

        # [36:39] Regime
        if self._regime_probs:
            regime = np.mean(self._regime_probs[-100:], axis=0)
            ctx[36:39] = regime * 2.0 - 1.0  # scale to [-1, 1]

        # [39:40] Volatility
        if self._volatility:
            ctx[39] = np.clip(np.mean(self._volatility[-100:]), -1.0, 1.0)

        # [40:44] Mean IC per horizon (across all models)
        for h_idx, h in enumerate(REWARD_HORIZONS):
            ics = []
            for v in range(1, 10):
                key = (v, h)
                if key in self._ic_per_model and len(self._ic_per_model[key]) > 0:
                    ics.append(float(np.mean(list(self._ic_per_model[key]))))
            if ics:
                ctx[40 + h_idx] = np.clip(np.mean(ics) * 10.0, -1.0, 1.0)

        # [44:46] IC variance (model disagreement) for horizons 1 and 4
        for h_idx_pair, h in enumerate(REWARD_HORIZONS[:2]):
            ics = []
            for v in range(1, 10):
                key = (v, h)
                if key in self._ic_per_model and len(self._ic_per_model[key]) > 0:
                    ics.append(float(np.mean(list(self._ic_per_model[key]))))
            if len(ics) > 1:
                ctx[44 + h_idx_pair] = np.clip(np.var(ics) * 100.0, 0.0, 1.0)

        return ctx

    def get_context_tensor(self, batch_size: int = 1, device: str = DEVICE) -> torch.Tensor:
        """Get context as a batched tensor."""
        ctx = self.get_context()
        return torch.from_numpy(ctx).unsqueeze(0).expand(batch_size, -1).to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Test router
    router = MetaRouter().to(DEVICE)
    print(f"MetaRouter parameters: {count_parameters(router):,}")

    ctx = torch.randn(4, META_CONTEXT_DIM).to(DEVICE)
    weights = router(ctx)
    for h in REWARD_HORIZONS:
        w = weights[h]
        print(f"  Horizon {h}: weights shape={w.shape}, sum={w.sum(dim=-1).mean():.4f}")

    # Test context computer
    cc = MetaContextComputer()
    for v in range(1, 10):
        for h in REWARD_HORIZONS:
            cc.update_ic(v, h, np.random.uniform(0.02, 0.10))
    cc.update_regime(np.array([0.2, 0.6, 0.2]))
    cc.update_volatility(0.5)

    ctx_vec = cc.get_context()
    print(f"\n  Context vector shape: {ctx_vec.shape}")
    print(f"  Context range: [{ctx_vec.min():.3f}, {ctx_vec.max():.3f}]")

    print("\n[OK] V10 Meta-Ensemble sanity check passed.")
    print("[NOTE] Full ensemble requires trained V1-V9 checkpoints.")
