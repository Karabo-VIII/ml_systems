"""
V1.E Cross-Model Ensemble -- Heterogeneous V1 Family Ensemble

Discovers and loads trained V1.x models (V1.0, V1.1, V1.4, V1.6),
averages their return predictions for improved IC via architectural diversity.

Key features:
  - Auto-discovers trained models from models/wm/v1/*/base/*_best_ema.pt
  - Handles heterogeneous input dims (f13 vs f18 vs f19 vs f22) via input slicing
  - All V1 models use TwoHot bins [-1.0, 1.0] (raw-return targets, V51+)
  - sys.path isolation prevents settings/components cross-contamination
  - Uniform or XD-gated averaging of return logits
  - All base models frozen (no gradients) -- inference only
  - Note: loss_dict keys differ across versions (V1.0 lacks kl_raw/regime_acc);
    this doesn't affect ensemble inference which only uses encode_sequence()

IC_ensemble = IC_single * sqrt(K / (1 + (K-1) * rho))
With K=3-6 diverse models and rho ~ 0.3, expect 30-60% IC boost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent           # src/wm/v1/
_V1_GROUP = _THIS_DIR                                 # src/wm/v1/ (ensemble lives at group root)
_SRC_DIR = _V1_GROUP.parent                           # src/
_PROJECT_ROOT = _SRC_DIR.parent                       # project root
_V1_0_DIR = _V1_GROUP / "v1_0_training"               # V1.0 base (settings, components)

for _p in [str(_V1_0_DIR), str(_V1_GROUP), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from settings import DEVICE, REWARD_HORIZONS, NUM_BINS, BIN_MIN, BIN_MAX
from components import TwoHotSymlog


# ---------------------------------------------------------------------------
# V1 Family Model Registry
# ---------------------------------------------------------------------------
# Each entry: {dir, n_features, ckpt_pattern, has_base_dim}
# has_base_dim: V1 base does NOT have base_dim param; V1.1+ do
_V1_FAMILY = {
    # V1.0: reference baseline (f13, no XD split)
    "v1_0": {
        "dir": "v1_0_training",
        "n_features": 13,
        "ckpt": "v1_0_f13_wm_best_ema.pt",
        "has_base_dim": False,
    },
    # V1.0 at f34: SOTA feature baseline
    "v1_0_f34": {
        "dir": "v1_0_training",
        "n_features": 34,
        "ckpt": "v1_0_f34_wm_best_ema.pt",
        "has_base_dim": False,
    },
    # V1.1 at f13: diversity variant (old base only, no XD)
    "v1_1_f13": {
        "dir": "v1_1_training",
        "n_features": 13,
        "ckpt": "v1_1_f13_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 13, "base_dim": 13},
    },
    # V1.1 at f22: default (17 base + 5 XD)
    # Ensemble tensor: [13 old base, 5 XD, xd_ma_dist, 4 new base]
    # Model expects: [17 base, 5 XD] → non-contiguous routing needed
    "v1_1": {
        "dir": "v1_1_training",
        "n_features": 22,
        "ckpt": "v1_1_f22_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 22, "base_dim": 17},
        "feat_indices": list(range(13)) + list(range(19, 23)) + list(range(13, 18)),
    },
    # V1.4 at f13: FeatureAttentionBlock (iTransformer-style cross-feature attention)
    "v1_4": {
        "dir": "v1_4_training",
        "n_features": 13,
        "ckpt": "v1_4_f13_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 13, "base_dim": 13},
    },
    # V1.1 at f18
    "v1_1_f18": {
        "dir": "v1_1_training",
        "n_features": 18,
        "ckpt": "v1_1_f18_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 18, "base_dim": 18},
    },
    # V1.1 at f25: best IC (+13% over f13)
    "v1_1_f25": {
        "dir": "v1_1_training",
        "n_features": 25,
        "ckpt": "v1_1_f25_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 25, "base_dim": 25},
    },
    # V1.1 "f37": trained with input_dim=37 but base_dim=30 (pre-SOTA schema).
    # obs_encoder: [256, 69] = 37+32 asset_embed. decoder: [30, 256].
    # posterior: [256, 286] = 256+30. Accepts 37 features, reconstructs 30.
    "v1_1_f37": {
        "dir": "v1_1_training",
        "n_features": 37,
        "ckpt": "v1_1_f37_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 37, "base_dim": 30},
    },
    # V1.4 at f18
    "v1_4_f18": {
        "dir": "v1_4_training",
        "n_features": 18,
        "ckpt": "v1_4_f18_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 18, "base_dim": 18},
    },
    # V1.4 at f25
    "v1_4_f25": {
        "dir": "v1_4_training",
        "n_features": 25,
        "ckpt": "v1_4_f25_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 25, "base_dim": 25},
    },
    # V1.6 at f18
    "v1_6_f18": {
        "dir": "v1_6_training",
        "n_features": 18,
        "ckpt": "v1_6_f18_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 18, "base_dim": 18},
    },
    # V1.6 at f25
    "v1_6_f25": {
        "dir": "v1_6_training",
        "n_features": 25,
        "ckpt": "v1_6_f25_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 25, "base_dim": 25},
    },
    # V1.2, V1.3, V1.5 archived (subsumed by V1.6 or failed experiments)
    # V1.6 at f13: all techniques (KL anneal, Gumbel, ATME)
    "v1_6": {
        "dir": "v1_6_training",
        "n_features": 13,
        "ckpt": "v1_6_f13_wm_best_ema.pt",
        "has_base_dim": True,
        "ctor_kwargs": {"input_dim": 13, "base_dim": 13},
    },
    # V1.7 archived (never trained, requires OOF prereq)
}

# ---------------------------------------------------------------------------
# Ensemble Feature List — union of all features in ensemble tensor order
# ---------------------------------------------------------------------------
# This ordering is the contract for feat_indices above. Both validate_ensemble.py
# and train_agent.py MUST load data in this order for feat_indices to work.
# Ordering: [13 old base, 5 XD, xd_ma_distance, 4 new base]
# V1.7 baseline predictions removed (V1.7 archived)
ENSEMBLE_FEATURE_LIST = [
    # Base features (0-12) -- shared by all V1 models
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    # Extended base (13-17) -- f18+ models
    "norm_ma_distance", "norm_whale", "norm_efficiency",
    "norm_return_4", "norm_return_16",
    # Tier 1 (18-20) -- f21+ models
    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
    # Hawkes (21-24) -- f25 models
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # IC-boost (25-29) -- f30+ models
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # Cross-asset (30-36) -- f37 models
    "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
    "xd_cross_return_mean", "xd_cross_vol_mean",
    "xd_ma_distance", "xd_momentum_rank",
]


# ---------------------------------------------------------------------------
# XD-Conditioned Gating (optional)
# ---------------------------------------------------------------------------
class EnsembleGating(nn.Module):
    """
    XD-conditioned soft weighting of K models.

    Linear(xd_dim, K) -> softmax -> per-model weights [B, K].
    Initialized near-uniform (zeros) so untrained gating = uniform averaging.
    ~xd_dim*K parameters -- structurally cannot overfit.

    xd_dim: 5 for models with 5 XD features, 6 when V1.5 (xd_ma_distance) is included.
    """

    def __init__(self, xd_dim: int = 6, n_models: int = 2):
        super().__init__()
        self.gate = nn.Linear(xd_dim, n_models, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, xd_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xd_features: [B, xd_dim] XD values from last bar of sequence (5 or 6)
        Returns:
            weights: [B, K] normalized gating weights (sum=1 per sample)
        """
        return F.softmax(self.gate(xd_features), dim=-1)


# ---------------------------------------------------------------------------
# Model Loading (sys.path isolated, reuses V10's proven pattern)
# ---------------------------------------------------------------------------
def _load_v1_model(
    model_key: str,
    spec: dict,
    project_root: Path,
    device: str = DEVICE,
) -> tuple:
    """
    Load a V1 family model with proper sys.path isolation.

    Returns:
        (model, n_features, model_key) or None if checkpoint not found.
    """
    training_dir = _V1_GROUP / spec["dir"]
    ckpt_path = project_root / "models" / "v1" / spec["dir"].replace("_training", "") / "base" / spec["ckpt"]

    if not ckpt_path.exists():
        return None

    # Save current sys.path and modules state
    saved_path = sys.path[:]
    saved_modules = {}
    for mod_name in ["settings", "components", "world_model"]:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    try:
        # Push this model's training dir to front of path
        sys.path.insert(0, str(training_dir))
        sys.path.insert(0, str(_V1_GROUP))

        # Import this version's world_model module
        wm_module = importlib.import_module("world_model")
        ModelClass = wm_module.TransformerWorldModel

        # Construct model -- use explicit kwargs if provided, else defaults
        ctor_kwargs = spec.get("ctor_kwargs", {})
        model = ModelClass(**ctor_kwargs)
        model = model.to(device)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt  # Flat state dict

        # Load with strict=False to handle minor mismatches
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [WARN] {model_key}: {len(missing)} missing keys (first 3: {missing[:3]})")
        if unexpected:
            print(f"  [WARN] {model_key}: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")

        # Freeze all parameters
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        return (model, spec["n_features"], model_key)

    except Exception as e:
        print(f"  [WARN] Failed to load {model_key}: {e}")
        return None

    finally:
        # Restore sys.path and modules
        sys.path[:] = saved_path
        for mod_name in ["settings", "components", "world_model"]:
            sys.modules.pop(mod_name, None)
        for mod_name, mod in saved_modules.items():
            sys.modules[mod_name] = mod


# ---------------------------------------------------------------------------
# Cross-Model Ensemble
# ---------------------------------------------------------------------------
class CrossModelEnsemble(nn.Module):
    """
    V1.E: Cross-Model Ensemble across V1 family variants.

    Loads trained V1.x models, averages their return predictions.
    Handles heterogeneous input dims (f13 vs f18) via automatic slicing.

    Usage:
        ens = CrossModelEnsemble()  # auto-discovers trained models
        outputs = ens.forward_train(obs_seq, asset_id)
        # outputs["return_logits"] = averaged predictions across all models
    """

    def __init__(
        self,
        model_keys: list = None,
        use_gating: bool = False,
        device: str = DEVICE,
    ):
        """
        Args:
            model_keys: List of keys from _V1_FAMILY to load. If None, auto-discovers
                        all models with existing checkpoints.
            use_gating: If True, create XD-conditioned gating (trainable).
                        If False, use uniform averaging (default, zero params).
            device: Device to load models on.
        """
        super().__init__()

        self.device = device
        self.bucketer = TwoHotSymlog(NUM_BINS, BIN_MIN, BIN_MAX, device)

        # Discover or load specified models
        if model_keys is None:
            model_keys = list(_V1_FAMILY.keys())

        self.model_entries = []  # List of (model, n_features, key)
        self.models = nn.ModuleList()
        self._has_heterogeneous_bins = False  # Track if any model has different bins

        print("[CrossModelEnsemble] Discovering V1 family models...")
        for key in model_keys:
            if key not in _V1_FAMILY:
                print(f"  [WARN] Unknown model key: {key}, skipping")
                continue

            result = _load_v1_model(key, _V1_FAMILY[key], _PROJECT_ROOT, device)
            if result is not None:
                model, n_features, model_key = result
                self.models.append(model)
                entry = {"n_features": n_features, "key": model_key}
                # Preserve base_dim for gating XD computation
                base_dim = _V1_FAMILY[key].get("ctor_kwargs", {}).get("base_dim", 13)
                entry["base_dim"] = base_dim
                # Preserve feat_indices for non-contiguous feature routing
                if "feat_indices" in _V1_FAMILY[key]:
                    entry["feat_indices"] = _V1_FAMILY[key]["feat_indices"]
                if _V1_FAMILY[key].get("narrow_bins", False):
                    entry["narrow_bins"] = True
                    self._has_heterogeneous_bins = True
                self.model_entries.append(entry)
                print(f"  [OK] {model_key} (f{n_features})")

        self.n_models = len(self.models)
        if self.n_models == 0:
            raise FileNotFoundError(
                "No V1 family models found. Train at least one model first.\n"
                "Expected checkpoints in models/wm/v1/*/base/*_best_ema.pt"
            )

        print(f"  [OK] Loaded {self.n_models} models for ensemble")

        # Optional XD-conditioned gating
        # xd_dim: max XD features across loaded models (5 for f18, 6 for f19)
        # Use base_dim from ctor_kwargs when available (V1.6 base_dim=17, not 13)
        self.gating = None
        if use_gating and self.n_models > 1:
            max_xd = max(
                e["n_features"] - e.get("base_dim", 13)
                for e in self.model_entries
            )
            self.gating = EnsembleGating(xd_dim=max_xd, n_models=self.n_models)
            print(f"  [OK] XD gating enabled (xd_dim={max_xd}, {max_xd * self.n_models} params)")

    def _route_input(self, obs_seq: torch.Tensor, model_idx: int) -> torch.Tensor:
        """
        Route input to model based on its expected n_features.

        f13 models use simple prefix slicing (obs[:, :, :13]).
        f22/f23 models need non-contiguous indices because their feature order
        [17 base, 5-6 XD] differs from the ensemble tensor order
        [13 old base, 5 XD, xd_ma_dist, 4 new base].
        """
        entry = self.model_entries[model_idx]
        feat_indices = entry.get("feat_indices")
        if feat_indices is not None:
            return obs_seq[:, :, feat_indices]
        n_feat = entry["n_features"]
        if n_feat < obs_seq.shape[-1]:
            return obs_seq[:, :, :n_feat]
        return obs_seq

    @torch.no_grad()
    def forward_train(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        masked_obs_seq: torch.Tensor = None,
        xd_features: torch.Tensor = None,
    ) -> dict:
        """
        Run all models, average return predictions.

        Args:
            obs_seq:        [B, T, F] -- full feature sequences (F=23 union of all V1 features)
            asset_id:       [B] -- integer asset indices
            masked_obs_seq: [B, T, F] -- masked observations (optional, for training)
            xd_features:    [B, xd_dim] -- XD features for gating (5 or 6, optional)

        Returns:
            Dict with averaged return_logits and first model's other outputs.
        """
        all_outputs = []

        for i, model in enumerate(self.models):
            # Route input based on model's expected features
            routed_obs = self._route_input(obs_seq, i)
            routed_mask = self._route_input(masked_obs_seq, i) if masked_obs_seq is not None else None

            out = model.forward_train(routed_obs, asset_id, routed_mask)
            all_outputs.append(out)

        # Average return predictions across models.
        # All active V1 models use bin range [-1.0, 1.0].
        # The heterogeneous bin path is retained for backward compatibility
        # in case models with different bin ranges are ever mixed.
        if self._has_heterogeneous_bins:
            # Per-model decode then average (handles different bin ranges)
            avg_return_preds = {}
            for h in REWARD_HORIZONS:
                decoded_list = []
                for i, (model, out) in enumerate(zip(self.models, all_outputs)):
                    decoded_list.append(model.bucketer.decode(out["return_logits"][h]))
                stacked = torch.stack(decoded_list, dim=0)  # [K, B, T]
                avg_return_preds[h] = stacked.mean(dim=0)   # [B, T]
            # Use first model's logits as reference (for backward compat code
            # that expects return_logits key, though decode should use _decoded_returns)
            avg_return_logits = {h: all_outputs[0]["return_logits"][h] for h in REWARD_HORIZONS}
        elif self.gating is not None and xd_features is not None:
            # XD-conditioned weighted average of logits (homogeneous bins)
            weights = self.gating(xd_features)  # [B, K]
            avg_return_logits = {}
            for h in REWARD_HORIZONS:
                stacked = torch.stack(
                    [out["return_logits"][h] for out in all_outputs], dim=1
                )  # [B, K, T, NUM_BINS]
                w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
                avg_return_logits[h] = (stacked * w).sum(dim=1)  # [B, T, NUM_BINS]
            avg_return_preds = None
        else:
            # Uniform average of logits (homogeneous bins)
            avg_return_logits = {}
            for h in REWARD_HORIZONS:
                stacked = torch.stack(
                    [out["return_logits"][h] for out in all_outputs], dim=0
                )  # [K, B, T, NUM_BINS]
                avg_return_logits[h] = stacked.mean(dim=0)  # [B, T, NUM_BINS]
            avg_return_preds = None

        # Use first model's non-return outputs as reference
        base = all_outputs[0]
        result = {
            "recon": base["recon"],
            "return_logits": avg_return_logits,
            "regime_logits": base["regime_logits"],
            "prior_logits": base["prior_logits"],
            "post_logits": base["post_logits"],
            "h_seq": base["h_seq"],
            "z_post": base["z_post"],
            "ret_trunk": base.get("ret_trunk"),
            "n_models": self.n_models,
        }
        if avg_return_preds is not None:
            result["_decoded_returns"] = avg_return_preds
        return result

    def decode_returns(self, return_logits: dict, pre_decoded: dict = None) -> dict:
        """Decode TwoHot logits to continuous return predictions.

        When models have heterogeneous bin ranges, pre_decoded contains
        per-model decoded then averaged scalar predictions (already computed
        in forward_train). Falls back to ensemble bucketer decode otherwise.
        """
        if pre_decoded is not None:
            return pre_decoded
        decoded = {}
        for h in REWARD_HORIZONS:
            if h in return_logits:
                decoded[h] = self.bucketer.decode(return_logits[h])
        return decoded

    # ------------------------------------------------------------------
    # Agent-compatible interface (duck-typing with individual world models)
    # ------------------------------------------------------------------
    # The trading agent environment calls:
    #   model.encode_sequence(obs_seq, asset_id) -> (h_seq, z_post, return_preds)
    #   model.regime_head(feat) -> regime_logits
    #   model.posterior_head(input) -> post_logits
    #   model.latent_dim, model.classes
    # These methods/properties proxy to forward_train + first model's attributes.

    @torch.no_grad()
    def encode_sequence(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
    ):
        """
        Agent-compatible encode: run ensemble forward, decode returns.

        Returns:
            h_seq:        [B, T, d_model] from first model
            z_post:       [B, T, flat_dim] from first model
            return_preds: dict {horizon: [B, T]} scalar predictions (ensemble-averaged)
        """
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = self.decode_returns(
            outputs["return_logits"],
            pre_decoded=outputs.get("_decoded_returns"),
        )

        # Cache full-sequence outputs for precomputation access.
        self._last_outputs = outputs  # {regime_logits: [B,T,3], post_logits: [B,T,flat], ...}

        # Cache last-timestep for per-step regime_head / posterior_head calls.
        self._cached_regime_logits = outputs["regime_logits"][:, -1]  # [B, 3]
        self._cached_post_logits = outputs["post_logits"][:, -1]     # [B, flat_dim]

        return outputs["h_seq"], outputs["z_post"], return_preds

    def regime_head(self, feat: torch.Tensor) -> torch.Tensor:
        """Return cached regime logits from last encode_sequence call."""
        if hasattr(self, "_cached_regime_logits") and self._cached_regime_logits is not None:
            return self._cached_regime_logits
        # Fallback: use first model's regime_head directly
        return self.models[0].regime_head(feat)

    def posterior_head(self, input_feat: torch.Tensor) -> torch.Tensor:
        """Return cached post logits from last encode_sequence call."""
        if hasattr(self, "_cached_post_logits") and self._cached_post_logits is not None:
            return self._cached_post_logits
        # Fallback: use first model
        return self.models[0].posterior_head(input_feat)

    @property
    def latent_dim(self) -> int:
        """Proxy to first model's latent_dim (for posterior entropy calculation)."""
        return self.models[0].latent_dim

    @property
    def classes(self) -> int:
        """Proxy to first model's classes (categorical RSSM classes)."""
        return self.models[0].classes

    def get_model_info(self) -> list:
        """Return info about loaded models for diagnostics."""
        return [
            {
                "key": entry["key"],
                "n_features": entry["n_features"],
                "params": sum(p.numel() for p in model.parameters()),
            }
            for entry, model in zip(self.model_entries, self.models)
        ]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Model Ensemble -- Smoke Test")
    print("=" * 70)

    try:
        ens = CrossModelEnsemble()
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)

    print("\nModel info:")
    for info in ens.get_model_info():
        print(f"  {info['key']}: f{info['n_features']}, {info['params']:,} params")

    # Test forward pass with dummy data
    # Use max feature dim across loaded models (19 if V1.5 present, else 18)
    max_feat = max(info['n_features'] for info in ens.get_model_info())
    B, T = 2, 16
    obs = torch.randn(B, T, max_feat, device=DEVICE)
    asset_id = torch.zeros(B, dtype=torch.long, device=DEVICE)

    outputs = ens.forward_train(obs, asset_id)

    print(f"\nForward pass OK:")
    print(f"  n_models: {outputs['n_models']}")
    for h in REWARD_HORIZONS:
        logits = outputs["return_logits"][h]
        print(f"  return_{h}: {logits.shape}")

    # Decode
    decoded = ens.decode_returns(outputs["return_logits"])
    for h in REWARD_HORIZONS:
        vals = decoded[h]
        print(f"  decoded_{h}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    print("\n[OK] Cross-model ensemble smoke test passed.")
