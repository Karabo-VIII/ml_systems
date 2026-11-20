"""
V10 Meta-Ensemble World Model.

Aggregates predictions from working V1-class world models (V1.1, V1.4, V1.6)
via weighted averaging. Weights are set by each model's validation IC
(higher IC → higher weight).

This is NOT trained from scratch — it loads frozen best_ema checkpoints
for each component, runs them on the same input, and averages return_logits.

Architecture choice: simple weighted average beats learned gating when
component models have similar IC profiles (theoretical result: Bates &
Granger 1969; the optimal weights for correlated-error forecasts are
inversely proportional to forecast-error variance, which is monotonic in
IC). Start simple.

For V10.1 (future): replace averaging with a small LightGBM gating model
that takes {market regime, volatility, current predictions} → dynamic
weights. Defer until V10.0 baseline established.

Interface: V1-compatible `forward_train()` and `get_loss()` — drop-in
replacement for any V1.x model in downstream code.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def _load_v1x_checkpoint(version: str, feature_count: int = 34,
                          device: str = "cuda") -> Optional[nn.Module]:
    """Load a V1.x best_ema checkpoint. Returns frozen model in eval mode.

    Args:
        version: "v1_1", "v1_4", "v1_6"
        feature_count: feature dimensionality (13 / 18 / 21 / 25 / 30 / 34)

    Returns:
        Loaded model or None if checkpoint missing.
    """
    ckpt_path = (MODEL_DIR / "v1" / version / "base"
                 / f"{version}_f{feature_count}_wm_best_ema.pt")
    if not ckpt_path.exists():
        return None

    # Path setup for loading V1.x modules
    v1x_path = str(PROJECT_ROOT / "src" / "wm" / "v1" / f"{version}_training")
    if v1x_path not in sys.path:
        sys.path.insert(0, v1x_path)

    from world_model import TransformerWorldModel
    import settings
    # Ensure feature count matches
    settings.INPUT_DIM = feature_count

    model = TransformerWorldModel()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # best_ema.pt is a state dict directly (not wrapped)
    sd = state if isinstance(state, dict) and "state_dict" not in state else state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Remove from path so next load uses its own settings
    sys.path.remove(v1x_path)
    return model


class V10MetaEnsemble(nn.Module):
    """Weighted average of V1.1, V1.4, V1.6 return predictions.

    Weights default to equal (1/3 each) at init — can be overridden via
    set_weights() based on validation IC measurement.
    """

    def __init__(self, versions: Optional[List[str]] = None,
                 weights: Optional[List[float]] = None,
                 feature_count: int = 34, device: str = "cuda"):
        super().__init__()
        self.versions = versions or ["v1_1", "v1_4", "v1_6"]
        self.feature_count = feature_count
        self.device_str = device

        components = nn.ModuleDict()
        for v in self.versions:
            m = _load_v1x_checkpoint(v, feature_count, device)
            if m is None:
                print(f"[V10] WARNING: missing checkpoint for {v}, skipping")
                continue
            components[v] = m
        self.components = components

        if not components:
            raise RuntimeError("V10 requires at least one V1.x checkpoint")

        # Weights (learnable, but init frozen)
        w = weights or [1.0] * len(components)
        if len(w) != len(components):
            w = [1.0] * len(components)
        self.register_buffer("weights", torch.tensor(w, dtype=torch.float32))
        self._normalize_weights()

    def _normalize_weights(self):
        with torch.no_grad():
            w = self.weights.clamp(min=1e-6)
            self.weights.copy_(w / w.sum())

    def set_weights(self, val_ic_per_version: Dict[str, float]):
        """Set ensemble weights ∝ validation IC of each component."""
        w = []
        for v in self.components.keys():
            ic = max(val_ic_per_version.get(v, 0.01), 0.01)
            w.append(ic)
        self.weights = torch.tensor(w, dtype=torch.float32, device=self.weights.device)
        self._normalize_weights()

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        """V1-compatible forward. Returns averaged predictions.

        Component models are frozen; this is pure inference + aggregation.
        """
        outputs = []
        for v, m in self.components.items():
            with torch.no_grad():
                o = m.forward_train(obs_seq, asset_id, masked_obs_seq)
            outputs.append(o)

        # Weighted average of return_logits per horizon
        avg_return_logits = {}
        sample = outputs[0]["return_logits"]
        for h in sample.keys():
            stacked = torch.stack([o["return_logits"][h] for o in outputs], dim=0)
            # [N_models, B, T, bins] -> [B, T, bins]
            w = self.weights.view(-1, *([1] * (stacked.dim() - 1)))
            avg_return_logits[h] = (stacked * w).sum(dim=0)

        # For auxiliary outputs, take first model's (they're not averaged)
        return {
            "return_logits": avg_return_logits,
            "regime_logits": outputs[0].get("regime_logits"),
            "h_seq": outputs[0].get("h_seq"),
            "recon": outputs[0].get("recon"),
            "prior_logits": outputs[0].get("prior_logits"),
            "post_logits": outputs[0].get("post_logits"),
        }

    def get_loss(self, obs_seq, asset_id, targets, **kwargs):
        """Inference-only: no loss (components are frozen)."""
        outputs = self.forward_train(obs_seq, asset_id)
        # Return near-zero loss to satisfy interface; training loop should skip
        zero = torch.zeros(1, device=obs_seq.device)
        return zero, {"loss_total": 0.0}, outputs


def build_v10(feature_count: int = 34, device: str = "cuda",
              weights_from_val_ic: Optional[Dict[str, float]] = None
              ) -> V10MetaEnsemble:
    """Factory for V10. Optionally set weights from validation IC dict."""
    m = V10MetaEnsemble(feature_count=feature_count, device=device)
    if weights_from_val_ic:
        m.set_weights(weights_from_val_ic)
    return m


if __name__ == "__main__":
    print("Building V10 meta-ensemble...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = build_v10(feature_count=34, device=device)
        print(f"V10 loaded {len(m.components)} components: {list(m.components.keys())}")
        print(f"Weights: {m.weights.tolist()}")
        total_params = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"Params: {total_params:,} total, {trainable:,} trainable")
    except Exception as e:
        print(f"V10 build FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
