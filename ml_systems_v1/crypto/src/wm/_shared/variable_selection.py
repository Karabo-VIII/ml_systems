"""src/wm/_shared/variable_selection.py -- shared, model-agnostic Variable Selection Network.

A per-timestep learnable feature gate (GRN-lite / VLSTM-class VSN) that any WM version can
reuse as a cheap input-side lever. It was first developed inline in V1.1
(src/wm/v1/v1_1_training/world_model.py, V1_VSN flag); generalized here 2026-06-10 so every
WM can import the SAME implementation instead of duplicating it per version.

Gate equation (causal, uses only x_t):
    g_t  = sigmoid(W_gate * x_t + b_gate)   -- [B, T, input_dim] in [0, 1]
    x'_t = g_t * x_t                        -- [B, T, input_dim]

Key properties:
  * CAUSAL: gate at t depends only on x_t -- no future timesteps leak in (point-wise linear).
  * PAST-ONLY: no look-ahead by construction.
  * Learnable: gate weights train end-to-end with whatever WM owns it.
  * Interpretable: get_weights(x) returns the [B, T, input_dim] gate scores so the operator
    can see which features are selected (mean over time/batch -> per-feature weight).
  * Cheap: one Linear(input_dim, input_dim) -- ~input_dim^2 + input_dim params.
  * Neutral start: bias=0 + tiny-weight init -> sigmoid(~0) = 0.5 (all features half-open),
    so it neither over-selects nor over-suppresses before training.

DEPENDENCY-LIGHT BY DESIGN: this module imports ONLY torch (no settings.py / components.py /
any version's symbols). That keeps it import-isolated so any WM version can adopt it without
coupling to another version's component registry. Construct it with an explicit input_dim.

A WM owner is responsible for the neutral-start init NOT being overwritten by a later
blanket _init_weights() pass: skip Linear modules whose qualified name contains "gate_proj"
(see V1.1 _init_weights for the canonical guard).

Self-test: python src/wm/_shared/variable_selection.py
  (causality at a perturbed timestep + neutral-start init + get_weights range + gate shape).
"""
from __future__ import annotations

import torch
import torch.nn as nn

__contract__ = {
    "kind": "model_lever",
    "version": "1.0",
    "inputs": ["x: [B,T,input_dim] raw features (before any asset embedding)"],
    "outputs": ["x_gated: [B,T,input_dim] element-wise sigmoid gate applied"],
    "invariants": [
        "CAUSAL -- gate at t uses only x_t; perturbing x_t' (t'!=t) cannot change gate[t]",
        "neutral start -- bias=0, weight std=0.01 -> initial gates ~0.5",
        "dependency-light -- imports only torch (no settings/components/version coupling)",
        "behaviour byte-for-byte equal to the original inline V1.1 VariableSelectionNetwork",
        "owning WM must skip 'gate_proj' in its _init_weights() to preserve neutral start",
    ],
}


class VariableSelectionNetwork(nn.Module):
    """Per-timestep learnable feature gate (GRN-lite / VLSTM-class VSN).

    Gate equation (causal, uses only x_t):
        g_t = sigmoid(W_gate * x_t + b_gate)   -- [B, T, input_dim] in [0, 1]
        x'_t = g_t * x_t                        -- [B, T, input_dim]

    Key properties:
    - CAUSAL: gate depends only on x_t, no future timesteps.
    - PAST-ONLY: no look-ahead by construction (point-wise linear).
    - Learnable: gate weights trained end-to-end with the world model.
    - Interpretable: get_weights(x) returns [B, T, input_dim] gate scores
      so the operator can see which features are selected.
    - Base-unchanged when OFF: module is not constructed when its flag is unset.

    The bias is initialized to 0 so sigmoid starts at 0.5 (all features
    half-open) -- a neutral initialization that neither over-selects nor
    over-suppresses at step 0.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        # Single linear layer: [input_dim] -> [input_dim] gate logits.
        # No hidden layer -- keeps the gate at ~100 extra params for f41
        # (well within the "cheap lever" budget).
        self.gate_proj = nn.Linear(input_dim, input_dim, bias=True)
        # Zero-init bias: sigmoid(0) = 0.5 -- neutral start.
        nn.init.zeros_(self.gate_proj.bias)
        # Small-weight init: gate starts near 0.5 everywhere before training.
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature gate.

        Args:
            x: [B, T, input_dim] -- raw features (before asset embedding)

        Returns:
            x_gated: [B, T, input_dim] -- element-wise gate applied
        """
        gate = torch.sigmoid(self.gate_proj(x))  # [B, T, input_dim]
        return gate * x

    def get_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return gate activations for inspection (no in-place changes).

        Args:
            x: [B, T, input_dim]

        Returns:
            gate: [B, T, input_dim] -- values in (0, 1); higher = feature selected
        """
        with torch.no_grad():
            return torch.sigmoid(self.gate_proj(x))


# ---------------------------------------------------------------------------
# Self-test -- causality, neutral-start init, get_weights range, gate shape.
# Does NOT require any WM version (torch-only).
# ---------------------------------------------------------------------------

def _selftest() -> int:
    torch.manual_seed(0)
    B, T, input_dim = 2, 10, 7
    vsn = VariableSelectionNetwork(input_dim)

    x = torch.randn(B, T, input_dim)

    # --- shape ---
    out = vsn(x)
    assert out.shape == (B, T, input_dim), f"forward shape {out.shape}"
    w = vsn.get_weights(x)
    assert w.shape == (B, T, input_dim), f"get_weights shape {w.shape}"

    # --- get_weights range (0,1) ---
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0, "gate must be in [0,1]"

    # --- neutral start: gates ~0.5 before any training ---
    # bias=0 and weight std=0.01 -> sigmoid(small) ~ 0.5; mean should be very close.
    assert abs(float(w.mean()) - 0.5) < 0.02, f"neutral-start mean {float(w.mean()):.4f} not ~0.5"

    # --- CAUSALITY: perturb t=5 only -> only the t=5 output column changes ---
    t_pert = 5
    x2 = x.clone()
    x2[:, t_pert, :] += 3.0  # large perturbation at a single timestep
    out2 = vsn(x2)
    diff = (out2 - out).abs().sum(dim=-1)  # [B, T] per-timestep total change
    changed = diff > 1e-6
    # the perturbed timestep MUST change; every other timestep MUST be identical.
    assert changed[:, t_pert].all(), "perturbed timestep t=5 did not change (gate not live)"
    other = torch.ones(T, dtype=torch.bool)
    other[t_pert] = False
    assert not changed[:, other].any(), \
        "CAUSALITY VIOLATION: a timestep other than t=5 changed when only t=5 was perturbed"

    print("[variable_selection] self-test PASSED")
    print(f"  gate shape [B,T,input_dim]={tuple(out.shape)} + get_weights range [0,1] OK")
    print(f"  neutral-start gates ~0.5 (mean={float(w.mean()):.4f}) OK")
    print(f"  CAUSAL: perturb t={t_pert} -> only t={t_pert} changes, all others byte-identical OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
