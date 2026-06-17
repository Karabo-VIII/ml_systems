"""forecast_bundle.py -- the FROZEN F->A1 contract (the single most important
boundary in the agent-layer taxonomy).

A `ForecastBundle` is the ONLY thing an A1 (WM-consuming agent) may import from a
forecaster F. It is FROZEN and DETACHED by construction:
  - every tensor is `.detach()`'d (no gradient path back into F);
  - the source forecaster is put in `eval()` before the forward pass;
  - `is_frozen` is asserted True at A1 train-time.

The three channels and their STRICT roles (doc SS1.3):
  1. `feat` = cat([h_seq, z_post])  -> A1's OBSERVATION (the Dreamer/MuZero state).
  2. `return_logits[h]`             -> A1's BELIEF INPUT (TwoHot DISTRIBUTIONS,
        decode for an uncertainty feature). It is NEVER the realized reward -- the
        critic regresses to REALIZED return (target_return_*). This is the GIGO
        firewall (CDAP no_predicted_return_as_realized_reward).
  3. `regime_logits`                -> A1's CONDITIONING / GATE (cheapest, slowest,
        most robust channel).

`genuine_learning` carries F's provenance (shic / held_out_ic / passed) so an A1
build can mechanically refuse to plan over a forecaster that has not PASSED Gate A.

No emoji (Windows cp1252). ASCII only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__contract__ = {
    "kind": "agent_layer_contract",
    "inputs": [
        "a forecaster F exposing forward_train(obs_seq, asset_id, ...) -> dict",
        "a batch dict with obs_seq[B,T,C] (+ asset_id, optional masked_obs_seq)",
    ],
    "outputs": [
        "ForecastBundle(feat, h_seq, z_post, return_logits, return_bins, "
        "regime_logits, forecaster_id, genuine_learning, is_frozen)",
    ],
    "invariants": [
        "FROZEN: source forecaster put in eval() before the forward pass",
        "DETACHED: every tensor field is .detach()'d (no grad path into F)",
        "is_frozen is True after from_forecaster()",
        "return_logits are DISTRIBUTIONS (belief input), never the realized reward",
        "frozen=True dataclass -- a bundle is immutable once built",
    ],
}


@dataclass(frozen=True)
class ForecastBundle:
    """The frozen, detached, versioned F->A1 contract. Immutable (frozen=True)."""

    # --- LATENT (the 'dynamics' channel -> A1's observation / state) ---
    feat: Any                     # [B,T,d_model+flat] = cat([h_seq, z_post])
    h_seq: Any                    # [B,T,d_model]
    z_post: Any                   # [B,T,flat]

    # --- BELIEF (DISTRIBUTIONS, not points -> A1's belief feature, NEVER reward) ---
    return_logits: Dict[int, Any] # {h: [B,T,NUM_BINS]} TwoHot logits per horizon
    return_bins: Any              # [NUM_BINS] bin centers for decode

    # --- REGIME (the cheap, robust gate channel) ---
    regime_logits: Any            # [B,T,3] bear/range/trend

    # --- PROVENANCE (the GIGO firewall, machine-checked) ---
    forecaster_id: str
    genuine_learning: Dict[str, Any] = field(default_factory=dict)
    is_frozen: bool = False       # asserted True at A1 train-time

    @classmethod
    def from_forecaster(
        cls,
        model: Any,
        batch: Dict[str, Any],
        forecaster_id: str = "",
        genuine_learning: Dict[str, Any] | None = None,
        return_bins: Any = None,
    ) -> "ForecastBundle":
        """Build a FROZEN, DETACHED bundle from a forecaster + a batch.

        Puts `model` in eval(), runs forward_train under no_grad, and .detach()s
        EVERY tensor so no gradient can flow back into F. Sets is_frozen=True.

        Args:
            model: a forecaster exposing forward_train(obs_seq, asset_id, [masked_obs_seq]).
            batch: dict with at least 'obs_seq' [B,T,C]; optional 'asset_id' [B],
                   'masked_obs_seq' [B,T,C].
            forecaster_id: provenance string (e.g. "v1_1_f41_seed0").
            genuine_learning: {"shic":..., "held_out_ic":..., "passed": bool}.
            return_bins: [NUM_BINS] bin centers; if None, left as None for the
                         caller to attach (decode is the A1's responsibility).
        """
        import torch  # local import -- the contract stays importable without torch

        genuine_learning = dict(genuine_learning or {})

        def _detach(x):
            if isinstance(x, torch.Tensor):
                return x.detach()
            if isinstance(x, dict):
                return {k: _detach(v) for k, v in x.items()}
            return x

        # FREEZE: eval mode so dropout/batchnorm are deterministic + no train-time
        # stochasticity leaks into the bundle.
        model.eval()

        obs_seq = batch["obs_seq"]
        asset_id = batch.get("asset_id")
        masked_obs_seq = batch.get("masked_obs_seq")

        with torch.no_grad():
            if masked_obs_seq is not None:
                out = model.forward_train(obs_seq, asset_id, masked_obs_seq)
            else:
                out = model.forward_train(obs_seq, asset_id)

        h_seq = _detach(out["h_seq"])
        z_post = _detach(out["z_post"])
        # feat = cat([h_seq, z_post]) -- reconstruct the A1 observation (doc SS1.3).
        feat = _detach(torch.cat([out["h_seq"], out["z_post"]], dim=-1))
        return_logits = _detach(out["return_logits"])
        regime_logits = _detach(out["regime_logits"])
        return_bins = _detach(return_bins) if return_bins is not None else None

        return cls(
            feat=feat,
            h_seq=h_seq,
            z_post=z_post,
            return_logits=return_logits,
            return_bins=return_bins,
            regime_logits=regime_logits,
            forecaster_id=forecaster_id,
            genuine_learning=genuine_learning,
            is_frozen=True,
        )


# ---------------------------------------------------------------------------
# Self-test: a tiny synthetic forecaster proves the freeze+detach contract
# end to end without the real V1.1 or any data on disk.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import torch
    import torch.nn as nn

    NUM_BINS = 255
    REWARD_HORIZONS = [1, 4, 16, 64]

    class _TinyForecaster(nn.Module):
        """Minimal forward_train mirroring V1.1's emit surface."""

        def __init__(self, c=8, d_model=16, flat=12):
            super().__init__()
            self.enc = nn.Linear(c, d_model)
            self.post = nn.Linear(d_model, flat)
            self.regime = nn.Linear(d_model + flat, 3)
            self.ret = nn.ModuleDict(
                {str(h): nn.Linear(d_model + flat, NUM_BINS) for h in REWARD_HORIZONS}
            )

        def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
            h_seq = torch.tanh(self.enc(obs_seq))            # [B,T,d_model]
            z_post = torch.tanh(self.post(h_seq))            # [B,T,flat]
            feat = torch.cat([h_seq, z_post], dim=-1)
            return {
                "h_seq": h_seq,
                "z_post": z_post,
                "regime_logits": self.regime(feat),
                "return_logits": {h: self.ret[str(h)](feat) for h in REWARD_HORIZONS},
            }

    torch.manual_seed(0)
    B, T, C = 4, 16, 8
    model = _TinyForecaster(c=C)
    batch = {"obs_seq": torch.randn(B, T, C, requires_grad=True),
             "asset_id": torch.zeros(B, dtype=torch.long)}
    bins = torch.linspace(-1.0, 1.0, NUM_BINS)

    bundle = ForecastBundle.from_forecaster(
        model, batch, forecaster_id="tiny_selftest",
        genuine_learning={"shic": 0.02, "held_out_ic": 0.03, "passed": True},
        return_bins=bins,
    )

    # 1. is_frozen True
    assert bundle.is_frozen is True, "is_frozen must be True"

    # 2. EVERY tensor detached (no grad, no grad_fn)
    def _assert_detached(x, name):
        if isinstance(x, torch.Tensor):
            assert not x.requires_grad, f"{name} still requires_grad"
            assert x.grad_fn is None, f"{name} still has grad_fn (not detached)"
        elif isinstance(x, dict):
            for k, v in x.items():
                _assert_detached(v, f"{name}[{k}]")

    for fname in ("feat", "h_seq", "z_post", "return_logits", "return_bins", "regime_logits"):
        _assert_detached(getattr(bundle, fname), fname)

    # 3. feat == cat([h_seq, z_post])
    assert torch.allclose(bundle.feat, torch.cat([bundle.h_seq, bundle.z_post], dim=-1)), \
        "feat must equal cat([h_seq, z_post])"

    # 4. shapes sane
    assert bundle.feat.shape[:2] == (B, T)
    assert set(bundle.return_logits.keys()) == set(REWARD_HORIZONS)
    assert bundle.regime_logits.shape == (B, T, 3)

    # 5. immutability (frozen dataclass)
    try:
        bundle.is_frozen = False  # type: ignore
        raise AssertionError("ForecastBundle must be immutable (frozen dataclass)")
    except Exception as e:
        assert "cannot assign" in str(e) or "FrozenInstanceError" in type(e).__name__, str(e)

    # 6. model left in eval mode (frozen-for-inference)
    assert model.training is False, "forecaster must be left in eval() mode"

    print("[forecast_bundle] self-test PASS: is_frozen=True, all tensors detached, "
          "feat=cat([h,z]), bundle immutable, forecaster in eval().")


if __name__ == "__main__":
    _selftest()
