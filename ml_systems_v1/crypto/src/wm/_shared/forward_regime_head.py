"""src/wm/_shared/forward_regime_head.py -- OFF-by-default forward regime / move-onset heads.

This is the model-side companion to regime_targets.py. It provides:

  * ForwardRegimeHead  -- two small MLP heads read from the SAME fused feature the
    existing return/regime heads read (feat = cat(h_seq, z_post), dim = d_model+flat_dim):
        - bear_logits   : [B,T,2]  P(forward K-bar drawdown > thresh)   (binary)
        - trend_logits  : [B,T,3]  forward K-bar trend {down,neutral,up} (3-class)
        - move_logits   : [B,T,2]  net-of-cost up-move available in [a,b] (binary)
      Trained against the labels from regime_targets.py.

  * attach_forward_regime_head(model, ...) -- mirrors apply_headline_upgrades
    (frontier_ml/v1_upgrades/headline_integration.py): mutates the model IN-PLACE,
    sets model._use_forward_regime = True, attaches model.forward_regime_head. DEFAULT
    OFF -- if you never call attach_*, the base model is byte-for-byte unchanged.

  * forward_regime_aux_loss(outputs, labels, ...) -- the masked CE aux loss the trainer
    would ADD with a small fixed weight (NaN label rows masked out). It does NOT touch
    the base loss; it is additive and gated.

WIRING (single guarded line in world_model.forward_train; see DESIGN spec in the
worker's final report). The block is:

    if getattr(self, "_use_forward_regime", False) and self.forward_regime_head is not None:
        out["forward_regime"] = self.forward_regime_head(feat)

When _use_forward_regime is False (the default, since attach_* is never called by base
training), the block is a no-op and out["forward_regime"] is absent -- identical to today.

NO IC OBJECTIVE. These heads are validated on held-out COMPOUND via the regime-gate path
in src/strat/wm_entry_producer.py (a new mode that thresholds bear/trend probabilities
instead of rolling the per-bar h16 return), scored by src/strat/wm_value_probe.py. IC h=1
remains a within-WM diagnostic only.

Self-test: python src/wm/_shared/forward_regime_head.py  (shapes + OFF-by-default no-op +
masked-loss correctness; does NOT require the full V1.1 model).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__contract__ = {
    "kind": "model_head",
    "version": "1.0",
    "inputs": ["feat: [B,T,d_model+flat_dim] fused (h_seq, z_post)"],
    "outputs": [
        "bear_logits [B,T,2]", "trend_logits [B,T,3]", "move_logits [B,T,2]",
    ],
    "invariants": [
        "OFF by default -- base model unchanged unless attach_forward_regime_head() is called",
        "reads the SAME feat as the existing return/regime heads (no new input plumbing)",
        "aux loss is ADDITIVE + masks NaN label rows; base loss path untouched",
        "validated on held-out COMPOUND (wm_value_probe), NOT IC",
    ],
}


class _MiniHead(nn.Module):
    """Linear -> RMSNorm-free SiLU MLP -> Linear. Small (keeps param budget tiny).

    Uses LayerNorm (stdlib) instead of the project RMSNorm to stay import-isolated from
    any version's components.py -- this module must not couple to a specific WM version.
    """

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class ForwardRegimeHead(nn.Module):
    """Three forward heads (bear / trend / move-onset) on the fused WM feature."""

    def __init__(self, feat_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bear_head = _MiniHead(feat_dim, hidden, 2, dropout)    # binary fwd drawdown
        self.trend_head = _MiniHead(feat_dim, hidden, 3, dropout)   # 3-class fwd trend
        self.move_head = _MiniHead(feat_dim, hidden, 2, dropout)    # binary move-onset

    def forward(self, feat: torch.Tensor) -> dict:
        return {
            "bear_logits": self.bear_head(feat),
            "trend_logits": self.trend_head(feat),
            "move_logits": self.move_head(feat),
        }


def attach_forward_regime_head(model: nn.Module, *, hidden: int = 128,
                               dropout: float = 0.1, verbose: bool = True) -> dict:
    """Attach the forward-regime head to a TransformerWorldModel IN-PLACE.

    Mirrors apply_headline_upgrades. Default training never calls this, so the base model
    is unchanged. feat_dim is read dynamically from the model (d_model + flat_dim).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    feat_dim = int(model.d_model + getattr(model, "flat_dim", 0))
    model.forward_regime_head = ForwardRegimeHead(feat_dim, hidden, dropout).to(
        device=device, dtype=dtype)
    model._use_forward_regime = True
    n_params = sum(p.numel() for p in model.forward_regime_head.parameters())
    if verbose:
        print(f"  [forward_regime] ATTACHED feat_dim={feat_dim} params+={n_params:,} "
              f"(bear/trend/move heads) -- OFF by default, now ON for this model instance")
    return {"params": n_params, "feat_dim": feat_dim}


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over rows where label is finite (NaN = no future window -> masked).

    logits: [..., C]; labels: [...] float with NaN for invalid rows. Returns scalar
    (mean over valid rows) or 0.0 if no valid rows.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = torch.isfinite(flat_labels)
    if valid.sum() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(flat_logits[valid], flat_labels[valid].long())


def forward_regime_aux_loss(outputs: dict, labels: dict, *,
                            w_bear: float = 1.0, w_trend: float = 1.0,
                            w_move: float = 1.0) -> torch.Tensor:
    """Additive masked-CE aux loss for the forward-regime head.

    Called by the trainer ONLY when model._use_forward_regime is True. Returns a scalar.
    labels = {"bear": [B,T], "trend": [B,T], "move": [B,T]} (any subset); NaN rows masked.
    The trainer adds `weight * forward_regime_aux_loss(...)` to total with a small weight,
    exactly like the CC-H5/H6 aux losses (world_model.get_loss lines 536-552).
    """
    fr = outputs.get("forward_regime")
    if fr is None:
        # head not attached / not in this forward -> no contribution
        dev = next(iter(outputs.values())).device if outputs else torch.device("cpu")
        return torch.tensor(0.0, device=dev)
    loss = fr["bear_logits"].new_tensor(0.0)
    if "bear" in labels:
        loss = loss + w_bear * _masked_ce(fr["bear_logits"], labels["bear"])
    if "trend" in labels:
        loss = loss + w_trend * _masked_ce(fr["trend_logits"], labels["trend"])
    if "move" in labels:
        loss = loss + w_move * _masked_ce(fr["move_logits"], labels["move"])
    return loss


# ---------------------------------------------------------------------------
# Self-test -- shapes, OFF-by-default no-op, masked-loss correctness.
# Does NOT need the full V1.1 model (uses a tiny mock with d_model/flat_dim).
# ---------------------------------------------------------------------------

def _selftest() -> int:
    torch.manual_seed(0)
    B, T, d_model, flat_dim = 2, 8, 16, 24
    feat_dim = d_model + flat_dim

    # --- head shapes ---
    head = ForwardRegimeHead(feat_dim, hidden=32)
    feat = torch.randn(B, T, feat_dim)
    out = head(feat)
    assert out["bear_logits"].shape == (B, T, 2), "bear_logits shape"
    assert out["trend_logits"].shape == (B, T, 3), "trend_logits shape"
    assert out["move_logits"].shape == (B, T, 2), "move_logits shape"

    # --- attach mutates a mock model in place + OFF-by-default ---
    class _Mock(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.flat_dim = flat_dim
            self.lin = nn.Linear(feat_dim, feat_dim)   # gives the module a parameter/device
            self._use_forward_regime = False
            self.forward_regime_head = None

    m = _Mock()
    # OFF: the guarded block (getattr default False) would skip -> no head present
    assert getattr(m, "_use_forward_regime", False) is False, "default must be OFF"
    assert m.forward_regime_head is None, "no head before attach"
    info = attach_forward_regime_head(m, hidden=32, verbose=False)
    assert m._use_forward_regime is True and m.forward_regime_head is not None, "attach failed"
    assert info["feat_dim"] == feat_dim and info["params"] > 0, "attach info wrong"

    # --- aux loss: NaN masking + no-head no-op ---
    # No "forward_regime" key in outputs -> 0.0 (off path)
    z = forward_regime_aux_loss({"x": torch.zeros(1)}, {"bear": torch.zeros(B, T)})
    assert float(z) == 0.0, "aux loss must be 0 when head absent"

    # With head outputs + labels containing NaN rows
    fr_out = {"forward_regime": head(feat)}
    bear_lab = torch.zeros(B, T)
    bear_lab[:, -3:] = float("nan")            # last 3 = no future -> masked
    trend_lab = torch.ones(B, T)
    trend_lab[:, -3:] = float("nan")
    loss = forward_regime_aux_loss(fr_out, {"bear": bear_lab, "trend": trend_lab})
    assert torch.isfinite(loss) and float(loss) > 0, "masked aux loss should be finite positive"

    # all-NaN labels -> 0 contribution (no valid rows)
    all_nan = torch.full((B, T), float("nan"))
    loss0 = forward_regime_aux_loss(fr_out, {"bear": all_nan})
    assert float(loss0) == 0.0, "all-NaN label must contribute 0"

    # gradient flows through valid rows only (sanity)
    loss.backward()
    assert head.bear_head.net[0].weight.grad is not None, "no grad through head"

    print("[forward_regime_head] self-test PASSED")
    print(f"  head shapes (bear[B,T,2] trend[B,T,3] move[B,T,2]) OK")
    print(f"  attach in-place + OFF-by-default (params+={info['params']:,}) OK")
    print(f"  masked aux loss (NaN rows masked, no-head no-op, all-NaN=0, grad flows) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
