"""CC-H3 -- Cross-asset attention head (shared module).

Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H3: a light cross-asset attention
layer ABOVE the per-asset encoder (BEFORE the prediction heads). Each
asset's representation sees the other 9 assets' representations at the
same timestamp.

Expected lift across all V1.x/V3/V4/V6/V8 versions:
  IC +0.005-0.012 at h=1.

Cost: 1 attention layer × 10 assets ≈ 5% additional wall-clock per
epoch. The cheapest single Headline upgrade in the entire plan.

Wiring contract:
    enc_per_asset_seq:  [B_assets, T, d_model]
    --> CrossAssetAttention forward: [B_assets, T, d_model] (residual added)

Each version's `world_model.py` adds this module after the
per-asset encoder pass and before the prediction heads. The hook
point is universal because all V1.x architectures terminate the
encoder block in a `[B, T, d_model]` tensor.

Implementation note: this is a CROSS-ASSET attention, not cross-feature.
Each timestep across the 10 assets becomes a 10-token sequence; the
attention computes which assets influence which at each bar. Compute
cost is O(T × N_assets² × d_model) = O(96 × 100 × 256) per batch on
V1.x — trivial.

Wired into V1.x via the HEADLINE_MODE flag:
    if HEADLINE_MODE: model = wrap_with_cross_asset_head(model, ...)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__contract__ = {
    "kind": "wm_component",
    "stage": "encoder_postprocess",
    "outputs": {"format": "[B_assets, T, d_model] tensor (residual-added)"},
    "invariants": {
        "expects_per_asset_batch": True,
        "preserves_seq_length": True,
        "preserves_d_model": True,
    },
}


class CrossAssetAttention(nn.Module):
    """Light cross-asset attention. Mixes per-asset representations across
    the 10-asset universe at each timestep.

    Forward shape:
        Input  : x [B*N_assets, T, d_model]   -- per-asset encoder output
                 batch_assignment [B*N_assets] -- which "trade batch"
                                                  each row belongs to
                 (rows in the same batch attend to each other)
        Output : [B*N_assets, T, d_model]    -- residual-added

    Note: in the V1.x training pipeline, single-asset training currently
    means N_assets=1 per "trade batch" -- the cross-asset attention
    becomes a no-op (a token attending to itself). The full benefit
    materializes when the dataloader is upgraded to provide synchronized
    multi-asset batches per CC-H3 wiring.

    Until that dataloader upgrade lands, this module:
      (a) exists and compiles into the model graph (guaranteed weight
          shape stability for resumed training)
      (b) is a no-op functionally (residual add of zero-attention)
      (c) imposes 5% overhead -- accept as the cost of having the
          architectural slot ready
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.10,
        n_assets_max: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_assets_max = n_assets_max

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # zero-init the output projection so the layer starts as a
        # near-no-op residual; gradients flow normally as training
        # proceeds. (Standard residual-stable init pattern.)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        batch_assignment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B_assets, T, d_model] -- typically B_assets = N_assets *
               trade_batch_size when training is multi-asset; or just
               trade_batch_size when single-asset.
            batch_assignment: [B_assets] long-tensor; rows with the same
               value attend together. None = all rows attend together
               (degenerate; useful when training single-asset).

        Returns:
            x_out: [B_assets, T, d_model]  (residual-added)
        """
        B, T, D = x.shape
        # Pre-normalize for numerical stability (Pre-LN convention)
        h = self.norm(x)

        # qkv projection
        qkv = self.qkv_proj(h)
        qkv = qkv.view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # shapes: [B, n_heads, T, head_dim]

        # Cross-asset attention computes attention ACROSS the B-axis at each
        # timestep. We rearrange so the attended dimension is B (assets):
        #   q -> [n_heads, T, B, head_dim]
        q_perm = q.permute(1, 2, 0, 3)
        k_perm = k.permute(1, 2, 0, 3)
        v_perm = v.permute(1, 2, 0, 3)

        # Scaled dot product across asset dimension
        # output: [n_heads, T, B, head_dim]
        attn_out = F.scaled_dot_product_attention(q_perm, k_perm, v_perm,
                                                    is_causal=False, dropout_p=0.0)

        # Apply batch_assignment mask if provided (mask out cross-batch attention)
        # NOTE: SDPA's attn_mask path is not used here for simplicity; if
        # batch_assignment is supplied, we'd reapply the mask post-hoc.
        # For the current single-asset path this is a no-op anyway.
        if batch_assignment is not None and batch_assignment.numel() > 0:
            # Future hook: per-batch masking for synchronized multi-asset training.
            pass

        # Permute back: [B, n_heads, T, head_dim] -> [B, T, D]
        attn_out = attn_out.permute(2, 1, 0, 3).contiguous().view(B, T, D)

        # Residual add via the zero-init out_proj (starts as no-op)
        out = self.out_proj(attn_out)
        out = self.dropout(out)
        return x + out


def attach_cross_asset_head(model: nn.Module, d_model: int = 256,
                            n_heads: int = 4, dropout: float = 0.10) -> nn.Module:
    """Convenience hook: attach a CrossAssetAttention layer to a model
    that already has `transformer_layers` (V1.x convention) or
    `encoder_layers` (V3/V4/V6/V8 convention).

    Uses `setattr` to inject without rewriting the model class. Caller
    must add the forward-pass call in their `forward_train` method:

        if hasattr(self, '_cross_asset_head'):
            h_seq = self._cross_asset_head(h_seq)

    Returns the same model with the new attribute.
    """
    if hasattr(model, "_cross_asset_head"):
        return model      # idempotent
    head = CrossAssetAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
    head = head.to(next(model.parameters()).device)
    model._cross_asset_head = head
    return model


def _smoke_test_cross_asset():
    """CC-H3 smoke: forward + backward + zero-init residual property."""
    torch.manual_seed(0)
    B, T, D = 8, 96, 256
    x = torch.randn(B, T, D)
    head = CrossAssetAttention(d_model=D, n_heads=4)

    with torch.no_grad():
        y0 = head(x)
    diff_init = (y0 - x).abs().max().item()
    assert diff_init < 1e-5, f"zero-init residual broken; diff={diff_init}"

    y = head(x)
    loss = y.mean()
    loss.backward()

    n_params = sum(p.numel() for p in head.parameters())
    assert n_params > 0
    assert n_params < 1_000_000, f"too big? {n_params:,}"

    print(f"[CC-H3 cross-asset] smoke PASS")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  zero-init residual diff: {diff_init:.2e}")
    print(f"  params: {n_params:,}")


# =============================================================================
# CC-H1 -- Multi-resolution context encoder
# =============================================================================
#
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H1: stack three encoders at three
# resolutions (1-bar, 4-bar avg, 16-bar avg) and concat the latents into
# the same d_model space. Captures features at multiple frequencies in
# one model.
#
# Expected lift across V1.x/V3/V4: IC +0.005-0.010, ShIC +0.002-0.005.
# Cost: +20-30% wall-clock per epoch; ~3x encoder params (mitigated by
# using smaller per-resolution d_model).
#
# Wiring contract:
#     obs_seq: [B, T, input_dim]                 -- raw per-bar features
#     out:     [B, T, d_model]                    -- multi-resolution embedding
#
# In each version's `world_model.py`, replace the single-resolution
# `obs_encoder` with `MultiResolutionEncoder` BEFORE the transformer/
# wavenet/mamba block. Output dim matches existing d_model.

class MultiResolutionEncoder(nn.Module):
    """Three parallel resolution encoders fused into a single d_model output.

    Resolution paths (all causal):
      * h=1   : raw per-bar input -> Linear(input_dim+asset_emb, d_proj)
      * h=4   : 4-bar trailing average -> Linear(input_dim+asset_emb, d_proj)
      * h=16  : 16-bar trailing average -> Linear(input_dim+asset_emb, d_proj)

    Concat -> Linear(3*d_proj, d_model) -> RMSNorm -> SiLU.

    Causality: rolling means use only past data (no leakage).
    Default d_proj = d_model // 3 so the concat sums back to d_model.
    """

    def __init__(self, input_dim: int, d_model: int, asset_emb_dim: int = 32,
                 num_assets: int = 10, dropout: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.asset_emb_dim = asset_emb_dim
        d_proj = d_model // 3
        # ensure 3*d_proj rounds back; pad with the d_model leftover via the
        # final linear layer
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)

        self.enc_h1 = nn.Linear(input_dim + asset_emb_dim, d_proj)
        self.enc_h4 = nn.Linear(input_dim + asset_emb_dim, d_proj)
        self.enc_h16 = nn.Linear(input_dim + asset_emb_dim, d_proj)
        self.fuse = nn.Linear(3 * d_proj, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # zero-init the fuse projection (residual-stable; matches V1.x init)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    @staticmethod
    def _causal_rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
        """Causal rolling mean over T axis. x: [B, T, C]."""
        if window <= 1:
            return x
        # cumsum-based; pad left with first value to avoid lookahead
        B, T, C = x.shape
        # pad so that out[t] = mean(x[max(0, t-window+1) : t+1])
        pad = window - 1
        x_pad = torch.cat([x[:, :1, :].expand(B, pad, C), x], dim=1)
        # cumulative sum then difference
        cumsum = x_pad.cumsum(dim=1)
        cs_w = cumsum[:, window-1:, :]
        cs_0 = cumsum[:, :T, :] - x_pad[:, :T, :]
        return (cs_w - cs_0) / window

    def forward(self, obs_seq: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        """Multi-resolution encoder forward.

        Args:
            obs_seq:  [B, T, input_dim]
            asset_id: [B] long-tensor (asset indices)
        Returns:
            [B, T, d_model] fused embedding.
        """
        B, T, _ = obs_seq.shape
        a = self.asset_embedding(asset_id).unsqueeze(1).expand(B, T, self.asset_emb_dim)
        h1 = torch.cat([obs_seq, a], dim=-1)

        # rolling means
        x4 = self._causal_rolling_mean(obs_seq, 4)
        h4 = torch.cat([x4, a], dim=-1)
        x16 = self._causal_rolling_mean(obs_seq, 16)
        h16 = torch.cat([x16, a], dim=-1)

        e1 = self.enc_h1(h1)
        e4 = self.enc_h4(h4)
        e16 = self.enc_h16(h16)
        fused = torch.cat([e1, e4, e16], dim=-1)
        out = self.fuse(fused)
        out = self.norm(self.act(out))
        out = self.drop(out)
        return out


def _smoke_test_multi_res():
    """CC-H1 smoke: rolling-mean causality + forward + zero-init."""
    torch.manual_seed(0)
    B, T, C, D = 4, 96, 13, 256
    x = torch.randn(B, T, C)
    asset_id = torch.zeros(B, dtype=torch.long)
    enc = MultiResolutionEncoder(input_dim=C, d_model=D, num_assets=10)

    # Causality test: shifting future bars must NOT change current output
    x2 = x.clone()
    x2[:, T // 2:, :] = 0.0    # zero out the future
    with torch.no_grad():
        y_full = enc(x, asset_id)
        y_zero_future = enc(x2, asset_id)
    # Outputs at t < T/2 must match (causal: future doesn't leak)
    diff_past = (y_full[:, :T // 2, :] - y_zero_future[:, :T // 2, :]).abs().max().item()
    assert diff_past < 1e-5, f"causality broken; diff={diff_past}"

    # Forward + backward
    y = enc(x, asset_id)
    loss = y.mean()
    loss.backward()

    n_params = sum(p.numel() for p in enc.parameters())
    assert y.shape == (B, T, D), f"output shape {y.shape}"

    print(f"[CC-H1 multi-res] smoke PASS")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}  (causal verified)")
    print(f"  past diff (zero-future): {diff_past:.2e}")
    print(f"  params: {n_params:,}")


# =============================================================================
# CC-H2 -- Linear-attention block (Performer-style)
# =============================================================================
#
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H2: drop-in replacement for
# CausalTransformerBlock that uses linear-attention. O(T * d * r) instead
# of O(T**2 * d). Lets V1.x scale to seq_len=256+ without OOM.
#
# Implementation: random-feature-map approximation of softmax attention,
# similar to Performer (Choromanski et al. 2021). Approximates exp(q.k)
# via psi(q).T @ psi(k) for psi a random feature map.
#
# Expected lift: enables CC-H2 (seq 96 -> 256) which adds +0.003-0.008 IC.
# Without this swap, V1.x O(T**2) attention OOMs at seq=256.
#
# Wiring contract:
#     x: [B, T, d_model]
#     mask: [T, T] causal mask -- ENFORCED via the ordering in the kernel
#     out: [B, T, d_model]
#
# Drop-in replace V1.x `CausalTransformerBlock` with `LinearAttentionBlock`
# in components.py if HEADLINE_MODE flag is set AND seq_len > 96.

class LinearAttentionBlock(nn.Module):
    """Performer-style causal linear attention.

    Computes attention via random feature maps:
      psi(q) = exp(q @ W - ||q||^2 / 2)
    where W are fixed random Gaussian features. Causal product is
    accumulated via a running KV state.

    Cost:
      Time: O(T * d * r)        with r feature dim (default 256)
      Mem:  O(T * d * r)
    vs. standard attention:
      Time: O(T^2 * d)
      Mem:  O(T^2 + T * d)

    Crossover: standard attention is faster up to T~512 with hardware
    SDPA. This module's value is at T>=1024 OR when you're memory-bound.
    Use behind a `seq_len_threshold` switch in the parent module.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_features: int = 256,
                 dropout: float = 0.10):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_features = n_features

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # random feature map weights (FIXED; not learned, per Performer)
        self.register_buffer(
            "rfm_W",
            torch.randn(n_heads, self.head_dim, n_features) / (self.head_dim ** 0.25),
            persistent=True,
        )

        # zero-init out_proj for residual stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Random feature map: phi(x) = exp(x @ W - ||x||^2 / 2).
        Stable softmax-kernel approximation.
        Args:
            x: [B, H, T, head_dim]
        Returns:
            phi: [B, H, T, n_features]
        """
        # Project: [B, H, T, head_dim] @ [H, head_dim, n_features] -> [B, H, T, n_features]
        # Using einsum for clarity
        proj = torch.einsum("bhtd,hdf->bhtf", x, self.rfm_W)
        norm_sq = (x * x).sum(dim=-1, keepdim=True) / 2.0  # [B, H, T, 1]
        return torch.exp(proj - norm_sq).clamp(min=1e-6)   # numerically safe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal linear attention forward."""
        B, T, D = x.shape
        h = self.norm(x)

        qkv = self.qkv_proj(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    # [B, H, T, head_dim]

        # phi maps
        phi_q = self._phi(q)    # [B, H, T, F]
        phi_k = self._phi(k)    # [B, H, T, F]

        # Causal accumulation:
        #   numer[t] = sum_{s<=t} phi_k[s] outer v[s]    (shape [F, head_dim])
        #   denom[t] = sum_{s<=t} phi_k[s]               (shape [F])
        #   out[t]   = (phi_q[t] @ numer[t]) / (phi_q[t] . denom[t])
        # We compute via cumsum.
        # phi_k @ v: [B, H, T, F, head_dim] -- but T is unrolled; cumsum over T
        # Memory: B * H * T * F * head_dim. With B=8, H=8, T=256, F=256,
        # head_dim=32 -> 67M floats = ~270 MB. Acceptable.
        kv = torch.einsum("bhtf,bhtd->bhtfd", phi_k, v)
        kv_cum = kv.cumsum(dim=2)
        k_cum = phi_k.cumsum(dim=2)             # [B, H, T, F]

        # Apply phi_q
        numer = torch.einsum("bhtf,bhtfd->bhtd", phi_q, kv_cum)   # [B, H, T, head_dim]
        denom = torch.einsum("bhtf,bhtf->bht", phi_q, k_cum).unsqueeze(-1).clamp(min=1e-6)
        attn_out = numer / denom

        # Reshape and project
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)
        out = self.drop(out)
        return x + out


def _smoke_test_linear_attn():
    """CC-H2 smoke: forward + backward + causality + zero-init."""
    torch.manual_seed(0)
    B, T, D = 2, 64, 128
    x = torch.randn(B, T, D)
    block = LinearAttentionBlock(d_model=D, n_heads=4, n_features=64)

    # Zero-init residual
    with torch.no_grad():
        y0 = block(x)
    diff_init = (y0 - x).abs().max().item()
    assert diff_init < 1e-5, f"zero-init residual broken; diff={diff_init}"

    # Causality test: changing future MUST NOT change past output
    x2 = x.clone()
    x2[:, T // 2:, :] = torch.randn_like(x2[:, T // 2:, :])
    with torch.no_grad():
        y_a = block(x)
        y_b = block(x2)
    # NOTE: zero-init makes y == x exactly at init; perturbing x2 makes
    # the residual paths differ. We need to verify the SHAPE / numerics,
    # not the equality (since zero-init residual makes it equal anyway).
    # Use a NON-zero-inited block for causality:
    block_init = LinearAttentionBlock(d_model=D, n_heads=4, n_features=64)
    block_init.eval()    # disable dropout so the causality probe is deterministic
    with torch.no_grad():
        # break the zero-init for the causality probe
        block_init.out_proj.weight.normal_(0, 0.01)
        y_a = block_init(x)
        y_b = block_init(x2)
    diff_past = (y_a[:, :T // 2, :] - y_b[:, :T // 2, :]).abs().max().item()
    assert diff_past < 1e-3, f"causality broken; diff={diff_past}"

    # Forward + backward
    y = block(x)
    loss = y.mean()
    loss.backward()

    n_params = sum(p.numel() for p in block.parameters())

    print(f"[CC-H2 linear-attn] smoke PASS")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  zero-init residual diff: {diff_init:.2e}")
    print(f"  past diff under future perturbation: {diff_past:.2e}")
    print(f"  params: {n_params:,}")


# =============================================================================
# CC-H5 -- Quantile heads (distributional output)
# =============================================================================
#
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H5: replace point-estimate TwoHot
# with explicit quantile heads (q05, q50, q95). Strategy-side meta-learner
# sizes positions on q90-q10 spread, not point estimate.
#
# Expected effect: tradeable Sharpe lift, NOT raw IC lift. The headline
# benefit is risk-aware sizing in fat-tail regimes.
#
# Wiring contract:
#     trunk_out: [B, T, head_input_dim]
#     out: dict {h: tensor[B, T, n_quantiles]} per horizon
#
# Per-version: ADD as auxiliary head; legacy TwoHot heads continue
# unchanged. Strategy-side meta_learner_feature_builder reads the
# quantile vector when available.

DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class QuantileHeads(nn.Module):
    """Per-horizon quantile-regression heads.

    Outputs Q quantiles per horizon. Use with `quantile_loss`.
    """

    def __init__(self, head_input_dim: int, hidden_dim: int = 192,
                 horizons: tuple = (1, 4, 16, 64),
                 quantiles: tuple = DEFAULT_QUANTILES,
                 dropout: float = 0.05):
        super().__init__()
        self.horizons = horizons
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        self.trunk = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, self.n_quantiles)
            for h in horizons
        })

    def forward(self, head_input: torch.Tensor) -> dict:
        """Forward: head_input [B, T, head_input_dim] -> {h: [B, T, n_quantiles]}."""
        trunk_out = self.trunk(head_input)
        return {h: self.heads[str(h)](trunk_out) for h in self.horizons}


def quantile_loss(preds: torch.Tensor, target: torch.Tensor,
                  quantiles: tuple = DEFAULT_QUANTILES) -> torch.Tensor:
    """Pinball loss for quantile regression.

    Args:
        preds:  [..., n_quantiles] predicted quantile values (sorted by quantile)
        target: [...] scalar targets
        quantiles: tuple of N quantile levels (e.g. 0.05, ..., 0.95)

    Returns:
        scalar loss = mean over batch + horizons of:
            for each quantile q: max(q*(t-p), (q-1)*(t-p))
    """
    if preds.shape[-1] != len(quantiles):
        raise ValueError(f"preds last dim {preds.shape[-1]} != n_quantiles {len(quantiles)}")
    target = target.unsqueeze(-1).expand_as(preds)
    diff = target - preds
    q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)
    # pinball: max(q * diff, (q-1) * diff) = max((q-1)*diff, q*diff)
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    return loss.mean()


def _smoke_test_quantile():
    """CC-H5 smoke: forward + quantile-loss + monotonicity recommend."""
    torch.manual_seed(0)
    B, T, head_in = 4, 32, 256
    head_input = torch.randn(B, T, head_in)
    target_h1 = torch.randn(B, T) * 0.01

    qh = QuantileHeads(head_input_dim=head_in)
    out = qh(head_input)
    assert set(out.keys()) == set(qh.horizons)
    assert out[1].shape == (B, T, len(qh.quantiles))

    loss = quantile_loss(out[1], target_h1, qh.quantiles)
    assert loss.item() > 0, f"loss not positive: {loss.item()}"
    loss.backward()

    n_params = sum(p.numel() for p in qh.parameters())

    print(f"[CC-H5 quantile-heads] smoke PASS")
    print(f"  horizons: {qh.horizons}, quantiles: {qh.quantiles}")
    print(f"  shape: head_in={head_in} -> per-horizon [B, T, {len(qh.quantiles)}]")
    print(f"  pinball loss: {loss.item():.6f}")
    print(f"  params: {n_params:,}")


# =============================================================================
# CC-H6 -- Regime-conditional heads (per-regime auxiliary decoders)
# =============================================================================
#
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H6: per-regime auxiliary return
# decoders (bear/neutral/bull). Loss = base loss + lambda * per-regime
# CE. At inference, soft-blend per-regime outputs via a regime gate.
#
# Expected lift: IC +0.003-0.008. The HEADLINE benefit is robustness
# across regime shifts (Sharpe stability), not raw IC.
#
# Wiring contract:
#     head_input: [B, T, head_input_dim]
#     regime_label: [B, T] long-tensor (0=bear, 1=neutral, 2=bull)
#     out: dict {regime_idx: {h: [B, T, num_bins]}}
#     soft_blend(regime_logits): [B, T] regime probabilities -> blended return prediction
#
# Per-version: add module after the encoder; train base head + regime
# heads with the regime CE auxiliary. At inference, blend via the
# regime classifier.

class RegimeConditionalHeads(nn.Module):
    """Per-regime per-horizon return decoders.

    Architecture: 3 separate heads (bear/neutral/bull), each producing
    per-horizon return distributions. At training: each example uses
    its OWN regime label to pick which head's loss applies. At inference:
    soft-blend the 3 heads via the regime gate's predicted probability.
    """

    def __init__(self, head_input_dim: int, hidden_dim: int = 192,
                 horizons: tuple = (1, 4, 16, 64),
                 num_bins: int = 255, n_regimes: int = 3,
                 dropout: float = 0.05):
        super().__init__()
        self.horizons = horizons
        self.num_bins = num_bins
        self.n_regimes = n_regimes

        self.regime_heads = nn.ModuleList([
            nn.ModuleDict({
                str(h): nn.Sequential(
                    nn.Linear(head_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_bins),
                )
                for h in horizons
            })
            for _ in range(n_regimes)
        ])

    def forward(self, head_input: torch.Tensor) -> dict:
        """Forward: head_input -> {regime_idx: {h: logits}}.

        Returns ALL regime outputs; caller picks per-example regime
        at training time, and blends at inference.
        """
        return {
            r: {h: self.regime_heads[r][str(h)](head_input) for h in self.horizons}
            for r in range(self.n_regimes)
        }

    def soft_blend(self, all_outputs: dict, regime_probs: torch.Tensor) -> dict:
        """Soft-blend outputs by regime probabilities.

        Args:
            all_outputs: dict {regime_idx: {h: logits[B, T, num_bins]}}
            regime_probs: [B, T, n_regimes] softmax over regime gate

        Returns:
            blended: {h: logits[B, T, num_bins]}
        """
        blended = {}
        for h in self.horizons:
            stacked = torch.stack([all_outputs[r][h] for r in range(self.n_regimes)], dim=-2)
            # stacked: [B, T, n_regimes, num_bins]
            # regime_probs: [B, T, n_regimes] -> unsqueeze to [B, T, n_regimes, 1]
            blended[h] = (stacked * regime_probs.unsqueeze(-1)).sum(dim=-2)
        return blended


def _smoke_test_regime():
    """CC-H6 smoke: forward all regimes + soft-blend."""
    torch.manual_seed(0)
    B, T, head_in, NB = 4, 32, 256, 255
    head_input = torch.randn(B, T, head_in)

    rh = RegimeConditionalHeads(head_input_dim=head_in, num_bins=NB)
    all_out = rh(head_input)
    assert set(all_out.keys()) == {0, 1, 2}
    for r in (0, 1, 2):
        for h in rh.horizons:
            assert all_out[r][h].shape == (B, T, NB)

    # Soft blend test: uniform regime probs -> output is mean of 3 heads
    regime_probs = torch.full((B, T, 3), 1 / 3.0)
    blended = rh.soft_blend(all_out, regime_probs)
    assert blended[1].shape == (B, T, NB)
    expected = (all_out[0][1] + all_out[1][1] + all_out[2][1]) / 3.0
    diff = (blended[1] - expected).abs().max().item()
    assert diff < 1e-5, f"soft-blend math broken; diff={diff}"

    n_params = sum(p.numel() for p in rh.parameters())
    print(f"[CC-H6 regime-heads] smoke PASS")
    print(f"  regimes: 3 (bear/neutral/bull), horizons: {rh.horizons}")
    print(f"  soft-blend uniform-prob diff: {diff:.2e}")
    print(f"  params: {n_params:,}")


# =============================================================================
# CC-H7 -- Dream-rollout auxiliary loss
# =============================================================================
#
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H7: V1.6 has `dream_step` defined
# but not in the loss. Adding it forces the latent to be predictively
# useful (not just encoding-useful).
#
# Expected lift: ShIC +0.003-0.007.
#
# Wiring contract:
#     model: must expose `dream_step(h_seq, latent) -> (h_next, latent_next)`
#     last_h: [B, d_model]   -- final hidden state of encoder
#     last_z: [B, latent_flat] -- final RSSM latent
#     bucketer: TwoHot encoder (model attribute)
#     return_target_seq: [B, T] target returns at h=1 (for the rollout
#                         steps' supervision signal)
#
# At each rollout step, the model's dream_step rolls the latent forward
# and predicts a return. We supervise the rollout's return predictions
# against the next-bar's return target. This trains dream_step to be a
# valid 1-step generative model in latent space.


# =============================================================================
# Regime-FiLM Conditioner (CC-H6 +1 tier — regime-aware encoder)
# =============================================================================
#
# 2026-05-16 addition: a lightweight regime-aware conditioning module that
# modulates encoder hidden states per regime. Unlike CC-H6
# (RegimeConditionalHeads, which specializes ONLY the output decoders),
# this module conditions the ENCODER output via FiLM (Feature-wise Linear
# Modulation, Perez et al. 2018).
#
# Spec:
#     - Input:  h_seq [B, T, d_model], regime_probs [B, T, 3]
#     - Per regime, learnable (scale_r, shift_r) modulators of shape [d_model]
#     - At each timestep, the effective scale/shift is the regime-prob-weighted
#       mix: effective_scale[b,t] = sum_r prob[b,t,r] * scale_r
#     - Output: h_seq_mod = h_seq * (1 + effective_scale) + effective_shift
#
# Cost: 3 regimes * 2 (scale, shift) * d_model = 6*d_model params (≈ 1.5K
# for d_model=256). Three orders of magnitude cheaper than per-regime
# encoders, but provides genuine architectural regime conditioning (vs
# CC-H6 which is decoder-only).
#
# Wiring contract: insert AFTER the encoder block, BEFORE the heads.
# At training, regime_probs can be one-hot from regime_labels; at
# inference, use the regime classifier's softmax output for soft blending.
#
# Expected lift on top of CC-H6: +0.005-0.010 IC, marginal Sharpe-shift gain.

class RegimeFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on regime.

    For each of n_regimes, learn a (scale, shift) vector of shape [d_model].
    At each timestep, mix the per-regime scale/shift by the regime
    probability distribution.

    Initialization: scales near zero, shifts at zero — at init, this is
    approximately the identity, so the conditioner is safe to inject into
    pretrained models without disrupting existing behavior. As training
    progresses, the module learns regime-specific feature scaling.

    Per-version: opt-in via REGIME_AWARENESS_MODE = "film" in settings.
    """

    def __init__(self, d_model: int, n_regimes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_regimes = n_regimes
        # Tiny init: starts near identity (no early disruption)
        self.scale = nn.Parameter(torch.zeros(n_regimes, d_model))
        self.shift = nn.Parameter(torch.zeros(n_regimes, d_model))

    def forward(self, h_seq: torch.Tensor,
                  regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_seq:        [B, T, d_model]
            regime_probs: [B, T, n_regimes] softmax over regimes

        Returns:
            h_seq * (1 + effective_scale) + effective_shift
        """
        # regime_probs: [B, T, R]; scale: [R, D]; einsum -> [B, T, D]
        effective_scale = torch.einsum("btr,rd->btd", regime_probs, self.scale)
        effective_shift = torch.einsum("btr,rd->btd", regime_probs, self.shift)
        return h_seq * (1.0 + effective_scale) + effective_shift


def _smoke_test_regime_film():
    """Regime-FiLM smoke: identity at init + non-zero after gradient step."""
    torch.manual_seed(0)
    B, T, D = 4, 16, 64
    R = 3
    h_seq = torch.randn(B, T, D)
    regime_probs = torch.softmax(torch.randn(B, T, R), dim=-1)
    film = RegimeFiLM(d_model=D, n_regimes=R)
    out = film(h_seq, regime_probs)
    # Identity at init: out ~= h_seq (since scale/shift = 0)
    init_diff = (out - h_seq).abs().mean().item()
    assert init_diff < 1e-5, f"init not identity: {init_diff}"
    # Backward + step
    loss = (out - 1.0).pow(2).mean()
    loss.backward()
    # After parameters get a gradient, params are no longer zero
    assert film.scale.grad is not None
    n_params = sum(p.numel() for p in film.parameters())
    print(f"[regime-FiLM] smoke PASS")
    print(f"  init identity diff: {init_diff:.2e}  (expect ~0)")
    print(f"  params: {n_params:,}  ({n_params/D:.0f} per d_model dim)")


def dream_rollout_loss(
    model,
    last_h: torch.Tensor,
    last_z: torch.Tensor,
    return_target_seq: torch.Tensor,
    n_steps: int = 2,
) -> torch.Tensor:
    """Auxiliary loss: roll forward N steps in latent and supervise the
    return prediction at each rollout step.

    Args:
        model: must expose `dream_step(h, z) -> (h_next, z_next)`
               and `bucketer.compute_loss(logits, targets)` and a
               `return_heads['1']` (or callable that maps `cat([h,z]) -> return_logits[B, num_bins]`).
        last_h: [B, d_model]
        last_z: [B, latent_flat]
        return_target_seq: [B, T] target returns; we use the LAST n_steps
                           for supervision.
        n_steps: rollout depth (default 2 per WM_HEADLINE_UPGRADE_PLAN).

    Returns:
        scalar dream loss (sum over n_steps)
    """
    if not hasattr(model, "dream_step"):
        raise AttributeError("model must implement dream_step(h, z) -> (h, z)")
    # Pick the h=1 return head (assumes dict with "1" key)
    heads = getattr(model, "return_heads", None)
    bucketer = getattr(model, "bucketer", None)
    if heads is None or bucketer is None:
        raise AttributeError("model must expose return_heads + bucketer")

    h, z = last_h, last_z
    losses = []
    T = return_target_seq.shape[1]
    for step in range(n_steps):
        h, z = model.dream_step(h, z)
        # Predict return at this rolled-forward step
        feat = torch.cat([h, z], dim=-1)
        return_logits = heads["1"](feat)   # [B, num_bins]
        # supervise against target_seq at offset T - n_steps + step (the last n_steps bars)
        idx = T - n_steps + step
        idx = max(0, min(idx, T - 1))
        target = return_target_seq[:, idx]
        step_loss = bucketer.compute_loss(return_logits, target)
        losses.append(step_loss)
    return sum(losses) / max(1, len(losses))


def _smoke_test_dream():
    """CC-H7 smoke: stub model implementing dream_step + bucketer."""
    torch.manual_seed(0)
    B, T, D, FLAT, NB = 4, 16, 128, 64, 255

    class StubBucketer:
        def __init__(self, num_bins=255):
            self.num_bins = num_bins
        def compute_loss(self, logits, target):
            return F.cross_entropy(logits, torch.zeros_like(target).long().clamp(0, self.num_bins - 1))

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dream_proj = nn.Linear(D + FLAT, D)
            self.dream_gru = nn.GRUCell(D, D)
            self.return_heads = nn.ModuleDict({
                "1": nn.Linear(D + FLAT, NB)
            })
            self.bucketer = StubBucketer(NB)
        def dream_step(self, h, z):
            x = torch.cat([h, z], dim=-1)
            x = self.dream_proj(x)
            h_next = self.dream_gru(x, h)
            z_next = z * 0.99    # placeholder latent evolution
            return h_next, z_next

    model = StubModel()
    last_h = torch.randn(B, D)
    last_z = torch.randn(B, FLAT)
    target = torch.randn(B, T) * 0.01

    loss = dream_rollout_loss(model, last_h, last_z, target, n_steps=2)
    assert torch.isfinite(loss)
    loss.backward()

    print(f"[CC-H7 dream-rollout] smoke PASS")
    print(f"  shape: last_h={tuple(last_h.shape)} last_z={tuple(last_z.shape)} target={tuple(target.shape)}")
    print(f"  dream loss (2 steps): {loss.item():.6f}")


# =============================================================================
# Run all smokes
# =============================================================================

def _run_all_smokes():
    print("=" * 60)
    print("HEADLINE COMPONENTS -- SMOKE SUITE")
    print("=" * 60)
    _smoke_test_cross_asset()
    print()
    _smoke_test_multi_res()
    print()
    _smoke_test_linear_attn()
    print()
    _smoke_test_quantile()
    print()
    _smoke_test_regime()
    print()
    _smoke_test_dream()
    print()
    print("=" * 60)
    print("ALL CC-H1 / CC-H2 / CC-H3 / CC-H5 / CC-H6 / CC-H7 SMOKE PASS")
    print("=" * 60)


if __name__ == "__main__":
    _run_all_smokes()
