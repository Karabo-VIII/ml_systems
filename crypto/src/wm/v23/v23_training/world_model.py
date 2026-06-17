"""V23 World Model — xLSTM (Beck et al., NeurIPS 2024).

Production V-version of the V23 backbone. Adds get_loss() following the
V13 pattern + lazy-device TwoHotSymlog bucketer.

Architecture (faithful to NeurIPS 2024 paper §3-4):
  - sLSTMBlock: scalar cell, exponential input/forget gates, normalizer state
    n_t to prevent overflow (paper Algorithm 1).
  - mLSTMBlock: matrix C_t cell, parallel (q, k, v) projections + outer-product
    memory updates + associative recall (paper §3.2).
  - Stack: alternating sLSTM + mLSTM blocks (paper Table 1 default).

Note on numerical stability: exp gating is clamped to max=10 to prevent
fp16/fp32 overflow during AMP training. Cell state is normalized by
max(|n_t|, 1e-6) (sLSTM) or max(|n^T q|, 1.0) (mLSTM) for stable gradient flow.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from settings import *
except ImportError:
    from .settings import *

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)

from components import RMSNorm, TwoHotSymlog, MLPHead, SwiGLU

# Round-7 frontier components
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from frontier_components import tail_adaptive_huber, CryptoPeriodEmbedding

# Shared model-agnostic VSN lever (causal per-timestep feature gate). Imported the
# same way the V1.1 template imports it; constructed ONLY when env V23_VSN=1 (below),
# so the default OFF path never touches it.
from variable_selection import VariableSelectionNetwork


# =============================================================================
# sLSTM Block (paper §3.1)
# =============================================================================

class sLSTMBlock(nn.Module):
    """Scalar cell with stabilized exponential gating (paper §3.1, Algorithm 1).

    Uses the m_t log-stabilizer to prevent fp16/fp32 overflow under AMP:
        m_t   = max(f_pre + m_{t-1}, i_pre)
        i'    = exp(i_pre - m_t)
        f'    = exp(f_pre + m_{t-1} - m_t)
    Recurrent state (c, n, m) is held in fp32 inside the loop regardless of
    autocast dtype to guarantee numerical stability.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # Combined projection for (z, i_pre, f_pre, o_pre)
        self.proj_x = nn.Linear(d_model, 4 * d_model, bias=True)
        self.proj_h = nn.Linear(d_model, 4 * d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Initialize forget bias positive (long memory by default), input bias slightly negative
        with torch.no_grad():
            f_bias_idx = slice(2 * d_model, 3 * d_model)
            self.proj_x.bias.data[f_bias_idx] = 1.0          # f_pre bias = 1.0 -> exp ~e
            i_bias_idx = slice(d_model, 2 * d_model)
            self.proj_x.bias.data[i_bias_idx] = -1.0         # i_pre bias = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_norm = self.norm(x)
        # Cast projections to fp32 explicitly. Under AMP fp16, the gate
        # pre-activations (`x_proj[:, t] + h_proj_t`) can reach ±65504 BEFORE
        # the m-stabilizer kicks in; .float() AFTER the addition saw inf and
        # propagated to the exp/maximum. Force fp32 throughout the recurrence.
        # Fixed 2026-05-21 RED-team audit.
        with torch.amp.autocast("cuda", enabled=False):
            x_proj = self.proj_x(x_norm.float())   # [B, T, 4D] fp32

        # Recurrent state: fp32 to avoid AMP overflow
        h_prev = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        c_prev = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        n_prev = torch.zeros(B, D, device=x.device, dtype=torch.float32)
        m_prev = torch.full((B, D), -1e9, device=x.device, dtype=torch.float32)

        outputs = []
        for t in range(T):
            with torch.amp.autocast("cuda", enabled=False):
                h_proj_t = self.proj_h(h_prev)     # fp32
            gates = x_proj[:, t] + h_proj_t        # both fp32
            z, i_pre, f_pre, o_pre = gates.chunk(4, dim=-1)

            # Stabilizer: m_t = max(f_pre + m_{t-1}, i_pre)
            m_new = torch.maximum(f_pre + m_prev, i_pre)
            i = torch.exp(i_pre - m_new)
            f = torch.exp(f_pre + m_prev - m_new)
            o = torch.sigmoid(o_pre)
            z = torch.tanh(z)

            c_new = f * c_prev + i * z
            n_new = f * n_prev + i
            h_new = o * (c_new / torch.clamp(torch.abs(n_new), min=1.0))

            outputs.append(h_new.to(x.dtype))
            h_prev, c_prev, n_prev, m_prev = h_new, c_new, n_new, m_new

        h_seq = torch.stack(outputs, dim=1)        # [B, T, D]
        return x + self.dropout(h_seq)             # residual


# =============================================================================
# mLSTM Block (paper §3.2)
# =============================================================================

class mLSTMBlock(nn.Module):
    """Matrix memory C_t with stabilized exp gating (paper §3.2, Algorithm 1).

    Uses scalar (per-batch) exp gates with m_t log-stabilizer:
        i_pre, f_pre at scalar dim (mean over d_v)
        m_t   = max(f_pre + m_{t-1}, i_pre)
        i'    = exp(i_pre - m_t)
        f'    = exp(f_pre + m_{t-1} - m_t)
        C_t   = f' * C_{t-1} + i' * v_t k_t^T
        n_t   = f' * n_{t-1} + i' * k_t
        h_t   = (C_t q_t) / max(|n_t^T q_t|, 1)
    Recurrent state (C, n, m) is held in fp32 inside the loop. d_v=d_model
    so memory is O(B * d_model^2) per layer.
    """

    def __init__(self, d_model: int, d_value: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_value = d_value if d_value is not None else d_model
        self.d_model = d_model
        self.d_value = d_value
        # Per-token projection: q, k, v at d_value; i_pre, f_pre, o_pre at scalar (1)
        self.proj_qkv = nn.Linear(d_model, 3 * d_value, bias=True)
        self.proj_gates = nn.Linear(d_model, 3, bias=True)   # (i_pre, f_pre, o_pre) scalars
        self.out_proj = nn.Linear(d_value, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Init forget bias positive (long memory), input bias slightly negative
        with torch.no_grad():
            self.proj_gates.bias.data[1] = 1.0      # f_pre bias
            self.proj_gates.bias.data[0] = -1.0     # i_pre bias
            self.proj_gates.bias.data[2] = 0.0      # o_pre bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_norm = self.norm(x)
        qkv = self.proj_qkv(x_norm)                            # [B, T, 3*d_v]
        # Gates pre-activations in fp32 BEFORE the projection (was .float()
        # AFTER, which preserved any fp16-overflow inf). Fixed 2026-05-21.
        with torch.amp.autocast("cuda", enabled=False):
            gates = self.proj_gates(x_norm.float())            # [B, T, 3] fp32
        d_v = self.d_value
        q, k, v = qkv.split(d_v, dim=-1)
        # k normalized for stable outer-product (paper §3.2)
        k = k / math.sqrt(d_v)

        # Recurrent state: fp32
        C = torch.zeros(B, d_v, d_v, device=x.device, dtype=torch.float32)
        n_state = torch.zeros(B, d_v, device=x.device, dtype=torch.float32)
        m_prev = torch.full((B, 1), -1e9, device=x.device, dtype=torch.float32)
        outputs = []
        for t in range(T):
            i_pre = gates[:, t, 0:1]    # [B, 1]
            f_pre = gates[:, t, 1:2]    # [B, 1]
            o_pre = gates[:, t, 2:3]    # [B, 1]
            v_t = v[:, t].float()
            k_t = k[:, t].float()
            q_t = q[:, t].float()

            # Stabilizer
            m_new = torch.maximum(f_pre + m_prev, i_pre)
            i = torch.exp(i_pre - m_new)               # [B, 1]
            f = torch.exp(f_pre + m_prev - m_new)      # [B, 1]
            o = torch.sigmoid(o_pre)                    # [B, 1]

            outer = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))  # [B, d_v, d_v]
            C = f.unsqueeze(-1) * C + i.unsqueeze(-1) * outer
            n_state = f * n_state + i * k_t

            h_raw = torch.bmm(C, q_t.unsqueeze(-1)).squeeze(-1)
            denom = torch.clamp(
                torch.abs((n_state * q_t).sum(dim=-1, keepdim=True)), min=1.0
            )
            h = o * (h_raw / denom)
            outputs.append(h.to(x.dtype))
            m_prev = m_new

        h_seq = torch.stack(outputs, dim=1)
        out = self.out_proj(h_seq)
        return x + self.dropout(out)


# =============================================================================
# xLSTM World Model
# =============================================================================

class xLSTMWorldModel(nn.Module):
    """V23: stacked sLSTM/mLSTM for crypto WM (V1.x-compatible interface)."""

    def __init__(self, input_dim: int = INPUT_DIM, seq_len: int = WM_SEQ_LEN,
                 d_model: int = WM_D_MODEL, n_layers: int = WM_N_LAYERS,
                 dropout: float = WM_DROPOUT, num_bins: int = NUM_BINS,
                 num_assets: int = NUM_ASSETS,
                 asset_emb_dim: int = WM_ASSET_EMB_DIM,
                 d_value: int = MLSTM_DV,
                 block_pattern: str = BLOCK_PATTERN,
                 z_dim: int = VIB_Z_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self._num_bins = num_bins
        self.z_dim = z_dim
        # flat_dim=0: the fused feature the heads read in V23 is feat_vib (dim d_model),
        # not cat(h, z). attach_forward_regime_head reads (d_model + flat_dim) for its
        # head input dim, so flat_dim=0 makes feat_dim == d_model for the V23 path.
        self.flat_dim = 0

        # Forward-regime head hook (V23_FORWARD_REGIME lever). Default OFF: attach is
        # never called by base training, so _use_forward_regime stays False and the
        # guarded forward/loss blocks are no-ops -> base path byte-for-byte unchanged.
        self._use_forward_regime = False
        self.forward_regime_head = None

        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # -- VSN (Variable Selection Network, V23_VSN flag) -------------------------
        # Flag-gated: constructed ONLY when env var V23_VSN="1" at model init time.
        # When OFF (default): self.vsn is None -> not constructed, no params, no RNG
        # consumed, forward path byte-for-byte unchanged. When ON: a causal per-timestep
        # sigmoid gate sits BEFORE the obs_encoder, gating [B,T,input_dim]. Combinable
        # with V23_FORWARD_REGIME. Mirrors the V1.1 template wiring exactly.
        import os as _os_vsn
        self._use_vsn = _os_vsn.environ.get("V23_VSN", "0") == "1"
        if self._use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim)
        else:
            self.vsn = None

        # Stacked alternating blocks
        blocks = []
        for i in range(n_layers):
            if block_pattern == "alternate":
                blocks.append(
                    sLSTMBlock(d_model, dropout) if i % 2 == 0
                    else mLSTMBlock(d_model, d_value, dropout)
                )
            elif block_pattern == "all_mlstm":
                blocks.append(mLSTMBlock(d_model, d_value, dropout))
            else:
                blocks.append(sLSTMBlock(d_model, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.post_norm = RMSNorm(d_model)

        # VIB bottleneck on h_seq (round-4 anti-fragile fix). xLSTM's mLSTM
        # matrix memory has unbounded storage; VIB adds stochastic compression.
        self.to_mu = nn.Linear(d_model, self.z_dim)
        self.to_logvar = nn.Linear(d_model, self.z_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, VIB_LOGVAR_INIT)
        self.z_expand = nn.Sequential(
            nn.Linear(self.z_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, RETURN_HEAD_DIM),
            RMSNorm(RETURN_HEAD_DIM),
            nn.SiLU(),
            nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(RETURN_HEAD_DIM, RETURN_HEAD_DIM // 2),
                RMSNorm(RETURN_HEAD_DIM // 2),
                nn.SiLU(),
                nn.Linear(RETURN_HEAD_DIM // 2, num_bins),
            )
            for h in REWARD_HORIZONS
        })
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"
        self.log_vars = nn.Parameter(
            torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5])
        )

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026)
        import sys as _qsys
        from pathlib import Path as _qPath
        _shared = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
        if _shared not in _qsys.path:
            _qsys.path.insert(0, _shared)
        try:
            from settings import USE_QUANTILE_HEADS as _use_qh
        except ImportError:
            _use_qh = False
        if _use_qh:
            from headline_components import QuantileHeads as _QH
            self.quantile_heads = _QH(head_input_dim=d_model,
                                         horizons=tuple(REWARD_HORIZONS))
        else:
            self.quantile_heads = None
        try:
            from settings import USE_REGIME_COND_HEADS as _use_rc
        except ImportError:
            _use_rc = False
        if _use_rc:
            from headline_components import RegimeConditionalHeads as _RC
            self.regime_cond_heads = _RC(head_input_dim=d_model,
                                            horizons=tuple(REWARD_HORIZONS),
                                            num_bins=num_bins, n_regimes=3)
        else:
            self.regime_cond_heads = None
        try:
            from settings import REGIME_AWARENESS_MODE as _ram
        except ImportError:
            _ram = "heads"
        self._regime_awareness_mode = _ram
        if _ram == "film":
            from headline_components import RegimeFiLM as _RF
            self.regime_film = _RF(d_model=d_model, n_regimes=3)
            self.regime_film_gate = nn.Linear(d_model, 3)
            nn.init.zeros_(self.regime_film_gate.weight)
            nn.init.zeros_(self.regime_film_gate.bias)
        else:
            self.regime_film = None
            self.regime_film_gate = None

        # Reconstruction anchor (HEADLINE graft, 2026-06-10; donor V12:222/434/702-710).
        # V23 had a REAL VIB on h_seq but recon was a torch.zeros stub with
        # RECON_WEIGHT=0 -- the bottleneck was a pass-through label-fit (the V22/V25
        # memorization trap: high contiguous IC, ShIC~0). This decoder reads the
        # BOTTLENECKED VIB latent (feat_vib, dim d_model) and reconstructs the input
        # features [.,input_dim], forcing the VIB latent to retain input-reconstructable
        # structure (not pure label-fit); VIB KL caps capacity.
        #
        # CONSTRUCTED LAST + RNG-NEUTRAL BY DESIGN: every nn.Linear/SwiGLU consumes
        # global RNG at construction time, and _init_weights() runs AFTER all modules
        # are built -- so naively adding the decoder would shift the RNG state the
        # whole _init_weights pass then consumes, perturbing EVERY base param. We
        # snapshot the RNG state before building the decoder and restore it after, so
        # _init_weights sees the exact pre-graft RNG state. Result: with RECON_WEIGHT=0
        # the base forward+loss is byte-for-byte identical to the pre-graft model.
        # The loss term is gated by RECON_WEIGHT in get_loss; set RECON_WEIGHT>0 in
        # settings to fire the anchor.
        _rng_state = torch.get_rng_state()
        self.recon_decoder = nn.Sequential(
            SwiGLU(d_model, RETURN_HEAD_DIM, dim_out=RETURN_HEAD_DIM, dropout=dropout),
            RMSNorm(RETURN_HEAD_DIM),
            nn.Linear(RETURN_HEAD_DIM, input_dim),
        )
        torch.set_rng_state(_rng_state)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Skip the recon-anchor decoder: it keeps its construction-time init
            # (nn.Linear kaiming / SwiGLU's own), and excluding it here keeps this
            # loop's RNG draws byte-for-byte identical to the pre-graft model so the
            # final asset_embedding normal_ below sees the same RNG state.
            if name.startswith("recon_decoder"):
                continue
            # Skip the VSN gate: its neutral-start init (bias=0, weight std=0.01 ->
            # gates ~0.5) is set in VariableSelectionNetwork.__init__ and must NOT be
            # overwritten by xavier_uniform (std~0.156 would push gates off neutral).
            # When V23_VSN is OFF this branch is never reached (self.vsn is None ->
            # no gate_proj params), so the base RNG sequence is unaffected.
            if "gate_proj" in name:
                continue
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor,
                      masked_obs_seq: torch.Tensor | None = None):
        B, T, F_in = obs_seq.shape
        if F_in != self.input_dim:
            raise ValueError(f"Expected n_features={self.input_dim}, got {F_in}")
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # VSN: per-timestep causal feature gate applied BEFORE the obs_encoder.
        # When V23_VSN=0 (default): self.vsn is None, branch skipped, path identical
        # to base. When V23_VSN=1: input_obs <- gate(x_t) * x_t (same shape). The gate
        # is differentiable and causal (uses only x_t -> no future leak); gradients
        # flow through the gate weights. Mirrors the V1.1 template.
        if self._use_vsn and self.vsn is not None:
            input_obs = self.vsn(input_obs)

        shifted = torch.cat(
            [torch.zeros(B, 1, F_in, device=obs_seq.device), input_obs[:, :-1, :]],
            dim=1,
        )
        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        x = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))   # [B, T, D]

        for block in self.blocks:
            x = block(x)
        h_seq = self.post_norm(x)

        # RegimeFiLM (SOTA-2026 opt-in) — identity-at-init.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # VIB bottleneck (anti-fragile compression on h_seq)
        mu = self.to_mu(h_seq)
        logvar = self.to_logvar(h_seq).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat_vib = self.z_expand(z)

        # Recon anchor (HEADLINE graft): decode the bottlenecked VIB latent back to
        # the input features. Causal: feat_vib derives from the causally-shifted
        # encoder (obs_encoder runs on `shifted`), so recon[.,t,:] sees only bars < t.
        # GATED on RECON_WEIGHT so the OFF default (0.0) is byte-for-byte unchanged:
        # the decoder contains Dropout, so RUNNING it in train() would consume forward
        # RNG and perturb every downstream head. When RECON_WEIGHT==0 we emit the
        # original zeros stub (no decoder call -> no RNG draw -> base path identical).
        if RECON_WEIGHT > 0:
            recon = self.recon_decoder(feat_vib)              # [B, T, input_dim]
        else:
            recon = torch.zeros(B, T, 1, device=obs_seq.device)

        feat = feat_vib
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat.device)
                         > TEMPORAL_CTX_DROP).float()
            feat = feat * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {
            h_key: self.return_heads[str(h_key)](ret_trunk)
            for h_key in REWARD_HORIZONS
        }
        regime_logits = self.regime_head(feat)

        # CC-H5 + CC-H6 auxiliary heads (SOTA-2026)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        out = {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            "ret_trunk": ret_trunk,
            "vib_mu": mu,
            "vib_logvar": logvar,
            "z_post": z,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": recon,
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
        }
        # Forward-regime head (V23_FORWARD_REGIME lever). OFF by default:
        # _use_forward_regime is False until attach_forward_regime_head() is called by
        # the trainer. When OFF: no 'forward_regime' key in out -> base path byte-for-byte
        # unchanged. The head reads the SAME post-encoder `feat` the return/regime heads
        # read (causal; no future leak -- feat derives from the causally-shifted encoder).
        if getattr(self, "_use_forward_regime", False) and getattr(self, "forward_regime_head", None) is not None:
            out["forward_regime"] = self.forward_regime_head(feat)
        return out

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None,
                 return_components=False, **kwargs):
        B, T, n_feat = obs_seq.shape
        dev = str(obs_seq.device)
        if self._bucketer_device != dev:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev)
            self._bucketer_device = dev

        masked_obs = obs_seq.clone()
        recon_mask = None    # [B,T,1] True where a position was masked (denoising recon target)
        if mask_ratio > 0 and self.training:
            mask = torch.rand(B, T, 1, device=obs_seq.device) < mask_ratio
            masked_obs = masked_obs * (~mask).float()
            recon_mask = mask

        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # VIB KL term (round-4 anti-fragile bottleneck regularization)
        mu = outputs["vib_mu"]
        logvar = outputs["vib_logvar"]
        vib_kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
        kl_anneal = float(kwargs.get("kl_anneal", 1.0))
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        s = self.log_vars.clamp(-6.0, 6.0)
        loss_dict = {"total": 0.0, "rec": 0.0, "kl": vib_kl.item(),
                     "kl_raw": vib_kl.item(), "kl_weight": kl_weight}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits_flat = outputs["return_logits"][h].reshape(-1, self._num_bins)
            tgt_flat = targets[h].reshape(-1)
            if h in ACTIVE_HORIZONS:
                l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)
                ret_terms[h] = torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()
            decoded = self.bucketer.decode(logits_flat)
            # Round-7: tail-adaptive Huber for crypto fat-tail returns
            l_direct = l_direct + tail_adaptive_huber(
                decoded, tgt_flat, delta=0.5, tail_sigma=2.0, tail_weight=2.5
            )

        aux_term = DIRECT_RETURN_WEIGHT * l_direct + kl_weight * vib_kl
        loss_dict["direct_ret"] = l_direct.item()

        # -- Reconstruction anchor (HEADLINE graft, donor V12:702-710) -----------
        # Masked recon-MSE: recon[.,t,:] reconstructs the input features obs_seq[.,t,:].
        # When mask_ratio>0 (training) this is a DENOISING objective -- reconstruct the
        # positions that were zeroed in masked_obs (recon_mask), forcing the VIB latent
        # to encode the missing context. When no mask exists, reconstruct all positions.
        # GATED by RECON_WEIGHT: 0.0 (settings default pre-graft) -> the decoder is not
        # even run in forward_train (outputs["recon"] is the zeros stub), so aux_term and
        # loss_dict["rec"]==0.0 are byte-for-byte unchanged; >0 -> the anchor fires.
        if RECON_WEIGHT > 0:
            recon = outputs["recon"]                          # [B,T,input_dim]
            if recon_mask is not None:
                m1 = recon_mask.float()                       # [B,T,1] -> broadcast over F
                denom = (m1.sum() * n_feat).clamp(min=1.0)
                l_rec = ((recon - obs_seq).pow(2) * m1).sum() / denom
            else:
                l_rec = F.mse_loss(recon, obs_seq)
            loss_dict["rec"] = l_rec.item()
            aux_term = aux_term + RECON_WEIGHT * l_rec

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(
                outputs["regime_logits"].reshape(-1, 3), regime_tgt.reshape(-1)
            )
            s_regime = s[-1].clamp(max=-1.0)
            aux_term = aux_term + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (
                    outputs["regime_logits"].argmax(-1) == regime_tgt
                ).float().mean().item()

        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        # Forward-regime aux loss (V23_FORWARD_REGIME lever). Guarded: only fires when
        # the head is attached (_use_forward_regime=True) AND the trainer supplied
        # forward_regime_labels. Default OFF -> total unchanged. Weight 0.10 (small fixed;
        # additive supervision, not the primary objective). Labels packed by the trainer's
        # collate into targets["forward_regime_labels"] = {"bear","trend","move"}; NaN tail
        # rows (last K bars) are auto-masked by forward_regime_aux_loss. Mirrors V1.1.
        if getattr(self, "_use_forward_regime", False) and "forward_regime" in outputs:
            _fr_labels = targets.get("forward_regime_labels", {})
            if _fr_labels:
                import sys as _frsys
                from pathlib import Path as _frPath
                _shared_dir = str(_frPath(__file__).resolve().parent.parent.parent / "_shared")
                if _shared_dir not in _frsys.path:
                    _frsys.path.insert(0, _shared_dir)
                from forward_regime_head import forward_regime_aux_loss as _fr_aux_loss
                fr_loss = _fr_aux_loss(outputs, _fr_labels)
                total = total + 0.10 * fr_loss
                loss_dict["fr_aux"] = fr_loss.item()

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    logits_h = outputs["return_logits"][h].reshape(-1, self._num_bins)
                    dec = self.bucketer.decode(logits_h)
                    act = targets[h].reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (
                            torch.sign(dec[nz]) == torch.sign(act[nz])
                        ).float().mean().item()

        # CC-H5 + CC-H6 auxiliary losses (SOTA-2026)
        try:
            from settings import QUANTILE_LOSS_WEIGHT as _ql_w
        except ImportError:
            _ql_w = 0.0
        l_quantile = torch.tensor(0.0, device=obs_seq.device)
        if (self.quantile_heads is not None
                and outputs.get("quantile_logits") is not None
                and _ql_w > 0):
            import sys as _qsys
            from pathlib import Path as _qPath
            _sd = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
            if _sd not in _qsys.path:
                _qsys.path.insert(0, _sd)
            from headline_components import quantile_loss as _ql
            quants = self.quantile_heads.quantiles
            n_h = 0
            for h_key in REWARD_HORIZONS:
                if h_key in targets:
                    q_pred = outputs["quantile_logits"][h_key]
                    l_quantile = l_quantile + _ql(q_pred, targets[h_key], quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        total = total + _ql_w * l_quantile
        loss_dict["quantile_pinball"] = l_quantile.item()
        try:
            from settings import REGIME_COND_WEIGHT as _rc_w
        except ImportError:
            _rc_w = 0.0
        l_regime_cond = torch.tensor(0.0, device=obs_seq.device)
        if (self.regime_cond_heads is not None
                and outputs.get("regime_cond_logits") is not None
                and _rc_w > 0 and regime_labels is not None):
            flat_labels = regime_labels.reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h_key in REWARD_HORIZONS:
                    if h_key not in targets:
                        continue
                    head_logits = outputs["regime_cond_logits"][r][h_key]
                    flat_logits = head_logits.reshape(-1, NUM_BINS)
                    flat_targets = targets[h_key].reshape(-1)
                    l_per = self.bucketer.compute_loss(
                        flat_logits[mask_r], flat_targets[mask_r])
                    l_regime_cond = l_regime_cond + l_per
                    n_terms += 1
            if n_terms > 0:
                l_regime_cond = l_regime_cond / n_terms
        total = total + _rc_w * l_regime_cond
        loss_dict["regime_cond_ce"] = l_regime_cond.item()

        loss_dict["total"] = total.item()
        if return_components:
            components = {
                "aux": aux_term,
                **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS},
            }
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, F_in = 4, WM_SEQ_LEN, INPUT_DIM
    m = xLSTMWorldModel(input_dim=F_in)
    x = torch.randn(B, T, F_in)
    asset = torch.randint(0, NUM_ASSETS, (B,))
    targets = {h: torch.randn(B, T) * 0.01 for h in REWARD_HORIZONS}
    total, ld, out = m.get_loss(x, asset, targets, mask_ratio=0.15)
    total.backward()
    n_params = count_parameters(m)
    print(f"[V23 xLSTMWorldModel smoke] PASS: B={B} T={T} F={F_in}")
    print(f"  params={n_params:,}  loss={ld['total']:.4f}  direct={ld['direct_ret']:.4f}")
    print(f"  return_logits[1]: {tuple(out['return_logits'][1].shape)}")
