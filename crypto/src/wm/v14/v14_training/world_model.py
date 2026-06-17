"""
V14: Diffusion Return Distribution Model
==========================================

Generates the FULL DISTRIBUTION of possible returns, not a point estimate.

Instead of predicting E[return] = +0.3%, predicts:
  "returns could be [-2%, -0.5%, +0.3%, +1%, +3%] with these probabilities"

Position sizing from distribution SHAPE:
  - High mean, low variance -> full size (confident directional)
  - High mean, high variance -> half size (right direction but risky)
  - Bimodal (crash or rally) -> small size or skip

Architecture:
  1. WaveNet encoder -> condition embedding c [B, T, D]
  2. Diffusion denoiser: takes noisy return r_t + condition c
     -> predicts noise epsilon (standard DDPM)
  3. Training: add noise to actual returns, learn to denoise
  4. Inference: start from pure noise, denoise N steps -> return samples

Also produces TwoHot logits for V1-compatible interface.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

_v1_comp = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_comp not in sys.path:
    sys.path.insert(0, _v1_comp)

from components import RMSNorm, TwoHotSymlog, MLPHead, SwiGLU  # SwiGLU added 2026-06-11 (recon anchor)


# =============================================================================
# WaveNet Encoder (condition extractor, same as V11/V12)
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # fp32 for gated activation — tanh/sigmoid backward overflows fp16
        _f, _g = self.filter_conv(x), self.gate_conv(x)
        h = torch.tanh(_f.float()) * torch.sigmoid(_g.float())
        h = h.to(x.dtype)
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
        return residual, skip


class ConditionEncoder(nn.Module):
    """WaveNet encoder that produces condition embeddings for denoiser."""

    def __init__(self, in_dim, d_model, channels, dilations, kernel=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels[0], 1)
        self.out_dim = d_model
        self.blocks = nn.ModuleList()
        self.ch_trans = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        out_ch = channels[-1]

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            self.ch_trans.append(nn.Conv1d(in_ch, ch, 1) if in_ch != ch else None)
            self.blocks.append(WaveNetBlock(ch, kernel, dil, dropout))
            self.skip_projs.append(nn.Conv1d(ch, out_ch, 1) if ch != out_ch else None)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.Conv1d(out_ch, d_model, 1),
        )

    def forward(self, x):
        """x: [B, T, F] -> [B, T, D]"""
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        skip_sum = 0
        for i, block in enumerate(self.blocks):
            if self.ch_trans[i] is not None:
                x = self.ch_trans[i](x)
            x, skip = block(x)
            if self.skip_projs[i] is not None:
                skip = self.skip_projs[i](skip)
            skip_sum = skip_sum + skip
        x = self.output_proj(x + skip_sum)
        return x.transpose(1, 2)


# =============================================================================
# Diffusion Denoiser
# =============================================================================

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal time embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B] float in [0, 1] -> [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ReturnDenoiser(nn.Module):
    """Denoises return values conditioned on encoder output.

    Input: noisy return r_t [B, T, 1] + condition c [B, T, D] + timestep t [B]
    Output: predicted noise epsilon [B, T, 1]
    """

    def __init__(self, cond_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # Input: noisy_return(1) + condition(cond_dim) -> hidden
        self.input_proj = nn.Linear(1 + cond_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "fc": nn.Linear(hidden_dim, hidden_dim),
                "act": nn.SiLU(),
                "drop": nn.Dropout(dropout),
            }))

        # 2026-05-09 score head bound: V22 autopsy showed `regime_ffn.X.3` outputs
        # in the 100-253 range — same architectural pattern as the diffusion score
        # output here. Pre-output LayerNorm bounds the score magnitude before
        # final 1-d projection. Diffusion score is sensitive to magnitude (sets
        # the gradient scale during sampling); unbounded → underfit + miscalibrated.
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, noisy_return, condition, timestep):
        """
        noisy_return: [B, T, 1]
        condition: [B, T, D]
        timestep: [B] float in [0, 1]
        Returns: [B, T, 1] predicted noise
        """
        # Time embedding
        t_emb = self.time_proj(F.silu(self.time_emb(timestep)))  # [B, H]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, H]

        # Combine inputs
        h = self.input_proj(torch.cat([noisy_return, condition], dim=-1))  # [B, T, H]
        h = h + t_emb  # Add time conditioning

        for layer in self.layers:
            residual = h
            h = layer["norm"](h)
            h = layer["act"](layer["fc"](h))
            h = layer["drop"](h)
            h = residual + h

        return self.output_proj(self.output_norm(h))  # [B, T, 1]


# =============================================================================
# V14 World Model
# =============================================================================

class DiffusionWorldModel(nn.Module):
    """V14: Diffusion Return Distribution.

    Trains: add noise to actual returns -> learn to denoise
    Inference: start from noise -> denoise -> return samples -> statistics
    Also produces TwoHot logits for V1-compatible evaluation.
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.z_dim = VIB_Z_DIM

        # Asset embedding
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        # Obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model), nn.SiLU(), nn.Dropout(dropout),
        )

        # Condition encoder (WaveNet)
        self.condition_encoder = ConditionEncoder(
            d_model, d_model, WAVENET_CHANNELS, WAVENET_DILATIONS,
            WAVENET_KERNEL, WAVENET_DROPOUT,
        )

        # Variational Information Bottleneck for the TwoHot path. The diffusion
        # denoiser path remains direct (it has its own noise injection regularizer
        # by construction). Without VIB on TwoHot, ShIC validation memorizes the
        # same way V3-clean did — same WaveNet -> direct head bug class.
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

        # ── RECON ANCHOR (keystone, 2026-06-11; donor V12:222/434/702-710) ───
        # Reconstruction decoder OFF the bottlenecked VIB latent on the TwoHot path
        # (z_expand(z), the d_model `feat`). Mirrors the V12 donor + the V13/V23
        # grafts: decode the bottlenecked feature back to the INPUT features
        # [B,T,input_dim]. V14 had a REAL Gaussian VIB on the TwoHot path (the ShIC-
        # validated path) but recon=torch.zeros stub + rec=0.0 + NO recon term -> the
        # bottleneck had no input-reconstruction pressure (the TwoHot heads could
        # route around z; the V22/V25 memorization trap: high contiguous IC, ShIC~0).
        # The recon term forces the 32-dim VIB latent to retain input-reconstructable
        # structure; the VIB KL (already computed + already in `total`, get_loss line
        # ~568) caps its capacity -> the bottleneck becomes REAL.
        #
        # SCOPE (honest): this anchors the TWOHOT path ONLY -- the path where ShIC is
        # measured. The headline DDPM denoiser reads `condition` UNBOTTLENECKED
        # (forward_train line ~345; sample/loss line ~604) by design (stable diffusion
        # training). Routing the denoiser off the bottleneck is a SEPARATE, larger
        # change that risks the diffusion path; see the DDPM-bottleneck TODO in
        # settings.py. For this graft the denoiser is intentionally left as-is.
        #
        # RNG-neutral construction (mirrors the V23 graft): snapshot RNG before
        # building the decoder + restore after, so the denoisers/heads constructed
        # below see the exact same RNG draws as the pre-graft model.
        _rng_state = torch.get_rng_state()
        self.recon_decoder = nn.Sequential(
            SwiGLU(d_model, RETURN_HEAD_DIM, dim_out=RETURN_HEAD_DIM, dropout=dropout),
            RMSNorm(RETURN_HEAD_DIM),
            nn.Linear(RETURN_HEAD_DIM, input_dim),
        )
        torch.set_rng_state(_rng_state)

        # Per-horizon denoisers
        self.denoisers = nn.ModuleDict({
            str(h): ReturnDenoiser(d_model, DENOISER_HIDDEN, DENOISER_LAYERS, dropout)
            for h in ACTIVE_HORIZONS
        })

        # Diffusion schedule (linear beta)
        steps = DIFFUSION_STEPS
        betas = torch.linspace(DIFFUSION_BETA_START, DIFFUSION_BETA_END, steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - alpha_cumprod))

        # TwoHot return heads (for V1-compatible evaluation)
        head_dim = RETURN_HEAD_DIM
        ret_mid = head_dim // 2
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, head_dim), RMSNorm(head_dim),
            nn.SiLU(), nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, ret_mid), RMSNorm(ret_mid),
                nn.SiLU(), nn.Linear(ret_mid, num_bins),
            )
            for h in REWARD_HORIZONS
        })
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"
        self.log_vars = nn.Parameter(torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5]))

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026 auxiliary heads / encoder cond)
        # V14 NATIVE FIT: DDPM produces distributional returns at inference via
        # N_SAMPLES; CC-H5 quantile heads add fast q05/q50/q95 outputs the
        # meta-learner can consume without expensive sampling.
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

    def _q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to clean data at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             input_obs[:, :-1, :]], dim=1)
        h = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))
        # WaveNet condition encoder in fp32 — same residual+skip GN:inf fix as V3/V11
        with torch.amp.autocast("cuda", enabled=False):
            condition = self.condition_encoder(h.float())

        h_seq = condition

        # RegimeFiLM (SOTA-2026 opt-in): condition h_seq BEFORE VIB.
        # Note: only modulates the TwoHot path; the DDPM denoiser still
        # reads `condition` (unmodulated) for stable diffusion training.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # Variational Information Bottleneck on TwoHot path (replaces no-op ATME).
        # Diffusion denoiser still reads condition directly downstream.
        mu = self.to_mu(h_seq)
        logvar = self.to_logvar(h_seq).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat = self.z_expand(z)

        # ── RECON ANCHOR (keystone, 2026-06-11) ──────────────────────────────
        # Decode the bottlenecked TwoHot-path latent back to the INPUT features
        # [B,T,input_dim]. Mirrors the V12 donor (decode off z_expand(z)) + the V13
        # graft. The masked recon-MSE in get_loss compares this vs the causally-
        # shifted clean input. NOTE: this anchors the TwoHot path only; the DDPM
        # denoiser is untouched (reads `condition` unbottlenecked, by design).
        recon = self.recon_decoder(feat)        # [B, T, input_dim]

        # TwoHot return predictions (V1-compatible)
        ret_trunk = self.return_trunk(feat)
        return_logits = {h_key: self.return_heads[str(h_key)](ret_trunk) for h_key in REWARD_HORIZONS}
        regime_logits = self.regime_head(h_seq)

        # CC-H5 + CC-H6 auxiliary heads (SOTA-2026)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        return {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "condition": condition, "ret_trunk": ret_trunk,
            "vib_mu": mu, "vib_logvar": logvar, "z_post": z,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": recon,  # [B, T, input_dim] -- REAL recon anchor (was torch.zeros stub)
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
        }

    @torch.no_grad()
    def sample_returns(self, condition, horizon=1, n_samples=None, num_steps=None):
        """Generate return distribution samples via reverse diffusion.

        Args:
            condition: [B, T, D] from encoder
            horizon: which horizon to sample
            n_samples: number of samples (default from settings)
            num_steps: number of sampling steps (default DIFFUSION_INFERENCE_STEPS)

        Returns: [B, T, n_samples] return samples

        Dispatches on HEADLINE_USE_DDIM:
          False  -> proper DDPM with evenly-subsampled timesteps
          True   -> DPM-Solver++ 2M (training-free, Lu et al. NeurIPS 2022,
                    arXiv:2211.01095). K=10-15 steps approximates DDPM K=50.

        2026-05-21: the prior `reversed(range(K))` only visited t=[K-1..0]
        — the bottom K bins of the noise schedule, missing the high-noise
        regime entirely. Both branches now subsample evenly from
        [0, DIFFUSION_STEPS-1].
        """
        n_samples = n_samples or DIFFUSION_N_SAMPLES
        num_steps = num_steps or DIFFUSION_INFERENCE_STEPS

        try:
            from settings import HEADLINE_USE_DDIM as _use_ddim
        except ImportError:
            _use_ddim = False

        if _use_ddim:
            return self._sample_dpmpp_2m(condition, horizon, n_samples, num_steps)
        return self._sample_ddpm_subsampled(condition, horizon, n_samples, num_steps)

    @torch.no_grad()
    def _sample_ddpm_subsampled(self, condition, horizon, n_samples, num_steps):
        """Vanilla DDPM with evenly-spaced timestep subsampling.

        Walks K+1 timesteps from T_train-1 down to 0, evenly spaced. Each
        step uses the DDPM update at the LARGER of the two boundary t's
        (the noise level we're starting from in this step).
        """
        B, T, D = condition.shape
        denoiser = self.denoisers[str(horizon)]
        T_train = DIFFUSION_STEPS

        # Evenly-spaced indices from T_train-1 down to 0 (inclusive on both)
        t_indices = torch.linspace(T_train - 1, 0, num_steps + 1).long().tolist()

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, T, 1, device=condition.device)
            for i in range(num_steps):
                t = t_indices[i]
                t_tensor = torch.full((B,), t / T_train, device=condition.device)
                noise_pred = denoiser(x, condition, t_tensor)
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_cumprod[t]
                beta_t = self.betas[t]
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
                )
                # Add noise except at the final step (t -> 0)
                if i < num_steps - 1:
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x)
            all_samples.append(x.squeeze(-1))
        return torch.stack(all_samples, dim=-1)

    @torch.no_grad()
    def _sample_dpmpp_2m(self, condition, horizon, n_samples, num_steps):
        """DPM-Solver++ 2M (multistep order-2). Training-free SOTA for
        fast diffusion sampling. Lu et al. NeurIPS 2022 (arXiv:2206.00927,
        arXiv:2211.01095). Drop-in for an epsilon-prediction DDPM.

        At K=10-15 steps, distributional quality matches DDPM K=50 on 1D
        scalar manifolds (simpler than images; faster convergence).

        Algorithm (x-prediction parameterization):
          1. Subsample K+1 timesteps evenly in [0, T_train-1].
          2. At each step from t_i to t_{i+1} (lower noise):
             a. eps_hat = denoiser(x_t, t)
             b. x0_hat = (x_t - sigma_t * eps_hat) / sqrt(alpha_bar_t)
             c. 1st-order: x_{t+1} = (sigma_{t+1}/sigma_t) * x_t
                            - sqrt(alpha_bar_{t+1}) * (exp(-h) - 1) * x0_hat
             d. 2M correction (i >= 1): use previous x0 to extrapolate D:
                D = (1 + 1/(2r)) * x0_cur - (1/(2r)) * x0_prev
                where r = h_cur / h_prev
        """
        B, T, D = condition.shape
        denoiser = self.denoisers[str(horizon)]
        T_train = DIFFUSION_STEPS

        # K+1 timestep indices from highest noise (T_train-1) to lowest (0)
        t_indices = torch.linspace(T_train - 1, 0, num_steps + 1).long().tolist()

        # Pre-compute log-SNR per index. lambda_t = 0.5*log(alpha_bar_t / (1-alpha_bar_t))
        ab = self.alpha_cumprod                                    # [T_train]
        log_snr = 0.5 * (torch.log(ab.clamp(min=1e-12))
                         - torch.log((1 - ab).clamp(min=1e-12)))  # [T_train]

        all_samples = []
        for _ in range(n_samples):
            x = torch.randn(B, T, 1, device=condition.device)
            prev_x0 = None
            prev_h = None

            for i in range(num_steps):
                t_cur = t_indices[i]
                t_next = t_indices[i + 1]
                t_norm = torch.full((B,), t_cur / T_train, device=condition.device)
                eps = denoiser(x, condition, t_norm)

                ab_cur = ab[t_cur]
                ab_next = ab[t_next]
                sigma_cur = torch.sqrt((1 - ab_cur).clamp(min=1e-12))
                sigma_next = torch.sqrt((1 - ab_next).clamp(min=1e-12))
                sqrt_ab_cur = torch.sqrt(ab_cur.clamp(min=1e-12))
                sqrt_ab_next = torch.sqrt(ab_next.clamp(min=1e-12))

                x0 = (x - sigma_cur * eps) / sqrt_ab_cur

                lam_cur = log_snr[t_cur]
                lam_next = log_snr[t_next]
                h = lam_next - lam_cur  # positive when going to lower noise

                if prev_x0 is not None and prev_h is not None and prev_h.abs() > 1e-6:
                    r = h / prev_h
                    D = (1.0 + 0.5 / r) * x0 - (0.5 / r) * prev_x0
                else:
                    D = x0

                # Final-step shortcut: at the last solver iteration, return
                # x0_hat directly (the de-noised estimate) rather than running
                # the update equation to ab[0] (which leaves a small residual
                # noise because alpha_bar[0] != 1.0 exactly). This is the
                # standard diffusers convention (DPMSolverMultistepScheduler
                # treats t=0 as fully clean).
                if i == num_steps - 1:
                    x = D
                else:
                    x = (sigma_next / sigma_cur) * x \
                        - sqrt_ab_next * (torch.exp(-h) - 1.0) * D

                prev_x0 = x0
                prev_h = h

            all_samples.append(x.squeeze(-1))
        return torch.stack(all_samples, dim=-1)

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None, **kwargs):
        B, T, n_feat = obs_seq.shape
        dev = str(obs_seq.device)
        if self._bucketer_device != dev:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev)
            self._bucketer_device = dev

        masked_obs = obs_seq.clone()
        if mask_ratio > 0 and self.training:
            mask = torch.rand(B, T, 1, device=obs_seq.device) < mask_ratio
            masked_obs = masked_obs * (~mask).float()

        outputs = self.forward_train(obs_seq, asset_id, masked_obs)
        condition = outputs["condition"]

        # VIB KL on TwoHot path
        mu = outputs["vib_mu"]
        logvar = outputs["vib_logvar"]
        vib_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        # default 1.0 (was 0.0): validate() calls get_loss WITHOUT kl_anneal, so the
        # val total excluded the KL term -> checkpoint selection could pick a
        # VIB-collapsed model. Training passes kl_anneal explicitly, so unaffected.
        kl_anneal = kwargs.get("kl_anneal", 1.0)
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        s = self.log_vars.clamp(-6.0, 6.0)
        total = kl_weight * vib_kl
        loss_dict = {"total": 0.0, "kl": vib_kl.item(), "kl_raw": vib_kl.item(),
                     "kl_weight": kl_weight}

        # ── RECON ANCHOR: reconstruction MSE (keystone, 2026-06-11) ──────────
        # recon[b,t,:] (decoded off the bottlenecked TwoHot-path latent at position
        # t) must reconstruct the CLEAN feature the encoder consumed at that
        # position. The encoder consumes `shifted` = causal-shift of the (possibly
        # mask_ratio-dropped) input (forward_train line ~340), so the recon TARGET
        # is the causal-shift of the CLEAN obs_seq (standard masked-autoencoding).
        # NO LOOK-AHEAD: position t reconstructs obs[t-1] (a PAST bar), never a
        # future one. RECON_WEIGHT>0 makes the TwoHot VIB bottleneck REAL -- the
        # latent must carry input-reconstructable structure, so the TwoHot heads
        # can no longer route around z. Together with the VIB KL (already in `total`
        # above) this is the anchor V14's TwoHot path was missing. Mirrors V13.
        # SCOPE: TwoHot path only; the DDPM denoiser is unanchored (see settings TODO).
        recon = outputs["recon"]                                 # [B, T, input_dim]
        recon_target = torch.cat(
            [torch.zeros(B, 1, n_feat, device=obs_seq.device), obs_seq[:, :-1, :]],
            dim=1)                                               # clean causal shift
        l_rec = F.mse_loss(recon, recon_target)
        total = total + RECON_WEIGHT * l_rec

        l_direct = torch.tensor(0.0, device=obs_seq.device)

        # TwoHot losses (standard, V1-compatible)
        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits_flat = outputs["return_logits"][h].reshape(-1, self._num_bins)
            tgt_flat = targets[h].reshape(-1)
            if h in ACTIVE_HORIZONS:
                l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)
                total = total + torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()
            decoded = self.bucketer.decode(logits_flat)
            l_direct = l_direct + F.huber_loss(decoded, tgt_flat, reduction="mean", delta=0.5)

        total = total + DIRECT_RETURN_WEIGHT * l_direct
        loss_dict["direct_ret"] = l_direct.item()

        # Diffusion denoising losses (per active horizon)
        l_diffusion = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h not in targets:
                continue
            denoiser = self.denoisers[str(h)]
            x_0 = targets[h].unsqueeze(-1)  # [B, T, 1]

            # Random diffusion timestep per sample
            t = torch.randint(0, DIFFUSION_STEPS, (B,), device=obs_seq.device)
            noisy, noise = self._q_sample(x_0, t)
            t_float = t.float() / DIFFUSION_STEPS

            # Predict noise
            noise_pred = denoiser(noisy, condition, t_float)
            l_diff_h = F.mse_loss(noise_pred, noise)
            l_diffusion = l_diffusion + l_diff_h
            loss_dict["diff_%d" % h] = l_diff_h.item()

        total = total + l_diffusion  # Equal weight with TwoHot
        loss_dict["diffusion"] = l_diffusion.item()

        # Regime
        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(outputs["regime_logits"].reshape(-1, 3), regime_tgt.reshape(-1))
            s_regime = s[-1].clamp(max=-1.0)
            total = total + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (outputs["regime_logits"].argmax(-1) == regime_tgt).float().mean().item()

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    logits_h = outputs["return_logits"][h].reshape(-1, self._num_bins)
                    dec = self.bucketer.decode(logits_h)
                    act = targets[h].reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (torch.sign(dec[nz]) == torch.sign(act[nz])).float().mean().item()

        # Recon anchor (2026-06-11 graft): report the REAL value (was hardcoded
        # 0.0). The recon term is in `total` (RECON_WEIGHT*l_rec above); the VIB KL
        # is in `total` (kl_weight*vib_kl at the top of get_loss).
        loss_dict["rec"] = l_rec.item()

        # CC-H5 quantile loss (SOTA-2026, auxiliary)
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

        # CC-H6 regime-conditional CE (SOTA-2026, auxiliary)
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
        return total, loss_dict, outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
