"""Shared trainer helpers for V1.x upgrade flag wiring.

Single source of truth for the 6 upgrade flags + their step-loop
plumbing. Each V1.x sibling (V1.0, V1.1, V1.4, V1.6) can call:

    from frontier_ml.v1_upgrades.trainer_helpers import (
        add_upgrade_args, build_upgrade_context, do_step_with_upgrades,
    )

    # In argparse setup:
    add_upgrade_args(parser)

    # After model + optimizer construction:
    ctx = build_upgrade_context(model, optimizer, args, device=DEVICE)

    # Replace the train-step backward block:
    do_step_with_upgrades(
        ctx, model, obs, asset, target_returns,
        get_loss_extra_kwargs=...
    )

The context object holds: SAM-wrapped optimizer (or original), FrAug
module, PCGrad module, use_amp flag, and a smart `step()` method that
routes to the right backward pattern based on which flags are active.

Upgrade applicability matrix (per-V; checked by add_upgrade_args
caller):
- V1.0/V1.1/V1.4/V1.6  — all 6 flags
- V12                  — all 6 flags (different model, same pattern)
- V4                   — all 6 flags except QKNorm-specific (already in code)
- V3                   — SAM/FrAug/MDN/AdaptiveBins (PCGrad/MTP n/a; single-head)
- V6                   — SAM/FrAug + VICReg-specific (no MTP/PCGrad/MDN/AdaptiveBins)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# Allow imports of v1_upgrades modules
_PKG = Path(__file__).resolve().parents[3]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_upgrade_args(parser: argparse.ArgumentParser) -> None:
    """Add the 6 frontier-ML V1.x upgrade flags to a parser.

    Default OFF so baseline training is unchanged. Idempotent if called
    multiple times (silent on conflict).
    """
    def _add(arg, **kwargs):
        try:
            parser.add_argument(arg, **kwargs)
        except argparse.ArgumentError:
            pass  # already added

    _add("--sam", action="store_true",
         help="[B003 R1] Wrap optimizer with Sharpness-Aware Minimization. "
              "Doubles wallclock per step; targets ShIC + IC simultaneously.")
    _add("--sam-rho", type=float, default=0.7,
         help="SAM perturbation radius. Default 0.7 per SAMformer official "
              "run.py (VERIFIED via github.com/romilbert/samformer/blob/main/run.py "
              "2026-05-02). Foret 2020 used 0.05 for vision; SAMformer validated "
              "0.7 specifically for time-series transformers. Override if a "
              "specific probe needs the vision default.")
    _add("--fraug", action="store_true",
         help="[B003 R2] Frequency-domain augmentation (FFT mask). "
              "~0 marginal cost; targets ShIC.")
    _add("--fraug-mask-ratio", type=float, default=0.10,
         help="Fraction of frequency components to mask (default 0.10).")
    _add("--fraug-p", type=float, default=0.5,
         help="Probability of applying FrAug per batch (default 0.5).")
    _add("--pcgrad", action="store_true",
         help="[B003 4.6] PCGrad gradient surgery across [aux, ret_1..ret_64]. "
              "Disables AMP (per-task backward incompatible with scaler).")
    _add("--mtp", action="store_true",
         help="[B002 R1] Multi-Token Prediction sequential causal-chain head.")
    _add("--adaptive-bins", action="store_true",
         help="[B001 R3] AdaptiveBucketer log-spaced bins. "
              "Same bin COUNT, denser PLACEMENT near zero.")
    _add("--adaptive-bins-mode", default="log_spaced",
         choices=["log_spaced", "quantile"])
    _add("--mdn", action="store_true",
         help="[B003 R3] MDN head replacement (mutually exclusive with --mtp).")
    _add("--mdn-mode", default="normal", choices=["normal", "skewed_t"])
    _add("--mdn-components", type=int, default=3)

    # B007 additions (post-2026-05-02 browser response)
    _add("--label-noise", action="store_true",
         help="[B007 E2] Calibrated Gaussian label noise during training. "
              "Suppresses noise memorization in low-SNR regression "
              "(arXiv 2510.17526). ~0 marginal compute.")
    _add("--label-noise-ratio", type=float, default=0.5,
         help="sigma_label = ratio * sigma_residual. Default 0.5 per B007 E2.")
    _add("--label-noise-sigma-residual", type=float, default=0.02,
         help="Empirical residual std proxy (default 0.02 = crypto h=1 return scale).")
    _add("--logit-clip", action="store_true",
         help="[B007 §5.2] LogitClip on bin head logits during training "
              "(arXiv 2212.04055). Bounds logit-vector L2 norm; reduces "
              "noisy-label memorization on the TwoHot 255-bin classifier.")
    _add("--logit-clip-tau", type=float, default=4.0,
         help="LogitClip max L2 norm. 4.0 is conservative for 255-bin TwoHot.")
    _add("--run-tag", type=str, default="",
         help="Optional checkpoint suffix for parallel-batch isolation. "
              "Without --run-tag, all variants write the same ckpt path "
              "and clobber each other. The trainer must consume args.run_tag "
              "and embed it in ckpt_prefix; per-version wiring is required "
              "(V1.1 wired 2026-05-02; others may need a small patch).")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class UpgradeContext:
    """Per-step state for the upgrade plumbing.

    Constructed by build_upgrade_context; consumed by do_step_with_upgrades.
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer            # may be SAM-wrapped
    base_optimizer: torch.optim.Optimizer       # pre-SAM optimizer (for grad clip etc.)
    use_amp: bool
    use_sam: bool
    use_pcgrad: bool
    use_mtp: bool
    use_mdn: bool
    use_adaptive_bins: bool
    fraug_module: Optional[nn.Module] = None
    pcgrad_module: Optional[object] = None
    grad_clip: float = 1.0
    horizons: Tuple[int, ...] = (1, 4, 16, 64)
    # B007 additions
    use_label_noise: bool = False
    label_noise_injector: Optional[object] = None
    use_logit_clip: bool = False
    logit_clip_module: Optional[nn.Module] = None


def build_upgrade_context(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    device: str = "cuda",
    grad_clip: float = 1.0,
    horizons: Tuple[int, ...] = (1, 4, 16, 64),
    quantile_returns: Optional[object] = None,
    verbose: bool = True,
) -> UpgradeContext:
    """Apply opt-in upgrades to model + optimizer; return context for step loop."""
    from frontier_ml.v1_upgrades.integration import apply_v1_upgrades

    # Model-side upgrades
    if args.mtp or args.adaptive_bins or args.mdn:
        apply_v1_upgrades(
            model,
            use_mtp=args.mtp,
            use_adaptive_bins=args.adaptive_bins,
            adaptive_bins_mode=args.adaptive_bins_mode,
            quantile_returns=quantile_returns,
            use_mdn=args.mdn, mdn_mode=args.mdn_mode,
            mdn_components=args.mdn_components,
            horizons=horizons,
            verbose=verbose,
        )

    # FrAug
    fraug_module = None
    if args.fraug:
        from frontier_ml.v1_upgrades.fraug import FrAug
        fraug_module = FrAug(mask_ratio=args.fraug_mask_ratio, mode="random",
                              p_aug=args.fraug_p).to(device)
        if verbose:
            print(f"  [B003 R2] FrAug ENABLED  mask_ratio={args.fraug_mask_ratio} "
                  f"p_aug={args.fraug_p}")

    # SAM
    base_optimizer = optimizer
    use_amp = True
    if args.sam:
        from frontier_ml.v1_upgrades.sam import SAM
        # Need to gather params from optimizer's param groups
        params = []
        for g in optimizer.param_groups:
            params.extend(g["params"])
        optimizer = SAM(params, optimizer, rho=args.sam_rho)
        use_amp = False  # SAM + AMP delicate; eager fp32 first revision
        if verbose:
            print(f"  [B003 R1] SAM ENABLED  rho={args.sam_rho}  (AMP off under SAM)")

    # PCGrad
    pcgrad_module = None
    if args.pcgrad:
        from frontier_ml.v1_upgrades.pcgrad import PCGrad
        pcgrad_module = PCGrad()
        use_amp = False  # PCGrad multi-backward incompatible with scaler
        if verbose:
            print(f"  [B003 4.6] PCGrad ENABLED  surgery across "
                  f"[aux, ret_1..ret_{horizons[-1]}]")

    if verbose:
        if args.mtp and getattr(model, "_use_mtp", False):
            print(f"  [B002 R1] MTP head ACTIVE")
        if args.mdn and getattr(model, "_use_mdn", False):
            print(f"  [B003 R3] MDN head ACTIVE  mode={args.mdn_mode}")
        if args.adaptive_bins and hasattr(model, "_original_bucketer"):
            print(f"  [B001 R3] AdaptiveBucketer ACTIVE")

    # B007: label-noise injector
    label_noise_injector = None
    use_label_noise = bool(getattr(args, "label_noise", False))
    if use_label_noise:
        from frontier_ml.v1_upgrades.label_noise import LabelNoiseInjector
        label_noise_injector = LabelNoiseInjector(
            sigma_residual=float(args.label_noise_sigma_residual),
            noise_ratio=float(args.label_noise_ratio),
        )
        if verbose:
            print(f"  [B007 E2] LabelNoise ENABLED  sigma_label="
                  f"{label_noise_injector.sigma_label:.5f} "
                  f"(ratio={args.label_noise_ratio} * res_std={args.label_noise_sigma_residual})")

    # B007: LogitClip module (caller wires into the bin head as appropriate)
    logit_clip_module = None
    use_logit_clip = bool(getattr(args, "logit_clip", False))
    if use_logit_clip:
        from frontier_ml.v1_upgrades.logit_clip import LogitClip
        logit_clip_module = LogitClip(tau=float(args.logit_clip_tau)).to(device)
        if verbose:
            print(f"  [B007 §5.2] LogitClip ENABLED  tau={args.logit_clip_tau}")

    return UpgradeContext(
        model=model,
        optimizer=optimizer,
        base_optimizer=base_optimizer,
        use_amp=use_amp,
        use_sam=args.sam,
        use_pcgrad=args.pcgrad,
        use_mtp=args.mtp,
        use_mdn=args.mdn,
        use_adaptive_bins=args.adaptive_bins,
        fraug_module=fraug_module,
        pcgrad_module=pcgrad_module,
        grad_clip=grad_clip,
        horizons=tuple(horizons),
        use_label_noise=use_label_noise,
        label_noise_injector=label_noise_injector,
        use_logit_clip=use_logit_clip,
        logit_clip_module=logit_clip_module,
    )


def maybe_label_noise(
    ctx: UpgradeContext,
    target: torch.Tensor,
    regime_label: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply calibrated Gaussian label noise to regression targets if enabled.

    Trainers should call this on `target_returns` BEFORE building the loss.
    Identity when --label-noise is OFF.
    """
    if not ctx.use_label_noise or ctx.label_noise_injector is None:
        return target
    return ctx.label_noise_injector(target, regime_label=regime_label)


def maybe_logit_clip(ctx: UpgradeContext, logits: torch.Tensor) -> torch.Tensor:
    """Apply LogitClip to bin-head logits during training if enabled.

    Trainers should call this on TwoHot/CE logits BEFORE the loss kernel.
    Identity when --logit-clip is OFF or when module is in eval mode.
    """
    if not ctx.use_logit_clip or ctx.logit_clip_module is None:
        return logits
    return ctx.logit_clip_module(logits)


def install_logit_clip_hooks(ctx: UpgradeContext, model: nn.Module, verbose: bool = True) -> int:
    """Install forward hooks on model.return_heads so LogitClip activates per-head.

    Returns the number of heads hooked. Skips quietly when --logit-clip is OFF
    or when the model has replaced return_heads with MTP/MDN (no clip needed).
    """
    if not ctx.use_logit_clip or ctx.logit_clip_module is None:
        return 0
    if not (hasattr(model, "return_heads") and isinstance(model.return_heads, nn.ModuleDict)):
        if verbose:
            print(f"  [B007 §5.2] WARN: --logit-clip set but model.return_heads not a "
                  f"ModuleDict (likely MTP/MDN active). LogitClip skipped.")
        return 0

    clip_mod = ctx.logit_clip_module

    def _hook(_mod, _inp, out):
        if model.training:
            return clip_mod(out)
        return out

    n = 0
    for _, head in model.return_heads.items():
        head.register_forward_hook(_hook)
        n += 1
    if verbose:
        print(f"  [B007 §5.2] LogitClip hooks installed on {n} return heads "
              f"(tau={clip_mod.tau})")
    return n


def make_run_tag_suffix(args) -> str:
    """Read args.run_tag (added by add_upgrade_args) and return '_<tag>' or ''.

    Trainers should embed this into ckpt_prefix and log_suffix to isolate
    parallel-batch variants. Without it, multi-variant batches clobber.
    """
    tag = str(getattr(args, "run_tag", "") or "")
    return f"_{tag}" if tag else ""


# ---------------------------------------------------------------------------
# Step plumbing
# ---------------------------------------------------------------------------

def maybe_fraug(ctx: UpgradeContext, obs: torch.Tensor) -> torch.Tensor:
    """Apply FrAug to obs if enabled and model is in train mode."""
    if ctx.fraug_module is None:
        return obs
    ctx.fraug_module.train()
    return ctx.fraug_module(obs)


def compute_loss_components(
    ctx: UpgradeContext,
    get_loss_fn: Callable,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Dict, object, Optional[Dict]]:
    """Wrap model.get_loss to optionally request per-horizon components.

    get_loss_fn is the model's get_loss method (already bound).

    Returns (total, loss_dict, outputs, components_or_None).
    """
    if ctx.use_pcgrad:
        return get_loss_fn(*args, **kwargs, return_components=True)
    res = get_loss_fn(*args, **kwargs)
    if len(res) == 4:
        # Caller passed return_components=True elsewhere; keep components
        return res
    total, loss_dict, outputs = res
    return total, loss_dict, outputs, None


def step_backward_and_update(
    ctx: UpgradeContext,
    loss: torch.Tensor,
    components: Optional[Dict],
    clip_params: List[torch.nn.Parameter],
    scaler: Optional[torch.amp.GradScaler],
    *,
    sam_recompute_fn: Optional[Callable] = None,
) -> float:
    """Execute the right backward + step pattern based on active upgrades.

    Args:
        ctx:                  the UpgradeContext
        loss:                 the total scalar loss (with all aux terms baked in)
        components:           per-task loss tensors dict if PCGrad active, else None
        clip_params:          list of params for clip_grad_norm_
        scaler:               AMP scaler (used only when ctx.use_amp=True and not SAM/PCGrad)
        sam_recompute_fn:     callable(no args) -> (loss2, components2) for SAM's
                              second pass. Required when ctx.use_sam=True.

    Returns:
        grad_norm (float) for logging.
    """
    optimizer = ctx.optimizer
    base_optim = ctx.base_optimizer
    pcgrad = ctx.pcgrad_module

    optimizer.zero_grad(set_to_none=True)

    # Path 1: SAM (always disables AMP)
    if ctx.use_sam:
        # Step 1: backward at w
        if pcgrad is not None and components is not None:
            pc_losses = _pcgrad_loss_list(components, ctx.horizons)
            pcgrad.pc_backward(pc_losses, ctx.model)
        else:
            loss.backward()
        gn = clip_grad_norm_(clip_params, ctx.grad_clip).item()
        optimizer.first_step(zero_grad=True)
        # Step 2: recompute at w + epsilon
        if sam_recompute_fn is None:
            raise RuntimeError("SAM requires sam_recompute_fn to recompute loss at w+eps")
        loss2, components2 = sam_recompute_fn()
        if pcgrad is not None and components2 is not None:
            pc_losses2 = _pcgrad_loss_list(components2, ctx.horizons)
            pcgrad.pc_backward(pc_losses2, ctx.model)
        else:
            loss2.backward()
        clip_grad_norm_(clip_params, ctx.grad_clip)
        optimizer.second_step(zero_grad=False)
        return gn

    # Path 2: PCGrad without SAM (also disables AMP)
    if pcgrad is not None and components is not None:
        pc_losses = _pcgrad_loss_list(components, ctx.horizons)
        pcgrad.pc_backward(pc_losses, ctx.model)
        gn = clip_grad_norm_(clip_params, ctx.grad_clip).item()
        optimizer.step()
        return gn

    # Path 3: standard AMP
    if scaler is not None and ctx.use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = clip_grad_norm_(clip_params, ctx.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
        return gn

    # Path 4: eager fp32 fallback (no AMP, no SAM, no PCGrad)
    loss.backward()
    gn = clip_grad_norm_(clip_params, ctx.grad_clip).item()
    optimizer.step()
    return gn


def _pcgrad_loss_list(components: Dict, horizons: Tuple[int, ...]) -> List[torch.Tensor]:
    """Pack components dict into ordered list for PCGrad."""
    out = [components["aux"]]
    for h in horizons:
        key = f"ret_{h}"
        if key in components:
            out.append(components[key])
    return out
