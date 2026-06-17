"""Integration helpers between V1.x trainer and frontier_ml/v1_upgrades modules.

Provides a single `apply_v1_upgrades(model, **flags)` that mutates a
TransformerWorldModel instance in-place to enable the requested upgrades.

The trainer calls this once after model construction; subsequent forward
passes through the model produce the upgraded outputs.

For the trainer-side step-loop hooks (SAM two-step, FrAug aug, PCGrad
backward), the trainer manages those directly via the per-flag pattern
already in place; this module only handles the model-internal swaps.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch.nn as nn

# Ensure the v1_upgrades package is importable
_PKG = Path(__file__).resolve().parents[3]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from frontier_ml.v1_upgrades.mtp_head import MTPHead
from frontier_ml.v1_upgrades.mdn_head import NormalMDNHead, SkewedStudentTHead


def apply_v1_upgrades(
    model: nn.Module,
    *,
    use_mtp: bool = False,
    use_adaptive_bins: bool = False,
    adaptive_bins_mode: str = "log_spaced",   # or "quantile"
    quantile_returns: object = None,           # numpy array, used if mode="quantile"
    use_mdn: bool = False,
    mdn_mode: str = "normal",                  # "normal" or "skewed_t"
    mdn_components: int = 3,
    horizons: tuple = (1, 4, 16, 64),
    verbose: bool = True,
) -> dict:
    """Mutate a TransformerWorldModel-class instance to enable upgrades.

    Returns a dict reporting what was applied.
    """
    applied = {}

    if use_mtp:
        # Find ret_dim from one of the existing return heads
        if not hasattr(model, "return_heads") or len(model.return_heads) == 0:
            raise RuntimeError(
                "Cannot apply MTP: model has no return_heads attribute"
            )
        # Each head is Sequential; first Linear's in_features is ret_dim
        first_h = list(model.return_heads.values())[0]
        in_dim = None
        for layer in first_h:
            if isinstance(layer, nn.Linear):
                in_dim = layer.in_features
                break
        if in_dim is None:
            raise RuntimeError("Could not infer ret_dim from existing return_heads")
        # num_bins from existing head's last Linear
        last_lin = None
        for layer in first_h:
            if isinstance(layer, nn.Linear):
                last_lin = layer  # last Linear in the Sequential
        num_bins = last_lin.out_features

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        mtp = MTPHead(d_model=in_dim, num_bins=num_bins,
                      horizons=tuple(horizons), share_head=True).to(device=device, dtype=dtype)
        model.mtp_head = mtp
        model._use_mtp = True
        applied["mtp"] = {
            "d_model": in_dim, "num_bins": num_bins,
            "horizons": tuple(horizons),
            "params_added": sum(p.numel() for p in mtp.parameters()),
        }
        if verbose:
            print(f"  [v1-upgrades] MTP head ATTACHED  d_model={in_dim} bins={num_bins} "
                  f"params+={applied['mtp']['params_added']:,}")

    if use_adaptive_bins:
        # Replace model.bucketer with AdaptiveBucketer (drop-in API).
        # CRITICAL: keep bin COUNT identical to model's existing bucketer so
        # the heads' output dimension still aligns with bucket count.
        # Adaptive bins changes PLACEMENT (denser near zero), not count.
        from frontier_ml.foundation.adaptive_bins import (
            make_log_spaced_bucketer, make_quantile_bucketer,
        )
        device = next(model.parameters()).device
        target_n_bins = int(getattr(model.bucketer, "num_bins", 255))
        if adaptive_bins_mode == "quantile":
            if quantile_returns is None:
                raise RuntimeError(
                    "adaptive_bins_mode='quantile' requires quantile_returns array"
                )
            new_bucketer = make_quantile_bucketer(quantile_returns, n_bins=target_n_bins,
                                                    device=str(device))
        else:
            new_bucketer = make_log_spaced_bucketer(n_bins=target_n_bins,
                                                      device=str(device))
        # Stash old bucketer for resume / fallback
        model._original_bucketer = model.bucketer
        model.bucketer = new_bucketer
        applied["adaptive_bins"] = {
            "mode": adaptive_bins_mode,
            "num_bins": new_bucketer.num_bins,
            "min_val": new_bucketer.min_val,
            "max_val": new_bucketer.max_val,
        }
        if verbose:
            print(f"  [v1-upgrades] AdaptiveBucketer ATTACHED  mode={adaptive_bins_mode} "
                  f"bins={new_bucketer.num_bins} range=[{new_bucketer.min_val:.5f}, "
                  f"{new_bucketer.max_val:.5f}]")

    if use_mdn:
        # MDN replaces TwoHot heads entirely. Mutually exclusive with MTP for now
        # (MDN expects (B, T, D) -> (B, T, K params); doesn't fit MTPHead's
        # bin-output causal-chain). Caller should not enable both.
        if use_mtp:
            raise RuntimeError("--mdn and --mtp are mutually exclusive at this revision")
        if not hasattr(model, "return_heads") or len(model.return_heads) == 0:
            raise RuntimeError("Cannot apply MDN: model has no return_heads attribute")
        first_h = list(model.return_heads.values())[0]
        in_dim = None
        for layer in first_h:
            if isinstance(layer, nn.Linear):
                in_dim = layer.in_features
                break
        if in_dim is None:
            raise RuntimeError("Could not infer ret_dim from existing return_heads")
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        cls = SkewedStudentTHead if mdn_mode == "skewed_t" else NormalMDNHead
        mdn_module = nn.ModuleDict({
            str(h): cls(d_in=in_dim, n_components=mdn_components).to(device=device, dtype=dtype)
            for h in horizons
        })
        # Stash original heads for resume / fallback
        model._original_return_heads = model.return_heads
        model.return_heads = mdn_module
        model._use_mdn = True
        model._mdn_mode = mdn_mode
        applied["mdn"] = {
            "mode": mdn_mode,
            "components": mdn_components,
            "params_total": sum(p.numel() for p in mdn_module.parameters()),
        }
        if verbose:
            print(f"  [v1-upgrades] MDNHead ATTACHED  mode={mdn_mode} "
                  f"K={mdn_components}  params={applied['mdn']['params_total']:,}  "
                  f"(get_loss MDN-aware path active)")

    return applied
