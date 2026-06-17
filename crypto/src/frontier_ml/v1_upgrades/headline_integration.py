"""Headline-tier component integration for V1.x (CC-H1/H2/H5/H6/H7).

Mirrors `apply_v1_upgrades`: mutate a TransformerWorldModel in-place to ATTACH the
Headline components (built + smoke-passing in `src/wm/_shared/headline_components.py`),
guarded by `_use_*` flags the forward/loss check. DEFAULT OFF -> base V1.1 behavior
unchanged (zero regression risk to the validated baseline).

Components wired:
  CC-H1 MultiResolutionEncoder  -> swaps obs_encoder (1/4/16-bar causal context)
  CC-H2 LinearAttentionBlock    -> appended after the transformer stack (long-context)
  CC-H5 QuantileHeads           -> auxiliary pinball head on feat (distributional)
  CC-H6 RegimeConditionalHeads  -> auxiliary regime-blended return head on feat
  CC-H7 dream                   -> flag only; trainer adds a dream-rollout aux loss using
                                   the model's NATIVE dream_step (returns return_preds)

The trainer calls this once after model construction (+ apply_v1_upgrades); forward/loss
then produce the upgraded outputs. Stores nothing on disk; checkpoint round-trips via the
attached submodules' state_dicts (strict=False on load handles the extra keys).
"""
from __future__ import annotations
from pathlib import Path
import sys

import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from wm._shared.headline_components import (
    MultiResolutionEncoder, LinearAttentionBlock, QuantileHeads, RegimeConditionalHeads,
)


def apply_headline_upgrades(
    model: nn.Module,
    *,
    use_multires: bool = False,
    use_linattn: bool = False,
    use_quantile: bool = False,
    use_regime_cond: bool = False,
    use_dream: bool = False,
    horizons: tuple = (1, 4, 16, 64),
    verbose: bool = True,
) -> dict:
    """Attach the requested CC-H components to a TransformerWorldModel in-place.

    Returns a dict reporting what was applied (params added per component).
    """
    applied = {}
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    d_model = int(model.d_model)
    input_dim = int(model.input_dim)
    head_input_dim = int(d_model + model.flat_dim)              # 320 + 576 = 896 (read dynamically)
    num_assets = int(model.asset_embedding.num_embeddings)
    num_bins = int(getattr(model.bucketer, "num_bins", 255))
    n_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)

    if use_multires:
        model.multires_encoder = MultiResolutionEncoder(
            input_dim, d_model, num_assets=num_assets).to(device=device, dtype=dtype)
        model._use_multires = True
        applied["multires"] = {"params": sum(p.numel() for p in model.multires_encoder.parameters())}

    if use_linattn:
        model.linattn_block = LinearAttentionBlock(
            d_model, n_heads=n_heads, n_features=d_model).to(device=device, dtype=dtype)
        model._use_linattn = True
        applied["linattn"] = {"params": sum(p.numel() for p in model.linattn_block.parameters())}

    if use_quantile:
        model.quantile_heads = QuantileHeads(
            head_input_dim, horizons=tuple(horizons)).to(device=device, dtype=dtype)
        model._use_quantile = True
        applied["quantile"] = {"params": sum(p.numel() for p in model.quantile_heads.parameters())}

    if use_regime_cond:
        model.regime_cond_heads = RegimeConditionalHeads(
            head_input_dim, horizons=tuple(horizons), num_bins=num_bins).to(device=device, dtype=dtype)
        model._use_regime_cond = True
        applied["regime_cond"] = {"params": sum(p.numel() for p in model.regime_cond_heads.parameters())}

    if use_dream:
        model._use_dream = True                                 # trainer adds the dream-rollout aux loss
        applied["dream"] = {"native_dream_step": hasattr(model, "dream_step")}

    if verbose:
        for k, v in applied.items():
            p = v.get("params")
            print(f"  [headline] CC-H {k:12s} ATTACHED" + (f"  params+={p:,}" if p else "  (flag)"))
    return applied
