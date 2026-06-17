"""
Frontier Loss Mixin (v2, SOTA 2026-04-13)
============================================

Drop-in mixin adding ceiling-breaking losses to ANY world model.

Usage: call init_frontier() in __init__, add_frontier_losses() in get_loss().

v2 upgrades:
- CurriculumWeighter: loss-conditioned difficulty (CRUCIAL-style)
- ContrastiveRegimeLoss: ProtoNCE with regime prototypes
- TradeWorthinessHead: asymmetric cost-sensitive CE (c_FP=3*c_FN)
- VolatilityHead: QLIKE loss (asymmetric vol underestimation penalty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

import sys
from pathlib import Path
_comp_path = str(Path(__file__).resolve().parent)
if _comp_path not in sys.path:
    sys.path.insert(0, _comp_path)

from frontier import (
    CurriculumWeighter, ContrastiveRegimeLoss,
    TradeWorthinessHead, VolatilityHead,
)


class FrontierLossMixin:
    """Mixin that adds frontier ceiling-breaking losses to any model.

    Call init_frontier() in __init__ and add_frontier_losses() in get_loss().
    """

    def init_frontier(self, d_model: int, cost_threshold: float = 0.0012,
                      dropout: float = 0.1, fp_fn_ratio: float = 3.0):
        """Initialize frontier components. Call in model __init__."""
        self.curriculum = CurriculumWeighter(scale=5.0, max_weight=10.0)
        self.contrastive = ContrastiveRegimeLoss(temperature=0.1, momentum=0.99)
        self.trade_head = TradeWorthinessHead(
            d_model, cost_threshold, dropout=dropout, fp_fn_ratio=fp_fn_ratio
        )
        self.vol_head = VolatilityHead(d_model, dropout=dropout)

        self.frontier_weights = {
            "contrastive": 0.5,
            "trade_worthiness": 1.0,
            "volatility": 0.5,
        }

    def add_frontier_losses(self, total: torch.Tensor, loss_dict: dict,
                             outputs: dict, targets: dict,
                             obs_seq: torch.Tensor = None,
                             regime_labels: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """Add frontier losses to existing total loss."""
        h_seq = outputs.get("h_seq")
        if h_seq is None:
            return total, loss_dict

        # 1. Contrastive regime loss (ProtoNCE)
        if regime_labels is not None and self.training:
            l_contrast = self.contrastive(h_seq, regime_labels)
            total = total + self.frontier_weights["contrastive"] * l_contrast
            loss_dict["frontier_contrastive"] = l_contrast.item()

        # 2. Trade worthiness (asymmetric cost-sensitive CE on h=1 returns)
        if 1 in targets:
            tw_logits = self.trade_head(h_seq)
            l_tw, tw_acc = self.trade_head.compute_loss(tw_logits, targets[1])
            total = total + self.frontier_weights["trade_worthiness"] * l_tw
            loss_dict["frontier_tw"] = l_tw.item()
            loss_dict["frontier_tw_acc"] = tw_acc
            outputs["trade_logits"] = tw_logits

        # 3. Volatility prediction (QLIKE loss)
        if 1 in targets:
            pred_vol = self.vol_head(h_seq)
            l_vol = self.vol_head.compute_loss(pred_vol, targets[1])
            total = total + self.frontier_weights["volatility"] * l_vol
            loss_dict["frontier_vol"] = l_vol.item()
            outputs["pred_vol"] = pred_vol

        return total, loss_dict
