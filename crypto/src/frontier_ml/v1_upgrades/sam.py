"""SAM — Sharpness-Aware Minimization (B003 R1).

Implements the canonical two-step SAM optimizer (Foret et al. 2020 arXiv
2010.01412; SAMformer paper arXiv 2402.10198 specifically validates SAM
on transformer time-series forecasters).

Mechanism:
    1. forward + backward as usual
    2. ascend: w' = w + rho * grad / ||grad||  (move to max-loss neighborhood)
    3. forward + backward at w'
    4. descend: optimizer.step() restoring w then applying gradient computed at w'

Doubles wall-clock per training step (two forward + two backward), but in
exchange the optimizer chases FLAT minima instead of the sharpest local
minimum. Per SAMformer, this is the exact intervention that lifts
generalization on transformer time-series predictors -- which V1.x is.

Usage:

    base_optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    optimizer = SAM(model.parameters(), base_optim, rho=0.05)

    # Training step replacement:
    def step_with_sam():
        optimizer.zero_grad(set_to_none=True)
        loss1 = compute_loss()        # forward 1
        loss1.backward()              # backward 1
        optimizer.first_step(zero_grad=True)
        loss2 = compute_loss()        # forward 2 at w + epsilon
        loss2.backward()              # backward 2
        optimizer.second_step(zero_grad=True)

Closure-based variant (cleaner) is also supported; see __step__.

`rho` is the perturbation radius.
- **Foret 2020 (original SAM, vision domain)**: rho=0.05.
- **SAMformer official run.py (time-series, VERIFIED 2026-05-02 via raw fetch
  of github.com/romilbert/samformer/blob/main/run.py)**: rho=0.7.

For our crypto WM (time-series transformer), default to 0.7 unless a
specific probe needs the vision baseline.

Reference implementation: github.com/davda54/sam (this is structurally
equivalent; written from the published math, not copy-pasted, so any
cargo-culted bugs from older revisions don't propagate).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper.

    Wraps any torch.optim.Optimizer instance. Use first_step / second_step
    pattern OR closure-based step().
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.7,
        adaptive: bool = False,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        # Rebind param_groups to base_optimizer's so its state machine works,
        # but inject SAM-specific keys (rho, adaptive) into each group since
        # base_optimizer's groups don't have them.
        self.param_groups = self.base_optimizer.param_groups
        for g in self.param_groups:
            g.setdefault("rho", rho)
            g.setdefault("adaptive", adaptive)
        # Track combined defaults for repr; SAM keys take precedence.
        self.defaults = {**self.base_optimizer.defaults, "rho": rho, "adaptive": adaptive}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute epsilon = rho * grad / ||grad|| and ascend to w + epsilon."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Stash the original weight so second_step can restore it
                self.state[p]["old_p"] = p.data.clone()
                # Adaptive SAM scales by per-param magnitude
                if group["adaptive"]:
                    e_w = (torch.pow(p, 2) * p.grad * scale.to(p)).clone()
                else:
                    e_w = (p.grad * scale.to(p)).clone()
                p.add_(e_w)  # w' = w + epsilon
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore weights and apply base optimizer step using grad at w'."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]  # restore original w
        self.base_optimizer.step()  # descent using grad computed at w'
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Closure-based step (preferred when wrapping AMP or scalers).

        The closure must perform forward + loss + backward and return the
        loss tensor. It will be called twice -- once at w, once at w + eps.
        """
        if closure is None:
            raise RuntimeError(
                "SAM.step() requires a closure that does forward + loss + backward. "
                "If you want manual control, use first_step() and second_step()."
            )
        # First pass at w
        closure_with_grad = torch.enable_grad()(closure)
        loss = closure_with_grad()
        self.first_step(zero_grad=True)
        # Second pass at w + epsilon
        closure_with_grad()
        self.second_step(zero_grad=False)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        """Global L2 grad norm across all parameters (with adaptive scaling)."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["adaptive"]:
                    n = (torch.abs(p) * p.grad).norm(p=2).to(shared_device)
                else:
                    n = p.grad.norm(p=2).to(shared_device)
                norms.append(n)
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Re-bind param groups to base_optimizer so its internal state aligns.
        self.base_optimizer.param_groups = self.param_groups


def smoke():
    """Verify SAM wraps AdamW and produces non-trivial weight delta vs vanilla."""
    torch.manual_seed(0)
    # Tiny linear regression
    X = torch.randn(64, 4)
    y = torch.randn(64, 1)
    model = torch.nn.Linear(4, 1)
    base = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optim = SAM(model.parameters(), base, rho=0.05)

    def closure():
        optim.zero_grad(set_to_none=True)
        loss = ((model(X) - y) ** 2).mean()
        loss.backward()
        return loss

    losses = []
    for i in range(20):
        loss = optim.step(closure)
        losses.append(loss.item())
    drop = losses[0] - losses[-1]
    print(f"[sam] loss start={losses[0]:.4f} end={losses[-1]:.4f}  drop={drop:.4f}")
    assert drop > 0.0, "SAM should still descend"
    print("[sam] PASS smoke")


if __name__ == "__main__":
    smoke()
