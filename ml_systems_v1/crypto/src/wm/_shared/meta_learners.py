"""Meta-learner intelligence layer — RESIDUAL implementations.

Round-8 (2026-05-08). Each meta-learner is a TOGGLEABLE OBSERVER that:
  1. Does NOT modify the base model's gradient flow (detached inputs)
  2. Has its own optimizer / state / checkpoints
  3. Produces a DIAGNOSTIC signal or PARALLEL prediction variant
  4. Can be enabled / disabled per training run via flags
  5. When ALL disabled: base trainer is byte-for-byte unchanged

This way one training run produces the BASE model + N residual variants. We
compare base vs each variant on the same val set to identify which meta-
implementation is helping or hurting.

Compute cost (overhead vs base training):
  ShICMonitor          : ~1% (periodic eval, no extra forward)
  MultiHeadConsistency : ~5-10% (3 small heads + consistency loss; trunk shared)
  ProbeCallback        : ~5% (probe runs every K epochs, ~30s out of 5 epochs)
  SlowTeacher          : ~10% (EMA copy in fp16, optional distillation loss)

Designs not implemented in this module (deferred for ROI reasons):
  D. AdversarialValidator: 50% overhead. Worth it if memorization persists.
  E. PopulationBasedTraining: 4× compute. Worth it for HP search; orthogonal
     to per-model meta-learners.
  F. EnsembleMetaNetwork: post-hoc, trains AFTER WMs frozen. Not a per-step
     residual.
"""
from __future__ import annotations

import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_RMSNorm():
    try:
        v1_path = Path(__file__).resolve().parent.parent / "v1" / "v1_0_training"
        if str(v1_path) not in sys.path:
            sys.path.insert(0, str(v1_path))
        from components import RMSNorm
        return RMSNorm
    except Exception:
        class _RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps
            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return _RMSNorm


# =============================================================================
# Design A — In-training ShIC monitor (passive observer, ~1% overhead)
# =============================================================================

class ShICMonitor:
    """Tracks ShIC alongside IC every N steps without affecting training.

    Passive observer: takes the base model and a held-out validation batch,
    computes IC on clean vs time-shuffled features, logs the trajectory.

    Triggers a warning callback when ShIC/IC ratio drops below threshold.
    Does NOT modify gradients or training loss.

    Usage:
        monitor = ShICMonitor(check_every=100, threshold=0.3)
        for step in training_loop:
            train_step(...)
            if monitor.should_check(step):
                metrics = monitor.evaluate(model, val_obs, val_asset, val_tgt)
                if metrics["shic_ratio"] < threshold:
                    print(f"  [ShIC WARN] step {step}: {metrics}")
    """

    def __init__(self, check_every: int = 100, threshold: float = 0.3,
                 warn_callback: Optional[callable] = None):
        self.check_every = check_every
        self.threshold = threshold
        self.warn_callback = warn_callback
        self.trajectory = []   # list of (step, ic, shic, ratio)

    def should_check(self, step: int) -> bool:
        return (step + 1) % self.check_every == 0

    @torch.no_grad()
    def evaluate(self, model, val_obs: torch.Tensor, val_asset: torch.Tensor,
                 val_tgt: torch.Tensor, step: int = 0) -> dict:
        """Compute IC + ShIC on val. val_obs: [B, T, F], val_tgt: [B, T] (h=1)."""
        was_training = model.training
        model.eval()
        try:
            with torch.amp.autocast("cuda"):
                out_clean = model.forward_train(val_obs, val_asset)
                nb = getattr(model, "_num_bins", None) or out_clean["return_logits"][1].shape[-1]
                pred_clean = model.bucketer.decode(
                    out_clean["return_logits"][1].reshape(-1, nb)
                )
                # ShIC: shuffle features per-sample along time axis
                B, T = val_obs.shape[:2]
                val_obs_sh = val_obs.clone()
                for b in range(B):
                    perm = torch.randperm(T, device=val_obs.device)
                    val_obs_sh[b] = val_obs_sh[b, perm]
                out_sh = model.forward_train(val_obs_sh, val_asset)
                pred_sh = model.bucketer.decode(out_sh["return_logits"][1].reshape(-1, nb))

            ic = self._ic(pred_clean, val_tgt)
            shic = self._ic(pred_sh, val_tgt)
            ratio = shic / ic if abs(ic) > 1e-6 else 0.0
            metrics = {"step": step, "ic": ic, "shic": shic, "shic_ratio": ratio}
            self.trajectory.append((step, ic, shic, ratio))
            if ratio < self.threshold and abs(ic) > 0.005 and self.warn_callback:
                self.warn_callback(metrics)
            return metrics
        finally:
            if was_training:
                model.train()

    @staticmethod
    def _ic(pred: torch.Tensor, tgt: torch.Tensor) -> float:
        p = pred.flatten().detach().cpu().numpy()
        t = tgt.flatten().detach().cpu().numpy()
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() < 50:
            return 0.0
        pc = (p[mask] - p[mask].mean()) / (p[mask].std() + 1e-8)
        tc = (t[mask] - t[mask].mean()) / (t[mask].std() + 1e-8)
        return float(np.mean(pc * tc))


# =============================================================================
# Design B — Multi-Head Consistency (residual variant, ~5-10% overhead)
# =============================================================================

class MultiHeadConsistency(nn.Module):
    """Aux heads on a SHARED but DETACHED trunk; consistency penalty as anti-memo.

    Adds three auxiliary prediction heads, each predicting a different target
    derived from the same return:
      - sign_head: binary (return > 0)
      - quantile_head: 3 quantiles (q05/q50/q95)
      - vol_head: |return| (magnitude)

    Each head takes the BASE model's ret_trunk as input but DETACHES IT —
    gradients from these aux losses do NOT flow into the base model. The aux
    heads + their consistency penalty are a SEPARATE optimization problem.

    The CONSISTENCY signal: if sign_head says +, q50_head says positive,
    vol_head says high → all consistent. If they disagree → memorization
    (model learned different patterns for different head losses).

    At inference the base model is unaffected. This module produces a
    PARALLEL diagnostic — we can compare base predictions to consistent-head
    predictions and see which generalize better OOS.

    Compute: 3 small linear heads + 1 KL consistency term. ~5% overhead.

    Returns dict with sign / quantile / vol predictions + consistency_loss
    that's optimized via meta_optimizer.
    """

    def __init__(self, trunk_dim: int, dropout: float = 0.1,
                 quantile_levels: tuple = (0.05, 0.50, 0.95)):
        super().__init__()
        RMSNorm = _get_RMSNorm()
        self.quantile_levels = quantile_levels
        # Sign head (binary)
        self.sign_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            RMSNorm(trunk_dim // 2),
            nn.SiLU(),
            nn.Linear(trunk_dim // 2, 1),
        )
        # Quantile head (3 outputs for q05/q50/q95)
        self.quantile_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            RMSNorm(trunk_dim // 2),
            nn.SiLU(),
            nn.Linear(trunk_dim // 2, len(quantile_levels)),
        )
        # Volatility head (|return|)
        self.vol_head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            RMSNorm(trunk_dim // 2),
            nn.SiLU(),
            nn.Linear(trunk_dim // 2, 1),
            nn.Softplus(),   # positive output for |return|
        )

    def forward(self, ret_trunk: torch.Tensor) -> dict:
        """ret_trunk: [B, T, D] — detached before this is called.

        Returns dict {sign_logits, quantile_preds, vol_preds}.
        """
        return {
            "sign_logits": self.sign_head(ret_trunk).squeeze(-1),     # [B, T]
            "quantile_preds": self.quantile_head(ret_trunk),           # [B, T, 3]
            "vol_preds": self.vol_head(ret_trunk).squeeze(-1),        # [B, T]
        }

    def compute_loss(self, outputs: dict, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """outputs from forward(); target: [B, T] actual returns.

        Returns (total_loss, components_dict).
        """
        # Sign loss
        sign_target = (target > 0).float()
        sign_loss = F.binary_cross_entropy_with_logits(outputs["sign_logits"], sign_target)

        # Quantile (pinball) loss
        q_preds = outputs["quantile_preds"]   # [B, T, 3]
        q_target = target.unsqueeze(-1)       # [B, T, 1]
        q_levels = torch.tensor(self.quantile_levels, device=target.device).view(1, 1, -1)
        diff = q_target - q_preds
        q_loss = torch.maximum(q_levels * diff, (q_levels - 1) * diff).mean()

        # Vol loss
        vol_target = target.abs()
        vol_loss = F.huber_loss(outputs["vol_preds"], vol_target, delta=0.5)

        # Consistency penalty (autocast-safe: use BCEWithLogits + fp32 cast):
        # sign predicted via sign_head ≈ sign of q50 (mid quantile)
        sign_from_q50_target = (q_preds[..., 1].detach() > 0).float()
        # Use logits directly (no sigmoid first) — BCEWithLogits is numerically
        # stable under autocast.
        consistency_q_sign = F.binary_cross_entropy_with_logits(
            outputs["sign_logits"].float(), sign_from_q50_target.float(),
            reduction="mean",
        )
        # vol predicted via vol_head ≈ |q50|
        vol_from_q50 = q_preds[..., 1].abs().detach()
        consistency_v_q = F.huber_loss(
            outputs["vol_preds"].float(), vol_from_q50.float(), delta=0.5,
        )
        consistency_loss = consistency_q_sign + 0.5 * consistency_v_q

        total = sign_loss + q_loss + vol_loss + 0.2 * consistency_loss
        return total, {
            "meta_b_sign": sign_loss.item(),
            "meta_b_quantile": q_loss.item(),
            "meta_b_vol": vol_loss.item(),
            "meta_b_consistency": consistency_loss.item(),
            "meta_b_total": total.item(),
        }


# =============================================================================
# Design C — Probe Callback (passive observer, ~5% overhead per K epochs)
# =============================================================================

class ProbeCallback:
    """Runs the architectural-ceiling probe inside training every K epochs.

    Reuses scripts/probe_architectural_ceiling.py infrastructure. Logs the
    BestIC / ShIC / ratio trajectory across training. Does NOT modify training.

    Use case: detect ceiling drift in real-time. If model's probe ceiling
    drops mid-training, something's wrong (LR too high, β too aggressive,
    optimizer issue).

    Note: the probe re-trains a small synthetic problem on the model. To avoid
    affecting the live training, we save+restore optimizer state.
    """

    def __init__(self, run_every_epochs: int = 5, n_steps: int = 100,
                 target_alpha: float = 0.20):
        self.run_every = run_every_epochs
        self.n_steps = n_steps
        self.target_alpha = target_alpha
        self.trajectory = []   # list of (epoch, best_ic, shic_ratio)

    def should_run(self, epoch: int) -> bool:
        return (epoch + 1) % self.run_every == 0

    def run_probe(self, model, epoch: int, device: str = "cuda") -> dict:
        """Run a quick synthetic probe on the current model state.

        Saves model state, runs probe, restores. Returns metrics.
        """
        # Save state
        state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            # Tiny in-line probe — fresh synthetic batch, brief training
            # We don't import the probe script (avoid module conflicts) but
            # implement a minimal version inline.
            model.train()
            B, T, F_in = 16, 96, model.input_dim if hasattr(model, "input_dim") else 29
            asset = torch.randint(0, 10, (B,), device=device)
            ic_traj = []
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            for s in range(self.n_steps):
                # Synthetic batch with planted signal
                obs = torch.randn(B, T, F_in, device=device)
                w = torch.randn(F_in, device=device)
                w = w / w.norm()
                signal = (obs * w).sum(dim=-1)
                lag = torch.cat([torch.zeros(B, 1, device=device), signal[:, :-1]], dim=1)
                tgt = self.target_alpha * lag + 0.05 * torch.randn_like(lag)
                targets = {1: tgt, 4: tgt, 16: tgt, 64: tgt}
                try:
                    with torch.amp.autocast("cuda"):
                        try:
                            loss, _, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15)
                        except TypeError:
                            loss, _, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15,
                                                         temporal_ctx_drop=0.15)
                    if torch.isfinite(loss).item():
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                except Exception:
                    pass

            # Quick IC measure on a fresh batch
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                obs = torch.randn(B, T, F_in, device=device)
                signal = (obs * w).sum(dim=-1)
                lag = torch.cat([torch.zeros(B, 1, device=device), signal[:, :-1]], dim=1)
                tgt = self.target_alpha * lag
                out = model.forward_train(obs, asset)
                nb = getattr(model, "_num_bins", None) or out["return_logits"][1].shape[-1]
                pred = model.bucketer.decode(out["return_logits"][1].reshape(-1, nb))
                ic = ShICMonitor._ic(pred, tgt)
            metrics = {"epoch": epoch, "probe_ic": ic, "n_steps": self.n_steps}
            self.trajectory.append((epoch, ic))
            return metrics
        finally:
            # Restore base state
            with torch.no_grad():
                for k, v in state_backup.items():
                    model.state_dict()[k].copy_(v)


# =============================================================================
# Design G — Slow-Teacher (EMA distillation, ~10% overhead — optional)
# =============================================================================

class SlowTeacher(nn.Module):
    """EMA teacher + optional distillation loss (residual).

    Maintains an EMA-averaged copy of the base model. The teacher's predictions
    are MORE STABLE than the live model's (averaged over recent timesteps).

    Two modes:
      passive : just track EMA, save teacher checkpoint, no training change
      active  : add distillation loss = α · MSE(student_preds, teacher_preds)
                to the training loss. Forces student to match teacher's
                stable predictions, reducing noise in trajectory.

    The distillation loss flows ONLY into the student, not into the teacher
    (teacher has no gradient — it's the EMA snapshot).
    """

    def __init__(self, base_model: nn.Module, ema_decay: float = 0.999,
                 distill_alpha: float = 0.0):
        super().__init__()
        self.ema_decay = ema_decay
        self.distill_alpha = distill_alpha   # 0.0 = passive (just track), >0 = active
        # Deep copy of base model — separate parameters (won't get gradients)
        self.teacher = deepcopy(base_model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    @torch.no_grad()
    def update_ema(self, base_model: nn.Module):
        """Call after each base model optimizer step."""
        for tp, sp in zip(self.teacher.parameters(), base_model.parameters()):
            tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1 - self.ema_decay)
        for tb, sb in zip(self.teacher.buffers(), base_model.buffers()):
            tb.data.copy_(sb.data)

    def distillation_loss(self, base_outputs: dict,
                          obs_seq: torch.Tensor,
                          asset_id: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Return distill loss term (student matches teacher's predictions).

        Returns (loss, metrics_dict). loss is 0 when distill_alpha == 0
        (passive mode).
        """
        if self.distill_alpha == 0.0:
            zero = torch.tensor(0.0, device=obs_seq.device)
            return zero, {"meta_g_distill": 0.0, "meta_g_alpha": 0.0}
        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = self.teacher.forward_train(obs_seq, asset_id)
            t_pred_logits = t_out["return_logits"][1].detach()
        # Student logits (from base_outputs)
        s_pred_logits = base_outputs["return_logits"][1]
        # Distillation = MSE between probability distributions
        t_probs = F.softmax(t_pred_logits, dim=-1)
        s_log_probs = F.log_softmax(s_pred_logits, dim=-1)
        distill = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        return self.distill_alpha * distill, {
            "meta_g_distill": distill.item(),
            "meta_g_alpha": self.distill_alpha,
        }


# =============================================================================
# Variant tracker — collects per-meta metrics across training
# =============================================================================

class MetaVariantTracker:
    """Aggregates per-meta metrics across training for end-of-run comparison.

    Each registered meta has its own checkpoint + IC trajectory. At training
    end, we compare base IC vs each variant IC to identify which meta is
    helping or hurting.
    """

    def __init__(self):
        self.metas: dict[str, dict] = {}    # name -> {"obj": meta_obj, "metrics": [...]}

    def register(self, name: str, meta_obj: Any, ckpt_dir: Path):
        self.metas[name] = {
            "obj": meta_obj,
            "metrics": [],
            "ckpt_dir": ckpt_dir,
        }

    def log_metrics(self, name: str, step_or_epoch: int, metrics: dict):
        if name in self.metas:
            self.metas[name]["metrics"].append((step_or_epoch, metrics))

    def save_meta_checkpoint(self, name: str, base_features: int, version: str):
        meta = self.metas.get(name)
        if meta is None:
            return
        ckpt_dir = meta["ckpt_dir"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{version}_f{base_features}_meta_{name}.pt"
        if hasattr(meta["obj"], "state_dict"):
            torch.save({
                "state_dict": meta["obj"].state_dict(),
                "metrics": meta["metrics"],
            }, path)

    def summary(self) -> dict:
        """Return per-meta summary metrics."""
        out = {}
        for name, info in self.metas.items():
            if info["metrics"]:
                last = info["metrics"][-1][1]
                out[name] = last
        return out


__all__ = [
    "ShICMonitor",
    "MultiHeadConsistency",
    "ProbeCallback",
    "SlowTeacher",
    "MetaVariantTracker",
]
