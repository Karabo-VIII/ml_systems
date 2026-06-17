"""Meta-runtime helper — minimal-boilerplate residual meta wiring.

Round-9 (2026-05-08). Lets every V-trainer add `--meta` support with ~10
lines of code, instead of duplicating V25's full wiring.

USAGE in a trainer:

    # 1. Import and add args
    from meta_runtime import MetaRuntime, add_meta_args
    add_meta_args(parser)   # adds --meta and --meta-distill-alpha

    # 2. After model is constructed, instantiate runtime
    meta_rt = MetaRuntime.from_args(args, model, MODEL_DIR,
                                     trunk_dim=RETURN_HEAD_DIM,
                                     device=DEVICE,
                                     version="v22")

    # 3. In training step, after base step:
    meta_rt.train_step_residual(model, base_outputs, targets_gpu, step)

    # 4. At end of each epoch:
    meta_rt.epoch_end(model, epoch)

    # 5. At end of training:
    meta_rt.save_and_summarize(version="v22", n_features=input_dim)

When --meta is empty (default): all four hooks are NO-OPS. Base trainer
is byte-for-byte unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

# Add _shared to path so we can import meta_learners
_self_dir = Path(__file__).resolve().parent
if str(_self_dir) not in sys.path:
    sys.path.insert(0, str(_self_dir))

from meta_learners import (
    ShICMonitor, MultiHeadConsistency, ProbeCallback, SlowTeacher,
    MetaVariantTracker,
)


def add_meta_args(parser):
    """Add --meta and --meta-distill-alpha to an argparse parser."""
    parser.add_argument(
        "--meta", type=str, default="",
        help="Comma-separated meta-learner residuals: shic_monitor, "
             "multi_head, probe_callback, slow_teacher. Default empty = "
             "base training unchanged."
    )
    parser.add_argument(
        "--meta-distill-alpha", type=float, default=0.0,
        help="If slow_teacher enabled, alpha for distillation loss. "
             "0.0 = passive EMA tracking only; >0 = active distill."
    )


class MetaRuntime:
    """Per-training-run meta-learner manager.

    All hooks are NO-OPS when no metas are enabled, so trainers can call them
    unconditionally without affecting the base trajectory.
    """

    def __init__(self, args, model, model_dir: Path, trunk_dim: int,
                 device: str, version: str = "vX"):
        self.version = version
        self.device = device
        self.model_dir = Path(model_dir)
        self.tracker = MetaVariantTracker()
        flags_str = getattr(args, "meta", "") or ""
        self.flags = set(f.strip() for f in flags_str.split(",") if f.strip())
        self.shic = None
        self.multi_head = None
        self.multi_head_optimizer = None
        self.probe = None
        self.slow_teacher = None
        self.log_freq = 100   # how often to log meta metrics

        if not self.flags:
            return   # base unchanged

        if "shic_monitor" in self.flags:
            self.shic = ShICMonitor(
                check_every=100, threshold=0.3,
                warn_callback=lambda m: print(f"  [META-A WARN] {m}"),
            )
            self.tracker.register("shic_monitor", self.shic, self.model_dir)
            print("  [META] Design A - ShICMonitor (passive, ~1% overhead)")

        if "multi_head" in self.flags:
            try:
                self.multi_head = MultiHeadConsistency(trunk_dim=trunk_dim).to(device)
                self.multi_head_optimizer = torch.optim.AdamW(
                    self.multi_head.parameters(), lr=3e-4, weight_decay=1e-4
                )
                self.tracker.register("multi_head", self.multi_head, self.model_dir)
                n_p = sum(p.numel() for p in self.multi_head.parameters())
                print(f"  [META] Design B - MultiHeadConsistency ({n_p:,} aux params)")
            except Exception as e:
                print(f"  [META] B failed to init: {str(e)[:60]}; skipping")
                self.multi_head = None

        if "probe_callback" in self.flags:
            self.probe = ProbeCallback(run_every_epochs=5, n_steps=80)
            self.tracker.register("probe_callback", self.probe, self.model_dir)
            print("  [META] Design C - ProbeCallback (every 5 epochs)")

        if "slow_teacher" in self.flags:
            try:
                alpha = float(getattr(args, "meta_distill_alpha", 0.0) or 0.0)
                self.slow_teacher = SlowTeacher(
                    model, ema_decay=0.999, distill_alpha=alpha
                ).to(device)
                self.tracker.register("slow_teacher", self.slow_teacher, self.model_dir)
                mode = "passive EMA" if alpha == 0 else f"active distill alpha={alpha}"
                print(f"  [META] Design G - SlowTeacher ({mode})")
            except Exception as e:
                print(f"  [META] G failed to init: {str(e)[:60]}; skipping")
                self.slow_teacher = None

    def add_distill_to_loss(self, base_loss: torch.Tensor, base_outputs: dict,
                             obs: torch.Tensor, asset: torch.Tensor) -> tuple:
        """If slow_teacher is active (alpha > 0), add distill term to base loss.

        Returns (loss, distill_metrics_dict). When inactive: returns base_loss + {}.
        """
        if self.slow_teacher is None or self.slow_teacher.distill_alpha <= 0:
            return base_loss, {}
        try:
            distill_loss, metrics = self.slow_teacher.distillation_loss(
                base_outputs, obs, asset
            )
            return base_loss + distill_loss, metrics
        except Exception as e:
            return base_loss, {"meta_g_distill_err": str(e)[:40]}

    def train_step_residual(self, model, base_outputs: dict, targets: dict, step: int):
        """Run residual meta computations after the base train step.

        Multi-head Consistency: uses DETACHED ret_trunk; backward into aux heads only.
        Slow Teacher: EMA update.
        ShIC: periodic check using a slice of current batch.

        SAFE to call even when no metas enabled (no-op).
        """
        # Design B: multi-head residual
        if self.multi_head is not None and self.multi_head_optimizer is not None:
            try:
                with torch.amp.autocast("cuda"):
                    if "ret_trunk" not in base_outputs:
                        return
                    rt = base_outputs["ret_trunk"]
                    if rt is None:
                        return
                    aux_out = self.multi_head(rt.detach())
                    tgt_h1 = targets.get(1, None)
                    if tgt_h1 is None:
                        return
                    aux_loss, aux_metrics = self.multi_head.compute_loss(aux_out, tgt_h1)
                self.multi_head_optimizer.zero_grad(set_to_none=True)
                aux_loss.backward()
                self.multi_head_optimizer.step()
                if step % self.log_freq == 0:
                    self.tracker.log_metrics("multi_head", step, aux_metrics)
            except Exception as e:
                if step == 0:
                    print(f"  [META-B] step error: {str(e)[:60]}")

        # Design G: slow teacher EMA update
        if self.slow_teacher is not None:
            try:
                self.slow_teacher.update_ema(model)
            except Exception as e:
                if step == 0:
                    print(f"  [META-G] EMA error: {str(e)[:60]}")

        # Design A: ShIC monitor (periodic; uses small batch slice)
        if self.shic is not None and self.shic.should_check(step):
            try:
                # We don't have direct access to val batch here; use the
                # last training batch's first few samples as a proxy.
                # Real trainers can pass their val batch directly via .check_shic()
                pass
            except Exception:
                pass

    def check_shic(self, model, val_obs: torch.Tensor, val_asset: torch.Tensor,
                   val_tgt: torch.Tensor, step: int):
        """Explicit ShIC check — call from trainer's val loop."""
        if self.shic is None:
            return
        try:
            metrics = self.shic.evaluate(model, val_obs, val_asset, val_tgt, step=step)
            self.tracker.log_metrics("shic_monitor", step, metrics)
        except Exception as e:
            if step == 0:
                print(f"  [META-A] check error: {str(e)[:60]}")

    def epoch_end(self, model, epoch: int):
        """End-of-epoch hooks (currently: probe callback)."""
        if self.probe is not None and self.probe.should_run(epoch):
            try:
                metrics = self.probe.run_probe(model, epoch=epoch, device=self.device)
                self.tracker.log_metrics("probe_callback", epoch, metrics)
                ic = metrics.get("probe_ic", 0)
                print(f"    [META-C] probe IC at epoch {epoch+1}: {ic:.4f}")
            except Exception as e:
                print(f"    [META-C] probe failed: {str(e)[:60]}")

    def save_and_summarize(self, version: str, n_features: int):
        """End-of-training: save residual variant checkpoints + print summary."""
        if not self.flags:
            return
        print()
        print("  [META] Residual variant summary:")
        for name, info in self.tracker.metas.items():
            self.tracker.save_meta_checkpoint(name, n_features, version)
            n_metrics = len(info["metrics"])
            print(f"    {name}: {n_metrics} metric snapshots saved")
            if info["metrics"]:
                last = info["metrics"][-1][1]
                summary_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in last.items()
                    if isinstance(v, (int, float)) and abs(v) < 1e6
                )
                print(f"      last: {summary_str}")

    @classmethod
    def from_args(cls, args, model, model_dir: Path, trunk_dim: int,
                  device: str, version: str = "vX"):
        return cls(args, model, model_dir, trunk_dim, device, version)


__all__ = ["MetaRuntime", "add_meta_args"]
