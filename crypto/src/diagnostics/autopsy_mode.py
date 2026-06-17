"""Autopsy Mode -- model internals diagnostic for high-risk training.

Complements `feature_autopsy.FeatureAutopsy` (feature-level: which inputs
matter). This module is INTERNALS-level: where activations explode, where
gradients vanish, where embeddings collapse, where loss-components diverge.

Designed for V22 (iTransformer + memorization vector), V25 (frontier
synthesis), V3/V11/V14/V23/V24 (high-experimental architectures).

What it captures:

  Per-layer per-step (sampled every N steps; cheap):
    - act_mean, act_std, act_absmax, dead_unit_frac (std<1e-6)
    - nan_count, inf_count, grad_norm, grad_absmax, nan_grad_count
    - weight stable-rank approximation (linear layers only)

  Per-step loss-component breakdown:
    - one row per call to step(); rolling window of last K steps detects
      explosions / oscillations / degenerate (NaN, identical) terms

  Triggered snapshots (auto-fire on):
    1. NaN/inf in any layer activation OR gradient
    2. Loss component > 3 std-dev above its rolling mean (explosion)
    3. ShIC/IC ratio drop below threshold (memorization)
    4. Embedding spectral collapse (top singular value > 100x median)

  Memorization probe (manual call):
    - Shuffled-batch forward pass: feed time-shuffled obs through model;
      shuffled-IC near zero with high contiguous-IC = memorization confirmed.
    - Per-feature input gradient on a batch + cosine of embedding rows.

Output:
    logs/{run_dir}/autopsy_mode_{ts}.jsonl      -- per-step records
    logs/{run_dir}/autopsy_snapshot_{step}_{reason}.json  -- triggered dumps

Hook safety:
    - All hook callbacks wrap their work in try/except; never crash training.
    - On NaN detection in a hook, the hook records the event but does NOT
      raise (training loop's grad-clip + NaN-recovery decides how to handle).
    - close() unregisters all hooks before final flush.

CLI integration pattern:
    parser.add_argument("--autopsy", action="store_true",
                        help="Enable internals-level autopsy mode (logs/run_dir/)")

    if args.autopsy:
        autopsy = AutopsyMode(model, log_dir=LOG_DIR / f"v{ver}",
                              run_tag=run_tag, sample_every=50,
                              loss_window=200)
    ...
    if autopsy is not None:
        autopsy.step(step_idx, loss_components={...})
        if step_idx % 500 == 0:
            autopsy.memorization_probe(model, val_batch)
    ...
    if autopsy is not None:
        autopsy.close()
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn


__contract__ = {
    "kind": "diagnostic",
    "stage": "wm_training",
    "inputs": {"args": ["model", "log_dir", "run_tag",
                         "sample_every (int)", "loss_window (int)"]},
    "outputs": {"side_effects": "JSONL records + triggered snapshots"},
    "invariants": {
        "no_training_crash_on_hook_error": True,
        "loss_explosion_triggers_snapshot": True,
        "nan_inf_triggers_snapshot": True,
    },
    "rationale": ("Internals-level diagnostic for high-risk WM training. "
                   "Complements feature_autopsy.FeatureAutopsy."),
}


def _safe_float(x: Any) -> float:
    """Return float(x) or 0.0 on any failure (incl. NaN-isfinite)."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _tensor_health(t: torch.Tensor) -> dict:
    """Cheap health snapshot of an activation/gradient tensor."""
    if t is None or t.numel() == 0:
        return {"n": 0}
    with torch.no_grad():
        t_f = t.detach()
        try:
            n_nan = int(torch.isnan(t_f).sum().item())
        except Exception:
            n_nan = 0
        try:
            n_inf = int(torch.isinf(t_f).sum().item())
        except Exception:
            n_inf = 0
        # Replace NaN/Inf with 0 for stat computation so single bad cell
        # doesn't poison mean/std summaries.
        if n_nan or n_inf:
            t_f = torch.where(torch.isfinite(t_f), t_f, torch.zeros_like(t_f))
        try:
            mean = _safe_float(t_f.float().mean())
            std = _safe_float(t_f.float().std(unbiased=False))
            absmax = _safe_float(t_f.float().abs().max())
        except Exception:
            mean, std, absmax = 0.0, 0.0, 0.0
        # Dead-unit fraction: along the last dim, fraction whose std < 1e-6
        try:
            if t_f.dim() >= 2:
                last_std = t_f.float().reshape(-1, t_f.shape[-1]).std(dim=0, unbiased=False)
                dead = _safe_float((last_std < 1e-6).float().mean())
            else:
                dead = 0.0
        except Exception:
            dead = 0.0
    return {
        "n": int(t.numel()),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "absmax": round(absmax, 6),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "dead_frac": round(dead, 4),
    }


def _stable_rank(weight: torch.Tensor) -> float:
    """sum(sigma^2) / max(sigma)^2 — stable-rank approximation."""
    if weight is None or weight.dim() != 2:
        return 0.0
    try:
        with torch.no_grad():
            w = weight.detach().float()
            # Skip if huge; SVD is O(min(m,n)^3 * max). Cap at 4096 dim.
            m, n = w.shape
            if max(m, n) > 4096:
                return -1.0  # sentinel: skipped
            sv = torch.linalg.svdvals(w)
            sv2 = (sv ** 2).sum().item()
            top2 = (sv.max() ** 2).item()
            return _safe_float(sv2 / max(top2, 1e-12))
    except Exception:
        return 0.0


class AutopsyMode:
    """Internals-level diagnostic logger.

    Hooks the model's `nn.Module` tree and records per-layer activation +
    gradient health every `sample_every` steps. Records per-step loss
    components; triggers snapshot dumps on anomalies.
    """

    # Layer types we track. Tuned to surface the highest-leverage layers
    # first; subclasses (Mamba blocks, RSSM cells, custom attention) inherit.
    TRACKED_TYPES = (nn.Linear, nn.Conv1d, nn.LayerNorm,
                      nn.MultiheadAttention, nn.Embedding)

    def __init__(
        self,
        model: nn.Module,
        log_dir: Path | str,
        run_tag: str = "",
        sample_every: int = 50,
        loss_window: int = 200,
        explosion_z_threshold: float = 3.0,
        max_layers: int = 64,
    ):
        """Args:
            model:               the WM under training
            log_dir:             where to write JSONL + snapshots
            run_tag:             prefix for output filenames (e.g., "v25_f29")
            sample_every:        snapshot per-layer health every N steps
            loss_window:         rolling window for loss-component anomaly z-score
            explosion_z_threshold: trigger snapshot if any loss term > z*std above mean
            max_layers:          cap on layers tracked (avoids exploding hook count
                                  on huge models; selects largest-by-param)
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_tag = run_tag or "run"
        self.sample_every = max(1, int(sample_every))
        self.loss_window = max(20, int(loss_window))
        self.explosion_z = float(explosion_z_threshold)
        self.max_layers = int(max_layers)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.log_dir / f"{self.run_tag}_autopsy_mode_{ts}.jsonl"
        self.snapshot_dir = self.log_dir / f"{self.run_tag}_autopsy_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Per-layer state captured by hooks
        self._latest_act: dict[str, dict] = {}
        self._latest_grad: dict[str, dict] = {}
        self._handles: list = []

        # Loss-component rolling state per term
        self._loss_history: dict[str, deque] = {}

        # Step counter
        self._step = 0
        self._n_snapshots = 0
        self._n_records = 0

        # Health flags raised by hooks; consumed by step()
        self._anomalies: list[tuple[str, str]] = []

        # Top-K largest layers by param-count, plus all LayerNorms (cheap).
        self._tracked_names = self._select_layers()
        self._register_hooks()

        self._write_jsonl({
            "type": "init",
            "ts": datetime.utcnow().isoformat() + "Z",
            "run_tag": self.run_tag,
            "model_class": type(model).__name__,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_tracked_layers": len(self._tracked_names),
            "tracked_layers": self._tracked_names[:30],   # first 30 for brevity
            "config": {
                "sample_every": self.sample_every,
                "loss_window": self.loss_window,
                "explosion_z_threshold": self.explosion_z,
                "max_layers": self.max_layers,
            },
        })

    # ---------------- Layer selection + hook registration ----------------

    def _select_layers(self) -> list[str]:
        """Select up to max_layers tracked modules. Linear/Conv by param-size desc,
        plus all LayerNorm + MultiheadAttention + Embedding."""
        candidates: list[tuple[str, nn.Module]] = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, self.TRACKED_TYPES):
                candidates.append((name, mod))
        # Always-keep: LayerNorm, MHA, Embedding (lightweight).
        always = [(n, m) for n, m in candidates
                   if isinstance(m, (nn.LayerNorm, nn.MultiheadAttention, nn.Embedding))]
        sized = [(n, m) for n, m in candidates if isinstance(m, (nn.Linear, nn.Conv1d))]
        # Sort linear/conv by param-count desc so we keep biggest = most leverage
        sized.sort(key=lambda x: sum(p.numel() for p in x[1].parameters()),
                    reverse=True)
        budget = max(0, self.max_layers - len(always))
        selected = always + sized[:budget]
        return [n for n, _ in selected]

    def _register_hooks(self):
        """Attach forward + full-backward hooks. Catches activation + grad health."""
        name_set = set(self._tracked_names)
        for name, mod in self.model.named_modules():
            if name not in name_set:
                continue
            try:
                h_fwd = mod.register_forward_hook(self._make_fwd_hook(name))
                self._handles.append(h_fwd)
                # full_backward_hook: only fires when gradients exist.
                h_bwd = mod.register_full_backward_hook(self._make_bwd_hook(name))
                self._handles.append(h_bwd)
            except Exception as e:
                # Some modules (e.g., MultiheadAttention) reject hooks under
                # certain forward signatures. Skip silently in production.
                self._anomalies.append((name, f"hook_register_failed: {type(e).__name__}"))

    def _make_fwd_hook(self, name: str) -> Callable:
        def _hook(module, inp, out):
            try:
                # MHA returns (out, attn_weights); take out only.
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor):
                    h = _tensor_health(t)
                    self._latest_act[name] = h
                    if h.get("n_nan", 0) > 0 or h.get("n_inf", 0) > 0:
                        self._anomalies.append((name, "fwd_nan_or_inf"))
            except Exception:
                pass
        return _hook

    def _make_bwd_hook(self, name: str) -> Callable:
        def _hook(module, grad_input, grad_output):
            try:
                # full_backward_hook gives grad_output as a tuple.
                g = grad_output[0] if grad_output else None
                if isinstance(g, torch.Tensor):
                    h = _tensor_health(g)
                    self._latest_grad[name] = h
                    if h.get("n_nan", 0) > 0 or h.get("n_inf", 0) > 0:
                        self._anomalies.append((name, "bwd_nan_or_inf"))
            except Exception:
                pass
        return _hook

    # ---------------- Public step + memorization probe ----------------

    def step(
        self,
        step_idx: int,
        loss_components: Optional[dict[str, float]] = None,
        extra: Optional[dict] = None,
    ):
        """Call every training step. Records loss components + flushes per-layer
        snapshot every `sample_every` steps. Triggers anomaly snapshots."""
        try:
            self._step = int(step_idx)
            anomaly_reasons: list[str] = []

            # Loss-component rolling check
            if loss_components:
                for k, v in loss_components.items():
                    v_f = _safe_float(v)
                    win = self._loss_history.setdefault(k, deque(maxlen=self.loss_window))
                    # NaN/Inf in loss term = immediate anomaly
                    raw = v
                    if isinstance(raw, (int, float)) and (math.isnan(raw) or math.isinf(raw)):
                        anomaly_reasons.append(f"loss_{k}_nan_or_inf")
                    win.append(v_f)
                    if len(win) >= self.loss_window // 2:
                        arr = np.array(win, dtype=np.float64)
                        m, s = float(arr.mean()), float(arr.std(ddof=1) + 1e-12)
                        if abs(v_f - m) > self.explosion_z * s and s > 1e-9:
                            anomaly_reasons.append(f"loss_{k}_explosion_z={(v_f-m)/s:.1f}")

            # Drain hook-side anomalies
            if self._anomalies:
                for name, why in self._anomalies:
                    anomaly_reasons.append(f"layer_{name}:{why}")
                self._anomalies = []

            # Periodic per-layer record + always on anomaly
            if (self._step % self.sample_every == 0) or anomaly_reasons:
                rec = {
                    "type": "step",
                    "step": self._step,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "loss": loss_components or {},
                    "extra": extra or {},
                    "act": dict(self._latest_act),
                    "grad": dict(self._latest_grad),
                }
                if anomaly_reasons:
                    rec["anomalies"] = anomaly_reasons
                self._write_jsonl(rec)
                self._n_records += 1

            # Triggered snapshot on any anomaly
            if anomaly_reasons:
                reason_tag = "_".join(r.split(":")[0] for r in anomaly_reasons[:3])[:60]
                self._dump_snapshot(reason_tag, anomaly_reasons,
                                     loss_components, extra)
        except Exception:
            # Diagnostics MUST NEVER crash the training loop.
            pass

    @torch.no_grad()
    def memorization_probe(
        self,
        model: nn.Module,
        sample_batch: tuple,
        n_features: Optional[int] = None,
    ) -> dict:
        """Probe for memorization. Forward a batch, then forward a TIME-SHUFFLED
        copy of the same batch through the same model. If the model relies on
        temporal patterns memorized (rather than predictive structure), the
        shuffled batch's IC will collapse near zero.

        Args:
            model: WM with .get_loss(...) signature.
            sample_batch: (obs, targets, asset) tuple from val_loader.
            n_features: optional cap; if obs has more, only the first n are
                shuffled (preserves cross-asset features that aren't temporal).

        Returns:
            dict with contiguous_ic_h1, shuffled_ic_h1, ratio, recon_mse_lift.
        """
        try:
            obs, targets, asset = sample_batch
            obs = obs.detach()
            B, T, F_total = obs.shape
            F_eff = n_features or F_total
            # Shuffle along time-axis per (B, F)
            idx = torch.randperm(T, device=obs.device)
            shuffled = obs.clone()
            shuffled[:, :, :F_eff] = obs[:, idx, :F_eff]

            tgt_gpu = {h: t.to(obs.device) for h, t in targets.items()}

            # Try to call model.get_loss; some have different signatures.
            def _ic_one(input_obs):
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    out = model.get_loss(
                        input_obs, asset.to(obs.device), tgt_gpu,
                        mask_ratio=0.0,
                        regime_labels=tgt_gpu.get("regime_label"),
                    )
                # Accept both 3-tuple and dict-style returns.
                outputs = out[2] if isinstance(out, tuple) else out
                if "return_logits" not in outputs:
                    return None, None
                logits = outputs["return_logits"][1].float()
                if logits.dim() == 3:
                    pred = logits.argmax(dim=-1).reshape(-1).float()
                else:
                    pred = logits.reshape(-1).float()
                real = tgt_gpu[1].reshape(-1).float()
                if pred.numel() != real.numel() or pred.numel() < 5:
                    return None, None
                # Spearman-style: rank-corr proxy via Pearson on demeaned.
                ic = float(torch.corrcoef(torch.stack([pred, real]))[0, 1].nan_to_num())
                # Rec loss as side health
                rec = outputs.get("recon")
                rec_mse = (
                    float(((rec.float() - input_obs[:, :, :rec.shape[-1]].float()) ** 2)
                                .mean().nan_to_num())
                    if rec is not None else 0.0
                )
                return ic, rec_mse

            ic_c, rec_c = _ic_one(obs)
            ic_s, rec_s = _ic_one(shuffled)
            if ic_c is None or ic_s is None:
                return {"error": "model.get_loss did not return return_logits"}

            ratio = ic_s / max(abs(ic_c), 1e-9) if ic_c else 0.0
            payload = {
                "type": "memorization_probe",
                "step": self._step,
                "ts": datetime.utcnow().isoformat() + "Z",
                "contiguous_ic_h1": round(ic_c, 6),
                "shuffled_ic_h1": round(ic_s, 6),
                "shic_to_ic_ratio": round(ratio, 4),
                "contiguous_recon_mse": round(rec_c, 6),
                "shuffled_recon_mse": round(rec_s, 6),
                "recon_lift_pct": round((rec_s - rec_c) / max(rec_c, 1e-9), 4),
            }
            # Memorization heuristic: shuffled-IC near zero AND contiguous-IC > 0.02
            if abs(ic_c) > 0.02 and abs(ratio) < 0.10:
                payload["verdict"] = "MEMORIZATION_LIKELY"
                self._dump_snapshot("memorization_probe_low_shic_ratio", [],
                                     None, payload)
            else:
                payload["verdict"] = "OK_OR_INCONCLUSIVE"
            self._write_jsonl(payload)
            return payload
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    @torch.no_grad()
    def embedding_health_probe(
        self,
        embedding_module: nn.Module,
        name: str = "embed",
    ) -> dict:
        """Diagnose memorization-vector style embeddings. Captures:
          - weight matrix stable rank
          - row-cosine collinearity (mean off-diag |cos|)
          - top-K singular value ratio (collapse indicator)
        """
        try:
            w = None
            for p in embedding_module.parameters():
                if p.dim() == 2:
                    w = p
                    break
            if w is None:
                return {"name": name, "error": "no 2D weight found"}
            sr = _stable_rank(w)
            # Row-cosine
            try:
                w_n = torch.nn.functional.normalize(w.float(), dim=1)
                cos = (w_n @ w_n.T).abs()
                n = cos.shape[0]
                off = cos.flatten()[~torch.eye(n, dtype=torch.bool, device=cos.device).flatten()]
                mean_off = float(off.mean()) if off.numel() else 0.0
            except Exception:
                mean_off = 0.0
            # Top SV ratio
            try:
                sv = torch.linalg.svdvals(w.float())
                top_ratio = float(sv[0] / max(sv.median().item(), 1e-12))
            except Exception:
                top_ratio = 0.0
            payload = {
                "type": "embedding_health",
                "step": self._step,
                "name": name,
                "shape": list(w.shape),
                "stable_rank": round(sr, 4),
                "mean_offdiag_abscos": round(mean_off, 4),
                "top_sv_to_median_ratio": round(top_ratio, 4),
            }
            if mean_off > 0.7 or top_ratio > 100:
                payload["verdict"] = "EMBEDDING_COLLAPSE_LIKELY"
                self._dump_snapshot("embedding_collapse", [], None, payload)
            self._write_jsonl(payload)
            return payload
        except Exception as e:
            return {"name": name, "error": f"{type(e).__name__}: {e}"}

    # ---------------- Snapshots + IO ----------------

    def _dump_snapshot(
        self, reason_tag: str, anomaly_reasons: list[str],
        loss_components: Optional[dict] = None,
        extra: Optional[dict] = None,
    ):
        """Write a full per-layer snapshot to disk. Caller continues training."""
        try:
            self._n_snapshots += 1
            payload = {
                "type": "snapshot",
                "step": self._step,
                "ts": datetime.utcnow().isoformat() + "Z",
                "reason_tag": reason_tag,
                "anomaly_reasons": anomaly_reasons,
                "loss": loss_components or {},
                "extra": extra or {},
                "act": dict(self._latest_act),
                "grad": dict(self._latest_grad),
            }
            fname = (f"step{self._step:08d}_{reason_tag.replace(' ', '_')[:60]}"
                     f"_{self._n_snapshots:04d}.json")
            (self.snapshot_dir / fname).write_text(
                json.dumps(payload, indent=2, default=_json_default),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _write_jsonl(self, record: dict):
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=_json_default) + "\n")
        except Exception:
            pass

    def close(self):
        """Unregister hooks + write final summary record."""
        try:
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._handles = []
            self._write_jsonl({
                "type": "close",
                "ts": datetime.utcnow().isoformat() + "Z",
                "n_records": self._n_records,
                "n_snapshots": self._n_snapshots,
                "final_step": self._step,
            })
        except Exception:
            pass

    # ---------------- CLI integration helper ----------------

    @staticmethod
    def add_argparse(parser):
        """Add --autopsy / --autopsy-every / --autopsy-loss-window flags."""
        parser.add_argument(
            "--autopsy", action="store_true",
            help="Enable internals-level autopsy mode (hooks every tracked layer).")
        parser.add_argument(
            "--autopsy-every", type=int, default=50,
            help="Per-layer snapshot frequency in steps (default 50).")
        parser.add_argument(
            "--autopsy-loss-window", type=int, default=200,
            help="Rolling window for loss-component anomaly z-score (default 200).")
        parser.add_argument(
            "--autopsy-z", type=float, default=3.0,
            help="Loss-component explosion z-threshold (default 3.0).")


def _json_default(o):
    """JSON serializer fallback: torch.Tensor scalars / numpy scalars."""
    try:
        import numpy as _np
        if isinstance(o, _np.generic):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, torch.Tensor):
        try:
            return o.item() if o.numel() == 1 else o.tolist()
        except Exception:
            return str(o)
    if isinstance(o, (set, deque)):
        return list(o)
    return str(o)


__all__ = ["AutopsyMode"]
