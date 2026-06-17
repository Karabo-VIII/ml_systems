"""Distillation training loop.

Loads:
    - data/_caches/distillation/windows.npz (fixed window set)
    - data/_caches/distillation/teacher_<name>_logits.npz (per teacher)

Constructs an ensemble teacher distribution by averaging teacher softmax
probabilities (NOT logits — averaging logits introduces calibration bias),
then re-encoding to logits for the student via log-prob.

Trains a 4-10M student to match the ensemble distribution + match
expected-return + match variance via HybridDistillLoss. Saves student
ckpt every 500 steps + supports resume.

Usage:
    # 1. Pick which teachers participate (must have caches present)
    python -m src.frontier_ml.distillation.train \
        --teachers foundation,v1_1,v1_4,v1_6,v3,v4 \
        --student-size small --max-steps 20000

    # 2. Resume
    python -m src.frontier_ml.distillation.train --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from frontier_ml.foundation.harmony import apply_harmony  # noqa: E402
apply_harmony(verbose=True)

from frontier_ml.distillation.student import make_student, STUDENT_CONFIGS  # noqa: E402
from frontier_ml.distillation.distill_loss import HybridDistillLoss  # noqa: E402
from frontier_ml.foundation.objectives import make_bucketer  # noqa: E402

CACHE_DIR = _PROJECT_ROOT / "data" / "_caches" / "distillation"
CKPT_DIR = _PROJECT_ROOT / "models" / "frontier_ml" / "distillation"
LOG_DIR = _PROJECT_ROOT / "logs" / "frontier_ml" / "distillation"
HORIZONS = (1, 4, 16, 64)


def _load_windows() -> dict:
    p = CACHE_DIR / "windows.npz"
    if not p.exists():
        raise FileNotFoundError(f"windows.npz not built; run teacher_inference --build-windows")
    return dict(np.load(p, allow_pickle=True))


def _load_teacher(name: str) -> Optional[np.ndarray]:
    p = CACHE_DIR / f"teacher_{name}_logits.npz"
    if not p.exists():
        print(f"[distill-train] WARN teacher {name} cache missing at {p}; skipping",
              flush=True)
        return None
    data = np.load(p, allow_pickle=True)
    return data["logits"]   # (N, H, NUM_BINS) fp16


def ensemble_logits(teacher_logits_list: List[np.ndarray]) -> np.ndarray:
    """Average teacher SOFTMAX probabilities, then re-log for student loss.

    Per LITERATURE.md Hole 5: averaging logits directly biases the
    distribution toward over-confident teachers.
    """
    # In float32 for numerical stability
    probs = np.zeros_like(teacher_logits_list[0], dtype=np.float32)
    for L in teacher_logits_list:
        # softmax along last axis, fp16 -> fp32
        L32 = L.astype(np.float32)
        L32 -= L32.max(axis=-1, keepdims=True)  # stability
        exp = np.exp(L32)
        s = exp.sum(axis=-1, keepdims=True)
        probs += exp / np.where(s > 0, s, 1.0)
    probs /= len(teacher_logits_list)
    # Convert back to logits (log p), clipped to avoid -inf
    probs = np.clip(probs, 1e-9, 1.0)
    return np.log(probs).astype(np.float16)


def train(
    teachers: List[str],
    student_size: str = "small",
    batch_size: int = 16,
    max_steps: int = 20_000,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 500,
    huber_weight: float = 0.5,
    ckpt_every: int = 500,
    log_every: int = 25,
    resume: bool = False,
    seed: int = 0,
) -> Dict:
    """Distillation training loop."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[distill-train] loading window set + teacher caches...", flush=True)
    win = _load_windows()
    features = win["features"]            # (N, S, F) fp16
    asset_ids = win["asset_ids"]
    targets = win["targets"]              # (N, H) fp32
    N, S, F = features.shape
    n_features = F
    print(f"[distill-train] N={N:,}  S={S}  F={F}  H={list(HORIZONS)}", flush=True)

    teacher_logits_list = []
    for name in teachers:
        L = _load_teacher(name)
        if L is not None:
            teacher_logits_list.append(L)
    if not teacher_logits_list:
        raise SystemExit("no teacher caches found; ran teacher_inference for at least one teacher first")
    print(f"[distill-train] using {len(teacher_logits_list)} teachers: {teachers}",
          flush=True)
    print(f"[distill-train] computing ensemble distribution (softmax-mean)...", flush=True)
    ens_logits = ensemble_logits(teacher_logits_list)   # (N, H, NUM_BINS) fp16

    # ---- Student + loss ------------------------------------------------
    student = make_student(size=student_size, n_features=n_features).to(device)
    print(f"[distill-train] student '{student_size}': {student.num_params():,} params  "
          f"({student.num_params()/1e6:.1f}M)", flush=True)

    cfg = STUDENT_CONFIGS[student_size]
    bucketer = make_bucketer(num_bins=cfg["num_bins"], device=device)
    loss_fn = HybridDistillLoss(bucketer, horizons=HORIZONS).to(device)

    optim = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Optional auxiliary supervised Huber loss on raw targets (small weight).
    huber = torch.nn.HuberLoss(reduction="mean", delta=0.005)

    # ---- Resume --------------------------------------------------------
    start_step = 0
    latest = CKPT_DIR / f"latest_{student_size}.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        student.load_state_dict(state["model"], strict=False)
        try: optim.load_state_dict(state["optim"])
        except Exception as e: print(f"[distill-train] optim load WARN: {e}")
        if scaler and state.get("scaler"):
            try: scaler.load_state_dict(state["scaler"])
            except Exception: pass
        start_step = int(state.get("step", 0)) + 1
        print(f"[distill-train] resumed at step {start_step}", flush=True)

    # ---- LR schedule ---------------------------------------------------
    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
        return lr * 0.5 * (1.0 + np.cos(np.pi * progress))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"distill_{student_size}_{int(time.time())}.jsonl"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rng = np.random.default_rng(seed)
    ema = float("nan")
    t0 = time.time()
    for step in range(start_step, max_steps):
        idx = rng.integers(0, N, size=batch_size)
        xb = torch.from_numpy(features[idx].astype(np.float32)).to(device, non_blocking=True)
        ab = torch.from_numpy(asset_ids[idx]).to(device, non_blocking=True)
        teacher_logits = {f"h{h}": torch.from_numpy(ens_logits[idx, hi].astype(np.float32))
                                            .to(device, non_blocking=True)
                           for hi, h in enumerate(HORIZONS)}
        target_returns = {h: torch.from_numpy(targets[idx, hi]).to(device, non_blocking=True)
                          for hi, h in enumerate(HORIZONS)}

        cur_lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = student(xb, asset_ids=ab)
                d = loss_fn(out["return_logits"], teacher_logits)
                # Auxiliary Huber on continuous expected return (helps when
                # teacher distributions are flat / uninformative).
                bin_centers = loss_fn.bin_centers.to(out["return_logits"]["h1"].dtype)
                aux = 0.0
                for hi, h in enumerate(HORIZONS):
                    logits_h = out["return_logits"][f"h{h}"]
                    p = torch.softmax(logits_h, dim=-1)
                    e_r = (p * bin_centers).sum(dim=-1)
                    aux = aux + huber(e_r.float(), target_returns[h].float())
                aux = aux / len(HORIZONS)
                total = d["total"] + huber_weight * aux
            scaler.scale(total).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            out = student(xb, asset_ids=ab)
            d = loss_fn(out["return_logits"], teacher_logits)
            bin_centers = loss_fn.bin_centers
            aux = 0.0
            for hi, h in enumerate(HORIZONS):
                logits_h = out["return_logits"][f"h{h}"]
                p = torch.softmax(logits_h, dim=-1)
                e_r = (p * bin_centers).sum(dim=-1)
                aux = aux + huber(e_r, target_returns[h])
            aux = aux / len(HORIZONS)
            total = d["total"] + huber_weight * aux
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()

        v = float(total.item())
        if np.isnan(ema):
            ema = v
        else:
            ema = 0.99 * ema + 0.01 * v

        if step % log_every == 0:
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
            entry = {
                "step": step, "loss_total": v,
                "loss_distill_total": float(d["total"].item()),
                "loss_kl": float(d["kl"].item()),
                "loss_l1_expected": float(d["l1_expected"].item()),
                "loss_l2_var": float(d["l2_var"].item()),
                "loss_aux_huber": float(aux.item()),
                "ema": ema, "lr": cur_lr,
                "peak_vram_gb": peak_gb,
            }
            with open(log_path, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
            print(f"[distill-train] step {step:5d}/{max_steps}  "
                  f"loss {v:.4f} ema {ema:.4f}  KL {d['kl'].item():.3f}  "
                  f"L1 {d['l1_expected'].item():.5f}  Huber {aux.item():.5f}  "
                  f"lr {cur_lr:.2e}  peak {peak_gb:.2f}GB",
                  flush=True)

        if (step + 1) % ckpt_every == 0:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = latest.with_suffix(".pt.tmp")
            torch.save({
                "step": step, "model": student.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "config": cfg, "n_features": n_features,
                "ema": ema, "teachers": teachers,
            }, tmp)
            if latest.exists():
                latest.unlink()
            tmp.rename(latest)

    elapsed = time.time() - t0
    print(f"\n[distill-train] DONE step={step} elapsed={elapsed:.0f}s ema={ema:.4f}",
          flush=True)
    return {"ema": ema, "elapsed_s": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default="foundation",
                    help="Comma-sep teacher names (must have cached logits)")
    ap.add_argument("--student-size", default="small", choices=list(STUDENT_CONFIGS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    train(
        teachers=teachers,
        student_size=args.student_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
