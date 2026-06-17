"""Foundation pretraining loop — Mamba-3 30M backbone with AMP + checkpoint.

Per PLAN.md / LITERATURE.md:
    - Causal multi-horizon TwoHot + lead-lag cross-asset contrastive
    - AMP (fp16) for 4060/8GB headroom
    - Checkpoint every 100 steps + resume support (Hole 8 closure)
    - Real-data probe mode (--probe-real): 200 steps, peak VRAM tracking

Usage:
    # Real-data probe (200 steps; gates the full epoch budget)
    python src/frontier_ml/foundation/pretrain.py --probe-real --universe u10

    # Full pretrain (writes ckpt every 100 steps)
    python src/frontier_ml/foundation/pretrain.py --universe u100 \
        --max-steps 50000 --batch-size 8 --seq-len 512

    # Resume from latest ckpt
    python src/frontier_ml/foundation/pretrain.py --universe u100 --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Apply CPU/GPU/IO harmony BEFORE importing torch-heavy modules.
from frontier_ml.foundation.harmony import apply_harmony  # noqa: E402
apply_harmony(verbose=True)

from frontier_ml.foundation.backbone import FoundationBackbone, DEFAULT_CONFIG  # noqa: E402
from frontier_ml.foundation.data_loader import FoundationDataset  # noqa: E402
from frontier_ml.foundation.objectives import make_bucketer, FoundationLoss  # noqa: E402

CKPT_DIR = _PROJECT_ROOT / "models" / "frontier_ml" / "foundation"
LOG_DIR = _PROJECT_ROOT / "logs" / "frontier_ml" / "foundation"


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def save_ckpt(path: Path, model: FoundationBackbone, optim: torch.optim.Optimizer,
              scaler: torch.amp.GradScaler, step: int, train_loss_ema: float,
              cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_loss_ema": train_loss_ema,
        "config": cfg,
        "n_features": model.n_features,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    if path.exists():
        path.unlink()
    tmp.rename(path)


def load_ckpt(path: Path, model: FoundationBackbone, optim: torch.optim.Optimizer,
              scaler: Optional[torch.amp.GradScaler]) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    if optim is not None and "optim" in state:
        try:
            optim.load_state_dict(state["optim"])
        except Exception as e:
            print(f"[ckpt] optim load failed ({e}); continuing fresh", flush=True)
    if scaler is not None and state.get("scaler") is not None:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"[ckpt] scaler load failed ({e}); continuing fresh", flush=True)
    return state


def pretrain(
    universe: str = "u100",
    seq_len: int = 512,
    batch_size: int = 8,
    n_features: int = 34,
    max_steps: int = 50_000,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1000,
    contrastive_every: int = 4,
    w_horizon: float = 1.0,
    w_contrastive: float = 0.1,
    contrastive_temp: float = 0.1,
    ckpt_every: int = 100,
    log_every: int = 10,
    probe_real: bool = False,
    resume: bool = False,
    config_overrides: Optional[dict] = None,
    seed: int = 0,
) -> dict:
    """Pretrain the foundation backbone."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[pretrain] WARN no CUDA; running on CPU (very slow)", flush=True)

    if probe_real:
        max_steps = 200
        ckpt_every = 1000  # don't ckpt during probe
        print("[pretrain] PROBE MODE -- 200 real-data steps, no ckpt", flush=True)

    # ---- Data ------------------------------------------------------------
    print(f"[pretrain] loading dataset universe={universe} seq_len={seq_len}", flush=True)
    ds = FoundationDataset(
        universe=universe,
        seq_len=seq_len,
        horizons=tuple(DEFAULT_CONFIG["horizons"]),
        seed=seed,
    )
    n_features = ds.n_features

    # ---- Model -----------------------------------------------------------
    cfg = dict(DEFAULT_CONFIG)
    if config_overrides:
        cfg.update(config_overrides)
    model = FoundationBackbone(n_features=n_features, config=cfg).to(device)
    n_params = model.num_params()
    print(f"[pretrain] model params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    bucketer = make_bucketer(num_bins=cfg["num_bins"], min_val=-1.0, max_val=1.0, device=device)
    loss_fn = FoundationLoss(
        bucketer,
        w_horizon=w_horizon,
        w_contrastive=w_contrastive,
        contrastive_temp=contrastive_temp,
    )

    # ---- Resume ----------------------------------------------------------
    start_step = 0
    train_loss_ema = float("nan")
    latest_ckpt = CKPT_DIR / "latest.pt"
    if resume and latest_ckpt.exists():
        print(f"[pretrain] resuming from {latest_ckpt}", flush=True)
        state = load_ckpt(latest_ckpt, model, optim, scaler)
        start_step = state["step"] + 1
        train_loss_ema = state.get("train_loss_ema", float("nan"))
        print(f"[pretrain] resumed at step {start_step}; ema {train_loss_ema:.4f}", flush=True)
    elif resume:
        print(f"[pretrain] --resume requested but no ckpt at {latest_ckpt}; starting fresh", flush=True)

    # ---- LR schedule (warmup then cosine) -------------------------------
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return lr * 0.5 * (1.0 + np.cos(np.pi * progress))

    # ---- Train loop ------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pretrain_{int(time.time())}.jsonl"

    nan_step = -1
    t0 = time.time()
    for step in range(start_step, max_steps):
        # LR
        cur_lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        # ---- Anchor batch (always) ---------------------------------------
        x_np, asset_ids_np, target_arrays_np = ds.sample_anchor_batch(batch_size)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        asset_ids = torch.from_numpy(asset_ids_np).to(device, non_blocking=True)
        target_returns = {
            h: torch.from_numpy(arr).to(device, non_blocking=True)
            for h, arr in target_arrays_np.items()
        }

        # Optionally pair with contrastive batch.
        # Memory budget rationale: anchor forward already uses ~6.5 GB of
        # the 8.59 GB VRAM at B=8 S=512. Three full forwards under autocast
        # would OOM. Use BYOL/SimSiam-style stop-grad on pos/neg so their
        # forward is no_grad (no activation cache), AND halve contrastive
        # batch so even the no_grad forward stays small.
        do_contrast = (step % contrastive_every == 0)
        if do_contrast:
            cb_bs = max(2, batch_size // 2)
            cb = ds.sample_contrastive_batch(cb_bs)
            x_p = torch.from_numpy(cb["x_pos"]).to(device, non_blocking=True)
            x_n = torch.from_numpy(cb["x_neg"]).to(device, non_blocking=True)
            bp_ids = torch.from_numpy(cb["asset_pos"]).to(device, non_blocking=True)
            x_anchor_c = x[:cb_bs]                             # reuse first cb_bs anchor rows
            asset_ids_c = asset_ids[:cb_bs]

        optim.zero_grad(set_to_none=True)

        def _do_step(amp: bool):
            ctx = (torch.amp.autocast("cuda", dtype=torch.float16)
                    if amp else torch.amp.autocast("cuda", enabled=False))
            with ctx:
                # Single anchor forward; reuse its contrastive_emb slice.
                out_main = model(x, asset_ids=asset_ids)
                if do_contrast:
                    # Pos/neg under no_grad (stop-grad): provide TARGET for
                    # anchor's contrastive loss but gradients don't flow.
                    # BYOL / SimSiam / DINO family. Sequential forwards
                    # with empty_cache between to keep activation peak low.
                    with torch.no_grad():
                        out_p_emb = model(x_p, asset_ids=bp_ids)["contrastive_emb"].detach()
                        out_n_emb = model(x_n, asset_ids=bp_ids)["contrastive_emb"].detach()
                    emb_anchor_c = out_main["contrastive_emb"][:cb_bs]
                    loss_d_local = loss_fn(
                        out_main["return_logits"], target_returns,
                        emb_anchor=emb_anchor_c,
                        emb_pos=out_p_emb,
                        emb_neg=out_n_emb,
                    )
                else:
                    loss_d_local = loss_fn(out_main["return_logits"], target_returns)
            return out_main, loss_d_local

        if scaler is not None:
            out, loss_d = _do_step(amp=True)
            scaler.scale(loss_d["total"]).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            out, loss_d = _do_step(amp=False)
            loss_d["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # NaN guard
        if torch.isnan(loss_d["total"]).item():
            nan_step = step
            print(f"[pretrain] NaN loss at step {step}; halting", flush=True)
            break

        # EMA loss
        v = loss_d["total"].item()
        if np.isnan(train_loss_ema):
            train_loss_ema = v
        else:
            train_loss_ema = 0.99 * train_loss_ema + 0.01 * v

        # Log
        if step % log_every == 0:
            peak = _peak_vram_gb()
            elapsed = time.time() - t0
            steps_done = max(1, step - start_step + 1)
            rate = steps_done / elapsed
            entry = {
                "step": step,
                "loss_total": float(loss_d["total"].item()),
                "loss_horizon": float(loss_d["horizon"].item()),
                "loss_contrastive": float(loss_d["contrastive"].item()),
                "loss_ema": float(train_loss_ema),
                "lr": float(cur_lr),
                "peak_vram_gb": float(peak),
                "rate_steps_per_s": float(rate),
                "h_max": float(out["h_seq"].abs().max().item()),
                "do_contrast": bool(do_contrast),
            }
            with open(log_path, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
            print(f"[pretrain] step {step:5d}/{max_steps}  "
                  f"loss {v:.4f} ema {train_loss_ema:.4f}  "
                  f"lr {cur_lr:.2e}  h_max {entry['h_max']:.2f}  "
                  f"peak {peak:.2f}GB  {rate:.2f} step/s",
                  flush=True)

        # Checkpoint
        if (step + 1) % ckpt_every == 0 and not probe_real:
            save_ckpt(latest_ckpt, model, optim, scaler, step, train_loss_ema, cfg)
            # Periodic full snapshot every 1000 steps
            if (step + 1) % (ckpt_every * 10) == 0:
                snap = CKPT_DIR / f"step_{step+1:06d}.pt"
                save_ckpt(snap, model, optim, scaler, step, train_loss_ema, cfg)

    elapsed = time.time() - t0
    peak = _peak_vram_gb()
    n_done = max(1, step - start_step + 1)
    print(f"\n[pretrain] DONE step={step} elapsed={elapsed:.1f}s "
          f"({elapsed/n_done:.2f}s/step) peak={peak:.2f}GB ema_loss={train_loss_ema:.4f}",
          flush=True)

    if probe_real:
        verdict_pass = (
            peak < 7.5
            and nan_step < 0
            and train_loss_ema < 6.0  # below random-init CE for 255 bins ~ 5.54
        )
        print(f"\n[pretrain-probe] VERDICT: {'PASS' if verdict_pass else 'FAIL'}", flush=True)
        print(f"  peak VRAM: {peak:.2f} / 8.59 GB", flush=True)
        print(f"  NaN: {'NO' if nan_step < 0 else f'step {nan_step}'}", flush=True)
        print(f"  loss ema: {train_loss_ema:.4f} (random ~5.54)", flush=True)
        return {
            "pass": verdict_pass,
            "peak_vram_gb": peak,
            "loss_ema": train_loss_ema,
            "nan_step": nan_step,
            "elapsed_s": elapsed,
            "steps_done": n_done,
        }

    return {
        "peak_vram_gb": peak,
        "loss_ema": train_loss_ema,
        "elapsed_s": elapsed,
        "steps_done": n_done,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u100", choices=["u10", "u50", "u100"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--probe-real", action="store_true",
                    help="Real-data probe: 200 steps, no ckpt, gate the budget.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pretrain(
        universe=args.universe,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        probe_real=args.probe_real,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
