"""Generic 200-step empirical probe for WM versions.

Per CLAUDE.md §12 Empirical Probe for Numerical Issues:
    "When fixing NaN, inf, crash, or numerical instability: write a standalone
    probe script: 200-300 real-data steps through the actual model under AMP.
    STRESS PROBE at B=32 (full batch), not just B=4."

This module provides a shared probe harness. Each version's training
directory can drive it by importing this module and passing in:
  - the model class + a constructor (no extra args)
  - a settings module (so we read FEATURE_LIST + REWARD_HORIZONS + DEVICE)
  - the version's data dir (so we feed real chimera bars not random)

The probe runs N steps under AMP, tracking per-step:
  - total loss + per-component losses
  - h_seq.abs().max() (magnitude growth)
  - sum of param grad norms (gradient health)
  - NaN / inf detection (HARD FAIL on first hit)

Returns a dict with PASS/FAIL plus the most-recent statistics, suitable
for a 1-line console summary.

Usage from a version dir:

    # src/wm/vN/vN_training/probe.py
    from settings import *
    from world_model import TransformerWorldModel
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
    from probe_runner import run_probe

    if __name__ == "__main__":
        result = run_probe(
            model_factory=lambda: TransformerWorldModel(input_dim=INPUT_DIM),
            data_dir=DATA_DIR,
            feature_list=FEATURE_LIST,
            asset_to_idx=ASSET_TO_IDX,
            reward_horizons=REWARD_HORIZONS,
            seq_len=WM_SEQ_LEN,
            batch_size=WM_BATCH_SIZE,
            device=DEVICE,
            n_steps=200,
            label="V1.0 f13",
        )
        sys.exit(0 if result["pass"] else 1)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.optim as optim


__contract__ = {
    "kind": "wm_probe_runner",
    "owner": "wm/_shared",
    "outputs": [],
    "invariants": [
        "200-step smoke under AMP at full batch",
        "loads real chimera data via data_api.load_full_data_for_training",
        "tracks NaN/inf + h_seq magnitude growth + grad-norm sum",
        "no side effects (no checkpoint writes, no log files)",
    ],
}


def run_probe(
    model_factory: Callable,
    data_dir: Path,
    feature_list: list,
    asset_to_idx: dict,
    reward_horizons: list,
    seq_len: int,
    batch_size: int,
    device: str,
    n_steps: int = 200,
    label: str = "probe",
    lr: float = 2e-4,
    grad_clip: float = 1.0,
    mask_ratio: float = 0.15,
) -> dict:
    """Run an N-step probe and return a dict with PASS/FAIL + statistics."""
    # Late import so this module is light to import.
    src_dir = Path(__file__).resolve().parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from data_api import load_full_data_for_training as load_full_data

    print(f"\n[probe:{label}] loading data from {data_dir}")
    segments = load_full_data(data_dir, feature_list, asset_to_idx, reward_horizons)
    if not segments:
        return {"pass": False, "reason": f"no data at {data_dir}"}

    # Build a tiny in-memory dataset by stacking random windows from the
    # first few assets — avoids dragging in AntifragileDataset machinery.
    rng = np.random.default_rng(42)
    samples = []
    for seg in segments[: max(3, min(5, len(segments)))]:
        feat = seg["features"]  # [T, F]
        asset_id = seg["asset_idx"]  # data_api uses asset_idx, not asset_id
        T = feat.shape[0]
        if T < seq_len + max(reward_horizons) + 1:
            continue
        # Sample 5000 windows per segment (capped by len)
        n_windows = min(5000, T - seq_len - max(reward_horizons) - 1)
        starts = rng.integers(0, T - seq_len - max(reward_horizons), size=n_windows)
        for s in starts:
            sample = {
                "obs": feat[s : s + seq_len],
                "asset_id": asset_id,
                "targets": {
                    h: seg[f"target_return_{h}"][s : s + seq_len] for h in reward_horizons
                },
            }
            if "regime_label" in seg:
                sample["regime_label"] = seg["regime_label"][s : s + seq_len]
            samples.append(sample)
        if len(samples) >= n_steps * batch_size * 2:
            break

    if len(samples) < n_steps * batch_size:
        return {
            "pass": False,
            "reason": f"insufficient windows ({len(samples)}) for {n_steps} steps × bs {batch_size}",
        }

    print(f"[probe:{label}] {len(samples):,} windows ready; building model on {device}", flush=True)

    model = model_factory().to(device)
    model.train()
    print(f"[probe:{label}] model built ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    print(f"[probe:{label}] optimizer + scaler ready; entering step loop", flush=True)

    # Statistics
    n_nan_loss = 0
    n_inf_loss = 0
    loss_history = []
    h_seq_max_history = []
    grad_norm_history = []
    t0 = time.time()

    rng_step = np.random.default_rng(123)

    for step in range(n_steps):
        try:
            idxs = rng_step.integers(0, len(samples), size=batch_size)
            batch = [samples[int(i)] for i in idxs]
            obs = torch.from_numpy(np.ascontiguousarray(np.stack([b["obs"] for b in batch]))).float().to(device)
            asset_arr = np.array([b["asset_id"] for b in batch], dtype=np.int64)
            asset = torch.from_numpy(asset_arr).long().to(device)
            targets = {
                h: torch.from_numpy(np.ascontiguousarray(np.stack([b["targets"][h] for b in batch]))).float().to(device)
                for h in reward_horizons
            }
            regime_labels = None
            if all("regime_label" in b for b in batch):
                regime_labels = torch.from_numpy(
                    np.ascontiguousarray(np.stack([b["regime_label"] for b in batch]))
                ).long().to(device)
        except Exception as e:
            print(f"[probe:{label}] batch-prep failed at step {step}: {type(e).__name__}: {e}", flush=True)
            raise

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            try:
                ret = model.get_loss(
                    obs, asset, targets,
                    mask_ratio=mask_ratio,
                    regime_labels=regime_labels,
                )
            except TypeError:
                # Some versions don't accept regime_labels kwarg
                ret = model.get_loss(obs, asset, targets, mask_ratio=mask_ratio)

        if isinstance(ret, tuple):
            loss = ret[0]
            outputs = ret[2] if len(ret) >= 3 else None
        else:
            loss = ret
            outputs = None

        loss_val = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)
        if not np.isfinite(loss_val):
            if np.isnan(loss_val):
                n_nan_loss += 1
            else:
                n_inf_loss += 1
            return {
                "pass": False,
                "reason": f"non-finite loss at step {step}: {loss_val}",
                "step": step,
                "loss_history": loss_history,
                "h_seq_max_history": h_seq_max_history,
                "grad_norm_history": grad_norm_history,
                "elapsed_s": time.time() - t0,
            }

        loss_history.append(loss_val)
        if outputs is not None and "h_seq" in outputs:
            h_max = float(outputs["h_seq"].detach().abs().max().cpu())
            if not np.isfinite(h_max):
                return {
                    "pass": False,
                    "reason": f"h_seq non-finite at step {step}: {h_max}",
                    "step": step,
                    "loss_history": loss_history,
                    "h_seq_max_history": h_seq_max_history,
                    "grad_norm_history": grad_norm_history,
                    "elapsed_s": time.time() - t0,
                }
            h_seq_max_history.append(h_max)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        gn = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        if not np.isfinite(gn):
            return {
                "pass": False,
                "reason": f"grad norm non-finite at step {step}: {gn}",
                "step": step,
                "loss_history": loss_history,
                "h_seq_max_history": h_seq_max_history,
                "grad_norm_history": grad_norm_history,
                "elapsed_s": time.time() - t0,
            }
        grad_norm_history.append(gn)
        scaler.step(optimizer)
        scaler.update()

        if step in (0, 9, 49, 99, 199) or step == n_steps - 1:
            h_msg = f"h_max={h_seq_max_history[-1]:.2f}" if h_seq_max_history else ""
            print(f"[probe:{label}] step {step:4d}  loss={loss_val:.4f}  gn={gn:.3f}  {h_msg}")

    elapsed = time.time() - t0
    # Magnitude-growth heuristic: if h_seq_max_history grew by >10x last10 vs first10,
    # consider it unstable.
    h_growth_factor = None
    if len(h_seq_max_history) >= 50:
        first10 = float(np.mean(h_seq_max_history[:10]))
        last10 = float(np.mean(h_seq_max_history[-10:]))
        if first10 > 1e-6:
            h_growth_factor = last10 / first10

    result = {
        "pass": True,
        "reason": "all checks passed",
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "loss_first10_mean": float(np.mean(loss_history[:10])) if loss_history else None,
        "loss_last10_mean": float(np.mean(loss_history[-10:])) if loss_history else None,
        "grad_norm_max": float(np.max(grad_norm_history)) if grad_norm_history else None,
        "h_seq_max_first10": float(np.mean(h_seq_max_history[:10])) if h_seq_max_history else None,
        "h_seq_max_last10": float(np.mean(h_seq_max_history[-10:])) if h_seq_max_history else None,
        "h_seq_growth_factor": h_growth_factor,
    }

    # Heuristic flags (don't fail, but note)
    if h_growth_factor is not None and h_growth_factor > 10.0:
        result["pass"] = False
        result["reason"] = f"h_seq magnitude grew {h_growth_factor:.1f}x — instability"

    print(f"\n[probe:{label}] DONE in {elapsed:.1f}s  PASS={result['pass']}")
    if h_growth_factor is not None:
        print(f"[probe:{label}]   h_seq growth: {h_growth_factor:.2f}x ({result['h_seq_max_first10']:.2f} -> {result['h_seq_max_last10']:.2f})")
    print(f"[probe:{label}]   loss: {result['loss_first10_mean']:.4f} -> {result['loss_last10_mean']:.4f}")
    print(f"[probe:{label}]   max grad norm: {result['grad_norm_max']:.3f}")
    return result
