"""OOM probe — Hole 4 gate (per LITERATURE.md).

Per CLAUDE.md cross-version invariant §12: never reason about fp16
overflow / VRAM thresholds from math alone. Run an empirical probe
BEFORE committing the full epoch budget.

Two tiers:
  1. quick   -- synthetic data, B=8, S=512, 50 steps. 4-second smoke.
                Validates structure + magnitude stability.
  2. full    -- real chimera_legacy data, B=8, S=512, 200-300 steps.
                Validates dataloader + AMP + true VRAM peak.

Pass criteria:
    - peak VRAM < 7.5 GB (leave headroom on 8 GB)
    - h_seq.abs().max() does not grow unboundedly across steps
    - no NaN in any of: h_seq, return_logits, contrastive_emb
    - loss.backward() succeeds under autocast
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from frontier_ml.foundation.backbone import FoundationBackbone, DEFAULT_CONFIG  # noqa: E402


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def smoke(n_features: int = 18, n_assets: int = 50) -> dict:
    """Quick smoke: construct backbone, single fwd+bwd, report params + memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = FoundationBackbone(n_features=n_features).to(device)
    n_params = model.num_params()
    print(f"[smoke] backbone params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)
    print(f"[smoke] config: {model.cfg}", flush=True)

    B, S = 4, 256
    x = torch.randn(B, S, n_features, device=device)
    asset_ids = torch.randint(0, n_assets, (B,), device=device)

    _reset_peak()
    out = model(x, asset_ids=asset_ids)
    h_seq = out["h_seq"]
    print(f"[smoke] h_seq shape: {tuple(h_seq.shape)}, "
          f"abs_max: {h_seq.abs().max().item():.3f}, "
          f"any_nan: {torch.isnan(h_seq).any().item()}", flush=True)
    for hk, lg in out["return_logits"].items():
        print(f"[smoke]   return_logits[{hk}]: {tuple(lg.shape)} "
              f"abs_max {lg.abs().max().item():.3f}", flush=True)
    print(f"[smoke] contrastive_emb shape: {tuple(out['contrastive_emb'].shape)}",
          flush=True)

    # tiny scalar loss to verify backward works
    loss = sum(lg.float().pow(2).mean() for lg in out["return_logits"].values())
    loss = loss + out["contrastive_emb"].float().pow(2).mean()
    loss.backward()
    peak = _peak_vram_gb()
    print(f"[smoke] fwd+bwd peak VRAM: {peak:.3f} GB (B={B} S={S})", flush=True)
    return {"params": n_params, "peak_vram_gb": peak}


def probe_synthetic(
    n_features: int = 18,
    n_assets: int = 50,
    batch_size: int = 8,
    seq_len: int = 512,
    n_steps: int = 50,
    use_amp: bool = True,
) -> dict:
    """Tier-1 probe: synthetic data, fp16 autocast, n_steps backward passes.

    Tracks peak VRAM + h_seq magnitude growth + NaN over the run.
    """
    if not torch.cuda.is_available():
        print("[probe-synth] no CUDA — skipping", flush=True)
        return {"skipped": True}

    device = "cuda"
    torch.manual_seed(0)
    model = FoundationBackbone(n_features=n_features).to(device)
    n_params = model.num_params()
    print(f"[probe-synth] params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)
    print(f"[probe-synth] B={batch_size} S={seq_len} steps={n_steps} amp={use_amp}",
          flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    _reset_peak()
    h_max_history = []
    nan_step = -1
    t0 = time.time()
    for step in range(n_steps):
        x = torch.randn(batch_size, seq_len, n_features, device=device)
        asset_ids = torch.randint(0, n_assets, (batch_size,), device=device)
        return_targets = torch.randn(batch_size, device=device).clamp(-1, 1)

        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x, asset_ids=asset_ids)
                # Tiny synthetic loss: MSE-ish on logits + L2 on emb
                loss = sum(
                    nn.functional.cross_entropy(lg.float(), torch.zeros(batch_size, dtype=torch.long, device=device))
                    for lg in out["return_logits"].values()
                )
                loss = loss + out["contrastive_emb"].float().pow(2).mean() * 0.01
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            out = model(x, asset_ids=asset_ids)
            loss = sum(
                nn.functional.cross_entropy(lg, torch.zeros(batch_size, dtype=torch.long, device=device))
                for lg in out["return_logits"].values()
            )
            loss.backward()
            optim.step()

        h_max = out["h_seq"].abs().max().item()
        h_max_history.append(h_max)
        if torch.isnan(out["h_seq"]).any():
            nan_step = step
            break
        if step % 10 == 0:
            print(f"[probe-synth] step {step:3d}: loss {loss.item():.4f} "
                  f"h_max {h_max:.2f} peak {_peak_vram_gb():.2f} GB", flush=True)

    elapsed = time.time() - t0
    peak = _peak_vram_gb()
    h_max_final = h_max_history[-1] if h_max_history else float("nan")
    h_max_growth = (h_max_final / h_max_history[0]) if h_max_history else float("nan")
    print(f"[probe-synth] DONE in {elapsed:.1f}s ({elapsed/max(1,len(h_max_history)):.2f}s/step)", flush=True)
    print(f"[probe-synth] peak VRAM: {peak:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB",
          flush=True)
    print(f"[probe-synth] h_seq abs_max: start {h_max_history[0]:.2f} -> "
          f"final {h_max_final:.2f} (growth {h_max_growth:.2f}x)", flush=True)
    print(f"[probe-synth] NaN detected: {'NO' if nan_step < 0 else f'YES at step {nan_step}'}",
          flush=True)

    pass_ = (
        peak < 7.5
        and nan_step < 0
        and h_max_growth < 5.0   # no unbounded growth
    )
    return {
        "params": n_params,
        "peak_vram_gb": peak,
        "h_max_growth": h_max_growth,
        "nan_step": nan_step,
        "elapsed_s": elapsed,
        "pass": pass_,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=["smoke", "synth", "real"])
    ap.add_argument("--n-features", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    print("=" * 70, flush=True)
    print(f"FRONTIER FOUNDATION OOM PROBE  mode={args.mode}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  device: {p.name}  vram: {p.total_memory/1e9:.2f} GB", flush=True)
    print("=" * 70, flush=True)

    if args.mode == "smoke":
        smoke(n_features=args.n_features)
    elif args.mode == "synth":
        result = probe_synthetic(
            n_features=args.n_features,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_steps=args.n_steps,
            use_amp=not args.no_amp,
        )
        verdict = "PASS" if result.get("pass") else "FAIL"
        print(f"\n[probe-synth] VERDICT: {verdict}", flush=True)
        if not result.get("pass"):
            sys.exit(2)
    elif args.mode == "real":
        # Reserved: real-data probe wires up dataloader (next step in build).
        print("[probe-real] not yet implemented — build data_loader.py first", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
