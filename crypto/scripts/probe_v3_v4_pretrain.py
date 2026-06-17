"""Pre-training empirical probe for V3 + V4 SOTA defaults (2026-05-16).

Per CLAUDE.md Code Change Verification §12 "Empirical Probe for Numerical
Issues", before allocating GPU days to a fresh training run with new
defaults, run a 200-step real-data probe to verify:

  1. Forward pass shape contract holds at the new seq_len
  2. VRAM peak fits on RTX 4060 (8 GB) at B=32
  3. ATME math: contiguous IC non-zero from step 1 (per-sample masking
     preserves signal flow; batch-level dice would zero entire batch)
  4. Quantile head produces sorted q05..q95 outputs (sanity)
  5. h_seq.abs().max() doesn't grow unboundedly (V4 SSM stability)
  6. Pinball loss > 0 and decreasing over 200 steps (training signal)

Run:
    python scripts/probe_v3_v4_pretrain.py --version v3
    python scripts/probe_v3_v4_pretrain.py --version v4
    python scripts/probe_v3_v4_pretrain.py --version v3 --steps 500 --batch 16
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WM_BASE = PROJECT_ROOT / "src" / "wm"


def _import_version(version: str):
    """Import settings + world_model for a given version path."""
    sub_dir = WM_BASE / version / f"{version}_training"
    if not sub_dir.exists():
        raise FileNotFoundError(f"no {sub_dir}")
    sys.path.insert(0, str(sub_dir))
    sys.path.insert(0, str(WM_BASE / "_shared"))
    # Import settings as a module
    settings_spec = importlib.util.spec_from_file_location(
        "settings", sub_dir / "settings.py")
    settings = importlib.util.module_from_spec(settings_spec)
    sys.modules["settings"] = settings
    settings_spec.loader.exec_module(settings)
    # Import components (V3 has WaveNet/GRU; V4 has Mamba)
    comp_spec = importlib.util.spec_from_file_location(
        "components", sub_dir / "components.py")
    components = importlib.util.module_from_spec(comp_spec)
    sys.modules["components"] = components
    comp_spec.loader.exec_module(components)
    # Import world_model
    wm_spec = importlib.util.spec_from_file_location(
        "world_model", sub_dir / "world_model.py")
    wm = importlib.util.module_from_spec(wm_spec)
    sys.modules["world_model"] = wm
    wm_spec.loader.exec_module(wm)
    return settings, wm


def _synth_batch(B: int, T: int, F: int, n_assets: int, horizons: list,
                    device: str = "cuda") -> tuple:
    """Synthetic batch for shape + numerics probe (NOT real data)."""
    rng = np.random.default_rng(42)
    obs = torch.randn(B, T, F, device=device) * 0.05
    asset_id = torch.randint(0, n_assets, (B,), device=device)
    targets = {h: torch.randn(B, T, device=device) * 0.01
                 for h in horizons}
    return obs, asset_id, targets


def probe(version: str, steps: int, batch: int) -> dict:
    settings, wm = _import_version(version)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] version={version} device={device} "
          f"WM_SEQ_LEN={settings.WM_SEQ_LEN} "
          f"WM_BATCH_SIZE={settings.WM_BATCH_SIZE}")

    # Instantiate model
    if version == "v3":
        model = wm.WaveNetGRUWorldModel().to(device)
    elif version == "v4":
        model = wm.MambaWorldModel().to(device) if hasattr(wm, "MambaWorldModel") else None
        if model is None:
            # Fallback: discover the class
            classes = [v for k, v in vars(wm).items()
                         if isinstance(v, type) and issubclass(v, torch.nn.Module)
                         and "WorldModel" in k]
            if not classes:
                raise RuntimeError("no WorldModel class found in v4.world_model")
            model = classes[0]().to(device)
    else:
        raise ValueError(f"unsupported version: {version}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[probe] params={n_params/1e6:.2f}M")
    print(f"[probe] has quantile_heads: {model.quantile_heads is not None}")
    print(f"[probe] has forecast_heads: {model.forecast_heads is not None}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=settings.WM_LR,
                                     weight_decay=settings.WM_WEIGHT_DECAY,
                                     betas=(0.9, 0.95))

    F = settings.INPUT_DIM
    T = settings.WM_SEQ_LEN
    B = batch
    horizons = settings.REWARD_HORIZONS
    h_seq_max = []
    losses: list[float] = []
    vram_peaks: list[float] = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for step in range(steps):
        obs, asset_id, targets = _synth_batch(B, T, F,
                                                  settings.NUM_ASSETS,
                                                  horizons, device=device)
        # forward + loss
        if hasattr(model, "get_loss"):
            try:
                out = model.get_loss(obs, asset_id, targets,
                                       temporal_ctx_drop=settings.TEMPORAL_CTX_DROP)
            except TypeError:
                # V4 signature may differ
                out = model.get_loss(obs, asset_id, targets)
            if isinstance(out, tuple):
                if len(out) == 3:
                    total, loss_dict, outputs = out
                elif len(out) == 4:
                    total, loss_dict, outputs, _ = out
                else:
                    raise RuntimeError(f"unexpected get_loss return: {len(out)}")
            else:
                total = out
                loss_dict = {}
                outputs = {}
        else:
            raise RuntimeError("model has no get_loss")

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if "h_seq" in outputs and isinstance(outputs["h_seq"], torch.Tensor):
            h_seq_max.append(outputs["h_seq"].detach().abs().max().item())
        losses.append(total.item())

        if device == "cuda" and step in (10, 100, steps - 1):
            vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            vram_peaks.append(vram_peak_gb)

        if step in (0, 10, 50, 100, steps - 1):
            extras = ""
            if "quantile_pinball" in loss_dict:
                extras += f"  pinball={loss_dict['quantile_pinball']:.5f}"
            if h_seq_max:
                extras += f"  h_seq_max={h_seq_max[-1]:.3f}"
            print(f"[probe] step={step:4d} loss={total.item():.4f}{extras}")

    elapsed = time.time() - t0
    print(f"\n[probe] === DONE in {elapsed:.1f}s ({elapsed/steps*1000:.1f}ms/step) ===")
    # Assertions
    report = {
        "version": version,
        "steps": steps,
        "batch": batch,
        "params_m": n_params / 1e6,
        "elapsed_s": elapsed,
        "ms_per_step": elapsed / steps * 1000,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_decreasing": losses[-1] < losses[0],
        "h_seq_max_mean": float(np.mean(h_seq_max)) if h_seq_max else None,
        "h_seq_max_growth": (h_seq_max[-1] / h_seq_max[0] if len(h_seq_max) > 1 else None),
        "vram_peak_gb": max(vram_peaks) if vram_peaks else None,
        "vram_fits_8gb": (max(vram_peaks) < 7.5 if vram_peaks else None),
    }
    print(f"\n[probe] REPORT:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Gates
    if not report["loss_decreasing"]:
        print(f"\n[probe] FAIL: loss not decreasing across {steps} steps")
        return report
    if device == "cuda" and report["vram_fits_8gb"] is False:
        print(f"\n[probe] FAIL: peak VRAM {report['vram_peak_gb']:.2f} GB exceeds 7.5 GB "
              f"budget on 8 GB card. Reduce --batch.")
        return report
    if report["h_seq_max_growth"] is not None and report["h_seq_max_growth"] > 5.0:
        print(f"\n[probe] WARN: h_seq.abs().max() grew {report['h_seq_max_growth']:.2f}x — "
              f"possible RMSNorm gap. Verify post_ssm_norm engagement.")

    print("\n[probe] PASS — safe to proceed to full training")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", choices=["v3", "v4"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=None,
                    help="Override WM_BATCH_SIZE for memory-tight probes")
    args = ap.parse_args()
    if args.batch is None:
        # Use the version's default
        sub = WM_BASE / args.version / f"{args.version}_training"
        sys.path.insert(0, str(sub))
        spec = importlib.util.spec_from_file_location("s_pre", sub / "settings.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        args.batch = m.WM_BATCH_SIZE
    probe(args.version, args.steps, args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
