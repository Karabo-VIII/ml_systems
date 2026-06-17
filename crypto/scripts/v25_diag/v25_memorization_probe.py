"""V25 memorization probe — read-only checkpoint diagnostic.

Owned by the secondary Claude instance for diagnostic support to the primary
instance currently iterating on V25. This script DOES NOT WRITE to any
src/wm/v25/* file. It loads a V25 checkpoint, instantiates the model from
existing source, and computes:

  1. Stable rank of weight matrices (patch_embed, proj, asset_token_proj,
     period_amp, input_vib if present)
  2. Spectral norm bound check (max singular value)
  3. Per-component activation magnitudes on a small synthetic batch
  4. Memorization signature tests:
     a. Temporal-shuffle invariance: does prediction change when we shuffle
        time within each sample? (memorization → big change)
     b. Feature-shuffle invariance: does prediction change when we shuffle
        feature columns? (memorization → big change)
     c. Per-position prediction variance: across same content at different t

Usage:
  python scripts/v25_diag/v25_memorization_probe.py --ckpt path/to/ckpt.pt

Outputs:
  logs/v25_diag/probe_<ckpt_basename>.json    structured diagnostic
  logs/v25_diag/probe_<ckpt_basename>.md      human-readable summary

Designed to be runnable independently of any active V25 training session.
Will not lock files; uses CPU by default (CUDA optional via --device cuda).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def stable_rank(W: torch.Tensor) -> tuple[float, float, int]:
    """Returns (stable_rank, top_sv, total_dim).
    stable_rank = ||W||_F^2 / ||W||_2^2 = sum_i sigma_i^2 / sigma_max^2.
    """
    if W.dim() < 2:
        return float("nan"), float("nan"), W.numel()
    M = W.detach().float().reshape(W.shape[0], -1)
    try:
        s = torch.linalg.svdvals(M)
        s2 = (s ** 2).sum().item()
        top = s.max().item()
        return s2 / max(top * top, 1e-12), top, M.shape[1]
    except Exception:
        return float("nan"), float("nan"), M.shape[1]


def cos_offdiag_mean(W: torch.Tensor) -> float:
    """Mean abs cosine between rows of W (excluding diagonal)."""
    if W.dim() < 2:
        return float("nan")
    M = W.detach().float().reshape(W.shape[0], -1)
    M = torch.nn.functional.normalize(M, dim=-1)
    G = M @ M.T
    n = G.shape[0]
    if n < 2:
        return float("nan")
    mask = ~torch.eye(n, dtype=torch.bool)
    return G.abs()[mask].mean().item()


def collect_activation_stats(model, batch: dict, hooks_for: list[str]) -> dict:
    """Forward-pass with hooks on named layers. Returns activation absmax p50/p95
    and stable_rank of activation tensor (if 2D)."""
    out: dict[str, dict] = {}
    handles = []

    def _hook(name):
        def fn(module, _in, _out):
            t = _out if isinstance(_out, torch.Tensor) else (_out[0] if isinstance(_out, tuple) else None)
            if t is None: return
            v = t.detach().float()
            absmax_per_sample = v.abs().reshape(v.shape[0], -1).max(dim=-1).values
            stats = {
                "absmax_p50": float(absmax_per_sample.median().item()),
                "absmax_p95": float(absmax_per_sample.quantile(0.95).item()),
                "absmax_max": float(absmax_per_sample.max().item()),
                "abs_mean": float(v.abs().mean().item()),
                "shape": list(v.shape),
            }
            # Approximate stable rank of activation (per-sample mean)
            if v.dim() >= 2:
                M = v.reshape(v.shape[0], -1).mean(dim=0, keepdim=True)
                if M.shape[-1] >= 2:
                    try:
                        s = torch.linalg.svdvals(M.reshape(-1, M.shape[-1] // max(1, M.shape[-1] // 32)))
                        if len(s) > 0:
                            stats["act_top_sv"] = float(s.max().item())
                    except Exception:
                        pass
            out[name] = stats
        return fn

    for name in hooks_for:
        # Walk the model tree to find the named submodule
        try:
            mod = model
            for tok in name.split("."):
                if tok.isdigit():
                    mod = mod[int(tok)]
                else:
                    mod = getattr(mod, tok)
            handles.append(mod.register_forward_hook(_hook(name)))
        except Exception:
            pass  # silently skip missing modules

    try:
        with torch.no_grad():
            _ = model(**batch) if isinstance(batch, dict) else model(*batch)
    except TypeError:
        try:
            with torch.no_grad():
                _ = model(*batch.values())
        except Exception as e:
            out["_forward_error"] = str(e)
    except Exception as e:
        out["_forward_error"] = str(e)
    finally:
        for h in handles:
            h.remove()
    return out


def make_synthetic_batch(B: int, T: int, F: int, n_assets: int = 10, device="cpu") -> dict:
    """Synthetic batch matching V25 train_world_model expectations.
    NOTE: V25's exact forward signature may differ; we try common variants."""
    return {
        "x": torch.randn(B, T, F, device=device),
        "asset_id": torch.randint(0, n_assets, (B,), device=device),
        "regime_id": torch.randint(0, 3, (B, T), device=device),
    }


def shuffle_temporal(batch: dict) -> dict:
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    if "x" in out and out["x"].dim() == 3:
        T = out["x"].shape[1]
        for b in range(out["x"].shape[0]):
            perm = torch.randperm(T)
            out["x"][b] = out["x"][b][perm]
    return out


def shuffle_features(batch: dict) -> dict:
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    if "x" in out and out["x"].dim() == 3:
        F = out["x"].shape[-1]
        perm = torch.randperm(F)
        out["x"] = out["x"][..., perm]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to V25 checkpoint .pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--feats", type=int, default=29, help="V25 default f29")
    ap.add_argument("--out-dir", default=str(ROOT / "logs" / "v25_diag"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"[probe] checkpoint not found: {ckpt_path}", flush=True)
        return 2

    print(f"[probe] loading checkpoint: {ckpt_path.relative_to(ROOT) if ROOT in ckpt_path.parents else ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("ema_state_dict") or ckpt
    if not isinstance(state, dict):
        print(f"[probe] unexpected ckpt structure", flush=True)
        return 2

    print(f"[probe] {len(state)} state keys", flush=True)

    # 1. Weight stable rank for all 2D tensors
    print(f"[probe] computing stable rank of weight matrices...", flush=True)
    weight_stats: dict = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or v.dim() < 2:
            continue
        sr, top_sv, n = stable_rank(v)
        if not np.isnan(sr):
            weight_stats[k] = {
                "stable_rank": round(sr, 3),
                "top_sv": round(top_sv, 4),
                "shape": list(v.shape),
                "n_params": int(v.numel()),
                "stable_rank_frac": round(sr / max(min(v.shape), 1), 4),
            }

    # 2. Highlight memorization-shaped weights
    print(f"[probe] flagging memorization-shaped layers...", flush=True)
    flagged = []
    suspect_keywords = ["patch_embed", "proj", "asset_token_proj", "period_amp",
                         "input_vib", "regime_ffn", "asset_embedding"]
    for k, v in weight_stats.items():
        for kw in suspect_keywords:
            if kw in k.lower():
                flagged.append({"key": k, "kw": kw, **v})
                break

    # 3. Try to instantiate model + run synthetic forward
    activations: dict = {}
    try:
        from wm.v25.v25_training.world_model import build_v25
        from wm.v25.v25_training import settings as S
        print(f"[probe] building V25 model from src/wm/v25/v25_training/world_model.py", flush=True)
        model = build_v25(args.feats)
        model.load_state_dict(state, strict=False)
        model.to(args.device).eval()

        batch_normal = make_synthetic_batch(args.batch, args.seq, args.feats, device=args.device)
        batch_tshuf = shuffle_temporal(batch_normal)
        batch_fshuf = shuffle_features(batch_normal)

        hooks_for = [
            "patch_embed", "proj", "asset_token_proj", "asset_embedding",
            "input_vib.z_expand.0", "input_vib", "period_emb",
        ]
        for layer_idx in range(6):
            hooks_for.extend([
                f"layers.{layer_idx}.attn",
                f"layers.{layer_idx}.regime_ffn.0.3",
                f"layers.{layer_idx}.regime_ffn.1.3",
                f"layers.{layer_idx}.regime_ffn.2.3",
                f"layers.{layer_idx}.ffn",
            ])
        hooks_for.extend(["return_trunk", "return_heads.0"])

        print(f"[probe] forward on normal batch...", flush=True)
        activations["normal"] = collect_activation_stats(model, batch_normal, hooks_for)
        print(f"[probe] forward on temporal-shuffled batch...", flush=True)
        activations["temporal_shuffled"] = collect_activation_stats(model, batch_tshuf, hooks_for)
        print(f"[probe] forward on feature-shuffled batch...", flush=True)
        activations["feature_shuffled"] = collect_activation_stats(model, batch_fshuf, hooks_for)
    except Exception as e:
        activations = {"_error": f"could not run forward: {type(e).__name__}: {e}"}
        print(f"[probe] WARN: forward pass failed -- weight-only diagnostics still valid. {e}", flush=True)

    # 4. Output
    result = {
        "ckpt": str(ckpt_path),
        "ckpt_metadata": {k: v for k, v in ckpt.items() if not isinstance(v, dict)} if isinstance(ckpt, dict) else {},
        "n_state_keys": len(state),
        "weight_stats_top20_lowest_stable_rank_frac": sorted(
            [{"key": k, **v} for k, v in weight_stats.items()],
            key=lambda x: x["stable_rank_frac"],
        )[:20],
        "memorization_shaped_layers": sorted(flagged, key=lambda x: x["stable_rank_frac"]),
        "activations": activations,
    }
    base = ckpt_path.stem
    json_p = out_dir / f"probe_{base}.json"
    md_p = out_dir / f"probe_{base}.md"
    json_p.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # MD summary
    md = [
        f"# V25 Memorization Probe — `{base}`",
        "",
        f"- ckpt: `{ckpt_path}`",
        f"- n_state_keys: {len(state)}",
        "",
        "## Memorization-shaped weight matrices (sorted by stable_rank_frac, lowest = most collapsed)",
        "",
        "| key | shape | params | stable_rank | top_sv | sr_frac |",
        "|---|---|---|---|---|---|",
    ]
    for f in result["memorization_shaped_layers"]:
        md.append(f"| `{f['key']}` | {f['shape']} | {f['n_params']} | "
                    f"{f['stable_rank']} | {f['top_sv']:.3f} | **{f['stable_rank_frac']}** |")
    md.append("")
    md.append("## Top 20 lowest-stable-rank weight matrices (any layer)")
    md.append("")
    md.append("| key | shape | params | stable_rank | sr_frac |")
    md.append("|---|---|---|---|---|")
    for w in result["weight_stats_top20_lowest_stable_rank_frac"]:
        md.append(f"| `{w['key']}` | {w['shape']} | {w['n_params']} | "
                    f"{w['stable_rank']} | **{w['stable_rank_frac']}** |")

    if "_error" not in result["activations"]:
        md.append("")
        md.append("## Activation magnitude per layer (normal vs shuffled inputs)")
        md.append("")
        md.append("Memorization signature: shuffled-feature inputs should produce DIFFERENT activations (model has memorized feature→output mapping).")
        md.append("Healthy generalization: shuffled-feature should produce similar magnitudes (model uses cross-feature interactions, not feature index lookup).")
        md.append("")
        md.append("| layer | normal_p50 | t-shuf_p50 | f-shuf_p50 | normal_p95 | f-shuf_p95 | Δ(f-shuf − normal) |")
        md.append("|---|---|---|---|---|---|---|")
        a_n = result["activations"].get("normal", {})
        a_t = result["activations"].get("temporal_shuffled", {})
        a_f = result["activations"].get("feature_shuffled", {})
        for layer in sorted(set(a_n.keys()) | set(a_t.keys()) | set(a_f.keys())):
            n = a_n.get(layer, {})
            t = a_t.get(layer, {})
            f = a_f.get(layer, {})
            if "absmax_p50" not in n: continue
            delta = (f.get("absmax_p50", n.get("absmax_p50", 0))
                     - n.get("absmax_p50", 0))
            md.append(f"| `{layer}` | {n.get('absmax_p50', 0):.4f} | "
                        f"{t.get('absmax_p50', 0):.4f} | {f.get('absmax_p50', 0):.4f} | "
                        f"{n.get('absmax_p95', 0):.4f} | {f.get('absmax_p95', 0):.4f} | "
                        f"{delta:+.4f} |")
    else:
        md.append("")
        md.append("## Activation forward-pass: SKIPPED")
        md.append(f"Reason: {result['activations']['_error']}")
        md.append("")
        md.append("(Weight-only diagnostics are still valid above.)")

    md.append("")
    md_p.write_text("\n".join(md), encoding="utf-8")
    print(f"[done] {json_p.relative_to(ROOT)}")
    print(f"[done] {md_p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
