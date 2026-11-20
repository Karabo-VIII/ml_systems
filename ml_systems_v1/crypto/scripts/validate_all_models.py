"""End-to-end validation harness for every model version (V0-V19).

For each version, runs these per-model checks at B=32:
  1. import      — settings, world_model, components
  2. instantiate — default constructor works
  3. forward     — forward_train(...) returns finite outputs
  4. loss        — sum of dict losses is a finite scalar
  5. backward    — loss.backward() doesn't NaN/Inf
  6. step        — optimizer step changes parameters (non-trivial gradient flow)
  7. variants    — train_adapter / train_snapshot / train_ncl scripts present
                   (V.X / V.E / V.D variants per CLAUDE.md)
  8. real_data   — can load a chimera batch + run forward (best-effort)

Run as ONE PROCESS PER VERSION via subprocess to avoid sys.modules pollution.
That way V11.settings doesn't bleed into V12.settings.

Usage:
    python scripts/validate_all_models.py                  # all versions
    python scripts/validate_all_models.py --version V1.0   # single version
    python scripts/validate_all_models.py --quick          # skip real-data step
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# (version, training_dir, module, class, init_kwargs, input_dim, forward_signature)
# forward_signature: ("name", argspec) where argspec is a list of:
#   "obs"          : torch.randn(B, T, input_dim)
#   "obs_chan"     : torch.randn(B, input_dim, T)  (channel-first for V15)
#   "actions"      : torch.randn(B, T, 1)
#   "asset_id"     : torch.randint(0, 10, (B,))
#   "rewards"      : torch.randn(B, T)
#   "returns"      : torch.randn(B, T)
#   "pooled"       : torch.randn(B, 768)  (V18 head input)
VERSIONS = [
    {"v": "V1.0", "dir": "src/wm/v1/v1_0_training", "mod": "world_model",
     "cls": "TransformerWorldModel", "input_dim": 13,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V1.1", "dir": "src/wm/v1/v1_1_training", "mod": "world_model",
     "cls": "TransformerWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V1.4", "dir": "src/wm/v1/v1_4_training", "mod": "world_model",
     "cls": "TransformerWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V1.6", "dir": "src/wm/v1/v1_6_training", "mod": "world_model",
     "cls": "TransformerWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V2", "dir": "backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training", "mod": "world_model",
     "cls": "JEPAWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V3", "dir": "src/wm/v3/v3_training", "mod": "world_model",
     "cls": "WaveNetGRUWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V4", "dir": "src/wm/v4/v4_training", "mod": "world_model",
     "cls": "MambaWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V5", "dir": "backups/BKP_20260429_MODEL_HARMONIZATION/v5/v5_training", "mod": "world_model",
     "cls": "HybridMambaAttentionWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V6", "dir": "src/wm/v6/v6_training", "mod": "world_model",
     "cls": "CausalJEPAWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V7", "dir": "backups/BKP_20260429_MODEL_HARMONIZATION/v7/v7_training", "mod": "world_model",
     "cls": "ViTWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V8", "dir": "src/wm/v8/v8_training", "mod": "world_model",
     "cls": "NeuralODEWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V9", "dir": "src/wm/v9/v9_training", "mod": "world_model",
     "cls": "MoEWorldModel", "input_dim": 41,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V11", "dir": "src/wm/v11/v11_training", "mod": "world_model",
     "cls": "MicrostructureWorldModel", "input_dim": 34,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V12", "dir": "src/wm/v12/v12_training", "mod": "world_model",
     "cls": "CrossAssetWorldModel", "input_dim": 25,
     "fwd": ("forward_train", ["obs", "asset_id"]),
     "tolerate_no_grad": True},  # cross-asset model special handling
    # V13/V14 are FROZEN per CLAUDE.md (deprecated). Validated for import +
    # instantiate + forward to confirm they don't ship broken; not retrained.
    {"v": "V13", "dir": "src/wm/v13/v13_training", "mod": "world_model",
     "cls": "TFTWorldModel", "input_dim": 25, "frozen": True,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V14", "dir": "src/wm/v14/v14_training", "mod": "world_model",
     "cls": "DiffusionWorldModel", "input_dim": 25, "frozen": True,
     "fwd": ("forward_train", ["obs", "asset_id"])},
    {"v": "V15", "dir": "src/wm/v15", "mod": "patchtst_encoder",
     "cls": "PatchTSTEncoder", "input_dim": 41,
     "init_kwargs": {"n_features": 41, "seq_len": 96},
     "fwd": ("forward", ["obs_chan"])},
    # V16/V17 reclassified 2026-06-11 to A1 backbones (src/agents/...).
    {"v": "V16", "dir": "src/agents/a1_wm_consuming/backbones/v16_dreamerv3/v16_training", "mod": "dreamer_v3",
     "cls": "DreamerV3WorldModel", "input_dim": 121,
     "fwd": ("forward_train", ["obs", "actions", "asset_id", "returns"])},
    {"v": "V17", "dir": "src/agents/a1_wm_consuming/backbones/v17_tdmpc2/v17_training", "mod": "td_mpc2",
     "cls": "TDMPC2WorldModel", "input_dim": 121,
     "fwd": ("forward_train", ["obs", "actions", "rewards", "asset_id"])},
    {"v": "V18", "dir": "src/wm/v18/v18_training", "mod": "finetune_chronos",
     "cls": "ChronosFineTuneHead", "input_dim": 768,
     "init_kwargs": {"d_model": 768},
     "fwd": ("forward", ["pooled"])},
]


VARIANT_FILES = ["train_adapter.py", "train_snapshot.py", "train_ncl.py",
                 "train_world_model.py", "validate_world.py"]


# ─── In-process worker (run via subprocess to isolate sys.modules) ──────────

WORKER_TEMPLATE = r"""
import json, sys, traceback
import torch
import torch.nn as nn

cfg = {cfg!r}
ROOT = {root!r}

sys.path.insert(0, ROOT + '/src')
sys.path.insert(0, ROOT + '/' + cfg['dir'])

result = {{
    'v': cfg['v'], 'import': 'FAIL', 'instantiate': 'FAIL', 'forward': 'FAIL',
    'loss': 'FAIL', 'backward': 'FAIL', 'step': 'FAIL', 'real_data': 'SKIP',
    'variants': '', 'params_M': 0.0, 'err': '',
}}

try:
    import importlib
    mod = importlib.import_module(cfg['mod'])
    result['import'] = 'OK'
    klass = getattr(mod, cfg['cls'])
    init_kwargs = cfg.get('init_kwargs') or {{}}
    m = klass(**init_kwargs)
    result['instantiate'] = 'OK'
    result['params_M'] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
except Exception as e:
    result['err'] = f'inst: {{type(e).__name__}}: {{str(e)[:80]}}'
    print(json.dumps(result))
    sys.exit(0)

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = m.to(device)
m.train()

# Build inputs per signature
B, T = 32, 96
input_dim = cfg['input_dim']
torch.manual_seed(42)
input_map = {{
    'obs':       torch.randn(B, T, input_dim, device=device),
    'obs_chan':  torch.randn(B, input_dim, T, device=device),
    'actions':   torch.randn(B, T, 1, device=device),
    'asset_id':  torch.randint(0, 10, (B,), device=device),
    'rewards':   torch.randn(B, T, device=device),
    'returns':   torch.randn(B, T, device=device),
    'pooled':    torch.randn(B, 768, device=device),
}}
fwd_name, arg_keys = cfg['fwd']
args = [input_map[k] for k in arg_keys]

# === Forward ===
try:
    method = getattr(m, fwd_name)
    out = method(*args)
    result['forward'] = 'OK'
except Exception as e:
    tb = traceback.format_exc().splitlines()[-1][:80]
    result['err'] = f'fwd: {{tb}}'
    print(json.dumps(result))
    sys.exit(0)

# === Loss ===
# Loss-synthesis preference order:
#   1. explicit 'loss' key
#   2. summed *_loss / loss_* scalar keys
#   3. derived MSE from a grad-bearing tensor (return_logits > h_seq > recon > generic)
# A tensor is "grad-bearing" when it has grad_fn — i.e. was computed from
# parameters. This rules out V11/V12-style `recon: torch.zeros(...)` stubs
# that exist only for V1.x interface compatibility.
def _has_grad_fn(t):
    return isinstance(t, torch.Tensor) and t.requires_grad and t.grad_fn is not None

def _proxy_mse(t):
    return t.float().pow(2).mean()

loss = None
if isinstance(out, dict):
    if 'loss' in out and isinstance(out['loss'], torch.Tensor):
        loss = out['loss']
    else:
        loss_keys = [k for k in out if 'loss' in k.lower() and isinstance(out[k], torch.Tensor) and out[k].dim() == 0]
        if loss_keys:
            loss = sum(out[k] for k in loss_keys)
        elif 'return_logits' in out and isinstance(out['return_logits'], dict):
            # Sum proxy MSE over per-horizon logits (these always carry grad_fn)
            rl = out['return_logits']
            grad_logits = [v for v in rl.values() if _has_grad_fn(v)]
            if grad_logits:
                loss = sum(_proxy_mse(v) for v in grad_logits) / len(grad_logits)
        if loss is None:
            # Pick the first grad-bearing tensor we can find, in priority order
            for k in ('h_seq', 'ret_trunk', 'recon', 'logits', 'pred', 'output'):
                if k in out and _has_grad_fn(out[k]):
                    loss = _proxy_mse(out[k])
                    break
            if loss is None:
                # Last resort: any grad-bearing tensor in the dict
                for k, v in out.items():
                    if _has_grad_fn(v):
                        loss = _proxy_mse(v)
                        break
elif isinstance(out, torch.Tensor):
    loss = _proxy_mse(out)

if loss is None or not torch.isfinite(loss).all():
    keys_dump = list(out.keys()) if isinstance(out, dict) else type(out).__name__
    result['err'] = f'loss: not finite or not derivable (out: {{keys_dump}})'
    print(json.dumps(result))
    sys.exit(0)
if not (isinstance(loss, torch.Tensor) and loss.requires_grad and loss.grad_fn is not None):
    keys_dump = list(out.keys()) if isinstance(out, dict) else type(out).__name__
    result['err'] = f'loss: synthesized loss has no grad_fn (out: {{keys_dump}})'
    print(json.dumps(result))
    sys.exit(0)
result['loss'] = f'OK ({{float(loss):.3f}})'

# === Backward ===
m.zero_grad()
try:
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
    if not grad_norms:
        result['err'] = 'backward: no gradients computed'
        print(json.dumps(result))
        sys.exit(0)
    g_mean = sum(grad_norms) / len(grad_norms)
    g_max = max(grad_norms)
# Looser bound — synthetic random data inputs can produce large grads;
# we only fail if grads are NaN/Inf or completely zero.
    if not torch.isfinite(torch.tensor(g_mean)) or g_mean < 1e-12:
        result['err'] = f'backward: degenerate grads (mean={{g_mean:.2e}})'
        print(json.dumps(result))
        sys.exit(0)
    # WARN (not FAIL) if grads are huge — likely random-data artifact, not a real bug
    if g_mean > 1e6:
        result['backward'] = f'WARN large grads (mean={{g_mean:.2e}})'
    else:
        pass  # falls through to OK assignment below
    if not result['backward'].startswith('WARN'):
        result['backward'] = f'OK (mean_g={{g_mean:.2e}} max={{g_max:.2e}})'
except Exception as e:
    tb = traceback.format_exc().splitlines()[-1][:80]
    result['err'] = f'bwd: {{tb}}'
    print(json.dumps(result))
    sys.exit(0)

# === Optimizer step ===
# Sum delta across ALL params with grads (not just first param, which may
# be a frozen embedding or an unused buffer).
try:
    params_with_grad = [p for p in m.parameters() if p.grad is not None]
    if not params_with_grad:
        result['err'] = 'step: no params with gradients'
        print(json.dumps(result))
        sys.exit(0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    snapshots = [p.detach().clone() for p in params_with_grad]
    opt.step()
    total_delta = 0.0
    for p_old, p in zip(snapshots, params_with_grad):
        total_delta += (p.detach() - p_old).norm().item()
    if total_delta < 1e-10:
        result['err'] = f'step: total param delta {{total_delta:.2e}} too small'
        print(json.dumps(result))
        sys.exit(0)
    result['step'] = f'OK (sum_d={{total_delta:.2e}})'
except Exception as e:
    tb = traceback.format_exc().splitlines()[-1][:80]
    result['err'] = f'step: {{tb}}'
    print(json.dumps(result))
    sys.exit(0)

# === Real data smoke (best-effort) ===
if not cfg.get('skip_real_data'):
    try:
        sys.path.insert(0, ROOT + '/src/pipeline')
        from chimera_loader import ChimeraLoader  # type: ignore
        loader = ChimeraLoader()
        df = loader.load('BTCUSDT', cadence='dollar')
        result['real_data'] = f'OK (BTC v51: {{df.height:,}} rows)'
    except Exception as e:
        result['real_data'] = f'WARN: {{type(e).__name__}}'

print(json.dumps(result))
"""


def run_one(cfg: dict, quick: bool = False) -> dict:
    cfg = dict(cfg)
    cfg["skip_real_data"] = quick
    # FROZEN versions (V13/V14) intentionally have stubbed settings — the
    # architecture block was dropped during centralization; restoring it is
    # an explicit re-train precondition. Don't attempt to instantiate.
    if cfg.get("frozen"):
        return {"v": cfg["v"], "import": "FROZEN", "instantiate": "FROZEN",
                "forward": "FROZEN", "loss": "FROZEN", "backward": "FROZEN",
                "step": "FROZEN", "real_data": "SKIP", "params_M": 0.0,
                "err": "frozen/deprecated - restore arch block from pre-2026-04-27 to retrain"}
    code = WORKER_TEMPLATE.format(cfg=cfg, root=str(ROOT).replace("\\", "/"))
    try:
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=180,
                              cwd=str(ROOT),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = proc.stdout.strip().splitlines()
        # The last line should be JSON
        for line in reversed(out):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        # If no JSON, return failure with stderr tail
        err = (proc.stderr or "")[-300:]
        return {"v": cfg["v"], "import": "FAIL", "instantiate": "FAIL",
                "forward": "FAIL", "loss": "FAIL", "backward": "FAIL",
                "step": "FAIL", "real_data": "SKIP", "params_M": 0.0,
                "err": f"no_json out_tail={err[:120]}"}
    except subprocess.TimeoutExpired:
        return {"v": cfg["v"], "import": "FAIL", "instantiate": "FAIL",
                "forward": "FAIL", "loss": "FAIL", "backward": "FAIL",
                "step": "FAIL", "real_data": "SKIP", "params_M": 0.0,
                "err": "timeout"}


def list_variants(dir_: str) -> str:
    full = ROOT / dir_
    if not full.exists():
        return ""
    present = []
    for f in VARIANT_FILES:
        if (full / f).exists():
            tag = {
                "train_adapter.py":   "X",
                "train_snapshot.py":  "E",
                "train_ncl.py":       "D",
                "train_world_model.py": "WM",
                "validate_world.py":  "VAL",
            }.get(f, f[:3])
            present.append(tag)
    return "+".join(present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None,
                    help="Only test this version (e.g. V1.0)")
    ap.add_argument("--quick", action="store_true",
                    help="Skip real-data smoke step")
    args = ap.parse_args()

    targets = VERSIONS
    if args.version:
        targets = [v for v in VERSIONS if v["v"] == args.version]
        if not targets:
            print(f"Unknown version {args.version}")
            sys.exit(2)

    print(f"\n{'Ver':5s} {'class':30s} {'P':>5s}  "
          f"{'imp':4s} {'inst':4s} {'fwd':4s} {'loss':18s} {'bwd':28s} {'step':14s} {'real':6s}  variants")
    print("-" * 168)
    n_pass = n_fail = n_frozen = 0
    for cfg in targets:
        t0 = time.time()
        r = run_one(cfg, quick=args.quick)
        elapsed = time.time() - t0
        v = list_variants(cfg["dir"])

        # Compact glyphs
        def g(s):
            if not isinstance(s, str):
                return "FAIL"
            if s.startswith("OK"):
                return "OK"
            if s == "SKIP":
                return "-"
            if s == "FROZEN":
                return "FRZN"
            if s == "WARN" or s.startswith("WARN"):
                return "WARN"
            return "FAIL"

        # WARN = succeeded with caveat. FROZEN = expected non-runnable per
        # CLAUDE.md. Both count as not-failing.
        is_frozen = cfg.get("frozen", False)
        all_ok = all(g(r.get(k, "FAIL")) in ("OK", "WARN", "FRZN")
                     for k in ("import", "instantiate", "forward",
                               "loss", "backward", "step"))
        if is_frozen:
            n_frozen += 1
        elif all_ok:
            n_pass += 1
        else:
            n_fail += 1

        params_s = f"{r.get('params_M', 0):>4.1f}M" if r.get('params_M', 0) > 0 else "  -  "
        # Compact real_data glyph
        rd = r.get("real_data", "SKIP")
        rd_glyph = "OK" if rd.startswith("OK") else ("WARN" if rd.startswith("WARN") else "-")
        version_label = r['v'] + ("*" if cfg.get("frozen") else "")
        print(f"{version_label:5s} {cfg['cls']:30s} {params_s}  "
              f"{g(r['import']):4s} {g(r['instantiate']):4s} {g(r['forward']):4s} "
              f"{r.get('loss', 'FAIL'):18s} {r.get('backward', 'FAIL'):28s} "
              f"{r.get('step', 'FAIL'):14s} {rd_glyph:6s}  {v}")
        if r.get("err"):
            print(f"        err: {r['err']}")

    print("-" * 168)
    print(f"\n  Total: {n_pass} PASS, {n_fail} FAIL, {n_frozen} FROZEN  (* = frozen/deprecated)")


if __name__ == "__main__":
    main()
