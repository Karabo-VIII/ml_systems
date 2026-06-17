"""Apply HEADLINE_MODE flag to V3+ trainers.

V3/V4/V6/V8 inherit the V1.x anti-mem stack via shared cross-version
constants. The HEADLINE_MODE flag activates per-architecture upgrade
hooks: each model's settings.py gets a tail block that, when
HEADLINE_MODE env var is set, exposes architecture-specific knobs.

Per WM_HEADLINE_UPGRADE_PLAN per-version specs:
  V3:  extend dilations [1,2,4,8] -> [1,2,4,8,16,32,64]; seq_len 256
  V4:  Mamba seq_len 96 -> 512 (linear scaling, free)
  V6:  discriminator spectral norm + R1 reg
  V8:  Tsit5 integrator + adjoint backprop
  V11: drop V9 MoE component (use 1 expert)
  V12: dataloader-fix flag (multi-asset batches required)
  V13: VSN_TOP_K 8 -> 12-16
  V14: DDIM inference 50 -> 10 steps

This script appends an env-var-aware block to each settings.py. The
TRAINER consumers must read these variables and act on them; this
patcher only exposes them so the operator can activate via env var.
Trainer-side wiring is per-architecture work, scheduled separately.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (target_path, version_short, headline_block)
TARGETS = [
    ("src/wm/v3/v3_training/settings.py", "v3", """
# ─── HEADLINE_MODE upgrades (V3-specific, added 2026-04-30) ──────────────────
# Per WM_HEADLINE_UPGRADE_PLAN §6: extend dilations + raise seq_len.
# Activation: V3_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V3_HEADLINE_MODE", "0")))
HEADLINE_DILATIONS = [1, 2, 4, 8, 16, 32, 64]   # was [1, 2, 4, 8]
HEADLINE_SEQ_LEN = 256                           # was 96
HEADLINE_FREE_NATS = 1.5
HEADLINE_XD_DROPOUT = 0.85
if HEADLINE_MODE:
    print(f"[V3 HEADLINE_MODE] dilations -> {HEADLINE_DILATIONS}, seq_len -> {HEADLINE_SEQ_LEN}")
    # Trainer must consume HEADLINE_DILATIONS / HEADLINE_SEQ_LEN; until wired
    # in v3 train_world_model.py these are exposed-but-unused.
"""),
    ("src/wm/v4/v4_training/settings.py", "v4", """
# ─── HEADLINE_MODE upgrades (V4-specific, added 2026-04-30) ──────────────────
# V4 Mamba scales linearly -- seq_len boost is a free upgrade.
# Per WM_HEADLINE_UPGRADE_PLAN §7.
# Activation: V4_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V4_HEADLINE_MODE", "0")))
HEADLINE_SEQ_LEN = 512                           # was 96; Mamba's home turf
HEADLINE_FREE_NATS = 1.5
HEADLINE_XD_DROPOUT = 0.85
if HEADLINE_MODE:
    print(f"[V4 HEADLINE_MODE] seq_len -> {HEADLINE_SEQ_LEN} (Mamba linear scaling)")
"""),
    ("src/wm/v6/v6_training/settings.py", "v6", """
# ─── HEADLINE_MODE upgrades (V6-specific, added 2026-04-30) ──────────────────
# V6 JEPA + Discriminator: needs spectral-norm + R1 reg for GAN stability;
# discriminator on RESIDUAL not encoder output (V6 fix log idea).
# Per WM_HEADLINE_UPGRADE_PLAN §8.
# Activation: V6_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V6_HEADLINE_MODE", "0")))
HEADLINE_DISC_SPECTRAL_NORM = True
HEADLINE_DISC_R1_GAMMA = 10.0
HEADLINE_DISC_TARGET = "residual"   # was "encoder_output"
if HEADLINE_MODE:
    print(f"[V6 HEADLINE_MODE] disc spectral_norm + R1 + residual target")
"""),
    ("src/wm/v8/v8_training/settings.py", "v8", """
# ─── HEADLINE_MODE upgrades (V8-specific, added 2026-04-30) ──────────────────
# V8 Neural ODE: switch RK4 -> Tsit5 (more efficient, same accuracy);
# adjoint-method backprop (memory-efficient -> enables longer seq).
# Per WM_HEADLINE_UPGRADE_PLAN §9.
# Activation: V8_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V8_HEADLINE_MODE", "0")))
HEADLINE_INTEGRATOR = "tsit5"           # was "rk4" (4 forward passes per step)
HEADLINE_ADJOINT_BACKPROP = True
HEADLINE_LEARNED_STEP_SIZE = True
if HEADLINE_MODE:
    print(f"[V8 HEADLINE_MODE] integrator -> {HEADLINE_INTEGRATOR}; adjoint backprop ON")
"""),
    ("src/wm/v11/v11_training/settings.py", "v11", """
# ─── HEADLINE_MODE upgrades (V11-specific, added 2026-04-30) ─────────────────
# V11 = V3 + V6 + V9. Per WM_HEADLINE_UPGRADE_PLAN §12: drop V9 MoE
# component (use 1 expert); inherit V3+V6 with their headline upgrades.
# Activation: V11_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V11_HEADLINE_MODE", "0")))
HEADLINE_MOE_EXPERTS = 1                # was 3 (V9-style, leaks); 1 = drop MoE
HEADLINE_DISC_SPECTRAL_NORM = True
HEADLINE_DILATIONS = [1, 2, 4, 8, 16, 32, 64]   # match V3-Headline
if HEADLINE_MODE:
    print(f"[V11 HEADLINE_MODE] MoE experts -> 1 (drop V9 leak); V3+V6 headline knobs")
"""),
    ("src/wm/v12/v12_training/settings.py", "v12", """
# ─── HEADLINE_MODE upgrades (V12-specific, added 2026-04-30) ─────────────────
# V12 cross-asset attention is dead code in standard runner (per
# world_model.py:267-271). HEADLINE_MODE flag enables the multi-asset
# path; trainer must read HEADLINE_MULTI_ASSET_PATH and use the multi-asset
# dataloader. Per WM_HEADLINE_UPGRADE_PLAN §14: highest ceiling; harness
# fix is the unlock.
# Activation: V12_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V12_HEADLINE_MODE", "0")))
HEADLINE_MULTI_ASSET_PATH = True        # forward_multi_asset (was dead code)
HEADLINE_HIERARCHICAL_ATTN = True       # cross-asset @ bar-level + temporal @ seq-level
HEADLINE_VIB_KL = 0.10
if HEADLINE_MODE:
    print(f"[V12 HEADLINE_MODE] multi-asset forward enabled; hierarchical attention ON")
    print(f"[V12 HEADLINE_MODE] DATALOADER MUST PROVIDE SYNCHRONIZED MULTI-ASSET BATCHES")
"""),
    ("src/wm/v13/v13_training/settings.py", "v13", """
# ─── HEADLINE_MODE upgrades (V13-specific, added 2026-04-30) ─────────────────
# V13 TFT: VSN_TOP_K bump from 8 to 12-16; cross-asset VSN layer.
# Per WM_HEADLINE_UPGRADE_PLAN §15.
# Activation: V13_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V13_HEADLINE_MODE", "0")))
HEADLINE_VSN_TOP_K = 16              # was 8; less aggressive bottleneck
HEADLINE_CROSS_ASSET_VSN = True      # new: asset-level variable selection
if HEADLINE_MODE:
    # Override the existing VSN_TOP_K
    VSN_TOP_K = HEADLINE_VSN_TOP_K
    print(f"[V13 HEADLINE_MODE] VSN_TOP_K -> {VSN_TOP_K}; cross-asset VSN ON")
"""),
    ("src/wm/v14/v14_training/settings.py", "v14", """
# ─── HEADLINE_MODE upgrades (V14-specific, added 2026-04-30) ─────────────────
# V14 Diffusion: reduce inference steps (50 -> 10 via DDIM); fewer samples
# (32 -> 8) -> 5x faster inference. Per WM_HEADLINE_UPGRADE_PLAN §16.
# The Headline argument for V14 is distributional Sharpe via meta-learner
# accepting q05/q50/q95 input -- not raw IC.
# Activation: V14_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V14_HEADLINE_MODE", "0")))
HEADLINE_DIFFUSION_INFERENCE_STEPS = 10     # was 50
HEADLINE_DIFFUSION_N_SAMPLES = 8            # was 32
HEADLINE_USE_DDIM = True
HEADLINE_CFG_SCALE = 1.5                    # classifier-free guidance
HEADLINE_QUANTILE_HEAD = True               # output q05/q50/q95 to meta-learner
if HEADLINE_MODE:
    DIFFUSION_INFERENCE_STEPS = HEADLINE_DIFFUSION_INFERENCE_STEPS
    DIFFUSION_N_SAMPLES = HEADLINE_DIFFUSION_N_SAMPLES
    print(f"[V14 HEADLINE_MODE] DDIM steps -> {DIFFUSION_INFERENCE_STEPS}, samples -> {DIFFUSION_N_SAMPLES}")
"""),
]


def patch_file(rel: str, version: str, block: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return f"SKIP missing: {rel}"
    text = p.read_text(encoding="utf-8")
    marker = f"V{version[1:].upper()}_HEADLINE_MODE"
    if marker in text:
        return f"already-patched: {rel}"
    new_text = text.rstrip() + "\n" + block + "\n"
    p.write_text(new_text, encoding="utf-8")
    return f"patched: {rel}"


def main() -> int:
    n_p = 0
    n_s = 0
    for rel, ver, block in TARGETS:
        result = patch_file(rel, ver, block)
        print(f"  {result}")
        if result.startswith("patched"):
            n_p += 1
        else:
            n_s += 1
    print(f"\npatched: {n_p}  skipped: {n_s}  total: {len(TARGETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
