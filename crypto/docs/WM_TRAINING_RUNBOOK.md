# WM Training Runbook — the exact commands, in order (so nothing goes wrong)

> 2026-06-10. Every command below is RWYB-verified (dry-run plan + a real first-step smoke for the V1.1 path).
> The two gotchas that would have wasted a run: (1) a bare `--model v1_1` **skips the base** (already trained) and
> trains only the secondary variants → you MUST pass `--force --only-base`; (2) the runs share one checkpoint path
> unless you pass `--run-tag` → use a distinct tag per run. Both are baked into the commands below.

## Phase 0 — Preflight (optional re-check; already passed)
Validates the data/config without training (gate + 5 validators + compile/data presence):
```
python src/run_all_training.py --features 41 --model v1_1 --dry-run
```
Expect: gate WARN-only (benign freshness debt), 0 FAILs, split clean. (Last run: PASS.)

## Phase 1 — V1.1 world-class A/B  (RUN THESE FIRST; ~3.5h each, ~0.5–3.3 GB VRAM, one at a time)

> **SHELL SYNTAX — READ (this project runs on PowerShell).** The `VAR=1 python ...` form below is **bash** inline-env
> syntax and **FAILS in PowerShell** (`V1_VSN=1 is not recognized as a cmdlet`). On PowerShell, set the env var FIRST,
> then run, then clean up so it doesn't leak into your next run:
> ```powershell
> $env:V1_VSN="1"; $env:V1_FORWARD_REGIME="1"; python src/run_all_training.py --features 41 --model v1_1 --force --only-base --run-tag vsn_fr; Remove-Item Env:V1_VSN, Env:V1_FORWARD_REGIME
> ```
> Apply the same `$env:VAR="1"; <cmd>; Remove-Item Env:VAR` pattern to every `VAR=1 ...` command below (the ablations,
> the V12 run). The `Remove-Item` matters: `$env:VAR` persists for the whole session and would silently leak into a
> later run. Commands with **no** `VAR=` prefix (the baseline, the decision gate) run as-is in both shells.

**1. Baseline (the control):**
```
python src/run_all_training.py --features 41 --model v1_1 --force --only-base --run-tag baseline
```

**2. World-class candidate (VSN gated-input + forward regime/move target):**
```
V1_VSN=1 V1_FORWARD_REGIME=1 python src/run_all_training.py --features 41 --model v1_1 --force --only-base --run-tag vsn_fr
```

**Optional ablations** — run ONLY if (2) beats (1) and you want to know *which* lever did it:
```
# VSN only:
V1_VSN=1 python src/run_all_training.py --features 41 --model v1_1 --force --only-base --run-tag vsn
# Forward-regime/move target only:
V1_FORWARD_REGIME=1 python src/run_all_training.py --features 41 --model v1_1 --force --only-base --run-tag fwd_regime
```
Checkpoints are isolated per tag: `models/wm/v1/v1_1/base/v1_1_f41_<tag>_wm_best_ema.pt`. No clobbering.

### DECISION GATE (after Phase 1) — compare on held-out COMPOUND, not IC
```
python src/wm/wm_promotion_gate.py        # the gate logic + champion record (runs/wm/champion.json)
```
- If **`vsn_fr` beats `baseline` on held-out compound** → the world-class direction works → do Phase 2 + wire the winner into the strat layer.
- If **not** → the daily-bar signal ceiling is confirmed empirically → the literal IC>0.10 path is HF/sub-bar data (a separate, larger build).

## Phase 2 — V12 cross-asset  (CONDITIONAL / a parallel cross-asset read; ~2h)
V12 is **BLOCKED in `run_all_training`**, so it runs via its trainer **directly** (this skips the pre-train gate, but the chimera already passed it in Phase 1 — same data). It has **no `--run-tag`** (its checkpoints live in their own `v12_*` namespace, so they can't clobber the V1.1 runs):
```
V12_HEADLINE_MODE=1 python src/wm/v12/v12_training/train_world_model.py --features 25
```
(Optionally run the gate first: `python src/pipeline/pre_train_gate.py --asset BTC`.)

## Memory fix (2026-06-10) — the ShIC OOM is fixed; re-runs are safe end-to-end
A V1.1 run died at ~1.5h in `compute_shuffled_ic` (the ShIC check fires every 10 epochs) with a host-RAM
OOM — it ran on the full multi-million-bar assets. Fixed: bars/asset capped at 90k + a float32 Pearson
(`src/anti_fragile.py`). The contiguous-IC `validate()` and other IC sites were scan-confirmed bounded
(batched), so the base path is clean end-to-end. **If a prior run died at the ShIC check, just re-run it.**

## What is NOT in this runbook (and why)
- **V22 / V24 / V25 — RETIRED, DO NOT RUN.** Dominated/broken: V22/V25 pure-iTransformer (`recon=torch.zeros`
  no anchor, ShIC=0.000 memorizing); V24's FFT clock-periodicity thesis is invalidated by its own dollar-bar
  data. See `docs/WM_COHORT_WORLDCLASS_VERDICT_2026_06_10.md`.
- **V3 / V4 / V6 / V8 / V13 / V23 — UPGRADED to world-class-capable** (real anchors + move/regime + VSN levers).
  Train them in **Phase 3, AFTER the V1.1 go/no-go validates the direction** (don't spend GPU-days replicating
  a refuted approach).

## Monitoring (all runs) — what's normal vs what to act on
- **NORMAL (don't panic):** the first ~25 steps show `gn=nan` and a frozen loss — that's the V1.1 LR-warmup + the NaN-guard skipping unstable steps. It recovers by ~step 26 (grad-norm finite, loss drops). Same warmup the IC-0.067 model went through.
- **AUTO-STOPS (let them work; don't kill early):** ShIC early-stop fires on memorization (ShIC declines); the V12 run also has a VIB-collapse guard (auto-stops if `kl` < 0.05 after anneal). Watch the `kl` line on the V12 run.
- A run stopping itself is the safety net working — read the final ShIC / compound before re-running.

## Phase 3 — the world-class cohort (ONLY after V1.1 `vsn_fr` wins the go/no-go)
The diverse anchored set (V3 gated-conv, V4 SSM, V6 JEPA, V8 Neural-ODE, V13 TFT-VSN) — each gets the same
flag-gated levers. **PINNED + VERIFIED 2026-06-11** (all 5 dry-run plan at f41 + preflight OK + `--run-tag` and
the env flags confirmed wired; the `run_all_training` registry was corrected to list f41 for the cohort, which
its `settings.py` + the f41 chimera data already supported). Run each (≈3.5–4.5 GPU-d, one at a time):
```
V3_VSN=1  V3_FORWARD_REGIME=1  python src/run_all_training.py --features 41 --model v3  --force --only-base --run-tag wc
V4_VSN=1  V4_FORWARD_REGIME=1  python src/run_all_training.py --features 41 --model v4  --force --only-base --run-tag wc
V6_VSN=1  V6_FORWARD_REGIME=1  python src/run_all_training.py --features 41 --model v6  --force --only-base --run-tag wc
V8_VSN=1  V8_FORWARD_REGIME=1  python src/run_all_training.py --features 41 --model v8  --force --only-base --run-tag wc
V13_VSN=1 V13_FORWARD_REGIME=1 python src/run_all_training.py --features 41 --model v13 --force --only-base --run-tag wc
```
Checkpoints isolate per model+tag (`v3_f41_wc_wm_*.pt`, …). f41 = 34 base + 7 cross-asset (same slice V1.1 uses),
so the cohort is directly comparable to V1.1. Each must clear **Gate A** (`wm_value_probe.py`) before any agent
is wired onto it.

V23 (xLSTM benchwarmer) is now trainable too (trainer fixed + anchor grafted) but is lower-priority; promote it
into the cohort if a recorded ShIC>0 lands. The C4 promotion gate (`wm_promotion_gate.py`) selects keepers on
held-out compound; V10 then ensembles the diverse winners.

## One-line summary of order
1. `baseline` → 2. `vsn_fr` → **decision gate (compound)** → (optional ablations) → Phase 2 `V12 HEADLINE` → **Phase 3 cohort (V3/V4/V6/V8/V13) if vsn_fr won** → wire the winner(s) into strat / V10 ensemble.
