# Model Layer (V4 Crypto System)

> **Single source of truth for the WM (world-model) subtree** -- where each
> version lives, how `run_all_training` orchestrates training, what guards
> run before any model touches data, and how to add a new version (V20+).
>
> Last touched: 2026-04-29 (Phase A harmonization: `src/v*` -> `src/wm/v*`).

## 1. Layout

```
src/wm/
├── __init__.py                 # marks the namespace
├── v0/                         # legacy baseline (no longer trained)
├── v1/
│   ├── v1_0_training/          # V1.0 Transformer+RSSM reference
│   ├── v1_1_training/          # V1.1 XD anti-memorization
│   ├── v1_4_training/          # V1.4 FeatureAttentionBlock
│   ├── v1_6_training/          # V1.6 KL+Gumbel+ATME+Dream
│   └── archive/                # V1.2/1.3/1.5/1.7 (frozen historical)
├── v3/  v4/  v6/  v8/          # ACTIVE  (RSSM/JEPA family)
├── v9/                         # ACTIVE-but-MARKED-ARCHIVED in run_all
├── v10/                        # meta-ensemble aggregator
├── v11/ v12/ v13/ v14/         # FROZEN/deprecated; settings drift suspected
└── v15/ ... v19/               # SOTA stubs (PatchTST/DreamerV3/TD-MPC2/Chronos)

backups/BKP_20260429_MODEL_HARMONIZATION/
├── MANIFEST.md
├── v2/                         # ARCHIVED -- not imported, not shipped
├── v5/
└── v7/
```

Inactive/archived versions live OUTSIDE `src/` so they cannot be picked
up by the model registry, run_all_training, or CDAP.

## 2. Pre-train guards (the entry point is `src/run_all_training.py`)

Order of operations for `python src/run_all_training.py --features 13`:

1. **`--auto-refresh`** *(opt-in)*
   Calls `src/pipeline/refresh.py --target chimera_v51 --scope u50`. The
   refresh runner walks the asset DAG (`config/asset_dag.yaml`),
   content-hashes every stage, and rebuilds only what's stale. This is the
   single command that brings raw_aggtrades -> chimera_v51 up to date.

2. **Pre-train gate** *(default ON; `--skip-gate` to bypass)*
   Calls `src/pipeline/pre_train_gate.py --asset BTC`. Composes 5 validators:
   `data_health` + `chimera_v51 schema` + `xd_consistency` + `e2e` +
   `split`. Exit 2 = hard fail (training aborts); exit 1 = warnings (proceed
   with caution); exit 0 = clean.

3. **Preflight** *(default ON; `--skip-preflight` to bypass)*
   Per training script: `py_compile` + `settings.get_feature_config(N)`
   resolves + chimera v50 files exist on disk.

4. **Execute** -- run each script with `--features N` (and any extras
   declared in `MODELS`).

5. **Coverage report** -- prints which active versions were
   `TRAINED` / `FAILED` / `fresh` (already complete) / `n/a` (don't
   support this f-count) / `archived`.

## 3. The MODELS registry

`src/run_all_training.py` declares 17 active versions in the `MODELS` list.
Each entry is `(model_id, label, script_path, extra_args, supported_features)`.
The runner filters by `supported_features` so `--features 13` only enumerates
versions that ship a 13-feature config in their `settings.py`.

A version is **archived** by adding its `model_id` to the `ARCHIVED_MODELS`
set at the top of run_all. Currently `{"v9"}` (memorized across retrains;
ShIC=0.007). To force-include archived versions: `--include-archived`.

## 4. Adding a new version (V20+)

1. Create the directory under `src/wm/v20/` with the standard subtree
   (`v20_training/{settings.py, train_world_model.py, world_model.py, __init__.py}`).
2. Mirror cross-version invariants from `config/_invariants.yaml`:
   `DIRECT_RETURN_WEIGHT=3.0`, `BIN_MIN/MAX=±1`, `NUM_BINS=255`,
   `WM_BATCH_SIZE=32`, `TWOHOT_FOCAL_GAMMA=0.0`, `WM_STEPS_PER_EPOCH=2000`,
   `ACTIVE_HORIZONS=[1, 4, 16, 64]`, `target_prefix="target_return"`.
3. Add an entry to `MODELS` in `src/run_all_training.py`:
   ```
   ("v20", "V20 (description)",
    "src/wm/v20/v20_training/train_world_model.py", [], [13, 25, 34]),
   ```
4. Run `python src/run_all_training.py --features 13 --dry-run --skip-gate`
   to confirm preflight passes.
5. CDAP audit must stay clean: `python src/audit/check_invariants.py`.

## 5. Migration history

* **2026-04-29 Phase A** -- `src/v*` -> `src/wm/v*`. v2/v5/v7 archived to
  `backups/BKP_20260429_MODEL_HARMONIZATION/`. ~870 files touched in one
  atomic commit (b859aad). 1113 .py files compile clean. Settings,
  imports, and CDAP globs updated. `run_all_training.preflight` and
  `_log_dir` updated to handle the new path layout.
* **2026-04-29 Phase C** -- pre_train_gate wired into run_all_training
  (was previously missing -- training would have run on broken data
  without warning). New flags: `--skip-gate`, `--gate-asset`,
  `--auto-refresh`. End-of-run coverage report added.
* Phase B (settings drift audit), D (data_api wiring inside training
  scripts), and remainder of E (richer CLAUDE.md) are deferred to a
  later session.

## 6. Where to look when something breaks

| Symptom | First file to read |
|---|---|
| `--auto-refresh` failed | `src/pipeline/refresh.py --status --target chimera_v51` and `data/_dag_state.json` |
| Pre-train gate exit 2 | `logs/pre_train_gate.log` |
| Preflight `feature config` error | `src/wm/<vN>/<vN>_training/settings.py::get_feature_config` |
| Preflight `no v50 chimera files` | rebuild via `src/pipeline/refresh.py --target chimera_legacy --scope u50` |
| Coverage shows `MISSING` for an active version | bug in run loop -- look for un-handled exception in `run_one()` |
| CDAP CRITICAL | `python src/audit/check_invariants.py`; rule names map directly to `config/_invariants.yaml` keys |
