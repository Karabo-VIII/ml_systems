# V12 — Cross-Asset Hierarchical WM (BLOCKED on MultiAssetDataset)

> **Role in cohort**: cross-asset hierarchical attention. Tests whether
> joint multi-asset features (BTC + ETH + … at synchronized timestamps)
> beat single-asset modeling.
>
> **Status: STRUCTURALLY BLOCKED**. V12's distinctive lever requires
> `MultiAssetDataset` (scaffolded but not built). Single-asset fallback
> path works but defeats V12's purpose.

## Purpose

V12 is the cohort's bet that **macro signals come from joint multi-asset
dynamics**, not individual asset histories. Hypothesis: "BTC broke out AND
ETH funding negative AND SOL VPIN spiking" is a stronger signal than any
single asset's features.

V12 architecture supports two paths:

| Path | Status | What it does |
|---|---|---|
| **forward_single_asset** | active (fallback) | Processes one asset at a time; effectively a small WaveNet+VIB model |
| **forward_multi_asset** | **DEAD CODE** | Hierarchical attention: cross-asset at bar-level + temporal at seq-level |

The dead-code path was flagged in iron-clad audit 2026-05-07. V12 is blocked
on the `MultiAssetDataset` build (1.5 weeks per scaffold in
`src/wm/_shared/multi_asset_dataset.py`).

## Architecture (when multi-asset path is live)

```
Per-asset obs (B, T, F)  ×  N=10 assets at SAME timestamp
                                  │
  ┌─ Per-asset encoder (shared) ──┐
  │ WaveNet TCN → encoder hidden  │ × N
  └───────────────────────────────┘
                  │
  ┌─ Cross-asset attention (HIERARCHICAL — bar-level) ────┐
  │ Each bar: attend across N assets                       │
  │ Output: [B, N, T, d_model] with cross-asset mixing     │
  └────────────────────────────────────────────────────────┘
                  │
  ┌─ Temporal attention (per-asset) ────┐
  │ Self-attention along T axis × N      │
  └──────────────────────────────────────┘
                  │
                  └── Per-asset heads (TwoHot, regime, CC-H5/H6) — same as cohort
```

### Design rationale

- **Why V12 matters**: of all 25 versions in `src/wm/v*/`, V12 is the ONLY
  one whose primary architecture exploits cross-asset structure. CC-H3
  cross-asset attention in `_shared/headline_components` is V12's natural
  home but currently a no-op on single-asset training.
- **Why d_model=128 (smaller per-asset)**: 10 assets in parallel = 10x
  memory cost vs single-asset. 128 keeps total params manageable.
- **Why BLOCKED on MultiAssetDataset**: V12 needs `[B, N_assets, T, F]`
  batches with synchronized timestamps. `AntifragileDataset` provides
  `[B, T, F]` only. Scaffold at `src/wm/_shared/multi_asset_dataset.py`
  documents the 1.5-week full-build plan.

## Files

```
src/wm/v12/v12_training/
├── settings.py
├── world_model.py           # CrossAssetWorldModel (single + multi paths)
└── train_world_model.py
```

## Usage

### Single-asset fallback (works today)

```bash
# Currently the ONLY working path. Treats V12 as a small single-asset model.
python src/wm/v12/v12_training/train_world_model.py --features 29
```

### Multi-asset Headline path (NOT YET RUNNABLE)

```bash
# Requires MultiAssetDataset full build first.
V12_HEADLINE_MODE=1 python src/wm/v12/v12_training/train_world_model.py --features 29
# Will currently fail with "DATALOADER MUST PROVIDE SYNCHRONIZED MULTI-ASSET BATCHES"
```

## Key settings (SOTA-2026)

| Setting | Value | Notes |
|---|---|---|
| `WM_D_MODEL` | 128 | small per-asset (10 in parallel) |
| `VIB_KL_WEIGHT` | 0.15 | raised 2026-04-22 (ShIC=0 fix) |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 (was 0.7) |
| `HEADLINE_MODE` | OFF | (no point ON without dataloader) |
| `HEADLINE_MULTI_ASSET_PATH` | True | flag for when dataloader lands |
| `HEADLINE_HIERARCHICAL_ATTN` | True | flag |
| `HEADLINE_VIB_KL` | 0.10 | headline KL |
| `USE_QUANTILE_HEADS` | True | CC-H5 (flags ready) |
| `USE_REGIME_COND_HEADS` | True | CC-H6 (flags ready) |
| `REGIME_AWARENESS_MODE` | "film" | (flags ready; deep wiring deferred) |

## What's queued vs done

| Item | Status |
|---|---|
| Cohort-wide invariants (BIN_*, batch, target_prefix, etc.) | ✓ canonical |
| CC-H5/H6/FiLM SETTINGS flags | ✓ defaults set |
| CC-H5/H6/FiLM world_model wiring | DEFERRED until multi-asset path lands (would be inconsistent to wire on single-asset only) |
| MultiAssetDataset full build | scaffold shipped (`src/wm/_shared/multi_asset_dataset.py`); ~1.5 weeks dev |
| forward_multi_asset path verification | DEAD CODE per iron-clad audit; needs dataloader |
| V12 first SOTA training | gated on MultiAssetDataset |

## Headline-tier projection

Per `WM_HEADLINE_UPGRADE_PLAN §14`:
> "V12 has the highest cohort Headline ceiling — but harness work (multi-asset
> dataloader) is non-trivial. Defer until V1.1-Headline + V3-Headline + V4-Headline ship."

Projected V12-Headline IC: **0.090 / ShIC 0.045** at 3.5 GPU-d, but only after
the dataloader is built and validated.
