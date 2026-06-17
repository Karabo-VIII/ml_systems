# V24 — TimesNet (Wu et al., ICLR 2023)

> **Role in cohort**: FFT-based period discovery → 2D conv on
> period-folded data. Tests whether explicit periodicity discovery beats
> attention on dollar bars.
>
> **Status**: SOTA-2026 wired (CC-H5/H6/FiLM + cohort flags). Not yet
> trained on new defaults.

## Purpose

The TimesNet paper's core insight: time series have multiple
periodicities (daily/weekly/seasonal), and you can DISCOVER them via
FFT + fold the 1D sequence into a 2D tensor where rows are periods and
columns are within-period positions, then apply 2D convolutions to
capture intra-period (column) AND inter-period (row) patterns.

For crypto:
- 8h funding cycle (3 bars/day at dollar-bar pace)
- 24h UTC cycle (8-12 bars/day)
- 7d weekly cycle

V24 discovers these via FFT instead of hardcoding (V22/V25's
`CryptoPeriodEmbedding` is the hardcoded alternative).

## Architecture (SOTA-2026)

```
Obs (B, T, F=29) + asset_emb
  └── obs_encoder → Linear → d_model
       └── N× TimesBlock:
            ├── FFT(x) → top-K periods (peak frequencies)
            ├── For each period p:
            │    ├── reshape: [B, T, D] → [B, T/p, p, D]
            │    ├── 2D conv → [B, T/p, p, D]
            │    └── reshape back: [B, T, D]
            ├── Weighted sum (gated by FFT magnitude at each period)
            └── Residual + RMSNorm
       └── post_norm → h_seq [B, T, D]
            ├── RegimeFiLM (h_seq pre-VIB; identity-at-init)
            └── VIB → feat
                 ├── ATME per-sample 0.15 → feat
                 │    ├── return_trunk + return_heads
                 │    ├── regime_head
                 │    ├── CC-H5 quantile_heads
                 │    └── CC-H6 regime_cond_heads
```

### Design rationale

- **Why FFT discovery**: doesn't require knowing the periods in advance;
  V22/V25 hardcode 8h/24h/7d which may not match the actual irregular
  dollar-bar period distribution
- **Why 2D conv on folded data**: captures both within-period and
  cross-period patterns natively
- **Why TimesBlock stack (not single)**: each block can specialize for
  different period scales; deeper stack = more period granularity

## Files

```
src/wm/v24/v24_training/
├── settings.py
├── world_model.py           # TimesNetWorldModel + TimesBlock
└── train_world_model.py
```

## Usage

```bash
# Train (SOTA-2026 defaults)
python src/wm/v24/v24_training/train_world_model.py --features 29

# Validate
python src/wm/v24/v24_training/validate_world.py
```

## Key settings (SOTA-2026)

| Setting | Value | Notes |
|---|---|---|
| Cohort invariants | canonical | |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 |
| `TEMPORAL_CTX_DROP` | 0.15 | per-sample ATME |
| `betas` | (0.9, 0.95) | fixed 2026-05-16 |
| `USE_QUANTILE_HEADS` | True | CC-H5 (NEW today) |
| `USE_REGIME_COND_HEADS` | True | CC-H6 (NEW today) |
| `REGIME_AWARENESS_MODE` | "film" | RegimeFiLM h_seq pre-VIB (NEW today) |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | Per-version Headline plan section | NOT in WM_HEADLINE_UPGRADE_PLAN (D10 fail) |
| 2 | CC-H3 cross-asset | hook deferred — needs MultiAssetDataset |
| 3 | First SOTA training | GPU-d pending |
| 4 | Period-discovery validity check | should check what periods FFT discovers; if all-period-1 (no periodicity), V24's mechanism evaporates |

## Risks

TimesNet's FFT period discovery assumes meaningful spectral peaks in the
sequence. For dollar bars (irregular wall-clock spacing), this assumption
may not hold — FFT could discover spurious periods that are artifacts of
the irregular sampling, not real market cycles.

Pre-flight check: at first training, log the top-K periods discovered per
batch. If they cluster at small integers (2, 3, 4) without alignment to
known cycles (3-4 bars/8h, 8-12 bars/24h), the discovery is noisy.
