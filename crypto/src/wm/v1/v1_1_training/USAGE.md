# V1.1 — Transformer-RSSM + XD-Split (SHIP-RECORD ★)

> **Role in cohort**: the cohort's RECORD HOLDER. Last verified
> **IC 0.073 / ShIC 0.032 (Trader tier)** at f29.
> Primary Headline target per `WM_COHORT_RETRAIN_SCHEDULE_2026_05_16.md`.

## Purpose

V1.1 = V1.0 + **XD-split anti-memorization**. The XD ("cross-asset") features
in chimera carry the strongest signal but also the highest temporal-fingerprint
risk — they're derived from BTC + cohort mean, so a single timestamp's XD
slice can uniquely identify that timestamp. V1.1 fights this with aggressive
per-timestep dropout + heavy noise on XD-channel inputs.

This is the SHIP architecture for the production v3 paper-trade-replay deploy
gate.

## Architecture

Identical to V1.0 EXCEPT:

1. **Posterior sees BASE features only**, not XD:
   ```python
   base_obs = obs_seq[:, :, :base_dim]   # NOT obs_seq
   post_input = torch.cat([h_seq, base_obs], dim=-1)
   ```
   Blocks the temporal-fingerprint shortcut where the posterior could memorize
   "which timestamp is this" via the XD slice.

2. **XD-channel anti-memo augmentation during training**:
   ```python
   xd_mask = (rand(B, T, xd_count) > 0.7).float()   # 70% dropout
   obs_seq[:,:,base_dim:] *= xd_mask
   obs_seq[:,:,base_dim:] += randn(...) * 0.3        # heavy noise
   ```

3. **Reconstruction target = base features only** (not XD) — XD remains "challenge" features the encoder must learn to use without memorizing.

### Why this works (design rationale)

- **The XD problem**: chimera's XD features are `(btc_ret, btc_vol, cohort_mean_ret, ...)` derived. Two assets at the same timestamp have IDENTICAL XD values. A model that learns "see XD slice → look up timestamp" gets cheap IC on train but ShIC = 0.
- **The XD solution**: blind the posterior to XD entirely (anti-memo through information bottleneck) + heavy dropout/noise on the encoder input (anti-memo through input corruption). The model is FORCED to learn from base features + corrupted XD.
- **Empirically validated**: ShIC=0.0305 on initial cohort training (the historical reference point). Best IC=0.0731 at f29 with full v1.1 stack.

## Files (slim — 16-17 .py)

```
src/wm/v1/v1_1_training/
├── settings.py
├── components.py            # shared bricks
├── world_model.py           # TransformerWorldModel (+ XD-split + ablation heads)
├── train_world_model.py     # main trainer (1254 lines — most-developed V1.x)
├── train_adapter.py         # .X (FiLM adapter) trainer
├── train_snapshot.py        # .E (snapshot ensemble) trainer
├── train_ncl.py             # .D (NCL diversity) trainer
├── train_diversity.py       # extra: multi-head diversity training
├── train_ensemble_gating.py # extra: V10 meta integration
├── adapter.py / snapshot_ensemble.py / ncl_model.py / diversity_model.py
├── validate_world.py / validate_adapter.py / validate_snapshot.py / validate_ncl.py
```

## Usage

### Train base

```bash
# DEFAULT recommended — f29 (Pattern P, no dead features)
python src/wm/v1/v1_1_training/train_world_model.py --features 29

# Headline mode (CC-H4 anti-mem ↑ + Pattern P+Q + projected ShIC ≥ 0.045)
V1_HEADLINE_MODE=1 python src/wm/v1/v1_1_training/train_world_model.py --features 29
```

### Train .X (FiLM adapter)

```bash
python src/wm/v1/v1_1_training/train_adapter.py --features 29
```

### Train .E (snapshot ensemble — cyclical LR snapshots)

```bash
python src/wm/v1/v1_1_training/train_snapshot.py --features 29
```

### Train .D (NCL diversity — multi-head)

```bash
python src/wm/v1/v1_1_training/train_ncl.py --features 29
```

### Validate

```bash
python src/wm/v1/v1_1_training/validate_world.py        # base
python src/wm/v1/v1_1_training/validate_adapter.py      # .X
python src/wm/v1/v1_1_training/validate_snapshot.py     # .E
python src/wm/v1/v1_1_training/validate_ncl.py          # .D
```

## Key settings

| Setting | Value | Notes |
|---|---|---|
| All canonical CLAUDE.md invariants | shared with V1.0 | |
| `XD_DROPOUT_RATE` | 0.7 | Per-timestep on XD features |
| `XD_NOISE_STD` | 0.3 | Heavy noise on dropped positions |
| `HEADLINE_MODE` | env var `V1_HEADLINE_MODE` | When 1: XD_DROPOUT=0.85, XD_NOISE=0.4, WM_FREE_NATS=1.5 |

## Last known metrics

- **IC = 0.073 / ShIC = 0.032** at f29 — cohort RECORD
- Ratio ShIC/IC = 0.44 (above 0.30 gate)
- Per `WM_COHORT_RETRAIN_SCHEDULE`: V1.1-Headline projected
  **IC ≥ 0.075 / ShIC ≥ 0.042** at 2.5 GPU-d cost (CC-H4 + CC-H6 + P+Q)

## Frontier-ML hooks (opt-in via `apply_v1_upgrades` helper)

V1.1's `world_model.py` defines hook points for experimental upgrades:
- **MTP head** (`_use_mtp=True`, `mtp_head` populated by `apply_v1_upgrades`)
- **MDN heads** (`_use_mdn=True`, swaps TwoHot for MDN log-prob)
- SAM / FrAug / PCGrad / label-noise / logit-clip — all monkey-patched
- Default: ALL OFF; behavior identical to documented baseline

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H3 cross-asset head | BLOCKED on MultiAssetDataset (1.5 weeks) |
| 2 | CC-H1 multi-resolution stack | NOT WIRED |
| 3 | CC-H6 regime-conditional heads | NOT WIRED (V3+V4 have it; queued for V1.x) |
| 4 | CC-H7 dream-rollout in loss | NOT WIRED (V1.6 is the lab for this) |
| 5 | CC-H4 anti-mem ↑ via HEADLINE_MODE | SHIPPED settings; needs retrain to validate |
