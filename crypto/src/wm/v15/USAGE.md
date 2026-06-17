# V15 — PatchTST Encoder Library (NOT a trained model)

> **Role in cohort**: LIBRARY ONLY. Drop-in PatchTST encoder + backbone
> available for any other version to import. No trainer; not a standalone
> model. Same status as V16 (DreamerV3 stub) and V17 (TD-MPC2 stub).

## Purpose

V15 ships the **PatchTST** (Nie et al., ICLR 2023) encoder as a reusable
component. PatchTST is the canonical "patches as tokens" time-series
transformer that beat the SOTA in the original paper.

It is provided as a library (no settings.py, no trainer) for two reasons:
1. **Architecture exploration**: V22 (iTransformer) and V25 (Frontier) both
   reference PatchTST mechanisms; V15 makes the encoder available as a
   single import point.
2. **Future trainer**: when a dedicated PatchTST world-model trainer is
   built, it'll use V15's encoder rather than re-implementing.

## Files

```
src/wm/v15/
├── __init__.py
├── README.md
├── patchtst_encoder.py      # PatchTSTEncoder (the main module)
└── patchtst_backbone.py     # full PatchTST backbone (encoder + heads)
```

## How to use V15

### As a building block (current path)

```python
import sys
sys.path.insert(0, "src/wm/v15")
from patchtst_encoder import PatchTSTEncoder

encoder = PatchTSTEncoder(
    input_dim=29,
    d_model=256,
    n_heads=8,
    n_layers=3,
    patch_len=12,        # bars per patch
    seq_len=96,          # total bars
    dropout=0.1,
)
out = encoder(obs_seq)   # [B, T_patches, d_model]
```

V22 settings.py references `USE_PATCH_EMBEDDING = True` + `PATCH_LEN = 12`
to enable PatchTST-style embedding. The actual patch encoder logic is
copied/reimplemented in V22 (legacy pattern); future work could refactor
V22 to import from V15 directly.

### As a trained world model (not yet possible)

NOT yet available. Would require:
- `v15_training/settings.py` with cohort canonical invariants
- `v15_training/world_model.py` wrapping `PatchTSTEncoder` + return heads
- `v15_training/train_world_model.py` with the V1.x training loop pattern

Estimated: 2-3 days of work. Tracked as "V15 trainer" in
`docs/WM_VERSION_INVENTORY_2026_04_29.md`.

## What V15 is NOT

| Misconception | Reality |
|---|---|
| "V15 is a stub I should archive" | NO — patchtst_encoder.py + patchtst_backbone.py are non-trivial modules used by other versions |
| "V15 needs CC-H5/H6/FiLM" | NO — not a trained model; no heads to add to |
| "V15 needs assert_canonical" | NO — no settings.py; no constants to validate |
| "V15 needs USAGE/architecture in cohort docs" | This doc IS the role description |

## Related (sibling stub-libraries)

| Version | What | Status |
|---|---|---|
| V15 | PatchTST encoder + backbone | this doc |
| V16 | DreamerV3 backbone | similar (in `src/wm/v16/dreamerv3_backbone.py`) |
| V17 | TD-MPC2 backbone | similar (in `src/wm/v17/tdmpc2_backbone.py`) |
| V21 | Cross-asset MASTER scaffold | similar (in `src/wm/v21/v21_mamba_node.py`) |

All four follow the same "library-only, no trainer yet" pattern. They are
NOT empty stubs; they hold useful architecture code that could be wrapped
into a trainer if needed.

## Cross-references

- `docs/WM_VERSION_INVENTORY_2026_04_29.md` — full version-tier inventory
- `docs/WM_VERSION_VERDICTS_2026_05_16.md` — V15 verdict: KEEP-as-library
- `src/wm/v22/v22_training/settings.py` (line 116) — references PatchTST patch embedding
