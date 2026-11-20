# V15 — PatchTST encoder (drop-in stub)

**Status**: stub only. No training pipeline yet.

**Source**: Nie et al. ICLR 2023, "A Time Series is Worth 64 Words" (arxiv 2211.14730).

**Files**:
- `patchtst_encoder.py` — channel-independent patch transformer encoder (605K params)

**Smoke test**: `python src/wm/v15/patchtst_encoder.py` — forward pass on dummy data, params count, num_patches.

**Drop-in usage** (in any V1-V14 trainer):
```python
from v15.patchtst_encoder import PatchTSTEncoder
encoder = PatchTSTEncoder(n_features=121, seq_len=96, patch_len=16, stride=8,
                           d_model=128, n_heads=4, n_layers=3)
# encoder(x) -> [B, C, d_model] (channel-independent, mean-pooled patches)
```

**Why a stub, not a full version**: V15 is intentionally not a parallel V-version with its own training pipeline. The encoder is meant to be ablated INSIDE V1.x or V19 by replacing the existing transformer encoder. Phase M2 of the model layer plan deliberately deferred building a V15 training pipeline because Chronos foundation-model finetune (V18) tests the same broad question (modern transformer for time series) at higher param scale.

**Source of truth**: docs/MODEL_LAYER_OPTION_B_2026_04_26.md
