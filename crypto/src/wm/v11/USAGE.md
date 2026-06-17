# V11 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏸ **STAY FROZEN (per B005)**

V11 (WaveNet + MoE + Discriminator) STAYS FROZEN per B005 verdict. The
sparse fine-grained MoE pattern (256 experts) is a 2025 trend but only
validated at ≥ 50M-param scale [REPORTED]; V11 at 2.9M is below the
threshold where sparse MoE is meaningfully sparse — adding more experts
at this scale shrinks per-expert capacity below dense-MLP equivalents.

## Decision rule to UNFREEZE

Revisit V11 ONLY if:
- Foundation prong scales to ≥ 50M params with MoE
- A published 2026-2027 paper measures sparse MoE at 2-5M params
  beating dense baselines on financial data with IC > 0.05

Until then, V11 stays as-is. V1.x upgrade flags (SAM/FrAug/MDN/etc.)
could in principle be wired to V11 (its WaveNet + multi-horizon return
heads structure is amenable), but the EV is low given the sparse-MoE
threshold issue.

## Architecture: WaveNet + MoE + Discriminator (combined V3+V6+V9)

Best-of-three combined architecture from V3/V6/V9 lineage. Frozen
checkpoints at `models/wm/v11/`.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.
