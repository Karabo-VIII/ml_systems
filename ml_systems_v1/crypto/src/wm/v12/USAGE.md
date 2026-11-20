# V12 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏳ **HIGH PRIORITY second-wave**

V12 (Cross-Asset Attention) is **architecturally aligned** with the
foundation prong's cross-asset attention design — the upgrade flags
should transfer cleanly and lift IC.

| Upgrade | Applicability | Wiring effort |
|---|---|---|
| `--sam` | ✅ applicable | port from V1.1 (1-2 hrs) |
| `--fraug` | ✅ applicable (input-side) | trivial (1 hr) |
| `--pcgrad` | ✅ applicable | port get_loss `return_components` (2-3 hrs) |
| `--mtp` | ✅ applicable | swap return_heads for MTPHead (1-2 hrs) |
| `--mdn` | ✅ applicable | head replacement (2-3 hrs) |
| `--adaptive-bins` | ✅ applicable | trivial bucketer drop-in (1 hr) |

**Why V12 is high-priority second-wave**:
- V12 already does cross-asset attention; the foundation prong scaled
  this to u100. A SAM+PCGrad+MTP V12 retrain is the closest test of
  whether those upgrades transfer at the smaller V12 size (841K params).
- V12's structure (single shared backbone + per-asset attention)
  benefits maximally from PCGrad if cross-asset gradients conflict.
- Per WM_FINDINGS, V12 is currently dead-code in the standard runner
  — that's a separate fix unrelated to upgrade wiring.

## Architecture: Cross-Asset Attention (10 assets jointly, per-timestep)

Smaller param count than V1.x family (841K vs 2M) but architecturally
distinct via cross-asset attention layer.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.
