# V13 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏸ **STAY FROZEN (per B005)**

V13 (Temporal Fusion Transformer) STAYS FROZEN per B005 verdict. The
2025-2026 frontier for graph-augmented financial forecasting is
**TFT-GNN hybrid** [REPORTED — preprint 202510.2481]: "achieved the best
overall results, with an average outperforming the standalone TFT in
11 of 12 evaluated periods" on stock market prediction.

This is a NEW architecture (graph-augmented), NOT a V13 retrofit.

## Decision rule

- V13 itself: STAY FROZEN. Do not wire V1.x upgrade flags.
- TFT-GNN hybrid: file as **V20+ candidate** for cross-asset graph
  reasoning. Will reuse V13's VSN (Variable Selection Network) machinery
  + add a graph-attention layer over assets.

## Architecture: Temporal Fusion Transformer (per-timestep VSN)

Variable Selection Network (VSN) blocks select features per timestep,
then a multi-head attention layer combines them. 2.2M params.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.
