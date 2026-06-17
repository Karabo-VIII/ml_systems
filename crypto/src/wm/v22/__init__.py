"""V22 — iTransformer (Liu et al., ICLR 2024). Inverted feature-as-token transformer.

Closes the cross-asset structural gap left by V12: iTransformer attends across
FEATURES (not time), so cross-asset modeling becomes a feature-attention
problem with no timestamp synchronization required.

Status: BACKBONE SCAFFOLD. Forward + backward + smoke verified.
"""
