"""V21 — Mamba + Latent NODE hybrid (per B005 §3, MODE arXiv 2601.00920).

The 2026 frontier successor to pure Neural ODE (V8) which is dominated by
Mamba on financial time-series. Mamba+NODE hybrid combines:
- Mamba SSD backbone for sub-quadratic temporal modeling
- Latent NODE residual for continuous-time correction (irregular dollar bars)

Per B005: "MODE (Low-Rank Neural ODE + Mamba) [REPORTED arxiv 2601.00920]
combines NODE with Mamba; the COMBINED model is the 2026 frontier, not pure NODE."

This is the V8 successor decision per UPGRADE_PLAN_PER_VERSION:
- V8 archived only if SSL-pretrain probe fails
- V21 is the architecturally-superior continuous-time WM
"""
