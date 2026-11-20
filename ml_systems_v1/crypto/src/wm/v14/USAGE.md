# V14 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏳ **REVIVE WITH CAUTION (per B005 R3)**

V14 (Diffusion return distribution) is on the conditional-revive list
per B005 R3. The 2025-2026 evidence:
- **Diffolio** [REPORTED — arXiv 2511.07014]: diffusion model for
  multivariate probabilistic financial time-series forecasting
  outperforms baselines on Sharpe + certainty equivalents
- "Leading diffusion forecasters achieve best or second-best performance
  across benchmarks with 9-47% relative improvement over prior SOTA"
  [REPORTED]
- FTS-Diffusion (ICLR 2024) [REPORTED]: existed in 2024 but performance
  vs XGBoost-class baselines was unclear

## Revival is GATED on two checks (per B005 R3)

1. **Quantile-vector consumption probe** at strategy layer (no GPU
   needed): can the meta-learner / strategy layer ingest 5-quantile
   vectors per horizon and improve sizing-side IC by ≥ +0.005 over
   scalar-mean baseline?
2. **Diffolio number verification**: WebFetch arXiv 2511.07014 paper
   body for actual IC numbers before committing GPU-h to a V14 retrain.

If both PASS → V14 retrain (~5 GPU-h). If either FAILS → V14 stays
frozen.

## V1.x upgrade flag applicability

| Upgrade | Applicability |
|---|---|
| `--sam` | ⏳ applicable (denoiser is still a transformer) |
| `--fraug` | ⏳ applicable (input-side) |
| `--pcgrad` | ❌ n/a (V14 has a single denoising objective per noise level) |
| `--mtp` | ❌ n/a (V14 outputs distributions, not horizon-specific bin logits) |
| `--mdn` | ❌ partial (V14 IS a parametric distribution head; mutually exclusive) |
| `--adaptive-bins` | ❌ n/a |

If revived, focus on `--sam` + `--fraug` only.

## Architecture: Diffusion over return distribution

DDPM-style denoiser predicts return at h=1; full distribution accessible
via repeated denoising sampling. 2.4M params.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.
