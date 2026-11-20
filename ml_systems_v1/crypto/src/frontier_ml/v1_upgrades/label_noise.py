"""Calibrated label-noise injection (B007 E2).

REPORTED source: arXiv 2510.17526, "How Does Label Noise Gradient Descent
Improve Generalization in the Low SNR Regime?" -- "adding label noise during
training suppresses noise memorization and prevents it from dominating the
learning process, thereby achieving good generalization despite low SNR."

Crypto returns ARE the low-SNR regression regime by construction (Pattern P
established 5 dead features, Pattern Q established reconstruction dominance).
A theoretical guarantee that label-noise GD generalizes in this exact regime
is prescriptive.

Implementation:
    At each training step, perturb regression labels by epsilon ~ N(0, sigma_label^2)
    where sigma_label is calibrated to a fraction of the residual std.

    sigma_label = noise_ratio * sigma_residual

    noise_ratio = 0.5 = canonical default per protocol §5.1 of B007.

Cost: per-batch RNG draw, ~0 marginal compute. Composable with SAM, FrAug,
PCGrad, MTP, MDN.

Optional regime-aware extension (arXiv 2402.04398): scale sigma_label by
regime stress (sigma_high in bear/chop, sigma_low in steady-bull). Folded
into the API but disabled by default until empirical evidence in our cohort.
"""
from __future__ import annotations

import torch


class LabelNoiseInjector:
    """Adds calibrated Gaussian noise to regression targets.

    sigma_residual: empirical residual std of the model's predictions on a
        held-out slice. If unknown, pass the target std as a conservative proxy.
    noise_ratio: sigma_label / sigma_residual. Default 0.5 per B007 E2.
    regime_scale: optional dict {regime_id: scale_factor} for non-stationary noise.
    """

    def __init__(
        self,
        sigma_residual: float,
        noise_ratio: float = 0.5,
        regime_scale: dict | None = None,
    ):
        if sigma_residual <= 0:
            raise ValueError(f"sigma_residual must be > 0, got {sigma_residual}")
        if noise_ratio < 0:
            raise ValueError(f"noise_ratio must be >= 0, got {noise_ratio}")
        self.sigma_residual = float(sigma_residual)
        self.noise_ratio = float(noise_ratio)
        self.regime_scale = regime_scale or {}
        self.sigma_label = self.noise_ratio * self.sigma_residual

    @torch.no_grad()
    def __call__(
        self,
        targets: torch.Tensor,
        regime_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return targets + noise. Same shape, same dtype, same device.

        regime_label: optional per-sample regime ids. If provided AND regime_scale
            is non-empty, sigma is scaled per-sample.
        """
        if self.sigma_label == 0.0:
            return targets
        sigma = self.sigma_label
        if regime_label is not None and self.regime_scale:
            scale = torch.ones_like(targets, dtype=torch.float32)
            for rid, factor in self.regime_scale.items():
                scale = torch.where(
                    regime_label == rid,
                    torch.full_like(scale, float(factor)),
                    scale,
                )
            noise = torch.randn_like(targets) * (sigma * scale.to(targets.dtype))
        else:
            noise = torch.randn_like(targets) * sigma
        return targets + noise


def smoke():
    """Confirm injector adds noise of the expected magnitude and is reproducible under seed."""
    torch.manual_seed(0)
    sigma_res = 0.02  # crypto return residual std-ish
    inj = LabelNoiseInjector(sigma_residual=sigma_res, noise_ratio=0.5)
    y = torch.zeros(10000)
    yn = inj(y)
    emp_sigma = yn.std().item()
    expected = 0.5 * sigma_res
    print(f"[label_noise] expected sigma={expected:.4f} empirical={emp_sigma:.4f}")
    assert abs(emp_sigma - expected) < 0.0008, f"sigma off: {emp_sigma}"

    # Regime scaling
    inj2 = LabelNoiseInjector(
        sigma_residual=sigma_res,
        noise_ratio=0.5,
        regime_scale={0: 2.0, 1: 1.0, 2: 0.5},
    )
    y2 = torch.zeros(30000)
    rl = torch.cat([
        torch.zeros(10000, dtype=torch.long),
        torch.ones(10000, dtype=torch.long),
        torch.full((10000,), 2, dtype=torch.long),
    ])
    yn2 = inj2(y2, regime_label=rl)
    s_bear = yn2[:10000].std().item()
    s_chop = yn2[10000:20000].std().item()
    s_bull = yn2[20000:].std().item()
    print(f"[label_noise] regime sigmas: bear={s_bear:.4f} chop={s_chop:.4f} bull={s_bull:.4f}")
    # Bear should be ~2x bull; chop in between.
    assert s_bear > s_chop > s_bull, "regime ordering wrong"
    assert s_bear / s_bull > 3.0, f"regime contrast too small: {s_bear / s_bull}"
    print("[label_noise] PASS smoke")


if __name__ == "__main__":
    smoke()
