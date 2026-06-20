"""Quick binomial significance test for smoothing lane results.
Is 57.6% (N=500) reliably above 55%?
"""
from scipy import stats
import numpy as np

N = 500
# Best ungated books
for label, wins in [
    ("EW_BH / VOL_TARGET / RISK_PARITY", 288),   # 57.6% of 500
    ("IVP", 287),                                   # 57.4%
    ("MIN_CORR", 284),                              # 56.8%
    ("BARBELL", 268),                               # 53.6%
]:
    # One-sided binomial: H0: p <= 0.55, H1: p > 0.55
    pval = stats.binomtest(wins, N, p=0.55, alternative='greater').pvalue
    wr = wins / N
    print(f"{label}: WR={wr*100:.1f}%  k={wins}  p-value (H0:p<=55%)={pval:.4f}  {'SIGNIFICANT' if pval < 0.05 else 'NOT SIGNIFICANT'}")

# Also test vs the stated standing result 55%
print()
# The standing baseline EW-BH is itself 57.6% -- so the standing "55%" was a DIFFERENT measurement
# (likely from earlier hand-coded engines). Let's check what the task said:
# "EW buy-hold wins ~55% of random 7d slices at +2.9% mean"
# Our EW-BH gets 57.6%/+2.35% over 2020-2026.
# This is CONSISTENT (the prior 55% was likely measured over a shorter period or different seed).
print("Note: Our EW-BH = 57.6%/+2.35% over 2020-2026 (N=500, seed=42)")
print("The standing '55%' was stated for a possibly shorter/different evaluation period.")
print("Vol-target and risk-parity tie EW-BH exactly -- no smoothing premium at all.")
print("Regime-gating HURTS win-rate by ~10pp (confirmed again: 47% vs 57%).")
