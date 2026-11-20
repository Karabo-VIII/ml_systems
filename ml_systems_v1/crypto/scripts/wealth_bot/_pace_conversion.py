"""Pace conversion helper: convert UNSEEN compound + days into per-period pace estimates
with gap-to-target tracking against PROJECT_NORTH_STAR.md target bands.

Mandated by docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md §EP2 (binding):
every wealth-bot audit report must include this conversion table.

Usage:
  from scripts.wealth_bot._pace_conversion import pace_table
  print(pace_table(compound_pct=51.86, days=141, name="1-strat deployed"))

Or CLI:
  python scripts/wealth_bot/_pace_conversion.py --compound 51.86 --days 141 --name "1-strat deployed"
"""
from __future__ import annotations

__contract__ = {
    "kind": "pace_conversion",
    "owner": "wealth_bot/pace_conversion",
    "purpose": "convert compound + days -> pace estimates with gap-to-target tracking",
}

import argparse
import sys

# Target bands from PROJECT_NORTH_STAR.md (user-mandated ROI targets)
# CRITICAL: these are LOWER-BOUND aim levels, NOT ceilings or limiters.
# Hitting the lower bound = minimum acceptable. Exceeding upper bound = keep pushing higher.
# Updated 2026-05-25: 3d band 1-5% -> 2-5% per user mandate
TARGETS = {
    "%/d":    (1.0, 5.0),   # aim AT LEAST 1%/d, push higher than 5%
    "%/3d":   (2.0, 5.0),   # aim AT LEAST 2%/3d, push higher than 5%  -- updated 2026-05-25
    "%/week": (3.0, 5.0),   # aim AT LEAST 3%/week, push higher than 5%
}


def compound_to_pace(compound_pct: float, days: float) -> dict:
    """Convert UNSEEN compound (%) over N days into per-period pace estimates.

    Pace is computed as the constant per-period compound rate that would equal
    the full-window compound. r_period = (1 + compound)^(period_days/days) - 1.
    """
    if days <= 0:
        return {"%/d": 0.0, "%/3d": 0.0, "%/week": 0.0, "%/month": 0.0}
    total_mult = 1.0 + compound_pct / 100.0
    if total_mult <= 0:
        return {"%/d": -100.0, "%/3d": -100.0, "%/week": -100.0, "%/month": -100.0}
    return {
        "%/d":     (total_mult ** (1.0    / days) - 1.0) * 100.0,
        "%/3d":    (total_mult ** (3.0    / days) - 1.0) * 100.0,
        "%/week":  (total_mult ** (7.0    / days) - 1.0) * 100.0,
        "%/month": (total_mult ** (30.0   / days) - 1.0) * 100.0,
    }


def gap_to_band(pace: float, target_low: float, target_high: float) -> tuple[str, float, float]:
    """Return (status, gap_to_low, gap_to_high) given a pace.

    Bands are LOWER-BOUND aim levels, NOT ceilings. Status reflects that:
    - Below lower bound = need to push up to floor (MISSED FLOOR)
    - At/above lower bound = met minimum, KEEP PUSHING higher
    - Above upper bound = good, but no ceiling — keep pushing
    """
    gap_low = target_low - pace
    gap_high = target_high - pace
    if pace >= target_high:
        status = "FLOOR + STRETCH MET (push higher)"
    elif pace >= target_low:
        status = "FLOOR MET (push to stretch)"
    elif gap_low <= 0.5:
        status = "BORDERLINE FLOOR"
    else:
        status = "MISSED FLOOR"
    return status, gap_low, gap_high


def pace_table(compound_pct: float, days: float, name: str = "bot") -> str:
    """Format the pace + gap-to-target table as ASCII text for audit reports."""
    pace = compound_to_pace(compound_pct, days)
    lines = []
    lines.append(f"## Pace conversion: {name}")
    lines.append(f"  UNSEEN compound: {compound_pct:+.2f}% over {days:.0f} days")
    lines.append("")
    lines.append(f"  Bands are LOWER-BOUND aim levels (minimum acceptable). Stretch beyond upper bound.")
    lines.append("")
    lines.append(f"  {'Period':<8}  {'Pace':>9}  {'Floor band':<14}  {'Status':<32}  {'Gap_low':>8}  {'Gap_high':>8}")
    lines.append(f"  {'-'*8}  {'-'*9}  {'-'*14}  {'-'*32}  {'-'*8}  {'-'*8}")
    for period_key in ["%/d", "%/3d", "%/week"]:
        p = pace[period_key]
        lo, hi = TARGETS[period_key]
        status, g_lo, g_hi = gap_to_band(p, lo, hi)
        band_str = f"{lo:.1f}-{hi:.1f}{period_key.replace('%','')}"
        lines.append(f"  {period_key:<8}  {p:>+8.2f}%  {band_str:<14}  {status:<32}  "
                     f"{g_lo:>+7.2f}  {g_hi:>+7.2f}")
    # Also show monthly for reporting context (no target band)
    lines.append(f"  {'%/month':<8}  {pace['%/month']:>+8.2f}%  {'(reference)':<14}  {'-':<32}  {'-':>8}  {'-':>8}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compound", type=float, required=True, help="UNSEEN compound percent")
    ap.add_argument("--days", type=float, required=True, help="UNSEEN window length in days")
    ap.add_argument("--name", default="bot", help="bot name for the table header")
    args = ap.parse_args()
    print(pace_table(args.compound, args.days, args.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
