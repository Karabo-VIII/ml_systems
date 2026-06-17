"""bootstrap_sharpe_ci.py -- B2.2: bootstrap confidence intervals on Sharpe.

PURPOSE
-------
Per validator-audit Priority 1: "Neither paper_trade_replay_v3 aggregate
nor CPCV summary computes a bootstrap CI on Sharpe. For windows of 9-30
days, se(Sharpe) ≈ sqrt((1 + Sharpe²/2)/N) gives a 95% CI of roughly ±1.0
for N=30. Any Sharpe reported from a 9-day window is statistically
indistinguishable from zero."

This module provides:
- `bootstrap_sharpe_ci(returns, n_boot=10000, alpha=0.05)` — empirical CI
- `sharpe_se_mertens(sharpe, n, skew, kurt_excess)` — analytical SE per Mertens
- `compare_sharpes(a_rets, b_rets)` — paired-difference bootstrap (is sleeve A
  significantly different from B?)
- CLI to scan all v3 JSONs and emit CI for each pillar's daily returns

USAGE
-----
    # Per pillar from v3 JSON
    python src/audit/bootstrap_sharpe_ci.py --blend REGIME_ROUTER_STRICT \\
        --json logs/strat_audit/paper_trade_replay_v3_REGIME_ROUTER_STRICT_u100_20260101_20260430.json

    # Full sweep across all v3 records
    python src/audit/bootstrap_sharpe_ci.py --sweep
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V3_LOGS = ROOT / "logs" / "strat_audit"
OUT_DIR = ROOT / "runs" / "sharpe_ci"
OUT_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "bootstrap_ci_calc",
    "owner": "audit/governance",
    "outputs": "runs/sharpe_ci/<blend>.json + summary.md",
    "invariants": [
        "bootstrap is empirical (resampling with replacement)",
        "default n_boot=10000 (chosen for 95% CI accuracy at 1% precision)",
        "Mertens analytical SE provided as cross-check",
    ],
}


SQRT_ANN = math.sqrt(365)


def annualized_sharpe(rets: np.ndarray) -> float:
    """Annualized Sharpe of a daily-returns array (crypto convention sqrt(365))."""
    s = float(rets.std())
    if s == 0 or not np.isfinite(s):
        return float("nan")
    return float(rets.mean() / s * SQRT_ANN)


def bootstrap_sharpe_ci(rets: np.ndarray, n_boot: int = 10000,
                         alpha: float = 0.05, seed: int = 42,
                         ) -> Tuple[float, float, float]:
    """Empirical bootstrap CI for annualized Sharpe.

    Returns (point_estimate, ci_low, ci_high) at (1-alpha) confidence level.
    """
    rng = np.random.default_rng(seed)
    n = len(rets)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    pt = annualized_sharpe(rets)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = annualized_sharpe(rets[idx])
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return pt, lo, hi


def sharpe_se_mertens(sharpe_daily: float, n: int, skew: float,
                       kurt_excess: float) -> float:
    """Analytical SE on Sharpe per Mertens (2002).

    sharpe_daily: NON-annualized (per-period) Sharpe.
    Returns SE on the per-period Sharpe; multiply by sqrt(365) for annualized.
    """
    if n <= 1:
        return float("nan")
    var_term = 1.0 - skew * sharpe_daily + ((kurt_excess + 2.0) / 4.0) * sharpe_daily ** 2
    if var_term <= 0:
        return float("nan")
    return float(math.sqrt(var_term / (n - 1)))


def compare_sharpes(a_rets: np.ndarray, b_rets: np.ndarray,
                     n_boot: int = 10000, seed: int = 42) -> Dict:
    """Paired-difference bootstrap on Sharpe(A) - Sharpe(B).

    Tests H0: Sharpe(A) == Sharpe(B). Both must have same length and dates.
    Returns dict: {diff, ci_low, ci_high, p_value_one_sided}.
    """
    if len(a_rets) != len(b_rets):
        raise ValueError("a_rets and b_rets must have same length")
    rng = np.random.default_rng(seed)
    n = len(a_rets)
    pt_a = annualized_sharpe(a_rets)
    pt_b = annualized_sharpe(b_rets)
    diff_pt = pt_a - pt_b
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = annualized_sharpe(a_rets[idx]) - annualized_sharpe(b_rets[idx])
    lo = float(np.nanpercentile(diffs, 2.5))
    hi = float(np.nanpercentile(diffs, 97.5))
    # one-sided p: prob that A's Sharpe <= B's
    p_one = float(np.mean(diffs <= 0))
    return {"diff": float(diff_pt), "ci_low": lo, "ci_high": hi,
              "p_value_one_sided": p_one}


# ============================================================================
# v3 JSON ingestion + sweep
# ============================================================================

def extract_returns_from_v3(json_path: Path) -> Optional[np.ndarray]:
    """Extract daily returns array from v3 JSON. Returns None if missing."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # v3 JSON usually has "equity_curve" or "daily_returns"
    for key in ("daily_returns", "daily_rets", "rets", "returns"):
        if key in data and isinstance(data[key], list):
            arr = np.asarray(data[key], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return arr if len(arr) > 0 else None
    # Reconstruct from equity_curve
    for key in ("equity_curve", "nav_curve", "equity"):
        if key in data and isinstance(data[key], list):
            eq = np.asarray(data[key], dtype=np.float64)
            eq = eq[np.isfinite(eq) & (eq > 0)]
            if len(eq) < 2:
                return None
            return np.diff(eq) / eq[:-1]
    # v3 per_day list (the actual v3 format): extract nav -> rets
    if "per_day" in data and isinstance(data["per_day"], list):
        navs = [r.get("nav") for r in data["per_day"]
                  if isinstance(r, dict) and r.get("nav") is not None]
        navs = np.asarray([n for n in navs if isinstance(n, (int, float))
                              and n > 0], dtype=np.float64)
        if len(navs) >= 2:
            return np.diff(navs) / navs[:-1]
        # Fallback: extract day_pnl_pct directly (already a return)
        pnls = [r.get("day_pnl_pct") for r in data["per_day"]
                  if isinstance(r, dict) and r.get("day_pnl_pct") is not None]
        pnls = np.asarray([p / 100.0 for p in pnls
                              if isinstance(p, (int, float))], dtype=np.float64)
        if len(pnls) >= 5:
            return pnls
    return None


def sweep_all_v3() -> List[Dict]:
    rows: List[Dict] = []
    for p in V3_LOGS.glob("paper_trade_replay_v3_*.json"):
        m = re.match(r"^paper_trade_replay_v3_(.+?)_u\d+_\d+_(\d+)\.json$", p.name)
        if not m:
            continue
        blend, end_date = m.group(1), m.group(2)
        rets = extract_returns_from_v3(p)
        if rets is None or len(rets) < 5:
            continue
        pt, lo, hi = bootstrap_sharpe_ci(rets, n_boot=2000)
        rows.append({
            "blend": blend, "window_end": end_date, "n": len(rets),
            "sharpe_pt": pt, "ci_low": lo, "ci_high": hi,
            "ci_width": hi - lo,
            "indistinguishable_from_zero": (lo <= 0 <= hi),
        })
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend", default=None)
    ap.add_argument("--json", default=None, help="Path to specific v3 JSON")
    ap.add_argument("--sweep", action="store_true", help="Sweep all v3 records")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.sweep:
        rows = sweep_all_v3()
        print(f"[sweep] {len(rows)} v3 records")
        rows.sort(key=lambda r: r.get("sharpe_pt", -99), reverse=True)
        summary_path = OUT_DIR / f"sweep_{dt.datetime.utcnow().strftime('%Y-%m-%d')}.md"
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Bootstrap Sharpe CI Sweep — {dt.datetime.utcnow().strftime('%Y-%m-%d')}\n\n")
            fh.write(f"n_boot=2000, alpha=0.05\n\n")
            fh.write(f"| Blend | Window | N | Sharpe | CI | Width | Indist 0? |\n")
            fh.write(f"|---|---|---|---|---|---|---|\n")
            for r in rows:
                flag = "🔴 YES" if r["indistinguishable_from_zero"] else "no"
                fh.write(f"| {r['blend']} | {r['window_end']} | {r['n']} | "
                          f"{r['sharpe_pt']:+.2f} | "
                          f"[{r['ci_low']:+.2f},{r['ci_high']:+.2f}] | "
                          f"{r['ci_width']:.2f} | {flag} |\n")
        n_indist = sum(1 for r in rows if r["indistinguishable_from_zero"])
        print(f"[sweep] indistinguishable from 0: {n_indist}/{len(rows)}")
        print(f"[sweep] summary at {summary_path}")
        return 0

    if args.json:
        rets = extract_returns_from_v3(Path(args.json))
        if rets is None:
            print(f"[boot] no returns array in {args.json}")
            return 2
        pt, lo, hi = bootstrap_sharpe_ci(rets, n_boot=args.n_boot)
        print(f"[boot] N={len(rets)} Sharpe={pt:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] "
              f"width={hi-lo:.3f} indist_0={lo <= 0 <= hi}")
        return 0

    print("Either --blend+--json or --sweep required", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
