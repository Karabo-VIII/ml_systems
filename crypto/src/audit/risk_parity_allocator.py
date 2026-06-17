"""risk_parity_allocator.py -- Block 3 Mech #6: risk-parity blend constructor.

PURPOSE
-------
Per STRATEGIC_OBJECTIVES.md Mech #6 (Bridgewater risk parity), the project
had `hrp_allocator.py` archived 2026-05-14. This is a CLEAN rewrite using
inverse-volatility (simple risk parity) + sleeve correlation penalty.

Extracts sleeve returns from v3 paper-trade-replay logs (per_day day_pnl_pct
arrays), computes covariance matrix, allocates weights inversely proportional
to volatility, then adjusts for pairwise correlation (down-weights highly-
correlated pairs).

ALGORITHM
---------
1. For each LIVE_PROMOTED + LIVE_PROBATION pillar with v3 history:
   - Extract daily return series from latest v3 JSON's per_day list
   - Compute annualized vol = std * sqrt(365)
2. Inverse-vol weights: w_i = (1/vol_i) / sum(1/vol_j)
3. Correlation adjustment: for pairs with rho > 0.5, halve weight
4. Renormalize, apply min/max caps (0.05 / 0.40 same as meta_allocator)

OUTPUT
------
- config/risk_parity_blend.yaml — weights overlay (separate from G4 dynamic_blend.yaml
  to allow comparison / A-B test)
- runs/risk_parity/<date>_explanation.md — covariance matrix + per-pillar metrics

USAGE
-----
    python src/audit/risk_parity_allocator.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_YAML = ROOT / "config" / "lifecycle_registry.yaml"
V3_LOGS = ROOT / "logs" / "strat_audit"
OUT_YAML = ROOT / "config" / "risk_parity_blend.yaml"
OUT_DIR = ROOT / "runs" / "risk_parity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "risk_parity_allocator",
    "owner": "audit/governance",
    "outputs": "config/risk_parity_blend.yaml + runs/risk_parity/explanation.md",
    "invariants": [
        "inverse-vol weighting from v3 return histories",
        "pairwise rho > 0.5 penalty (halve weight)",
        "min 0.05 / max 0.40 caps",
        "Mech #6 (Bridgewater) implementation; NOT replacement for G4 meta-allocator",
    ],
}


MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.40
RHO_PENALTY_THRESHOLD = 0.5
SQRT_ANN = float(np.sqrt(365))


def _load_lifecycle_universe() -> List[str]:
    if not LIFECYCLE_YAML.exists():
        return []
    data = yaml.safe_load(LIFECYCLE_YAML.read_text(encoding="utf-8")) or {}
    pillars = data.get("pillars", {}) or {}
    return [n for n, info in pillars.items()
              if info.get("state") in ("LIVE_PROMOTED", "LIVE_PROBATION")]


def _extract_returns_and_dates(blend: str) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Daily returns + dates from latest v3 JSON (single-pass, row-aligned).

    R32+++ auditor-HIGH fix: previously had TWO filter passes (one for pnls,
    one for dates) with DIFFERENT predicates (pnls dropped None pnl + non-
    numeric; dates dropped rows without 'date'). When a non-end row had a
    None pnl, the date list was longer than pnls and `dates[-len(r):]`
    trimmed wrong dates → silent date-misalignment in correlation matrix.
    Single-pass extraction guarantees pnls[i] corresponds to dates[i].
    """
    pattern = re.compile(rf"^paper_trade_replay_v3_{re.escape(blend)}_u\d+_(\d+)_(\d+)\.json$")
    candidates = []
    for p in V3_LOGS.glob(f"paper_trade_replay_v3_{blend}_*.json"):
        m = pattern.match(p.name)
        if m:
            candidates.append((p, m.group(2)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    try:
        data = json.loads(candidates[0][0].read_text(encoding="utf-8"))
    except Exception:
        return None
    if "per_day" not in data or not isinstance(data["per_day"], list):
        return None
    pnls: List[float] = []
    dates: List[str] = []
    for row in data["per_day"]:
        if not isinstance(row, dict):
            continue
        pnl = row.get("day_pnl_pct")
        date = row.get("date")
        if pnl is None or date is None or not isinstance(pnl, (int, float)):
            continue
        pnls.append(float(pnl) / 100.0)
        dates.append(str(date))
    if len(pnls) < 5:
        return None
    return np.asarray(pnls, dtype=np.float64), dates


def _extract_returns(blend: str) -> Optional[np.ndarray]:
    """Backwards-compat wrapper -- returns only the pnl array."""
    pair = _extract_returns_and_dates(blend)
    return pair[0] if pair is not None else None


def compute_weights(pillars: List[str]) -> Tuple[Dict[str, float], Dict]:
    """Run inverse-vol allocation with correlation penalty."""
    # R32+++ auditor-HIGH fix: extract returns + dates as a single row-aligned
    # pair, eliminating the prior dual-filter row-misalignment bug.
    aligned_dates: Dict[str, List[str]] = {}
    aligned_rets: Dict[str, np.ndarray] = {}
    for p in pillars:
        pair = _extract_returns_and_dates(p)
        if pair is None:
            continue
        r, dates = pair
        aligned_rets[p] = r
        aligned_dates[p] = dates
    if not aligned_rets:
        return {}, {"error": "no v3 returns available for any pillar"}

    if not aligned_rets:
        return {}, {"error": "no date-aligned return series available"}

    # Build date-indexed DataFrame; intersect by date
    import pandas as _pd
    df_dict = {p: _pd.Series(aligned_rets[p], index=aligned_dates[p])
                 for p in aligned_rets}
    df = _pd.DataFrame(df_dict).dropna(how="any")
    if len(df) < 5:
        return {}, {"error": f"intersected return series too short ({len(df)} days)"}
    aligned = {p: df[p].values for p in df.columns}

    # Vol + corr matrix
    vols: Dict[str, float] = {}
    for p, r in aligned.items():
        s = float(r.std())
        vols[p] = max(s * SQRT_ANN, 1e-6)

    # Inverse-vol weights
    inv = {p: 1.0 / v for p, v in vols.items()}
    total_inv = sum(inv.values())
    raw = {p: w / total_inv for p, w in inv.items()}

    # Correlation matrix
    names = list(aligned.keys())
    mat = np.stack([aligned[n] for n in names], axis=0)
    if len(names) > 1:
        corr = np.corrcoef(mat)
    else:
        corr = np.array([[1.0]])

    # Correlation penalty: for each pair (i,j) with rho > threshold, halve the LARGER.
    # R32 RED-TEAM-fix H4: previously halved the SMALLER → increased concentration in
    # the dominant pillar of correlated pairs (opposite of risk parity intent).
    # Now: penalize the LARGER pillar so concentration reduces with correlation.
    penalty = {p: 1.0 for p in names}
    high_corr_pairs: List[Tuple[str, str, float]] = []
    for i, p_i in enumerate(names):
        for j, p_j in enumerate(names):
            if i >= j:
                continue
            rho = float(corr[i, j])
            if rho > RHO_PENALTY_THRESHOLD:
                high_corr_pairs.append((p_i, p_j, rho))
                larger = p_i if raw[p_i] >= raw[p_j] else p_j
                penalty[larger] *= 0.5

    penalized = {p: raw[p] * penalty[p] for p in names}
    total = sum(penalized.values())
    if total <= 0:
        return {}, {"error": "all weights penalized to zero"}
    weights = {p: w / total for p, w in penalized.items()}

    # Apply min/max caps iteratively
    for _ in range(20):
        viol_min = [p for p, w in weights.items() if w < MIN_WEIGHT]
        viol_max = [p for p, w in weights.items() if w > MAX_WEIGHT]
        if not viol_min and not viol_max:
            break
        for p in viol_min:
            weights[p] = MIN_WEIGHT
        for p in viol_max:
            weights[p] = MAX_WEIGHT
        others = [p for p in weights if p not in viol_min + viol_max]
        capped = sum(weights[p] for p in viol_min + viol_max)
        remaining = 1.0 - capped
        if not others or remaining < 0:
            break
        other_sum = sum(weights[p] for p in others)
        if other_sum == 0:
            break
        for p in others:
            weights[p] = remaining * (weights[p] / other_sum)

    # R32 RED-TEAM-fix C1: post-loop guard against overconstrained allocations.
    # When N*MIN_WEIGHT > 1.0 (too many pillars), the iteration can leave
    # weights summing to > 1, after which residual-rounding (-gap to largest)
    # could produce NEGATIVE weights. Now: renormalize sum and assert bounds.
    total = sum(weights.values())
    if total > 0:
        weights = {p: w / total for p, w in weights.items()}
    # Clamp any rounding-induced micro-violations
    for p in list(weights.keys()):
        if weights[p] < 0:
            weights[p] = 0.0
        elif weights[p] > MAX_WEIGHT:
            weights[p] = MAX_WEIGHT
    # Final renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {p: w / total for p, w in weights.items()}

    # Residual rounding fix
    rounded = {p: round(float(w), 4) for p, w in weights.items()}
    gap = round(1.0 - sum(rounded.values()), 6)
    if rounded and abs(gap) > 0:
        largest = max(rounded.keys(), key=lambda k: rounded[k])
        rounded[largest] = round(rounded[largest] + gap, 4)

    explanation = {
        "n_pillars": len(names),
        "vols": {p: round(v, 4) for p, v in vols.items()},
        "high_corr_pairs": [(a, b, round(r, 3)) for a, b, r in high_corr_pairs],
        "penalty": {p: round(penalty[p], 3) for p in names},
    }
    return rounded, explanation


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    universe = _load_lifecycle_universe()
    print(f"[risk_parity] LIVE_PROMOTED+PROBATION universe: {len(universe)} pillars")
    weights, expl = compute_weights(universe)
    if not weights:
        print(f"[risk_parity] FAIL: {expl.get('error')}")
        return 1

    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    out = {
        "version": 1,
        "generated_by": "src/audit/risk_parity_allocator.py",
        "generated_at": today,
        "method": "inverse-vol with rho>0.5 correlation penalty (Mech #6)",
        "weights": weights,
        "weight_sum": round(sum(weights.values()), 4),
    }
    OUT_YAML.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")

    expl_md = OUT_DIR / f"{today}_explanation.md"
    with open(expl_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Risk-Parity Allocator Explanation — {today}\n\n")
        fh.write(f"Pillars: {expl['n_pillars']}\n\n")
        fh.write(f"## Annualized volatility\n\n")
        for p, v in expl["vols"].items():
            fh.write(f"- {p}: {v}\n")
        fh.write(f"\n## High-correlation pairs (rho > {RHO_PENALTY_THRESHOLD})\n\n")
        for a, b, r in expl["high_corr_pairs"]:
            fh.write(f"- {a} <-> {b}: rho={r}\n")
        fh.write(f"\n## Final weights\n\n")
        for p, w in weights.items():
            fh.write(f"- {p}: {w:.4f}\n")

    print(f"[risk_parity] weights -> {OUT_YAML}")
    print(f"[risk_parity] explanation -> {expl_md}")
    for p, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {p}: {w:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
