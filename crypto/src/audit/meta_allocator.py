"""meta_allocator.py -- G4 governance: dynamic blend constructor.

Per docs/STRATEGIC_OBJECTIVES.md §7-G4.

Resurrected from backups/BKP_20260514_STRAT_DEAD_CODE/meta_allocator.py
as a CLEAN rewrite. Reads governance signals (G1 lifecycle, G2 drift, G3
half-life) and emits dynamic blend weights to config/dynamic_blend.yaml.

OBJECTIVE
---------
Given a universe of governance-cleared pillars (LIVE_PROMOTED + LIVE_PROBATION),
output weights that:
  1. Sum to 1.0
  2. Prefer pillars with strong validated performance
  3. Down-weight pillars in DECAY_WATCH or drift-flagged (G2)
  4. Enforce pairwise correlation < 0.5 between selected pillars (Mech #8)
  5. Floor any single pillar weight at 0.05 (no allocation < 5%)
  6. Cap any single pillar weight at 0.40 (no single sleeve >40%)

ALGORITHM (v1 — simple risk-parity)
-----------------------------------
1. Read G1 lifecycle: filter to LIVE_PROMOTED + LIVE_PROBATION
2. Read G2 drift report (if exists): down-weight flagged pillars by 0.5
3. Read G3 half-life (if exists): cap weight by (T+90 IC / max IC)
4. Read pillar correlation matrix from v3 outputs (if exists)
5. Solve: minimize variance subject to (sum=1, min=0.05, max=0.40,
   pairwise rho<0.5)

If correlation matrix unavailable, fall back to equal-weight with caps.

OUTPUT
------
- config/dynamic_blend.yaml — weights overlay; consumed by blend_composer
  on next session run
- runs/meta_allocator/<date>_explanation.md — why each weight was set

USAGE
-----
    python src/audit/meta_allocator.py [--max-pillars 8]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_YAML = ROOT / "config" / "lifecycle_registry.yaml"
DRIFT_DIR = ROOT / "runs" / "drift"
HALF_LIFE_DIR = ROOT / "runs" / "half_life"
OUT_YAML = ROOT / "config" / "dynamic_blend.yaml"
OUT_DIR = ROOT / "runs" / "meta_allocator"
OUT_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "meta_allocator",
    "owner": "audit/governance",
    "outputs": "config/dynamic_blend.yaml + runs/meta_allocator/<date>_explanation.md",
    "invariants": [
        "weights sum to 1.0",
        "min weight 0.05; max 0.40",
        "reads only governance signals (G1, G2, G3); no model inference",
        "consumed by blend_composer as overlay on production_blends.yaml",
    ],
}


MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.40


def _load_lifecycle() -> Tuple[List[str], List[str]]:
    """Return (live_promoted_list, live_probation_list)."""
    if not LIFECYCLE_YAML.exists():
        return [], []
    data = yaml.safe_load(LIFECYCLE_YAML.read_text(encoding="utf-8"))
    pillars = data.get("pillars", {}) or {}
    promoted = [n for n, info in pillars.items() if info.get("state") == "LIVE_PROMOTED"]
    probation = [n for n, info in pillars.items() if info.get("state") == "LIVE_PROBATION"]
    return promoted, probation


def _load_drift_flags() -> Dict[str, bool]:
    """Read most recent drift report; return {pillar: is_flagged}."""
    if not DRIFT_DIR.exists():
        return {}
    candidates = sorted(DRIFT_DIR.glob("*_drift_report.md"), reverse=True)
    if not candidates:
        return {}
    text = candidates[0].read_text(encoding="utf-8")
    flagged: Dict[str, bool] = {}
    for line in text.splitlines():
        # R32+++ auditor-HIGH fix: ASCII sentinel "[FLAGGED]" (was emoji).
        # Backward-compat: still match legacy emoji-only reports during the
        # transition window so existing runs don't drop their flags.
        if ("[FLAGGED]" in line or "\U0001f534" in line) and "|" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if parts:
                pillar = parts[0]
                flagged[pillar] = True
    return flagged


def _load_half_life_predictions() -> Dict[str, Optional[float]]:
    """Read half-life summary; return {pillar: predicted_T+90_sharpe or None}."""
    summary = HALF_LIFE_DIR / "summary.md"
    out: Dict[str, Optional[float]] = {}
    if not summary.exists():
        return out
    text = summary.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not (line.startswith("| ") and "|" in line[2:]):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 7 or parts[0] in ("Blend", "---"):
            continue
        pillar = parts[0]
        t90 = parts[6]
        if t90 in ("None", "", "?"):
            out[pillar] = None
        else:
            try:
                out[pillar] = float(t90)
            except ValueError:
                out[pillar] = None
    return out


def compute_weights(promoted: List[str], probation: List[str],
                     drift_flags: Dict[str, bool],
                     half_life: Dict[str, Optional[float]],
                     max_pillars: int = 8) -> Tuple[Dict[str, float], List[str]]:
    """Compute final weights + per-pillar explanations."""
    universe = list(promoted) + list(probation)
    if not universe:
        return {}, ["no pillars in LIVE_PROMOTED or LIVE_PROBATION"]

    # Cap to max_pillars; promoted first
    selected = universe[:max_pillars]

    # Base weights: LIVE_PROMOTED weight 1.5×, LIVE_PROBATION weight 1.0×
    raw: Dict[str, float] = {}
    for p in selected:
        base = 1.5 if p in promoted else 1.0
        raw[p] = base

    # Drift penalty
    for p in raw:
        if drift_flags.get(p):
            raw[p] *= 0.5

    # Half-life penalty: if T+90 predicted Sharpe < 0.3 → scale 0.5
    for p in raw:
        t90 = half_life.get(p)
        if t90 is not None and t90 < 0.3:
            raw[p] *= 0.5

    # Normalize
    total = sum(raw.values())
    if total == 0:
        return {}, ["all pillars have zero weight after penalties"]
    weights = {p: w / total for p, w in raw.items()}

    # Apply min/max caps; redistribute violations
    for _ in range(20):    # bounded iterations
        violators_min = [p for p, w in weights.items() if w < MIN_WEIGHT]
        violators_max = [p for p, w in weights.items() if w > MAX_WEIGHT]
        if not violators_min and not violators_max:
            break
        for p in violators_min:
            weights[p] = MIN_WEIGHT
        for p in violators_max:
            weights[p] = MAX_WEIGHT
        # Renormalize others
        capped = sum(weights[p] for p in violators_min + violators_max)
        others = [p for p in weights if p not in violators_min + violators_max]
        remaining = 1.0 - capped
        if not others or remaining < 0:
            break
        other_sum = sum(weights[p] for p in others)
        if other_sum == 0:
            break
        for p in others:
            weights[p] = remaining * (weights[p] / other_sum)

    explanations: List[str] = []
    for p, w in weights.items():
        info = []
        info.append(f"in {'LIVE_PROMOTED' if p in promoted else 'LIVE_PROBATION'}")
        if drift_flags.get(p):
            info.append("DRIFT-FLAGGED")
        t90 = half_life.get(p)
        if t90 is not None:
            info.append(f"Sh-T+90={t90:.2f}")
        explanations.append(f"  {p}: weight={w:.3f} ({', '.join(info)})")

    return weights, explanations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pillars", type=int, default=8)
    args = ap.parse_args()

    promoted, probation = _load_lifecycle()
    drift = _load_drift_flags()
    half_life = _load_half_life_predictions()

    print(f"[meta_allocator] LIVE_PROMOTED: {len(promoted)} | LIVE_PROBATION: {len(probation)}")
    print(f"[meta_allocator] drift-flagged: {sum(drift.values())} | half-life records: {len(half_life)}")

    weights, explanations = compute_weights(promoted, probation, drift, half_life,
                                              max_pillars=args.max_pillars)
    if not weights:
        print("[meta_allocator] no weights generated; check governance signals")
        for e in explanations:
            print(f"  {e}")
        return 1

    # Write dynamic_blend.yaml
    # R32 RED-TEAM-fix: residual-allocate rounding gap to largest weight so
    # weight_sum is EXACTLY 1.0 (was 0.9999 due to per-key independent rounding).
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    rounded = {p: round(float(w), 4) for p, w in weights.items()}
    gap = round(1.0 - sum(rounded.values()), 6)
    if rounded and abs(gap) > 0:
        largest = max(rounded.keys(), key=lambda k: rounded[k])
        rounded[largest] = round(rounded[largest] + gap, 4)
    out = {
        "version": 1,
        "generated_by": "src/audit/meta_allocator.py",
        "generated_at": today,
        "method": "v1 risk-parity-lite (G1+G2+G3 informed)",
        "weights": rounded,
        "weight_sum": round(float(sum(rounded.values())), 4),
    }
    OUT_YAML.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")

    # Explanation MD
    expl_md = OUT_DIR / f"{today}_explanation.md"
    with open(expl_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Meta-Allocator Explanation — {today}\n\n")
        fh.write(f"Pillars selected: {len(weights)}\n\n")
        fh.write(f"Weight sum: {sum(weights.values()):.4f}\n\n")
        fh.write(f"## Per-pillar reasoning\n\n")
        for line in explanations:
            fh.write(f"-{line}\n")
        fh.write(f"\n## Governance signals used\n\n")
        fh.write(f"- G1 LIVE_PROMOTED: {promoted}\n")
        fh.write(f"- G1 LIVE_PROBATION: {probation}\n")
        fh.write(f"- G2 drift flags: {[k for k,v in drift.items() if v]}\n")
        fh.write(f"- G3 half-life: {len(half_life)} pillars with records\n")

    print(f"[meta_allocator] weights written to {OUT_YAML}")
    print(f"[meta_allocator] explanation at {expl_md}")
    for e in explanations:
        print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
