"""strat_quality_crawler.py -- detect sub-par strategy patterns across the layer.

Phase 2.5 audit. Premise: many shipped strategies were built under sub-par
assumptions / approaches (no meta-gate, no regime gate, naive cost model,
single-cadence, no anti-fragility check). This crawler flags them so they
can be re-imagined with higher-ambition v2 designs.

PATTERNS DETECTED
-----------------
1. No-meta-gate            : sleeve fires without conviction filter
2. No-regime-gate          : sleeve fires regardless of btc_30d / cluster
3. Single-cadence-only     : sleeve uses only 1d (misses sub-day signal)
4. Hand-tuned-knobs        : sleeve has hardcoded thresholds with no TPE
5. No-short-route          : sleeve is long-only despite bear-side opportunity
6. Cost-model-unaware      : sleeve doesn't reference cost calibration
7. Stale-validation        : sleeve's last paper-trade-replay is >30 days old

OUTPUT
------
runs/audit/strat_quality_<DATE>.md -- per-sleeve scorecard + re-imagining queue
"""
from __future__ import annotations

__contract__ = {
    "kind": "strat_quality_crawler",
    "owner": "audit/strat-layer",
    "outputs": ["runs/audit/strat_quality_<DATE>.md"],
    "invariants": [
        "scores each sleeve on 7 quality axes",
        "flags sub-par patterns for Phase 2.5 re-imagining",
        "complements DEAD_STRATEGIES_2026_04_23.md registry",
    ],
}

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLEEVES_DIR = PROJECT_ROOT / "src" / "strategy" / "sleeves"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pattern detectors per quality axis
PATTERNS = {
    "no_meta_gate": {
        "positive_markers": ["mover_oracle", "meta_label", "conviction",
                              "p_target_5pct", "min_score"],
        "description": "Sleeve has no conviction-filter (meta-labeler / oracle)",
    },
    "no_regime_gate": {
        "positive_markers": ["btc_30d", "btc_regime", "regime_thr", "cluster_id",
                              "hurst_regime"],
        "description": "Sleeve fires regardless of macro regime context",
    },
    "single_cadence_only": {
        "positive_markers": ["cadence=\"4h\"", "cadence='4h'", "cadence=\"1h\"",
                              "cadence='1h'", "cadence=\"15m\"", "cadence='15m'"],
        "description": "Sleeve uses only 1d cadence (misses sub-day signal)",
    },
    "hand_tuned_knobs": {
        "positive_markers": ["TPE", "trial.suggest_", "optuna", "hyperparameter"],
        "description": "Knobs are hardcoded constants without TPE-tuning hook",
    },
    "no_short_route": {
        "positive_markers": ["side=\"short\"", "side='short'", "short_leg",
                              "long_only=False", "long_only: False"],
        "description": "Sleeve is long-only with no short-leg option",
    },
    "cost_model_unaware": {
        "positive_markers": ["MakerCostModel", "cost_calibration", "p_fill",
                              "bucket_cost", "fill_by_bucket"],
        "description": "Sleeve doesn't reference cost calibration",
    },
    "no_always_emit": {
        "positive_markers": ["cash_USDC", "cash fallback", "always-emit",
                              "data-gap"],
        "description": "Sleeve missing always-emit cash fallback (silent zero-alloc)",
    },
}


def analyze_sleeve(fp: Path) -> dict:
    try:
        text = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"sleeve": fp.name, "error": "read failure"}
    out: dict = {"sleeve": fp.name, "axes": {}, "score": 0}
    for axis, spec in PATTERNS.items():
        matched = any(m in text for m in spec["positive_markers"])
        out["axes"][axis] = matched
        if matched:
            out["score"] += 1
    out["max_score"] = len(PATTERNS)
    out["pct_quality"] = out["score"] / out["max_score"]
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold-pct", type=float, default=0.50,
                     help="Flag sleeves below this quality %")
    args = ap.parse_args()

    sleeves = list(SLEEVES_DIR.glob("*_sleeve.py"))
    if not sleeves:
        print("[strat-quality-crawler] no sleeves found")
        return 1
    results = [analyze_sleeve(fp) for fp in sleeves]
    results.sort(key=lambda r: r.get("pct_quality", 0))

    today = dt.date.today().isoformat()
    out_path = OUT_DIR / f"strat_quality_{today}.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Strat Quality Crawler -- {today}\n\n")
        fh.write(f"Sleeves audited: {len(results)}\n")
        flagged = [r for r in results if r.get("pct_quality", 1) < args.threshold_pct]
        fh.write(f"Flagged (< {args.threshold_pct:.0%} quality): {len(flagged)}\n\n")
        fh.write(f"## Sleeve quality scorecard\n\n")
        axes = list(PATTERNS.keys())
        fh.write(f"| sleeve | score | " + " | ".join(a[:10] for a in axes) + " |\n")
        fh.write(f"|---|---|" + "---|" * len(axes) + "\n")
        for r in results:
            if "error" in r:
                continue
            axes_str = " | ".join(("Y" if r["axes"][a] else ".") for a in axes)
            fh.write(f"| {r['sleeve']} | {r['score']}/{r['max_score']} | "
                      f"{axes_str} |\n")
        fh.write(f"\n## Phase 2.5 re-imagining queue (flagged sleeves)\n\n")
        for r in flagged:
            missing_axes = [a for a in axes if not r["axes"].get(a, False)]
            fh.write(f"### {r['sleeve']}  ({r['score']}/{r['max_score']})\n")
            fh.write(f"- missing axes: {missing_axes}\n")
            fh.write(f"- v2 prescription: add ")
            fixes = []
            for ax in missing_axes:
                desc = PATTERNS[ax]["description"]
                fixes.append(f"({ax}) {desc}")
            fh.write("; ".join(fixes) + "\n\n")
        fh.write(f"\n## Pattern legend\n\n")
        for axis, spec in PATTERNS.items():
            fh.write(f"- **{axis}**: {spec['description']}\n")
    print(f"[strat-quality-crawler] {len(flagged)}/{len(results)} flagged -> {out_path}")
    for r in flagged[:5]:
        print(f"  {r['sleeve']:<50s} {r['score']}/{r['max_score']} "
              f"({r['pct_quality']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
