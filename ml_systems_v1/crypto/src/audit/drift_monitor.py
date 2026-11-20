"""drift_monitor.py -- G2 governance: yaml-vs-v3 drift detector.

Per docs/STRATEGIC_OBJECTIVES.md §7-G2.

Compares aggregate_metrics_* claims in config/production_blends.yaml against
the most recent v3 paper-trade-replay outputs in logs/strat_audit/. Flags
deviations >= 30%. Outputs Markdown drift report.

Background: 2026-05-14 audit (R14) found 7.6× inflation gap between yaml
claims and v3 reality. This monitor detects that drift automatically.

USAGE
-----
    python src/audit/drift_monitor.py
        # Generates runs/drift/<date>_drift_report.md

CONTRACT
--------
- Reads config/production_blends.yaml (claims)
- Reads logs/strat_audit/paper_trade_replay_v3_*.json (reality)
- Maps yaml metric keys -> v3 JSON keys (Sharpe, total_ret, DD, hit)
- Flags drift >= 30% on any metric as [FLAGGED] (was emoji; ASCII per R32+++)
- Auto-routes flagged pillars to DECAY_WATCH (G1)
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
BLENDS_YAML = ROOT / "config" / "production_blends.yaml"
V3_LOGS = ROOT / "logs" / "strat_audit"
OUT_DIR = ROOT / "runs" / "drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "drift_monitor",
    "owner": "audit/governance",
    "outputs": "runs/drift/<date>_drift_report.md",
    "invariants": [
        "compares claimed yaml metrics to measured v3 JSONs",
        "30% deviation is the flag threshold",
        "drift triggers G1 LIVE_PROMOTED -> DECAY_WATCH transition",
    ],
}

DRIFT_FLAG_PCT = 30.0   # deviation above which a metric is flagged


def _latest_v3_for_blend(blend: str) -> Optional[Path]:
    """Find most recent v3 JSON for this blend."""
    pattern = re.compile(rf"^paper_trade_replay_v3_{re.escape(blend)}_(u\d+)_(\d+)_(\d+)\.json$")
    candidates: List[Tuple[Path, str]] = []
    for p in V3_LOGS.glob(f"paper_trade_replay_v3_{blend}_*.json"):
        m = pattern.match(p.name)
        if m:
            end_date = m.group(3)  # YYYYMMDD
            candidates.append((p, end_date))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _extract_yaml_claims(blend_def: Dict) -> Dict[str, float]:
    """Pull metric claims from any aggregate_metrics_* dict in a blend definition."""
    claims: Dict[str, float] = {}
    for k, v in blend_def.items():
        if not k.startswith("aggregate_metrics"):
            continue
        if not isinstance(v, dict):
            continue
        for mk, mv in v.items():
            if mk == "note":
                continue
            if isinstance(mv, (int, float)):
                # Normalize key names
                kk = mk.replace("total_ret_pct", "total_ret_pct") \
                       .replace("sharpe", "sharpe") \
                       .replace("max_dd_pct", "max_dd_pct") \
                       .replace("hit_rate_pct", "hit_rate_pct")
                if kk not in claims:
                    claims[kk] = float(mv)
    return claims


def _v3_metrics(json_path: Path) -> Dict[str, float]:
    """R32++ Lane A: read v3 metrics via canonical schema registry, NOT
    hand-coded aliases. Eliminates the drift class of bug where this file
    and v3_schema disagree on key names."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # Lazy import to avoid circular
    import sys as _sys
    from pathlib import Path as _Path
    _src = _Path(__file__).resolve().parents[1]
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from audit.v3_schema import extract_v3_metrics, derive_hit_rate_pct
    raw = extract_v3_metrics(
        data,
        canonical_keys=("sharpe_ann", "total_ret_pct", "max_dd_pct",
                          "n_trades", "hit_rate_pct"),
    )
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    # hit_rate_pct derived from per_day if not at top-level
    if "hit_rate_pct" not in out:
        hr = derive_hit_rate_pct(data)
        if hr is not None:
            out["hit_rate_pct"] = float(hr)
    return out


def _deviation_pct(claimed: float, measured: float) -> float:
    """Return percentage drift: positive if claim > measured."""
    if measured == 0:
        return float("inf") if claimed != 0 else 0.0
    return 100.0 * (claimed - measured) / abs(measured)


def main() -> int:
    with open(BLENDS_YAML, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    root_key = next((k for k in data if k.startswith("production_blends")), None)
    if not root_key:
        print(f"[drift] no production_blends key in {BLENDS_YAML}")
        return 2
    blends = data[root_key]

    rows: List[Dict] = []
    for name, defn in blends.items():
        if not isinstance(defn, dict):
            continue
        claims = _extract_yaml_claims(defn)
        v3_path = _latest_v3_for_blend(name)
        v3_metrics = _v3_metrics(v3_path) if v3_path else {}
        row = {
            "blend": name,
            "v3_path": str(v3_path) if v3_path else None,
            "claims": claims,
            "v3": v3_metrics,
            "drift": {},
        }
        # Compute drift
        for key in ("sharpe_ann", "total_ret_pct", "max_dd_pct"):
            claim_keys = [k for k in claims if key.replace("_ann", "") in k.lower()
                            or key.replace("_pct", "_pct") in k.lower()]
            if not claim_keys:
                continue
            if key not in v3_metrics:
                continue
            claimed = claims[claim_keys[0]]
            measured = v3_metrics[key]
            row["drift"][key] = {
                "claimed": claimed,
                "measured": measured,
                "drift_pct": round(_deviation_pct(claimed, measured), 2),
            }
        rows.append(row)

    # Report
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    out_md = OUT_DIR / f"{today}_drift_report.md"
    n_flagged = 0
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Drift Monitor Report — {today}\n\n")
        fh.write(f"Total blends scanned: {len(rows)}\n")
        n_with_v3 = sum(1 for r in rows if r["v3_path"])
        n_with_claims = sum(1 for r in rows if r["claims"])
        fh.write(f"With v3 JSON: {n_with_v3} | With yaml claims: {n_with_claims}\n\n")
        fh.write(f"Drift flag threshold: ±{DRIFT_FLAG_PCT}%\n\n")
        fh.write(f"| Blend | Metric | Claimed | Measured | Drift% | Flag |\n")
        fh.write(f"|---|---|---|---|---|---|\n")
        for r in rows:
            for metric, info in r["drift"].items():
                # R32+++ auditor-HIGH fix: ASCII sentinels (was emoji). CLAUDE.md
                # bans emoji in py files (Windows cp1252 crash) AND meta_allocator
                # reads this token via substring match — both ends now use ASCII.
                flag = "[FLAGGED]" if abs(info["drift_pct"]) >= DRIFT_FLAG_PCT else "[OK]"
                if abs(info["drift_pct"]) >= DRIFT_FLAG_PCT:
                    n_flagged += 1
                fh.write(f"| {r['blend']} | {metric} | {info['claimed']} | "
                          f"{info['measured']} | {info['drift_pct']:+.1f}% | {flag} |\n")
        fh.write(f"\n## Summary\n\n")
        fh.write(f"- Blends with drift flagged: {n_flagged}\n")
        fh.write(f"- Blends with no v3 record: {len(rows) - n_with_v3}\n")
        fh.write(f"\n## Action items\n\n")
        fh.write(f"- Flagged blends should be re-validated via v3 OR have their yaml "
                  f"`aggregate_metrics_*` corrected.\n")
        fh.write(f"- Blends with no v3 record (BIRTH state) need at least one v3 "
                  f"window before promotion to PAPER.\n")
    print(f"[drift] report written to {out_md}")
    print(f"[drift] {n_flagged} metrics flagged ({DRIFT_FLAG_PCT}%+ drift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
