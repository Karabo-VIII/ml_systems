"""yaml_claim_corrector.py -- B2.1: replace inflated yaml aggregate_metrics_*
claims with v3 measured values.

PURPOSE
-------
Per R14 audit, yaml aggregate_metrics_* fields are systematically inflated
versus v3 paper-trade-replay reality (7.6x mean inflation found). G2 drift
monitor now detects 13+ drift-flagged metrics. This script generates a
patch yaml `config/production_blends_v3_corrected.yaml` that replaces
inflated claims with v3-measured numbers, with provenance comments.

OUTPUT
------
- Writes config/production_blends_v3_corrected.yaml (NEW file, not in-place
  overwrite). Operator reviews + manually merges into production_blends.yaml
  after sanity check.
- Writes runs/drift/yaml_correction_<DATE>.md summary of changes.

CONTRACT
--------
- Reads config/production_blends.yaml (claims) + latest v3 JSON per blend
- For each blend with a v3 record AND yaml claim, replaces claim with
  v3 measurement
- Preserves `note`/`caveats` fields unchanged
- Annotates corrections with `_v3_source` field citing the v3 JSON
- Does NOT delete original blends (safe to run; output is separate file)

USAGE
-----
    python src/audit/yaml_claim_corrector.py [--dry-run]
"""
from __future__ import annotations

import argparse
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
OUT_YAML = ROOT / "config" / "production_blends_v3_corrected.yaml"
OUT_DIR = ROOT / "runs" / "drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "yaml_corrector",
    "owner": "audit/governance",
    "outputs": "config/production_blends_v3_corrected.yaml + runs/drift/correction_summary.md",
    "invariants": [
        "never overwrites config/production_blends.yaml in-place",
        "replaces aggregate_metrics_* claims with v3 measured values",
        "preserves note/caveats fields; annotates with _v3_source",
    ],
}


# R32++ Lane A: V3_KEY_ALIASES superseded by canonical registry at
# src/audit/v3_schema.py V3_FIELD_MAP. Kept for backward compat with any
# external caller; new code should import from v3_schema.
import sys as _sys
from pathlib import Path as _Path
_src = _Path(__file__).resolve().parents[1]
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))
from audit.v3_schema import V3_FIELD_MAP as V3_KEY_ALIASES  # noqa: E402,F401


def _latest_v3_for_blend(blend: str) -> Optional[Tuple[Path, str]]:
    """Return (json_path, window_end_date) for the most recent v3 of this blend."""
    pattern = re.compile(rf"^paper_trade_replay_v3_{re.escape(blend)}_u\d+_(\d+)_(\d+)\.json$")
    candidates: List[Tuple[Path, str]] = []
    for p in V3_LOGS.glob(f"paper_trade_replay_v3_{blend}_*.json"):
        m = pattern.match(p.name)
        if m:
            end_date = m.group(2)
            candidates.append((p, end_date))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def _extract_v3_metrics(json_path: Path) -> Dict[str, float]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for canonical, aliases in V3_KEY_ALIASES.items():
        for a in aliases:
            if a in data and isinstance(data[a], (int, float)):
                out[canonical] = float(data[a])
                break
    return out


def _v3_aggregate_block(metrics: Dict[str, float], v3_json: Path,
                          window_end: str) -> Dict[str, Any]:
    """Compose a corrected aggregate_metrics_v3 block."""
    block: Dict[str, Any] = {
        "_v3_source": str(v3_json.relative_to(ROOT)).replace("\\", "/"),
        "_v3_window_end": window_end,
        "_corrected_by": "src/audit/yaml_claim_corrector.py",
        "_corrected_at": dt.datetime.utcnow().strftime("%Y-%m-%d"),
    }
    if "sharpe_ann" in metrics:
        block["sharpe"] = round(metrics["sharpe_ann"], 4)
    if "total_ret_pct" in metrics:
        block["total_ret_pct"] = round(metrics["total_ret_pct"], 4)
    if "max_dd_pct" in metrics:
        block["max_dd_pct"] = round(metrics["max_dd_pct"], 4)
    if "hit_rate_pct" in metrics:
        block["hit_rate_pct"] = round(metrics["hit_rate_pct"], 4)
    if "n_trades" in metrics:
        block["n_trades"] = int(metrics["n_trades"])
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="Print changes without writing files")
    args = ap.parse_args()

    with open(BLENDS_YAML, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root_key = next((k for k in data if k.startswith("production_blends")), None)
    if not root_key:
        print(f"[corrector] no production_blends key in {BLENDS_YAML}")
        return 2

    blends = data[root_key]
    corrections: List[Dict] = []

    for blend_name, defn in blends.items():
        if not isinstance(defn, dict):
            continue
        v3_info = _latest_v3_for_blend(blend_name)
        if not v3_info:
            continue
        v3_path, window_end = v3_info
        v3_metrics = _extract_v3_metrics(v3_path)
        if not v3_metrics:
            continue
        # Build corrected block + attach to blend (under new key
        # `aggregate_metrics_v3_corrected` so original yaml claims are
        # preserved for diff/audit)
        corrected_block = _v3_aggregate_block(v3_metrics, v3_path, window_end)
        defn["aggregate_metrics_v3_corrected"] = corrected_block
        corrections.append({
            "blend": blend_name,
            "v3_source": str(v3_path.name),
            "v3_window_end": window_end,
            "v3_metrics": v3_metrics,
        })

    # Write output yaml
    if not args.dry_run:
        with open(OUT_YAML, "w", encoding="utf-8") as fh:
            fh.write(f"# Generated by src/audit/yaml_claim_corrector.py — "
                      f"{dt.datetime.utcnow().strftime('%Y-%m-%d')}\n")
            fh.write("# Replaces inflated aggregate_metrics_* claims with v3 measured values.\n")
            fh.write("# Each blend retains its original claims; v3-corrected block added.\n")
            fh.write("# Operator reviews this file, then merges relevant entries into\n")
            fh.write("# config/production_blends.yaml.\n\n")
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        print(f"[corrector] wrote {OUT_YAML} ({len(corrections)} blends corrected)")

    # Write summary
    summary_path = OUT_DIR / f"yaml_correction_{dt.datetime.utcnow().strftime('%Y-%m-%d')}.md"
    if not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Yaml Claim Correction Summary — {dt.datetime.utcnow().strftime('%Y-%m-%d')}\n\n")
            fh.write(f"Total blends corrected: {len(corrections)}\n\n")
            fh.write(f"| Blend | V3 Window | V3 Sharpe | V3 Total% | V3 DD% | V3 Trades |\n")
            fh.write(f"|---|---|---|---|---|---|\n")
            for c in corrections:
                m = c["v3_metrics"]
                fh.write(f"| {c['blend']} | {c['v3_window_end']} | "
                          f"{m.get('sharpe_ann', '?')} | "
                          f"{m.get('total_ret_pct', '?')} | "
                          f"{m.get('max_dd_pct', '?')} | "
                          f"{m.get('n_trades', '?')} |\n")
        print(f"[corrector] summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
