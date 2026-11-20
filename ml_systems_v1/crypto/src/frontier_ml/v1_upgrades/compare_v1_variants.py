"""Side-by-side variant comparison from V1.1 training logs.

Reads logs in `logs/v1/v1_1/v1_1_<feat>_<run-tag>_train_<ts>.log` and emits a
table of best-ShIC / best-IC / final-Gap per variant, plus a winner-by-ShIC
ranking and a per-variant decision relative to the V1.1 single-model baseline
record (IC=0.0674 / ShIC=0.0330).

Usage:
    python -m frontier_ml.v1_upgrades.compare_v1_variants \
        --features 29 --run-tags baseline sam mtp mdn fraug label_noise logit_clip
        # OR autodiscover all matching logs:
    python -m frontier_ml.v1_upgrades.compare_v1_variants --features 29 --auto

Output: stdout table + JSON summary at logs/frontier_ml/v1_variant_compare/<ts>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Parametrized on --version (default v1_1 for back-compat); resolved at main()
DEFAULT_VERSION = "v1_1"


def _log_root_for(version: str) -> Path:
    """Map version_id to its logs directory.

    Layout:  logs/{family}/{version}/  for V1.x and V3-V14
        v1_0..v1_6 -> family 'v1'
        v3..v14    -> own family
    """
    family = "v1" if version.startswith("v1_") else version
    return PROJECT_ROOT / "logs" / family / version

# V1.1 single-model baseline (from MEMORY.md feature table, prior to flag-batch round)
BASELINE_IC = 0.0674
BASELINE_SHIC = 0.0330

# Decision thresholds per /un Iterate-Until-Beats
SHIP_SHIC_DELTA = 0.005   # ShIC must exceed baseline by at least +0.005
SHIP_IC_TOLERANCE = -0.005  # IC may drop by at most 0.005 to still ship


@dataclass
class VariantSummary:
    run_tag: str
    log_path: str
    n_epochs: int = 0
    best_shic: float = float("nan")
    best_ic_h1: float = float("nan")
    final_ic_h1: float = float("nan")
    final_shic: float = float("nan")
    final_gap: float = float("nan")
    gate_pass_count: int = 0
    cycles: list[dict] = field(default_factory=list)


_RE_VAL = re.compile(
    r"Loss:\s*([\d.\-]+)\s*\|\s*Rec:\s*([\d.\-]+).*?IC1:([\d.\-]+).*?IC4:([\d.\-]+).*?IC16:([\d.\-]+).*?IC64:([\d.\-]+)"
)
_RE_SHIC = re.compile(r"ShIC:\s*([\d.\-]+)\s*Gap:\s*([\d.\-]+)")
_RE_NEW_BEST_SHIC = re.compile(r"\[NEW BEST SHUFFLED IC\]\s+([\d.\-]+)")
_RE_GATE_PASS = re.compile(r"\[GATE PASS\]")
_RE_EPOCH = re.compile(r"^\s*Ep\s+(\d+)")
_RE_CYCLE_LINE = re.compile(
    r"Epoch\s+(\d+)\s+\(C(\d+)\):\s*Contiguous=([\d.\-]+)\s+Shuffled=([\d.\-]+)\s+Gap=([\d.\-]+)"
)


def parse_log(path: Path) -> VariantSummary:
    summary = VariantSummary(run_tag=_run_tag_from_filename(path.name), log_path=str(path))
    text = path.read_text(errors="replace").splitlines()
    last_val = None
    last_shic = None
    last_gap = None
    for line in text:
        m_ep = _RE_EPOCH.match(line)
        if m_ep:
            summary.n_epochs = max(summary.n_epochs, int(m_ep.group(1)))
        m_val = _RE_VAL.search(line)
        if m_val:
            last_val = {
                "loss": float(m_val.group(1)),
                "rec": float(m_val.group(2)),
                "ic1": float(m_val.group(3)),
                "ic4": float(m_val.group(4)),
                "ic16": float(m_val.group(5)),
                "ic64": float(m_val.group(6)),
            }
            if (last_val["ic1"] >
                (summary.best_ic_h1 if not _isnan(summary.best_ic_h1) else -1)):
                summary.best_ic_h1 = last_val["ic1"]
            summary.final_ic_h1 = last_val["ic1"]
        m_shic = _RE_SHIC.search(line)
        if m_shic:
            last_shic = float(m_shic.group(1))
            last_gap = float(m_shic.group(2))
            summary.final_shic = last_shic
            summary.final_gap = last_gap
            if (last_shic >
                (summary.best_shic if not _isnan(summary.best_shic) else -1)):
                summary.best_shic = last_shic
        m_new = _RE_NEW_BEST_SHIC.search(line)
        if m_new:
            v = float(m_new.group(1))
            if v > (summary.best_shic if not _isnan(summary.best_shic) else -1):
                summary.best_shic = v
        if _RE_GATE_PASS.search(line):
            summary.gate_pass_count += 1
        m_cyc = _RE_CYCLE_LINE.search(line)
        if m_cyc:
            summary.cycles.append({
                "epoch": int(m_cyc.group(1)),
                "cycle": int(m_cyc.group(2)),
                "ic_contiguous": float(m_cyc.group(3)),
                "shic": float(m_cyc.group(4)),
                "gap": float(m_cyc.group(5)),
            })
    return summary


def _isnan(x: float) -> bool:
    return x != x


def _run_tag_from_filename(name: str, version: str = "v1_1") -> str:
    """Extract run-tag between feat-tag and `_train_`. Returns 'baseline' if absent."""
    pattern = re.compile(rf"{re.escape(version)}_f\d+(?:_revin)?(?:_abl)?(?:_(.+?))?_train_\d")
    m = pattern.match(name)
    if m and m.group(1):
        return m.group(1)
    return "baseline"


def decide(s: VariantSummary) -> str:
    if _isnan(s.best_shic) or _isnan(s.best_ic_h1):
        return "INCOMPLETE"
    shic_delta = s.best_shic - BASELINE_SHIC
    ic_delta = s.best_ic_h1 - BASELINE_IC
    if shic_delta >= SHIP_SHIC_DELTA and ic_delta >= SHIP_IC_TOLERANCE:
        return "SHIP"
    if shic_delta >= SHIP_SHIC_DELTA and ic_delta < SHIP_IC_TOLERANCE:
        return "MIXED (ShIC up, IC drop)"
    if shic_delta < 0:
        return "REGRESS"
    return "FLAT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, default=29,
                   help="Feature count to filter logs by (matches f<N> in filename).")
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help="Version to analyze (e.g. v1_0/v1_1/v1_4/v1_6/v3/v4/v6/v8/v11/v12/v13/v14). Default v1_1.")
    p.add_argument("--run-tags", nargs="+", default=None,
                   help="Specific run-tags to compare. Default: --auto-discover all matching logs.")
    p.add_argument("--auto", action="store_true",
                   help="Auto-discover all matching logs at the given feature count.")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "logs" / "frontier_ml" / "v_variant_compare"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    log_root = _log_root_for(args.version)
    feat_pattern = f"{args.version}_f{args.features}*_train_*.log"
    candidates = sorted(log_root.glob(feat_pattern))
    if not candidates:
        print(f"[compare] no logs found at {log_root}/{feat_pattern}", file=sys.stderr)
        sys.exit(2)

    if args.run_tags:
        wanted = set(args.run_tags)
        kept = [c for c in candidates if _run_tag_from_filename(c.name, args.version) in wanted]
    else:
        kept = candidates

    # Pick latest log per (run_tag) to handle reruns
    latest_by_tag: dict[str, Path] = {}
    for c in kept:
        tag = _run_tag_from_filename(c.name, args.version)
        if tag not in latest_by_tag or c.stat().st_mtime > latest_by_tag[tag].stat().st_mtime:
            latest_by_tag[tag] = c

    summaries = []
    for tag, path in sorted(latest_by_tag.items()):
        s = parse_log(path)
        summaries.append(s)

    # Side-by-side table
    print(f"{'Variant':<20} {'Epochs':>7} {'BestIC':>8} {'BestShIC':>9} "
          f"{'FinalIC':>8} {'FinalShIC':>10} {'FinalGap':>9} {'GatePass':>9} "
          f"{'dShIC':>8} {'Decision':<22}")
    print("-" * 130)
    for s in sorted(summaries, key=lambda x: (-x.best_shic if not _isnan(x.best_shic) else 0)):
        d = decide(s)
        d_shic = (s.best_shic - BASELINE_SHIC) if not _isnan(s.best_shic) else float("nan")
        print(f"{s.run_tag:<20} {s.n_epochs:>7} "
              f"{s.best_ic_h1:>8.4f} {s.best_shic:>9.4f} "
              f"{s.final_ic_h1:>8.4f} {s.final_shic:>10.4f} {s.final_gap:>9.4f} "
              f"{s.gate_pass_count:>9} {d_shic:>+8.4f} {d:<22}")

    print("-" * 130)
    print(f"Baseline reference: V1.1 single-model record IC={BASELINE_IC} ShIC={BASELINE_SHIC}")
    print(f"  (override per-version by editing BASELINE_IC/BASELINE_SHIC at top of compare_v1_variants.py)")
    print(f"Ship gate: ShIC delta >= +{SHIP_SHIC_DELTA}  AND  IC delta >= {SHIP_IC_TOLERANCE}")

    # JSON summary
    out_json = {
        "ts": ts,
        "version": args.version,
        "features": args.features,
        "log_root": str(log_root),
        "baseline": {"ic": BASELINE_IC, "shic": BASELINE_SHIC},
        "ship_gate": {"shic_delta_min": SHIP_SHIC_DELTA, "ic_tolerance": SHIP_IC_TOLERANCE},
        "variants": [
            {
                **s.__dict__,
                "decision": decide(s),
                "shic_delta": (s.best_shic - BASELINE_SHIC) if not _isnan(s.best_shic) else None,
                "ic_delta": (s.best_ic_h1 - BASELINE_IC) if not _isnan(s.best_ic_h1) else None,
            }
            for s in summaries
        ],
    }
    out_path = out_dir / f"compare_{args.version}_f{args.features}_{ts}.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"\n[compare] summary written to {out_path}")


if __name__ == "__main__":
    main()
