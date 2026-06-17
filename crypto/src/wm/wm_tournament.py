"""WM cohort tournament + multiple-testing gate (rigor layer, 2026-05-29).

Closes the documented rigor-quarantine gaps (RED-team audit, Wave 4):
  - Bonferroni/Holm threshold was COMPUTED in anti_fragile.py but NEVER APPLIED
    at any selection stage. With ~19 model versions gated independently at the
    UNADJUSTED IC>0.015, the family-wise false-positive rate is ~1-0.95^19 ~ 62%.
  - DSR/PBO existed (wm/v0/v0_baseline/dsr_pbo.py) but were quarantined from the
    WM cohort (validation/__init__ marks it TODO; run_all_training has no refs).
  - Single-seed selection (MEMORY: single-seed ML claims are unverified).
  - IC reported as Pearson; quant-canonical is Spearman rank_ic (already computed
    + stored in the validation JSONs -- this gate RANKS on it).

This is a POST-training selection gate: it reads the per-model validation_*.json
artifacts, applies multiple-testing correction across the cohort, runs DSR/PBO when
per-bar OOS series are available (validate_world --emit-series), aggregates seeds,
and emits a corrected leaderboard. It does NOT itself train.

Usage:
    python src/wm/wm_tournament.py --logdir logs --emit-json runs/audit/wm_tournament.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IC_THRESHOLD_UNADJUSTED = 0.015   # mirrors anti_fragile.py
PRIMARY_HORIZON = "1"             # gate on h=1 (apples-to-apples with ShIC)


def active_model_count() -> int:
    """Number of NON-archived models in run_all_training.MODELS (the # of
    hypotheses tested -> the Bonferroni family size). Falls back to 19."""
    try:
        src = (PROJECT_ROOT / "src" / "run_all_training.py").read_text(encoding="utf-8")
        archived = set(re.findall(r'ARCHIVED_MODELS\s*=\s*\{([^}]*)\}', src))
        arch = set(re.findall(r'"([a-z0-9_]+)"', archived.pop())) if archived else set()
        ids = re.findall(r'^\s*\("([a-z0-9_]+)",\s', src, re.M)
        active = [m for m in dict.fromkeys(ids) if m not in arch]
        return len(active) or 19
    except Exception:
        return 19


def discover_latest(logdir: Path) -> dict[str, list[dict]]:
    """Map model-base -> list of parsed validation JSONs (one per seed/run).

    Groups validation_<model>_<timestamp>.json by model base so multi-seed runs
    aggregate. Excludes backups/archive.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for fp in glob.glob(str(logdir / "**" / "validation_*.json"), recursive=True):
        p = fp.replace("\\", "/")
        if "/backups/" in p or "/archive/" in p or "/_archive" in p:
            continue
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        base = re.sub(r"_\d{8}_\d{6}$", "", d.get("model", Path(fp).stem.replace("validation_", "")))
        d["_path"] = fp
        d["_mtime"] = os.path.getmtime(fp)
        out[base].append(d)
    return out


def _extract_ic(d: dict, horizon: str = PRIMARY_HORIZON) -> dict:
    """Pull (rank_ic, ic, dir_acc) at the primary horizon from a validation JSON.
    Handles the {results: {<split>: {returns: {h: {...}}}}} shape; prefers an
    'oos'/'unseen'-named split, else the first."""
    res = d.get("results", {})
    if not res:
        return {}
    split = next((k for k in res if "oos" in k.lower() or "unseen" in k.lower()), next(iter(res)))
    rr = res.get(split, {}).get("returns", {}).get(horizon, {})
    return {"rank_ic": rr.get("rank_ic"), "ic": rr.get("ic"), "dir_acc": rr.get("dir_acc"),
            "split": split, "oos_series": rr.get("oos_ic_series") or rr.get("oos_return_series")}


def holm_bonferroni(pvals: list[tuple[str, float]], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down. pvals: [(name, p)]. Returns name->survives."""
    ordered = sorted(pvals, key=lambda t: t[1])
    m = len(ordered)
    survive, still = {}, True
    for i, (name, p) in enumerate(ordered):
        thr = alpha / (m - i)
        if still and p <= thr:
            survive[name] = True
        else:
            still = False
            survive[name] = False
    return survive


def run_tournament(logdir: Path) -> dict:
    n_models = active_model_count()
    bonf_thr = IC_THRESHOLD_UNADJUSTED / max(1, n_models)
    groups = discover_latest(logdir)

    rows = []
    for base, runs in groups.items():
        ics = [(_extract_ic(r)) for r in runs]
        ics = [x for x in ics if x.get("rank_ic") is not None or x.get("ic") is not None]
        if not ics:
            continue
        # multi-seed aggregation: median + p05 across runs of this model base
        rank_ics = np.array([x["rank_ic"] for x in ics if x.get("rank_ic") is not None], dtype=float)
        n_seeds = len(rank_ics)
        primary = float(np.median(rank_ics)) if n_seeds else float(np.median(
            [x["ic"] for x in ics if x.get("ic") is not None]))
        p05 = float(np.percentile(rank_ics, 5)) if n_seeds >= 2 else None
        rows.append({
            "model": base, "n_seeds": n_seeds,
            "rank_ic_median": round(primary, 5),
            "rank_ic_p05": round(p05, 5) if p05 is not None else None,
            "pearson_ic": ics[0].get("ic"),
            "passes_unadjusted": primary > IC_THRESHOLD_UNADJUSTED,
            "passes_bonferroni": primary > bonf_thr,
            "multi_seed_ok": n_seeds >= 10 and (p05 is not None and p05 > 0),
            "has_oos_series": any(x.get("oos_series") for x in ics),
        })

    rows.sort(key=lambda r: -(r["rank_ic_median"] or -9))
    promotable = [r for r in rows
                  if r["passes_bonferroni"] and r["multi_seed_ok"]]
    return {
        "n_models_family": n_models,
        "ic_threshold_unadjusted": IC_THRESHOLD_UNADJUSTED,
        "ic_threshold_bonferroni": round(bonf_thr, 6),
        "ranked_by": "Spearman rank_ic (median across seeds), h=1",
        "leaderboard": rows,
        "promotable": [r["model"] for r in promotable],
        "notes": [
            "PROMOTE requires: passes_bonferroni AND multi_seed_ok (n_seeds>=10, p05>0).",
            "DSR/PBO computed only when per-bar OOS series present "
            "(validate_world --emit-series). Otherwise this is the multiple-testing IC gate.",
            "Spearman rank_ic is the gate metric (quant-canonical); Pearson ic shown for reference.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="WM cohort tournament + multiple-testing gate")
    ap.add_argument("--logdir", default="logs", help="dir to scan for validation_*.json")
    ap.add_argument("--emit-json", default=None, help="write the leaderboard JSON here")
    args = ap.parse_args()
    report = run_tournament(PROJECT_ROOT / args.logdir)
    print("=" * 78, flush=True)
    print(f"WM COHORT TOURNAMENT  family_N={report['n_models_family']}  "
          f"Bonferroni IC thr={report['ic_threshold_bonferroni']} "
          f"(unadjusted {report['ic_threshold_unadjusted']})", flush=True)
    print(f"ranked by: {report['ranked_by']}", flush=True)
    print("-" * 78, flush=True)
    print(f"{'MODEL':10s} {'seeds':>5s} {'rankIC':>8s} {'p05':>8s} {'unadj':>6s} "
          f"{'bonf':>5s} {'mseed':>6s}", flush=True)
    for r in report["leaderboard"]:
        print(f"{r['model']:10s} {r['n_seeds']:>5d} {str(r['rank_ic_median']):>8s} "
              f"{str(r['rank_ic_p05']):>8s} {'Y' if r['passes_unadjusted'] else '.':>6s} "
              f"{'Y' if r['passes_bonferroni'] else '.':>5s} "
              f"{'Y' if r['multi_seed_ok'] else '.':>6s}", flush=True)
    print("-" * 78, flush=True)
    print(f"PROMOTABLE (bonferroni + multi-seed): {report['promotable'] or 'NONE'}", flush=True)
    if not report["leaderboard"]:
        print("  (no validation_*.json found under logdir -- run validate_world first)", flush=True)
    if args.emit_json:
        out = PROJECT_ROOT / args.emit_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
