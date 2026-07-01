"""Data inspector — the 'what do I have?' coverage report across all layers.

Scans raw ticks, silver bars/features, cross-pair panel, gold dataset, and the
rates panel for a universe, and prints a coverage table + writes a JSON. This is
the fast situational-awareness tool (complements data_quality.py, which is the
correctness gate).

    python src/pipeline/inspect_data.py --universe u7
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout

__contract__ = {
    "kind": "pipeline_validator",
    "stage": "inspect_data",
    "inputs": {"args": ["--pairs", "--universe", "--cadences"]},
    "outputs": {"report": "runs/coverage/coverage_<ts>.json + stdout table"},
    "invariants": {"cheap_metadata_row_counts": True, "read_only": True},
    "rationale": "Fast coverage/situational-awareness across every pipeline layer.",
}


def _rows(path) -> int:
    if path is None or not Path(path).exists():
        return 0
    try:
        return pl.scan_parquet(str(path)).select(pl.len()).collect().item()
    except Exception:
        return -1


def inspect_pair(pair: str, cadences: list[str]) -> dict:
    tick_days = layout.list_tick_days(pair)
    rec: dict = {"pair": pair, "tick_days": len(tick_days)}
    if tick_days:
        rec["tick_span"] = f"{tick_days[0]}..{tick_days[-1]}"
    for cad in cadences:
        rec[f"bars_{cad}"] = _rows(layout.bars_latest(pair, cad))
        rec[f"feat_{cad}"] = _rows(layout.features_latest(pair, cad))
        rec[f"data_{cad}"] = _rows(layout.dataset_latest(pair, cad))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Forex data coverage inspector")
    cli.add_standard_args(ap, default_workers=1, date_window=False)
    ap.add_argument("--cadences", nargs="+", default=["1h", "1d"])
    args = ap.parse_args()
    pairs = cli.resolve_pairs(args, stage_name="inspect")
    cads = args.cadences

    header = f"{'pair':<8} {'tickdays':>8} {'span':<24}"
    for cad in cads:
        header += f" {'bar_' + cad:>9} {'feat_' + cad:>9} {'data_' + cad:>9}"
    print(header)
    print("-" * len(header))
    recs = []
    for p in pairs:
        r = inspect_pair(p, cads)
        recs.append(r)
        line = f"{p:<8} {r['tick_days']:>8} {r.get('tick_span', '-'):<24}"
        for cad in cads:
            line += f" {r[f'bars_{cad}']:>9} {r[f'feat_{cad}']:>9} {r[f'data_{cad}']:>9}"
        print(line)

    # cross-pair panel + rates
    for cad in cads:
        pn = layout.panel_latest("panel", cad)
        print(f"panel {cad}: {_rows(pn)} rows" + (f" ({Path(pn).name})" if pn else " (none)"))
    rates = sorted((layout.PROCESSED / "rates").glob("rates_daily_*.parquet"))
    print(f"rates panel: {_rows(rates[-1]) if rates else 0} rows"
          + (f" ({rates[-1].name})" if rates else " (none)"))

    out_dir = layout.PROJECT_ROOT / "runs" / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out_dir / f"coverage_{stamp}.json").write_text(
        json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), "pairs": recs},
                   indent=2), encoding="utf-8")
    print(f"\n[inspect] coverage -> runs/coverage/coverage_{stamp}.json")


if __name__ == "__main__":
    main()
