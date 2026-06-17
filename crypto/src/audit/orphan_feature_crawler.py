"""orphan_feature_crawler.py -- detect chimera columns with ZERO sleeve consumer.

Crawler-style audit. Runs over the chimera schema + the src/strategy/sleeves/
tree and flags any feature family (by prefix) that:
  - has columns shipped in chimera (so the pipeline pays the compute cost)
  - has zero sleeve runner that reads it
  - OR has data-population < threshold (silent dead feature)

OUTPUT
------
runs/audit/orphan_features_<DATE>.md     -- triage doc per crawler convention

INVOKE
------
    python src/audit/orphan_feature_crawler.py
    python src/audit/orphan_feature_crawler.py --threshold-pct 0.30
"""
from __future__ import annotations

__contract__ = {
    "kind": "orphan_feature_crawler",
    "owner": "audit/strat-layer",
    "outputs": ["runs/audit/orphan_features_<DATE>.md"],
    "invariants": [
        "flags chimera column families with zero src/strategy/sleeves/ consumer",
        "flags features with population < threshold (silent dead)",
        "complements per_feature std check in pre_train_gate.py",
    ],
}

import argparse
import datetime as dt
import re
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHIMERA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "1d"
SLEEVES_DIR = PROJECT_ROOT / "src" / "strategy" / "sleeves"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefixes we EXPECT a sleeve to consume; the audit yells if none does.
TRACKED_PREFIXES = [
    "lob_", "mv_", "xex_", "bd_", "hurst_", "soc_",
    "te_", "hbr_", "wh_", "etf_", "stbl_", "liq_",
    "norm_kyle_", "norm_efficiency", "norm_perm_entropy",
    "norm_fd_close", "is_", "rv_bpv", "rv_rv_",
]

# Some prefixes are KNOWN-consumed via composite sleeves we don't have to flag
KNOWN_CONSUMED = {
    "etf_": "etf_flow_sleeve",
    "stbl_": "stablecoin_supply / stbl_v2 sleeves",
    "liq_": "liquidation_flow / reflexivity_cascade sleeves",
    "wh_": "whale_flow sleeve",
    "hbr_": "hawkes_branching (oracle Layer-3)",
    "te_": "xsec rankers (chimera feature consumer)",
    "lob_": "lob_microstructure_sleeve (Phase 2)",
    "mv_": "multi_venue_listing_sleeve (Phase 2)",
    "xex_": "cross_exchange_spread_sleeve (Phase 2)",
    "hurst_": "hurst_regime_sleeve (Phase 2)",
}


def find_sleeve_consumers(prefix: str) -> list[str]:
    """Return list of sleeve files that reference this prefix.

    Uses \\w* (zero-or-more) so prefixes that ARE complete column names
    (e.g. 'norm_efficiency') match as bare references too.
    """
    out = []
    pattern = re.compile(re.escape(prefix) + r"\w*")
    for fp in SLEEVES_DIR.glob("*_sleeve.py"):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if pattern.search(text):
            out.append(fp.name)
    return out


def measure_population(prefix: str, sample_assets: int = 3) -> dict:
    """Sample chimera files; return per-column mean non-null fraction."""
    fps = sorted(CHIMERA_DIR.glob("btcusdt_*.parquet"))
    if not fps:
        return {}
    fps_to_sample = fps[-1:]
    extras = []
    for sym in ("ethusdt", "solusdt"):
        f2 = sorted(CHIMERA_DIR.glob(f"{sym}_*.parquet"))
        if f2:
            extras.append(f2[-1])
    fps_to_sample.extend(extras[:sample_assets - 1])
    cols_with_data: dict[str, list[float]] = defaultdict(list)
    for fp in fps_to_sample:
        try:
            df = pl.read_parquet(fp).to_pandas()
        except Exception:
            continue
        n = len(df)
        for c in df.columns:
            if c.startswith(prefix):
                non_null = df[c].notna().sum()
                cols_with_data[c].append(float(non_null / n) if n else 0.0)
    return {c: float(sum(v) / len(v)) for c, v in cols_with_data.items()}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold-pct", type=float, default=0.30,
                     help="Population fraction below which feature is dead-flagged")
    args = ap.parse_args()

    findings = []
    for pfx in TRACKED_PREFIXES:
        consumers = find_sleeve_consumers(pfx)
        pop = measure_population(pfx)
        avg_pop = sum(pop.values()) / len(pop) if pop else 0.0
        if not pop:
            continue   # prefix not in chimera schema
        # Two failure modes:
        if not consumers:
            findings.append({
                "prefix": pfx, "category": "no-consumer",
                "consumers": [], "known": KNOWN_CONSUMED.get(pfx),
                "n_cols": len(pop), "avg_population": avg_pop,
                "fix": KNOWN_CONSUMED.get(pfx) or
                          f"Build a sleeve_runner that consumes {pfx}*",
            })
        elif avg_pop < args.threshold_pct:
            findings.append({
                "prefix": pfx, "category": "low-population",
                "consumers": consumers, "known": KNOWN_CONSUMED.get(pfx),
                "n_cols": len(pop), "avg_population": avg_pop,
                "fix": f"avg_pop={avg_pop:.1%} < {args.threshold_pct:.0%}; "
                          f"check pipeline producer for {pfx}",
            })

    # Write report
    today = dt.date.today().isoformat()
    out_path = OUT_DIR / f"orphan_features_{today}.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Orphan-Feature Crawler -- {today}\n\n")
        fh.write(f"Tracked prefixes: {len(TRACKED_PREFIXES)}\n")
        fh.write(f"Findings: {len(findings)}\n\n")
        no_con = [f for f in findings if f["category"] == "no-consumer"]
        low_pop = [f for f in findings if f["category"] == "low-population"]
        if no_con:
            fh.write(f"## No-Consumer ({len(no_con)})\n\n")
            for f in no_con:
                fh.write(f"- **{f['prefix']}*** ({f['n_cols']} cols, "
                          f"avg_pop {f['avg_population']:.1%})\n")
                fh.write(f"  - known: {f['known']}\n")
                fh.write(f"  - fix: {f['fix']}\n\n")
        if low_pop:
            fh.write(f"## Low-Population ({len(low_pop)})\n\n")
            for f in low_pop:
                fh.write(f"- **{f['prefix']}*** ({f['n_cols']} cols, "
                          f"avg_pop {f['avg_population']:.1%})\n")
                fh.write(f"  - consumers: {f['consumers']}\n")
                fh.write(f"  - fix: {f['fix']}\n\n")
        if not findings:
            fh.write("All tracked prefixes have consumers AND sufficient "
                      "population. No action needed.\n")

    print(f"[orphan-feature-crawler] {len(findings)} findings -> {out_path}")
    for f in findings:
        print(f"  [{f['category']}] {f['prefix']:<12s} avg_pop={f['avg_population']:.1%} "
              f"consumers={len(f['consumers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
