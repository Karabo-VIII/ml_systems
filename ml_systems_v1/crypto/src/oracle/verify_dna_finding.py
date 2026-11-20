"""Reproducible verifier for the 2026-06-08 Oracle-Decomposer DNA finding.

Closes the provenance gap flagged by the oracle_falsify red-team audit (finding H1): the headline
"top feature raw AUC 0.606 -> 0.533 date-demeaned" number previously lived ONLY in frontier.json
prose with no checked-in script. This script RE-DERIVES the entire DNA finding from the committed
DNA panel (`runs/oracle/dna_panel.parquet`) in one command, plus the data-level causality checks,
so the finding is bit-reproducible and self-auditing.

The finding (docs/ORACLE_DNA_FINDINGS_2026_06_08.md): NO entry-day chimera feature predicts whether
the MA driver captures a top-performer's move well at the ASSET level. The one raw "winner"
(xd_btc_volatility) collapses toward 0.5 when date-demeaned -> it is a per-DATE/cohort REGIME effect,
not a per-asset signal.

Run:  python src/oracle/verify_dna_finding.py
Exit: 0 if the finding reproduces within tolerance AND causality holds; 2 otherwise.
No emoji (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "runs" / "oracle" / "dna_panel.parquet"

# Expected values from the committed finding (tolerances are generous; panel re-gens drift ~0.01).
EXPECT = {
    "raw_auc_top": (0.606, 0.04),       # raw AUC of the strongest ctx_entry feature
    "demean_auc_top": (0.533, 0.04),    # collapses toward 0.5 after date-demean
    "n_above_noise": 1,                 # count of ctx_entry features with |AUC-0.5| > 0.10
}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average-rank tie handling. labels in {0,1}."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sr = scores[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    npos = int(pos.sum())
    nneg = int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main() -> int:
    if not PANEL.exists():
        print(f"FAIL: DNA panel not found at {PANEL}")
        return 2
    df = pl.read_parquet(PANEL)
    cr = df["capture_rate"].to_numpy().astype(float)
    good = (cr >= 0.286).astype(int)  # GOOD = top tercile of capture_rate

    # --- data-level causality (the keystone: no look-ahead in the oracle's entry selection) ---
    caus = df.select([
        (pl.col("entry_date") <= pl.col("query_date")).all().alias("entry_le_query"),
        (pl.col("days_back") >= 0).all().alias("days_back_nonneg"),
        (pl.col("days_back") <= pl.col("max_days_back")).all().alias("within_bound"),
        (pl.col("peak_date") >= pl.col("entry_date")).all().alias("peak_after_entry"),
    ]).row(0)
    causality_ok = all(caus)

    # --- the 47-feature noise floor + the top feature's raw/demeaned AUC ---
    ctx = [c for c in df.columns if c.startswith("ctx_entry__")]
    qd = df["query_date"].to_numpy()
    devs = []
    for c in ctx:
        x = df[c].to_numpy().astype(float)
        m = ~np.isnan(x)
        if m.sum() < 20:
            continue
        a = _auc(x[m], good[m])
        if not np.isnan(a):
            devs.append((abs(a - 0.5), c, a))
    devs.sort(reverse=True)
    top_dev, top_feat, top_auc = devs[0]

    # date-demean the top feature: subtract per-query_date mean, recompute AUC
    x = df[top_feat].to_numpy().astype(float)
    m = ~np.isnan(x)
    xd = x.copy()
    for d in np.unique(qd[m]):
        sel = (qd == d) & m
        xd[sel] = x[sel] - np.nanmean(x[sel])
    demean_auc = _auc(xd[m], good[m])

    n_above = sum(1 for d, _, _ in devs if d > 0.10)
    median_dev = float(np.median([d for d, _, _ in devs]))
    zero_frac = float((cr == 0.0).mean())

    print("=== DNA FINDING VERIFIER (runs/oracle/dna_panel.parquet) ===")
    print(f"rows={df.height}  ctx_entry features scanned={len(devs)}")
    print(f"causality (entry<=query, days_back in [0,bound], peak>=entry): "
          f"{'PASS' if causality_ok else 'FAIL'}  {dict(zip(['entry_le_query','days_back_nonneg','within_bound','peak_after_entry'], caus))}")
    print(f"top feature              : {top_feat}")
    print(f"  raw AUC                : {top_auc:.4f}  (expect ~{EXPECT['raw_auc_top'][0]})")
    print(f"  date-demeaned AUC      : {demean_auc:.4f}  (expect ~{EXPECT['demean_auc_top'][0]}; collapse toward 0.5 = regime not asset)")
    print(f"  |AUC-0.5| > 0.10 count : {n_above}  (expect {EXPECT['n_above_noise']} = chance over 47 tests)")
    print(f"  median |AUC-0.5|       : {median_dev:.4f}  (noise floor)")
    print(f"capture_rate zero-inflation (label note H3): {zero_frac:.3f} exactly 0.0")

    ok = (
        causality_ok
        and abs(top_auc - EXPECT["raw_auc_top"][0]) <= EXPECT["raw_auc_top"][1]
        and abs(demean_auc - EXPECT["demean_auc_top"][0]) <= EXPECT["demean_auc_top"][1]
        and n_above == EXPECT["n_above_noise"]
    )
    print(f"\nRESULT: {'REPRODUCED + CAUSAL (null is honest)' if ok else 'DEVIATION -- investigate'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
