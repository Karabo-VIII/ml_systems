"""Render the full TI x TF DNA JSON into a markdown report — NO exclusions.

Robustness score (transparent, not a black box):
  + VAL edge > 0           (+1)
  + OOS edge > 0           (+1)
  + VAL null >= 95         (+1)
  + OOS null >= 95         (+1)
  + sign(VAL edge)==sign(OOS edge) AND both>0  (+1)  -> consistency
  + selective (deploy<70% in BOTH)             (+1)  -> the rare/valuable property
Max 6. This ranks ROBUST engines (consistent across both windows + beats null + selective)
above one-window flukes — without dropping anyone from the table.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

jpaths = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
if jpaths is None:
    jdir = (Path(__file__).resolve().parents[1].parent / "runs" / "engines" / "V5_dna_report")
    jpaths = sorted(jdir.glob("ti_dna_*.json"), key=lambda p: p.stat().st_mtime)[-2:]
rows = []
data = None
for jp in jpaths:
    data = json.load(open(jp))
    rows.extend(data["rows"])
jpath = jpaths[-1]


def robustness(r):
    v, o = r.get("VAL"), r.get("OOS2024")
    if not v or not o:
        return -1, {}
    ve, oe = v["edge_pp"], o["edge_pp"]
    vn, on = r["VAL_null_pct"], r["OOS_null_pct"]
    sel = v["deploy"] < 0.70 and o["deploy"] < 0.70
    s = 0
    s += ve > 0
    s += oe > 0
    s += vn >= 95
    s += on >= 95
    s += (ve > 0 and oe > 0)
    s += sel
    return s, {"ve": ve, "oe": oe, "vn": vn, "on": on, "sel": sel}


scored = []
for r in rows:
    s, d = robustness(r)
    scored.append((s, r, d))
scored.sort(key=lambda x: (-x[0], -(x[1].get("OOS2024") or {}).get("edge_pp", -9)))

lines = []
lines.append("# TI × TIMEFRAME — FULL PERFORMANCE DNA (no exclusions)\n")
lines.append(f"Source: `{jpath.name}` | VAL {data['windows']['VAL']} | "
             f"OOS-2024 {data['windows']['OOS2024']} (2025 sealed)\n")
lines.append(f"Cost {data['cost_rt']} RT maker | gate BTC-SMA100 | EW u100 | "
             f"null draws {data['n_null']} | {len(rows)} cells\n")
lines.append("Pipeline audited 19/19 (PIPELINE_AUDIT_RESULT.md): causal, walled, "
             "cost-correct, null verified.\n")
lines.append("\n**Robustness score (0-6):** VALedge>0 + OOSedge>0 + VALnull≥95 + OOSnull≥95 "
             "+ both-edges-positive(consistency) + selective(<70% both). Higher = more robust.\n")
lines.append("\n**Reads:** edge = linear %/wk vs always-long (honest). null = engine compound "
             "percentile vs random@same-deploy (≥95 = genuine timing). deploy = fraction of bars "
             "long (low = selective). Sharpe annualized. Compound shown for context (OOS-2024 "
             "rebalancing premium ≈1.1x per audit, so ~honest; VAL multi-month is mildly inflated).\n")

# Main table
lines.append("\n## Full DNA table (ranked by robustness, then OOS edge)\n")
lines.append("| Rk | Robust | Family | TF | VAL edge/wk | VAL null | VAL dep | VAL Shrp | "
             "OOS edge/wk | OOS null | OOS dep | OOS Shrp | VAL comp | OOS comp | profile |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for rank, (s, r, d) in enumerate(scored, 1):
    v, o = r.get("VAL"), r.get("OOS2024")
    if not v or not o:
        lines.append(f"| {rank} | n/a | {r['family']} | {r['tf']} | INSUFFICIENT | | | | | | | | | | |")
        continue
    # profile tag
    consistent = d["ve"] > 0 and d["oe"] > 0
    if d["sel"] and consistent and d["on"] >= 95:
        prof = "**SELECTIVE+ROBUST**"
    elif consistent and d["vn"] >= 95 and d["on"] >= 95:
        prof = "robust tilt (non-sel)"
    elif consistent:
        prof = "consistent (weak null)"
    elif d["oe"] > 0 and d["on"] >= 95:
        prof = "OOS-only (VAL≤0)"
    elif d["ve"] > 0 and d["vn"] >= 95:
        prof = "VAL-only (OOS≤0)"
    elif o["deploy"] < 1e-3:
        prof = "abstain (2h synth)"
    else:
        prof = "null/bleed"
    lines.append(
        f"| {rank} | {s} | {r['family']} | {r['tf']} | {v['edge_pp']:+.3f} | {r['VAL_null_pct']:.0f} | "
        f"{v['deploy']:.0%} | {v['sharpe']:.2f} | {o['edge_pp']:+.3f} | {r['OOS_null_pct']:.0f} | "
        f"{o['deploy']:.0%} | {o['sharpe']:.2f} | {v['compound']:+.0f}% | {o['compound']:+.0f}% | {prof} |")

# Per-TF best summary
lines.append("\n## Best per timeframe (by OOS edge, among consistent cells)\n")
for tf in ["1d", "4h", "2h"]:
    tf_rows = [(s, r, d) for s, r, d in scored if r["tf"] == tf and r.get("OOS2024")
               and d.get("ve", -9) > 0 and d.get("oe", -9) > 0]
    tf_rows.sort(key=lambda x: -x[1]["OOS2024"]["edge_pp"])
    lines.append(f"\n**{tf}** (consistent cells, top 6):")
    if not tf_rows:
        lines.append("  none consistent across both windows.")
    for s, r, d in tf_rows[:6]:
        o = r["OOS2024"]
        lines.append(f"  - {r['family']:<10} OOS {o['edge_pp']:+.3f}pp/wk, null {r['OOS_null_pct']:.0f}, "
                     f"deploy {o['deploy']:.0%}, Sharpe {o['sharpe']:.2f}")

# Honest summary stats
n_consistent = sum(1 for s, r, d in scored if d.get("ve", -9) > 0 and d.get("oe", -9) > 0)
n_selective = sum(1 for s, r, d in scored if d.get("sel"))
n_robust_tilt = sum(1 for s, r, d in scored
                    if d.get("ve", -9) > 0 and d.get("oe", -9) > 0 and d.get("on", 0) >= 95)
lines.append("\n## Honest summary\n")
lines.append(f"- {len(rows)} cells total. **{n_consistent}** are DEV/OOS-consistent (edge>0 both windows).")
lines.append(f"- **{n_robust_tilt}** are consistent AND beat OOS random-null ≥95th pct (genuine timing).")
lines.append(f"- **{n_selective}** are selective (deploy<70% both windows) — these are the rare ones.")
lines.append("- Most TI×TF cells deploy 95-100% (always-long-with-tilt under the bull gate): they "
             "refine WHICH longs, not WHETHER long. Consistent with the standing finding that the "
             "regime gate is the dominant value lever (D78 CASH THEOREM).")
lines.append("- VAL→OOS sign flips are common at sub-daily TFs (2h especially) — the noise signature. "
             "1d and 4h carry the most VAL/OOS-consistent cells.")
lines.append("- Volume families (OBV/VWAP/CMF/MFI/FLOW) abstain at 2h (chimera volume features have no "
             "native-2h baseline → NaN, correctly treated as unavailable, not faked).")

out = jpath.with_name("DNA_REPORT.md")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
print(f"\nTop 12 by robustness:")
for s, r, d in scored[:12]:
    o = r.get("OOS2024"); v = r.get("VAL")
    if o and v:
        print(f"  [{s}] {r['family']:<10}|{r['tf']:<3} VAL {v['edge_pp']:+.2f} OOS {o['edge_pp']:+.2f} "
              f"dep {o['deploy']:.0%} null V{r['VAL_null_pct']:.0f}/O{r['OOS_null_pct']:.0f}")
