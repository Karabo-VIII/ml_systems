"""Analyze 15m mining output: setup histogram + capture_rate table + signature aggregation."""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/mover_day/pepe_train_mover_day_15m_2026_05_27.json"

with open(DATA) as f:
    d = json.load(f)

per_day = d["phase2_per_day"]
ok = [r for r in per_day if r["status"] == "OK"]
print(f"Mover days OK: {len(ok)}/{len(per_day)}")

# Setup histogram
setup_counter = Counter()
for r in ok:
    s = r["best_setup"]
    setup_counter[(s["kind"], s["fast"], s["slow"])] += 1

print("\n=== Setup histogram (top 20) ===")
for (kind, fast, slow), n in setup_counter.most_common(20):
    print(f"  {kind:3s} {fast:3d}/{slow:3d}: {n:3d} days ({100.0*n/len(ok):.1f}%)")

# Capture rate stats
caps = [r["capture_rate"] for r in ok]
cap_pcts = [r["captured_move_pct"] for r in ok]
avail_pcts = [r["available_move_pct"] for r in ok]
print(f"\n=== Capture rate stats ===")
print(f"  median capture_rate: {np.median(caps):.3f}")
print(f"  mean capture_rate:   {np.mean(caps):.3f}")
print(f"  p25 / p75:           {np.quantile(caps, 0.25):.3f} / {np.quantile(caps, 0.75):.3f}")
print(f"  median captured_pct: {np.median(cap_pcts):.4f}")
print(f"  median available_pct: {np.median(avail_pcts):.4f}")

# Entry offset
offsets_h = [r["entry_offset_hours"] for r in ok]
offsets_b = [r["entry_offset_bars"] for r in ok]
print(f"\n=== Entry offset (hours before daily close) ===")
print(f"  median: {np.median(offsets_h):.2f}h  ({np.median(offsets_b):.0f} 15m bars)")
print(f"  mean:   {np.mean(offsets_h):.2f}h")
print(f"  p25/p75: {np.quantile(offsets_h, 0.25):.2f}h / {np.quantile(offsets_h, 0.75):.2f}h")
print(f"  max:    {max(offsets_h):.2f}h, min: {min(offsets_h):.2f}h")

# Top 30 by captured_pct
print("\n=== Top 30 days by captured_pct ===")
sorted_top = sorted(ok, key=lambda r: -r["captured_move_pct"])[:30]
print(f"  {'date':12s} {'daily%':>8s} {'best_setup':>14s} {'off_h':>7s} {'cap%':>8s} {'avail%':>8s} {'cap_rate':>9s}")
for r in sorted_top:
    s = r["best_setup"]
    setup_str = f"{s['kind']} {s['fast']}/{s['slow']}"
    print(
        f"  {r['date']:12s} {100*r['daily_return']:>7.2f}% {setup_str:>14s} "
        f"{r['entry_offset_hours']:>6.2f}h {100*r['captured_move_pct']:>7.2f}% "
        f"{100*r['available_move_pct']:>7.2f}% {100*r['capture_rate']:>7.2f}%"
    )

# Bottom 5
print("\n=== Bottom 5 days by captured_pct ===")
sorted_bot = sorted(ok, key=lambda r: r["captured_move_pct"])[:5]
for r in sorted_bot:
    s = r["best_setup"]
    setup_str = f"{s['kind']} {s['fast']}/{s['slow']}"
    print(
        f"  {r['date']:12s} {100*r['daily_return']:>7.2f}% {setup_str:>14s} "
        f"{r['entry_offset_hours']:>6.2f}h {100*r['captured_move_pct']:>7.2f}% "
        f"{100*r['available_move_pct']:>7.2f}% {100*r['capture_rate']:>7.2f}%"
    )

# Correlations
print("\n=== Feature-vs-capture_rate correlations (elevated bool) ===")
corr = d["phase4_signature"]["correlation_elevated_vs_capture_rate"]
sorted_corr = sorted(corr.items(), key=lambda kv: -abs(kv[1]) if kv[1] is not None else 0)
for f, c in sorted_corr:
    if c is not None:
        print(f"  {f:32s}: {c:+.4f}")
    else:
        print(f"  {f:32s}: N/A")

print("\n=== Feature-vs-capture_rate correlations (z_12h) ===")
corr_z = d["phase4_signature"]["correlation_z12h_vs_capture_rate"]
sorted_corr_z = sorted(corr_z.items(), key=lambda kv: -abs(kv[1]) if kv[1] is not None else 0)
for f, c in sorted_corr_z:
    if c is not None:
        print(f"  {f:32s}: {c:+.4f}")
    else:
        print(f"  {f:32s}: N/A")

print("\n=== Feature-vs-capture_rate correlations (delta_12h) ===")
corr_d = d["phase4_signature"]["correlation_delta12h_vs_capture_rate"]
sorted_corr_d = sorted(corr_d.items(), key=lambda kv: -abs(kv[1]) if kv[1] is not None else 0)
for f, c in sorted_corr_d:
    if c is not None:
        print(f"  {f:32s}: {c:+.4f}")
    else:
        print(f"  {f:32s}: N/A")

# Test top-setup × top-exit combo
print("\n=== Most-recurring setup detailed PnL by exit ===")
top_setup = setup_counter.most_common(1)[0][0]
top_kind, top_fast, top_slow = top_setup
print(f"Top setup: {top_kind} {top_fast}/{top_slow} ({setup_counter[top_setup]} days)")

per_day_exits = d["phase3_exits"]["per_day"]
top_days_indices = [
    i for i, r in enumerate(per_day)
    if r["status"] == "OK" and r["best_setup"] is not None
    and r["best_setup"]["kind"] == top_kind
    and r["best_setup"]["fast"] == top_fast
    and r["best_setup"]["slow"] == top_slow
]
print(f"  Days with this top setup: {len(top_days_indices)}")
for exit_name in ["E1_opp_cross", "E2_6h", "E3_12h", "E4_24h", "E5_48h", "E6_mfe_trail50"]:
    vals = [per_day_exits[exit_name][i] for i in top_days_indices if per_day_exits[exit_name][i] is not None]
    if vals:
        arr = np.array(vals)
        print(f"  {exit_name:18s}: n={len(arr):3d} med={np.median(arr):+.4f} mean={np.mean(arr):+.4f} win={(arr>0).mean():.2%}")

print("\n=== Daily return distribution (TRAIN) ===")
p1 = d["phase1"]
print(f"  Total TRAIN days: {p1['n_train_days']}")
print(f"  >=2% movers: {p1['n_movers_2pct']} ({100*p1['n_movers_2pct']/p1['n_train_days']:.1f}%)")
print(f"  >=5% movers: {p1['n_movers_5pct']} ({100*p1['n_movers_5pct']/p1['n_train_days']:.1f}%)")
print(f"  >=10% movers: {p1['n_movers_10pct']} ({100*p1['n_movers_10pct']/p1['n_train_days']:.1f}%)")
