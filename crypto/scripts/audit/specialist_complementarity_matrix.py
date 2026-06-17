"""specialist_complementarity_matrix.py -- are our specialists actually complementary?

The "engine of specialists" thesis says we capture more by stacking specialists
with disjoint permission zones. That thesis is FALSIFIABLE: if specialists fire
on the same (asset, day) cells and have high PnL correlation, the union adds
nothing.

This tool quantifies the thesis using ACTUAL v3 paper-trade-replay 8Q sidecars
(2024-Q1 to 2025-Q4, ~730 days), not estimates.

OUTPUTS (per pair, per quarter, and union):
  - PnL correlation (Pearson)
  - Fire-day Jaccard (Jaccard on dates where new_entries > 0)
  - Win-day complementarity (fraction of days where A wins big & B is flat/neg)
  - Naive union NAV (equal-weight average daily) -- no cap, no asset dedup
  - Per-day union pass-rate at 7d floor (+5.25%)

The UNION NAV is an approximation -- v3 enforces per-blend isolation, so a
realistic union would need a position-cap simulator. The naive number is an
UPPER BOUND for what equal-weight combination could deliver. RELATIVE rank
between unions IS valid; absolute should be confirmed with a portfolio sim.

INPUTS:
  logs/strat_audit/paper_trade_replay_v3_<BLEND>_u<N>_<YYYYMMDD_YYYYMMDD>.json
  Per-day records contain: date, day_pnl_pct, new_entries, btc_30d
"""
from __future__ import annotations

import json
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "strat_audit"
OUT_MD = ROOT / "runs" / "audit" / "SPECIALIST_COMPLEMENTARITY_2026_05_20.md"
OUT_JSON = ROOT / "runs" / "audit" / "specialist_complementarity_2026_05_20.json"

# 8Q quarter windows (matches canonical state's 24Q1-25Q4)
QUARTERS = [
    ("2024Q1", "20240101", "20240331"),
    ("2024Q2", "20240401", "20240630"),
    ("2024Q3", "20240701", "20240930"),
    ("2024Q4", "20241001", "20241231"),
    ("2025Q1", "20250101", "20250331"),
    ("2025Q2", "20250401", "20250630"),
    ("2025Q3", "20250701", "20250930"),
    ("2025Q4", "20251001", "20251231"),
]

# Specialists with full 8Q v3 coverage
SPECIALISTS = [
    ("STRICT",      "REGIME_ROUTER_STRICT_LO_SETUP60", "u100"),
    ("MOVER_LO",    "MOVER_CONTINUATION_LO",            "u100"),
    ("TA_SML_MOE",  "TA_SML_MOE",                       "u50"),
    ("TA_SML_SOLO", "TA_SML_SOLO",                      "u50"),
]

# User-mandated gate (2026-05-19)
GATE_7D_PCT = 5.25


def load_specialist_per_day(blend: str, universe: str) -> pd.DataFrame:
    """Concat per_day records across 8 quarters for one blend."""
    rows = []
    for label, ws, we in QUARTERS:
        # Each quarter may have multiple matches (different start dates);
        # take the latest by mtime.
        candidates = sorted(
            LOGS.glob(f"paper_trade_replay_v3_{blend}_{universe}_*_{we}.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print(f"  [WARN] {blend} {universe} {label}: no sidecar matching *_{we}.json")
            continue
        p = candidates[-1]
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] {blend} {label}: parse fail {e}")
            continue
        for d in j.get("per_day", []):
            rows.append({
                "date": d["date"],
                "quarter": label,
                "pnl_pct": float(d.get("day_pnl_pct", 0.0)),
                "new_entries": int(d.get("new_entries", 0)),
                "open_book": int(d.get("open_book_after", 0)),
                "closed_today": int(d.get("closed_today", 0)),
                "btc_30d": float(d.get("btc_30d", 0.0)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def regime(btc_30d: float) -> str:
    if btc_30d <= -0.15: return "crash"
    if btc_30d <= -0.05: return "bear"
    if btc_30d >= 0.05:  return "bull"
    return "chop"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 78)
    print("SPECIALIST COMPLEMENTARITY MATRIX (v3 8Q sidecars)")
    print("=" * 78)

    series: dict[str, pd.DataFrame] = {}
    for label, blend, universe in SPECIALISTS:
        print(f"\nLoading {label} ({blend} on {universe})...")
        df = load_specialist_per_day(blend, universe)
        if df.empty:
            print(f"  [SKIP] no data")
            continue
        n_fire = (df["new_entries"] > 0).sum()
        print(f"  {len(df)} days; {n_fire} fire days ({n_fire/len(df)*100:.1f}%); "
              f"sum PnL {df['pnl_pct'].sum():+.2f}%")
        series[label] = df

    if len(series) < 2:
        print("[FATAL] need at least 2 specialists with data")
        return 2

    # Build merged daily panel on common dates
    common = None
    for label, df in series.items():
        d = df[["date", "pnl_pct", "new_entries"]].rename(
            columns={"pnl_pct": f"pnl_{label}", "new_entries": f"fire_{label}"})
        common = d if common is None else common.merge(d, on="date", how="outer")
    common = common.sort_values("date").reset_index(drop=True)
    # Fill missing with 0 (specialist had no data that day)
    for c in common.columns:
        if c != "date":
            common[c] = common[c].fillna(0)
    # Add regime via STRICT's btc_30d (most reliable u100 reference)
    if "STRICT" in series:
        regime_df = series["STRICT"][["date", "btc_30d"]].copy()
        common = common.merge(regime_df, on="date", how="left")
        common["regime"] = common["btc_30d"].apply(regime)
    else:
        common["regime"] = "unknown"

    labels = list(series.keys())
    print(f"\nCommon panel: {len(common)} days × {len(labels)} specialists ({labels})")

    # ----- 1. PNL CORRELATION MATRIX -----
    print("\n" + "=" * 78)
    print("1. PNL CORRELATION (Pearson, full 8Q)")
    print("=" * 78)
    pnl_cols = [f"pnl_{l}" for l in labels]
    corr_pnl = common[pnl_cols].corr()
    print(f"\n  {'':<14}" + "".join(f"{l:>14}" for l in labels))
    for i, l1 in enumerate(labels):
        print(f"  {l1:<14}" + "".join(
            f"{corr_pnl.loc[f'pnl_{l1}', f'pnl_{l2}']:>+13.3f} " for l2 in labels))

    # ----- 2. FIRE-DAY JACCARD -----
    print("\n" + "=" * 78)
    print("2. FIRE-DAY JACCARD (days where new_entries > 0)")
    print("=" * 78)
    fire_masks = {l: (common[f"fire_{l}"] > 0).astype(bool) for l in labels}
    jaccard = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i == j: continue
            a = fire_masks[l1]; b = fire_masks[l2]
            inter = (a & b).sum(); uni = (a | b).sum()
            jaccard.loc[l1, l2] = inter / uni if uni > 0 else 0
    print(f"\n  {'':<14}" + "".join(f"{l:>14}" for l in labels))
    for l1 in labels:
        print(f"  {l1:<14}" + "".join(
            f"{jaccard.loc[l1, l2]:>+13.3f} " for l2 in labels))
    print("\n  (1.0 = perfectly overlapping fire-days; 0 = disjoint)")

    # ----- 3. WIN-DAY COMPLEMENTARITY -----
    # For each pair: fraction of days where A wins big (PnL > +1%) AND B is flat/neg (PnL < +0.2%)
    print("\n" + "=" * 78)
    print("3. WIN-DAY COMPLEMENTARITY (A wins big & B flat/neg)")
    print("=" * 78)
    print("Read row→col: 'when ROW wins >+1%, what fraction of those days is COL flat/neg (<+0.2%)?'")
    print("Higher = MORE complementary (ROW catches days COL misses)")
    win_comp = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    for l1 in labels:
        for l2 in labels:
            if l1 == l2: continue
            a_wins = common[f"pnl_{l1}"] > 1.0
            b_flat = common[f"pnl_{l2}"] < 0.2
            if a_wins.sum() == 0:
                win_comp.loc[l1, l2] = float("nan")
            else:
                win_comp.loc[l1, l2] = (a_wins & b_flat).sum() / a_wins.sum()
    print(f"\n  {'':<14}" + "".join(f"{l:>14}" for l in labels))
    for l1 in labels:
        cells = []
        for l2 in labels:
            v = win_comp.loc[l1, l2]
            cells.append(f"{v:>+13.3f}" if not pd.isna(v) else f"{'—':>13}")
        print(f"  {l1:<14}" + " ".join(cells))

    # ----- 4. NAIVE UNION SIMULATION -----
    # Equal-weight daily average across specialists (assumes 1/N NAV per specialist)
    # then compound. This is an UPPER BOUND -- real union needs per-bucket cap sim.
    print("\n" + "=" * 78)
    print("4. NAIVE UNION (equal-weight daily avg, NO cap, NO dedup) — UPPER BOUND")
    print("=" * 78)
    common["union_pnl_naive"] = common[pnl_cols].mean(axis=1)
    union_comp = ((1 + common["union_pnl_naive"]/100).prod() - 1) * 100
    n_days = len(common)
    union_mean_d = common["union_pnl_naive"].mean()
    union_ann = ((1 + union_mean_d/100)**365 - 1) * 100
    common["pnl_7d_union"] = common["union_pnl_naive"].rolling(7).sum()
    union_7d_pass = (common["pnl_7d_union"] >= GATE_7D_PCT).mean() * 100

    print(f"\n  Naive union (n={n_days} days):")
    print(f"    daily mean      : {union_mean_d:+.4f}%")
    print(f"    compound 8Q     : {union_comp:+.2f}%")
    print(f"    annualized      : {union_ann:+.2f}%")
    print(f"    7d floor pass % : {union_7d_pass:.1f}%  (gate +{GATE_7D_PCT:.2f}%)")

    # Individual baselines for comparison
    print("\n  Per-specialist (same horizon):")
    print(f"  {'spec':<14}{'daily mean':>12}{'comp 8Q':>11}{'annualized':>12}{'7d pass%':>11}")
    for l in labels:
        s = common[f"pnl_{l}"]
        d_mean = s.mean()
        comp = ((1 + s/100).prod() - 1) * 100
        ann = ((1 + d_mean/100)**365 - 1) * 100
        roll7 = s.rolling(7).sum()
        p7 = (roll7 >= GATE_7D_PCT).mean() * 100
        print(f"  {l:<14}{d_mean:>+11.4f}%{comp:>+10.2f}%{ann:>+11.2f}%{p7:>10.1f}%")

    # ----- 5. PER-REGIME COMPLEMENTARITY -----
    print("\n" + "=" * 78)
    print("5. PER-REGIME PERFORMANCE (mean daily PnL)")
    print("=" * 78)
    print(f"  {'spec':<14}" + "".join(f"{r:>11}" for r in ("crash", "bear", "chop", "bull")))
    for l in labels:
        row = []
        for r in ("crash", "bear", "chop", "bull"):
            mask = common["regime"] == r
            n = mask.sum()
            if n == 0:
                row.append(f"{'n/a':>10} ")
                continue
            m = common.loc[mask, f"pnl_{l}"].mean()
            row.append(f"{m:>+9.3f}%")
        print(f"  {l:<14}" + " ".join(row))

    # ----- 6. HONEST INTERPRETATION -----
    interp = []
    interp.append("\n" + "=" * 78)
    interp.append("6. HONEST INTERPRETATION")
    interp.append("=" * 78)
    # Find highest-correlation pair
    upper_tri = []
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i < j:
                upper_tri.append((l1, l2, corr_pnl.loc[f"pnl_{l1}", f"pnl_{l2}"]))
    upper_tri.sort(key=lambda x: -x[2])
    interp.append(f"\nHighest-correlation pair: {upper_tri[0][0]} <-> {upper_tri[0][1]} "
                  f"(r = {upper_tri[0][2]:+.3f})")
    interp.append(f"Lowest-correlation pair:  {upper_tri[-1][0]} <-> {upper_tri[-1][1]} "
                  f"(r = {upper_tri[-1][2]:+.3f})")
    interp.append("")
    if upper_tri[-1][2] < 0.2:
        interp.append(f"GOOD: at least one pair (corr < 0.2) -- these specialists ARE")
        interp.append(f"firing on different cells. Stacking adds signal.")
    elif upper_tri[-1][2] < 0.4:
        interp.append(f"MIXED: lowest corr is {upper_tri[-1][2]:.2f}. Some complementarity")
        interp.append(f"but not maximally diverse.")
    else:
        interp.append(f"WARN: even the lowest-correlation pair is {upper_tri[-1][2]:.2f}.")
        interp.append(f"Specialists are largely firing on the same days; union adds little.")
    interp.append("")
    interp.append(f"Union 7d-floor pass-rate: {union_7d_pass:.1f}% vs target 50%+.")
    if union_7d_pass < 25:
        interp.append("Naive equal-weight union does NOT bridge to the 7d floor. The gap")
        interp.append("is hit-rate, not coverage. See realistic_ceiling.py for the ceiling")
        interp.append("at varying hit rates -- closing the gap to +1%/d needs hit rate 60%+,")
        interp.append("not more specialists at 50-55% hit each.")
    else:
        interp.append("Naive union approaches the 7d floor. With per-bucket cap relax (P5-3)")
        interp.append("the realistic estimate should improve further.")
    for line in interp:
        print(line)

    # ----- WRITE REPORT -----
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    mdlines = [
        "# Specialist Complementarity Matrix (v3 8Q, 2026-05-20)",
        "",
        f"**Method**: load per-day PnL from v3 paper_trade_replay sidecars for "
        f"each verified specialist across 8 quarters (24Q1-25Q4). Compute pairwise "
        f"correlations + Jaccard + win-day complementarity + naive equal-weight union.",
        "",
        f"**Specialists analysed** ({len(labels)} with full 8Q coverage):",
        ""
    ]
    for l in labels:
        df = series[l]
        spec_blend = next(b for ll, b, _ in SPECIALISTS if ll == l)
        spec_univ = next(u for ll, _, u in SPECIALISTS if ll == l)
        mdlines.append(f"- **{l}**: {spec_blend} ({spec_univ}), "
                        f"{len(df)} days, {(df['new_entries']>0).sum()} fire days")
    mdlines += [
        "",
        "## 1. PnL correlation (full 8Q)",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for l1 in labels:
        row = [f"{corr_pnl.loc[f'pnl_{l1}', f'pnl_{l2}']:+.3f}" for l2 in labels]
        mdlines.append(f"| **{l1}** | " + " | ".join(row) + " |")
    mdlines += [
        "",
        "## 2. Fire-day Jaccard",
        "",
        "Fraction of days where both specialists fired (`new_entries > 0`). "
        "1.0 = perfect overlap; 0 = disjoint.",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for l1 in labels:
        row = [f"{jaccard.loc[l1, l2]:.3f}" for l2 in labels]
        mdlines.append(f"| **{l1}** | " + " | ".join(row) + " |")
    mdlines += [
        "",
        "## 3. Win-day complementarity (row wins +1%, col flat/neg)",
        "",
        "Higher = ROW catches days that COL misses. Read row→col.",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for l1 in labels:
        row = []
        for l2 in labels:
            v = win_comp.loc[l1, l2]
            row.append(f"{v:.3f}" if not pd.isna(v) else "—")
        mdlines.append(f"| **{l1}** | " + " | ".join(row) + " |")
    mdlines += [
        "",
        "## 4. Naive equal-weight union (UPPER BOUND — no cap, no dedup)",
        "",
        f"- Common-panel days: {n_days}",
        f"- Daily mean (union): {union_mean_d:+.4f}%",
        f"- Compound 8Q (union): {union_comp:+.2f}%",
        f"- Annualized (union): {union_ann:+.2f}%",
        f"- 7d floor (+{GATE_7D_PCT}%) pass rate: **{union_7d_pass:.1f}%**",
        "",
        "### Per-specialist for comparison",
        "",
        "| spec | daily mean | comp 8Q | annualized | 7d pass% |",
        "|---|---:|---:|---:|---:|",
    ]
    for l in labels:
        s = common[f"pnl_{l}"]
        d_mean = s.mean()
        comp = ((1 + s/100).prod() - 1) * 100
        ann = ((1 + d_mean/100)**365 - 1) * 100
        roll7 = s.rolling(7).sum()
        p7 = (roll7 >= GATE_7D_PCT).mean() * 100
        mdlines.append(f"| {l} | {d_mean:+.4f}% | {comp:+.2f}% | {ann:+.2f}% | {p7:.1f}% |")
    mdlines += [
        "",
        "## 5. Per-regime mean daily PnL",
        "",
        "| spec | crash | bear | chop | bull |",
        "|---|---:|---:|---:|---:|",
    ]
    for l in labels:
        cells = []
        for r in ("crash", "bear", "chop", "bull"):
            mask = common["regime"] == r
            if mask.sum() == 0:
                cells.append("n/a")
            else:
                m = common.loc[mask, f"pnl_{l}"].mean()
                cells.append(f"{m:+.3f}%")
        mdlines.append(f"| {l} | " + " | ".join(cells) + " |")
    mdlines += interp
    OUT_MD.write_text("\n".join(mdlines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_MD}")

    # Save matrices
    out = {
        "pnl_correlation": corr_pnl.to_dict(),
        "fire_day_jaccard": jaccard.to_dict(),
        "win_complementarity": win_comp.fillna(-999).to_dict(),
        "naive_union": {
            "n_days": int(n_days),
            "daily_mean_pct": float(union_mean_d),
            "comp_8Q_pct": float(union_comp),
            "annualized_pct": float(union_ann),
            "gate_7d_pass_rate_pct": float(union_7d_pass),
        },
        "per_specialist": {
            l: {
                "daily_mean_pct": float(common[f"pnl_{l}"].mean()),
                "comp_pct": float(((1 + common[f"pnl_{l}"]/100).prod() - 1) * 100),
                "ann_pct": float(((1 + common[f"pnl_{l}"].mean()/100)**365 - 1) * 100),
                "gate_7d_pass_pct": float((common[f"pnl_{l}"].rolling(7).sum() >= GATE_7D_PCT).mean() * 100),
                "n_fire_days": int((common[f"fire_{l}"] > 0).sum()),
            }
            for l in labels
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[OK] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
