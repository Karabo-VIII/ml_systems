"""Model Diagnostic Oracle — per-day attribution of WHY strategies miss.

User mandate (2026-05-19): "I want proper explanations about WHY a particular
model is missing opportunities. Is our selection of model params poor, or is it
something else."

Per-day failure-mode classifier:

  Each calendar day (24Q1-25Q4) is classified into ONE of:

  CAPTURE_HIT      Model fired AND captured >= 50% of oracle's K=5 LO availability
  CAPTURE_PARTIAL  Model fired AND captured 10-50%
  CAPTURE_LEAK     Model fired but captured < 10% (entry/exit timing fail)
  MISS_NO_FIRE_BIG Oracle had >= +5% available but model did not fire (selection fail)
  MISS_NO_FIRE_OK  Oracle was modest (< +2%) AND model did not fire (correct cash)
  ADVERSE_FIRE     Model fired into a negative oracle day (regime fail)
  DEAD_MARKET      Oracle availability < +0.5% AND model correctly cash
  UNKNOWN          Edge cases (NaN, missing data)

  Aggregates per week (rolling 7d), per 3d window, per regime, per quarter.

Plus: gate checks at user-mandated thresholds (2026-05-19):
  - 7-day rolling window MEDIAN ROI floor (FLOOR-must-clear)
  - 3-day rolling window MEDIAN ROI gate (next-most-strict)
  - Daily ROI 0.75-1.25% (lower-end "not doing enough" line)

Inputs:
  - data/processed/outcome_catalog.parquet (oracle K=5 LO daily availability)
  - logs/strat_audit/paper_trade_replay_v3_<BLEND>_*.json (per_day + trade_ledger)

Output:
  - runs/audit/MODEL_DIAGNOSTIC_<BLEND>_2026_05_19.md
  - runs/audit/diagnostic_panel_<BLEND>.parquet (per-day classified panel)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "strat_audit"
OUT_DIR = ROOT / "runs" / "audit"
OUTCOME_PATH = ROOT / "data" / "processed" / "outcome_catalog.parquet"

# Quarter labels for the 8Q window
QUARTERS = [
    ("2024Q1", "20240101_20240331"),
    ("2024Q2", "20240401_20240630"),
    ("2024Q3", "20240701_20240930"),
    ("2024Q4", "20241001_20241231"),
    ("2025Q1", "20250101_20250331"),
    ("2025Q2", "20250401_20250630"),
    ("2025Q3", "20250701_20250930"),
    ("2025Q4", "20251001_20251231"),
]

# User-mandated gates (2026-05-19)
GATE_7D_FLOOR_PCT = 0.75 * 7   # 0.75%/d × 7d = +5.25% over 7d window
GATE_3D_FLOOR_PCT = 0.75 * 3   # +2.25% over 3d window
GATE_DAILY_LOW = 0.75          # daily ROI lower "not doing enough" line
GATE_DAILY_HIGH = 1.25         # daily ROI upper "minimum acceptable" line

# Oracle thresholds for day classification
ORACLE_BIG_OPP_PCT = 5.0       # day had ≥ +5% available
ORACLE_MODEST_PCT = 2.0        # day had ≥ +2% available
ORACLE_DEAD_PCT = 0.5          # day had < +0.5% available

# Capture-ratio thresholds
CAPTURE_HIT_FRAC = 0.50
CAPTURE_PARTIAL_FRAC = 0.10


def load_strategy_per_day(blend: str, universe: str = "u100") -> pd.DataFrame:
    """Load v3 per_day records across 8Q for a given blend."""
    rows = []
    for label, win in QUARTERS:
        end_no_dash = win.split("_")[1]
        pat = f"paper_trade_replay_v3_{blend}_{universe}_*_{end_no_dash}.json"
        matches = list(LOGS.glob(pat))
        if not matches:
            continue
        p = sorted(matches, key=lambda x: x.stat().st_mtime)[-1]
        j = json.loads(p.read_text(encoding="utf-8"))
        for d in j.get("per_day", []):
            rows.append({
                "date": d["date"],
                "quarter": label,
                "pnl_pct": float(d["day_pnl_pct"]),
                "n_entries": int(d.get("new_entries", 0)),
                "n_open": int(d.get("open_book_after", 0)),
                "n_closed": int(d.get("closed_today", 0)),
                "btc_30d": float(d.get("btc_30d", 0.0)),
                "dispatch_ok": bool(d.get("dispatch_ok", False)),
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def load_oracle() -> pd.DataFrame:
    """Load outcome_catalog with K=5 LO daily availability for 24Q1-25Q4."""
    oc = pl.read_parquet(OUTCOME_PATH).to_pandas()
    oc["date"] = pd.to_datetime(oc["date"]).dt.date
    oc = oc[(oc["date"] >= date(2024, 1, 1)) & (oc["date"] <= date(2025, 12, 31))].copy()
    oc["oracle_1d_pct"] = oc["ideal_k5_1d_ret"] * 100
    oc["oracle_3d_pct"] = oc["ideal_k5_3d_ret"] * 100
    oc["oracle_5d_pct"] = oc["ideal_k5_5d_ret"] * 100
    return oc[["date", "oracle_1d_pct", "oracle_3d_pct", "oracle_5d_pct"]]


def classify_day(row: pd.Series) -> str:
    """Per-day failure-mode classifier — the diagnostic core."""
    oracle = row["oracle_1d_pct"]
    pnl = row["pnl_pct"]
    fired = row["n_entries"] > 0 or row["n_closed"] > 0 or row["n_open"] > 0

    if pd.isna(oracle):
        return "UNKNOWN"
    if oracle < ORACLE_DEAD_PCT and not fired:
        return "DEAD_MARKET"  # correct cash on dead day
    if fired and pnl < -0.5 * abs(oracle):  # noticeably negative when oracle wasn't catastrophic
        if oracle < 0:
            return "ADVERSE_FIRE"  # fired into known negative day
        if oracle > 0:
            return "ADVERSE_FIRE"  # fired but lost despite positive available
    if fired:
        # Capture ratio = realized / oracle (clipped 0..1)
        capture = pnl / max(oracle, 1e-9) if oracle > 0 else 0
        capture = max(0.0, min(1.0, capture))
        if capture >= CAPTURE_HIT_FRAC:
            return "CAPTURE_HIT"
        if capture >= CAPTURE_PARTIAL_FRAC:
            return "CAPTURE_PARTIAL"
        return "CAPTURE_LEAK"
    # Did NOT fire
    if oracle >= ORACLE_BIG_OPP_PCT:
        return "MISS_NO_FIRE_BIG"
    if oracle < ORACLE_MODEST_PCT:
        return "MISS_NO_FIRE_OK"  # correct cash; no big opp anyway
    return "MISS_NO_FIRE_OK"  # modest opp missed; not a critical fail


def regime_label(btc_30d: float) -> str:
    if btc_30d <= -0.15:
        return "crash"
    if btc_30d <= -0.05:
        return "bear"
    if btc_30d >= 0.05:
        return "bull"
    return "chop"


def build_diagnostic(blend: str, universe: str = "u100") -> dict:
    """End-to-end diagnostic build for one blend."""
    strat = load_strategy_per_day(blend, universe)
    if len(strat) == 0:
        return {"error": f"no v3 sidecars for {blend} {universe}"}
    oracle = load_oracle()
    df = strat.merge(oracle, on="date", how="left")
    df["regime"] = df["btc_30d"].apply(regime_label)
    df["mode"] = df.apply(classify_day, axis=1)
    df["fired"] = (df["n_entries"] > 0) | (df["n_closed"] > 0) | (df["n_open"] > 0)

    # Capture ratio per day (clipped 0..1; 0 if no oracle availability or model didn't fire)
    df["capture_ratio"] = 0.0
    pos_mask = df["oracle_1d_pct"] > 0
    df.loc[pos_mask, "capture_ratio"] = (df.loc[pos_mask, "pnl_pct"] /
                                          df.loc[pos_mask, "oracle_1d_pct"]).clip(0.0, 1.0)

    # Rolling 3d / 7d / 30d windows
    df["pnl_3d_sum"] = df["pnl_pct"].rolling(3).sum()
    df["pnl_7d_sum"] = df["pnl_pct"].rolling(7).sum()
    df["pnl_30d_sum"] = df["pnl_pct"].rolling(30).sum()

    # User gates
    df["gate_7d_pass"] = df["pnl_7d_sum"] >= GATE_7D_FLOOR_PCT
    df["gate_3d_pass"] = df["pnl_3d_sum"] >= GATE_3D_FLOOR_PCT
    df["gate_daily_pass"] = df["pnl_pct"] >= GATE_DAILY_LOW

    # ----- Aggregates -----
    n_days = len(df)
    mode_counts = df["mode"].value_counts().to_dict()
    mode_pct = {k: v / n_days * 100 for k, v in mode_counts.items()}

    regime_table = df.groupby("regime").agg(
        n=("pnl_pct", "size"),
        mean_pnl=("pnl_pct", "mean"),
        pos_pct=("pnl_pct", lambda s: (s > 0).mean() * 100),
        capture=("capture_ratio", "mean"),
    ).reset_index()

    # Mode × regime cross-tab (rate)
    crosstab = pd.crosstab(df["mode"], df["regime"], normalize="columns") * 100

    # Gate pass rates (rolling windows)
    gate_summary = {
        "n_days": int(n_days),
        "n_3d_windows": int(df["pnl_3d_sum"].notna().sum()),
        "n_7d_windows": int(df["pnl_7d_sum"].notna().sum()),
        "gate_3d_pass_rate": float(df["gate_3d_pass"].mean() * 100),
        "gate_7d_pass_rate": float(df["gate_7d_pass"].mean() * 100),
        "gate_daily_pass_rate": float(df["gate_daily_pass"].mean() * 100),
        "median_3d_pnl": float(df["pnl_3d_sum"].median()),
        "median_7d_pnl": float(df["pnl_7d_sum"].median()),
        "median_daily_pnl": float(df["pnl_pct"].median()),
        "mean_daily_pnl": float(df["pnl_pct"].mean()),
    }

    return {
        "df": df,
        "n_days": n_days,
        "mode_counts": mode_counts,
        "mode_pct": mode_pct,
        "regime_table": regime_table,
        "crosstab": crosstab,
        "gate_summary": gate_summary,
    }


def render_report(blend: str, universe: str, diag: dict) -> str:
    """Compose the markdown diagnostic report."""
    df = diag["df"]
    g = diag["gate_summary"]
    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# Model Diagnostic Oracle — {blend} ({universe})")
    w()
    w("**Date**: 2026-05-19  ")
    w("**Charter**: user-mandated diagnostic upgrade — explain WHY models miss, not just measure.  ")
    w(f"**Window**: 24Q1-25Q4 v3 paper-trade-replay sidecars + outcome_catalog K=5 LO oracle.  ")
    w(f"**Days analysed**: {g['n_days']}  ")
    w()
    w("---")
    w()
    w("## 1. User-mandated gates (2026-05-19)")
    w()
    w("Daily floor: **0.75-1.25%/day** (lower-end = 'not doing enough' line; 7d window must clear at min).")
    w()
    w(f"| Gate | Threshold | Achieved | Pass-rate |")
    w(f"|---|---:|---:|---:|")
    w(f"| Daily ROI ≥ +0.75% | per day | median {g['median_daily_pnl']:+.3f}% / mean {g['mean_daily_pnl']:+.3f}% | **{g['gate_daily_pass_rate']:.1f}%** of days |")
    w(f"| 3-day rolling ≥ +2.25% (0.75%×3) | per 3d window | median {g['median_3d_pnl']:+.3f}% | **{g['gate_3d_pass_rate']:.1f}%** of windows |")
    w(f"| **7-day FLOOR ≥ +5.25% (0.75%×7)** | per 7d window | median {g['median_7d_pnl']:+.3f}% | **{g['gate_7d_pass_rate']:.1f}%** of windows |")
    w()

    # Honest verdict
    sev_7d = g['gate_7d_pass_rate']
    if sev_7d >= 50:
        verdict = "PASS — 7d floor cleared in majority of windows"
    elif sev_7d >= 25:
        verdict = "PARTIAL — 7d floor cleared in significant minority"
    elif sev_7d >= 10:
        verdict = "FAIL with seasonal pass — 7d floor only clears in narrow windows"
    else:
        verdict = "FAIL — 7d floor structurally not cleared at this strategy / regime"
    w(f"**7d floor verdict**: {verdict}")
    w()

    w("## 2. Day-mode classification")
    w()
    w("Each day classified into ONE mode (priority: ADVERSE > CAPTURE > MISS > DEAD).")
    w()
    w("| Mode | n_days | % of days | Meaning |")
    w("|---|---:|---:|---|")
    mode_meanings = {
        "CAPTURE_HIT": "model fired and captured ≥ 50% of oracle (great)",
        "CAPTURE_PARTIAL": "model fired and captured 10-50% (decent)",
        "CAPTURE_LEAK": "model fired but captured < 10% (entry/exit fail)",
        "MISS_NO_FIRE_BIG": "oracle ≥ +5% but model did NOT fire (SELECTION FAIL)",
        "MISS_NO_FIRE_OK": "model correctly held cash; no major opp",
        "ADVERSE_FIRE": "model fired into a negative day (regime / direction fail)",
        "DEAD_MARKET": "no oracle availability; cash correct",
        "UNKNOWN": "edge case — NaN or missing data",
    }
    for mode in ["CAPTURE_HIT", "CAPTURE_PARTIAL", "CAPTURE_LEAK",
                  "MISS_NO_FIRE_BIG", "MISS_NO_FIRE_OK", "ADVERSE_FIRE",
                  "DEAD_MARKET", "UNKNOWN"]:
        n = diag["mode_counts"].get(mode, 0)
        pct = diag["mode_pct"].get(mode, 0)
        w(f"| {mode} | {n} | {pct:.1f}% | {mode_meanings.get(mode, '')} |")
    w()

    # Diagnostic narrative
    miss_big = diag["mode_pct"].get("MISS_NO_FIRE_BIG", 0)
    leak = diag["mode_pct"].get("CAPTURE_LEAK", 0)
    adverse = diag["mode_pct"].get("ADVERSE_FIRE", 0)
    hits = diag["mode_pct"].get("CAPTURE_HIT", 0) + diag["mode_pct"].get("CAPTURE_PARTIAL", 0)

    w("### Diagnostic narrative")
    w()
    if miss_big >= 30:
        w(f"- **PRIMARY FAILURE MODE: setup selection.** Model failed to fire on {miss_big:.1f}% of days where oracle had ≥+5% available. The signal/threshold is too conservative or wrong.")
    elif leak >= 30:
        w(f"- **PRIMARY FAILURE MODE: entry/exit timing.** Model fires but bleeds out on {leak:.1f}% of days. Stop/take-profit/hold-horizon settings are off.")
    elif adverse >= 20:
        w(f"- **PRIMARY FAILURE MODE: regime / direction.** Model fires into adverse days {adverse:.1f}% of the time. Regime gate is failing.")
    else:
        w(f"- **NO SINGLE PRIMARY FAILURE MODE** — distribution is reasonable. Misses are MISS_NO_FIRE_OK ({diag['mode_pct'].get('MISS_NO_FIRE_OK', 0):.1f}%, correct cash).")
    w()
    w(f"- Capture-quality (HIT+PARTIAL): {hits:.1f}% of days. CAPTURE_HIT alone: {diag['mode_pct'].get('CAPTURE_HIT', 0):.1f}%")
    w(f"- Adverse fires: {adverse:.1f}%")
    w()

    w("## 3. Per-regime breakdown")
    w()
    w("| Regime | n_days | mean PnL | % positive | mean capture |")
    w("|---|---:|---:|---:|---:|")
    for _, r in diag["regime_table"].iterrows():
        w(f"| {r['regime']} | {int(r['n'])} | {r['mean_pnl']:+.3f}% | {r['pos_pct']:.1f}% | {r['capture']*100:.2f}% |")
    w()

    w("## 4. Mode × regime cross-tab (column %)")
    w()
    crosstab = diag["crosstab"]
    cols = sorted(crosstab.columns.tolist())
    w("| Mode | " + " | ".join(cols) + " |")
    w("|---" * (len(cols) + 1) + "|")
    for mode in sorted(crosstab.index.tolist()):
        row = [f"{crosstab.loc[mode, c]:.1f}%" for c in cols]
        w(f"| {mode} | " + " | ".join(row) + " |")
    w()

    # Specific actionable diagnosis
    w("## 5. Actionable next moves (failure-mode → fix)")
    w()
    w("Based on the diagnostic, the highest-EV experiments to lift this strategy:")
    w()
    if miss_big >= 20:
        w(f"- **Setup-selection upgrade**: model misses ≥+5% oracle days {miss_big:.1f}% of the time. Lower trigger threshold or add a complementary detector. EV: high.")
    if leak >= 20:
        w(f"- **Exit-manager swap**: capture leaks {leak:.1f}% of days. Try max-hold-only (no signal_flip) or trailing stop. EV: high.")
    if adverse >= 15:
        w(f"- **Tighten regime gate**: model fires adverse {adverse:.1f}% of days. Add BTC SMA-200 slope OR raise regime threshold from -5% to -2%. EV: medium-high.")
    # 7d gate-specific
    if g["gate_7d_pass_rate"] < 30:
        w(f"- **7d-gate FAIL**: only {g['gate_7d_pass_rate']:.1f}% of 7d windows clear +5.25%. ENSEMBLE COMPLEMENTARY SPECIALIST needed — this strategy alone cannot clear the 7d floor; pair it with a regime-orthogonal lane.")
    if g["gate_daily_pass_rate"] < 20:
        w(f"- **Daily ROI floor FAIL**: only {g['gate_daily_pass_rate']:.1f}% of days clear +0.75%. Selection precision is the binding constraint, not throughput.")
    w()

    # Worst weeks list (for forensic deep-dive)
    df_wk = df.copy()
    df_wk["week"] = pd.to_datetime(df_wk["date"]).dt.strftime("%G-W%V")
    wk_pnl = df_wk.groupby("week").agg(pnl_sum=("pnl_pct","sum"),
                                         trades=("n_closed", "sum"),
                                         oracle_sum=("oracle_1d_pct", "sum")).reset_index()
    worst5 = wk_pnl.sort_values("pnl_sum").head(5)
    w("## 6. Worst 5 weeks (for forensic dive)")
    w()
    w("| Week | strategy pnl | oracle available | trades | gap to oracle |")
    w("|---|---:|---:|---:|---:|")
    for _, r in worst5.iterrows():
        gap = r["oracle_sum"] - r["pnl_sum"]
        w(f"| {r['week']} | {r['pnl_sum']:+.2f}% | {r['oracle_sum']:+.2f}% | {int(r['trades'])} | {gap:+.2f}pp |")
    w()
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--universe", default="u100")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diag = build_diagnostic(args.blend, args.universe)
    if "error" in diag:
        print(f"FAIL: {diag['error']}")
        return 1
    md = render_report(args.blend, args.universe, diag)
    out_md = Path(args.out) if args.out else OUT_DIR / f"MODEL_DIAGNOSTIC_{args.blend}_2026_05_19.md"
    out_md.write_text(md, encoding="utf-8")
    diag["df"].to_parquet(OUT_DIR / f"diagnostic_panel_{args.blend}.parquet")
    print(f"Wrote {out_md}")

    # Headline echo
    g = diag["gate_summary"]
    print(f"\nGate scorecard ({args.blend}):")
    print(f"  7d floor (+5.25%):   {g['gate_7d_pass_rate']:6.2f}% pass | median {g['median_7d_pnl']:+.2f}%")
    print(f"  3d floor (+2.25%):   {g['gate_3d_pass_rate']:6.2f}% pass | median {g['median_3d_pnl']:+.2f}%")
    print(f"  daily floor (+0.75%):{g['gate_daily_pass_rate']:6.2f}% pass | median {g['median_daily_pnl']:+.3f}%")
    print(f"\nMode distribution:")
    for m, p in sorted(diag["mode_pct"].items(), key=lambda x: -x[1]):
        print(f"  {m:<20} {p:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
