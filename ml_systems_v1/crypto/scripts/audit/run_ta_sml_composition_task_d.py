"""Task D: TA_SML composition analysis (Stack / Parallel / Replace) from per-day v3 sidecars.

Computes:
  - PARALLEL: convex combinations of STRICT and TA_SML_SOLO day-PnL streams at
    multiple weights (10/15/20/25/30% TA_SML, rest STRICT).
  - REPLACE: TA_SML_SOLO standalone (already measured at +71.84%).
  - STACK (analytical proxy): on days when TA_SML had a trade (fire-day),
    use TA_SML PnL; otherwise use STRICT. Approximates "TA_SML inside STRICT
    regime gate" without re-running v3 (which would be 4hrs).
  - Correlation, joint DD, joint Sharpe per composition.

Reads v3 sidecar JSONs:
  - STRICT_LO_SETUP60 u100 8Q (8 logs)
  - TA_SML_SOLO       u50  8Q (8 logs from this session's batch)

Universe note: STRICT runs u100, TA_SML runs u50. The per-day PnL series for
each is at portfolio level (NAV %), so the composition math is universe-agnostic
when treating each as a capital-silo.

Output: runs/audit/TA_SML_COMPOSITION_TASK_D_2026_05_18.md + parquet ledgers.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "strat_audit"
OUT_DIR = ROOT / "runs" / "audit"

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

STRICT_TAG = "REGIME_ROUTER_STRICT_LO_SETUP60"
TA_SML_TAG = "TA_SML_SOLO"


def load_per_day(blend: str, universe: str, window: str) -> pd.DataFrame:
    """Read v3 sidecar JSON, return per_day dataframe."""
    p = LOGS / f"paper_trade_replay_v3_{blend}_{universe}_{window}.json"
    js = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for d in js.get("per_day", []):
        rows.append({
            "date": d["date"],
            "day_pnl_pct": float(d["day_pnl_pct"]),
            "nav": float(d["nav"]),
            "n_entries": int(d.get("new_entries", 0)),
            "n_open": int(d.get("open_book_after", 0)),
            "n_closed": int(d.get("closed_today", 0)),
            "dispatch_ok": bool(d.get("dispatch_ok", False)),
        })
    return pd.DataFrame(rows)


def find_ta_sml_log(window: str) -> Path:
    """TA_SML_SOLO logs have varying start dates (window_start) per v3."""
    # window like "20240101_20240331" -> use end date
    end = window.split("_")[1]
    matches = list(LOGS.glob(f"paper_trade_replay_v3_TA_SML_SOLO_u50_*_{end}.json"))
    if not matches:
        return None
    # If multiple, take the most recent
    return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]


def main():
    # 1. Load STRICT per-day across 8Q
    strict_dfs = []
    for label, w in QUARTERS:
        df = load_per_day(STRICT_TAG, "u100", w)
        df["quarter"] = label
        df["strategy"] = "STRICT"
        strict_dfs.append(df)
    strict = pd.concat(strict_dfs, ignore_index=True)
    strict["date"] = pd.to_datetime(strict["date"]).dt.date

    # 2. Load TA_SML_SOLO per-day across 8Q
    ta_sml_dfs = []
    for label, w in QUARTERS:
        p = find_ta_sml_log(w)
        if p is None:
            print(f"[WARN] no TA_SML log for {label}")
            continue
        js = json.loads(p.read_text(encoding="utf-8"))
        rows = []
        for d in js.get("per_day", []):
            rows.append({
                "date": d["date"],
                "day_pnl_pct": float(d["day_pnl_pct"]),
                "nav": float(d["nav"]),
                "n_entries": int(d.get("new_entries", 0)),
                "n_open": int(d.get("open_book_after", 0)),
                "n_closed": int(d.get("closed_today", 0)),
            })
        df = pd.DataFrame(rows)
        df["quarter"] = label
        df["strategy"] = "TA_SML_SOLO"
        ta_sml_dfs.append(df)
    ta_sml = pd.concat(ta_sml_dfs, ignore_index=True)
    ta_sml["date"] = pd.to_datetime(ta_sml["date"]).dt.date

    # 3. Join on date
    j = strict.merge(ta_sml, on="date", suffixes=("_S", "_T"), how="outer").sort_values("date")
    j["pnl_S"] = j["day_pnl_pct_S"].fillna(0.0)
    j["pnl_T"] = j["day_pnl_pct_T"].fillna(0.0)
    j["n_closed_T"] = j["n_closed_T"].fillna(0)
    j["quarter"] = j["quarter_T"].fillna(j["quarter_S"])

    print(f"Joined panel: {len(j)} days  ")
    print(f"STRICT-only days: {(j['day_pnl_pct_T'].isna()).sum()}  ")
    print(f"TA_SML-only days: {(j['day_pnl_pct_S'].isna()).sum()}  ")
    print(f"Both-active days: {(j['day_pnl_pct_T'].notna() & j['day_pnl_pct_S'].notna()).sum()}")

    # 4. Correlation
    both_mask = j["day_pnl_pct_T"].notna() & j["day_pnl_pct_S"].notna()
    rho = j.loc[both_mask, ["day_pnl_pct_S", "day_pnl_pct_T"]].corr().iloc[0, 1]
    print(f"\nCorrelation STRICT vs TA_SML (both-active days): {rho:+.4f}")

    # 5. Summary helper
    def metrics(daily_pnl: pd.Series, label: str) -> dict:
        """Compute total, mean Sh, worst DD, positive quarters."""
        s = daily_pnl.fillna(0.0)
        total = s.sum()
        # Running NAV for DD
        nav = (1 + s / 100).cumprod()
        peak = nav.cummax()
        dd = (nav / peak - 1) * 100
        worst_dd = dd.min()
        # Per-quarter
        df = pd.DataFrame({"pnl": s, "quarter": j["quarter"]}).groupby("quarter")["pnl"].sum()
        n_pos = (df > 0).sum()
        n_total = (df != 0).count()
        # Daily Sharpe annualized (sqrt(365) crypto convention)
        if s.std() > 0:
            sh = s.mean() / s.std() * np.sqrt(365)
        else:
            sh = 0.0
        return {
            "label": label,
            "total_pnl_pct": float(total),
            "mean_sh_ann": float(sh),
            "worst_dd_pct": float(worst_dd),
            "pos_quarters": f"{int(n_pos)}/{int(n_total)}",
            "per_quarter": {q: float(v) for q, v in df.items()},
        }

    # 6. Build composition variants
    out = []

    # 6a. STRICT alone (anchor)
    out.append(metrics(j["pnl_S"], "STRICT (anchor)"))

    # 6b. TA_SML alone (REPLACE mode)
    out.append(metrics(j["pnl_T"], "REPLACE: TA_SML_SOLO standalone"))

    # 6c. PARALLEL: weighted convex combination at various weights
    for w_ta in [0.10, 0.15, 0.20, 0.25, 0.30]:
        w_s = 1.0 - w_ta
        combined = w_s * j["pnl_S"] + w_ta * j["pnl_T"]
        out.append(metrics(combined, f"PARALLEL: {int(w_ta*100)}% TA_SML / {int(w_s*100)}% STRICT"))

    # 6d. STACK (analytical proxy): on TA_SML fire-days use TA_SML, else STRICT.
    # Fire-day defined as TA_SML had >=1 entry that day.
    fire_day = j["n_entries_T"].fillna(0) > 0
    stack_pnl = np.where(fire_day, j["pnl_T"], j["pnl_S"])
    out.append(metrics(pd.Series(stack_pnl, index=j.index),
                       "STACK proxy: TA_SML on fire-days, STRICT otherwise"))

    # 6e. UNION OVERLAY: STRICT always-on, ADDITIONAL TA_SML allocation on fire-days
    # at 15% sleeve weight (compounded as separate position)
    for w_ta in [0.10, 0.15, 0.20]:
        overlay_pnl = j["pnl_S"] + w_ta * np.where(fire_day, j["pnl_T"], 0)
        out.append(metrics(pd.Series(overlay_pnl, index=j.index),
                           f"OVERLAY: STRICT base + {int(w_ta*100)}% TA_SML on fire-days"))

    # 7. Write report
    lines = []
    def w(s=""):
        lines.append(s)

    w("# TA_SML Composition Test — Task D")
    w()
    w("**Date**: 2026-05-19  ")
    w("**Charter**: docs/TA_SML_REFRESH_BRIEF_2026_05_18.md §3 Task D  ")
    w("**Sources**: 8Q v3 sidecar logs (STRICT_LO_SETUP60 u100 + TA_SML_SOLO u50)  ")
    w(f"**Joined panel**: {len(j)} days; both-active {both_mask.sum()}; "
      f"STRICT-only {(j['day_pnl_pct_T'].isna()).sum()}; "
      f"TA_SML-only {(j['day_pnl_pct_S'].isna()).sum()}")
    w()
    w(f"**Correlation (both-active)**: rho = {rho:+.4f}")
    w()
    w("---")
    w()
    w("## 1. Composition results (8Q WF, 24Q1-25Q4)")
    w()
    w("| Configuration | total_pnl | mean Sh ann | worst DD | pos Q | Lift vs STRICT |")
    w("|---|---:|---:|---:|---:|---:|")
    base = next(r for r in out if r["label"] == "STRICT (anchor)")
    base_pnl = base["total_pnl_pct"]
    for r in out:
        lift = r["total_pnl_pct"] - base_pnl
        lift_str = f"{lift:+.2f}pp" if r["label"] != "STRICT (anchor)" else "—"
        w(f"| {r['label']} | {r['total_pnl_pct']:+.2f}% | {r['mean_sh_ann']:+.2f} | "
          f"{r['worst_dd_pct']:+.2f}% | {r['pos_quarters']} | {lift_str} |")
    w()

    # 2. Per-quarter detail for the winner
    winner = max(out, key=lambda r: r["total_pnl_pct"])
    w(f"## 2. Per-quarter — best config: `{winner['label']}`")
    w()
    w(f"Total: **{winner['total_pnl_pct']:+.2f}%** / Sh {winner['mean_sh_ann']:+.2f} / "
      f"DD {winner['worst_dd_pct']:+.2f}% / pos {winner['pos_quarters']}")
    w()
    w("| Quarter | PnL % |")
    w("|---|---:|")
    for q, v in winner["per_quarter"].items():
        w(f"| {q} | {v:+.3f}% |")
    w()

    # 3. Verdict
    w("## 3. Verdict and recommendation")
    w()
    w("**Acceptance gate** (brief §3 Task D): composition must beat standalone STRICT on Sharpe AND DD, "
      "OR pareto-dominate at lower correlation.")
    w()
    parallel_15 = next(r for r in out if "PARALLEL: 15%" in r["label"])
    parallel_20 = next(r for r in out if "PARALLEL: 20%" in r["label"])
    overlay_15 = next(r for r in out if "OVERLAY: STRICT base + 15%" in r["label"])
    stack = next(r for r in out if "STACK proxy" in r["label"])
    replace = next(r for r in out if "REPLACE" in r["label"])

    w(f"- **REPLACE** (TA_SML alone): {replace['total_pnl_pct']:+.2f}% / "
      f"DD {replace['worst_dd_pct']:+.2f}% / pos {replace['pos_quarters']}. "
      f"Highest absolute return but DD breach in 25Q3.")
    w(f"- **STACK proxy** (TA_SML on fire-days else STRICT): {stack['total_pnl_pct']:+.2f}% / "
      f"DD {stack['worst_dd_pct']:+.2f}% / pos {stack['pos_quarters']}. "
      f"Strong if TA_SML fire-day selection picks the right days.")
    w(f"- **OVERLAY** (15% TA_SML on top of STRICT base): {overlay_15['total_pnl_pct']:+.2f}% / "
      f"DD {overlay_15['worst_dd_pct']:+.2f}% / pos {overlay_15['pos_quarters']}. "
      f"Likely best of (DD, return) tradeoff because STRICT carries through 25Q3.")
    w(f"- **PARALLEL 15%/85%** silo: {parallel_15['total_pnl_pct']:+.2f}% / "
      f"DD {parallel_15['worst_dd_pct']:+.2f}% / pos {parallel_15['pos_quarters']}.")
    w(f"- **PARALLEL 20%/80%** silo: {parallel_20['total_pnl_pct']:+.2f}% / "
      f"DD {parallel_20['worst_dd_pct']:+.2f}% / pos {parallel_20['pos_quarters']}.")
    w()
    w("**Recommendation**: choose between PARALLEL silo and STACK proxy based on:")
    w("- Operational simplicity: PARALLEL is simpler (two strategies running in parallel)")
    w("- Capital efficiency: STACK uses 100% capital one-or-the-other")
    w("- DD floor: whichever clears -10% worst DD")
    w()
    w("**Caveat**: STACK proxy assumes TA_SML fire-days are correctly identified "
      "(used n_entries > 0 here). Real STACK in v3 requires a regime-router blend that "
      "switches between STRICT and TA_SML per day — a separate ~1 session of integration work.")
    w()
    w("---")
    w("Generated by `scripts/audit/run_ta_sml_composition_task_d.py`")

    out_md = OUT_DIR / "TA_SML_COMPOSITION_TASK_D_2026_05_18.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")

    # Persist ledger
    j.to_parquet(OUT_DIR / "ta_sml_composition_joined_panel.parquet")
    pd.DataFrame(out).drop(columns=["per_quarter"]).to_parquet(
        OUT_DIR / "ta_sml_composition_summary.parquet")
    print(f"Wrote ledger parquets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
