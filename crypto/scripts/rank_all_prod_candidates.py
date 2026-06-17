"""Comprehensive PROD-candidate ranker.

Scans paper_trader_v2 seed snapshots across the project, computes full-history
+ April window metrics, and ranks every deployable / alternate / legacy strategy.

Seed categories:
    CORE       - the 10 sleeves actively included in deploy_all_top_strats.py
    ALTERNATE  - same strategy, different sizing (k15 variants, _enh variants,
                 v9meta variants)
    DEAD       - officially dropped (confirmed negative in 2025-2026 regime)
    LEGACY     - historical prod / frozen snapshots (reference points)
    RELATED    - other ranked-tier entries from config/deployment_ranking.yaml

Output:
    logs/deployment/<UTC-date>/prod_candidates_ranking.csv
    logs/deployment/<UTC-date>/prod_candidates_ranking.md (operator-readable)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "logs" / "paper_trader_v2" / "seeds"

# Curated: what to rank (category, seed_dir, display_name, notes)
CANDIDATES = [
    # ============ CORE (10 sleeves actively wired into deploy_all_top_strats) ============
    ("CORE", "pt_xsec_K5_5_FULL_dneut",         "xsec_K5_5_FULL_dneut",
        "xsec ranker, K=5L + K=5S, delta-neutral, meta+regime gate, 10% stop. U50."),
    ("CORE", "pt_xgb_K3_long_WEALTH40",         "xgb_K3_long_WEALTH40",
        "XGB ranker, K=3 long-only, WEALTH40 sizing. Wealth amplifier."),
    ("CORE", "pt_cat_K1_stop_no_macro",         "cat_K1_stop_no_macro",
        "CatBoost ranker, K=1 long, 10% stop, no macro gate. Aggressive."),
    ("CORE", "pt_frontier_dib_flow_both",       "frontier_dib_flow_both",
        "Dollar-Imbalance-Bars BTC+ETH duo. Trigger=BOTH>0. Sharpe-optimal."),
    ("CORE", "pt_frontier_dib_flow_any",        "frontier_dib_flow_any",
        "DIB duo, permissive trigger (BTC OR ETH>0)."),
    ("CORE", "pt_frontier_dib_flow_avg",        "frontier_dib_flow_avg",
        "DIB duo, averaged trigger."),
    ("CORE", "pt_meta_combined",                "prod_meta_combined",
        "Meta-labeled swing book, S-tier Rank 1 champion."),
    ("CORE", "pt_meta_full",                    "prod_meta_full",
        "Meta-labeled full (swing+short), A-tier Rank 3."),
    ("CORE", "pt_frontier_stable_flow",         "frontier_stable_flow",
        "USDT supply shock overlay, K=10 H=5 mom_select."),
    ("CORE", "pt_frontier_etf_flow",            "frontier_etf_flow",
        "BTC+ETH ETF inflow shock overlay, ethbtc_duo_shock K=2 H=2."),

    # ============ ALTERNATE (same strat, different sizing) ============
    ("ALTERNATE", "pt_xsec_K5_5_FULL_dneut_enh",   "xsec_K5_5_FULL_dneut_enh",
        "xsec with enhanced gate experimental variant."),
    ("ALTERNATE", "pt_xgb_K3_long_WEALTH40_enh",   "xgb_K3_long_WEALTH40_enh",
        "xgb_K3 with enhanced regime gate."),
    ("ALTERNATE", "pt_cat_K1_stop_no_macro_enh",   "cat_K1_stop_no_macro_enh",
        "cat_K1 with enhanced logic."),
    ("ALTERNATE", "pt_xsec_K5_5_FULL_dneut_v9meta", "xsec_K5_5_FULL_dneut_v9meta",
        "xsec gated by v9 adaptive-TB meta-labeler (strict threshold)."),
    ("ALTERNATE", "pt_xsec_K5_5_FULL_dneut_v9meta_loose", "xsec_K5_5_FULL_dneut_v9meta_loose",
        "xsec gated by v9 meta-labeler, loose threshold 0.35."),
    # Newly aligned paper_trader_v2 profiles (2026-04-23 alignment run)
    ("ALTERNATE", "pt_meta_combined_k15",        "prod_meta_combined_k15 (aligned)",
        "Rank-1 champion with k=1.5 Kelly sizing (post-fix aligned 2026-04-23)."),
    ("ALTERNATE", "pt_meta_full_k15",            "prod_meta_full_k15 (aligned)",
        "Rank-3 with k=1.5 Kelly sizing (post-fix aligned)."),
    ("ALTERNATE", "pt_meta_medium",              "prod_meta_medium (aligned)",
        "Tier-A Rank-5 meta-labeled medium (trend) book."),
    ("ALTERNATE", "pt_meta_short",               "prod_meta_short (aligned)",
        "Short-only meta-labeler variant."),

    # ============ DEAD (officially dropped or confirmed negative in current regime) ============
    ("DEAD", "pt_perp_dna",                     "perp_dna_long_short",
        "Bidirectional PERP carry. Worked 2022-2023, -13% in 2025-2026 bull."),
    ("DEAD", "pt_regime_routed",                "regime_routed_full",
        "Regime-overlay router. -16% in 2025-2026 bull, DD -24%."),

    # ============ LEGACY (frozen / historical reference points) ============
    ("LEGACY", "frozen_prod_meta_combined",      "frozen_prod_meta_combined",
        "Frozen 2026-Q1 snapshot of prod_meta_combined."),
    ("LEGACY", "frozen_prod_meta_full",          "frozen_prod_meta_full",
        "Frozen 2026-Q1 snapshot of prod_meta_full."),
    ("LEGACY", "frozen_prod_meta_medium",        "frozen_prod_meta_medium",
        "Frozen 2026-Q1 snapshot of prod_meta_medium."),
    ("LEGACY", "meta13_combined",                "meta13_combined",
        "Initial meta13 combined (pre-reconciliation)."),
    ("LEGACY", "meta13_full",                    "meta13_full"  ,
        "Initial meta13 full."),
    ("LEGACY", "meta13_medium",                  "meta13_medium",
        "Initial meta13 medium."),

    # ============ BASE (newly aligned 2026-04-23 — non-meta profiles) ============
    ("BASE", "pt_prod_combined",                 "prod_combined (aligned)",
        "Base combined swing+short book (post-fix simulator)."),
    ("BASE", "pt_prod_swing",                    "prod_swing (aligned)",
        "Base swing book only."),
    ("BASE", "pt_prod_short",                    "prod_short (aligned)",
        "Base short book only."),
    ("BASE", "pt_prod_medium",                   "prod_medium (aligned)",
        "Base medium (trend) book."),
    ("BASE", "pt_prod_trend",                    "prod_trend (aligned)",
        "Base trend book."),
    ("BASE", "pt_prod_floor_combined",           "prod_floor_combined (aligned)",
        "Floor sizing combined."),
    ("BASE", "pt_prod_floor_medium",             "prod_floor_medium (aligned)",
        "Floor sizing medium."),
    ("BASE", "pt_prod_size_xsec",                "prod_size_xsec (aligned)",
        "Size13 xsec sizing variant."),
    ("BASE", "pt_multi_stream",                  "multi_stream (aligned)",
        "Multi-stream variant."),
    ("BASE", "pt_capture_v1_cons",               "capture_v1_conservative (aligned)",
        "Capture v1 conservative profile."),
    ("BASE", "pt_subday_combined",               "subday_combined (aligned)",
        "Subday bundled (expected degraded)."),
    ("BASE", "pt_v7sd_combined",                 "prod_v7sd_combined (aligned)",
        "v7 signal-direction combined."),
    ("BASE", "pt_te_leadlag",                    "prod_te_leadlag (aligned)",
        "Time-shuffled lead-lag."),
    ("BASE", "pt_multi_prod_combined",           "multi_prod_combined (aligned)",
        "Rank-8 multi-stream combined (post-fix)."),
    ("BASE", "pt_kelly_vol_break",               "kelly_vol_break (aligned)",
        "Rank-14 vol-break with Kelly sizing (post-fix)."),
    # ============ RELATED (other ranked-tier entries from yaml) ============
    ("RELATED", "size13_prod_floor_combined",    "size13_prod_floor_combined",
        "SUSPECT (MtM bug-dependent equity, per-trade negative). Do NOT deploy."),
    ("RELATED", "size13_prod_combined",          "size13_prod_combined",
        "Rank-7 singleton. De-rated post simulator fix."),
    ("RELATED", "size13_prod_floor_medium",      "size13_prod_floor_medium",
        "Rank-9 singleton."),
    ("RELATED", "size13_prod_size_medium",       "size13_prod_size_medium",
        "Rank-10 singleton."),
    ("RELATED", "size13_v4_prod_full",           "size13_v4_prod_full",
        "Rank-11 singleton."),
    ("RELATED", "multi_v4_prod_full",            "multi_v4_prod_full",
        "Rank-12 singleton."),
    ("RELATED", "multi_prod_combined",           "multi_prod_combined",
        "Rank-8 multi-variant combined."),
    ("RELATED", "kelly_vol_break",               "kelly_vol_break",
        "Rank-14 vol-break with Kelly sizing."),
    ("RELATED", "prod_jan2026_primary",          "prod_jan2026_primary",
        "Historical Jan-2026 primary prod run."),
    ("RELATED", "prod_live_maker",               "prod_live_maker",
        "Historical live-maker prod run."),

    # ============ OPS-ONLY (cost / validation runs, reference only) ============
    ("OPS", "cost_val_maker",                    "cost_val_maker",
        "Maker cost-model validation run."),
    ("OPS", "cost_val_taker",                    "cost_val_taker",
        "Taker cost-model validation run."),
    ("OPS", "fixed_maker",                       "fixed_maker",
        "Post-MtM-fix maker validation."),
    ("OPS", "fixed_taker",                       "fixed_taker",
        "Post-MtM-fix taker validation."),
]

WINDOW_START = "2026-04-01"
WINDOW_END = "2026-04-22"


def metrics_from_csv(fp: Path, window_start: str | None = None,
                     window_end: str | None = None) -> dict:
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        return {"status": "UNREADABLE", "err": str(e)}
    if df.empty or "date" not in df.columns or "total_equity" not in df.columns:
        return {"status": "BAD_SCHEMA"}
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Group by date (paper_trader_v2 writes multiple bars/day; take last)
    df = df.groupby(df["date"].dt.date).last().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if len(df) < 10:
        return {"status": "TOO_SHORT", "n": len(df)}
    eq = df["total_equity"].values.astype(float)
    if eq[0] <= 0:
        return {"status": "ZERO_START"}
    # rebase to 10k for cross-seed comparison
    eq_r = eq / eq[0] * 10000.0
    r = np.diff(eq_r) / eq_r[:-1]
    total_ret = (eq_r[-1] / 10000.0 - 1) * 100
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days or 1
    cagr = ((eq_r[-1] / 10000.0) ** (365.0 / days) - 1) * 100
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
    cm = np.maximum.accumulate(eq_r)
    dd = ((eq_r - cm) / cm).min() * 100
    calmar = cagr / abs(dd) if dd < 0 else float("inf")
    log_r = np.log1p(r)
    g_annual = log_r.mean() * 365

    result = {
        "status": "OK",
        "n_days": len(df),
        "first_date": df["date"].iloc[0].date().isoformat(),
        "last_date": df["date"].iloc[-1].date().isoformat(),
        "end_nav_on_10k": float(eq_r[-1]),
        "total_ret_pct": float(total_ret),
        "cagr_pct": float(cagr),
        "sharpe": float(sharpe),
        "max_dd_pct": float(dd),
        "calmar": float(calmar) if calmar != float("inf") else None,
        "g_annual_nats": float(g_annual),
    }

    # Window slice
    if window_start and window_end:
        ws = pd.to_datetime(window_start)
        we = pd.to_datetime(window_end)
        sub = df[(df["date"] >= ws) & (df["date"] <= we)]
        if len(sub) >= 2:
            sub_eq = sub["total_equity"].values.astype(float)
            sub_r = np.diff(sub_eq) / sub_eq[:-1]
            wret = (sub_eq[-1] / sub_eq[0] - 1) * 100
            wsh = sub_r.mean() / sub_r.std() * np.sqrt(365) if sub_r.std() > 0 else 0.0
            cm = np.maximum.accumulate(sub_eq)
            wdd = ((sub_eq - cm) / cm).min() * 100
            result["win_days"] = len(sub)
            result["win_ret_pct"] = float(wret)
            result["win_sharpe"] = float(wsh)
            result["win_dd_pct"] = float(wdd)
        else:
            result["win_days"] = len(sub)
    return result


def main():
    today = datetime.now(timezone.utc).date()
    out_dir = ROOT / "logs" / "deployment" / str(today)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat, seed, display, notes in CANDIDATES:
        fp = SEEDS / seed / "daily_snapshot.csv"
        if not fp.exists():
            rows.append({"category": cat, "seed": seed, "display": display,
                         "status": "NO_SNAPSHOT", "notes": notes})
            continue
        m = metrics_from_csv(fp, WINDOW_START, WINDOW_END)
        rows.append({"category": cat, "seed": seed, "display": display,
                     "notes": notes, **m})

    df = pd.DataFrame(rows)
    valid = df[df.get("status") == "OK"].copy()
    csv_path = out_dir / "prod_candidates_ranking.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Rank by composite: 0.5*sharpe + 0.3*cagr_normalized + 0.2*calmar_normalized
    # Simpler: order by sharpe descending; flag calmar / DD.
    valid = valid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 130)
    print(f"TOP 10 BY FULL-HISTORY SHARPE  (window: {WINDOW_START} -> {WINDOW_END})")
    print("=" * 130)
    hdr = f"{'rank':>4} {'category':<10} {'display':<36} {'days':>5} " \
          f"{'cagr%':>8} {'Sh':>6} {'DD%':>7} {'calmar':>7} {'win_d':>5} {'winR%':>7} {'winSh':>6}"
    print(hdr)
    print("-" * 130)
    for i, r in valid.head(10).iterrows():
        print(f"{i+1:>4} {r['category']:<10} {r['display']:<36} {int(r['n_days']):>5} "
              f"{r['cagr_pct']:>+7.2f} {r['sharpe']:>+5.2f} {r['max_dd_pct']:>+6.2f} "
              f"{r.get('calmar') or float('nan'):>+6.2f} "
              f"{int(r.get('win_days', 0)):>5} "
              f"{r.get('win_ret_pct', float('nan')):>+6.2f} "
              f"{r.get('win_sharpe', float('nan')):>+5.2f}")

    print("\n" + "=" * 130)
    print(f"ALL OTHERS (ranked 11+) -- in order of Sharpe")
    print("=" * 130)
    print(hdr)
    print("-" * 130)
    for i, r in valid.iloc[10:].iterrows():
        print(f"{i+1:>4} {r['category']:<10} {r['display']:<36} {int(r['n_days']):>5} "
              f"{r['cagr_pct']:>+7.2f} {r['sharpe']:>+5.2f} {r['max_dd_pct']:>+6.2f} "
              f"{r.get('calmar') or float('nan'):>+6.2f} "
              f"{int(r.get('win_days', 0)):>5} "
              f"{r.get('win_ret_pct', float('nan')):>+6.2f} "
              f"{r.get('win_sharpe', float('nan')):>+5.2f}")

    # Missing / invalid
    missing = df[df["status"] != "OK"]
    if len(missing):
        print("\n" + "=" * 130)
        print(f"MISSING / INVALID  ({len(missing)} entries)")
        print("=" * 130)
        for _, r in missing.iterrows():
            print(f"  [{r['category']:<10}] {r['display']:<36}  {r.get('status','?'):<15}  ({r['seed']})")

    # Write markdown summary
    md_path = out_dir / "prod_candidates_ranking.md"
    with open(md_path, "w") as f:
        f.write(f"# PROD-Candidate Ranking — {today}\n\n")
        f.write(f"Window: {WINDOW_START} -> {WINDOW_END}  |  Capital baseline $10K rebased per seed.\n\n")
        f.write(f"Count: {len(valid)} valid / {len(df)} total.\n\n")
        f.write("## Top 10 by full-history Sharpe\n\n")
        f.write("| # | Cat | Strategy | Days | CAGR % | Sharpe | DD % | Calmar | Win d | Win ret% | Win Sh | Notes |\n")
        f.write("|--:|-----|----------|----:|------:|------:|-----:|------:|-----:|--------:|-------:|-------|\n")
        for i, r in valid.head(10).iterrows():
            f.write(f"| {i+1} | {r['category']} | `{r['display']}` | {int(r['n_days'])} | "
                    f"{r['cagr_pct']:+.2f} | {r['sharpe']:+.2f} | {r['max_dd_pct']:+.2f} | "
                    f"{r.get('calmar') if r.get('calmar') is not None else 'inf'} | "
                    f"{int(r.get('win_days',0))} | "
                    f"{r.get('win_ret_pct', float('nan')):+.2f} | {r.get('win_sharpe', float('nan')):+.2f} | "
                    f"{r['notes']} |\n")
        f.write(f"\n## Ranks 11-{len(valid)}\n\n")
        f.write("| # | Cat | Strategy | Days | CAGR % | Sharpe | DD % | Calmar | Win d | Win ret% | Win Sh | Notes |\n")
        f.write("|--:|-----|----------|----:|------:|------:|-----:|------:|-----:|--------:|-------:|-------|\n")
        for i, r in valid.iloc[10:].iterrows():
            f.write(f"| {i+1} | {r['category']} | `{r['display']}` | {int(r['n_days'])} | "
                    f"{r['cagr_pct']:+.2f} | {r['sharpe']:+.2f} | {r['max_dd_pct']:+.2f} | "
                    f"{r.get('calmar') if r.get('calmar') is not None else 'inf'} | "
                    f"{int(r.get('win_days',0))} | "
                    f"{r.get('win_ret_pct', float('nan')):+.2f} | {r.get('win_sharpe', float('nan')):+.2f} | "
                    f"{r['notes']} |\n")
        if len(missing):
            f.write(f"\n## Missing / invalid ({len(missing)})\n\n")
            for _, r in missing.iterrows():
                f.write(f"- **{r['display']}** [{r['category']}]: {r.get('status','?')} ({r['seed']}) — {r['notes']}\n")
    print(f"\n[saved] {md_path}")


if __name__ == "__main__":
    main()
