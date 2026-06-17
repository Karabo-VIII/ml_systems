"""V8: Smart-grid cells + V6 architecture (confirmation gate + 3-level ranker).

V6 used the deployed (brute-force, per-asset-mined) cells and got +118%. V8
swaps in smart-grid cells (Fibonacci + golden + log + decorrelation) while
keeping the V6 architecture.

WHY: smart-grid cells found PEPE Fibonacci pairs (e.g., SMA(5,13)) that the
brute-force adjacent-period grid missed. If V6 architecture is the lever
and smart-grid is the better substrate, V8 should match or beat V6's +118%.

INPUTS:
  - per_asset_smart_grid_profile.parquet from master_csv_smart_rebuild.py
    (top 3 cells per asset per cadence, n_train>=10)
  - asset_own_regime_panel.parquet (for regime tier multiplier; smart-grid
    cells are treated as ALL_WEATHER, no regime gating)

OUTPUT:
  runs/audit/MASTER_CSV_SMART_REBUILD_2026_05_20/V8_OOS_RESULTS.json
  runs/audit/MASTER_CSV_SMART_REBUILD_2026_05_20/V8_REPORT.md
"""
from __future__ import annotations
import sys
import json
import glob
from pathlib import Path
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from union_oos_fix_variants import (
    OOS_START, OOS_END, K_MAX_BASE, PER_ASSET_CAP, TOTAL_DEPLOY_CAP, COST_RT,
    EXIT_HOLD_MAX, EXIT_STOP_BASE, EXIT_TRAIL_ARM, EXIT_TRAIL_DROP,
    REGIME_TIER_MULT, BUCKET_VOL_MULT,
    walk_forward_exit, compute_ma, load_chimera_1d, load_universe,
)

SMART_PROFILE_PATH = ROOT / "runs" / "audit" / "MASTER_CSV_SMART_REBUILD_2026_05_20" / "per_asset_smart_grid_profile.parquet"
OWN_REGIME = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
OUT_DIR = ROOT / "runs" / "audit" / "MASTER_CSV_SMART_REBUILD_2026_05_20"


def build_smart_grid_cells(asset_to_bucket, chimera, smart_profile_df):
    """Build per-asset cells from smart-grid profile. All cells treated as
    ALL_WEATHER (no regime gating)."""
    asset_cells = {}
    grouped = smart_profile_df[smart_profile_df["cadence"] == "1d"].groupby("asset")
    for asset, sub in grouped:
        if asset not in chimera:
            continue
        chim = chimera[asset]
        closes = chim["close"].values
        cells = []
        for _, c in sub.iterrows():
            mf, ml = compute_ma(closes, c["ma_type"], int(c["fast"]), int(c["slow"]))
            cells.append({
                "source": "smart_grid",
                "regime_qualifies": {"bull", "chop", "bear", "crash"},  # ALL_WEATHER
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "sharpe": float(c.get("sharpe_train", 0)),
                "tier_priority": 1,
                "active": mf > ml,
            })
        if cells:
            asset_cells[asset] = (chim, cells)
    return asset_cells


def load_subday_smart_active(asset_to_bucket, smart_profile_df, cadence: str) -> dict:
    """Per-day boolean: 'any of asset's smart-grid cells on this cadence is
    active today'."""
    chim_subday = {}
    chim_dir = ROOT / "data" / "processed" / "chimera" / cadence
    for f in glob.glob(str(chim_dir / f"*_v51_chimera_{cadence}_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["dt"].dt.normalize()
        df = df.sort_values("dt").reset_index(drop=True)
        chim_subday[sym] = df

    out = {}
    grouped = smart_profile_df[smart_profile_df["cadence"] == cadence].groupby("asset")
    for asset, sub in grouped:
        if asset not in chim_subday:
            continue
        chim = chim_subday[asset]
        closes = chim["close"].values
        date_arr = chim["date"].values
        per_day = {}
        for _, c in sub.iterrows():
            mf, ml = compute_ma(closes, c["ma_type"], int(c["fast"]), int(c["slow"]))
            active = mf > ml
            n = min(len(active), len(date_arr))
            if n == 0:
                continue
            tmp = pd.DataFrame({"date": date_arr[:n], "active": active[:n]})
            for d, v in tmp.groupby("date")["active"].any().items():
                if bool(v):
                    per_day.setdefault(pd.Timestamp(d).normalize(), True)
        out[asset] = per_day
    return out


def simulate_v8(asset_cells_1d, sub_4h_active, sub_1h_active,
                own_lookup, asset_to_bucket, use_multi_tf: bool = True):
    """V6 architecture: confirmation gate (>=2 cells active) + 3-level ranker
    (cadences_agree, confluence, deploy_score). Smart-grid cells, no regime
    gating (ALL_WEATHER).
    """
    median_sharpe = float(np.median([c["sharpe"] for _, (_, cs) in asset_cells_1d.items()
                                       for c in cs if c["sharpe"] > 0])) or 0.1
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    cur = OOS_START
    cal_dates = []
    while cur <= OOS_END:
        cal_dates.append(cur)
        cur += timedelta(days=1)

    for sim_date in cal_dates:
        sim_dt = pd.Timestamp(sim_date)

        new_open = []
        for pos in open_positions:
            chim, _ = asset_cells_1d.get(pos["asset"], (None, []))
            if chim is None:
                new_open.append(pos)
                continue
            fwd = chim[(chim["date"] > pos["entry_date"]) & (chim["date"] <= sim_dt)]
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd["close"].values]
            if not fwd_closes:
                new_open.append(pos)
                continue
            rret, d_held, reason = walk_forward_exit(
                pos["entry_price"], fwd_closes,
                hold_max=EXIT_HOLD_MAX, stop=EXIT_STOP_BASE,
                trail_arm=EXIT_TRAIL_ARM, trail_drop=EXIT_TRAIL_DROP, cost=COST_RT,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": rret,
                    "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset, (chim, cells) in asset_cells_1d.items():
            if asset in open_assets:
                continue
            date_mask = chim["date"] == sim_dt
            if not date_mask.any():
                continue
            idx = int(np.where(date_mask)[0][0])
            qualifying_cells = []
            for cell in sorted(cells, key=lambda c: c["tier_priority"]):
                if idx >= len(cell["active"]):
                    continue
                if not cell["active"][idx]:
                    continue
                qualifying_cells.append(cell)
            # CONFIRMATION GATE: >=2 cells active
            if len(qualifying_cells) < 2:
                continue
            confluence = len(qualifying_cells)
            best_cell = qualifying_cells[0]
            own_r = own_lookup.get((asset, sim_dt), "chop")  # default for tier mult only
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = (best_cell["sharpe"] * tier_mult * vol_mult * quality_mult
                             * (1 + 0.1 * (confluence - 1)))
            if use_multi_tf:
                cad_agree = 1
                if sub_4h_active.get(asset, {}).get(sim_dt):
                    cad_agree += 1
                if sub_1h_active.get(asset, {}).get(sim_dt):
                    cad_agree += 1
            else:
                cad_agree = 1
            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "deploy_score": deploy_score, "confluence": confluence,
                "cadences_agree": cad_agree,
                "tier_mult": tier_mult, "vol_mult": vol_mult, "quality_mult": quality_mult,
                "chim": chim,
            })

        if candidates:
            candidates.sort(key=lambda x: (-x["cadences_agree"], -x["confluence"], -x["deploy_score"]))
            slots_remaining = K_MAX_BASE - len(open_positions)
            budget_remaining = portfolio_value * TOTAL_DEPLOY_CAP - sum(p["bet_size"] for p in open_positions)
            for cand in candidates:
                if slots_remaining <= 0:
                    break
                if budget_remaining <= 0.001 * portfolio_value:
                    break
                base = (TOTAL_DEPLOY_CAP / K_MAX_BASE) * portfolio_value
                target = base * cand["tier_mult"] * cand["vol_mult"] * cand["quality_mult"]
                hard_cap = portfolio_value * PER_ASSET_CAP
                actual = min(target, hard_cap, budget_remaining, available_cash)
                if actual < 0.005 * portfolio_value:
                    continue
                today_row = cand["chim"][cand["chim"]["date"] == sim_dt]
                if today_row.empty:
                    continue
                ep = float(today_row.iloc[0]["close"])
                if ep <= 0 or not np.isfinite(ep):
                    continue
                available_cash -= actual
                budget_remaining -= actual
                slots_remaining -= 1
                open_positions.append({
                    "asset": cand["asset"], "entry_date": sim_dt,
                    "entry_price": ep, "bet_size": actual,
                })

        omtm = 0
        for pos in open_positions:
            chim, _ = asset_cells_1d.get(pos["asset"], (None, []))
            if chim is None:
                omtm += pos["bet_size"]
                continue
            av = chim[chim["date"] <= sim_dt]
            if not len(av):
                omtm += pos["bet_size"]
                continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp):
                omtm += pos["bet_size"]
                continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio_value = available_cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio_value,
                              "n_open": len(open_positions)})

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)
    pv = daily_df["portfolio_value"].values
    total_pct = float((pv[-1] / pv[0] - 1) * 100)
    window_days = (OOS_END - OOS_START).days
    ann_pct = float(((1 + total_pct / 100) ** (365 / max(window_days, 1)) - 1) * 100)
    drs = pv[1:] / pv[:-1] - 1
    sortino = float((drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs < 0].std() > 0 else 0)
    sharpe = float((drs.mean() / drs.std() * np.sqrt(252)) if drs.std() > 0 else 0)
    cum = pv / pv[0]
    cm = np.maximum.accumulate(cum)
    max_dd = float(((cum / cm - 1) * 100).min())
    calmar = float(ann_pct / abs(max_dd)) if max_dd != 0 else 0
    win = float((trades_df["realized_ret"] > 0).mean() * 100) if len(trades_df) else 0
    if len(pv) >= 7:
        r7 = (pv[7:] / pv[:-7] - 1) * 100
        p7 = float((r7 >= 5.25).mean() * 100)
    else:
        p7 = 0
    return {
        "total_pct": total_pct, "ann_pct": ann_pct,
        "sortino": sortino, "sharpe": sharpe,
        "max_dd_pct": max_dd, "calmar": calmar,
        "n_trades": int(len(trades_df)), "win_rate_pct": win,
        "pct_7d_above_5_25": p7,
    }


def main():
    print("=" * 78)
    print("V8: Smart-grid cells + V6 architecture")
    print("=" * 78)
    if not SMART_PROFILE_PATH.exists():
        print(f"ERROR: {SMART_PROFILE_PATH} not found. Run master_csv_smart_rebuild.py first.")
        return
    smart_profile = pd.read_parquet(SMART_PROFILE_PATH)
    print(f"  smart-grid profile cells: {len(smart_profile)}")

    asset_to_bucket = load_universe()
    own_regime = pl.read_parquet(OWN_REGIME).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                   for _, r in own_regime.iterrows()}
    chim_1d = load_chimera_1d()

    asset_cells_1d = build_smart_grid_cells(asset_to_bucket, chim_1d, smart_profile)
    print(f"  built 1d smart-grid cells for {len(asset_cells_1d)} assets")

    print("  loading 4h smart-grid active map...")
    sub_4h_active = load_subday_smart_active(asset_to_bucket, smart_profile, "4h")
    print(f"    {len(sub_4h_active)} assets")
    print("  loading 1h smart-grid active map...")
    sub_1h_active = load_subday_smart_active(asset_to_bucket, smart_profile, "1h")
    print(f"    {len(sub_1h_active)} assets")

    print("\nRunning V8 variants...")
    results = {}
    r1 = simulate_v8(asset_cells_1d, sub_4h_active, sub_1h_active, own_lookup, asset_to_bucket, use_multi_tf=True)
    r2 = simulate_v8(asset_cells_1d, sub_4h_active, sub_1h_active, own_lookup, asset_to_bucket, use_multi_tf=False)
    results["V8_smart_grid_3level_multi_tf"] = r1
    results["V8_smart_grid_2level_no_mtf"] = r2
    for name, r in results.items():
        print(f"  {name:35s}: NAV={r['total_pct']:+8.2f}%  Sortino={r['sortino']:+.2f}  "
              f"DD={r['max_dd_pct']:+.1f}%  trades={r['n_trades']}  win={r['win_rate_pct']:.1f}%  "
              f"7d>=5.25%: {r['pct_7d_above_5_25']:.1f}%")

    (OUT_DIR / "V8_OOS_RESULTS.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {OUT_DIR / 'V8_OOS_RESULTS.json'}")

    # Comparison report
    lines = ["# V8 Smart-grid + V6 architecture results\n",
             "\n## Comparison to other variants\n",
             "| Variant | Cells | Ranker | NAV | Sortino | DD | Trades |",
             "|---|---|---|---:|---:|---:|---:|",
             "| Published fix1 (deployed) | brute-force adjacent cousin-set | 1-level deploy_score | +91.40% | +3.65 | -13.25% | 385 |",
             "| fix6_3level_multi_tf (production patch) | brute-force adjacent cousin-set | 3-level (cad,conf,score) | +118.52% | +4.92 | -12.2% | 362 |",
             f"| **V8 smart-grid + 3-level multi-TF** | Fibonacci/golden/log + decorr | 3-level | **{r1['total_pct']:+.2f}%** | **{r1['sortino']:+.2f}** | **{r1['max_dd_pct']:+.2f}%** | {r1['n_trades']} |",
             f"| V8 smart-grid + 2-level no multi-TF | Fibonacci/golden/log + decorr | 2-level (conf, score) | {r2['total_pct']:+.2f}% | {r2['sortino']:+.2f} | {r2['max_dd_pct']:+.2f}% | {r2['n_trades']} |",
             "\n## Interpretation\n",
             "- Smart-grid cells without regime tagging (ALL_WEATHER).",
             "- If V8 beats V6 (+118.52%): smart-grid IS a better substrate.",
             "- If V8 underperforms V6: deployed cousin-set cells already capture the best of the smart grid.",
             ]
    (OUT_DIR / "V8_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'V8_REPORT.md'}")


if __name__ == "__main__":
    main()
