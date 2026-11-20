"""V6/V7: Add deployed CONFIRMATION GATE (>=2 cells active per asset) to V5.

V5 replicated half of the deployed architecture; V6 adds the missing piece.
V7 = V6 + multi-TF gate (>=2 cadences with at least one active cell).

If V6 ~~ +91% (deployed baseline), my harness is validated.
If V7 > V6, multi-TF on top of confirmation gate is the breakthrough.
If V7 < V6, multi-TF doesn't help and we ship V6 == deployed.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from multi_tf_v5_deployed_cells import (
    load_inputs, load_universe, compute_ma_active, walk_forward_exit,
    build_cell_active_matrix, build_per_day_cell_active, collapse_subday_to_daily,
    metrics, OOS_START, OOS_END,
    K_MAX, PER_ASSET_CAP, TOTAL_DEPLOY_CAP, COST_RT,
    REGIME_TIER_MULT, BUCKET_VOL_MULT, REGIME_QUALIFIES,
)

OUT_DIR = ROOT / "runs" / "audit" / "MULTI_TF_BREAKTHROUGH_2026_05_20"


def simulate_v67(
    own_lookup, asset_to_bucket,
    active_1d_full, active_4h_full, active_1h_full,
    chim_by_asset,
    require_confirmation_gate: bool = True,
    require_multi_tf: bool = False,
    mode: str = "signal", rng_seed: int = 42,
):
    """Same as V5 but:
      - require_confirmation_gate: at least 2 cells in the asset's 1d cell-list
        must be active today (the deployed fix1 gate).
      - require_multi_tf: at least one cell active on 1d AND on 4h (cadence
        confluence).
    """
    rng = np.random.default_rng(rng_seed)
    portfolio = 1.0
    cash = 1.0
    open_positions: list[dict] = []
    trades: list[dict] = []
    daily_records: list[dict] = []

    # Median cell sharpe for normalization
    all_sharpes = []
    for asset, per_day in active_1d_full.items():
        for d, cs in per_day.items():
            for c in cs:
                if c["sharpe"] > 0:
                    all_sharpes.append(c["sharpe"])
    median_sharpe = float(np.median(all_sharpes)) if all_sharpes else 0.1

    cur = OOS_START
    cal_dates = []
    while cur <= OOS_END:
        cal_dates.append(cur)
        cur += timedelta(days=1)

    for sim_date in cal_dates:
        sim_dt = pd.Timestamp(sim_date).normalize()

        # Close positions
        new_open = []
        for pos in open_positions:
            chim = chim_by_asset.get(pos["asset"])
            if chim is None:
                new_open.append(pos)
                continue
            fwd = chim[(chim["date"] > pos["entry_date"]) & (chim["date"] <= sim_dt)]
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd["close"].values]
            if not fwd_closes:
                new_open.append(pos)
                continue
            r, d_held, reason = walk_forward_exit(pos["entry_price"], fwd_closes)
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * r
                cash += pos["bet_size"] + pnl
                trades.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": r,
                    "exit_reason": reason,
                    "active_cells_count": pos.get("active_cells_count", 1),
                    "cadences_agree": pos.get("cadences_agree", 1),
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset in asset_to_bucket.keys():
            if asset in open_assets:
                continue
            active_1d_cells = active_1d_full.get(asset, {}).get(sim_dt, [])
            if not active_1d_cells:
                continue

            # CONFIRMATION GATE
            if require_confirmation_gate and len(active_1d_cells) < 2:
                continue

            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None:
                continue
            qualifying_cells = [c for c in active_1d_cells if own_r in c["regime_qualifies"]]
            if not qualifying_cells:
                continue
            # Additional confirmation: >=2 cells that QUALIFY for current regime
            if require_confirmation_gate and len(qualifying_cells) < 2:
                continue

            # Multi-TF gate
            cadences_agree = 1  # 1d already active
            if active_4h_full.get(asset, {}).get(sim_dt):
                cadences_agree += 1
            if active_1h_full.get(asset, {}).get(sim_dt):
                cadences_agree += 1
            if require_multi_tf and cadences_agree < 2:
                continue

            best_cell = sorted(qualifying_cells, key=lambda c: (c["tier_priority"], -c["sharpe"]))[0]
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = best_cell["sharpe"] * tier_mult * vol_mult * quality_mult
            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "cell_sharpe": best_cell["sharpe"],
                "deploy_score": deploy_score,
                "active_cells_count": len(qualifying_cells),
                "cadences_agree": cadences_agree,
            })

        if not candidates:
            portfolio = cash + sum(p["bet_size"] for p in open_positions)
            daily_records.append({"date": sim_date, "portfolio_value": portfolio,
                                  "n_open": len(open_positions)})
            continue

        cand_df = pd.DataFrame(candidates)
        if mode in ("best", "worst"):
            fwd_rets = []
            for _, r in cand_df.iterrows():
                chim = chim_by_asset.get(r["asset"])
                if chim is None:
                    fwd_rets.append(np.nan)
                    continue
                sub_oos = chim[chim["date"] >= sim_dt].head(6)
                if len(sub_oos) < 6:
                    fwd_rets.append(np.nan)
                    continue
                ep = float(sub_oos.iloc[0]["close"])
                fp = float(sub_oos.iloc[5]["close"])
                if ep > 0 and np.isfinite(ep) and np.isfinite(fp):
                    fwd_rets.append(fp / ep - 1)
                else:
                    fwd_rets.append(np.nan)
            cand_df["fwd_5d"] = fwd_rets
            cand_df["rank_sig"] = cand_df["fwd_5d"].fillna(-999 if mode == "best" else 999)
            asc = (mode == "worst")
            cand_df = cand_df.sort_values("rank_sig", ascending=asc)
        elif mode == "signal":
            cand_df = cand_df.sort_values(
                ["cadences_agree", "active_cells_count", "deploy_score"],
                ascending=[False, False, False])
        elif mode == "random":
            cand_df["rng"] = rng.random(len(cand_df))
            cand_df = cand_df.sort_values("rng")
        else:
            raise ValueError(mode)

        cand_df = cand_df.drop_duplicates(subset="asset", keep="first")

        for _, c in cand_df.iterrows():
            if len(open_positions) >= K_MAX:
                break
            current_deploy = sum(p["bet_size"] / portfolio for p in open_positions)
            if current_deploy >= TOTAL_DEPLOY_CAP:
                break
            bet = PER_ASSET_CAP * portfolio
            if (current_deploy + bet / portfolio) > TOTAL_DEPLOY_CAP:
                bet = (TOTAL_DEPLOY_CAP - current_deploy) * portfolio
            if bet <= 0:
                break
            if cash < bet:
                break
            chim = chim_by_asset.get(c["asset"])
            if chim is None:
                continue
            drow = chim[chim["date"] == sim_dt]
            if not len(drow):
                continue
            ep = float(drow.iloc[0]["close"])
            if not np.isfinite(ep) or ep <= 0:
                continue
            cash -= bet
            open_positions.append({
                "asset": c["asset"], "entry_date": sim_dt,
                "entry_price": ep, "bet_size": bet,
                "active_cells_count": c["active_cells_count"],
                "cadences_agree": c["cadences_agree"],
            })

        omtm = 0.0
        for p in open_positions:
            chim = chim_by_asset.get(p["asset"])
            if chim is None:
                omtm += p["bet_size"]
                continue
            av = chim[chim["date"] <= sim_dt]
            if not len(av):
                omtm += p["bet_size"]
                continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp):
                omtm += p["bet_size"]
                continue
            omtm += p["bet_size"] * (cp / p["entry_price"])
        portfolio = cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio,
                              "n_open": len(open_positions)})

    return pd.DataFrame(daily_records), pd.DataFrame(trades)


def main():
    print("=" * 78)
    print("V6/V7: Confirmation gate (>=2 cells) + optional multi-TF")
    print("=" * 78)
    profile, winners, fallback, own_regime = load_inputs()
    asset_to_bucket = load_universe()
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                   for _, r in own_regime.iterrows()}

    chim_dir_1d = ROOT / "data" / "processed" / "chimera" / "1d"
    chim_dir_4h = ROOT / "data" / "processed" / "chimera" / "4h"
    chim_dir_1h = ROOT / "data" / "processed" / "chimera" / "1h"
    cells_1d = build_cell_active_matrix(asset_to_bucket, profile, winners, fallback, "1d", chim_dir_1d)
    cells_4h = build_cell_active_matrix(asset_to_bucket, profile, winners, fallback, "4h", chim_dir_4h)
    cells_1h = build_cell_active_matrix(asset_to_bucket, profile, winners, fallback, "1h", chim_dir_1h)

    active_1d = build_per_day_cell_active(cells_1d)
    active_4h = collapse_subday_to_daily(cells_4h)
    active_1h = collapse_subday_to_daily(cells_1h)

    chim_by_asset = {a: chim for a, (chim, _) in cells_1d.items()}

    window = (OOS_END - OOS_START).days
    all_res: dict = {}

    variants = [
        ("V6_confirmation_gate_only",     True,  False),
        ("V7_confirmation_AND_multi_tf",  True,  True),
        ("V5A_NO_gates",                   False, False),  # baseline replication
    ]
    for vname, conf, mtf in variants:
        print(f"\n--- {vname}  confirmation={conf}  multi_tf={mtf} ---")
        var_res = {}
        for mode in ["best", "signal", "random", "worst"]:
            daily, trades = simulate_v67(
                own_lookup, asset_to_bucket,
                active_1d, active_4h, active_1h, chim_by_asset,
                require_confirmation_gate=conf, require_multi_tf=mtf,
                mode=mode,
            )
            m = metrics(daily, trades, window)
            var_res[mode] = m
            print(f"  {mode:7s}-K: NAV={m['total_pct']:+8.2f}%  Sortino={m['sortino']:+.2f}  "
                  f"DD={m['max_dd_pct']:+.1f}%  n={m['n_trades']}  win={m['win_rate_pct']:.1f}%  "
                  f"7d>=5.25%: {m['pct_7d_above_5_25']:.1f}%")
        all_res[vname] = var_res

    (OUT_DIR / "v67_oos_results.json").write_text(json.dumps(all_res, indent=2, default=str))
    print(f"\nSaved to {OUT_DIR / 'v67_oos_results.json'}")


if __name__ == "__main__":
    main()
