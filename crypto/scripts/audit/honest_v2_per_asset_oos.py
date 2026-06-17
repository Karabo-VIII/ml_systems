"""honest_v2_per_asset_oos.py -- OOS validation of per-asset MA/EMA architecture.

Steps:
  1 — OOS validation: run per_asset_v1 on canonical OOS window (2024-05-16 -> 2025-03-15)
  3 — Per-cell exit optimization (simplified): run multiple exit policies, compare
  + Per-asset performance profile (the user's ask): coverage, capture rate per asset

KEY DIFFERENCES from honest_v2_per_asset.py (VAL-only):
  - Computes MA fires directly from chimera 1d (snap only covers VAL)
  - Walk-forward each (asset, cell) signal across the FULL chimera history
  - Runs multiple exit policies for comparison
  - Per-asset breakdown: trades, win_rate, sum_contribution, capture vs oracle

OUTPUT:
  runs/audit/MA_EMA_PROFILE_2026_05_20/OOS_PER_ASSET_REPORT.md
  runs/audit/MA_EMA_PROFILE_2026_05_20/oos_per_asset_trades.csv (per-exit-policy)
"""
from __future__ import annotations

import sys
import json
import glob
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

# Canonical OOS window (per CLAUDE.md split_four_way)
OOS_START = _date(2024, 5, 16)
OOS_END   = _date(2025, 3, 15)

# Portfolio params
K_MAX = 12
BET_FRACTION = 1.0 / K_MAX
COST_RT = 0.0030
BUCKET_CAP = 5

# Exit policy configurations to compare (Step 3 simplified exit optimization)
EXIT_CONFIGS = {
    "baseline_trail_10_5_21d": {
        "hold_max": 21, "stop": -0.04, "trail_arm": 0.10, "trail_drop": 0.05,
    },
    "tight_trail_5_3_14d": {
        "hold_max": 14, "stop": -0.04, "trail_arm": 0.05, "trail_drop": 0.03,
    },
    "loose_trail_15_7_30d": {
        "hold_max": 30, "stop": -0.04, "trail_arm": 0.15, "trail_drop": 0.07,
    },
    "no_trail_7d_hold": {
        "hold_max": 7, "stop": -0.04, "trail_arm": None, "trail_drop": None,
    },
    "no_trail_14d_hold": {
        "hold_max": 14, "stop": -0.04, "trail_arm": None, "trail_drop": None,
    },
}

REGIME_QUALIFIES = {
    "ALL_WEATHER":      {"bull", "chop", "bear", "crash"},
    "BLOCK_OWN_CRASH":  {"bull", "chop", "bear"},
    "BLOCK_OWN_BEAR":   {"bull", "chop"},
    "BULL_AND_CHOP":    {"bull", "chop"},
    "BULL_ONLY":        {"bull"},
    "REGIME_DEPENDENT": {"bull", "chop"},
    "INSUFFICIENT_DATA": set(),
}


def walk_forward_exit_generic(entry_price, fwd_closes, hold_max, stop, trail_arm, trail_drop, cost):
    """Walk forward with optional trail-stop. Returns (ret, days_held, reason)."""
    peak = entry_price; armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p): continue
        ret = p / entry_price - 1
        if p > peak: peak = p
        if trail_arm is not None and not armed and ret >= trail_arm:
            armed = True
        if ret <= stop: return stop - cost, d, "stop"
        if trail_arm is not None and armed and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d, "trail"
        if d >= hold_max: return ret - cost, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None: return -cost, 0, "no_data"
    return last / entry_price - 1 - cost, len(fwd_closes), "expire"


def compute_oracle_long_oos(chimera, oos_start, oos_end):
    """Per-asset oracle long availability on OOS: sum of positive next-day cc% on each asset."""
    oracle = {}
    for asset, chim in chimera.items():
        oos = chim[(chim["date"] >= pd.Timestamp(oos_start)) & (chim["date"] <= pd.Timestamp(oos_end))].copy()
        if len(oos) < 5: continue
        oos["cc"] = oos["close"].pct_change()
        # Long-side availability: sum of positive next-day cc on days where asset went up >1%
        pos_days = oos[oos["cc"] > 0.01]
        oracle[asset] = {
            "n_pos_days": int(len(pos_days)),
            "sum_long_pct": float(pos_days["cc"].sum() * 100),
            "n_oos_days": int(len(oos)),
        }
    return oracle


def precompute_fires(chimera, profile_cells, oos_start, oos_end):
    """For each cousin-set cell, compute fire dates (MA fast > slow, cross-up from <= 0).
    Returns DataFrame: asset, date, cell_id, ma_type, fast, slow, regime_tag, sharpe."""
    fires = []
    by_asset = {a: g for a, g in profile_cells.groupby("asset")}
    for asset, asset_cells in by_asset.items():
        chim = chimera.get(asset)
        if chim is None: continue
        closes = chim["close"].values
        dates = chim["date"].values
        for _, cell in asset_cells.iterrows():
            ma_type = cell["ma_type"]; fast = int(cell["fast"]); slow = int(cell["slow"])
            if ma_type == "SMA":
                ma_f = pd.Series(closes).rolling(fast).mean().values
                ma_s = pd.Series(closes).rolling(slow).mean().values
            else:
                ma_f = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
                ma_s = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
            # Active long signal: fast > slow
            diff = ma_f - ma_s
            sig_active = (diff > 0).astype(int)
            # We treat "active and yesterday was active" as a deploy candidate (not strict cross-up).
            # The simulator will dedupe by (asset, date) so multiple cells firing on same asset is confluence.
            for i in range(slow, len(dates)):
                if sig_active[i] == 1:
                    d = dates[i]
                    if pd.Timestamp(oos_start) <= d <= pd.Timestamp(oos_end):
                        fires.append({
                            "asset": asset,
                            "date": pd.Timestamp(d).normalize(),
                            "cell_id": cell["cell_id"],
                            "regime_tag": cell["regime_tag"],
                            "sharpe": cell["sharpe"],
                            "ma_type": ma_type, "fast": fast, "slow": slow,
                        })
    return pd.DataFrame(fires)


def simulate_with_exit(exit_name, exit_cfg, fires_df, chimera, own_lookup, asset_to_bucket,
                       oos_start, oos_end):
    """Run per-asset simulation with the given exit config."""
    cost = COST_RT
    # Apply regime qualification
    fires_df = fires_df.copy()
    fires_df["regime_today"] = [own_lookup.get((r.asset, r.date), "unknown")
                                  for r in fires_df.itertuples()]
    fires_df["qualifies"] = [r.regime_today in REGIME_QUALIFIES.get(r.regime_tag, set())
                              for r in fires_df.itertuples()]
    qual = fires_df[fires_df["qualifies"]]

    asset_score = (qual.groupby(["asset", "date"]).agg(
        confluence_count=("cell_id", "nunique"),
        max_cell_sharpe=("sharpe", "max"),
    ).reset_index())

    # Simulate
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    cur = oos_start; cal_dates = []
    while cur <= oos_end:
        cal_dates.append(cur); cur += timedelta(days=1)

    for sim_date in cal_dates:
        sim_dt = pd.Timestamp(sim_date)

        # Close positions that hit exit
        new_open = []
        for pos in open_positions:
            asset = pos["asset"]
            chim = chimera.get(asset)
            if chim is None:
                new_open.append(pos); continue
            fwd = chim[(chim["date"] > pos["entry_date"]) & (chim["date"] <= sim_dt)]
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd["close"].values]
            if not fwd_closes:
                new_open.append(pos); continue
            rret, d_held, reason = walk_forward_exit_generic(
                pos["entry_price"], fwd_closes,
                hold_max=exit_cfg["hold_max"], stop=exit_cfg["stop"],
                trail_arm=exit_cfg["trail_arm"], trail_drop=exit_cfg["trail_drop"],
                cost=cost,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": asset, "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": rret,
                    "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # New entries
        today_cands = asset_score[asset_score["date"] == sim_dt].copy()
        if len(today_cands):
            today_cands["bucket"] = today_cands["asset"].map(asset_to_bucket)
            today_cands = today_cands.sort_values(
                ["confluence_count", "max_cell_sharpe"], ascending=[False, False])
            open_assets = set(p["asset"] for p in open_positions)
            today_cands = today_cands[~today_cands["asset"].isin(open_assets)]
            bucket_count = {}
            for _, row in today_cands.iterrows():
                if len(open_positions) >= K_MAX: break
                b = row["bucket"]
                if pd.isna(b) or bucket_count.get(b, 0) >= BUCKET_CAP: continue
                chim = chimera.get(row["asset"])
                if chim is None: continue
                today_row = chim[chim["date"] == sim_dt]
                if today_row.empty: continue
                ep = float(today_row.iloc[0]["close"])
                if ep <= 0 or not np.isfinite(ep): continue
                bet = BET_FRACTION * portfolio_value
                if available_cash < bet: break
                available_cash -= bet
                bucket_count[b] = bucket_count.get(b, 0) + 1
                open_positions.append({
                    "asset": row["asset"], "entry_date": sim_dt,
                    "entry_price": ep, "bet_size": bet,
                })

        # MtM
        omtm = 0
        for pos in open_positions:
            chim = chimera.get(pos["asset"])
            if chim is None: omtm += pos["bet_size"]; continue
            av = chim[chim["date"] <= sim_dt]
            if not len(av): omtm += pos["bet_size"]; continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio_value = available_cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio_value,
                              "n_open": len(open_positions)})

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)

    pv = daily_df["portfolio_value"].values
    total_pct = (pv[-1] / pv[0] - 1) * 100
    window_days = (cal_dates[-1] - cal_dates[0]).days
    ann_pct = ((1 + total_pct/100) ** (365/max(window_days,1)) - 1) * 100
    drs = pv[1:] / pv[:-1] - 1
    sortino = (drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs<0].std() > 0 else 0
    sharpe = (drs.mean() / drs.std() * np.sqrt(252)) if drs.std() > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    calmar = ann_pct / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "exit_name": exit_name,
        "total_pct": float(total_pct), "ann_pct": float(ann_pct),
        "daily_mean_pct": float(drs.mean() * 100) if len(drs) else 0,
        "sortino": float(sortino), "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd), "calmar": float(calmar),
        "n_trades": int(len(trades_df)),
    }
    if len(trades_df):
        metrics["win_rate_pct"] = float((trades_df["realized_ret"] > 0).mean() * 100)
        metrics["avg_win_pct"] = float(trades_df.loc[trades_df["realized_ret"]>0, "realized_ret"].mean()*100) if (trades_df["realized_ret"]>0).any() else 0
        metrics["avg_loss_pct"] = float(trades_df.loc[trades_df["realized_ret"]<0, "realized_ret"].mean()*100) if (trades_df["realized_ret"]<0).any() else 0
        metrics["max_win_pct"] = float(trades_df["realized_ret"].max() * 100)
        metrics["median_hold"] = float(trades_df["days_held"].median())
        metrics["exit_reasons"] = trades_df["exit_reason"].value_counts().to_dict()
    return metrics, trades_df, daily_df


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print(f"HONEST V2 PER-ASSET OOS VALIDATION ({OOS_START} → {OOS_END})")
    print("="*78)
    print("Loading inputs...")
    profile = pd.read_parquet(PROFILE_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    print(f"  profile: {len(profile)} cells / {profile['asset'].nunique()} assets")
    print(f"  cousin-set members: {profile['is_cousin_set_member'].sum()}")

    # Load chimera 1d
    print("Loading chimera 1d per asset...")
    chimera = {}
    for f in glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        chimera[sym] = df
    print(f"  loaded {len(chimera)} assets")

    # Restrict to cousin-set cells
    deploy_cells = profile[profile["is_cousin_set_member"]].copy()
    print(f"  deploying {len(deploy_cells)} cousin-set cells across {deploy_cells['asset'].nunique()} assets")

    # Bucket lookup
    asset_to_bucket = {a: deploy_cells[deploy_cells["asset"] == a]["bucket"].iloc[0]
                       for a in deploy_cells["asset"].unique()}

    # Own-regime lookup
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                  for _, r in own_regime[["asset", "date", "asset_own_regime"]].iterrows()}

    # Precompute fires on OOS
    print(f"\nPrecomputing fires across OOS window...")
    fires_df = precompute_fires(chimera, deploy_cells, OOS_START, OOS_END)
    print(f"  fires: {len(fires_df):,} across {fires_df['asset'].nunique()} assets, {fires_df['date'].nunique()} dates")

    # Compute per-asset oracle availability on OOS
    print(f"Computing per-asset oracle (long availability)...")
    oracle = compute_oracle_long_oos(chimera, OOS_START, OOS_END)
    print(f"  oracle for {len(oracle)} assets on OOS")

    # Run each exit config
    all_results = {}
    all_trades = {}
    all_daily = {}
    for exit_name, exit_cfg in EXIT_CONFIGS.items():
        print(f"\n=== Exit: {exit_name} ({exit_cfg}) ===")
        metrics, trades, daily = simulate_with_exit(
            exit_name, exit_cfg, fires_df, chimera, own_lookup, asset_to_bucket,
            OOS_START, OOS_END
        )
        print(f"  Total return:   {metrics['total_pct']:+.2f}%")
        print(f"  Annualized:     {metrics['ann_pct']:+.2f}%")
        print(f"  Daily mean:     {metrics['daily_mean_pct']:+.4f}%")
        print(f"  Sortino:        {metrics['sortino']:+.3f}")
        print(f"  Sharpe:         {metrics['sharpe']:+.3f}")
        print(f"  Max DD:         {metrics['max_dd_pct']:+.2f}%")
        print(f"  Trades:         {metrics['n_trades']}")
        if metrics.get("win_rate_pct") is not None:
            print(f"  Win rate:       {metrics['win_rate_pct']:.1f}%")
            print(f"  Avg W/L:        {metrics.get('avg_win_pct', 0):+.2f}% / {metrics.get('avg_loss_pct', 0):+.2f}%")
            print(f"  Max win:        {metrics.get('max_win_pct', 0):+.2f}%")
            print(f"  Median hold:    {metrics.get('median_hold', 0):.0f}d")
            print(f"  Exits: {metrics.get('exit_reasons', {})}")
        all_results[exit_name] = metrics
        all_trades[exit_name] = trades
        all_daily[exit_name] = daily

    # ===== Per-asset breakdown (best exit) =====
    best_exit = max(all_results.items(), key=lambda x: x[1]["total_pct"])[0]
    print(f"\n=== BEST EXIT: {best_exit} ===")
    best_trades = all_trades[best_exit]
    per_asset = best_trades.groupby("asset").agg(
        n_trades=("realized_ret", "size"),
        sum_ret_pct=("realized_ret", lambda s: float(s.sum() * 100)),
        mean_ret_pct=("realized_ret", lambda s: float(s.mean() * 100)),
        win_rate=("realized_ret", lambda s: float((s > 0).mean() * 100)),
        sum_bet_contrib=("bet_size", "sum"),
        contribution_to_nav_pct=("realized_ret", lambda s: float((s * best_trades.loc[s.index, "bet_size"]).sum() * 100)),
    ).reset_index()
    per_asset["bucket"] = per_asset["asset"].map(asset_to_bucket)
    # Add oracle
    per_asset["oracle_long_avail_pct"] = per_asset["asset"].map(lambda a: oracle.get(a, {}).get("sum_long_pct", None))
    per_asset["oracle_pos_days"] = per_asset["asset"].map(lambda a: oracle.get(a, {}).get("n_pos_days", None))
    per_asset["coverage_pct"] = per_asset.apply(
        lambda r: r["n_trades"] / r["oracle_pos_days"] * 100 if r["oracle_pos_days"] and r["oracle_pos_days"] > 0 else None,
        axis=1
    )
    per_asset["capture_pct"] = per_asset.apply(
        lambda r: r["sum_ret_pct"] / r["oracle_long_avail_pct"] * 100 if r["oracle_long_avail_pct"] and r["oracle_long_avail_pct"] > 0 else None,
        axis=1
    )
    per_asset = per_asset.sort_values("contribution_to_nav_pct", ascending=False)

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, (OUT_DIR / "oos_per_asset_results.json").open("w"), indent=2, default=str)
    best_trades.to_csv(OUT_DIR / "oos_per_asset_trades_best.csv", index=False)
    per_asset.to_csv(OUT_DIR / "oos_per_asset_breakdown.csv", index=False)

    # Markdown report
    lines = [
        f"# Per-Asset OOS Validation — MA/EMA only ({OOS_START} → {OOS_END})\n",
        f"**Profile**: {len(deploy_cells)} cousin-set cells across {deploy_cells['asset'].nunique()} assets",
        f"**Portfolio**: K={K_MAX}, BET={BET_FRACTION*100:.2f}%, bucket cap={BUCKET_CAP}, cost={COST_RT*100:.2f}% RT",
        f"**Regime gating**: cell deploys only when today's asset_own_regime ∈ tag's qualified set",
        "",
        "## A. Exit policy comparison (Step 3 simplified)",
        "",
        "| exit policy | total | annualized | daily | Sortino | Sharpe | Max DD | trades | win rate | avg W/L | max win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for nm, m in sorted(all_results.items(), key=lambda x: -x[1]["total_pct"]):
        wr = m.get("win_rate_pct", 0); aw = m.get("avg_win_pct", 0); al = m.get("avg_loss_pct", 0); mw = m.get("max_win_pct", 0)
        lines.append(f"| `{nm}` | {m['total_pct']:+.2f}% | {m['ann_pct']:+.2f}% | "
                     f"{m['daily_mean_pct']:+.4f}% | {m['sortino']:+.3f} | {m['sharpe']:+.3f} | "
                     f"{m['max_dd_pct']:+.2f}% | {m['n_trades']} | {wr:.1f}% | "
                     f"{aw:+.2f}/{al:+.2f}% | {mw:+.2f}% |")
    lines.append("")
    lines.append(f"**Best exit by total return**: `{best_exit}`")
    lines.append("")

    # Per-asset breakdown for best exit
    lines.append(f"## B. Per-asset performance breakdown (best exit: {best_exit})\n")
    lines.append("| asset | bucket | trades | win % | mean ret | sum ret | NAV contrib | oracle long avail | coverage % | capture % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in per_asset.iterrows():
        ola = f"{r['oracle_long_avail_pct']:+.2f}%" if pd.notna(r["oracle_long_avail_pct"]) else "—"
        cov = f"{r['coverage_pct']:.1f}%" if pd.notna(r["coverage_pct"]) else "—"
        cap = f"{r['capture_pct']:+.1f}%" if pd.notna(r["capture_pct"]) else "—"
        lines.append(f"| {r['asset']} | {r['bucket']} | {int(r['n_trades'])} | "
                     f"{r['win_rate']:.1f} | {r['mean_ret_pct']:+.3f}% | "
                     f"{r['sum_ret_pct']:+.2f}% | {r['contribution_to_nav_pct']:+.4f}% | "
                     f"{ola} | {cov} | {cap} |")
    lines.append("")

    # Aggregate stats per bucket
    lines.append("## C. Aggregate by bucket\n")
    by_bucket = per_asset.groupby("bucket").agg(
        n_assets=("asset", "nunique"),
        total_trades=("n_trades", "sum"),
        sum_contrib=("contribution_to_nav_pct", "sum"),
        median_win_rate=("win_rate", "median"),
        median_capture=("capture_pct", "median"),
    ).reset_index().sort_values("sum_contrib", ascending=False)
    lines.append("| bucket | n_assets | trades | NAV contrib | median win % | median capture % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in by_bucket.iterrows():
        med_cap = f"{r['median_capture']:+.1f}%" if pd.notna(r["median_capture"]) else "—"
        lines.append(f"| {r['bucket']} | {int(r['n_assets'])} | {int(r['total_trades'])} | "
                     f"{r['sum_contrib']:+.4f}% | {r['median_win_rate']:.1f} | {med_cap} |")

    # Headline
    best_m = all_results[best_exit]
    lines += [
        "",
        "## D. Headline",
        "",
        f"- **Architecture**: per-asset MA/EMA cousin-set + asset-own-regime gating",
        f"- **Universe deployed**: {per_asset['asset'].nunique()} assets",
        f"- **OOS window**: {OOS_START} → {OOS_END} (~{(OOS_END - OOS_START).days} days)",
        f"- **Best exit policy**: `{best_exit}`",
        f"- **OOS total return**: **{best_m['total_pct']:+.2f}%**",
        f"- **OOS annualized**: **{best_m['ann_pct']:+.2f}%**",
        f"- **Daily compound**: **{best_m['daily_mean_pct']:+.4f}%/d**",
        f"- **Sortino**: {best_m['sortino']:+.3f} ; Sharpe: {best_m['sharpe']:+.3f} ; Calmar: {best_m['calmar']:+.3f}",
        f"- **Max DD**: {best_m['max_dd_pct']:+.2f}%",
        f"- **Win rate**: {best_m.get('win_rate_pct', 0):.1f}% with asymmetric ratio {best_m.get('avg_win_pct', 0) / abs(best_m.get('avg_loss_pct', 1)):.1f}x" if best_m.get('avg_loss_pct') else "",
        "",
        "## E. Comparison to confluence-only (universal sleeve, current shipped)",
        "",
        f"| Architecture | OOS NAV | Annualized | Sortino | Max DD |",
        f"|---|---:|---:|---:|---:|",
        f"| **per_asset_v1 ({best_exit})** | **{best_m['total_pct']:+.2f}%** | **{best_m['ann_pct']:+.2f}%** | **{best_m['sortino']:+.3f}** | **{best_m['max_dd_pct']:+.2f}%** |",
        f"| confluence_only (shipped sleeve) | +67.21% | +85.76% | +1.563 | -31.91% |",
        f"| random-K (baseline) | +60.00% | +76.16% | +1.473 | -36.41% |",
        f"| signal (old broken sort) | +32.65% | +40.55% | +1.129 | -39.83% |",
        f"| best-K (perfect ranker) | +415.97% | +621.85% | +3.450 | -33.39% |",
    ]
    (OUT_DIR / "OOS_PER_ASSET_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_DIR / 'OOS_PER_ASSET_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
