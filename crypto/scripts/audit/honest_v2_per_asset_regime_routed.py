"""honest_v2_per_asset_regime_routed.py -- OOS sim with REGIME-ROUTED per-asset cells.

User insight (2026-05-20):
  Each asset now has a cell for each own-regime (when data permits). Deploy the
  regime-matched cell. Per-asset confluence is no longer the primary signal;
  the per-regime cell IS the signal for that asset on that day.

ARCHITECTURE:
  At each (sim_date, asset):
    1. Look up asset's own_regime today.
    2. Look up the winner cell for (asset, own_regime). If none, asset is cash.
    3. Check if the cell's MA cross signal fires today. If yes, asset is candidate.
  Global rank: rank candidate assets by their regime-cell's per-regime Sharpe;
  K=12 cap with per-bucket cap=5.

CONTRAST vs cousin-set architecture:
  - Cousin: confluence (multiple cells per asset can fire) → confluence score
  - Regime-routed: 1 cell per asset per day (the regime-matched one) → single score

Output: runs/audit/MA_EMA_PROFILE_2026_05_20/OOS_REGIME_ROUTED_REPORT.md
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
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
WINNERS_PATH = ROOT / "data" / "processed" / "per_asset_regime_winners.parquet"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

OOS_START = _date(2024, 5, 16)
OOS_END   = _date(2025, 3, 15)

K_MAX = 12
BET_FRACTION = 1.0 / K_MAX
COST_RT = 0.0030
BUCKET_CAP = 5

# Best exit from previous OOS run
EXIT_CFG = {"hold_max": 14, "stop": -0.04, "trail_arm": 0.05, "trail_drop": 0.03}


def walk_forward_exit(entry_price, fwd_closes, hold_max, stop, trail_arm, trail_drop, cost):
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


def precompute_regime_routed_fires(chimera, winners_df, own_regime, oos_start, oos_end):
    """For each (sim_date, asset), look up asset's regime and the winner cell;
    check if cell's MA cross is active today. Return DataFrame of qualifying fires."""
    fires = []
    # Build lookup: (asset, regime) -> winner cell row
    winner_lookup = {(r["asset"], r["own_regime"]): r for _, r in winners_df.iterrows()}
    own_lookup = {(r["asset"], r["date"].normalize()): r["asset_own_regime"]
                  for _, r in own_regime.iterrows()}

    # For each asset, precompute MA fast/slow series for ALL its winner cells (across regimes)
    asset_to_cells = {}
    for _, w in winners_df.iterrows():
        asset_to_cells.setdefault(w["asset"], []).append({
            "regime": w["own_regime"],
            "ma_type": w["ma_type"], "fast": int(w["fast"]), "slow": int(w["slow"]),
            "sharpe_in_regime": w["sharpe_in_regime"],
            "bucket": w["bucket"],
        })

    for asset, cells in asset_to_cells.items():
        chim = chimera.get(asset)
        if chim is None: continue
        closes = chim["close"].values
        dates = chim["date"].values

        # Compute each cell's MA series once
        cell_active = {}  # (regime) -> bool array of signal active
        for cell in cells:
            ma_type = cell["ma_type"]; fast = cell["fast"]; slow = cell["slow"]
            if ma_type == "SMA":
                ma_f = pd.Series(closes).rolling(fast).mean().values
                ma_s = pd.Series(closes).rolling(slow).mean().values
            else:
                ma_f = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
                ma_s = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
            cell_active[cell["regime"]] = (ma_f > ma_s)

        for i in range(len(dates)):
            d = pd.Timestamp(dates[i]).normalize()
            if not (pd.Timestamp(oos_start) <= d <= pd.Timestamp(oos_end)):
                continue
            own_reg = own_lookup.get((asset, d))
            if own_reg is None: continue
            if own_reg not in cell_active: continue  # no winner cell for this regime
            if cell_active[own_reg][i]:
                cell_info = next(c for c in cells if c["regime"] == own_reg)
                fires.append({
                    "asset": asset, "date": d,
                    "regime": own_reg,
                    "ma_type": cell_info["ma_type"],
                    "fast": cell_info["fast"], "slow": cell_info["slow"],
                    "sharpe": cell_info["sharpe_in_regime"],
                    "bucket": cell_info["bucket"],
                })
    return pd.DataFrame(fires)


def simulate(fires_df, chimera, oos_start, oos_end):
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
            rret, d_held, reason = walk_forward_exit(
                pos["entry_price"], fwd_closes,
                hold_max=EXIT_CFG["hold_max"], stop=EXIT_CFG["stop"],
                trail_arm=EXIT_CFG["trail_arm"], trail_drop=EXIT_CFG["trail_drop"],
                cost=COST_RT,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": asset, "regime": pos["regime"],
                    "entry_date": pos["entry_date"], "exit_date": sim_date,
                    "days_held": d_held, "bet_size": pos["bet_size"],
                    "realized_ret": rret, "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        today_cands = fires_df[fires_df["date"] == sim_dt].copy()
        if len(today_cands):
            today_cands = today_cands.sort_values("sharpe", ascending=False)
            open_assets = set(p["asset"] for p in open_positions)
            today_cands = today_cands[~today_cands["asset"].isin(open_assets)]
            today_cands = today_cands.drop_duplicates(subset="asset", keep="first")
            bucket_count = {}
            for _, row in today_cands.iterrows():
                if len(open_positions) >= K_MAX: break
                b = row["bucket"]
                if bucket_count.get(b, 0) >= BUCKET_CAP: continue
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
                    "regime": row["regime"],
                })

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
    return pd.DataFrame(daily_records), pd.DataFrame(trade_log)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print(f"REGIME-ROUTED PER-ASSET OOS ({OOS_START} → {OOS_END})")
    print("="*78)
    winners = pd.read_parquet(WINNERS_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    print(f"Winners: {len(winners)} cells / {winners['asset'].nunique()} assets")
    print(f"Per-regime: " + ", ".join(f"{r}={n}" for r, n in winners["own_regime"].value_counts().items()))
    print(f"Exit: {EXIT_CFG}")

    print("\nLoading chimera...")
    chimera = {}
    for f in glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        chimera[sym] = df
    print(f"  {len(chimera)} assets")

    print("\nPrecomputing regime-routed fires...")
    fires_df = precompute_regime_routed_fires(chimera, winners, own_regime, OOS_START, OOS_END)
    print(f"  total fires: {len(fires_df):,} across {fires_df['asset'].nunique()} assets / {fires_df['date'].nunique()} dates")
    print(f"  fires by regime: " + ", ".join(f"{r}={n}" for r, n in fires_df["regime"].value_counts().items()))

    print(f"\nSimulating...")
    daily_df, trades_df = simulate(fires_df, chimera, OOS_START, OOS_END)

    pv = daily_df["portfolio_value"].values
    total_pct = (pv[-1] / pv[0] - 1) * 100
    window_days = (OOS_END - OOS_START).days
    ann_pct = ((1 + total_pct/100) ** (365/max(window_days,1)) - 1) * 100
    drs = pv[1:] / pv[:-1] - 1
    sortino = (drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs<0].std() > 0 else 0
    sharpe = (drs.mean() / drs.std() * np.sqrt(252)) if drs.std() > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    calmar = ann_pct / abs(max_dd) if max_dd != 0 else 0

    print(f"\n=== RESULTS regime-routed (OOS) ===")
    print(f"  Total return:    {total_pct:+.2f}%")
    print(f"  Annualized:      {ann_pct:+.2f}%")
    print(f"  Daily mean:      {drs.mean()*100:+.4f}%")
    print(f"  Sortino:         {sortino:+.3f}")
    print(f"  Sharpe:          {sharpe:+.3f}")
    print(f"  Max DD:          {max_dd:+.2f}%")
    print(f"  Calmar:          {calmar:+.3f}")
    print(f"  Total trades:    {len(trades_df)}")
    if len(trades_df):
        win_rate = (trades_df["realized_ret"] > 0).mean() * 100
        avg_win = trades_df.loc[trades_df["realized_ret"]>0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]>0).any() else 0
        avg_loss = trades_df.loc[trades_df["realized_ret"]<0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]<0).any() else 0
        max_win = trades_df["realized_ret"].max() * 100
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Avg W/L:         {avg_win:+.2f}% / {avg_loss:+.2f}%")
        print(f"  Max win:         {max_win:+.2f}%")
        # Per-regime trade breakdown
        print(f"  Per-regime trades:")
        for r, g in trades_df.groupby("regime"):
            wr = (g["realized_ret"] > 0).mean() * 100
            sm = g["realized_ret"].sum() * 100
            print(f"    own_{r:<6} n={len(g):3d} win={wr:5.1f}% sum_ret={sm:+.2f}%")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(trades_df):
        trades_df.to_csv(OUT_DIR / "oos_regime_routed_trades.csv", index=False)

    lines = [
        f"# OOS Validation — Regime-Routed Per-Asset Architecture (2026-05-20)\n",
        f"**Window**: {OOS_START} → {OOS_END} ({window_days} days, canonical OOS)",
        f"**Architecture**: each asset deploys its per-own-regime winner cell when own-regime matches today",
        f"**Universe**: {winners['asset'].nunique()} assets × up to 4 regime-specific cells = {len(winners)} cells",
        f"**Exit**: `{EXIT_CFG}` (best from prior OOS sweep)",
        "",
        "## Headline",
        f"- Total return: **{total_pct:+.2f}%**",
        f"- Annualized: **{ann_pct:+.2f}%**",
        f"- Daily compound: **{drs.mean()*100:+.4f}%/d**",
        f"- Sortino: {sortino:+.3f} ; Sharpe: {sharpe:+.3f} ; Calmar: {calmar:+.3f}",
        f"- Max DD: {max_dd:+.2f}%",
        f"- Trades: {len(trades_df)} ; Win rate: {(trades_df['realized_ret'] > 0).mean()*100 if len(trades_df) else 0:.1f}%",
        "",
        "## Comparison table",
        "",
        "| Architecture | OOS NAV | Annualized | Sortino | Max DD | Trades | Win % |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **per_asset_regime_routed (THIS)** | **{total_pct:+.2f}%** | **{ann_pct:+.2f}%** | **{sortino:+.3f}** | **{max_dd:+.2f}%** | **{len(trades_df)}** | **{(trades_df['realized_ret'] > 0).mean()*100 if len(trades_df) else 0:.1f}%** |",
        f"| per_asset_cousin_set (tight_trail) | +159.33% | +215.17% | +4.176 | -19.12% | 550 | 39.5% |",
        f"| confluence_only (current shipped) | +67.21% | +85.76% | +1.563 | -31.91% | — | 37.0% |",
        f"| random-K (baseline) | +60.00% | +76.16% | +1.473 | -36.41% | — | 34.8% |",
        f"| best-K (perfect foresight) | +415.97% | +621.85% | +3.450 | -33.39% | — | 46.8% |",
        "",
        "## Per-regime trade breakdown",
        "",
        "| regime | trades | win rate | sum ret | mean ret |",
        "|---|---:|---:|---:|---:|",
    ]
    if len(trades_df):
        for r, g in trades_df.groupby("regime"):
            wr = (g["realized_ret"] > 0).mean() * 100
            sm = g["realized_ret"].sum() * 100
            mn = g["realized_ret"].mean() * 100
            lines.append(f"| own_{r} | {len(g)} | {wr:.1f}% | {sm:+.2f}% | {mn:+.3f}% |")
    (OUT_DIR / "OOS_REGIME_ROUTED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_DIR / 'OOS_REGIME_ROUTED_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
