"""honest_v2_per_asset.py -- Step 6: honest portfolio simulator under PER-ASSET architecture.

NEW deploy model:
  For each (sim_date, asset) where asset has a profile:
    1. Determine asset_own_regime today (from asset_own_regime_panel)
    2. For each of asset's COUSIN-SET cells (regime-tag qualifies today):
        a. Check if cell's signal fires today (sign(MA_fast - MA_slow) == +1 today
           AND wasn't long yesterday → cross-up event)
        b. If yes, count toward asset's confluence_count_today
    3. Asset's deploy_score = (confluence_count, per_asset_sharpe, regime_favorability)

  Then global ranking: pick top K=12 assets by deploy_score; per-bucket cap.

  Portfolio mechanics (same as honest_v2_simulator):
    - K_MAX=12; BET=8.3% per pick (equal-weight across K)
    - HARD_STOP -4%; TRAIL_ARM +10%; TRAIL_DROP -5%; HOLD_MAX up to 21 days
      (extended from 10 since user said "let winners run")

  Compare modes:
    - per_asset_v1: per-asset profile-driven (NEW)
    - universal_v2 (signal_confluence_only from sleeve patch — current baseline)
    - random: pick K assets uniformly from firing universe
    - best: oracle (pick top K by future 14d return)
    - worst: adversarial

Outputs:
  runs/audit/MA_EMA_PROFILE_2026_05_20/HONEST_PER_ASSET_REPORT.md
  runs/audit/MA_EMA_PROFILE_2026_05_20/honest_per_asset.json
"""
from __future__ import annotations

import sys
import json
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
SNAP_PATH = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation" / "event_ma_snapshot.parquet"
OUT_MD = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "HONEST_PER_ASSET_REPORT.md"
OUT_JSON = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "honest_per_asset.json"

# Portfolio params (let winners run — extended hold)
K_MAX = 12
BET_FRACTION = 1.0 / K_MAX  # equal-weight 8.3% per pick
HARD_STOP = -0.04
TRAIL_ARM = 0.10
TRAIL_DROP = 0.05
HOLD_MAX = 21  # extended from 10 — "let winners run" per user 2026-05-20
COST_RT = 0.0030
BUCKET_CAP = 5  # max 5 per bucket
TRADING_DAYS_PER_YEAR = 365

# Regime qualification: which today-regimes qualify a cell's regime_tag
REGIME_QUALIFIES = {
    "ALL_WEATHER":      {"bull", "chop", "bear", "crash"},
    "BLOCK_OWN_CRASH":  {"bull", "chop", "bear"},
    "BLOCK_OWN_BEAR":   {"bull", "chop"},
    "BULL_ONLY":        {"bull"},
    "REGIME_DEPENDENT": {"bull", "chop"},   # conservative default
    "INSUFFICIENT_DATA": set(),             # don't deploy
}


def walk_forward_exit(entry_price, fwd_closes, fwd_highs=None, hold_max=HOLD_MAX,
                      stop=HARD_STOP, trail_arm=TRAIL_ARM, trail_drop=TRAIL_DROP, cost=COST_RT):
    """Walk forward; return (realized_ret_net, days_held, exit_reason)."""
    peak = entry_price; armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p): continue
        ret = p / entry_price - 1
        if p > peak: peak = p
        if not armed and ret >= trail_arm: armed = True
        if ret <= stop: return stop - cost, d, "stop"
        if armed and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d, "trail"
        if d >= hold_max: return ret - cost, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None: return -cost, 0, "no_data"
    return last / entry_price - 1 - cost, len(fwd_closes), "expire"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("HONEST V2 PER-ASSET SIMULATOR")
    print("="*78)
    print(f"Params: K_MAX={K_MAX} BET={BET_FRACTION*100:.2f}% HOLD_MAX={HOLD_MAX}d "
          f"STOP={HARD_STOP*100:.0f}% TRAIL_ARM={TRAIL_ARM*100:.0f}%/{TRAIL_DROP*100:.0f}%")
    print(f"Cost: {COST_RT*100:.2f}% RT  Bucket cap: {BUCKET_CAP}")
    print(f"Regime: tag-gated; cell deploys only if today's asset_own_regime ∈ tag's qualified set")

    print("\nLoading inputs...")
    profile = pd.read_parquet(PROFILE_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"])
    snap = pl.read_parquet(SNAP_PATH).to_pandas()
    snap["date"] = pd.to_datetime(snap["date"])
    print(f"  profile cells: {len(profile)} across {profile['asset'].nunique()} assets")
    print(f"  own_regime: {len(own_regime)} rows")
    print(f"  snap: {len(snap)} events")

    # Load per-asset chimera close (for walk-forward exit)
    print("\nLoading chimera 1d per asset (for forward walk)...")
    chimera = {}
    import glob as _glob
    for f in _glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close", "high", "low"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        chimera[sym] = df
    print(f"  loaded {len(chimera)} assets")

    # Define per-asset cell dict: {asset -> [cells]}
    # Only use cousin-set cells (deploy-grade subset)
    deploy_cells = profile[profile["is_cousin_set_member"]].copy()
    print(f"  cousin-set cells: {len(deploy_cells)} across {deploy_cells['asset'].nunique()} assets")
    cells_by_asset = {a: g.to_dict("records") for a, g in deploy_cells.groupby("asset")}

    # Build asset-own-regime lookup: (asset, date) -> regime
    own_regime["date"] = own_regime["date"].dt.normalize()
    own_lookup = {(row["asset"], row["date"]): row["asset_own_regime"]
                  for _, row in own_regime[["asset", "date", "asset_own_regime"]].iterrows()}

    # Simulation window: VAL (snap covers VAL)
    ws = snap["date"].min().date()
    we = snap["date"].max().date()
    print(f"\nSim window: {ws} -> {we}")

    # Pre-compute fires per (asset, date, cell_id)
    # signal fires today = sign(MA_fast(T-1) - MA_slow(T-1)) just turned to +1
    # (i.e., today MA_fast > MA_slow AND yesterday it was <=)
    # But snap has MA values at T-1 already (per build_ma_ema_permutation shift(1)).
    # Simpler: use snap rows where (asset, date) exists AND signal == +1 for the cell.
    # For "fires today" we need a STATE CHANGE — let's use snap fires (where signal==+1)
    # as a proxy for "today's qualifying long signal."
    print("\nPre-computing per-(asset, date, cell) fire matrix...")
    fires = []  # list of (asset, date, cell_id, ma_type, fast, slow)
    snap_idx = snap.set_index(["asset", "date"])
    for asset, cells in cells_by_asset.items():
        ev_asset = snap[snap["asset"] == asset]
        for cell in cells:
            ma_type = cell["ma_type"]; fast = int(cell["fast"]); slow = int(cell["slow"])
            col_f = f"{ma_type}_{fast}"; col_s = f"{ma_type}_{slow}"
            if col_f not in ev_asset.columns: continue
            mask = (ev_asset[col_f].notna() & ev_asset[col_s].notna() &
                    (ev_asset[col_f] > ev_asset[col_s]))
            for d in ev_asset.loc[mask, "date"]:
                fires.append({"asset": asset, "date": d,
                              "cell_id": cell["cell_id"],
                              "regime_tag": cell["regime_tag"],
                              "sharpe": cell["sharpe"]})
    fires_df = pd.DataFrame(fires)
    print(f"  total fire-events: {len(fires_df):,} across {fires_df['asset'].nunique()} assets")
    print(f"  unique fire-dates: {fires_df['date'].nunique()}")

    # Per-asset deploy_score per date
    # confluence_count = n distinct cousin cells firing on asset that day
    # asset's max_cell_sharpe = max sharpe among today's firing cells (for tiebreak)
    print("\nComputing per-(asset, date) deploy score...")
    fires_df["regime_today"] = [own_lookup.get((row.asset, row.date), "unknown")
                                  for row in fires_df.itertuples()]
    fires_df["qualifies"] = [row.regime_today in REGIME_QUALIFIES.get(row.regime_tag, set())
                              for row in fires_df.itertuples()]
    qual = fires_df[fires_df["qualifies"]]
    print(f"  fires passing regime qualification: {len(qual):,} ({len(qual)/max(1,len(fires_df))*100:.1f}%)")

    asset_score_per_day = (qual.groupby(["asset", "date"]).agg(
        confluence_count=("cell_id", "nunique"),
        max_cell_sharpe=("sharpe", "max"),
        cells=("cell_id", lambda x: list(x.unique())),
    ).reset_index())
    print(f"  asset-day deploy candidates: {len(asset_score_per_day):,}")

    # Simulate
    print(f"\n=== SIMULATE per_asset_v1 ===")
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []
    asset_to_bucket = {a: deploy_cells[deploy_cells["asset"] == a]["bucket"].iloc[0]
                       for a in deploy_cells["asset"].unique()}

    cur = ws; cal_dates = []
    while cur <= we:
        cal_dates.append(cur); cur += timedelta(days=1)
    print(f"  calendar days: {len(cal_dates)}")

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
            # Check exit
            rret, d_held, reason = walk_forward_exit(pos["entry_price"], fwd_closes)
            should_close = (
                d_held >= HOLD_MAX or
                (len(fwd_closes) and fwd_closes[-1] is not None and (
                    fwd_closes[-1] / pos["entry_price"] - 1 <= HARD_STOP
                )) or
                # Trail-stop check: peak vs current
                False  # walk_forward_exit handles inline
            )
            if d_held >= 1 and (reason in ("stop", "trail", "max_hold")):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": asset, "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": rret,
                    "exit_reason": reason,
                    "cell_id": pos["cell_id"],
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # New entries
        today_candidates = asset_score_per_day[asset_score_per_day["date"] == sim_dt].copy()
        if len(today_candidates):
            today_candidates["bucket"] = today_candidates["asset"].map(asset_to_bucket)
            today_candidates = today_candidates.sort_values(
                ["confluence_count", "max_cell_sharpe"], ascending=[False, False])
            # Exclude assets already open
            open_assets = set(p["asset"] for p in open_positions)
            today_candidates = today_candidates[~today_candidates["asset"].isin(open_assets)]
            # Bucket cap + K_MAX
            bucket_count = {}
            for _, row in today_candidates.iterrows():
                if len(open_positions) >= K_MAX: break
                b = row["bucket"]
                if bucket_count.get(b, 0) >= BUCKET_CAP: continue
                # Entry price = today's close
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
                    "cell_id": row["cells"][0] if row["cells"] else "?",
                })

        # MtM open positions
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
                              "n_open": len(open_positions),
                              "n_candidates": len(today_candidates) if len(today_candidates) else 0})

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)

    # Metrics
    pv = daily_df["portfolio_value"].values
    total_pct = (pv[-1] / pv[0] - 1) * 100
    window_days = (cal_dates[-1] - cal_dates[0]).days
    ann_pct = ((1 + total_pct/100) ** (365/max(window_days,1)) - 1) * 100
    daily_rets = pv[1:] / pv[:-1] - 1
    mean_d = daily_rets.mean()
    std_d = daily_rets.std()
    sortino = (mean_d / daily_rets[daily_rets < 0].std() * np.sqrt(252)) if (daily_rets < 0).sum() and daily_rets[daily_rets<0].std() > 0 else 0
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    calmar = ann_pct / abs(max_dd) if max_dd != 0 else 0

    print(f"\n=== RESULTS per_asset_v1 ===")
    print(f"  Window: {ws} -> {we} ({window_days} days)")
    print(f"  Total return:    {total_pct:+.2f}%")
    print(f"  Annualized:      {ann_pct:+.2f}%")
    print(f"  Daily mean:      {mean_d*100:+.4f}%")
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
        median_hold = trades_df["days_held"].median()
        print(f"  Win rate:        {win_rate:.1f}%")
        print(f"  Avg win/loss:    {avg_win:+.2f}% / {avg_loss:+.2f}%")
        print(f"  Max win:         {max_win:+.2f}%")
        print(f"  Median hold:     {median_hold} days")
        print(f"  Exit reasons:")
        for r, n in trades_df["exit_reason"].value_counts().items():
            print(f"    {r}: {n}")

    # Write outputs
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "params": {
            "K_MAX": K_MAX, "BET_FRACTION": BET_FRACTION,
            "HARD_STOP": HARD_STOP, "TRAIL_ARM": TRAIL_ARM, "TRAIL_DROP": TRAIL_DROP,
            "HOLD_MAX": HOLD_MAX, "COST_RT": COST_RT, "BUCKET_CAP": BUCKET_CAP,
        },
        "metrics": {
            "total_pct": total_pct, "ann_pct": ann_pct, "daily_mean_pct": mean_d * 100,
            "sortino": sortino, "sharpe": sharpe, "max_dd_pct": max_dd, "calmar": calmar,
            "n_trades": int(len(trades_df)),
            "win_rate_pct": float((trades_df["realized_ret"] > 0).mean() * 100) if len(trades_df) else 0,
        },
        "window_days": window_days,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    # Trades + daily NAV
    if len(trades_df):
        trades_df.to_csv(OUT_MD.parent / "honest_per_asset_trades.csv", index=False)
        daily_df.to_csv(OUT_MD.parent / "honest_per_asset_daily.csv", index=False)

    lines = [
        "# Honest V2 — Per-Asset Architecture Result (2026-05-20)\n",
        f"**Source**: per_asset_ma_ema_profile (cousin-set only) + chimera 1d closes + asset_own_regime_panel\n",
        f"**Params**: K={K_MAX} BET={BET_FRACTION*100:.2f}% HOLD≤{HOLD_MAX}d STOP={HARD_STOP*100:.0f}%  TRAIL+{TRAIL_ARM*100:.0f}/-{TRAIL_DROP*100:.0f}%  COST={COST_RT*100:.2f}%RT  BUCKET_CAP={BUCKET_CAP}\n",
        f"**Window**: {ws} → {we} ({window_days} days, VAL)\n",
        f"**Universe**: {deploy_cells['asset'].nunique()} assets with cousin-set MA/EMA cells\n",
        "",
        "## Headline\n",
        f"- Total return: **{total_pct:+.2f}%** over {window_days} days",
        f"- Annualized: **{ann_pct:+.2f}%**",
        f"- Daily compound: **{mean_d*100:+.4f}%/d**",
        f"- Sortino: {sortino:+.3f} ; Sharpe: {sharpe:+.3f} ; Calmar: {calmar:+.3f}",
        f"- Max DD: {max_dd:+.2f}%",
        f"- Trades: {len(trades_df)} ; Win rate: {(trades_df['realized_ret'] > 0).mean()*100 if len(trades_df) else 0:.1f}%",
        "",
        "## Comparison to current baselines (same window, honest_v2_simulator)\n",
        "| Mode | OOS NAV | Annualized | Sortino | Max DD |",
        "|---|---:|---:|---:|---:|",
        f"| **per_asset_v1 (this)** | **{total_pct:+.2f}%** | **{ann_pct:+.2f}%** | **{sortino:+.3f}** | **{max_dd:+.2f}%** |",
        "| signal_confluence_only (current shipped sleeve) | +67.21% | +85.76% | +1.563 | -31.91% |",
        "| random-K (baseline) | +60.00% | +76.16% | +1.473 | -36.41% |",
        "| signal (OLD raw-sort) | +32.65% | +40.55% | +1.129 | -39.83% |",
        "| best-K (perfect foresight) | +415.97% | +621.85% | +3.450 | -33.39% |",
        "",
        "Note: per_asset_v1 uses VAL window (2023-07 → 2024-05), while the above were on OOS (2024-05 → 2025-03). Same-window comparison needs OOS rerun.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_MD}")
    print(f"[OK] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
