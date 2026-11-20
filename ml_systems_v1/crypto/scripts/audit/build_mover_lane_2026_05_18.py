"""+25%-Mover Sleeve + RVOL Exhaustion Exit overlay (Builds #1 and #3).

Trades the +25%-mover continuation lane discovered in the oracle exercise.
Composes three Gemini-recommended modifications + my earlier regime-gate finding:

  Build #1: Regime-gated, bucket-filtered, multi-day hold sleeve.
    - Trigger:     ret_1d >= TRIGGER_THRESH (default +15% with optional +25%)
    - Regime gate: btc_30d >= REGIME_MIN (default 0%, i.e. NOT bear/crash)
    - Bucket gate: DEGEN or VOLATILE only (where alpha lives per oracle trace)
    - Entry:       t+1 close (no look-ahead)
    - Exit:        fixed N-day hold (default 5d)
    - Cost:        24bps RT
    - Size:        4% NAV per entry
    - Cap:         max 5 simultaneous positions (capital lockup is binding)

  Build #3: RVOL exhaustion exit overlay.
    - On each held position, evaluate daily RVOL = volume[t] / vol_10d_ma[t]
    - Exit early if RVOL drops below 0.5 for 2 consecutive days (mania has died)
    - Otherwise hold to max_N
    - Same cost, size, cap

Both runs are compared against:
  - Naive baseline (no gate, no bucket filter)
  - Fixed-N-hold gated (Build #1)
  - Fixed-N-hold + RVOL exit (Build #1 + #3)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "runs" / "audit" / "oracle_panel_binance_2026_05_18.parquet"
OUT_DIR = ROOT / "runs" / "audit"


@dataclass
class SleeveConfig:
    name: str
    trigger_thresh: float = 0.15          # ret_1d >= this
    regime_min_btc_30d: float = -0.05     # exclude bear/crash (-5% threshold)
    bucket_filter: tuple = ("DEGEN", "VOLATILE")
    max_hold_days: int = 5
    rvol_exit_enabled: bool = False
    rvol_exit_threshold: float = 0.5      # rvol drops below this
    rvol_exit_consecutive_days: int = 2
    cost_rt: float = 0.0024
    size_per_entry: float = 0.04
    max_simultaneous: int = 5


def load_panel():
    """Load the 8Q WF panel built in run_gemini_oracle_brief_real.py."""
    df = pd.read_parquet(PANEL_PATH)
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    # Per-asset 10-day rolling mean volume for RVOL
    df["vol_10d_ma"] = df.groupby("asset")["volume"].transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    df["rvol"] = df["volume"] / df["vol_10d_ma"]
    # Build close-by-(asset,date) lookup for fast forward indexing
    return df


def simulate(panel: pd.DataFrame, cfg: SleeveConfig) -> dict:
    """Run sleeve config over the panel. Returns aggregated metrics + trade ledger.

    Methodology:
      1. Enumerate trigger events (asset, date) where ret_1d >= cfg.trigger_thresh.
      2. Apply regime + bucket filters.
      3. For each event:
         - Entry at close of t+1 (use close_t1 column).
         - Evaluate hold daily: at each k in 1..max_hold_days, check exit conditions.
         - Exit on: (a) max_hold reached, (b) RVOL exhaustion if enabled.
         - Trade PnL = exit_close / entry_close - 1, net of cost.
      4. Aggregate to per-quarter + 8Q totals.

    Position cap: cfg.max_simultaneous concurrent positions enforced FIFO. Triggers
    fired while at cap are SKIPPED (counted in `skipped_capacity`).
    """
    # 1. Trigger filter (entry-eligible events)
    ev = panel[panel["ret_1d"] >= cfg.trigger_thresh].copy()
    ev = ev[ev["close_t1"].notna()]  # need a valid t+1 close
    # 2. Regime + bucket
    ev = ev[ev["btc_30d"] >= cfg.regime_min_btc_30d]
    ev = ev[ev["bucket"].isin(cfg.bucket_filter)]
    # Sort by date for FIFO position-cap simulation
    ev = ev.sort_values(["date", "asset"]).reset_index(drop=True)

    # Build a date-indexed close panel for fast forward lookups
    close_by = {}
    for asset, sub in panel.groupby("asset"):
        s = sub.set_index("date")["close"].sort_index()
        close_by[asset] = s
    vol_by = {}
    for asset, sub in panel.groupby("asset"):
        v = sub.set_index("date")["rvol"].sort_index()
        vol_by[asset] = v

    open_positions = []   # list of dicts: {asset, entry_date_t1, entry_close, days_low_rvol}
    trade_log = []
    skipped_capacity = 0
    triggers_seen = len(ev)

    # Walk trigger events in date order
    # For each event date, advance open positions one day; check exits; then attempt new entry.
    if triggers_seen == 0:
        return {"trade_log": pd.DataFrame(), "n_triggers": 0, "n_skipped_capacity": 0,
                "summary": {}, "per_quarter": pd.DataFrame(),
                "per_exit_reason": pd.DataFrame(), "per_bucket": pd.DataFrame()}

    # Pre-build all unique dates in the panel (for position aging)
    unique_dates = sorted(panel["date"].unique())
    ev_idx_by_date = {d: [] for d in unique_dates}
    for i, r in ev.iterrows():
        ev_idx_by_date.setdefault(r["date"], []).append(i)

    def try_exit(pos, today):
        """Check whether position pos exits at `today`. Returns (should_exit, exit_close, exit_reason)."""
        days_held = (today - pos["entry_date_t1"]).days
        if days_held <= 0:
            return False, None, None
        # Get today's close + rvol
        try:
            today_close = close_by[pos["asset"]].loc[today]
            today_rvol = vol_by[pos["asset"]].loc[today]
        except KeyError:
            return False, None, None  # asset has no bar today — keep holding
        # Hard max-hold check
        if days_held >= cfg.max_hold_days:
            return True, today_close, "max_hold"
        # RVOL exhaustion check
        if cfg.rvol_exit_enabled and pd.notna(today_rvol):
            if today_rvol < cfg.rvol_exit_threshold:
                pos["days_low_rvol"] += 1
                if pos["days_low_rvol"] >= cfg.rvol_exit_consecutive_days:
                    return True, today_close, "rvol_exit"
            else:
                pos["days_low_rvol"] = 0
        return False, None, None

    # Drive simulation forward day-by-day
    for today in unique_dates:
        # 1. Age open positions, check exits
        still_open = []
        for pos in open_positions:
            should_exit, exit_close, reason = try_exit(pos, today)
            if should_exit:
                gross_ret = exit_close / pos["entry_close"] - 1
                net_ret = gross_ret - cfg.cost_rt
                days_held = (today - pos["entry_date_t1"]).days
                trade_log.append({
                    "asset": pos["asset"],
                    "bucket": pos["bucket"],
                    "trigger_date": pos["trigger_date"],
                    "entry_date_t1": pos["entry_date_t1"],
                    "exit_date": today,
                    "days_held": days_held,
                    "entry_close": pos["entry_close"],
                    "exit_close": exit_close,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "exit_reason": reason,
                    "quarter": pos["quarter"],
                    "regime": pos["regime"],
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Try to enter new positions if there are triggers today
        for idx in ev_idx_by_date.get(today, []):
            if len(open_positions) >= cfg.max_simultaneous:
                skipped_capacity += 1
                continue
            row = ev.loc[idx]
            entry_close = row["close_t1"]
            if pd.isna(entry_close):
                continue
            # Find entry date (= today + 1 calendar day position-wise but we use date+1 row)
            # Use the date of the t+1 bar = next row's date
            asset_dates = close_by[row["asset"]].index
            if today not in asset_dates:
                continue
            today_pos = list(asset_dates).index(today)
            if today_pos + 1 >= len(asset_dates):
                continue
            entry_date_t1 = asset_dates[today_pos + 1]
            open_positions.append({
                "asset": row["asset"],
                "bucket": row["bucket"],
                "trigger_date": today,
                "entry_date_t1": entry_date_t1,
                "entry_close": entry_close,
                "days_low_rvol": 0,
                "quarter": row["quarter"],
                "regime": row["regime"],
            })

    # Force close any remaining positions at last available close (window end)
    last_date = unique_dates[-1]
    for pos in open_positions:
        try:
            close_at_end = close_by[pos["asset"]].loc[last_date]
        except KeyError:
            close_at_end = pos["entry_close"]
        gross_ret = close_at_end / pos["entry_close"] - 1
        net_ret = gross_ret - cfg.cost_rt
        days_held = (last_date - pos["entry_date_t1"]).days
        trade_log.append({
            "asset": pos["asset"],
            "bucket": pos["bucket"],
            "trigger_date": pos["trigger_date"],
            "entry_date_t1": pos["entry_date_t1"],
            "exit_date": last_date,
            "days_held": days_held,
            "entry_close": pos["entry_close"],
            "exit_close": close_at_end,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "exit_reason": "window_end",
            "quarter": pos["quarter"],
            "regime": pos["regime"],
        })

    tl = pd.DataFrame(trade_log)
    # Aggregate
    if len(tl) == 0:
        return {"trade_log": tl, "n_triggers": triggers_seen,
                "n_skipped_capacity": skipped_capacity,
                "summary": {}, "per_quarter": pd.DataFrame(),
                "per_exit_reason": pd.DataFrame(), "per_bucket": pd.DataFrame()}

    nav_per_trade = cfg.size_per_entry * tl["net_ret"]
    total_nav_pct = (nav_per_trade.sum()) * 100

    per_q = tl.groupby("quarter").agg(
        n=("net_ret", "size"),
        mean_net=("net_ret", "mean"),
        sum_net=("net_ret", "sum"),
        win_rate=("net_ret", lambda s: (s > 0).mean()),
        mean_days_held=("days_held", "mean"),
    ).reset_index()
    per_q["nav_q_pct"] = per_q["sum_net"] * cfg.size_per_entry * 100

    per_exit_reason = tl.groupby("exit_reason").agg(
        n=("net_ret", "size"),
        mean_net=("net_ret", "mean"),
        sum_net=("net_ret", "sum"),
        mean_days_held=("days_held", "mean"),
    ).reset_index()

    per_bucket = tl.groupby("bucket").agg(
        n=("net_ret", "size"),
        mean_net=("net_ret", "mean"),
        sum_net=("net_ret", "sum"),
    ).reset_index()
    per_bucket["nav_q_pct"] = per_bucket["sum_net"] * cfg.size_per_entry * 100

    summary = {
        "config_name": cfg.name,
        "n_triggers_eligible": triggers_seen,
        "n_trades_closed": len(tl),
        "mean_days_held": float(tl["days_held"].mean()),
        "win_rate": float((tl["net_ret"] > 0).mean()),
        "mean_net_per_trade": float(tl["net_ret"].mean()),
        "total_gross_pct": float((tl["gross_ret"]).sum() * 100),
        "total_net_pct": float((tl["net_ret"]).sum() * 100),
        "total_nav_8q_pct": float(total_nav_pct),
        "best_trade": float(tl["net_ret"].max() * 100),
        "worst_trade": float(tl["net_ret"].min() * 100),
        "median_trade": float(tl["net_ret"].median() * 100),
    }
    return {
        "trade_log": tl,
        "summary": summary,
        "n_skipped_capacity": skipped_capacity,
        "per_quarter": per_q,
        "per_exit_reason": per_exit_reason,
        "per_bucket": per_bucket,
    }


def main():
    panel = load_panel()
    print(f"Loaded panel: {len(panel)} rows, {panel['asset'].nunique()} assets, "
          f"dates {panel['date'].min()} to {panel['date'].max()}")

    configs = [
        # Naive baseline (no gate, no bucket filter, fixed 5d hold)
        SleeveConfig(
            name="NAIVE_+15%_5d_NO_GATE",
            trigger_thresh=0.15,
            regime_min_btc_30d=-1.0,   # accept everything
            bucket_filter=("BLUE", "STEADY", "DEGEN", "VOLATILE"),
            max_hold_days=5,
            rvol_exit_enabled=False,
        ),
        # Build #1a: +15% trigger, Bull/Chop only, DEGEN+VOLATILE, 5d hold
        SleeveConfig(
            name="BUILD1_+15%_5d_GATED_BUCKETED",
            trigger_thresh=0.15,
            regime_min_btc_30d=-0.05,  # exclude bear/crash
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=5,
            rvol_exit_enabled=False,
        ),
        # Build #1b: +25% trigger, Bull/Chop only, DEGEN+VOLATILE, 5d hold
        SleeveConfig(
            name="BUILD1_+25%_5d_GATED_BUCKETED",
            trigger_thresh=0.25,
            regime_min_btc_30d=-0.05,
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=5,
            rvol_exit_enabled=False,
        ),
        # Build #1c: +15% trigger, BULL ONLY (>=+5%), DEGEN+VOLATILE, 5d hold
        SleeveConfig(
            name="BUILD1_+15%_5d_BULLONLY_BUCKETED",
            trigger_thresh=0.15,
            regime_min_btc_30d=0.05,
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=5,
            rvol_exit_enabled=False,
        ),
        # Build #3: +15%, Bull/Chop, DEGEN+VOLATILE, 5d max with RVOL exit
        SleeveConfig(
            name="BUILD3_+15%_5d_GATED_RVOL",
            trigger_thresh=0.15,
            regime_min_btc_30d=-0.05,
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=5,
            rvol_exit_enabled=True,
            rvol_exit_threshold=0.5,
            rvol_exit_consecutive_days=2,
        ),
        # Build #1+#3 combined: tightest config — +25%, BULL ONLY, DEGEN+VOLATILE, RVOL exit
        SleeveConfig(
            name="BUILD1+3_+25%_5d_BULLONLY_RVOL",
            trigger_thresh=0.25,
            regime_min_btc_30d=0.05,
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=5,
            rvol_exit_enabled=True,
            rvol_exit_threshold=0.5,
            rvol_exit_consecutive_days=2,
        ),
        # Sensitivity: 3-day max hold variant
        SleeveConfig(
            name="BUILD1_+15%_3d_GATED_BUCKETED",
            trigger_thresh=0.15,
            regime_min_btc_30d=-0.05,
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=3,
            rvol_exit_enabled=False,
        ),
        # Sensitivity: 7-day max hold variant
        SleeveConfig(
            name="BUILD1_+15%_7d_GATED_BUCKETED",
            trigger_thresh=0.15,
            regime_min_btc_30d=-0.05,
            bucket_filter=("DEGEN", "VOLATILE"),
            max_hold_days=7,
            rvol_exit_enabled=False,
        ),
    ]

    results = {}
    print()
    print("=" * 78)
    print("RUNNING CONFIGS")
    print("=" * 78)
    for cfg in configs:
        print(f"\n[{cfg.name}]")
        res = simulate(panel, cfg)
        results[cfg.name] = res
        s = res["summary"]
        if s:
            print(f"  triggers_eligible={s['n_triggers_eligible']} "
                  f"trades_closed={s['n_trades_closed']} "
                  f"skipped_cap={res['n_skipped_capacity']}")
            print(f"  mean_days_held={s['mean_days_held']:.2f} "
                  f"win_rate={s['win_rate']*100:.1f}% "
                  f"mean_net_per_trade={s['mean_net_per_trade']*100:+.3f}%")
            print(f"  total_gross={s['total_gross_pct']:+.2f}% "
                  f"total_net={s['total_net_pct']:+.2f}% "
                  f"NAV_8Q@4%={s['total_nav_8q_pct']:+.2f}%")
            print(f"  best_trade={s['best_trade']:+.2f}% "
                  f"worst_trade={s['worst_trade']:+.2f}% "
                  f"median={s['median_trade']:+.3f}%")
        else:
            print("  NO TRADES")

    # Generate audit report
    lines = []
    def w(line=""):
        lines.append(line)

    w("# Mover-Lane Builds (#1 + #3) — Real-Data Walk-Forward")
    w()
    w("**Date**: 2026-05-18  ")
    w(f"**Window**: 24Q1 → 25Q4 (UNSEEN NOT TOUCHED)  ")
    w(f"**Universe**: 81 USDT-quoted spot symbols (Binance public klines)  ")
    w(f"**Panel**: {len(panel)} rows  ")
    w("**Cost**: 24bps RT, 4% NAV per entry, max 5 simultaneous positions  ")
    w()
    w("Predecessors:")
    w("- [ORACLE_25PCT_MOVER_TRACE_2026_05_18.md](ORACLE_25PCT_MOVER_TRACE_2026_05_18.md) — original trace")
    w("- [ORACLE_EXERCISE_BINANCE_REAL_RESULTS_2026_05_18.md](ORACLE_EXERCISE_BINANCE_REAL_RESULTS_2026_05_18.md) — Binance verification")
    w()

    w("## 1. Headline comparison")
    w()
    w("| Config | Trigger | Hold | Regime gate | Bucket | RVOL exit | Trades | Win rate | Mean net/trade | NAV 8Q @4% |")
    w("|---|---|---|---|---|:---:|---:|---:|---:|---:|")
    for cfg in configs:
        s = results[cfg.name]["summary"]
        if not s:
            continue
        gate_label = (f"BTC30d≥{cfg.regime_min_btc_30d:+.0%}" if cfg.regime_min_btc_30d > -0.99
                      else "none")
        buck_label = "+".join(cfg.bucket_filter) if len(cfg.bucket_filter) < 4 else "ALL"
        rvol_label = "YES" if cfg.rvol_exit_enabled else "no"
        w(f"| `{cfg.name}` | +{cfg.trigger_thresh*100:.0f}% | {cfg.max_hold_days}d | {gate_label} | "
          f"{buck_label} | {rvol_label} | {s['n_trades_closed']} | {s['win_rate']*100:.1f}% | "
          f"{s['mean_net_per_trade']*100:+.3f}% | **{s['total_nav_8q_pct']:+.2f}%** |")
    w()

    # Per-quarter breakdown for the headline config
    headline = "BUILD1_+15%_5d_GATED_BUCKETED"
    if results[headline]["summary"]:
        w(f"## 2. Per-quarter breakdown — `{headline}`")
        w()
        w("| Quarter | Trades | Mean Net | Win Rate | Mean Days Held | NAV q% @4% |")
        w("|---|---:|---:|---:|---:|---:|")
        pq = results[headline]["per_quarter"]
        for _, r in pq.iterrows():
            w(f"| {r['quarter']} | {int(r['n'])} | {r['mean_net']*100:+.3f}% | "
              f"{r['win_rate']*100:.1f}% | {r['mean_days_held']:.2f} | {r['nav_q_pct']:+.2f}% |")
        w()

    # Exit reason breakdown for RVOL config
    rvol_cfg = "BUILD3_+15%_5d_GATED_RVOL"
    if results[rvol_cfg]["summary"]:
        w(f"## 3. Exit-reason breakdown — `{rvol_cfg}`")
        w()
        w("| Exit Reason | n | Mean Net | Sum Net | Mean Days Held |")
        w("|---|---:|---:|---:|---:|")
        for _, r in results[rvol_cfg]["per_exit_reason"].iterrows():
            w(f"| {r['exit_reason']} | {int(r['n'])} | {r['mean_net']*100:+.3f}% | "
              f"{r['sum_net']*100:+.2f}% | {r['mean_days_held']:.2f} |")
        w()
        w("Interpretation: if `rvol_exit` rows have higher mean net than `max_hold`, the RVOL overlay "
          "is capturing value by exiting before mean-reversion. If lower, it's exiting too soon.")
        w()

    # Per-bucket breakdown for headline config
    if results[headline]["summary"]:
        w(f"## 4. Per-bucket breakdown — `{headline}`")
        w()
        w("| Bucket | n | Mean Net | Sum Net | NAV q% @4% |")
        w("|---|---:|---:|---:|---:|")
        for _, r in results[headline]["per_bucket"].iterrows():
            w(f"| {r['bucket']} | {int(r['n'])} | {r['mean_net']*100:+.3f}% | "
              f"{r['sum_net']*100:+.2f}% | {r['nav_q_pct']:+.2f}% |")
        w()

    # Critical comparison: build vs Gemini's "optimistic +250-300%" projection
    w("## 5. Verdict")
    w()
    w("**Comparator anchors**:")
    w("- STRICT_LO_SETUP60 8Q: +20.948% / Sh 0.631 / DD -10.31% / 6/8 pos Q  ")
    w("- Naive +25% Binance: NAV @4% / 5d = +38.16% (Bull+Chop+Bear+Crash all-events)  ")
    w("- Gemini optimistic projection: +250-300% with full optimization  ")
    w()
    w("**Our shipped builds**:")
    naive = results.get("NAIVE_+15%_5d_NO_GATE", {}).get("summary", {})
    b1a   = results.get("BUILD1_+15%_5d_GATED_BUCKETED", {}).get("summary", {})
    b1b   = results.get("BUILD1_+25%_5d_GATED_BUCKETED", {}).get("summary", {})
    b1c   = results.get("BUILD1_+15%_5d_BULLONLY_BUCKETED", {}).get("summary", {})
    b3    = results.get("BUILD3_+15%_5d_GATED_RVOL", {}).get("summary", {})
    b13   = results.get("BUILD1+3_+25%_5d_BULLONLY_RVOL", {}).get("summary", {})
    if naive: w(f"- Naive (no gate/bucket): {naive['total_nav_8q_pct']:+.2f}% ({naive['n_trades_closed']} trades)")
    if b1a:   w(f"- Build #1 (+15% gated bucketed 5d): **{b1a['total_nav_8q_pct']:+.2f}%** ({b1a['n_trades_closed']} trades)")
    if b1b:   w(f"- Build #1 (+25% gated bucketed 5d): **{b1b['total_nav_8q_pct']:+.2f}%** ({b1b['n_trades_closed']} trades)")
    if b1c:   w(f"- Build #1 (+15% Bull-only bucketed 5d): **{b1c['total_nav_8q_pct']:+.2f}%** ({b1c['n_trades_closed']} trades)")
    if b3:    w(f"- Build #3 (+15% gated + RVOL exit): **{b3['total_nav_8q_pct']:+.2f}%** ({b3['n_trades_closed']} trades)")
    if b13:   w(f"- Build #1+#3 (+25% Bull-only + RVOL): **{b13['total_nav_8q_pct']:+.2f}%** ({b13['n_trades_closed']} trades)")
    w()
    w("**Annualized (mid-window)**: best config NAV ÷ 2 years = ~equivalent annual ROI.")
    w()
    w("**Capital lockup note**: at max 5 simultaneous positions × 5-day max hold, throughput cap = "
      "~1 trade/day worst case. The skipped-capacity counter tells you how many triggers we missed.")

    out_path = OUT_DIR / "MOVER_LANE_BUILDS_2026_05_18.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Save trade ledgers for downstream analysis
    for name, res in results.items():
        if "trade_log" in res and isinstance(res["trade_log"], pd.DataFrame) and len(res["trade_log"]):
            p = OUT_DIR / f"mover_lane_trades_{name}.parquet"
            res["trade_log"].to_parquet(p)
    print(f"Wrote trade ledgers under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
