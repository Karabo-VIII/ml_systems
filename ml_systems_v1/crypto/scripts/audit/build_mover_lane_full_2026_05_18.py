"""Composed Mover-Lane Builds A/B/C/D — final empirical battery.

Items tested in one pass:
  A. Build #2 + #3 composed: intraday +N% cross entry + RVOL-exhaustion exit
  B. Cap relaxation: 2% size x 10 simultaneous OR per-bucket parallel cap
  C. Bull-only gate applied to Build #2 (BTC 30d >= +5%)
  D. Intraday cross sensitivity: thresholds +10%, +15%, +20%

Data:
  - 1h panel (Binance pull): entry detection (cumulative-day-return cross)
  - 1d panel (Binance pull): exit evaluation, regime, RVOL

Output:
  - runs/audit/MOVER_LANE_FULL_BUILDS_2026_05_18.md (results table)
  - runs/audit/mover_lane_full_trades_<NAME>.parquet (per-config ledger)
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PANEL_PATH = OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"

# 31 DEGEN/VOLATILE symbols already pulled for Build #2
SYMBOLS_1H = [
    "BONK", "PEPE", "SHIB", "WIF", "WLD",
    "ADA", "ALGO", "ATOM", "AVAX", "DOGE", "DOT", "FET", "FIL",
    "LINK", "LTC", "NEAR", "RENDER", "TAO", "ZEC",
    "ARKM", "AR", "ENA", "FLOKI", "PNUT", "SUI", "SUPER", "SEI", "TIA", "ORDI",
    "ZEN", "PENGU",
]

START_MS = 1704067200000
END_MS   = 1767225599999


def fetch_klines_1h_cached(sym):
    """Fetch all 1h klines for a symbol; cache on disk in runs/audit/1h_cache/{sym}.parquet."""
    cache_dir = OUT_DIR / "1h_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_p = cache_dir / f"{sym}.parquet"
    if cache_p.exists():
        return pd.read_parquet(cache_p), "cached"
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        for try_sym in [f"{sym}USDT", f"1000{sym}USDT"]:
            params = {"symbol": try_sym, "interval": "1h",
                      "startTime": cur, "endTime": END_MS, "limit": 1000}
            try:
                r = requests.get(url, params=params, timeout=20)
            except Exception:
                return None, "net-error"
            if r.status_code == 200 and r.json():
                data = r.json()
                for k in data:
                    all_rows.append({
                        "asset": sym, "openTime": k[0],
                        "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]),  "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                cur = data[-1][0] + 3600_000
                break
        else:
            break
        time.sleep(0.06)
    if not all_rows:
        return None, "empty"
    df = pd.DataFrame(all_rows)
    df.to_parquet(cache_p)
    return df, "fetched"


def load_panels():
    """Load 1h (entry detection) + 1d (exit/regime/RVOL) panels."""
    # 1h panel
    frames_1h = []
    for s in SYMBOLS_1H:
        df, info = fetch_klines_1h_cached(s)
        if df is None or len(df) < 200:
            continue
        frames_1h.append(df)
    p1h = pd.concat(frames_1h, ignore_index=True)
    p1h["dt"]   = pd.to_datetime(p1h["openTime"], unit="ms", utc=True)
    p1h["date"] = p1h["dt"].dt.date
    p1h = p1h.sort_values(["asset", "dt"]).reset_index(drop=True)
    p1h["day_open_close"] = p1h.groupby(["asset", "date"])["close"].transform("first")
    p1h["cum_day_ret"] = p1h["close"] / p1h["day_open_close"] - 1
    # 1d panel (already has vol_10d_ma + rvol from earlier; recompute if missing)
    p1d = pd.read_parquet(DAILY_PANEL_PATH)
    if "vol_10d_ma" not in p1d.columns:
        p1d["vol_10d_ma"] = p1d.groupby("asset")["volume"].transform(
            lambda s: s.rolling(10, min_periods=3).mean())
        p1d["rvol"] = p1d["volume"] / p1d["vol_10d_ma"]
    return p1h, p1d


@dataclass
class CfgFull:
    name: str
    trigger_thresh: float = 0.15
    regime_min_btc_30d: float = -0.05
    bucket_filter: tuple = ("DEGEN", "VOLATILE")
    max_hold_days: int = 5
    rvol_exit_enabled: bool = False
    rvol_exit_threshold: float = 0.5
    rvol_exit_consecutive_days: int = 2
    cost_rt: float = 0.0024
    size_per_entry: float = 0.04
    max_simultaneous: int = 5
    per_bucket_cap: Optional[dict] = None    # e.g. {"DEGEN":5,"VOLATILE":5}
    entry_hour_cap: Optional[int] = None     # only accept crosses at hour <= N


def simulate_full(p1h: pd.DataFrame, p1d: pd.DataFrame, cfg: CfgFull) -> dict:
    """Walk-forward sim. Intraday entry, daily exit eval."""
    # Build 1d close + rvol index per asset
    close_1d = {}
    rvol_1d = {}
    for a, sub in p1d.groupby("asset"):
        s = sub.set_index("date").sort_index()
        close_1d[a] = s["close"]
        rvol_1d[a]  = s["rvol"]
    # BTC 30d per date
    btc30d = p1d[p1d["asset"] == "BTC"].set_index("date").sort_index()["btc_30d"]
    # Bucket lookup
    bucket_of = p1d.drop_duplicates("asset").set_index("asset")["bucket"].to_dict()

    # Find FIRST intraday cross per (asset, date)
    fc = (p1h[p1h["cum_day_ret"] >= cfg.trigger_thresh]
            .groupby(["asset", "date"]).first().reset_index())
    fc["entry_dt"] = fc["dt"]
    fc["entry_hour"] = fc["entry_dt"].dt.hour
    fc["entry_close"] = fc["close"]
    fc["bucket"] = fc["asset"].map(bucket_of).fillna("VOLATILE")
    # Filters
    fc = fc[fc["bucket"].isin(cfg.bucket_filter)]
    if cfg.entry_hour_cap is not None:
        fc = fc[fc["entry_hour"] <= cfg.entry_hour_cap]
    # Regime gate (join btc30d at trigger date)
    fc["btc30d"] = fc["date"].map(btc30d.to_dict())
    fc = fc[fc["btc30d"] >= cfg.regime_min_btc_30d]
    fc = fc.sort_values("entry_dt").reset_index(drop=True)

    if len(fc) == 0:
        return {"summary": {}, "trades": pd.DataFrame(), "n_skipped_cap": 0}

    open_pos = {}  # asset -> dict, keyed by asset to avoid double-positions
    bucket_open = {b: 0 for b in cfg.bucket_filter}
    trades = []
    skipped_cap = 0

    def cap_blocked(bucket):
        if cfg.per_bucket_cap is not None:
            return bucket_open.get(bucket, 0) >= cfg.per_bucket_cap.get(bucket, 999)
        return sum(bucket_open.values()) >= cfg.max_simultaneous

    def close_position(pos, today_date, today_close, reason):
        gross = today_close / pos["entry_close"] - 1
        net = gross - cfg.cost_rt
        trades.append({
            "asset": pos["asset"],
            "bucket": pos["bucket"],
            "trigger_date": pos["trigger_date"],
            "entry_dt": pos["entry_dt"],
            "exit_date": today_date,
            "days_held": (today_date - pos["trigger_date"]).days,
            "entry_close": pos["entry_close"],
            "exit_close": today_close,
            "gross_ret": gross,
            "net_ret": net,
            "exit_reason": reason,
            "regime_btc30d": pos["btc30d"],
            "quarter": pos["quarter"],
        })
        bucket_open[pos["bucket"]] -= 1

    quarters_by_date = {}
    for _, r in p1d.drop_duplicates("date").iterrows():
        quarters_by_date[r["date"]] = r.get("quarter", None)

    # Drive simulation day-by-day on 1d panel dates
    unique_dates = sorted(p1d["date"].unique())
    fc_by_date = {d: [] for d in unique_dates}
    for i, r in fc.iterrows():
        fc_by_date.setdefault(r["date"], []).append(i)

    for today in unique_dates:
        # 1. Age open positions, check exits
        for asset in list(open_pos.keys()):
            pos = open_pos[asset]
            if today <= pos["trigger_date"]:
                continue
            days_held = (today - pos["trigger_date"]).days
            try:
                today_close = close_1d[asset].loc[today]
                today_rvol = rvol_1d[asset].loc[today]
            except KeyError:
                continue
            if days_held >= cfg.max_hold_days:
                close_position(pos, today, today_close, "max_hold")
                del open_pos[asset]
                continue
            if cfg.rvol_exit_enabled and pd.notna(today_rvol):
                if today_rvol < cfg.rvol_exit_threshold:
                    pos["days_low_rvol"] = pos.get("days_low_rvol", 0) + 1
                    if pos["days_low_rvol"] >= cfg.rvol_exit_consecutive_days:
                        close_position(pos, today, today_close, "rvol_exit")
                        del open_pos[asset]
                        continue
                else:
                    pos["days_low_rvol"] = 0

        # 2. Enter new positions
        for idx in fc_by_date.get(today, []):
            row = fc.loc[idx]
            asset = row["asset"]
            if asset in open_pos:
                continue  # already in position
            if cap_blocked(row["bucket"]):
                skipped_cap += 1
                continue
            open_pos[asset] = {
                "asset": asset,
                "bucket": row["bucket"],
                "trigger_date": today,
                "entry_dt": row["entry_dt"],
                "entry_close": row["entry_close"],
                "btc30d": row["btc30d"],
                "quarter": quarters_by_date.get(today, "unk"),
                "days_low_rvol": 0,
            }
            bucket_open[row["bucket"]] += 1

    # Close any still-open positions at last available close
    last_date = unique_dates[-1]
    for asset, pos in list(open_pos.items()):
        try:
            today_close = close_1d[asset].loc[last_date]
        except KeyError:
            today_close = pos["entry_close"]
        close_position(pos, last_date, today_close, "window_end")
        del open_pos[asset]

    tl = pd.DataFrame(trades)
    if len(tl) == 0:
        return {"summary": {}, "trades": tl, "n_skipped_cap": skipped_cap}

    total_nav = cfg.size_per_entry * tl["net_ret"].sum() * 100
    summary = {
        "n_triggers_eligible": int(len(fc)),
        "n_skipped_capacity": int(skipped_cap),
        "n_trades_closed": int(len(tl)),
        "mean_days_held": float(tl["days_held"].mean()),
        "win_rate": float((tl["net_ret"] > 0).mean()),
        "mean_net_per_trade": float(tl["net_ret"].mean()),
        "total_gross_pct": float(tl["gross_ret"].sum() * 100),
        "total_net_pct": float(tl["net_ret"].sum() * 100),
        "total_nav_8q_pct": float(total_nav),
        "best_trade": float(tl["net_ret"].max() * 100),
        "worst_trade": float(tl["net_ret"].min() * 100),
        "median_trade": float(tl["net_ret"].median() * 100),
    }
    per_q = tl.groupby("quarter").agg(
        n=("net_ret","size"),
        mean_net=("net_ret","mean"),
        sum_net=("net_ret","sum"),
        win_rate=("net_ret", lambda s: (s > 0).mean()),
    ).reset_index()
    per_q["nav_q_pct"] = per_q["sum_net"] * cfg.size_per_entry * 100
    return {"summary": summary, "trades": tl, "n_skipped_cap": skipped_cap,
            "per_quarter": per_q}


def main():
    print("Loading panels...")
    p1h, p1d = load_panels()
    print(f"1h: {len(p1h)} rows / {p1h['asset'].nunique()} assets")
    print(f"1d: {len(p1d)} rows / {p1d['asset'].nunique()} assets")

    configs = [
        # ---- D: intraday cross sensitivity (+10%, +15%, +20%) ----
        CfgFull(name="D_intraday_+10pct_5d_4pct_cap5",
                trigger_thresh=0.10, size_per_entry=0.04, max_simultaneous=5),
        CfgFull(name="D_intraday_+15pct_5d_4pct_cap5",
                trigger_thresh=0.15, size_per_entry=0.04, max_simultaneous=5),
        CfgFull(name="D_intraday_+20pct_5d_4pct_cap5",
                trigger_thresh=0.20, size_per_entry=0.04, max_simultaneous=5),
        # ---- C: Bull-only gate on +15% intraday ----
        CfgFull(name="C_intraday_+15pct_BULLONLY_5d_4pct_cap5",
                trigger_thresh=0.15, regime_min_btc_30d=0.05),
        # ---- A: Compose #2 + #3 (intraday + RVOL exit) ----
        CfgFull(name="A_intraday_+15pct_RVOLexit_5d_4pct_cap5",
                trigger_thresh=0.15, rvol_exit_enabled=True),
        CfgFull(name="A_intraday_+20pct_RVOLexit_5d_4pct_cap5",
                trigger_thresh=0.20, rvol_exit_enabled=True),
        # ---- B: Cap relaxation ----
        # B1: 2% size x 10 positions (same risk, doubled throughput)
        CfgFull(name="B1_intraday_+15pct_5d_2pct_cap10",
                trigger_thresh=0.15, size_per_entry=0.02, max_simultaneous=10),
        # B2: per-bucket cap (DEGEN 5 + VOLATILE 5 = 10 total, 4% each = 40% max exposure)
        CfgFull(name="B2_intraday_+15pct_5d_4pct_perbucket5",
                trigger_thresh=0.15, max_simultaneous=999,
                per_bucket_cap={"DEGEN":5, "VOLATILE":5}),
        # B3: 2% size x 20 positions
        CfgFull(name="B3_intraday_+15pct_5d_2pct_cap20",
                trigger_thresh=0.15, size_per_entry=0.02, max_simultaneous=20),
        # ---- Best composed (A + B + C) ----
        CfgFull(name="BEST_intraday_+15pct_BULL_RVOL_2pct_cap10",
                trigger_thresh=0.15, regime_min_btc_30d=0.05,
                rvol_exit_enabled=True,
                size_per_entry=0.02, max_simultaneous=10),
        # Tightest signal-quality
        CfgFull(name="TIGHT_intraday_+20pct_BULL_RVOL_4pct_cap5",
                trigger_thresh=0.20, regime_min_btc_30d=0.05,
                rvol_exit_enabled=True),
    ]

    results = {}
    print()
    print("=" * 78)
    print(f"RUNNING {len(configs)} CONFIGURATIONS")
    print("=" * 78)
    for cfg in configs:
        res = simulate_full(p1h, p1d, cfg)
        results[cfg.name] = res
        s = res["summary"]
        if not s:
            print(f"\n[{cfg.name}] NO TRADES")
            continue
        print(f"\n[{cfg.name}]")
        print(f"  triggers={s['n_triggers_eligible']} trades={s['n_trades_closed']} "
              f"skipped_cap={s['n_skipped_capacity']} "
              f"days_held_mean={s['mean_days_held']:.2f}")
        print(f"  win_rate={s['win_rate']*100:.1f}% mean_net={s['mean_net_per_trade']*100:+.3f}% "
              f"median={s['median_trade']:+.3f}%")
        print(f"  total_gross={s['total_gross_pct']:+.2f}% total_net={s['total_net_pct']:+.2f}% "
              f"NAV_8Q={s['total_nav_8q_pct']:+.2f}%")

    # ----- Generate report -----
    lines = []
    def w(s=""):
        lines.append(s)

    w("# Mover-Lane Full Build Battery — Real Binance 8Q Walk-Forward")
    w()
    w("**Date**: 2026-05-18  ")
    w(f"**Window**: 24Q1 -> 25Q4 (UNSEEN NOT TOUCHED)  ")
    w(f"**Universe**: 31 DEGEN/VOLATILE symbols (1h intraday entry + 1d exit eval)  ")
    w("**Cost**: 24bps RT  ")
    w()
    w("Items tested (per prior unified verdict §8):")
    w("- A: Compose #2 + #3 (intraday entry + RVOL exhaustion exit)")
    w("- B: Cap relaxation (2% x 10, 4% x per-bucket-5, 2% x 20)")
    w("- C: Bull-only gate applied to Build #2")
    w("- D: Intraday cross sensitivity (+10%, +15%, +20%)")
    w()
    w("---")
    w()
    w("## 1. Headline ranking")
    w()
    w("| Config | Trigger | Bull? | RVOL? | Size | Cap | Trades | Win % | Mean Net/Trade | NAV 8Q |")
    w("|---|---|:---:|:---:|---:|---|---:|---:|---:|---:|")
    sorted_results = sorted(
        [(name, results[name]) for name in [c.name for c in configs] if results[name]["summary"]],
        key=lambda x: x[1]["summary"]["total_nav_8q_pct"], reverse=True)
    cfg_by_name = {c.name: c for c in configs}
    for name, res in sorted_results:
        s = res["summary"]
        c = cfg_by_name[name]
        bull = "YES" if c.regime_min_btc_30d >= 0.05 else "no"
        rvol = "YES" if c.rvol_exit_enabled else "no"
        cap_label = (f"per-bucket {c.per_bucket_cap}" if c.per_bucket_cap
                     else f"{c.max_simultaneous}")
        w(f"| `{name}` | +{c.trigger_thresh*100:.0f}% | {bull} | {rvol} | "
          f"{c.size_per_entry*100:.0f}% | {cap_label} | "
          f"{s['n_trades_closed']} | {s['win_rate']*100:.1f}% | "
          f"{s['mean_net_per_trade']*100:+.3f}% | **{s['total_nav_8q_pct']:+.2f}%** |")
    w()

    # Per-quarter for best
    best_name = sorted_results[0][0]
    best_res = sorted_results[0][1]
    w(f"## 2. Per-quarter — best config: `{best_name}`")
    w()
    w(f"NAV 8Q: **{best_res['summary']['total_nav_8q_pct']:+.2f}%** | "
      f"trades: {best_res['summary']['n_trades_closed']} | "
      f"mean days held: {best_res['summary']['mean_days_held']:.2f} | "
      f"skipped @ cap: {best_res['summary']['n_skipped_capacity']}")
    w()
    w("| Quarter | n | Win % | Mean Net | NAV Q% @size |")
    w("|---|---:|---:|---:|---:|")
    for _, r in best_res["per_quarter"].iterrows():
        w(f"| {r['quarter']} | {int(r['n'])} | {r['win_rate']*100:.1f}% | "
          f"{r['mean_net']*100:+.3f}% | {r['nav_q_pct']:+.2f}% |")
    pos_q = (best_res["per_quarter"]["nav_q_pct"] > 0).sum()
    w(f"\nPositive quarters: {pos_q}/8 | "
      f"Worst quarter: {best_res['per_quarter']['nav_q_pct'].min():+.2f}%")
    w()

    # Build-by-item interpretation
    w("## 3. Item-by-item interpretation")
    w()
    w("### A — Compose #2 + #3 (intraday entry + RVOL exhaustion exit)")
    a_v15 = results.get("A_intraday_+15pct_RVOLexit_5d_4pct_cap5", {}).get("summary", {})
    a_v20 = results.get("A_intraday_+20pct_RVOLexit_5d_4pct_cap5", {}).get("summary", {})
    if a_v15:
        w(f"- +15% trigger: NAV={a_v15['total_nav_8q_pct']:+.2f}%, "
          f"mean_days_held={a_v15['mean_days_held']:.2f}, "
          f"win_rate={a_v15['win_rate']*100:.1f}%")
    if a_v20:
        w(f"- +20% trigger: NAV={a_v20['total_nav_8q_pct']:+.2f}%, "
          f"mean_days_held={a_v20['mean_days_held']:.2f}, "
          f"win_rate={a_v20['win_rate']*100:.1f}%")
    w()
    w("### B — Cap relaxation")
    b1 = results.get("B1_intraday_+15pct_5d_2pct_cap10", {}).get("summary", {})
    b2 = results.get("B2_intraday_+15pct_5d_4pct_perbucket5", {}).get("summary", {})
    b3 = results.get("B3_intraday_+15pct_5d_2pct_cap20", {}).get("summary", {})
    if b1: w(f"- 2% size x cap 10: NAV={b1['total_nav_8q_pct']:+.2f}%, trades={b1['n_trades_closed']}, skipped={b1['n_skipped_capacity']}")
    if b2: w(f"- 4% size x per-bucket-5: NAV={b2['total_nav_8q_pct']:+.2f}%, trades={b2['n_trades_closed']}, skipped={b2['n_skipped_capacity']}")
    if b3: w(f"- 2% size x cap 20: NAV={b3['total_nav_8q_pct']:+.2f}%, trades={b3['n_trades_closed']}, skipped={b3['n_skipped_capacity']}")
    w()
    w("### C — Bull-only gate applied to Build #2")
    c_v15 = results.get("C_intraday_+15pct_BULLONLY_5d_4pct_cap5", {}).get("summary", {})
    if c_v15: w(f"- Intraday +15% Bull-only: NAV={c_v15['total_nav_8q_pct']:+.2f}%, trades={c_v15['n_trades_closed']}")
    w()
    w("### D — Intraday cross sensitivity")
    d10 = results.get("D_intraday_+10pct_5d_4pct_cap5", {}).get("summary", {})
    d15 = results.get("D_intraday_+15pct_5d_4pct_cap5", {}).get("summary", {})
    d20 = results.get("D_intraday_+20pct_5d_4pct_cap5", {}).get("summary", {})
    if d10: w(f"- +10% threshold: NAV={d10['total_nav_8q_pct']:+.2f}%, trades={d10['n_trades_closed']}")
    if d15: w(f"- +15% threshold: NAV={d15['total_nav_8q_pct']:+.2f}%, trades={d15['n_trades_closed']}")
    if d20: w(f"- +20% threshold: NAV={d20['total_nav_8q_pct']:+.2f}%, trades={d20['n_trades_closed']}")
    w()

    # Composed final
    w("## 4. Best fully-composed config")
    best = results.get("BEST_intraday_+15pct_BULL_RVOL_2pct_cap10", {}).get("summary", {})
    if best:
        w(f"`BEST_intraday_+15pct_BULL_RVOL_2pct_cap10`: NAV **{best['total_nav_8q_pct']:+.2f}%** "
          f"over {best['n_trades_closed']} trades. "
          f"Combines: intraday entry + Bull-regime gate + RVOL exhaustion exit + 2% size x 10-position cap.")
    w()

    out_md = OUT_DIR / "MOVER_LANE_FULL_BUILDS_2026_05_18.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")

    # Persist trade ledgers
    for name, res in results.items():
        if isinstance(res.get("trades"), pd.DataFrame) and len(res["trades"]):
            res["trades"].to_parquet(OUT_DIR / f"mover_lane_full_trades_{name}.parquet")

    # Also dump summaries JSON
    summary_dump = {name: res.get("summary", {}) for name, res in results.items()}
    (OUT_DIR / "mover_lane_full_summaries_2026_05_18.json").write_text(
        json.dumps(summary_dump, indent=2, default=str))
    print(f"Wrote summaries JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
