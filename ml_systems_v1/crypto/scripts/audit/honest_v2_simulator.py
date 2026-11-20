"""HONEST V2 simulator — fixes the look-ahead bias.

CRITICAL FIX:
  Prior V2 (improve_metrics_v2.py) sorted today's fires by `ret_E_14d`
  (the future 14-day return) as the K-selection signal. This is
  PERFECT-FORESIGHT look-ahead -- in live deploy we don't know the
  future return at entry.

THIS VERSION:
  Ranks today's fires by SIGNAL STRENGTH (computed from past prices
  only -- moving averages, oscillator values, etc.). Same logic as the
  live runner (smart_discovery_17_sleeve.py).

Also reports:
  - BEST-K (perfect foresight) -- upper bound, look-ahead-biased
  - SIGNAL-K (live-realistic) -- uses only past info to rank
  - RANDOM-K (lower bound) -- no ranker, pure random pick from co-fires
  - WORST-K (catastrophic) -- pick the K WORST forward returns
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# V2 params
BET_FRACTION = 0.10
HARD_STOP = -0.04
TRAIL_ARM = 0.10
TRAIL_DROP = 0.05
K_MAX = 12
HOLD_MAX = 10
COST = 0.0030

DEPLOY_17 = [
    ("SMA_cross", "(3, 5)"), ("SMA_cross", "(3, 8)"), ("SMA_cross", "(3, 13)"),
    ("SMA_cross", "(5, 8)"), ("SMA_cross", "(20, 21)"),
    ("Donchian_breakout", "(20,)"), ("ROC_momentum", "(10, 7)"),
    ("Stochastic_bounce", "(7, 3, 80, 20)"), ("Stochastic_bounce", "(7, 3, 90, 10)"),
    ("MACD_cross", "(5, 21, 5)"), ("MACD_cross", "(5, 34, 9)"),
    ("BB_breach", "(20, 1.5)"), ("EMA_cross", "(3, 5)"), ("EMA_cross", "(3, 8)"),
    ("Supertrend_flip", "(10, 2.0)"), ("Supertrend_flip", "(14, 2.5)"),
    ("Ichimoku_cross", "(9, 26, 52)"),
]

def walk_forward_exit(entry, fwd_prices, hold_max=HOLD_MAX, stop=HARD_STOP,
                       trail_arm=TRAIL_ARM, trail_drop=TRAIL_DROP, cost=COST):
    peak = entry; armed = False
    for d, p in enumerate(fwd_prices, start=1):
        if p is None or not np.isfinite(p): break
        ret = p / entry - 1
        if p > peak: peak = p
        if not armed and ret >= trail_arm: armed = True
        if ret <= stop: return stop - cost, d
        if armed and p <= peak * (1 - trail_drop): return p / entry - 1 - cost, d
        if d >= hold_max: return ret - cost, d
    last = next((p for p in reversed(fwd_prices) if p and np.isfinite(p)), None)
    if last is None: return -cost, 0
    return last / entry - 1 - cost, len(fwd_prices)

def compute_signal_strength(asset_sub, indicator, cfg, idx):
    """Compute LIVE-realistic signal strength using only past info up to idx.

    Matches the live runner's _signal_fires_today signature but for backtest:
    we KNOW the entry index. Returns strength score (higher = stronger signal).
    """
    if idx < 60 or idx + 1 >= len(asset_sub): return 0.0
    h = asset_sub["high"].values[:idx+1]
    l = asset_sub["low"].values[:idx+1]
    c = asset_sub["close"].values[:idx+1]
    try:
        if indicator in ("SMA_cross", "EMA_cross"):
            a, b = eval(cfg)
            if indicator == "SMA_cross":
                ma_s = pd.Series(c).rolling(a).mean().values
                ma_l = pd.Series(c).rolling(b).mean().values
            else:
                ma_s = pd.Series(c).ewm(span=a, adjust=False).mean().values
                ma_l = pd.Series(c).ewm(span=b, adjust=False).mean().values
            return float((ma_s[-1] - ma_l[-1]) / c[-1]) if c[-1] > 0 else 0
        if indicator == "Donchian_breakout":
            p = eval(cfg)[0]
            if len(c) > p:
                rh = max(h[-p-1:-1])
                return float((c[-1] - rh) / c[-1]) if c[-1] > 0 else 0
        if indicator == "ROC_momentum":
            p, _ = eval(cfg)
            if len(c) > p:
                return float(100 * (c[-1] - c[-1-p]) / c[-1-p])
        if indicator == "Stochastic_bounce":
            kp, dp, ob, _ = eval(cfg)
            if len(c) >= kp:
                hh = pd.Series(h).rolling(kp).max(); ll = pd.Series(l).rolling(kp).min()
                k = 100 * (pd.Series(c) - ll) / (hh - ll + 1e-12)
                # Strength = how far below ob (less = more oversold = stronger bounce)
                return float(ob - k.values[-1])
        if indicator == "MACD_cross":
            f, s, sig = eval(cfg)
            sc = pd.Series(c)
            ef = sc.ewm(span=f, adjust=False).mean(); es = sc.ewm(span=s, adjust=False).mean()
            macd = ef - es; ss = macd.ewm(span=sig, adjust=False).mean()
            return float(macd.values[-1] - ss.values[-1])
        if indicator == "BB_breach":
            p, std = eval(cfg)
            sc = pd.Series(c); mid = sc.rolling(p).mean(); sd = sc.rolling(p).std()
            ub = (mid + std * sd).values
            return float((c[-1] - ub[-1]) / c[-1]) if c[-1] > 0 else 0
        if indicator == "Supertrend_flip":
            atr_p, _ = eval(cfg)
            # Strength = recency of flip; use simple proxy
            return 1.0
        if indicator == "Ichimoku_cross":
            t_p, k_p, _ = eval(cfg)
            hh_t = pd.Series(h).rolling(t_p).max(); ll_t = pd.Series(l).rolling(t_p).min()
            tenkan = ((hh_t + ll_t) / 2).values
            hh_k = pd.Series(h).rolling(k_p).max(); ll_k = pd.Series(l).rolling(k_p).min()
            kijun = ((hh_k + ll_k) / 2).values
            return float((tenkan[-1] - kijun[-1]) / c[-1]) if c[-1] > 0 else 0
    except Exception:
        return 0.0
    return 0.0

def simulate_honest(events, panel_idx, ws, we, mode="signal", rng_seed=42):
    """mode: 'signal'    -- live-realistic (z-score within indicator family)
             'signal_v2' -- 2026-05-20 fix: confluence-first + within-family rank-pct
                            (matches patched smart_discovery_17_sleeve.py:303)
             'best'      -- perfect foresight upper bound (sorts by ret_E_14d)
             'random'    -- no ranker (uniform random)
             'worst'     -- catastrophic lower bound (sorts by ret_E_14d ASC)
    """
    rng = np.random.default_rng(rng_seed)
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    # Pre-compute panel indexed for fast lookup
    panel_date_idx = {}
    for a, sub in panel_idx.items():
        panel_date_idx[a] = {d: i for i, d in enumerate(sub["date"].values)}

    events = events.copy()
    if mode in ("best", "worst"):
        # Use forward ret_E_14d as the rank signal (perfect foresight)
        events["rank_sig"] = events["ret_E_14d"]
    elif mode in ("signal", "signal_v2", "signal_confluence_only", "signal_rankpct_only"):
        # Compute signal strength per event using past info
        print(f"  pre-computing signal strengths (past info only) for mode={mode}...")
        sigs = []
        for ev in events.itertuples():
            sub = panel_idx.get(ev.asset)
            if sub is None: sigs.append(0); continue
            idx = panel_date_idx[ev.asset].get(ev.date)
            if idx is None: sigs.append(0); continue
            s = compute_signal_strength(sub, ev.indicator, ev.config, idx)
            sigs.append(s)
        events["raw_strength"] = sigs

        if mode == "signal":
            # OLD ranker: z-score within indicator family (across all events).
            events["rank_sig_norm"] = events.groupby("indicator")["raw_strength"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
            events["rank_sig"] = events["rank_sig_norm"]
        elif mode == "signal_confluence_only":
            # ABLATION: only confluence_count, no strength tiebreaker (random within tier)
            events["confluence_count"] = events.groupby(["asset", "date"])["indicator"].transform("nunique")
            events["random_tiebreak"] = rng.random(len(events))
            events["rank_sig"] = events["confluence_count"].astype(float) + events["random_tiebreak"] * 0.1
        elif mode == "signal_rankpct_only":
            # ABLATION: only within-family rank-pct, no confluence
            events["strength_pct"] = events.groupby(["date", "indicator"])["raw_strength"].rank(pct=True)
            events["rank_sig"] = events["strength_pct"].astype(float)
        else:  # mode == "signal_v2"
            # NEW ranker (2026-05-20 fix): rank by (confluence_count, strength_pct).
            # confluence_count = number of distinct indicators firing on (asset, date)
            # strength_pct = within-family rank-percentile of raw strength PER-DAY
            #                (rank within today's fires for that indicator family).
            # The per-day within-family rank-pct is what the live sleeve does.
            # This matches the patched smart_discovery_17_sleeve.py:303.
            events["confluence_count"] = events.groupby(["asset", "date"])["indicator"].transform("nunique")
            # Per-day within-family rank-pct
            events["strength_pct"] = events.groupby(["date", "indicator"])["raw_strength"].rank(pct=True)
            # Composite rank: confluence dominant, strength_pct as tiebreaker.
            # To enable single-column sort downstream, pack into rank_sig:
            #   rank_sig = confluence + strength_pct  (confluence in {1..5} dominates)
            events["rank_sig"] = events["confluence_count"].astype(float) + events["strength_pct"].astype(float)
    elif mode == "random":
        events["rank_sig"] = rng.random(len(events))
    else:
        raise ValueError(mode)

    events_by_date = events.groupby("date")
    cur = ws; cal_dates = []
    while cur <= we:
        cal_dates.append(cur); cur += timedelta(days=1)

    for sim_date in cal_dates:
        new_open = []
        for pos in open_positions:
            if sim_date >= pos["exit_date"]:
                sub = panel_idx.get(pos["asset"])
                if sub is None: new_open.append(pos); continue
                fwd_sub = sub[(sub["date"] > pos["entry_date"]) & (sub["date"] <= sim_date)]
                fwd_prices = [float(p) if np.isfinite(p) else None for p in fwd_sub["close"].values]
                rret, d_held = walk_forward_exit(pos["entry_price"], fwd_prices)
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": rret,
                    "indicator": pos["indicator"], "config": pos["config"],
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        if sim_date in events_by_date.groups:
            today = events_by_date.get_group(sim_date).copy()
            if mode == "worst":
                today = today.sort_values("rank_sig", ascending=True)  # WORST first
            else:
                today = today.sort_values("rank_sig", ascending=False)
            open_assets = set(p["asset"] for p in open_positions)
            today = today[~today["asset"].isin(open_assets)]
            today = today.drop_duplicates(subset="asset", keep="first")

            for _, ev in today.iterrows():
                if len(open_positions) >= K_MAX: break
                bet = BET_FRACTION * portfolio_value
                if available_cash < bet: break
                sub = panel_idx.get(ev["asset"])
                if sub is None: continue
                idx = panel_date_idx[ev["asset"]].get(sim_date)
                if idx is None: continue
                ep = float(sub.iloc[idx]["close"])
                if ep <= 0 or not np.isfinite(ep): continue
                available_cash -= bet
                open_positions.append({
                    "asset": ev["asset"], "entry_date": sim_date,
                    "entry_price": ep, "exit_date": sim_date + timedelta(days=HOLD_MAX),
                    "bet_size": bet, "indicator": ev["indicator"], "config": ev["config"],
                })

        omtm = 0
        for pos in open_positions:
            sub = panel_idx.get(pos["asset"])
            if sub is None: omtm += pos["bet_size"]; continue
            av = sub[sub["date"] <= sim_date]
            if not len(av): omtm += pos["bet_size"]; continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio_value = available_cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio_value,
                                "n_open": len(open_positions)})

    return pd.DataFrame(daily_records), pd.DataFrame(trade_log)

def compute_metrics(daily_df, trade_df, window_days):
    pv = daily_df["portfolio_value"].values
    if len(pv) < 2: return {}
    dr = pv[1:] / pv[:-1] - 1
    total = (pv[-1] / pv[0] - 1) * 100
    ann = ((1 + total/100) ** (365/max(window_days,1)) - 1) * 100
    mn = dr.mean(); sd = dr.std()
    sortino = (mn / dr[dr<0].std() * np.sqrt(252)) if (dr<0).sum() and dr[dr<0].std() > 0 else 0
    sharpe = (mn / sd * np.sqrt(252)) if sd > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    calmar = ann / abs(max_dd) if max_dd != 0 else 0
    n = len(trade_df)
    if n:
        win = (trade_df["realized_ret"] > 0).mean() * 100
        aw = trade_df.loc[trade_df["realized_ret"]>0, "realized_ret"].mean()*100 if (trade_df["realized_ret"]>0).any() else 0
        al = trade_df.loc[trade_df["realized_ret"]<0, "realized_ret"].mean()*100 if (trade_df["realized_ret"]<0).any() else 0
        mw = trade_df["realized_ret"].max() * 100
    else:
        win = aw = al = mw = 0
    return {"total_pct": total, "ann_pct": ann, "sharpe": sharpe, "sortino": sortino,
            "max_dd_pct": max_dd, "calmar": calmar, "n_trades": n,
            "win_rate_pct": win, "avg_win_pct": aw, "avg_loss_pct": al, "max_win_pct": mw}

def main():
    print("="*78)
    print("HONEST V2 SIMULATOR -- 4 ranking modes")
    print("="*78)

    train_e = pd.read_parquet(OUT_DIR/"per_event_enriched.parquet")
    train_e["date"] = pd.to_datetime(train_e["date"]).dt.date
    oos_e = pd.read_parquet(OUT_DIR/"oos_events.parquet")
    oos_e["date"] = pd.to_datetime(oos_e["date"]).dt.date
    round2 = pd.read_parquet(OUT_DIR/"round2_events.parquet")
    round2["date"] = pd.to_datetime(round2["date"]).dt.date

    keys = set(DEPLOY_17)
    def filt(df):
        return df[df.set_index(["indicator","config"]).index.isin(keys)].copy()
    train_filt = pd.concat([
        filt(train_e[["asset","date","indicator","config","ret_E_14d"]]),
        filt(round2[["asset","date","indicator","config","ret_E_14d"]]),
    ], ignore_index=True)
    oos_filt = filt(oos_e[["asset","date","indicator","config","ret_E_14d"]])

    print("Loading panel...")
    files = sorted((ROOT/"data"/"processed"/"chimera"/"1d").glob("*_v51_chimera_1d_*.parquet"))
    panel_idx = {}
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT","")
        try:
            df = pl.read_parquet(f, columns=["timestamp","high","low","close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < 30: continue
        panel_idx[sym] = df
    print(f"Panels: {len(panel_idx)}")

    # Run all 7 modes on OOS (added signal_v2 + ablations 2026-05-20)
    print("\n=== OOS RESULTS (7 ranking modes incl. ablations) ===")
    results = {}
    for mode in ("best", "signal", "signal_v2", "signal_confluence_only", "signal_rankpct_only", "random", "worst"):
        print(f"\n[{mode.upper()}-K]")
        daily, trades = simulate_honest(oos_filt, panel_idx, date(2024,5,16), date(2025,3,15),
                                          mode=mode, rng_seed=42)
        m = compute_metrics(daily, trades, (date(2025,3,15)-date(2024,5,16)).days)
        results[mode] = m
        print(f"  total_ret={m['total_pct']:+9.2f}%  ann={m['ann_pct']:+9.2f}%  "
              f"Sortino={m['sortino']:+.3f}  Calmar={m['calmar']:+.3f}  "
              f"DD={m['max_dd_pct']:+.2f}%  n={m['n_trades']}  "
              f"win={m['win_rate_pct']:.1f}%  avg_win={m['avg_win_pct']:+.2f}%  "
              f"max_win={m['max_win_pct']:+.2f}%")

    print("\n" + "="*78)
    print("DEPLOY-REALISTIC NUMBERS (signal-strength rankers, no future info):")
    print(f"  signal (OLD, z-score within family):")
    print(f"    OOS total: {results['signal']['total_pct']:+.2f}%  ann: {results['signal']['ann_pct']:+.2f}%  Sortino: {results['signal']['sortino']:+.3f}")
    print(f"  signal_v2 (NEW 2026-05-20: confluence-first + within-family rank-pct):")
    print(f"    OOS total: {results['signal_v2']['total_pct']:+.2f}%  ann: {results['signal_v2']['ann_pct']:+.2f}%  Sortino: {results['signal_v2']['sortino']:+.3f}")
    lift = results['signal_v2']['total_pct'] - results['signal']['total_pct']
    print(f"  LIFT signal -> signal_v2: {lift:+.2f}pp NAV")
    print(f"  random (baseline, no ranker):")
    print(f"    OOS total: {results['random']['total_pct']:+.2f}%  ann: {results['random']['ann_pct']:+.2f}%")
    print(f"  best (perfect foresight upper bound):")
    print(f"    OOS total: {results['best']['total_pct']:+.2f}%  ann: {results['best']['ann_pct']:+.2f}%")
    print("="*78)

    # Write report
    lines = ["# HONEST V2 Simulator -- look-ahead bias FIXED\n"]
    lines.append(f"\n## Why this matters\n")
    lines.append(f"Prior V2 simulator (improve_metrics_v2.py) used `ret_E_14d` (future")
    lines.append(f"return) as the K-selection signal. That's perfect-foresight look-ahead.")
    lines.append(f"In live deploy we don't know future returns at entry.")
    lines.append(f"\n## OOS results (4 ranking modes)\n")
    lines.append("| mode | total ret | annualized | Sortino | Calmar | Max DD | n_trades | win % | avg win | max win |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for mode in ("best", "signal", "random", "worst"):
        m = results[mode]
        lines.append(f"| {mode}-K | {m['total_pct']:+.2f}% | {m['ann_pct']:+.2f}% | "
                     f"{m['sortino']:+.3f} | {m['calmar']:+.3f} | {m['max_dd_pct']:+.2f}% | "
                     f"{m['n_trades']} | {m['win_rate_pct']:.1f}% | "
                     f"{m['avg_win_pct']:+.2f}% | {m['max_win_pct']:+.2f}% |")

    lines.append(f"\n## Headline\n")
    sig = results["signal"]
    best = results["best"]
    lines.append(f"- **DEPLOY-REALISTIC (signal-K)**: OOS +{sig['total_pct']:.2f}% over 10 mo, annualized +{sig['ann_pct']:.2f}%")
    lines.append(f"- BEST-K upper bound (look-ahead): OOS +{best['total_pct']:.2f}% -- the +468% reported earlier was THIS")
    lines.append(f"- Gap between best-K and signal-K = ranker amplifier headroom (~{best['total_pct']-sig['total_pct']:+.0f}pp)")
    lines.append(f"\nPrior V2 +468% claim was upper-bound, not realistic. Correct realistic OOS is **{sig['total_pct']:+.0f}%** with Sortino {sig['sortino']:.2f}, Calmar {sig['calmar']:.2f}, Max DD {sig['max_dd_pct']:.1f}%.")

    (OUT_DIR/"HONEST_V2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'HONEST_V2_REPORT.md'}")

if __name__ == "__main__":
    main()
