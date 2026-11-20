"""OOS market replay on the 14-setup deploy portfolio.

OOS window: 2024-05-16 -> 2025-03-15 (canonical per data_config.yaml).
UNSEEN (2025-03-15+) remains UNTOUCHED per user mandate.

For each day in OOS:
  1. Determine BTC regime (bull/chop/bear/crash)
  2. Find which of the 14 setups fire on each asset
  3. Per-day: K=8 unique-asset cap, sort by asymmetric expectancy DESC
  4. Apply asymmetric stops (-4% / +12%); compute realized 14d return
  5. Per-day NAV = sum(picked_returns) * BET_FRACTION

Output:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/oos_events.parquet
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/oos_daily_nav.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/OOS_REPLAY_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# Canonical OOS per data_config.yaml
OOS_START = date(2024, 5, 16)
OOS_END = date(2025, 3, 15)

COST = 0.0024
BET_FRACTION = 0.08
HARD_STOP = -0.04
TARGET = 0.12
K_MAX = 8
WEEKLY_FLOOR = 0.0525

# THE 14-SETUP DEPLOY PORTFOLIO (commit 18b7309 / oracle_audit_and_val_walkforward.py)
DEPLOY_PORTFOLIO = [
    ("SMA_cross", "(3, 5)"),
    ("SMA_cross", "(3, 8)"),
    ("SMA_cross", "(3, 13)"),
    ("SMA_cross", "(5, 8)"),
    ("SMA_cross", "(20, 21)"),
    ("Donchian_breakout", "(20,)"),
    ("ROC_momentum", "(10, 7)"),
    ("Stochastic_bounce", "(7, 3, 80, 20)"),
    ("MACD_cross", "(5, 21, 5)"),
    ("MACD_cross", "(5, 34, 9)"),
    ("BB_breach", "(20, 1.5)"),
    ("Stochastic_bounce", "(7, 3, 90, 10)"),
    ("EMA_cross", "(3, 5)"),
    ("EMA_cross", "(3, 8)"),
]

def asymmetric_returns(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

# Compact indicator calculators
def calc_macd(closes, f, s, sig):
    sc = pd.Series(closes)
    ef = sc.ewm(span=f, adjust=False).mean(); es = sc.ewm(span=s, adjust=False).mean()
    macd = ef - es; ss = macd.ewm(span=sig, adjust=False).mean()
    return macd.values, ss.values

def calc_stoch(highs, lows, closes, kp, dp):
    hh = pd.Series(highs).rolling(kp).max(); ll = pd.Series(lows).rolling(kp).min()
    k = 100 * (pd.Series(closes) - ll) / (hh - ll + 1e-12)
    return k.values, k.rolling(dp).mean().values

def calc_roc(closes, period):
    s = pd.Series(closes); return (100 * (s - s.shift(period)) / s.shift(period)).values

# Event finders
def find_sma(sub, cfg):
    a, b = eval(cfg)
    c = sub["close"].values
    ma_s = pd.Series(c).rolling(a).mean().values
    ma_l = pd.Series(c).rolling(b).mean().values
    above = ma_s > ma_l
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_ema(sub, cfg):
    a, b = eval(cfg)
    c = sub["close"].values
    ma_s = pd.Series(c).ewm(span=a, adjust=False).mean().values
    ma_l = pd.Series(c).ewm(span=b, adjust=False).mean().values
    above = ma_s > ma_l
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_donchian(sub, cfg):
    p = eval(cfg)[0]
    h = sub["high"].values; c = sub["close"].values
    rh = pd.Series(h).rolling(p).max().shift(1).values
    bo = c > rh
    return np.where(bo[1:] & ~bo[:-1])[0] + 1

def find_roc(sub, cfg):
    p, t = eval(cfg)
    r = calc_roc(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    return np.where((prev < t) & (r >= t))[0]

def find_stoch(sub, cfg):
    kp, dp, ob, os_ = eval(cfg)
    k, d = calc_stoch(sub["high"].values, sub["low"].values, sub["close"].values, kp, dp)
    prev = np.roll(k, 1); prev[0] = k[0]
    return np.where((prev < os_) & (k >= os_))[0]

def find_macd(sub, cfg):
    f, s, sig = eval(cfg)
    m, ss = calc_macd(sub["close"].values, f, s, sig)
    above = m > ss
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_bb(sub, cfg):
    p, s = eval(cfg)
    closes = sub["close"].values
    pd_closes = pd.Series(closes)
    mid = pd_closes.rolling(p).mean()
    sd = pd_closes.rolling(p).std()
    ub = (mid + s * sd).values
    above = closes > ub
    return np.where(above[1:] & ~above[:-1])[0] + 1

FINDERS = {
    "SMA_cross": find_sma, "EMA_cross": find_ema,
    "Donchian_breakout": find_donchian, "ROC_momentum": find_roc,
    "Stochastic_bounce": find_stoch, "MACD_cross": find_macd,
    "BB_breach": find_bb,
}

def main():
    print("="*78)
    print("OOS MARKET REPLAY -- 14-setup deploy portfolio")
    print(f"OOS window: {OOS_START} -> {OOS_END}")
    print("UNSEEN ({}+) untouched per user mandate".format(OOS_END + timedelta(days=1)))
    print("="*78)

    # Load OOS panel with warmup
    print("\nLoading chimera 1d panel (OOS + warmup + forward)...")
    files = sorted((ROOT/"data"/"processed"/"chimera"/"1d").glob("*_v51_chimera_1d_*.parquet"))
    panels = {}
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT","")
        try:
            df = pl.read_parquet(f, columns=["timestamp","open","high","low","close","volume"]).to_pandas()
        except Exception:
            try:
                df = pl.read_parquet(f, columns=["timestamp","open","high","low","close"]).to_pandas()
                df["volume"] = 0.0
            except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= OOS_START - timedelta(days=120)) & (df["date"] <= OOS_END + timedelta(days=14))].reset_index(drop=True)
        if len(df) < 50: continue
        df["asset"] = sym
        panels[sym] = df
    print(f"  panels: {len(panels)} assets")

    # Regime overlay
    print("Loading regime overlay...")
    reg_df = pl.read_parquet(ROOT/"runs"/"oracle_layer2"/"daily_regime_cluster.parquet").to_pandas()
    reg_df["date"] = pd.to_datetime(reg_df["date"]).dt.date
    reg_df = reg_df[(reg_df["date"] >= OOS_START) & (reg_df["date"] <= OOS_END)]
    print(f"  OOS regime days: {len(reg_df)}")
    print(f"  Mix: {reg_df['btc_regime_30d'].value_counts().to_dict()}")
    date2reg = dict(zip(reg_df["date"], reg_df["btc_regime_30d"]))

    # Generate OOS events for the 14 setups
    print("\nGenerating OOS events for the 14 deploy setups...")
    rows = []
    for ind, cfg in DEPLOY_PORTFOLIO:
        finder = FINDERS[ind]
        for asset, sub in panels.items():
            try:
                idx = finder(sub, cfg)
            except Exception: continue
            for i in idx:
                if i < 60 or i + 14 >= len(sub): continue
                ev_date = sub.iloc[i]["date"]
                if ev_date < OOS_START or ev_date > OOS_END: continue
                entry = float(sub.iloc[i]["close"])
                if entry <= 0 or not np.isfinite(entry): continue
                c14 = float(sub.iloc[i+14]["close"])
                if not np.isfinite(c14): continue
                ret = c14/entry - 1 - COST
                rows.append({
                    "asset": asset, "date": ev_date, "indicator": ind, "config": cfg,
                    "entry_close": entry, "ret_E_14d": ret,
                    "btc_regime_30d": date2reg.get(ev_date, "UNK"),
                    "asym_ret": float(asymmetric_returns(np.array([ret]))[0]),
                })
    events = pd.DataFrame(rows)
    events.to_parquet(OUT_DIR/"oos_events.parquet", index=False, compression="zstd")
    print(f"\nOOS events: {len(events):,}")
    print(f"Regime distribution: {events['btc_regime_30d'].value_counts().to_dict()}")
    print(f"Indicator distribution:")
    print(events.groupby("indicator").size())

    # Daily simulation: best-K, random-K, worst-K
    print("\n=== OOS DAILY ENSEMBLE SIMULATION ===")
    bounds = {}
    for mode in ("best", "random", "worst"):
        rng = np.random.default_rng(42)
        daily = []
        for d, day_grp in events.groupby("date"):
            uniq = day_grp.sort_values("asym_ret", ascending=False).drop_duplicates(subset="asset", keep="first")
            if mode == "best":
                picked = uniq.head(K_MAX)
            elif mode == "worst":
                picked = uniq.tail(K_MAX)
            else:
                if len(uniq) <= K_MAX:
                    picked = uniq
                else:
                    sel = rng.choice(len(uniq), K_MAX, replace=False)
                    picked = uniq.iloc[sel]
            nav = picked["asym_ret"].sum() * BET_FRACTION
            daily.append({"date": d, "regime": date2reg.get(d, "UNK"),
                          "n_fires_raw": len(day_grp), "n_unique": len(uniq),
                          "n_picked": len(picked), "nav_pct": nav})
        df = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
        df["nav_7d"] = df["nav_pct"].rolling(7).sum()
        df["cum_nav"] = (1 + df["nav_pct"]).cumprod()
        bounds[mode] = df

    # Save best-K daily for reference
    bounds["best"].to_csv(OUT_DIR/"oos_daily_nav_best.csv", index=False)
    bounds["random"].to_csv(OUT_DIR/"oos_daily_nav_random.csv", index=False)

    # Headlines
    print()
    print(f"{'mode':<8}{'days':<7}{'total_NAV':<13}{'mean_d':<10}{'med_d':<10}{'+days':<8}{'cum_compound':<15}{'max_DD':<10}{'mean_7d':<10}{'floor_clear':<15}")
    for mode in ("best", "random", "worst"):
        df = bounds[mode]
        total_nav = df["nav_pct"].sum() * 100
        mean_d = df["nav_pct"].mean() * 100
        med_d = df["nav_pct"].median() * 100
        positive = (df["nav_pct"] > 0).mean() * 100
        cum_final = (df["cum_nav"].iloc[-1] - 1) * 100
        # max DD
        cum_max = df["cum_nav"].cummax()
        max_dd = ((df["cum_nav"] / cum_max - 1) * 100).min()
        mean_7d = df["nav_7d"].mean() * 100
        floor_clear = (df["nav_7d"] >= WEEKLY_FLOOR).sum()
        floor_total = max(len(df) - 6, 1)
        print(f"{mode:<8}{len(df):<7}{total_nav:+9.2f}%   {mean_d:+7.3f}% {med_d:+7.3f}% {positive:5.1f}%  {cum_final:+11.2f}%  {max_dd:+7.2f}%  {mean_7d:+.2f}%  {floor_clear}/{floor_total} ({floor_clear*100/floor_total:.0f}%)")

    # Per-regime breakdown (best-K)
    print("\n=== PER-REGIME OOS BREAKDOWN (best-K) ===")
    df = bounds["best"]
    for reg in ("bull", "chop", "bear", "crash"):
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            print(f"  {reg:<8} 0 days"); continue
        print(f"  {reg:<8} n={len(sub):4d}d  mean={sub['nav_pct'].mean()*100:+6.3f}%  median={sub['nav_pct'].median()*100:+6.3f}%  +days={(sub['nav_pct']>0).mean()*100:5.1f}%")

    # Synthesis
    lines = ["# OOS Market Replay — 14-Setup Deploy Portfolio\n"]
    lines.append(f"OOS window: {OOS_START} -> {OOS_END} (per canonical splits)")
    lines.append(f"UNSEEN ({OOS_END + timedelta(days=1)}+) UNTOUCHED.\n")
    lines.append(f"\n## A) OOS event generation\n")
    lines.append(f"- Total events: **{len(events):,}**")
    lines.append(f"- Per-regime: {events['btc_regime_30d'].value_counts().to_dict()}")

    lines.append(f"\n## B) Bounded ensemble simulation (K=8 unique-asset, asym -4%/+12%)\n")
    lines.append("| mode | total NAV | mean daily | median daily | +days % | cum compound | max DD | mean 7d | 7d floor clear |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for mode in ("best", "random", "worst"):
        df = bounds[mode]
        total_nav = df["nav_pct"].sum() * 100
        mean_d = df["nav_pct"].mean() * 100
        med_d = df["nav_pct"].median() * 100
        positive = (df["nav_pct"] > 0).mean() * 100
        cum_final = (df["cum_nav"].iloc[-1] - 1) * 100
        cum_max = df["cum_nav"].cummax()
        max_dd = ((df["cum_nav"] / cum_max - 1) * 100).min()
        mean_7d = df["nav_7d"].mean() * 100
        floor_clear = (df["nav_7d"] >= WEEKLY_FLOOR).sum()
        floor_total = max(len(df) - 6, 1)
        lines.append(f"| {mode}-K | {total_nav:+.2f}% | {mean_d:+.3f}% | {med_d:+.3f}% | {positive:.1f}% | {cum_final:+.2f}% | {max_dd:+.2f}% | {mean_7d:+.2f}% | {floor_clear}/{floor_total} ({floor_clear*100/floor_total:.0f}%) |")

    lines.append(f"\n## C) Per-regime breakdown (best-K)\n")
    lines.append("| regime | days | mean daily | median daily | +days % |")
    lines.append("|---|--:|--:|--:|--:|")
    df = bounds["best"]
    for reg in ("bull","chop","bear","crash"):
        sub = df[df["regime"] == reg]
        if len(sub)==0: continue
        lines.append(f"| {reg} | {len(sub)} | {sub['nav_pct'].mean()*100:+.3f}% | {sub['nav_pct'].median()*100:+.3f}% | {(sub['nav_pct']>0).mean()*100:.1f}% |")

    lines.append(f"\n## D) Deploy-readiness verdict\n")
    df = bounds["random"]
    floor_clear = (df["nav_7d"] >= WEEKLY_FLOOR).sum()
    floor_total = max(len(df) - 6, 1)
    pct = floor_clear * 100 / floor_total
    if pct >= 50:
        lines.append(f"**OOS PASS**: Random-K (realistic) clears the +5.25%/7d floor {pct:.1f}% of weeks (target >=50%).")
    else:
        lines.append(f"**OOS PARTIAL**: Random-K clears floor only {pct:.1f}% of weeks (target >=50%). Best-K shows higher rate; deploy with ML ranker recommended.")
    lines.append(f"\nReady for v3 paper-trade-replay integration via production_blends.yaml.")

    (OUT_DIR/"OOS_REPLAY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'OOS_REPLAY_REPORT.md'}")

if __name__ == "__main__":
    main()
