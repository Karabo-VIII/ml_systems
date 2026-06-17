"""N3: VAL window confirmation.

Builds events on VAL window (2023-07-02 -> 2024-05-15) using the SAME
smart-candidate set as TRAIN, then runs each TRAIN-qualifying setup
against VAL to confirm survival.

A setup is VAL-CONFIRMED if:
  - it had positive expectancy on TRAIN in regime R AND
  - on VAL it still qualifies in regime R (n>=15, hit>=33%, asym_ratio>=1.2)
    (gates slightly relaxed for VAL given shorter window)

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/val_events.parquet
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/val_confirmation_v2.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/N3_VAL_CONFIRMATION_REPORT.md
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# VAL window per config/data_config.yaml
VAL_START = date(2023, 7, 2)
VAL_END = date(2024, 5, 15)

COST = 0.0024
HARD_STOP = -0.04
TARGET = 0.12

# VAL gates (slightly relaxed for shorter window)
N_VAL = 15
HIT_VAL = 0.33
ASYM_VAL = 1.2
EXPECT_VAL = 0.002

REGIMES = ("bull", "chop", "bear", "crash")

# Compact indicator calculators (mirror universal_setup_library)
def calc_rsi(closes, period):
    d = np.diff(closes, prepend=closes[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    up_s = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().values
    dn_s = pd.Series(dn).ewm(alpha=1/period, adjust=False).mean().values
    return 100 - 100 / (1 + (up_s / (dn_s + 1e-12)))

def calc_bb(closes, period, std):
    s = pd.Series(closes); mid = s.rolling(period).mean(); sd = s.rolling(period).std()
    return mid.values, (mid + std*sd).values, (mid - std*sd).values

def calc_macd(closes, f, s, sig):
    sc = pd.Series(closes)
    ef = sc.ewm(span=f, adjust=False).mean(); es = sc.ewm(span=s, adjust=False).mean()
    macd = ef - es; ss = macd.ewm(span=sig, adjust=False).mean()
    return macd.values, ss.values

def calc_stoch(highs, lows, closes, kp, dp):
    hh = pd.Series(highs).rolling(kp).max(); ll = pd.Series(lows).rolling(kp).min()
    k = 100 * (pd.Series(closes) - ll) / (hh - ll + 1e-12)
    return k.values, k.rolling(dp).mean().values

def calc_williams(highs, lows, closes, period):
    hh = pd.Series(highs).rolling(period).max(); ll = pd.Series(lows).rolling(period).min()
    return (-100 * (hh - pd.Series(closes)) / (hh - ll + 1e-12)).values

def calc_roc(closes, period):
    s = pd.Series(closes); return (100 * (s - s.shift(period)) / s.shift(period)).values

def calc_obv(closes, vols):
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    return np.cumsum(direction * vols)

# Event finders (compact versions)
def find_sma_events(sub, cfg):
    a, b = eval(cfg) if isinstance(cfg, str) else cfg
    c = sub["close"].values
    ma_s = pd.Series(c).rolling(a).mean().values
    ma_l = pd.Series(c).rolling(b).mean().values
    above = ma_s > ma_l
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_ema_events(sub, cfg):
    a, b = eval(cfg) if isinstance(cfg, str) else cfg
    c = sub["close"].values
    ma_s = pd.Series(c).ewm(span=a, adjust=False).mean().values
    ma_l = pd.Series(c).ewm(span=b, adjust=False).mean().values
    above = ma_s > ma_l
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_rsi_events(sub, cfg):
    p, t = eval(cfg) if isinstance(cfg, str) else cfg
    r = calc_rsi(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    return np.where((prev < t) & (r >= t))[0]

def find_bb_events(sub, cfg):
    p, s = eval(cfg) if isinstance(cfg, str) else cfg
    mid, ub, lb = calc_bb(sub["close"].values, p, s)
    c = sub["close"].values
    above = c > ub
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_donchian_events(sub, cfg):
    p = eval(cfg)[0] if isinstance(cfg, str) else cfg[0]
    h = sub["high"].values; c = sub["close"].values
    rh = pd.Series(h).rolling(p).max().shift(1).values
    bo = c > rh
    return np.where(bo[1:] & ~bo[:-1])[0] + 1

def find_obv_events(sub, cfg):
    p, t = eval(cfg) if isinstance(cfg, str) else cfg
    c = sub["close"].values
    v = sub["volume"].values if "volume" in sub.columns else np.ones_like(c)
    if not np.any(v > 0): return np.array([])
    o = calc_obv(c, v); s = pd.Series(o)
    z = ((s - s.rolling(p).mean()) / (s.rolling(p).std() + 1e-12)).values
    above = z > t
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_macd_events(sub, cfg):
    f, s, sig = eval(cfg) if isinstance(cfg, str) else cfg
    m, ss = calc_macd(sub["close"].values, f, s, sig)
    above = m > ss
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_stoch_events(sub, cfg):
    kp, dp, ob, os_ = eval(cfg) if isinstance(cfg, str) else cfg
    k, d = calc_stoch(sub["high"].values, sub["low"].values, sub["close"].values, kp, dp)
    prev = np.roll(k, 1); prev[0] = k[0]
    return np.where((prev < os_) & (k >= os_))[0]

def find_williams_events(sub, cfg):
    p, t = eval(cfg) if isinstance(cfg, str) else cfg
    w = calc_williams(sub["high"].values, sub["low"].values, sub["close"].values, p)
    prev = np.roll(w, 1); prev[0] = w[0]
    return np.where((prev < t) & (w >= t))[0]

def find_roc_events(sub, cfg):
    p, t = eval(cfg) if isinstance(cfg, str) else cfg
    r = calc_roc(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    return np.where((prev < t) & (r >= t))[0]

FINDERS = {
    "SMA_cross": find_sma_events, "EMA_cross": find_ema_events,
    "RSI_oversold": find_rsi_events, "BB_breach": find_bb_events,
    "Donchian_breakout": find_donchian_events, "OBV_zscore": find_obv_events,
    "MACD_cross": find_macd_events, "Stochastic_bounce": find_stoch_events,
    "Williams_R": find_williams_events, "ROC_momentum": find_roc_events,
}

def asymmetric_returns(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

def main():
    print("="*78)
    print("N3: VAL CONFIRMATION")
    print(f"VAL window: {VAL_START} -> {VAL_END}")
    print("="*78)

    # Load library V2
    library = pd.read_csv(OUT_DIR / "universal_library_v2.csv")
    survivors = library[library["class"] != "UNRELIABLE"]
    print(f"TRAIN survivors to confirm: {len(survivors)}")

    # Load regime overlay for VAL
    print("\nLoading VAL regime overlay...")
    reg_df = pl.read_parquet(ROOT / "runs" / "oracle_layer2" / "daily_regime_cluster.parquet").to_pandas()
    reg_df["date"] = pd.to_datetime(reg_df["date"]).dt.date
    reg_df = reg_df[(reg_df["date"] >= VAL_START - timedelta(days=1)) & (reg_df["date"] <= VAL_END)]
    print(f"  VAL regime days: {len(reg_df)}")
    print(f"  Regime mix: {reg_df['btc_regime_30d'].value_counts().to_dict()}")
    date_to_regime = dict(zip(reg_df["date"], reg_df["btc_regime_30d"]))

    # Load chimera VAL panel
    print("\nLoading chimera VAL panel (with 120d warmup + 30d forward)...")
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    panels = {}
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
        except Exception:
            try:
                df = pl.read_parquet(f, columns=["timestamp", "open", "high", "low", "close"]).to_pandas()
                df["volume"] = 0.0
            except Exception:
                continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= VAL_START - timedelta(days=120)) &
                  (df["date"] <= VAL_END + timedelta(days=30))].reset_index(drop=True)
        if len(df) < 50: continue
        df["asset"] = sym
        panels[sym] = df
    print(f"  Panel: {len(panels)} assets")

    # Generate VAL events per setup
    print("\nGenerating VAL events per setup...")
    val_rows = []
    for i, (_, r) in enumerate(survivors.iterrows()):
        ind = r["indicator"]; cfg = r["config"]
        finder = FINDERS.get(ind)
        if finder is None: continue
        for asset, sub in panels.items():
            try:
                entry_idx = finder(sub, cfg)
            except Exception:
                continue
            for idx in entry_idx:
                if idx < 60 or idx + 14 >= len(sub): continue
                ev_date = sub.iloc[idx]["date"]
                if ev_date < VAL_START or ev_date > VAL_END: continue
                entry_close = float(sub.iloc[idx]["close"])
                if entry_close <= 0 or not np.isfinite(entry_close): continue
                c14 = float(sub.iloc[idx + 14]["close"]) if idx + 14 < len(sub) else None
                if c14 is None or not np.isfinite(c14): continue
                ret_14d = c14 / entry_close - 1 - COST
                val_rows.append({
                    "asset": asset, "date": ev_date, "indicator": ind, "config": cfg,
                    "entry_close": entry_close, "ret_E_14d": ret_14d,
                    "btc_regime_30d": date_to_regime.get(ev_date, "UNK"),
                })
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(survivors)} setups, {len(val_rows):,} val events so far")
    val_df = pd.DataFrame(val_rows)
    val_df.to_parquet(OUT_DIR / "val_events.parquet", index=False, compression="zstd")
    print(f"\nVAL events: {len(val_df):,}")
    print(f"VAL regime distribution: {val_df['btc_regime_30d'].value_counts().to_dict()}")

    # Per-setup VAL qualification
    print("\nConfirming setups on VAL...")
    confirm_rows = []
    for _, r in survivors.iterrows():
        ind, cfg = r["indicator"], r["config"]
        sub_v = val_df[(val_df["indicator"] == ind) & (val_df["config"] == cfg)]
        per_reg = {}
        for reg in REGIMES:
            sub_reg = sub_v[sub_v["btc_regime_30d"] == reg]
            rets = sub_reg["ret_E_14d"].dropna().values
            if len(rets) < N_VAL:
                per_reg[reg] = False
                continue
            asym = asymmetric_returns(rets)
            pos = rets[rets > 0]; neg = rets[rets < 0]
            asym_ratio = pos.mean() / abs(neg.mean()) if len(neg) and neg.mean() != 0 else float('inf')
            per_reg[reg] = (
                asym.mean() >= EXPECT_VAL and
                (rets > 0).mean() >= HIT_VAL and
                asym_ratio >= ASYM_VAL
            )
        n_val_qual = sum(per_reg.values())
        # TRAIN-VAL match: which TRAIN-qualifying regimes also qualify on VAL?
        train_qual = {reg: r[f"{reg}_qualifies"] for reg in REGIMES}
        match = sum(1 for reg in REGIMES if train_qual.get(reg) and per_reg[reg])
        confirm_rows.append({
            "indicator": ind, "config": cfg, "train_class": r["class"],
            "val_n": int(len(sub_v)),
            "val_qualifies_n_regimes": n_val_qual,
            "train_qual_regimes": ",".join(reg for reg in REGIMES if train_qual.get(reg)),
            "val_qual_regimes": ",".join(reg for reg in REGIMES if per_reg[reg]),
            "match_regimes_count": match,
            "stable": match >= 1,
        })

    confirm = pd.DataFrame(confirm_rows)
    confirm.to_csv(OUT_DIR / "val_confirmation_v2.csv", index=False)

    n_stable = confirm["stable"].sum()
    print(f"\n=== VAL CONFIRMATION RESULTS ===")
    print(f"TRAIN-qualifying setups: {len(survivors)}")
    print(f"VAL-stable (TRAIN regime AND VAL regime match): {n_stable}/{len(confirm)} ({n_stable*100/len(confirm):.1f}%)")
    print()
    print("By TRAIN class:")
    by_class = confirm.groupby("train_class")["stable"].agg(["count", "sum", "mean"])
    by_class["pct"] = (by_class["mean"] * 100).round(1)
    print(by_class)

    # Top survivors by VAL qualification
    top_stable = confirm[confirm["stable"]].sort_values("val_qualifies_n_regimes", ascending=False)
    print("\nTop-15 VAL-stable setups:")
    print(top_stable.head(15)[["indicator", "config", "train_class", "val_n",
                                 "val_qualifies_n_regimes", "match_regimes_count"]].to_string(index=False))

    # Report
    lines = ["# N3: VAL Window Confirmation\n"]
    lines.append(f"VAL: {VAL_START} -> {VAL_END}")
    lines.append(f"VAL gates: n>={N_VAL}, hit>={HIT_VAL*100:.0f}%, asym_ratio>={ASYM_VAL}, expect>={EXPECT_VAL*100:.2f}%")
    lines.append(f"\n## A) Headline\n")
    lines.append(f"TRAIN-qualifying setups: {len(survivors)}")
    lines.append(f"VAL-stable (qualify in matching regime on VAL): **{n_stable}/{len(confirm)} ({n_stable*100/len(confirm):.1f}%)**")

    lines.append(f"\n## B) Stability by TRAIN class\n")
    lines.append("| TRAIN class | tested | stable | pct |")
    lines.append("|---|--:|--:|--:|")
    for c in by_class.index:
        row = by_class.loc[c]
        lines.append(f"| {c} | {int(row['count'])} | {int(row['sum'])} | {row['pct']:.1f}% |")

    lines.append(f"\n## C) Top-15 VAL-stable setups (TRAIN+VAL both qualify)\n")
    lines.append("| indicator | config | train class | val_n | val regimes qualified | match |")
    lines.append("|---|---|---|--:|--:|--:|")
    for _, r in top_stable.head(15).iterrows():
        lines.append(f"| {r['indicator']} | `{r['config']}` | {r['train_class']} | {int(r['val_n'])} | {int(r['val_qualifies_n_regimes'])} | {int(r['match_regimes_count'])} |")

    (OUT_DIR / "N3_VAL_CONFIRMATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'N3_VAL_CONFIRMATION_REPORT.md'}")

if __name__ == "__main__":
    main()
