"""Round-2 indicator extension: 5 new indicator families mined on TRAIN.

Adds:
  - Ichimoku Cloud (tenkan/kijun cross, price-above-cloud)
  - Keltner Channel (EMA + ATR multiplier breakout)
  - Supertrend (ATR-trailed direction flip)
  - ADX (trend strength filter; long when ADX>25 + price above SMA)
  - Aroon (Aroon up > Aroon down cross)

Mining methodology: same as the original (smart candidate selection per
indicator, classify by per-regime qualification, n>=20 / hit>=35% / asym>=1.3).

TRAIN window: 2020-01-01 -> 2023-07-01.

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/round2_events.parquet
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/round2_library.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/round2_complementarity.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/ROUND2_INDICATORS_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)

COST = 0.0024
HARD_STOP = -0.04
TARGET = 0.12
N_MIN = 20
HIT_MIN = 0.35
ASYM_MIN = 1.3
EXPECT_MIN = 0.003

REGIMES = ("bull", "chop", "bear", "crash")

# ============================================================================
# New indicator calculators
# ============================================================================

def calc_atr(highs, lows, closes, period):
    tr = np.maximum.reduce([
        highs - lows,
        np.abs(highs - np.roll(closes, 1)),
        np.abs(lows - np.roll(closes, 1)),
    ])
    tr[0] = (highs[0] - lows[0])
    return pd.Series(tr).rolling(period).mean().values

def calc_ichimoku(highs, lows, closes, tenkan_p=9, kijun_p=26, senkou_p=52):
    hh_t = pd.Series(highs).rolling(tenkan_p).max(); ll_t = pd.Series(lows).rolling(tenkan_p).min()
    tenkan = ((hh_t + ll_t) / 2).values
    hh_k = pd.Series(highs).rolling(kijun_p).max(); ll_k = pd.Series(lows).rolling(kijun_p).min()
    kijun = ((hh_k + ll_k) / 2).values
    senkou_a = ((tenkan + kijun) / 2)  # NOT projected forward (avoiding lookahead in entry)
    hh_s = pd.Series(highs).rolling(senkou_p).max(); ll_s = pd.Series(lows).rolling(senkou_p).min()
    senkou_b = ((hh_s + ll_s) / 2).values
    return tenkan, kijun, senkou_a, senkou_b

def calc_keltner(closes, highs, lows, ema_p=20, atr_p=10, multiplier=2.0):
    ema = pd.Series(closes).ewm(span=ema_p, adjust=False).mean().values
    atr = calc_atr(highs, lows, closes, atr_p)
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return ema, upper, lower

def calc_supertrend(highs, lows, closes, atr_p=10, multiplier=3.0):
    """Compact Supertrend (Olson) — returns direction (+1 long, -1 short) and trend line."""
    atr = calc_atr(highs, lows, closes, atr_p)
    src = (highs + lows) / 2
    upper_basic = src + multiplier * atr
    lower_basic = src - multiplier * atr
    upper = np.copy(upper_basic)
    lower = np.copy(lower_basic)
    direction = np.ones(len(closes), dtype=int)  # 1 = long
    for i in range(1, len(closes)):
        if closes[i-1] > upper[i-1]:
            direction[i] = 1
        elif closes[i-1] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        # Update bands
        upper[i] = min(upper_basic[i], upper[i-1]) if closes[i-1] <= upper[i-1] else upper_basic[i]
        lower[i] = max(lower_basic[i], lower[i-1]) if closes[i-1] >= lower[i-1] else lower_basic[i]
    return direction

def calc_adx(highs, lows, closes, period=14):
    """ADX trend strength (0-100)."""
    h_prev = np.roll(highs, 1); l_prev = np.roll(lows, 1); c_prev = np.roll(closes, 1)
    up = highs - h_prev
    down = l_prev - lows
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum.reduce([highs - lows, np.abs(highs - c_prev), np.abs(lows - c_prev)])
    tr[0] = highs[0] - lows[0]
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / (tr_smooth + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / (tr_smooth + 1e-12)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx, plus_di, minus_di

def calc_aroon(highs, lows, period=14):
    """Aroon indicator: % of bars since highest high / lowest low in last `period`."""
    aroon_up = np.full(len(highs), np.nan)
    aroon_dn = np.full(len(lows), np.nan)
    for i in range(period, len(highs)):
        window_h = highs[i-period+1:i+1]
        window_l = lows[i-period+1:i+1]
        aroon_up[i] = 100 * (period - 1 - np.argmax(window_h[::-1])) / period
        aroon_dn[i] = 100 * (period - 1 - np.argmin(window_l[::-1])) / period
    return aroon_up, aroon_dn

# ============================================================================
# Event finders
# ============================================================================

def find_ichimoku(sub, cfg):
    """Long: tenkan crosses up through kijun AND price above senkou_a."""
    t_p, k_p, s_p = eval(cfg)
    h = sub["high"].values; l = sub["low"].values; c = sub["close"].values
    tenkan, kijun, sa, sb = calc_ichimoku(h, l, c, t_p, k_p, s_p)
    above = tenkan > kijun
    cross_up = np.where(above[1:] & ~above[:-1])[0] + 1
    # Filter to price above senkou_a
    out = [i for i in cross_up if i < len(c) and not np.isnan(sa[i]) and c[i] > sa[i]]
    return np.array(out)

def find_keltner(sub, cfg):
    """Long: close breaks above upper Keltner band."""
    ema_p, atr_p, mult = eval(cfg)
    c = sub["close"].values; h = sub["high"].values; l = sub["low"].values
    ema, upper, lower = calc_keltner(c, h, l, ema_p, atr_p, mult)
    above = c > upper
    return np.where(above[1:] & ~above[:-1])[0] + 1

def find_supertrend(sub, cfg):
    """Long: Supertrend direction flips from -1 to +1."""
    atr_p, mult = eval(cfg)
    h = sub["high"].values; l = sub["low"].values; c = sub["close"].values
    d = calc_supertrend(h, l, c, atr_p, mult)
    flip_up = np.where((d[1:] == 1) & (d[:-1] == -1))[0] + 1
    return flip_up

def find_adx_trend(sub, cfg):
    """Long: ADX > threshold AND +DI > -DI AND price > SMA(50)."""
    adx_p, thresh = eval(cfg)
    h = sub["high"].values; l = sub["low"].values; c = sub["close"].values
    adx, pdi, mdi = calc_adx(h, l, c, adx_p)
    sma50 = pd.Series(c).rolling(50).mean().values
    condition = (adx > thresh) & (pdi > mdi) & (c > sma50)
    return np.where(condition[1:] & ~condition[:-1])[0] + 1

def find_aroon(sub, cfg):
    """Long: Aroon up crosses up through Aroon down."""
    p, thresh = eval(cfg)
    h = sub["high"].values; l = sub["low"].values
    aroon_up, aroon_dn = calc_aroon(h, l, p)
    diff = aroon_up - aroon_dn
    above = diff > thresh
    return np.where(above[1:] & ~above[:-1])[0] + 1

# ============================================================================
# Candidate configs
# ============================================================================

INDICATOR_CANDIDATES = {
    "Ichimoku_cross":     [(9, 26, 52), (7, 21, 42), (5, 15, 30), (12, 30, 60)],
    "Keltner_breakout":   [(20, 10, 1.5), (20, 10, 2.0), (20, 14, 2.0), (10, 10, 2.0), (30, 14, 2.5)],
    "Supertrend_flip":    [(10, 3.0), (14, 3.0), (10, 2.0), (14, 2.5), (21, 3.0), (7, 2.0)],
    "ADX_trend":          [(14, 20), (14, 25), (14, 30), (21, 25), (10, 25)],
    "Aroon_cross":        [(14, 0), (14, 20), (21, 0), (21, 20), (28, 0)],
}

FINDERS = {
    "Ichimoku_cross": find_ichimoku, "Keltner_breakout": find_keltner,
    "Supertrend_flip": find_supertrend, "ADX_trend": find_adx_trend,
    "Aroon_cross": find_aroon,
}

# ============================================================================
# Asymmetric stop
# ============================================================================

def asymmetric_returns(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

def main():
    print("="*78)
    print("ROUND-2 INDICATOR EXTENSION (5 new families)")
    print("="*78)

    # Load chimera 1d panel + regime overlay
    print("Loading chimera panel (TRAIN)...")
    files = sorted((ROOT/"data"/"processed"/"chimera"/"1d").glob("*_v51_chimera_1d_*.parquet"))
    panels = {}
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT","")
        try:
            df = pl.read_parquet(f, columns=["timestamp","open","high","low","close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].reset_index(drop=True)
        if len(df) < 100: continue
        df["asset"] = sym
        panels[sym] = df
    print(f"  panels: {len(panels)} assets")

    print("Loading regime overlay...")
    reg = pl.read_parquet(ROOT/"runs"/"oracle_layer2"/"daily_regime_cluster.parquet").to_pandas()
    reg["date"] = pd.to_datetime(reg["date"]).dt.date
    reg = reg[(reg["date"] >= TRAIN_START) & (reg["date"] <= TRAIN_END)]
    date2reg = dict(zip(reg["date"], reg["btc_regime_30d"]))

    # Generate events
    print("\nGenerating round-2 events...")
    rows = []
    for ind, cfgs in INDICATOR_CANDIDATES.items():
        finder = FINDERS[ind]
        for cfg in cfgs:
            cfg_str = str(cfg) if isinstance(cfg, tuple) else cfg
            for asset, sub in panels.items():
                try:
                    idx = finder(sub, cfg_str)
                except Exception as e:
                    continue
                for i in idx:
                    if i < 60 or i + 14 >= len(sub): continue
                    ev_date = sub.iloc[i]["date"]
                    entry = float(sub.iloc[i]["close"])
                    if entry <= 0 or not np.isfinite(entry): continue
                    c14 = float(sub.iloc[i+14]["close"])
                    if not np.isfinite(c14): continue
                    ret = c14/entry - 1 - COST
                    rows.append({
                        "asset": asset, "date": ev_date, "indicator": ind, "config": cfg_str,
                        "entry_close": entry, "ret_E_14d": ret,
                        "btc_regime_30d": date2reg.get(ev_date, "UNK"),
                    })
            print(f"  {ind} {cfg_str}: done")
    events = pd.DataFrame(rows)
    events.to_parquet(OUT_DIR/"round2_events.parquet", index=False, compression="zstd")
    print(f"\nRound-2 events: {len(events):,}")
    print(events.groupby("indicator").size())

    # Build library v2-style classification
    print("\nClassifying round-2 setups by regime qualification...")
    lib_rows = []
    for (ind, cfg), grp in events.groupby(["indicator", "config"]):
        qual = {}
        stats = {}
        for r in REGIMES:
            sub = grp[grp["btc_regime_30d"] == r]
            rets = sub["ret_E_14d"].dropna().values
            if len(rets) < N_MIN:
                qual[r] = False; stats[r] = None; continue
            asym = asymmetric_returns(rets)
            pos = rets[rets > 0]; neg = rets[rets < 0]
            ar = pos.mean()/abs(neg.mean()) if len(neg) and neg.mean() != 0 else float('inf')
            ok = (asym.mean() >= EXPECT_MIN and (rets > 0).mean() >= HIT_MIN
                  and ar >= ASYM_MIN and len(rets) >= N_MIN)
            qual[r] = ok
            stats[r] = dict(n=len(rets), asym_mean=asym.mean()*100,
                             hit=(rets>0).mean()*100, asym_ratio=min(ar, 10))
        n_q = sum(qual.values())
        cls = ("ALL_WEATHER" if n_q >= 3 else
               f"MULTI_REGIME_{'_'.join(r for r in REGIMES if qual[r])}" if n_q == 2 else
               f"{[r for r in REGIMES if qual[r]][0].upper()}_DOMINANT" if n_q == 1 else
               "UNRELIABLE")
        row = {"indicator": ind, "config": cfg, "class": cls, "n_qualifying": n_q,
                "n_total": sum(s["n"] for s in stats.values() if s)}
        for r in REGIMES:
            s = stats[r]
            row[f"{r}_n"] = s["n"] if s else 0
            row[f"{r}_asym_mean_pct"] = round(s["asym_mean"], 3) if s else None
            row[f"{r}_hit_pct"] = round(s["hit"], 1) if s else None
            row[f"{r}_qual"] = qual[r]
        lib_rows.append(row)
    lib = pd.DataFrame(lib_rows).sort_values(["n_qualifying", "n_total"], ascending=[False, False])
    lib.to_csv(OUT_DIR/"round2_library.csv", index=False)
    print(f"\nClass distribution:")
    print(lib["class"].value_counts())

    print("\n=== ROUND-2 QUALIFYING SETUPS (n_qualifying >=1) ===")
    qual_setups = lib[lib["class"] != "UNRELIABLE"]
    print(f"Total qualifying: {len(qual_setups)}")
    print(qual_setups[["indicator","config","class","bull_asym_mean_pct","chop_asym_mean_pct",
                        "bear_asym_mean_pct","crash_asym_mean_pct"]].to_string(index=False))

    # Compute lift vs current 14-setup portfolio
    print("\n=== COMPLEMENTARITY VS EXISTING 14-SETUP PORTFOLIO ===")
    existing = pd.read_parquet(OUT_DIR/"per_event_enriched.parquet")
    existing["date"] = pd.to_datetime(existing["date"]).dt.date
    movers = pd.read_csv(OUT_DIR/"movers_from_panel_train.csv")
    movers["date"] = pd.to_datetime(movers["date"]).dt.date
    movers_set = set(zip(movers["asset"], movers["date"]))

    # Existing 14-portfolio firings
    DEPLOY = [
        ("SMA_cross", "(3, 5)"), ("SMA_cross", "(3, 8)"), ("SMA_cross", "(3, 13)"),
        ("SMA_cross", "(5, 8)"), ("SMA_cross", "(20, 21)"),
        ("Donchian_breakout", "(20,)"), ("ROC_momentum", "(10, 7)"),
        ("Stochastic_bounce", "(7, 3, 80, 20)"), ("Stochastic_bounce", "(7, 3, 90, 10)"),
        ("MACD_cross", "(5, 21, 5)"), ("MACD_cross", "(5, 34, 9)"),
        ("BB_breach", "(20, 1.5)"), ("EMA_cross", "(3, 5)"), ("EMA_cross", "(3, 8)"),
    ]
    existing_fires = set()
    for ind, cfg in DEPLOY:
        sub = existing[(existing["indicator"] == ind) & (existing["config"] == cfg)]
        existing_fires |= set(zip(sub["asset"], sub["date"]))
    existing_cov = len(existing_fires & movers_set) / max(len(movers_set), 1) * 100
    print(f"Existing 14-portfolio coverage: {existing_cov:.2f}% of {len(movers_set)} movers")

    # Per round-2 setup: marginal lift on top of existing portfolio
    lift_rows = []
    for _, r in qual_setups.iterrows():
        ind, cfg = r["indicator"], r["config"]
        sub = events[(events["indicator"] == ind) & (events["config"] == cfg)]
        new_fires = set(zip(sub["asset"], sub["date"]))
        union_fires = existing_fires | new_fires
        union_cov = len(union_fires & movers_set) / max(len(movers_set), 1) * 100
        marginal_lift = union_cov - existing_cov
        lift_rows.append({"indicator": ind, "config": cfg, "class": r["class"],
                           "n_fires": len(new_fires),
                           "alone_cov_pct": len(new_fires & movers_set) * 100 / max(len(movers_set), 1),
                           "union_cov_pct": union_cov,
                           "marginal_lift_pp": marginal_lift})
    lift_df = pd.DataFrame(lift_rows).sort_values("marginal_lift_pp", ascending=False)
    lift_df.to_csv(OUT_DIR/"round2_complementarity.csv", index=False)
    print("\nMarginal lift vs existing 14-portfolio:")
    print(lift_df.to_string(index=False))

    # Report
    lines = ["# Round-2 Indicators: Ichimoku / Keltner / Supertrend / ADX / Aroon\n"]
    lines.append(f"\n## A) Event mining\n")
    lines.append(f"- Round-2 events generated: {len(events):,}")
    lines.append(f"- Indicators tested: {list(INDICATOR_CANDIDATES.keys())}")
    lines.append(f"- Configs per indicator: {[len(c) for c in INDICATOR_CANDIDATES.values()]} ({sum(len(c) for c in INDICATOR_CANDIDATES.values())} total)")

    lines.append(f"\n## B) Class distribution\n")
    lines.append("| class | count |\n|---|--:|")
    for c, n in lib["class"].value_counts().items():
        lines.append(f"| {c} | {n} |")

    lines.append(f"\n## C) Qualifying setups (n_qualifying >=1)\n")
    lines.append("| indicator | config | class | bull % | chop % | bear % | crash % |")
    lines.append("|---|---|---|--:|--:|--:|--:|")
    for _, r in qual_setups.iterrows():
        b = r["bull_asym_mean_pct"]; c = r["chop_asym_mean_pct"]
        be = r["bear_asym_mean_pct"]; cr = r["crash_asym_mean_pct"]
        lines.append(f"| {r['indicator']} | `{r['config']}` | {r['class']} | "
                     f"{(f'{b:+.2f}%' if pd.notna(b) else '—')} | "
                     f"{(f'{c:+.2f}%' if pd.notna(c) else '—')} | "
                     f"{(f'{be:+.2f}%' if pd.notna(be) else '—')} | "
                     f"{(f'{cr:+.2f}%' if pd.notna(cr) else '—')} |")

    lines.append(f"\n## D) Complementarity vs existing 14-portfolio\n")
    lines.append(f"- Existing 14-portfolio coverage: **{existing_cov:.2f}%** of TRAIN top-movers")
    lines.append("\n| indicator | config | class | n_fires | alone cov | union cov | marginal lift |")
    lines.append("|---|---|---|--:|--:|--:|--:|")
    for _, r in lift_df.iterrows():
        lines.append(f"| {r['indicator']} | `{r['config']}` | {r['class']} | "
                     f"{int(r['n_fires'])} | {r['alone_cov_pct']:.2f}% | "
                     f"{r['union_cov_pct']:.2f}% | +{r['marginal_lift_pp']:.2f}pp |")

    lines.append(f"\n## E) Recommendation\n")
    top_lift = lift_df.head(5)
    if len(top_lift) and top_lift["marginal_lift_pp"].max() >= 0.5:
        lines.append(f"Top 5 round-2 setups by marginal lift add **+{top_lift['marginal_lift_pp'].sum():.2f}pp** combined.")
        lines.append("Consider adding to the deploy portfolio AFTER VAL+OOS confirmation on these specifically.")
    else:
        lines.append("Round-2 setups add minimal marginal lift; the original 10-indicator family covers the gap well.")

    (OUT_DIR/"ROUND2_INDICATORS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'ROUND2_INDICATORS_REPORT.md'}")

if __name__ == "__main__":
    main()
