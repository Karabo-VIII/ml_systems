"""Smart Discovery — EXHAUSTIVE, information-saturated, TRAIN window only.

Per user direction (2026-05-20):
  1. Two ranking modes: PER-ASSET (for cousin-finding) + GLOBAL (for unseen assets).
  2. Top-25 PER INDICATOR (not globally) in BOTH modes.
  3. Per-day regime overlay (NOT asset-static) on TRAIN.
  4. Classify ideal exit mechanism PER indicator × PER regime.
  5. Information saturation: per-event raw rows emitted for future data-mining.

Inputs:
  - data/processed/chimera/1d/*.parquet (OHLCV)
  - runs/oracle_layer2/daily_regime_cluster.parquet (per-day regime + cluster)
  - config/universes/u100.yaml (per-asset DNA bucket)
  - runs/oracle_layer2/sector_map.json (per-asset sector)

Outputs in runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/:
  per_event_raw.parquet            -- 1 row per entry event (full metadata + per-exit returns)
  per_asset_indicator_top25.csv    -- top-25 configs PER asset PER indicator
  global_indicator_top25.csv       -- top-25 configs GLOBAL PER indicator
  regime_exit_decomposition.csv    -- regime × indicator × exit → n,mean,hit,sharpe
  bucket_exit_decomposition.csv    -- dna_bucket × indicator × exit
  best_exit_per_indicator_per_regime.csv -- the recommendation table
  cousins.csv                      -- per-asset top-3 setups + asset metadata for cousin-matching
  REPORT.md                        -- synthesis + recommendations
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)
COST = 0.0024
SIZE = 0.04
MIN_EVENTS = 30      # per (asset, indicator, config) cell to be ranked
MIN_GLOBAL_EVENTS = 200  # global cell
MIN_REGIME_EVENTS = 30   # per-regime cell

# ============================================================================
# SMART CANDIDATE GENERATORS
# ============================================================================

def fibonacci_pairs(max_period=120):
    fibs = [3, 5, 8, 13, 21, 34, 55, 89]
    return [(a, b) for a in fibs for b in fibs if a < b and b <= max_period]

def golden_pairs(max_period=120):
    base = [3, 5, 8, 13, 21, 34, 55, 89]
    out = []
    for a in base:
        b = int(round(a * 1.618))
        if b > a and b <= max_period:
            out.append((a, b))
    return out

def log_spaced_pairs(max_period=120):
    shorts = sorted(set(np.round(np.logspace(np.log10(3), np.log10(20), 4)).astype(int)))
    longs = sorted(set(np.round(np.logspace(np.log10(21), np.log10(max_period), 5)).astype(int)))
    return [(int(a), int(b)) for a in shorts for b in longs if a < b]

def smart_ma_candidates():
    return sorted(set(fibonacci_pairs() + golden_pairs() + log_spaced_pairs()))

def rsi_configs():     return [(p, t) for p in (7, 9, 14, 21, 28) for t in (20, 25, 30, 35, 40)]
def bb_configs():      return [(p, s) for p in (10, 20, 30, 50) for s in (1.5, 2.0, 2.5, 3.0)]
def donchian_configs(): return [(p,) for p in (8, 13, 20, 30, 50, 89)]
def obv_configs():     return [(p, t) for p in (14, 20, 30, 50, 100) for t in (1.5, 2.0, 2.5, 3.0)]
def macd_configs():    return [(f, s, sig) for f in (5, 8, 12) for s in (21, 26, 34) for sig in (5, 9, 13) if f < s]
def stoch_configs():   return [(k, d, ob, os_) for k in (7, 14, 21) for d in (3, 5) for ob, os_ in [(80,20),(85,15),(90,10)]]
def williams_configs(): return [(p, t) for p in (7, 14, 21, 28) for t in (-80, -85, -90)]
def roc_configs():     return [(p, t) for p in (5, 10, 14, 20, 30) for t in (5, 7, 10, 15)]

# ============================================================================
# INDICATOR CALCULATORS
# ============================================================================

def calc_rsi(closes, period):
    d = np.diff(closes, prepend=closes[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    up_s = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().values
    dn_s = pd.Series(dn).ewm(alpha=1/period, adjust=False).mean().values
    return 100 - 100 / (1 + (up_s / (dn_s + 1e-12)))

def calc_bb(closes, period, std_mult):
    s = pd.Series(closes); mid = s.rolling(period).mean(); sd = s.rolling(period).std()
    return mid.values, (mid + std_mult * sd).values, (mid - std_mult * sd).values

def calc_obv(closes, vols):
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    return np.cumsum(direction * vols)

def calc_macd(closes, fast, slow, sig):
    s = pd.Series(closes); ef = s.ewm(span=fast, adjust=False).mean(); es = s.ewm(span=slow, adjust=False).mean()
    macd = ef - es; ss = macd.ewm(span=sig, adjust=False).mean()
    return macd.values, ss.values

def calc_stoch(highs, lows, closes, kp, dp):
    hh = pd.Series(highs).rolling(kp).max(); ll = pd.Series(lows).rolling(kp).min()
    k = 100 * (pd.Series(closes) - ll) / (hh - ll + 1e-12)
    return k.values, k.rolling(dp).mean().values

def calc_williams(highs, lows, closes, p):
    hh = pd.Series(highs).rolling(p).max(); ll = pd.Series(lows).rolling(p).min()
    return (-100 * (hh - pd.Series(closes)) / (hh - ll + 1e-12)).values

def calc_roc(closes, p):
    s = pd.Series(closes); return (100 * (s - s.shift(p)) / s.shift(p)).values

# ============================================================================
# DATA LOADERS
# ============================================================================

def load_train_panel():
    print("Loading chimera 1d panel (TRAIN window)...")
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    panel_rows = []
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp","open","high","low","close","volume"]).to_pandas()
        except Exception:
            df = pl.read_parquet(f, columns=["timestamp","open","high","low","close"]).to_pandas()
            df["volume"] = 0.0
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].reset_index(drop=True)
        if len(df) < 200: continue
        df["asset"] = sym
        df["high_low"] = df["high"] - df["low"]
        df["high_pc"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_pc"] = (df["low"] - df["close"].shift(1)).abs()
        df["tr"] = df[["high_low","high_pc","low_pc"]].max(axis=1)
        df["atr30_pct"] = df["tr"].rolling(30).mean() / df["close"]
        # Realized vol regime per-asset rolling
        df["ret_1d"] = df["close"].pct_change()
        df["rv30"] = df["ret_1d"].rolling(30).std()
        panel_rows.append(df[["asset","date","open","high","low","close","volume","atr30_pct","rv30"]])
    panel = pd.concat(panel_rows, ignore_index=True)
    print(f"  panel: {len(panel):,} rows, {panel['asset'].nunique()} assets, {panel['date'].min()} -> {panel['date'].max()}")
    return panel

def load_regime_overlay():
    print("Loading daily regime overlay...")
    df = pl.read_parquet(ROOT / "runs" / "oracle_layer2" / "daily_regime_cluster.parquet").to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].reset_index(drop=True)
    overlay = df[["date", "btc_regime_30d", "btc_regime_chimera", "cluster_id", "breadth_pct_long", "btc_30d_ret", "btc_vol_30d"]].rename(
        columns={"cluster_id": "daily_market_cluster"})
    print(f"  regime overlay: {len(overlay)} rows, {overlay['btc_regime_30d'].value_counts().to_dict()}")
    return overlay

def load_asset_metadata():
    print("Loading per-asset metadata (DNA + sector)...")
    out = {}
    for u in ("u50.yaml", "u100.yaml"):
        try:
            with open(ROOT / "config" / "universes" / u) as f:
                data = yaml.safe_load(f)
            for key in ("assets", "extra_assets"):
                for entry in data.get(key, []) or []:
                    sym = entry["symbol"].replace("USDT", "")
                    out[sym] = {"dna_bucket": entry.get("dna", "UNK"), "tier": entry.get("tier", "UNK"),
                                "pos_cap": entry.get("pos_cap", 0.04)}
        except Exception as e:
            print(f"  WARN: could not load {u}: {e}")
    try:
        sector_map = json.loads((ROOT / "runs" / "oracle_layer2" / "sector_map.json").read_text())
        for sym, sec in sector_map.items():
            out.setdefault(sym, {})["sector"] = sec
    except Exception as e:
        print(f"  WARN: sector_map: {e}")
    print(f"  metadata for {len(out)} assets")
    return out

# ============================================================================
# MULTI-EXIT EVALUATION
# ============================================================================

EXIT_NAMES = ("B_3d", "C_5d", "D_7d", "E_14d", "S_setup_toxic", "F_atr30_K3_TP30", "G_trail_5_3")

def evaluate_event(asset_sub, entry_idx, signal_active_fn, max_horizon=14):
    if entry_idx + 1 >= len(asset_sub):
        return None
    entry_close = float(asset_sub.iloc[entry_idx]["close"])
    if entry_close <= 0 or not np.isfinite(entry_close):
        return None
    atr30 = float(asset_sub.iloc[entry_idx].get("atr30_pct") or 0.0)

    fwd = []
    for k in range(1, max_horizon + 1):
        if entry_idx + k < len(asset_sub):
            c = float(asset_sub.iloc[entry_idx + k]["close"])
            fwd.append(c if np.isfinite(c) else None)
        else:
            fwd.append(None)

    def pct(c): return c / entry_close - 1 - COST
    rets = {}
    if len(fwd) >= 3 and fwd[2] is not None: rets["B_3d"] = pct(fwd[2])
    if len(fwd) >= 5 and fwd[4] is not None: rets["C_5d"] = pct(fwd[4])
    if len(fwd) >= 7 and fwd[6] is not None: rets["D_7d"] = pct(fwd[6])
    if len(fwd) >= 14 and fwd[13] is not None: rets["E_14d"] = pct(fwd[13])

    if signal_active_fn is not None:
        exit_s = None
        for k in range(max_horizon):
            if fwd[k] is None: continue
            if not signal_active_fn(entry_idx + 1 + k):
                exit_s = fwd[k]; break
        if exit_s is None and fwd[-1] is not None: exit_s = fwd[-1]
        if exit_s is not None: rets["S_setup_toxic"] = pct(exit_s)

    if atr30 > 0:
        stop_pct = 3.0 * atr30; tp_pct = 0.30
        exit_f = None
        for c in fwd:
            if c is None: continue
            r = c / entry_close - 1
            if r <= -stop_pct or r >= tp_pct: exit_f = c; break
        if exit_f is None and fwd[-1] is not None: exit_f = fwd[-1]
        if exit_f is not None: rets["F_atr30_K3_TP30"] = pct(exit_f)

    peak = entry_close; exit_g = None; armed = False
    for c in fwd:
        if c is None: continue
        if c > peak: peak = c
        if not armed and (c / entry_close - 1) >= 0.05: armed = True
        if armed and c <= peak * 0.97: exit_g = c; break
    if exit_g is None and fwd[-1] is not None: exit_g = fwd[-1]
    if exit_g is not None: rets["G_trail_5_3"] = pct(exit_g)

    return rets

# ============================================================================
# EVENT FINDERS
# ============================================================================

def find_ma(sub, cfg, kind):
    short_p, long_p = cfg
    c = sub["close"].values
    if kind == "sma":
        a = pd.Series(c).rolling(short_p).mean().values
        b = pd.Series(c).rolling(long_p).mean().values
    else:
        a = pd.Series(c).ewm(span=short_p, adjust=False).mean().values
        b = pd.Series(c).ewm(span=long_p, adjust=False).mean().values
    above = a > b
    crosses = np.where(above[1:] & ~above[:-1])[0] + 1
    return list(crosses), (lambda i: i < len(above) and above[i])

def find_sma(sub, cfg): return find_ma(sub, cfg, "sma")
def find_ema(sub, cfg): return find_ma(sub, cfg, "ema")

def find_rsi(sub, cfg):
    p, t = cfg
    r = calc_rsi(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    crosses = np.where((prev < t) & (r >= t))[0]
    return list(crosses), (lambda i: i < len(r) and t < r[i] < 70)

def find_bb(sub, cfg):
    p, s = cfg
    mid, ub, lb = calc_bb(sub["close"].values, p, s)
    c = sub["close"].values
    above = c > ub
    crosses = np.where(above[1:] & ~above[:-1])[0] + 1
    return list(crosses), (lambda i: i < len(c) and c[i] > mid[i])

def find_donchian(sub, cfg):
    (p,) = cfg
    h = sub["high"].values; c = sub["close"].values
    rh = pd.Series(h).rolling(p).max().shift(1).values
    bo = c > rh
    crosses = np.where(bo[1:] & ~bo[:-1])[0] + 1
    return list(crosses), (lambda i: i < len(c) and i < len(rh) and not np.isnan(rh[i]) and c[i] >= rh[i] * 0.98)

def find_obv(sub, cfg):
    p, t = cfg
    c = sub["close"].values
    v = sub["volume"].values if "volume" in sub.columns else np.ones_like(c)
    if not np.any(v > 0): return [], None
    o = calc_obv(c, v); s = pd.Series(o)
    z = ((s - s.rolling(p).mean()) / (s.rolling(p).std() + 1e-12)).values
    above = z > t
    crosses = np.where(above[1:] & ~above[:-1])[0] + 1
    return list(crosses), (lambda i: i < len(z) and z[i] > 0)

def find_macd(sub, cfg):
    f, s, sig = cfg
    m, ss = calc_macd(sub["close"].values, f, s, sig)
    above = m > ss
    crosses = np.where(above[1:] & ~above[:-1])[0] + 1
    return list(crosses), (lambda i: i < len(m) and m[i] > ss[i])

def find_stoch(sub, cfg):
    kp, dp, ob, os_ = cfg
    k, d = calc_stoch(sub["high"].values, sub["low"].values, sub["close"].values, kp, dp)
    prev = np.roll(k, 1); prev[0] = k[0]
    crosses = np.where((prev < os_) & (k >= os_))[0]
    return list(crosses), (lambda i: i < len(k) and k[i] < ob)

def find_williams(sub, cfg):
    p, t = cfg
    w = calc_williams(sub["high"].values, sub["low"].values, sub["close"].values, p)
    prev = np.roll(w, 1); prev[0] = w[0]
    crosses = np.where((prev < t) & (w >= t))[0]
    return list(crosses), (lambda i: i < len(w) and w[i] < -20)

def find_roc(sub, cfg):
    p, t = cfg
    r = calc_roc(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    crosses = np.where((prev < t) & (r >= t))[0]
    return list(crosses), (lambda i: i < len(r) and r[i] > 0)

# ============================================================================
# MAIN DISPATCH — emit per-event raw rows
# ============================================================================

INDICATOR_REG = [
    ("SMA_cross",        smart_ma_candidates(),  find_sma),
    ("EMA_cross",        smart_ma_candidates(),  find_ema),
    ("RSI_oversold",     rsi_configs(),          find_rsi),
    ("BB_breach",        bb_configs(),           find_bb),
    ("Donchian_breakout", donchian_configs(),    find_donchian),
    ("OBV_zscore",       obv_configs(),          find_obv),
    ("MACD_cross",       macd_configs(),         find_macd),
    ("Stochastic_bounce", stoch_configs(),       find_stoch),
    ("Williams_R",       williams_configs(),     find_williams),
    ("ROC_momentum",     roc_configs(),          find_roc),
]

def run_all_events(panel, regime_overlay, asset_meta):
    """Emit one row per entry event with ALL per-exit returns + metadata."""
    panel_idx = {a: sub.sort_values("date").reset_index(drop=True) for a, sub in panel.groupby("asset")}
    regime_idx = regime_overlay.set_index("date").to_dict("index")

    rows = []
    for indicator_name, candidates, finder in INDICATOR_REG:
        print(f"\n  [{indicator_name}] {len(candidates)} configs", flush=True)
        n_events_total = 0
        for cfg in candidates:
            cfg_str = str(cfg)
            for asset, sub in panel_idx.items():
                entries, sig_fn = finder(sub, cfg)
                for ent_idx in entries:
                    if ent_idx < 60: continue  # min warmup
                    rets = evaluate_event(sub, ent_idx, sig_fn)
                    if rets is None: continue
                    row = sub.iloc[ent_idx]
                    ev_date = row["date"]
                    reg = regime_idx.get(ev_date, {})
                    am = asset_meta.get(asset, {})
                    out_row = {
                        "asset": asset,
                        "date": ev_date,
                        "entry_close": float(row["close"]),
                        "indicator": indicator_name,
                        "config": cfg_str,
                        "atr30_pct": float(row.get("atr30_pct") or 0.0),
                        "rv30_at_entry": float(row.get("rv30") or 0.0),
                        # Regime metadata
                        "btc_regime_30d": reg.get("btc_regime_30d", "UNK"),
                        "daily_market_cluster": reg.get("daily_market_cluster", -1),
                        "breadth_pct_long_at_entry": reg.get("breadth_pct_long", None),
                        # Asset metadata
                        "dna_bucket": am.get("dna_bucket", "UNK"),
                        "tier": am.get("tier", "UNK"),
                        "sector": am.get("sector", "UNK"),
                    }
                    for ex in EXIT_NAMES:
                        out_row[f"ret_{ex}"] = rets.get(ex, np.nan)
                    rows.append(out_row)
                    n_events_total += 1
        print(f"    -> {n_events_total} events", flush=True)
    return pd.DataFrame(rows)

# ============================================================================
# AGGREGATIONS
# ============================================================================

def agg_global(events_df):
    """Top-25 per indicator (global). Best exit found per config."""
    out = []
    for ind, grp in events_df.groupby("indicator"):
        for cfg, cfg_grp in grp.groupby("config"):
            if len(cfg_grp) < MIN_GLOBAL_EVENTS: continue
            best_exit = None; best_nav = -1e18; metrics = {}
            for ex in EXIT_NAMES:
                col = f"ret_{ex}"
                arr = cfg_grp[col].dropna().values
                if len(arr) < MIN_GLOBAL_EVENTS: continue
                nav = arr.sum() * SIZE * 100
                mean_pct = arr.mean() * 100
                hit = (arr > 0).mean() * 100
                sh = arr.mean() / (arr.std() + 1e-9)
                metrics[ex] = dict(n=int(len(arr)), nav_pct=round(nav, 3),
                                    mean_pct=round(mean_pct, 4), hit_pct=round(hit, 2),
                                    sharpe=round(sh, 4))
                if nav > best_nav: best_nav = nav; best_exit = ex
            if best_exit is None: continue
            m = metrics[best_exit]
            out.append({"indicator": ind, "config": cfg, "n_events": m["n"], "best_exit": best_exit,
                         "best_nav_pct": m["nav_pct"], "best_mean_pct": m["mean_pct"],
                         "best_hit_pct": m["hit_pct"], "best_sharpe": m["sharpe"],
                         "all_exits": json.dumps(metrics, default=str)})
    df = pd.DataFrame(out).sort_values(["indicator","best_nav_pct"], ascending=[True, False])
    # top-25 per indicator
    df = df.groupby("indicator", group_keys=False).head(25).reset_index(drop=True)
    df["rank_in_indicator"] = df.groupby("indicator").cumcount() + 1
    return df

def agg_per_asset(events_df):
    """Top-25 per (asset, indicator)."""
    out = []
    for (asset, ind), grp in events_df.groupby(["asset", "indicator"]):
        for cfg, cfg_grp in grp.groupby("config"):
            if len(cfg_grp) < MIN_EVENTS: continue
            best_exit = None; best_nav = -1e18; m = None
            for ex in EXIT_NAMES:
                arr = cfg_grp[f"ret_{ex}"].dropna().values
                if len(arr) < MIN_EVENTS: continue
                nav = arr.sum() * SIZE * 100
                if nav > best_nav:
                    best_nav = nav; best_exit = ex
                    m = dict(n=int(len(arr)), nav_pct=round(nav, 3),
                             mean_pct=round(arr.mean()*100, 4),
                             hit_pct=round((arr>0).mean()*100, 2),
                             sharpe=round(arr.mean()/(arr.std()+1e-9), 4))
            if best_exit is None: continue
            out.append({"asset": asset, "indicator": ind, "config": cfg,
                         "n_events": m["n"], "best_exit": best_exit,
                         "best_nav_pct": m["nav_pct"], "best_mean_pct": m["mean_pct"],
                         "best_hit_pct": m["hit_pct"], "best_sharpe": m["sharpe"]})
    df = pd.DataFrame(out).sort_values(["asset","indicator","best_nav_pct"], ascending=[True, True, False])
    df = df.groupby(["asset","indicator"], group_keys=False).head(25).reset_index(drop=True)
    df["rank_in_asset_indicator"] = df.groupby(["asset","indicator"]).cumcount() + 1
    return df

def agg_regime_exit(events_df):
    """regime × indicator × exit → metrics."""
    out = []
    for (reg, ind), grp in events_df.groupby(["btc_regime_30d", "indicator"]):
        for ex in EXIT_NAMES:
            arr = grp[f"ret_{ex}"].dropna().values
            if len(arr) < MIN_REGIME_EVENTS: continue
            out.append({"btc_regime_30d": reg, "indicator": ind, "exit_class": ex,
                         "n": int(len(arr)),
                         "mean_pct": round(arr.mean()*100, 4),
                         "median_pct": round(np.median(arr)*100, 4),
                         "hit_pct": round((arr>0).mean()*100, 2),
                         "sharpe": round(arr.mean()/(arr.std()+1e-9), 4),
                         "sum_pct_at_4size": round(arr.sum()*SIZE*100, 3)})
    return pd.DataFrame(out)

def agg_bucket(events_df):
    """dna_bucket × indicator × exit → metrics."""
    out = []
    for (b, ind), grp in events_df.groupby(["dna_bucket", "indicator"]):
        for ex in EXIT_NAMES:
            arr = grp[f"ret_{ex}"].dropna().values
            if len(arr) < MIN_REGIME_EVENTS: continue
            out.append({"dna_bucket": b, "indicator": ind, "exit_class": ex,
                         "n": int(len(arr)),
                         "mean_pct": round(arr.mean()*100, 4),
                         "hit_pct": round((arr>0).mean()*100, 2),
                         "sharpe": round(arr.mean()/(arr.std()+1e-9), 4),
                         "sum_pct_at_4size": round(arr.sum()*SIZE*100, 3)})
    return pd.DataFrame(out)

def best_exit_recommendation(regime_exit_df):
    """For each (indicator, regime), which exit wins? Sharpe + sum_pct combined rank."""
    out = []
    for (ind, reg), grp in regime_exit_df.groupby(["indicator", "btc_regime_30d"]):
        ranked = grp.sort_values(["sharpe","sum_pct_at_4size"], ascending=[False, False])
        if len(ranked) == 0: continue
        winner = ranked.iloc[0]
        out.append({"indicator": ind, "btc_regime_30d": reg,
                     "best_exit": winner["exit_class"],
                     "n": int(winner["n"]),
                     "sharpe": winner["sharpe"],
                     "mean_pct": winner["mean_pct"],
                     "sum_pct_at_4size": winner["sum_pct_at_4size"]})
    return pd.DataFrame(out).sort_values(["indicator","btc_regime_30d"]).reset_index(drop=True)

def asset_cousins(per_asset_df, asset_meta):
    """For each asset, top-3 (indicator, config) with asset metadata. Substrate for cousin-matching."""
    top3 = per_asset_df.groupby(["asset","indicator"]).head(1).copy()  # best per (asset, indicator)
    top3 = top3.sort_values(["asset","best_nav_pct"], ascending=[True, False])
    top3 = top3.groupby("asset", group_keys=False).head(5)
    top3["dna_bucket"] = top3["asset"].map(lambda a: asset_meta.get(a, {}).get("dna_bucket", "UNK"))
    top3["sector"] = top3["asset"].map(lambda a: asset_meta.get(a, {}).get("sector", "UNK"))
    top3["tier"] = top3["asset"].map(lambda a: asset_meta.get(a, {}).get("tier", "UNK"))
    return top3.reset_index(drop=True)

# ============================================================================
# REPORT
# ============================================================================

def write_report(events_df, global_df, per_asset_df, regime_exit_df, bucket_df, best_exit_df, cousins_df):
    lines = ["# Smart Discovery — EXHAUSTIVE TRAIN data-mining\n"]
    lines.append(f"TRAIN window: {TRAIN_START} -> {TRAIN_END}\n")
    lines.append(f"Total events: {len(events_df):,}\n")
    lines.append(f"Assets: {events_df['asset'].nunique()} | Indicators: {events_df['indicator'].nunique()} | Configs: {events_df['config'].nunique()}\n")
    lines.append(f"Regime distribution: {events_df['btc_regime_30d'].value_counts().to_dict()}\n")
    lines.append(f"DNA bucket distribution: {events_df['dna_bucket'].value_counts().to_dict()}\n")

    lines.append("\n## A) GLOBAL top-25 per indicator (universal — for unseen assets)\n")
    for ind, grp in global_df.groupby("indicator"):
        lines.append(f"\n### {ind}")
        lines.append("| rank | config | n | best_exit | NAV @4% | mean | hit | sharpe |")
        lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
        for _, r in grp.head(5).iterrows():
            lines.append(f"| {int(r['rank_in_indicator'])} | `{r['config']}` | {int(r['n_events'])} | {r['best_exit']} | {r['best_nav_pct']:+.2f}% | {r['best_mean_pct']:+.3f}% | {r['best_hit_pct']:.1f}% | {r['best_sharpe']:+.3f} |")

    lines.append("\n## B) Best exit per indicator per regime (recommendation table)\n")
    lines.append("| indicator | regime | best_exit | n | sharpe | mean | sum @4% |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for _, r in best_exit_df.iterrows():
        lines.append(f"| {r['indicator']} | {r['btc_regime_30d']} | {r['best_exit']} | {int(r['n'])} | {r['sharpe']:+.3f} | {r['mean_pct']:+.3f}% | {r['sum_pct_at_4size']:+.2f}% |")

    lines.append("\n## C) Top-15 across all per-asset / per-indicator (best by NAV)\n")
    top15 = per_asset_df.nlargest(15, "best_nav_pct")
    lines.append("| asset | indicator | config | n | best_exit | NAV @4% | hit |")
    lines.append("|---|---|---|---:|---|---:|---:|")
    for _, r in top15.iterrows():
        lines.append(f"| {r['asset']} | {r['indicator']} | `{r['config']}` | {int(r['n_events'])} | {r['best_exit']} | {r['best_nav_pct']:+.2f}% | {r['best_hit_pct']:.1f}% |")

    lines.append("\n## D) Regime-conditional exit decomposition (sharpe leaders per indicator)\n")
    pivot = regime_exit_df.pivot_table(index=["indicator","btc_regime_30d"], columns="exit_class", values="sharpe").round(3)
    lines.append("```")
    lines.append(pivot.to_string())
    lines.append("```")

    lines.append("\n## E) DNA bucket × indicator (mean return)\n")
    pv2 = bucket_df.pivot_table(index="indicator", columns="dna_bucket", values="mean_pct", aggfunc="mean").round(3)
    lines.append("```")
    lines.append(pv2.to_string())
    lines.append("```")

    lines.append("\n## F) Cousins substrate — top 5 setups per asset (for cousin-finding on new assets)\n")
    lines.append("File: cousins.csv (full table). Sample:\n")
    lines.append("| asset | dna | sector | indicator | config | best_exit | NAV @4% |")
    lines.append("|---|---|---|---|---|---|---:|")
    for _, r in cousins_df.head(20).iterrows():
        lines.append(f"| {r['asset']} | {r['dna_bucket']} | {r['sector']} | {r['indicator']} | `{r['config']}` | {r['best_exit']} | {r['best_nav_pct']:+.2f}% |")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*78)
    print("SMART DISCOVERY — EXHAUSTIVE / TRAIN")
    print("="*78)
    panel = load_train_panel()
    regime_overlay = load_regime_overlay()
    asset_meta = load_asset_metadata()

    print("\nDispatching event generation across 10 indicator classes...")
    events = run_all_events(panel, regime_overlay, asset_meta)
    print(f"\nTotal per-event rows: {len(events):,}")
    print(f"Regime breakdown of events: {events['btc_regime_30d'].value_counts().to_dict()}")
    print(f"DNA-bucket breakdown: {events['dna_bucket'].value_counts().to_dict()}")

    events_path = OUT_DIR / "per_event_raw.parquet"
    events.to_parquet(events_path, index=False, compression="zstd")
    print(f"Wrote {events_path} ({events_path.stat().st_size/1024/1024:.1f} MB)")

    print("\nAggregating: global top-25 per indicator...")
    global_df = agg_global(events)
    global_df.to_csv(OUT_DIR / "global_indicator_top25.csv", index=False)

    print("Aggregating: per-asset top-25 per indicator...")
    per_asset_df = agg_per_asset(events)
    per_asset_df.to_csv(OUT_DIR / "per_asset_indicator_top25.csv", index=False)

    print("Aggregating: regime × indicator × exit decomposition...")
    regime_exit_df = agg_regime_exit(events)
    regime_exit_df.to_csv(OUT_DIR / "regime_exit_decomposition.csv", index=False)

    print("Aggregating: bucket × indicator × exit decomposition...")
    bucket_df = agg_bucket(events)
    bucket_df.to_csv(OUT_DIR / "bucket_exit_decomposition.csv", index=False)

    print("Building best-exit recommendation table...")
    best_exit_df = best_exit_recommendation(regime_exit_df)
    best_exit_df.to_csv(OUT_DIR / "best_exit_per_indicator_per_regime.csv", index=False)

    print("Building cousins substrate...")
    cousins_df = asset_cousins(per_asset_df, asset_meta)
    cousins_df.to_csv(OUT_DIR / "cousins.csv", index=False)

    print("Writing REPORT.md...")
    write_report(events, global_df, per_asset_df, regime_exit_df, bucket_df, best_exit_df, cousins_df)

    print(f"\nALL OUTPUTS in {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            sz = f.stat().st_size
            print(f"  {f.name:50s} {sz/1024:.1f} KB")

if __name__ == "__main__":
    main()
