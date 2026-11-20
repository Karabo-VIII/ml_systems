"""Smart Discovery — all technical indicators, TRAIN window only, top-25 per indicator.

Per user direction (2026-05-20):
  1. Discover on TRAIN window only (canonical split: 2020-01 → 2023-07-01).
  2. Rank top-25 setups per indicator (not top-5).
  3. Cluster/regime/DNA overlaid as POST-HOC metadata, NOT used as filter.
  4. Smart candidate generation (Fibonacci/golden/log-spaced/decorrelation)
     replaces brute-force grid.

Indicators in scope (Round 1):
  - SMA cross (Fibonacci-like periods + golden ratio)
  - EMA cross (same)
  - RSI threshold
  - Bollinger Band breach (long on close > UB)
  - Donchian breakout
  - OBV z-score
  - MACD signal cross
  - Stochastic oversold-bounce
  - Williams %R oversold-bounce
  - ROC momentum-start

Multi-exit framework (from V2):
  - B_3d, C_5d, D_7d (fixed periods)
  - S_setup_toxic (signal-flip back / overbought) + 14d cap
  - F_atr30_K3_TP30 (best ATR variant from V2)
  - G_trail_5pct_3pct (best trail from V2)

Output:
  runs/oracle_layer3/SMART_DISCOVERY_ALL_TRAIN/
    {indicator}/top_25.csv    -- ranked top-25 per indicator
    {indicator}/all_configs.json  -- full result set
    SUMMARY.md                   -- cross-indicator summary
    BEST_SETUPS_TOP_25_OVERALL.csv -- combined top-25 across indicators
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_ALL_TRAIN"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical TRAIN window per config/data_config.yaml
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)

COST = 0.0024
SIZE = 0.04
MIN_EVENTS = 100  # configs with <100 events dropped from ranking

# ============================================================================
# SMART CANDIDATE GENERATORS
# ============================================================================

def fibonacci_pairs(max_period=120):
    fibs = [3, 5, 8, 13, 21, 34, 55, 89]
    fibs = [f for f in fibs if f <= max_period]
    return [(a, b) for a in fibs for b in fibs if a < b]

def golden_pairs(max_period=120):
    """Pairs where long/short ≈ φ=1.618."""
    base = [3, 5, 8, 13, 21, 34, 55, 89]
    pairs = []
    for a in base:
        b_target = int(round(a * 1.618))
        if b_target > a and b_target <= max_period and b_target not in [p[1] for p in pairs if p[0] == a]:
            pairs.append((a, b_target))
    return pairs

def log_spaced_pairs(n_short=4, n_long=5, max_period=120):
    """Log-spaced pair grid: short in [3,20], long in [21,89]."""
    shorts = np.unique(np.round(np.logspace(np.log10(3), np.log10(20), n_short)).astype(int))
    longs = np.unique(np.round(np.logspace(np.log10(21), np.log10(max_period), n_long)).astype(int))
    return [(int(a), int(b)) for a in shorts for b in longs if a < b]

def smart_ma_candidates():
    pairs = set(fibonacci_pairs() + golden_pairs() + log_spaced_pairs())
    return sorted(pairs)

def rsi_configs():
    """Period × oversold-threshold."""
    return [(p, t) for p in (7, 9, 14, 21, 28) for t in (20, 25, 30, 35, 40)]

def bb_configs():
    """Period × std-multiplier. Long on close > upper band."""
    return [(p, s) for p in (10, 20, 30, 50) for s in (1.5, 2.0, 2.5, 3.0)]

def donchian_configs():
    return [(p,) for p in (8, 13, 20, 30, 50, 89)]

def obv_configs():
    return [(p, t) for p in (14, 20, 30, 50, 100) for t in (1.5, 2.0, 2.5, 3.0)]

def macd_configs():
    """Fast × slow × signal."""
    return [(f, s, sig) for f in (5, 8, 12) for s in (21, 26, 34) for sig in (5, 9, 13) if f < s]

def stoch_configs():
    return [(k, d, ob, os_) for k in (7, 14, 21) for d in (3, 5) for ob, os_ in [(80,20),(85,15),(90,10)]]

def williams_configs():
    return [(p, t) for p in (7, 14, 21, 28) for t in (-80, -85, -90)]

def roc_configs():
    return [(p, t) for p in (5, 10, 14, 20, 30) for t in (5, 7, 10, 15)]

# ============================================================================
# INDICATOR SIGNAL CALCULATORS
# ============================================================================

def rsi_signal(closes: np.ndarray, period: int) -> np.ndarray:
    d = np.diff(closes, prepend=closes[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    up_s = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().values
    dn_s = pd.Series(dn).ewm(alpha=1/period, adjust=False).mean().values
    rs = up_s / (dn_s + 1e-12)
    return 100 - 100 / (1 + rs)

def bb_bands(closes: np.ndarray, period: int, std_mult: float):
    s = pd.Series(closes)
    mid = s.rolling(period).mean()
    sd = s.rolling(period).std()
    return mid.values, (mid + std_mult * sd).values, (mid - std_mult * sd).values

def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    return np.cumsum(direction * volumes)

def macd_signal(closes: np.ndarray, fast: int, slow: int, signal: int):
    s = pd.Series(closes)
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, sig.values

def stochastic(highs, lows, closes, k_period, d_period):
    hh = pd.Series(highs).rolling(k_period).max()
    ll = pd.Series(lows).rolling(k_period).min()
    k = 100 * (pd.Series(closes) - ll) / (hh - ll + 1e-12)
    d = k.rolling(d_period).mean()
    return k.values, d.values

def williams_r(highs, lows, closes, period):
    hh = pd.Series(highs).rolling(period).max()
    ll = pd.Series(lows).rolling(period).min()
    return (-100 * (hh - pd.Series(closes)) / (hh - ll + 1e-12)).values

def roc(closes: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(closes)
    return (100 * (s - s.shift(period)) / s.shift(period)).values

# ============================================================================
# DATA LOADER
# ============================================================================

def load_train_panel():
    """Load chimera 1d with OHLCV + ATR pre-computed, TRAIN window only."""
    print("Loading chimera 1d panel (TRAIN window only)...")
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    print(f"  {len(files)} asset files found")
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
        if len(df) < 200:
            continue
        df["asset"] = sym
        # ATR
        df["high_low"] = df["high"] - df["low"]
        df["high_pc"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_pc"] = (df["low"] - df["close"].shift(1)).abs()
        df["tr"] = df[["high_low","high_pc","low_pc"]].max(axis=1)
        df["atr30_pct"] = df["tr"].rolling(30).mean() / df["close"]
        panel_rows.append(df[["asset","date","open","high","low","close","volume","atr30_pct"]])
    panel = pd.concat(panel_rows, ignore_index=True)
    print(f"  TRAIN panel: {len(panel):,} rows × {panel['asset'].nunique()} assets")
    print(f"  Date range:  {panel['date'].min()} -> {panel['date'].max()}")
    return panel

# ============================================================================
# REGIME / CLUSTER / DNA OVERLAY (post-hoc metadata, NOT filter)
# ============================================================================

def overlay_metadata():
    """Load asset → cluster/bucket/regime metadata for post-hoc tagging."""
    meta = {}
    # asset_dna lookup
    try:
        u100 = pl.read_parquet(ROOT / "config" / "universes" / "u100_lookup.parquet").to_pandas()
        for _, r in u100.iterrows():
            meta[r["asset"]] = {"bucket": r.get("asset_dna", "UNK")}
    except Exception:
        pass
    # cluster_centroids
    try:
        cl = pd.read_csv(ROOT / "runs" / "oracle_layer2" / "cluster_centroids.csv")
        for _, r in cl.iterrows():
            a = r.get("asset", r.get("symbol", None))
            if a:
                meta.setdefault(a, {})["cluster"] = r.get("cluster", r.get("cluster_id", "UNK"))
    except Exception:
        pass
    return meta

# ============================================================================
# EXIT FRAMEWORK (from V2)
# ============================================================================

def apply_multi_exits(asset_sub, entry_idx, signal_func=None, max_horizon=14):
    """Compute NAV under each exit on a single entry event.

    asset_sub: DataFrame for one asset, sorted by date
    entry_idx: index in asset_sub where entry triggered (long opens at this close)
    signal_func: optional callable(idx) → bool (True = setup still active)
                  if None, setup-toxic exit unused
    """
    if entry_idx + 1 >= len(asset_sub):
        return {}
    entry_close = float(asset_sub.iloc[entry_idx]["close"])
    if entry_close <= 0 or not np.isfinite(entry_close):
        return {}
    atr30 = float(asset_sub.iloc[entry_idx].get("atr30_pct", 0.0)) or 0.0

    fwd = []
    for k in range(1, max_horizon + 1):
        if entry_idx + k < len(asset_sub):
            c = float(asset_sub.iloc[entry_idx + k]["close"])
            fwd.append(c if np.isfinite(c) else None)
        else:
            fwd.append(None)

    def pct(c):
        return c / entry_close - 1 - COST

    out = {}
    # Fixed-period holds
    if len(fwd) >= 3 and fwd[2] is not None: out["B_3d"] = pct(fwd[2])
    if len(fwd) >= 5 and fwd[4] is not None: out["C_5d"] = pct(fwd[4])
    if len(fwd) >= 7 and fwd[6] is not None: out["D_7d"] = pct(fwd[6])
    if len(fwd) >= 14 and fwd[13] is not None: out["E_14d"] = pct(fwd[13])

    # Setup-toxic (signal-flip back) + 14d cap
    if signal_func is not None:
        exit_s = None
        for k in range(max_horizon):
            if fwd[k] is None: continue
            if not signal_func(entry_idx + 1 + k):
                exit_s = fwd[k]; break
        if exit_s is None and fwd[-1] is not None:
            exit_s = fwd[-1]
        if exit_s is not None:
            out["S_setup_toxic"] = pct(exit_s)

    # ATR(30) K=3 stop + 30% TP (best from V2)
    if atr30 > 0:
        stop_pct = 3.0 * atr30
        tp_pct = 0.30
        exit_f = None
        for c in fwd:
            if c is None: continue
            r = c / entry_close - 1
            if r <= -stop_pct: exit_f = c; break
            if r >= tp_pct:   exit_f = c; break
        if exit_f is None and fwd[-1] is not None:
            exit_f = fwd[-1]
        if exit_f is not None:
            out["F_atr30_K3_TP30"] = pct(exit_f)

    # Trail 5%/3%
    peak = entry_close
    exit_g = None
    armed = False
    for c in fwd:
        if c is None: continue
        if c > peak: peak = c
        if not armed and (c / entry_close - 1) >= 0.05: armed = True
        if armed and c <= peak * 0.97: exit_g = c; break
    if exit_g is None and fwd[-1] is not None:
        exit_g = fwd[-1]
    if exit_g is not None:
        out["G_trail_5_3"] = pct(exit_g)

    return out

# ============================================================================
# EVENT FINDERS (per indicator)
# ============================================================================

def run_indicator_class(panel, indicator_name, candidate_gen, event_finder):
    """Generic dispatcher: for each candidate config, find events + score.

    candidate_gen: list of tuples
    event_finder: callable(asset_sub, config) → (list of entry indices, signal_func or None)
    """
    print(f"\n=== {indicator_name} ===")
    candidates = candidate_gen
    print(f"  candidates: {len(candidates)}")
    results = []

    panel_idx = {a: sub.sort_values("date").reset_index(drop=True) for a, sub in panel.groupby("asset")}

    for cfg in candidates:
        all_pnls = {}
        n_events = 0
        for asset, sub in panel_idx.items():
            entries, sig_fn = event_finder(sub, cfg)
            for ent_idx in entries:
                exits = apply_multi_exits(sub, ent_idx, sig_fn)
                for k, v in exits.items():
                    all_pnls.setdefault(k, []).append(v)
                if exits:
                    n_events += 1
        if n_events < MIN_EVENTS:
            continue
        # Find best exit type for this config
        best_exit = None
        best_nav = -1e9
        exit_summary = {}
        for ex, pnls in all_pnls.items():
            arr = np.array(pnls)
            if len(arr) < MIN_EVENTS: continue
            nav = arr.sum() * SIZE * 100
            mean_pct = arr.mean() * 100
            hit = (arr > 0).mean() * 100
            sharpe = arr.mean() / (arr.std() + 1e-9)
            exit_summary[ex] = {"n": int(len(arr)), "nav_pct": nav, "mean_pct": mean_pct,
                                "hit_pct": hit, "sharpe": sharpe}
            if nav > best_nav:
                best_nav = nav; best_exit = ex
        results.append({
            "indicator": indicator_name,
            "config": str(cfg),
            "n_events": n_events,
            "best_exit": best_exit,
            "best_nav_pct": best_nav,
            "best_mean_pct": exit_summary.get(best_exit, {}).get("mean_pct", None),
            "best_hit_pct": exit_summary.get(best_exit, {}).get("hit_pct", None),
            "best_sharpe": exit_summary.get(best_exit, {}).get("sharpe", None),
            "all_exits": exit_summary,
        })

    results.sort(key=lambda r: -r["best_nav_pct"])
    print(f"  scored: {len(results)} configs (>= {MIN_EVENTS} events)")
    if results:
        top3 = results[:3]
        for r in top3:
            print(f"    {r['config']:<25s}  best={r['best_exit']:<18s} NAV={r['best_nav_pct']:+8.2f}%  n={r['n_events']:5d}")
    return results

# ============================================================================
# PER-INDICATOR EVENT FINDERS
# ============================================================================

def find_ma_events(sub, cfg, kind="sma"):
    short_p, long_p = cfg
    closes = sub["close"].values
    if kind == "sma":
        s_short = pd.Series(closes).rolling(short_p).mean().values
        s_long = pd.Series(closes).rolling(long_p).mean().values
    else:
        s_short = pd.Series(closes).ewm(span=short_p, adjust=False).mean().values
        s_long = pd.Series(closes).ewm(span=long_p, adjust=False).mean().values
    above = s_short > s_long
    cross_up = np.where(above[1:] & ~above[:-1])[0] + 1  # idx
    sig_fn = lambda i: i < len(above) and above[i]
    return list(cross_up), sig_fn

def find_sma_events(sub, cfg): return find_ma_events(sub, cfg, "sma")
def find_ema_events(sub, cfg): return find_ma_events(sub, cfg, "ema")

def find_rsi_events(sub, cfg):
    p, t = cfg
    r = rsi_signal(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    crosses = np.where((prev < t) & (r >= t))[0]
    sig_fn = lambda i: i < len(r) and r[i] < 70 and r[i] > t
    return list(crosses), sig_fn

def find_bb_events(sub, cfg):
    p, std = cfg
    mid, ub, lb = bb_bands(sub["close"].values, p, std)
    closes = sub["close"].values
    above = closes > ub
    cross_up = np.where(above[1:] & ~above[:-1])[0] + 1
    sig_fn = lambda i: i < len(closes) and closes[i] > mid[i]
    return list(cross_up), sig_fn

def find_donchian_events(sub, cfg):
    (p,) = cfg
    highs = sub["high"].values
    closes = sub["close"].values
    rolling_high = pd.Series(highs).rolling(p).max().shift(1).values
    breakout = closes > rolling_high
    cross_up = np.where(breakout[1:] & ~breakout[:-1])[0] + 1
    sig_fn = lambda i: i < len(closes) and closes[i] >= rolling_high[i] * 0.98 if i < len(rolling_high) else False
    return list(cross_up), sig_fn

def find_obv_events(sub, cfg):
    p, t = cfg
    closes = sub["close"].values
    vols = sub["volume"].values if "volume" in sub.columns else np.ones_like(closes)
    if not np.any(vols > 0):
        return [], None
    obv_v = obv(closes, vols)
    s = pd.Series(obv_v)
    z = (s - s.rolling(p).mean()) / (s.rolling(p).std() + 1e-12)
    z = z.values
    above = z > t
    cross_up = np.where(above[1:] & ~above[:-1])[0] + 1
    sig_fn = lambda i: i < len(z) and z[i] > 0
    return list(cross_up), sig_fn

def find_macd_events(sub, cfg):
    f, s, sig_p = cfg
    macd, sig = macd_signal(sub["close"].values, f, s, sig_p)
    above = macd > sig
    cross_up = np.where(above[1:] & ~above[:-1])[0] + 1
    sig_fn = lambda i: i < len(macd) and macd[i] > sig[i]
    return list(cross_up), sig_fn

def find_stoch_events(sub, cfg):
    k_p, d_p, ob, os_ = cfg
    k, d = stochastic(sub["high"].values, sub["low"].values, sub["close"].values, k_p, d_p)
    prev = np.roll(k, 1); prev[0] = k[0]
    crosses = np.where((prev < os_) & (k >= os_))[0]
    sig_fn = lambda i: i < len(k) and k[i] < ob
    return list(crosses), sig_fn

def find_williams_events(sub, cfg):
    p, t = cfg
    w = williams_r(sub["high"].values, sub["low"].values, sub["close"].values, p)
    prev = np.roll(w, 1); prev[0] = w[0]
    crosses = np.where((prev < t) & (w >= t))[0]
    sig_fn = lambda i: i < len(w) and w[i] < -20
    return list(crosses), sig_fn

def find_roc_events(sub, cfg):
    p, t = cfg
    r = roc(sub["close"].values, p)
    prev = np.roll(r, 1); prev[0] = r[0]
    crosses = np.where((prev < t) & (r >= t))[0]
    sig_fn = lambda i: i < len(r) and r[i] > 0
    return list(crosses), sig_fn

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*78)
    print("SMART DISCOVERY — ALL INDICATORS, TRAIN WINDOW ONLY")
    print(f"TRAIN: {TRAIN_START} -> {TRAIN_END}")
    print("="*78)
    panel = load_train_panel()
    meta = overlay_metadata()
    print(f"  metadata available for {len(meta)} assets")

    indicator_runs = [
        ("SMA_cross",        smart_ma_candidates(),  find_sma_events),
        ("EMA_cross",        smart_ma_candidates(),  find_ema_events),
        ("RSI_oversold",     rsi_configs(),          find_rsi_events),
        ("BB_breach",        bb_configs(),           find_bb_events),
        ("Donchian_breakout", donchian_configs(),    find_donchian_events),
        ("OBV_zscore",       obv_configs(),          find_obv_events),
        ("MACD_cross",       macd_configs(),         find_macd_events),
        ("Stochastic_bounce", stoch_configs(),       find_stoch_events),
        ("Williams_R",       williams_configs(),     find_williams_events),
        ("ROC_momentum",     roc_configs(),          find_roc_events),
    ]

    all_top_25 = []
    all_results = {}

    for name, cand, finder in indicator_runs:
        results = run_indicator_class(panel, name, cand, finder)
        if not results:
            continue
        top_25 = results[:25]
        all_results[name] = results

        # Write per-indicator outputs
        ind_dir = OUT_DIR / name
        ind_dir.mkdir(parents=True, exist_ok=True)

        top_df = pd.DataFrame([{
            "rank": i + 1,
            "indicator": r["indicator"],
            "config": r["config"],
            "n_events": r["n_events"],
            "best_exit": r["best_exit"],
            "nav_pct": round(r["best_nav_pct"], 3),
            "mean_pct": round(r["best_mean_pct"], 4) if r["best_mean_pct"] is not None else None,
            "hit_pct": round(r["best_hit_pct"], 2) if r["best_hit_pct"] is not None else None,
            "sharpe": round(r["best_sharpe"], 4) if r["best_sharpe"] is not None else None,
        } for i, r in enumerate(top_25)])
        top_df.to_csv(ind_dir / "top_25.csv", index=False)
        (ind_dir / "all_configs.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

        for r in top_25:
            all_top_25.append({**{k: r[k] for k in ["indicator", "config", "n_events", "best_exit", "best_nav_pct", "best_mean_pct", "best_hit_pct", "best_sharpe"]}})

    # Cross-indicator combined top
    if all_top_25:
        combined = pd.DataFrame(all_top_25).sort_values("best_nav_pct", ascending=False).reset_index(drop=True)
        combined.insert(0, "rank", combined.index + 1)
        combined.to_csv(OUT_DIR / "BEST_SETUPS_TOP_25_OVERALL.csv", index=False)

    # SUMMARY.md
    lines = ["# Smart Discovery — All Indicators (TRAIN window)\n"]
    lines.append(f"TRAIN window: {TRAIN_START} -> {TRAIN_END}\n")
    lines.append(f"Multi-exit framework from V2 synthesis.\n")
    lines.append("\n## Per-indicator top-3\n")
    for name in all_results:
        lines.append(f"\n### {name}\n")
        lines.append("| rank | config | n_events | best_exit | NAV @4% | mean | hit |")
        lines.append("|---:|---|---:|---|---:|---:|---:|")
        for i, r in enumerate(all_results[name][:3]):
            lines.append(f"| {i+1} | `{r['config']}` | {r['n_events']} | {r['best_exit']} | {r['best_nav_pct']:+.2f}% | {r['best_mean_pct']:+.3f}% | {r['best_hit_pct']:.1f}% |")
    if all_top_25:
        lines.append("\n## Cross-indicator top-25\n")
        lines.append("| rank | indicator | config | n_events | best_exit | NAV @4% |")
        lines.append("|---:|---|---|---:|---|---:|")
        for _, r in combined.head(25).iterrows():
            lines.append(f"| {int(r['rank'])} | {r['indicator']} | `{r['config']}` | {r['n_events']} | {r['best_exit']} | {r['best_nav_pct']:+.2f}% |")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'SUMMARY.md'}")
    print(f"Wrote {OUT_DIR / 'BEST_SETUPS_TOP_25_OVERALL.csv'}")

if __name__ == "__main__":
    main()
