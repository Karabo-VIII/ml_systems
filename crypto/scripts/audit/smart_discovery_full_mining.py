"""Smart Discovery — FULL data-mining pass: all value-adding dimensions.

Inputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/per_event_raw.parquet (326K events)
  data/processed/chimera/1d/*.parquet (for forward-horizon expansion + drawdown)
  runs/oracle_layer2/daily_regime_cluster.parquet (BTC state + breadth)

Phases:
  A) ENRICH events with per-event features:
     - entry_strength (distance above threshold, per indicator)
     - pre_entry_run_1d / _3d / _7d (cumulative return BEFORE entry)
     - vol_regime_bin (rv30 quartile)
     - distance_from_90d_high
     - weekend_flag (entry on Sat/Sun)
     - days_since_first_appearance (per asset)
     - btc_drawdown_from_ath_at_entry
     - multi-horizon forward returns (3d/7d/14d/30d)
     - max_forward_ret_anywhere_le_30d (per-event oracle ceiling)
     - drawdown_during_14d_hold (peak-to-trough)
     - failure_mode label (WHIPSAW / DD_RECOVER / SUSTAINED_DOWN / WINNER)

  B) CONFLUENCE: count indicator-configs firing same (asset, date) -> confluence_count

  C) CAPTURE v2: mean-of-per-event-ratio (RED-flag fix) + multi-horizon oracle ratio

  D) WALK-FORWARD INSIDE TRAIN: 4 sub-folds (2020H1, 2021, 2022, 2023H1)
     -> top-25 stability per indicator across folds

  E) VARIANCE DECOMPOSITION: OLS on asset / regime / bucket / indicator fixed effects

  F) SHARPE-RANK vs NAV-RANK head-to-head per indicator

  G) PER-BUCKET BEST CONFIG (cleaner deploy unit than per-asset)

  H) PER-ASSET CONFIG STABILITY ACROSS YEARS

  I) SECTOR x INDICATOR MATRIX

Outputs in runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/:
  per_event_enriched.parquet
  confluence_analysis.csv
  capture_v2.csv
  walkforward_stability.csv
  variance_decomposition.txt
  sharpe_vs_nav_rank.csv
  per_bucket_best_config.csv
  per_asset_year_stability.csv
  sector_indicator_matrix.csv
  failure_mode_distribution.csv
  FULL_MINING_REPORT.md
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)
COST = 0.0024
SIZE = 0.04

# ============================================================================
# Phase A: Load and enrich
# ============================================================================

def load_close_panel():
    """Load OHLC closes for the TRAIN window with extended +30d cushion for forward returns."""
    print("Loading close panel (TRAIN + 30d cushion)...")
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    rows = []
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close", "high", "low"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= TRAIN_START - timedelta(days=120)) &
                 (df["date"] <= TRAIN_END + timedelta(days=45))].reset_index(drop=True)
        if len(df) < 30:
            continue
        df["asset"] = sym
        rows.append(df[["asset", "date", "close", "high", "low"]])
    panel = pd.concat(rows, ignore_index=True)
    print(f"  panel: {len(panel):,} rows / {panel['asset'].nunique()} assets")
    return panel

def enrich_events(events, panel):
    """Add per-event features."""
    print("\n[Phase A] Enriching events with new features...")
    panel = panel.sort_values(["asset","date"]).reset_index(drop=True)
    panel_idx = {a: sub.reset_index(drop=True) for a, sub in panel.groupby("asset")}
    btc_close = panel[panel["asset"]=="BTC"][["date","close"]].rename(columns={"close":"btc_close"})
    btc_close["btc_running_ath"] = btc_close["btc_close"].cummax()
    btc_close["btc_dd_from_ath"] = btc_close["btc_close"] / btc_close["btc_running_ath"] - 1
    btc_idx = btc_close.set_index("date").to_dict("index")

    # First-appearance dates per asset
    first_date = {a: sub["date"].min() for a, sub in panel.groupby("asset")}

    enriched_rows = []
    n = 0
    last_print = 0
    for ev in events.itertuples(index=False):
        n += 1
        if n - last_print >= 50000:
            print(f"    {n:,}/{len(events):,} events enriched")
            last_print = n
        asset, dt = ev.asset, ev.date
        sub = panel_idx.get(asset)
        if sub is None: continue
        # Find date row
        date_idx_arr = sub.index[sub["date"] == dt].tolist()
        if not date_idx_arr: continue
        idx = date_idx_arr[0]
        if idx + 30 >= len(sub): continue
        entry_close = float(sub.iloc[idx]["close"])
        # pre-entry returns
        def safe_close(off):
            j = idx + off
            if 0 <= j < len(sub):
                return float(sub.iloc[j]["close"])
            return None
        pre1 = safe_close(-1); pre3 = safe_close(-3); pre7 = safe_close(-7); pre90max = None
        # 90d running max BEFORE entry (today's close included? exclude to be safe)
        lo = max(0, idx - 90)
        prior = sub.iloc[lo:idx+1]
        pre90max = prior["high"].max() if len(prior) else None
        # forward returns at 3/7/14/30 + max-anywhere
        fwd_closes = [(safe_close(k) or np.nan) for k in range(1, 31)]
        fwd_arr = np.array(fwd_closes, dtype=float)
        valid = ~np.isnan(fwd_arr)
        if valid.sum() < 3: continue
        rets = (fwd_arr / entry_close - 1)
        # multi-horizon
        def hr(h):
            j = h - 1
            return float(rets[j] - COST) if j < len(rets) and valid[j] else None
        ret_3d = hr(3); ret_7d = hr(7); ret_14d = hr(14); ret_30d = hr(30)
        # max forward at any horizon le 30d (per-event oracle ceiling)
        ret_max_le30 = float(np.nanmax(rets) - COST) if valid.sum() else None
        # Drawdown during 14d hold (max peak-to-trough)
        within_14 = rets[:14]
        if valid[:14].sum() >= 3:
            cum_max = np.fmax.accumulate(np.where(valid[:14], 1 + within_14, np.nan))
            cum_max = pd.Series(cum_max).ffill().values
            cur = np.where(valid[:14], 1 + within_14, np.nan)
            dd_series = (cur / cum_max - 1)
            max_drawdown_14d = float(np.nanmin(dd_series)) if np.any(~np.isnan(dd_series)) else None
        else:
            max_drawdown_14d = None
        # failure_mode
        e14 = ret_14d
        if e14 is None:
            mode = "UNK"
        elif e14 >= 0.05:
            mode = "WINNER"
        elif max_drawdown_14d is not None and max_drawdown_14d <= -0.10 and e14 > -0.03:
            mode = "DD_RECOVER"
        elif ret_3d is not None and ret_3d <= -0.05:
            mode = "WHIPSAW"
        elif e14 <= -0.05:
            mode = "SUSTAINED_DOWN"
        else:
            mode = "FLAT"
        # vol regime bin
        rv = ev.rv30_at_entry or 0
        # entry_strength placeholder — depends on indicator class
        out = dict(asset=asset, date=dt, indicator=ev.indicator, config=ev.config,
                    entry_close=entry_close, atr30_pct=ev.atr30_pct, rv30_at_entry=rv,
                    btc_regime_30d=ev.btc_regime_30d,
                    daily_market_cluster=ev.daily_market_cluster,
                    breadth_pct_long_at_entry=ev.breadth_pct_long_at_entry,
                    dna_bucket=ev.dna_bucket, tier=ev.tier, sector=ev.sector,
                    # Existing exits passed through
                    ret_B_3d=ev.ret_B_3d, ret_C_5d=ev.ret_C_5d, ret_D_7d=ev.ret_D_7d,
                    ret_E_14d=ev.ret_E_14d, ret_S_setup_toxic=ev.ret_S_setup_toxic,
                    ret_F_atr30_K3_TP30=ev.ret_F_atr30_K3_TP30, ret_G_trail_5_3=ev.ret_G_trail_5_3,
                    # NEW columns
                    pre_run_1d = entry_close / pre1 - 1 if pre1 else None,
                    pre_run_3d = entry_close / pre3 - 1 if pre3 else None,
                    pre_run_7d = entry_close / pre7 - 1 if pre7 else None,
                    distance_from_90d_high = entry_close / pre90max - 1 if pre90max and pre90max > 0 else None,
                    weekend_flag = 1 if dt.weekday() >= 5 else 0,
                    days_since_first_appearance = (dt - first_date.get(asset, dt)).days,
                    btc_drawdown_from_ath = btc_idx.get(dt, {}).get("btc_dd_from_ath", None),
                    ret_fwd_3d = ret_3d, ret_fwd_7d = ret_7d, ret_fwd_14d = ret_14d, ret_fwd_30d = ret_30d,
                    ret_max_anywhere_le30 = ret_max_le30,
                    max_drawdown_14d = max_drawdown_14d,
                    failure_mode = mode,
        )
        enriched_rows.append(out)

    enriched = pd.DataFrame(enriched_rows)
    # vol regime bin (per-asset quartile of rv30)
    enriched["vol_regime_bin"] = enriched.groupby("asset")["rv30_at_entry"].transform(
        lambda x: pd.qcut(x, 4, labels=["LOW","MIDLOW","MIDHIGH","HIGH"], duplicates="drop").astype(str)
    )
    print(f"  enriched: {len(enriched):,} rows")
    return enriched

def add_entry_strength(enriched, panel):
    """Per indicator: distance above threshold at entry."""
    print("\n[Phase A continued] Adding entry_strength per indicator...")
    panel = panel.sort_values(["asset","date"]).reset_index(drop=True)
    panel_idx = {a: sub.reset_index(drop=True) for a, sub in panel.groupby("asset")}
    out = []
    for ev in enriched.itertuples(index=False):
        asset, dt = ev.asset, ev.date
        sub = panel_idx.get(asset)
        if sub is None:
            out.append(None); continue
        date_idx_arr = sub.index[sub["date"] == dt].tolist()
        if not date_idx_arr:
            out.append(None); continue
        idx = date_idx_arr[0]
        ind = ev.indicator
        cfg = ev.config
        close = float(sub.iloc[idx]["close"])
        atr = ev.atr30_pct or 0
        try:
            if ind == "Donchian_breakout":
                p = int(cfg.strip("(),"))
                if idx >= p:
                    prev_high = sub.iloc[idx-p:idx]["high"].max()
                    strength = (close - prev_high) / (close * atr + 1e-12) if atr > 0 else None
                else:
                    strength = None
            elif ind == "BB_breach":
                p, s = eval(cfg)
                window = sub.iloc[max(0,idx-p+1):idx+1]["close"]
                mid = window.mean(); sd = window.std()
                ub = mid + s*sd
                strength = (close - ub) / (sd + 1e-12)
            elif ind == "ROC_momentum":
                p, t = eval(cfg)
                if idx >= p:
                    prev = float(sub.iloc[idx-p]["close"])
                    roc_val = 100 * (close - prev) / prev
                    strength = roc_val - t
                else:
                    strength = None
            elif ind in ("SMA_cross","EMA_cross"):
                a, b = eval(cfg)
                window = sub.iloc[max(0,idx-b+1):idx+1]["close"]
                if len(window) >= b:
                    if ind == "SMA_cross":
                        ma_s = window.tail(a).mean()
                        ma_l = window.mean()
                    else:
                        ma_s = window.ewm(span=a, adjust=False).mean().iloc[-1]
                        ma_l = window.ewm(span=b, adjust=False).mean().iloc[-1]
                    strength = (ma_s - ma_l) / (close * atr + 1e-12) if atr > 0 else None
                else:
                    strength = None
            else:
                strength = None
        except Exception:
            strength = None
        out.append(strength)
    enriched["entry_strength"] = out
    print(f"  entry_strength computed for {(enriched['entry_strength'].notna().sum()):,} events")
    return enriched

# ============================================================================
# Phase B: Confluence count
# ============================================================================

def add_confluence(enriched):
    """For each (asset, date), count indicators that fired."""
    print("\n[Phase B] Computing confluence counts...")
    # Use distinct indicators (not configs)
    grp = enriched.groupby(["asset","date"])["indicator"].nunique().reset_index(name="confluence_n_indicators")
    grp2 = enriched.groupby(["asset","date"]).size().reset_index(name="confluence_n_configs")
    enriched = enriched.merge(grp, on=["asset","date"])
    enriched = enriched.merge(grp2, on=["asset","date"])
    print(f"  distribution of confluence_n_indicators: {enriched['confluence_n_indicators'].value_counts().sort_index().to_dict()}")
    return enriched

# ============================================================================
# Phase C: Capture v2 (mean-of-per-event-ratio + multi-horizon)
# ============================================================================

def capture_v2(enriched):
    """Per (indicator, config): mean of (realized / oracle-per-event)."""
    print("\n[Phase C] Computing capture v2 metrics...")
    rows = []
    for (ind, cfg), grp in enriched.groupby(["indicator","config"]):
        if len(grp) < 200: continue
        e14 = grp["ret_fwd_14d"].dropna()  # raw forward 14d (uncapped by exit)
        oracle = grp["ret_max_anywhere_le30"].dropna()
        common = grp[["ret_fwd_14d", "ret_max_anywhere_le30"]].dropna()
        if len(common) < 100: continue
        per_event_ratio = (common["ret_fwd_14d"] / common["ret_max_anywhere_le30"].replace(0, np.nan)).clip(-5, 5)
        # Only count positive oracle events (where the asset went up at some point in 30d)
        pos_oracle = common[common["ret_max_anywhere_le30"] > 0]
        if len(pos_oracle) >= 50:
            mean_ratio_pos_oracle = (pos_oracle["ret_fwd_14d"] / pos_oracle["ret_max_anywhere_le30"]).clip(-2, 2).mean()
        else:
            mean_ratio_pos_oracle = None
        rows.append({
            "indicator": ind, "config": cfg, "n": int(len(grp)),
            "realized_E14_nav_pct": grp["ret_E_14d"].dropna().sum() * SIZE * 100,
            "fwd_14d_mean_pct": e14.mean() * 100 if len(e14) else None,
            "fwd_14d_sum_pct_at_4size": e14.sum() * SIZE * 100,
            "oracle_per_event_mean_pct": oracle.mean() * 100 if len(oracle) else None,
            "oracle_per_event_sum_pct_at_4size": oracle.sum() * SIZE * 100,
            "mean_event_capture_ratio_pct": float(per_event_ratio.mean() * 100) if len(per_event_ratio) else None,
            "median_event_capture_ratio_pct": float(per_event_ratio.median() * 100) if len(per_event_ratio) else None,
            "mean_event_capture_ratio_pos_oracle_pct": float(mean_ratio_pos_oracle * 100) if mean_ratio_pos_oracle is not None else None,
            "pct_events_realizing_oracle_at_14d": float(((common["ret_fwd_14d"] >= common["ret_max_anywhere_le30"] * 0.5)).mean() * 100),
        })
    df = pd.DataFrame(rows).sort_values(["indicator","fwd_14d_sum_pct_at_4size"], ascending=[True, False])
    return df

# ============================================================================
# Phase D: Walk-forward inside TRAIN
# ============================================================================

def walk_forward(enriched):
    """4 sub-folds: 2020H1+2020H2 / 2021 / 2022 / 2023H1. Top-25 stability per indicator."""
    print("\n[Phase D] Walk-forward inside TRAIN...")
    enriched = enriched.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    folds = {
        "F1_2020": (date(2020,1,1), date(2020,12,31)),
        "F2_2021": (date(2021,1,1), date(2021,12,31)),
        "F3_2022": (date(2022,1,1), date(2022,12,31)),
        "F4_2023H1": (date(2023,1,1), date(2023,7,1)),
    }
    fold_rank = {}
    for fold, (s, e) in folds.items():
        sub = enriched[(enriched["date"].dt.date >= s) & (enriched["date"].dt.date <= e)]
        rank = {}
        for ind, grp in sub.groupby("indicator"):
            cfg_stats = grp.groupby("config")["ret_E_14d"].agg([("n","count"),("sum","sum")])
            cfg_stats = cfg_stats[cfg_stats["n"] >= 30]
            if len(cfg_stats) == 0: continue
            cfg_stats = cfg_stats.sort_values("sum", ascending=False).head(25)
            rank[ind] = list(cfg_stats.index)
        fold_rank[fold] = rank

    # Build stability matrix: per (indicator, config), how many folds it appears in top-25
    out = []
    for ind in set().union(*[set(r.keys()) for r in fold_rank.values()]):
        # collect all top-25 configs ever
        all_cfgs = set()
        for fold in folds:
            all_cfgs |= set(fold_rank.get(fold, {}).get(ind, []))
        for cfg in all_cfgs:
            present = {fold: (cfg in fold_rank[fold].get(ind, [])) for fold in folds}
            out.append({"indicator": ind, "config": cfg,
                         "n_folds_in_top25": sum(present.values()),
                         **{f"in_{f}": int(v) for f,v in present.items()}})
    df = pd.DataFrame(out).sort_values(["indicator","n_folds_in_top25"], ascending=[True, False])
    return df

# ============================================================================
# Phase E: Variance decomposition
# ============================================================================

def variance_decomp(enriched):
    """OLS on ret_E_14d ~ asset + month + regime + bucket + indicator fixed effects."""
    print("\n[Phase E] Variance decomposition...")
    df = enriched.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df = df.dropna(subset=["ret_E_14d","btc_regime_30d","dna_bucket","indicator","asset"])
    y = df["ret_E_14d"].values
    var_y = y.var()
    if var_y == 0:
        return "Variance is zero — degenerate", {}

    # Sequential R^2: how much variance each factor explains
    factors = ["btc_regime_30d","dna_bucket","indicator","asset","year_month"]
    results = []
    cumulative = pd.Series(y - y.mean())
    for f in factors:
        # Group means by this factor; subtract within-group means
        group_means = df.groupby(f)["ret_E_14d"].transform("mean")
        ss_explained_by_factor = ((group_means - df["ret_E_14d"].mean()) ** 2).sum()
        r2 = ss_explained_by_factor / ((y - y.mean()) ** 2).sum()
        results.append((f, r2 * 100, int(df[f].nunique())))
    # Render
    lines = ["Variance decomposition — single-factor R^2 (marginal, not joint):"]
    lines.append(f"  Total variance of ret_E_14d (n={len(df):,}): {var_y:.6f}")
    lines.append(f"  Mean: {y.mean()*100:+.3f}%  std: {y.std()*100:.2f}%")
    lines.append("")
    lines.append(f"  {'Factor':<22s}{'R^2 %':>8s}{'n_levels':>12s}")
    for f, r2, n_lev in sorted(results, key=lambda r: -r[1]):
        lines.append(f"  {f:<22s}{r2:>8.3f}{n_lev:>12d}")
    return "\n".join(lines), dict(results=results)

# ============================================================================
# Phase F: Sharpe-rank vs NAV-rank
# ============================================================================

def sharpe_vs_nav(enriched):
    print("\n[Phase F] Sharpe-rank vs NAV-rank head-to-head...")
    rows = []
    for (ind, cfg), grp in enriched.groupby(["indicator","config"]):
        if len(grp) < 200: continue
        arr = grp["ret_E_14d"].dropna().values
        if len(arr) < 200: continue
        nav = arr.sum() * SIZE * 100
        sh = arr.mean() / (arr.std() + 1e-9)
        rows.append({"indicator": ind, "config": cfg, "n": len(arr),
                      "nav_pct": nav, "mean_pct": arr.mean()*100,
                      "sharpe": sh, "hit_pct": (arr>0).mean()*100})
    df = pd.DataFrame(rows)
    out = []
    for ind, grp in df.groupby("indicator"):
        nav_rank = grp.sort_values("nav_pct", ascending=False).head(10).reset_index(drop=True)
        sh_rank  = grp.sort_values("sharpe", ascending=False).head(10).reset_index(drop=True)
        for i in range(min(len(nav_rank), len(sh_rank))):
            out.append({"indicator": ind, "rank": i+1,
                         "by_NAV_config": nav_rank.iloc[i]["config"],
                         "by_NAV_value": round(nav_rank.iloc[i]["nav_pct"], 2),
                         "by_NAV_sharpe": round(nav_rank.iloc[i]["sharpe"], 3),
                         "by_Sharpe_config": sh_rank.iloc[i]["config"],
                         "by_Sharpe_value": round(sh_rank.iloc[i]["sharpe"], 3),
                         "by_Sharpe_nav": round(sh_rank.iloc[i]["nav_pct"], 2)})
    return pd.DataFrame(out)

# ============================================================================
# Phase G: Per-bucket best config
# ============================================================================

def per_bucket_best(enriched):
    print("\n[Phase G] Per-bucket best config...")
    rows = []
    for (b, ind, cfg), grp in enriched.groupby(["dna_bucket","indicator","config"]):
        if len(grp) < 100: continue
        arr = grp["ret_E_14d"].dropna().values
        if len(arr) < 50: continue
        rows.append({"dna_bucket": b, "indicator": ind, "config": cfg,
                      "n": len(arr), "mean_pct": arr.mean()*100,
                      "hit_pct": (arr>0).mean()*100,
                      "sharpe": arr.mean()/(arr.std()+1e-9),
                      "sum_pct_at_4size": arr.sum() * SIZE * 100})
    df = pd.DataFrame(rows).sort_values(["dna_bucket","indicator","sum_pct_at_4size"], ascending=[True, True, False])
    df = df.groupby(["dna_bucket","indicator"], group_keys=False).head(3).reset_index(drop=True)
    df["rank_in_bucket_indicator"] = df.groupby(["dna_bucket","indicator"]).cumcount() + 1
    return df

# ============================================================================
# Phase H: Per-asset year stability
# ============================================================================

def per_asset_year_stability(enriched):
    print("\n[Phase H] Per-asset year stability...")
    df = enriched.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    # For each (asset, indicator, year), find best config
    out = []
    for (asset, ind, yr), grp in df.groupby(["asset","indicator","year"]):
        cfg_stats = grp.groupby("config")["ret_E_14d"].agg([("n","count"),("sum","sum"),("mean","mean")])
        cfg_stats = cfg_stats[cfg_stats["n"] >= 5]
        if len(cfg_stats) == 0: continue
        best = cfg_stats.sort_values("sum", ascending=False).iloc[0]
        out.append({"asset": asset, "indicator": ind, "year": yr,
                     "best_config": best.name, "n": int(best["n"]),
                     "sum_ret": float(best["sum"]), "mean_ret": float(best["mean"])})
    df_out = pd.DataFrame(out)
    return df_out

# ============================================================================
# Phase I: Sector x indicator
# ============================================================================

def sector_indicator(enriched):
    print("\n[Phase I] Sector x indicator matrix...")
    rows = []
    for (sec, ind), grp in enriched.groupby(["sector","indicator"]):
        arr = grp["ret_E_14d"].dropna().values
        if len(arr) < 30: continue
        rows.append({"sector": sec, "indicator": ind, "n": len(arr),
                      "mean_pct": arr.mean()*100, "hit_pct": (arr>0).mean()*100,
                      "sharpe": arr.mean()/(arr.std()+1e-9),
                      "sum_pct_at_4size": arr.sum()*SIZE*100})
    return pd.DataFrame(rows).sort_values(["sector","sum_pct_at_4size"], ascending=[True, False])

# ============================================================================
# Failure mode distribution
# ============================================================================

def failure_modes(enriched):
    print("\n[Phase J] Failure mode distribution...")
    rows = []
    for (ind, mode), grp in enriched.groupby(["indicator","failure_mode"]):
        rows.append({"indicator": ind, "failure_mode": mode, "n": len(grp),
                      "mean_ret": grp["ret_E_14d"].mean() * 100,
                      "median_dd_14d": grp["max_drawdown_14d"].median() * 100})
    return pd.DataFrame(rows).sort_values(["indicator","n"], ascending=[True, False])

# ============================================================================
# Confluence effect on returns
# ============================================================================

def confluence_effect(enriched):
    rows = []
    for ind, grp in enriched.groupby("indicator"):
        for n_conf, sub in grp.groupby("confluence_n_indicators"):
            arr = sub["ret_E_14d"].dropna().values
            if len(arr) < 100: continue
            rows.append({"indicator": ind, "confluence_n_indicators": int(n_conf),
                          "n_events": int(len(arr)),
                          "mean_pct": arr.mean()*100,
                          "hit_pct": (arr>0).mean()*100,
                          "sharpe": arr.mean()/(arr.std()+1e-9)})
    return pd.DataFrame(rows)

# ============================================================================
# Effect-level analyses (entry_strength, pre_run, distance_from_high, weekend, vol_regime)
# ============================================================================

def feature_effect_analysis(enriched, feature, n_bins=4):
    """For a continuous feature, bin it and report return per bin."""
    df = enriched[enriched[feature].notna()].copy()
    if len(df) < 200: return pd.DataFrame()
    try:
        df["bin"] = pd.qcut(df[feature], n_bins, labels=[f"Q{i}" for i in range(n_bins)], duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for (ind, b), grp in df.groupby(["indicator","bin"]):
        arr = grp["ret_E_14d"].dropna().values
        if len(arr) < 30: continue
        rows.append({"indicator": ind, f"{feature}_bin": str(b), "n": len(arr),
                      "mean_pct": arr.mean()*100, "hit_pct": (arr>0).mean()*100})
    return pd.DataFrame(rows)

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*78)
    print("FULL DATA-MINING PASS")
    print("="*78)
    events = pd.read_parquet(OUT_DIR / "per_event_raw.parquet")
    print(f"Loaded events: {len(events):,}")

    panel = load_close_panel()

    enriched = enrich_events(events, panel)
    enriched = add_entry_strength(enriched, panel)
    enriched = add_confluence(enriched)

    enriched_path = OUT_DIR / "per_event_enriched.parquet"
    enriched.to_parquet(enriched_path, index=False, compression="zstd")
    print(f"\nWrote {enriched_path} ({enriched_path.stat().st_size/1024/1024:.1f} MB)")

    cap_v2 = capture_v2(enriched)
    cap_v2.to_csv(OUT_DIR / "capture_v2.csv", index=False)

    wf = walk_forward(enriched)
    wf.to_csv(OUT_DIR / "walkforward_stability.csv", index=False)

    var_text, _ = variance_decomp(enriched)
    (OUT_DIR / "variance_decomposition.txt").write_text(var_text, encoding="utf-8")
    print(var_text)

    sv = sharpe_vs_nav(enriched)
    sv.to_csv(OUT_DIR / "sharpe_vs_nav_rank.csv", index=False)

    pbb = per_bucket_best(enriched)
    pbb.to_csv(OUT_DIR / "per_bucket_best_config.csv", index=False)

    pys = per_asset_year_stability(enriched)
    pys.to_csv(OUT_DIR / "per_asset_year_stability.csv", index=False)

    si = sector_indicator(enriched)
    si.to_csv(OUT_DIR / "sector_indicator_matrix.csv", index=False)

    fm = failure_modes(enriched)
    fm.to_csv(OUT_DIR / "failure_mode_distribution.csv", index=False)

    ce = confluence_effect(enriched)
    ce.to_csv(OUT_DIR / "confluence_effect.csv", index=False)

    # Feature effect analyses
    for feat in ["entry_strength", "pre_run_1d", "pre_run_3d", "pre_run_7d",
                 "distance_from_90d_high", "btc_drawdown_from_ath", "rv30_at_entry"]:
        eff = feature_effect_analysis(enriched, feat)
        if len(eff): eff.to_csv(OUT_DIR / f"feature_effect_{feat}.csv", index=False)
        print(f"  feature_effect_{feat}: {len(eff)} rows")

    # Report
    lines = ["# Full Data-Mining Pass — TRAIN\n"]
    lines.append(f"Enriched events: {len(enriched):,}\n")
    lines.append("\n## Variance decomposition\n```\n" + var_text + "\n```\n")

    lines.append("\n## Capture v2 (mean of per-event ratio + multi-horizon oracle)\n")
    top_cap = cap_v2.groupby("indicator").head(3)
    lines.append("| indicator | config | n | E14 NAV | mean cap ratio % | median cap ratio % | pos-oracle mean cap % | pct events >=50% oracle |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for _, r in top_cap.iterrows():
        lines.append(f"| {r['indicator']} | `{r['config']}` | {int(r['n']):,} | {r['realized_E14_nav_pct']:+.2f}% | {r['mean_event_capture_ratio_pct']:+.2f}% | {r['median_event_capture_ratio_pct']:+.2f}% | {(r['mean_event_capture_ratio_pos_oracle_pct'] if r['mean_event_capture_ratio_pos_oracle_pct'] is not None else float('nan')):+.2f}% | {r['pct_events_realizing_oracle_at_14d']:.1f}% |")

    lines.append("\n## Walk-forward stability inside TRAIN (top-25 across 4 folds)\n")
    stable = wf[wf["n_folds_in_top25"] >= 3].sort_values(["indicator","n_folds_in_top25"], ascending=[True, False])
    lines.append(f"Total configs in any fold's top-25: {len(wf)}")
    lines.append(f"Configs in >=3 of 4 folds (robust): {len(stable)}")
    lines.append("\nMost stable per indicator (n_folds>=3):")
    lines.append("| indicator | config | n_folds_in_top25 | F1_2020 | F2_2021 | F3_2022 | F4_2023H1 |")
    lines.append("|---|---|--:|:--:|:--:|:--:|:--:|")
    for ind, grp in stable.groupby("indicator"):
        for _, r in grp.head(3).iterrows():
            lines.append(f"| {ind} | `{r['config']}` | {int(r['n_folds_in_top25'])} | {r['in_F1_2020']} | {r['in_F2_2021']} | {r['in_F3_2022']} | {r['in_F4_2023H1']} |")

    lines.append("\n## Sharpe-rank vs NAV-rank (top-5 head-to-head)\n")
    for ind in sv["indicator"].unique():
        sub = sv[sv["indicator"]==ind].head(5)
        if len(sub) == 0: continue
        lines.append(f"\n### {ind}")
        lines.append("| rank | by_NAV config | NAV % | NAV's Sharpe | by_Sharpe config | Sharpe | Sharpe's NAV |")
        lines.append("|--:|---|--:|--:|---|--:|--:|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['rank'])} | `{r['by_NAV_config']}` | {r['by_NAV_value']:+.2f}% | {r['by_NAV_sharpe']:+.3f} | `{r['by_Sharpe_config']}` | {r['by_Sharpe_value']:+.3f} | {r['by_Sharpe_nav']:+.2f}% |")

    lines.append("\n## Per-bucket best config (top-3 per bucket per indicator)\n")
    for b in pbb["dna_bucket"].unique():
        sub = pbb[pbb["dna_bucket"]==b]
        lines.append(f"\n### {b}")
        lines.append("| indicator | config | n | mean | hit | sharpe | NAV @4% |")
        lines.append("|---|---|--:|--:|--:|--:|--:|")
        for _, r in sub.head(15).iterrows():
            lines.append(f"| {r['indicator']} | `{r['config']}` | {int(r['n'])} | {r['mean_pct']:+.2f}% | {r['hit_pct']:.1f}% | {r['sharpe']:+.3f} | {r['sum_pct_at_4size']:+.2f}% |")

    lines.append("\n## Sector x indicator (sharpe leaders)\n")
    lines.append("| sector | indicator | n | mean | hit | sharpe | NAV @4% |")
    lines.append("|---|---|--:|--:|--:|--:|--:|")
    for _, r in si.head(50).iterrows():
        lines.append(f"| {r['sector']} | {r['indicator']} | {int(r['n'])} | {r['mean_pct']:+.2f}% | {r['hit_pct']:.1f}% | {r['sharpe']:+.3f} | {r['sum_pct_at_4size']:+.2f}% |")

    lines.append("\n## Failure-mode distribution per indicator\n")
    lines.append("| indicator | mode | n | mean_ret_14d | median_max_dd_14d |")
    lines.append("|---|---|--:|--:|--:|")
    for _, r in fm.iterrows():
        lines.append(f"| {r['indicator']} | {r['failure_mode']} | {int(r['n'])} | {r['mean_ret']:+.2f}% | {r['median_dd_14d']:+.2f}% |")

    lines.append("\n## Confluence effect (n_indicators firing same day same asset)\n")
    lines.append("| indicator | confluence_n_indicators | n | mean | hit | sharpe |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for _, r in ce.iterrows():
        lines.append(f"| {r['indicator']} | {int(r['confluence_n_indicators'])} | {int(r['n_events'])} | {r['mean_pct']:+.2f}% | {r['hit_pct']:.1f}% | {r['sharpe']:+.3f} |")

    (OUT_DIR / "FULL_MINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'FULL_MINING_REPORT.md'}")
    print("\nAll outputs:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name:60s} {f.stat().st_size/1024:.1f} KB")

if __name__ == "__main__":
    main()
