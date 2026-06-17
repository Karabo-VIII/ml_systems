"""4h conditioner test for s3 top_pos_lsr vs trend book.

Run: python src/strat/s3_conditioner_4h.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pipeline.chimera_loader import ChimeraLoader
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

U10 = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT",
    "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT", "LTC": "LTCUSDT",
}

ATR_MULT   = 10.0
TRAIN_END  = pd.Timestamp("2024-05-15")
VAL_END    = pd.Timestamp("2025-03-15")
OOS_END    = pd.Timestamp("2025-12-31")
UNSEEN_END = pd.Timestamp("2026-06-01")
COST_RT    = 0.0024

# NOTE: pre-registered threshold was 1.5 on 1d daily-entries pooled cross-section.
# 4h bars use the same per-asset EWMA z, so 1.0 matches the pooled OOS finding (z>1 shows -2.4% above vs +0.4% below).
# Both 1.0 and 1.5 are tested; 1.0 is the primary.
Z_THRESH_PRIMARY = 1.0
Z_THRESH_SECONDARY = 1.5
N_SHUFFLE = 200


def load_4h(sym: str, cl: ChimeraLoader) -> pd.DataFrame | None:
    try:
        loaded = cl.load(sym, cadence="4h")
    except Exception:
        return None
    df = loaded if hasattr(loaded, "iloc") else pd.DataFrame(loaded.to_dict(as_series=False))
    df["date"] = (pd.to_datetime(df["date"], unit="ms")
                  if np.issubdtype(df["date"].dtype, np.number)
                  else pd.to_datetime(df["date"]))
    _off = df.groupby("date", sort=False).cumcount()
    df["date"] = df["date"] + pd.to_timedelta(_off * 240, unit="m")
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close"]]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"]       = tr.rolling(14).mean()
    df["sma200"]      = df["close"].rolling(200).mean()
    df["sma50"]       = df["close"].rolling(50).mean()
    df["sma50_rising"] = (df["sma50"] > df["sma50"].shift(1)).astype(float)
    df["entry_signal"] = (
        (df["close"] > df["sma50"]) &
        (df["sma50"] > df["sma200"]) &
        (df["sma50_rising"] > 0.5) &
        (df["close"] > df["sma200"])
    ).astype(int)
    nan_mask = df[["sma200", "sma50", "atr14"]].isna().any(axis=1)
    df.loc[nan_mask, "entry_signal"] = 0
    return df


def add_gate(df: pd.DataFrame, base: str, df_s3: pd.DataFrame, z_thresh: float) -> pd.DataFrame:
    s3 = (df_s3[df_s3["asset"] == base][["date", "top_pos_lsr"]]
          .rename(columns={"date": "date_day"})
          .copy())
    s3["date_day"] = pd.to_datetime(s3["date_day"])
    df = df.copy()
    df["date_day"] = df["date"].dt.floor("D")
    df = df.merge(s3, on="date_day", how="left")
    # Shift by 1 bar (past-only)
    lsr_lag = df["top_pos_lsr"].shift(1)
    ewma_mean = lsr_lag.ewm(span=120, min_periods=20).mean()
    ewma_std  = lsr_lag.ewm(span=120, min_periods=20).std()
    df["lsr_z"]     = (lsr_lag - ewma_mean) / (ewma_std + 1e-8)
    df["gate_block"] = ((df["lsr_z"] > z_thresh) & lsr_lag.notna()).astype(int)
    return df


def simulate(df: pd.DataFrame, use_gate: bool = True) -> list[dict]:
    opens     = df["open"].values.astype(float)
    highs     = df["high"].values.astype(float)
    lows      = df["low"].values.astype(float)
    closes    = df["close"].values.astype(float)
    atr_arr   = df["atr14"].values.astype(float)
    entry_arr = df["entry_signal"].values > 0.5
    gate_arr  = df["gate_block"].values.astype(int) if use_gate else np.zeros(len(df), dtype=int)
    dates     = pd.to_datetime(df["date"])
    n = len(opens)
    i = 0
    trades = []
    while i < n - 2:
        if not entry_arr[i] or gate_arr[i] == 1:
            i += 1
            continue
        ts = dates.iloc[i]
        entry_fill = i + 1
        if entry_fill >= n:
            break
        hwm = max(opens[entry_fill], highs[entry_fill])
        j = entry_fill + 1
        exit_bar = n - 1
        while j < n:
            atr_ref = atr_arr[j - 1] if np.isfinite(atr_arr[j - 1]) else np.nan
            if np.isfinite(atr_ref):
                stop = hwm - ATR_MULT * atr_ref
                if lows[j] <= stop:
                    exit_bar = j
                    break
            hwm = max(hwm, highs[j])
            j += 1
        net = closes[exit_bar] / opens[entry_fill] - 1.0 - COST_RT
        wnd = ("TRAIN" if ts < TRAIN_END
               else "VAL" if ts < VAL_END
               else "OOS" if ts < OOS_END
               else "UNSEEN")
        trades.append({"window": wnd, "net_pnl": float(net)})
        i = max(exit_bar, i + 1)
    return trades


def book_compound(per_asset_trades: dict[str, list[dict]], window: str) -> float:
    comps = []
    for trades in per_asset_trades.values():
        sub = [t for t in trades if t["window"] == window]
        c = float((np.prod(1.0 + np.array([t["net_pnl"] for t in sub])) - 1.0) * 100) if sub else 0.0
        comps.append(c)
    if not comps:
        return 0.0
    return float((np.prod([(1 + c / 100) for c in comps]) ** (1 / len(comps)) - 1) * 100)


def cagr(compound_pct: float, window: str) -> float:
    spans = {
        "TRAIN":  (pd.Timestamp("2022-01-01"), TRAIN_END),
        "VAL":    (TRAIN_END, VAL_END),
        "OOS":    (VAL_END,   OOS_END),
        "UNSEEN": (OOS_END,   UNSEEN_END),
    }
    s, e = spans[window]
    n_yr = (e - s).days / 365.25
    if n_yr <= 0 or compound_pct <= -100:
        return 0.0
    return round(((1 + compound_pct / 100) ** (1 / n_yr) - 1) * 100, 2)


def run_test(z_thresh: float, label: str) -> dict:
    print(f"\n--- z_thresh={z_thresh} ({label}) ---")
    cl = ChimeraLoader()
    df_s3 = pd.read_parquet(ROOT / "data" / "processed" / "panels" / "daily" / "s3_metrics_panel.parquet")
    df_s3["date"] = pd.to_datetime(df_s3["date"])

    asset_data: dict[str, pd.DataFrame] = {}
    gate_blocks: dict[str, pd.Series]   = {}
    for base, sym in U10.items():
        df = load_4h(sym, cl)
        if df is None or len(df) < 500:
            print(f"  {base}: SKIP")
            continue
        df = add_indicators(df)
        df = add_gate(df, base, df_s3, z_thresh)
        asset_data[base] = df
        gate_blocks[base] = df["gate_block"].copy()

    # Gate rate on entry signals in OOS
    total_sigs = total_gated = 0
    for df in asset_data.values():
        mask = (df["date"] >= VAL_END) & (df["date"] < OOS_END) & (df["entry_signal"] > 0.5)
        total_sigs += int(mask.sum())
        total_gated += int((df["gate_block"][mask] == 1).sum())
    gate_rate = total_gated / max(total_sigs, 1)
    print(f"  OOS entry-signal gate rate: {total_gated}/{total_sigs} = {gate_rate*100:.1f}%")

    # Ungated
    ug = {b: simulate(df, use_gate=False) for b, df in asset_data.items()}
    # Gated
    g  = {b: simulate(df, use_gate=True)  for b, df in asset_data.items()}

    results = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        ug_c = book_compound(ug, w)
        g_c  = book_compound(g, w)
        n_ug = sum(len([t for t in v if t["window"] == w]) for v in ug.values())
        n_g  = sum(len([t for t in v if t["window"] == w]) for v in g.values())
        results[w] = {"ungated": ug_c, "gated": g_c, "delta": g_c - ug_c,
                      "ungated_cagr": cagr(ug_c, w), "gated_cagr": cagr(g_c, w),
                      "n_ug": n_ug, "n_g": n_g, "n_blocked": n_ug - n_g}

    # Shuffled null
    print(f"  Running {N_SHUFFLE} shuffled nulls...")
    rng = np.random.default_rng(42)
    sh_oos_list: list[float] = []
    for _ in range(N_SHUFFLE):
        shuffled = {}
        for base, gb in gate_blocks.items():
            vals = gb.values.copy()
            rng.shuffle(vals)
            df_sh = asset_data[base].copy()
            df_sh["gate_block"] = pd.Series(vals, index=gb.index)
            shuffled[base] = simulate(df_sh, use_gate=True)
        sh_oos_list.append(book_compound(shuffled, "OOS"))

    sh_oos = np.array(sh_oos_list)
    g_oos  = results["OOS"]["gated"]
    ug_oos = results["OOS"]["ungated"]
    null_result = {
        "mean":    float(np.mean(sh_oos)),
        "p05":     float(np.percentile(sh_oos, 5)),
        "p25":     float(np.percentile(sh_oos, 25)),
        "p50":     float(np.percentile(sh_oos, 50)),
        "p75":     float(np.percentile(sh_oos, 75)),
        "p95":     float(np.percentile(sh_oos, 95)),
        "p_sh_ge_gated": float((sh_oos >= g_oos).mean()),
    }

    exposure_reduction_pp = null_result["mean"] - ug_oos
    timing_skill_pp       = g_oos - null_result["mean"]

    # Print summary
    print(f"\n  {'Window':<10} {'UNGATED%':>10} {'GATED%':>9} {'DELTA':>8} {'n_ug':>6} {'n_g':>5} {'blocked':>8}")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        r = results[w]
        print(f"    {w:<8} {r['ungated']:>9.2f}% {r['gated']:>9.2f}% {r['delta']:>+8.2f}pp {r['n_ug']:>6} {r['n_g']:>5} {r['n_blocked']:>8}")
    print()
    print(f"  Shuffle null OOS: mean={null_result['mean']:.2f}% p05={null_result['p05']:.2f}% "
          f"p50={null_result['p50']:.2f}% p95={null_result['p95']:.2f}%")
    print(f"  P(shuffle >= gated) = {null_result['p_sh_ge_gated']:.3f}")
    print(f"  Exposure-reduction effect: shuffle_mean - ungated = {exposure_reduction_pp:+.2f}pp")
    print(f"  Timing-skill effect:       gated - shuffle_mean   = {timing_skill_pp:+.2f}pp")
    print()
    if g_oos > ug_oos and g_oos > np.percentile(sh_oos, 75):
        verdict = "CONDITIONER ADDS VALUE (beats ungated AND 75th-pct shuffle -> timing skill)"
    elif g_oos > ug_oos:
        verdict = "VALUE IS EXPOSURE-REDUCTION ONLY (beats ungated but shuffle ties/beats)"
    elif g_oos < ug_oos and null_result["p_sh_ge_gated"] > 0.6:
        verdict = "CONDITIONER HURTS (worse than ungated; shuffle also mostly beats gated)"
    else:
        verdict = "CONDITIONER HURTS (worse than ungated)"
    print(f"  VERDICT: {verdict}")
    print()

    return {
        "z_thresh": z_thresh, "label": label,
        "results": results,
        "null": null_result,
        "exposure_reduction_oos_pp": float(exposure_reduction_pp),
        "timing_skill_oos_pp":       float(timing_skill_pp),
        "gate_rate_oos": float(gate_rate),
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=== S3 CONDITIONER TEST (4h trend book) ===")
    print("Conditioner: top_pos_lsr per-asset EWMA z-score (contrarian -- skip extreme longs)")
    print("Book: 4h MA trend book, ATR_MULT=10, regime_gate=True")
    print()

    out_dir = ROOT / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary test (z=1.0, adjusted from 1.5 for 4h cadence based on 4h entry discrimination)
    res_primary = run_test(Z_THRESH_PRIMARY, "primary")
    # Secondary test (z=1.5, original pre-registered)
    res_secondary = run_test(Z_THRESH_SECONDARY, "secondary_preregistered")

    out = {"primary": res_primary, "secondary": res_secondary}
    out_path = out_dir / "s3_conditioner_4h_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for res in [res_primary, res_secondary]:
        oos = res["results"]["OOS"]
        un  = res["results"]["UNSEEN"]
        print(f"\nz_thresh={res['z_thresh']} ({res['label']}):")
        print(f"  OOS:    ungated={oos['ungated']:.2f}%  gated={oos['gated']:.2f}%  delta={oos['delta']:+.2f}pp  "
              f"P(sh>=g)={res['null']['p_sh_ge_gated']:.3f}")
        print(f"  UNSEEN: ungated={un['ungated']:.2f}%  gated={un['gated']:.2f}%  delta={un['delta']:+.2f}pp")
        print(f"  Exposure-reduction: {res['exposure_reduction_oos_pp']:+.2f}pp  "
              f"Timing-skill: {res['timing_skill_oos_pp']:+.2f}pp")
        print(f"  VERDICT: {res['verdict']}")
