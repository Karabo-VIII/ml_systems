"""s3_gate_sweep.py -- sweep gate directions/thresholds on TRAIN+VAL, then evaluate OOS+UNSEEN.

Purpose: determine if ANY direction of the top_pos_lsr conditioner adds value on the 4h trend book.
Pre-registration: the direction is selected on TRAIN+VAL only, then OOS/UNSEEN is evaluated once.

Run: python src/strat/s3_gate_sweep.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

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
N_SHUFFLE  = 500


def load_4h_with_gate(sym: str, base: str, cl: ChimeraLoader, df_s3: pd.DataFrame) -> pd.DataFrame | None:
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
    # Indicators
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["sma50"]  = df["close"].rolling(50).mean()
    df["sma50_rising"] = (df["sma50"] > df["sma50"].shift(1)).astype(float)
    df["entry_signal"] = (
        (df["close"] > df["sma50"]) &
        (df["sma50"]  > df["sma200"]) &
        (df["sma50_rising"] > 0.5) &
        (df["close"] > df["sma200"])
    ).astype(int)
    nan_mask = df[["sma200", "sma50", "atr14"]].isna().any(axis=1)
    df.loc[nan_mask, "entry_signal"] = 0
    # s3 merge
    df["date_day"] = df["date"].dt.floor("D")
    s3 = (df_s3[df_s3["asset"] == base][["date", "top_pos_lsr"]]
          .rename(columns={"date": "date_day"}).copy())
    s3["date_day"] = pd.to_datetime(s3["date_day"])
    df = df.merge(s3, on="date_day", how="left")
    lsr_lag = df["top_pos_lsr"].shift(1)
    ewma_m = lsr_lag.ewm(span=120, min_periods=20).mean()
    ewma_s = lsr_lag.ewm(span=120, min_periods=20).std()
    df["lsr_z"] = (lsr_lag - ewma_m) / (ewma_s + 1e-8)
    df["lsr_avail"] = lsr_lag.notna().astype(int)
    return df


def simulate(df: pd.DataFrame, gate_arr: np.ndarray) -> list[dict]:
    opens     = df["open"].values.astype(float)
    highs     = df["high"].values.astype(float)
    lows      = df["low"].values.astype(float)
    closes    = df["close"].values.astype(float)
    atr_arr   = df["atr14"].values.astype(float)
    entry_arr = df["entry_signal"].values > 0.5
    dates     = pd.to_datetime(df["date"])
    n = len(opens); i = 0; trades = []
    while i < n - 2:
        if not entry_arr[i] or gate_arr[i] == 1:
            i += 1; continue
        ts = dates.iloc[i]; entry_fill = i + 1
        if entry_fill >= n: break
        hwm = max(opens[entry_fill], highs[entry_fill])
        j = entry_fill + 1; exit_bar = n - 1
        while j < n:
            atr_ref = atr_arr[j - 1] if np.isfinite(atr_arr[j - 1]) else np.nan
            if np.isfinite(atr_ref):
                stop = hwm - ATR_MULT * atr_ref
                if lows[j] <= stop: exit_bar = j; break
            hwm = max(hwm, highs[j]); j += 1
        net = closes[exit_bar] / opens[entry_fill] - 1.0 - COST_RT
        wnd = ("TRAIN" if ts < TRAIN_END else "VAL" if ts < VAL_END
               else "OOS" if ts < OOS_END else "UNSEEN")
        trades.append({"window": wnd, "net_pnl": float(net)})
        i = max(exit_bar, i + 1)
    return trades


def book_compound(per_asset: dict[str, list[dict]], window: str) -> float:
    comps = []
    for trades in per_asset.values():
        sub = [t for t in trades if t["window"] == window]
        c = float((np.prod(1.0 + np.array([t["net_pnl"] for t in sub])) - 1.0) * 100) if sub else 0.0
        comps.append(c)
    return float((np.prod([(1 + c / 100) for c in comps]) ** (1 / len(comps)) - 1) * 100) if comps else 0.0


def cagr(pct: float, window: str) -> float:
    spans = {"TRAIN": (pd.Timestamp("2022-01-01"), TRAIN_END),
             "VAL": (TRAIN_END, VAL_END),
             "OOS": (VAL_END, OOS_END),
             "UNSEEN": (OOS_END, UNSEEN_END)}
    s, e = spans[window]; n_yr = (e - s).days / 365.25
    if n_yr <= 0 or pct <= -100: return 0.0
    return round(((1 + pct / 100) ** (1 / n_yr) - 1) * 100, 2)


def run_strategy(asset_data: dict, gate_fn) -> dict[str, float]:
    per_asset = {}
    for base, df in asset_data.items():
        gate_arr = gate_fn(df)
        per_asset[base] = simulate(df, gate_arr)
    result = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        c = book_compound(per_asset, w)
        result[w] = c
        result[f"{w}_cagr"] = cagr(c, w)
        result[f"{w}_n"] = sum(len([t for t in v if t["window"] == w]) for v in per_asset.values())
    return result


def run_shuffled_null(asset_data: dict, gate_fn_name: str, gate_fns: dict, n_shuffle: int) -> np.ndarray:
    """Shuffle the gate_block COLUMN per-asset independently, preserving gate structure/rate."""
    rng = np.random.default_rng(seed=42)
    gate_fn = gate_fns[gate_fn_name]
    # Pre-compute gate arrays per asset, then shuffle
    gate_arrs = {base: gate_fn(df) for base, df in asset_data.items()}
    sh_oos = []
    for _ in range(n_shuffle):
        per_asset = {}
        for base, df in asset_data.items():
            arr = gate_arrs[base].copy()
            rng.shuffle(arr)
            per_asset[base] = simulate(df, arr)
        sh_oos.append(book_compound(per_asset, "OOS"))
    return np.array(sh_oos)


if __name__ == "__main__":
    print("=== S3 GATE DIRECTION SWEEP (4h trend book) ===")
    print("Book: ATR_MULT=10, MA regime gate, taker 0.24% RT, U10")
    print()

    cl = ChimeraLoader()
    df_s3 = pd.read_parquet(ROOT / "data" / "processed" / "panels" / "daily" / "s3_metrics_panel.parquet")
    df_s3["date"] = pd.to_datetime(df_s3["date"])

    print("Loading 4h data with s3 gate signal...")
    asset_data = {}
    for base, sym in U10.items():
        df = load_4h_with_gate(sym, base, cl, df_s3)
        if df is not None and len(df) > 500:
            asset_data[base] = df

    # Define gate functions (all to be tested on TRAIN+VAL only)
    gate_fns = {
        "ungated":       lambda df: np.zeros(len(df), dtype=int),
        "contra_z>0.5":  lambda df: ((df["lsr_z"] > 0.5) & (df["lsr_avail"] == 1)).astype(int).values,
        "contra_z>1.0":  lambda df: ((df["lsr_z"] > 1.0) & (df["lsr_avail"] == 1)).astype(int).values,
        "contra_z>1.5":  lambda df: ((df["lsr_z"] > 1.5) & (df["lsr_avail"] == 1)).astype(int).values,
        "protrend_z<-0.5": lambda df: ((df["lsr_z"] < -0.5) & (df["lsr_avail"] == 1)).astype(int).values,
        "protrend_z<-1.0": lambda df: ((df["lsr_z"] < -1.0) & (df["lsr_avail"] == 1)).astype(int).values,
        "mid_band_only": lambda df: (((df["lsr_z"] > 1.5) | (df["lsr_z"] < -1.5)) & (df["lsr_avail"] == 1)).astype(int).values,
        "lsr_na_block":  lambda df: (df["lsr_avail"] == 0).astype(int).values,
    }

    print("\nSweep results (ALL windows shown for transparency):")
    print(f"  {'Strategy':<22} {'TRAIN%':>8} {'VAL%':>8} {'OOS%':>8} {'UNSEEN%':>9} {'TV_combined':>12}")

    results_table = {}
    for name, gate_fn in gate_fns.items():
        r = run_strategy(asset_data, gate_fn)
        # TRAIN+VAL combined compound
        trainval_comps = []
        for trades_list in [simulate(df, gate_fn(df)) for df in asset_data.values()]:
            sub = [t for t in trades_list if t["window"] in ("TRAIN", "VAL")]
            c = float((np.prod(1.0 + np.array([t["net_pnl"] for t in sub])) - 1.0) * 100) if sub else 0.0
            trainval_comps.append(c)
        tv = float((np.prod([(1 + c / 100) for c in trainval_comps]) ** (1 / len(trainval_comps)) - 1) * 100)
        r["TRAINVAL"] = tv
        results_table[name] = r
        print(f"  {name:<22} {r['TRAIN']:>8.1f}% {r['VAL']:>8.1f}% {r['OOS']:>8.1f}% {r['UNSEEN']:>9.1f}% {tv:>12.1f}%")

    # Select best strategy on TRAIN+VAL (honest selection criterion)
    best_name = max(gate_fns.keys(), key=lambda n: results_table[n]["TRAINVAL"])
    best_oos   = results_table[best_name]["OOS"]
    ungated_oos = results_table["ungated"]["OOS"]
    ungated_un  = results_table["ungated"]["UNSEEN"]
    print(f"\n  Best on TRAIN+VAL: '{best_name}' (TV={results_table[best_name]['TRAINVAL']:.1f}%)")
    print(f"  OOS: best={best_oos:.2f}%  ungated={ungated_oos:.2f}%  delta={best_oos-ungated_oos:+.2f}pp")
    print(f"  UNSEEN: best={results_table[best_name]['UNSEEN']:.2f}%  ungated={ungated_un:.2f}%  delta={results_table[best_name]['UNSEEN']-ungated_un:+.2f}pp")

    # Shuffled null on the BEST strategy
    print(f"\nRunning {N_SHUFFLE}-shuffle null on best strategy '{best_name}'...")
    sh_oos = run_shuffled_null(asset_data, best_name, gate_fns, N_SHUFFLE)
    print(f"  Shuffle null OOS: mean={np.mean(sh_oos):.2f}% p05={np.percentile(sh_oos,5):.2f}%",
          f"p50={np.percentile(sh_oos,50):.2f}% p95={np.percentile(sh_oos,95):.2f}%")
    print(f"  Gated OOS: {best_oos:.2f}%  Ungated OOS: {ungated_oos:.2f}%")
    print(f"  P(shuffle >= gated): {(sh_oos >= best_oos).mean():.3f}")
    print(f"  Exposure-reduction: {np.mean(sh_oos) - ungated_oos:+.2f}pp")
    print(f"  Timing skill:       {best_oos - np.mean(sh_oos):+.2f}pp")

    # Multiple-comparisons correction note
    n_strategies = len(gate_fns) - 1  # excluding ungated
    print(f"\n  Note: {n_strategies} gate strategies tested -> Bonferroni-corrected alpha = {0.05/n_strategies:.4f}")
    print(f"  Best OOS p-value vs shuffle: {(sh_oos >= best_oos).mean():.4f}")
    print(f"  (Multiple-comparisons inflation present; best-on-TRAIN+VAL selection is the guard)")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    oos_delta = best_oos - ungated_oos
    p_sh = float((sh_oos >= best_oos).mean())
    timing_skill_pp = best_oos - float(np.mean(sh_oos))
    exp_red_pp = float(np.mean(sh_oos)) - ungated_oos

    if best_name == "ungated":
        print("  NULL: No gate strategy beats the ungated book on TRAIN+VAL.")
        print("  top_pos_lsr as a conditioner adds NO value to the 4h trend book.")
    elif oos_delta > 0 and p_sh < 0.1 and timing_skill_pp > 0.5:
        print(f"  CONDITIONAL VALUE: '{best_name}' adds +{oos_delta:.2f}pp OOS.")
        print(f"  Timing skill = {timing_skill_pp:.2f}pp (not just exposure reduction).")
        print(f"  BUT: threshold selected on TRAIN+VAL (7 strategies tested -- multiple-comparisons risk).")
    elif oos_delta > 0:
        print(f"  WEAK: '{best_name}' adds +{oos_delta:.2f}pp OOS but P(sh>=g)={p_sh:.3f} -- exposure-reduction dominated.")
    else:
        print(f"  NULL: Best strategy '{best_name}' HURTS OOS ({oos_delta:.2f}pp).")
        print(f"  top_pos_lsr conditioner does NOT improve the 4h trend book.")

    # Save
    out_dir = ROOT / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "results": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                        for kk, vv in v.items()} for k, v in results_table.items()},
        "best_on_trainval": best_name,
        "null_oos": {
            "mean": float(np.mean(sh_oos)), "p05": float(np.percentile(sh_oos, 5)),
            "p50": float(np.percentile(sh_oos, 50)), "p95": float(np.percentile(sh_oos, 95)),
            "p_sh_ge_gated": float((sh_oos >= best_oos).mean()),
        },
        "timing_skill_oos_pp": float(timing_skill_pp),
        "exposure_reduction_oos_pp": float(exp_red_pp),
    }
    out_path = out_dir / "s3_gate_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")
