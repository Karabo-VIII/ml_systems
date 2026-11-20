"""s3_global_lsr_test.py -- Test global_lsr as a gate on the 4h trend book.

Run: python src/strat/s3_global_lsr_test.py
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


def load_prep(sym: str, base: str, cl: ChimeraLoader, df_s3: pd.DataFrame) -> pd.DataFrame | None:
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
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["sma50"]  = df["close"].rolling(50).mean()
    df["sma50_rising"] = (df["sma50"] > df["sma50"].shift(1)).astype(float)
    df["entry_signal"] = (
        (df["close"] > df["sma50"]) & (df["sma50"] > df["sma200"]) &
        (df["sma50_rising"] > 0.5) & (df["close"] > df["sma200"])
    ).astype(int)
    df.loc[df[["sma200", "sma50", "atr14"]].isna().any(axis=1), "entry_signal"] = 0
    df["date_day"] = df["date"].dt.floor("D")
    s3 = (df_s3[df_s3["asset"] == base][["date", "global_lsr"]]
          .rename(columns={"date": "date_day"}).copy())
    s3["date_day"] = pd.to_datetime(s3["date_day"])
    df = df.merge(s3, on="date_day", how="left")
    lag = df["global_lsr"].shift(1)
    df["g_z"]     = (lag - lag.ewm(span=120, min_periods=20).mean()) / (lag.ewm(span=120, min_periods=20).std() + 1e-8)
    df["g_avail"] = lag.notna().astype(int)
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
                if lows[j] <= stop:
                    exit_bar = j; break
            hwm = max(hwm, highs[j]); j += 1
        net = closes[exit_bar] / opens[entry_fill] - 1.0 - COST_RT
        wnd = ("TRAIN" if ts < TRAIN_END else "VAL" if ts < VAL_END
               else "OOS" if ts < OOS_END else "UNSEEN")
        trades.append({"window": wnd, "net_pnl": float(net)})
        i = max(exit_bar, i + 1)
    return trades


def book_compound(per_asset: dict, window: str) -> float:
    comps = []
    for trades in per_asset.values():
        sub = [t for t in trades if t["window"] == window]
        c = float((np.prod(1.0 + np.array([t["net_pnl"] for t in sub])) - 1.0) * 100) if sub else 0.0
        comps.append(c)
    return float((np.prod([(1 + c / 100) for c in comps]) ** (1 / len(comps)) - 1) * 100) if comps else 0.0


def trainval_compound(per_asset: dict) -> float:
    comps = []
    for trades in per_asset.values():
        sub = [t for t in trades if t["window"] in ("TRAIN", "VAL")]
        c = float((np.prod(1.0 + np.array([t["net_pnl"] for t in sub])) - 1.0) * 100) if sub else 0.0
        comps.append(c)
    return float((np.prod([(1 + c / 100) for c in comps]) ** (1 / len(comps)) - 1) * 100) if comps else 0.0


if __name__ == "__main__":
    print("=== global_lsr GATE TEST (4h trend book) ===")
    print()
    cl = ChimeraLoader()
    df_s3 = pd.read_parquet(ROOT / "data" / "processed" / "panels" / "daily" / "s3_metrics_panel.parquet")
    df_s3["date"] = pd.to_datetime(df_s3["date"])

    print("Loading assets...")
    asset_data = {}
    for base, sym in U10.items():
        df = load_prep(sym, base, cl, df_s3)
        if df is not None:
            asset_data[base] = df

    # Gate strategies to test
    gate_fns = {
        "ungated":       lambda df: np.zeros(len(df), dtype=int),
        "contra_g>0.5":  lambda df: ((df["g_z"] > 0.5) & (df["g_avail"] == 1)).astype(int).values,
        "contra_g>1.0":  lambda df: ((df["g_z"] > 1.0) & (df["g_avail"] == 1)).astype(int).values,
        "contra_g>1.5":  lambda df: ((df["g_z"] > 1.5) & (df["g_avail"] == 1)).astype(int).values,
        "na_block":      lambda df: (df["g_avail"] == 0).astype(int).values,
    }

    print()
    print(f"  {'Strategy':<20} {'TRAIN%':>8} {'VAL%':>8} {'OOS%':>8} {'UNSEEN%':>9} {'TV%':>10}")
    results = {}
    for name, gate_fn in gate_fns.items():
        per_asset = {base: simulate(df, gate_fn(df)) for base, df in asset_data.items()}
        r = {w: book_compound(per_asset, w) for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
        r["TV"] = trainval_compound(per_asset)
        r["n_trades"] = {w: sum(len([t for t in v if t["window"] == w]) for v in per_asset.values())
                         for w in ("TRAIN", "VAL", "OOS", "UNSEEN")}
        results[name] = r
        print(f"  {name:<20} {r['TRAIN']:>7.1f}% {r['VAL']:>8.1f}% {r['OOS']:>8.1f}% {r['UNSEEN']:>9.1f}% {r['TV']:>10.1f}%")

    # Best on TRAIN+VAL
    best_name = max((n for n in gate_fns), key=lambda n: results[n]["TV"])
    ug_oos    = results["ungated"]["OOS"]
    best_oos  = results[best_name]["OOS"]
    ug_un     = results["ungated"]["UNSEEN"]
    best_un   = results[best_name]["UNSEEN"]
    print(f"\n  Best on TRAIN+VAL: '{best_name}'")
    print(f"  OOS:    {best_oos:.2f}% vs ungated {ug_oos:.2f}% -> delta={best_oos - ug_oos:+.2f}pp")
    print(f"  UNSEEN: {best_un:.2f}% vs ungated {ug_un:.2f}% -> delta={best_un - ug_un:+.2f}pp")

    # Shuffled null for best
    print(f"\n  Running 200-shuffle null for '{best_name}'...")
    gate_fn   = gate_fns[best_name]
    gate_arrs = {base: gate_fn(df) for base, df in asset_data.items()}
    rng = np.random.default_rng(42)
    sh_oos = []
    for _ in range(200):
        per_asset = {}
        for base, df in asset_data.items():
            arr = gate_arrs[base].copy(); rng.shuffle(arr)
            per_asset[base] = simulate(df, arr)
        sh_oos.append(book_compound(per_asset, "OOS"))
    sh_oos = np.array(sh_oos)
    p_sh = float((sh_oos >= best_oos).mean())
    timing = best_oos - float(np.mean(sh_oos))
    exp_red = float(np.mean(sh_oos)) - ug_oos
    print(f"  Shuffle OOS: mean={np.mean(sh_oos):.2f}% p50={np.percentile(sh_oos,50):.2f}% p05={np.percentile(sh_oos,5):.2f}%")
    print(f"  P(shuffle >= gated): {p_sh:.3f}")
    print(f"  Timing skill: {timing:+.2f}pp  Exposure-reduction: {exp_red:+.2f}pp")
    n_gates = len(gate_fns) - 1  # excluding ungated
    print(f"  Multiple-comparisons: {n_gates} strategies -> Bonferroni alpha={0.05/n_gates:.4f}, raw p={p_sh:.4f}")

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if best_name == "ungated":
        print("  NULL: ungated wins on TRAIN+VAL. global_lsr adds no value.")
    elif best_oos > ug_oos and p_sh < 0.05 / n_gates:
        print(f"  VALUE: '{best_name}' beats ungated OOS AND survives MC correction.")
    elif best_oos > ug_oos and p_sh < 0.1:
        print(f"  WEAK: '{best_name}' beats ungated by {best_oos-ug_oos:.2f}pp but p={p_sh:.3f} >> MC-alpha={0.05/n_gates:.4f}.")
        print("  global_lsr conditioner does NOT survive multiple-comparisons correction.")
    else:
        print(f"  NULL: '{best_name}' hurts OOS ({best_oos-ug_oos:.2f}pp). global_lsr adds no value.")

    # Save
    out_dir = ROOT / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "results": {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                        for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in results.items()},
        "best": best_name,
        "null": {"mean": float(np.mean(sh_oos)), "p05": float(np.percentile(sh_oos, 5)),
                 "p50": float(np.percentile(sh_oos, 50)), "p_sh_ge_gated": float(p_sh)},
        "timing_skill": float(timing),
        "exposure_reduction": float(exp_red),
    }
    out_path = out_dir / "s3_global_lsr_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")
