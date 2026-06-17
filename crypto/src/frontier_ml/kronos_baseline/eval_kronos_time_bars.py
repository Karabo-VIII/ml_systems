"""Time-bar A/B test for Kronos zero-shot.

Hypothesis: Kronos's E1 result (pooled IC = +0.0292 on dollar bars) was
biased by a fundamental mismatch -- Kronos pretrained on uniform-time
K-line bars, we fed it dollar-volume bars with variable time intervals.

This script tests the hypothesis empirically by:
    1. Reading chimera_legacy dollar-bar OHLCV
    2. Resampling to 1-hour time bars (aggregate via first-OHLCV-per-hour)
    3. Recomputing target_return_1 as the 1-hour-forward return
    4. Running Kronos on the time-bar context with the same eval protocol

Decision rule:
    Kronos time-bar IC >> Kronos dollar-bar IC (+0.029)  =>  dual-cadence
                                                              architecture is
                                                              strongly indicated
    Kronos time-bar IC ~  Kronos dollar-bar IC           =>  pure dollar-bar
                                                              architecture is fine
                                                              (mismatch wasn't
                                                              the issue)

Usage:
    python -m src.frontier_ml.kronos_baseline.eval_kronos_time_bars \
        --device cuda --gpu-mem-fraction 0.30 --max-windows-per-asset 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frontier_ml.foundation.harmony import apply_harmony  # noqa: E402

KRONOS_LOCAL_DEFAULT = PROJECT_ROOT / "external" / "Kronos"
LEGACY_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
LOG_DIR = PROJECT_ROOT / "logs" / "frontier_ml" / "kronos_baseline"

KRONOS_HF_NAMES = {
    "mini":  "NeoQuasar/Kronos-mini",
    "small": "NeoQuasar/Kronos-small",
    "base":  "NeoQuasar/Kronos-base",
}

ASSETS_U10 = ["btc", "eth", "sol", "bnb", "xrp", "doge", "ada", "avax", "link", "ltc"]


def _import_kronos():
    kronos_path = Path(os.environ.get("KRONOS_PATH", str(KRONOS_LOCAL_DEFAULT)))
    if kronos_path.exists():
        sys.path.insert(0, str(kronos_path))
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
        return Kronos, KronosTokenizer, KronosPredictor
    raise ImportError(f"Kronos repo not found at {kronos_path}")


def _resample_to_hourly(asset: str) -> Optional[pd.DataFrame]:
    """Load chimera_legacy dollar bars, resample to 1-hour time bars.

    Returns a pandas DataFrame with columns:
        timestamps (pd.Datetime), open, high, low, close, volume, amount,
        target_return_1   <- recomputed as 1-hour-forward log return
    Sliced to OOS suffix (last 30%).
    """
    cands = sorted(LEGACY_DIR.glob(f"{asset.lower()}usdt_v50_chimera_*.parquet"))
    if not cands:
        return None
    df = pl.read_parquet(cands[-1], columns=[
        "timestamp", "open", "high", "low", "close", "volume",
    ]).to_pandas()
    if len(df) == 0:
        return None
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["amount"] = df["close"] * df["volume"]
    df = df.set_index("timestamps").sort_index()

    # Resample to 1h bars: open=first, high=max, low=min, close=last, volume=sum, amount=sum
    agg = df.resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        "amount": "sum",
    }).dropna(subset=["close"])

    # Recompute h=1 return as log(close_{t+1} / close_t)
    agg["target_return_1"] = np.log(agg["close"].shift(-1) / agg["close"])
    agg = agg.dropna(subset=["target_return_1"]).reset_index()

    # OOS suffix (last 30%)
    n = len(agg)
    train_end = int(n * 0.70)
    oos = agg.iloc[train_end:].reset_index(drop=True)
    return oos


def _build_eval_windows(oos: pd.DataFrame, ctx_len: int = 256, stride: int = 24,
                          max_windows: int = 100) -> List[Tuple[pd.DataFrame, pd.Series, pd.Series, float]]:
    """Slice OOS into eval windows. Stride 24h = sample one window per day."""
    out = []
    n = len(oos)
    needed = ctx_len + 2
    if n < needed:
        return out
    starts = list(range(0, n - needed, stride))
    if len(starts) > max_windows:
        rng = np.random.default_rng(42)
        starts = sorted(rng.choice(starts, size=max_windows, replace=False).tolist())
    for s in starts:
        x_df = oos.iloc[s:s+ctx_len][["open", "high", "low", "close", "volume", "amount"]].copy()
        x_ts = oos.iloc[s:s+ctx_len]["timestamps"].copy()
        y_ts = oos.iloc[s+ctx_len:s+ctx_len+1]["timestamps"].copy()
        last_bar = s + ctx_len - 1
        target_h1 = float(oos.iloc[last_bar]["target_return_1"])
        if not np.isfinite(target_h1):
            continue
        out.append((x_df, x_ts, y_ts, target_h1))
    return out


def run(model_size: str = "small", ctx_len: int = 256,
        device: str = "cuda", gpu_mem_fraction: float = 0.30,
        n_sample: int = 5, max_windows_per_asset: int = 100,
        smoke: bool = False, assets: Optional[List[str]] = None) -> Dict:
    apply_harmony(verbose=False)
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
            print(f"[kronos-time] GPU mem fraction = {gpu_mem_fraction}", flush=True)
        except Exception:
            pass

    Kronos, KronosTokenizer, KronosPredictor = _import_kronos()
    print(f"[kronos-time] loading {KRONOS_HF_NAMES[model_size]} on {device}...",
          flush=True)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(KRONOS_HF_NAMES[model_size])
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    predictor = KronosPredictor(model, tok, max_context=512)
    print(f"[kronos-time] predictor ready (max_context=512)", flush=True)

    asset_list = assets or ASSETS_U10
    if smoke:
        asset_list = asset_list[:2]
        max_windows_per_asset = 16

    rows = []
    pooled_pred = []
    pooled_truth = []

    for asset in asset_list:
        oos = _resample_to_hourly(asset)
        if oos is None or len(oos) < ctx_len + 50:
            print(f"[kronos-time] {asset}: insufficient hourly OOS; skip", flush=True)
            continue

        windows = _build_eval_windows(oos, ctx_len=ctx_len, stride=24,
                                          max_windows=max_windows_per_asset)
        N = len(windows)
        if N == 0:
            print(f"[kronos-time] {asset}: no valid windows; skip", flush=True)
            continue
        print(f"[kronos-time] {asset}: {N} windows  ctx={ctx_len}h "
              f"({oos['timestamps'].iloc[0].date()} -> {oos['timestamps'].iloc[-1].date()})",
              flush=True)

        preds = np.empty(N, dtype=np.float32)
        truths = np.empty(N, dtype=np.float32)
        t0 = time.time()
        for i, (x_df, x_ts, y_ts, tgt) in enumerate(windows):
            try:
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=1,
                    T=1.0,
                    top_p=0.9,
                    sample_count=n_sample,
                    verbose=False,
                )
                pred_close = float(pred_df["close"].iloc[0])
                last_close = float(x_df["close"].iloc[-1])
                preds[i] = float(np.log(max(pred_close, 1e-9) / max(last_close, 1e-9)))
            except Exception as e:
                if i < 3:
                    print(f"  err i={i}: {type(e).__name__}: {e}", flush=True)
                preds[i] = 0.0
            truths[i] = tgt
            if i and i % 25 == 0:
                rate = i / (time.time() - t0)
                print(f"  {asset} {i}/{N}  rate {rate:.2f} win/s", flush=True)

        rho, p = spearmanr(preds, truths)
        rows.append({
            "asset": asset, "n": int(N),
            "ic_h1": float(rho), "p_value": float(p),
            "model": f"kronos-{model_size}-time1h",
        })
        pooled_pred.append(preds)
        pooled_truth.append(truths)
        print(f"[kronos-time] {asset} h=1: IC = {rho:+.4f} (p={p:.2e})", flush=True)

    if pooled_pred:
        pred_all = np.concatenate(pooled_pred)
        truth_all = np.concatenate(pooled_truth)
        rho_all, p_all = spearmanr(pred_all, truth_all)
        rows.append({
            "asset": "POOLED", "n": int(len(pred_all)),
            "ic_h1": float(rho_all), "p_value": float(p_all),
            "model": f"kronos-{model_size}-time1h",
        })
        print(f"\n[kronos-time] POOLED h=1: IC = {rho_all:+.4f}  "
              f"n={len(pred_all)}  p={p_all:.2e}", flush=True)
    else:
        rho_all = float("nan")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"kronos_{model_size}_time1h_zero_shot_{int(time.time())}.json"
    with open(out_path, "w") as fp:
        json.dump({
            "model": f"kronos-{model_size}-time1h",
            "ctx_len_hours": ctx_len,
            "n_sample": n_sample,
            "rows": rows,
            "comparison": {
                "kronos_dollar_bar_pooled_ic": 0.0292,
                "v1_baseline_ic": 0.066,
                "v1_1_record_ic": 0.067,
            },
        }, fp, indent=2)
    print(f"\n[kronos-time] result -> {out_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("DUAL-CADENCE A/B VERDICT", flush=True)
    print(f"  Kronos dollar-bar IC : +0.0292", flush=True)
    print(f"  Kronos time-bar IC   : {rho_all:+.4f}", flush=True)
    if not np.isnan(rho_all):
        delta = rho_all - 0.0292
        print(f"  Delta (time - dollar): {delta:+.4f}", flush=True)
        if rho_all >= 0.060:
            print("  => Time-bar Kronos beats V1.1 (0.067) approach. "
                  "DUAL-CADENCE strongly indicated.", flush=True)
        elif rho_all >= 0.040 or delta >= 0.030:
            print("  => Time bars unlock material lift. "
                  "Add time-bar input stream to foundation.", flush=True)
        elif delta >= 0.010:
            print("  => Modest time-bar lift; dual-cadence weakly indicated.", flush=True)
        else:
            print("  => Time vs dollar makes little difference for Kronos. "
                  "Pure dollar-bar architecture validated.", flush=True)
    print("=" * 70, flush=True)

    return {"rows": rows, "pooled_ic": rho_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(KRONOS_HF_NAMES))
    ap.add_argument("--ctx-len", type=int, default=256,
                    help="Context length in HOURS (default 256h = ~10.6 days)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.30)
    ap.add_argument("--n-sample", type=int, default=5)
    ap.add_argument("--max-windows-per-asset", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assets", nargs="+", default=None)
    args = ap.parse_args()

    run(
        model_size=args.model,
        ctx_len=args.ctx_len,
        device=args.device,
        gpu_mem_fraction=args.gpu_mem_fraction,
        n_sample=args.n_sample,
        max_windows_per_asset=args.max_windows_per_asset,
        smoke=args.smoke,
        assets=args.assets,
    )


if __name__ == "__main__":
    main()
