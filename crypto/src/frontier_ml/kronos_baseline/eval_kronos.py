"""Run Kronos zero-shot on chimera_legacy OOS, compute IC vs V1.x baseline.

This is E1 from FRONTIER_RESEARCH_RESPONSE_2026_05_02.md -- the gate that
determines whether Prong 1 pivots from "scratch pretrain" to "Kronos finetune."

The eval protocol:
    1. For each asset in u10:
       a. Load chimera_legacy dollar bars (OHLCV columns + target_return_h)
       b. Restrict to the OOS suffix (last 30% per asset)
       c. Sliding window of 400 context bars; predict next-bar close
       d. Convert predicted close -> predicted return at h=1
       e. Spearman IC vs target_return_1
    2. Aggregate IC per (asset) and overall pooled

Decision rule (per browser response R1):
    Kronos-small IC >= 0.060 zero-shot  -> PIVOT (finetune Kronos)
    Kronos-small IC >= 0.080 zero-shot  -> SHIP zero-shot directly
    Kronos-small IC <  0.030            -> stay on plan (pretrain from scratch)

Run:
    # Smoke (cheap; works alongside V1 training):
    python -m src.frontier_ml.kronos_baseline.eval_kronos --smoke

    # Full eval (after V1 training completes; uses GPU):
    python -m src.frontier_ml.kronos_baseline.eval_kronos --device cuda

Resource hygiene:
    - GPU memory fraction defaults to 0.30 to stay under V1 training's allocation
    - Override with --gpu-mem-fraction 0.85 when GPU is free
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
    "large": "NeoQuasar/Kronos-large",
}

ASSETS_U10 = ["btc", "eth", "sol", "bnb", "xrp", "doge", "ada", "avax", "link", "ltc"]


def _import_kronos():
    """Import Kronos / KronosTokenizer / KronosPredictor from local repo."""
    kronos_path = Path(os.environ.get("KRONOS_PATH", str(KRONOS_LOCAL_DEFAULT)))
    if kronos_path.exists():
        sys.path.insert(0, str(kronos_path))
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
            return Kronos, KronosTokenizer, KronosPredictor
        except Exception as e:
            raise ImportError(
                f"Kronos repo at {kronos_path} but import failed: {type(e).__name__}: {e}"
            )
    raise ImportError(
        f"Kronos local clone not found at {kronos_path}. "
        f"Run `git clone https://github.com/shiyu-coder/Kronos.git` into "
        f"{kronos_path.parent}."
    )


def _load_asset_oos(asset: str) -> Optional[pd.DataFrame]:
    """Load asset OHLCV from chimera_legacy, return OOS suffix as pandas df."""
    cands = sorted(LEGACY_DIR.glob(f"{asset.lower()}usdt_v50_chimera_*.parquet"))
    if not cands:
        return None
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "target_return_1"]
    df = pl.read_parquet(cands[-1], columns=cols).to_pandas()
    n = len(df)
    train_end = int(n * 0.70)
    oos = df.iloc[train_end:].reset_index(drop=True)
    # Synthesize 'amount' column = price*volume (Kronos expects 6 OHLCVA)
    oos["amount"] = oos["close"] * oos["volume"]
    # Convert ts ms -> pandas datetime
    oos["timestamps"] = pd.to_datetime(oos["timestamp"], unit="ms")
    return oos


def _build_eval_windows(oos: pd.DataFrame, ctx_len: int = 400, stride: int = 100,
                          max_windows: int = 100) -> List[Tuple[pd.DataFrame, pd.Series, pd.Series, float]]:
    """Slice OOS into eval windows for Kronos.

    Each window is a tuple of:
        (x_df, x_timestamp, y_timestamp, target_h1)

    Returns at most `max_windows` to keep runtime tractable.
    """
    out = []
    n = len(oos)
    needed = ctx_len + 2  # context + 1 prediction bar + 1 to read next-bar return
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
        # target: realized 1-bar return AT bar s+ctx_len-1 (the bar AFTER context end)
        # i.e., target_return_1 at the LAST bar of context
        last_bar_idx = s + ctx_len - 1
        target_h1 = float(oos.iloc[last_bar_idx]["target_return_1"])
        out.append((x_df, x_ts, y_ts, target_h1))
    return out


def run_kronos_zero_shot(
    model_size: str = "small",
    ctx_len: int = 400,
    device: str = "cuda",
    gpu_mem_fraction: float = 0.30,
    n_sample: int = 5,
    smoke: bool = False,
    assets: List[str] = None,
    max_windows_per_asset: int = 100,
) -> Dict:
    """Run Kronos zero-shot on chimera_legacy OOS for u10 assets."""
    apply_harmony(verbose=False)
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
            print(f"[kronos] GPU memory fraction capped at {gpu_mem_fraction} "
                  f"(V1 training in flight; staying out of its way)", flush=True)
        except Exception:
            pass

    Kronos, KronosTokenizer, KronosPredictor = _import_kronos()

    print(f"[kronos] loading {KRONOS_HF_NAMES[model_size]} on {device}...", flush=True)
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(KRONOS_HF_NAMES[model_size])
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    predictor = KronosPredictor(model, tok, max_context=512)
    print(f"[kronos] predictor ready (max_context=512)", flush=True)

    asset_list = assets or ASSETS_U10
    if smoke:
        asset_list = asset_list[:2]
        max_windows_per_asset = 16

    rows = []
    pooled_pred = []
    pooled_truth = []

    for asset in asset_list:
        oos = _load_asset_oos(asset)
        if oos is None or len(oos) < ctx_len + 50:
            print(f"[kronos] {asset}: insufficient OOS ({0 if oos is None else len(oos)}); skip",
                  flush=True)
            continue

        windows = _build_eval_windows(oos, ctx_len=ctx_len, stride=200,
                                          max_windows=max_windows_per_asset)
        N = len(windows)
        if N == 0:
            print(f"[kronos] {asset}: no valid windows; skip", flush=True)
            continue
        print(f"[kronos] {asset}: {N} windows  ctx={ctx_len}", flush=True)

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
                # pred_df has 'close' column for the predicted bar
                pred_close = float(pred_df["close"].iloc[0])
                last_close = float(x_df["close"].iloc[-1])
                preds[i] = (pred_close - last_close) / max(last_close, 1e-9)
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
            "asset": asset,
            "n": int(N),
            "ic_h1": float(rho),
            "p_value": float(p),
            "model": f"kronos-{model_size}",
        })
        pooled_pred.append(preds)
        pooled_truth.append(truths)
        print(f"[kronos] {asset} h=1: IC = {rho:+.4f} (p={p:.2e})", flush=True)

    if pooled_pred:
        pred_all = np.concatenate(pooled_pred)
        truth_all = np.concatenate(pooled_truth)
        rho_all, p_all = spearmanr(pred_all, truth_all)
        rows.append({
            "asset": "POOLED", "n": int(len(pred_all)),
            "ic_h1": float(rho_all), "p_value": float(p_all),
            "model": f"kronos-{model_size}",
        })
        print(f"\n[kronos] POOLED h=1: IC = {rho_all:+.4f}  n={len(pred_all)}", flush=True)
    else:
        rho_all = float("nan")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"kronos_{model_size}_zero_shot_{int(time.time())}.json"
    with open(out_path, "w") as fp:
        json.dump({
            "model": f"kronos-{model_size}",
            "ctx_len": ctx_len,
            "n_sample": n_sample,
            "rows": rows,
            "v1_baseline_ic": 0.066,
            "v1_1_record_ic": 0.067,
            "headline_target_ic": 0.10,
            "decision_rule": {
                "ge_0.080": "SHIP zero-shot",
                "ge_0.060": "PIVOT to Kronos finetune",
                "ge_0.030": "stack stays on plan",
                "lt_0.030": "Kronos doesn't help; stack stays on plan",
            },
        }, fp, indent=2)
    print(f"\n[kronos] result -> {out_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"DECISION  pooled IC = {rho_all:+.4f}", flush=True)
    if not np.isnan(rho_all):
        if rho_all >= 0.080:
            print("  => SHIP zero-shot Kronos as foundation.", flush=True)
        elif rho_all >= 0.060:
            print("  => PIVOT Prong 1: finetune Kronos instead of scratch pretrain.",
                  flush=True)
        elif rho_all >= 0.030:
            print("  => Mixed; consider Kronos-base finetune.", flush=True)
        else:
            print("  => Kronos does not help on our data; stay on scratch pretrain.",
                  flush=True)
    print("=" * 70, flush=True)

    return {"rows": rows, "pooled_ic": rho_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(KRONOS_HF_NAMES))
    ap.add_argument("--ctx-len", type=int, default=400)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.30)
    ap.add_argument("--n-sample", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="2 assets, 16 windows each. CPU-friendly.")
    ap.add_argument("--max-windows-per-asset", type=int, default=100)
    ap.add_argument("--assets", nargs="+", default=None)
    args = ap.parse_args()

    run_kronos_zero_shot(
        model_size=args.model,
        ctx_len=args.ctx_len,
        device=args.device,
        gpu_mem_fraction=args.gpu_mem_fraction,
        n_sample=args.n_sample,
        smoke=args.smoke,
        assets=args.assets,
        max_windows_per_asset=args.max_windows_per_asset,
    )


if __name__ == "__main__":
    main()
