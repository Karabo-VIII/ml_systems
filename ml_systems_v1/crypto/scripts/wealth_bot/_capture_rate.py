"""Capture-rate helper: the L2 KPI of the Layered Strategy Decomposition.

For each fired signal:
  signal_valid_start = bar where rule + filter first co-fire (entry moment)
  signal_valid_end   = bar where signal decays (rule flips OR filter fails OR fwd_bars elapsed)
  available_move     = max(close[t] for t in [signal_valid_start, signal_valid_end])
                       - close[signal_valid_start]
  realized_move      = close[exit_idx] - close[entry_idx]
  capture_rate       = realized_move / available_move  if available_move > 0 else NaN

Interpretation:
  >= 0.80 : L2 near-optimal -- refine L1/L4
  0.40-0.80: L2 has headroom -- trailing stop / signal-flip-exit could lift
  <  0.40 : L2 bottleneck -- wrong fwd_bars or deferred-entry
  <  0    : L2 broken -- entering AFTER best moments

Spec: docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md §Layered Strategy Decomposition
     + runs/oracle/LAYER_DECOMPOSITION_TEMPLATE.md
"""
from __future__ import annotations

__contract__ = {
    "kind": "capture_rate_decomposer",
    "owner": "wealth_bot/capture_rate (L2 KPI canonical)",
    "purpose": "Per-trade capture rate + aggregate statistics per Layered Strategy Decomposition L2",
}

from typing import Iterable
import math

import numpy as np


def capture_rate_per_trade(
    entry_idx: int,
    exit_idx: int,
    signal_valid_start: int,
    signal_valid_end: int,
    closes: np.ndarray,
) -> tuple[float, float, float]:
    """Compute L2 capture rate for a single trade.

    Returns: (capture_rate, realized_move_abs, available_move_abs)

    Edge cases:
      - signal_valid_end > len(closes) - 1: clamped to last available bar
      - available_move <= 0: returns NaN capture_rate (no upward move available)
    """
    n = len(closes)
    s = max(0, int(signal_valid_start))
    e = min(n - 1, int(signal_valid_end))
    if s >= e:
        return (float("nan"), 0.0, 0.0)
    entry_close = float(closes[int(entry_idx)])
    exit_close = float(closes[int(exit_idx)])
    realized_move = exit_close - entry_close
    window = closes[s:e + 1]
    available_move = float(window.max() - closes[s])
    if available_move <= 0:
        return (float("nan"), realized_move, available_move)
    return (realized_move / available_move, realized_move, available_move)


def aggregate_capture_rates(rates: Iterable[float]) -> dict:
    """Aggregate per-trade capture rates -> summary stats + interpretation."""
    arr = np.array([r for r in rates if not math.isnan(r)])
    if len(arr) == 0:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"),
            "p05": float("nan"), "p95": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "interpretation": "no_valid_trades",
        }
    mean = float(arr.mean())
    if mean >= 0.80:
        interp = "L2 near-optimal (>=0.80) -- refine L1 or L4"
    elif mean >= 0.40:
        interp = "L2 has headroom (0.40-0.80) -- trailing stop / signal-flip-exit could lift"
    elif mean >= 0:
        interp = "L2 bottleneck (<0.40) -- wrong fwd_bars or deferred entries"
    else:
        interp = "L2 broken (mean negative) -- entries AFTER best moments; signal-decay handling broken"
    return {
        "n": int(len(arr)),
        "mean": mean,
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "interpretation": interp,
    }


def decay_aware_signal_valid_end(
    entry_idx: int,
    raw_signal: np.ndarray,
    fwd_bars: int,
    n_bars: int,
) -> int:
    """Find the signal_valid_end via 'first bar where raw signal flips back'.

    Falls back to entry_idx + fwd_bars if signal never flips.
    """
    n = len(raw_signal)
    cap = min(n - 1, entry_idx + fwd_bars * 4)  # search up to 4x fwd_bars
    for t in range(entry_idx + 1, cap + 1):
        if raw_signal[t] == 0:
            return t - 1   # last bar where signal was still valid
    # fallback: fixed-bar definition
    return min(n - 1, entry_idx + fwd_bars)


def fixed_bar_signal_valid_end(entry_idx: int, fwd_bars: int, n_bars: int) -> int:
    """Fixed-bar definition: signal_valid_end = entry + fwd_bars."""
    return min(n_bars - 1, entry_idx + fwd_bars)


# ---------- Level 1 (within-TI family) capture-rate sweep ----------
#
# Per framework r5 §Capture Hierarchy: Level 1 = best WITHIN-TI config could
# have achieved from the same entry bar. NOT cross-indicator. The sweep takes
# a TI-family rule generator + parameter grid and returns the max-capture
# config + its capture for each fire.

def within_ti_family_ceiling(
    closes: np.ndarray,
    fire_bars: list[int],
    fwd_bars: int,
    family_signal_generator,
    param_grid: list[dict],
) -> list[dict]:
    """For each fire_bar, compute the max-capture achievable across all configs
    in the within-TI family parameter grid.

    Args:
      closes: full asset closes array
      fire_bars: bar indices where ANY config in the family fired (union of fire sets)
      fwd_bars: hold horizon
      family_signal_generator: callable(closes, **params) -> np.ndarray binary signal
          MUST return the family rule's bar-by-bar signal for the given params
      param_grid: list of dicts, each a param combo (e.g. [{"fast": 7, "slow": 15, "ma": "EMA"}, ...])

    Returns: list of dicts, one per fire_bar, each with:
      - entry_idx
      - best_config: param dict that achieved highest capture from this entry
      - best_realized_move: realized move under best_config (entry_close -> best_exit_close)
      - best_capture_rate: best_realized_move / (its own signal-valid available_move)
      - within_family_best_available: max(close in [entry, entry+fwd_bars])
                                       - close[entry]
      - L1_ceiling: same as within_family_best_available; the denominator for OUR-config L1 ratio

    Notes:
      - "Best" = max realized move under that config's own signal-valid window
      - The signal-valid window is config-specific (different (fast, slow) flip at different bars)
      - For the purposes of L1 vs OUR-config ratio, the user computes:
          L1_ratio_for_our_config = our_realized_move / L1_ceiling
        where L1_ceiling = max realized move achievable by ANY config from this entry
      - This sweeps WITHIN-TI only; never cross-indicator-family
    """
    rows = []
    for entry_idx in fire_bars:
        entry_close = float(closes[entry_idx])
        end_idx = min(len(closes) - 1, entry_idx + fwd_bars)
        # The "within-family best available" is the max close in [entry, entry+fwd_bars]
        # - close[entry]. This is the same as available_move for any config that holds
        # the full fwd_bars window starting from entry.
        window_max = float(closes[entry_idx:end_idx + 1].max())
        within_family_best_available = window_max - entry_close

        best_realized = float("-inf")
        best_config = None
        for params in param_grid:
            # Generate this config's signal across the whole price path
            try:
                sig = family_signal_generator(closes, **params)
            except Exception:
                continue
            # Find this config's exit: first bar after entry where signal flips OR fwd_bars elapsed
            exit_idx = end_idx  # default: fwd_bars elapsed
            for t in range(entry_idx + 1, end_idx + 1):
                if sig[t] == 0:
                    exit_idx = t - 1
                    break
            realized = float(closes[exit_idx]) - entry_close
            if realized > best_realized:
                best_realized = realized
                best_config = params

        if best_realized == float("-inf"):
            best_realized = 0.0
            best_config = {}

        if within_family_best_available > 0:
            best_capture_rate = best_realized / within_family_best_available
        else:
            best_capture_rate = float("nan")

        rows.append({
            "entry_idx": entry_idx,
            "entry_close": entry_close,
            "fwd_bars": fwd_bars,
            "best_config": best_config,
            "best_realized_move": best_realized,
            "within_family_best_available": within_family_best_available,
            "best_capture_rate": best_capture_rate,
            "L1_ceiling": within_family_best_available,
        })
    return rows


def compute_our_config_l1_ratio(
    our_realized_moves: list[float],
    l1_ceilings: list[float],
) -> dict:
    """Given OUR-config's realized moves per fire + the L1 within-TI ceilings,
    compute the L1 capture ratio (how much of within-family-best we captured).

    L1_ratio_per_fire = our_realized / L1_ceiling   (if L1_ceiling > 0 else NaN)

    Returns aggregate stats + per-fire detail.
    """
    rows = []
    ratios = []
    for our, ceil in zip(our_realized_moves, l1_ceilings):
        if ceil > 0:
            r = our / ceil
            ratios.append(r)
        else:
            r = float("nan")
        rows.append({"our_realized": our, "L1_ceiling": ceil, "L1_ratio": r})
    arr = np.array([x for x in ratios if not (isinstance(x, float) and math.isnan(x))])
    if len(arr) == 0:
        agg = {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p05": float("nan"), "p95": float("nan"), "interpretation": "no_valid_fires"}
    else:
        mean = float(arr.mean())
        if mean >= 0.50:
            interp = "L1 ratio >= 0.50 -- our config is roughly best-in-family; minor param-tweak headroom"
        elif mean >= 0.30:
            interp = "L1 ratio 0.30-0.50 -- our config underperforms within-TI best; Phase 3 within-TI param/filter sweep recommended"
        else:
            interp = "L1 ratio < 0.30 -- WRONG CONFIG within TI; Phase 3 within-TI expansion is high-EV"
        agg = {
            "n": int(len(arr)), "mean": mean, "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
            "interpretation": interp,
        }
    return {"per_fire": rows, "summary": agg}


# ---------- CLI for offline computation ----------
if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path
    import polars as pl

    ap = argparse.ArgumentParser(description="Compute L2 capture rate for a fired-trade list.")
    ap.add_argument("--chimera", required=True, help="Path to chimera parquet")
    ap.add_argument("--trades-json", required=True,
                    help="Path to JSON with list of trades. Each trade has fields: "
                         "entry_idx, exit_idx, signal_valid_start (optional), signal_valid_end (optional). "
                         "If signal_valid_start missing, defaults to entry_idx. "
                         "If signal_valid_end missing, defaults to entry_idx + fwd_bars.")
    ap.add_argument("--fwd-bars", type=int, default=7, help="Default fwd_bars for fixed-bar valid-end")
    ap.add_argument("--decay-mode", choices=["fixed", "signal_flip"], default="fixed",
                    help="How to define signal_valid_end")
    ap.add_argument("--raw-signal-path", help="(if decay-mode=signal_flip) path to JSON with raw_signal array")
    args = ap.parse_args()

    df = pl.read_parquet(args.chimera, columns=["timestamp", "close"]).to_pandas()
    closes = df["close"].values
    trades = json.loads(Path(args.trades_json).read_text())

    raw_signal = None
    if args.decay_mode == "signal_flip" and args.raw_signal_path:
        raw_signal = np.array(json.loads(Path(args.raw_signal_path).read_text()))

    rates = []
    for t in trades:
        entry_idx = int(t["entry_idx"])
        exit_idx = int(t["exit_idx"])
        s = int(t.get("signal_valid_start", entry_idx))
        if "signal_valid_end" in t:
            e = int(t["signal_valid_end"])
        elif args.decay_mode == "signal_flip" and raw_signal is not None:
            e = decay_aware_signal_valid_end(entry_idx, raw_signal, args.fwd_bars, len(closes))
        else:
            e = fixed_bar_signal_valid_end(entry_idx, args.fwd_bars, len(closes))
        cr, rm, am = capture_rate_per_trade(entry_idx, exit_idx, s, e, closes)
        rates.append({"entry_idx": entry_idx, "exit_idx": exit_idx,
                       "signal_valid_start": s, "signal_valid_end": e,
                       "realized_move": rm, "available_move": am,
                       "capture_rate": None if math.isnan(cr) else cr})

    summary = aggregate_capture_rates([r["capture_rate"] for r in rates
                                         if r["capture_rate"] is not None])
    print(json.dumps({"per_trade": rates, "summary": summary}, indent=2, default=str))
    sys.exit(0)
