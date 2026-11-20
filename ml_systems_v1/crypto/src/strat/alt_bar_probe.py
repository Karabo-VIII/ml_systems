"""Alt-bar-type probe: does a simple MA trend/breakout SETUP capture BTC moves
differently (better net, different structure) on DIB / Range bars vs the
dollar/time-bar baseline?

DESIGN (sealed honest gate, not a hunt):
- Bar types tested: dollar (v51 chimera), 1d time, 4h time, DIB, Range.
- Setup: WMA crossover (fast/slow) -- the same design tested in the MA campaign.
- Target: held-out COMPOUND return (TRAIN/VAL/OOS/UNSEEN split).
- Gate: cost-matched random-ENTRY null (taker 24bps RT, same trade count / hold
  distribution), using the firewall.random_entry_null apparatus.
- UNSEEN is kept sealed -- we report it but do not optimize on it.
- NO optimization across bar types or params: we pick ONE canonical config
  (WMA-10/30 or WMA-50/200 equiv normalized to bar count) and run it.
  We derive "equivalent" lookbacks by matching the wall-clock horizon:
    4h bar WMA-10 = 40h; on DIB (388 bars/day) = ~647 bars for 40h;
    but DIB bars are variable so we instead use fraction-of-universe approach:
    Use WMA lookbacks proportional to bars-per-day ratio.
- The probe answers: does the bar geometry CHANGE the de-risked-beta verdict?

HARVESTABILITY FRAMING:
- We check: is compound_OOS/UNSEEN positive vs a random-entry null?
- We do NOT claim edge until the full battery passes (that would require the
  robustness battery). This probe's verdict is limited to:
  (a) direction of the effect: alt bars better / worse / same as time bars
  (b) whether the null is beaten on held-out (the primary harvestability test)
  (c) the bar-type geometry: bar duration distribution, IID assumptions

BINDING CONSTRAINTS: long-only, spot, lev=1, taker cost 24bps RT.
"""
from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "strat"))
sys.path.insert(0, str(ROOT / "src" / "wealth_bot"))

from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, wma_past_only
from strat.firewall import random_entry_null

# ---- WINDOWS (sealed OOS/UNSEEN boundaries) ---
WIN = WindowSpec(
    train_end="2024-05-15",
    val_end="2025-03-15",
    oos_end="2025-12-31",
    unseen_end="2026-05-22",
)

TAKER_COST_RT = 0.0024   # spot taker 24bps RT (binding)


# ---- BAR LOADERS ----

def _to_datetime(col):
    """Convert date/timestamp column to pandas DatetimeIndex."""
    if col.dtype.kind in ("i", "u"):
        return pd.to_datetime(col, unit="ms")
    return pd.to_datetime(col)


def _prep_df(raw: dict, ts_col: str = "date") -> pd.DataFrame:
    """Common dataframe preparation: OHLC + date."""
    dt = _to_datetime(np.asarray(raw[ts_col]))
    df = pd.DataFrame({
        "date": dt,
        "open": np.asarray(raw["open"], float),
        "high": np.asarray(raw.get("high", raw["close"]), float),
        "low": np.asarray(raw.get("low", raw["close"]), float),
        "close": np.asarray(raw["close"], float),
    })
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_chimera_df(asset: str, cadence: str) -> pd.DataFrame:
    """Load a time/dollar-bar chimera via ChimeraLoader."""
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset, cadence=cadence)
    raw = g.to_dict(as_series=False)
    ts_col = "date" if "date" in raw else "timestamp"
    return _prep_df(raw, ts_col=ts_col)


def load_dib_chimera_df(asset: str) -> pd.DataFrame:
    """Load BTC/ETH DIB chimera from processed/chimera/dib/."""
    sym = asset.lower() if asset.lower().endswith("usdt") else asset.lower() + "usdt"
    dib_dir = ROOT / "data" / "processed" / "chimera" / "dib"
    files = sorted(dib_dir.glob(f"{sym}_v51_chimera_dib_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No DIB chimera for {sym} in {dib_dir}")
    import polars as pl
    raw_pl = pl.read_parquet(files[-1])
    raw = raw_pl.to_dict(as_series=False)
    # DIB uses epoch-ms timestamp
    ts_col = "timestamp" if "timestamp" in raw else "date"
    return _prep_df(raw, ts_col=ts_col)


def load_range_chimera_df(asset: str) -> pd.DataFrame:
    """Load BTC/ETH range chimera from processed/chimera/range/."""
    sym = asset.lower() if asset.lower().endswith("usdt") else asset.lower() + "usdt"
    rng_dir = ROOT / "data" / "processed" / "chimera" / "range"
    files = sorted(rng_dir.glob(f"{sym}_v51_chimera_range_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No range chimera for {sym} in {rng_dir}")
    import polars as pl
    raw_pl = pl.read_parquet(files[-1])
    raw = raw_pl.to_dict(as_series=False)
    ts_col = "timestamp" if "timestamp" in raw else "date"
    return _prep_df(raw, ts_col=ts_col)


# ---- COARSEN + INDICATOR ----

def coarsen(df: pd.DataFrame, target_bars: int) -> pd.DataFrame:
    """Reduce a high-freq df to ~target_bars via groupby-step aggregation."""
    n = len(df)
    step = max(1, n // target_bars)
    df = df.copy()
    df["grp"] = np.arange(n) // step
    a = df.groupby("grp").agg(
        date=("date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index(drop=True)
    return a


def add_wma(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Add WMA fast/slow past-only indicators."""
    df = df.copy()
    df["wma_fast"] = wma_past_only(df["close"], fast)
    df["wma_slow"] = wma_past_only(df["close"], slow)
    return df


# ---- COARSEN STRATEGY PER BAR TYPE ----
# Reference: 4h-time-bar WMA-10/30 = canonical baseline from the MA campaign.
# Approximate bars per day:
#   4h time bar: 6 bars/day
#   1d time bar: 1 bar/day
#   dollar bar: ~670 bars/day (based on 2.7M bars / ~4000 days)
#   DIB bar: ~388 bars/day (median ~222s duration)
#   range bar: ~137K bars / ~2300 days = ~60 bars/day
#
# We target the SAME wall-clock horizon for fast/slow:
#   fast = 40h, slow = 120h (WMA-10@4h, WMA-30@4h)
#   Bars-equivalent:
#     4h:     fast=10,    slow=30
#     1d:     fast=2,     slow=5
#     dollar: fast=1117,  slow=3350   (too many; use coarsened at ~target_bars)
#     DIB:    fast=647,   slow=1940   (coarsen to 6676 = dollar coarse level)
#     range:  fast=100,   slow=300    (coarsen to 6676)
#
# SIMPLIFICATION for the probe: coarsen ALL bar types to ~6676 bars TOTAL
# (the same target as the dollar-bar canonical) then use WMA-10/30 throughout.
# This ensures the COARSENING STEP is the same for all -- what changes is the
# aggregation path (how bars sample price action).

TARGET_BARS = 6676   # canonical coarse level (matches dollar bar discover.py)
WMA_FAST, WMA_SLOW = 10, 30
MAX_HOLD = 18   # bars in coarsened space


def run_one(df: pd.DataFrame, label: str, max_hold: int = MAX_HOLD,
            n_books: int = 300) -> dict:
    """Run WMA-10/30 crossover on df (already coarsened), honest taker cost, firewall."""
    df = add_wma(df, WMA_FAST, WMA_SLOW)
    # Drop NaN head from WMA computation
    df = df.dropna(subset=["wma_fast", "wma_slow"]).reset_index(drop=True)

    spec = StrategySpec(
        fast_col="wma_fast",
        slow_col="wma_slow",
        signal="crossover",
        filter_col=None,
        exit_policy="signal_flip_or_filter",
        cost_rt=TAKER_COST_RT,
        use_funding=False,
        max_hold_bars=max_hold,
        max_hold_ext_bars=max_hold * 3,
    )

    h = CanonicalHarness(df, spec, WIN, chimera_path=f"alt_bar_probe::{label}")
    res = h.run()
    comps = {w: res.window_stats[w].compound_pct for w in h.WINDOWS}
    n_trades = {w: res.window_stats[w].n_trades for w in h.WINDOWS}
    dds = {w: res.window_stats[w].max_dd_pct for w in h.WINDOWS}

    # Firewall: cost-matched random-entry null on held-out windows
    fw = random_entry_null(h, n_books=n_books, seed=42)
    beats_held = bool(fw.get("beats_held"))
    per_win = fw.get("per_window", {})  # the keyed dict: per_window["OOS"], per_window["UNSEEN"]

    print(f"\n[{label}] bars_total={len(df)}")
    print(f"  COMPOUND (%): TRAIN={comps['TRAIN']:.1f}  VAL={comps['VAL']:.1f}  OOS={comps['OOS']:.1f}  UNSEEN={comps['UNSEEN']:.1f}")
    print(f"  N_TRADES:     TRAIN={n_trades['TRAIN']}  VAL={n_trades['VAL']}  OOS={n_trades['OOS']}  UNSEEN={n_trades['UNSEEN']}")
    print(f"  MAX_DD (%):   TRAIN={dds['TRAIN']:.1f}  VAL={dds['VAL']:.1f}  OOS={dds['OOS']:.1f}  UNSEEN={dds['UNSEEN']:.1f}")

    # Firewall per-window (correct sub-key)
    for w in ["OOS", "UNSEEN"]:
        fw_w = per_win.get(w, {})
        if fw_w:
            real_c = fw_w.get("real", float("nan"))
            null_p50 = fw_w.get("null_p50")
            null_p95 = fw_w.get("null_p95")
            b = fw_w.get("beats_null")
            real_s = f"{real_c:.1f}" if isinstance(real_c, (int, float)) else str(real_c)
            p50_s = f"{null_p50:.1f}" if isinstance(null_p50, (int, float)) else str(null_p50)
            p95_s = f"{null_p95:.1f}" if isinstance(null_p95, (int, float)) else str(null_p95)
            print(f"  FIREWALL {w}: real={real_s}% | null_p50={p50_s}% null_p95={p95_s}% | beats_p95={b}")
        else:
            print(f"  FIREWALL {w}: no data")

    print(f"  FIREWALL beats_held_out (OOS+UNSEEN both > null_p95): {beats_held}")
    print(f"  FIREWALL verdict: {fw.get('verdict','?')}")

    return {
        "label": label,
        "n_bars": len(df),
        "comps": {w: round(comps[w], 2) for w in h.WINDOWS},
        "n_trades": {w: int(n_trades[w]) for w in h.WINDOWS},
        "max_dd": {w: round(dds[w], 2) for w in h.WINDOWS},
        "beats_held_out": beats_held,
        "firewall_per_window": per_win,
        "firewall_verdict": fw.get("verdict", "?"),
        "all_4_positive": res.all_4_positive,
    }


# ---- MAIN ----

def main():
    asset = "BTC"
    print(f"Alt-bar probe: {asset}  WMA-{WMA_FAST}/{WMA_SLOW}  taker {TAKER_COST_RT*10000:.0f}bps RT  target_bars={TARGET_BARS}")
    print("=" * 70)

    results = []

    # 1. Dollar bar (canonical baseline)
    print("\n[Loading] dollar bars (BTC v51 chimera)...")
    try:
        df_dollar = load_chimera_df("BTC", cadence="dollar")
        df_dollar_c = coarsen(df_dollar, TARGET_BARS)
        r_dollar = run_one(df_dollar_c, "dollar_bar")
        results.append(r_dollar)
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. 4h time bar
    print("\n[Loading] 4h time bars (BTC v51 chimera)...")
    try:
        df_4h = load_chimera_df("BTC", cadence="4h")
        # 4h chimera has far fewer bars; still coarsen to standard level
        df_4h_c = coarsen(df_4h, TARGET_BARS)
        r_4h = run_one(df_4h_c, "4h_time_bar")
        results.append(r_4h)
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3. 1d time bar
    print("\n[Loading] 1d time bars (BTC v51 chimera)...")
    try:
        df_1d = load_chimera_df("BTC", cadence="1d")
        df_1d_c = coarsen(df_1d, TARGET_BARS)
        r_1d = run_one(df_1d_c, "1d_time_bar")
        results.append(r_1d)
    except Exception as e:
        print(f"  FAILED: {e}")

    # 4. DIB (Dollar Imbalance Bars)
    print("\n[Loading] DIB chimera (BTC)...")
    try:
        df_dib = load_dib_chimera_df("BTC")
        df_dib_c = coarsen(df_dib, TARGET_BARS)
        r_dib = run_one(df_dib_c, "dib_bar")
        results.append(r_dib)
    except Exception as e:
        print(f"  FAILED: {e}")

    # 5. Range bars
    print("\n[Loading] Range chimera (BTC)...")
    try:
        df_range = load_range_chimera_df("BTC")
        df_range_c = coarsen(df_range, TARGET_BARS)
        r_range = run_one(df_range_c, "range_bar")
        results.append(r_range)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print(f"{'Bar type':<18}  {'TRAIN%':>7} {'VAL%':>7} {'OOS%':>7} {'UNS%':>7}  {'beats_held':>10}  {'all4pos':>7}")
    print("-" * 70)
    for r in results:
        c = r["comps"]
        print(f"{r['label']:<18}  {c.get('TRAIN',0):>7.1f} {c.get('VAL',0):>7.1f} {c.get('OOS',0):>7.1f} {c.get('UNSEEN',0):>7.1f}  {str(r['beats_held_out']):>10}  {str(r['all_4_positive']):>7}")

    print()
    print("VERDICT FRAMING:")
    print("  - If alt-bar beats_held_out=False: same de-risked-beta verdict, bar geometry irrelevant to harvestability.")
    print("  - If alt-bar beats_held_out=True AND time bars beat_held=False: alt-bar geometry changes the answer.")
    print("  - In any case: discrimination != harvestability. Full battery required before SHIP.")

    # Save JSON for audit
    out_path = ROOT / "runs" / "alt_bar_probe_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
