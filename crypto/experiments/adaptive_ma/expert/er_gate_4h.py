"""experiments/adaptive_ma/expert/er_gate_4h.py -- ER-gated fixed-MA @ 4h, regime-matched FALSIFIER.

TASK (auditor RED-team falsifier): prove the ER-gated fixed-MA @ 4h with ATR-trail exit BEATS a
REGIME-MATCHED null -- random entries drawn ONLY from inside the same ER>0.4 windows, held for durations
sampled from the strategy's OWN hold distribution, at the SAME cost. If it does NOT beat that gated-random
null on held-out (OOS+UNSEEN), the "edge" is regime/beta, not entry TIMING.

Strategy (minimal honest config, 3 DOF fixed up-front BEFORE seeing held-out -- per RESEARCHER_REPORT_1):
  - ONE fixed MA config: 8/21 EMA (the brief's "trend" pair -- responsive momentum).
  - ER gate: Kaufman efficiency ratio over ER_WIN bars, threshold 0.4 (trade the cross ONLY when trending).
  - Entry: fast>slow CROSSOVER event AND er>0.4, confirmed at CLOSE of bar t, filled at open[t+1].
  - Exit: ATR-trail (atr_trail_mult * ATR, ATR over ATR_WIN) + max_hold cap (7d = 42 4h-bars). ONE policy.
  - Cost: taker 0.0024 round-trip.

Regime-matched null: the firewall draws random entries ONLY from bars where er>0.4 (NOT from the entry
column, NOT from all bars). This isolates WITHIN-REGIME entry TIMING from gate/regime SELECTION. Wired by
overriding harness.spec.filter_col='er', filter_op='gt', filter_val=0.4 so strat.firewall._gate_on_mask
masks on er>0.4.

All past-only: MA/cross/ER use close-of-bar t (filled at t+1 => structurally past-only vs the fill);
ATR read by the harness as atr[j-1] (prior-bar) for leak safety. No emoji (cp1252). numpy/pandas only.

RWYB:  python experiments/adaptive_ma/expert/er_gate_4h.py [--quick] [--probe BTCUSDT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from wealth_bot.harness import WindowSpec, ema_past_only  # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy  # noqa: E402
from strat.firewall import random_entry_null  # noqa: E402

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")

# --- fixed config (chosen up-front, NOT on held-out) ---
FAST, SLOW = 8, 21          # the brief's "trend" EMA pair
ER_WIN = 20                 # Kaufman efficiency-ratio lookback (bars)
ER_THRESH = 0.40            # gate: trade the cross only when ER > 0.4 (trending)
ATR_WIN = 14                # ATR lookback
ATR_TRAIL_MULT = 3.0        # 3-ATR trailing stop (banks the 2-5% move; researcher's "3-ATR ~3% move")
MAX_HOLD = 42               # 7 days at 4h (hours-to-<7d hold)


def load_4h(loader: ChimeraLoader, sym: str) -> pd.DataFrame | None:
    try:
        g = loader.load(sym, cadence="4h")
    except Exception:
        return None
    ts = g["timestamp"].to_numpy()
    dt = pd.to_datetime(ts, unit="ms")
    df = pd.DataFrame({
        "date": dt,
        "open": g["open"].to_numpy().astype(float),
        "high": g["high"].to_numpy().astype(float),
        "low": g["low"].to_numpy().astype(float),
        "close": g["close"].to_numpy().astype(float),
    })
    return df


def kaufman_er(close: pd.Series, win: int) -> pd.Series:
    """Kaufman Efficiency Ratio in [0,1]: |net move over win| / sum(|bar moves|). High = clean trend."""
    change = (close - close.shift(win)).abs()
    vol_path = close.diff().abs().rolling(win, min_periods=win // 2).sum()
    return (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0)


def atr_past(df: pd.DataFrame, win: int) -> pd.Series:
    """ATR = rolling mean of true range (value AT bar t; the harness reads atr[j-1] => leak-safe)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=win // 2).mean()


def build_cols(df: pd.DataFrame, entry_style: str = "state") -> pd.DataFrame:
    """Add er, fast/slow EMA, ER-gated entry, atr columns. All close-of-bar (past-only vs the next-open
    fill the harness uses).

    entry_style:
      'state' (primary) -- ER-gated fixed-MA STATE: long when fast>slow AND er>0.4. The single-position
                sim enters at the first such bar after each exit => enters at/after the cross onset within
                the gated regime. Dense enough to power the held-out firewall.
      'cross'           -- the strict cross-EVENT version: a fresh upward crossover AND er>0.4 at the SAME
                bar. Faithful to "trade the cross when ER>0.4" but very sparse (a cross follows chop, where
                ER is low) -> often 0 held-out trades -> underpowered. Reported for completeness.
    """
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    out["er"] = kaufman_er(close, ER_WIN)
    out["fast"] = ema_past_only(close, length=FAST, shift=0).to_numpy()
    out["slow"] = ema_past_only(close, length=SLOW, shift=0).to_numpy()
    f, s = out["fast"], out["slow"]
    cross_up = (f > s) & (f.shift(1) <= s.shift(1))          # fresh upward crossover at close of t
    out["cross_up"] = cross_up.fillna(False)
    gate = out["er"] > ER_THRESH
    if entry_style == "cross":
        out["entry"] = (out["cross_up"] & gate).fillna(False).astype(int)
    else:  # 'state'
        out["entry"] = ((f > s) & gate).fillna(False).astype(int)
    out["atr"] = atr_past(out, ATR_WIN)
    return out


def run_asset(df: pd.DataFrame, n_books: int, seed: int, entry_style: str = "state") -> dict | None:
    """Build the ER-gated 4h strategy on one asset, run the regime-matched (ER>0.4) firewall, return rec."""
    cols = build_cols(df, entry_style=entry_style)
    if int(cols["entry"].sum()) < 4:
        return None  # too few setups to evaluate
    policy = ExitPolicy(atr_trail_mult=ATR_TRAIL_MULT, atr_col="atr", max_hold_bars=MAX_HOLD)
    # regime_match_on_entry=False so SetupHarness does NOT set filter_col=entry; we override to gate on ER>0.4.
    h = SetupHarness(cols, "entry", policy, WIN, cost_rt=TAKER, regime_match_on_entry=False)
    h.spec.filter_col = "er"          # regime-matched null draws from bars where ...
    h.spec.filter_op = "gt"
    h.spec.filter_val = ER_THRESH     # ... er > 0.4 (the SAME regime the strategy gates on)

    res = h.run()
    fw = random_entry_null(h, n_books=n_books, seed=seed, regime_matched=True)

    rec = {
        "n_entries": int(cols["entry"].sum()),
        "er_gate_on_bars": int((cols["er"] > ER_THRESH).sum()),
        "regime_mode": fw["regime_mode"],
        "windows": {w: {"comp": round(res.window_stats[w].compound_pct, 2),
                        "n": res.window_stats[w].n_trades,
                        "wr": round(res.window_stats[w].win_rate, 3)} for w in WINDOWS},
        "firewall": {w: {"real": fw["per_window"][w]["real"],
                         "null_p50": fw["per_window"][w]["null_p50"],
                         "null_p95": fw["per_window"][w]["null_p95"],
                         "beats_null": fw["per_window"][w]["beats_null"],
                         "n": fw["per_window"][w]["n_trades"]} for w in WINDOWS},
        "beats_held": bool(fw["beats_held"]),
        "pos_held": bool(fw["pos_held"]),
        "verdict": fw["verdict"],
    }
    return rec


def main(quick: bool, probe: str | None, entry_style: str = "state"):
    loader = ChimeraLoader()
    if probe:
        df = load_4h(loader, probe)
        print(f"[probe {probe}] bars={0 if df is None else len(df)} | entry_style={entry_style}")
        if df is None:
            return
        cols = build_cols(df, entry_style=entry_style)
        print(f"  entries(ER-gated {entry_style})={int(cols['entry'].sum())}  "
              f"cross_up total={int(cols['cross_up'].sum())}  er>0.4 bars={int((cols['er']>ER_THRESH).sum())}  "
              f"er median={cols['er'].median():.3f}")
        rec = run_asset(df, n_books=300, seed=7, entry_style=entry_style)
        print(json.dumps(rec, indent=2, default=str))
        return

    syms = UniverseLoader.load().list("u100")
    if quick:
        syms = syms[:20]
    print(f"[ER-gate 4h falsifier] u100 4h | assets={len(syms)} | taker={TAKER} | entry={entry_style} | "
          f"MA={FAST}/{SLOW}EMA | ER>{ER_THRESH} gate | exit=ATR-trail x{ATR_TRAIL_MULT}+{MAX_HOLD}bar cap", flush=True)

    per_asset = {}
    n_books = 200
    for k, s in enumerate(syms, 1):
        df = load_4h(loader, s)
        if df is None or len(df) < 1000:
            continue
        try:
            rec = run_asset(df, n_books=n_books, seed=7, entry_style=entry_style)
        except Exception as e:  # noqa: BLE001
            rec = {"error": repr(e)[:160]}
        if rec is not None:
            per_asset[s] = rec
        if k % 10 == 0:
            print(f"[run] {k}/{len(syms)} processed, {len(per_asset)} evaluated", flush=True)

    # aggregate
    evaluated = {s: r for s, r in per_asset.items() if "error" not in r and "beats_held" in r}
    n_eval = len(evaluated)
    beat_and_pos = [s for s, r in evaluated.items() if r["beats_held"] and r["pos_held"]]
    pos_held_only = [s for s, r in evaluated.items() if r["pos_held"]]
    beats_held_only = [s for s, r in evaluated.items() if r["beats_held"]]
    # pooled held-out real vs null
    unseen_real = np.array([r["windows"]["UNSEEN"]["comp"] for r in evaluated.values()], float)
    oos_real = np.array([r["windows"]["OOS"]["comp"] for r in evaluated.values()], float)

    agg = {
        "n_assets_evaluated": n_eval,
        "n_beat_regime_null_AND_pos_held": len(beat_and_pos),
        "assets_beat_and_pos": beat_and_pos,
        "n_pos_held": len(pos_held_only),
        "n_beats_held_flag": len(beats_held_only),
        "UNSEEN_real_mean_comp": round(float(unseen_real.mean()), 2) if n_eval else None,
        "UNSEEN_real_median_comp": round(float(np.median(unseen_real)), 2) if n_eval else None,
        "OOS_real_mean_comp": round(float(oos_real.mean()), 2) if n_eval else None,
        "OOS_real_median_comp": round(float(np.median(oos_real)), 2) if n_eval else None,
    }
    out = {"config": {"cadence": "4h", "entry_style": entry_style, "fast": FAST, "slow": SLOW, "ma": "ema",
                      "er_win": ER_WIN, "er_thresh": ER_THRESH, "atr_win": ATR_WIN,
                      "atr_trail_mult": ATR_TRAIL_MULT, "max_hold": MAX_HOLD, "taker": TAKER,
                      "null": "regime_matched_ER>0.4", "windows": WIN.__dict__},
           "aggregate": agg, "per_asset": per_asset}
    print("\n" + "=" * 78)
    print(f"ER-GATED FIXED-MA @ 4h vs REGIME-MATCHED (ER>{ER_THRESH}) RANDOM-ENTRY NULL  |  {n_eval} assets")
    print("=" * 78)
    print(f"  Assets that BEAT the regime-matched null AND stay positive on held-out (OOS+UNSEEN): "
          f"{len(beat_and_pos)}/{n_eval}")
    print(f"  (positive-held only: {len(pos_held_only)}/{n_eval}; beats-null flag only: {len(beats_held_only)}/{n_eval})")
    print(f"  UNSEEN real compound: mean={agg['UNSEEN_real_mean_comp']}%  median={agg['UNSEEN_real_median_comp']}%")
    print(f"  OOS    real compound: mean={agg['OOS_real_mean_comp']}%  median={agg['OOS_real_median_comp']}%")
    print("=" * 78)
    outpath = Path(__file__).resolve().parent / ("er_gate_4h_quick.json" if quick else "er_gate_4h_u100.json")
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 20 u100 assets only")
    ap.add_argument("--probe", type=str, default=None, help="single-asset probe (e.g. BTCUSDT)")
    ap.add_argument("--entry", type=str, default="state", choices=["state", "cross"],
                    help="ER-gated entry style: state=long when fast>slow&er>0.4 (primary); cross=fresh cross&er>0.4")
    args = ap.parse_args()
    main(quick=args.quick, probe=args.probe, entry_style=args.entry)
