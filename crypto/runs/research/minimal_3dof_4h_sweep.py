"""runs/research/minimal_3dof_4h_sweep.py -- robustness sweep + leak-guard for the 3-DOF 4h candidate.

Is the held-out REFUTATION (0/77 firewall on the pre-registered config) a knife-edge of one bad config,
or robust across the natural neighbourhood? Sweep ER-gate x breakout-N x ATR-mult on a fixed 25-asset
subset; for each config report pooled UNSEEN per-trade expectancy, median per-asset compound, and the
firewall beat-count (regime-matched + plain) on held-out. Also run the SetupHarness leak_guard on a
positive-base asset to confirm no look-ahead inflates the positives.

RWYB:  python runs/research/minimal_3dof_4h_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader            # noqa: E402
from pipeline.universe_loader import UniverseLoader          # noqa: E402
from wealth_bot.harness import WindowSpec, ema_past_only     # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy     # noqa: E402
from strat.firewall import random_entry_null                 # noqa: E402

import importlib.util
spec = importlib.util.spec_from_file_location("m3", ROOT / "runs" / "research" / "minimal_3dof_4h_breakout.py")
m3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m3)

WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
TAKER = 0.0024


def build_entry_cfg(df, ma_fast, ma_slow, er_win, er_gate, break_n, atr_win):
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float); high, low = out["high"].astype(float), out["low"].astype(float)
    fast = ema_past_only(close, length=ma_fast, shift=0)
    slow = ema_past_only(close, length=ma_slow, shift=0)
    change = (close - close.shift(er_win)).abs()
    vol_path = close.diff().abs().rolling(er_win, min_periods=er_win // 2).sum()
    er = (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0).shift(1)
    prior_high = high.rolling(break_n).max().shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(atr_win, min_periods=atr_win // 2).mean()
    out["entry"] = ((close > prior_high) & (fast > slow) & (er > er_gate)).fillna(False).astype(int)
    return out


def main():
    loader = ChimeraLoader()
    syms = UniverseLoader.load().list("u100")[:25]
    frames = {}
    for s in syms:
        df = m3._load_ohlc(loader, s)
        if df is not None and len(df) >= 600:
            frames[s] = df
    print(f"[sweep] {len(frames)} assets, 4h. cols: ER-gate x breakout-N x ATR-mult (MA 8/21 EMA, "
          f"ER_WIN 20, ATR_WIN 14, time-stop 42 fixed)\n", flush=True)

    grid = [(eg, bn, am) for eg in (0.30, 0.40, 0.50) for bn in (10, 20, 30) for am in (2.0, 3.0, 4.0)]
    rows = []
    for (eg, bn, am) in grid:
        pool_u, comps_u = [], []
        beat_reg, beat_plain, n_eval = 0, 0, 0
        for s, df in frames.items():
            d = build_entry_cfg(df, 8, 21, 20, eg, bn, 14)
            policy = ExitPolicy(atr_trail_mult=am, atr_col="atr", max_hold_bars=42)
            h = SetupHarness(d, "entry", policy, WIN, cost_rt=TAKER, regime_match_on_entry=True)
            res = h.run()
            u = [t["net_pnl"] for t in res.trades if t["window"] == "UNSEEN"]
            pool_u += u
            comps_u.append(res.window_stats["UNSEEN"].compound_pct)
            try:
                fw = random_entry_null(h, n_books=120, seed=7, regime_matched=True)
                fwp = random_entry_null(h, n_books=120, seed=7, regime_matched=False)
                beat_reg += int(fw["beats_held"]); beat_plain += int(fwp["beats_held"]); n_eval += 1
            except Exception:
                pass
        pu = np.asarray(pool_u, float); cu = np.asarray(comps_u, float)
        rows.append({"er_gate": eg, "break_n": bn, "atr_mult": am,
                     "unseen_pool_exp_pct": round(float(pu.mean() * 100), 3) if pu.size else None,
                     "unseen_pool_n": int(pu.size),
                     "unseen_median_comp": round(float(np.median(cu)), 2),
                     "unseen_pos_assets": int((cu > 0).sum()), "n_assets": int(cu.size),
                     "fw_regime_beats_held": f"{beat_reg}/{n_eval}",
                     "fw_plain_beats_held": f"{beat_plain}/{n_eval}"})
        r = rows[-1]
        print(f"ER>{eg} N={bn:>2} ATR={am} | UNSEEN exp={str(r['unseen_pool_exp_pct']):>7}% "
              f"n={r['unseen_pool_n']:>4} medComp={r['unseen_median_comp']:>7} "
              f"pos={r['unseen_pos_assets']}/{r['n_assets']} | "
              f"FW regime={r['fw_regime_beats_held']} plain={r['fw_plain_beats_held']}", flush=True)

    # ---- leak guard on a positive-base asset (default config) ----
    print("\n[leak-guard] default config, assets with positive UNSEEN base:", flush=True)
    checked = 0
    for s, df in frames.items():
        d = m3.build_entry(df)
        policy = ExitPolicy(atr_trail_mult=3.0, atr_col="atr", max_hold_bars=42)
        h = SetupHarness(d, "entry", policy, WIN, cost_rt=TAKER)
        lg = h.leak_guard()
        if not str(lg["verdict"]).startswith("INSUFFICIENT"):
            print(f"  {s}: {lg['verdict']}  base_held={lg['held_compound_base_pp']}pp "
                  f"ratio={lg['ratio_lead_over_lag']}", flush=True)
            checked += 1
        if checked >= 4:
            break
    if checked == 0:
        print("  (all assets had INSUFFICIENT held-out edge -> leak guard defers to the structural "
              "guarantee; no positive result to leak-test, consistent with refutation)", flush=True)

    outp = ROOT / "runs" / "research" / "minimal_3dof_4h_sweep_result.json"
    outp.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {outp}", flush=True)

    # summary
    best_plain = max(rows, key=lambda r: int(r["fw_plain_beats_held"].split("/")[0]))
    print(f"\n[summary] best plain-firewall config over grid: ER>{best_plain['er_gate']} "
          f"N={best_plain['break_n']} ATR={best_plain['atr_mult']} -> "
          f"plain {best_plain['fw_plain_beats_held']} regime {best_plain['fw_regime_beats_held']} "
          f"(of {best_plain['n_assets']} assets)")


if __name__ == "__main__":
    main()
