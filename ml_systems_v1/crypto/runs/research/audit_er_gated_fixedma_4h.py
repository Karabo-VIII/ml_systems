"""runs/research/audit_er_gated_fixedma_4h.py -- AUDITOR verification (RWYB, NOT a ship claim).

Independently verifies the redirect claim: does an ER-GATED FIXED-MA at 4h cadence BEAT the
REGIME-MATCHED random-entry null on the held-out (OOS+UNSEEN) splits, with per-trade UNSEEN net
expectancy in the 2-5%/move band?  No rig produced a committed 4h result; this builds the minimal
3-DOF config the researcher prescribed (ER threshold gate + 1 fixed MA + 1 exit policy) and runs the
KEPT apparatus (src/strat/firewall.regime_matched=True + battery) over u100 4h.

Minimal config (pre-registered, fixed before reading held-out):
  - entry  : fixed-MA crossover (fast>slow), gated by Kaufman ER > ER_THR (trade only when trending)
  - exit   : signal_flip_or_filter (opposite cross OR ER-gate drops) -- the harness's gate-aware exit
  - cost   : taker 0.0024 round-trip (honest)
  - null   : firewall regime_matched=True -> random entries drawn ONLY from ER-gate-ON bars
             (isolates WITHIN-GATE entry timing from the gate/regime SELECTION).

Decisive held-out tests (per the brief's audit protocol):
  (1) per-asset: real held-out compound beats the gate-ON null p95 AND positive on OOS+UNSEEN
  (2) pooled UNSEEN per-trade net expectancy in [2%,5%] band
  (3) battery on pooled UNSEEN: block-bootstrap p05>0, jackknife jk3 stable

RWYB:  python runs/research/audit_er_gated_fixedma_4h.py [--quick] [--er-thr 0.4] [--ma 10 30 sma]
No emoji (cp1252). numpy/pandas + kept apparatus only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from wealth_bot.harness import (  # noqa: E402
    CanonicalHarness, StrategySpec, WindowSpec, sma_past_only, ema_past_only,
)
from strat.firewall import random_entry_null  # noqa: E402
from strat.battery import evaluate, block_bootstrap_p05_p95, jackknife, herfindahl_neff  # noqa: E402

TAKER = 0.0024
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
ER_WIN = 20


def _load_4h(loader: ChimeraLoader, sym: str) -> pd.DataFrame | None:
    try:
        g = loader.load(sym, cadence="4h")
    except Exception:
        return None
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    df = pd.DataFrame({"date": dt,
                       "open": np.asarray(d["open"], float), "high": np.asarray(d["high"], float),
                       "low": np.asarray(d["low"], float), "close": np.asarray(d["close"], float)})
    return df.sort_values("date").reset_index(drop=True)


def _kaufman_er(close: pd.Series, win: int) -> pd.Series:
    """Causal Kaufman efficiency ratio in [0,1], .shift(1) -> strictly past at decision bar."""
    change = (close - close.shift(win)).abs()
    vol_path = close.diff().abs().rolling(win, min_periods=win // 2).sum()
    er = (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0)
    return er.shift(1)


def _ma(close: pd.Series, length: int, mtype: str) -> np.ndarray:
    if mtype == "ema":
        return ema_past_only(close, length=length, shift=0).to_numpy()
    return sma_past_only(close, length=length, shift=0).to_numpy()


def run(quick: bool, er_thr: float, fast: int, slow: int, mtype: str, n_books: int) -> dict:
    loader = ChimeraLoader()
    syms = UniverseLoader.load().list("u100")
    if quick:
        syms = syms[:20]
    print(f"[audit ER-gated fixed-MA 4h] assets={len(syms)} | taker={TAKER} | MA={fast}/{slow} {mtype} | "
          f"ER>{er_thr} gate | exit=signal_flip_or_filter | null=REGIME_MATCHED | n_books={n_books}", flush=True)

    per_asset = {}
    uns_pool, oos_pool = [], []
    uns_pairs = []
    n_loaded = 0
    for k, s in enumerate(syms, 1):
        df = _load_4h(loader, s)
        if df is None or len(df) < 500:
            continue
        n_loaded += 1
        close = df["close"].astype(float)
        df["er"] = _kaufman_er(close, ER_WIN)
        df["ma_fast"] = _ma(close, fast, mtype)
        df["ma_slow"] = _ma(close, slow, mtype)
        spec = StrategySpec(fast_col="ma_fast", slow_col="ma_slow", signal="crossover",
                            filter_col="er", filter_op="gt", filter_val=er_thr,
                            exit_policy="signal_flip_or_filter", cost_rt=TAKER,
                            use_funding=False, funding_scale=0.0, max_hold_bars=None, max_hold_ext_bars=None)
        h = CanonicalHarness(df, spec, WIN, chimera_path=f"er_gated::{s}")
        try:
            fw = random_entry_null(h, n_books=n_books, seed=7, regime_matched=True)
        except Exception as e:  # noqa: BLE001
            per_asset[s] = {"error": repr(e)}
            continue
        res = h.run()
        uns = [t["net_pnl"] for t in res.trades if t["window"] == "UNSEEN"]
        oos = [t["net_pnl"] for t in res.trades if t["window"] == "OOS"]
        uns_pool += uns
        oos_pool += oos
        uns_pairs += [(t["entry_ts"], t["net_pnl"]) for t in res.trades if t["window"] == "UNSEEN"]
        comps = {w: res.window_stats[w].compound_pct for w in h.WINDOWS}
        per_asset[s] = {
            "regime_mode": fw["regime_mode"],
            "beats_held": fw["beats_held"], "pos_held": fw["pos_held"],
            "ships_real_edge": bool(fw["beats_held"] and fw["pos_held"]),
            "per_window": {w: {"real": fw["per_window"][w]["real"], "null_p95": fw["per_window"][w]["null_p95"],
                               "beats_null": fw["per_window"][w]["beats_null"], "n": fw["per_window"][w]["n_trades"]}
                          for w in h.WINDOWS},
            "comp": {w: round(comps[w], 2) for w in h.WINDOWS},
            "n_unseen": len(uns), "unseen_exp_pct": round(float(np.mean(uns) * 100), 4) if uns else None,
        }
        if k % 10 == 0 or quick:
            pa = per_asset[s]
            print(f"  [{k}/{len(syms)}] {s:12} ships={pa['ships_real_edge']} "
                  f"beats_held={pa['beats_held']} pos_held={pa['pos_held']} "
                  f"n_uns={pa['n_unseen']} uns_exp={pa['unseen_exp_pct']}", flush=True)

    # aggregate
    evals = [v for v in per_asset.values() if "ships_real_edge" in v]
    n_ship = sum(1 for v in evals if v["ships_real_edge"])
    n_beats = sum(1 for v in evals if v["beats_held"])
    uns_arr = np.asarray(uns_pool, float)
    oos_arr = np.asarray(oos_pool, float)
    pooled = {
        "n_assets_loaded": n_loaded, "n_assets_evaluated": len(evals),
        "n_assets_beat_regime_null_held": n_beats,
        "n_assets_ship_real_edge_held": n_ship,
        "pooled_unseen_n": int(uns_arr.size),
        "pooled_unseen_exp_pct": round(float(uns_arr.mean() * 100), 4) if uns_arr.size else None,
        "pooled_unseen_median_pct": round(float(np.median(uns_arr) * 100), 4) if uns_arr.size else None,
        "pooled_unseen_winrate": round(float((uns_arr > 0).mean()), 4) if uns_arr.size else None,
        "in_2_5_band": bool(uns_arr.size and 2.0 <= float(uns_arr.mean() * 100) <= 5.0),
        "pooled_oos_n": int(oos_arr.size),
        "pooled_oos_exp_pct": round(float(oos_arr.mean() * 100), 4) if oos_arr.size else None,
    }
    # battery on pooled UNSEEN
    if uns_arr.size >= 10:
        bb = block_bootstrap_p05_p95(uns_arr)
        pooled["battery_pooled_unseen"] = {
            "p05": bb["p05"], "p50": bb["p50"], "p95": bb["p95"],
            "jk2": round(jackknife(uns_arr, 2), 2), "jk3": round(jackknife(uns_arr, 3), 2),
            "n_eff": round(herfindahl_neff(uns_arr), 1),
            "p05_gt_0": bool(bb["p05"] is not None and bb["p05"] > 0),
        }
    out = {"config": {"cadence": "4h", "taker": TAKER, "ma": [fast, slow, mtype], "er_thr": er_thr,
                      "exit": "signal_flip_or_filter", "null": "regime_matched_gate_on", "n_books": n_books,
                      "windows": WIN.__dict__},
           "pooled": pooled, "per_asset": per_asset}
    print("\n" + "=" * 78)
    print(f"ER-GATED FIXED-MA 4h vs REGIME-MATCHED NULL  |  {len(evals)} assets  |  MA {fast}/{slow} {mtype}  ER>{er_thr}")
    print("=" * 78)
    print(f"  assets beating regime-matched null on held-out (OOS+UNSEEN): {n_beats}/{len(evals)}")
    print(f"  assets w/ REAL EDGE (beats null AND positive held-out)     : {n_ship}/{len(evals)}")
    print(f"  pooled UNSEEN per-trade exp = {pooled['pooled_unseen_exp_pct']}%  "
          f"(median {pooled['pooled_unseen_median_pct']}%, wr {pooled['pooled_unseen_winrate']}, "
          f"n={pooled['pooled_unseen_n']})  in 2-5% band: {pooled['in_2_5_band']}")
    if "battery_pooled_unseen" in pooled:
        b = pooled["battery_pooled_unseen"]
        print(f"  battery pooled UNSEEN: p05={b['p05']} (>0: {b['p05_gt_0']}) jk3={b['jk3']} n_eff={b['n_eff']}")
    print("=" * 78 + "\n", flush=True)
    tag = "quick" if quick else "full"
    outp = Path(__file__).resolve().parent / f"audit_er_gated_4h_{mtype}{fast}_{slow}_er{int(er_thr*100)}_{tag}.json"
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outp}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--er-thr", type=float, default=0.4)
    ap.add_argument("--ma", nargs=3, default=["10", "30", "sma"], help="fast slow type")
    ap.add_argument("--n-books", type=int, default=200)
    args = ap.parse_args()
    run(quick=args.quick, er_thr=args.er_thr,
        fast=int(args.ma[0]), slow=int(args.ma[1]), mtype=args.ma[2], n_books=args.n_books)
