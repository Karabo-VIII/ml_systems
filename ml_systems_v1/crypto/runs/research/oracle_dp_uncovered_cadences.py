"""oracle_dp_uncovered_cadences.py -- DP oracle for ONLY the cadences the
capturable_*_catalog.parquet files do NOT cover (RWYB, 2026-06-06).

WHY THIS SCRIPT EXISTS
----------------------
Task: "confirm capturable_4h_catalog + capturable_win_catalog encode per-asset
oracle entries (max per-move capture, hold<=42 bars, net taker 0.0024); build
the DP oracle only for cadences the catalogs do not cover."

Catalog audit verdict (RWYB-confirmed): REFUTED.
  * capturable_4h_catalog  -> cadence 4h: BINARY win-label catalog
      (win_3pct_12bar base 0.4097, win_5pct_18bar base 0.3287); GROSS max_gain
      over a FIXED 12/18-bar horizon; NO cost column, NO DP non-overlap,
      NO max-per-move-capture compound. Not an oracle.
  * capturable_win_catalog -> cadence 1d: BINARY win-label catalog
      (win_5pct base 0.2991); GROSS max_gain over FIXED 3d horizon; no cost.
      Not an oracle.
So the catalogs ADDRESS cadences {4h, 1d} (as win-label data), but encode NO
oracle. The DP oracle's exact cadences are {1d, 4h, 1h, 30m, 15m} (+dollar).
=> Cadences the catalogs do not cover = {1h, 30m, 15m, dollar}. Build only those.

Reuses the AUDITED perfect-foresight high-capture DP from oracle_ceiling_builder
(selftest PASS: single-move / flat / pullback-two-move exact). Spec verbatim:
entry=open[k], exit=high[j] j>k, 1h<=hold<7d, non-overlap, net round-trip TAKER
cost 0.0024, objective = max compound. NB: at 4h, 7d == 42 bars -- the task's
"hold<=42 bars" is the 4h-expressed form of this 7-day band.

RWYB:  .venv/Scripts/python.exe runs/research/oracle_dp_uncovered_cadences.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from oracle_ceiling_builder import (  # noqa: E402
    oracle_high_capture, _onp_unconstrained, _windows_ms, summarize,
    COST_RT, MIN_HOLD_HOURS, WIN, MS_PER_HOUR,
)

ASSETS = ["SOL", "BTC", "ETH", "BNB", "AVAX"]          # parity with existing maps
DP_CADENCES = ["1h", "30m", "15m"]                      # exact high-capture DP
ONP_CADENCES = ["dollar"]                               # O(n) close-to-close (degenerate UB)
# Deliberately EXCLUDED: 4h, 1d -> covered by the catalogs (task scope).


def summarize_actionable(ts_ms, trades, open_, high):
    """Honest per-move ceiling (compound explodes under perfect reinvestment)."""
    wins = _windows_ms(ts_ms)
    nets, holds_h = [], []
    per_win = {w: [] for w in WIN}
    for (i, j) in trades:
        net = high[j] / open_[i] - 1.0 - COST_RT
        nets.append(net)
        holds_h.append((ts_ms[j] - ts_ms[i]) / MS_PER_HOUR)
        ent = ts_ms[i]
        for w, (lo, hi) in wins.items():
            if lo <= ent < hi:
                per_win[w].append(net)
                break
    nets = np.array(nets) if nets else np.array([])
    win_out = {}
    for w in WIN:
        wn = np.array(per_win[w]) if per_win[w] else np.array([])
        win_out[w] = {
            "n_moves": int(len(wn)),
            "mean_net_pct": float(wn.mean() * 100) if len(wn) else 0.0,
            "median_net_pct": float(np.median(wn) * 100) if len(wn) else 0.0,
            "sum_net_pct": float(wn.sum() * 100) if len(wn) else 0.0,
        }
    return {
        "n_moves": int(len(trades)),
        "mean_net_per_move_pct": float(nets.mean() * 100) if len(nets) else 0.0,
        "median_net_per_move_pct": float(np.median(nets) * 100) if len(nets) else 0.0,
        "sum_net_per_move_pct": float(nets.sum() * 100) if len(nets) else 0.0,
        "total_capturable_compound_pct_DEGENERATE": float(
            (np.prod(1.0 + nets) - 1.0) * 100) if len(nets) else 0.0,
        "median_hold_hours": float(np.median(holds_h)) if holds_h else 0.0,
        "UNSEEN": win_out["UNSEEN"],
        "OOS": win_out["OOS"],
    }


def main():
    import pandas as pd
    from pipeline.chimera_loader import ChimeraLoader

    L = ChimeraLoader()
    out = {
        "cost_rt": COST_RT, "min_hold_hours": MIN_HOLD_HOURS, "max_hold_days": 7,
        "scope": "DP oracle for cadences the capturable_*_catalog files do NOT cover",
        "catalogs_cover_cadences": ["4h (capturable_4h_catalog)", "1d (capturable_win_catalog)"],
        "catalog_oracle_verdict": "REFUTED: both catalogs are BINARY win-label catalogs "
                                  "(no cost column, fixed horizon, no DP non-overlap) -- NOT max-capture oracles.",
        "uncovered_cadences_built": DP_CADENCES + ONP_CADENCES,
        "method_dp": "perfect-foresight high-capture DP (entry=open[k], exit=high[j], non-overlap, "
                     "1h<=hold<7d via real ts); EXACT max-compound; selftest PASS.",
        "method_dollar": "O(n) unconstrained close-to-close oracle (looser UB; granularity-degenerate).",
        "note_compound_degenerate": "perfect-foresight compounding explodes (1e20+ %); the ACTIONABLE "
                                    "ceiling is mean/median/sum net per-move % + n_moves.",
        "results": {},
    }
    print(f"{'cad':6} {'asset':5} {'n_bars':>9} {'moves':>7} {'mean%':>6} {'med%':>6} {'sum%':>10} "
          f"{'medHold_h':>9} | UNSEEN n/mean%/sum%")
    print("-" * 110)

    for cad in DP_CADENCES:
        out["results"][cad] = {}
        t0 = time.time()
        for a in ASSETS:
            sym = a + "USDT"
            try:
                g = L.load(sym, cadence=cad)
            except Exception as e:
                out["results"][cad][a] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
                print(f"{cad:6} {a:5} LOAD-ERR {str(e)[:50]}")
                continue
            cols = g.columns
            tcol = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
            ts_raw = g[tcol].to_numpy()
            if np.issubdtype(ts_raw.dtype, np.number):
                ts_ms = ts_raw.astype(np.int64)
            else:
                ts_ms = (pd.to_datetime(ts_raw).astype("int64") // 1_000_000).to_numpy().astype(np.int64)
            op = g["open"].to_numpy().astype(np.float64)
            hi = g["high"].to_numpy().astype(np.float64)
            n = len(op)
            if not np.all(np.diff(ts_ms) > 0):
                order = np.argsort(ts_ms, kind="stable")
                ts_ms, op, hi = ts_ms[order], op[order], hi[order]
            f, trades = oracle_high_capture(ts_ms, op, hi)
            s = summarize_actionable(ts_ms, trades, op, hi)
            s["n_bars"] = n
            s["span_days"] = float((ts_ms[-1] - ts_ms[0]) / (24 * 3600 * 1000))
            out["results"][cad][a] = s
            u = s["UNSEEN"]
            print(f"{cad:6} {a:5} {n:>9} {s['n_moves']:>7} {s['mean_net_per_move_pct']:>6.2f} "
                  f"{s['median_net_per_move_pct']:>6.2f} {s['sum_net_per_move_pct']:>10.0f} "
                  f"{s['median_hold_hours']:>9.2f} | {u['n_moves']}/{u['mean_net_pct']:.2f}%/{u['sum_net_pct']:.0f}%")
        print(f"   [{cad} done in {time.time()-t0:.1f}s]")

    for cad in ONP_CADENCES:
        out["results"][cad] = {}
        t0 = time.time()
        for a in ASSETS:
            sym = a + "USDT"
            try:
                g = L.load(sym, cadence=cad)
            except Exception as e:
                out["results"][cad][a] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
                print(f"{cad:6} {a:5} LOAD-ERR {str(e)[:50]}")
                continue
            cl = g["close"].to_numpy().astype(np.float64)
            n = len(cl)
            mult, legs = _onp_unconstrained(cl, COST_RT)
            out["results"][cad][a] = {
                "n_bars": n,
                "method": "onp_close_to_close_unconstrained",
                "total_capturable_compound_pct_DEGENERATE": float((mult - 1.0) * 100),
                "n_round_trips": int(legs),
                "CAVEAT": "granularity-degenerate upper bound; not comparable to high-capture cadences",
            }
            print(f"{cad:6} {a:5} {n:>9}  ONP comp(DEGENERATE)={(mult-1)*100:.2e}%  legs={legs}")
        print(f"   [{cad} done in {time.time()-t0:.1f}s]")

    outp = ROOT / "runs" / "research" / "oracle_dp_uncovered_cadences.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
    return out


if __name__ == "__main__":
    main()
