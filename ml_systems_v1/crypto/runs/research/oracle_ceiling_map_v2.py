"""oracle_ceiling_map_v2.py -- per-(cadence x asset) REALIZABLE-CEILING map (RWYB, 2026-06-06).

Reuses the AUDITED perfect-foresight high-capture DP from oracle_ceiling_builder.py
(selftest PASS: single-move, flat, and pullback-two-move cases all exact). Adds the
HONEST, ACTIONABLE ceiling metrics the raw "total compound" lacks:

  * total_capturable_compound_pct  -- the REQUESTED metric. Emitted, but DEGENERATE:
      perfect-foresight compounding of hundreds-to-thousands of perfectly-timed moves
      blows up to 1e20..1e120 %. It is mathematically correct but NOT a target anyone
      (or any MA-DNA) can "approach" -- you cannot reinvest perfectly N hundred times.
  * sum_net_per_move_pct            -- additive (no compounding explosion); the honest
      "if you captured each move on a fixed unit stake" ceiling.
  * mean/median net per-move %      -- the PER-TRADE QUALITY ceiling. THIS is the number
      the MA-DNA must approach: capture_rate = MA_mean_net / oracle_mean_net.
  * per-window UNSEEN {n_moves, mean/median net%, compound%} -- the held-out slice; the
      only window the MA-DNA is judged on. Still explodes on compound (flagged), so the
      actionable UNSEEN ceiling is (n_moves, mean/median net%, sum_net%).

Construction choice (vs the task's two offered paths):
  - capturable_4h_catalog is a BINARY win-LABEL catalog (win_3pct_12bar/win_5pct_18bar,
    base rates 0.41/0.33) -- NOT a max-per-move-capture oracle. Reuse branch REFUTED.
  - setup_harness with a FIXED TP/SL exit caps capture BELOW the true high -> a LOOSER
    (lower) ceiling. The DP high-capture (exit at the exact future high) is the TIGHT
    perfect-foresight ceiling and matches the task spec verbatim ("max per-move capture").

RWYB:  .venv/Scripts/python.exe runs/research/oracle_ceiling_map_v2.py
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

from oracle_ceiling_builder import oracle_high_capture, COST_RT, MIN_HOLD_HOURS, _windows_ms, WIN, MS_PER_HOUR

ASSETS = ["SOL", "BTC", "ETH", "BNB", "AVAX"]
CADENCES = ["1d", "4h", "1h", "30m", "15m"]   # exact high-capture DP cadences (dollar is degenerate -> excluded)


def summarize_v2(ts_ms, trades, open_, high):
    wins = _windows_ms(ts_ms)
    nets, holds_h, ent_ts = [], [], []
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
    comp = float(np.prod(1.0 + nets) - 1.0) if len(nets) else 0.0
    win_out = {}
    for w in WIN:
        wn = np.array(per_win[w]) if per_win[w] else np.array([])
        win_out[w] = {
            "n_moves": int(len(wn)),
            "mean_net_pct": float(wn.mean() * 100) if len(wn) else 0.0,
            "median_net_pct": float(np.median(wn) * 100) if len(wn) else 0.0,
            "sum_net_pct": float(wn.sum() * 100) if len(wn) else 0.0,
            "compound_pct_DEGENERATE": float((np.prod(1.0 + wn) - 1.0) * 100) if len(wn) else 0.0,
        }
    return {
        "n_moves": int(len(trades)),
        "mean_net_per_move_pct": float(nets.mean() * 100) if len(nets) else 0.0,
        "median_net_per_move_pct": float(np.median(nets) * 100) if len(nets) else 0.0,
        "sum_net_per_move_pct": float(nets.sum() * 100) if len(nets) else 0.0,
        "total_capturable_compound_pct_DEGENERATE": comp * 100,
        "mean_hold_hours": float(np.mean(holds_h)) if holds_h else 0.0,
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
        "method": "perfect-foresight high-capture DP (entry=open[k], exit=high[j], non-overlap, "
                  "hours<=hold<7d via real ts); EXACT max-compound (selftest PASS)",
        "note_compound_degenerate": "total/UNSEEN compound explode under perfect-foresight reinvestment; "
                                    "use mean/median/sum net per-move % + n_moves as the ACTIONABLE ceiling.",
        "catalog_reuse_verdict": "REFUTED: capturable_4h_catalog is a binary win-label catalog "
                                 "(win_3pct_12bar/win_5pct_18bar, base 0.41/0.33), not a max-capture oracle.",
        "results": {},
    }
    print(f"{'cad':5} {'asset':5} {'n_bars':>8} {'moves':>6} {'mean%':>6} {'med%':>6} {'sum%':>9} "
          f"{'medHold_h':>9} | UNSEEN: {'n':>4} {'mean%':>6} {'med%':>6} {'sum%':>8}")
    print("-" * 120)
    for cad in CADENCES:
        out["results"][cad] = {}
        t_cad = time.time()
        for a in ASSETS:
            sym = a + "USDT"
            try:
                g = L.load(sym, cadence=cad)
            except Exception as e:
                out["results"][cad][a] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
                print(f"{cad:5} {a:5} LOAD-ERR {str(e)[:50]}")
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
            s = summarize_v2(ts_ms, trades, op, hi)
            s["n_bars"] = n
            s["span_days"] = float((ts_ms[-1] - ts_ms[0]) / (24 * 3600 * 1000))
            out["results"][cad][a] = s
            u = s["UNSEEN"]
            print(f"{cad:5} {a:5} {n:>8} {s['n_moves']:>6} {s['mean_net_per_move_pct']:>6.2f} "
                  f"{s['median_net_per_move_pct']:>6.2f} {s['sum_net_per_move_pct']:>9.0f} "
                  f"{s['median_hold_hours']:>9.2f} | {u['n_moves']:>10} {u['mean_net_pct']:>6.2f} "
                  f"{u['median_net_pct']:>6.2f} {u['sum_net_pct']:>8.0f}")
        print(f"   [{cad} done in {time.time()-t_cad:.1f}s]")

    outp = ROOT / "runs" / "research" / "oracle_ceiling_map_v2.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
    return out


if __name__ == "__main__":
    main()
