"""ma_dna_cadence_fullgrid.py -- FULL feasible cadence grid for the OVERSEER cadence-search task.

Driver around runs/research/ma_dna_capture_vs_firewall_by_cadence.scan_cell (the vetted cell scorer).
Runs ONE comparable asset panel across the time cadences so the cadence RANKING is apples-to-apples,
the 3-asset exotic panel for range/dib (their only built assets), and a small panel for dollar (its
oracle DP is degenerate -> firewall-compound verdict only, no capture-rate).

Feasibility (verified 2026-06-06): runs_volume + adaptive_vol have NO chimera files (raw bars only) ->
INFEASIBLE through the ChimeraLoader apparatus; reported as such, not silently dropped.

RWYB: every number here is produced by running scan_cell on real chimera data. Read-only; writes ONE
JSON. No commit/deploy. Rank key = median held-out capture_MA and median (capture_MA - capture_null_p50);
the decision test is capture_MA > capture_null_p95 (MA crossover timing beats random-among-bullish).
"""
from __future__ import annotations

import glob
import json
import os
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

import ma_dna_capture_vs_firewall_by_cadence as M   # noqa: E402
from pipeline.chimera_loader import ChimeraLoader    # noqa: E402


def chimera_assets(cad):
    fs = glob.glob(str(ROOT / "data" / "processed" / "chimera" / cad / "*.parquet"))
    return set(os.path.basename(f).split("_v51")[0].upper() for f in fs)


def common_time_panel():
    """assets present across ALL time cadences (15m/30m/1h/1d/dollar) -> apples-to-apples panel."""
    sets = [chimera_assets(c) for c in ("15m", "30m", "1h", "1d", "dollar")]
    common = set.intersection(*sets)
    # liquid-first ordering so a truncated panel is still the most-liquid subset
    liquid_order = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
                    "DOGEUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "ARBUSDT", "OPUSDT", "APTUSDT",
                    "INJUSDT", "SUIUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT", "AAVEUSDT", "UNIUSDT"]
    head = [a for a in liquid_order if a in common]
    tail = sorted(common - set(head))
    return head + tail


def run():
    t0 = time.time()
    L = ChimeraLoader()
    panel = common_time_panel()
    # PLAN: per-cadence (panel, n_books, run_oracle).  '4h' intentionally omitted (already done upstream).
    time_panel = panel                          # full 77-asset common panel
    dollar_panel = panel[:8]                     # dollar: small liquid panel, verdict-only (no capture-rate)
    exotic_panel = ["BTC", "ETH", "PEPE"]        # range/dib: only built assets
    PLAN = [
        ("1d",     [a.replace("USDT", "") for a in time_panel],  250, True),
        ("1h",     [a.replace("USDT", "") for a in time_panel],  250, True),
        ("30m",    [a.replace("USDT", "") for a in time_panel],  250, True),
        ("15m",    [a.replace("USDT", "") for a in time_panel],  250, True),
        ("range",  exotic_panel,                                 250, True),
        ("dib",    exotic_panel,                                 250, True),
        ("dollar", [a.replace("USDT", "") for a in dollar_panel],120, False),
    ]

    out = {
        "task": "OVERSEER cadence-as-search-dimension: MA-DNA capture-rate vs regime-matched firewall null (FULL feasible grid)",
        "cost_rt": M.oracle_high_capture.__globals__["COST_RT"], "ma_configs": M.MA_CONFIGS,
        "hold_constraint": "1h<=hold<7d, max_hold_bars = floor(7d / median_bar_dur)",
        "panel_time": time_panel, "panel_dollar": dollar_panel, "panel_exotic": exotic_panel,
        "infeasible_cadences": {
            "runs_volume": "no chimera v51 files (raw bars only at data/processed/bars/runs_volume) -> not loadable via ChimeraLoader",
            "adaptive_vol": "no chimera v51 files (raw bars only at data/processed/bars/adaptive_vol) -> not loadable via ChimeraLoader",
        },
        "note_4h": "4h already done upstream (runs/research/firewall_4h_regime_matched_result.json + MA_DNA_CADENCE_SCAN_REPORT.md)",
        "cells": {},
    }

    print(f"panel(time, n={len(time_panel)}): {time_panel}")
    print(f"{'cad':7} {'asset':6} {'ma':>7} | {'MA%':>8} {'np50%':>8} {'np95%':>8} {'capMA':>7} {'cap95':>7} {'beat95':>6} | {'FW':>5}")
    print("-" * 110)

    for (cad, assets, n_books, run_oracle) in PLAN:
        out["cells"][cad] = {}
        ct0 = time.time()
        for a in assets:
            sym = a + "USDT"
            for (f, sl) in M.MA_CONFIGS:
                key = f"{a}_{f}/{sl}"
                try:
                    cell = M.scan_cell(L, sym, cad, f, sl, n_books, run_oracle)
                except Exception as e:
                    cell = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
                out["cells"][cad][key] = cell
                if "error" in cell:
                    print(f"{cad:7} {a:6} {f}/{sl:<4} ERR {cell['error'][:55]}")
                    continue
                capMA = cell.get("capture_MA")
                cp95 = cell.get("capture_null_p95")
                b95 = cell.get("beats_null_p95_sumnet")
                fwv = "EDGE" if (cell.get("firewall_beats_held") and cell.get("firewall_pos_held")) else "BETA"

                def _p(x, w=8, d=2):
                    return (f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}")
                print(f"{cad:7} {a:6} {f}/{sl:<4} | {_p(cell.get('ma_held_sumnet_pct'))} "
                      f"{_p(cell.get('null_p50_sumnet_pct'))} {_p(cell.get('null_p95_sumnet_pct'))} "
                      f"{_p(capMA,7,3)} {_p(cp95,7,3)} {str(b95):>6} | {fwv:>5}")
        out["cells"][cad]["_AGG"] = M._aggregate(out["cells"][cad])
        agg = out["cells"][cad]["_AGG"]
        print(f"  [{cad} AGG] n={agg['n_cells']} medCapMA={agg.get('median_capture_MA')} "
              f"medEdge_p50={agg.get('median_edge_vs_p50')} beats95={agg['n_beats_p95']}/{agg['n_cells']} "
              f"FW-EDGE={agg['n_fw_edge']}/{agg['n_cells']}  ({time.time()-ct0:.0f}s)\n")

    # rank (capture cadences only -- dollar has no oracle denominator)
    ranking = []
    for cad in out["cells"]:
        agg = out["cells"][cad]["_AGG"]
        if agg["n_cells"] > 0 and agg.get("median_capture_MA") is not None:
            ranking.append((cad, agg["median_capture_MA"], agg.get("median_edge_vs_p50"),
                            agg["n_beats_p95"], agg["n_cells"], agg["n_fw_edge"]))
    ranking.sort(key=lambda r: (r[1] if r[1] is not None else -9e9), reverse=True)
    out["ranking_by_median_capture_MA"] = [
        {"cadence": r[0], "median_capture_MA": r[1], "median_edge_vs_null_p50": r[2],
         "n_beats_null_p95": r[3], "n_cells": r[4], "n_firewall_edge": r[5]} for r in ranking]

    print("=" * 110)
    print("CADENCE RANKING by held-out median MA capture-rate (capture cadences; dollar = verdict-only):")
    print(f"  {'rank':>4} {'cad':7} {'medCapMA':>9} {'medEdge_p50':>12} {'beats_p95':>11} {'FW_edge':>9}")
    for r, row in enumerate(out["ranking_by_median_capture_MA"], 1):
        e = row["median_edge_vs_null_p50"]
        print(f"  {r:>4} {row['cadence']:7} {row['median_capture_MA']:>9.3f} "
              f"{(e if e is not None else float('nan')):>12.4f} "
              f"{row['n_beats_null_p95']:>4}/{row['n_cells']:<6} {row['n_firewall_edge']:>3}/{row['n_cells']}")
    # dollar verdict
    dagg = out["cells"].get("dollar", {}).get("_AGG", {})
    if dagg:
        print(f"\n  dollar (no capture-rate): FW-EDGE {dagg.get('n_fw_edge')}/{dagg.get('n_cells')} cells")
    print(f"\n  INFEASIBLE: runs_volume, adaptive_vol (no chimera files)")

    out["elapsed_s"] = time.time() - t0
    outp = ROOT / "runs" / "research" / "ma_dna_cadence_fullgrid_result.json"
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] wrote {outp}  ({out['elapsed_s']:.0f}s)")
    return out


if __name__ == "__main__":
    run()
