"""ma_dna_capture_vs_firewall_by_cadence.py -- CADENCE-as-search-dimension scan (RWYB, 2026-06-06).

TASK (OVERSEER): "Cadence is a SEARCH dimension. Run the oracle-DNA scan across the
{15m,30m,1h,1d,dollar,range,dib,runs_volume,adaptive_vol} x u100 grid (4h already done), MA as
indicator, hold 1h-<7d enforced as a trade-level exit constraint. Rank each cadence by held-out
CAPTURE-RATE vs the regime-matched random-entry firewall null. Find where MA-DNA best captures the
oracle, then deepen."

WHAT THIS MEASURES (per cadence x asset), all on the SAME held-out split the apparatus uses everywhere:
  * MA-DNA = a long-only MOVING-AVERAGE CROSSOVER setup (fast SMA crosses above slow SMA, confirmed at
    close -> next-bar-open fill), exited on the bearish cross OR a <7d max-hold cap (the hold-time
    constraint, expressed per cadence in bars). Scored by src/strat/setup_harness.SetupHarness
    (IC-independent compound; structural look-ahead guards built in).
  * ORACLE = the AUDITED perfect-foresight high-capture DP (oracle_ceiling_builder.oracle_high_capture:
    entry=open[k], exit=high[j], 1h<=hold<7d, non-overlap, net taker 0.0024). selftest PASS.
  * CAPTURE RATE = MA realized held-out (OOS+UNSEEN) sum-of-net-per-trade / ORACLE held-out
    sum-of-net-per-move. (Additive fixed-stake basis -- the honest aggregate; the *compound* oracle is
    degenerate, 1e20+%, per the ceiling map.) capture in [0,1]-ish; the fraction of the clairvoyant
    opportunity the MA timing actually realizes.
  * FIREWALL NULL = regime-matched random-entry null. "Regime" = MA bullish alignment (fast>slow); the
    null draws random entries ONLY from those bars, count matched to the MA per-window trade count,
    hold-durations sampled from the MA's OWN per-window distribution, same cost. This isolates the MA
    CROSSOVER TIMING from the MA REGIME SELECTION (being long while fast>slow). Reported as the null
    capture-rate p50/p95. The AUDITED src/strat/firewall.random_entry_null (compound basis) is ALSO run
    as a cross-check verdict.

DECISION SURFACE: a cadence where MA-DNA "captures the oracle" beyond beta/regime requires
  capture_MA  >  capture_null_p95   on held-out (the MA crossover timing beats random-among-bullish).
Rank cadences by median capture_MA and by median (capture_MA - capture_null_p50) across the asset panel.

RWYB:  python runs/research/ma_dna_capture_vs_firewall_by_cadence.py [--quick] [--cadences a,b] [--assets X,Y]
SAFETY: read-only on real data; writes one JSON under runs/research/. No commit/deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pipeline.chimera_loader import ChimeraLoader          # noqa: E402
from wealth_bot.harness import WindowSpec, sma_past_only    # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy    # noqa: E402
from strat.firewall import random_entry_null                # noqa: E402
from oracle_ceiling_builder import oracle_high_capture, _windows_ms, COST_RT, MS_PER_HOUR  # noqa: E402

WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
HELD = ("OOS", "UNSEEN")
MAX_HOLD_MS = 7 * 24 * 3600 * 1000

# pre-registered MA configs (MOVING-AVERAGE crossover family; small to keep family_n honest)
MA_CONFIGS = [(10, 30), (20, 50)]

# liquid panel parity with the oracle ceiling map; range/dib only have BTC/ETH/PEPE built
PANEL_LIQUID = ["SOL", "BTC", "ETH", "BNB", "AVAX"]
PANEL_EXOTIC = ["BTC", "ETH", "PEPE"]


def load_ohlc(L, sym, cadence):
    g = L.load(sym, cadence=cadence)
    cols = g.columns
    # PREFER the ms `timestamp` (Int64) -- the chimera `date` col is Date-truncated (lossy intraday),
    # which collapses the oracle DP's hold-time windows to day-integers. Use full-precision ms for BOTH
    # the oracle DP and full-datetime window labeling.
    tcol = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
    raw = g[tcol].to_numpy()
    if np.issubdtype(raw.dtype, np.number):
        ts_ms = raw.astype(np.int64)
        dt = pd.to_datetime(ts_ms, unit="ms")
    else:
        dt = pd.to_datetime(raw)
        ts_ms = (dt.astype("int64") // 1_000_000).to_numpy().astype(np.int64)
    df = pd.DataFrame({
        "date": dt,
        "open": g["open"].to_numpy().astype(float),
        "high": g["high"].to_numpy().astype(float),
        "low": g["low"].to_numpy().astype(float),
        "close": g["close"].to_numpy().astype(float),
    })
    if not np.all(np.diff(ts_ms) > 0):
        order = np.argsort(ts_ms, kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        ts_ms = ts_ms[order]
    return df, ts_ms


def bars_per_7d(ts_ms):
    """Median bar duration -> the <7d hold cap expressed in bars for this cadence (>=1)."""
    if len(ts_ms) < 3:
        return 1
    med_dur = float(np.median(np.diff(ts_ms)))
    if med_dur <= 0:
        return 1
    return max(1, int(MAX_HOLD_MS / med_dur))


def build_ma_setup(df, fast, slow):
    """Add MA-cross entry/exit/regime columns (past-only). entry = bullish cross-up confirmed at close."""
    c = df["close"]
    sf = sma_past_only(c, fast).to_numpy()
    ss = sma_past_only(c, slow).to_numpy()
    bull = (sf > ss) & np.isfinite(sf) & np.isfinite(ss)               # bullish alignment (NaN warmup -> False)
    bull_prev = np.concatenate([[False], bull[:-1]])
    df = df.copy()
    df["ma_x_up"] = (bull & ~bull_prev)                                # cross-up bar (confirmed at close)
    df["ma_x_dn"] = (~bull & bull_prev)                                # cross-down bar (exit signal)
    df["ma_bullish"] = bull.astype(int)                               # the REGIME (long while fast>slow)
    return df


def regime_null_sumnet(h, regime_col, n_books=300, seed=7):
    """Regime-matched random-entry null on the SUM-OF-NET-PER-TRADE basis (additive, capture-rate-ready).
    Mirrors strat.firewall.random_entry_null logic but accumulates sum(nets) per held-out window instead
    of compound. Random entries drawn ONLY from regime-ON (fast>slow) bars; count + durations matched."""
    rng = np.random.default_rng(seed)
    real = h.run()
    df = h.df
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    cost = float(h.spec.cost_rt)
    wlab = np.array([h._window_label(pd.Timestamp(dates.iloc[i])) for i in range(n)])
    regime = df[regime_col].to_numpy(float) > 0.5

    real_n = {w: 0 for w in h.WINDOWS}
    real_durs = {w: [] for w in h.WINDOWS}
    real_sum = {w: 0.0 for w in h.WINDOWS}
    for t in real.trades:
        w = t["window"]
        real_n[w] += 1
        real_durs[w].append(max(1, int(t["duration_bars"])))
        real_sum[w] += float(t["net_pnl"])

    eligible = {w: np.array([i for i in range(1, n - 2) if wlab[i] == w and regime[i]]) for w in h.WINDOWS}

    out = {}
    for w in h.WINDOWS:
        nw = real_n[w]
        if nw == 0 or len(eligible[w]) == 0:
            out[w] = {"real_sum": real_sum[w], "null_p50": 0.0, "null_p95": 0.0, "n": nw, "n_eligible": int(len(eligible[w]))}
            continue
        durs = np.array(real_durs[w]) if real_durs[w] else np.array([3])
        sums = np.empty(n_books)
        for b in range(n_books):
            entries = rng.choice(eligible[w], size=nw, replace=True)
            dsamp = rng.choice(durs, size=nw, replace=True)
            s = 0.0
            for e, d in zip(entries, dsamp):
                ef = e + 1
                xf = min(ef + int(d), n - 1)
                if xf <= ef:
                    continue
                s += opens[xf] / opens[ef] - 1.0 - cost
            sums[b] = s
        out[w] = {"real_sum": real_sum[w], "null_p50": float(np.percentile(sums, 50)),
                  "null_p95": float(np.percentile(sums, 95)), "n": nw, "n_eligible": int(len(eligible[w]))}
    return out, real


def oracle_held_sumnet(ts_ms, df):
    """Oracle DP held-out (OOS+UNSEEN) sum-of-net-per-move % (the capture denominator). None for too-fine."""
    op = df["open"].to_numpy().astype(np.float64)
    hi = df["high"].to_numpy().astype(np.float64)
    _, trades = oracle_high_capture(ts_ms, op, hi)
    wins = _windows_ms(ts_ms)
    held = {w: 0.0 for w in HELD}
    nmoves = {w: 0 for w in HELD}
    for (i, j) in trades:
        net = hi[j] / op[i] - 1.0 - COST_RT
        ent = ts_ms[i]
        for w in HELD:
            lo, hiw = wins[w]
            if lo <= ent < hiw:
                held[w] += net
                nmoves[w] += 1
                break
    return (held["OOS"] + held["UNSEEN"]) * 100.0, {w: nmoves[w] for w in HELD}, len(trades)


def scan_cell(L, sym, cadence, fast, slow, n_books, run_oracle):
    df, ts_ms = load_ohlc(L, sym, cadence)
    n = len(df)
    cap = bars_per_7d(ts_ms)
    df = build_ma_setup(df, fast, slow)
    n_setups = int(df["ma_x_up"].sum())
    if n_setups < 4:
        return {"error": f"only {n_setups} MA cross-ups", "n_bars": n}
    policy = ExitPolicy(exit_signal_col="ma_x_dn", max_hold_bars=cap)
    h = SetupHarness(df, "ma_x_up", policy, WIN, cost_rt=COST_RT)
    # firewall regime mask = MA bullish alignment (NOT the sparse entry col)
    h.spec.filter_col = "ma_bullish"
    h.spec.filter_op = "gt"
    h.spec.filter_val = 0.5

    null_sum, real = regime_null_sumnet(h, "ma_bullish", n_books=n_books)
    ma_held_sum = (null_sum["OOS"]["real_sum"] + null_sum["UNSEEN"]["real_sum"]) * 100.0
    null_p50_held = (null_sum["OOS"]["null_p50"] + null_sum["UNSEEN"]["null_p50"]) * 100.0
    null_p95_held = (null_sum["OOS"]["null_p95"] + null_sum["UNSEEN"]["null_p95"]) * 100.0
    med_hold = float(np.median([t["duration_bars"] for t in real.trades])) if real.trades else 0.0
    held_n = sum(null_sum[w]["n"] for w in HELD)

    # audited compound-basis firewall cross-check verdict
    fw = random_entry_null(h, n_books=n_books, seed=7, regime_matched=True)

    cell = {
        "n_bars": n, "n_setups": n_setups, "max_hold_cap_bars": cap, "median_hold_bars": med_hold,
        "ma_held_sumnet_pct": ma_held_sum, "null_p50_sumnet_pct": null_p50_held,
        "null_p95_sumnet_pct": null_p95_held, "held_n_trades": held_n,
        "per_window_compound": {w: real.window_stats[w].compound_pct for w in ("TRAIN", "VAL", "OOS", "UNSEEN")},
        "firewall_compound_verdict": fw["verdict"], "firewall_beats_held": fw["beats_held"],
        "firewall_pos_held": fw["pos_held"], "regime_mode": fw["regime_mode"],
    }
    if run_oracle:
        try:
            orc_held, orc_nmoves, orc_total = oracle_held_sumnet(ts_ms, df)
            cell["oracle_held_sumnet_pct"] = orc_held
            cell["oracle_held_nmoves"] = orc_nmoves
            cell["oracle_total_moves"] = orc_total
            if orc_held > 1e-9:
                cell["capture_MA"] = ma_held_sum / orc_held
                cell["capture_null_p50"] = null_p50_held / orc_held
                cell["capture_null_p95"] = null_p95_held / orc_held
                cell["capture_edge_vs_p50"] = cell["capture_MA"] - cell["capture_null_p50"]
                cell["beats_null_p95_sumnet"] = ma_held_sum > null_p95_held
        except Exception as e:
            cell["oracle_error"] = f"{type(e).__name__}: {str(e)[:60]}"
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="n_books=120, panel=BTC,SOL only")
    ap.add_argument("--cadences", default="15m,30m,1h,4h,1d,dollar,range,dib")
    ap.add_argument("--assets", default="")
    ap.add_argument("--nbooks", type=int, default=300)
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()

    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    n_books = 120 if args.quick else args.nbooks
    L = ChimeraLoader()
    t_start = time.time()

    out = {"task": "cadence-as-search-dimension: MA-DNA capture-rate vs regime-matched firewall null",
           "cost_rt": COST_RT, "ma_configs": MA_CONFIGS, "n_books": n_books,
           "hold_constraint": "1h <= hold < 7d (max_hold_bars = floor(7d/median_bar_dur))",
           "window": {"train_end": WIN.train_end, "val_end": WIN.val_end, "oos_end": WIN.oos_end, "unseen_end": WIN.unseen_end},
           "cells": {}, "infeasible": {}}

    print(f"{'cad':7} {'asset':5} {'ma':>7} | {'MA_held%':>9} {'nullp50%':>9} {'nullp95%':>9} "
          f"{'capMA':>7} {'cap_p50':>7} {'cap_p95':>7} {'beat95':>6} | {'FWcompound':>12}")
    print("-" * 120)

    for cad in cadences:
        run_oracle = (not args.no_oracle) and cad not in ("dollar",)  # dollar DP degenerate
        if args.assets:
            assets = [a.strip() for a in args.assets.split(",")]
        elif cad in ("range", "dib"):
            assets = PANEL_EXOTIC
        else:
            assets = (["BTC", "SOL"] if args.quick else PANEL_LIQUID)
        out["cells"][cad] = {}
        for a in assets:
            sym = a + "USDT"
            for (f, sl) in MA_CONFIGS:
                key = f"{a}_{f}/{sl}"
                try:
                    cell = scan_cell(L, sym, cad, f, sl, n_books, run_oracle)
                except FileNotFoundError as e:
                    out["infeasible"][cad] = f"FileNotFoundError: {str(e)[:80]}"
                    print(f"{cad:7} {a:5} {f}/{sl:<4} INFEASIBLE (no bars): {str(e)[:50]}")
                    break
                except Exception as e:
                    cell = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
                out["cells"][cad][key] = cell
                if "error" in cell:
                    print(f"{cad:7} {a:5} {f}/{sl:<4} ERR {cell['error'][:60]}")
                    continue
                capMA = cell.get("capture_MA")
                cp50 = cell.get("capture_null_p50")
                cp95 = cell.get("capture_null_p95")
                b95 = cell.get("beats_null_p95_sumnet")
                fwv = "EDGE" if cell["firewall_beats_held"] and cell["firewall_pos_held"] else "BETA"
                def _p(x, w=9, d=2):
                    return (f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}")
                print(f"{cad:7} {a:5} {f}/{sl:<4} | {_p(cell['ma_held_sumnet_pct'])} {_p(cell['null_p50_sumnet_pct'])} "
                      f"{_p(cell['null_p95_sumnet_pct'])} {_p(capMA,7,3)} {_p(cp50,7,3)} {_p(cp95,7,3)} "
                      f"{str(b95):>6} | {fwv:>12}")
            else:
                continue
            # FileNotFoundError broke the config loop -> mark cadence infeasible, skip remaining assets
            if cad in out["infeasible"]:
                break
        # cadence aggregate
        agg = _aggregate(out["cells"][cad])
        out["cells"][cad]["_AGG"] = agg
        if agg.get("n_cells", 0) > 0:
            print(f"  [{cad} AGG] median capMA={agg.get('median_capture_MA')}  median edge={agg.get('median_edge_vs_p50')}  "
                  f"beats_p95 {agg.get('n_beats_p95')}/{agg.get('n_cells')}  FW-EDGE {agg.get('n_fw_edge')}/{agg.get('n_cells')}")
        print()

    # rank cadences
    ranking = []
    for cad in cadences:
        agg = out["cells"].get(cad, {}).get("_AGG", {})
        if agg.get("n_cells", 0) > 0 and agg.get("median_capture_MA") is not None:
            ranking.append((cad, agg["median_capture_MA"], agg.get("median_edge_vs_p50"),
                            agg.get("n_beats_p95"), agg.get("n_cells"), agg.get("n_fw_edge")))
    ranking.sort(key=lambda r: (r[1] if r[1] is not None else -9e9), reverse=True)
    out["ranking_by_median_capture_MA"] = [
        {"cadence": r[0], "median_capture_MA": r[1], "median_edge_vs_null_p50": r[2],
         "n_beats_null_p95": r[3], "n_cells": r[4], "n_firewall_edge": r[5]} for r in ranking]

    print("=" * 120)
    print("CADENCE RANKING by held-out median MA capture-rate (and edge over regime-matched null):")
    print(f"  {'rank':>4} {'cad':7} {'medCapMA':>9} {'medEdge_p50':>12} {'beats_p95':>10} {'FW_edge':>8}")
    for r, row in enumerate(out["ranking_by_median_capture_MA"], 1):
        print(f"  {r:>4} {row['cadence']:7} {row['median_capture_MA']:>9.3f} "
              f"{(row['median_edge_vs_null_p50'] if row['median_edge_vs_null_p50'] is not None else float('nan')):>12.3f} "
              f"{row['n_beats_null_p95']:>4}/{row['n_cells']:<5} {row['n_firewall_edge']:>3}/{row['n_cells']}")
    if out["infeasible"]:
        print("\nINFEASIBLE cadences (no bars built):")
        for c, m in out["infeasible"].items():
            print(f"  {c}: {m}")

    out["elapsed_s"] = time.time() - t_start
    outp = ROOT / "runs" / "research" / "ma_dna_capture_vs_firewall_by_cadence.json"
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] wrote {outp}  ({out['elapsed_s']:.1f}s)")
    return out


def _aggregate(cells):
    caps, edges, beats, fw_edges = [], [], 0, 0
    n = 0
    for key, c in cells.items():
        if key == "_AGG" or not isinstance(c, dict) or "error" in c:
            continue
        n += 1
        if c.get("capture_MA") is not None:
            caps.append(c["capture_MA"])
        if c.get("capture_edge_vs_p50") is not None:
            edges.append(c["capture_edge_vs_p50"])
        if c.get("beats_null_p95_sumnet"):
            beats += 1
        if c.get("firewall_beats_held") and c.get("firewall_pos_held"):
            fw_edges += 1
    return {
        "n_cells": n,
        "median_capture_MA": float(np.median(caps)) if caps else None,
        "median_edge_vs_p50": float(np.median(edges)) if edges else None,
        "n_beats_p95": beats, "n_fw_edge": fw_edges,
    }


if __name__ == "__main__":
    main()
