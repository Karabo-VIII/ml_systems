"""
experiments/adaptive_ma/dna_to_setup_bridge.py

THE BRIDGE: predictability  ->  held-out COMPOUND RETURN.

WHAT GAP THIS CLOSES
--------------------
`oracle_dna_shuffled_falsifier.py` proves whether a DNA classifier P(oracle-entry | past-only feats)
has GENUINE selection skill (AUC > shuffled, capture_skill, beats firewall). But its `capture_skill` /
`dna_compound_pct` are measured with TWO conveniences that a live book does NOT have:
  (1) a GLOBAL top-k selection inside the held-out window (rank ALL held-out bars, take the best
      k_frac) -- that needs to see every bar's prediction at once = not realizable in real time;
  (2) a fixed-H OPEN-TO-OPEN proxy exit -- no take-profit / stop / trail / time policy, no intra-bar
      fills, no fill-model cost realism, no benchmark-excess-vs-passive question.

This module is the missing bridge. It converts the SAME DNA predictions into a REALIZABLE setup:
  * freeze a probability THRESHOLD on the FIT window only (enter at bar t iff P[t] >= thr) -- a
    real-time decision rule, NOT a global top-k peek;
  * feed that boolean entry column into src/strat/setup_harness.SetupHarness with a declarative
    multi-candle MOVE exit policy (next-bar-open fill, intra-bar TP/SL/trail/time, pessimistic);
  * apply src/strat/fill_model (taker + maker_pessimistic + ideal_ref) for cost realism;
  * run src/strat/benchmark.benchmark_excess (per-window vs beta-matched passive, incl. BEAR);
  * run src/strat/firewall.random_entry_null (cost-matched random-entry null, regime-matched);
  * run src/strat/battery.evaluate (Lens A/B/C robustness) + the harness's own leak_guard;
  * and finally MEASURE the realized CAPTURE-RATE = realized held-out compound / oracle-ceiling
    compound, per window -- the number that says how much of the perfect-foresight ceiling a
    realizable DNA->policy book actually banks.

REUSES (does NOT re-implement) the audited apparatus:
  experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py :: fit_predict, _window_mask, _feature_cols
  runs/research/oracle_ceiling_builder.py                      :: oracle_high_capture, summarize, WIN
  src/strat/setup_harness.py                                   :: SetupHarness, ExitPolicy
  src/strat/{fill_model,benchmark,firewall,battery}.py         :: apply_fill_model, benchmark_excess,
                                                                   random_entry_null, evaluate

HARD CONSTRAINTS (inherited): LONG-ONLY, SPOT, leverage 1, honest taker round-trip 0.0024, UNSEEN
touched once, objective = WEALTH (held-out compound) under the robustness battery. The DNA threshold
is set on FIT data only; features are past-only norm_/xd_ chimera cols (target_*/forward NEVER in X).

RWYB:
  .venv/Scripts/python.exe experiments/adaptive_ma/dna_to_setup_bridge.py --selftest
  .venv/Scripts/python.exe experiments/adaptive_ma/dna_to_setup_bridge.py --asset ADA --cadence 1d --min-move-net 0.03
  .venv/Scripts/python.exe experiments/adaptive_ma/dna_to_setup_bridge.py --panel ADA,BNB,LINK,PEPE --cadence 1d --min-move-net 0.03
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
sys.path.insert(0, str(ROOT / "runs" / "research"))
sys.path.insert(0, str(ROOT / "experiments" / "adaptive_ma" / "sol"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- reused apparatus -------------------------------------------------------
import oracle_dna_shuffled_falsifier as fal          # fit_predict, _window_mask, _feature_cols, COST_RT
from oracle_ceiling_builder import oracle_high_capture, summarize  # audited perfect-foresight ceiling
from strat.setup_harness import SetupHarness, ExitPolicy
from strat.fill_model import apply_fill_model, MODES
from strat.benchmark import benchmark_excess
from strat.firewall import random_entry_null
from strat.battery import evaluate
from wealth_bot.harness import WindowSpec

COST_RT = 0.0024
WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
# canonical split (parity with strat.DEFAULT_WINDOWS + oracle_ceiling_builder.WIN)
SPEC_WINDOWS = WindowSpec(train_end="2024-05-15", val_end="2025-03-15",
                          oos_end="2025-12-31", unseen_end="2026-05-22")

__contract__ = {
    "kind": "dna_to_setup_capture_bridge",
    "version": "1.0",
    "inputs": ["chimera (asset,cadence) past-only norm_/xd_ feats", "oracle-entry labels (min_move_net)",
               "ExitPolicy (multi-candle MOVE), fill mode"],
    "outputs": ["per-window realized compound (SetupHarness), realized capture-rate vs oracle ceiling, "
                "fill_model 3-mode, benchmark_excess incl bear, firewall, battery, leak_guard; JSON+verdict"],
    "invariants": [
        "DNA threshold frozen on FIT (TRAIN+VAL) only -> realizable real-time rule, NOT a global top-k peek",
        "features past-only norm_/xd_ only; target_*/high/forward NEVER in X (inherited from falsifier)",
        "entry fill = next-bar open; TP/SL/trail breach via intra-bar high/low (inherited from setup_harness)",
        "oracle ceiling + DNA labels share the SAME min_move_net -> capture-rate is apples-to-apples",
        "capture_rate = realized_compound / oracle_ceiling_compound per window (held-out is the verdict)",
        "honest taker 0.0024 baseline; maker modes are pessimistic stress (provisional calibration)",
    ],
}


# ---------------------------------------------------------------------------
def _load_full(asset, cadence):
    """Mirror falsifier.load_asset (same sort/feature contract) but ALSO return low + date for the
    intra-bar setup_harness. Returns ts(int64 ms), date(datetime), o,h,l,c (float64), X, feats."""
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset + "USDT", cadence=cadence)
    cols = list(g.columns)
    ts = g["timestamp"].to_numpy().astype(np.int64)
    o = g["open"].to_numpy().astype(np.float64)
    h = g["high"].to_numpy().astype(np.float64)
    lo = g["low"].to_numpy().astype(np.float64)
    c = g["close"].to_numpy().astype(np.float64)
    feats = fal._feature_cols(cols)
    X = g.select(feats).to_numpy().astype(np.float64)
    if not np.all(np.diff(ts) > 0):
        order = np.argsort(ts, kind="stable")
        ts, o, h, lo, c, X = ts[order], o[order], h[order], lo[order], c[order], X[order]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    date = pd.to_datetime(ts, unit="ms")
    return ts, date, o, h, lo, c, X, feats


def _capture_rate(realized_pct, ceiling_pct):
    """realized / ceiling, guarded. Ceiling is perfect-foresight POSITIVE compound; a realizable book
    can be negative (capture < 0). None when no oracle moves in the window (ceiling ~ 0)."""
    if ceiling_pct is None or abs(ceiling_pct) < 1e-9:
        return None
    return round(realized_pct / ceiling_pct, 4)


# ---------------------------------------------------------------------------
def run_cell(asset, cadence, min_move_net, policy_name, n_books, seed=7, model="logistic", verbose=True):
    ts, date, o, h, lo, c, X, feats = _load_full(asset, cadence)
    n = len(o)

    # ---- 1. oracle labels + ceiling (SAME min_move_net for both) ----------
    f_dp, trades = oracle_high_capture(ts, o, h, min_move_net=min_move_net)
    y = np.zeros(n, dtype=int)
    ent = np.array([i for i, j in trades], dtype=int)
    if len(ent):
        y[ent] = 1
    holds = np.array([j - i for i, j in trades], dtype=int) if trades else np.array([1])
    oracle_summary = summarize(ts, trades, o, h)
    ceiling = {w: oracle_summary["per_window"][w]["compound_pct"] for w in WINDOWS}
    # PER-MOVE ceiling: the bounded, NON-degenerate capture denominator. The compound ceiling
    # over-compounds dozens of non-overlapping perfect moves (TRAIN ~1e14%), so a realizable single-sleeve
    # book scores ~0 capture by construction -- the per-move mean is the honest "quality of one up-leg" bar.
    oracle_mean_net_per_move = {w: oracle_summary["per_window"][w]["mean_net_pct"] for w in WINDOWS}
    oracle_med_hold = int(np.median(holds))

    # ---- 2. DNA fit on TRAIN+VAL, predict P for ALL bars ------------------
    m_tr = fal._window_mask(ts, "TRAIN") | fal._window_mask(ts, "VAL")
    Xtr, ytr = X[m_tr], y[m_tr]
    k_frac = float(ytr.mean()) if ytr.size else 0.0          # oracle base rate on fit window
    p_all = fal.fit_predict(Xtr, ytr, X, seed=seed, shuffle=False, model=model)

    # ---- 3. freeze threshold on FIT only -> realizable boolean entry ------
    p_fit = p_all[m_tr]
    if k_frac <= 0 or k_frac >= 1 or p_fit.size < 10:
        return {"asset": asset, "cadence": cadence, "error": f"degenerate base rate k_frac={k_frac:.4f}"}
    thr = float(np.quantile(p_fit, 1.0 - k_frac))            # fit selection rate ~ oracle base rate
    entry = (p_all >= thr)
    fit_rate = float(entry[m_tr].mean())
    held_rate = float(entry[fal._window_mask(ts, "OOS") | fal._window_mask(ts, "UNSEEN")].mean())

    # ---- 4. build df + SetupHarness with a multi-candle MOVE policy -------
    df = pd.DataFrame({"date": date, "open": o, "high": h, "low": lo, "close": c, "dna_entry": entry})
    policy = _make_policy(policy_name, cadence)
    har = SetupHarness(df, "dna_entry", policy, SPEC_WINDOWS, cost_rt=COST_RT)
    res = har.run()
    realized = {w: float(res.window_stats[w].compound_pct) for w in WINDOWS}
    n_trades = {w: int(res.window_stats[w].n_trades) for w in WINDOWS}
    # realized mean net per trade per window (the per-move numerator)
    realized_mean_net = {}
    for w in WINDOWS:
        sub = [t["net_pnl"] for t in res.trades if t["window"] == w]
        realized_mean_net[w] = float(np.mean(sub) * 100) if sub else 0.0

    # ---- 5. realized CAPTURE-RATE vs oracle ceiling ----------------------
    # (a) COMPOUND capture-rate -- reported but DEGENERATE (ceiling over-compounds; see note above)
    cap_rate = {w: _capture_rate(realized[w], ceiling[w]) for w in WINDOWS}
    held_real = sum(realized[w] for w in HELD)
    held_ceil = sum(ceiling[w] for w in HELD)
    cap_rate_held = _capture_rate(held_real, held_ceil)
    # (b) PER-MOVE capture-rate -- the BOUNDED, MEANINGFUL bridge metric (realized mean net per trade /
    #     oracle mean net per move): "of the average up-leg the oracle banks, what fraction does the
    #     realizable DNA->policy book bank per trade?"  Held-out = trade-weighted across OOS+UNSEEN.
    cap_per_move = {w: _capture_rate(realized_mean_net[w], oracle_mean_net_per_move[w]) for w in WINDOWS}
    held_n = sum(n_trades[w] for w in HELD)
    held_real_mean = (sum(realized_mean_net[w] * n_trades[w] for w in HELD) / held_n) if held_n else 0.0
    held_oracle_mean = float(np.mean([oracle_mean_net_per_move[w] for w in HELD
                                      if oracle_mean_net_per_move[w] != 0.0]) or 0.0)
    cap_per_move_held = _capture_rate(held_real_mean, held_oracle_mean)

    # ---- 6. fill_model 3-mode (cost realism) -----------------------------
    fillm = {mode: apply_fill_model(har, mode, n_mc=300, seed=seed) for mode in MODES}

    # ---- 7. benchmark-excess per regime (incl bear) ----------------------
    bench = benchmark_excess(har)

    # ---- 8. firewall (cost-matched random-entry null, regime-matched) ----
    try:
        fw = random_entry_null(har, n_books=n_books, seed=seed, regime_matched=True)
    except Exception as e:
        fw = {"error": repr(e)[:160]}

    # ---- 9. robustness battery on UNSEEN trades --------------------------
    uns = [t["net_pnl"] for t in res.trades if t["window"] == "UNSEEN"]
    try:
        bat = evaluate(uns, realized, res.window_stats["UNSEEN"].max_dd_pct, family_n=1)
    except Exception as e:
        bat = {"error": repr(e)[:160]}

    # ---- 10. measured leak guard -----------------------------------------
    try:
        lg = har.leak_guard()
    except Exception as e:
        lg = {"verdict": f"ERROR {repr(e)[:120]}"}

    # ---- verdict ---------------------------------------------------------
    beats_fw_held = bool(fw.get("beats_held")) if isinstance(fw, dict) and "error" not in fw else None
    beats_beta_held = bool(bench.get("beats_beta_held"))
    pos_held = all(realized[w] > 0 for w in HELD)
    cap_positive_held = cap_per_move_held is not None and cap_per_move_held > 0
    bridge_survives = bool(pos_held and beats_beta_held and (beats_fw_held is True)
                           and isinstance(bat, dict) and bat.get("verdict", "").upper().startswith("SHIP"))

    out = {
        "asset": asset, "cadence": cadence, "model": model, "min_move_net": float(min_move_net),
        "policy": {"name": policy_name, **_policy_dict(policy)},
        "n_bars": n, "n_features": len(feats), "oracle_base_rate": k_frac,
        "oracle_median_hold_bars": oracle_med_hold,
        "dna_threshold": round(thr, 4), "fit_entry_rate": round(fit_rate, 4),
        "held_entry_rate": round(held_rate, 4),
        "oracle_ceiling_compound_pct": {w: round(ceiling[w], 2) for w in WINDOWS},
        "realized_compound_pct": {w: round(realized[w], 2) for w in WINDOWS},
        "realized_n_trades": n_trades,
        "oracle_mean_net_per_move_pct": {w: round(oracle_mean_net_per_move[w], 3) for w in WINDOWS},
        "realized_mean_net_per_trade_pct": {w: round(realized_mean_net[w], 3) for w in WINDOWS},
        "capture_rate_compound": cap_rate,
        "capture_rate_compound_held_out": cap_rate_held,
        "capture_rate_per_move": cap_per_move,
        "capture_rate_per_move_held_out": cap_per_move_held,
        "held_realized_pct": round(held_real, 2), "held_ceiling_pct": round(held_ceil, 2),
        "fill_model": fillm,
        "benchmark_excess": bench,
        "firewall": fw,
        "battery": bat,
        "leak_guard": lg,
        "VERDICT": {
            "positive_held_out": pos_held,
            "capture_rate_per_move_positive_held": cap_positive_held,
            "beats_beta_benchmark_held": beats_beta_held,
            "beats_firewall_held": beats_fw_held,
            "bear_preserved": bench.get("bear_preserved"),
            "battery_verdict": bat.get("verdict") if isinstance(bat, dict) else None,
            "leak_guard": lg.get("verdict"),
            "BRIDGE_SURVIVES": bridge_survives,
        },
    }
    if verbose:
        _print_cell(out)
    return out


def _make_policy(name, cadence):
    """Multi-candle MOVE exit policies. bars-per-7-days time cap per cadence (matches oracle's 7d hold cap)."""
    bars_7d = {"1d": 7, "4h": 42, "1h": 168, "30m": 336, "15m": 672, "dib": 7, "range": 7}.get(cadence, 7)
    if name == "trail":      # ride the up-leg: trailing stop + the oracle's own 7d time cap
        return ExitPolicy(trail_pct=0.10, max_hold_bars=bars_7d)
    if name == "tp_sl":      # fixed take-profit / stop / time (setup_harness RWYB convention)
        return ExitPolicy(tp_pct=0.12, sl_pct=0.06, max_hold_bars=bars_7d)
    if name == "tp_trail":   # TP cap + trail + time (compromise)
        return ExitPolicy(tp_pct=0.15, trail_pct=0.10, max_hold_bars=bars_7d)
    raise ValueError(f"unknown policy {name}")


def _policy_dict(p: ExitPolicy):
    return {k: getattr(p, k) for k in ("tp_pct", "sl_pct", "trail_pct", "max_hold_bars")
            if getattr(p, k) is not None}


def _print_cell(o):
    print("\n" + "=" * 92)
    print(f"DNA->SETUP BRIDGE  {o['asset']} {o['cadence']}  min_move_net={o['min_move_net']}  "
          f"policy={o['policy']['name']}{ {k:v for k,v in o['policy'].items() if k!='name'} }")
    print("=" * 92)
    print(f"  oracle base rate={o['oracle_base_rate']:.3f}  median hold={o['oracle_median_hold_bars']} bars  "
          f"DNA thr={o['dna_threshold']}  fit_entry_rate={o['fit_entry_rate']}  held_entry_rate={o['held_entry_rate']}")
    print(f"  {'window':8} {'realized%':>11} {'realMean/trd':>12} {'oracMean/mv':>12} "
          f"{'cap_per_move':>13} {'n_trd':>6}")
    for w in WINDOWS:
        cm = o["capture_rate_per_move"][w]
        print(f"  {w:8} {o['realized_compound_pct'][w]:>+11.2f} {o['realized_mean_net_per_trade_pct'][w]:>+12.3f} "
              f"{o['oracle_mean_net_per_move_pct'][w]:>+12.3f} "
              f"{(f'{cm:+.3f}' if cm is not None else 'n/a'):>13} {o['realized_n_trades'][w]:>6}")
    print(f"  HELD-OUT  CAPTURE-RATE per-move={o['capture_rate_per_move_held_out']}  "
          f"(realized {o['held_realized_pct']:+.2f}% vs ceiling {o['held_ceiling_pct']:.2e}% compound "
          f"-> compound cap={o['capture_rate_compound_held_out']} [DEGENERATE: ceiling over-compounds])")
    fm = o["fill_model"]
    print(f"  fill_model held-out medians: " + "  ".join(
        f"{m}=OOS{fm[m]['OOS']['median']:+.1f}/UNS{fm[m]['UNSEEN']['median']:+.1f}" for m in MODES))
    be = o["benchmark_excess"]["per_window"]
    print("  benchmark excess (cand - beta-matched passive), is_bear:")
    for w in WINDOWS:
        b = be[w]
        if b["cand_pct"] is not None:
            print(f"     {w:8} cand={b['cand_pct']:>+8.2f}  beta={b['beta_matched_pct']:>+8.2f}  "
                  f"buyhold={b['buyhold_pct']:>+8.2f}  excess={b['excess_pp']:>+8.2f}pp  "
                  f"bear={b['is_bear']}  beats_beta={b['beats_beta']}")
    fw = o["firewall"]
    if isinstance(fw, dict) and "error" not in fw:
        print(f"  firewall: beats_held={fw.get('beats_held')}  pos_held={fw.get('pos_held')}  "
              f"verdict={fw.get('verdict')}")
    print(f"  battery={o['VERDICT']['battery_verdict']}  leak_guard={o['VERDICT']['leak_guard']}")
    v = o["VERDICT"]
    print(f"  VERDICT: pos_held={v['positive_held_out']}  cap/move+={v['capture_rate_per_move_positive_held']}  "
          f"beats_beta={v['beats_beta_benchmark_held']}  beats_fw={v['beats_firewall_held']}  "
          f"bear_pres={v['bear_preserved']}  ==> BRIDGE_SURVIVES={v['BRIDGE_SURVIVES']}")


# ---------------------------------------------------------------------------
def _selftest():
    """No-market checks of the bridge's own glue: capture-rate math + threshold realizability."""
    print("=" * 70); print("[bridge selftest]"); print("=" * 70)
    ok = True
    # capture-rate math
    assert _capture_rate(50.0, 200.0) == 0.25
    assert _capture_rate(-10.0, 200.0) == -0.05
    assert _capture_rate(10.0, 0.0) is None
    print("  capture-rate math: OK")
    # threshold realizability: thr from a fit slice reproduces ~base-rate selection on that slice
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    base = 0.12
    thr = float(np.quantile(p, 1 - base))
    sel = (p >= thr).mean()
    print(f"  threshold@base_rate={base}: fit selection rate={sel:.3f} (expect ~{base})")
    ok &= abs(sel - base) < 0.02
    # policies are valid ExitPolicy objects with a time cap
    for nm in ("trail", "tp_sl", "tp_trail"):
        pol = _make_policy(nm, "1d")
        ok &= isinstance(pol, ExitPolicy) and pol.max_hold_bars == 7
    print("  policies build with 7-bar (7d) time cap on 1d: OK")
    print(f"\n[bridge selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="ADA")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--panel", default="", help="comma assets; overrides --asset")
    ap.add_argument("--min-move-net", type=float, default=0.03,
                    help="oracle per-move net floor: 0.0=scalp, 0.03-0.05=SWING multi-day MOVE (default 0.03)")
    ap.add_argument("--policy", default="trail", choices=["trail", "tp_sl", "tp_trail"])
    ap.add_argument("--model", default="logistic", choices=["logistic", "gbm"])
    ap.add_argument("--n-books", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not _selftest():
        print("SELFTEST FAILED -- aborting before market data"); sys.exit(1)

    panel = [x.strip().upper() for x in args.panel.split(",") if x.strip()] or [args.asset.upper()]
    rows = []
    for a in panel:
        try:
            rows.append(run_cell(a, args.cadence, args.min_move_net, args.policy, args.n_books,
                                 model=args.model))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"asset": a, "cadence": args.cadence, "error": repr(e)[:200]})

    # panel summary
    ok = [r for r in rows if "error" not in r]
    print("\n" + "=" * 92)
    print(f"PANEL SUMMARY  cadence={args.cadence}  min_move_net={args.min_move_net}  policy={args.policy}  "
          f"model={args.model}")
    print("=" * 92)
    print(f"  {'asset':6} {'cap/move_held':>13} {'held_real%':>11} {'beats_beta':>11} "
          f"{'beats_fw':>9} {'bear_pres':>10} {'battery':>10} {'SURVIVES':>9}")
    for r in ok:
        v = r["VERDICT"]
        print(f"  {r['asset']:6} {str(r['capture_rate_per_move_held_out']):>13} {r['held_realized_pct']:>+11.2f} "
              f"{str(v['beats_beta_benchmark_held']):>11} "
              f"{str(v['beats_firewall_held']):>9} {str(v['bear_preserved']):>10} "
              f"{str(v['battery_verdict']):>10} {str(v['BRIDGE_SURVIVES']):>9}")
    n_survive = sum(1 for r in ok if r["VERDICT"]["BRIDGE_SURVIVES"])
    print(f"\n  bridge survives in {n_survive}/{len(ok)} cells "
          f"(survive = pos held-out AND beats beta-benchmark AND beats firewall AND battery SHIP).")

    tag = f"_swing{int(args.min_move_net*100)}" if args.min_move_net > 0 else "_scalp"
    blob = {"meta": {"cadence": args.cadence, "min_move_net": args.min_move_net, "policy": args.policy,
                     "model": args.model, "panel": panel, "n_books": args.n_books, "cost_rt": COST_RT,
                     "capture_rate_def": "realized_setup_harness_compound / oracle_ceiling_compound per window",
                     "reused": ["oracle_dna_shuffled_falsifier.fit_predict", "oracle_ceiling_builder.oracle_high_capture",
                                "strat.setup_harness.SetupHarness", "strat.fill_model", "strat.benchmark",
                                "strat.firewall", "strat.battery"]},
            "n_survive": n_survive, "cells": rows}
    outp = Path(__file__).resolve().parent / f"dna_to_setup_bridge_{args.cadence}{tag}_{args.policy}.json"
    outp.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
    return blob


if __name__ == "__main__":
    main()
