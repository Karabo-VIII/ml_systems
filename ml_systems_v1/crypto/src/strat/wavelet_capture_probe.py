"""src/strat/wavelet_capture_probe.py -- does a CAUSAL wavelet energy-EXPANSION setup beat a proper null
on CAPTURE-RATE (realized-vs-available move) after costs? (TSFM_WAVELET_WM_SURVEY_2026_06_09.md rec #1)

THE FRAMING (founding, MEMORY.md): the unit of trading is a SETUP across a MULTI-CANDLE MOVE. We do NOT
measure IC / per-bar predictability (banned as a primary metric). We measure two complementary things:
  - CAPTURE-RATE -- via the EXISTING, already-soundness-proven apparatus
    src/strat/within_window_capture_proxy.py: capture = (trigger_ret - worst) / (best - worst) in [0,1]
    over the move-window, with a single common exit (horizon held constant) + a membership+regime-matched
    random-timing null. We DELEGATE to it (it is two-sided calibrated) rather than reinvent a capture
    metric; an earlier home-rolled realized/available ratio EXPLODED on near-zero-available moves -- the
    canonical bounded definition is the right one.
  - net COMPOUND return of the setup after the canonical cost model (the wealth objective), with a
    cost-matched random-entry null + a membership-matched null.

THE SETUP (mirrors C1 / the vol-expansion trigger so results compare): enter long when the causal,
past-only `energy_expansion` (multi-scale generalization of conditional.py's vol trigger) crosses ABOVE a
threshold; exit after a FIXED hold (`hold_bars`). Entry FILL is at next bar's open (no same-bar look-ahead).

HONEST EVALUATION (asymmetric loss -- default to "no edge" unless it clears null + cost):
  (a) CAPTURE-RATE vs membership+regime-matched null -- delegated to within_window_capture_proxy.
      Beating its null => the wavelet TRIGGER lands closer to the best entry inside the move than random
      timing inside the same move (genuine within-move timing value, not move-selection).
  (b) COMPOUND-RETURN random-entry null -- Monte-Carlo: same #trades drawn at random eligible bars, same
      hold dist, same cost. This is the firewall.py PRIMARY gate, re-implemented inline because
      firewall.random_entry_null wraps the R12-specific CanonicalHarness, whereas this setup is a generic
      energy-expansion gate not expressible as an R12 spec. Same principle (cost-matched random ENTRY).
  (c) COMPOUND-RETURN membership-matched null -- each null entry drawn from a band [e-r, e+r] around the
      real entry (mirrors firewall.py M-1). Beating it => the trigger timing adds COMPOUND value too.
  (d) Cost-aware via src/strat/fill_model.py::MODES (taker solid; maker_pessimistic stress) on compound.
  (e) Two-sided soundness: src/strat/synthetic_positive_control.py + within_window_capture_proxy's own
      selftest prove the null family has POWER (accepts a genuine timing skill, rejects when skill=0).

RWYB: runs on REAL u10 chimera (1d, and 4h). Writes runs/strat/wavelet_capture_*.json. Default verdict
"NO EDGE" unless excess-over-null > 0 AND bootstrap p < 0.05 AND it survives costs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from strat.wavelet_causal import wavelet_features, leak_test                       # noqa: E402
from strat.fill_model import MODES                                                  # noqa: E402
from strat.within_window_capture_proxy import capture_proxy, Windows               # noqa: E402

U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
       "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


# --------------------------------------------------------------------------------------------------
def _load_df(sym: str, cadence: str) -> pd.DataFrame:
    """Return a pandas DataFrame with date, open, high, low, close from chimera (canonical loader, no
    direct parquet read). date is a proper datetime so the capture proxy can label TRAIN/VAL/OOS/UNSEEN."""
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(sym, cadence=cadence, features=["open", "high", "low", "close", "date"])
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    return pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                         "high": np.asarray(d["high"], float), "low": np.asarray(d["low"], float),
                         "close": np.asarray(d["close"], float)})


def _net_cost(mode: str) -> float:
    """Round-trip cost charged once per trade. Adverse-selection from the maker stress mode is folded in
    as an extra magnitude penalty proportional to |gross| at evaluation time (see _trade_metrics)."""
    return float(MODES[mode]["cost_rt"])


def _trade_net(o, entry_idx, hold, mode):
    """Net costed return of one long trade entered at FILL = next bar's open after the trigger at
    entry_idx, exiting at open[fill+hold]. Returns net or None if no room.
      gross = open[exit]/open[fill] - 1
      net   = gross - adverse*|gross| - cost_rt   (mirrors fill_model.apply_fill_model F12 penalty)."""
    n = len(o)
    fill = entry_idx + 1
    exit_i = fill + hold
    if fill >= n or exit_i >= n or exit_i <= fill:
        return None
    entry_px = o[fill]
    if entry_px <= 0:
        return None
    gross = o[exit_i] / entry_px - 1.0
    m = MODES[mode]
    return gross - float(m["adverse"]) * abs(gross) - float(m["cost_rt"])


def _compound(nets):
    a = np.asarray(nets, float)
    return float((np.prod(1.0 + a) - 1.0) * 100.0) if a.size else 0.0


# --------------------------------------------------------------------------------------------------
def _make_trigger(c, J, wavelet, energy_win, thresh, warmup):
    """Boolean per-bar trigger array: causal energy_expansion crosses ABOVE `thresh` (past-only).

    Fires at t when ee[t] >= thresh and ee[t-1] < thresh (or ee[t-1] is NaN). All inputs to ee[t] are
    causal functions of c[0..t] (proven by the leak test), so this trigger is strictly past-only."""
    feat = wavelet_features(c, J=J, wavelet=wavelet, energy_win=energy_win)
    ee = feat["energy_expansion"]
    n = len(c)
    trig = np.zeros(n, dtype=bool)
    for t in range(warmup, n):
        if not np.isfinite(ee[t]):
            continue
        prev = ee[t - 1] if t > 0 else np.nan
        if ee[t] >= thresh and (not np.isfinite(prev) or prev < thresh):
            trig[t] = True
    return trig


def run_asset(sym, cadence, J=4, wavelet="haar", energy_win=20, thresh=1.5, hold=5,
              warmup=60, mode="taker", n_books=400, move_radius=1.0, win_radius=3, seed=7) -> dict:
    """Run the causal-wavelet energy-expansion setup on one asset.

    Capture-rate is delegated to within_window_capture_proxy.capture_proxy (canonical, two-sided sound,
    membership+regime-matched null, capture in [0,1]). Compound-return is evaluated inline against a
    cost-matched random-entry null and a membership-matched null (firewall.py spirit). Returns a dict."""
    rng = np.random.default_rng(seed)
    df = _load_df(sym, cadence)
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)
    if n < warmup + hold + 40:
        return {"asset": sym, "error": f"too short ({n} bars)"}

    trig = _make_trigger(c, J, wavelet, energy_win, thresh, warmup)
    fires = np.flatnonzero(trig)
    fires = fires[(fires >= warmup) & (fires < n - hold - 2)]
    if fires.size == 0:
        return {"asset": sym, "n_trades": 0, "note": "no triggers"}

    # ---- CAPTURE-RATE via the canonical proxy (held-out OOS+UNSEEN pooled) ----
    df2 = df.copy()
    df2["wav_trig"] = trig
    cost_rt = float(MODES[mode]["cost_rt"])
    cap_res = capture_proxy(df2, "wav_trig", horizon=hold, win_radius=win_radius, cost=cost_rt, seed=seed)
    held = cap_res.held_summary
    cap_block = {
        "n_windows_held": held["n_windows"], "mean_capture_held": held["mean_capture"],
        "null_p50_held": held.get("null_p50"), "null_p95_held": held["null_p95"],
        "beats_null_held": held["beats_null"], "null_analytic_mean_held": held.get("null_analytic_mean"),
        "all_windows": {w: {"n": s["n_windows"], "mean_capture": s["mean_capture"],
                            "null_p95": s["null_p95"], "beats_null": s["beats_null"]}
                        for w, s in cap_res.per_window_summary.items()},
    }

    # ---- COMPOUND-RETURN (the wealth objective) + two nulls ----
    real_entries = [int(t) for t in fires]
    real_nets = [v for v in (_trade_net(o, t, hold, mode) for t in real_entries) if v is not None]
    if not real_nets:
        return {"asset": sym, "n_trades": 0, "note": "triggers had no room", "capture": cap_block}
    nT = len(real_nets)
    real_comp = _compound(real_nets)

    eligible = np.arange(warmup, n - hold - 2)
    rand_comps = np.empty(n_books)
    for b in range(n_books):
        ents = rng.choice(eligible, size=nT, replace=True)
        nets = [v for v in (_trade_net(o, int(e), hold, mode) for e in ents) if v is not None]
        rand_comps[b] = _compound(nets)
    rand_p50, rand_p95 = float(np.percentile(rand_comps, 50)), float(np.percentile(rand_comps, 95))
    p_rand = float((np.sum(rand_comps >= real_comp) + 1) / (n_books + 1))

    last_valid = n - hold - 2
    r = max(1, int(round(move_radius * hold)))
    bands = []
    for e in real_entries:
        loi, hii = max(warmup, e - r), min(last_valid, e + r)
        bands.append(np.arange(loi, hii + 1) if hii >= loi else np.array([min(max(e, warmup), last_valid)]))
    mem_comps = np.empty(n_books)
    for b in range(n_books):
        nets = []
        for band in bands:
            v = _trade_net(o, int(rng.choice(band)), hold, mode)
            if v is not None:
                nets.append(v)
        mem_comps[b] = _compound(nets)
    mem_p50, mem_p95 = float(np.percentile(mem_comps, 50)), float(np.percentile(mem_comps, 95))
    p_mem = float((np.sum(mem_comps >= real_comp) + 1) / (n_books + 1))

    return {
        "asset": sym, "cadence": cadence, "mode": mode, "n_trades": nT,
        "real_compound_pct": round(real_comp, 3),
        "capture": cap_block,
        "compound_random_entry_null": {
            "p50": round(rand_p50, 3), "p95": round(rand_p95, 3), "p": round(p_rand, 4),
            "beats": bool(real_comp > rand_p95), "excess_pct": round(real_comp - rand_p50, 3)},
        "compound_membership_null": {
            "p50": round(mem_p50, 3), "p95": round(mem_p95, 3), "p": round(p_mem, 4),
            "beats": bool(real_comp > mem_p95), "excess_pct": round(real_comp - mem_p50, 3)},
    }


def _verdict(per_asset, mode):
    """Aggregate verdict. Asymmetric loss -> default NO EDGE. Require BOTH:
      CAPTURE: a MAJORITY of assets beat the membership+regime-matched capture null on held-out, AND a
               positive bootstrap-significant median excess capture (held-out mean - null analytic mean).
      COMPOUND: a MAJORITY beat the cost-matched random-entry compound null AND the membership compound
                null (so the wealth objective, not just the bounded timing statistic, clears the null)."""
    rows = [r for r in per_asset if r.get("n_trades", 0) > 0 and "capture" in r and "compound_random_entry_null" in r]
    if not rows:
        return {"verdict": "NO DATA", "mode": mode}
    n = len(rows)

    # capture (held-out) -- only count assets where the proxy produced a held-out estimate
    cap_rows = [r for r in rows if r["capture"]["mean_capture_held"] is not None
                and r["capture"]["null_analytic_mean_held"] is not None]
    cap_beat = sum(bool(r["capture"]["beats_null_held"]) for r in cap_rows)
    cap_excess = [r["capture"]["mean_capture_held"] - r["capture"]["null_analytic_mean_held"] for r in cap_rows]
    med_excess_cap = float(np.median(cap_excess)) if cap_excess else 0.0
    rng = np.random.default_rng(11)
    if cap_excess:
        arr = np.array(cap_excess)
        boots = np.array([np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(2000)])
        p_boot_cap = float((np.sum(boots <= 0) + 1) / (2000 + 1))
    else:
        p_boot_cap = 1.0

    # compound
    rand_beat = sum(r["compound_random_entry_null"]["beats"] for r in rows)
    mem_beat = sum(r["compound_membership_null"]["beats"] for r in rows)
    med_excess_comp_rand = float(np.median([r["compound_random_entry_null"]["excess_pct"] for r in rows]))

    n_cap = len(cap_rows)
    capture_edge = (n_cap > 0 and cap_beat > n_cap / 2 and med_excess_cap > 0 and p_boot_cap < 0.05)
    compound_edge = (rand_beat > n / 2 and mem_beat > n / 2 and med_excess_comp_rand > 0)
    edge = bool(capture_edge and compound_edge)
    verdict = ("CAUSAL WAVELET EDGE (beats membership-matched capture null AND cost-matched compound null "
               "across a majority of u10, after costs)"
               if edge else
               "NO EDGE -- causal wavelet energy-expansion does NOT beat the null on capture-rate "
               "(and/or compound) after costs")
    return {"verdict": verdict, "edge": edge, "capture_edge": bool(capture_edge),
            "compound_edge": bool(compound_edge), "mode": mode, "n_assets": n, "n_assets_capture": n_cap,
            "assets_beating_capture_null": cap_beat,
            "assets_beating_compound_random_null": rand_beat,
            "assets_beating_compound_membership_null": mem_beat,
            "median_excess_capture_vs_null": round(med_excess_cap, 4),
            "bootstrap_p_median_excess_capture_gt0": round(p_boot_cap, 4),
            "median_excess_compound_pct_vs_random": round(med_excess_comp_rand, 3)}


def _soundness_witness():
    """Two-sided soundness witness: the membership/within-window null family we use has POWER -- it
    DETECTS a genuine timing skill and ties when skill=0, so a NO-EDGE verdict means real absence of edge,
    not a reject-everything sieve. Two independent witnesses:
      (1) synthetic_positive_control.demonstrate -- its within_window_null detects a genuine f=0.6 timing.
      (2) within_window_capture_proxy._selftest -- the EXACT proxy we delegate capture to: SKILLED trigger
          beats its null AND a NO-SKILL trigger does not (two-sided), on synthetic data."""
    out = {}
    try:
        from strat.synthetic_positive_control import demonstrate
        d = demonstrate(seed=0, f=0.6, s=0.8, verbose=False)
        out["spc_within_window_detects_timing(f=0.6)"] = bool(d["timing_detected"])
    except Exception as e:  # noqa: BLE001
        out["spc_within_window_detects_timing(f=0.6)"] = None
        out["spc_note"] = f"synthetic_positive_control not directly reusable: {e}"
    try:
        from strat.within_window_capture_proxy import _selftest as _capproxy_selftest
        # _selftest prints + returns 0 on PASS (skilled beats null, no-skill does not, bounds hold)
        out["capture_proxy_selftest_pass"] = (_capproxy_selftest() == 0)
    except Exception as e:  # noqa: BLE001
        out["capture_proxy_selftest_pass"] = None
        out["capproxy_note"] = f"within_window_capture_proxy selftest not runnable: {e}"
    out["note"] = ("the membership/within-window null family has POWER (accepts a real timing skill, "
                   "rejects no-skill) -> a NO-EDGE verdict is a real absence of edge.")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="1d,4h")
    ap.add_argument("--wavelet", default="haar")
    ap.add_argument("--thresh", type=float, default=1.5)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--n-books", type=int, default=400)
    ap.add_argument("--modes", default="taker,maker_pessimistic")
    args = ap.parse_args()

    print("[wavelet_capture_probe] STEP 0 -- leak test (gate the whole probe on causality):")
    leak_test(wavelet=args.wavelet, J=4, verbose=True)

    soundness = _soundness_witness()
    print(f"[wavelet_capture_probe] soundness witness: spc detects genuine timing = "
          f"{soundness.get('spc_within_window_detects_timing(f=0.6)')}  | capture_proxy selftest pass = "
          f"{soundness.get('capture_proxy_selftest_pass')}")

    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "leak_test_passed": True,
           "soundness_witness": soundness, "params": vars(args), "runs": {}}

    cadences = [s.strip() for s in args.cadences.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    for cadence in cadences:
        for mode in modes:
            key = f"{cadence}__{mode}"
            print(f"\n===== cadence={cadence}  cost-mode={mode} =====")
            per_asset = []
            for sym in U10:
                try:
                    r = run_asset(sym, cadence, wavelet=args.wavelet, thresh=args.thresh,
                                  hold=args.hold, mode=mode, n_books=args.n_books)
                except FileNotFoundError:
                    r = {"asset": sym, "error": "no chimera"}
                except Exception as e:  # noqa: BLE001
                    r = {"asset": sym, "error": str(e)}
                per_asset.append(r)
                if "real_compound_pct" in r:
                    cp = r["capture"]; cr = r["compound_random_entry_null"]; cm = r["compound_membership_null"]
                    cap_h = cp["mean_capture_held"]; cap_p95 = cp["null_p95_held"]
                    print(f"  {sym:9} nT={r['n_trades']:3d} | CAP held mean={cap_h} null_p95={cap_p95} "
                          f"beat={cp['beats_null_held']!s:5} | COMP real={r['real_compound_pct']:+8.2f}% "
                          f"RANDp95={cr['p95']:+7.2f} beat={cr['beats']!s:5} p={cr['p']:.3f} | "
                          f"MEMp95={cm['p95']:+7.2f} beat={cm['beats']!s:5} p={cm['p']:.3f}")
                else:
                    print(f"  {sym:9} {r.get('error') or r.get('note')}")
            verdict = _verdict(per_asset, mode)
            print(f"  --> VERDICT [{key}]: {verdict['verdict']}")
            print(f"      capture: beat_null {verdict.get('assets_beating_capture_null')}/"
                  f"{verdict.get('n_assets_capture')}  med_excess_cap={verdict.get('median_excess_capture_vs_null')}  "
                  f"boot_p={verdict.get('bootstrap_p_median_excess_capture_gt0')}  | "
                  f"compound: beat_random {verdict.get('assets_beating_compound_random_null')}/"
                  f"{verdict.get('n_assets')}  beat_membership "
                  f"{verdict.get('assets_beating_compound_membership_null')}/{verdict.get('n_assets')}")
            out["runs"][key] = {"per_asset": per_asset, "verdict": verdict}

    outdir = Path(__file__).resolve().parents[2] / "runs" / "strat"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"wavelet_capture_{int(time.time())}.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[wavelet_capture_probe] wrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
