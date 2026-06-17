"""experiments/adaptive_ma/expert/er_gate_4h_bootstrap.py -- the EXPLICIT block-bootstrap p05>0 + jackknife
falsifier on top of the regime-matched (ER>0.4) firewall.

TASK (RED-team falsifier, the statistical-rigour layer): the regime-matched firewall (er_gate_4h.py) already
showed the ER-gated fixed-MA @ 4h does not clear the random-entry null's p95 on held-out. This script adds the
two tests the brief NAMES BY HAND -- block-bootstrap p05>0 and jackknife -- to decide, with confidence
intervals (not just point estimates), whether the MA entry TIMING is a genuine signal ABOVE a regime-matched
null whose entries are drawn ONLY from the SAME ER>0.4 bars.

Two-sided contract (the signal is real ONLY if ALL hold on HELD-OUT = OOS+UNSEEN):
  (1) SIGNAL ROBUSTLY POSITIVE  : block-bootstrap p05 of the REAL strategy's held-out trade returns > 0.
  (2) NOT ONE-TRADE LUCK        : jackknife jk2>0 AND jk3>0 (compound survives dropping the 2/3 biggest trades).
  (3) SIGNAL > NULL             : real held-out compound > regime-matched null p95  AND
                                  block-bootstrap p05 of the (real - null) paired difference > 0.

NULL = regime-matched: random entries drawn ONLY from held-out bars where ER>0.4 (the SAME regime the gate
selects), held for durations sampled from the strategy's OWN held-out hold distribution, at the SAME taker
cost. This isolates WITHIN-regime entry TIMING from gate/regime SELECTION.

Reuses the EXACT falsifier apparatus (er_gate_4h.build_cols/load_4h, ER>0.4 gate, 8/21 EMA, ATR-trail) and the
battery's audited primitives (strat.battery.block_bootstrap_p05_p95 + jackknife -- F6/F7 hardened). The
positive control (positive_control_4h.py) already proved this firewall HAS POWER at 4h, so a null here is a
real refutation, not a dead test.

All past-only (MA/ER close-of-bar; next-open fill; ATR read as atr[j-1]). No emoji (cp1252). numpy/pandas only.

RWYB:  python experiments/adaptive_ma/expert/er_gate_4h_bootstrap.py [--quick] [--probe BTCUSDT]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import er_gate_4h as EG  # noqa: E402  reuse exact apparatus: load_4h, build_cols, ER>0.4 gate, ATR-trail policy
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy  # noqa: E402
from strat.battery import block_bootstrap_p05_p95, jackknife, compound, expectancy  # noqa: E402

HELD = ("OOS", "UNSEEN")
N_NULL_TRADES = 6000   # Monte-Carlo size for the pooled-null trade-return distribution (per asset cap)
N_BOOT = 2000          # bootstrap resamples


def held_trade_returns(res, durations_too: bool = False):
    """Real per-trade net returns on the held-out windows (OOS+UNSEEN), time-ordered by entry index."""
    sub = [t for t in res.trades if t["window"] in HELD]
    sub.sort(key=lambda t: t["entry_idx"])
    rets = np.array([t["net_pnl"] for t in sub], float)
    if durations_too:
        durs = np.array([max(1, int(t["duration_bars"])) for t in sub], int)
        return rets, durs
    return rets


def regime_null_trade_returns(harness, durs: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Monte-Carlo the regime-matched null's per-trade net returns on HELD-OUT bars.

    Draw `n_draws` random entries from held-out bars where ER>0.4 (the harness gate, filter_col='er'),
    each held for a duration sampled from the strategy's OWN held-out hold distribution, filled next-open,
    same taker cost. Returns the per-trade net-return sample (the null's trade-level distribution)."""
    df = harness.df
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    cost = float(harness.spec.cost_rt)
    # held-out, gate-ON eligible entry bars (need room for fill i+1 and an exit)
    er = df[harness.spec.filter_col].to_numpy(float)
    thr = float(harness.spec.filter_val)
    elig = []
    for i in range(1, n - 2):
        ts = pd.Timestamp(dates.iloc[i])
        if harness._window_label(ts) in HELD and np.isfinite(er[i]) and er[i] > thr:
            elig.append(i)
    elig = np.array(elig, int)
    if elig.size == 0 or durs.size == 0:
        return np.array([], float)
    rng = np.random.default_rng(seed)
    ent = rng.choice(elig, size=n_draws, replace=True)
    dd = rng.choice(durs, size=n_draws, replace=True)
    nets = []
    for e, d in zip(ent, dd):
        ef = e + 1
        xf = min(ef + int(d), n - 1)
        if xf <= ef:
            continue
        nets.append(opens[xf] / opens[ef] - 1.0 - cost)
    return np.array(nets, float)


def diff_bootstrap_p05(real: np.ndarray, null: np.ndarray, n_boot: int, block: int, seed: int) -> dict:
    """Block-bootstrap p05 of the paired (real_compound - null_compound) difference.

    Each iteration: block-resample the REAL held-out trade series (preserving local time order) -> real_comp;
    i.i.d.-resample an equal-size draw from the NULL trade pool -> null_comp; record the difference. p05>0 means
    the real strategy's held-out compound robustly EXCEEDS the regime-matched null. Real uses block resampling
    (autocorrelated trade series); null is an i.i.d. Monte-Carlo pool so a plain resample is correct for it."""
    if real.size < block * 2 or null.size < 2:
        return {"p05": None, "p50": None, "p95": None, "frac_pos": None}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(real.size / block))
    sp = real.size - block + 1
    m = real.size  # match the null book to the real held-out trade count
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, sp, size=nb)
        rs = np.concatenate([real[st:st + block] for st in idx])[:real.size]
        rc = (np.prod(1.0 + rs) - 1.0) * 100
        ns = null[rng.integers(0, null.size, size=m)]
        nc = (np.prod(1.0 + ns) - 1.0) * 100
        diffs[b] = rc - nc
    return {"p05": round(float(np.percentile(diffs, 5)), 2),
            "p50": round(float(np.percentile(diffs, 50)), 2),
            "p95": round(float(np.percentile(diffs, 95)), 2),
            "frac_pos": round(float((diffs > 0).mean()), 3)}


def run_one(df: pd.DataFrame, seed: int = 7) -> dict | None:
    """Build the ER-gated 4h strategy, score it, and run the block-bootstrap p05 + jackknife vs regime null."""
    cols = EG.build_cols(df, entry_style="state")
    if int(cols["entry"].sum()) < 4:
        return None
    policy = ExitPolicy(atr_trail_mult=EG.ATR_TRAIL_MULT, atr_col="atr", max_hold_bars=EG.MAX_HOLD)
    h = SetupHarness(cols, "entry", policy, EG.WIN, cost_rt=EG.TAKER, regime_match_on_entry=False)
    h.spec.filter_col, h.spec.filter_op, h.spec.filter_val = "er", "gt", EG.ER_THRESH
    res = h.run()

    real, durs = held_trade_returns(res, durations_too=True)
    if real.size < 6:           # too few held-out trades to bootstrap meaningfully
        return {"skip": "n_held<6", "n_held": int(real.size)}

    real_comp = compound(real)
    bb_real = block_bootstrap_p05_p95(real, block=5, n=N_BOOT, seed=seed)
    jk2, jk3 = jackknife(real, 2), jackknife(real, 3)

    null = regime_null_trade_returns(h, durs, n_draws=min(N_NULL_TRADES, max(2000, real.size * 200)), seed=seed + 1)
    # null compound for an equal-size random book (point estimate) + its bootstrap band
    bb_null = block_bootstrap_p05_p95(null, block=5, n=N_BOOT, seed=seed + 2) if null.size >= 10 else {"p05": None, "p50": None, "p95": None}
    # null distribution of equal-size (real.size) books -> p95 the real must clear
    null_book_p95 = null_book_p50 = None
    if null.size >= real.size:
        rng = np.random.default_rng(seed + 3)
        books = np.array([compound(null[rng.integers(0, null.size, size=real.size)]) for _ in range(N_BOOT)])
        null_book_p50, null_book_p95 = float(np.percentile(books, 50)), float(np.percentile(books, 95))

    dbp = diff_bootstrap_p05(real, null, n_boot=N_BOOT, block=5, seed=seed + 4)

    # the three contract conditions on HELD-OUT
    cond1_p05_pos = bool(bb_real["p05"] is not None and bb_real["p05"] > 0)
    cond2_jk = bool(jk2 > 0 and jk3 > 0)
    cond3_beats = bool(null_book_p95 is not None and real_comp > null_book_p95 and dbp["p05"] is not None and dbp["p05"] > 0)
    genuine = bool(cond1_p05_pos and cond2_jk and cond3_beats)

    return {
        "n_held": int(real.size),
        "real_held_compound_pct": round(real_comp, 2),
        "real_expectancy_pct": round(expectancy(real), 3),
        "bb_real_p05": bb_real["p05"], "bb_real_p50": bb_real["p50"], "bb_real_p95": bb_real["p95"],
        "jk2": round(jk2, 2), "jk3": round(jk3, 2),
        "null_expectancy_pct": round(expectancy(null), 3) if null.size else None,
        "null_book_p50": round(null_book_p50, 2) if null_book_p50 is not None else None,
        "null_book_p95": round(null_book_p95, 2) if null_book_p95 is not None else None,
        "diff_real_minus_null_p05": dbp["p05"], "diff_p50": dbp["p50"], "diff_frac_pos": dbp["frac_pos"],
        "cond1_bootstrap_p05_pos": cond1_p05_pos,
        "cond2_jackknife_pos": cond2_jk,
        "cond3_beats_null": cond3_beats,
        "GENUINE_SIGNAL": genuine,
    }


def main(quick: bool, probe: str | None):
    loader = ChimeraLoader()
    if probe:
        df = EG.load_4h(loader, probe)
        if df is None:
            print(f"[probe {probe}] no data")
            return
        rec = run_one(df)
        print(f"[probe {probe}]  bars={len(df)}")
        print(json.dumps(rec, indent=2, default=str))
        return

    syms = UniverseLoader.load().list("u100")
    if quick:
        syms = syms[:20]
    print(f"[ER-gate 4h BOOTSTRAP falsifier] u100 4h | assets={len(syms)} | block-bootstrap p05 + jackknife vs "
          f"regime-matched(ER>{EG.ER_THRESH}) null | taker={EG.TAKER}", flush=True)

    per_asset = {}
    pooled_real, pooled_null = [], []
    for k, s in enumerate(syms, 1):
        df = EG.load_4h(loader, s)
        if df is None or len(df) < 1000:
            continue
        try:
            rec = run_one(df, seed=7)
        except Exception as e:  # noqa: BLE001
            rec = {"error": repr(e)[:160]}
        if rec is None:
            continue
        per_asset[s] = rec
        # accumulate pooled trade returns (re-derive cheaply for the pool)
        if "GENUINE_SIGNAL" in rec:
            cols = EG.build_cols(df, entry_style="state")
            policy = ExitPolicy(atr_trail_mult=EG.ATR_TRAIL_MULT, atr_col="atr", max_hold_bars=EG.MAX_HOLD)
            h = SetupHarness(cols, "entry", policy, EG.WIN, cost_rt=EG.TAKER, regime_match_on_entry=False)
            h.spec.filter_col, h.spec.filter_op, h.spec.filter_val = "er", "gt", EG.ER_THRESH
            r2 = h.run()
            rr, dd = held_trade_returns(r2, durations_too=True)
            if rr.size:
                pooled_real.append(rr)
                pooled_null.append(regime_null_trade_returns(h, dd, n_draws=min(N_NULL_TRADES, max(2000, rr.size * 200)), seed=11))
        if k % 10 == 0:
            print(f"[run] {k}/{len(syms)} processed, {len(per_asset)} evaluated", flush=True)

    evaluated = {s: r for s, r in per_asset.items() if "GENUINE_SIGNAL" in r}
    n_eval = len(evaluated)
    genuine = [s for s, r in evaluated.items() if r["GENUINE_SIGNAL"]]
    cond1 = [s for s, r in evaluated.items() if r["cond1_bootstrap_p05_pos"]]
    cond3 = [s for s, r in evaluated.items() if r["cond3_beats_null"]]

    # POOLED universe-level bootstrap (per-trade expectancy is the pooling-safe statistic)
    pr = np.concatenate(pooled_real) if pooled_real else np.array([])
    pn = np.concatenate(pooled_null) if pooled_null else np.array([])
    pooled = {}
    if pr.size and pn.size:
        rng = np.random.default_rng(7)
        # i.i.d. bootstrap of per-trade expectancy (mean) -- pooling across assets makes compound ordering
        # arbitrary, so expectancy (mean net return per trade) is the honest pooled statistic.
        exp_r = np.array([pr[rng.integers(0, pr.size, size=pr.size)].mean() for _ in range(N_BOOT)]) * 100
        exp_n = np.array([pn[rng.integers(0, pn.size, size=pn.size)].mean() for _ in range(N_BOOT)]) * 100
        diff = exp_r - exp_n
        pooled = {
            "n_real_trades": int(pr.size), "n_null_trades": int(pn.size),
            "real_expectancy_pct": round(float(pr.mean() * 100), 4),
            "null_expectancy_pct": round(float(pn.mean() * 100), 4),
            "real_expectancy_p05": round(float(np.percentile(exp_r, 5)), 4),
            "real_expectancy_p95": round(float(np.percentile(exp_r, 95)), 4),
            "null_expectancy_p05": round(float(np.percentile(exp_n, 5)), 4),
            "null_expectancy_p95": round(float(np.percentile(exp_n, 95)), 4),
            "diff_real_minus_null_p05": round(float(np.percentile(diff, 5)), 4),
            "diff_real_minus_null_p50": round(float(np.percentile(diff, 50)), 4),
            "diff_frac_pos": round(float((diff > 0).mean()), 3),
            "real_jk3_on_pooled": round(jackknife(pr, 3), 2),
        }

    agg = {
        "n_assets_evaluated": n_eval,
        "n_GENUINE_SIGNAL": len(genuine), "assets_genuine": genuine,
        "n_cond1_bootstrap_p05_pos": len(cond1),
        "n_cond3_beats_null": len(cond3),
        "pooled": pooled,
    }
    out = {"config": {"cadence": "4h", "ma": "8/21EMA", "er_thresh": EG.ER_THRESH, "atr_trail": EG.ATR_TRAIL_MULT,
                      "max_hold": EG.MAX_HOLD, "taker": EG.TAKER, "n_boot": N_BOOT, "null": "regime_matched_ER>0.4",
                      "held_windows": list(HELD)},
           "aggregate": agg, "per_asset": per_asset}
    print("\n" + "=" * 80)
    print(f"ER-GATED FIXED-MA @4h -- BLOCK-BOOTSTRAP p05>0 + JACKKNIFE vs REGIME-MATCHED NULL | {n_eval} assets")
    print("=" * 80)
    print(f"  GENUINE SIGNAL (p05>0 AND jk2,jk3>0 AND beats regime-null on held-out): {len(genuine)}/{n_eval}")
    print(f"    cond1 (block-bootstrap p05 of real held-out > 0)       : {len(cond1)}/{n_eval}")
    print(f"    cond3 (real held-out compound > null p95 AND diff p05>0): {len(cond3)}/{n_eval}")
    if pooled:
        print(f"\n  POOLED (universe, {pooled['n_real_trades']} real vs {pooled['n_null_trades']} null held-out trades):")
        print(f"    real per-trade expectancy = {pooled['real_expectancy_pct']}%  "
              f"[p05 {pooled['real_expectancy_p05']}, p95 {pooled['real_expectancy_p95']}]")
        print(f"    null per-trade expectancy = {pooled['null_expectancy_pct']}%  "
              f"[p05 {pooled['null_expectancy_p05']}, p95 {pooled['null_expectancy_p95']}]")
        print(f"    (real - null) expectancy p05 = {pooled['diff_real_minus_null_p05']}%  "
              f"(frac_pos {pooled['diff_frac_pos']})  -> signal>null robust iff p05>0")
    print("=" * 80)
    outpath = Path(__file__).resolve().parent / ("er_gate_4h_bootstrap_quick.json" if quick else "er_gate_4h_bootstrap_u100.json")
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 20 u100 assets only")
    ap.add_argument("--probe", type=str, default=None, help="single-asset probe (e.g. BTCUSDT)")
    args = ap.parse_args()
    main(quick=args.quick, probe=args.probe)
