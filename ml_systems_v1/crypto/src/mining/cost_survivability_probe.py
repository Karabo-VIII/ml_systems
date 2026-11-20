"""src/mining/cost_survivability_probe.py -- C1: does the vol-EXPANSION lead survive transaction costs?

WHAT THIS PROBES
----------------
The mining run (src/mining/conditional.py :: vol_expansion_setup, P12b) reported the project's
STRONGEST descriptive lead: a vol-EXPANSION bar predicts a LARGER next move. Output magnitudes
(runs/mining/cond_vol_expansion_*.json):
    1d  : ratio 1.51  (next |ret| 581.9 vs 386.3 bps),  up-rate EXP 0.492 vs CALM 0.484
    4h  : ratio 1.81  (270.1 vs 149.0 bps),             up-rate 0.493 vs 0.484
    30m : ratio 2.02  (104.9 vs 51.8  bps),             up-rate 0.495 vs 0.481

EXACT TRIGGER (read from code, not guessed) -- src/mining/conditional.py:88-95:
    ret = log-return of close
    amag = |ret|
    rv[t]  = amag.rolling(20).mean()                          # local 20-bar realized vol
    med[t] = rv.expanding(50).median().shift(1)               # past-only threshold (no look-ahead)
    EXPANSION at bar t  <=>  rv[t] > 1.5 * med[t]
The mining then measured |ret[t+1]| conditional on the trigger. KEY: the magnitude is bigger, but
DIRECTION is a coin flip (up-rate ~0.49 ~= 0.50). A magnitude edge with NO direction edge cannot be
monetized by a naive long/short -- the signed expectation is ~0. This probe quantifies that.

THE REALIZABLE TRADE
--------------------
Enter at the trigger bar t (signal known at close of t, no look-ahead), exit by a simple fixed policy
(hold N bars). Two honest variants:
  (A) directional-long  : the naive realizable bot -- go LONG at trigger, capture SIGNED hold return.
                          This is what a real long-only spot/perp bot actually earns.
  (B) magnitude-ceiling : |hold return| -- the UPPER BOUND if you had an oracle for direction.
                          Not realizable (no direction edge exists), but bounds the edge from above.

NULL: random-entry. Same asset, same number of trades, same hold policy, entry bars drawn uniformly
at random from the eligible range. The "edge" is measured as EXCESS over this null (gross and net),
NOT as an absolute number -- a positive absolute net that does not beat random is NOT an edge.

COST MODEL (canonical, not hardcoded)
-------------------------------------
Sourced from src/strat/fill_model.py :: MODES (the repo's canonical post-hoc cost+fill realism layer,
itself derived from config/maker_cost_calibration.yaml). round-trip cost_rt and p_fill:
    taker             : cost_rt 0.0024, p_fill 1.00            (solid spot taker)
    maker_pessimistic : cost_rt 0.0010, p_fill 0.30, adverse 0.96
Per CLAUDE.md the MakerCostModel p_fill=0.80 default is OPTIMISTIC (empirical 0.21-0.40); we budget
p_fill in {0.25, 0.40, 0.80} and report all three. cost_rt for the maker leg = MODES['maker_pessimistic'].
p_fill scales the EXPECTED per-attempted-trade net: with prob p_fill the trade fills and earns
(gross - cost - adverse*|gross|); with prob (1-p_fill) it does not fill and earns 0 (no exposure).
So expected_net_per_attempt = p_fill * (gross - adverse*|gross| - cost_rt).

VERDICT: survives <=> expected NET excess-over-null per attempted trade > 0 at that p_fill.

RWYB. Reads via pipeline.chimera_loader.ChimeraLoader.load (NOT direct parquet). No emoji.
Run:  python src/mining/cost_survivability_probe.py --universe u10 --cadence 1d --hold 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.chimera_loader import ChimeraLoader
from pipeline.universe_loader import UniverseLoader

# Canonical cost source -- import the repo's MODES dict, do NOT hardcode fees.
from strat.fill_model import MODES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "mining"

# Trigger constants -- mirror src/mining/conditional.py:88-95 exactly.
RV_WINDOW = 20          # rolling mean of |ret| (local vol)
MED_MIN_PERIODS = 50    # expanding-median min periods
EXPANSION_MULT = 1.5    # rv > 1.5 * past-median => expansion
WARMUP = 50             # mining starts the scan at t=50

MAKER_COST_RT = MODES["maker_pessimistic"]["cost_rt"]   # canonical round-trip maker cost
MAKER_ADVERSE = MODES["maker_pessimistic"]["adverse"]   # canonical adverse-selection penalty
TAKER_COST_RT = MODES["taker"]["cost_rt"]               # solid spot-taker round-trip (no adverse, p_fill 1.0)


def _expansion_mask(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (expansion_bool[t], log_ret[t]) using the EXACT mining trigger (past-only, no look-ahead)."""
    ret = np.zeros(len(close))
    ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
    amag = np.abs(ret)
    rv = pd.Series(amag).rolling(RV_WINDOW).mean().to_numpy()
    med = pd.Series(rv).expanding(MED_MIN_PERIODS).median().shift(1).to_numpy()
    exp = np.zeros(len(close), dtype=bool)
    valid = np.isfinite(rv) & np.isfinite(med) & (med > 0)
    exp[valid] = rv[valid] > EXPANSION_MULT * med[valid]
    return exp, ret


def _hold_return(ret: np.ndarray, entry_t: int, hold: int) -> float | None:
    """Signed log return from entering at close of entry_t, holding `hold` bars (sum of forward log-rets)."""
    end = entry_t + hold
    if end >= len(ret):
        return None
    seg = ret[entry_t + 1: end + 1]
    if not np.all(np.isfinite(seg)):
        return None
    return float(np.sum(seg))   # log-return additivity over the hold


def _collect_trades(close: np.ndarray, hold: int):
    """Return (trig_signed, trig_abs, pool_signed, pool_abs) hold-return arrays for one asset.

    trig_*  = entries at expansion-trigger bars.
    pool_*  = hold-returns at EVERY eligible bar (the universe random entry draws from). The null is
              built downstream by averaging many same-size random draws from the per-asset pool, so a
              single lucky/unlucky draw cannot move the verdict (a single random draw of the null is
              itself ~as noisy as the trigger sample -- see the seed-sweep that motivated this).
    signed  = long hold return; abs = |hold return| (magnitude-capture ceiling).
    """
    exp, ret = _expansion_mask(close)
    n = len(close)
    last_entry = n - hold - 1            # need `hold` forward bars
    if last_entry <= WARMUP:
        return None
    eligible = np.arange(WARMUP, last_entry + 1)
    if eligible.size == 0:
        return None

    trig_idx = eligible[exp[eligible]]
    if trig_idx.size == 0:
        return None

    trig_signed, trig_abs = [], []
    for t in trig_idx:
        r = _hold_return(ret, int(t), hold)
        if r is not None:
            trig_signed.append(r)
            trig_abs.append(abs(r))
    if not trig_signed:
        return None

    pool_signed, pool_abs = [], []
    for t in eligible:
        r = _hold_return(ret, int(t), hold)
        if r is not None:
            pool_signed.append(r)
            pool_abs.append(abs(r))

    return (np.array(trig_signed), np.array(trig_abs),
            np.array(pool_signed), np.array(pool_abs))


def _net_after_cost(gross: np.ndarray, p_fill: float) -> float:
    """Expected NET per ATTEMPTED trade at the given p_fill, using the canonical maker cost+adverse.

    Per fill: net = gross - adverse*|gross| - cost_rt.  With prob (1-p_fill) no fill -> 0.
    Expected per attempt = p_fill * mean(net_per_fill).
    """
    if gross.size == 0:
        return float("nan")
    net_per_fill = gross - MAKER_ADVERSE * np.abs(gross) - MAKER_COST_RT
    return float(p_fill * np.mean(net_per_fill))


def run(universe: str, cadence: str, hold: int, seed: int, n_null: int = 2000) -> dict:
    cl = ChimeraLoader()
    u = UniverseLoader.load()
    syms = u.list(universe)
    rng = np.random.default_rng(seed)

    all_trig_signed, all_trig_abs = [], []
    all_pool_signed, all_pool_abs = [], []
    per_asset = {}
    used, skipped = [], []

    for sym in syms:
        try:
            df = cl.load(sym, cadence=cadence)
        except Exception as e:
            skipped.append((sym, f"load_error:{type(e).__name__}"))
            continue
        if df is None or "close" not in df.columns or df.height < 300:
            skipped.append((sym, "too_short_or_no_close"))
            continue
        close = df["close"].to_numpy().astype(float)
        out = _collect_trades(close, hold)
        if out is None:
            skipped.append((sym, "no_eligible_trigger"))
            continue
        ts, ta, ps, pa = out
        all_trig_signed.append(ts); all_trig_abs.append(ta)
        all_pool_signed.append(ps); all_pool_abs.append(pa)
        per_asset[sym] = dict(n_trig=int(ts.size),
                              n_pool=int(ps.size),
                              mean_signed_bps=round(1e4 * float(np.mean(ts)), 1),
                              mean_abs_bps=round(1e4 * float(np.mean(ta)), 1),
                              pool_signed_bps=round(1e4 * float(np.mean(ps)), 1),
                              pool_abs_bps=round(1e4 * float(np.mean(pa)), 1))
        used.append(sym)

    if not all_trig_signed:
        return {"universe": universe, "cadence": cadence, "hold": hold,
                "error": "no trades collected", "skipped": skipped}

    trig_signed = np.concatenate(all_trig_signed)
    trig_abs = np.concatenate(all_trig_abs)
    pool_signed = np.concatenate(all_pool_signed)
    pool_abs = np.concatenate(all_pool_abs)
    n_trades = int(trig_signed.size)
    n_pool = int(pool_signed.size)

    # MONTE-CARLO NULL: draw n_trades random entries from the eligible pool, n_null times. The null
    # statistic is the MEAN over draws (the EXPECTED random entry) -- a single draw is too noisy to
    # anchor a verdict (seed-sweep showed the single-draw null swinging 30 bps). Bootstrap p-value =
    # fraction of random draws whose mean >= the observed trigger mean (one-sided).
    boot_signed = np.empty(n_null)
    boot_abs = np.empty(n_null)
    for i in range(n_null):
        idx = rng.integers(0, n_pool, size=n_trades)
        boot_signed[i] = np.mean(pool_signed[idx])
        boot_abs[i] = np.mean(pool_abs[idx])
    # null arrays for the cost layer = the FULL pool (cost is per-trade and linear, so the expected
    # net of a random entry uses the pool mean; we pass the pool itself to _net_after_cost).
    null_signed = pool_signed
    null_abs = pool_abs

    obs_signed = float(np.mean(trig_signed))
    obs_abs = float(np.mean(trig_abs))
    p_signed = float(np.mean(boot_signed >= obs_signed))   # one-sided: trigger long beats random?
    p_abs = float(np.mean(boot_abs >= obs_abs))            # trigger |move| beats random?

    # GROSS edges (bps). Null = MEAN of the Monte-Carlo random-entry distribution.
    gross_signed_bps = 1e4 * obs_signed
    gross_abs_bps = 1e4 * obs_abs
    null_signed_bps = 1e4 * float(np.mean(boot_signed))
    null_abs_bps = 1e4 * float(np.mean(boot_abs))
    null_signed_std_bps = 1e4 * float(np.std(boot_signed))
    null_abs_std_bps = 1e4 * float(np.std(boot_abs))
    # EXCESS over random null (the real edge).
    excess_signed_bps = gross_signed_bps - null_signed_bps
    excess_abs_bps = gross_abs_bps - null_abs_bps

    # TAKER baseline (the SOLID cost default per fill_model.py: cost_rt 0.0024, p_fill 1.0, NO adverse).
    # This gives the directional case a clean read free of the contested maker adverse=0.96 stress term.
    # Survives ONLY if the directional excess is also statistically significant (p_signed < 0.05);
    # a positive excess that does not beat random at p<0.05 is sampling noise (drift), not an edge.
    SIG = 0.05
    taker_signed_trig = obs_signed - TAKER_COST_RT
    taker_signed_null = float(np.mean(boot_signed)) - TAKER_COST_RT
    taker_signed_excess = taker_signed_trig - taker_signed_null
    taker_signed_survives = bool(taker_signed_excess > 0 and taker_signed_trig > 0 and p_signed < SIG)

    # Breakeven cost (round-trip) the GROSS magnitude edge can pay before net excess hits 0.
    # For the magnitude-ceiling variant the excess |move| over null is the budget for cost+adverse.
    breakeven_abs = float(np.mean(trig_abs) - np.mean(null_abs))   # in return units (fraction)

    # NET (after canonical cost) at the three p_fill budgets, for BOTH variants.
    p_fills = [0.25, 0.40, 0.80]
    net = {}
    for pf in p_fills:
        net_signed_trig = _net_after_cost(trig_signed, pf)
        net_signed_null = _net_after_cost(null_signed, pf)
        net_abs_trig = _net_after_cost(trig_abs, pf)
        net_abs_null = _net_after_cost(null_abs, pf)
        net[str(pf)] = dict(
            # directional-long (the realizable bot) -- requires p_signed<0.05 to count as survival
            signed_net_trig_bps=round(1e4 * net_signed_trig, 2),
            signed_net_null_bps=round(1e4 * net_signed_null, 2),
            signed_net_excess_bps=round(1e4 * (net_signed_trig - net_signed_null), 2),
            signed_survives=bool((net_signed_trig - net_signed_null) > 0 and net_signed_trig > 0
                                 and p_signed < 0.05),
            # magnitude-ceiling (oracle direction; upper bound) -- requires p_abs<0.05
            abs_net_trig_bps=round(1e4 * net_abs_trig, 2),
            abs_net_null_bps=round(1e4 * net_abs_null, 2),
            abs_net_excess_bps=round(1e4 * (net_abs_trig - net_abs_null), 2),
            abs_survives=bool((net_abs_trig - net_abs_null) > 0 and net_abs_trig > 0 and p_abs < 0.05),
        )

    return {
        "universe": universe,
        "cadence": cadence,
        "hold_bars": hold,
        "trigger": "rv20[t] > 1.5 * expanding-median(min50).shift(1)  [src/mining/conditional.py:88-95]",
        "cost_source": "src/strat/fill_model.py::MODES['maker_pessimistic']",
        "cost_rt": MAKER_COST_RT,
        "adverse_selection": MAKER_ADVERSE,
        "n_trades": n_trades,
        "n_pool_eligible": n_pool,
        "n_null_resamples": n_null,
        "n_assets_used": len(used),
        "assets_used": used,
        "skipped": skipped,
        "up_rate_trigger_signed": round(float(np.mean(trig_signed > 0)), 3),
        "up_rate_pool_signed": round(float(np.mean(pool_signed > 0)), 3),
        "bootstrap_p_signed_long_beats_random": round(p_signed, 4),
        "bootstrap_p_abs_move_beats_random": round(p_abs, 4),
        "gross": {
            "signed_long_bps": round(gross_signed_bps, 1),
            "abs_magnitude_bps": round(gross_abs_bps, 1),
            "null_signed_bps": round(null_signed_bps, 1),
            "null_signed_std_bps": round(null_signed_std_bps, 1),
            "null_abs_bps": round(null_abs_bps, 1),
            "null_abs_std_bps": round(null_abs_std_bps, 1),
            "excess_signed_bps": round(excess_signed_bps, 1),
            "excess_abs_bps": round(excess_abs_bps, 1),
        },
        "breakeven_roundtrip_cost_bps": round(1e4 * breakeven_abs, 1),
        "taker_baseline_directional": {
            "cost_source": "src/strat/fill_model.py::MODES['taker'] (solid; p_fill 1.0, no adverse)",
            "cost_rt": TAKER_COST_RT,
            "signed_net_trig_bps": round(1e4 * taker_signed_trig, 2),
            "signed_net_null_bps": round(1e4 * taker_signed_null, 2),
            "signed_net_excess_bps": round(1e4 * taker_signed_excess, 2),
            "survives": taker_signed_survives,
        },
        "net_after_cost_by_p_fill": net,
        "per_asset": per_asset,
        "verdict_note": (
            "DIRECTION is a coin flip at the trigger (up_rate ~0.49-0.51) -> the realizable directional-long "
            "edge is the SIGNED excess over a MONTE-CARLO random-entry null. The magnitude-ceiling variant is "
            "the UPPER bound earnable only WITH a (non-existent) direction edge. Survives = NET excess-over-null "
            "> 0 AND bootstrap p < 0.05 (a positive excess that is not significant vs random = drift, not edge)."
        ),
    }


def _print_report(res: dict) -> None:
    if "error" in res:
        print("[probe] ERROR:", res["error"])
        print("[probe] skipped:", res.get("skipped"))
        return
    print("=" * 78)
    print("VOL-EXPANSION COST-SURVIVABILITY PROBE (C1)")
    print("=" * 78)
    print(f"universe={res['universe']}  cadence={res['cadence']}  hold={res['hold_bars']} bars")
    print(f"trigger : {res['trigger']}")
    print(f"cost    : {res['cost_source']}  cost_rt={res['cost_rt']}  adverse={res['adverse_selection']}")
    print(f"n_trades={res['n_trades']}  n_pool_eligible={res['n_pool_eligible']}  "
          f"n_null_resamples={res['n_null_resamples']}  assets_used={res['n_assets_used']}")
    print(f"up-rate (trigger long)={res['up_rate_trigger_signed']}  "
          f"up-rate (pool/random)={res['up_rate_pool_signed']}   <- direction is a coin flip")
    print(f"bootstrap p (trigger long beats random) = {res['bootstrap_p_signed_long_beats_random']}   "
          f"p (trigger |move| beats random) = {res['bootstrap_p_abs_move_beats_random']}")
    g = res["gross"]
    print("-" * 78)
    print("GROSS (bps per trade):  null = Monte-Carlo random-entry MEAN (+/- std over draws)")
    print(f"  directional-long  : trigger {g['signed_long_bps']:>8}   "
          f"null {g['null_signed_bps']:>7} +/-{g['null_signed_std_bps']:<5}   EXCESS {g['excess_signed_bps']:>8}")
    print(f"  magnitude-ceiling : trigger {g['abs_magnitude_bps']:>8}   "
          f"null {g['null_abs_bps']:>7} +/-{g['null_abs_std_bps']:<5}   EXCESS {g['excess_abs_bps']:>8}")
    print(f"  breakeven round-trip cost the magnitude EXCESS can pay: {res['breakeven_roundtrip_cost_bps']} bps "
          f"(canonical cost_rt = {1e4*res['cost_rt']:.0f} bps + adverse {res['adverse_selection']})")
    print("-" * 78)
    tb = res["taker_baseline_directional"]
    print(f"TAKER baseline (directional-long, solid cost, no adverse; cost_rt={1e4*tb['cost_rt']:.0f} bps):")
    print(f"  net {tb['signed_net_trig_bps']:>8}   null {tb['signed_net_null_bps']:>8}   "
          f"EXCESS {tb['signed_net_excess_bps']:>8}   survives={'YES' if tb['survives'] else 'NO'}")
    print("-" * 78)
    print("NET after canonical cost, per attempted trade (bps), by p_fill budget:")
    for pf, d in res["net_after_cost_by_p_fill"].items():
        print(f"  p_fill={pf}")
        print(f"    directional-long  : net {d['signed_net_trig_bps']:>8}   null {d['signed_net_null_bps']:>8}   "
              f"EXCESS {d['signed_net_excess_bps']:>8}   survives={'YES' if d['signed_survives'] else 'NO'}")
        print(f"    magnitude-ceiling : net {d['abs_net_trig_bps']:>8}   null {d['abs_net_null_bps']:>8}   "
              f"EXCESS {d['abs_net_excess_bps']:>8}   survives={'YES' if d['abs_survives'] else 'NO'}")
    print("-" * 78)
    # overall verdict
    any_dir = any(d["signed_survives"] for d in res["net_after_cost_by_p_fill"].values())
    any_abs = any(d["abs_survives"] for d in res["net_after_cost_by_p_fill"].values())
    taker_dir = res["taker_baseline_directional"]["survives"]
    if any_dir or taker_dir:
        verdict = ("SURVIVES (directional-long realizable edge clears costs at >=1 cost scenario; "
                   f"taker={'YES' if taker_dir else 'no'}, maker_p_fill={'YES' if any_dir else 'no'})")
    elif any_abs:
        verdict = ("DOES NOT SURVIVE as a tradeable edge: only the (non-realizable) magnitude-ceiling "
                   "clears costs. No direction edge -> nothing to monetize.")
    else:
        verdict = "DOES NOT SURVIVE: neither directional nor magnitude-ceiling clears costs."
    print(f"VERDICT: {verdict}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d", help="1d|4h|1h|30m|15m")
    ap.add_argument("--hold", type=int, default=1, help="hold N bars (matches mining's t+1 at N=1)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-null", type=int, default=2000, help="Monte-Carlo random-entry resamples")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    res = run(args.universe, args.cadence, args.hold, args.seed, n_null=args.n_null)
    res["_elapsed_s"] = round(time.time() - t0, 1)
    _print_report(res)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUT / f"cost_survivability_{args.universe}_{args.cadence}_hold{args.hold}.json")
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
