"""src/strat/ma_exposure_control.py -- THE DECISIVE SKEPTIC TEST for the MA move-CATCH "drawdown-insurance" claim.

THE QUESTION (skeptic 1, the single most important check). The 8 MA move-catch engines, at coarse TFs (1d/4h),
HALVE the drawdown (median maxDD ~ -35% vs TRUE-HODL ~ -82%) and "preserve the 2022 bear" (+57pp vs the
always-long index). The claimed deliverable is "deployable drawdown-insurance de-risked-beta". BUT: the engine's
AVERAGE GROSS EXPOSURE is only ~0.40 -- it sits in CASH ~60% of the time. A long-only book that is flat 60% of
the time MECHANICALLY has a far lower drawdown -- that is NOT timing skill and NOT a success.

So: is the drawdown reduction TIMING SKILL (the engine is in cash at the RIGHT times) or trivially "lower
exposure -> mechanically lower drawdown"? This module builds the honest EXPOSURE-MATCHED control and quantifies
the SKILL DELTA = (engine's drawdown-reduction) - (a dumb same-exposure book's drawdown-reduction).

THE CONTROLS (both hold the SAME average exposure phi as the engine, with NO MA timing):

  CONTROL A -- CFAL (Constant-Fraction Always-Long).  At EVERY bar deploy exactly phi of capital into the
    rebalanced EW-listed index (1/N_listed per listed name), the rest in CASH. Same average exposure phi, ZERO
    timing decision, CHURN-IMMUNE (the only turnover is the index's own listing drift -- no in/out toggling).
    This is the cleanest "just hold phi of the market, always" dumb book. Its drawdown is the PURELY MECHANICAL
    consequence of running at exposure phi. (maxDD_CFAL ~= phi * maxDD_alwayslong by construction.)

  CONTROL B -- REHTM (Random-Entry Hold-To-Maturity).  Per asset, draw random NON-OVERLAPPING entry bars and HOLD
    each for the engine's hold length H = window_days * bars_per_day, sized so the realized per-asset average
    exposure matches the engine's. HOLD-TO-MATURITY (entries fixed, no per-bar reselection) = the CHURN-IMMUNE
    null mandated by dead-list A9 (the same-exposure SHUFFLE / per-bar-reselection null is BROKEN -- churn-
    penalized -- and must NOT be used). Averaged over N_SEED random books. This is "deploy the same amount, at the
    same hold-length, but at RANDOM times instead of MA-timed times".

THE VERDICT METRIC -- the SKILL DELTA (extra drawdown-reduction beyond exposure-matching):
    dd_reduction(book)        = maxDD_HODL - maxDD_book           (pp of drawdown the book avoids vs full HODL)
    skill_delta_vs_CFAL_pp    = dd_reduction(engine) - dd_reduction(CFAL)
    skill_delta_vs_REHTM_pp   = dd_reduction(engine) - dd_reduction(REHTM)
    bear_skill_delta_vs_CFAL  = bear2022_preservation(engine) - bear2022_preservation(CFAL)   (same idea, bear leg)
  POSITIVE & material  => the engine is in cash at the RIGHT (drawdown) times = REAL TIMING SKILL = success.
  ~0                    => the drawdown reduction is the trivial mechanical low-exposure effect = NOT a success.

ACCOUNTING: reuses ma_wealth_driver's verified-correct machinery byte-for-byte (same _load_asset panel, same
  _compound MtM-no-double-count, same _window_compound bear/up masks, same maker 0.0006 RT, same DEV wall,
  same TRAIN+VAL band selection -- OOS never read). The controls are evaluated on the IDENTICAL admitted panel,
  identical bars, identical cost, identical reconciliation. Apples-to-apples by construction.

RWYB (smoke): python -m strat.ma_exposure_control --ma-type EMA --tf 1d --window 7d
No emoji (cp1252). No git commit (meta reviews).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl                                              # noqa: E402
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402
from strat.ma_wealth_driver import (                                      # noqa: E402  reuse verified machinery
    _u100_symbols, _load_asset, _compound, _window_compound, _true_hodl,
    MAKER_RT, BEAR_2022_START, BEAR_2022_END,
)

N_SEED = 24                                                               # REHTM random books to average


def _maxdd_from_net(net: np.ndarray) -> float:
    eq = np.cumprod(1.0 + net)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def _compound_pct_from_net(net: np.ndarray) -> float:
    if len(net) == 0:
        return 0.0
    return float((np.prod(1.0 + net) - 1.0) * 100)


def _book_metrics(W: pd.DataFrame, R: pd.DataFrame, cost_rt: float,
                  bear_mask: np.ndarray, up_mask: np.ndarray,
                  along_bear: float, along_up: float) -> dict:
    """Full MtM read of a weight book on the shared panel: compound, maxDD, avg gross exposure, and the
    2022-bear / up-market window compounds (and their pp-vs-always-long deltas, computed by the caller's
    along_* references). Same accounting as ma_wealth_driver._compound (MtM-no-double-count, lag-1, maker)."""
    pct, net, eq, avg_gross, dd = _compound(W, R, cost_rt)
    bear = _window_compound(W, R, cost_rt, bear_mask)
    up = _window_compound(W, R, cost_rt, up_mask)
    return {
        "compound_pct": round(pct, 2),
        "maxdd_pct": round(dd, 1),
        "avg_gross": round(avg_gross, 4),
        "bear2022_compound_pct": round(bear, 2) if np.isfinite(bear) else None,
        "upmkt_compound_pct": round(up, 2) if np.isfinite(up) else None,
        "bear2022_preservation_pp": round(bear - along_bear, 2)
            if (np.isfinite(bear) and np.isfinite(along_bear)) else None,
        "upmkt_timing_alpha_pp": round(up - along_up, 2)
            if (np.isfinite(up) and np.isfinite(along_up)) else None,
    }


def run_control(ma_type: str, tf: str, window: str, *, cost_rt: float = MAKER_RT,
                n_seed: int = N_SEED, seed0: int = 0, rehtm_hold_mode: str = "engine_run",
                verbose: bool = False) -> dict:
    """Build the engine + the two exposure-matched dumb controls (CFAL, REHTM) on the SAME admitted u100 panel,
    same DEV window, same cost. Return the SKILL DELTA (extra drawdown-reduction / bear-preservation the engine
    earns BEYOND exposure-matching). All three legs share the exact ma_wealth_driver machinery.

    rehtm_hold_mode: the REHTM random-segment hold length.
      'engine_run' (DEFAULT, most apples-to-apples) = the engine's EMPIRICAL mean hold-run length on this cell
        (the engine holds long continuous trend-runs, ~23 bars at EMA|1d, NOT the 7-bar lookback). Matching the
        random book's hold length to the engine's run length is the adversarially-fair churn match -- it gives
        REHTM the SAME number of entries/exits and the SAME exposure-to-a-trend duration, so any remaining
        drawdown gap is TIMING (entering at the right times), not a hold-length artifact.
      'window' = hold = window_days*bpd (the lookback). Chops the same exposure into more, shorter segments
        (more churn) -- kept as a cross-check.
    """
    window_days = int(str(window).rstrip("d"))
    bpd = fl.BARS_PER_DAY[tf]
    hold_bars = window_days * bpd
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)

    # --- band selection + ensemble hold provider = the COMMITTED capture infra (TRAIN+VAL only; OOS unread) ---
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        return {"error": f"sub-band < 2 for {ma_type}|{tf}", "selection": sel_meta}
    provider = make_hold_provider(sub_band, ma_type)

    # --- load the SAME u100 panel as ma_wealth_driver (identical _load_asset, identical 2h synth, DEV wall) ---
    closes, holds = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        hh = provider(sym, o, h, l, c).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]
    if not closes:
        return {"error": f"no assets loaded for {ma_type}|{tf}|{window}"}

    cp = pd.DataFrame(closes).sort_index()
    _max_rows = max((s.notna().sum() for s in closes.values()), default=0)
    assert len(cp) <= 1.5 * _max_rows + 5, f"date-alignment regression: {len(cp)} >> {_max_rows}"
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)
    assert cp.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"

    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)

    bidx = cp.index
    bear_mask = np.asarray((bidx >= pd.Timestamp(BEAR_2022_START)) & (bidx < pd.Timestamp(BEAR_2022_END)))
    up_mask = ~bear_mask
    n_bars, n_assets = cp.shape

    # ============================ LEG: ALWAYS-LONG INDEX (phi = 1.0 reference) ============================
    # the daily-rebalanced EW index (1/N_listed per LISTED name). NOT a buy-hold; the timing-skill reference and
    # the bear/up baseline that every preservation-pp is measured against (same as ma_wealth_driver).
    w_along = listed.div(n_listed, axis=0).fillna(0.0)
    along_bear = _window_compound(w_along, R, cost_rt, bear_mask)
    along_up = _window_compound(w_along, R, cost_rt, up_mask)
    along = _book_metrics(w_along, R, cost_rt, bear_mask, up_mask, along_bear, along_up)

    # ============================ LEG: ENGINE (MA-timed; phi = avg exposure) ============================
    w_eng = held.div(n_listed, axis=0).fillna(0.0)
    eng = _book_metrics(w_eng, R, cost_rt, bear_mask, up_mask, along_bear, along_up)
    phi = eng["avg_gross"]                                          # the exposure ALL controls must match

    # ============================ LEG: TRUE HODL (phi -> 1, the primary baseline) ============================
    hodl_pct, hodl_eq, hodl_dd, n_total = _true_hodl(cp, R, listed, cost_rt)
    hodl_dd = round(hodl_dd, 1)

    # ============================ CONTROL A: CFAL (constant-fraction always-long) ============================
    # deploy exactly phi of capital into the rebalanced EW index every bar; rest in cash. Same avg exposure phi,
    # ZERO timing, CHURN-IMMUNE (only the index's listing-drift turnover, no in/out toggle).
    w_cfal = w_along * phi
    cfal = _book_metrics(w_cfal, R, cost_rt, bear_mask, up_mask, along_bear, along_up)

    # ============================ CONTROL B: REHTM (random-entry hold-to-maturity) ============================
    # per asset, random NON-OVERLAPPING entries held H=hold_bars (the engine hold length), at the engine's
    # fixed-N EW sizing (1/N_listed per held name). HOLD-TO-MATURITY = churn-immune (A9-safe). The PER-ASSET
    # number of entry-segments is matched to the engine's realized holding so realized exposure ~= phi. Average
    # the maxDD / bear-preservation over n_seed random books.
    eng_hold_bool = (held.to_numpy() > 0)                          # (n_bars, n_assets) engine held mask
    listed_np = listed.to_numpy() > 0
    eng_held_bars_per_asset = eng_hold_bool.sum(axis=0)            # how many bars the engine held each asset
    col_names = list(cp.columns)

    # --- the REHTM segment hold length (apples-to-apples churn match) ---
    # measure the engine's EMPIRICAL hold-run length (consecutive-held bars), so the random book holds for the
    # same duration the engine actually holds (the engine rides long trend-runs ~23 bars at EMA|1d, NOT 7).
    run_lens = []
    for j in range(n_assets):
        col = eng_hold_bool[:, j]
        i = 0
        L = len(col)
        while i < L:
            if col[i]:
                k = i
                while k < L and col[k]:
                    k += 1
                run_lens.append(k - i)
                i = k
            else:
                i += 1
    eng_mean_run = int(round(float(np.mean(run_lens)))) if run_lens else hold_bars
    seg_hold = max(1, eng_mean_run) if rehtm_hold_mode == "engine_run" else hold_bars

    rehtm_runs = []
    for s in range(n_seed):
        rng = np.random.default_rng(seed0 + 1000 + s)
        rh = np.zeros((n_bars, n_assets), dtype=bool)
        for j in range(n_assets):
            valid = np.flatnonzero(listed_np[:, j])
            if valid.size == 0:
                continue
            lo, hi = int(valid[0]), int(valid[-1])                # listed span [lo, hi]
            span = hi - lo + 1
            target_held = int(eng_held_bars_per_asset[j])         # match the engine's bars-held on this asset
            if target_held <= 0 or span < 2:
                continue
            n_seg = max(1, int(round(target_held / seg_hold)))    # #hold-to-maturity segments to place
            placed = 0
            attempts = 0
            occupied = np.zeros(span, dtype=bool)
            while placed < n_seg and attempts < n_seg * 12:
                attempts += 1
                start = int(rng.integers(0, span))                # entry offset within the listed span
                end = min(start + seg_hold, span)                 # hold seg_hold bars (clipped at listed end)
                if occupied[start:end].any():                     # NON-OVERLAPPING (hold-to-maturity, A9-safe)
                    continue
                occupied[start:end] = True
                rh[lo + start: lo + end, j] = True
                placed += 1
        rh = rh & listed_np                                       # only while listed
        rh_df = pd.DataFrame(rh.astype(float), index=cp.index, columns=col_names)
        w_re = rh_df.div(n_listed, axis=0).fillna(0.0)            # same fixed-N EW sizing as the engine
        m = _book_metrics(w_re, R, cost_rt, bear_mask, up_mask, along_bear, along_up)
        rehtm_runs.append(m)

    def _agg(key):
        vals = [r[key] for r in rehtm_runs if r.get(key) is not None]
        return (round(float(np.mean(vals)), 2), round(float(np.std(vals)), 2)) if vals else (None, None)

    re_dd_mean, re_dd_std = _agg("maxdd_pct")
    re_phi_mean, re_phi_std = _agg("avg_gross")
    re_bear_mean, re_bear_std = _agg("bear2022_preservation_pp")
    re_comp_mean, re_comp_std = _agg("compound_pct")
    re_up_mean, re_up_std = _agg("upmkt_timing_alpha_pp")

    # ============================ THE VERDICT: SKILL DELTAS ============================
    # maxDD is stored NEGATIVE (deeper = more negative). "drawdown avoided vs full HODL" = a POSITIVE magnitude =
    # book_dd - hodl_dd  (e.g. engine -37.5 minus HODL -82.3 = +44.8pp shallower). A bigger positive = the book
    # avoids more of HODL's drawdown.
    dd_red_engine = round(eng["maxdd_pct"] - hodl_dd, 1)           # pp of full-HODL drawdown the engine avoids (+)
    dd_red_cfal = round(cfal["maxdd_pct"] - hodl_dd, 1)
    dd_red_rehtm = round(re_dd_mean - hodl_dd, 1) if re_dd_mean is not None else None

    skill_delta_vs_cfal_pp = round(dd_red_engine - dd_red_cfal, 1)
    skill_delta_vs_rehtm_pp = (round(dd_red_engine - dd_red_rehtm, 1) if dd_red_rehtm is not None else None)
    bear_skill_delta_vs_cfal_pp = (round(eng["bear2022_preservation_pp"] - cfal["bear2022_preservation_pp"], 2)
                                   if (eng["bear2022_preservation_pp"] is not None
                                       and cfal["bear2022_preservation_pp"] is not None) else None)
    bear_skill_delta_vs_rehtm_pp = (round(eng["bear2022_preservation_pp"] - re_bear_mean, 2)
                                    if (eng["bear2022_preservation_pp"] is not None
                                        and re_bear_mean is not None) else None)

    # REHTM maxDD as a +/- band (mean +/- 1 std): is the engine OUTSIDE the random band (skill) or inside (luck)?
    engine_dd_z = (round((eng["maxdd_pct"] - re_dd_mean) / re_dd_std, 2)
                   if (re_dd_std and re_dd_std > 1e-9) else None)

    verdict = "INCONCLUSIVE"
    if skill_delta_vs_cfal_pp is not None and skill_delta_vs_rehtm_pp is not None:
        # SUCCESS = engine reduces drawdown MATERIALLY more than BOTH dumb same-exposure controls.
        material = 5.0                                            # pp threshold for "material" extra DD-reduction
        if skill_delta_vs_cfal_pp >= material and skill_delta_vs_rehtm_pp >= material:
            verdict = "TIMING_SKILL (engine beats same-exposure dumb controls -> real drawdown-insurance)"
        elif skill_delta_vs_cfal_pp <= 2.0 and skill_delta_vs_rehtm_pp <= 2.0:
            verdict = "TRIVIAL_LOW_EXPOSURE (drawdown reduction ~= mechanical exposure effect -> NOT a success)"
        else:
            verdict = "PARTIAL (some extra DD-reduction but below the material bar; weak skill)"

    out = {
        "cell": f"{ma_type}|{tf}|{window}",
        "ma_type": ma_type, "tf": tf, "window": window, "hold_bars": hold_bars,
        "n_assets": int(n_assets), "n_bars": int(n_bars), "cost_rt": cost_rt,
        "engine_avg_exposure_phi": phi,
        # --- the three legs' drawdowns (the heart of the test) ---
        "maxdd_HODL_pct": hodl_dd,
        "maxdd_alwayslong_phi1_pct": along["maxdd_pct"],
        "maxdd_engine_pct": eng["maxdd_pct"],
        "maxdd_CFAL_pct": cfal["maxdd_pct"],
        "maxdd_REHTM_pct_mean": re_dd_mean,
        "maxdd_REHTM_pct_std": re_dd_std,
        "REHTM_realized_exposure_mean": re_phi_mean,
        "CFAL_realized_exposure": cfal["avg_gross"],
        "engine_mean_hold_run_bars": eng_mean_run,
        "REHTM_hold_mode": rehtm_hold_mode,
        "REHTM_seg_hold_bars": seg_hold,
        # CFAL mechanical floor: a phi-scaled rebalanced index ~ phi*HODL-index drawdown. Engine beating THIS = skill.
        "CFAL_mechanical_floor_check_phi_x_alwayslong_dd": round(phi * along["maxdd_pct"], 1),
        # --- drawdown-reduction decomposition ---
        "dd_reduction_engine_pp": dd_red_engine,
        "dd_reduction_CFAL_pp": dd_red_cfal,
        "dd_reduction_REHTM_pp": dd_red_rehtm,
        # --- THE SKILL DELTAS (the verdict numbers) ---
        "skill_delta_dd_vs_CFAL_pp": skill_delta_vs_cfal_pp,
        "skill_delta_dd_vs_REHTM_pp": skill_delta_vs_rehtm_pp,
        "engine_maxdd_z_vs_REHTM": engine_dd_z,
        # --- bear-preservation skill (the other claimed value) ---
        "bear2022_preservation_engine_pp": eng["bear2022_preservation_pp"],
        "bear2022_preservation_CFAL_pp": cfal["bear2022_preservation_pp"],
        "bear2022_preservation_REHTM_pp_mean": re_bear_mean,
        "bear2022_preservation_REHTM_pp_std": re_bear_std,
        "bear_skill_delta_vs_CFAL_pp": bear_skill_delta_vs_cfal_pp,
        "bear_skill_delta_vs_REHTM_pp": bear_skill_delta_vs_rehtm_pp,
        # --- compound + up-market context (does the cash-timing cost up-market return?) ---
        "compound_engine_pct": eng["compound_pct"],
        "compound_CFAL_pct": cfal["compound_pct"],
        "compound_REHTM_pct_mean": re_comp_mean,
        "upmkt_timing_alpha_engine_pp": eng["upmkt_timing_alpha_pp"],
        "upmkt_timing_alpha_CFAL_pp": cfal["upmkt_timing_alpha_pp"],
        "upmkt_timing_alpha_REHTM_pp_mean": re_up_mean,
        # --- meta ---
        "n_seed_REHTM": n_seed,
        "selection_rule": sel_meta.get("selection_rule"),
        "split": f"DEV (TRAIN+VAL, < {fl.DEV_END}); OOS+UNSEEN SEALED; band TRAIN+VAL only",
        "controls_note": (
            "CFAL = constant-fraction (phi) always-long rebalanced EW index, zero timing, churn-immune. "
            "REHTM = random-entry HOLD-TO-MATURITY (H=hold_bars) sized to engine exposure, A9-churn-immune "
            "(NOT the broken same-exposure per-bar shuffle), averaged over n_seed books. Both match the "
            "engine's avg exposure phi with NO MA timing. skill_delta = engine's extra DD-reduction beyond them."),
        "verdict": verdict,
    }
    if verbose:
        print(f"[control] {out['cell']}  phi={phi}  REHTM hold={rehtm_hold_mode}({seg_hold}b, eng-run={eng_mean_run}b)")
        print(f"          maxDD: HODL {hodl_dd}  engine {eng['maxdd_pct']}  CFAL {cfal['maxdd_pct']}  "
              f"REHTM {re_dd_mean}+/-{re_dd_std}")
        print(f"          dd-reduction: engine {dd_red_engine}pp  CFAL {dd_red_cfal}pp  REHTM {dd_red_rehtm}pp")
        print(f"          SKILL DELTA  vs CFAL {skill_delta_vs_cfal_pp}pp   vs REHTM {skill_delta_vs_rehtm_pp}pp"
              f"   (engine maxDD z vs REHTM band = {engine_dd_z})")
        print(f"          bear-preservation: engine {eng['bear2022_preservation_pp']}pp  CFAL "
              f"{cfal['bear2022_preservation_pp']}pp  REHTM {re_bear_mean}+/-{re_bear_std}  "
              f"=> bear skill delta vs CFAL {bear_skill_delta_vs_cfal_pp}pp  vs REHTM {bear_skill_delta_vs_rehtm_pp}pp")
        print(f"          VERDICT: {verdict}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_exposure_control")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--window", default="7d")
    ap.add_argument("--n-seed", type=int, default=N_SEED)
    ap.add_argument("--rehtm-hold-mode", default="engine_run", choices=["engine_run", "window"])
    ap.add_argument("--out", default=None, help="write JSON to this path")
    a = ap.parse_args()
    r = run_control(a.ma_type, a.tf, a.window, n_seed=a.n_seed,
                    rehtm_hold_mode=a.rehtm_hold_mode, verbose=True)
    print(json.dumps(r, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=str))
        print(f"[wrote] {a.out}")
    return 0 if "error" not in r else 2


if __name__ == "__main__":
    sys.exit(main())
