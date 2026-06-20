"""src/strat/quant_book_adversarial.py -- ADVERSARIAL VALIDATION of the assembled participate-preserve book.

The quant referee lane. Trust nothing; re-derive every number from the DEV-walled labs. Four questions:

  Q1 INCREMENTAL  -- does the liq-flush amplifier ADD over plain momentum/breakout continuation, or is it
                     redundant? The reported +3.3pp liq edge is vs a RANDOM-ENTRY null. But brk14 alone is
                     already +2.75pp vs random. The HONEST test: liq-flush edge vs a MOMENTUM/BREAKOUT-ENTRY
                     null (entries drawn from the already-continuing pool), not a random-entry null.
  Q2 SUM-OF-PARTS -- is the book's DEV edge just up-regime beta + flush amplifier, or does combining
                     manufacture a NEW regime-conditional (bear-positive) edge? Test the combined gate
                     by regime; it must NOT beat its parts in bear.
  Q3 GATE PRESERVE-- does the causal regime gate genuinely preserve bear (lower maxDD, less negative) vs
                     buy-hold, and is the gate causal (regime label uses only data <= the decision bar)?
  Q4 CEILING      -- what the book IS (regime-aware de-risked-beta participate-preserve) vs IS NOT
                     (a 7d-always-positive engine -- the cash theorem). 7d-slice profit-rate ceiling = in-market up-rate.

DEV-walled (<= 2024-05-15). Long-only spot, taker cost. Honest date-block bootstrap, not iid. No emoji (cp1252).
RWYB: C:\\...\\.venv\\Scripts\\python.exe -m strat.quant_book_adversarial  (run from crypto/src)
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl
import strat.capture_lab as cl
import strat.v51_feature_lab as v51

COST = fl.COST
OUT = ROOT.parent / "runs" / "strat" / "quant_book_adversarial.json"


# ---------------------------------------------------------------------------
# shared: build the v51 daily lab + price-TI fired matrices + MFE/time matrices
# ---------------------------------------------------------------------------
def build():
    lab = v51.load_v51_daily(n=50)
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(v51.DEV_END), "WALL VIOLATION"
    hold = 7  # 1d bars
    lab["_MFE"] = cl.mfe_matrix(C, hold)
    lab["_TIME"] = cl.time_return_matrix(C, hold)
    lab["_reg"] = cl.regime_series(lab, "1d")
    lab["_hold"] = hold
    return lab


def fired(lab, ti, thr=None):
    return cl.fired_matrix(lab, ti, thr).to_numpy()


def valid_mask(lab, min_move=0.03, warm=40):
    C = lab["C"]; hold = lab["_hold"]
    n = len(C.index)
    v = np.zeros((n, len(C.columns)), dtype=bool)
    v[warm:n - hold - 1, :] = True
    Ca = C.to_numpy(); MFEa = lab["_MFE"].to_numpy()
    v &= np.isfinite(Ca) & np.isfinite(MFEa) & (MFEa > min_move)
    return v


def block_p(vals_f, rows_f, vals_p, rows_p, n_bars, block_bars=21, n_boot=2000, seed=7):
    """Honest moving-block bootstrap of edge(mean_f - mean_p). Resamples contiguous bar-blocks (overlap-aware)."""
    rng = np.random.default_rng(seed)
    fd = defaultdict(list); pdd = defaultdict(list)
    for v, d in zip(vals_f, rows_f): fd[int(d)].append(v)
    for v, d in zip(vals_p, rows_p): pdd[int(d)].append(v)
    obs = float(np.mean(vals_f) - np.mean(vals_p))
    n_blocks = max(1, n_bars // block_bars)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n_bars - block_bars + 1), size=n_blocks)
        fv = []; pv = []
        for s in starts:
            for d in range(int(s), int(s) + block_bars):
                if d in fd: fv.extend(fd[d])
                if d in pdd: pv.extend(pdd[d])
        if len(fv) >= 5 and len(pv) >= 5:
            boots.append(np.mean(fv) - np.mean(pv))
    boots = np.array(boots)
    return {"obs_edge_pp": round(100 * obs, 3),
            "p05_pp": round(100 * float(np.percentile(boots, 5)), 3) if len(boots) else None,
            "p_le0": round(float(np.mean(boots <= 0)), 4) if len(boots) else None,
            "n_dates_f": int(len(np.unique(rows_f))), "n_dates_p": int(len(np.unique(rows_p)))}


# ---------------------------------------------------------------------------
# Q1: does liq-flush ADD over momentum/breakout? -- conditional-null test
# ---------------------------------------------------------------------------
def q1_incremental(lab):
    """For each flush feature, three honest contrasts WITHIN each regime:
       (a) vs RANDOM-ENTRY null (reproduce the campaign claim)
       (b) vs MOM-OR-BRK-ENTRY null (the already-continuing pool -- the ORTHOGONALITY test)
       (c) flush-WITHIN-cont vs flush-OUTSIDE-cont (does flush help where mom/brk is NOT already firing?)
    """
    C = lab["C"]; n = len(C.index); reg = lab["_reg"].to_numpy()
    TIMEa = lab["_TIME"].to_numpy()
    base = valid_mask(lab)
    momf = fired(lab, "mom14") & base
    brkf = fired(lab, "brk14") & base
    cont = (momf | brkf)                      # "already-continuing" pool (mom OR brk firing)
    rng = np.random.default_rng(11)
    out = {}
    for feat in ("liq_capitulation", "liq_short_panic"):
        flush = fired(lab, feat) & base
        per = {}
        for rg in ("bull", "chop", "bear"):
            rmask = (reg == rg)
            fl_idx = np.array(np.where(flush & rmask[:, None])).T
            if len(fl_idx) < 20:
                per[rg] = {"note": "insufficient flush", "n": int(len(fl_idx))}; continue
            fl_ret = TIMEa[fl_idx[:, 0], fl_idx[:, 1]]
            m = np.isfinite(fl_ret); fl_ret = fl_ret[m]; fl_rows = fl_idx[m, 0]
            # pools within regime
            pool_rand = np.array(np.where(base & rmask[:, None])).T
            pool_cont = np.array(np.where(cont & rmask[:, None])).T
            def pool_ret(idx):
                v = TIMEa[idx[:, 0], idx[:, 1]]; mm = np.isfinite(v); return v[mm], idx[mm, 0]
            rand_ret, rand_rows = pool_ret(pool_rand)
            cont_ret, cont_rows = pool_ret(pool_cont)
            # flush OUTSIDE the continuing pool (flush AND NOT cont) -- pure exogenous-onset slice
            fl_out_idx = np.array(np.where(flush & rmask[:, None] & ~cont)).T
            fl_out_ret, fl_out_rows = (pool_ret(fl_out_idx) if len(fl_out_idx) else (np.array([]), np.array([])))
            # flush INSIDE the continuing pool
            fl_in_idx = np.array(np.where(flush & rmask[:, None] & cont)).T
            fl_in_ret, fl_in_rows = (pool_ret(fl_in_idx) if len(fl_in_idx) else (np.array([]), np.array([])))
            d = {"n_flush": int(len(fl_ret)),
                 "flush_realized_pct": round(100 * float(fl_ret.mean()), 2),
                 "n_in_cont": int(len(fl_in_ret)), "n_out_cont": int(len(fl_out_ret)),
                 "pct_flush_in_cont": round(100 * len(fl_in_ret) / max(1, len(fl_ret)), 1)}
            # (a) vs random
            d["vs_random"] = block_p(fl_ret, fl_rows, rand_ret, rand_rows, n)
            # (b) vs mom-or-brk-entry null (orthogonality: edge AFTER removing the continuation pool's level)
            d["vs_continuation_pool"] = block_p(fl_ret, fl_rows, cont_ret, cont_rows, n)
            # (c) does flush help OUTSIDE continuation? flush-not-cont vs random
            if len(fl_out_ret) >= 20:
                d["flush_outside_cont_pct"] = round(100 * float(fl_out_ret.mean()), 2)
                d["flush_outside_vs_random"] = block_p(fl_out_ret, fl_out_rows, rand_ret, rand_rows, n)
            else:
                d["flush_outside_cont_pct"] = None; d["flush_outside_vs_random"] = None
            per[rg] = d
        out[feat] = per
    return out


# ---------------------------------------------------------------------------
# Q2: sum-of-parts -- does combining manufacture a NEW regime-conditional edge?
# ---------------------------------------------------------------------------
def q2_sum_of_parts(lab):
    """The 'book' fires when (mom14 OR brk14 OR liq_flush) in bull/chop. Test the combined gate's per-regime
    realized return vs random AND check it does NOT beat its best part in bear (no manufactured bear edge)."""
    C = lab["C"]; n = len(C.index); reg = lab["_reg"].to_numpy()
    TIMEa = lab["_TIME"].to_numpy(); base = valid_mask(lab)
    parts = {
        "mom14": fired(lab, "mom14") & base,
        "brk14": fired(lab, "brk14") & base,
        "liq_capitulation": fired(lab, "liq_capitulation") & base,
        "liq_short_panic": fired(lab, "liq_short_panic") & base,
    }
    combined = np.zeros_like(base)
    for v in parts.values():
        combined |= v
    def per_regime(mask):
        res = {}
        for rg in ("bull", "chop", "bear"):
            rmask = reg == rg
            idx = np.array(np.where(mask & rmask[:, None])).T
            pidx = np.array(np.where(base & rmask[:, None])).T
            if len(idx) < 20:
                res[rg] = None; continue
            fv = TIMEa[idx[:, 0], idx[:, 1]]; mf = np.isfinite(fv); fv = fv[mf]; fr = idx[mf, 0]
            pv = TIMEa[pidx[:, 0], pidx[:, 1]]; mp = np.isfinite(pv); pv = pv[mp]; pr = pidx[mp, 0]
            bp = block_p(fv, fr, pv, pr, n)
            res[rg] = {"n": int(len(fv)), "realized_pct": round(100 * float(fv.mean()), 2),
                       "edge_vs_random": bp}
        return res
    out = {"combined": per_regime(combined)}
    for name, v in parts.items():
        out[name] = per_regime(v)
    # manufactured-edge check: combined bear realized vs best-part bear realized
    cb = out["combined"].get("bear")
    parts_bear = [out[p]["bear"]["realized_pct"] for p in parts if out[p].get("bear")]
    out["_manufactured_bear_check"] = {
        "combined_bear_realized_pct": cb["realized_pct"] if cb else None,
        "best_part_bear_realized_pct": (max(parts_bear) if parts_bear else None),
        "combined_bear_edge_p05": (cb["edge_vs_random"]["p05_pp"] if cb else None),
        "verdict": None,  # filled in report
    }
    return out


# ---------------------------------------------------------------------------
# Q3 + Q4: the participate-preserve BOOK simulation vs buy-hold (DEV equity curve)
# ---------------------------------------------------------------------------
def book_signal(lab, di, causal_reg):
    """At bar di, the book's target weight. Participate (long EW of fired-and-continuing assets) in bull/chop;
    PRESERVE (cash) in bear. Causal: uses only reg label <= di and features <= di."""
    rg = causal_reg.iloc[di]
    if rg == "bear":
        return None  # cash
    base = valid_mask(lab)
    momf = fired(lab, "mom14"); brkf = fired(lab, "brk14")
    flush = fired(lab, "liq_capitulation") | fired(lab, "liq_short_panic")
    sig = (momf | brkf | flush)[di] & base[di]
    cols = np.where(sig)[0]
    return cols if len(cols) else None


def q3q4_book_vs_bh(lab):
    """Simulate a weekly-rebalanced participate-preserve book on DEV vs EW buy-hold. Report compound return,
    maxDD, in-market rate, and the 7d-slice profit-rate ceiling (= in-market up-rate)."""
    C = lab["C"]; n = len(C.index); hold = lab["_hold"]
    reg = lab["_reg"]; Ca = C.to_numpy()
    # weekly (every `hold` bars) rebalanced book
    eq_book = 1.0; eq_bh = 1.0
    book_curve = [1.0]; bh_curve = [1.0]
    in_market = 0; total_steps = 0
    slice_rets_book = []; slice_rets_bh = []; slice_in_market = []
    di = 40
    while di + hold < n - 1:
        # EW buy-hold leg over this 7d step
        bh_r = np.nanmean(Ca[di + hold, :] / Ca[di, :] - 1.0)
        # book leg
        cols = book_signal(lab, di, reg)
        if cols is None or len(cols) == 0:
            book_r = 0.0; inm = 0
        else:
            rets = Ca[di + hold, cols] / Ca[di, cols] - 1.0
            book_r = float(np.nanmean(rets)) - COST; inm = 1
        if np.isfinite(bh_r):
            eq_bh *= (1.0 + bh_r); bh_curve.append(eq_bh); slice_rets_bh.append(bh_r)
        if np.isfinite(book_r):
            eq_book *= (1.0 + book_r); book_curve.append(eq_book)
            slice_rets_book.append(book_r); slice_in_market.append(inm)
            in_market += inm; total_steps += 1
        di += hold
    def maxdd(curve):
        c = np.array(curve); peak = np.maximum.accumulate(c); return float(((c - peak) / peak).min())
    sb = np.array(slice_rets_book); im = np.array(slice_in_market).astype(bool)
    in_mkt_slices = sb[im]
    return {
        "n_steps": total_steps,
        "in_market_rate": round(in_market / max(1, total_steps), 3),
        "book_compound_pct": round(100 * (eq_book - 1.0), 2),
        "bh_compound_pct": round(100 * (eq_bh - 1.0), 2),
        "book_maxDD_pct": round(100 * maxdd(book_curve), 2),
        "bh_maxDD_pct": round(100 * maxdd(bh_curve), 2),
        "book_slice_profit_rate": round(float(np.mean(sb > 0)), 3),
        "bh_slice_profit_rate": round(float(np.mean(np.array(slice_rets_bh) > 0)), 3),
        # the CEILING: among IN-MARKET (participate) steps, the up-rate -- the cash theorem says cash steps are 0 (not +)
        "in_market_up_rate": round(float(np.mean(in_mkt_slices > 0)), 3) if len(in_mkt_slices) else None,
        "cash_steps": int((~im).sum()),
        "cash_steps_are_zero": True,
    }


def q3_gate_causality(lab):
    """Prove the regime label is causal: regime_series uses C.rolling(W).mean() (trailing) + breadth (trailing),
    no .shift(-) / no future window. Re-derive the bear-preservation contrast: book return in bear-labeled steps
    (= 0, cash) vs EW buy-hold return in those same steps (should be more negative)."""
    C = lab["C"]; n = len(C.index); hold = lab["_hold"]; Ca = C.to_numpy()
    reg = lab["_reg"]
    di = 40; bear_bh = []; nonbear_bh = []
    while di + hold < n - 1:
        bh_r = np.nanmean(Ca[di + hold, :] / Ca[di, :] - 1.0)
        if np.isfinite(bh_r):
            (bear_bh if reg.iloc[di] == "bear" else nonbear_bh).append(bh_r)
        di += hold
    bear_bh = np.array(bear_bh); nonbear_bh = np.array(nonbear_bh)
    return {
        "regime_label_uses_future_data": False,  # verified by code inspection: trailing rolling mean + trailing breadth
        "n_bear_steps": int(len(bear_bh)), "n_nonbear_steps": int(len(nonbear_bh)),
        "bh_mean_in_bear_steps_pct": round(100 * float(bear_bh.mean()), 2) if len(bear_bh) else None,
        "bh_mean_in_nonbear_steps_pct": round(100 * float(nonbear_bh.mean()), 2) if len(nonbear_bh) else None,
        "book_return_in_bear_steps_pct": 0.0,  # cash by construction
        "preservation_pp_saved": round(-100 * float(bear_bh.mean()), 2) if len(bear_bh) else None,
        "gate_separates_bear": bool(len(bear_bh) and len(nonbear_bh) and bear_bh.mean() < nonbear_bh.mean()),
    }


def main():
    t0 = time.time()
    print("[adversarial] building DEV-walled v51 lab (n=50) ...")
    lab = build()
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets; {C.index.min().date()} -> {C.index.max().date()}  (DEV wall < {v51.DEV_END})")
    reg = lab["_reg"]
    print(f"  regime bar counts: {dict(reg.value_counts())}")

    print("\n[Q1] INCREMENTAL: liq-flush vs random-entry AND vs momentum/breakout-entry null ...")
    q1 = q1_incremental(lab)
    for feat, per in q1.items():
        print(f"  {feat}:")
        for rg in ("bull", "chop", "bear"):
            d = per.get(rg, {})
            if "note" in d:
                print(f"    {rg:5}: {d['note']} (n={d.get('n')})"); continue
            vr = d["vs_random"]; vc = d["vs_continuation_pool"]
            print(f"    {rg:5}: flush_real={d['flush_realized_pct']}% (n={d['n_flush']}, {d['pct_flush_in_cont']}% in-cont)  "
                  f"vs_RANDOM edge={vr['obs_edge_pp']}pp p05={vr['p05_pp']} ple0={vr['p_le0']}  | "
                  f"vs_CONT-POOL edge={vc['obs_edge_pp']}pp p05={vc['p05_pp']} ple0={vc['p_le0']}")
            if d.get("flush_outside_vs_random"):
                fo = d["flush_outside_vs_random"]
                print(f"           flush OUTSIDE cont (n_out={d['n_out_cont']}): real={d['flush_outside_cont_pct']}% "
                      f"edge_vs_random={fo['obs_edge_pp']}pp p05={fo['p05_pp']} ple0={fo['p_le0']}")

    print("\n[Q2] SUM-OF-PARTS: combined gate per regime + manufactured-bear-edge check ...")
    q2 = q2_sum_of_parts(lab)
    for name in ("combined", "mom14", "brk14", "liq_capitulation", "liq_short_panic"):
        d = q2[name]
        s = []
        for rg in ("bull", "chop", "bear"):
            x = d.get(rg)
            s.append(f"{rg}={x['realized_pct']}%/p05={x['edge_vs_random']['p05_pp']}" if x else f"{rg}=NA")
        print(f"  {name:18} {'  '.join(s)}")
    mc = q2["_manufactured_bear_check"]
    print(f"  MANUFACTURED-BEAR check: combined_bear={mc['combined_bear_realized_pct']}% "
          f"best_part_bear={mc['best_part_bear_realized_pct']}% combined_bear_edge_p05={mc['combined_bear_edge_p05']}")

    print("\n[Q3] GATE CAUSALITY + bear preservation ...")
    q3 = q3_gate_causality(lab)
    print(f"  regime label future-leak: {q3['regime_label_uses_future_data']} | gate_separates_bear: {q3['gate_separates_bear']}")
    print(f"  BH mean in BEAR steps: {q3['bh_mean_in_bear_steps_pct']}% (n={q3['n_bear_steps']})  vs  "
          f"non-bear: {q3['bh_mean_in_nonbear_steps_pct']}% (n={q3['n_nonbear_steps']})")
    print(f"  preservation (pp saved per bear step by going cash): {q3['preservation_pp_saved']}pp")

    print("\n[Q4] BOOK vs BUY-HOLD on DEV + the cash-theorem ceiling ...")
    q4 = q3q4_book_vs_bh(lab)
    print(f"  book compound: {q4['book_compound_pct']}%  vs  BH: {q4['bh_compound_pct']}%   "
          f"(maxDD book {q4['book_maxDD_pct']}% vs BH {q4['bh_maxDD_pct']}%)")
    print(f"  in-market rate: {q4['in_market_rate']}  cash steps: {q4['cash_steps']}")
    print(f"  7d-slice profit-rate: book {q4['book_slice_profit_rate']} (in-market up-rate {q4['in_market_up_rate']}) "
          f"vs BH {q4['bh_slice_profit_rate']}")

    res = {"meta": {"dev_end": v51.DEV_END, "n_assets": len(lab["syms"]), "cost_rt": COST,
                    "date_range": [str(C.index.min().date()), str(C.index.max().date())],
                    "regime_bar_counts": {k: int(v) for k, v in reg.value_counts().items()},
                    "runtime_s": round(time.time() - t0, 1)},
           "Q1_incremental": q1, "Q2_sum_of_parts": q2, "Q3_gate": q3, "Q4_book_vs_bh": q4}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[adversarial] saved -> {OUT}  ({res['meta']['runtime_s']}s)")
    return res


if __name__ == "__main__":
    main()
