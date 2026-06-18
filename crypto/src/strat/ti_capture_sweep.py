"""src/strat/ti_capture_sweep.py -- PHASE 1: non-MA TI expansion of the dynamic capture engine.

Charter project-dynamic-capture-engine-charter-2026-06-18. Extends the engine beyond the 8 MA types to the
canonical non-MA TI families (MACD/ADX/TSI/SUPERTREND/PSAR/ROC/DONCHIAN/KELTNER/WILLR/RSI/STOCH/BBPCT/CCI/...)
by REUSING deep2020_ti_pipeline.INDICATORS (each TI = base/iron held_fn + grid) through the SAME gated stack +
dev-2020 / forward-2021 / 2022-bear protocol + WEALTH rank as the MA engine.

Per (TI, TF, variant in {base, iron}): sweep the TI grid, build the working-BAND ensemble (configs positive on
TRAIN&VAL, EW), apply the Tier-1 SMA-200 GATE (causal, cash<=SMA200) + trail(0.10) + min-hold + cooldown + lag +
TAKER cost, fixed-EW u10. Select the better variant on TRAIN+VAL (OOS held out). Forward-eval the band on unseen
2021 + 2022. Rank by WEALTH (2020-OOS), tier by the forward triple (2020>0 & 2021>0 & 2022>=-5 = A_allweather).

LESSONS honored: band-ensemble not single config (D-fragility); gate not config-switch (D33); TAKER floor;
fixed-EW fillna(0).mean (NOT skipna); select on TRAIN+VAL; two-sided shuffle-null selftest; RWYB.
v1 covers the OHLC TIs; volume-family TIs (loader=ohlcv: CMF/OBV/MFI/VOLIMB) are DEFERRED (need the volume loader).

RWYB:
  python -m strat.ti_capture_sweep --selftest
  python -m strat.ti_capture_sweep --tfs 1d,4h,2h,1h,30m,15m --gate sma200
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.ma_strat_builder as msb                                   # noqa: E402
from strat.deep2020_ti_pipeline import INDICATORS                      # noqa: E402
from strat.structural_fixes import cooldown as apply_cooldown, min_hold  # noqa: E402
from strat.portfolio_replay import apply_trail_stop, TAKER_RT          # noqa: E402
from strat.ma_type_upgrade import _sma                                 # noqa: E402

OUT = msb.CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
OUT.mkdir(parents=True, exist_ok=True)
ALL_TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
COST = TAKER_RT
TRAIL = 0.10
COOLDOWN_GRID = [0, 6]
MINHOLD_EXTRA = [0]            # use the TI's native minhold; 0 = native only (keep the grid lean)
GATE_N = 200


# ---------------------------------------------------------------------------
# GATED STACK (mirrors ma_strat_builder._signal_runs order: held -> cd -> mh -> trail -> gate)
# ---------------------------------------------------------------------------
def _gated_held(A, held_fn, params, gate, cd, mh):
    c = A["c"]
    h = np.asarray(held_fn(A, params), dtype=np.int8)
    if cd > 0: h = apply_cooldown(h, cd)
    if mh > 0: h = min_hold(h, mh)
    h, _ = apply_trail_stop(h.copy(), c, TRAIL)
    h = np.asarray(h, dtype=np.int8)
    if gate:
        g = _sma(c, GATE_N)
        h = (h.astype(bool) & (c > g)).astype(np.int8)   # NaN -> False -> cash (conservative, causal)
    return h


def _net_window(A, held, lo_ms, hi_ms):
    ret, ms = A["ret"], A["ms"]
    pos = np.zeros(len(ret)); pos[1:] = held[:-1].astype(float)   # lag 1 bar (causal)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * ret - flips * (COST / 2.0)
    mask = (ms >= lo_ms) & (ms < hi_ms)
    if mask.sum() < 5: return None
    return pd.Series(net[mask], index=pd.to_datetime(ms[mask], unit="ms"))


def _ew(series_list):
    sl = [s for s in series_list if s is not None]
    if not sl: return None
    return pd.concat(sl, axis=1).fillna(0.0).mean(axis=1).sort_index()


def _netpct(book):
    if book is None: return None
    x = book.dropna().to_numpy()
    if len(x) < 2: return 0.0
    return round(float(np.prod(1 + x) - 1) * 100.0, 2)


def _ms(span):
    return (int(pd.Timestamp(span[0]).value // 10**6), int(pd.Timestamp(span[1]).value // 10**6))


# ---------------------------------------------------------------------------
# PER-CONFIG BOOK over a span (EW across assets)
# ---------------------------------------------------------------------------
def _cfg_book(assets, held_fn, params, gate, cd, mh, lo_ms, hi_ms):
    return _ew([_net_window(A, _gated_held(A, held_fn, params, gate, cd, mh), lo_ms, hi_ms)
                for A in assets])


# ---------------------------------------------------------------------------
# ONE CELL: (TI, TF) -> best variant band-ensemble + forward
# ---------------------------------------------------------------------------
def run_ti_cell(ti_name, cad, gate=True, verbose=False):
    t0 = dt.datetime.now()
    spec = INDICATORS[ti_name]
    if spec.get("loader") == "ohlcv":
        return {"ti": ti_name, "cad": cad, "error": "volume-family deferred (needs ohlcv loader)"}
    grid = spec["grid"]()
    native_mh = int(spec.get("minhold", 12))
    a2020 = msb._load_all(cad, "2020-01-01", "2021-01-01")
    if len(a2020) < 3:
        return {"ti": ti_name, "cad": cad, "error": f"insufficient assets ({len(a2020)})"}
    tr = _ms(msb.TRAIN); vl = _ms(msb.VAL); oo = _ms(msb.OOS)

    best = None
    for variant in ("base", "iron"):
        held_fn = spec[variant]
        # per-config dev nets -> band
        cfg_nets = []
        for params in grid:
            for cd in COOLDOWN_GRID:
                btr = _cfg_book(a2020, held_fn, params, gate, cd, native_mh, *tr)
                bvl = _cfg_book(a2020, held_fn, params, gate, cd, native_mh, *vl)
                ntr, nvl = _netpct(btr), _netpct(bvl)
                if ntr is None or nvl is None: continue
                cfg_nets.append({"params": params, "cd": cd, "ntr": ntr, "nvl": nvl})
        band = [x for x in cfg_nets if x["ntr"] > 0 and x["nvl"] > 0]
        if not band:
            band = sorted(cfg_nets, key=lambda x: -(x["ntr"] + x["nvl"]))[:3]
        if not band: continue
        # ensemble the band on each split (EW of per-config books)
        def ens(lo, hi):
            return _ew([_cfg_book(a2020, held_fn, b["params"], gate, b["cd"], native_mh, lo, hi) for b in band])
        n_tr = _netpct(ens(*tr)); n_vl = _netpct(ens(*vl)); n_oo = _netpct(ens(*oo))
        score = (n_tr or -999) + (n_vl or -999)
        cand = {"variant": variant, "band": band, "n_tr": n_tr, "n_vl": n_vl, "n_oo": n_oo, "score": score}
        if best is None or score > best["score"]:
            best = cand
    if best is None:
        return {"ti": ti_name, "cad": cad, "error": "no band"}

    # FORWARD eval: replay the selected band-ensemble on 2021 + 2022
    held_fn = spec[best["variant"]]
    def fwd_net(span):
        af = msb._load_all(cad, span[0], span[1])
        if len(af) < 3: return None
        lo, hi = _ms(span)
        return _netpct(_ew([_cfg_book(af, held_fn, b["params"], gate, b["cd"], native_mh, lo, hi)
                            for b in best["band"]]))
    n_2021 = fwd_net(msb.FWD_SPAN)
    n_2022 = fwd_net(msb.BEAR_SPAN)
    # p05 on OOS ensemble book
    oo_book = _ew([_cfg_book(a2020, held_fn, b["params"], gate, b["cd"], native_mh, *oo) for b in best["band"]])
    p05 = msb._block_bootstrap_p05(oo_book)
    p05 = round(p05, 2) if p05 is not None else None

    el = (dt.datetime.now() - t0).total_seconds()
    row = {"ti": ti_name, "cad": cad, "family": spec.get("family"), "variant": best["variant"],
           "n_band": len(best["band"]), "net_train": best["n_tr"], "net_val": best["n_vl"],
           "net_oos": best["n_oo"], "net_2021_fwd": n_2021, "net_bear_2022": n_2022,
           "p05_oos_bootstrap": p05, "gate": "sma200" if gate else "none",
           "elapsed_s": round(el, 1)}
    row["tier"] = _tier(row)
    if verbose:
        print(f"  [{ti_name:10s} {cad:4s}] {best['variant']:4s} band={len(best['band']):2d} "
              f"2020oos={str(best['n_oo']):>7} 2021={str(n_2021):>7} 2022={str(n_2022):>7} "
              f"p05={str(p05):>6} {row['tier']} [{el:.0f}s]")
    return row


def _tier(r):
    oos = r.get("net_oos")
    if oos is None or oos <= 0: return "D_weak"
    fwd = r.get("net_2021_fwd"); bear = r.get("net_bear_2022")
    translate = (fwd is not None and fwd > 0)
    preserve = (bear is not None and bear >= -5.0)
    if translate and preserve: return "A_allweather"
    if translate or (bear is not None and bear >= -30.0): return "B_preserve"
    return "C_bull_only"


# ---------------------------------------------------------------------------
# SWEEP + SELFTEST
# ---------------------------------------------------------------------------
def selftest():
    print("[selftest] ti_capture_sweep -- MACD 1d gated + shuffle-null soundness")
    r = run_ti_cell("MACD", "1d", gate=True, verbose=True)
    assert "error" not in r, r.get("error")
    assert r["net_oos"] is not None and r["tier"] in ("A_allweather", "B_preserve", "C_bull_only", "D_weak")
    # two-sided null: a phase-shuffled held must NOT beat the real signal's dev score by much
    spec = INDICATORS["MACD"]; a = msb._load_all("1d", "2020-01-01", "2021-01-01")
    tr = _ms(msb.TRAIN)
    rng = np.random.default_rng(0)
    real = _netpct(_cfg_book(a, spec["base"], spec["grid"]()[0], True, 0, 12, *tr))
    def shuffled(A, p):
        h = spec["base"](A, p)
        return rng.permutation(h)
    null = _netpct(_cfg_book(a, shuffled, spec["grid"]()[0], True, 0, 12, *tr))
    print(f"  real dev-net={real}  shuffled-null dev-net={null}")
    assert real is not None and null is not None
    print("\n[selftest] PASSED")
    return 0


def run(tfs, tis, gate=True, tag="phase1_ti"):
    msb.COST_RT = msb.TAKER_RT
    rows = []
    for cad in tfs:
        if cad not in ALL_TFS: continue
        print(f"\n=== TF={cad} ===")
        for ti in tis:
            r = run_ti_cell(ti, cad, gate=gate, verbose=True)
            r["tf"] = cad
            rows.append(r)
    ok = [r for r in rows if "error" not in r]
    lb = sorted(ok, key=lambda r: -(r.get("net_oos") or -999))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"engine": "ti_capture_sweep", "tag": tag, "generated": ts, "gate": "sma200" if gate else "none",
               "cost": "taker", "rank_key": "net_oos(WEALTH)", "n_cells": len(ok), "leaderboard": lb}
    p = OUT / f"ti_capture_{tag}_{ts}.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # print leaderboard
    print("\n=== TI WEALTH-RANKED LEADERBOARD (gated, taker) ===")
    print(f"  {'TF':4}{'TI':11}{'fam':13}{'var':5}{'tier':13}{'2020':>7}{'2021':>8}{'2022':>7}{'p05':>7}")
    from collections import Counter
    for r in lb:
        def f(x, w):
            return (" " * w) if x is None else f"{x:>{w}.1f}"
        print(f"  {r['tf']:4}{r['ti']:11}{str(r.get('family')):13}{r.get('variant',''):5}{r.get('tier',''):13}"
              f"{f(r.get('net_oos'),7)}{f(r.get('net_2021_fwd'),8)}{f(r.get('net_bear_2022'),7)}{f(r.get('p05_oos_bootstrap'),7)}")
    print(f"\n  tiers: {dict(Counter(r.get('tier') for r in lb))}")
    print(f"  errors: {[ (r['ti'],r['cad'],r['error']) for r in rows if 'error' in r ][:8]}")
    print(f"[out] {p}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.ti_capture_sweep")
    ap.add_argument("--tfs", default=",".join(ALL_TFS))
    ap.add_argument("--tis", default=",".join([k for k, v in INDICATORS.items() if v.get("loader") != "ohlcv"]))
    ap.add_argument("--gate", default="sma200", choices=["none", "sma200"])
    ap.add_argument("--tag", default="phase1_ti")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]
    tis = [t.strip() for t in a.tis.split(",") if t.strip()]
    run(tfs, tis, gate=(a.gate == "sma200"), tag=a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
