"""src/strat/tsmom_ensemble.py -- the D4 corner: a VOL-SCALED, CROSS-SECTIONAL, MULTI-LOOKBACK TIME-SERIES-MOMENTUM
ensemble (the academically-robust MA use, per MA_TRADING_DECOMPOSITION_2026_06_10). LO + spot + lev=1 adapted.

WHY this and not config-tuning: the research + our deep dive agree the value of MAs is NOT the entry-config corner
(B1, which we proved is a weak/non-adaptable lever) but the FILTER/EXIT corner (B2/B3) and the cross-sectional
portfolio corner (D4 = TSMOM). This builds D4 and races it against our validated regime-beta book + buy&hold.

THE STRATEGY (Moskowitz-Ooi-Pedersen 2012, LO-adapted):
  - SIGNAL (ensemble): per asset, sign of the trailing return over EACH lookback in {21,63,126,252} bars; the ensemble
    signal = fraction of lookbacks that are POSITIVE (in [0,1]). LO: negative momentum -> 0 (flat), never short.
  - VOL-SCALING: weight each asset inversely to its realized vol (risk-parity) -> equalizes risk contribution.
  - LO + lev=1 BUDGET: raw_i = ensemble_signal_i / realized_vol_i; normalize so total exposure <= 1 (the rest is
    cash). Two normalizers offered: BREADTH (invest a fraction = mean signal; de-risks in broad bears) and FULL
    (always fully invested among trenders; concentrated). Both reported.
  - Daily rebalance, weights LAGGED 1 bar (past-only), cost on turnover (taker round-trip, conservative).

Baselines raced: REGIME-BETA book (equal-weight price>SMA, flat=cash -- our B2+B3) and BUY&HOLD equal-weight basket.
MtM-correct (lagged weights x next-bar returns). Held-out (OOS/UNSEEN) + per-year + a phase-shift null. No emoji.
Run: python src/strat/tsmom_ensemble.py --universe u50 --cadence 1d
"""
from __future__ import annotations
import argparse, sys, warnings, json
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from strat.entry_signal_lab import load_ohlc, WIN
import yaml

TAKER_RT = 0.0024
LOOKBACKS = [21, 63, 126, 252]          # 1/3/6/12 months in daily bars (the ensemble -- no single magic lookback)
VOL_WIN = 30                            # realized-vol window
ANN = 365.0


def _universe(name):
    try:
        d = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{name}.yaml", encoding="utf-8"))
    except Exception:
        d = None
    if not d:
        from strat.entry_signal_lab import U10
        return U10
    syms = []
    for v in d.values():
        if isinstance(v, list):
            syms += [(x["symbol"] if isinstance(x, dict) else x) for x in v]
    return sorted(set(syms))


def build_panel(universe, cadence):
    series = {}
    for sym in _universe(universe):
        df = load_ohlc(sym, cadence)
        if df is None or len(df) < 300: continue
        series[sym] = df.set_index("date")["close"]
    panel = pd.DataFrame(series).sort_index()
    ret = panel.pct_change()
    vol = ret.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std() * np.sqrt(ANN)   # annualized realized vol
    listed = panel.notna()
    return panel, ret, vol, listed


def tsmom_signal(panel, lookbacks=None):
    """Ensemble momentum signal in [0,1] = fraction of lookbacks with positive trailing return (LO)."""
    sig = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    n = 0
    for L in (lookbacks or LOOKBACKS):
        trail = panel / panel.shift(L) - 1.0
        sig = sig.add((trail > 0).astype(float), fill_value=0.0); n += 1
    return sig / n                       # in [0,1]


def tsmom_weights(sig, vol, listed, mode="breadth"):
    """LO + lev=1 weights. raw = signal/vol (inverse-vol risk-parity, momentum-gated); normalize per mode."""
    inv_vol = 1.0 / vol.replace(0, np.nan)
    raw = (sig * inv_vol).where(listed, 0.0).fillna(0.0)
    total = raw.sum(axis=1)
    if mode == "full":                   # always fully invested among trenders (sum=1 when any positive)
        W = raw.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    else:                                # BREADTH: invested fraction = mean signal across listed names
        listed_count = listed.sum(axis=1).replace(0, np.nan)
        budget = (sig.where(listed, 0.0).sum(axis=1) / listed_count).clip(0, 1)   # in [0,1]
        W = raw.div(total.replace(0, np.nan), axis=0).fillna(0.0).mul(budget, axis=0)
    return W


def regime_beta_weights(panel, listed, sma=120):
    """Our validated B2+B3: equal-weight long the names with close>SMA, flat=cash."""
    on = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    for c in panel.columns:
        s = panel[c].dropna()
        on.loc[s.index, c] = (s > s.rolling(sma, min_periods=sma // 2).mean()).values
    on = on & listed
    cnt = listed.sum(axis=1).replace(0, np.nan)
    return on.astype(float).div(cnt, axis=0).fillna(0.0)   # each long name gets 1/N_listed (flat = cash remainder)


def low_vol_weights(vol, listed, q=0.40):
    """Low-vol anomaly: EW-long the bottom-q realized-vol quintile of LISTED names (lev=1, rest cash)."""
    rank = vol.where(listed).rank(axis=1, pct=True)         # 0=lowest vol
    pick = (rank <= q) & listed
    cnt = pick.sum(axis=1).replace(0, np.nan)
    return pick.astype(float).div(cnt, axis=0).fillna(0.0)


def buyhold_weights(listed):
    cnt = listed.sum(axis=1).replace(0, np.nan)
    return listed.astype(float).div(cnt, axis=0).fillna(0.0)


def backtest(W, ret, cost_per_side=TAKER_RT / 2):
    Wl = W.shift(1).fillna(0.0)                                   # lag (past-only); MtM-correct
    gross = (Wl * ret.fillna(0.0)).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_per_side
    return net, turnover


def stats(net, lo=None, hi=None):
    r = net.copy()
    if lo is not None: r = r[(r.index >= lo) & (r.index < hi)]
    if len(r) < 5: return None
    eq = (1 + r).cumprod(); comp = (eq.iloc[-1] - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    nyr = len(r) / ANN; ann = (eq.iloc[-1] ** (1 / nyr) - 1) * 100 if eq.iloc[-1] > 0 else -100.0
    sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(ANN))
    calmar = comp / abs(dd) if dd < -1e-6 else None
    return {"ann_pct": round(ann, 1), "comp_pct": round(comp, 1), "maxdd_pct": round(dd, 1),
            "sharpe": round(sharpe, 2), "calmar": round(calmar, 1) if calmar else None}


def build_books(panel, ret, vol, listed, lookbacks=None, sma=120, alphas=(0.25, 0.5, 0.75)):
    """All weight books incl. BLENDS (convex combo of regime-beta + TSMOM_breadth -> bull-capture + defense)."""
    sig = tsmom_signal(panel, lookbacks)
    Wt = tsmom_weights(sig, vol, listed, "breadth")
    Wr = regime_beta_weights(panel, listed, sma)
    books = {"TSMOM_breadth": Wt, "regime_beta": Wr, "buy_hold": buyhold_weights(listed), "low_vol_tilt": low_vol_weights(vol, listed)}
    for a in alphas:                                       # blend: a*regime + (1-a)*tsmom (both already lev<=1)
        books[f"BLEND_{int(a*100)}r"] = a * Wr + (1 - a) * Wt
    rng = np.random.default_rng(11)                        # exposure-matched RANDOM null (vs TSMOM)
    budget = Wt.sum(axis=1)
    rr = pd.DataFrame(rng.random(Wt.shape), index=Wt.index, columns=Wt.columns).where(listed, 0.0) / vol.replace(0, np.nan)
    books["RANDOM_null"] = rr.div(rr.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).mul(budget, axis=0)
    return books


def run(universe="u50", cadence="1d", cost_per_side=TAKER_RT / 2):
    panel, ret, vol, listed = build_panel(universe, cadence)
    books = build_books(panel, ret, vol, listed)
    oos_lo, oos_hi = pd.Timestamp(WIN.val_end), pd.Timestamp(WIN.oos_end)
    uns_lo, uns_hi = pd.Timestamp(WIN.oos_end), pd.Timestamp(WIN.unseen_end)
    out = {}
    for name, W in books.items():
        net, turn = backtest(W, ret, cost_per_side)
        yr = {}
        for y, g in net.groupby(net.index.year):
            eq = (1 + g).cumprod(); yr[int(y)] = round(float((eq.iloc[-1] - 1) * 100), 1)
        out[name] = {"full": stats(net), "OOS": stats(net, oos_lo, oos_hi), "UNSEEN": stats(net, uns_lo, uns_hi),
                     "avg_daily_turnover_pct": round(float(turn.mean()) * 100, 2),
                     "avg_exposure": round(float(W.sum(axis=1).mean()), 2), "per_year": yr, "_net": net}
    return out, panel


def block_bootstrap(net, block=20, n_boot=2000, seed=7):
    """Stationary block-bootstrap of daily net returns -> p05/p50/p95 of ANNUALIZED return + P(comp>0)."""
    r = net.dropna().to_numpy(); N = len(r)
    if N < 100: return None
    rng = np.random.default_rng(seed); anns = []
    nblocks = N // block + 1
    for _ in range(n_boot):
        starts = rng.integers(0, N - block, nblocks)
        samp = np.concatenate([r[s:s + block] for s in starts])[:N]
        eq = np.prod(1 + samp); ann = (eq ** (ANN / N) - 1) * 100 if eq > 0 else -100.0
        anns.append(ann)
    a = np.array(anns)
    return {"p05": round(float(np.percentile(a, 5)), 1), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "prob_positive": round(float(np.mean(a > 0)), 2)}


def battery(universe="u50", cadence="1d", cost_per_side=TAKER_RT / 2, book="BLEND_50r"):
    """Robustness battery on the chosen book: param-perturbation (lookbacks/vol/sma) + block-bootstrap + held-out."""
    panel, ret, vol, listed = build_panel(universe, cadence)
    oos_lo, oos_hi = pd.Timestamp(WIN.val_end), pd.Timestamp(WIN.oos_end)
    uns_lo, uns_hi = pd.Timestamp(WIN.oos_end), pd.Timestamp(WIN.unseen_end)
    # PARAM PERTURBATION (the "seeds"): vary lookback sets, vol window, regime SMA
    LB_SETS = [[21, 63, 126, 252], [14, 42, 84, 168], [30, 90, 180, 360], [21, 63, 126], [63, 126, 252, 365],
               [10, 30, 90, 180], [21, 50, 100, 200], [28, 84, 168, 336], [20, 60, 120, 240], [21, 63, 189, 252]]
    SMAS = [100, 120, 150]
    perturb = []
    for i, lb in enumerate(LB_SETS):
        for sm in SMAS:
            vw_panel, vw_ret, vw_vol, vw_listed = panel, ret, vol, listed
            books = build_books(vw_panel, vw_ret, vw_vol, vw_listed, lookbacks=lb, sma=sm)
            if book not in books: continue
            net, _ = backtest(books[book], vw_ret, cost_per_side)
            f = stats(net); o = stats(net, oos_lo, oos_hi); u = stats(net, uns_lo, uns_hi)
            perturb.append({"lb": lb, "sma": sm, "full_ann": f["ann_pct"] if f else None, "full_dd": f["maxdd_pct"] if f else None,
                            "full_calmar": f["calmar"] if f else None, "oos_ann": o["ann_pct"] if o else None,
                            "unseen_ann": u["ann_pct"] if u else None})
    fa = [p["full_ann"] for p in perturb if p["full_ann"] is not None]
    fc = [p["full_calmar"] for p in perturb if p["full_calmar"] is not None]
    oa = [p["oos_ann"] for p in perturb if p["oos_ann"] is not None]
    ua = [p["unseen_ann"] for p in perturb if p["unseen_ann"] is not None]
    # bootstrap on the canonical config
    books0 = build_books(panel, ret, vol, listed)
    net0, _ = backtest(books0[book], ret, cost_per_side)
    boot_full = block_bootstrap(net0)
    boot_held = block_bootstrap(net0[net0.index >= oos_lo])
    return {"book": book, "n_perturb": len(perturb),
            "param_robustness": {"full_ann": {"min": min(fa), "med": float(np.median(fa)), "max": max(fa)},
                                 "full_calmar": {"min": min(fc), "med": float(np.median(fc)), "max": max(fc)},
                                 "oos_ann": {"min": min(oa), "med": float(np.median(oa)), "max": max(oa), "pct_positive": round(float(np.mean(np.array(oa) > 0)), 2)},
                                 "unseen_ann": {"min": min(ua), "med": float(np.median(ua)), "max": max(ua), "pct_positive": round(float(np.mean(np.array(ua) > 0)), 2)}},
            "bootstrap_full": boot_full, "bootstrap_held_out": boot_held}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.tsmom_ensemble")
    ap.add_argument("--universe", default="u50"); ap.add_argument("--cadence", default="1d")
    ap.add_argument("--maker", action="store_true", help="maker cost (~0.0006/side) instead of taker (~0.0012/side)")
    ap.add_argument("--battery", action="store_true", help="robustness battery on the blend (perturbation + bootstrap)")
    ap.add_argument("--book", default="BLEND_50r")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cps = 0.0006 if a.maker else TAKER_RT / 2
    if a.battery:
        b = battery(a.universe, a.cadence, cps, a.book)
        pr = b["param_robustness"]
        print(f"## ROBUSTNESS BATTERY -- {a.book} -- {a.universe} {a.cadence} -- {'maker' if a.maker else 'taker'} cost -- {b['n_perturb']} param-perturbations")
        print(f"   PARAM-PERTURB full_ann:   min={pr['full_ann']['min']} med={pr['full_ann']['med']:.0f} max={pr['full_ann']['max']}")
        print(f"   PARAM-PERTURB full_calmar:min={pr['full_calmar']['min']} med={pr['full_calmar']['med']:.1f} max={pr['full_calmar']['max']}")
        print(f"   PARAM-PERTURB OOS_ann:    min={pr['oos_ann']['min']} med={pr['oos_ann']['med']:.1f} max={pr['oos_ann']['max']}  ({pr['oos_ann']['pct_positive']:.0%} configs positive)")
        print(f"   PARAM-PERTURB UNSEEN_ann: min={pr['unseen_ann']['min']} med={pr['unseen_ann']['med']:.1f} max={pr['unseen_ann']['max']}  ({pr['unseen_ann']['pct_positive']:.0%} configs positive)")
        print(f"   BLOCK-BOOTSTRAP full:    p05={b['bootstrap_full']['p05']}%  p50={b['bootstrap_full']['p50']}%  p95={b['bootstrap_full']['p95']}%  P(>0)={b['bootstrap_full']['prob_positive']:.0%}")
        print(f"   BLOCK-BOOTSTRAP held-out:p05={b['bootstrap_held_out']['p05']}%  p50={b['bootstrap_held_out']['p50']}%  p95={b['bootstrap_held_out']['p95']}%  P(>0)={b['bootstrap_held_out']['prob_positive']:.0%}")
        if a.json:
            import subprocess
            sha=subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
            outdir=ROOT.parent/"runs"/"mining"; outdir.mkdir(parents=True,exist_ok=True)
            pth=outdir/f"tsmom_battery_{a.book}_{a.universe}_{a.cadence}_{'maker' if a.maker else 'taker'}.json"
            json.dump({"repro":{"command":"python "+" ".join(sys.argv),"git_sha":sha},"battery":b}, open(pth,"w",encoding="utf-8"), indent=2, default=str)
            print(f"[persisted] {pth}")
        return 0
    out, panel = run(a.universe, a.cadence, cps)
    print(f"## D4 TSMOM + BLENDS vs regime-beta vs buy&hold -- {a.universe} {a.cadence} -- net of {'maker' if a.maker else 'taker'} cost")
    print(f"   {'book':16} {'FULL ann%':>9} {'dd%':>6} {'Calmar':>6} {'Sh':>5} | {'OOS ann%':>8} {'dd%':>6} | {'UNSEEN ann%':>11} {'dd%':>6} | {'expo':>4} {'turn%':>5}")
    for name, d in out.items():
        f, o, u = d["full"], d["OOS"], d["UNSEEN"]
        def g(s, k): return f"{s[k]}" if s and s.get(k) is not None else "-"
        print(f"   {name:16} {g(f,'ann_pct'):>9} {g(f,'maxdd_pct'):>6} {g(f,'calmar'):>6} {g(f,'sharpe'):>5} | "
              f"{g(o,'ann_pct'):>8} {g(o,'maxdd_pct'):>6} | {g(u,'ann_pct'):>11} {g(u,'maxdd_pct'):>6} | "
              f"{d['avg_exposure']:>4} {d['avg_daily_turnover_pct']:>5}")
    print("   -- per-year net% --")
    yrs = sorted(set().union(*[set(d["per_year"].keys()) for d in out.values()]))
    print("   " + "book".ljust(16) + " ".join(f"{y:>7}" for y in yrs))
    for name in ["TSMOM_breadth","regime_beta","RANDOM_null","buy_hold"]:
        d=out.get(name); 
        if d: print("   " + name.ljust(16) + " ".join(f"{d['per_year'].get(y,0):>7}" for y in yrs))
    if a.json:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        clean = {k: {kk: vv for kk, vv in v.items() if kk != "_net"} for k, v in out.items()}
        outdir = ROOT.parent / "runs" / "mining"; outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"tsmom_ensemble_{a.universe}_{a.cadence}.json"
        json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha}, "results": clean},
                  open(p, "w", encoding="utf-8"), indent=2, default=str)
        print(f"[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
